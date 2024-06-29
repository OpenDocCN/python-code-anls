# `.\numpy\numpy\_core\fromnumeric.py`

```
"""
Module containing non-deprecated functions borrowed from Numeric.

"""
# 引入必要的模块和库
import functools  # 提供函数式编程的工具
import types  # 提供类型检查和操作类型的工具
import warnings  # 提供警告处理相关的功能

import numpy as np  # 引入 NumPy 库，并使用 np 别名
from .._utils import set_module  # 从上级模块中引入 set_module 函数
from . import multiarray as mu  # 从当前包中引入 multiarray 模块，并使用 mu 别名
from . import overrides  # 从当前包中引入 overrides 模块
from . import umath as um  # 从当前包中引入 umath 模块，并使用 um 别名
from . import numerictypes as nt  # 从当前包中引入 numerictypes 模块，并使用 nt 别名
from .multiarray import asarray, array, asanyarray, concatenate  # 从 multiarray 模块中引入特定函数
from ._multiarray_umath import _array_converter  # 从 _multiarray_umath 模块中引入 _array_converter 函数
from . import _methods  # 从当前包中引入 _methods 模块

_dt_ = nt.sctype2char  # 将 sctype2char 函数结果赋值给 _dt_

# functions that are methods
# 定义一个列表，包含当前模块中可供外部使用的函数名
__all__ = [
    'all', 'amax', 'amin', 'any', 'argmax',
    'argmin', 'argpartition', 'argsort', 'around', 'choose', 'clip',
    'compress', 'cumprod', 'cumsum', 'diagonal', 'mean',
    'max', 'min', 'matrix_transpose',
    'ndim', 'nonzero', 'partition', 'prod', 'ptp', 'put',
    'ravel', 'repeat', 'reshape', 'resize', 'round',
    'searchsorted', 'shape', 'size', 'sort', 'squeeze',
    'std', 'sum', 'swapaxes', 'take', 'trace', 'transpose', 'var',
]

_gentype = types.GeneratorType  # 将 GeneratorType 类型赋值给 _gentype
# save away Python sum
_sum_ = sum  # 将内置的 sum 函数保存到 _sum_

# 定义一个偏函数 array_function_dispatch，使用 overrides 模块中的 array_function_dispatch 函数，并指定 module='numpy'
array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


# functions that are now methods

# 定义一个函数 _wrapit，接受一个对象 obj、一个方法 method，以及其他参数
def _wrapit(obj, method, *args, **kwds):
    conv = _array_converter(obj)  # 调用 _array_converter 函数将 obj 转换为数组
    # 获取数组对象 arr，并调用其 method 方法，传入 args 和 kwds 参数
    arr, = conv.as_arrays(subok=False)
    result = getattr(arr, method)(*args, **kwds)  # 调用 arr 对象的 method 方法，并传入 args 和 kwds 参数

    return conv.wrap(result, to_scalar=False)  # 将结果 wrap 起来，并返回


# 定义一个函数 _wrapfunc，接受一个对象 obj、一个方法 method，以及其他参数
def _wrapfunc(obj, method, *args, **kwds):
    bound = getattr(obj, method, None)  # 尝试从 obj 中获取名为 method 的方法，并赋值给 bound
    if bound is None:
        return _wrapit(obj, method, *args, **kwds)  # 如果 bound 为 None，则调用 _wrapit 函数处理

    try:
        return bound(*args, **kwds)  # 尝试调用 bound 方法，传入 args 和 kwds 参数
    except TypeError:
        # 处理 TypeError 异常，通常出现在对象的类中有该方法，但其签名与 NumPy 的不同的情况下
        # 在 except 子句中调用 _wrapit 函数，确保异常链中包含 traceback
        return _wrapit(obj, method, *args, **kwds)


# 定义一个函数 _wrapreduction，接受一个对象 obj、一个 ufunc、一个方法 method，以及其他参数
def _wrapreduction(obj, ufunc, method, axis, dtype, out, **kwargs):
    passkwargs = {k: v for k, v in kwargs.items()
                  if v is not np._NoValue}  # 创建 passkwargs 字典，筛选出值不是 np._NoValue 的键值对

    if type(obj) is not mu.ndarray:  # 如果 obj 的类型不是 mu.ndarray
        try:
            reduction = getattr(obj, method)  # 尝试从 obj 中获取名为 method 的属性，并赋值给 reduction
        except AttributeError:
            pass
        else:
            # 这个分支用于像 any 这样不支持 dtype 参数的归约操作
            if dtype is not None:
                return reduction(axis=axis, dtype=dtype, out=out, **passkwargs)  # 调用 reduction 方法，传入特定参数
            else:
                return reduction(axis=axis, out=out, **passkwargs)  # 调用 reduction 方法，传入特定参数

    return ufunc.reduce(obj, axis, dtype, out, **passkwargs)  # 调用 ufunc 的 reduce 方法，传入特定参数


# 定义一个函数 _wrapreduction_any_all，接受一个对象 obj、一个 ufunc、一个方法 method，以及其他参数
def _wrapreduction_any_all(obj, ufunc, method, axis, out, **kwargs):
    # 与上面的函数相同，但是 dtype 参数始终为 bool 类型（但不会传递）
    # 创建一个字典 passkwargs，其中包含所有非 numpy._NoValue 值的关键字参数
    passkwargs = {k: v for k, v in kwargs.items()
                  if v is not np._NoValue}
    
    # 检查 obj 对象的类型是否不是 mu.ndarray（即不是 numpy.ndarray 类型）
    if type(obj) is not mu.ndarray:
        # 尝试从 obj 对象中获取指定方法（method）的归约函数
        try:
            reduction = getattr(obj, method)
        # 如果 obj 对象中没有该方法（AttributeError 异常）
        except AttributeError:
            pass  # 如果出现异常则什么都不做
        else:
            # 如果成功获取到方法，则调用该方法进行归约操作，传递给该方法的参数为 axis, out, 和 passkwargs 字典
            return reduction(axis=axis, out=out, **passkwargs)
    
    # 如果 obj 是 numpy.ndarray 类型，则使用 ufunc.reduce 方法进行归约操作
    return ufunc.reduce(obj, axis, bool, out, **passkwargs)
# 定义一个简单的分发函数，返回传入的参数 a 和 out
def _take_dispatcher(a, indices, axis=None, out=None, mode=None):
    return (a, out)

# 使用 array_function_dispatch 装饰器将 take 函数与 _take_dispatcher 分发函数关联起来
@array_function_dispatch(_take_dispatcher)
def take(a, indices, axis=None, out=None, mode='raise'):
    """
    Take elements from an array along an axis.

    When axis is not None, this function does the same thing as "fancy"
    indexing (indexing arrays using arrays); however, it can be easier to use
    if you need elements along a given axis. A call such as
    ``np.take(arr, indices, axis=3)`` is equivalent to
    ``arr[:,:,:,indices,...]``.

    Explained without fancy indexing, this is equivalent to the following use
    of `ndindex`, which sets each of ``ii``, ``jj``, and ``kk`` to a tuple of
    indices::

        Ni, Nk = a.shape[:axis], a.shape[axis+1:]
        Nj = indices.shape
        for ii in ndindex(Ni):
            for jj in ndindex(Nj):
                for kk in ndindex(Nk):
                    out[ii + jj + kk] = a[ii + (indices[jj],) + kk]

    Parameters
    ----------
    a : array_like (Ni..., M, Nk...)
        The source array.
    indices : array_like (Nj...)
        The indices of the values to extract.

        .. versionadded:: 1.8.0

        Also allow scalars for indices.
    axis : int, optional
        The axis over which to select values. By default, the flattened
        input array is used.
    out : ndarray, optional (Ni..., Nj..., Nk...)
        If provided, the result will be placed in this array. It should
        be of the appropriate shape and dtype. Note that `out` is always
        buffered if `mode='raise'`; use other modes for better performance.
    mode : {'raise', 'wrap', 'clip'}, optional
        Specifies how out-of-bounds indices will behave.

        * 'raise' -- raise an error (default)
        * 'wrap' -- wrap around
        * 'clip' -- clip to the range

        'clip' mode means that all indices that are too large are replaced
        by the index that addresses the last element along that axis. Note
        that this disables indexing with negative numbers.

    Returns
    -------
    out : ndarray (Ni..., Nj..., Nk...)
        The returned array has the same type as `a`.

    See Also
    --------
    compress : Take elements using a boolean mask
    ndarray.take : equivalent method
    take_along_axis : Take elements by matching the array and the index arrays

    Notes
    -----

    By eliminating the inner loop in the description above, and using `s_` to
    build simple slice objects, `take` can be expressed  in terms of applying
    fancy indexing to each 1-d slice::

        Ni, Nk = a.shape[:axis], a.shape[axis+1:]
        for ii in ndindex(Ni):
            for kk in ndindex(Nj):
                out[ii + s_[...,] + kk] = a[ii + s_[:,] + kk][indices]

    For this reason, it is equivalent to (but faster than) the following use
    of `apply_along_axis`::

        out = np.apply_along_axis(lambda a_1d: a_1d[indices], axis, a)

    Examples
    --------
    >>> a = [4, 3, 5, 7, 6, 8]
    """
    # 创建一个包含索引的列表
    indices = [0, 1, 4]
    # 使用 `np.take` 函数从数组 `a` 中按照给定的索引取值，并返回一个新的数组
    # 结果是从 `a` 中取出索引为 0, 1, 4 的元素，组成的一维数组
    >>> np.take(a, indices)
    array([4, 3, 6])

    # 如果 `a` 是一个 ndarray，可以使用 "fancy" 索引。
    # 将 `a` 转换为 ndarray 类型
    >>> a = np.array(a)
    # 使用索引数组 `indices` 从 `a` 中获取元素，返回一个新的数组
    # 结果同上，从 `a` 中取出索引为 0, 1, 4 的元素，组成的一维数组
    >>> a[indices]
    array([4, 3, 6])

    # 如果 `indices` 不是一维的，输出的数组也会有相应的维度
    >>> np.take(a, [[0, 1], [2, 3]])
    array([[4, 3],
           [5, 7]])
    """
    # 调用 `_wrapfunc` 函数，对数组 `a` 进行 `take` 操作，返回结果
    return _wrapfunc(a, 'take', indices, axis=axis, out=out, mode=mode)
# 定义一个函数 _reshape_dispatcher，用于分发参数至 reshape 函数
def _reshape_dispatcher(a, /, shape=None, *, newshape=None, order=None, copy=None):
    # 返回参数 a 的元组形式
    return (a,)

# 使用 array_function_dispatch 装饰器，将 _reshape_dispatcher 函数与 reshape 函数关联起来
@array_function_dispatch(_reshape_dispatcher)
# 定义 reshape 函数，用于改变数组的形状而不修改其数据
def reshape(a, /, shape=None, *, newshape=None, order='C', copy=None):
    """
    给数组重新定义形状，但不改变其数据。

    Parameters
    ----------
    a : array_like
        待重新形状的数组。
    shape : int 或 int 元组
        新的形状应该与原始形状兼容。如果是整数，则结果将是该长度的一维数组。
        形状的一个维度可以是 -1。在这种情况下，值将从数组的长度和剩余维度中推断出来。
    newshape : int 或 int 元组
        .. 已弃用:: 2.1
            已由 ``shape`` 参数取代。保留向后兼容性。
    order : {'C', 'F', 'A'}, 可选
        使用此索引顺序读取 ``a`` 的元素，并使用此索引顺序将元素放入重新形状的数组中。
        'C' 表示使用类似 C 的索引顺序读取/写入元素，最后一个轴索引最快变化，第一个轴索引最慢变化。
        'F' 表示使用类似 Fortran 的索引顺序读取/写入元素，第一个索引最快变化，最后一个索引最慢变化。
        注意，'C' 和 'F' 选项不考虑底层数组的内存布局，仅指索引顺序。
        'A' 表示如果 ``a`` 在内存中是 Fortran 连续的，则以类似 Fortran 的索引顺序读取/写入元素，否则以 C 的顺序。
    copy : bool, 可选
        如果为 ``True``，则复制数组数据。如果为 ``None``，只有在 ``order`` 所需时才会进行复制。
        对于 ``False``，如果无法避免复制，则会引发 ``ValueError``。默认值： ``None``。

    Returns
    -------
    reshaped_array : ndarray
        如果可能，这将是一个新的视图对象；否则，它将是一个副本。请注意，无法保证返回数组的内存布局（C 或 Fortran 连续性）。

    See Also
    --------
    ndarray.reshape : 等效的方法。

    Notes
    -----
    不总是能够在不复制数据的情况下改变数组的形状。

    ``order`` 关键字同时指定了从 ``a`` 中 *获取* 值的索引顺序，以及将值 *放置* 到输出数组中的索引顺序。
    例如，假设你有一个数组：

    >>> a = np.arange(6).reshape((3, 2))
    >>> a
    array([[0, 1],
           [2, 3],
           [4, 5]])

    可以将重新形状视为首先拉平数组（使用给定的索引顺序），然后将拉平数组中的元素按照相同的索引顺序插入到新数组中。

    >>> np.reshape(a, (2, 3)) # 使用类似 C 的索引顺序
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.reshape(np.ravel(a), (2, 3)) # equivalent to C ravel then C reshape
    # 将数组展平（按行展平），然后按指定形状重新排列，结果与原始数组相同
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.reshape(a, (2, 3), order='F') # Fortran-like index ordering
    # 按照 Fortran 风格的索引顺序重新排列数组
    array([[0, 4, 3],
           [2, 1, 5]])
    >>> np.reshape(np.ravel(a, order='F'), (2, 3), order='F')
    # 先按 Fortran 风格展平数组，然后按照同样的顺序重新排列成指定形状
    array([[0, 4, 3],
           [2, 1, 5]])

    Examples
    --------
    >>> a = np.array([[1,2,3], [4,5,6]])
    >>> np.reshape(a, 6)
    # 将数组展平为一维数组
    array([1, 2, 3, 4, 5, 6])
    >>> np.reshape(a, 6, order='F')
    # 按 Fortran 风格将数组展平为一维数组
    array([1, 4, 2, 5, 3, 6])

    >>> np.reshape(a, (3,-1))       # the unspecified value is inferred to be 2
    # 将数组按照指定形状重新排列，其中 -1 表示自动推断维度大小
    array([[1, 2],
           [3, 4],
           [5, 6]])
    """
    if newshape is None and shape is None:
        # 如果没有指定 newshape 和 shape 参数，则抛出 TypeError
        raise TypeError(
            "reshape() missing 1 required positional argument: 'shape'")
    if newshape is not None:
        if shape is not None:
            # 如果同时指定了 newshape 和 shape 参数，则抛出 TypeError
            raise TypeError(
                "You cannot specify 'newshape' and 'shape' arguments "
                "at the same time.")
        # 在 NumPy 2.1 版本中已弃用，2024-04-18
        # 发出警告，建议使用 shape=... 或者直接传递形状参数
        warnings.warn(
            "`newshape` keyword argument is deprecated, "
            "use `shape=...` or pass shape positionally instead. "
            "(deprecated in NumPy 2.1)",
            DeprecationWarning,
            stacklevel=2,
        )
        shape = newshape
    if copy is not None:
        # 如果指定了 copy 参数，则使用 _wrapfunc 函数进行操作
        return _wrapfunc(a, 'reshape', shape, order=order, copy=copy)
    # 否则直接使用 _wrapfunc 函数进行操作
    return _wrapfunc(a, 'reshape', shape, order=order)
# 定义一个生成器函数 _choose_dispatcher，用于生成参数 a、choices 和 out 的值
def _choose_dispatcher(a, choices, out=None, mode=None):
    # 返回生成器对象，依次生成 a、choices 中的元素以及 out
    yield a
    yield from choices
    yield out

# 用装饰器 array_function_dispatch 将 _choose_dispatcher 与 choose 函数关联起来
@array_function_dispatch(_choose_dispatcher)
# 定义函数 choose，用于从索引数组和选择数组中构造数组
def choose(a, choices, out=None, mode='raise'):
    """
    Construct an array from an index array and a list of arrays to choose from.
    
    首先，如果感到困惑或不确定，请务必查看示例 - 从其完整的一般性来看，
    该函数比从下面的代码描述（下面的 ndi = `numpy.lib.index_tricks`）更简单。
    
    ``np.choose(a,c) == np.array([c[a[I]][I] for I in ndi.ndindex(a.shape)])``。
    
    但这省略了一些细微之处。这里是一个完全一般的摘要：
    
    给定一个整数“索引”数组（`a`）和一个选择数组序列（`choices`），首先将 `a` 和每个选择数组广播（如果需要）到一个共同形状的数组；
    将它们分别称为 *Ba* 和 *Bchoices[i], i = 0,...,n-1*，这里必然有 ``Ba.shape == Bchoices[i].shape`` 对每个 ``i`` 成立。
    然后，根据如下方式创建一个形状为 ``Ba.shape`` 的新数组：
    
    * 如果 ``mode='raise'``（默认），则首先 `a` 的每个元素（因此 `Ba`）必须在 ``[0, n-1]`` 范围内；
      现在假设 `i`（在该范围内）是 `Ba` 中 ``(j0, j1, ..., jm)`` 位置上的值 - 则在新数组中相同位置的值是 `Bchoices[i]` 中相同位置的值；
    
    * 如果 ``mode='wrap'``，`a` 中的值（因此 `Ba`）可以是任何（有符号）整数；
      使用模算术将超出范围 ``[0, n-1]`` 的整数映射回该范围；然后像上面一样构造新数组；
    
    * 如果 ``mode='clip'``，`a` 中的值（因此 `Ba`）可以是任何（有符号）整数；
      负整数映射为 0；大于 ``n-1`` 的值映射为 ``n-1``；然后像上面一样构造新数组。
    
    Parameters
    ----------
    a : int array
        此数组必须包含 ``[0, n-1]`` 范围内的整数，其中 ``n`` 是选择数量，除非 ``mode=wrap`` 或 ``mode=clip``，在这些情况下任何整数都是允许的。
    choices : sequence of arrays
        选择数组。`a` 和所有选择数组必须可广播到相同的形状。
        如果 `choices` 本身是一个数组（不推荐），则其最外层维度（即对应于 ``choices.shape[0]`` 的维度）被视为定义的“序列”。
    out : array, optional
        如果提供，结果将插入到此数组中。它应该具有适当的形状和 dtype。
        注意，如果 ``mode='raise'``，则始终会缓冲 `out`；对于更好的性能，请使用其他模式。
    """
    pass  # 该函数暂未实现任何具体逻辑，仅有文档字符串提供函数说明
    # mode参数用于指定超出索引范围 `[0, n-1]` 的处理方式：
    # * 'raise'：抛出异常
    # * 'wrap'：值变为 `value mod n`
    # * 'clip'：值 < 0 映射为 0，值 > n-1 映射为 n-1

    # 返回合并后的数组结果。
    merged_array : array
        The merged result.

    # 如果 `a` 和每个选择数组的形状不可广播到相同的形状，则引发 ValueError。
    ValueError: shape mismatch
        If `a` and each choice array are not all broadcastable to the same
        shape.

    # 参见等效的方法 `ndarray.choose`。
    See Also
    --------
    ndarray.choose : equivalent method
    numpy.take_along_axis : Preferable if `choices` is an array

    # 为了减少误解的可能性，即使支持以下所谓的 "滥用"，`choices` 也不应该被认为是单个数组，
    # 即最外层的类似序列的容器应该是列表或元组。
    Notes
    -----
    To reduce the chance of misinterpretation, even though the following
    "abuse" is nominally supported, `choices` should neither be, nor be
    thought of as, a single array, i.e., the outermost sequence-like container
    should be either a list or a tuple.

    # 示例
    Examples
    --------

    # choices 是一个包含四个数组的列表，每个数组有四个元素
    >>> choices = [[0, 1, 2, 3], [10, 11, 12, 13],
    ...   [20, 21, 22, 23], [30, 31, 32, 33]]
    # np.choose([2, 3, 1, 0], choices) 的结果是从 choices 中第三个数组的第一个元素开始，
    # 第四个数组的第二个元素等。结果是一个一维数组 [20, 31, 12,  3]
    >>> np.choose([2, 3, 1, 0], choices)
    array([20, 31, 12,  3])
    
    # np.choose([2, 4, 1, 0], choices, mode='clip') 中，超出索引 4 被映射为 3，结果与上例相同
    >>> np.choose([2, 4, 1, 0], choices, mode='clip')
    array([20, 31, 12,  3])
    
    # np.choose([2, 4, 1, 0], choices, mode='wrap') 中，超出索引 4 被视为 4 对 4 取模，结果是 [20, 1, 12,  3]
    >>> np.choose([2, 4, 1, 0], choices, mode='wrap')
    array([20,  1, 12,  3])
    
    # 示例说明 np.choose 如何进行广播
    >>> a = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
    >>> choices = [-10, 10]
    # np.choose(a, choices) 结果是一个二维数组，通过广播选择填充结果
    >>> np.choose(a, choices)
    array([[ 10, -10,  10],
           [-10,  10, -10],
           [ 10, -10,  10]])

    >>> a = np.array([0, 1]).reshape((2,1,1))
    >>> c1 = np.array([1, 2, 3]).reshape((1,3,1))
    >>> c2 = np.array([-1, -2, -3, -4, -5]).reshape((1,1,5))
    # np.choose(a, (c1, c2)) 结果是一个三维数组，res[0,:,:]=c1, res[1,:,:]=c2
    >>> np.choose(a, (c1, c2))
    array([[[ 1,  1,  1,  1,  1],
            [ 2,  2,  2,  2,  2],
            [ 3,  3,  3,  3,  3]],
           [[-1, -2, -3, -4, -5],
            [-1, -2, -3, -4, -5],
            [-1, -2, -3, -4, -5]]])

    """
    return _wrapfunc(a, 'choose', choices, out=out, mode=mode)
# 定义一个函数 `_repeat_dispatcher`，它接受参数 `a`、`repeats` 和 `axis`，但是它仅仅返回元组 `(a,)`，未使用后续参数
def _repeat_dispatcher(a, repeats, axis=None):
    return (a,)


# 使用装饰器 `array_function_dispatch`，将函数 `repeat` 关联到 `_repeat_dispatcher` 函数
@array_function_dispatch(_repeat_dispatcher)
def repeat(a, repeats, axis=None):
    """
    重复数组中每个元素

    Parameters
    ----------
    a : array_like
        输入数组。
    repeats : int or array of ints
        每个元素的重复次数。`repeats` 会被广播以适应指定轴的形状。
    axis : int, optional
        沿着其重复值的轴。默认情况下，使用扁平化的输入数组，并返回一个扁平化的输出数组。

    Returns
    -------
    repeated_array : ndarray
        输出数组，形状与 `a` 相同，除了沿着给定轴。

    See Also
    --------
    tile : 平铺数组。
    unique : 查找数组的唯一元素。

    Examples
    --------
    >>> np.repeat(3, 4)
    array([3, 3, 3, 3])
    >>> x = np.array([[1,2],[3,4]])
    >>> np.repeat(x, 2)
    array([1, 1, 2, 2, 3, 3, 4, 4])
    >>> np.repeat(x, 3, axis=1)
    array([[1, 1, 1, 2, 2, 2],
           [3, 3, 3, 4, 4, 4]])
    >>> np.repeat(x, [1, 2], axis=0)
    array([[1, 2],
           [3, 4],
           [3, 4]])

    """
    # 调用 `_wrapfunc` 函数，执行实际的重复操作，并返回结果
    return _wrapfunc(a, 'repeat', repeats, axis=axis)


# 定义一个函数 `_put_dispatcher`，它接受参数 `a`、`ind`、`v` 和 `mode`，但是它仅仅返回元组 `(a, ind, v)`
def _put_dispatcher(a, ind, v, mode=None):
    return (a, ind, v)


# 使用装饰器 `array_function_dispatch`，将函数 `put` 关联到 `_put_dispatcher` 函数
@array_function_dispatch(_put_dispatcher)
def put(a, ind, v, mode='raise'):
    """
    替换数组的指定元素为给定的值。

    索引操作在扁平化的目标数组上进行。`put` 大致相当于:

    ::

        a.flat[ind] = v

    Parameters
    ----------
    a : ndarray
        目标数组。
    ind : array_like
        目标索引，解释为整数。
    v : array_like
        要放置在 `a` 中目标索引处的值。如果 `v` 比 `ind` 短，则必要时会重复。
    mode : {'raise', 'wrap', 'clip'}, optional
        指定超出边界索引的行为。

        * 'raise' -- 抛出错误（默认）
        * 'wrap' -- 环绕
        * 'clip' -- 裁剪到范围

        'clip' 模式意味着所有超出范围的索引都被替换为指向沿该轴的最后一个元素的索引。
        注意，这会禁用使用负数索引。在 'raise' 模式下，如果发生异常，目标数组仍可能被修改。

    See Also
    --------
    putmask, place
    put_along_axis : 通过匹配数组和索引数组来放置元素

    Examples
    --------
    >>> a = np.arange(5)
    >>> np.put(a, [0, 2], [-44, -55])
    >>> a
    array([-44,   1, -55,   3,   4])

    >>> a = np.arange(5)
    >>> np.put(a, 22, -5, mode='clip')
    >>> a
    array([ 0,  1,  2,  3, -5])

    """
    try:
        # 尝试获取数组 `a` 的 `put` 方法
        put = a.put
    except AttributeError as e:
        # 如果出现属性错误，抛出类型错误，说明参数 `a` 不是 `numpy.ndarray` 类型
        raise TypeError("argument 1 must be numpy.ndarray, "
                        "not {name}".format(name=type(a).__name__)) from e
    # 调用 put 函数并返回其结果
    return put(ind, v, mode=mode)
# 根据输入的参数 `a`、`axis1` 和 `axis2` 调度函数选择合适的处理函数
def _swapaxes_dispatcher(a, axis1, axis2):
    return (a,)


@array_function_dispatch(_swapaxes_dispatcher)
# 实现数组的轴交换操作
def swapaxes(a, axis1, axis2):
    """
    Interchange two axes of an array.

    Parameters
    ----------
    a : array_like
        Input array.
    axis1 : int
        First axis.
    axis2 : int
        Second axis.

    Returns
    -------
    a_swapped : ndarray
        For NumPy >= 1.10.0, if `a` is an ndarray, then a view of `a` is
        returned; otherwise a new array is created. For earlier NumPy
        versions a view of `a` is returned only if the order of the
        axes is changed, otherwise the input array is returned.

    Examples
    --------
    >>> x = np.array([[1,2,3]])
    >>> np.swapaxes(x,0,1)
    array([[1],
           [2],
           [3]])

    >>> x = np.array([[[0,1],[2,3]],[[4,5],[6,7]]])
    >>> x
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])

    >>> np.swapaxes(x,0,2)
    array([[[0, 4],
            [2, 6]],
           [[1, 5],
            [3, 7]]])

    """
    # 调用包装函数 `_wrapfunc` 执行实际的轴交换操作
    return _wrapfunc(a, 'swapaxes', axis1, axis2)


# 根据输入的参数 `a` 和 `axes` 调度函数选择合适的处理函数
def _transpose_dispatcher(a, axes=None):
    return (a,)


@array_function_dispatch(_transpose_dispatcher)
# 实现数组的轴转置操作
def transpose(a, axes=None):
    """
    Returns an array with axes transposed.

    For a 1-D array, this returns an unchanged view of the original array, as a
    transposed vector is simply the same vector.
    To convert a 1-D array into a 2-D column vector, an additional dimension
    must be added, e.g., ``np.atleast_2d(a).T`` achieves this, as does
    ``a[:, np.newaxis]``.
    For a 2-D array, this is the standard matrix transpose.
    For an n-D array, if axes are given, their order indicates how the
    axes are permuted (see Examples). If axes are not provided, then
    ``transpose(a).shape == a.shape[::-1]``.

    Parameters
    ----------
    a : array_like
        Input array.
    axes : tuple or list of ints, optional
        If specified, it must be a tuple or list which contains a permutation
        of [0,1,...,N-1] where N is the number of axes of `a`. The `i`'th axis
        of the returned array will correspond to the axis numbered ``axes[i]``
        of the input. If not specified, defaults to ``range(a.ndim)[::-1]``,
        which reverses the order of the axes.

    Returns
    -------
    p : ndarray
        `a` with its axes permuted. A view is returned whenever possible.

    See Also
    --------
    ndarray.transpose : Equivalent method.
    moveaxis : Move axes of an array to new positions.
    argsort : Return the indices that would sort an array.

    Notes
    -----
    Use ``transpose(a, argsort(axes))`` to invert the transposition of tensors
    when using the `axes` keyword argument.

    Examples
    --------
    >>> a = np.array([[1, 2], [3, 4]])
    >>> a
    array([[1, 2],
           [3, 4]])
    >>> np.transpose(a)
    array([[1, 3],
           [2, 4]])

    >>> a = np.array([1, 2, 3, 4])
    >>> a

    """
    # 创建一个包含四个元素的一维数组 [1, 2, 3, 4]
    >>> array([1, 2, 3, 4])
    # 对该数组进行转置操作，由于是一维数组，转置后仍然返回原数组
    >>> np.transpose(a)
    array([1, 2, 3, 4])
    
    # 创建一个形状为 (1, 2, 3) 的全为 1 的三维数组
    >>> a = np.ones((1, 2, 3))
    # 对该数组进行转置操作，指定轴的顺序为 (1, 0, 2)，即原第0轴变为第1轴，原第1轴变为第0轴，第2轴保持不变
    >>> np.transpose(a, (1, 0, 2)).shape
    (2, 1, 3)
    
    # 创建一个形状为 (2, 3, 4, 5) 的全为 1 的四维数组
    >>> a = np.ones((2, 3, 4, 5))
    # 对该数组进行转置操作，返回的数组形状变为 (5, 4, 3, 2)，即原数组的各轴顺序完全颠倒
    >>> np.transpose(a).shape
    (5, 4, 3, 2)
    
    """
    返回经过 _wrapfunc 函数处理后的结果，调用了 'transpose' 操作，并传入了 axes 参数
    """
    return _wrapfunc(a, 'transpose', axes)
# 返回一个包含 x 的元组，用于矩阵转置的分派器函数
def _matrix_transpose_dispatcher(x):
    return (x,)

# 使用 array_function_dispatch 装饰器将函数注册为 _matrix_transpose_dispatcher 的分派函数
@array_function_dispatch(_matrix_transpose_dispatcher)
# 定义矩阵转置函数，用于将矩阵（或矩阵堆叠）x 进行转置操作
def matrix_transpose(x, /):
    """
    Transposes a matrix (or a stack of matrices) ``x``.

    This function is Array API compatible.

    Parameters
    ----------
    x : array_like
        Input array having shape (..., M, N) and whose two innermost
        dimensions form ``MxN`` matrices.

    Returns
    -------
    out : ndarray
        An array containing the transpose for each matrix and having shape
        (..., N, M).

    See Also
    --------
    transpose : Generic transpose method.

    Examples
    --------
    >>> np.matrix_transpose([[1, 2], [3, 4]])
    array([[1, 3],
           [2, 4]])

    >>> np.matrix_transpose([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    array([[[1, 3],
            [2, 4]],
           [[5, 7],
            [6, 8]]])

    """
    # 将输入 x 转换为 ndarray 类型
    x = asanyarray(x)
    # 如果 x 的维度小于 2，抛出异常
    if x.ndim < 2:
        raise ValueError(
            f"Input array must be at least 2-dimensional, but it is {x.ndim}"
        )
    # 交换 x 的倒数第一和倒数第二个轴，实现转置操作
    return swapaxes(x, -1, -2)


# 返回一个包含 a 的元组，用于 partition 函数的分派器函数
def _partition_dispatcher(a, kth, axis=None, kind=None, order=None):
    return (a,)

# 使用 array_function_dispatch 装饰器将函数注册为 _partition_dispatcher 的分派函数
@array_function_dispatch(_partition_dispatcher)
# 定义数组分区函数，返回数组的一个分区副本
def partition(a, kth, axis=-1, kind='introselect', order=None):
    """
    Return a partitioned copy of an array.

    Creates a copy of the array and partially sorts it in such a way that
    the value of the element in k-th position is in the position it would be
    in a sorted array. In the output array, all elements smaller than the k-th
    element are located to the left of this element and all equal or greater
    are located to its right. The ordering of the elements in the two
    partitions on the either side of the k-th element in the output array is
    undefined.

    .. versionadded:: 1.8.0

    Parameters
    ----------
    a : array_like
        Array to be sorted.
    kth : int or sequence of ints
        Element index to partition by. The k-th value of the element
        will be in its final sorted position and all smaller elements
        will be moved before it and all equal or greater elements behind
        it. The order of all elements in the partitions is undefined. If
        provided with a sequence of k-th it will partition all elements
        indexed by k-th  of them into their sorted position at once.

        .. deprecated:: 1.22.0
            Passing booleans as index is deprecated.
    axis : int or None, optional
        Axis along which to sort. If None, the array is flattened before
        sorting. The default is -1, which sorts along the last axis.
    kind : {'introselect'}, optional
        Selection algorithm. Default is 'introselect'.

    """
    # 返回数组 a 的一个分区副本，根据 kth 参数指定的位置进行分区
    return np.partition(a, kth, axis=axis, kind=kind, order=order)
    order : str or list of str, optional
        当 `a` 是一个包含字段定义的数组时，此参数指定首先比较哪些字段，第二个字段等等。
        可以将单个字段指定为字符串。不需要指定所有字段，但未指定的字段仍将按照它们在dtype中出现的顺序使用，用于解决平局。

    Returns
    -------
    partitioned_array : ndarray
        与 `a` 相同类型和形状的数组。

    See Also
    --------
    ndarray.partition : 就地对数组进行排序的方法。
    argpartition : 间接分区。
    sort : 完全排序

    Notes
    -----
    不同的选择算法由它们的平均速度、最坏情况性能、工作空间大小以及它们是否稳定来表征。稳定排序保持具有相同键的项的相对顺序不变。
    可用的算法具有以下属性：

    ================= ======= ============= ============ =======
       kind            speed   worst case    work space  stable
    ================= ======= ============= ============ =======
    'introselect'        1        O(n)           0         no
    ================= ======= ============= ============ =======

    所有分区算法在除了最后一个轴以外的任何轴上进行分区时都会对数据进行临时复制。
    因此，沿着最后一个轴进行分区比沿着其他任何轴快，并且使用的空间更少。

    对于复数，排序顺序是词典顺序的。如果实部和虚部均为非nan，则顺序由实部决定，除非它们相等，在这种情况下，顺序由虚部决定。

    Examples
    --------
    >>> a = np.array([7, 1, 7, 7, 1, 5, 7, 2, 3, 2, 6, 2, 3, 0])
    >>> p = np.partition(a, 4)
    >>> p
    array([0, 1, 2, 1, 2, 5, 2, 3, 3, 6, 7, 7, 7, 7]) # 结果可能有所不同

    ``p[4]`` 是 2； ``p[:4]`` 中的所有元素都小于或等于 ``p[4]``， ``p[5:]`` 中的所有元素都大于或等于 ``p[4]``。分区如下：

        [0, 1, 2, 1], [2], [5, 2, 3, 3, 6, 7, 7, 7, 7]

    下面的示例展示了对 `kth` 传递多个值的使用。

    >>> p2 = np.partition(a, (4, 8))
    >>> p2
    array([0, 1, 2, 1, 2, 3, 3, 2, 5, 6, 7, 7, 7, 7])

    ``p2[4]`` 是 2， ``p2[8]`` 是 5。 ``p2[:4]`` 中的所有元素都小于或等于 ``p2[4]``，
    ``p2[5:8]`` 中的所有元素大于或等于 ``p2[4]`` 且小于或等于 ``p2[8]``， ``p2[9:]`` 中的所有元素大于或等于 ``p2[8]``。分区如下：

        [0, 1, 2, 1], [2], [3, 3, 2], [5], [6, 7, 7, 7, 7]
    """
    if axis is None:
        # 对于 np.matrix，flatten 返回 (1, N)，因此始终使用最后一个轴
        a = asanyarray(a).flatten()
        axis = -1
    else:
        # 使用 'K' 顺序对数组进行复制，保证 C 和 Fortran 顺序的最优性能
        a = asanyarray(a).copy(order="K")
    # 使用数组 a 的 partition 方法，根据给定的 kth 值对数组进行分区操作
    a.partition(kth, axis=axis, kind=kind, order=order)
    # 返回分区后的数组 a，该操作会直接修改数组 a 的内容
    return a
# 定义一个分派函数，用于确定 argpartition 函数的参数类型和位置
def _argpartition_dispatcher(a, kth, axis=None, kind=None, order=None):
    # 返回参数 a，这里仅用于分派作用
    return (a,)


# 使用 array_function_dispatch 装饰器将 _argpartition_dispatcher 与 argpartition 函数关联起来
@array_function_dispatch(_argpartition_dispatcher)
# 定义 argpartition 函数，用于沿指定轴使用给定的算法进行间接分区
def argpartition(a, kth, axis=-1, kind='introselect', order=None):
    """
    Perform an indirect partition along the given axis using the
    algorithm specified by the `kind` keyword. It returns an array of
    indices of the same shape as `a` that index data along the given
    axis in partitioned order.

    .. versionadded:: 1.8.0

    Parameters
    ----------
    a : array_like
        Array to sort.
    kth : int or sequence of ints
        Element index to partition by. The k-th element will be in its
        final sorted position and all smaller elements will be moved
        before it and all larger elements behind it. The order of all
        elements in the partitions is undefined. If provided with a
        sequence of k-th it will partition all of them into their sorted
        position at once.

        .. deprecated:: 1.22.0
            Passing booleans as index is deprecated.
    axis : int or None, optional
        Axis along which to sort. The default is -1 (the last axis). If
        None, the flattened array is used.
    kind : {'introselect'}, optional
        Selection algorithm. Default is 'introselect'
    order : str or list of str, optional
        When `a` is an array with fields defined, this argument
        specifies which fields to compare first, second, etc. A single
        field can be specified as a string, and not all fields need be
        specified, but unspecified fields will still be used, in the
        order in which they come up in the dtype, to break ties.

    Returns
    -------
    index_array : ndarray, int
        Array of indices that partition `a` along the specified axis.
        If `a` is one-dimensional, ``a[index_array]`` yields a partitioned `a`.
        More generally, ``np.take_along_axis(a, index_array, axis=axis)``
        always yields the partitioned `a`, irrespective of dimensionality.

    See Also
    --------
    partition : Describes partition algorithms used.
    ndarray.partition : Inplace partition.
    argsort : Full indirect sort.
    take_along_axis : Apply ``index_array`` from argpartition
                      to an array as if by calling partition.

    Notes
    -----
    See `partition` for notes on the different selection algorithms.

    Examples
    --------
    One dimensional array:

    >>> x = np.array([3, 4, 2, 1])
    >>> x[np.argpartition(x, 3)]
    array([2, 1, 3, 4]) # may vary
    >>> x[np.argpartition(x, (1, 3))]
    array([1, 2, 3, 4]) # may vary

    >>> x = [3, 4, 2, 1]
    >>> np.array(x)[np.argpartition(x, 3)]
    array([2, 1, 3, 4]) # may vary

    Multi-dimensional array:

    >>> x = np.array([[3, 4, 2], [1, 3, 1]])
    >>> index_array = np.argpartition(x, kth=1, axis=-1)
    >>> # below is the same as np.partition(x, kth=1)
    >>> np.take_along_axis(x, index_array, axis=-1)
    """
    # 创建一个二维数组，包含两个行和三个列的整数数据
    array([[2, 3, 4],
           [1, 1, 3]])
    
    # 返回调用 _wrapfunc 函数的结果，调用参数包括数组 a、函数名 'argpartition'、
    # 分割位置 kth、轴向 axis、分割方式 kind、排序方式 order
    return _wrapfunc(a, 'argpartition', kth, axis=axis, kind=kind, order=order)
# 定义一个排序调度器函数，接受多个参数并返回它们的元组
def _sort_dispatcher(a, axis=None, kind=None, order=None, *, stable=None):
    return (a,)


# 使用装饰器array_function_dispatch将_sort_dispatcher函数与sort函数关联起来
@array_function_dispatch(_sort_dispatcher)
def sort(a, axis=-1, kind=None, order=None, *, stable=None):
    """
    Return a sorted copy of an array.

    Parameters
    ----------
    a : array_like
        Array to be sorted.
    axis : int or None, optional
        Axis along which to sort. If None, the array is flattened before
        sorting. The default is -1, which sorts along the last axis.
    kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
        Sorting algorithm. The default is 'quicksort'. Note that both 'stable'
        and 'mergesort' use timsort or radix sort under the covers and,
        in general, the actual implementation will vary with data type.
        The 'mergesort' option is retained for backwards compatibility.

        .. versionchanged:: 1.15.0.
           The 'stable' option was added.

    order : str or list of str, optional
        When `a` is an array with fields defined, this argument specifies
        which fields to compare first, second, etc.  A single field can
        be specified as a string, and not all fields need be specified,
        but unspecified fields will still be used, in the order in which
        they come up in the dtype, to break ties.
    stable : bool, optional
        Sort stability. If ``True``, the returned array will maintain
        the relative order of ``a`` values which compare as equal.
        If ``False`` or ``None``, this is not guaranteed. Internally,
        this option selects ``kind='stable'``. Default: ``None``.

        .. versionadded:: 2.0.0

    Returns
    -------
    sorted_array : ndarray
        Array of the same type and shape as `a`.

    See Also
    --------
    ndarray.sort : Method to sort an array in-place.
    argsort : Indirect sort.
    lexsort : Indirect stable sort on multiple keys.
    searchsorted : Find elements in a sorted array.
    partition : Partial sort.

    Notes
    -----
    The various sorting algorithms are characterized by their average speed,
    worst case performance, work space size, and whether they are stable. A
    stable sort keeps items with the same key in the same relative
    order. The four algorithms implemented in NumPy have the following
    properties:

    =========== ======= ============= ============ ========
       kind      speed   worst case    work space   stable
    =========== ======= ============= ============ ========
    'quicksort'    1     O(n^2)            0          no
    'heapsort'     3     O(n*log(n))       0          no
    'mergesort'    2     O(n*log(n))      ~n/2        yes
    'timsort'      2     O(n*log(n))      ~n/2        yes
    =========== ======= ============= ============ ========

    .. note:: The datatype determines which of 'mergesort' or 'timsort'
       is actually used, even if 'mergesort' is specified. User selection
       at a finer scale is not currently available.
    """
    # 返回通过指定参数进行排序后的数组的副本
    return (a,)
    For performance, ``sort`` makes a temporary copy if needed to make the data
    `contiguous <https://numpy.org/doc/stable/glossary.html#term-contiguous>`_
    in memory along the sort axis. For even better performance and reduced
    memory consumption, ensure that the array is already contiguous along the
    sort axis.

    The sort order for complex numbers is lexicographic. If both the real
    and imaginary parts are non-nan then the order is determined by the
    real parts except when they are equal, in which case the order is
    determined by the imaginary parts.

    Previous to numpy 1.4.0 sorting real and complex arrays containing nan
    values led to undefined behaviour. In numpy versions >= 1.4.0 nan
    values are sorted to the end. The extended sort order is:

      * Real: [R, nan]
      * Complex: [R + Rj, R + nanj, nan + Rj, nan + nanj]

    where R is a non-nan real value. Complex values with the same nan
    placements are sorted according to the non-nan part if it exists.
    Non-nan values are sorted as before.

    .. versionadded:: 1.12.0

    quicksort has been changed to:
    `introsort <https://en.wikipedia.org/wiki/Introsort>`_.
    When sorting does not make enough progress it switches to
    `heapsort <https://en.wikipedia.org/wiki/Heapsort>`_.
    This implementation makes quicksort O(n*log(n)) in the worst case.

    'stable' automatically chooses the best stable sorting algorithm
    for the data type being sorted.
    It, along with 'mergesort' is currently mapped to
    `timsort <https://en.wikipedia.org/wiki/Timsort>`_
    or `radix sort <https://en.wikipedia.org/wiki/Radix_sort>`_
    depending on the data type.
    API forward compatibility currently limits the
    ability to select the implementation and it is hardwired for the different
    data types.

    .. versionadded:: 1.17.0

    Timsort is added for better performance on already or nearly
    sorted data. On random data timsort is almost identical to
    mergesort. It is now used for stable sort while quicksort is still the
    default sort if none is chosen. For timsort details, refer to
    `CPython listsort.txt
    <https://github.com/python/cpython/blob/3.7/Objects/listsort.txt>`_
    'mergesort' and 'stable' are mapped to radix sort for integer data types.
    Radix sort is an O(n) sort instead of O(n log n).

    .. versionchanged:: 1.18.0

    NaT now sorts to the end of arrays for consistency with NaN.

    Examples
    --------
    >>> a = np.array([[1,4],[3,1]])
    >>> np.sort(a)                # sort along the last axis
    array([[1, 4],
           [1, 3]])
    >>> np.sort(a, axis=None)     # sort the flattened array
    array([1, 1, 3, 4])
    >>> np.sort(a, axis=0)        # sort along the first axis
    array([[1, 1],
           [3, 4]])

    Use the `order` keyword to specify a field to use when sorting a
    structured array:

    >>> dtype = [('name', 'S10'), ('height', float), ('age', int)]
    if axis is None:
        # 如果未指定轴，则将数组展平为一维数组
        a = asanyarray(a).flatten()
        # 设定轴为最后一个维度
        axis = -1
    else:
        # 如果指定了轴，则创建数组的副本，并按照内存中的顺序复制元素
        a = asanyarray(a).copy(order="K")
    # 对数组按照指定的轴进行排序，支持不同的排序算法和顺序
    a.sort(axis=axis, kind=kind, order=order, stable=stable)
    # 返回排序后的数组
    return a
# 为 _argsort_dispatcher 函数创建一个分发器，用于根据参数类型调度合适的处理函数
def _argsort_dispatcher(a, axis=None, kind=None, order=None, *, stable=None):
    # 返回参数元组 (a,)
    return (a,)

# 使用 array_function_dispatch 装饰器将 argsort 函数注册到数组函数的分发系统中
@array_function_dispatch(_argsort_dispatcher)
def argsort(a, axis=-1, kind=None, order=None, *, stable=None):
    """
    返回对数组排序后的索引。

    在指定的轴上使用给定的排序算法执行间接排序。它返回与数组 `a` 相同形状的索引数组，
    该数组按排序顺序索引数据。

    Parameters
    ----------
    a : array_like
        待排序的数组。
    axis : int or None, optional
        要排序的轴。默认为 -1（最后一个轴）。如果为 None，则使用扁平化的数组。
    kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
        排序算法。默认为 'quicksort'。注意 'stable' 和 'mergesort' 选项在内部使用 timsort，
        并且实际实现会根据数据类型而变化。保留 'mergesort' 选项是为了向后兼容性。

        .. versionchanged:: 1.15.0.
           添加了 'stable' 选项。
    order : str or list of str, optional
        当 `a` 是一个有字段定义的数组时，此参数指定首先比较哪些字段、第二个字段等。
        可以将单个字段指定为字符串，不需要指定所有字段，但未指定的字段仍将按它们在 dtype 中出现的顺序使用来打破平局。
    stable : bool, optional
        排序稳定性。如果为 ``True``，返回的数组将保持 ``a`` 值的相对顺序，这些值被视为相等。
        如果为 ``False`` 或 ``None``，则不能保证这一点。在内部，此选项选择 ``kind='stable'``。默认为 ``None``。

        .. versionadded:: 2.0.0

    Returns
    -------
    index_array : ndarray, int
        沿指定 `axis` 排序 `a` 的索引数组。
        如果 `a` 是一维的，则 ``a[index_array]`` 返回一个排序后的 `a`。
        更一般地，``np.take_along_axis(a, index_array, axis=axis)`` 始终返回排序后的 `a`，无论其维度如何。

    See Also
    --------
    sort : 描述使用的排序算法。
    lexsort : 使用多个键进行间接稳定排序。
    ndarray.sort : 原地排序。
    argpartition : 间接部分排序。
    take_along_axis : 将来自 argsort 的 ``index_array`` 应用于数组，就像调用 sort 一样。

    Notes
    -----
    有关不同排序算法的注意事项，请参见 `sort`。

    从 NumPy 1.4.0 开始，`argsort` 可与包含 NaN 值的实数/复数数组一起工作。增强的排序顺序在 `sort` 中有文档记录。

    Examples
    --------
    一维数组:

    >>> x = np.array([3, 1, 2])
    >>> np.argsort(x)
    array([1, 2, 0])

    二维数组:

    >>> x = np.array([[0, 3], [2, 2]])
    >>> x
    array([[0, 3],
           [2, 2]])

    """
    # 对数组 x 按照第一个轴（沿着列）进行排序，返回排序后元素的索引
    ind = np.argsort(x, axis=0)
    # 输出排序后的索引数组
    ind
    # 沿着第一个轴（列）取出排序后的数组元素，等同于 np.sort(x, axis=0)
    np.take_along_axis(x, ind, axis=0)

    # 对数组 x 按照最后一个轴（沿着行）进行排序，返回排序后元素的索引
    ind = np.argsort(x, axis=1)
    # 输出排序后的索引数组
    ind
    # 沿着最后一个轴（行）取出排序后的数组元素，等同于 np.sort(x, axis=1)
    np.take_along_axis(x, ind, axis=1)

    # 返回一个元组，包含按照展开数组 x 后的元素排序的索引数组
    ind = np.unravel_index(np.argsort(x, axis=None), x.shape)
    # 输出展开数组排序后的索引元组
    ind
    # 根据排序后的索引元组获取排序后的数组元素，等同于 np.sort(x, axis=None)
    x[ind]

    # 使用自定义字段进行排序的示例：
    # 创建一个结构化数组 x，包含两个字段 'x' 和 'y'
    x = np.array([(1, 0), (0, 1)], dtype=[('x', '<i4'), ('y', '<i4')])
    # 输出结构化数组 x
    x

    # 按照字段 ('x', 'y') 的顺序对数组 x 进行排序，返回排序后元素的索引
    np.argsort(x, order=('x', 'y'))

    # 按照字段 ('y', 'x') 的顺序对数组 x 进行排序，返回排序后元素的索引
    np.argsort(x, order=('y', 'x'))

    """
    # 调用 _wrapfunc 函数，传递参数进行数组排序操作
    return _wrapfunc(
        a, 'argsort', axis=axis, kind=kind, order=order, stable=stable
    )
# 根据参数生成一个分发函数 _argmax_dispatcher，返回元组 (a, out)
def _argmax_dispatcher(a, axis=None, out=None, *, keepdims=np._NoValue):
    return (a, out)


# 使用装饰器 array_function_dispatch 将 argmax 函数与 _argmax_dispatcher 分发函数关联起来
@array_function_dispatch(_argmax_dispatcher)
def argmax(a, axis=None, out=None, *, keepdims=np._NoValue):
    """
    Returns the indices of the maximum values along an axis.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        By default, the index is into the flattened array, otherwise
        along the specified axis.
    out : array, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the array.

        .. versionadded:: 1.22.0

    Returns
    -------
    index_array : ndarray of ints
        Array of indices into the array. It has the same shape as ``a.shape``
        with the dimension along `axis` removed. If `keepdims` is set to True,
        then the size of `axis` will be 1 with the resulting array having same
        shape as ``a.shape``.

    See Also
    --------
    ndarray.argmax, argmin
    amax : The maximum value along a given axis.
    unravel_index : Convert a flat index into an index tuple.
    take_along_axis : Apply ``np.expand_dims(index_array, axis)``
                      from argmax to an array as if by calling max.

    Notes
    -----
    In case of multiple occurrences of the maximum values, the indices
    corresponding to the first occurrence are returned.

    Examples
    --------
    >>> a = np.arange(6).reshape(2,3) + 10
    >>> a
    array([[10, 11, 12],
           [13, 14, 15]])
    >>> np.argmax(a)
    5
    >>> np.argmax(a, axis=0)
    array([1, 1, 1])
    >>> np.argmax(a, axis=1)
    array([2, 2])

    Indexes of the maximal elements of a N-dimensional array:

    >>> ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
    >>> ind
    (1, 2)
    >>> a[ind]
    15

    >>> b = np.arange(6)
    >>> b[1] = 5
    >>> b
    array([0, 5, 2, 3, 4, 5])
    >>> np.argmax(b)  # Only the first occurrence is returned.
    1

    >>> x = np.array([[4,2,3], [1,0,3]])
    >>> index_array = np.argmax(x, axis=-1)
    >>> # Same as np.amax(x, axis=-1, keepdims=True)
    >>> np.take_along_axis(x, np.expand_dims(index_array, axis=-1), axis=-1)
    array([[4],
           [3]])
    >>> # Same as np.amax(x, axis=-1)
    >>> np.take_along_axis(x, np.expand_dims(index_array, axis=-1),
    ...     axis=-1).squeeze(axis=-1)
    array([4, 3])

    Setting `keepdims` to `True`,

    >>> x = np.arange(24).reshape((2, 3, 4))
    >>> res = np.argmax(x, axis=1, keepdims=True)
    >>> res.shape
    (2, 1, 4)
    """
    # 如果 keepdims 是 np._NoValue，将其作为空字典传递给 _wrapfunc 函数
    kwds = {'keepdims': keepdims} if keepdims is not np._NoValue else {}
    # 调用 _wrapfunc 函数，传递相应的参数和关键字参数
    return _wrapfunc(a, 'argmax', axis=axis, out=out, **kwds)
# 定义一个分派函数 _argmin_dispatcher，返回元组 (a, out)
def _argmin_dispatcher(a, axis=None, out=None, *, keepdims=np._NoValue):
    return (a, out)


# 使用 array_function_dispatch 装饰器将 argmin 函数分派给 _argmin_dispatcher 函数
@array_function_dispatch(_argmin_dispatcher)
def argmin(a, axis=None, out=None, *, keepdims=np._NoValue):
    """
    返回沿指定轴的最小值的索引。

    Parameters
    ----------
    a : array_like
        输入数组。
    axis : int, optional
        默认情况下，索引是在展平数组中，否则沿指定的轴。
    out : array, optional
        如果提供，则结果将插入到此数组中。它应该具有适当的形状和dtype。
    keepdims : bool, optional
        如果设置为True，则减少的轴将作为大小为一的维度保留在结果中。使用此选项，
        结果将正确地与数组广播。

        .. versionadded:: 1.22.0

    Returns
    -------
    index_array : ndarray of ints
        数组中的索引数组。它具有与 `a.shape` 相同的形状，沿 `axis` 的维度被移除。如果 `keepdims` 设置为True，
        那么轴的大小将为1，生成的数组将具有与 `a.shape` 相同的形状。

    See Also
    --------
    ndarray.argmin, argmax
    amin : 沿指定轴的最小值。
    unravel_index : 将平坦索引转换为索引元组。
    take_along_axis : 将 ``np.expand_dims(index_array, axis)`` 从 argmin 应用于数组，就像调用 min 一样。

    Notes
    -----
    对于最小值的多个出现情况，返回与第一次出现相对应的索引。

    Examples
    --------
    >>> a = np.arange(6).reshape(2,3) + 10
    >>> a
    array([[10, 11, 12],
           [13, 14, 15]])
    >>> np.argmin(a)
    0
    >>> np.argmin(a, axis=0)
    array([0, 0, 0])
    >>> np.argmin(a, axis=1)
    array([0, 0])

    N 维数组的最小元素的索引：

    >>> ind = np.unravel_index(np.argmin(a, axis=None), a.shape)
    >>> ind
    (0, 0)
    >>> a[ind]
    10

    >>> b = np.arange(6) + 10
    >>> b[4] = 10
    >>> b
    array([10, 11, 12, 13, 10, 15])
    >>> np.argmin(b)  # 仅返回第一次出现的最小值的索引。
    0

    >>> x = np.array([[4,2,3], [1,0,3]])
    >>> index_array = np.argmin(x, axis=-1)
    >>> # 等同于 np.amin(x, axis=-1, keepdims=True)
    >>> np.take_along_axis(x, np.expand_dims(index_array, axis=-1), axis=-1)
    array([[2],
           [0]])
    >>> # 等同于 np.amax(x, axis=-1)
    >>> np.take_along_axis(x, np.expand_dims(index_array, axis=-1),
    ...     axis=-1).squeeze(axis=-1)
    array([2, 0])

    设置 `keepdims` 为 `True`，

    >>> x = np.arange(24).reshape((2, 3, 4))
    >>> res = np.argmin(x, axis=1, keepdims=True)
    >>> res.shape
    (2, 1, 4)
    """
    # 如果 keepdims 是 np._NoValue，则创建空字典 kwds，否则创建包含 keepdims 的字典 kwds
    kwds = {'keepdims': keepdims} if keepdims is not np._NoValue else {}
    # 调用 _wrapfunc 函数，传递参数 a, 'argmin', axis=axis, out=out 和 kwds 中的关键字参数
    return _wrapfunc(a, 'argmin', axis=axis, out=out, **kwds)
# 定义一个分派函数 _searchsorted_dispatcher，用于将参数传递给 searchsorted 函数
def _searchsorted_dispatcher(a, v, side=None, sorter=None):
    return (a, v, sorter)

# 使用装饰器 array_function_dispatch 将 _searchsorted_dispatcher 作为分派函数与 searchsorted 函数关联
@array_function_dispatch(_searchsorted_dispatcher)
def searchsorted(a, v, side='left', sorter=None):
    """
    Find indices where elements should be inserted to maintain order.

    Find the indices into a sorted array `a` such that, if the
    corresponding elements in `v` were inserted before the indices, the
    order of `a` would be preserved.

    Assuming that `a` is sorted:

    ======  ============================
    `side`  returned index `i` satisfies
    ======  ============================
    left    ``a[i-1] < v <= a[i]``
    right   ``a[i-1] <= v < a[i]``
    ======  ============================

    Parameters
    ----------
    a : 1-D array_like
        Input array. If `sorter` is None, then it must be sorted in
        ascending order, otherwise `sorter` must be an array of indices
        that sort it.
    v : array_like
        Values to insert into `a`.
    side : {'left', 'right'}, optional
        If 'left', the index of the first suitable location found is given.
        If 'right', return the last such index.  If there is no suitable
        index, return either 0 or N (where N is the length of `a`).
    sorter : 1-D array_like, optional
        Optional array of integer indices that sort array a into ascending
        order. They are typically the result of argsort.

        .. versionadded:: 1.7.0

    Returns
    -------
    indices : int or array of ints
        Array of insertion points with the same shape as `v`,
        or an integer if `v` is a scalar.

    See Also
    --------
    sort : Return a sorted copy of an array.
    histogram : Produce histogram from 1-D data.

    Notes
    -----
    Binary search is used to find the required insertion points.

    As of NumPy 1.4.0 `searchsorted` works with real/complex arrays containing
    `nan` values. The enhanced sort order is documented in `sort`.

    This function uses the same algorithm as the builtin python
    `bisect.bisect_left` (``side='left'``) and `bisect.bisect_right`
    (``side='right'``) functions, which is also vectorized
    in the `v` argument.

    Examples
    --------
    >>> np.searchsorted([11,12,13,14,15], 13)
    2
    >>> np.searchsorted([11,12,13,14,15], 13, side='right')
    3
    >>> np.searchsorted([11,12,13,14,15], [-10, 20, 12, 13])
    array([0, 5, 1, 2])

    """
    # 调用 _wrapfunc 函数，将参数传递给其它函数进行处理，返回处理结果
    return _wrapfunc(a, 'searchsorted', v, side=side, sorter=sorter)


# 定义一个分派函数 _resize_dispatcher，用于将参数传递给 resize 函数
def _resize_dispatcher(a, new_shape):
    return (a,)

# 使用装饰器 array_function_dispatch 将 _resize_dispatcher 作为分派函数与 resize 函数关联
@array_function_dispatch(_resize_dispatcher)
def resize(a, new_shape):
    """
    Return a new array with the specified shape.

    If the new array is larger than the original array, then the new
    array is filled with repeated copies of `a`.  Note that this behavior
    is different from a.resize(new_shape) which fills with zeros instead
    of repeated copies of `a`.

    Parameters
    ----------
    a : array_like
        Array to be resized.

    new_shape : tuple of ints
        The new shape should be compatible with the original shape. If
        an integer, then the result will be a 1-D array of that length.
        One shape dimension can be -1. In this case, the value is
        inferred from the length of the array and remaining dimensions.

    Returns
    -------
    reshaped_array : ndarray
        The new array with the specified shape.

    Examples
    --------
    >>> a = np.array([[0, 1], [2, 3]])
    >>> np.resize(a, (2,3))
    array([[0, 1, 2],
           [3, 0, 1]])
    >>> np.resize(a, (1,4))
    array([[0, 1, 2, 3]])
    >>> np.resize(a,(2, 1))
    array([[0],
           [1]])

    """
    # 返回调用 _wrapfunc 函数后的结果，将参数传递给其它函数进行处理
    return _wrapfunc(a, 'resize', new_shape)
    new_shape : int or tuple of int
        # 定义参数 new_shape，可以是单个整数或整数元组，表示重置后数组的形状。

    Returns
    -------
    reshaped_array : ndarray
        # 返回重塑后的数组，从旧数组的数据中形成，必要时重复以填充所需的元素数。数据按C顺序重复。

    See Also
    --------
    numpy.reshape : 不改变总大小的情况下重塑数组。
    numpy.pad : 扩展和填充数组。
    numpy.repeat : 重复数组的元素。
    ndarray.resize : 就地调整数组大小。

    Notes
    -----
    当数组的总大小不变时，应使用 `~numpy.reshape`。在大多数其他情况下，索引（用于减小大小）或填充（用于增加大小）可能是更合适的解决方案。

    Warning: 此功能 **不** 单独考虑轴，即不应用插值/外推。它填充返回数组以满足所需的元素数，在C顺序下迭代 `a`，忽略轴（如果新形状较大，则从开始处循环）。因此，此功能不适合调整图像或每个轴表示独立和不同实体的数据。

    Examples
    --------
    >>> a=np.array([[0,1],[2,3]])
    >>> np.resize(a,(2,3))
    array([[0, 1, 2],
           [3, 0, 1]])
    >>> np.resize(a,(1,4))
    array([[0, 1, 2, 3]])
    >>> np.resize(a,(2,4))
    array([[0, 1, 2, 3],
           [0, 1, 2, 3]])

    """
    if isinstance(new_shape, (int, nt.integer)):
        # 如果 new_shape 是整数或整数类型，则转换为元组形式
        new_shape = (new_shape,)

    a = ravel(a)

    new_size = 1
    for dim_length in new_shape:
        # 计算新形状下的总大小
        new_size *= dim_length
        if dim_length < 0:
            raise ValueError(
                'all elements of `new_shape` must be non-negative'
            )

    if a.size == 0 or new_size == 0:
        # 如果原始数组或新大小为零，返回一个与 a 类型和形状相同的零数组
        return np.zeros_like(a, shape=new_shape)

    repeats = -(-new_size // a.size)  # ceil division，计算重复次数
    a = concatenate((a,) * repeats)[:new_size]  # 将数组按重复次数连接，并截取所需长度

    return reshape(a, new_shape)
# 创建一个分发器函数 _squeeze_dispatcher，接受参数 a 和 axis，并返回一个包含 a 的元组
def _squeeze_dispatcher(a, axis=None):
    # 返回一个元组，包含参数 a
    return (a,)


# 使用 array_function_dispatch 装饰器将 _squeeze_dispatcher 应用于 squeeze 函数
@array_function_dispatch(_squeeze_dispatcher)
def squeeze(a, axis=None):
    """
    Remove axes of length one from `a`.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : None or int or tuple of ints, optional
        .. versionadded:: 1.7.0

        Selects a subset of the entries of length one in the
        shape. If an axis is selected with shape entry greater than
        one, an error is raised.

    Returns
    -------
    squeezed : ndarray
        The input array, but with all or a subset of the
        dimensions of length 1 removed. This is always `a` itself
        or a view into `a`. Note that if all axes are squeezed,
        the result is a 0d array and not a scalar.

    Raises
    ------
    ValueError
        If `axis` is not None, and an axis being squeezed is not of length 1

    See Also
    --------
    expand_dims : The inverse operation, adding entries of length one
    reshape : Insert, remove, and combine dimensions, and resize existing ones

    Examples
    --------
    >>> x = np.array([[[0], [1], [2]]])
    >>> x.shape
    (1, 3, 1)
    >>> np.squeeze(x).shape
    (3,)
    >>> np.squeeze(x, axis=0).shape
    (3, 1)
    >>> np.squeeze(x, axis=1).shape
    Traceback (most recent call last):
    ...
    ValueError: cannot select an axis to squeeze out which has size
    not equal to one
    >>> np.squeeze(x, axis=2).shape
    (1, 3)
    >>> x = np.array([[1234]])
    >>> x.shape
    (1, 1)
    >>> np.squeeze(x)
    array(1234)  # 0d array
    >>> np.squeeze(x).shape
    ()
    >>> np.squeeze(x)[()]
    1234

    """
    # 尝试获取参数 a 的 squeeze 方法，如果不存在则调用 _wrapit 函数包装并返回结果
    try:
        squeeze = a.squeeze
    except AttributeError:
        return _wrapit(a, 'squeeze', axis=axis)
    # 如果 axis 为 None，则调用 squeeze() 方法
    if axis is None:
        return squeeze()
    else:
        # 否则，调用 squeeze(axis=axis) 方法
        return squeeze(axis=axis)


# 创建一个分发器函数 _diagonal_dispatcher，接受参数 a, offset, axis1, axis2，并返回一个包含 a 的元组
def _diagonal_dispatcher(a, offset=None, axis1=None, axis2=None):
    # 返回一个元组，包含参数 a
    return (a,)


# 使用 array_function_dispatch 装饰器将 _diagonal_dispatcher 应用于 diagonal 函数
@array_function_dispatch(_diagonal_dispatcher)
def diagonal(a, offset=0, axis1=0, axis2=1):
    """
    Return specified diagonals.

    If `a` is 2-D, returns the diagonal of `a` with the given offset,
    i.e., the collection of elements of the form ``a[i, i+offset]``.  If
    `a` has more than two dimensions, then the axes specified by `axis1`
    and `axis2` are used to determine the 2-D sub-array whose diagonal is
    returned.  The shape of the resulting array can be determined by
    removing `axis1` and `axis2` and appending an index to the right equal
    to the size of the resulting diagonals.

    In versions of NumPy prior to 1.7, this function always returned a new,
    independent array containing a copy of the values in the diagonal.

    In NumPy 1.7 and 1.8, it continues to return a copy of the diagonal,
    but depending on this fact is deprecated. Writing to the resulting
    array continues to work as it used to, but a FutureWarning is issued.
    """
    # 这里缺少返回值的部分，需要注意补充上相关代码和注释
    # Starting in NumPy 1.9, np.diagonal() returns a read-only view of the original array's diagonals.
    # Attempts to modify this view will result in an error.
    
    # In a future release of NumPy, np.diagonal() may return a read/write view, allowing modifications
    # to affect the original array. The returned array will maintain the same type as the input array.
    
    # If you do not intend to modify the array returned by np.diagonal(), you can disregard the above
    # considerations.
    
    # If your code relies on the current read-only behavior, it's recommended to explicitly copy the
    # returned array, e.g., use np.diagonal(a).copy() instead of np.diagonal(a). This ensures
    # compatibility with both current and future versions of NumPy.
    
    # Parameters:
    # a : array_like
    #     Input array from which diagonals are extracted.
    # offset : int, optional
    #     Offset of the diagonal from the main diagonal. Can be positive or negative. Defaults to 0.
    # axis1 : int, optional
    #     First axis of the 2-D sub-arrays from which diagonals should be taken. Defaults to 0.
    # axis2 : int, optional
    #     Second axis of the 2-D sub-arrays from which diagonals should be taken. Defaults to 1.
    
    # Returns:
    # -------
    # array_of_diagonals : ndarray
    #     If `a` is 2-D, a 1-D array containing the diagonals of `a` of the same type.
    #     If `a` has more than 2 dimensions, dimensions specified by `axis1` and `axis2` are removed,
    #     and a new axis is inserted at the end corresponding to the diagonal.
    
    # Raises:
    # ------
    # ValueError
    #     If `a` has less than 2 dimensions.
    
    # See Also:
    # --------
    # diag : Extract diagonals as a new 1-D array.
    # diagflat : Create diagonal arrays.
    # trace : Sum along diagonals.
    
    # Examples:
    # ---------
    # Example 1: 2-D array
    # >>> a = np.arange(4).reshape(2,2)
    # >>> a
    # array([[0, 1],
    #        [2, 3]])
    # >>> a.diagonal()
    # array([0, 3])
    # >>> a.diagonal(1)
    # array([1])
    
    # Example 2: 3-D array
    # >>> a = np.arange(8).reshape(2,2,2)
    # >>> a
    # array([[[0, 1],
    #         [2, 3]],
    #        [[4, 5],
    #         [6, 7]]])
    # >>> a.diagonal(0, 0, 1)
    # array([[0, 6],
    #        [1, 7]])
    
    # Explanation:
    # The function `np.diagonal()` extracts diagonals from arrays of various dimensions,
    # supporting multi-dimensional slicing through `axis1` and `axis2` parameters.
    The anti-diagonal can be obtained by reversing the order of elements
    using either `numpy.flipud` or `numpy.fliplr`.

    >>> a = np.arange(9).reshape(3, 3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> np.fliplr(a).diagonal()  # Horizontal flip
    array([2, 4, 6])
    >>> np.flipud(a).diagonal()  # Vertical flip
    array([6, 4, 2])

    Note that the order in which the diagonal is retrieved varies depending
    on the flip function.
    """
    # 如果输入的数组是一个 numpy 矩阵（matrix），则将其对角线转换为 1 维数组，以保持向后兼容性。
    if isinstance(a, np.matrix):
        return asarray(a).diagonal(offset=offset, axis1=axis1, axis2=axis2)
    else:
        # 如果输入的数组不是矩阵，则将其转换为任意数组（asanyarray）再提取对角线元素。
        return asanyarray(a).diagonal(offset=offset, axis1=axis1, axis2=axis2)
# 定义一个函数 _trace_dispatcher，用于调度 trace 函数的参数
def _trace_dispatcher(
        a, offset=None, axis1=None, axis2=None, dtype=None, out=None):
    # 返回元组 (a, out)，作为 trace 函数的参数
    return (a, out)


# 使用 array_function_dispatch 装饰器定义 trace 函数
@array_function_dispatch(_trace_dispatcher)
def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    """
    返回数组对角线上的元素之和。

    如果 `a` 是二维数组，返回指定偏移量的主对角线元素之和，即所有 `a[i, i+offset]` 的和。

    如果 `a` 的维度高于二维，则使用指定的 axis1 和 axis2 参数确定要返回其对角线的二维子数组。
    返回数组的形状与 `a` 的形状相同，但移除了 axis1 和 axis2。

    参数
    ----------
    a : array_like
        输入数组，提取对角线元素。
    offset : int, optional
        主对角线的偏移量。可以是正数或负数。默认为 0。
    axis1, axis2 : int, optional
        用于提取对角线的二维子数组的轴。默认为 `a` 的前两个轴。
    dtype : dtype, optional
        决定返回数组和累加器的数据类型。如果 dtype 为 None 并且 `a` 是整数类型且精度低于默认整数精度，
        则使用默认整数精度。否则，精度与 `a` 相同。
    out : ndarray, optional
        存放输出结果的数组。其类型保持不变，必须具有正确的形状来容纳输出。

    返回
    -------
    sum_along_diagonals : ndarray
        如果 `a` 是二维数组，返回沿对角线的和。如果 `a` 的维度更高，则返回沿对角线的和组成的数组。

    参见
    --------
    diag, diagonal, diagflat

    示例
    --------
    >>> np.trace(np.eye(3))
    3.0
    >>> a = np.arange(8).reshape((2,2,2))
    >>> np.trace(a)
    array([6, 8])

    >>> a = np.arange(24).reshape((2,2,2,3))
    >>> np.trace(a).shape
    (2, 3)

    """
    # 如果 `a` 是 matrix 类型，则通过 asarray 转换为数组再调用 trace 方法获取对角线元素之和，以保持向后兼容性
    if isinstance(a, np.matrix):
        return asarray(a).trace(
            offset=offset, axis1=axis1, axis2=axis2, dtype=dtype, out=out
        )
    else:
        # 否则，将 `a` 转换为数组并调用 trace 方法获取对角线元素之和
        return asanyarray(a).trace(
            offset=offset, axis1=axis1, axis2=axis2, dtype=dtype, out=out
        )


# 定义一个函数 _ravel_dispatcher，用于调度 ravel 函数的参数
def _ravel_dispatcher(a, order=None):
    # 返回元组 (a,)，作为 ravel 函数的参数
    return (a,)


# 使用 array_function_dispatch 装饰器定义 ravel 函数
@array_function_dispatch(_ravel_dispatcher)
def ravel(a, order='C'):
    """返回一个连续展平的数组。

    返回一个包含输入元素的一维数组。仅在需要时才会复制。

    从 NumPy 1.10 开始，返回的数组将具有与输入数组相同的类型。
    （例如，对于输入的掩码数组，将返回掩码数组）

    参数
    ----------
    a : array_like
        输入数组，要展平。
    order : {'C', 'F', 'A', 'K'}, optional
        指定数组元素在展平时的顺序（C 表示按行，F 表示按列，A 表示按原顺序，K 表示按内存顺序）。默认为 'C'。

    返回
    -------
    raveled_array : ndarray
        连续展平后的一维数组。

    """
    # 返回使用 asarray 转换的数组的展平结果，以保持向后兼容性
    return asarray(a).ravel(order=order)
    # 定义函数 `ravel`
    def ravel(a, order='C'):
        # 返回输入数组 `a` 的展平版本，按照指定的顺序 `order` 进行展平
        y = a.ravel(order=order)
        # 返回展平后的数组 `y`，保持 `a` 的相同子类型，并且是一个连续的 1-D 数组，形状为 `(a.size,)`
        return y
    # 如果输入的参数 a 是 numpy 矩阵类型，则将其转换为数组类型，并按指定的顺序展平
    if isinstance(a, np.matrix):
        return asarray(a).ravel(order=order)
    else:
        # 如果输入的参数 a 不是 numpy 矩阵类型，则将其转换为任意数组类型，并按指定的顺序展平
        return asanyarray(a).ravel(order=order)
# 定义一个函数 _nonzero_dispatcher，接受一个参数 a，返回包含 a 的元组
def _nonzero_dispatcher(a):
    return (a,)

# 使用 array_function_dispatch 装饰器，将 _nonzero_dispatcher 与 nonzero 函数关联起来
@array_function_dispatch(_nonzero_dispatcher)
# 定义函数 nonzero，返回非零元素的索引
def nonzero(a):
    """
    Return the indices of the elements that are non-zero.

    Returns a tuple of arrays, one for each dimension of `a`,
    containing the indices of the non-zero elements in that
    dimension. The values in `a` are always tested and returned in
    row-major, C-style order.

    To group the indices by element, rather than dimension, use `argwhere`,
    which returns a row for each non-zero element.

    .. note::

       When called on a zero-d array or scalar, ``nonzero(a)`` is treated
       as ``nonzero(atleast_1d(a))``.

       .. deprecated:: 1.17.0

          Use `atleast_1d` explicitly if this behavior is deliberate.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    tuple_of_arrays : tuple
        Indices of elements that are non-zero.

    See Also
    --------
    flatnonzero :
        Return indices that are non-zero in the flattened version of the input
        array.
    ndarray.nonzero :
        Equivalent ndarray method.
    count_nonzero :
        Counts the number of non-zero elements in the input array.

    Notes
    -----
    While the nonzero values can be obtained with ``a[nonzero(a)]``, it is
    recommended to use ``x[x.astype(bool)]`` or ``x[x != 0]`` instead, which
    will correctly handle 0-d arrays.

    Examples
    --------
    >>> x = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
    >>> x
    array([[3, 0, 0],
           [0, 4, 0],
           [5, 6, 0]])
    >>> np.nonzero(x)
    (array([0, 1, 2, 2]), array([0, 1, 0, 1]))

    >>> x[np.nonzero(x)]
    array([3, 4, 5, 6])
    >>> np.transpose(np.nonzero(x))
    array([[0, 0],
           [1, 1],
           [2, 0],
           [2, 1]])

    A common use for ``nonzero`` is to find the indices of an array, where
    a condition is True.  Given an array `a`, the condition `a` > 3 is a
    boolean array and since False is interpreted as 0, np.nonzero(a > 3)
    yields the indices of the `a` where the condition is true.

    >>> a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> a > 3
    array([[False, False, False],
           [ True,  True,  True],
           [ True,  True,  True]])
    >>> np.nonzero(a > 3)
    (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))

    Using this result to index `a` is equivalent to using the mask directly:

    >>> a[np.nonzero(a > 3)]
    array([4, 5, 6, 7, 8, 9])
    >>> a[a > 3]  # prefer this spelling
    array([4, 5, 6, 7, 8, 9])

    ``nonzero`` can also be called as a method of the array.

    >>> (a > 3).nonzero()
    (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))

    """
    # 调用 _wrapfunc 函数，传入参数 a 和字符串 'nonzero'，返回结果
    return _wrapfunc(a, 'nonzero')


# 定义一个函数 _shape_dispatcher，接受一个参数 a，返回包含 a 的元组
def _shape_dispatcher(a):
    return (a,)

# 使用 array_function_dispatch 装饰器，将 _shape_dispatcher 与 shape 函数关联起来
@array_function_dispatch(_shape_dispatcher)
# 定义函数 shape，返回数组的形状
def shape(a):
    """
    Return the shape of an array.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    """
    shape : tuple of ints
        The elements of the shape tuple give the lengths of the
        corresponding array dimensions.

    See Also
    --------
    len : ``len(a)`` is equivalent to ``np.shape(a)[0]`` for N-D arrays with
          ``N>=1``.
    ndarray.shape : Equivalent array method.

    Examples
    --------
    >>> np.shape(np.eye(3))
    (3, 3)
    >>> np.shape([[1, 3]])
    (1, 2)
    >>> np.shape([0])
    (1,)
    >>> np.shape(0)
    ()

    >>> a = np.array([(1, 2), (3, 4), (5, 6)],
    ...              dtype=[('x', 'i4'), ('y', 'i4')])
    >>> np.shape(a)
    (3,)
    >>> a.shape
    (3,)

    """
    try:
        # 尝试获取数组 a 的形状
        result = a.shape
    except AttributeError:
        # 若 a 没有 shape 属性，则将 a 转换为 ndarray 后再获取其形状
        result = asarray(a).shape
    # 返回获取到的形状结果
    return result
# 定义一个函数 _compress_dispatcher，返回一个包含条件、数组及输出的元组
def _compress_dispatcher(condition, a, axis=None, out=None):
    return (condition, a, out)

# 使用 array_function_dispatch 装饰器将 compress 函数与 _compress_dispatcher 函数关联
@array_function_dispatch(_compress_dispatcher)
def compress(condition, a, axis=None, out=None):
    """
    返回数组沿指定轴的选定片段。

    在沿指定轴工作时，对于每个条件为 True 的索引，从 `a` 中返回一个 `output` 切片。
    在处理 1-D 数组时，`compress` 等效于 `extract`。

    参数
    ----------
    condition : 1-D 布尔数组
        选择要返回的条目的数组。如果 len(condition) 小于给定轴上 `a` 的大小，则输出被截断为条件数组的长度。
    a : array_like
        要从中提取部分的数组。
    axis : int, optional
        沿其获取切片的轴。如果为 None（默认），则在展平数组上工作。
    out : ndarray, optional
        输出数组。其类型被保留，必须具有正确的形状以容纳输出。

    返回
    -------
    compressed_array : ndarray
        `a` 的副本，删除了在轴上条件为 False 的切片。

    另请参见
    --------
    take, choose, diag, diagonal, select
    ndarray.compress : 数组中的等效方法
    extract : 在处理 1-D 数组时的等效方法
    :ref:`ufuncs-output-type`

    示例
    --------
    >>> a = np.array([[1, 2], [3, 4], [5, 6]])
    >>> a
    array([[1, 2],
           [3, 4],
           [5, 6]])
    >>> np.compress([0, 1], a, axis=0)
    array([[3, 4]])
    >>> np.compress([False, True, True], a, axis=0)
    array([[3, 4],
           [5, 6]])
    >>> np.compress([False, True], a, axis=1)
    array([[2],
           [4],
           [6]])

    在展平数组上工作不会返回沿轴的切片，而是选择元素。

    >>> np.compress([False, True], a)
    array([2])

    """

# 定义一个函数 _clip_dispatcher，返回一个包含数组、最小值、最大值及输出的元组
def _clip_dispatcher(a, a_min, a_max, out=None, **kwargs):
    return (a, a_min, a_max)

# 使用 array_function_dispatch 装饰器将 clip 函数与 _clip_dispatcher 函数关联
@array_function_dispatch(_clip_dispatcher)
def clip(a, a_min, a_max, out=None, **kwargs):
    """
    对数组中的值进行裁剪（限制）。

    给定一个区间，将超出该区间的值裁剪到区间的边缘。例如，如果指定区间为 ``[0, 1]``，
    小于 0 的值变为 0，大于 1 的值变为 1。

    等效于但比 ``np.minimum(a_max, np.maximum(a, a_min))`` 更快。

    不执行任何检查以确保 ``a_min < a_max``。

    参数
    ----------
    a : array_like
        包含要裁剪元素的数组。
    a_min, a_max : array_like 或 None
        最小值和最大值。如果为 ``None``，则不在相应边缘执行裁剪。`a_min` 和 `a_max` 中只能有一个为 ``None``。两者与 `a` 广播。

    """
    out : ndarray, optional
        结果将被放置在这个数组中。它可以是用于原地裁剪的输入数组。`out` 必须具有正确的形状来容纳输出。其类型将被保留。
    **kwargs
        其他关键字参数，请参阅:ref:`ufunc docs <ufuncs.kwargs>`。

        .. versionadded:: 1.17.0

    Returns
    -------
    clipped_array : ndarray
        元素为 `a` 的数组，但其中小于 `a_min` 的值被替换为 `a_min`，大于 `a_max` 的值被替换为 `a_max`。

    See Also
    --------
    :ref:`ufuncs-output-type`

    Notes
    -----
    当 `a_min` 大于 `a_max` 时，`clip` 返回一个数组，其中所有值都等于 `a_max`，如第二个示例所示。

    Examples
    --------
    >>> a = np.arange(10)
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> np.clip(a, 1, 8)
    array([1, 1, 2, 3, 4, 5, 6, 7, 8, 8])
    >>> np.clip(a, 8, 1)
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    >>> np.clip(a, 3, 6, out=a)
    array([3, 3, 3, 3, 4, 5, 6, 6, 6, 6])
    >>> a
    array([3, 3, 3, 3, 4, 5, 6, 6, 6, 6])
    >>> a = np.arange(10)
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> np.clip(a, [3, 4, 1, 1, 1, 4, 4, 4, 4, 4], 8)
    array([3, 4, 2, 3, 4, 5, 6, 7, 8, 8])

    """
    return _wrapfunc(a, 'clip', a_min, a_max, out=out, **kwargs)
# 定义一个分派函数 `_sum_dispatcher`，用于调度参数 `a`，返回一个元组 `(a, out)`
def _sum_dispatcher(a, axis=None, dtype=None, out=None, keepdims=None,
                    initial=None, where=None):
    return (a, out)

# 装饰器 `array_function_dispatch` 用于将 `_sum_dispatcher` 与函数 `sum` 关联
@array_function_dispatch(_sum_dispatcher)
def sum(a, axis=None, dtype=None, out=None, keepdims=np._NoValue,
        initial=np._NoValue, where=np._NoValue):
    """
    Sum of array elements over a given axis.

    Parameters
    ----------
    a : array_like
        Elements to sum.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a sum is performed.  The default,
        axis=None, will sum all of the elements of the input array.  If
        axis is negative it counts from the last to the first axis.

        .. versionadded:: 1.7.0

        If axis is a tuple of ints, a sum is performed on all of the axes
        specified in the tuple instead of a single axis or all the axes as
        before.
    dtype : dtype, optional
        The type of the returned array and of the accumulator in which the
        elements are summed.  The dtype of `a` is used by default unless `a`
        has an integer dtype of less precision than the default platform
        integer.  In that case, if `a` is signed then the platform integer
        is used while if `a` is unsigned then an unsigned integer of the
        same precision as the platform integer is used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output, but the type of the output
        values will be cast if necessary.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `sum` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.
    initial : scalar, optional
        Starting value for the sum. See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.15.0

    where : array_like of bool, optional
        Elements to include in the sum. See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.17.0

    Returns
    -------
    sum_along_axis : ndarray
        An array with the same shape as `a`, with the specified
        axis removed.   If `a` is a 0-d array, or if `axis` is None, a scalar
        is returned.  If an output array is specified, a reference to
        `out` is returned.

    See Also
    --------
    ndarray.sum : Equivalent method.
    add: ``numpy.add.reduce`` equivalent function.
    cumsum : Cumulative sum of array elements.
    trapezoid : Integration of array values using composite trapezoidal rule.

    mean, average

    Notes
    -----
    """
    # 函数 `sum` 实现了对数组元素沿指定轴的求和操作
    # 具体参数解释详见上方的 docstring 文档
    pass  # 这里是函数体的占位符，实际上没有实现其他功能
    if isinstance(a, _gentype):
        # 如果参数 a 是 _gentype 类型的实例
        warnings.warn(
            # 发出警告，提示调用 np.sum(generator) 已被弃用，并且未来会产生不同的结果。
            # 建议使用 np.sum(np.fromiter(generator)) 或者 Python 的内置 sum 函数代替。
            "Calling np.sum(generator) is deprecated, and in the future will "
            "give a different result. Use np.sum(np.fromiter(generator)) or "
            "the python sum builtin instead.",
            DeprecationWarning, stacklevel=2
        )

        # 使用 _sum_ 函数对参数 a 进行求和
        res = _sum_(a)
        if out is not None:
            # 如果指定了输出数组 out，则将结果 res 复制给 out
            out[...] = res
            return out
        # 如果没有指定输出数组 out，则直接返回结果 res
        return res

    # 调用 _wrapreduction 函数进行归约操作
    return _wrapreduction(
        a, np.add, 'sum', axis, dtype, out,
        keepdims=keepdims, initial=initial, where=where
    )
# 定义了一个私有函数 `_any_dispatcher`，用于分发参数 `a`, `axis`, `out`, `keepdims`, `where` 到元组中
def _any_dispatcher(a, axis=None, out=None, keepdims=None, *, where=np._NoValue):
    # 返回元组 `(a, where, out)`，将输入参数 `a`, `where`, `out` 打包成元组返回
    return (a, where, out)

# 使用装饰器 `array_function_dispatch` 装饰的函数 `any`，用于测试数组在指定轴向上是否有任意元素为 `True`
@array_function_dispatch(_any_dispatcher)
def any(a, axis=None, out=None, keepdims=np._NoValue, *, where=np._NoValue):
    """
    Test whether any array element along a given axis evaluates to True.

    Returns single boolean if `axis` is ``None``

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a logical OR reduction is performed.
        The default (``axis=None``) is to perform a logical OR over all
        the dimensions of the input array. `axis` may be negative, in
        which case it counts from the last to the first axis.

        .. versionadded:: 1.7.0

        If this is a tuple of ints, a reduction is performed on multiple
        axes, instead of a single axis or all the axes as before.
    out : ndarray, optional
        Alternate output array in which to place the result.  It must have
        the same shape as the expected output and its type is preserved
        (e.g., if it is of type float, then it will remain so, returning
        1.0 for True and 0.0 for False, regardless of the type of `a`).
        See :ref:`ufuncs-output-type` for more details.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `any` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.
    where : array_like of bool, optional
        Elements to include in checking for any `True` values.
        See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.20.0

    Returns
    -------
    any : bool or ndarray
        A new boolean or `ndarray` is returned unless `out` is specified,
        in which case a reference to `out` is returned.

    See Also
    --------
    ndarray.any : equivalent method

    all : Test whether all elements along a given axis evaluate to True.

    Notes
    -----
    Not a Number (NaN), positive infinity and negative infinity evaluate
    to `True` because these are not equal to zero.

    .. versionchanged:: 2.0
       Before NumPy 2.0, ``any`` did not return booleans for object dtype
       input arrays.
       This behavior is still available via ``np.logical_or.reduce``.

    Examples
    --------
    >>> np.any([[True, False], [True, True]])
    True

    >>> np.any([[True,  False, True ],
    ...         [False, False, False]], axis=0)
    array([ True, False, True])

    >>> np.any([-1, 0, 5])
    True
    """
    >>> np.any([[np.nan], [np.inf]], axis=1, keepdims=True)
    array([[ True],
           [ True]])

# 在给定的二维数组中，沿着指定的轴（axis=1，即按行）检查是否存在任何非零元素。
# 返回一个布尔数组，表示每行中是否至少存在一个非零元素。

    >>> np.any([[True, False], [False, False]], where=[[False], [True]])
    False

# 在给定的二维数组中，使用where参数指定条件，检查是否存在任何非零元素。
# 返回False，因为第一行的where条件为False，第二行无非零元素。

    >>> a = np.array([[1, 0, 0],
    ...               [0, 0, 1],
    ...               [0, 0, 0]])
    >>> np.any(a, axis=0)
    array([ True, False,  True])
    >>> np.any(a, axis=1)
    array([ True,  True, False])

# 创建一个3x3的NumPy数组a，表示一个稀疏的矩阵。
# np.any(a, axis=0) 沿着列轴（axis=0）检查数组a中是否存在任何非零元素。
# 返回一个布尔数组，每个元素表示对应列中是否至少存在一个非零元素。
# np.any(a, axis=1) 沿着行轴（axis=1）检查数组a中是否存在任何非零元素。
# 返回一个布尔数组，每个元素表示对应行中是否至少存在一个非零元素。

    >>> o=np.array(False)
    >>> z=np.any([-1, 4, 5], out=o)
    >>> z, o
    (array(True), array(True))

# 创建一个布尔数组o，初始值为False。
# 使用np.any函数检查[-1, 4, 5]数组中是否存在任何非零元素，并将结果存储在z变量中，同时更新布尔数组o。
# 返回z为True（因为数组中有非零元素），o也为True。

    >>> # Check now that z is a reference to o
    >>> z is o
    True

# 检查变量z和o是否引用了相同的对象（即它们是否是同一个对象）。
# 返回True，表明z和o指向同一个布尔数组对象。

    >>> id(z), id(o) # identity of z and o              # doctest: +SKIP
    (191614240, 191614240)

# 获取变量z和o的内存地址。
# 返回它们的内存地址，这里的具体地址数字可能会因为环境不同而有所变化。
# （注：由于文档测试模式被跳过，这段代码不会在文档测试中执行。）

    """
    return _wrapreduction_any_all(a, np.logical_or, 'any', axis, out,
                                  keepdims=keepdims, where=where)

# 返回调用_wrapreduction_any_all函数的结果，该函数用于对数组a进行逻辑或（logical_or）操作，
# 在指定轴上检查是否有任何元素满足条件，可指定输出out、保持维度keepdims和where条件。
# 定义一个调度器函数，用于决定如何分派参数 `a`, `axis`, `out`, `keepdims`, `where`
def _all_dispatcher(a, axis=None, out=None, keepdims=None, *, where=None):
    # 返回参数 `a`, `where`, `out` 的元组
    return (a, where, out)


# 使用 array_function_dispatch 装饰器将 _all_dispatcher 函数与 all 函数关联起来
@array_function_dispatch(_all_dispatcher)
def all(a, axis=None, out=None, keepdims=np._NoValue, *, where=np._NoValue):
    """
    Test whether all array elements along a given axis evaluate to True.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a logical AND reduction is performed.
        The default (``axis=None``) is to perform a logical AND over all
        the dimensions of the input array. `axis` may be negative, in
        which case it counts from the last to the first axis.

        .. versionadded:: 1.7.0

        If this is a tuple of ints, a reduction is performed on multiple
        axes, instead of a single axis or all the axes as before.
    out : ndarray, optional
        Alternate output array in which to place the result.
        It must have the same shape as the expected output and its
        type is preserved (e.g., if ``dtype(out)`` is float, the result
        will consist of 0.0's and 1.0's). See :ref:`ufuncs-output-type`
        for more details.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `all` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.
    where : array_like of bool, optional
        Elements to include in checking for all `True` values.
        See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.20.0

    Returns
    -------
    all : ndarray, bool
        A new boolean or array is returned unless `out` is specified,
        in which case a reference to `out` is returned.

    See Also
    --------
    ndarray.all : equivalent method

    any : Test whether any element along a given axis evaluates to True.

    Notes
    -----
    Not a Number (NaN), positive infinity and negative infinity
    evaluate to `True` because these are not equal to zero.

    .. versionchanged:: 2.0
       Before NumPy 2.0, ``all`` did not return booleans for object dtype
       input arrays.
       This behavior is still available via ``np.logical_and.reduce``.

    Examples
    --------
    >>> np.all([[True,False],[True,True]])
    False

    >>> np.all([[True,False],[True,True]], axis=0)
    array([ True, False])

    >>> np.all([-1, 4, 5])
    True

    >>> np.all([1.0, np.nan])
    True

    >>> np.all([[True, True], [False, True]], where=[[True], [False]])
    True

    >>> o=np.array(False)
    """
    # 函数体中不需要添加额外的注释，因为文档字符串已经充分解释了函数的功能和用法
    # 调用 NumPy 的 all 函数，检查数组中所有元素是否都为真
    >>> z=np.all([-1, 4, 5], out=o)
    # 获取 z 和 o 对象的内存地址，并输出 z 的值
    >>> id(z), id(o), z
    # 输出结果显示 z 和 o 对象的内存地址以及 z 的值为数组中所有元素是否都为真的结果
    (28293632, 28293632, array(True)) # may vary
    
    """
    # 返回调用 _wrapreduction_any_all 函数的结果，执行数组的逻辑与操作，用于 'all' 操作
    return _wrapreduction_any_all(a, np.logical_and, 'all', axis, out,
                                  keepdims=keepdims, where=where)
# 定义一个分派器函数，用于 `cumsum` 函数，将传入的参数 `a`、`axis`、`dtype` 和 `out` 包装成元组返回
def _cumsum_dispatcher(a, axis=None, dtype=None, out=None):
    return (a, out)


# 使用 `array_function_dispatch` 装饰器标记的 `cumsum` 函数，实现了累积求和操作
@array_function_dispatch(_cumsum_dispatcher)
def cumsum(a, axis=None, dtype=None, out=None):
    """
    Return the cumulative sum of the elements along a given axis.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        Axis along which the cumulative sum is computed. The default
        (None) is to compute the cumsum over the flattened array.
    dtype : dtype, optional
        Type of the returned array and of the accumulator in which the
        elements are summed.  If `dtype` is not specified, it defaults
        to the dtype of `a`, unless `a` has an integer dtype with a
        precision less than that of the default platform integer.  In
        that case, the default platform integer is used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type will be cast if necessary. See :ref:`ufuncs-output-type`
        for more details.

    Returns
    -------
    cumsum_along_axis : ndarray.
        A new array holding the result is returned unless `out` is
        specified, in which case a reference to `out` is returned. The
        result has the same size as `a`, and the same shape as `a` if
        `axis` is not None or `a` is a 1-d array.

    See Also
    --------
    sum : Sum array elements.
    trapezoid : Integration of array values using composite trapezoidal rule.
    diff : Calculate the n-th discrete difference along given axis.

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.

    ``cumsum(a)[-1]`` may not be equal to ``sum(a)`` for floating-point
    values since ``sum`` may use a pairwise summation routine, reducing
    the roundoff-error. See `sum` for more information.

    Examples
    --------
    >>> a = np.array([[1,2,3], [4,5,6]])
    >>> a
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> np.cumsum(a)
    array([ 1,  3,  6, 10, 15, 21])
    >>> np.cumsum(a, dtype=float)     # specifies type of output value(s)
    array([  1.,   3.,   6.,  10.,  15.,  21.])

    >>> np.cumsum(a,axis=0)      # sum over rows for each of the 3 columns
    array([[1, 2, 3],
           [5, 7, 9]])
    >>> np.cumsum(a,axis=1)      # sum over columns for each of the 2 rows
    array([[ 1,  3,  6],
           [ 4,  9, 15]])

    ``cumsum(b)[-1]`` may not be equal to ``sum(b)``

    >>> b = np.array([1, 2e-9, 3e-9] * 1000000)
    >>> b.cumsum()[-1]
    1000000.0050045159
    >>> b.sum()
    1000000.0050000029

    """
    # 调用 `_wrapfunc` 函数，传递参数 `a`, 'cumsum', `axis`, `dtype`, `out`，并返回其结果
    return _wrapfunc(a, 'cumsum', axis=axis, dtype=dtype, out=out)


# 定义一个分派器函数，用于 `ptp` 函数，将传入的参数 `a`, `axis`, `out`, `keepdims` 包装成元组返回
def _ptp_dispatcher(a, axis=None, out=None, keepdims=None):
    return (a, out)


# 使用 `array_function_dispatch` 装饰器标记的 `ptp` 函数，封装了计算数组中元素沿指定轴的峰值到峰值的范围的操作
@array_function_dispatch(_ptp_dispatcher)
def ptp(a, axis=None, out=None, keepdims=np._NoValue):
    """
    # 创建一个空字典用于存储参数
    kwargs = {}
    # 如果 keepdims 参数不是默认值 np._NoValue，则将其加入 kwargs 字典中
    if keepdims is not np._NoValue:
        kwargs['keepdims'] = keepdims
    # 调用底层的 _ptp 方法来计算给定数组的峰值到峰值范围
    return _methods._ptp(a, axis=axis, out=out, **kwargs)
# 定义一个分发器函数 `_max_dispatcher`，用于选择正确的输入参数和输出参数
def _max_dispatcher(a, axis=None, out=None, keepdims=None, initial=None,
                    where=None):
    # 返回输入数组 `a` 和输出数组 `out`
    return (a, out)

# 使用 `array_function_dispatch` 装饰器注册 `_max_dispatcher` 函数
# 设置模块为 `numpy`
@array_function_dispatch(_max_dispatcher)
@set_module('numpy')
# 定义 `max` 函数，计算数组或沿着指定轴的最大值
def max(a, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue,
        where=np._NoValue):
    """
    Return the maximum of an array or maximum along an axis.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate.  By default, flattened input is
        used.

        .. versionadded:: 1.7.0

        If this is a tuple of ints, the maximum is selected over multiple axes,
        instead of a single axis or all the axes as before.
    out : ndarray, optional
        Alternative output array in which to place the result.  Must
        be of the same shape and buffer length as the expected output.
        See :ref:`ufuncs-output-type` for more details.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the ``max`` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    initial : scalar, optional
        The minimum value of an output element. Must be present to allow
        computation on empty slice. See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.15.0

    where : array_like of bool, optional
        Elements to compare for the maximum. See `~numpy.ufunc.reduce`
        for details.

        .. versionadded:: 1.17.0

    Returns
    -------
    max : ndarray or scalar
        Maximum of `a`. If `axis` is None, the result is a scalar value.
        If `axis` is an int, the result is an array of dimension
        ``a.ndim - 1``. If `axis` is a tuple, the result is an array of
        dimension ``a.ndim - len(axis)``.

    See Also
    --------
    amin :
        The minimum value of an array along a given axis, propagating any NaNs.
    nanmax :
        The maximum value of an array along a given axis, ignoring any NaNs.
    maximum :
        Element-wise maximum of two arrays, propagating any NaNs.
    fmax :
        Element-wise maximum of two arrays, ignoring any NaNs.
    argmax :
        Return the indices of the maximum values.

    nanmin, minimum, fmin

    Notes
    -----
    NaN values are propagated, that is if at least one item is NaN, the
    corresponding max value will be NaN as well. To ignore NaN values
    (MATLAB behavior), please use nanmax.

    Don't use `~numpy.max` for element-wise comparison of 2 arrays; when
    ``a.shape[0]`` is 2, ``maximum(a[0], a[1])`` is faster than
    """
    # 函数文档字符串描述了 `max` 函数的输入参数和返回值
    # 返回数组 `a` 沿指定轴的最大值。
    
    return _wrapreduction(a, np.maximum, 'max', axis, None, out,
                          keepdims=keepdims, initial=initial, where=where)
    
    
    这段代码是 NumPy 中的 `np.max` 函数的核心实现。它调用了 `_wrapreduction` 函数，用于执行数组 `a` 的归约操作，具体功能包括：
    
    - `a`: 要计算最大值的输入数组。
    - `np.maximum`: 归约函数，用于比较两个元素并返回较大值的函数。
    - `'max'`: 字符串标识，表示进行最大值计算。
    - `axis`: 指定的轴，沿着该轴计算最大值。
    - `None`: 在此处没有使用。
    - `out`: 可选的输出数组，用于存放结果。
    - `keepdims`: 布尔值，指示是否保持归约操作后的维度。
    - `initial`: 可选的初始值，在进行归约操作时使用，用于处理空切片或特定条件下的最大值计算。
    - `where`: 可选的条件数组，用于指定进行归约操作的元素范围。
    
    这段代码对应的注释简明扼要地描述了函数的参数和功能，确保了读者能够快速理解代码的作用和用法。
@array_function_dispatch(_max_dispatcher)
def amax(a, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue,
         where=np._NoValue):
    """
    Return the maximum of an array or maximum along an axis.

    `amax` is an alias of `~numpy.max`.

    See Also
    --------
    max : alias of this function
    ndarray.max : equivalent method
    """
    # 使用 _wrapreduction 函数对数组进行约简操作，使用 np.maximum 函数找到数组的最大值
    return _wrapreduction(a, np.maximum, 'max', axis, None, out,
                          keepdims=keepdims, initial=initial, where=where)


def _min_dispatcher(a, axis=None, out=None, keepdims=None, initial=None,
                    where=None):
    # 返回输入参数 a 和 out
    return (a, out)


@array_function_dispatch(_min_dispatcher)
def min(a, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue,
        where=np._NoValue):
    """
    Return the minimum of an array or minimum along an axis.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate.  By default, flattened input is
        used.

        .. versionadded:: 1.7.0

        If this is a tuple of ints, the minimum is selected over multiple axes,
        instead of a single axis or all the axes as before.
    out : ndarray, optional
        Alternative output array in which to place the result.  Must
        be of the same shape and buffer length as the expected output.
        See :ref:`ufuncs-output-type` for more details.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the ``min`` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    initial : scalar, optional
        The maximum value of an output element. Must be present to allow
        computation on empty slice. See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.15.0

    where : array_like of bool, optional
        Elements to compare for the minimum. See `~numpy.ufunc.reduce`
        for details.

        .. versionadded:: 1.17.0

    Returns
    -------
    min : ndarray or scalar
        Minimum of `a`. If `axis` is None, the result is a scalar value.
        If `axis` is an int, the result is an array of dimension
        ``a.ndim - 1``.  If `axis` is a tuple, the result is an array of
        dimension ``a.ndim - len(axis)``.

    See Also
    --------
    amax :
        The maximum value of an array along a given axis, propagating any NaNs.
    nanmin :
        The minimum value of an array along a given axis, ignoring any NaNs.
    minimum :
        Element-wise minimum of two arrays, propagating any NaNs.
    """
    # 使用 _wrapreduction 函数对数组进行约简操作，使用 np.minimum 函数找到数组的最小值
    return _wrapreduction(a, np.minimum, 'min', axis, None, out,
                          keepdims=keepdims, initial=initial, where=where)
    fmin :
        # 计算两个数组的逐元素最小值，忽略任何 NaN 值。
    argmin :
        # 返回最小值的索引数组。

    nanmax, maximum, fmax

    Notes
    -----
    # NaN 值会被传播，即如果至少有一个元素是 NaN，则对应的最小值也将是 NaN。要忽略 NaN 值（类似 MATLAB 的行为），请使用 nanmin。

    Don't use `~numpy.min` for element-wise comparison of 2 arrays; when
    ``a.shape[0]`` is 2, ``minimum(a[0], a[1])`` is faster than
    ``min(a, axis=0)``.
    # 不要使用 `~numpy.min` 进行两个数组的逐元素比较；当 ``a.shape[0]`` 为 2 时，``minimum(a[0], a[1])`` 比 ``min(a, axis=0)`` 更快。

    Examples
    --------
    >>> a = np.arange(4).reshape((2,2))
    >>> a
    array([[0, 1],
           [2, 3]])
    >>> np.min(a)           # Minimum of the flattened array
    0
    # 扁平化数组的最小值
    >>> np.min(a, axis=0)   # Minima along the first axis
    array([0, 1])
    # 沿第一个轴的最小值
    >>> np.min(a, axis=1)   # Minima along the second axis
    array([0, 2])
    # 沿第二个轴的最小值
    >>> np.min(a, where=[False, True], initial=10, axis=0)
    array([10,  1])
    # 在指定条件下沿第零轴的最小值

    >>> b = np.arange(5, dtype=float)
    >>> b[2] = np.nan
    >>> np.min(b)
    np.float64(nan)
    # 返回数组中的最小值，包括 NaN 值
    >>> np.min(b, where=~np.isnan(b), initial=10)
    0.0
    # 在忽略 NaN 值的情况下返回数组中的最小值
    >>> np.nanmin(b)
    0.0
    # 返回数组中的最小值，忽略 NaN 值

    >>> np.min([[-50], [10]], axis=-1, initial=0)
    array([-50,   0])
    # 沿最后一个轴的最小值，指定初始值为 0

    Notice that the initial value is used as one of the elements for which the
    minimum is determined, unlike for the default argument Python's max
    function, which is only used for empty iterables.
    # 注意，初始值将作为决定最小值的元素之一，与 Python 的 max 函数默认参数不同，后者仅用于空可迭代对象。

    Notice that this isn't the same as Python's ``default`` argument.
    # 请注意，这与 Python 的 ``default`` 参数不同。

    >>> np.min([6], initial=5)
    5
    # 返回数组的最小值，指定初始值为 5
    >>> min([6], default=5)
    6
    # 返回数组的最小值，使用默认参数 5
    """
    return _wrapreduction(a, np.minimum, 'min', axis, None, out,
                          keepdims=keepdims, initial=initial, where=where)
    # 调用内部函数 _wrapreduction，使用 np.minimum 函数计算数组 a 沿指定轴的最小值，并返回结果
# 使用装饰器 `_min_dispatcher` 将该函数注册为 `amin` 的分派函数
@array_function_dispatch(_min_dispatcher)
# 定义函数 `amin`，返回数组或沿着指定轴的最小值
def amin(a, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue,
         where=np._NoValue):
    """
    Return the minimum of an array or minimum along an axis.

    `amin` is an alias of `~numpy.min`.

    See Also
    --------
    min : alias of this function
    ndarray.min : equivalent method
    """
    # 调用 `_wrapreduction` 函数，对数组进行最小值规约操作
    return _wrapreduction(a, np.minimum, 'min', axis, None, out,
                          keepdims=keepdims, initial=initial, where=where)


# 使用 `_prod_dispatcher` 装饰器将该函数注册为 `prod` 的分派函数
@array_function_dispatch(_prod_dispatcher)
# 定义函数 `prod`，返回沿指定轴的数组元素的乘积
def prod(a, axis=None, dtype=None, out=None, keepdims=np._NoValue,
         initial=np._NoValue, where=np._NoValue):
    """
    Return the product of array elements over a given axis.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a product is performed.  The default,
        axis=None, will calculate the product of all the elements in the
        input array. If axis is negative it counts from the last to the
        first axis.

        .. versionadded:: 1.7.0

        If axis is a tuple of ints, a product is performed on all of the
        axes specified in the tuple instead of a single axis or all the
        axes as before.
    dtype : dtype, optional
        The type of the returned array, as well as of the accumulator in
        which the elements are multiplied.  The dtype of `a` is used by
        default unless `a` has an integer dtype of less precision than the
        default platform integer.  In that case, if `a` is signed then the
        platform integer is used while if `a` is unsigned then an unsigned
        integer of the same precision as the platform integer is used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output, but the type of the output
        values will be cast if necessary.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result
        will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `prod` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.
    initial : scalar, optional
        The starting value for this product. See `~numpy.ufunc.reduce`
        for details.

        .. versionadded:: 1.15.0

    where : array_like of bool, optional
        Elements to include in the product. See `~numpy.ufunc.reduce`
        for details.

        .. versionadded:: 1.17.0

    Returns
    -------
    product_along_axis : ndarray, see `dtype` parameter above.
        An array shaped as `a` but with the specified axis removed.
        Returns a reference to `out` if specified.

    See Also
    --------
    ndarray.prod : equivalent method
    :ref:`ufuncs-output-type`

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.  That means that, on a 32-bit platform:

    >>> x = np.array([536870910, 536870910, 536870910, 536870910])
    >>> np.prod(x)
    16 # may vary

    The product of an empty array is the neutral element 1:

    >>> np.prod([])
    1.0

    Examples
    --------
    By default, calculate the product of all elements:

    >>> np.prod([1.,2.])
    2.0

    Even when the input array is two-dimensional:

    >>> a = np.array([[1., 2.], [3., 4.]])
    >>> np.prod(a)
    24.0

    But we can also specify the axis over which to multiply:

    >>> np.prod(a, axis=1)
    array([  2.,  12.])
    >>> np.prod(a, axis=0)
    array([3., 8.])

    Or select specific elements to include:

    >>> np.prod([1., np.nan, 3.], where=[True, False, True])
    3.0

    If the type of `x` is unsigned, then the output type is
    the unsigned platform integer:

    >>> x = np.array([1, 2, 3], dtype=np.uint8)
    >>> np.prod(x).dtype == np.uint
    True

    If `x` is of a signed integer type, then the output type
    is the default platform integer:

    >>> x = np.array([1, 2, 3], dtype=np.int8)
    >>> np.prod(x).dtype == int
    True

    You can also start the product with a value other than one:

    >>> np.prod([1, 2], initial=5)
    10
    """
    return _wrapreduction(a, np.multiply, 'prod', axis, dtype, out,
                          keepdims=keepdims, initial=initial, where=where)


注释：


# 返回一个沿指定轴删除后的数组，其形状与输入数组 `a` 相同。如果指定了 `out`，则返回其引用。
# 定义一个分发器函数，返回输入参数和输出参数元组
def _cumprod_dispatcher(a, axis=None, dtype=None, out=None):
    return (a, out)


# 使用 array_function_dispatch 装饰器将 cumprod 函数与 _cumprod_dispatcher 分发器关联起来
@array_function_dispatch(_cumprod_dispatcher)
# 定义 cumprod 函数，计算沿指定轴的累积乘积
def cumprod(a, axis=None, dtype=None, out=None):
    """
    返回沿给定轴的元素的累积乘积。

    参数
    ----------
    a : array_like
        输入数组。
    axis : int, optional
        计算累积乘积的轴。默认情况下会展平输入。
    dtype : dtype, optional
        返回数组的类型，以及元素相乘时累加器的类型。如果未指定 *dtype*，则默认为 `a` 的 dtype，
        除非 `a` 具有低于默认平台整数精度的整数 dtype。在这种情况下，将使用默认平台整数。
    out : ndarray, optional
        替代输出数组，用于放置结果。它必须具有与预期输出相同的形状和缓冲区长度，但如果需要会强制转换结果值的类型。

    返回
    -------
    cumprod : ndarray
        返回一个新数组，其中包含结果，除非指定了 `out`，在这种情况下将返回对 `out` 的引用。
    """
    # 调用 _wrapfunc 函数，执行实际的累积乘积计算
    return _wrapfunc(a, 'cumprod', axis=axis, dtype=dtype, out=out)


# 定义一个分发器函数，返回输入参数元组
def _ndim_dispatcher(a):
    return (a,)


# 使用 array_function_dispatch 装饰器将 ndim 函数与 _ndim_dispatcher 分发器关联起来
@array_function_dispatch(_ndim_dispatcher)
# 定义 ndim 函数，返回数组的维数
def ndim(a):
    """
    返回数组的维数。

    参数
    ----------
    a : array_like
        输入数组。如果它还不是 ndarray，则尝试进行转换。

    返回
    -------
    number_of_dimensions : int
        `a` 的维数。标量为零维。

    示例
    --------
    >>> np.ndim([[1,2,3],[4,5,6]])
    2
    >>> np.ndim(np.array([[1,2,3],[4,5,6]]))
    2
    >>> np.ndim(1)
    0
    """
    try:
        # 尝试返回 a 的维数属性
        return a.ndim
    except AttributeError:
        # 如果 a 没有 ndim 属性，则将其转换为 ndarray 并返回其维数
        return asarray(a).ndim
    # 返回一个包含单个元素 a 的元组
    return (a,)
# 使用装饰器进行函数分派，将_size_dispatcher作为分派函数
@array_function_dispatch(_size_dispatcher)
def size(a, axis=None):
    """
    Return the number of elements along a given axis.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : int, optional
        Axis along which the elements are counted.  By default, give
        the total number of elements.

    Returns
    -------
    element_count : int
        Number of elements along the specified axis.

    See Also
    --------
    shape : dimensions of array
    ndarray.shape : dimensions of array
    ndarray.size : number of elements in array

    Examples
    --------
    >>> a = np.array([[1,2,3],[4,5,6]])
    >>> np.size(a)
    6
    >>> np.size(a,1)
    3
    >>> np.size(a,0)
    2

    """
    # 如果axis为None，返回数组a的元素总数
    if axis is None:
        try:
            return a.size
        except AttributeError:
            return asarray(a).size
    else:
        # 返回数组a在指定轴上的元素数
        try:
            return a.shape[axis]
        except AttributeError:
            return asarray(a).shape[axis]


# 使用装饰器进行函数分派，将_round_dispatcher作为分派函数
def _round_dispatcher(a, decimals=None, out=None):
    return (a, out)

@array_function_dispatch(_round_dispatcher)
def round(a, decimals=0, out=None):
    """
    Evenly round to the given number of decimals.

    Parameters
    ----------
    a : array_like
        Input data.
    decimals : int, optional
        Number of decimal places to round to (default: 0).  If
        decimals is negative, it specifies the number of positions to
        the left of the decimal point.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output, but the type of the output
        values will be cast if necessary. See :ref:`ufuncs-output-type`
        for more details.

    Returns
    -------
    rounded_array : ndarray
        An array of the same type as `a`, containing the rounded values.
        Unless `out` was specified, a new array is created.  A reference to
        the result is returned.

        The real and imaginary parts of complex numbers are rounded
        separately.  The result of rounding a float is a float.

    See Also
    --------
    ndarray.round : equivalent method
    around : an alias for this function
    ceil, fix, floor, rint, trunc


    Notes
    -----
    For values exactly halfway between rounded decimal values, NumPy
    rounds to the nearest even value. Thus 1.5 and 2.5 round to 2.0,
    -0.5 and 0.5 round to 0.0, etc.

    ``np.round`` uses a fast but sometimes inexact algorithm to round
    floating-point datatypes. For positive `decimals` it is equivalent to
    ``np.true_divide(np.rint(a * 10**decimals), 10**decimals)``, which has
    error due to the inexact representation of decimal fractions in the IEEE
    floating point standard [1]_ and errors introduced when scaling by powers
    of ten. For instance, note the extra "1" in the following:

        >>> np.round(56294995342131.5, 3)
        56294995342131.51


    """
    # 将输入数据a按指定小数位数decimals进行四舍五入
    # 将结果存储在输出数组out中，如果未指定out，则创建新数组
    # 返回与a相同类型的数组，包含四舍五入后的值的引用
    return np.around(a, decimals=decimals, out=out)
    # 使用 `_wrapfunc` 函数将输入数组 `a` 中的每个元素进行四舍五入处理，返回处理后的结果。
    def _round(a, decimals=0, out=None):
        # 调用 `_wrapfunc` 函数，将 `a` 数组中的元素进行四舍五入处理，指定小数位数为 `decimals`，输出到 `out` 中。
        return _wrapfunc(a, 'round', decimals=decimals, out=out)
# 使用装饰器将该函数注册为数组函数分发器，并指定调度器为 _round_dispatcher
@array_function_dispatch(_round_dispatcher)
def around(a, decimals=0, out=None):
    """
    Round an array to the given number of decimals.

    `around` is an alias of `~numpy.round`.

    See Also
    --------
    ndarray.round : equivalent method
    round : alias for this function
    ceil, fix, floor, rint, trunc

    """
    # 调用 _wrapfunc 函数，将 'round' 方法应用于数组 a，指定小数位数为 decimals，输出结果存储在 out 中
    return _wrapfunc(a, 'round', decimals=decimals, out=out)


# 定义 _mean_dispatcher 函数，用于分发计算均值的请求
def _mean_dispatcher(a, axis=None, dtype=None, out=None, keepdims=None, *,
                     where=None):
    return (a, where, out)


# 使用装饰器将 mean 函数注册为数组函数分发器，并指定调度器为 _mean_dispatcher
@array_function_dispatch(_mean_dispatcher)
def mean(a, axis=None, dtype=None, out=None, keepdims=np._NoValue, *,
         where=np._NoValue):
    """
    Compute the arithmetic mean along the specified axis.

    Returns the average of the array elements.  The average is taken over
    the flattened array by default, otherwise over the specified axis.
    `float64` intermediate and return values are used for integer inputs.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose mean is desired. If `a` is not an
        array, a conversion is attempted.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the means are computed. The default is to
        compute the mean of the flattened array.

        .. versionadded:: 1.7.0

        If this is a tuple of ints, a mean is performed over multiple axes,
        instead of a single axis or all the axes as before.
    dtype : data-type, optional
        Type to use in computing the mean.  For integer inputs, the default
        is `float64`; for floating point inputs, it is the same as the
        input dtype.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.
        See :ref:`ufuncs-output-type` for more details.
        See :ref:`ufuncs-output-type` for more details.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `mean` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    where : array_like of bool, optional
        Elements to include in the mean. See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.20.0

    Returns
    -------
    m : ndarray, see dtype parameter above
        If `out=None`, returns a new array containing the mean values,
        otherwise a reference to the output array is returned.

    See Also
    --------
    average : Weighted average
    std, var, nanmean, nanstd, nanvar

    """
    # 调用 _wrapfunc 函数，将 'mean' 方法应用于数组 a，指定轴 axis、数据类型 dtype、输出 out、keepdims 和 where 参数
    return _wrapfunc(a, 'mean', axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where)
    """
    Calculate the mean along a specified axis of an array `a`, considering optional parameters.
    
    Parameters
    ----------
    a : array_like
        Input array to compute the mean.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the means are computed. The default is to compute
        the mean of the flattened array.
    dtype : data-type, optional
        Type to use in computing the mean. For integer inputs, the default is
        `float64`; for floating-point inputs, it is the same as the input dtype.
    out : ndarray, optional
        Alternate output array in which to place the result. The default is `None`;
        if provided, it must have the same shape as the expected output, but the
        type will be cast if necessary.
    keepdims : bool, optional
        If this is set to `True`, the axes which are reduced are left in the result
        as dimensions with size one.
    where : array_like of bool, optional
        Elements to include in the mean calculation. Only elements that correspond
        to `True` in this boolean mask are included. If `where` is `None`, then all
        elements are included by default.
    
    Returns
    -------
    mean : ndarray
        If `a` is not a masked array, returns the mean of the array elements. If
        `a` is a masked array, returns the mean of the non-masked elements.
        The output data type depends on the input data type and the precision of
        the mean calculation.
    
    Raises
    ------
    TypeError
        If the input `a` is not an ndarray.
    
    See Also
    --------
    average : Weighted average.
    
    Notes
    -----
    The arithmetic mean is the sum of the elements along the axis divided
    by the number of elements.
    
    Note that for floating-point input, the mean is computed using the
    same precision the input has. Depending on the input data, this can
    cause the results to be inaccurate, especially for `float32` (see
    example below). Specifying a higher-precision accumulator using the
    `dtype` keyword can alleviate this issue.
    
    By default, `float16` results are computed using `float32` intermediates
    for extra precision.
    
    Examples
    --------
    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.mean(a)
    2.5
    >>> np.mean(a, axis=0)
    array([2., 3.])
    >>> np.mean(a, axis=1)
    array([1.5, 3.5])
    
    In single precision, `mean` can be inaccurate:
    
    >>> a = np.zeros((2, 512*512), dtype=np.float32)
    >>> a[0, :] = 1.0
    >>> a[1, :] = 0.1
    >>> np.mean(a)
    0.54999924
    
    Computing the mean in float64 is more accurate:
    
    >>> np.mean(a, dtype=np.float64)
    0.55000000074505806 # may vary
    
    Specifying a where argument:
    
    >>> a = np.array([[5, 9, 13], [14, 10, 12], [11, 15, 19]])
    >>> np.mean(a)
    12.0
    >>> np.mean(a, where=[[True], [False], [False]])
    9.0
    """
    kwargs = {}
    if keepdims is not np._NoValue:
        kwargs['keepdims'] = keepdims
    if where is not np._NoValue:
        kwargs['where'] = where
    if type(a) is not mu.ndarray:
        try:
            mean = a.mean
        except AttributeError:
            pass
        else:
            return mean(axis=axis, dtype=dtype, out=out, **kwargs)
    
    return _methods._mean(a, axis=axis, dtype=dtype,
                          out=out, **kwargs)
# 定义一个私有函数 _std_dispatcher，用于分发参数，并返回元组 (a, where, out, mean)
def _std_dispatcher(a, axis=None, dtype=None, out=None, ddof=None,
                    keepdims=None, *, where=None, mean=None, correction=None):
    return (a, where, out, mean)


# 使用装饰器 array_function_dispatch，将 _std_dispatcher 注册为 std 函数的分发器
@array_function_dispatch(_std_dispatcher)
# 定义 std 函数，计算沿指定轴的标准差
def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue, *,
        where=np._NoValue, mean=np._NoValue, correction=np._NoValue):
    r"""
    Compute the standard deviation along the specified axis.

    Returns the standard deviation, a measure of the spread of a distribution,
    of the array elements. The standard deviation is computed for the
    flattened array by default, otherwise over the specified axis.

    Parameters
    ----------
    a : array_like
        Calculate the standard deviation of these values.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the standard deviation is computed. The
        default is to compute the standard deviation of the flattened array.

        .. versionadded:: 1.7.0

        If this is a tuple of ints, a standard deviation is performed over
        multiple axes, instead of a single axis or all the axes as before.
    dtype : dtype, optional
        Type to use in computing the standard deviation. For arrays of
        integer type the default is float64, for arrays of float types it is
        the same as the array type.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type (of the calculated
        values) will be cast if necessary.
        See :ref:`ufuncs-output-type` for more details.
    ddof : {int, float}, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
        By default `ddof` is zero. See Notes for details about use of `ddof`.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `std` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.
    where : array_like of bool, optional
        Elements to include in the standard deviation.
        See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.20.0

    mean : array_like, optional
        Provide the mean to prevent its recalculation. The mean should have
        a shape as if it was calculated with ``keepdims=True``.
        The axis for the calculation of the mean should be the same as used in
        the call to this std function.

        .. versionadded:: 1.26.0
    correction : {int, float}, optional
        # 参数 correction 可以是 int 或 float 类型，可选的 Array API 兼容名称，用于参数 ddof。
        # 同一时间只能提供其中之一。

        .. versionadded:: 2.0.0
        # 添加于版本 2.0.0

    Returns
    -------
    standard_deviation : ndarray, see dtype parameter above.
        # 返回标准差的 ndarray，参见上述的 dtype 参数。
        # 如果 `out` 为 None，返回包含标准差的新数组；否则返回对输出数组的引用。

    See Also
    --------
    var, mean, nanmean, nanstd, nanvar
    :ref:`ufuncs-output-type`
        # 参见也有 var, mean, nanmean, nanstd, nanvar 和 :ref:`ufuncs-output-type`

    Notes
    -----
    There are several common variants of the array standard deviation
    calculation. Assuming the input `a` is a one-dimensional NumPy array
    and ``mean`` is either provided as an argument or computed as
    ``a.mean()``, NumPy computes the standard deviation of an array as::

        N = len(a)
        d2 = abs(a - mean)**2  # abs is for complex `a`
        var = d2.sum() / (N - ddof)  # note use of `ddof`
        std = var**0.5
        # 计算数组标准差的常见变体之一。假设输入 `a` 是一维 NumPy 数组，
        # `mean` 作为参数提供或计算为 `a.mean()`，NumPy 计算数组的标准差为：

    Different values of the argument `ddof` are useful in different
    contexts. NumPy's default ``ddof=0`` corresponds with the expression:

    .. math::

        \sqrt{\frac{\sum_i{|a_i - \bar{a}|^2 }}{N}}

    which is sometimes called the "population standard deviation" in the field
    of statistics because it applies the definition of standard deviation to
    `a` as if `a` were a complete population of possible observations.
    # 参数 `ddof` 的不同值在不同的上下文中很有用。NumPy 的默认值 `ddof=0`
    # 对应于表达式：

    .. math::

        \sqrt{\frac{\sum_i{|a_i - \bar{a}|^2 }}{N}}
        # 这在统计学领域有时被称为 "总体标准差"，因为它将标准差的定义应用于 `a`，
        # 就好像 `a` 是可能观察到的全部总体。

    Many other libraries define the standard deviation of an array
    differently, e.g.:

    .. math::

        \sqrt{\frac{\sum_i{|a_i - \bar{a}|^2 }}{N - 1}}
        # 许多其他库以不同的方式定义数组的标准差，例如：

    In statistics, the resulting quantity is sometimed called the "sample
    standard deviation" because if `a` is a random sample from a larger
    population, this calculation provides the square root of an unbiased
    estimate of the variance of the population. The use of :math:`N-1` in the
    denominator is often called "Bessel's correction" because it corrects for
    bias (toward lower values) in the variance estimate introduced when the
    sample mean of `a` is used in place of the true mean of the population.
    The resulting estimate of the standard deviation is still biased, but less
    than it would have been without the correction. For this quantity, use
    ``ddof=1``.
    # 在统计学中，由于 `a` 是来自更大总体的随机样本，该计算提供了总体方差的无偏估计的平方根，
    # 因此所得的数量有时被称为 "样本标准差"。在分母中使用 :math:`N-1` 通常被称为 "贝塞尔校正"，
    # 因为它校正了当使用 `a` 的样本均值代替总体真实均值时引入的方差估计偏差（向更小值偏移）。
    # 所得的标准差估计仍然是有偏的，但比没有校正时要小。对于这个量，使用 `ddof=1`。

    Note that, for complex numbers, `std` takes the absolute
    value before squaring, so that the result is always real and nonnegative.
    # 注意，对于复数，`std` 在平方之前取绝对值，以便结果始终是实数且非负数。

    For floating-point input, the standard deviation is computed using the same
    precision the input has. Depending on the input data, this can cause
    the results to be inaccurate, especially for float32 (see example below).
    Specifying a higher-accuracy accumulator using the `dtype` keyword can
    alleviate this issue.
    # 对于浮点数输入，标准差使用与输入相同的精度计算。根据输入数据，这可能导致结果不准确，
    # 特别是对于 float32（参见下面的示例）。使用 `dtype` 关键字指定更高精度的累加器可以缓解此问题。

    Examples
    --------
    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.std(a)
    1.1180339887498949 # may vary
    >>> np.std(a, axis=0)
    array([1.,  1.])
    >>> np.std(a, axis=1)
    array([0.5,  0.5])
    # 示例
    """
    Construct kwargs dictionary based on optional arguments for calculating standard deviation.
    
    :param a: Input array for which standard deviation is calculated.
    :param axis: Axis or axes along which the standard deviation is computed.
    :param dtype: Data type used in computing standard deviation.
    :param out: Output array.
    :param ddof: Delta degrees of freedom. The divisor used in calculations (default is 0).
    :param keepdims: If True, the axes which are reduced are left in the result as dimensions with size one.
    :param where: Array-like or boolean condition defining where the standard deviation should be computed.
    :param mean: Array of means to be used in standard deviation computation to save computation time.
    
    :return: Standard deviation of input array 'a' along specified axis/axes.
    """
    
    kwargs = {}
    
    # Check if keepdims is specified, add to kwargs if so
    if keepdims is not np._NoValue:
        kwargs['keepdims'] = keepdims
    
    # Check if where is specified, add to kwargs if so
    if where is not np._NoValue:
        kwargs['where'] = where
    
    # Check if mean is specified, add to kwargs if so
    if mean is not np._NoValue:
        kwargs['mean'] = mean
    
    # Handle correction and ddof arguments
    if correction != np._NoValue:
        # If correction is provided, ensure ddof is set accordingly
        if ddof != 0:
            raise ValueError("ddof and correction can't be provided simultaneously.")
        else:
            ddof = correction
    
    # Check if 'a' is not a numpy ndarray, try to retrieve std attribute
    if type(a) is not np.ndarray:
        try:
            std = a.std
        except AttributeError:
            pass
        else:
            # If 'std' attribute is found, call it with additional kwargs
            return std(axis=axis, dtype=dtype, out=out, ddof=ddof, **kwargs)
    
    # If 'a' is a numpy ndarray or 'std' attribute retrieval fails, use _methods._std function
    return _methods._std(a, axis=axis, dtype=dtype, out=out, ddof=ddof, **kwargs)
# 定义一个变量分派函数，用于返回参数元组 (a, where, out, mean)
def _var_dispatcher(a, axis=None, dtype=None, out=None, ddof=None,
                    keepdims=None, *, where=None, mean=None, correction=None):
    return (a, where, out, mean)

# 使用装饰器将 _var_dispatcher 函数注册为 var 函数的分派函数
@array_function_dispatch(_var_dispatcher)
# 定义计算方差的函数 var
def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue, *,
        where=np._NoValue, mean=np._NoValue, correction=np._NoValue):
    r"""
    Compute the variance along the specified axis.

    Returns the variance of the array elements, a measure of the spread of a
    distribution.  The variance is computed for the flattened array by
    default, otherwise over the specified axis.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose variance is desired.  If `a` is not an
        array, a conversion is attempted.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the variance is computed.  The default is to
        compute the variance of the flattened array.

        .. versionadded:: 1.7.0

        If this is a tuple of ints, a variance is performed over multiple axes,
        instead of a single axis or all the axes as before.
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
        ``N - ddof``, where ``N`` represents the number of elements. By
        default `ddof` is zero. See notes for details about use of `ddof`.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `var` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.
    where : array_like of bool, optional
        Elements to include in the variance. See `~numpy.ufunc.reduce` for
        details.

        .. versionadded:: 1.20.0

    mean : array like, optional
        Provide the mean to prevent its recalculation. The mean should have
        a shape as if it was calculated with ``keepdims=True``.
        The axis for the calculation of the mean should be the same as used in
        the call to this var function.

        .. versionadded:: 1.26.0
    correction : {int, float}, optional
        # 参数 correction 可以是 int 或 float 类型，可选的，用于指定 ddof 参数的数组 API 兼容名称。同一时间只能提供其中之一。

        .. versionadded:: 2.0.0
        # 引入版本 2.0.0 添加了此参数。

    Returns
    -------
    variance : ndarray, see dtype parameter above
        # 返回值是一个 ndarray 数组，具体类型可以参考上面的 dtype 参数说明。
        如果 ``out=None``，返回一个包含方差的新数组；
        否则，返回对输出数组的引用。

    See Also
    --------
    std, mean, nanmean, nanstd, nanvar
    :ref:`ufuncs-output-type`
        # 参见其他函数：std、mean、nanmean、nanstd、nanvar，以及 ufuncs 输出类型的文档。

    Notes
    -----
    # 注意事项：

    There are several common variants of the array variance calculation.
    # 数组方差计算有几种常见的变体。

    Assuming the input `a` is a one-dimensional NumPy array and ``mean`` is
    either provided as an argument or computed as ``a.mean()``, NumPy
    computes the variance of an array as::
        # 假设输入 `a` 是一个一维的 NumPy 数组，并且 `mean` 要么作为参数提供，要么通过 `a.mean()` 计算，NumPy 计算数组的方差如下所示：

        N = len(a)
        # 计算数组 `a` 的长度为 N
        d2 = abs(a - mean)**2  # abs is for complex `a`
        # 计算绝对值平方后的差值，对于复数 `a` 使用 abs 函数
        var = d2.sum() / (N - ddof)  # note use of `ddof`
        # 计算方差，注意使用 `ddof`

    Different values of the argument `ddof` are useful in different
    contexts. NumPy's default ``ddof=0`` corresponds with the expression:
        # 参数 `ddof` 的不同值在不同的情况下有不同的用处。NumPy 默认的 `ddof=0` 对应以下表达式：

    .. math::

        \frac{\sum_i{|a_i - \bar{a}|^2 }}{N}
        # 这个表达式有时在统计学中称为 "总体方差"，因为它将方差的定义应用于 `a`，就好像 `a` 是可能观察结果的完整总体。

    Many other libraries define the variance of an array differently, e.g.:
        # 许多其他库以不同的方式定义数组的方差，例如：

    .. math::

        \frac{\sum_i{|a_i - \bar{a}|^2}}{N - 1}
        # 在统计学中，这种结果有时称为 "样本方差"，因为如果 `a` 是来自更大总体的随机样本，则此计算提供了总体方差的无偏估计。分母使用 `N-1` 是为了校正偏差，因为使用 `a` 的样本均值代替总体真实均值时引入了方差估计的偏向（朝向较低的值）。这种校正通常称为 "贝塞尔校正"。

    Note that for complex numbers, the absolute value is taken before
    squaring, so that the result is always real and nonnegative.
        # 注意，对于复数，取绝对值后再进行平方，以确保结果始终是实数且非负数。

    For floating-point input, the variance is computed using the same
    precision the input has.  Depending on the input data, this can cause
    the results to be inaccurate, especially for `float32` (see example
    below).  Specifying a higher-accuracy accumulator using the ``dtype``
    keyword can alleviate this issue.
        # 对于浮点数输入，方差的计算精度与输入数据一致。根据输入数据的不同，这可能会导致结果不准确，特别是对于 `float32`（参见下面的示例）。可以通过指定更高精度的累加器来缓解这个问题，使用 `dtype` 关键字。

    Examples
    --------
    # 示例：

    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.var(a)
    1.25
        # 计算数组 `a` 的方差，结果为 1.25

    >>> np.var(a, axis=0)
    array([1.,  1.])
        # 沿着 axis=0 方向计算数组 `a` 的方差，结果为 [1., 1.]

    >>> np.var(a, axis=1)
    array([0.25,  0.25])
        # 沿着 axis=1 方向计算数组 `a` 的方差，结果为 [0.25, 0.25]

    In single precision, var() can be inaccurate:
        # 在单精度浮点数情况下，var() 可能不准确：

    >>> a = np.zeros((2, 512*512), dtype=np.float32)
    >>> a[0, :] = 1.0
    >>> a[1, :] = 0.1
    >>> np.var(a)
    0.20250003
        # 计算单精度下数组 `a` 的方差，结果为 0.20250003

    Computing the variance in float64 is more accurate:
        # 使用 float64 计算的方差更加精确：

    >>> np.var(a, dtype=np.float64)
    0.20249999932944759 # may vary
        # 使用 float64 数据类型计算数组 `a` 的方差，结果为 0.20249999932944759（可能有所变化）
    # 计算方差，使用公式 ((1-0.55)**2 + (0.1-0.55)**2)/2
    ((1-0.55)**2 + (0.1-0.55)**2)/2

    # 指定 where 参数的使用示例
    a = np.array([[14, 8, 11, 10], [7, 9, 10, 11], [10, 15, 5, 10]])
    # 计算数组 a 的方差
    np.var(a)
    # 输出结果可能会有所不同
    6.833333333333333 # may vary
    # 使用 where 参数限制计算的范围
    np.var(a, where=[[True], [True], [False]])
    # 返回结果 4.0

    # 使用 mean 关键字以节省计算时间
    import numpy as np
    from timeit import timeit

    a = np.array([[14, 8, 11, 10], [7, 9, 10, 11], [10, 15, 5, 10]])
    # 计算每行的平均值并保持维度
    mean = np.mean(a, axis=1, keepdims=True)

    g = globals()
    n = 10000
    # 使用 mean 关键字计算方差，并计算执行时间
    t1 = timeit("var = np.var(a, axis=1, mean=mean)", globals=g, number=n)
    # 普通计算方差的执行时间
    t2 = timeit("var = np.var(a, axis=1)", globals=g, number=n)
    # 打印节省的执行时间百分比
    print(f'Percentage execution time saved {100*(t2-t1)/t2:.0f}%')
    # doctest: +SKIP
    # 输出示例：Percentage execution time saved 32%

    """
    # 初始化一个空的关键字参数字典
    kwargs = {}
    # 如果 keepdims 不是 np._NoValue，则将其加入参数字典中
    if keepdims is not np._NoValue:
        kwargs['keepdims'] = keepdims
    # 如果 where 不是 np._NoValue，则将其加入参数字典中
    if where is not np._NoValue:
        kwargs['where'] = where
    # 如果 mean 不是 np._NoValue，则将其加入参数字典中
    if mean is not np._NoValue:
        kwargs['mean'] = mean

    # 如果 correction 不是 np._NoValue
    if correction != np._NoValue:
        # 如果 ddof 不为 0，则抛出 ValueError 异常
        if ddof != 0:
            raise ValueError(
                "ddof and correction can't be provided simultaneously."
            )
        else:
            # 否则将 correction 赋值给 ddof
            ddof = correction

    # 如果 a 的类型不是 mu.ndarray
    if type(a) is not mu.ndarray:
        try:
            # 尝试获取 a 的 var 属性
            var = a.var
        # 如果出现 AttributeError 异常则跳过
        except AttributeError:
            pass
        else:
            # 如果成功获取 var 属性，则调用其方法计算方差并返回结果
            return var(axis=axis, dtype=dtype, out=out, ddof=ddof, **kwargs)

    # 否则调用 _methods 模块中的 _var 方法计算方差并返回结果
    return _methods._var(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
                         **kwargs)
    ```
```