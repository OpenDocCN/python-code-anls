# `.\numpy\numpy\lib\_stride_tricks_impl.py`

```
"""
Utilities that manipulate strides to achieve desirable effects.

An explanation of strides can be found in the :ref:`arrays.ndarray`.

Functions
---------

.. autosummary::
   :toctree: generated/

"""
import numpy as np
from numpy._core.numeric import normalize_axis_tuple
from numpy._core.overrides import array_function_dispatch, set_module

__all__ = ['broadcast_to', 'broadcast_arrays', 'broadcast_shapes']


class DummyArray:
    """Dummy object that just exists to hang __array_interface__ dictionaries
    and possibly keep alive a reference to a base array.
    """

    def __init__(self, interface, base=None):
        self.__array_interface__ = interface
        self.base = base


def _maybe_view_as_subclass(original_array, new_array):
    if type(original_array) is not type(new_array):
        # if input was an ndarray subclass and subclasses were OK,
        # then view the result as that subclass.
        new_array = new_array.view(type=type(original_array))
        # Since we have done something akin to a view from original_array, we
        # should let the subclass finalize (if it has it implemented, i.e., is
        # not None).
        if new_array.__array_finalize__:
            new_array.__array_finalize__(original_array)
    return new_array


@set_module("numpy.lib.stride_tricks")
def as_strided(x, shape=None, strides=None, subok=False, writeable=True):
    """
    Create a view into the array with the given shape and strides.

    .. warning:: This function has to be used with extreme care, see notes.

    Parameters
    ----------
    x : ndarray
        Array to create a new.
    shape : sequence of int, optional
        The shape of the new array. Defaults to ``x.shape``.
    strides : sequence of int, optional
        The strides of the new array. Defaults to ``x.strides``.
    subok : bool, optional
        .. versionadded:: 1.10

        If True, subclasses are preserved.
    writeable : bool, optional
        .. versionadded:: 1.12

        If set to False, the returned array will always be readonly.
        Otherwise it will be writable if the original array was. It
        is advisable to set this to False if possible (see Notes).

    Returns
    -------
    view : ndarray

    See also
    --------
    broadcast_to : broadcast an array to a given shape.
    reshape : reshape an array.
    lib.stride_tricks.sliding_window_view :
        userfriendly and safe function for a creation of sliding window views.

    Notes
    -----
    ``as_strided`` creates a view into the array given the exact strides
    and shape. This means it manipulates the internal data structure of
    ndarray and, if done incorrectly, the array elements can point to
    invalid memory and can corrupt results or crash your program.
    It is advisable to always use the original ``x.strides`` when
    calculating new strides to avoid reliance on a contiguous memory
    layout.
    """
    # 设置所返回数组的默认形状为输入数组的形状
    shape = x.shape if shape is None else shape
    # 设置所返回数组的默认步幅为输入数组的步幅
    strides = x.strides if strides is None else strides
    # 返回一个数组视图，这个函数要谨慎使用，遵循函数中的警告和注意事项
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides, subok=subok, writeable=writeable)
    """
        Furthermore, arrays created with this function often contain self
        overlapping memory, so that two elements are identical.
        Vectorized write operations on such arrays will typically be
        unpredictable. They may even give different results for small, large,
        or transposed arrays.
    
        Since writing to these arrays has to be tested and done with great
        care, you may want to use ``writeable=False`` to avoid accidental write
        operations.
    
        For these reasons it is advisable to avoid ``as_strided`` when
        possible.
    """
    # 首先将输入转换为数组，可能保留子类
    x = np.array(x, copy=None, subok=subok)
    # 获取数组的 __array_interface__，并存储为字典
    interface = dict(x.__array_interface__)
    # 如果指定了形状，则将其设置到接口字典中
    if shape is not None:
        interface['shape'] = tuple(shape)
    # 如果指定了步幅，则将其设置到接口字典中
    if strides is not None:
        interface['strides'] = tuple(strides)
    
    # 创建一个虚拟的 DummyArray 对象，并转换为 NumPy 数组
    array = np.asarray(DummyArray(interface, base=x))
    # 由于通过 `__interface__` 创建的数组不保留结构化 dtype，我们显式设置 dtype
    array.dtype = x.dtype
    
    # 尝试将结果视图返回为原始数组的子类视图
    view = _maybe_view_as_subclass(x, array)
    
    # 如果视图可写而且不允许写入，则将视图设置为不可写
    if view.flags.writeable and not writeable:
        view.flags.writeable = False
    
    # 返回视图对象
    return view
# 定义函数 `_sliding_window_view_dispatcher`，用于分派到适当的函数处理器
def _sliding_window_view_dispatcher(x, window_shape, axis=None, *,
                                    subok=None, writeable=None):
    # 返回输入数组 `x` 本身，表示暂时只做了分派的处理
    return (x,)


# 使用装饰器 `array_function_dispatch`，将函数 `_sliding_window_view_dispatcher` 注册到模块 `numpy.lib.stride_tricks`
@array_function_dispatch(
    _sliding_window_view_dispatcher, module="numpy.lib.stride_tricks"
)
# 定义函数 `sliding_window_view`，创建一个滑动窗口视图，从给定的数组 `x` 中生成
def sliding_window_view(x, window_shape, axis=None, *,
                        subok=False, writeable=False):
    """
    Create a sliding window view into the array with the given window shape.

    Also known as rolling or moving window, the window slides across all
    dimensions of the array and extracts subsets of the array at all window
    positions.

    .. versionadded:: 1.20.0

    Parameters
    ----------
    x : array_like
        Array to create the sliding window view from.
    window_shape : int or tuple of int
        Size of window over each axis that takes part in the sliding window.
        If `axis` is not present, must have same length as the number of input
        array dimensions. Single integers `i` are treated as if they were the
        tuple `(i,)`.
    axis : int or tuple of int, optional
        Axis or axes along which the sliding window is applied.
        By default, the sliding window is applied to all axes and
        `window_shape[i]` will refer to axis `i` of `x`.
        If `axis` is given as a `tuple of int`, `window_shape[i]` will refer to
        the axis `axis[i]` of `x`.
        Single integers `i` are treated as if they were the tuple `(i,)`.
    subok : bool, optional
        If True, sub-classes will be passed-through, otherwise the returned
        array will be forced to be a base-class array (default).
    writeable : bool, optional
        When true, allow writing to the returned view. The default is false,
        as this should be used with caution: the returned view contains the
        same memory location multiple times, so writing to one location will
        cause others to change.

    Returns
    -------
    view : ndarray
        Sliding window view of the array. The sliding window dimensions are
        inserted at the end, and the original dimensions are trimmed as
        required by the size of the sliding window.
        That is, ``view.shape = x_shape_trimmed + window_shape``, where
        ``x_shape_trimmed`` is ``x.shape`` with every entry reduced by one less
        than the corresponding window size.

    See Also
    --------
    lib.stride_tricks.as_strided: A lower-level and less safe routine for
        creating arbitrary views from custom shape and strides.
    broadcast_to: broadcast an array to a given shape.

    Notes
    -----
    For many applications using a sliding window view can be convenient, but
    potentially very slow. Often specialized solutions exist, for example:

    - `scipy.signal.fftconvolve`

    - filtering functions in `scipy.ndimage`

    - moving window functions provided by
      `bottleneck <https://github.com/pydata/bottleneck>`_.
    """
    # 返回滑动窗口视图的 ndarray 对象，通过 numpy 的库函数来实现
    # 参数 `subok` 和 `writeable` 默认为 `False`
    return np.lib.stride_tricks.sliding_window_view(x, window_shape, axis, subok=subok, writeable=writeable)
    """
    As a rough estimate, a sliding window approach with an input size of `N`
    and a window size of `W` will scale as `O(N*W)` where frequently a special
    algorithm can achieve `O(N)`. That means that the sliding window variant
    for a window size of 100 can be a 100 times slower than a more specialized
    version.

    Nevertheless, for small window sizes, when no custom algorithm exists, or
    as a prototyping and developing tool, this function can be a good solution.

    Examples
    --------
    >>> from numpy.lib.stride_tricks import sliding_window_view
    >>> x = np.arange(6)
    >>> x.shape
    (6,)
    >>> v = sliding_window_view(x, 3)
    >>> v.shape
    (4, 3)
    >>> v
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])

    This also works in more dimensions, e.g.

    >>> i, j = np.ogrid[:3, :4]
    >>> x = 10*i + j
    >>> x.shape
    (3, 4)
    >>> x
    array([[ 0,  1,  2,  3],
           [10, 11, 12, 13],
           [20, 21, 22, 23]])
    >>> shape = (2,2)
    >>> v = sliding_window_view(x, shape)
    >>> v.shape
    (2, 3, 2, 2)
    >>> v
    array([[[[ 0,  1],
             [10, 11]],
            [[ 1,  2],
             [11, 12]],
            [[ 2,  3],
             [12, 13]]],
           [[[10, 11],
             [20, 21]],
            [[11, 12],
             [21, 22]],
            [[12, 13],
             [22, 23]]]])

    The axis can be specified explicitly:

    >>> v = sliding_window_view(x, 3, 0)
    >>> v.shape
    (1, 4, 3)
    >>> v
    array([[[ 0, 10, 20],
            [ 1, 11, 21],
            [ 2, 12, 22],
            [ 3, 13, 23]]])

    The same axis can be used several times. In that case, every use reduces
    the corresponding original dimension:

    >>> v = sliding_window_view(x, (2, 3), (1, 1))
    >>> v.shape
    (3, 1, 2, 3)
    >>> v
    array([[[[ 0,  1,  2],
             [ 1,  2,  3]]],
           [[[10, 11, 12],
             [11, 12, 13]]],
           [[[20, 21, 22],
             [21, 22, 23]]]])

    Combining with stepped slicing (`::step`), this can be used to take sliding
    views which skip elements:

    >>> x = np.arange(7)
    >>> sliding_window_view(x, 5)[:, ::2]
    array([[0, 2, 4],
           [1, 3, 5],
           [2, 4, 6]])

    or views which move by multiple elements

    >>> x = np.arange(7)
    >>> sliding_window_view(x, 3)[::2, :]
    array([[0, 1, 2],
           [2, 3, 4],
           [4, 5, 6]])

    A common application of `sliding_window_view` is the calculation of running
    statistics. The simplest example is the
    `moving average <https://en.wikipedia.org/wiki/Moving_average>`_:

    >>> x = np.arange(6)
    >>> x.shape
    (6,)
    >>> v = sliding_window_view(x, 3)
    >>> v.shape
    (4, 3)
    >>> v
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    >>> moving_average = v.mean(axis=-1)
    >>> moving_average
    array([1., 2., 3., 4.])
    """
    Note that a sliding window approach is often **not** optimal (see Notes).
    """
    # 将窗口形状转换为元组，如果window_shape是可迭代对象的话；否则转换为单元素元组
    window_shape = (tuple(window_shape)
                    if np.iterable(window_shape)
                    else (window_shape,))

    # 将输入数组转换为numpy数组，保留可能存在的子类
    x = np.array(x, copy=None, subok=subok)

    # 将窗口形状转换为numpy数组
    window_shape_array = np.array(window_shape)

    # 检查窗口形状数组中是否有负值，若有则引发ValueError异常
    if np.any(window_shape_array < 0):
        raise ValueError('`window_shape` cannot contain negative values')

    # 如果axis为None，则默认使用所有维度的轴
    if axis is None:
        axis = tuple(range(x.ndim))
        # 检查window_shape与x的维度数是否一致，若不一致则引发ValueError异常
        if len(window_shape) != len(axis):
            raise ValueError(f'Since axis is `None`, must provide '
                             f'window_shape for all dimensions of `x`; '
                             f'got {len(window_shape)} window_shape elements '
                             f'and `x.ndim` is {x.ndim}.')
    else:
        # 标准化axis，确保所有轴都在有效范围内
        axis = normalize_axis_tuple(axis, x.ndim, allow_duplicate=True)
        # 检查window_shape与axis的长度是否一致，若不一致则引发ValueError异常
        if len(window_shape) != len(axis):
            raise ValueError(f'Must provide matching length window_shape and '
                             f'axis; got {len(window_shape)} window_shape '
                             f'elements and {len(axis)} axes elements.')

    # 计算输出数组的步幅
    out_strides = x.strides + tuple(x.strides[ax] for ax in axis)

    # 调整输入数组的形状，以确保每个轴上窗口的合法性
    x_shape_trimmed = list(x.shape)
    for ax, dim in zip(axis, window_shape):
        if x_shape_trimmed[ax] < dim:
            raise ValueError(
                'window shape cannot be larger than input array shape')
        x_shape_trimmed[ax] -= dim - 1
    out_shape = tuple(x_shape_trimmed) + window_shape

    # 返回按需切片的数组，以形成窗口视图
    return as_strided(x, strides=out_strides, shape=out_shape,
                      subok=subok, writeable=writeable)
# 将形状参数转换为元组，如果形状是可迭代的，则转换为元组，否则创建只包含一个元素的元组
shape = tuple(shape) if np.iterable(shape) else (shape,)

# 使用给定的参数创建一个新的 NumPy 数组，将 array 转换为数组，不复制数据，允许子类化
array = np.array(array, copy=None, subok=subok)

# 如果形状为空且 array 的形状不为空，则抛出值错误异常，说明无法将非标量广播到标量数组
if not shape and array.shape:
    raise ValueError('cannot broadcast a non-scalar to a scalar array')

# 如果形状中有任何负数元素，则抛出值错误异常，说明广播形状的所有元素必须是非负的
if any(size < 0 for size in shape):
    raise ValueError('all elements of broadcast shape must be non-negative')

# 定义额外参数列表为空列表
extras = []

# 使用 np.nditer 迭代器创建一个迭代器对象 it，以便按照形状 shape 广播 array
it = np.nditer(
    (array,), flags=['multi_index', 'refs_ok', 'zerosize_ok'] + extras,
    op_flags=['readonly'], itershape=shape, order='C')

# 进入迭代器上下文
with it:
    # 从迭代器获取广播后的视图，即 itviews 列表的第一个元素
    broadcast = it.itviews[0]

# 调用 _maybe_view_as_subclass 函数，返回 array 或 broadcast 的可能子类视图
result = _maybe_view_as_subclass(array, broadcast)

# 如果 readonly 为 False 且 array 具有 _writeable_no_warn 标志，则设置 result 的写标志和写警告标志
# 这部分代码将来会被移除
if not readonly and array.flags._writeable_no_warn:
    result.flags.writeable = True
    result.flags._warn_on_write = True

# 返回广播后的结果数组 result
return result
    # 从索引 32 开始，每隔 31 个元素取一个位置进行迭代
    for pos in range(32, len(args), 31):
        # 使用广播（broadcasting）来避免分配完整的数组
        # 具讽刺意味的是，np.broadcast 不能正确处理 np.broadcast 对象（它将它们视为标量）
        b = broadcast_to(0, b.shape)
        # 将广播应用于参数数组的子集（从 pos 到 pos+31），包括 b 本身
        b = np.broadcast(b, *args[pos:(pos + 31)])
    # 返回广播后的数组的形状
    return b.shape
# 定义一个空的 NumPy 数据类型作为默认值
_size0_dtype = np.dtype([])


@set_module('numpy')
# 将输入的形状广播成单个形状。
def broadcast_shapes(*args):
    """
    Broadcast the input shapes into a single shape.

    :ref:`Learn more about broadcasting here <basics.broadcasting>`.

    .. versionadded:: 1.20.0

    Parameters
    ----------
    *args : tuples of ints, or ints
        The shapes to be broadcast against each other.

    Returns
    -------
    tuple
        Broadcasted shape.

    Raises
    ------
    ValueError
        If the shapes are not compatible and cannot be broadcast according
        to NumPy's broadcasting rules.

    See Also
    --------
    broadcast
    broadcast_arrays
    broadcast_to

    Examples
    --------
    >>> np.broadcast_shapes((1, 2), (3, 1), (3, 2))
    (3, 2)

    >>> np.broadcast_shapes((6, 7), (5, 6, 1), (7,), (5, 1, 7))
    (5, 6, 7)
    """
    # 创建一个包含指定形状的空数组列表，使用默认的空数据类型 _size0_dtype
    arrays = [np.empty(x, dtype=_size0_dtype) for x in args]
    # 调用内部函数 _broadcast_shape 对数组进行广播操作，并返回结果
    return _broadcast_shape(*arrays)


# 定义 _broadcast_arrays_dispatcher 函数，用于分发 broadcast_arrays 函数的参数
def _broadcast_arrays_dispatcher(*args, subok=None):
    return args


@array_function_dispatch(_broadcast_arrays_dispatcher, module='numpy')
# 广播任意数量的数组。
def broadcast_arrays(*args, subok=False):
    """
    Broadcast any number of arrays against each other.

    Parameters
    ----------
    *args : array_likes
        The arrays to broadcast.

    subok : bool, optional
        If True, then sub-classes will be passed-through, otherwise
        the returned arrays will be forced to be a base-class array (default).

    Returns
    -------
    broadcasted : tuple of arrays
        These arrays are views on the original arrays.  They are typically
        not contiguous.  Furthermore, more than one element of a
        broadcasted array may refer to a single memory location. If you need
        to write to the arrays, make copies first. While you can set the
        ``writable`` flag True, writing to a single output value may end up
        changing more than one location in the output array.

        .. deprecated:: 1.17
            The output is currently marked so that if written to, a deprecation
            warning will be emitted. A future version will set the
            ``writable`` flag False so writing to it will raise an error.

    See Also
    --------
    broadcast
    broadcast_to
    broadcast_shapes

    Examples
    --------
    >>> x = np.array([[1,2,3]])
    >>> y = np.array([[4],[5]])
    >>> np.broadcast_arrays(x, y)
    (array([[1, 2, 3],
            [1, 2, 3]]),
     array([[4, 4, 4],
            [5, 5, 5]]))

    Here is a useful idiom for getting contiguous copies instead of
    non-contiguous views.

    >>> [np.array(a) for a in np.broadcast_arrays(x, y)]
    [array([[1, 2, 3],
            [1, 2, 3]]),
     array([[4, 4, 4],
            [5, 5, 5]])]

    """
    # nditer is not used here to avoid the limit of 32 arrays.
    # Otherwise, something like the following one-liner would suffice:
    # return np.nditer(args, flags=['multi_index', 'zerosize_ok'],
    #                  order='C').itviews
    # 列表推导式，对传入的每个参数进行处理，将其转换为 NumPy 数组
    args = [np.array(_m, copy=None, subok=subok) for _m in args]

    # 计算广播后的数组形状，_broadcast_shape 是一个自定义函数用于计算广播形状
    shape = _broadcast_shape(*args)

    # 列表推导式，对每个数组进行检查和处理：
    # 如果数组形状与广播后的形状相同，则保持不变
    # 否则，使用 _broadcast_to 函数将数组广播到指定形状
    result = [array if array.shape == shape
              else _broadcast_to(array, shape, subok=subok, readonly=False)
                              for array in args]
    # 返回结果元组，其中包含广播后的所有数组
    return tuple(result)
```