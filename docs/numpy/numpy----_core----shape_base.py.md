# `.\numpy\numpy\_core\shape_base.py`

```
# 定义导出的函数和变量名列表，用于模块导入时指定可导出的内容
__all__ = ['atleast_1d', 'atleast_2d', 'atleast_3d', 'block', 'hstack',
           'stack', 'vstack']

# 导入必要的模块和函数
import functools
import itertools
import operator
import warnings

# 从当前包中导入特定模块或函数
from . import numeric as _nx
from . import overrides
from .multiarray import array, asanyarray, normalize_axis_index
from . import fromnumeric as _from_nx

# 使用 functools.partial 创建一个函数 array_function_dispatch，作为数组函数调度器
array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')

# 定义一个函数 _atleast_1d_dispatcher，该函数简单地返回传入的所有参数数组
def _atleast_1d_dispatcher(*arys):
    return arys

# 使用 array_function_dispatch 装饰器注册 _atleast_1d_dispatcher 函数为 atleast_1d 的调度器
@array_function_dispatch(_atleast_1d_dispatcher)
def atleast_1d(*arys):
    """
    Convert inputs to arrays with at least one dimension.

    Scalar inputs are converted to 1-dimensional arrays, whilst
    higher-dimensional inputs are preserved.

    Parameters
    ----------
    arys1, arys2, ... : array_like
        One or more input arrays.

    Returns
    -------
    ret : ndarray
        An array, or tuple of arrays, each with ``a.ndim >= 1``.
        Copies are made only if necessary.

    See Also
    --------
    atleast_2d, atleast_3d

    Examples
    --------
    >>> np.atleast_1d(1.0)
    array([1.])

    >>> x = np.arange(9.0).reshape(3,3)
    >>> np.atleast_1d(x)
    array([[0., 1., 2.],
           [3., 4., 5.],
           [6., 7., 8.]])
    >>> np.atleast_1d(x) is x
    True

    >>> np.atleast_1d(1, [3, 4])
    (array([1]), array([3, 4]))

    """
    # 如果只有一个输入数组，则将其转换为 ndarray，如果是标量则转换成一维数组
    if len(arys) == 1:
        result = asanyarray(arys[0])
        if result.ndim == 0:
            result = result.reshape(1)
        return result
    # 如果有多个输入数组，则逐个转换为 ndarray，标量转换成一维数组
    res = []
    for ary in arys:
        result = asanyarray(ary)
        if result.ndim == 0:
            result = result.reshape(1)
        res.append(result)
    return tuple(res)

# 定义一个函数 _atleast_2d_dispatcher，该函数简单地返回传入的所有参数数组
def _atleast_2d_dispatcher(*arys):
    return arys

# 使用 array_function_dispatch 装饰器注册 _atleast_2d_dispatcher 函数为 atleast_2d 的调度器
@array_function_dispatch(_atleast_2d_dispatcher)
def atleast_2d(*arys):
    """
    View inputs as arrays with at least two dimensions.

    Parameters
    ----------
    arys1, arys2, ... : array_like
        One or more array-like sequences.  Non-array inputs are converted
        to arrays.  Arrays that already have two or more dimensions are
        preserved.

    Returns
    -------
    res, res2, ... : ndarray
        An array, or tuple of arrays, each with ``a.ndim >= 2``.
        Copies are avoided where possible, and views with two or more
        dimensions are returned.

    See Also
    --------
    atleast_1d, atleast_3d

    Examples
    --------
    >>> np.atleast_2d(3.0)
    array([[3.]])

    >>> x = np.arange(3.0)
    >>> np.atleast_2d(x)
    array([[0., 1., 2.]])
    >>> np.atleast_2d(x).base is x
    True

    >>> np.atleast_2d(1, [1, 2], [[1, 2]])
    (array([[1]]), array([[1, 2]]), array([[1, 2]]))

    """
    # 将输入数组转换为至少二维的 ndarray 视图
    res = []
    for ary in arys:
        ary = asanyarray(ary)
        if ary.ndim == 0:
            result = ary.reshape(1, 1)
        elif ary.ndim == 1:
            result = ary[_nx.newaxis, :]
        else:
            result = ary
        res.append(result)
    # 如果结果列表的长度为1，则返回列表中的唯一元素
    if len(res) == 1:
        return res[0]
    # 否则，返回结果列表转换成元组的形式
    else:
        return tuple(res)
# 定义一个分派函数 `_atleast_3d_dispatcher`，接受任意数量的参数并将它们作为元组返回
def _atleast_3d_dispatcher(*arys):
    return arys


# 使用装饰器 `array_function_dispatch` 将 `_atleast_3d_dispatcher` 与 `atleast_3d` 函数关联
@array_function_dispatch(_atleast_3d_dispatcher)
# 定义函数 `atleast_3d`，将输入视为至少三维的数组
def atleast_3d(*arys):
    """
    View inputs as arrays with at least three dimensions.

    Parameters
    ----------
    arys1, arys2, ... : array_like
        One or more array-like sequences.  Non-array inputs are converted to
        arrays.  Arrays that already have three or more dimensions are
        preserved.

    Returns
    -------
    res1, res2, ... : ndarray
        An array, or tuple of arrays, each with ``a.ndim >= 3``.  Copies are
        avoided where possible, and views with three or more dimensions are
        returned.  For example, a 1-D array of shape ``(N,)`` becomes a view
        of shape ``(1, N, 1)``, and a 2-D array of shape ``(M, N)`` becomes a
        view of shape ``(M, N, 1)``.

    See Also
    --------
    atleast_1d, atleast_2d

    Examples
    --------
    >>> np.atleast_3d(3.0)
    array([[[3.]]])

    >>> x = np.arange(3.0)
    >>> np.atleast_3d(x).shape
    (1, 3, 1)

    >>> x = np.arange(12.0).reshape(4,3)
    >>> np.atleast_3d(x).shape
    (4, 3, 1)
    >>> np.atleast_3d(x).base is x.base  # x is a reshape, so not base itself
    True

    >>> for arr in np.atleast_3d([1, 2], [[1, 2]], [[[1, 2]]]):
    ...     print(arr, arr.shape) # doctest: +SKIP
    ...
    [[[1]
      [2]]] (1, 2, 1)
    [[[1]
      [2]]] (1, 2, 1)
    [[[1 2]]] (1, 1, 2)

    """
    # 初始化结果列表
    res = []
    # 遍历输入的每个数组
    for ary in arys:
        # 将每个数组转换为 ndarray 类型
        ary = asanyarray(ary)
        # 根据数组的维度进行不同的处理
        if ary.ndim == 0:
            result = ary.reshape(1, 1, 1)
        elif ary.ndim == 1:
            result = ary[_nx.newaxis, :, _nx.newaxis]
        elif ary.ndim == 2:
            result = ary[:, :, _nx.newaxis]
        else:
            result = ary
        # 将处理后的结果添加到结果列表中
        res.append(result)
    # 如果结果列表中只有一个元素，则返回该元素
    if len(res) == 1:
        return res[0]
    else:
        # 否则返回结果元组
        return tuple(res)


# 定义一个分派函数 `_arrays_for_stack_dispatcher`，用于处理堆叠操作中的数组序列
def _arrays_for_stack_dispatcher(arrays):
    # 如果输入的数组不具备 `__getitem__` 方法，则抛出类型错误
    if not hasattr(arrays, "__getitem__"):
        raise TypeError('arrays to stack must be passed as a "sequence" type '
                        'such as list or tuple.')
    # 将输入的数组转换为元组并返回
    return tuple(arrays)


# 定义一个分派函数 `_vhstack_dispatcher`，接受元组参数 `tup` 并调用 `_arrays_for_stack_dispatcher` 处理
def _vhstack_dispatcher(tup, *, dtype=None, casting=None):
    return _arrays_for_stack_dispatcher(tup)


# 使用装饰器 `array_function_dispatch` 将 `_vhstack_dispatcher` 与 `vstack` 函数关联
@array_function_dispatch(_vhstack_dispatcher)
# 定义函数 `vstack`，用于沿第一个轴垂直堆叠数组序列
def vstack(tup, *, dtype=None, casting="same_kind"):
    """
    Stack arrays in sequence vertically (row wise).

    This is equivalent to concatenation along the first axis after 1-D arrays
    of shape `(N,)` have been reshaped to `(1,N)`. Rebuilds arrays divided by
    `vsplit`.

    This function makes most sense for arrays with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions `concatenate`, `stack` and
    `block` provide more general stacking and concatenation operations.

    Parameters
    ----------

    """
    # tup 是包含多个 ndarray 的序列
    # 所有的 ndarray 必须在除第一个轴以外的维度上具有相同的形状
    # 1-D 数组必须具有相同的长度
    tup : sequence of ndarrays
        The arrays must have the same shape along all but the first axis.
        1-D arrays must have the same length.

    # dtype 可以是字符串或 dtype 对象
    # 如果提供了 dtype，则返回的数组将具有此数据类型
    # 不能与参数 `out` 同时提供
    # 自 NumPy 版本 1.24 起添加
    dtype : str or dtype
        If provided, the destination array will have this dtype. Cannot be
        provided together with `out`.

        .. versionadded:: 1.24

    # casting 控制数据类型转换的方式
    # 可选值包括 {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}，默认为 'same_kind'
    # 自 NumPy 版本 1.24 起添加
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur. Defaults to 'same_kind'.

        .. versionadded:: 1.24

    # 返回由给定数组堆叠而成的 ndarray
    # 返回的数组至少是二维的
    Returns
    -------
    stacked : ndarray
        The array formed by stacking the given arrays, will be at least 2-D.

    # 参见下列函数，用于在不同的情况下连接多个数组：
    # concatenate: 沿着已存在的轴连接数组序列
    # stack: 沿着新轴连接数组序列
    # block: 从块的嵌套列表中组装 nd-array
    # hstack: 按列方向（水平方向）连接数组序列
    # dstack: 按深度方向（第三个轴）连接数组序列
    # column_stack: 将1-D 数组按列堆叠成一个 2-D 数组
    # vsplit: 按行方向（垂直方向）将数组分割成多个子数组

    See Also
    --------
    concatenate : Join a sequence of arrays along an existing axis.
    stack : Join a sequence of arrays along a new axis.
    block : Assemble an nd-array from nested lists of blocks.
    hstack : Stack arrays in sequence horizontally (column wise).
    dstack : Stack arrays in sequence depth wise (along third axis).
    column_stack : Stack 1-D arrays as columns into a 2-D array.
    vsplit : Split an array into multiple sub-arrays vertically (row-wise).

    # 示例
    # 堆叠两个一维数组 `a` 和 `b`，生成一个二维数组
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([4, 5, 6])
    >>> np.vstack((a,b))
    array([[1, 2, 3],
           [4, 5, 6]])

    # 堆叠两个二维数组 `a` 和 `b`，生成一个二维数组
    >>> a = np.array([[1], [2], [3]])
    >>> b = np.array([[4], [5], [6]])
    >>> np.vstack((a,b))
    array([[1],
           [2],
           [3],
           [4],
           [5],
           [6]])

    """
    # 将 tup 中的数组至少转换为二维数组
    arrs = atleast_2d(*tup)
    # 如果转换后的结果不是一个元组，则将其包装成一个元组
    if not isinstance(arrs, tuple):
        arrs = (arrs,)
    # 调用底层函数 `_nx.concatenate` 进行数组堆叠操作
    return _nx.concatenate(arrs, 0, dtype=dtype, casting=casting)
@array_function_dispatch(_vhstack_dispatcher)
# 使用装饰器将函数与_dispatcher函数关联，用于分发不同输入的处理
def hstack(tup, *, dtype=None, casting="same_kind"):
    """
    Stack arrays in sequence horizontally (column wise).

    This is equivalent to concatenation along the second axis, except for 1-D
    arrays where it concatenates along the first axis. Rebuilds arrays divided
    by `hsplit`.

    This function makes most sense for arrays with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions `concatenate`, `stack` and
    `block` provide more general stacking and concatenation operations.

    Parameters
    ----------
    tup : sequence of ndarrays
        The arrays must have the same shape along all but the second axis,
        except 1-D arrays which can be any length.

    dtype : str or dtype
        If provided, the destination array will have this dtype. Cannot be
        provided together with `out`.

        .. versionadded:: 1.24

    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur. Defaults to 'same_kind'.

        .. versionadded:: 1.24

    Returns
    -------
    stacked : ndarray
        The array formed by stacking the given arrays.

    See Also
    --------
    concatenate : Join a sequence of arrays along an existing axis.
    stack : Join a sequence of arrays along a new axis.
    block : Assemble an nd-array from nested lists of blocks.
    vstack : Stack arrays in sequence vertically (row wise).
    dstack : Stack arrays in sequence depth wise (along third axis).
    column_stack : Stack 1-D arrays as columns into a 2-D array.
    hsplit : Split an array into multiple sub-arrays 
             horizontally (column-wise).

    Examples
    --------
    >>> a = np.array((1,2,3))
    >>> b = np.array((4,5,6))
    >>> np.hstack((a,b))
    array([1, 2, 3, 4, 5, 6])
    >>> a = np.array([[1],[2],[3]])
    >>> b = np.array([[4],[5],[6]])
    >>> np.hstack((a,b))
    array([[1, 4],
           [2, 5],
           [3, 6]])

    """
    arrs = atleast_1d(*tup)
    # 将输入数组至少视为一维数组
    if not isinstance(arrs, tuple):
        arrs = (arrs,)
    # 作为特殊情况，对于一维数组，第0维被视为“水平”的
    if arrs and arrs[0].ndim == 1:
        return _nx.concatenate(arrs, 0, dtype=dtype, casting=casting)
    else:
        return _nx.concatenate(arrs, 1, dtype=dtype, casting=casting)


def _stack_dispatcher(arrays, axis=None, out=None, *,
                      dtype=None, casting=None):
    arrays = _arrays_for_stack_dispatcher(arrays)
    if out is not None:
        # optimize for the typical case where only arrays is provided
        arrays = list(arrays)
        arrays.append(out)
    return arrays


@array_function_dispatch(_stack_dispatcher)
# 使用装饰器将函数与_dispatcher函数关联，用于分发不同输入的处理
def stack(arrays, axis=0, out=None, *, dtype=None, casting="same_kind"):
    """
    Join a sequence of arrays along a new axis.
    """
    """
    The `axis` parameter specifies the index of the new axis in the
    dimensions of the result. For example, if `axis=0` it will be the first
    dimension and if `axis=-1` it will be the last dimension.
    
    .. versionadded:: 1.10.0
    
    Parameters
    ----------
    arrays : sequence of array_like
        Each array must have the same shape.
    
    axis : int, optional
        The axis in the result array along which the input arrays are stacked.
    
    out : ndarray, optional
        If provided, the destination to place the result. The shape must be
        correct, matching that of what stack would have returned if no
        out argument were specified.
    
    dtype : str or dtype
        If provided, the destination array will have this dtype. Cannot be
        provided together with `out`.
    
        .. versionadded:: 1.24
    
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur. Defaults to 'same_kind'.
    
        .. versionadded:: 1.24
    
    
    Returns
    -------
    stacked : ndarray
        The stacked array has one more dimension than the input arrays.
    
    See Also
    --------
    concatenate : Join a sequence of arrays along an existing axis.
    block : Assemble an nd-array from nested lists of blocks.
    split : Split array into a list of multiple sub-arrays of equal size.
    
    Examples
    --------
    >>> rng = np.random.default_rng()
    >>> arrays = [rng.normal(size=(3,4)) for _ in range(10)]
    >>> np.stack(arrays, axis=0).shape
    (10, 3, 4)
    
    >>> np.stack(arrays, axis=1).shape
    (3, 10, 4)
    
    >>> np.stack(arrays, axis=2).shape
    (3, 4, 10)
    
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([4, 5, 6])
    >>> np.stack((a, b))
    array([[1, 2, 3],
           [4, 5, 6]])
    
    >>> np.stack((a, b), axis=-1)
    array([[1, 4],
           [2, 5],
           [3, 6]])
    
    """
    
    arrays = [asanyarray(arr) for arr in arrays]
    # 转换所有输入数组为数组对象
    
    if not arrays:
        raise ValueError('need at least one array to stack')
    # 如果数组列表为空，则抛出数值错误
    
    shapes = {arr.shape for arr in arrays}
    # 获取所有输入数组的形状集合
    
    if len(shapes) != 1:
        raise ValueError('all input arrays must have the same shape')
    # 如果形状集合的长度不为1，则抛出数值错误，要求所有输入数组必须具有相同的形状
    
    result_ndim = arrays[0].ndim + 1
    # 计算结果数组的维数为第一个数组的维数加1
    
    axis = normalize_axis_index(axis, result_ndim)
    # 根据结果数组的维数规范化轴索引
    
    sl = (slice(None),) * axis + (_nx.newaxis,)
    # 创建切片对象以扩展数组，添加一个新轴
    
    expanded_arrays = [arr[sl] for arr in arrays]
    # 使用切片对象扩展所有输入数组
    
    return _nx.concatenate(expanded_arrays, axis=axis, out=out,
                           dtype=dtype, casting=casting)
    # 使用 numpy 的 concatenate 函数连接扩展后的数组，指定轴、输出、数据类型和数据类型转换方式
# Internal functions to eliminate the overhead of repeated dispatch in one of
# the two possible paths inside np.block.
# Use getattr to protect against __array_function__ being disabled.
# 使用 getattr 来保护免受 __array_function__ 被禁用的影响。
_size = getattr(_from_nx.size, '__wrapped__', _from_nx.size)
# 获取 _from_nx.size 的包装版本，如果不存在，则返回原始版本 _from_nx.size。
_ndim = getattr(_from_nx.ndim, '__wrapped__', _from_nx.ndim)
# 获取 _from_nx.ndim 的包装版本，如果不存在，则返回原始版本 _from_nx.ndim。
_concatenate = getattr(_from_nx.concatenate,
                       '__wrapped__', _from_nx.concatenate)
# 获取 _from_nx.concatenate 的包装版本，如果不存在，则返回原始版本 _from_nx.concatenate。

def _block_format_index(index):
    """
    Convert a list of indices ``[0, 1, 2]`` into ``"arrays[0][1][2]"``.
    将索引列表 ``[0, 1, 2]`` 转换为 ``"arrays[0][1][2]"`` 的字符串形式。
    """
    idx_str = ''.join('[{}]'.format(i) for i in index if i is not None)
    # 生成索引的字符串表示，跳过为 None 的索引。
    return 'arrays' + idx_str
    # 返回格式化后的索引字符串。

def _block_check_depths_match(arrays, parent_index=[]):
    """
    Recursive function checking that the depths of nested lists in `arrays`
    all match. Mismatch raises a ValueError as described in the block
    docstring below.

    The entire index (rather than just the depth) needs to be calculated
    for each innermost list, in case an error needs to be raised, so that
    the index of the offending list can be printed as part of the error.

    Parameters
    ----------
    arrays : nested list of arrays
        The arrays to check
    parent_index : list of int
        The full index of `arrays` within the nested lists passed to
        `_block_check_depths_match` at the top of the recursion.

    Returns
    -------
    first_index : list of int
        The full index of an element from the bottom of the nesting in
        `arrays`. If any element at the bottom is an empty list, this will
        refer to it, and the last index along the empty axis will be None.
    max_arr_ndim : int
        The maximum of the ndims of the arrays nested in `arrays`.
    final_size: int
        The number of elements in the final array. This is used the motivate
        the choice of algorithm used using benchmarking wisdom.
    
    递归函数，用于检查 `arrays` 中嵌套列表的深度是否匹配。如果不匹配，会抛出一个 ValueError，
    如下面 block 文档字符串中描述的那样。

    需要为每个最内层列表计算整个索引（而不仅仅是深度），以防需要引发错误，以便可以打印错误的列表索引作为错误的一部分。

    参数
    ----------
    arrays : 嵌套数组的列表
        要检查的数组
    parent_index : int 列表
        传递给 `_block_check_depths_match` 函数的嵌套列表中 `arrays` 的完整索引。

    返回
    -------
    first_index : int 列表
        `arrays` 最底层嵌套元素的完整索引。如果底层有任何空列表，则会引用它，并且沿着空轴的最后一个索引将为 None。
    max_arr_ndim : int
        `arrays` 中嵌套数组的最大维度。
    final_size: int
        最终数组中的元素数目。这是用于选择使用基准智慧的算法的动机。
    """
    if type(arrays) is tuple:
        # not strictly necessary, but saves us from:
        #  - more than one way to do things - no point treating tuples like
        #    lists
        #  - horribly confusing behaviour that results when tuples are
        #    treated like ndarray
        raise TypeError(
            '{} is a tuple. '
            'Only lists can be used to arrange blocks, and np.block does '
            'not allow implicit conversion from tuple to ndarray.'.format(
                _block_format_index(parent_index)
            )
        )
    # 如果数组类型为元组，则抛出类型错误，因为 np.block 不允许将元组隐式转换为 ndarray。
    # 如果 arrays 是列表且长度大于 0
    elif type(arrays) is list and len(arrays) > 0:
        # 生成一个生成器表达式，用于检查每个数组的深度匹配情况
        idxs_ndims = (_block_check_depths_match(arr, parent_index + [i])
                      for i, arr in enumerate(arrays))

        # 获取第一个数组的索引、最大维度和最终大小
        first_index, max_arr_ndim, final_size = next(idxs_ndims)
        
        # 遍历生成器中的剩余元组
        for index, ndim, size in idxs_ndims:
            # 累加最终大小
            final_size += size
            # 更新最大维度
            if ndim > max_arr_ndim:
                max_arr_ndim = ndim
            # 检查索引的长度是否与第一个索引相同，若不同则抛出 ValueError
            if len(index) != len(first_index):
                raise ValueError(
                    "List depths are mismatched. First element was at depth "
                    "{}, but there is an element at depth {} ({})".format(
                        len(first_index),
                        len(index),
                        _block_format_index(index)
                    )
                )
            # 传播标志，指示底部的空列表
            if index[-1] is None:
                first_index = index
        
        # 返回第一个索引、最大维度和最终大小
        return first_index, max_arr_ndim, final_size
    
    # 如果 arrays 是列表但长度为 0
    elif type(arrays) is list and len(arrays) == 0:
        # 已经达到空列表的底部
        return parent_index + [None], 0, 0
    
    # 如果 arrays 是标量或者数组
    else:
        # 计算 arrays 的大小
        size = _size(arrays)
        # 返回父索引、arrays 的维度和大小
        return parent_index, _ndim(arrays), size
# 确保数组 `a` 至少具有 `ndim` 维度，通过必要时在 `a.shape` 前面添加一些维度为1的方式来实现
def _atleast_nd(a, ndim):
    return array(a, ndmin=ndim, copy=None, subok=True)


# 对给定的值列表进行累加操作，返回累加后的结果列表
def _accumulate(values):
    return list(itertools.accumulate(values))


# 给定数组的形状列表和轴向，返回连接后的形状和切片前缀
def _concatenate_shapes(shapes, axis):
    """Given array shapes, return the resulting shape and slices prefixes.

    These help in nested concatenation.

    Returns
    -------
    shape: tuple of int
        This tuple satisfies::

            shape, _ = _concatenate_shapes([arr.shape for shape in arrs], axis)
            shape == concatenate(arrs, axis).shape

    slice_prefixes: tuple of (slice(start, end), )
        For a list of arrays being concatenated, this returns the slice
        in the larger array at axis that needs to be sliced into.

        For example, the following holds::

            ret = concatenate([a, b, c], axis)
            _, (sl_a, sl_b, sl_c) = concatenate_slices([a, b, c], axis)

            ret[(slice(None),) * axis + sl_a] == a
            ret[(slice(None),) * axis + sl_b] == b
            ret[(slice(None),) * axis + sl_c] == c

        These are called slice prefixes since they are used in the recursive
        blocking algorithm to compute the left-most slices during the
        recursion. Therefore, they must be prepended to rest of the slice
        that was computed deeper in the recursion.

        These are returned as tuples to ensure that they can quickly be added
        to existing slice tuple without creating a new tuple every time.

    """
    # 缓存将会被重复使用的结果。
    shape_at_axis = [shape[axis] for shape in shapes]

    # 选择任意一个形状
    first_shape = shapes[0]
    first_shape_pre = first_shape[:axis]
    first_shape_post = first_shape[axis+1:]

    # 检查是否存在任何形状与第一个形状的前缀或后缀不匹配
    if any(shape[:axis] != first_shape_pre or
           shape[axis+1:] != first_shape_post for shape in shapes):
        raise ValueError(
            'Mismatched array shapes in block along axis {}.'.format(axis))

    # 计算连接后的形状
    shape = (first_shape_pre + (sum(shape_at_axis),) + first_shape[axis+1:])

    # 计算轴向上的偏移量
    offsets_at_axis = _accumulate(shape_at_axis)
    
    # 返回切片前缀列表，以确保在递归中计算最左边的切片时使用
    slice_prefixes = [(slice(start, end),)
                      for start, end in zip([0] + offsets_at_axis,
                                            offsets_at_axis)]
    return shape, slice_prefixes


# 递归地计算数组的信息，包括最终数组的形状、切片列表和可以用于在新数组内赋值的数组列表
def _block_info_recursion(arrays, max_depth, result_ndim, depth=0):
    """
    Returns the shape of the final array, along with a list
    of slices and a list of arrays that can be used for assignment inside the
    new array

    Parameters
    ----------
    arrays : nested list of arrays
        The arrays to check
    max_depth : list of int
        The number of nested lists
    result_ndim : int
        The number of dimensions in the final array.

    Returns
    -------
    shape : tuple of int
        The shape that the final array will take on.
    """
    # slices: list of tuple of slices
    #        The slices into the full array required for assignment. These are
    #        required to be prepended with ``(Ellipsis, )`` to obtain to correct
    #        final index.
    # arrays: list of ndarray
    #        The data to assign to each slice of the full array
    """
    if depth < max_depth:
        # Recursively call _block_info_recursion for each array in arrays
        shapes, slices, arrays = zip(
            *[_block_info_recursion(arr, max_depth, result_ndim, depth+1)
              for arr in arrays])

        # Calculate the axis and shape after concatenating shapes
        axis = result_ndim - max_depth + depth
        shape, slice_prefixes = _concatenate_shapes(shapes, axis)

        # Prepend the slice prefix to each inner slice and flatten them
        slices = [slice_prefix + the_slice
                  for slice_prefix, inner_slices in zip(slice_prefixes, slices)
                  for the_slice in inner_slices]

        # Flatten the list of arrays into a single list
        arrays = functools.reduce(operator.add, arrays)

        return shape, slices, arrays
    else:
        # Base case: depth >= max_depth, return shape, slices, and arrays as they are
        # We've 'bottomed out' - arrays is either a scalar or an array
        # type(arrays) is not list
        # Return the slice and the array inside a list to be consistent with
        # the recursive case.
        arr = _atleast_nd(arrays, result_ndim)
        return arr.shape, [()], [arr]
# 定义一个内部函数 `_block`，用于基于重复连接实现块的操作。
# `arrays` 是传递给 `block` 函数的参数，表示需要处理的数组列表。
# `max_depth` 是 `arrays` 中嵌套列表的最大深度。
# `result_ndim` 是 `arrays` 中数组的最大维度和列表的深度之间的较大者。
# `depth` 是当前递归的深度，默认为0。

if depth < max_depth:
    # 如果当前深度还未达到最大深度，则递归调用 `_block` 函数处理每个数组，并返回结果。
    arrs = [_block(arr, max_depth, result_ndim, depth+1)
            for arr in arrays]
    # 将处理后的数组列表按照轴 `-(max_depth-depth)` 进行连接。
    return _concatenate(arrs, axis=-(max_depth-depth))
else:
    # 如果已经到达最大深度，则调用 `_atleast_nd` 函数确保数组至少具有 `result_ndim` 维度。
    # 这表示处理到了数组的最底层，`arrays` 可能是标量或数组。
    return _atleast_nd(arrays, result_ndim)


def _block_dispatcher(arrays):
    # 使用 `type(arrays) is list` 来匹配 `np.block()` 的行为，特别处理列表而不是泛型可迭代对象或元组。
    # 同时，我们知道 `list.__array_function__` 永远不会存在。
    if type(arrays) is list:
        # 遍历输入的列表 `arrays` 中的子列表，使用递归方式调用 `_block_dispatcher`。
        for subarrays in arrays:
            yield from _block_dispatcher(subarrays)
    else:
        # 如果 `arrays` 不是列表，则直接生成该数组。
        yield arrays


@array_function_dispatch(_block_dispatcher)
def block(arrays):
    """
    Assemble an nd-array from nested lists of blocks.

    Blocks in the innermost lists are concatenated (see `concatenate`) along
    the last dimension (-1), then these are concatenated along the
    second-last dimension (-2), and so on until the outermost list is reached.

    Blocks can be of any dimension, but will not be broadcasted using
    the normal rules. Instead, leading axes of size 1 are inserted, 
    to make ``block.ndim`` the same for all blocks. This is primarily useful
    for working with scalars, and means that code like ``np.block([v, 1])``
    is valid, where ``v.ndim == 1``.

    When the nested list is two levels deep, this allows block matrices to be
    constructed from their components.

    .. versionadded:: 1.13.0

    Parameters
    ----------
    arrays : nested list of array_like or scalars (but not tuples)
        If passed a single ndarray or scalar (a nested list of depth 0), this
        is returned unmodified (and not copied).

        Elements shapes must match along the appropriate axes (without
        broadcasting), but leading 1s will be prepended to the shape as
        necessary to make the dimensions match.

    Returns
    -------
    block_array : ndarray
        The array assembled from the given blocks.

        The dimensionality of the output is equal to the greatest of:

        * the dimensionality of all the inputs
        * the depth to which the input list is nested

    Raises
    ------
    ValueError
        * If list depths are mismatched - for instance, ``[[a, b], c]`` is
          illegal, and should be spelt ``[[a, b], [c]]``
        * If lists are empty - for instance, ``[[a, b], []]``

    See Also
    --------
    # concatenate : Join a sequence of arrays along an existing axis.
    # stack : Join a sequence of arrays along a new axis.
    # vstack : Stack arrays in sequence vertically (row wise).
    # hstack : Stack arrays in sequence horizontally (column wise).
    # dstack : Stack arrays in sequence depth wise (along third axis).
    # column_stack : Stack 1-D arrays as columns into a 2-D array.
    # vsplit : Split an array into multiple sub-arrays vertically (row-wise).
    
    # Notes
    # -----
    
    # When called with only scalars, ``np.block`` is equivalent to an ndarray
    # call. So ``np.block([[1, 2], [3, 4]])`` is equivalent to
    # ``np.array([[1, 2], [3, 4]])``.
    
    # This function does not enforce that the blocks lie on a fixed grid.
    # ``np.block([[a, b], [c, d]])`` is not restricted to arrays of the form::
    
    #     AAAbb
    #     AAAbb
    #     cccDD
    
    # But is also allowed to produce, for some ``a, b, c, d``::
    
    #     AAAbb
    #     AAAbb
    #     cDDDD
    
    # Since concatenation happens along the last axis first, `block` is *not*
    # capable of producing the following directly::
    
    #     AAAbb
    #     cccbb
    #     cccDD
    
    # Matlab's "square bracket stacking", ``[A, B, ...; p, q, ...]``, is
    # equivalent to ``np.block([[A, B, ...], [p, q, ...]])``.
    
    # Examples
    # --------
    # The most common use of this function is to build a block matrix
    
    # >>> A = np.eye(2) * 2
    # >>> B = np.eye(3) * 3
    # >>> np.block([
    # ...     [A,               np.zeros((2, 3))],
    # ...     [np.ones((3, 2)), B               ]
    # ... ])
    # array([[2., 0., 0., 0., 0.],
    #        [0., 2., 0., 0., 0.],
    #        [1., 1., 3., 0., 0.],
    #        [1., 1., 0., 3., 0.],
    #        [1., 1., 0., 0., 3.]])
    
    # With a list of depth 1, `block` can be used as `hstack`
    
    # >>> np.block([1, 2, 3])              # hstack([1, 2, 3])
    # array([1, 2, 3])
    
    # >>> a = np.array([1, 2, 3])
    # >>> b = np.array([4, 5, 6])
    # >>> np.block([a, b, 10])             # hstack([a, b, 10])
    # array([ 1,  2,  3,  4,  5,  6, 10])
    
    # >>> A = np.ones((2, 2), int)
    # >>> B = 2 * A
    # >>> np.block([A, B])                 # hstack([A, B])
    # array([[1, 1, 2, 2],
    #        [1, 1, 2, 2]])
    
    # With a list of depth 2, `block` can be used in place of `vstack`:
    
    # >>> a = np.array([1, 2, 3])
    # >>> b = np.array([4, 5, 6])
    # >>> np.block([[a], [b]])             # vstack([a, b])
    # array([[1, 2, 3],
    #        [4, 5, 6]])
    
    # >>> A = np.ones((2, 2), int)
    # >>> B = 2 * A
    # >>> np.block([[A], [B]])             # vstack([A, B])
    # array([[1, 1],
    #        [1, 1],
    #        [2, 2],
    #        [2, 2]])
    
    # It can also be used in places of `atleast_1d` and `atleast_2d`
    
    # >>> a = np.array(0)
    # >>> b = np.array([1])
    # >>> np.block([a])                    # atleast_1d(a)
    # array([0])
    # >>> np.block([b])                    # atleast_1d(b)
    # array([1])
    
    # >>> np.block([[a]])                  # atleast_2d(a)
    # array([[0]])
    # >>> np.block([[b]])                  # atleast_2d(b)
    # array([[1]])
    array([[1]])



"""
arrays, list_ndim, result_ndim, final_size = _block_setup(arrays)
"""

"""
# 通过调用 _block_setup 函数设置必要的变量
# arrays: 输入的数组列表
# list_ndim: 输入数组中最高维度的数量
# result_ndim: 结果数组的维度
# final_size: 最终拼接的数组的大小
# 这些变量将在后续的条件判断和函数调用中使用
"""

"""
# 根据性能测试结果，发现在 i7-7700HQ 处理器和双通道 2400MHz 内存上，
# 通过直接拼接生成大小约为 256x256 的数组会更快。
# dtype 的选择并不明显影响性能。
#
# 使用重复拼接生成二维数组需要对数组进行两次复制。
#
# 最快的算法取决于 CPU 功率和内存速度的比例。
# 可以通过监控基准测试的结果
# https://pv.github.io/numpy-bench/#bench_shape_base.Block2D.time_block2d
# 来调整这个参数，直到实现 `_block_info_recursion` 算法的 C 版本，
# 这可能会比 Python 版本更快。
"""

if list_ndim * final_size > (2 * 512 * 512):
    return _block_slicing(arrays, list_ndim, result_ndim)
else:
    return _block_concatenate(arrays, list_ndim, result_ndim)
# 这些辅助函数主要用于测试。
# 它们允许我们编写测试，直接调用 `_block_slicing` 或 `_block_concatenate`，
# 而不需要阻塞大数组以触发所需的路径。

def _block_setup(arrays):
    """
    返回 (`arrays`, list_ndim, result_ndim, final_size)
    """
    # 调用 `_block_check_depths_match` 函数检查数组的深度匹配情况，并返回底部索引、数组维度、最终大小
    bottom_index, arr_ndim, final_size = _block_check_depths_match(arrays)
    # 计算列表的维度
    list_ndim = len(bottom_index)
    # 如果 bottom_index 不为空且最后一个元素为 None，则抛出 ValueError 异常
    if bottom_index and bottom_index[-1] is None:
        raise ValueError(
            'List at {} cannot be empty'.format(
                _block_format_index(bottom_index)
            )
        )
    # 计算结果的维度，为数组维度和列表维度的较大值
    result_ndim = max(arr_ndim, list_ndim)
    # 返回 arrays、列表维度、结果维度和最终大小的元组
    return arrays, list_ndim, result_ndim, final_size


def _block_slicing(arrays, list_ndim, result_ndim):
    # 通过 `_block_info_recursion` 函数获取形状、切片和处理后的数组
    shape, slices, arrays = _block_info_recursion(
        arrays, list_ndim, result_ndim)
    # 计算数组的数据类型
    dtype = _nx.result_type(*[arr.dtype for arr in arrays])

    # 测试是否优先使用 F（Fortran）顺序，仅在所有输入数组都为 F 顺序且非 C（连续）顺序时选择 F
    F_order = all(arr.flags['F_CONTIGUOUS'] for arr in arrays)
    C_order = all(arr.flags['C_CONTIGUOUS'] for arr in arrays)
    order = 'F' if F_order and not C_order else 'C'
    # 使用 `_nx.empty` 创建一个指定形状、数据类型和顺序的空数组
    result = _nx.empty(shape=shape, dtype=dtype, order=order)
    # 注意：在 C 实现中，可以使用函数 `PyArray_CreateMultiSortedStridePerm` 来更高级地猜测所需的顺序。

    # 将处理后的数组填充到结果数组的相应切片位置
    for the_slice, arr in zip(slices, arrays):
        result[(Ellipsis,) + the_slice] = arr
    # 返回填充后的结果数组
    return result


def _block_concatenate(arrays, list_ndim, result_ndim):
    # 调用 `_block` 函数处理数组
    result = _block(arrays, list_ndim, result_ndim)
    # 如果列表维度为 0，处理一个特殊情况，其中 `_block` 返回一个视图，因为 `arrays` 是单个 numpy 数组而不是 numpy 数组列表。
    # 这可能会复制标量或列表两次，但对于关注性能的用户来说，这不太可能是一个常见情况。
    if list_ndim == 0:
        result = result.copy()  # 复制结果，以防 `arrays` 是单个数组的情况
    # 返回处理后的结果
    return result
```