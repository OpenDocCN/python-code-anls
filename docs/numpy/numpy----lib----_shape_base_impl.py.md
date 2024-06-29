# `.\numpy\numpy\lib\_shape_base_impl.py`

```
# 导入 functools 模块，用于创建偏函数
import functools
# 导入 warnings 模块，用于警告处理
import warnings

# 导入 numpy 库的部分子模块和函数
# 从 numpy 的核心数值计算模块中导入特定函数和类
import numpy._core.numeric as _nx
# 从 numpy 的核心数值计算模块中导入 asarray、zeros、zeros_like、array、asanyarray 函数
from numpy._core.numeric import asarray, zeros, zeros_like, array, asanyarray
# 从 numpy 的核心数组处理模块中导入 reshape、transpose 函数
from numpy._core.fromnumeric import reshape, transpose
# 从 numpy 的核心多维数组模块中导入 normalize_axis_index 函数
from numpy._core.multiarray import normalize_axis_index
# 从 numpy 的核心数学计算模块中导入 _array_converter 函数
from numpy._core._multiarray_umath import _array_converter
# 从 numpy 的核心模块中导入 overrides 模块
from numpy._core import overrides
# 从 numpy 的核心模块中导入 vstack、atleast_3d 函数
from numpy._core import vstack, atleast_3d
# 从 numpy 的核心数值计算模块中导入 normalize_axis_tuple 函数
from numpy._core.numeric import normalize_axis_tuple
# 从 numpy 的核心模块中导入 set_module 函数
from numpy._core.overrides import set_module
# 从 numpy 的核心形状基础模块中导入 _arrays_for_stack_dispatcher 函数
from numpy._core.shape_base import _arrays_for_stack_dispatcher
# 从 numpy 的核心索引技巧实现模块中导入 ndindex 函数
from numpy.lib._index_tricks_impl import ndindex
# 从 numpy 的矩阵库中导入 matrix 类
from numpy.matrixlib.defmatrix import matrix  # this raises all the right alarm bells

# 定义 __all__ 变量，包含导出的函数和类名列表
__all__ = [
    'column_stack', 'row_stack', 'dstack', 'array_split', 'split',
    'hsplit', 'vsplit', 'dsplit', 'apply_over_axes', 'expand_dims',
    'apply_along_axis', 'kron', 'tile', 'take_along_axis',
    'put_along_axis'
    ]

# 创建 array_function_dispatch 变量，为 numpy 中函数分派的偏函数
array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


def _make_along_axis_idx(arr_shape, indices, axis):
    # 计算用于迭代的维度
    # 检查 indices 是否为整数数组的子类型，否则引发错误
    if not _nx.issubdtype(indices.dtype, _nx.integer):
        raise IndexError('`indices` must be an integer array')
    # 检查 indices 和 arr 是否具有相同的维度数，否则引发错误
    if len(arr_shape) != indices.ndim:
        raise ValueError(
            "`indices` and `arr` must have the same number of dimensions")
    # 创建形状全为 1 的元组 shape_ones
    shape_ones = (1,) * indices.ndim
    # 构建目标维度列表 dest_dims，插入 None 在指定的轴上
    dest_dims = list(range(axis)) + [None] + list(range(axis+1, indices.ndim))

    # 构建一个花式索引 fancy_index，包含正交 arange，请求的索引插入到正确的位置
    fancy_index = []
    for dim, n in zip(dest_dims, arr_shape):
        if dim is None:
            fancy_index.append(indices)
        else:
            ind_shape = shape_ones[:dim] + (-1,) + shape_ones[dim+1:]
            fancy_index.append(_nx.arange(n).reshape(ind_shape))

    return tuple(fancy_index)


def _take_along_axis_dispatcher(arr, indices, axis):
    # 返回 arr 和 indices 的元组，用于 take_along_axis 函数的分派
    return (arr, indices)


@array_function_dispatch(_take_along_axis_dispatcher)
def take_along_axis(arr, indices, axis):
    """
    通过匹配 1 维索引和数据切片，从输入数组中取值。

    这将沿着指定轴迭代匹配的索引和数据数组中的 1 维切片，并使用前者查找后者中的值。这些切片可以具有不同的长度。

    返回一个由匹配的数组组成的新数组。

    .. versionadded:: 1.15.0

    Parameters
    ----------
    arr : ndarray (Ni..., M, Nk...)
        源数组
    indices : ndarray (Ni..., J, Nk...)
        沿着 `arr` 的每个 1 维切片取出的索引。它必须与 `arr` 的维度匹配，
        但 Ni 和 Nj 只需要广播至 `arr`。

    """
    # normalize inputs
    # 如果 axis 参数为 None，则将输入数组视为展平后的一维数组，以保持与 `sort` 和 `argsort` 的一致性
    if axis is None:
        # 如果 indices 不是一维数组，则抛出 ValueError 异常
        if indices.ndim != 1:
            raise ValueError(
                'when axis=None, `indices` must have a single dimension.')
        arr = arr.flat  # 将 arr 展平为一维数组
        arr_shape = (len(arr),)  # 获取展平后数组的长度，因为 flatiter 没有 .shape 属性
        axis = 0  # 将 axis 设置为 0
    else:
        # 规范化 axis 参数的值，确保其在有效范围内
        axis = normalize_axis_index(axis, arr.ndim)
        arr_shape = arr.shape  # 获取数组的形状信息

    # use the fancy index
    # 使用 _make_along_axis_idx 函数根据给定的 axis 和 indices 创建沿指定轴的索引
    return arr[_make_along_axis_idx(arr_shape, indices, axis)]
# 分发器函数，返回传递给 `put_along_axis` 函数的参数元组 (arr, indices, values)
def _put_along_axis_dispatcher(arr, indices, values, axis):
    return (arr, indices, values)

# 装饰器函数，用于分发 `_put_along_axis_dispatcher` 函数的结果到 `put_along_axis` 函数
@array_function_dispatch(_put_along_axis_dispatcher)
def put_along_axis(arr, indices, values, axis):
    """
    将值插入到目标数组中，通过匹配1维索引和数据切片。

    这个函数沿着指定轴迭代匹配的1维切片，使用索引数组将值放入数据数组中。这些切片可以是不同长度的。

    返回沿轴产生索引的函数，如 `argsort` 和 `argpartition`，可以生成适合此函数使用的索引。

    .. versionadded:: 1.15.0

    Parameters
    ----------
    arr : ndarray (Ni..., M, Nk...)
        目标数组。
    indices : ndarray (Ni..., J, Nk...)
        要在 `arr` 的每个1维切片上更改的索引。它必须与 `arr` 的维度匹配，但 Ni 和 Nj 中的维度可以是 1，以便与 `arr` 广播。
    values : array_like (Ni..., J, Nk...)
        要插入到这些索引处的值。其形状和维度会广播以匹配 `indices`。
    axis : int
        要沿其获取1维切片的轴。如果 `axis` 是 None，则将目标数组视为其已创建的展平1维视图。

    Notes
    -----
    这相当于（但比）`ndindex` 和 `s_` 的以下用法更快，它设置 `ii` 和 `kk` 的每个值为索引元组：

        Ni, M, Nk = a.shape[:axis], a.shape[axis], a.shape[axis+1:]
        J = indices.shape[axis]  # 可能不等于 M

        for ii in ndindex(Ni):
            for kk in ndindex(Nk):
                a_1d       = a      [ii + s_[:,] + kk]
                indices_1d = indices[ii + s_[:,] + kk]
                values_1d  = values [ii + s_[:,] + kk]
                for j in range(J):
                    a_1d[indices_1d[j]] = values_1d[j]

    或者等效地，消除内部循环，最后两行将是：

                a_1d[indices_1d] = values_1d

    See Also
    --------
    take_along_axis :
        通过匹配1维索引和数据切片从输入数组中获取值

    Examples
    --------

    对于这个示例数组

    >>> a = np.array([[10, 30, 20], [60, 40, 50]])

    我们可以用以下代码将最大值替换为：

    >>> ai = np.argmax(a, axis=1, keepdims=True)
    >>> ai
    array([[1],
           [0]])
    >>> np.put_along_axis(a, ai, 99, axis=1)
    >>> a
    array([[10, 99, 20],
           [99, 40, 50]])

    """
    # 规范化输入
    if axis is None:
        if indices.ndim != 1:
            raise ValueError(
                '当 `axis=None` 时，`indices` 必须具有单个维度。')
        arr = arr.flat
        axis = 0
        arr_shape = (len(arr),)  # flatiter 没有 .shape
    else:
        axis = normalize_axis_index(axis, arr.ndim)
        arr_shape = arr.shape

    # 使用高级索引
    arr[_make_along_axis_idx(arr_shape, indices, axis)] = values
# 定义一个调度函数，用于应用在给定轴向上的1维切片
def _apply_along_axis_dispatcher(func1d, axis, arr, *args, **kwargs):
    # 直接返回输入数组arr
    return (arr,)


# 使用装饰器进行分派，以支持不同的数据类型和参数组合
@array_function_dispatch(_apply_along_axis_dispatcher)
def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    Apply a function to 1-D slices along the given axis.

    Execute `func1d(a, *args, **kwargs)` where `func1d` operates on 1-D arrays
    and `a` is a 1-D slice of `arr` along `axis`.

    This is equivalent to (but faster than) the following use of `ndindex` and
    `s_`, which sets each of ``ii``, ``jj``, and ``kk`` to a tuple of indices::

        Ni, Nk = a.shape[:axis], a.shape[axis+1:]
        for ii in ndindex(Ni):
            for kk in ndindex(Nk):
                f = func1d(arr[ii + s_[:,] + kk])
                Nj = f.shape
                for jj in ndindex(Nj):
                    out[ii + jj + kk] = f[jj]

    Equivalently, eliminating the inner loop, this can be expressed as::

        Ni, Nk = a.shape[:axis], a.shape[axis+1:]
        for ii in ndindex(Ni):
            for kk in ndindex(Nk):
                out[ii + s_[...,] + kk] = func1d(arr[ii + s_[:,] + kk])

    Parameters
    ----------
    func1d : function (M,) -> (Nj...)
        This function should accept 1-D arrays. It is applied to 1-D
        slices of `arr` along the specified axis.
    axis : integer
        Axis along which `arr` is sliced.
    arr : ndarray (Ni..., M, Nk...)
        Input array.
    args : any
        Additional arguments to `func1d`.
    kwargs : any
        Additional named arguments to `func1d`.

        .. versionadded:: 1.9.0


    Returns
    -------
    out : ndarray  (Ni..., Nj..., Nk...)
        The output array. The shape of `out` is identical to the shape of
        `arr`, except along the `axis` dimension. This axis is removed, and
        replaced with new dimensions equal to the shape of the return value
        of `func1d`. So if `func1d` returns a scalar `out` will have one
        fewer dimensions than `arr`.

    See Also
    --------
    apply_over_axes : Apply a function repeatedly over multiple axes.

    Examples
    --------
    >>> def my_func(a):
    ...     \"\"\"Average first and last element of a 1-D array\"\"\"
    ...     return (a[0] + a[-1]) * 0.5
    >>> b = np.array([[1,2,3], [4,5,6], [7,8,9]])
    >>> np.apply_along_axis(my_func, 0, b)
    array([4., 5., 6.])
    >>> np.apply_along_axis(my_func, 1, b)
    array([2.,  5.,  8.])

    For a function that returns a 1D array, the number of dimensions in
    `outarr` is the same as `arr`.

    >>> b = np.array([[8,1,7], [4,3,9], [5,2,6]])
    >>> np.apply_along_axis(sorted, 1, b)
    array([[1, 7, 8],
           [3, 4, 9],
           [2, 5, 6]])

    For a function that returns a higher dimensional array, those dimensions
    are inserted in place of the `axis` dimension.

    >>> b = np.array([[1,2,3], [4,5,6], [7,8,9]])
    >>> np.apply_along_axis(np.diag, -1, b)
    """
    # 函数体已经在文档字符串中详细解释了其功能和实现方式，因此无需额外注释
    array([[[1, 0, 0],
            [0, 2, 0],
            [0, 0, 3]],
           [[4, 0, 0],
            [0, 5, 0],
            [0, 0, 6]],
           [[7, 0, 0],
            [0, 8, 0],
            [0, 0, 9]]])
    """
    # 处理负轴索引
    conv = _array_converter(arr)
    # 将处理后的数组赋值给arr
    arr = conv[0]

    # 获取数组的维度
    nd = arr.ndim
    # 规范化轴索引，确保axis在有效范围内
    axis = normalize_axis_index(axis, nd)

    # 将迭代轴放置在数组维度的末尾
    in_dims = list(range(nd))
    inarr_view = transpose(arr, in_dims[:axis] + in_dims[axis+1:] + [axis])

    # 计算迭代轴的索引，并在末尾添加省略号，以防止0维数组退化为标量，从而修复gh-8642
    inds = ndindex(inarr_view.shape[:-1])
    inds = (ind + (Ellipsis,) for ind in inds)

    # 在第一个项目上调用函数
    try:
        ind0 = next(inds)
    except StopIteration:
        raise ValueError(
            'Cannot apply_along_axis when any iteration dimensions are 0'
        ) from None
    # 对inarr_view[ind0]应用func1d函数，*args和**kwargs是可选的额外参数
    res = asanyarray(func1d(inarr_view[ind0], *args, **kwargs))

    # 构建一个缓冲区用于存储func1d的评估结果。
    # 删除请求的轴，并在末尾添加新的轴，使得每次写操作都是连续的。
    if not isinstance(res, matrix):
        buff = zeros_like(res, shape=inarr_view.shape[:-1] + res.shape)
    else:
        # 矩阵在重塑时可能会出现问题，所以这里不保留它们。
        buff = zeros(inarr_view.shape[:-1] + res.shape, dtype=res.dtype)

    # 轴的排列，以便out = buff.transpose(buff_permute)
    buff_dims = list(range(buff.ndim))
    buff_permute = (
        buff_dims[0 : axis] +
        buff_dims[buff.ndim-res.ndim : buff.ndim] +
        buff_dims[axis : buff.ndim-res.ndim]
    )

    # 保存第一个结果，然后计算并保存所有剩余结果
    buff[ind0] = res
    for ind in inds:
        buff[ind] = asanyarray(func1d(inarr_view[ind], *args, **kwargs))

    # 对buff进行转置，使用buff_permute重新排列轴，得到最终结果res
    res = transpose(buff, buff_permute)
    # 将结果进行包装并返回
    return conv.wrap(res)
# 用于分发给定函数的轴参数，返回输入数组 `a` 的元组
def _apply_over_axes_dispatcher(func, a, axes):
    return (a,)


# 应用函数到多个轴上的分发器装饰器
@array_function_dispatch(_apply_over_axes_dispatcher)
def apply_over_axes(func, a, axes):
    """
    Apply a function repeatedly over multiple axes.

    `func` is called as `res = func(a, axis)`, where `axis` is the first
    element of `axes`.  The result `res` of the function call must have
    either the same dimensions as `a` or one less dimension.  If `res`
    has one less dimension than `a`, a dimension is inserted before
    `axis`.  The call to `func` is then repeated for each axis in `axes`,
    with `res` as the first argument.

    Parameters
    ----------
    func : function
        This function must take two arguments, `func(a, axis)`.
    a : array_like
        Input array.
    axes : array_like
        Axes over which `func` is applied; the elements must be integers.

    Returns
    -------
    apply_over_axis : ndarray
        The output array.  The number of dimensions is the same as `a`,
        but the shape can be different.  This depends on whether `func`
        changes the shape of its output with respect to its input.

    See Also
    --------
    apply_along_axis :
        Apply a function to 1-D slices of an array along the given axis.

    Notes
    -----
    This function is equivalent to tuple axis arguments to reorderable ufuncs
    with keepdims=True. Tuple axis arguments to ufuncs have been available since
    version 1.7.0.

    Examples
    --------
    >>> a = np.arange(24).reshape(2,3,4)
    >>> a
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])

    Sum over axes 0 and 2. The result has same number of dimensions
    as the original array:

    >>> np.apply_over_axes(np.sum, a, [0,2])
    array([[[ 60],
            [ 92],
            [124]]])

    Tuple axis arguments to ufuncs are equivalent:

    >>> np.sum(a, axis=(0,2), keepdims=True)
    array([[[ 60],
            [ 92],
            [124]]])

    """
    # 将输入数组 `a` 转换为 ndarray 类型
    val = asarray(a)
    # 获取输入数组 `a` 的维度数
    N = a.ndim
    # 如果轴参数 `axes` 的维度为0，则转换为元组
    if array(axes).ndim == 0:
        axes = (axes,)
    # 遍历每个轴参数 `axes` 中的轴
    for axis in axes:
        # 如果轴为负数，将其转换为对应的非负数索引
        if axis < 0:
            axis = N + axis
        # 构造函数调用的参数元组
        args = (val, axis)
        # 调用给定函数 `func`，将结果保存为 `res`
        res = func(*args)
        # 如果结果 `res` 的维度与输入数组 `val` 的维度相同，则更新 `val`
        if res.ndim == val.ndim:
            val = res
        # 否则，在指定轴前插入新维度，并更新 `val`
        else:
            res = expand_dims(res, axis)
            # 如果插入新维度后 `res` 的维度与输入数组 `val` 的维度相同，则更新 `val`
            if res.ndim == val.ndim:
                val = res
            # 否则，抛出值错误，指示函数未返回正确形状的数组
            else:
                raise ValueError("function is not returning "
                                 "an array of the correct shape")
    # 返回最终处理结果 `val`
    return val


# 用于分发给定数组和轴参数的装饰器，返回输入数组 `a` 的元组
def _expand_dims_dispatcher(a, axis):
    return (a,)


# 插入新轴以扩展数组形状的函数
@array_function_dispatch(_expand_dims_dispatcher)
def expand_dims(a, axis):
    """
    Expand the shape of an array.

    Insert a new axis that will appear at the `axis` position in the expanded
    array shape.

    Parameters
    ----------
    a : array_like
        Input array.

    """
    # 如果输入的数组 `a` 是矩阵类型，将其转换为 ndarray 类型
    if isinstance(a, matrix):
        a = asarray(a)
    else:
        # 否则，将输入的数组 `a` 转换为任意数组类型
        a = asanyarray(a)

    # 如果输入的轴 `axis` 不是 tuple 或者 list 类型，转换为 tuple 类型
    if type(axis) not in (tuple, list):
        axis = (axis,)

    # 计算输出数组的维度，为输入数组的维度加上 `axis` 的长度
    out_ndim = len(axis) + a.ndim

    # 根据输入的轴 `axis` 和输出的维度 `out_ndim`，规范化轴元组
    axis = normalize_axis_tuple(axis, out_ndim)

    # 创建一个迭代器用于输入数组 `a` 的形状
    shape_it = iter(a.shape)
    # 根据输入的轴 `axis` 和输出的维度 `out_ndim`，生成输出数组的形状
    shape = [1 if ax in axis else next(shape_it) for ax in range(out_ndim)]

    # 将输入数组 `a` 按照生成的形状 `shape` 进行重塑，并返回结果
    return a.reshape(shape)
# NOTE: Remove once deprecation period passes
# 设置模块为 "numpy"，这在 NumPy 中用于一些内部操作
@set_module("numpy")
# 定义函数 `row_stack`，用于堆叠数组的行，已被弃用，将在 NumPy 2.0 中删除，预计在 2023-08-18 弃用
def row_stack(tup, *, dtype=None, casting="same_kind"):
    # 发出警告，提示用户 `row_stack` 已弃用，建议直接使用 `np.vstack`
    warnings.warn(
        "`row_stack` alias is deprecated. "
        "Use `np.vstack` directly.",
        DeprecationWarning,
        stacklevel=2
    )
    # 调用 `vstack` 函数进行实际的堆叠操作
    return vstack(tup, dtype=dtype, casting=casting)


# 将 `row_stack` 函数的文档字符串设置为 `vstack` 函数的文档字符串
row_stack.__doc__ = vstack.__doc__


# 定义 `_column_stack_dispatcher` 函数，用于分派给 `_arrays_for_stack_dispatcher` 函数
def _column_stack_dispatcher(tup):
    return _arrays_for_stack_dispatcher(tup)


# 使用 `array_function_dispatch` 装饰器，定义 `column_stack` 函数
@array_function_dispatch(_column_stack_dispatcher)
# `column_stack` 函数用于将多个一维数组或二维数组按列堆叠成二维数组
def column_stack(tup):
    """
    Stack 1-D arrays as columns into a 2-D array.

    Take a sequence of 1-D arrays and stack them as columns
    to make a single 2-D array. 2-D arrays are stacked as-is,
    just like with `hstack`.  1-D arrays are turned into 2-D columns
    first.

    Parameters
    ----------
    tup : sequence of 1-D or 2-D arrays.
        Arrays to stack. All of them must have the same first dimension.

    Returns
    -------
    stacked : 2-D array
        The array formed by stacking the given arrays.

    See Also
    --------
    stack, hstack, vstack, concatenate

    Examples
    --------
    >>> a = np.array((1,2,3))
    >>> b = np.array((2,3,4))
    >>> np.column_stack((a,b))
    array([[1, 2],
           [2, 3],
           [3, 4]])

    """
    arrays = []
    # 遍历输入的数组序列
    for v in tup:
        # 将每个元素转换为数组对象
        arr = asanyarray(v)
        # 如果数组的维度小于 2，则将其转置为二维数组的列
        if arr.ndim < 2:
            arr = array(arr, copy=None, subok=True, ndmin=2).T
        # 将处理后的数组添加到数组列表中
        arrays.append(arr)
    # 使用 `_nx.concatenate` 函数在第二维度上连接所有数组，形成最终的堆叠数组
    return _nx.concatenate(arrays, 1)


# 定义 `_dstack_dispatcher` 函数，用于分派给 `_arrays_for_stack_dispatcher` 函数
def _dstack_dispatcher(tup):
    return _arrays_for_stack_dispatcher(tup)


# 使用 `array_function_dispatch` 装饰器，定义 `dstack` 函数
@array_function_dispatch(_dstack_dispatcher)
# `dstack` 函数用于沿着第三个轴（深度方向）堆叠数组序列
def dstack(tup):
    """
    Stack arrays in sequence depth wise (along third axis).

    This is equivalent to concatenation along the third axis after 2-D arrays
    of shape `(M,N)` have been reshaped to `(M,N,1)` and 1-D arrays of shape
    `(N,)` have been reshaped to `(1,N,1)`. Rebuilds arrays divided by
    `dsplit`.

    This function makes most sense for arrays with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions `concatenate`, `stack` and
    `block` provide more general stacking and concatenation operations.

    Parameters
    ----------
    tup : sequence of arrays
        The arrays must have the same shape along all but the third axis.
        1-D or 2-D arrays must have the same shape.

    Returns
    -------
    stacked : ndarray
        The array formed by stacking the given arrays, will be at least 3-D.

    See Also
    --------
    concatenate : Join a sequence of arrays along an existing axis.
    stack : Join a sequence of arrays along a new axis.
    block : Assemble an nd-array from nested lists of blocks.
    vstack : Stack arrays in sequence vertically (row wise).
    hstack : Stack arrays in sequence horizontally (column wise).

    """
    # 为了深度堆叠，将每个数组扩展到至少三维，然后沿第三个轴连接它们
    return concatenate([atleast_3d(_m) for _m in tup], axis=2)
    # column_stack函数：将多个一维数组按列堆叠成一个二维数组。
    # dsplit函数：沿着第三个轴（即深度方向）分割数组。

    Examples
    --------
    >>> a = np.array((1,2,3))
    >>> b = np.array((2,3,4))
    >>> np.dstack((a,b))
    array([[[1, 2],
            [2, 3],
            [3, 4]]])

    >>> a = np.array([[1],[2],[3]])
    >>> b = np.array([[2],[3],[4]])
    >>> np.dstack((a,b))
    array([[[1, 2]],
           [[2, 3]],
           [[3, 4]]])

    """
    # 将输入的数组tup中的各个数组至少转换为三维数组
    arrs = atleast_3d(*tup)
    # 如果转换后的结果不是一个元组，则将其包装成元组
    if not isinstance(arrs, tuple):
        arrs = (arrs,)
    # 沿着第三个轴（深度方向）连接所有数组，形成结果数组
    return _nx.concatenate(arrs, 2)
# 将传入的数组中所有零元素替换为与其相同类型的空数组
def _replace_zero_by_x_arrays(sub_arys):
    for i in range(len(sub_arys)):
        # 检查当前子数组的维度是否为零
        if _nx.ndim(sub_arys[i]) == 0:
            # 如果是零维数组，将其替换为一个空数组，数据类型与原数组相同
            sub_arys[i] = _nx.empty(0, dtype=sub_arys[i].dtype)
        elif _nx.sometrue(_nx.equal(_nx.shape(sub_arys[i]), 0)):
            # 如果子数组的形状中有任何一个维度为零，同样替换为一个空数组
            sub_arys[i] = _nx.empty(0, dtype=sub_arys[i].dtype)
    return sub_arys


# 将传入的参数直接返回，用于分派数组分割操作
def _array_split_dispatcher(ary, indices_or_sections, axis=None):
    return (ary, indices_or_sections)


@array_function_dispatch(_array_split_dispatcher)
def array_split(ary, indices_or_sections, axis=0):
    """
    将数组分割成多个子数组。

    请参考“split”文档。这两个函数之间唯一的区别在于，
    “array_split”允许“indices_or_sections”是一个整数，而不需要等分轴。
    对于长度为l的数组，应该分割为n个部分，它返回l % n个大小为l//n + 1的子数组，
    和其余大小为l//n的子数组。

    参见
    --------
    split : 将数组等分成多个子数组。

    示例
    --------
    >>> x = np.arange(8.0)
    >>> np.array_split(x, 3)
    [array([0.,  1.,  2.]), array([3.,  4.,  5.]), array([6.,  7.])]

    >>> x = np.arange(9)
    >>> np.array_split(x, 4)
    [array([0, 1, 2]), array([3, 4]), array([5, 6]), array([7, 8])]

    """
    try:
        Ntotal = ary.shape[axis]
    except AttributeError:
        Ntotal = len(ary)
    try:
        # 处理数组情况。
        Nsections = len(indices_or_sections) + 1
        div_points = [0] + list(indices_or_sections) + [Ntotal]
    except TypeError:
        # indices_or_sections 是标量，而不是数组。
        Nsections = int(indices_or_sections)
        if Nsections <= 0:
            raise ValueError('number sections must be larger than 0.') from None
        Neach_section, extras = divmod(Ntotal, Nsections)
        section_sizes = ([0] +
                         extras * [Neach_section+1] +
                         (Nsections-extras) * [Neach_section])
        div_points = _nx.array(section_sizes, dtype=_nx.intp).cumsum()

    sub_arys = []
    sary = _nx.swapaxes(ary, axis, 0)
    for i in range(Nsections):
        st = div_points[i]
        end = div_points[i + 1]
        sub_arys.append(_nx.swapaxes(sary[st:end], axis, 0))

    return sub_arys


# 将传入的参数直接返回，用于分派数组分割操作
def _split_dispatcher(ary, indices_or_sections, axis=None):
    return (ary, indices_or_sections)


@array_function_dispatch(_split_dispatcher)
def split(ary, indices_or_sections, axis=0):
    """
    将数组分割成多个作为视图的子数组。

    参数
    ----------
    ary : ndarray
        要分割成子数组的数组。
    try:
        # 尝试获取 `indices_or_sections` 的长度，如果无法获取则抛出 TypeError 异常
        len(indices_or_sections)
    except TypeError:
        # 如果无法获取长度，说明 `indices_or_sections` 是一个整数，将其作为 `sections`
        sections = indices_or_sections
        # 获取数组 `ary` 在指定轴 `axis` 上的长度
        N = ary.shape[axis]
        # 如果数组长度不能被 `sections` 整除，抛出 ValueError 异常
        if N % sections:
            raise ValueError(
                'array split does not result in an equal division') from None
    # 调用 `array_split` 函数进行数组分割，并返回分割后的结果
    return array_split(ary, indices_or_sections, axis)
# 定义一个分发函数，用于处理 hsplit 和 vsplit 函数的分派逻辑
def _hvdsplit_dispatcher(ary, indices_or_sections):
    return (ary, indices_or_sections)


# 使用装饰器将 _hvdsplit_dispatcher 注册为 hsplit 函数的分发函数
@array_function_dispatch(_hvdsplit_dispatcher)
def hsplit(ary, indices_or_sections):
    """
    将数组沿水平方向（列方向）分割为多个子数组。

    请参考 `split` 的文档。`hsplit` 相当于 `split`，其中 ``axis=1``，
    数组始终沿第二轴分割，对于 1-D 数组，则在 ``axis=0`` 处分割。

    See Also
    --------
    split : 将数组分割为多个等大小的子数组。

    Examples
    --------
    >>> x = np.arange(16.0).reshape(4, 4)
    >>> x
    array([[ 0.,   1.,   2.,   3.],
           [ 4.,   5.,   6.,   7.],
           [ 8.,   9.,  10.,  11.],
           [12.,  13.,  14.,  15.]])
    >>> np.hsplit(x, 2)
    [array([[  0.,   1.],
           [  4.,   5.],
           [  8.,   9.],
           [12.,  13.]]),
     array([[  2.,   3.],
           [  6.,   7.],
           [10.,  11.],
           [14.,  15.]])]
    >>> np.hsplit(x, np.array([3, 6]))
    [array([[ 0.,   1.,   2.],
           [ 4.,   5.,   6.],
           [ 8.,   9.,  10.],
           [12.,  13.,  14.]]),
     array([[ 3.],
           [ 7.],
           [11.],
           [15.]]),
     array([], shape=(4, 0), dtype=float64)]

    对于更高维度的数组，分割仍沿第二轴进行。

    >>> x = np.arange(8.0).reshape(2, 2, 2)
    >>> x
    array([[[0.,  1.],
            [2.,  3.]],
           [[4.,  5.],
            [6.,  7.]]])
    >>> np.hsplit(x, 2)
    [array([[[0.,  1.]],
           [[4.,  5.]]]),
     array([[[2.,  3.]],
           [[6.,  7.]]])]

    对于 1-D 数组，分割沿轴 0 进行。

    >>> x = np.array([0, 1, 2, 3, 4, 5])
    >>> np.hsplit(x, 2)
    [array([0, 1, 2]), array([3, 4, 5])]

    """
    # 如果数组维度为 0，则引发错误，hsplit 只适用于至少 1 维的数组
    if _nx.ndim(ary) == 0:
        raise ValueError('hsplit only works on arrays of 1 or more dimensions')
    # 如果数组维度大于 1，则使用 split 函数在第二轴（axis=1）进行分割
    if ary.ndim > 1:
        return split(ary, indices_or_sections, 1)
    else:
        # 对于 1-D 数组，使用 split 函数在轴 0 进行分割
        return split(ary, indices_or_sections, 0)


# 使用装饰器将 _hvdsplit_dispatcher 注册为 vsplit 函数的分发函数
@array_function_dispatch(_hvdsplit_dispatcher)
def vsplit(ary, indices_or_sections):
    """
    将数组沿垂直方向（行方向）分割为多个子数组。

    请参考 `split` 的文档。`vsplit` 相当于 `split`，其中 `axis=0`（默认），
    数组始终沿第一轴进行分割，无论数组的维度如何。

    See Also
    --------
    split : 将数组分割为多个等大小的子数组。

    Examples
    --------
    >>> x = np.arange(16.0).reshape(4, 4)
    >>> x
    array([[ 0.,   1.,   2.,   3.],
           [ 4.,   5.,   6.,   7.],
           [ 8.,   9.,  10.,  11.],
           [12.,  13.,  14.,  15.]])
    >>> np.vsplit(x, 2)
    [array([[0., 1., 2., 3.],
            [4., 5., 6., 7.]]),
     array([[ 8.,  9., 10., 11.],
            [12., 13., 14., 15.]])]
    >>> np.vsplit(x, np.array([3, 6]))

    """
    # vsplit 与 hsplit 类似，但始终在第一轴（axis=0）进行分割
    return split(ary, indices_or_sections, 0)
    [array([[ 0.,  1.,  2.,  3.],
            [ 4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11.]]),
     array([[12., 13., 14., 15.]]),
     array([], shape=(0, 4), dtype=float64)]

    With a higher dimensional array the split is still along the first axis.

    >>> x = np.arange(8.0).reshape(2, 2, 2)
    >>> x
    array([[[0.,  1.],
            [2.,  3.]],
           [[4.,  5.],
            [6.,  7.]]])
    >>> np.vsplit(x, 2)
    [array([[[0., 1.],
             [2., 3.]]]),
     array([[[4., 5.],
             [6., 7.]]])]

    """
    # 检查输入数组的维度是否小于2，如果是则抛出异常
    if _nx.ndim(ary) < 2:
        raise ValueError('vsplit only works on arrays of 2 or more dimensions')
    # 在第一轴（axis=0）上按照指定的 indices_or_sections 进行分割数组 ary
    return split(ary, indices_or_sections, 0)
@array_function_dispatch(_hvdsplit_dispatcher)
def dsplit(ary, indices_or_sections):
    """
    Split array into multiple sub-arrays along the 3rd axis (depth).

    Please refer to the `split` documentation.  `dsplit` is equivalent
    to `split` with ``axis=2``, the array is always split along the third
    axis provided the array dimension is greater than or equal to 3.

    See Also
    --------
    split : Split an array into multiple sub-arrays of equal size.

    Examples
    --------
    >>> x = np.arange(16.0).reshape(2, 2, 4)
    >>> x
    array([[[ 0.,   1.,   2.,   3.],
            [ 4.,   5.,   6.,   7.]],
           [[ 8.,   9.,  10.,  11.],
            [12.,  13.,  14.,  15.]]])
    >>> np.dsplit(x, 2)
    [array([[[ 0.,  1.],
            [ 4.,  5.]],
           [[ 8.,  9.],
            [12., 13.]]]), array([[[ 2.,  3.],
            [ 6.,  7.]],
           [[10., 11.],
            [14., 15.]]])]
    >>> np.dsplit(x, np.array([3, 6]))
    [array([[[ 0.,   1.,   2.],
            [ 4.,   5.,   6.]],
           [[ 8.,   9.,  10.],
            [12.,  13.,  14.]]]),
     array([[[ 3.],
            [ 7.]],
           [[11.],
            [15.]]]),
    array([], shape=(2, 2, 0), dtype=float64)]
    """
    # 检查数组的维度是否小于3，如果是则抛出异常
    if _nx.ndim(ary) < 3:
        raise ValueError('dsplit only works on arrays of 3 or more dimensions')
    # 调用split函数，按照指定的轴（第3轴）进行分割
    return split(ary, indices_or_sections, 2)


def get_array_wrap(*args):
    """Find the wrapper for the array with the highest priority.

    In case of ties, leftmost wins. If no wrapper is found, return None.

    .. deprecated:: 2.0
    """

    # 在NumPy 2.0中已弃用，将在2023年7月11日弃用
    warnings.warn(
        "`get_array_wrap` is deprecated. "
        "(deprecated in NumPy 2.0)",
        DeprecationWarning,
        stacklevel=2
    )

    # 根据数组对象的`__array_priority__`属性排序，找到优先级最高的数组包装器
    wrappers = sorted((getattr(x, '__array_priority__', 0), -i,
                 x.__array_wrap__) for i, x in enumerate(args)
                                   if hasattr(x, '__array_wrap__'))
    # 如果找到包装器，则返回优先级最高的包装器
    if wrappers:
        return wrappers[-1][-1]
    # 如果没有找到包装器，则返回None
    return None


def _kron_dispatcher(a, b):
    return (a, b)


@array_function_dispatch(_kron_dispatcher)
def kron(a, b):
    """
    Kronecker product of two arrays.

    Computes the Kronecker product, a composite array made of blocks of the
    second array scaled by the first.

    Parameters
    ----------
    a, b : array_like

    Returns
    -------
    out : ndarray

    See Also
    --------
    outer : The outer product

    Notes
    -----
    The function assumes that the number of dimensions of `a` and `b`
    are the same, if necessary prepending the smallest with ones.
    If ``a.shape = (r0,r1,..,rN)`` and ``b.shape = (s0,s1,...,sN)``,
    the Kronecker product has shape ``(r0*s0, r1*s1, ..., rN*SN)``.
    The elements are products of elements from `a` and `b`, organized
    explicitly by::

        kron(a,b)[k0,k1,...,kN] = a[i0,i1,...,iN] * b[j0,j1,...,jN]

    where::

        kt = it * st + jt,  t = 0,...,N


    """
    # 返回由两个数组的Kronecker积构成的新数组
    return _multiarray_umath.c_kron(a, b)
    """
    In the common 2-D case (N=1), the block structure can be visualized::
    
        [[ a[0,0]*b,   a[0,1]*b,  ... , a[0,-1]*b  ],
         [  ...                              ...   ],
         [ a[-1,0]*b,  a[-1,1]*b, ... , a[-1,-1]*b ]]
    
    
    Examples
    --------
    >>> np.kron([1,10,100], [5,6,7])
    array([  5,   6,   7, ..., 500, 600, 700])
    >>> np.kron([5,6,7], [1,10,100])
    array([  5,  50, 500, ...,   7,  70, 700])
    
    >>> np.kron(np.eye(2), np.ones((2,2)))
    array([[1.,  1.,  0.,  0.],
           [1.,  1.,  0.,  0.],
           [0.,  0.,  1.,  1.],
           [0.,  0.,  1.,  1.]])
    
    >>> a = np.arange(100).reshape((2,5,2,5))
    >>> b = np.arange(24).reshape((2,3,4))
    >>> c = np.kron(a,b)
    >>> c.shape
    (2, 10, 6, 20)
    >>> I = (1,3,0,2)
    >>> J = (0,2,1)
    >>> J1 = (0,) + J             # extend to ndim=4
    >>> S1 = (1,) + b.shape
    >>> K = tuple(np.array(I) * np.array(S1) + np.array(J1))
    >>> c[K] == a[I]*b[J]
    True
    
    """
    # Equalise the shapes by prepending smaller array with 1s
    # Expand shapes of both the arrays by adding new axes at odd positions for 1st array and even positions for 2nd
    # Compute the product of the modified arrays
    # The innermost array elements now contain the rows of the Kronecker product
    # Reshape the result to kron's shape, which is the product of shapes of the two arrays.
    b = asanyarray(b)
    a = array(a, copy=None, subok=True, ndmin=b.ndim)
    is_any_mat = isinstance(a, matrix) or isinstance(b, matrix)
    ndb, nda = b.ndim, a.ndim
    nd = max(ndb, nda)
    
    if (nda == 0 or ndb == 0):
        return _nx.multiply(a, b)
    
    as_ = a.shape
    bs = b.shape
    if not a.flags.contiguous:
        a = reshape(a, as_)
    if not b.flags.contiguous:
        b = reshape(b, bs)
    
    # Equalise the shapes by prepending smaller one with 1s
    as_ = (1,)*max(0, ndb-nda) + as_
    bs = (1,)*max(0, nda-ndb) + bs
    
    # Insert empty dimensions
    a_arr = expand_dims(a, axis=tuple(range(ndb-nda)))
    b_arr = expand_dims(b, axis=tuple(range(nda-ndb)))
    
    # Compute the product
    a_arr = expand_dims(a_arr, axis=tuple(range(1, nd*2, 2)))
    b_arr = expand_dims(b_arr, axis=tuple(range(0, nd*2, 2)))
    # In case of `mat`, convert result to `array`
    result = _nx.multiply(a_arr, b_arr, subok=(not is_any_mat))
    
    # Reshape back
    result = result.reshape(_nx.multiply(as_, bs))
    
    return result if not is_any_mat else matrix(result, copy=False)
# 定义一个函数 _tile_dispatcher，用于返回输入参数 A 和 reps
def _tile_dispatcher(A, reps):
    return (A, reps)

# 使用装饰器 array_function_dispatch，将 _tile_dispatcher 函数与 tile 函数关联起来
@array_function_dispatch(_tile_dispatcher)
# 定义 tile 函数，用于构造通过重复 A 组成的数组
def tile(A, reps):
    """
    Construct an array by repeating A the number of times given by reps.

    If `reps` has length ``d``, the result will have dimension of
    ``max(d, A.ndim)``.

    If ``A.ndim < d``, `A` is promoted to be d-dimensional by prepending new
    axes. So a shape (3,) array is promoted to (1, 3) for 2-D replication,
    or shape (1, 1, 3) for 3-D replication. If this is not the desired
    behavior, promote `A` to d-dimensions manually before calling this
    function.

    If ``A.ndim > d``, `reps` is promoted to `A`.ndim by prepending 1's to it.
    Thus for an `A` of shape (2, 3, 4, 5), a `reps` of (2, 2) is treated as
    (1, 1, 2, 2).

    Note : Although tile may be used for broadcasting, it is strongly
    recommended to use numpy's broadcasting operations and functions.

    Parameters
    ----------
    A : array_like
        The input array.
    reps : array_like
        The number of repetitions of `A` along each axis.

    Returns
    -------
    c : ndarray
        The tiled output array.

    See Also
    --------
    repeat : Repeat elements of an array.
    broadcast_to : Broadcast an array to a new shape

    Examples
    --------
    >>> a = np.array([0, 1, 2])
    >>> np.tile(a, 2)
    array([0, 1, 2, 0, 1, 2])
    >>> np.tile(a, (2, 2))
    array([[0, 1, 2, 0, 1, 2],
           [0, 1, 2, 0, 1, 2]])
    >>> np.tile(a, (2, 1, 2))
    array([[[0, 1, 2, 0, 1, 2]],
           [[0, 1, 2, 0, 1, 2]]])

    >>> b = np.array([[1, 2], [3, 4]])
    >>> np.tile(b, 2)
    array([[1, 2, 1, 2],
           [3, 4, 3, 4]])
    >>> np.tile(b, (2, 1))
    array([[1, 2],
           [3, 4],
           [1, 2],
           [3, 4]])

    >>> c = np.array([1,2,3,4])
    >>> np.tile(c,(4,1))
    array([[1, 2, 3, 4],
           [1, 2, 3, 4],
           [1, 2, 3, 4],
           [1, 2, 3, 4]])
    """
    try:
        # 尝试将 reps 转换为元组
        tup = tuple(reps)
    except TypeError:
        # 如果转换失败，则将 reps 包装成元组
        tup = (reps,)
    # 计算 tup 的长度
    d = len(tup)
    if all(x == 1 for x in tup) and isinstance(A, _nx.ndarray):
        # 如果 tup 中所有元素都为 1，并且 A 是 numpy 数组，则进行特定处理
        # 以确保在所有维度上重复次数为 1 时也会进行复制
        return _nx.array(A, copy=True, subok=True, ndmin=d)
    else:
        # 否则，根据输入的 A 创建一个数组，保留原始数据结构
        c = _nx.array(A, copy=None, subok=True, ndmin=d)
    if (d < c.ndim):
        # 如果 tup 的长度小于 c 的维度数，则在 tup 前面填充 1，以匹配 c 的维度
        tup = (1,)*(c.ndim-d) + tup
    # 计算输出数组的形状，通过将 c 的每个维度乘以对应的重复次数得到
    shape_out = tuple(s*t for s, t in zip(c.shape, tup))
    n = c.size
    if n > 0:
        # 如果 c 的大小大于 0，则进行重复操作
        for dim_in, nrep in zip(c.shape, tup):
            if nrep != 1:
                c = c.reshape(-1, n).repeat(nrep, 0)
            n //= dim_in
    # 最终将 c 重新整形为 shape_out 形状的数组，并返回结果
    return c.reshape(shape_out)
```