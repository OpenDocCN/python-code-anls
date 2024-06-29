# `.\numpy\numpy\lib\_arraypad_impl.py`

```
"""
The arraypad module contains a group of functions to pad values onto the edges
of an n-dimensional array.

"""
import numpy as np
from numpy._core.overrides import array_function_dispatch
from numpy.lib._index_tricks_impl import ndindex


__all__ = ['pad']


###############################################################################
# Private utility functions.


def _round_if_needed(arr, dtype):
    """
    Rounds arr inplace if destination dtype is integer.

    Parameters
    ----------
    arr : ndarray
        Input array.
    dtype : dtype
        The dtype of the destination array.
    """
    if np.issubdtype(dtype, np.integer):
        arr.round(out=arr)


def _slice_at_axis(sl, axis):
    """
    Construct tuple of slices to slice an array in the given dimension.

    Parameters
    ----------
    sl : slice
        The slice for the given dimension.
    axis : int
        The axis to which `sl` is applied. All other dimensions are left
        "unsliced".

    Returns
    -------
    sl : tuple of slices
        A tuple with slices matching `shape` in length.

    Examples
    --------
    >>> _slice_at_axis(slice(None, 3, -1), 1)
    (slice(None, None, None), slice(None, 3, -1), (...,))
    """
    return (slice(None),) * axis + (sl,) + (...,)


def _view_roi(array, original_area_slice, axis):
    """
    Get a view of the current region of interest during iterative padding.

    When padding multiple dimensions iteratively corner values are
    unnecessarily overwritten multiple times. This function reduces the
    working area for the first dimensions so that corners are excluded.

    Parameters
    ----------
    array : ndarray
        The array with the region of interest.
    original_area_slice : tuple of slices
        Denotes the area with original values of the unpadded array.
    axis : int
        The currently padded dimension assuming that `axis` is padded before
        `axis` + 1.

    Returns
    -------
    roi : ndarray
        The region of interest of the original `array`.
    """
    axis += 1
    sl = (slice(None),) * axis + original_area_slice[axis:]
    return array[sl]


def _pad_simple(array, pad_width, fill_value=None):
    """
    Pad array on all sides with either a single value or undefined values.

    Parameters
    ----------
    array : ndarray
        Array to grow.
    pad_width : sequence of tuple[int, int]
        Pad width on both sides for each dimension in `arr`.
    fill_value : scalar, optional
        If provided the padded area is filled with this value, otherwise
        the pad area left undefined.

    Returns
    -------
    padded : ndarray
        The padded array with the same dtype as`array`. Its order will default
        to C-style if `array` is not F-contiguous.
    original_area_slice : tuple
        A tuple of slices pointing to the area of the original array.
    """
    # Allocate grown array
    pad_width = np.asarray(pad_width)
    new_shape = tuple((np.array(array.shape) + pad_width.sum(axis=1)).tolist())
    padded = np.empty(new_shape, dtype=array.dtype, order='C' if not array.flags.f_contiguous else 'F')

    # Copy original array into padded array
    original_area_slice = tuple(slice(pad_width[i, 0], pad_width[i, 0] + array.shape[i]) for i in range(array.ndim))
    padded[original_area_slice] = array

    # Fill padded areas if fill_value is provided
    if fill_value is not None:
        padded[...] = fill_value

    return padded, original_area_slice
    # 计算扩展后的数组形状，将左边界、数组原始大小、右边界相加
    new_shape = tuple(
        left + size + right
        for size, (left, right) in zip(array.shape, pad_width)
    )
    # 确定数组的存储顺序为 'F'（Fortran顺序）或 'C'（C顺序）
    order = 'F' if array.flags.fnc else 'C'  # Fortran and not also C-order
    # 创建一个空数组，使用指定的数据类型和存储顺序
    padded = np.empty(new_shape, dtype=array.dtype, order=order)

    # 如果指定了填充值，用填充值填充扩展后的数组
    if fill_value is not None:
        padded.fill(fill_value)

    # 将原始数组复制到扩展后的数组的正确位置
    # 计算原始数组在扩展后数组中的切片范围
    original_area_slice = tuple(
        slice(left, left + size)
        for size, (left, right) in zip(array.shape, pad_width)
    )
    # 将原始数组复制到扩展后数组的指定切片位置
    padded[original_area_slice] = array

    # 返回扩展后的数组和原始数组在扩展后数组中的切片范围
    return padded, original_area_slice
# 设置给定维度中的空白填充区域。
def _set_pad_area(padded, axis, width_pair, value_pair):
    # 创建左侧切片，以指定维度上的前width_pair[0]个位置
    left_slice = _slice_at_axis(slice(None, width_pair[0]), axis)
    # 将左侧填充区域设置为value_pair[0]
    padded[left_slice] = value_pair[0]

    # 创建右侧切片，以指定维度上从padded.shape[axis] - width_pair[1]到结尾的位置
    right_slice = _slice_at_axis(
        slice(padded.shape[axis] - width_pair[1], None), axis)
    # 将右侧填充区域设置为value_pair[1]
    padded[right_slice] = value_pair[1]


# 从给定维度的空白填充数组中检索边缘值。
def _get_edges(padded, axis, width_pair):
    # 左侧边缘的索引为width_pair[0]
    left_index = width_pair[0]
    # 创建左侧切片，指定维度上的[left_index, left_index + 1)范围
    left_slice = _slice_at_axis(slice(left_index, left_index + 1), axis)
    # 获取左侧边缘值
    left_edge = padded[left_slice]

    # 右侧边缘的索引为padded.shape[axis] - width_pair[1]
    right_index = padded.shape[axis] - width_pair[1]
    # 创建右侧切片，指定维度上的[right_index - 1, right_index)范围
    right_slice = _slice_at_axis(slice(right_index - 1, right_index), axis)
    # 获取右侧边缘值
    right_edge = padded[right_slice]

    return left_edge, right_edge


# 在给定维度的空白填充数组中构造线性斜坡。
def _get_linear_ramps(padded, axis, width_pair, end_value_pair):
    # 获取边缘值对
    edge_pair = _get_edges(padded, axis, width_pair)

    # 生成左侧和右侧线性斜坡
    left_ramp, right_ramp = (
        np.linspace(
            start=end_value,
            stop=edge.squeeze(axis),  # 使用edge的指定维度上的值
            num=width,
            endpoint=False,
            dtype=padded.dtype,
            axis=axis
        )
        for end_value, edge, width in zip(
            end_value_pair, edge_pair, width_pair
        )
    )
    # 在指定维度上反转线性空间
    right_ramp = right_ramp[_slice_at_axis(slice(None, None, -1), axis)]
    # 返回更新后的左线性空间和右线性空间
    return left_ramp, right_ramp
# 计算空填充数组在给定维度上的统计量。

def _get_stats(padded, axis, width_pair, length_pair, stat_func):
    # 计算包含原始值区域的边界索引
    left_index = width_pair[0]
    right_index = padded.shape[axis] - width_pair[1]
    # 计算有效区域的长度
    max_length = right_index - left_index

    # 限制统计长度不超过 max_length
    left_length, right_length = length_pair
    if left_length is None or max_length < left_length:
        left_length = max_length
    if right_length is None or max_length < right_length:
        right_length = max_length

    if (left_length == 0 or right_length == 0) \
            and stat_func in {np.amax, np.amin}:
        # 如果左右统计长度为 0，并且统计函数是 np.amax 或 np.amin，
        # 抛出更具描述性的异常信息
        raise ValueError("stat_length of 0 yields no value for padding")

    # 计算左侧的统计量
    left_slice = _slice_at_axis(
        slice(left_index, left_index + left_length), axis)
    left_chunk = padded[left_slice]
    left_stat = stat_func(left_chunk, axis=axis, keepdims=True)
    _round_if_needed(left_stat, padded.dtype)

    if left_length == right_length == max_length:
        # 如果左右统计长度相等且等于 max_length，则右侧的统计量必须与左侧相同，
        # 直接返回左侧统计量
        return left_stat, left_stat

    # 计算右侧的统计量
    right_slice = _slice_at_axis(
        slice(right_index - right_length, right_index), axis)
    right_chunk = padded[right_slice]
    right_stat = stat_func(right_chunk, axis=axis, keepdims=True)
    _round_if_needed(right_stat, padded.dtype)

    return left_stat, right_stat


def _set_reflect_both(padded, axis, width_pair, method, 
                      original_period, include_edge=False):
    """
    用反射方式填充数组 `padded` 在指定轴 `axis` 上。

    Parameters
    ----------
    padded : ndarray
        任意形状的输入数组。
    axis : int
        要填充的 `padded` 的轴。
    width_pair : (int, int)
        在给定维度上标记填充区域两侧的宽度对。
    method : str
        # 控制反射方法的选择；选项为 'even' 或 'odd'。
        Controls method of reflection; options are 'even' or 'odd'.
    original_period : int
        # `arr` 的 `axis` 上数据的原始长度。
        Original length of data on `axis` of `arr`.
    include_edge : bool
        # 如果为真，则在反射中包含边缘值；否则，边缘值形成对称轴的一部分。
        If true, edge value is included in reflection, otherwise the edge
        value forms the symmetric axis to the reflection.

    Returns
    -------
    pad_amt : tuple of ints, length 2
        # 沿 `axis` 要进行填充的新索引位置。如果这两个值都为0，则在此维度上进行填充。
        New index positions of padding to do along the `axis`. If these are
        both 0, padding is done in this dimension.
    """
    left_pad, right_pad = width_pair
    # 计算未填充数组的有效长度
    old_length = padded.shape[axis] - right_pad - left_pad
    
    if include_edge:
        # 如果要包含边缘值，在计算反射量时需要进行偏移
        # 避免只使用原始区域的子集来进行包装
        old_length = old_length // original_period * original_period
        # 边缘值被包含，需要将填充量偏移1
        edge_offset = 1
    else:
        # 如果不包含边缘值，在计算反射量时也需要进行偏移
        # 避免只使用原始区域的子集来进行包装
        old_length = ((old_length - 1) // (original_period - 1)
            * (original_period - 1) + 1)
        edge_offset = 0  # 不包含边缘值，填充量无需偏移
        old_length -= 1  # 但必须从块中省略

    if left_pad > 0:
        # 在左侧使用反射值进行填充：
        # 首先限制块的大小，不能大于填充区域
        chunk_length = min(old_length, left_pad)
        # 从右到左切片，停在或靠近边缘，相对于停止开始
        stop = left_pad - edge_offset
        start = stop + chunk_length
        left_slice = _slice_at_axis(slice(start, stop, -1), axis)
        left_chunk = padded[left_slice]

        if method == "odd":
            # 反转块并与边缘对齐，如果方法为 'odd'
            edge_slice = _slice_at_axis(slice(left_pad, left_pad + 1), axis)
            left_chunk = 2 * padded[edge_slice] - left_chunk

        # 将块插入填充区域
        start = left_pad - chunk_length
        stop = left_pad
        pad_area = _slice_at_axis(slice(start, stop), axis)
        padded[pad_area] = left_chunk
        # 调整指向下一次迭代的左边缘
        left_pad -= chunk_length
    if right_pad > 0:
        # 如果右侧需要填充
        # 首先限制填充区域的长度，不能大于需要填充的长度
        chunk_length = min(old_length, right_pad)
        # 从右向左切片，从边缘或其相邻处开始，向左切片的终点相对于开始的位置
        start = -right_pad + edge_offset - 2
        stop = start - chunk_length
        # 在指定的轴上创建切片对象，用于获取右侧的数据块
        right_slice = _slice_at_axis(slice(start, stop, -1), axis)
        right_chunk = padded[right_slice]

        if method == "odd":
            # 如果使用奇数方法（odd），对块进行取反并与边缘对齐
            edge_slice = _slice_at_axis(
                slice(-right_pad - 1, -right_pad), axis)
            right_chunk = 2 * padded[edge_slice] - right_chunk

        # 将右侧数据块插入到填充区域
        start = padded.shape[axis] - right_pad
        stop = start + chunk_length
        pad_area = _slice_at_axis(slice(start, stop), axis)
        padded[pad_area] = right_chunk
        # 调整右侧填充的指针位置，为下一次迭代做准备
        right_pad -= chunk_length

    # 返回更新后的左侧和右侧填充值
    return left_pad, right_pad
# 在输入数组 `x` 上操作，将其转换为形状符合 `ndim` 维度的元组对
def _as_pairs(x, ndim, as_index=False):
    # 如果输入 `x` 是整数，则将其视为在所有维度上相同的元组对
    if isinstance(x, int):
        return tuple((x,) * ndim)
    # 如果输入 `x` 是一个可迭代对象且 `as_index` 为真，则返回其前 `ndim` 个元素作为索引
    elif isinstance(x, collections.Iterable):
        return tuple(x[:ndim]) if as_index else tuple(x)
    # 其他情况，引发类型错误
    else:
        raise TypeError('Expected integer or iterable, got {}'.format(type(x)))
    # 如果 x 为 None，则返回一个形状为 (ndim, 2) 的嵌套迭代器，每个元素为 (None, None)
    if x is None:
        # Pass through None as a special case, otherwise np.round(x) fails
        # with an AttributeError
        return ((None, None),) * ndim

    # 将 x 转换为 numpy 数组，以便后续处理
    x = np.array(x)

    # 如果需要将 x 转换为索引形式（整数并确保非负），则进行相应处理
    if as_index:
        x = np.round(x).astype(np.intp, copy=False)

    # 如果 x 的维度小于 3，进行优化处理
    if x.ndim < 3:
        # 优化：对于 x 只有一个或两个元素的情况，可能采用更快的路径处理。
        # `np.broadcast_to` 也可以处理这些情况，但目前速度较慢

        # 如果 x 只有一个元素
        if x.size == 1:
            # 将 x 拉平，确保对 x.ndim == 0, 1, 2 的情况都适用
            x = x.ravel()
            # 如果 as_index 为 True 且 x 小于 0，则抛出 ValueError
            if as_index and x < 0:
                raise ValueError("index can't contain negative values")
            # 返回形状为 (ndim, 2) 的嵌套迭代器，每个元素为 (x[0], x[0])
            return ((x[0], x[0]),) * ndim

        # 如果 x 有两个元素且不是 (2, 1) 的形状
        if x.size == 2 and x.shape != (2, 1):
            # 将 x 拉平，确保对 x[0], x[1] 的操作适用
            x = x.ravel()
            # 如果 as_index 为 True 且 x[0] 或 x[1] 小于 0，则抛出 ValueError
            if as_index and (x[0] < 0 or x[1] < 0):
                raise ValueError("index can't contain negative values")
            # 返回形状为 (ndim, 2) 的嵌套迭代器，每个元素为 (x[0], x[1])
            return ((x[0], x[1]),) * ndim

    # 如果 as_index 为 True 且 x 中最小值小于 0，则抛出 ValueError
    if as_index and x.min() < 0:
        raise ValueError("index can't contain negative values")

    # 将 x 广播到形状为 (ndim, 2)，并转换为列表形式返回
    # 使用 `tolist` 转换数组为列表似乎可以提高迭代和索引结果时的性能（见 `pad` 中的使用）
    return np.broadcast_to(x, (ndim, 2)).tolist()
# 定义一个内部函数 _pad_dispatcher，用于分发数组填充操作
def _pad_dispatcher(array, pad_width, mode=None, **kwargs):
    # 返回一个元组，其中包含输入的数组 array
    return (array,)


###############################################################################
# Public functions

# 使用装饰器 array_function_dispatch 将 _pad_dispatcher 函数与 numpy 模块关联起来
@array_function_dispatch(_pad_dispatcher, module='numpy')
# 定义公共函数 pad，用于对数组进行填充操作
def pad(array, pad_width, mode='constant', **kwargs):
    """
    Pad an array.

    Parameters
    ----------
    array : array_like of rank N
        The array to pad.
    pad_width : {sequence, array_like, int}
        Number of values padded to the edges of each axis.
        ``((before_1, after_1), ... (before_N, after_N))`` unique pad widths
        for each axis.
        ``(before, after)`` or ``((before, after),)`` yields same before
        and after pad for each axis.
        ``(pad,)`` or ``int`` is a shortcut for before = after = pad width
        for all axes.
    mode : str or function, optional
        One of the following string values or a user supplied function.

        'constant' (default)
            Pads with a constant value.
        'edge'
            Pads with the edge values of array.
        'linear_ramp'
            Pads with the linear ramp between end_value and the
            array edge value.
        'maximum'
            Pads with the maximum value of all or part of the
            vector along each axis.
        'mean'
            Pads with the mean value of all or part of the
            vector along each axis.
        'median'
            Pads with the median value of all or part of the
            vector along each axis.
        'minimum'
            Pads with the minimum value of all or part of the
            vector along each axis.
        'reflect'
            Pads with the reflection of the vector mirrored on
            the first and last values of the vector along each
            axis.
        'symmetric'
            Pads with the reflection of the vector mirrored
            along the edge of the array.
        'wrap'
            Pads with the wrap of the vector along the axis.
            The first values are used to pad the end and the
            end values are used to pad the beginning.
        'empty'
            Pads with undefined values.

            .. versionadded:: 1.17

        <function>
            Padding function, see Notes.
    stat_length : sequence or int, optional
        Used in 'maximum', 'mean', 'median', and 'minimum'.  Number of
        values at edge of each axis used to calculate the statistic value.

        ``((before_1, after_1), ... (before_N, after_N))`` unique statistic
        lengths for each axis.

        ``(before, after)`` or ``((before, after),)`` yields same before
        and after statistic lengths for each axis.

        ``(stat_length,)`` or ``int`` is a shortcut for
        ``before = after = statistic`` length for all axes.

        Default is ``None``, to use the entire axis.
    """
    constant_values : sequence or scalar, optional
        # 用于 'constant' 模式的参数。设置每个轴的填充值。
        # 格式为 ((before_1, after_1), ... (before_N, after_N))，每个轴有唯一的填充常量。
        # 如果是 (before, after) 或 ((before, after),)，则每个轴使用相同的填充常量。
        # 如果是 (constant,) 或 constant，则所有轴的填充常量相同。
        默认值为 0.

    end_values : sequence or scalar, optional
        # 用于 'linear_ramp' 模式的参数。设置线性斜坡的结束值，形成填充数组的边缘。
        # 格式为 ((before_1, after_1), ... (before_N, after_N))，每个轴有唯一的结束值。
        # 如果是 (before, after) 或 ((before, after),)，则每个轴使用相同的结束值。
        # 如果是 (constant,) 或 constant，则所有轴的结束值相同。
        默认值为 0.

    reflect_type : {'even', 'odd'}, optional
        # 用于 'reflect' 和 'symmetric' 模式的参数。
        # 'even' 表示默认的反射模式，围绕边缘值没有改变的反射。
        # 'odd' 表示扩展部分由边缘值的两倍减去反射值得到。
    # 导入 NumPy 库，用于科学计算和数组操作
    import numpy as np
    
    # 将输入参数 `array` 转换为 NumPy 数组
    array = np.asarray(array)
    
    # 将输入参数 `pad_width` 转换为 NumPy 数组
    pad_width = np.asarray(pad_width)
    
    # 检查 `pad_width` 数组元素的数据类型是否为整数，若不是则抛出类型错误异常
    if not pad_width.dtype.kind == 'i':
        raise TypeError('`pad_width` must be of integral type.')
    
    # 将 `pad_width` 数组广播为形状 (array.ndim, 2)
    pad_width = _as_pairs(pad_width, array.ndim, as_index=True)
    if callable(mode):
        # 如果 mode 是可调用对象，则使用用户提供的函数进行处理，结合 np.apply_along_axis 的旧行为
        function = mode
        # 创建一个新的零填充数组
        padded, _ = _pad_simple(array, pad_width, fill_value=0)
        # 然后沿着每个轴应用函数

        for axis in range(padded.ndim):
            # 使用 ndindex 迭代，类似于 apply_along_axis，但假设函数在填充数组上原地操作。

            # 视图，将迭代轴放在最后
            view = np.moveaxis(padded, axis, -1)

            # 计算迭代轴的索引，并添加一个尾随省略号，以防止 0 维数组衰减为标量 (gh-8642)
            inds = ndindex(view.shape[:-1])
            inds = (ind + (Ellipsis,) for ind in inds)
            for ind in inds:
                function(view[ind], pad_width[axis], axis, kwargs)

        return padded

    # 确保对于当前模式没有传递不支持的关键字参数
    allowed_kwargs = {
        'empty': [], 'edge': [], 'wrap': [],
        'constant': ['constant_values'],
        'linear_ramp': ['end_values'],
        'maximum': ['stat_length'],
        'mean': ['stat_length'],
        'median': ['stat_length'],
        'minimum': ['stat_length'],
        'reflect': ['reflect_type'],
        'symmetric': ['reflect_type'],
    }
    try:
        unsupported_kwargs = set(kwargs) - set(allowed_kwargs[mode])
    except KeyError:
        # 如果出现不支持的模式，抛出 ValueError 异常
        raise ValueError("mode '{}' is not supported".format(mode)) from None
    if unsupported_kwargs:
        # 如果有不支持的关键字参数，抛出 ValueError 异常
        raise ValueError("unsupported keyword arguments for mode '{}': {}"
                         .format(mode, unsupported_kwargs))

    # 统计函数字典，用于 mode 是 "maximum", "minimum", "mean", "median" 时选择对应的 numpy 统计函数
    stat_functions = {"maximum": np.amax, "minimum": np.amin,
                      "mean": np.mean, "median": np.median}

    # 创建具有最终形状和原始值的填充数组（填充区域未定义）
    padded, original_area_slice = _pad_simple(array, pad_width)
    # 准备在所有维度上进行迭代（使用 zip 比使用 enumerate 更可读）
    axes = range(padded.ndim)

    if mode == "constant":
        # 如果 mode 是 "constant"
        values = kwargs.get("constant_values", 0)
        values = _as_pairs(values, padded.ndim)
        for axis, width_pair, value_pair in zip(axes, pad_width, values):
            # 获取感兴趣区域（ROI），即填充数组上的视图
            roi = _view_roi(padded, original_area_slice, axis)
            # 设置填充区域的值
            _set_pad_area(roi, axis, width_pair, value_pair)

    elif mode == "empty":
        # 如果 mode 是 "empty"，则什么都不做，因为 _pad_simple 已经返回了正确的结果
        pass
    elif array.size == 0:
        # 如果数组为空
        # 只有 "constant" 和 "empty" 模式可以扩展空轴，其它模式都要求数组非空
        # -> 确保每个空轴只能 "用0填充"
        for axis, width_pair in zip(axes, pad_width):
            if array.shape[axis] == 0 and any(width_pair):
                raise ValueError(
                    "can't extend empty axis {} using modes other than "
                    "'constant' or 'empty'".format(axis)
                )
        # 通过检查，不需要进行更多操作，因为 _pad_simple 已经返回了正确的结果

    elif mode == "edge":
        # 如果模式为 "edge"
        for axis, width_pair in zip(axes, pad_width):
            # 获取填充后区域的视图
            roi = _view_roi(padded, original_area_slice, axis)
            # 获取边缘值对
            edge_pair = _get_edges(roi, axis, width_pair)
            # 设置填充区域
            _set_pad_area(roi, axis, width_pair, edge_pair)

    elif mode == "linear_ramp":
        # 如果模式为 "linear_ramp"
        # 获取端点值，默认为0
        end_values = kwargs.get("end_values", 0)
        end_values = _as_pairs(end_values, padded.ndim)
        for axis, width_pair, value_pair in zip(axes, pad_width, end_values):
            # 获取填充后区域的视图
            roi = _view_roi(padded, original_area_slice, axis)
            # 获取线性斜坡
            ramp_pair = _get_linear_ramps(roi, axis, width_pair, value_pair)
            # 设置填充区域
            _set_pad_area(roi, axis, width_pair, ramp_pair)

    elif mode in stat_functions:
        # 如果模式在统计函数中
        # 获取统计函数
        func = stat_functions[mode]
        # 获取统计长度，默认为None
        length = kwargs.get("stat_length", None)
        length = _as_pairs(length, padded.ndim, as_index=True)
        for axis, width_pair, length_pair in zip(axes, pad_width, length):
            # 获取填充后区域的视图
            roi = _view_roi(padded, original_area_slice, axis)
            # 获取统计值对
            stat_pair = _get_stats(roi, axis, width_pair, length_pair, func)
            # 设置填充区域
            _set_pad_area(roi, axis, width_pair, stat_pair)

    elif mode in {"reflect", "symmetric"}:
        # 如果模式是 "reflect" 或 "symmetric"
        # 获取反射类型，默认为 "even"
        method = kwargs.get("reflect_type", "even")
        # 是否包含边缘
        include_edge = True if mode == "symmetric" else False
        for axis, (left_index, right_index) in zip(axes, pad_width):
            if array.shape[axis] == 1 and (left_index > 0 or right_index > 0):
                # 如果数组在该轴上的形状为1，并且左右索引大于0
                # 对于 'reflect' 扩展单例维度是旧行为，它实际上应该引发错误。
                # 获取边缘值对
                edge_pair = _get_edges(padded, axis, (left_index, right_index))
                # 设置填充区域
                _set_pad_area(padded, axis, (left_index, right_index), edge_pair)
                continue

            # 获取填充后区域的视图
            roi = _view_roi(padded, original_area_slice, axis)
            while left_index > 0 or right_index > 0:
                # 反复填充，直到该维度的填充区域被反射值填满。
                # 如果填充区域大于当前维度中原始值的长度，则此过程是必要的。
                left_index, right_index = _set_reflect_both(
                    roi, axis, (left_index, right_index),
                    method, array.shape[axis], include_edge
                )
    # 如果模式为 "wrap"，则执行以下操作
    elif mode == "wrap":
        # 遍历每个轴以及其对应的填充宽度
        for axis, (left_index, right_index) in zip(axes, pad_width):
            # 获取当前轴上原始区域的视图
            roi = _view_roi(padded, original_area_slice, axis)
            # 计算当前轴上原始值的周期长度
            original_period = padded.shape[axis] - right_index - left_index
            # 当左右两侧的填充数量大于0时，进行循环填充
            while left_index > 0 or right_index > 0:
                # 调用函数 _set_wrap_both，设置填充方式为 "wrap"
                left_index, right_index = _set_wrap_both(
                    roi, axis, (left_index, right_index), original_period)

    # 返回填充后的数组
    return padded
```