# `D:\src\scipysrc\scipy\scipy\stats\_axis_nan_policy.py`

```
# 导入警告模块，用于发出运行时警告
# 导入NumPy库，并为了方便起见，仅导入其数组操作部分
# 导入用于函数文档解析和参数处理的模块和类
# 导入一些与轴相关的异常和辅助函数
# 导入inspect模块，用于获取对象的信息

# 在一维情况下，样本参数太小时，所有返回值将为NaN。
# 请参阅样本大小要求的文档。
too_small_1d_not_omit = (
    "One or more sample arguments is too small; all "
    "returned values will be NaN. "
    "See documentation for sample size requirements.")

# 在一维情况下，经过排除NaN后，一个或多个样本参数仍然太小，
# 所有返回值将为NaN。请参阅样本大小要求的文档。
too_small_1d_omit = (
    "After omitting NaNs, one or more sample arguments "
    "is too small; all returned values will be NaN. "
    "See documentation for sample size requirements.")

# 在多维情况下，所有轴切片中的一个或多个样本参数都太小，
# 返回数组的所有元素将为NaN。请参阅样本大小要求的文档。
too_small_nd_not_omit = (
    "All axis-slices of one or more sample arguments are "
    "too small; all elements of returned arrays will be NaN. "
    "See documentation for sample size requirements.")

# 在多维情况下，经过排除NaN后，一个或多个轴切片中的一个或多个样本参数仍然太小，
# 返回数组的相应元素将为NaN。请参阅样本大小要求的文档。
too_small_nd_omit = (
    "After omitting NaNs, one or more axis-slices of one "
    "or more sample arguments is too small; corresponding "
    "elements of returned arrays will be NaN. "
    "See documentation for sample size requirements.")

# 定义一个自定义的运行时警告类，用于特定的小样本警告
class SmallSampleWarning(RuntimeWarning):
    pass

# _broadcast_arrays函数：广播数组的形状，忽略指定轴的不兼容性
def _broadcast_arrays(arrays, axis=None, xp=None):
    """
    Broadcast shapes of arrays, ignoring incompatibility of specified axes
    """
    if not arrays:
        return arrays
    
    # 确定要使用的数组命名空间（NumPy或其他数组库）
    xp = array_namespace(*arrays) if xp is None else xp
    
    # 将输入的每个数组转换为所选的数组库的数组形式
    arrays = [xp.asarray(arr) for arr in arrays]
    
    # 获取输入数组的形状列表
    shapes = [arr.shape for arr in arrays]
    
    # 根据指定的轴广播数组形状，并返回新的形状列表
    new_shapes = _broadcast_shapes(shapes, axis)
    
    # 如果未指定轴，则所有数组都使用同一组广播后的形状
    if axis is None:
        new_shapes = [new_shapes]*len(arrays)
    
    # 将每个数组广播到相应的新形状，并返回广播后的数组列表
    return [xp.broadcast_to(array, new_shape)
            for array, new_shape in zip(arrays, new_shapes)]

# _broadcast_shapes函数：广播数组的形状，忽略指定轴的不兼容性
def _broadcast_shapes(shapes, axis=None):
    """
    Broadcast shapes, ignoring incompatibility of specified axes
    """
    if not shapes:
        return shapes
    
    # 输入验证：确保轴参数符合要求
    if axis is not None:
        axis = np.atleast_1d(axis)
        axis_int = axis.astype(int)
        if not np.array_equal(axis_int, axis):
            raise AxisError('`axis` must be an integer, a '
                            'tuple of integers, or `None`.')
        axis = axis_int
    
    # 确保所有形状具有相同的维数，并在需要时前置1以对齐
    n_dims = max([len(shape) for shape in shapes])
    new_shapes = np.ones((len(shapes), n_dims), dtype=int)
    for row, shape in zip(new_shapes, shapes):
        row[len(row)-len(shape):] = shape  # can't use negative indices (-0:)
    
    # 返回广播后的新形状列表
    return new_shapes

# 以下部分代码被省略，因为没有完整的代码块需要注释
    # 如果给定了轴（axis），执行以下操作
    if axis is not None:
        # 将负值的轴索引转换为正值的索引
        axis[axis < 0] = n_dims + axis[axis < 0]
        # 对轴索引进行排序
        axis = np.sort(axis)
        
        # 检查最大和最小的轴索引是否超出数组的维度范围
        if axis[-1] >= n_dims or axis[0] < 0:
            message = (f"`axis` is out of bounds "
                       f"for array of dimension {n_dims}")
            # 如果超出范围，抛出 AxisError 异常
            raise AxisError(message)

        # 检查轴索引是否包含重复元素
        if len(np.unique(axis)) != len(axis):
            # 如果有重复元素，抛出 AxisError 异常
            raise AxisError("`axis` must contain only distinct elements")

        # 从 new_shapes 中移除指定的轴，并将这些移除的形状保存起来
        removed_shapes = new_shapes[:, axis]
        new_shapes = np.delete(new_shapes, axis, axis=1)

    # 如果数组是可广播的，可以将形状中的 1 替换为相应的非 1 形状元素。
    # 假设数组是可广播的，可以通过以下方式找到最终的形状元素：
    new_shape = np.max(new_shapes, axis=0)
    
    # 除非数组为空，否则可以用以下方法找到最终形状元素：
    new_shape *= new_shapes.all(axis=0)

    # 在所有数组中，只能有一个唯一的非 1 形状元素。
    # 因此，如果任何非 1 形状元素与上述找到的不匹配，则数组最终不可广播。
    if np.any(~((new_shapes == 1) | (new_shapes == new_shape))):
        # 如果形状不兼容以进行广播，抛出 ValueError 异常
        raise ValueError("Array shapes are incompatible for broadcasting.")

    # 如果给定了轴（axis），则执行以下操作
    if axis is not None:
        # 将被忽略的形状元素重新添加回去
        new_axis = axis - np.arange(len(axis))
        # 将重新插入后的形状元素组成元组，并返回新的形状列表
        new_shapes = [tuple(np.insert(new_shape, new_axis, removed_shape))
                      for removed_shape in removed_shapes]
        return new_shapes
    else:
        # 如果没有指定轴，则返回新形状的元组
        return tuple(new_shape)
# 广播数组的形状并删除指定的轴

def _broadcast_array_shapes_remove_axis(arrays, axis=None):
    """
    给定数组序列 `arrays` 和整数或元组 `axis`，计算在消耗/删除 `axis` 后的广播结果的形状。
    换句话说，返回沿 `axis` 向量化的 `arrays` 的假设测试的输出形状。

    Parameters
    ----------
    arrays : sequence of ndarray
        输入的数组序列
    axis : int or tuple of int, optional
        要消耗/删除的轴。默认为 None，表示不消耗/删除任何轴，不将数组展平后再广播。

    Returns
    -------
    tuple
        广播后的形状

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats._axis_nan_policy import _broadcast_array_shapes_remove_axis
    >>> a = np.zeros((5, 2, 1))
    >>> b = np.zeros((9, 3))
    >>> _broadcast_array_shapes_remove_axis((a, b), 1)
    (5, 3)
    """

    # 获取所有数组的形状
    shapes = [arr.shape for arr in arrays]
    # 调用函数计算并返回广播后的形状，删除指定的轴
    return _broadcast_shapes_remove_axis(shapes, axis)


def _broadcast_shapes_remove_axis(shapes, axis=None):
    """
    广播形状，并删除指定的轴

    Parameters
    ----------
    shapes : list of tuple
        输入的数组形状序列
    axis : int or tuple of int, optional
        要删除的轴的索引。如果为 None，则不删除任何轴。

    Returns
    -------
    tuple
        广播后的形状

    Notes
    -----
    这个函数与 `_broadcast_array_shapes_remove_axis` 类似，但是接受形状序列而不是数组本身。
    """

    # 调用函数计算广播后的形状序列
    shapes = _broadcast_shapes(shapes, axis)
    # 获取第一个形状
    shape = shapes[0]
    # 如果指定了要删除的轴，从形状中删除该轴
    if axis is not None:
        shape = np.delete(shape, axis)
    # 返回形状的元组形式
    return tuple(shape)


def _broadcast_concatenate(arrays, axis, paired=False):
    """
    沿着指定轴广播并连接数组

    Parameters
    ----------
    arrays : sequence of ndarray
        输入的数组序列
    axis : int
        沿着哪个轴连接数组
    paired : bool, optional
        是否需要配对连接，默认为 False

    Returns
    -------
    ndarray
        连接后的数组

    Notes
    -----
    这个函数用于沿着指定轴进行广播和连接数组。
    """

    # 根据是否需要配对连接调用不同的广播数组函数
    arrays = _broadcast_arrays(arrays, axis if not paired else None)
    # 在指定轴上连接数组
    res = np.concatenate(arrays, axis=axis)
    # 返回连接后的结果数组
    return res


# TODO: 添加对 `axis` 元组的支持
def _remove_nans(samples, paired):
    """
    从配对或非配对的一维样本中移除 NaN 值

    Parameters
    ----------
    samples : list of ndarray
        输入的样本数组序列
    paired : bool
        是否为配对样本

    Returns
    -------
    list of ndarray
        移除 NaN 值后的样本数组序列

    Notes
    -----
    如果 `paired` 为 False，只移除不包含 NaN 的数组。
    如果 `paired` 为 True，当任何一个部分包含 NaN 时，移除整个配对。

    可能的优化：不复制不含 NaN 的数组。
    """

    if not paired:
        # 对于非配对样本，移除不包含 NaN 的数组元素
        return [sample[~np.isnan(sample)] for sample in samples]

    # 对于配对样本，如果任何部分包含 NaN，则移除整个配对
    nans = np.isnan(samples[0])
    for sample in samples[1:]:
        nans = nans | np.isnan(sample)
    not_nans = ~nans
    return [sample[not_nans] for sample in samples]


def _remove_sentinel(samples, paired, sentinel):
    """
    从配对或非配对的一维样本中移除标记值

    Parameters
    ----------
    samples : list of ndarray
        输入的样本数组序列
    paired : bool
        是否为配对样本
    sentinel : int or float or bool
        要移除的标记值

    Returns
    -------
    list of ndarray
        移除标记值后的样本数组序列

    Notes
    -----
    可能与 `_remove_nans` 合并，但是不像简单地传递 `sentinel=np.nan` 那样简单，
    因为 `(np.nan == np.nan) is False`。

    可能的优化：不复制不含标记值的数组。
    """

    if not paired:
        # 对于非配对样本，移除不包含指定标记值的数组元素
        return [sample[sample != sentinel] for sample in samples]

    # 对于配对样本，如果任何部分包含标记值，则移除整个配对
    sentinels = (samples[0] == sentinel)
    for sample in samples[1:]:
        sentinels = sentinels | (sample == sentinel)
    not_sentinels = ~sentinels
    return [sample[not_sentinels] for sample in samples]


def _masked_arrays_2_sentinel_arrays(samples):
    """
    将 `samples` 中的掩码数组转换为普通数组，并保留值
    """
    # 掩码数组在 `samples` 中被转换为普通数组，并且保留值
    # 根据遮罩元素替换成哨兵值

    # 如果没有任何数组有遮罩，则原样返回数组
    has_mask = False
    for sample in samples:
        mask = getattr(sample, 'mask', False)  # 获取每个样本的遮罩，如果没有则为 False
        has_mask = has_mask or np.any(mask)  # 检查是否存在任何一个样本有遮罩
    if not has_mask:
        return samples, None  # 如果没有遮罩，则返回原样本数组和 None，表示没有哨兵值

    # 选择一个哨兵值。不能使用 `np.nan`，因为哨兵（遮罩）值总是被省略，但有不同的 NaN 策略。
    dtype = np.result_type(*samples)  # 确定所有样本的数据类型
    dtype = dtype if np.issubdtype(dtype, np.number) else np.float64  # 如果数据类型不是数字类型，则使用 np.float64
    for i in range(len(samples)):
        # 如果数组类型不同，则事情变得更加复杂。
        # 我们可以为每个数组选择不同的哨兵值，但是这段代码的目的是方便，而不是效率。
        samples[i] = samples[i].astype(dtype, copy=False)  # 将每个数组转换为相同的数据类型，以便进行遮罩替换

    inexact = np.issubdtype(dtype, np.inexact)
    info = np.finfo if inexact else np.iinfo
    max_possible, min_possible = info(dtype).max, info(dtype).min
    nextafter = np.nextafter if inexact else (lambda x, _: x - 1)

    sentinel = max_possible
    # 为了简单起见，min_possible/np.infs 不是候选的哨兵值
    while sentinel > min_possible:
        for sample in samples:
            if np.any(sample == sentinel):  # 选择一个新的哨兵值
                sentinel = nextafter(sentinel, -np.inf)
                break
        else:  # 当哨兵值合适时，跳出 while 循环
            break
    else:
        message = ("This function replaces masked elements with sentinel "
                   "values, but the data contains all distinct values of this "
                   "data type. Consider promoting the dtype to `np.float64`.")
        raise ValueError(message)

    # 将遮罩元素替换为哨兵值
    out_samples = []
    for sample in samples:
        mask = getattr(sample, 'mask', None)
        if mask is not None:  # 将所有有遮罩的数组转换为带有哨兵值的数组
            mask = np.broadcast_to(mask, sample.shape)
            sample = sample.data.copy() if np.any(mask) else sample.data
            sample = np.asarray(sample)  # `sample.data` 可能是一个 memoryview？
            sample[mask] = sentinel
        out_samples.append(sample)

    return out_samples, sentinel
def _check_empty_inputs(samples, axis):
    """
    Check for empty sample; return appropriate output for a vectorized hypotest
    """
    # 如果样本都不为空，则需要执行检验
    if not any(sample.size == 0 for sample in samples):
        return None
    # 否则，统计量和 p 值要么是空数组，要么是包含 NaN 的数组。生成相应的数组并返回。
    output_shape = _broadcast_array_shapes_remove_axis(samples, axis)
    output = np.ones(output_shape) * _get_nan(*samples)
    return output


def _add_reduced_axes(res, reduced_axes, keepdims):
    """
    Add reduced axes back to all the arrays in the result object
    if keepdims = True.
    """
    # 如果 keepdims 为 True，则将减少的轴添加回结果对象中的所有数组中
    return ([np.expand_dims(output, reduced_axes) 
             if not isinstance(output, int) else output for output in res]
            if keepdims else res)


# Standard docstring / signature entries for `axis`, `nan_policy`, `keepdims`
_name = 'axis'
_desc = (
    """If an int, the axis of the input along which to compute the statistic.
The statistic of each axis-slice (e.g. row) of the input will appear in a
corresponding element of the output.
If ``None``, the input will be raveled before computing the statistic."""
    .split('\n'))

# 定义参数文档和参数本身，用于描述 `axis` 参数
def _get_axis_params(default_axis=0, _name=_name, _desc=_desc):  # bind NOW
    _type = f"int or None, default: {default_axis}"
    _axis_parameter_doc = Parameter(_name, _type, _desc)
    _axis_parameter = inspect.Parameter(_name,
                                        inspect.Parameter.KEYWORD_ONLY,
                                        default=default_axis)
    return _axis_parameter_doc, _axis_parameter


_name = 'nan_policy'
_type = "{'propagate', 'omit', 'raise'}"
_desc = (
    """Defines how to handle input NaNs.

- ``propagate``: if a NaN is present in the axis slice (e.g. row) along
  which the  statistic is computed, the corresponding entry of the output
  will be NaN.
- ``omit``: NaNs will be omitted when performing the calculation.
  If insufficient data remains in the axis slice along which the
  statistic is computed, the corresponding entry of the output will be
  NaN.
- ``raise``: if a NaN is present, a ``ValueError`` will be raised."""
    .split('\n'))

# 定义参数文档和参数本身，用于描述 `nan_policy` 参数
_nan_policy_parameter_doc = Parameter(_name, _type, _desc)
_nan_policy_parameter = inspect.Parameter(_name,
                                          inspect.Parameter.KEYWORD_ONLY,
                                          default='propagate')

_name = 'keepdims'
_type = "bool, default: False"
_desc = (
    """If this is set to True, the axes which are reduced are left
in the result as dimensions with size one. With this option,
the result will broadcast correctly against the input array."""
    .split('\n'))

# 定义参数文档和参数本身，用于描述 `keepdims` 参数
_keepdims_parameter_doc = Parameter(_name, _type, _desc)
# 使用`
_keepdims_parameter = inspect.Parameter(_name,  # 创建一个 inspect.Parameter 对象，名称为 _name，表示关键字参数，默认值为 False
                                        inspect.Parameter.KEYWORD_ONLY,  # 参数为关键字参数
                                        default=False)  # 默认值为 False

_standard_note_addition = (  # 定义一个多行字符串，包含了 SciPy 1.9 之后 np.matrix 的行为说明
    """\nBeginning in SciPy 1.9, ``np.matrix`` inputs (not recommended for new
code) are converted to ``np.ndarray`` before the calculation is performed. In
this case, the output will be a scalar or ``np.ndarray`` of appropriate shape
rather than a 2D ``np.matrix``. Similarly, while masked elements of masked
arrays are ignored, the output will be a scalar or ``np.ndarray`` rather than a
masked array with ``mask=False``.""").split('\n')  # 将多行字符串按行分割成列表

def _axis_nan_policy_factory(tuple_to_result, default_axis=0,  # 定义一个工厂函数，接受多个参数，返回一个包装函数
                             n_samples=1, paired=False,  # 其中 n_samples 为样本数量，paired 表示是否成对数据
                             result_to_tuple=None, too_small=0,  # result_to_tuple 为结果转换函数，too_small 为最小值限制
                             n_outputs=2, kwd_samples=[], override=None):  # n_outputs 为输出数量，kwd_samples 为关键字样本，override 为覆盖参数
    """Factory for a wrapper that adds axis/nan_policy params to a function.

    Parameters
    ----------
    tuple_to_result : callable
        Callable that returns an object of the type returned by the function
        being wrapped (e.g. the namedtuple or dataclass returned by a
        statistical test) provided the separate components (e.g. statistic,
        pvalue).
    default_axis : int, default: 0
        The default value of the axis argument. Standard is 0 except when
        backwards compatibility demands otherwise (e.g. `None`).
    n_samples : int or callable, default: 1
        The number of data samples accepted by the function
        (e.g. `mannwhitneyu`), a callable that accepts a dictionary of
        parameters passed into the function and returns the number of data
        samples (e.g. `wilcoxon`), or `None` to indicate an arbitrary number
        of samples (e.g. `kruskal`).
    paired : {False, True}
        Whether the function being wrapped treats the samples as paired (i.e.
        corresponding elements of each sample should be considered as different
        components of the same sample.)
    result_to_tuple : callable, optional
        Function that unpacks the results of the function being wrapped into
        a tuple. This is essentially the inverse of `tuple_to_result`. Default
        is `None`, which is appropriate for statistical tests that return a
        statistic, pvalue tuple (rather than, e.g., a non-iterable datalass).
    """
    # 函数体，定义返回的包装函数的逻辑
    pass
    # 指定装饰器必须覆盖的现有行为
    temp = override or {}
    override = {'vectorization': False,
                'nan_propagation': True}
    override.update(temp)

    # 如果 result_to_tuple 为 None，则定义一个将结果转换为元组的函数
    if result_to_tuple is None:
        def result_to_tuple(res):
            return res

    # 如果 too_small 不是可调用对象，则定义一个函数检查样本是否太小
    if not callable(too_small):
        def is_too_small(samples, *ts_args, axis=-1, **ts_kwargs):
            for sample in samples:
                if sample.shape[axis] <= too_small:
                    return True
            return False
    else:
        # 如果 too_small 是可调用对象，则直接使用它
        is_too_small = too_small

    # 返回 axis_nan_policy_decorator 函数
    return axis_nan_policy_decorator
```