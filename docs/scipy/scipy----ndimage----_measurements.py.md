# `D:\src\scipysrc\scipy\scipy\ndimage\_measurements.py`

```
# 导入必要的库和模块
import numpy as np
from . import _ni_support
from . import _ni_label
from . import _nd_image
from . import _morphology

# 定义模块的公开接口列表
__all__ = ['label', 'find_objects', 'labeled_comprehension', 'sum', 'mean',
           'variance', 'standard_deviation', 'minimum', 'maximum', 'median',
           'minimum_position', 'maximum_position', 'extrema', 'center_of_mass',
           'histogram', 'watershed_ift', 'sum_labels', 'value_indices']

# 定义标签函数，用于在数组中标记特征
def label(input, structure=None, output=None):
    """
    Label features in an array.

    Parameters
    ----------
    input : array_like
        An array-like object to be labeled. Any non-zero values in `input` are
        counted as features and zero values are considered the background.
    structure : array_like, optional
        A structuring element that defines feature connections.
        `structure` must be centrosymmetric
        (see Notes).
        If no structuring element is provided,
        one is automatically generated with a squared connectivity equal to
        one.  That is, for a 2-D `input` array, the default structuring element
        is::

            [[0,1,0],
             [1,1,1],
             [0,1,0]]
    """
    output : (None, data-type, array_like), optional
        # 输出参数，可以是 None、数据类型或类似数组
        If `output` is a data type, it specifies the type of the resulting
        labeled feature array.
        If `output` is an array-like object, then `output` will be updated
        with the labeled features from this function.  This function can
        operate in-place, by passing output=input.
        Note that the output must be able to store the largest label, or this
        function will raise an Exception.

    Returns
    -------
    label : ndarray or int
        # 返回值之一是整数数组或整数
        An integer ndarray where each unique feature in `input` has a unique
        label in the returned array.
    num_features : int
        # 返回值之一是整数，表示找到的对象数量
        How many objects were found.

        If `output` is None, this function returns a tuple of
        (`labeled_array`, `num_features`).

        If `output` is a ndarray, then it will be updated with values in
        `labeled_array` and only `num_features` will be returned by this
        function.

    See Also
    --------
    find_objects : generate a list of slices for the labeled features (or
                   objects); useful for finding features' position or
                   dimensions

    Notes
    -----
    A centrosymmetric matrix is a matrix that is symmetric about the center.
    See [1]_ for more information.

    The `structure` matrix must be centrosymmetric to ensure
    two-way connections.
    For instance, if the `structure` matrix is not centrosymmetric
    and is defined as::

        [[0,1,0],
         [1,1,0],
         [0,0,0]]

    and the `input` is::

        [[1,2],
         [0,3]]

    then the structure matrix would indicate the
    entry 2 in the input is connected to 1,
    but 1 is not connected to 2.

    References
    ----------
    .. [1] James R. Weaver, "Centrosymmetric (cross-symmetric)
       matrices, their basic properties, eigenvalues, and
       eigenvectors." The American Mathematical Monthly 92.10
       (1985): 711-717.

    Examples
    --------
    Create an image with some features, then label it using the default
    (cross-shaped) structuring element:

    >>> from scipy.ndimage import label, generate_binary_structure
    >>> import numpy as np
    >>> a = np.array([[0,0,1,1,0,0],
    ...               [0,0,0,1,0,0],
    ...               [1,1,0,0,1,0],
    ...               [0,0,0,1,0,0]])
    >>> labeled_array, num_features = label(a)

    Each of the 4 features are labeled with a different integer:

    >>> num_features
    4
    >>> labeled_array
    array([[0, 0, 1, 1, 0, 0],
           [0, 0, 0, 1, 0, 0],
           [2, 2, 0, 0, 3, 0],
           [0, 0, 0, 4, 0, 0]])

    Generate a structuring element that will consider features connected even
    if they touch diagonally:

    >>> s = generate_binary_structure(2,2)

    or,

    >>> s = [[1,1,1],
    ...      [1,1,1],
    ...      [1,1,1]]

    Label the image using the new structuring element:

    >>> labeled_array, num_features = label(a, structure=s)
    # 将输入转换为 NumPy 数组
    input = np.asarray(input)
    
    # 检查输入是否为复数类型，如果是，则引发类型错误
    if np.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    
    # 如果未提供结构参数，则生成一个默认的二进制结构
    if structure is None:
        structure = _morphology.generate_binary_structure(input.ndim, 1)
    
    # 将结构参数转换为布尔类型的 NumPy 数组
    structure = np.asarray(structure, dtype=bool)
    
    # 检查结构参数的维度是否与输入的维度相同，如果不同，则引发运行时错误
    if structure.ndim != input.ndim:
        raise RuntimeError('structure and input must have equal rank')
    
    # 检查结构参数的每个维度是否都为 3，如果不是，则引发值错误
    for ii in structure.shape:
        if ii != 3:
            raise ValueError('structure dimensions must be equal to 3')
    
    # 如果输入图像的大小超过 2^31 - 2，则需要使用 64 位整数
    need_64bits = input.size >= (2**31 - 2)
    
    # 如果调用方提供了输出数组，并且其形状与输入数组不匹配，则引发值错误
    if isinstance(output, np.ndarray):
        if output.shape != input.shape:
            raise ValueError("output shape not correct")
        caller_provided_output = True
    else:
        caller_provided_output = False
        # 如果未提供输出数组，则根据需要选择使用 intp 或 int32 类型创建空数组
        if output is None:
            output = np.empty(input.shape, np.intp if need_64bits else np.int32)
        else:
            output = np.empty(input.shape, output)
    
    # 处理标量或者零维数组的情况
    if input.ndim == 0 or input.size == 0:
        if input.ndim == 0:
            # 对于标量，最大标签值为 1 或 0
            maxlabel = 1 if (input != 0) else 0
            output[...] = maxlabel
        else:
            # 对于零维数组，最大标签值为 0
            maxlabel = 0
        if caller_provided_output:
            # 如果调用方提供了输出数组，则返回最大标签值
            return maxlabel
        else:
            # 否则返回输出数组和最大标签值
            return output, maxlabel
    
    try:
        # 尝试使用 _ni_label._label() 标签函数进行标记
        max_label = _ni_label._label(input, structure, output)
    except _ni_label.NeedMoreBits as e:
        # 如果需要更多比特，则使用足够的位数创建临时输出数组
        tmp_output = np.empty(input.shape, np.intp if need_64bits else np.int32)
        max_label = _ni_label._label(input, structure, tmp_output)
        # 将临时输出数组的内容复制到输出数组
        output[...] = tmp_output[...]
        # 如果输出数组与临时输出数组不全等，则引发运行时错误
        if not np.all(output == tmp_output):
            raise RuntimeError(
                "insufficient bit-depth in requested output type"
            ) from e
    
    if caller_provided_output:
        # 如果调用方提供了输出数组，则返回最大标签值
        return max_label
    else:
        # 否则返回输出数组和最大标签值
        return output, max_label
def find_objects(input, max_label=0):
    """
    Find objects in a labeled array.

    Parameters
    ----------
    input : ndarray of ints
        Array containing objects defined by different labels. Labels with
        value 0 are ignored.
    max_label : int, optional
        Maximum label to be searched for in `input`. If max_label is not
        given, the positions of all objects are returned.

    Returns
    -------
    object_slices : list of tuples
        A list of tuples, with each tuple containing N slices (with N the
        dimension of the input array). Slices correspond to the minimal
        parallelepiped that contains the object. If a number is missing,
        None is returned instead of a slice. The label ``l`` corresponds to
        the index ``l-1`` in the returned list.

    See Also
    --------
    label, center_of_mass

    Notes
    -----
    This function is very useful for isolating a volume of interest inside
    a 3-D array, that cannot be "seen through".

    Examples
    --------
    >>> from scipy import ndimage
    >>> import numpy as np
    >>> a = np.zeros((6,6), dtype=int)
    >>> a[2:4, 2:4] = 1
    >>> a[4, 4] = 1
    >>> a[:2, :3] = 2
    >>> a[0, 5] = 3
    >>> a
    array([[2, 2, 2, 0, 0, 3],
           [2, 2, 2, 0, 0, 0],
           [0, 0, 1, 1, 0, 0],
           [0, 0, 1, 1, 0, 0],
           [0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0]])
    >>> ndimage.find_objects(a)
    [(slice(2, 5, None), slice(2, 5, None)),
     (slice(0, 2, None), slice(0, 3, None)),
     (slice(0, 1, None), slice(5, 6, None))]
    >>> ndimage.find_objects(a, max_label=2)
    [(slice(2, 5, None), slice(2, 5, None)), (slice(0, 2, None), slice(0, 3, None))]
    >>> ndimage.find_objects(a == 1, max_label=2)
    [(slice(2, 5, None), slice(2, 5, None)), None]

    >>> loc = ndimage.find_objects(a)[0]
    >>> a[loc]
    array([[1, 1, 0],
           [1, 1, 0],
           [0, 0, 1]])

    """
    # 将输入数组转换为 NumPy 数组
    input = np.asarray(input)
    # 检查输入数组是否为复数类型，如果是则引发异常
    if np.iscomplexobj(input):
        raise TypeError('Complex type not supported')

    # 如果未指定最大标签，则将其设为输入数组中的最大值
    if max_label < 1:
        max_label = input.max()

    # 调用 SciPy 的内部函数 _nd_image.find_objects 来查找对象
    return _nd_image.find_objects(input, max_label)


def value_indices(arr, *, ignore_value=None):
    """
    Find indices of each distinct value in given array.

    Parameters
    ----------
    arr : ndarray of ints
        Array containing integer values.
    ignore_value : int, optional
        This value will be ignored in searching the `arr` array. If not
        given, all values found will be included in output. Default
        is None.

    Returns
    -------
    indices : dictionary
        A Python dictionary of array indices for each distinct value. The
        dictionary is keyed by the distinct values, the entries are array
        index tuples covering all occurrences of the value within the
        array.

        This dictionary can occupy significant memory, usually several times
        the size of the input array.

    See Also
    --------

    """
    # 处理忽略值为 None 的情况，以简化 C 代码的复杂性。如果 ignore_value 不是 None，
    # 则将其作为与 arr 相同 dtype 的 numpy 数组传入。
    ignore_value_arr = np.zeros((1,), dtype=arr.dtype)
    ignoreIsNone = (ignore_value is None)
    # 如果 ignoreIsNone 不为真，则将 ignore_value_arr 数组的第一个元素设为 ignore_value 的数据类型
    if not ignoreIsNone:
        ignore_value_arr[0] = ignore_value_arr.dtype.type(ignore_value)

    # 调用 _nd_image 模块中的 value_indices 函数，传入 arr 数组、ignoreIsNone 值和 ignore_value_arr 数组
    val_indices = _nd_image.value_indices(arr, ignoreIsNone, ignore_value_arr)
    
    # 返回 value_indices 函数的结果
    return val_indices
# 定义一个函数 `labeled_comprehension`，用于处理带标签的压缩操作
def labeled_comprehension(input, labels, index, func, out_dtype, default,
                          pass_positions=False):
    """
    Roughly equivalent to [func(input[labels == i]) for i in index].

    Sequentially applies an arbitrary function (that works on array_like input)
    to subsets of an N-D image array specified by `labels` and `index`.
    The option exists to provide the function with positional parameters as the
    second argument.

    Parameters
    ----------
    input : array_like
        Data from which to select `labels` to process.
    labels : array_like or None
        Labels to objects in `input`.
        If not None, array must be same shape as `input`.
        If None, `func` is applied to raveled `input`.
    index : int, sequence of ints or None
        Subset of `labels` to which to apply `func`.
        If a scalar, a single value is returned.
        If None, `func` is applied to all non-zero values of `labels`.
    func : callable
        Python function to apply to `labels` from `input`.
    out_dtype : dtype
        Dtype to use for `result`.
    default : int, float or None
        Default return value when an element of `index` does not exist
        in `labels`.
    pass_positions : bool, optional
        If True, pass linear indices to `func` as a second argument.
        Default is False.

    Returns
    -------
    result : ndarray
        Result of applying `func` to each of `labels` to `input` in `index`.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, 2, 0, 0],
    ...               [5, 3, 0, 4],
    ...               [0, 0, 0, 7],
    ...               [9, 3, 0, 0]])
    >>> from scipy import ndimage
    >>> lbl, nlbl = ndimage.label(a)
    >>> lbls = np.arange(1, nlbl+1)
    >>> ndimage.labeled_comprehension(a, lbl, lbls, np.mean, float, 0)
    array([ 2.75,  5.5 ,  6.  ])

    Falling back to `default`:

    >>> lbls = np.arange(1, nlbl+2)
    >>> ndimage.labeled_comprehension(a, lbl, lbls, np.mean, float, -1)
    array([ 2.75,  5.5 ,  6.  , -1.  ])

    Passing positions:

    >>> def fn(val, pos):
    ...     print("fn says: %s : %s" % (val, pos))
    ...     return (val.sum()) if (pos.sum() % 2 == 0) else (-val.sum())
    ...
    >>> ndimage.labeled_comprehension(a, lbl, lbls, fn, float, 0, True)
    fn says: [1 2 5 3] : [0 1 4 5]
    fn says: [4 7] : [ 7 11]
    fn says: [9 3] : [12 13]
    array([ 11.,  11., -12.,   0.])

    """

    # 检查 `index` 是否是标量
    as_scalar = np.isscalar(index)
    # 将输入数据转换为 NumPy 数组
    input = np.asarray(input)

    # 如果 `pass_positions` 参数为 True，生成位置索引数组
    if pass_positions:
        positions = np.arange(input.size).reshape(input.shape)

    # 如果 `labels` 为 None，处理条件
    if labels is None:
        # 如果 `index` 不为 None，则引发错误
        if index is not None:
            raise ValueError("index without defined labels")
        # 如果不传递位置信息，直接对扁平化的输入数据应用函数 `func`
        if not pass_positions:
            return func(input.ravel())
        else:
            # 否则，将位置信息也传递给 `func`
            return func(input.ravel(), positions.ravel())

    try:
        # 广播 `input` 和 `labels` 数组，使它们具有相同的形状
        input, labels = np.broadcast_arrays(input, labels)
    # 如果捕获到值错误异常，则抛出带有详细信息的值错误异常
    except ValueError as e:
        raise ValueError("input and labels must have the same shape "
                            "(excepting dimensions with width 1)") from e

    # 如果索引为 None
    if index is None:
        # 如果不传递位置信息
        if not pass_positions:
            # 返回函数应用于输入中标签大于0的部分的结果
            return func(input[labels > 0])
        else:
            # 返回函数应用于输入中标签大于0的部分以及对应位置的结果
            return func(input[labels > 0], positions[labels > 0])

    # 将索引转换为至少是一维的数组
    index = np.atleast_1d(index)
    # 如果索引数组中有任何值不能以标签数组的数据类型转换为索引数组的数据类型，抛出值错误异常
    if np.any(index.astype(labels.dtype).astype(index.dtype) != index):
        raise ValueError(f"Cannot convert index values from <{index.dtype}> to "
                         f"<{labels.dtype}> (labels' type) without loss of precision")

    # 将索引数组转换为标签数组的数据类型
    index = index.astype(labels.dtype)

    # 优化步骤：找到索引中的最小值和最大值，选择标签、输入和位置中对应部分
    lo = index.min()
    hi = index.max()
    # 创建一个掩码，用于选择标签、输入和位置中在最小值和最大值之间的部分
    mask = (labels >= lo) & (labels <= hi)

    # 对标签、输入和位置进行切片，以便只保留最小值和最大值之间的部分
    labels = labels[mask]
    input = input[mask]
    if pass_positions:
        positions = positions[mask]

    # 按照标签对所有数组进行排序
    label_order = labels.argsort()
    labels = labels[label_order]
    input = input[label_order]
    if pass_positions:
        positions = positions[label_order]

    # 对索引进行排序，并记录排序后的索引
    index_order = index.argsort()
    sorted_index = index[index_order]

    # 定义一个函数，用于将函数应用于输入数据和输出数组
    def do_map(inputs, output):
        """labels must be sorted"""
        # 计算索引数组中的元素个数
        nidx = sorted_index.size

        # 找到每个连续标签区间的边界
        lo = np.searchsorted(labels, sorted_index, side='left')
        hi = np.searchsorted(labels, sorted_index, side='right')

        # 遍历索引数组，应用函数到每个标签区间
        for i, l, h in zip(range(nidx), lo, hi):
            if l == h:
                continue
            output[i] = func(*[inp[l:h] for inp in inputs])

    # 创建一个临时数组，用于存储结果，默认初始化为指定的默认值
    temp = np.empty(index.shape, out_dtype)
    temp[:] = default
    # 如果不传递位置信息，将函数应用于输入数组并存储结果到临时数组中
    if not pass_positions:
        do_map([input], temp)
    else:
        # 否则，将函数应用于输入数组和位置数组，并存储结果到临时数组中
        do_map([input, positions], temp)

    # 创建一个输出数组，用于存储最终结果，并根据索引排序将临时数组的结果复制到输出数组中
    output = np.zeros(index.shape, out_dtype)
    output[index_order] = temp
    # 如果需要将输出作为标量返回，则返回输出数组的第一个元素
    if as_scalar:
        output = output[0]

    # 返回最终的输出数组
    return output
def _safely_castable_to_int(dt):
    """Test whether the NumPy data type `dt` can be safely cast to an int."""
    # 获取整数类型的字节大小
    int_size = np.dtype(int).itemsize
    # 检查是否可以安全地将数据类型 `dt` 强制转换为整数类型
    safe = ((np.issubdtype(dt, np.signedinteger) and dt.itemsize <= int_size) or
            (np.issubdtype(dt, np.unsignedinteger) and dt.itemsize < int_size))
    return safe


def _stats(input, labels=None, index=None, centered=False):
    """Count, sum, and optionally compute (sum - centre)^2 of input by label

    Parameters
    ----------
    input : array_like, N-D
        The input data to be analyzed.
    labels : array_like (N-D), optional
        The labels of the data in `input`. This array must be broadcast
        compatible with `input`; typically, it is the same shape as `input`.
        If `labels` is None, all nonzero values in `input` are treated as
        the single labeled group.
    index : label or sequence of labels, optional
        These are the labels of the groups for which the stats are computed.
        If `index` is None, the stats are computed for the single group where
        `labels` is greater than 0.
    centered : bool, optional
        If True, the centered sum of squares for each labeled group is
        also returned. Default is False.

    Returns
    -------
    counts : int or ndarray of ints
        The number of elements in each labeled group.
    sums : scalar or ndarray of scalars
        The sums of the values in each labeled group.
    sums_c : scalar or ndarray of scalars, optional
        The sums of mean-centered squares of the values in each labeled group.
        This is only returned if `centered` is True.

    """
    def single_group(vals):
        """Compute statistics for a single group of values."""
        if centered:
            # 计算均值中心化后的平方和
            vals_c = vals - vals.mean()
            return vals.size, vals.sum(), (vals_c * vals_c.conjugate()).sum()
        else:
            return vals.size, vals.sum()

    if labels is None:
        # 若没有提供标签，统计整个输入数据的信息
        return single_group(input)

    # 确保输入数据和标签的大小匹配
    input, labels = np.broadcast_arrays(input, labels)

    if index is None:
        # 若未指定索引，统计标签大于0的单个组的信息
        return single_group(input[labels > 0])

    if np.isscalar(index):
        # 若索引是标量，统计指定标签的单个组的信息
        return single_group(input[labels == index])

    def _sum_centered(labels):
        """Compute sum of mean-centered squares for labeled groups."""
        # `labels` 应为与 `input` 相同形状的 ndarray，包含标签索引
        means = sums / counts
        centered_input = input - means[labels]
        # 使用 `np.bincount` 计算均值中心化后的平方和
        bc = np.bincount(labels.ravel(),
                         weights=(centered_input *
                                  centered_input.conjugate()).ravel())
        return bc

    # 若需要，将标签重新映射为唯一整数
    # 检查标签是否无法安全转换为整数、最小值小于0或最大值超出标签数组大小
    if (not _safely_castable_to_int(labels.dtype) or
            labels.min() < 0 or labels.max() > labels.size):
        # 使用 np.unique 生成唯一的标签索引。`new_labels` 将是一个一维数组，
        # 但应解释为展平的 N 维数组的标签索引。
        unique_labels, new_labels = np.unique(labels, return_inverse=True)
        new_labels = np.reshape(new_labels, (-1,))  # 展平数组，因为可能是 >1 维
        counts = np.bincount(new_labels)  # 计算每个标签的出现次数
        sums = np.bincount(new_labels, weights=input.ravel())  # 计算每个标签在输入中的加权和
        if centered:
            # 计算去中心化平方和。
            # 在传递给 _sum_centered 之前，我们必须将 new_labels 重塑为与 `input` 相同的 N 维形状。
            sums_c = _sum_centered(new_labels.reshape(labels.shape))
        idxs = np.searchsorted(unique_labels, index)
        # 使所有的 idxs 值有效
        idxs[idxs >= unique_labels.size] = 0
        found = (unique_labels[idxs] == index)  # 检查索引是否匹配某个唯一标签
    else:
        # 标签是 bincount 允许的整数类型，并且数量不多，因此直接调用 bincount。
        counts = np.bincount(labels.ravel())  # 计算每个标签的出现次数
        sums = np.bincount(labels.ravel(), weights=input.ravel())  # 计算每个标签在输入中的加权和
        if centered:
            sums_c = _sum_centered(labels)  # 计算去中心化的和
        # 确保所有索引值都是有效的
        idxs = np.asanyarray(index, np.int_).copy()
        found = (idxs >= 0) & (idxs < counts.size)
        idxs[~found] = 0

    # 根据索引 idxs 过滤 counts 和 sums 数组
    counts = counts[idxs]
    counts[~found] = 0
    sums = sums[idxs]
    sums[~found] = 0

    if not centered:
        return (counts, sums)  # 如果没有进行中心化，返回 counts 和 sums
    else:
        sums_c = sums_c[idxs]
        sums_c[~found] = 0
        return (counts, sums, sums_c)  # 如果进行了中心化，返回 counts、sums 和 sums_c
def sum(input, labels=None, index=None):
    """
    Calculate the sum of the values of the array.

    Notes
    -----
    This is an alias for `ndimage.sum_labels` kept for backwards compatibility
    reasons, for new code please prefer `sum_labels`.  See the `sum_labels`
    docstring for more details.

    """
    # 调用 `sum_labels` 函数并返回结果
    return sum_labels(input, labels, index)


def sum_labels(input, labels=None, index=None):
    """
    Calculate the sum of the values of the array.

    Parameters
    ----------
    input : array_like
        Values of `input` inside the regions defined by `labels`
        are summed together.
    labels : array_like of ints, optional
        Assign labels to the values of the array. Has to have the same shape as
        `input`.
    index : array_like, optional
        A single label number or a sequence of label numbers of
        the objects to be measured.

    Returns
    -------
    sum : ndarray or scalar
        An array of the sums of values of `input` inside the regions defined
        by `labels` with the same shape as `index`. If 'index' is None or scalar,
        a scalar is returned.

    See Also
    --------
    mean, median

    Examples
    --------
    >>> from scipy import ndimage
    >>> input =  [0,1,2,3]
    >>> labels = [1,1,2,2]
    >>> ndimage.sum_labels(input, labels, index=[1,2])
    [1.0, 5.0]
    >>> ndimage.sum_labels(input, labels, index=1)
    1
    >>> ndimage.sum_labels(input, labels)
    6


    """
    # 调用内部函数 `_stats` 计算统计信息，返回和值 `sum`
    count, sum = _stats(input, labels, index)
    return sum


def mean(input, labels=None, index=None):
    """
    Calculate the mean of the values of an array at labels.

    Parameters
    ----------
    input : array_like
        Array on which to compute the mean of elements over distinct
        regions.
    labels : array_like, optional
        Array of labels of same shape, or broadcastable to the same shape as
        `input`. All elements sharing the same label form one region over
        which the mean of the elements is computed.
    index : int or sequence of ints, optional
        Labels of the objects over which the mean is to be computed.
        Default is None, in which case the mean for all values where label is
        greater than 0 is calculated.

    Returns
    -------
    out : list
        Sequence of same length as `index`, with the mean of the different
        regions labeled by the labels in `index`.

    See Also
    --------
    variance, standard_deviation, minimum, maximum, sum, label

    Examples
    --------
    >>> from scipy import ndimage
    >>> import numpy as np
    >>> a = np.arange(25).reshape((5,5))
    >>> labels = np.zeros_like(a)
    >>> labels[3:5,3:5] = 1
    >>> index = np.unique(labels)
    >>> labels
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1],
           [0, 0, 0, 1, 1]])
    >>> index
    array([0, 1])
    >>> ndimage.mean(a, labels=labels, index=index)
    """
    # 在指定标签处计算输入数组的均值，并返回结果
    return NotImplemented  # Placeholder; actual implementation was truncated
    # 定义一个包含两个元素的列表，这里是一个示例数据
    [10.285714285714286, 21.0]
    
    """
    
    # 调用 _stats 函数，返回计数和总和
    count, sum = _stats(input, labels, index)
    # 将总和除以计数，并将结果转换为 numpy 的 float64 类型
    return sum / np.asanyarray(count).astype(np.float64)
# 计算N维图像数组值的方差，可选地在指定的子区域进行计算
def variance(input, labels=None, index=None):
    # 调用_stats函数计算统计信息：count为像素数目，sum为总和，sum_c_sq为中心化平方和
    count, sum, sum_c_sq = _stats(input, labels, index, centered=True)
    # 返回中心化平方和除以像素数目的结果作为方差
    return sum_c_sq / np.asanyarray(count).astype(float)


# 计算N维图像数组值的标准差，可选地在指定的子区域进行计算
def standard_deviation(input, labels=None, index=None):
    # 调用_stats函数计算统计信息：count为像素数目，sum为总和，sum_c_sq为中心化平方和
    count, sum, sum_c_sq = _stats(input, labels, index, centered=True)
    # 返回中心化平方和除以像素数目的结果的平方根作为标准差
    return np.sqrt(sum_c_sq / np.asanyarray(count).astype(float))
    # 返回给定输入、标签和索引的方差的平方根
    return np.sqrt(variance(input, labels, index))
# 将输入转换为 NumPy 数组，确保能够进行数组操作
input = np.asanyarray(input)

# 根据需求确定是否需要查找位置信息
find_positions = find_min_positions or find_max_positions
positions = None
if find_positions:
    # 创建与输入数组形状相同的索引数组
    positions = np.arange(input.size).reshape(input.shape)

# 定义内部函数，根据需求返回最小值、最大值、中位数等信息
def single_group(vals, positions):
    result = []
    if find_min:
        # 返回最小值
        result += [vals.min()]
    if find_min_positions:
        # 返回最小值的位置
        result += [positions[vals == vals.min()][0]]
    if find_max:
        # 返回最大值
        result += [vals.max()]
    if find_max_positions:
        # 返回最大值的位置
        result += [positions[vals == vals.max()][0]]
    if find_median:
        # 返回中位数
        result += [np.median(vals)]
    return result

if labels is None:
    # 如果没有标签，则对整个输入进行处理
    return single_group(input, positions)

# 确保输入数组和标签数组可以广播到相同的形状
input, labels = np.broadcast_arrays(input, labels)

if index is None:
    # 如果未指定索引，根据标签大于0的条件创建掩码
    mask = (labels > 0)
    masked_positions = None
    if find_positions:
        # 如果需要位置信息，使用掩码筛选位置
        masked_positions = positions[mask]
    return single_group(input[mask], masked_positions)

if np.isscalar(index):
    # 如果索引是标量，根据标签等于索引值创建掩码
    mask = (labels == index)
    masked_positions = None
    if find_positions:
        # 如果需要位置信息，使用掩码筛选位置
        masked_positions = positions[mask]
    return single_group(input[mask], masked_positions)

# 如果标签类型不安全地转换为整数，或者标签范围超出值的数量，重新映射标签到唯一整数
if (not _safely_castable_to_int(labels.dtype) or
        labels.min() < 0 or labels.max() > labels.size):
    # 获取唯一的标签和它们的逆向映射
    unique_labels, labels = np.unique(labels, return_inverse=True)
    idxs = np.searchsorted(unique_labels, index)

    # 确保所有的索引值有效
    idxs[idxs >= unique_labels.size] = 0
    found = (unique_labels[idxs] == index)
else:
    # 如果标签是整数类型且数量适中，则直接使用索引
    idxs = np.asanyarray(index, np.int_).copy()
    found = (idxs >= 0) & (idxs <= labels.max())

# 将未找到的索引值设置为标签的最大值加一
idxs[~ found] = labels.max() + 1

if find_median:
    # 如果需要计算中位数，则根据输入和标签的复合排序创建顺序
    order = np.lexsort((input.ravel(), labels.ravel()))
else:
    # 否则，仅根据输入数组的值进行排序
    order = input.ravel().argsort()
input = input.ravel()[order]
labels = labels.ravel()[order]
if find_positions:
    # 根据排序后的顺序重新排列位置数组
    positions = positions.ravel()[order]

result = []
if find_min:
    # 创建数组存储每个标签对应的最小值
    mins = np.zeros(labels.max() + 2, input.dtype)
    mins[labels[::-1]] = input[::-1]
    result += [mins[idxs]]
if find_min_positions:
    # 创建数组存储每个标签对应的最小值位置
    minpos = np.zeros(labels.max() + 2, int)
    minpos[labels[::-1]] = positions[::-1]
    result += [minpos[idxs]]
if find_max:
    # 创建数组存储每个标签对应的最大值
    maxs = np.zeros(labels.max() + 2, input.dtype)
    maxs[labels] = input
    result += [maxs[idxs]]
    # 如果需要找到最大位置
    if find_max_positions:
        # 创建一个数组 maxpos，长度为 labels 中的最大值加2，元素类型为整数，初始值为零
        maxpos = np.zeros(labels.max() + 2, int)
        # 将 positions 的值按照 labels 中的索引填入 maxpos 数组
        maxpos[labels] = positions
        # 将 maxpos 中 idxs 指定位置的值添加到结果列表 result 中
        result += [maxpos[idxs]]

    # 如果需要找到中位数
    if find_median:
        # 创建一个位置数组 locs，长度为 labels 的长度，元素为从 0 到 len(labels)-1 的整数
        locs = np.arange(len(labels))
        # 创建 lo 数组，长度为 labels 中的最大值加2，元素类型为整数，初始化为零，按照 labels 逆序填充
        lo = np.zeros(labels.max() + 2, np.int_)
        lo[labels[::-1]] = locs[::-1]
        # 创建 hi 数组，长度为 labels 中的最大值加2，元素类型为整数，初始化为零，按照 labels 顺序填充
        hi = np.zeros(labels.max() + 2, np.int_)
        hi[labels] = locs
        # 从 lo 和 hi 中提取出 idxs 指定位置的值
        lo = lo[idxs]
        hi = hi[idxs]
        
        # lo 是每个标签对应的输入中最小值的索引，
        # hi 是每个标签对应的输入中最大值的索引。
        # 调整它们使它们要么相等 ((hi - lo) % 2 == 0)，要么相邻 ((hi - lo) % 2 == 1)，然后求平均值。
        step = (hi - lo) // 2
        lo += step
        hi -= step
        
        # 如果输入的数据类型是整数或布尔值，避免整数溢出或布尔值加法问题
        if (np.issubdtype(input.dtype, np.integer)
                or np.issubdtype(input.dtype, np.bool_)):
            # 将 lo 和 hi 处的输入数据转换为双精度浮点型后求平均值，然后添加到结果列表 result 中
            result += [(input[lo].astype('d') + input[hi].astype('d')) / 2.0]
        else:
            # 将 lo 和 hi 处的输入数据求平均值，然后添加到结果列表 result 中
            result += [(input[lo] + input[hi]) / 2.0]

    # 返回结果列表
    return result
# 计算数组在标记区域上的最小值。
def minimum(input, labels=None, index=None):
    """
    Calculate the minimum of the values of an array over labeled regions.

    Parameters
    ----------
    input : array_like
        Array_like of values. For each region specified by `labels`, the
        minimal values of `input` over the region is computed.
    labels : array_like, optional
        An array_like of integers marking different regions over which the
        minimum value of `input` is to be computed. `labels` must have the
        same shape as `input`. If `labels` is not specified, the minimum
        over the whole array is returned.
    index : array_like, optional
        A list of region labels that are taken into account for computing the
        minima. If index is None, the minimum over all elements where `labels`
        is non-zero is returned.

    Returns
    -------
    minimum : float or list of floats
        List of minima of `input` over the regions determined by `labels` and
        whose index is in `index`. If `index` or `labels` are not specified, a
        float is returned: the minimal value of `input` if `labels` is None,
        and the minimal value of elements where `labels` is greater than zero
        if `index` is None.

    See Also
    --------
    label, maximum, median, minimum_position, extrema, sum, mean, variance,
    standard_deviation

    Notes
    -----
    The function returns a Python list and not a NumPy array, use
    `np.array` to convert the list to an array.

    Examples
    --------
    >>> from scipy import ndimage
    >>> import numpy as np
    >>> a = np.array([[1, 2, 0, 0],
    ...               [5, 3, 0, 4],
    ...               [0, 0, 0, 7],
    ...               [9, 3, 0, 0]])
    >>> labels, labels_nb = ndimage.label(a)
    >>> labels
    array([[1, 1, 0, 0],
           [1, 1, 0, 2],
           [0, 0, 0, 2],
           [3, 3, 0, 0]])
    >>> ndimage.minimum(a, labels=labels, index=np.arange(1, labels_nb + 1))
    [1.0, 4.0, 3.0]
    >>> ndimage.minimum(a)
    0.0
    >>> ndimage.minimum(a, labels=labels)
    1.0

    """
    # 调用内部函数 `_select`，返回计算出的最小值列表的第一个元素
    return _select(input, labels, index, find_min=True)[0]


def maximum(input, labels=None, index=None):
    """
    Calculate the maximum of the values of an array over labeled regions.

    Parameters
    ----------
    input : array_like
        Array_like of values. For each region specified by `labels`, the
        maximal values of `input` over the region is computed.
    labels : array_like, optional
        An array of integers marking different regions over which the
        maximum value of `input` is to be computed. `labels` must have the
        same shape as `input`. If `labels` is not specified, the maximum
        over the whole array is returned.
    index : array_like, optional
        A list of region labels that are taken into account for computing the
        maxima. If index is None, the maximum over all elements where `labels`
        is non-zero is returned.

    Returns
    -------

    """
    # 返回输入数组 `input` 在由 `labels` 确定的区域中的最大值列表或单个最大值。
    # 如果未指定 `index` 或 `labels`，则返回一个单独的浮点数：
    # 如果 `labels` 为 None，则返回 `input` 的最大值；
    # 如果 `index` 为 None，则返回在 `labels` 大于零的元素中的最大值。
    
    # 参见
    # --------
    # label, minimum, median, maximum_position, extrema, sum, mean, variance,
    # standard_deviation
    
    # 注意
    # -----
    # 此函数返回一个 Python 列表而不是 NumPy 数组，可以使用 `np.array` 将列表转换为数组。
    
    # 示例
    # --------
    # >>> import numpy as np
    # >>> a = np.arange(16).reshape((4,4))
    # >>> a
    # array([[ 0,  1,  2,  3],
    #        [ 4,  5,  6,  7],
    #        [ 8,  9, 10, 11],
    #        [12, 13, 14, 15]])
    # >>> labels = np.zeros_like(a)
    # >>> labels[:2,:2] = 1
    # >>> labels[2:, 1:3] = 2
    # >>> labels
    # array([[1, 1, 0, 0],
    #        [1, 1, 0, 0],
    #        [0, 2, 2, 0],
    #        [0, 2, 2, 0]])
    # >>> from scipy import ndimage
    # >>> ndimage.maximum(a)
    # 15.0
    # >>> ndimage.maximum(a, labels=labels, index=[1,2])
    # [5.0, 14.0]
    # >>> ndimage.maximum(a, labels=labels)
    # 14.0
    
    # >>> b = np.array([[1, 2, 0, 0],
    # ...               [5, 3, 0, 4],
    # ...               [0, 0, 0, 7],
    # ...               [9, 3, 0, 0]])
    # >>> labels, labels_nb = ndimage.label(b)
    # >>> labels
    # array([[1, 1, 0, 0],
    #        [1, 1, 0, 2],
    #        [0, 0, 0, 2],
    #        [3, 3, 0, 0]])
    # >>> ndimage.maximum(b, labels=labels, index=np.arange(1, labels_nb + 1))
    # [5.0, 7.0, 9.0]
    
    """
    return _select(input, labels, index, find_max=True)[0]
# 计算数组中指定区域的值的中位数
def median(input, labels=None, index=None):
    """
    Calculate the median of the values of an array over labeled regions.

    Parameters
    ----------
    input : array_like
        Array_like of values. For each region specified by `labels`, the
        median value of `input` over the region is computed.
    labels : array_like, optional
        An array_like of integers marking different regions over which the
        median value of `input` is to be computed. `labels` must have the
        same shape as `input`. If `labels` is not specified, the median
        over the whole array is returned.
    index : array_like, optional
        A list of region labels that are taken into account for computing the
        medians. If index is None, the median over all elements where `labels`
        is non-zero is returned.

    Returns
    -------
    median : float or list of floats
        List of medians of `input` over the regions determined by `labels` and
        whose index is in `index`. If `index` or `labels` are not specified, a
        float is returned: the median value of `input` if `labels` is None,
        and the median value of elements where `labels` is greater than zero
        if `index` is None.

    See Also
    --------
    label, minimum, maximum, extrema, sum, mean, variance, standard_deviation

    Notes
    -----
    The function returns a Python list and not a NumPy array, use
    `np.array` to convert the list to an array.

    Examples
    --------
    >>> from scipy import ndimage
    >>> import numpy as np
    >>> a = np.array([[1, 2, 0, 1],
    ...               [5, 3, 0, 4],
    ...               [0, 0, 0, 7],
    ...               [9, 3, 0, 0]])
    >>> labels, labels_nb = ndimage.label(a)
    >>> labels
    array([[1, 1, 0, 2],
           [1, 1, 0, 2],
           [0, 0, 0, 2],
           [3, 3, 0, 0]])
    >>> ndimage.median(a, labels=labels, index=np.arange(1, labels_nb + 1))
    [2.5, 4.0, 6.0]
    >>> ndimage.median(a)
    1.0
    >>> ndimage.median(a, labels=labels)
    3.0

    """
    # 调用内部函数 _select，返回计算中位数的结果列表的第一个元素
    return _select(input, labels, index, find_median=True)[0]


# 查找数组中数值的最小位置
def minimum_position(input, labels=None, index=None):
    """
    Find the positions of the minimums of the values of an array at labels.

    Parameters
    ----------
    input : array_like
        Array_like of values.
    labels : array_like, optional
        An array of integers marking different regions over which the
        position of the minimum value of `input` is to be computed.
        `labels` must have the same shape as `input`. If `labels` is not
        specified, the location of the first minimum over the whole
        array is returned.

        The `labels` argument only works when `index` is specified.

    """
    # 将输入数据的形状转换为数组，并存储在 dims 中
    dims = np.array(np.asarray(input).shape)
    # 使用 np.cumprod 计算维度的累积乘积，以及 np[::-1] 反转数组
    dim_prod = np.cumprod([1] + list(dims[:0:-1]))[::-1]

    # 调用 _select 函数，获取最小位置的结果
    result = _select(input, labels, index, find_min_positions=True)[0]

    # 如果结果是标量（单个值），则计算其在多维数组中的索引位置并返回
    if np.isscalar(result):
        return tuple((result // dim_prod) % dims

    # 如果结果是数组，则对每个结果应用索引计算，并返回位置元组列表
    return [tuple(v) for v in (result.reshape(-1, 1) // dim_prod) % dims]
def maximum_position(input, labels=None, index=None):
    """
    Find the positions of the maximums of the values of an array at labels.

    For each region specified by `labels`, the position of the maximum
    value of `input` within the region is returned.

    Parameters
    ----------
    input : array_like
        Array_like of values.
    labels : array_like, optional
        An array of integers marking different regions over which the
        position of the maximum value of `input` is to be computed.
        `labels` must have the same shape as `input`. If `labels` is not
        specified, the location of the first maximum over the whole
        array is returned.

        The `labels` argument only works when `index` is specified.
    index : array_like, optional
        A list of region labels that are taken into account for finding the
        location of the maxima. If `index` is None, the first maximum
        over all elements where `labels` is non-zero is returned.

        The `index` argument only works when `labels` is specified.

    Returns
    -------
    output : list of tuples of ints
        List of tuples of ints that specify the location of maxima of
        `input` over the regions determined by `labels` and whose index
        is in `index`.

        If `index` or `labels` are not specified, a tuple of ints is
        returned specifying the location of the ``first`` maximal value
        of `input`.

    See Also
    --------
    label, minimum, median, maximum_position, extrema, sum, mean, variance,
    standard_deviation

    Examples
    --------
    >>> from scipy import ndimage
    >>> import numpy as np
    >>> a = np.array([[1, 2, 0, 0],
    ...               [5, 3, 0, 4],
    ...               [0, 0, 0, 7],
    ...               [9, 3, 0, 0]])
    >>> ndimage.maximum_position(a)
    (3, 0)

    Features to process can be specified using `labels` and `index`:

    >>> lbl = np.array([[0, 1, 2, 3],
    ...                 [0, 1, 2, 3],
    ...                 [0, 1, 2, 3],
    ...                 [0, 1, 2, 3]])
    >>> ndimage.maximum_position(a, lbl, 1)
    (1, 1)

    If no index is given, non-zero `labels` are processed:

    >>> ndimage.maximum_position(a, lbl)
    (2, 3)

    If there are no maxima, the position of the first element is returned:

    >>> ndimage.maximum_position(a, lbl, 2)
    (0, 2)

    """
    # Compute the dimensions of the input array
    dims = np.array(np.asarray(input).shape)
    
    # Calculate the products of dimensions for unraveling indices
    dim_prod = np.cumprod([1] + list(dims[:0:-1]))[::-1]

    # Call _select function to find positions of maximum values
    result = _select(input, labels, index, find_max_positions=True)[0]

    # If the result is a scalar, compute its position
    if np.isscalar(result):
        return tuple((result // dim_prod) % dims)

    # If the result is not scalar, reshape and compute positions for each value
    return [tuple(v) for v in (result.reshape(-1, 1) // dim_prod) % dims]
    # 将输入数据转换为 numpy 数组，并获取其形状的维度
    dims = np.array(np.asarray(input).shape)
    # 计算用于多维索引转换的乘积数组，以便后续将线性索引转换为多维坐标
    dim_prod = np.cumprod([1] + list(dims[:0:-1]))[::-1]

    # 调用 _select 函数选择要处理的特征的最小值、最小位置、最大值和最大位置
    minimums, min_positions, maximums, max_positions = _select(input, labels,
                                                               index,
                                                               find_min=True,
                                                               find_max=True,
                                                               find_min_positions=True,
                                                               find_max_positions=True)

    # 如果最小值是标量，则返回单个最小值、最大值及其位置
    if np.isscalar(minimums):
        return (minimums, maximums, tuple((min_positions // dim_prod) % dims),
                tuple((max_positions // dim_prod) % dims))

    # 如果最小值不是标量，则将所有最小位置和最大位置转换为多维坐标
    min_positions = [
        tuple(v) for v in (min_positions.reshape(-1, 1) // dim_prod) % dims
    ]
    max_positions = [
        tuple(v) for v in (max_positions.reshape(-1, 1) // dim_prod) % dims
    ]

    # 返回所有最小值、最大值及其多维坐标
    return minimums, maximums, min_positions, max_positions
def center_of_mass(input, labels=None, index=None):
    """
    Calculate the center of mass of the values of an array at labels.

    Parameters
    ----------
    input : ndarray
        Data from which to calculate center-of-mass. The masses can either
        be positive or negative.
    labels : ndarray, optional
        Labels for objects in `input`, as generated by `ndimage.label`.
        Only used with `index`. Dimensions must be the same as `input`.
    index : int or sequence of ints, optional
        Labels for which to calculate centers-of-mass. If not specified,
        the combined center of mass of all labels greater than zero
        will be calculated. Only used with `labels`.

    Returns
    -------
    center_of_mass : tuple, or list of tuples
        Coordinates of centers-of-mass.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array(([0,0,0,0],
    ...               [0,1,1,0],
    ...               [0,1,1,0],
    ...               [0,1,1,0]))
    >>> from scipy import ndimage
    >>> ndimage.center_of_mass(a)
    (2.0, 1.5)

    Calculation of multiple objects in an image

    >>> b = np.array(([0,1,1,0],
    ...               [0,1,0,0],
    ...               [0,0,0,0],
    ...               [0,0,1,1],
    ...               [0,0,1,1]))
    >>> lbl = ndimage.label(b)[0]
    >>> ndimage.center_of_mass(b, lbl, [1,2])
    [(0.33333333333333331, 1.3333333333333333), (3.5, 2.5)]

    Negative masses are also accepted, which can occur for example when
    bias is removed from measured data due to random noise.

    >>> c = np.array(([-1,0,0,0],
    ...               [0,-1,-1,0],
    ...               [0,1,-1,0],
    ...               [0,1,1,0]))
    >>> ndimage.center_of_mass(c)
    (-4.0, 1.0)

    If there are division by zero issues, the function does not raise an
    error but rather issues a RuntimeWarning before returning inf and/or NaN.

    >>> d = np.array([-1, 1])
    >>> ndimage.center_of_mass(d)
    (inf,)
    """
    # 计算输入数据在给定标签下的总和，作为归一化因子
    normalizer = sum(input, labels, index)
    # 创建与输入数据形状相匹配的坐标网格
    grids = np.ogrid[[slice(0, i) for i in input.shape]]

    # 计算每个维度上的质心坐标
    results = [sum(input * grids[dir].astype(float), labels, index) / normalizer
               for dir in range(input.ndim)]

    # 如果结果是标量，则返回单个坐标元组；否则返回多个坐标元组的列表
    if np.isscalar(results[0]):
        return tuple(results)

    return [tuple(v) for v in np.array(results).T]


def histogram(input, min, max, bins, labels=None, index=None):
    """
    Calculate the histogram of the values of an array, optionally at labels.

    Histogram calculates the frequency of values in an array within bins
    determined by `min`, `max`, and `bins`. The `labels` and `index`
    keywords can limit the scope of the histogram to specified sub-regions
    within the array.

    Parameters
    ----------
    input : array_like
        Data for which to calculate histogram.
    min, max : int
        Minimum and maximum values of range of histogram bins.
    bins : int
        Number of bins.
    labels : ndarray, optional
        Labels for objects in `input`, as generated by `ndimage.label`.
        Only used with `index`. Dimensions must be the same as `input`.
    index : int or sequence of ints, optional
        Labels for which to calculate histogram. If not specified,
        the entire `input` array will be used.
    # `labels`：array_like，可选参数
    # `input` 中对象的标签。
    # 如果不为 None，则必须与 `input` 具有相同的形状。
    # `index`：int 或 int 序列，可选参数
    # 要计算直方图的标签或标签。如果为 None，则使用所有标签大于零的值。

    # 返回
    # -------
    # hist：ndarray
    # 直方图计数。

    # 示例
    # --------
    # 导入 numpy 库
    >>> import numpy as np
    # 创建示例数组 `a`
    >>> a = np.array([[ 0.    ,  0.2146,  0.5962,  0.    ],
    ...               [ 0.    ,  0.7778,  0.    ,  0.    ],
    ...               [ 0.    ,  0.    ,  0.    ,  0.    ],
    ...               [ 0.    ,  0.    ,  0.7181,  0.2787],
    ...               [ 0.    ,  0.    ,  0.6573,  0.3094]])
    # 导入 scipy 的 ndimage 模块
    >>> from scipy import ndimage
    # 对数组 `a` 计算直方图
    >>> ndimage.histogram(a, 0, 1, 10)
    # 返回数组 `[13,  0,  2,  1,  0,  1,  1,  2,  0,  0]`

    # 使用标签 `lbl` 并且没有索引，计数非零元素：
    >>> lbl, nlbl = ndimage.label(a)
    >>> ndimage.histogram(a, 0, 1, 10, lbl)
    # 返回数组 `[0, 0, 2, 1, 0, 1, 1, 2, 0, 0]`

    # 可以使用索引仅计数特定对象：
    >>> ndimage.histogram(a, 0, 1, 10, lbl, 2)
    # 返回数组 `[0, 0, 1, 1, 0, 0, 1, 1, 0, 0]`
    
    """
    # 根据指定的 `min`，`max` 和 `bins` 参数生成等间隔的 bins 数组
    _bins = np.linspace(min, max, bins + 1)

    # 定义内部函数 `_hist`，用于计算输入值 `vals` 的直方图
    def _hist(vals):
        return np.histogram(vals, _bins)[0]

    # 使用 `labeled_comprehension` 函数应用 `_hist` 函数到 `input` 上，
    # 使用指定的 `labels` 和 `index` 参数，不传递位置信息
    return labeled_comprehension(input, labels, index, _hist, object, None,
                                 pass_positions=False)
# 应用基于图像森林变换算法的分水岭算法，从标记点开始进行分水岭变换

def watershed_ift(input, markers, structure=None, output=None):
    """
    Apply watershed from markers using image foresting transform algorithm.

    Parameters
    ----------
    input : array_like
        输入图像或数组。
    markers : array_like
        标记数组，表示每个分水岭的起始点。负标记被视为背景标记，将在其他标记之后处理。
    structure : structure element, optional
        结构元素，定义对象的连通性。如果为None，则生成一个具有方形连通性的元素。
    output : ndarray, optional
        可选的输出数组，与输入具有相同的形状。

    Returns
    -------
    watershed_ift : ndarray
        输出数组，与`input`具有相同的形状。

    References
    ----------
    .. [1] A.X. Falcao, J. Stolfi and R. de Alencar Lotufo, "The image
           foresting transform: theory, algorithms, and applications",
           Pattern Analysis and Machine Intelligence, vol. 26, pp. 19-29, 2004.

    """
    # 将输入转换为NumPy数组
    input = np.asarray(input)
    # 检查输入的数据类型是否为无符号8位或16位整数类型
    if input.dtype.type not in [np.uint8, np.uint16]:
        raise TypeError('only 8 and 16 unsigned inputs are supported')

    # 如果未提供结构元素，则生成一个与输入维度相同的结构元素，其连通性为1
    if structure is None:
        structure = _morphology.generate_binary_structure(input.ndim, 1)
    # 将结构元素转换为布尔类型的NumPy数组
    structure = np.asarray(structure, dtype=bool)
    # 检查结构元素和输入数组的维度是否相同
    if structure.ndim != input.ndim:
        raise RuntimeError('structure and input must have equal rank')
    # 检查结构元素的每个维度是否为3
    for ii in structure.shape:
        if ii != 3:
            raise RuntimeError('structure dimensions must be equal to 3')

    # 如果结构元素不是连续的，则复制它以确保连续性
    if not structure.flags.contiguous:
        structure = structure.copy()
    # 将标记数组转换为NumPy数组
    markers = np.asarray(markers)
    # 检查输入数组和标记数组的形状是否相同
    if input.shape != markers.shape:
        raise RuntimeError('input and markers must have equal shape')

    # 定义支持的整数类型列表
    integral_types = [np.int8,
                      np.int16,
                      np.int32,
                      np.int64,
                      np.intc,
                      np.intp]

    # 检查标记数组的数据类型是否为支持的整数类型之一
    if markers.dtype.type not in integral_types:
        raise RuntimeError('marker should be of integer type')

    # 如果提供了输出数组，则检查其数据类型是否为支持的整数类型之一；否则，使用标记数组的数据类型作为输出数据类型
    if isinstance(output, np.ndarray):
        if output.dtype.type not in integral_types:
            raise RuntimeError('output should be of integer type')
    else:
        output = markers.dtype

    # 获取输出数组，其形状与输入数组相同
    output = _ni_support._get_output(output, input)
    # 调用底层函数进行图像森林变换分水岭算法
    _nd_image.watershed_ift(input, markers, structure, output)
    # 返回分水岭算法的输出数组
    return output
```