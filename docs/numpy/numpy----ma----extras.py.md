# `.\numpy\numpy\ma\extras.py`

```py
# 定义一个函数，用于创建一个指定形状和数据类型的全遮蔽数组
def masked_all(shape, dtype=float):
    # 返回一个全遮蔽的数组，形状由参数 shape 指定，数据类型由参数 dtype 指定
    return masked_array(np.empty(shape, dtype),
                        mask=True, dtype=dtype)
    # 创建一个形状为 shape，数据类型为 dtype 的空的掩码数组（Masked Array）。
    # 所有的数据都被掩盖（masked）。
    # Parameters 参数:
    # shape : int or tuple of ints
    #     所需的 MaskedArray 的形状，例如 ``(2, 3)`` 或 ``2``。
    # dtype : dtype, optional
    #     输出的数据类型。
    # Returns 返回:
    # a : MaskedArray
    #     所有数据都被掩盖的掩码数组。
    # Notes 注意:
    # 和其他创建掩码数组的函数（例如 `numpy.ma.zeros`, `numpy.ma.ones`, `numpy.ma.full`）不同，
    # `masked_all` 不会初始化数组的值，因此可能会稍微更快。
    # 然而，新分配的数组中存储的值是任意的。为了可重现的行为，请确保在读取之前设置数组的每个元素。
    # Examples 示例:
    # >>> np.ma.masked_all((3, 3))
    # masked_array(
    #   data=[[--, --, --],
    #         [--, --, --],
    #         [--, --, --]],
    #   mask=[[ True,  True,  True],
    #         [ True,  True,  True],
    #         [ True,  True,  True]],
    #   fill_value=1e+20,
    #   dtype=float64)
    # `dtype` 参数定义了底层数据类型。
    # >>> a = np.ma.masked_all((3, 3))
    # >>> a.dtype
    # dtype('float64')
    # >>> a = np.ma.masked_all((3, 3), dtype=np.int32)
    # >>> a.dtype
    # dtype('int32')
    a = masked_array(np.empty(shape, dtype),
                     mask=np.ones(shape, make_mask_descr(dtype)))
    # 返回创建的掩码数组
    return a
# 创建一个函数 masked_all_like，返回一个与给定数组 arr 具有相同形状和数据类型的空的掩码数组。
def masked_all_like(arr):
    """
    Empty masked array with the properties of an existing array.

    Return an empty masked array of the same shape and dtype as
    the array `arr`, where all the data are masked.

    Parameters
    ----------
    arr : ndarray
        An array describing the shape and dtype of the required MaskedArray.

    Returns
    -------
    a : MaskedArray
        A masked array with all data masked.

    Raises
    ------
    AttributeError
        If `arr` doesn't have a shape attribute (i.e. not an ndarray)

    See Also
    --------
    masked_all : Empty masked array with all elements masked.

    Notes
    -----
    Unlike other masked array creation functions (e.g. `numpy.ma.zeros_like`,
    `numpy.ma.ones_like`, `numpy.ma.full_like`), `masked_all_like` does not
    initialize the values of the array, and may therefore be marginally
    faster. However, the values stored in the newly allocated array are
    arbitrary. For reproducible behavior, be sure to set each element of the
    array before reading.

    Examples
    --------
    >>> arr = np.zeros((2, 3), dtype=np.float32)
    >>> arr
    array([[0., 0., 0.],
           [0., 0., 0.]], dtype=float32)
    >>> np.ma.masked_all_like(arr)
    masked_array(
      data=[[--, --, --],
            [--, --, --]],
      mask=[[ True,  True,  True],
            [ True,  True,  True]],
      fill_value=np.float64(1e+20),
      dtype=float32)

    The dtype of the masked array matches the dtype of `arr`.

    >>> arr.dtype
    dtype('float32')
    >>> np.ma.masked_all_like(arr).dtype
    dtype('float32')

    """

    # 创建一个空的与 arr 具有相同形状和数据类型的数组
    a = np.empty_like(arr).view(MaskedArray)
    # 初始化掩码数组，将所有元素都掩码化
    a._mask = np.ones(a.shape, dtype=make_mask_descr(a.dtype))
    return a


#####--------------------------------------------------------------------------
#---- --- Standard functions ---
#####--------------------------------------------------------------------------
class _fromnxfunction:
    """
    Defines a wrapper to adapt NumPy functions to masked arrays.


    An instance of `_fromnxfunction` can be called with the same parameters
    as the wrapped NumPy function. The docstring of `newfunc` is adapted from
    the wrapped function as well, see `getdoc`.

    This class should not be used directly. Instead, one of its extensions that
    provides support for a specific type of input should be used.

    Parameters
    ----------
    funcname : str
        The name of the function to be adapted. The function should be
        in the NumPy namespace (i.e. ``np.funcname``).

    """

    # 定义一个类 `_fromnxfunction`，用于适配将 NumPy 函数应用到掩码数组的情况
    def __init__(self, funcname):
        # 设置实例的名称为 funcname，即被适配的函数名
        self.__name__ = funcname
        # 根据被适配函数的文档，设置自己的文档
        self.__doc__ = self.getdoc()
    # 定义一个方法，用于获取函数的文档字符串和签名信息
    def getdoc(self):
        """
        Retrieve the docstring and signature from the function.
    
        The ``__doc__`` attribute of the function is used as the docstring for
        the new masked array version of the function. A note on application
        of the function to the mask is appended.
    
        Parameters
        ----------
        None
    
        """
        # 获取 NumPy 中与当前函数名对应的函数对象
        npfunc = getattr(np, self.__name__, None)
        # 获取函数对象的文档字符串
        doc = getattr(npfunc, '__doc__', None)
        # 如果存在文档字符串
        if doc:
            # 获取函数对象的签名信息
            sig = ma.get_object_signature(npfunc)
            # 向文档字符串添加应用于数据和掩码（如果有）的说明
            doc = ma.doc_note(doc, "The function is applied to both the _data "
                                   "and the _mask, if any.")
            # 如果存在签名信息，将函数名和签名信息添加到文档字符串开头
            if sig:
                sig = self.__name__ + sig + "\n\n"
            # 返回组合后的文档字符串
            return sig + doc
        # 如果不存在文档字符串，返回空值
        return
    
    # 定义一个特殊方法，使得对象可以像函数一样被调用，但该方法暂未实现任何功能
    def __call__(self, *args, **params):
        pass
class _fromnxfunction_single(_fromnxfunction):
    """
    A version of `_fromnxfunction` that is called with a single array
    argument followed by auxiliary args that are passed verbatim for
    both the data and mask calls.
    """
    # 重载 __call__ 方法，处理单个 ndarray 参数及其它辅助参数，返回一个 masked_array 对象
    def __call__(self, x, *args, **params):
        # 获取 numpy 中与 self.__name__ 对应的函数对象
        func = getattr(np, self.__name__)
        # 如果 x 是 ndarray 类型
        if isinstance(x, ndarray):
            # 对 x.__array__() 调用 func 函数，并传递其它参数 args 和 params
            _d = func(x.__array__(), *args, **params)
            # 对 getmaskarray(x) 调用 func 函数，并传递其它参数 args 和 params
            _m = func(getmaskarray(x), *args, **params)
            # 返回一个 masked_array 对象，数据为 _d，掩码为 _m
            return masked_array(_d, mask=_m)
        else:
            # 将 x 转换为 ndarray 类型，然后调用 func 函数，并传递其它参数 args 和 params
            _d = func(np.asarray(x), *args, **params)
            # 对 getmaskarray(x) 调用 func 函数，并传递其它参数 args 和 params
            _m = func(getmaskarray(x), *args, **params)
            # 返回一个 masked_array 对象，数据为 _d，掩码为 _m
            return masked_array(_d, mask=_m)


class _fromnxfunction_seq(_fromnxfunction):
    """
    A version of `_fromnxfunction` that is called with a single sequence
    of arrays followed by auxiliary args that are passed verbatim for
    both the data and mask calls.
    """
    # 重载 __call__ 方法，处理单个数组序列参数及其它辅助参数，返回一个 masked_array 对象
    def __call__(self, x, *args, **params):
        # 获取 numpy 中与 self.__name__ 对应的函数对象
        func = getattr(np, self.__name__)
        # 将 x 中的每个数组转换为 ndarray 类型，并传递给 func 函数，同时传递其它参数 args 和 params
        _d = func(tuple([np.asarray(a) for a in x]), *args, **params)
        # 对 x 中的每个数组获取掩码并传递给 func 函数，同时传递其它参数 args 和 params
        _m = func(tuple([getmaskarray(a) for a in x]), *args, **params)
        # 返回一个 masked_array 对象，数据为 _d，掩码为 _m
        return masked_array(_d, mask=_m)


class _fromnxfunction_args(_fromnxfunction):
    """
    A version of `_fromnxfunction` that is called with multiple array
    arguments. The first non-array-like input marks the beginning of the
    arguments that are passed verbatim for both the data and mask calls.
    Array arguments are processed independently and the results are
    returned in a list. If only one array is found, the return value is
    just the processed array instead of a list.
    """
    # 重载 __call__ 方法，处理多个数组参数及其它辅助参数，返回一个列表或单个 processed array 对象
    def __call__(self, *args, **params):
        # 获取 numpy 中与 self.__name__ 对应的函数对象
        func = getattr(np, self.__name__)
        # 初始化一个空列表用于存储处理后的数组结果
        arrays = []
        # 将 args 转换为列表
        args = list(args)
        # 只要 args 的长度大于 0 且 args[0] 是一个序列
        while len(args) > 0 and issequence(args[0]):
            # 将 args[0] 弹出并添加到 arrays 列表中
            arrays.append(args.pop(0))
        # 初始化一个空列表用于存储最终的处理结果
        res = []
        # 遍历 arrays 列表中的每个数组 x
        for x in arrays:
            # 将 x 转换为 ndarray 类型，并调用 func 函数，同时传递其它参数 args 和 params
            _d = func(np.asarray(x), *args, **params)
            # 对 getmaskarray(x) 调用 func 函数，同时传递其它参数 args 和 params
            _m = func(getmaskarray(x), *args, **params)
            # 将处理后的 masked_array 对象添加到 res 列表中
            res.append(masked_array(_d, mask=_m))
        # 如果 arrays 中只有一个数组，直接返回该 processed array 对象而不是列表
        if len(arrays) == 1:
            return res[0]
        # 否则返回处理后的列表 res
        return res


class _fromnxfunction_allargs(_fromnxfunction):
    """
    A version of `_fromnxfunction` that is called with multiple array
    arguments. Similar to `_fromnxfunction_args` except that all args
    are converted to arrays even if they are not so already. This makes
    it possible to process scalars as 1-D arrays. Only keyword arguments
    are passed through verbatim for the data and mask calls. Arrays
    arguments are processed independently and the results are returned
    in a list. If only one arg is present, the return value is just the
    processed array instead of a list.
    """
    # 重载 __call__ 方法，处理多个数组参数，即使它们不是数组也会转换为数组，返回一个列表或单个 processed array 对象
    def __call__(self, *args, **params):
        # 获取 numpy 中与 self.__name__ 对应的函数对象
        func = getattr(np, self.__name__)
        # 初始化一个空列表用于存储处理后的数组结果
        arrays = []
        # 将 args 转换为列表
        args = list(args)
        # 只要 args 的长度大于 0 且 args[0] 是一个序列
        while len(args) > 0 and issequence(args[0]):
            # 将 args[0] 弹出并添加到 arrays 列表中
            arrays.append(args.pop(0))
        # 初始化一个空列表用于存储最终的处理结果
        res = []
        # 遍历 arrays 列表中的每个数组 x
        for x in arrays:
            # 将 x 转换为 ndarray 类型，并调用 func 函数，同时传递其它参数 args 和 params
            _d = func(np.asarray(x), *args, **params)
            # 对 getmaskarray(x) 调用 func 函数，同时传递其它参数 args 和 params
            _m = func(getmaskarray(x), *args, **params)
            # 将处理后的 masked_array 对象添加到 res 列表中
            res.append(masked_array(_d, mask=_m))
        # 如果 arrays 中只有一个数组，直接返回该 processed array 对象而不是列表
        if len(arrays) == 1:
            return res[0]
        # 否则返回处理后的列表 res
        return res
    # 定义一个特殊方法 __call__，使得对象可以像函数一样被调用
    def __call__(self, *args, **params):
        # 获取 numpy 库中与对象的名称相对应的函数，并赋值给变量 func
        func = getattr(np, self.__name__)
        # 初始化结果列表
        res = []
        # 遍历所有传入的参数 args
        for x in args:
            # 将参数 x 转换为 numpy 数组，并使用 func 调用相应的函数进行处理，传入额外的参数 params
            _d = func(np.asarray(x), **params)
            # 使用 getmaskarray 函数获取 x 的掩码，然后同样使用 func 调用相应的函数进行处理
            _m = func(getmaskarray(x), **params)
            # 使用 _d 和 _m 创建一个 masked_array 对象，并添加到结果列表 res 中
            res.append(masked_array(_d, mask=_m))
        # 如果参数个数为 1，则直接返回结果列表中的第一个元素
        if len(args) == 1:
            return res[0]
        # 否则返回整个结果列表
        return res
atleast_1d = _fromnxfunction_allargs('atleast_1d')
atleast_2d = _fromnxfunction_allargs('atleast_2d')
atleast_3d = _fromnxfunction_allargs('atleast_3d')

vstack = row_stack = _fromnxfunction_seq('vstack')
hstack = _fromnxfunction_seq('hstack')
column_stack = _fromnxfunction_seq('column_stack')
dstack = _fromnxfunction_seq('dstack')
stack = _fromnxfunction_seq('stack')

hsplit = _fromnxfunction_single('hsplit')

diagflat = _fromnxfunction_single('diagflat')


#####--------------------------------------------------------------------------
#----
#####--------------------------------------------------------------------------
# 将输入序列扁平化（in-place），即将多层嵌套的序列展开成一维
def flatten_inplace(seq):
    """Flatten a sequence in place."""
    k = 0
    while (k != len(seq)):
        while hasattr(seq[k], '__iter__'):
            seq[k:(k + 1)] = seq[k]
        k += 1
    return seq


def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    (This docstring should be overwritten)
    """
    # 将输入数组转换为ndarray类型，如果已经是，复制为True，允许子类
    arr = array(arr, copy=False, subok=True)
    # 获取数组的维度
    nd = arr.ndim
    # 标准化轴索引，确保轴在有效范围内
    axis = normalize_axis_index(axis, nd)
    # 初始化索引
    ind = [0] * (nd - 1)
    i = np.zeros(nd, 'O')
    indlist = list(range(nd))
    indlist.remove(axis)
    # 创建切片对象
    i[axis] = slice(None, None)
    # 计算输出形状
    outshape = np.asarray(arr.shape).take(indlist)
    # 设置初始索引
    i.put(indlist, ind)
    # 应用func1d函数到选定轴上的切片，并传递额外参数
    res = func1d(arr[tuple(i.tolist())], *args, **kwargs)
    # 如果res是一个数值，说明输出是一个较小的数组
    asscalar = np.isscalar(res)
    if not asscalar:
        try:
            len(res)
        except TypeError:
            asscalar = True
    # 注意：我们不应该从第一个结果中设置输出的dtype
    # 所以我们强制类型为对象，并建立一个dtype列表。我们将取最大的，以避免某些向下转型
    dtypes = []
    if asscalar:
        dtypes.append(np.asarray(res).dtype)
        # 创建一个对象类型的全零数组
        outarr = zeros(outshape, object)
        # 将结果放入相应的位置
        outarr[tuple(ind)] = res
        Ntot = np.prod(outshape)
        k = 1
        while k < Ntot:
            # 增加索引
            ind[-1] += 1
            n = -1
            while (ind[n] >= outshape[n]) and (n > (1 - nd)):
                ind[n - 1] += 1
                ind[n] = 0
                n -= 1
            i.put(indlist, ind)
            # 再次应用func1d函数到选定轴上的切片，并传递额外参数
            res = func1d(arr[tuple(i.tolist())], *args, **kwargs)
            outarr[tuple(ind)] = res
            dtypes.append(asarray(res).dtype)
            k += 1
    else:
        # 将 res 转换为数组，确保没有复制，子类化允许
        res = array(res, copy=False, subok=True)
        # 复制索引 i 到 j
        j = i.copy()
        # 将 j[axis] 设置为一个切片列表，以便在 res 的维度上进行索引
        j[axis] = ([slice(None, None)] * res.ndim)
        # 将 indlist 中的索引 ind 替换为 indlist 中的 ind
        j.put(indlist, ind)
        # 计算总共的元素个数
        Ntot = np.prod(outshape)
        # 保存原始的输出形状
        holdshape = outshape
        # 将输出形状转换为数组 arr 的形状
        outshape = list(arr.shape)
        # 将输出形状的轴设置为 res 的形状
        outshape[axis] = res.shape
        # 将 asarray(res) 的数据类型添加到 dtypes 中
        dtypes.append(asarray(res).dtype)
        # 将输出形状扁平化处理
        outshape = flatten_inplace(outshape)
        # 创建一个对象类型的全零数组 outarr
        outarr = zeros(outshape, object)
        # 将 res 放入 outarr 的相应位置
        outarr[tuple(flatten_inplace(j.tolist()))] = res
        k = 1
        while k < Ntot:
            # 增加索引
            ind[-1] += 1
            n = -1
            while (ind[n] >= holdshape[n]) and (n > (1 - nd)):
                # 如果索引超出 holdshape[n]，则增加前一维的索引
                ind[n - 1] += 1
                ind[n] = 0
                n -= 1
            # 将 indlist 中的索引 ind 替换为 indlist 中的 ind
            i.put(indlist, ind)
            j.put(indlist, ind)
            # 对 arr[tuple(i.tolist())] 应用 func1d 函数，并将结果保存到 res
            res = func1d(arr[tuple(i.tolist())], *args, **kwargs)
            # 将结果 res 放入 outarr 的相应位置
            outarr[tuple(flatten_inplace(j.tolist()))] = res
            # 将 asarray(res) 的数据类型添加到 dtypes 中
            dtypes.append(asarray(res).dtype)
            k += 1
    # 将 dtypes 转换为数组，找出最大的数据类型
    max_dtypes = np.dtype(np.asarray(dtypes).max())
    # 如果 arr 没有属性 '_mask'
    if not hasattr(arr, '_mask'):
        # 将 outarr 转换为数组，使用最大的数据类型
        result = np.asarray(outarr, dtype=max_dtypes)
    else:
        # 将 outarr 转换为数组，使用最大的数据类型
        result = asarray(outarr, dtype=max_dtypes)
        # 设置 result 的填充值为其默认的填充值
        result.fill_value = ma.default_fill_value(result)
    # 返回结果数组 result
    return result
# 将 `apply_along_axis` 函数的文档字符串设置为 `np.apply_along_axis` 的文档字符串
apply_along_axis.__doc__ = np.apply_along_axis.__doc__


# 定义函数 `apply_over_axes`，用于沿指定轴应用函数
def apply_over_axes(func, a, axes):
    """
    Apply a function `func` over specified axes of array `a`.

    Parameters
    ----------
    func : function
        Function to be applied.
    a : array_like
        Input array.
    axes : int or tuple of ints
        Axes along which `func` is applied.

    Returns
    -------
    array_like
        Result of applying `func` over the specified axes.

    Notes
    -----
    This function modifies the input array `a` by applying `func`
    over the specified axes.

    Examples
    --------
    >>> a = np.ma.arange(24).reshape(2,3,4)
    >>> a[:,0,1] = np.ma.masked
    >>> a[:,1,:] = np.ma.masked
    >>> a
    masked_array(
      data=[[[0, --, 2, 3],
             [--, --, --, --],
             [8, 9, 10, 11]],
            [[12, --, 14, 15],
             [--, --, --, --],
             [20, 21, 22, 23]]],
      mask=[[[False,  True, False, False],
             [ True,  True,  True,  True],
             [False, False, False, False]],
            [[False,  True, False, False],
             [ True,  True,  True,  True],
             [False, False, False, False]]],
      fill_value=999999)
    >>> np.ma.apply_over_axes(np.ma.sum, a, [0,2])
    masked_array(
      data=[[[46],
             [--],
             [124]]],
      mask=[[[False],
             [ True],
             [False]]],
      fill_value=999999)

    Tuple axis arguments to ufuncs are equivalent:

    >>> np.ma.sum(a, axis=(0,2)).reshape((1,-1,1))
    masked_array(
      data=[[[46],
             [--],
             [124]]],
      mask=[[[False],
             [ True],
             [False]]],
      fill_value=999999)
    """
    # 将输入数组 `a` 转换为数组形式
    val = asarray(a)
    # 获取数组 `a` 的维度
    N = a.ndim
    # 如果 `axes` 是一个标量，则转换为元组形式
    if array(axes).ndim == 0:
        axes = (axes,)
    # 遍历每个轴进行操作
    for axis in axes:
        # 处理负数轴索引
        if axis < 0:
            axis = N + axis
        # 准备传递给函数 `func` 的参数
        args = (val, axis)
        # 应用函数 `func` 并获取结果
        res = func(*args)
        # 如果结果的维度与输入数组 `a` 的维度相同，则更新 `val`
        if res.ndim == val.ndim:
            val = res
        else:
            # 如果结果维度不正确，则展开结果并更新 `val`
            res = ma.expand_dims(res, axis)
            if res.ndim == val.ndim:
                val = res
            else:
                # 如果仍然维度不匹配，则抛出值错误
                raise ValueError("function is not returning "
                        "an array of the correct shape")
    # 返回最终结果 `val`
    return val


# 如果 `apply_over_axes` 函数已有文档字符串，则更新为 `np.apply_over_axes` 的文档字符串（排除 'Notes' 部分）
if apply_over_axes.__doc__ is not None:
    apply_over_axes.__doc__ = np.apply_over_axes.__doc__[
        :np.apply_over_axes.__doc__.find('Notes')].rstrip() + \
    """

    Examples
    --------
    >>> a = np.ma.arange(24).reshape(2,3,4)
    >>> a[:,0,1] = np.ma.masked
    >>> a[:,1,:] = np.ma.masked
    >>> a
    masked_array(
      data=[[[0, --, 2, 3],
             [--, --, --, --],
             [8, 9, 10, 11]],
            [[12, --, 14, 15],
             [--, --, --, --],
             [20, 21, 22, 23]]],
      mask=[[[False,  True, False, False],
             [ True,  True,  True,  True],
             [False, False, False, False]],
            [[False,  True, False, False],
             [ True,  True,  True,  True],
             [False, False, False, False]]],
      fill_value=999999)
    >>> np.ma.apply_over_axes(np.ma.sum, a, [0,2])
    masked_array(
      data=[[[46],
             [--],
             [124]]],
      mask=[[[False],
             [ True],
             [False]]],
      fill_value=999999)

    Tuple axis arguments to ufuncs are equivalent:

    >>> np.ma.sum(a, axis=(0,2)).reshape((1,-1,1))
    masked_array(
      data=[[[46],
             [--],
             [124]]],
      mask=[[[False],
             [ True],
             [False]]],
      fill_value=999999)
    """
    weights : array_like, optional
        # 可选参数，与 `a` 中的值相关联的权重数组。每个 `a` 中的值根据其关联的权重contributing to the average according to its associated weight. 
        # 如果未指定轴，则权重数组必须与 `a` 具有相同的形状，否则在指定轴上，权重必须具有与 `a` 一致的维度和形状consistent with `a`.
        # 如果 `weights=None`，则假定 `a` 中的所有数据权重均为一。
        # 计算公式为::
        # 
        #     avg = sum(a * weights) / sum(weights)
        # 
        # 其中 sum 是针对所有包括的元素进行的。
        # 对 `weights` 的唯一约束是 `sum(weights)` 不能为0。
    returned : bool, optional
        # 指示是否应返回一个元组 `(result, sum of weights)` 作为输出（True），或仅返回结果（False）的标志。默认为 False.
    keepdims : bool, optional
        # 如果设置为 True，则被减少的轴会作为大小为1的维度保留在结果中。使用此选项，结果将正确地广播到原始 `a` 上。
        # *注意:* `keepdims` 不适用于 `numpy.matrix` 的实例或其它不支持 `keepdims` 方法的类。
        # 
        # .. versionadded:: 1.23.0

    Returns
    -------
    average, [sum_of_weights] : (tuple of) scalar or MaskedArray
        # 沿指定轴的平均值。当 `returned` 为 `True` 时，返回一个元组，第一个元素是平均值，第二个元素是权重的总和。
        # 如果 `a` 是整数类型且浮点数小于 `float64`，则返回类型为 `np.float64`；否则返回输入数据类型。如果返回，则 `sum_of_weights` 总是 `float64`.

    Raises
    ------
    ZeroDivisionError
        # 当轴上所有权重为零时。参见 `numpy.ma.average` 以处理此类错误的版本。
    TypeError
        # 当 `weights` 与 `a` 的形状不同时，并且 `axis=None` 时。
    ValueError
        # 当 `weights` 与指定轴上的 `a` 的维度和形状不一致时。

    Examples
    --------
    >>> a = np.ma.array([1., 2., 3., 4.], mask=[False, False, True, True])
    >>> np.ma.average(a, weights=[3, 1, 0, 0])
    1.25

    >>> x = np.ma.arange(6.).reshape(3, 2)
    >>> x
    masked_array(
      data=[[0., 1.],
            [2., 3.],
            [4., 5.]],
      mask=False,
      fill_value=1e+20)
    >>> data = np.arange(8).reshape((2, 2, 2))
    >>> data
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> np.ma.average(data, axis=(0, 1), weights=[[1./4, 3./4], [1., 1./2]])
    masked_array(data=[3.4, 4.4],
             mask=[False, False],
       fill_value=1e+20)
    >>> np.ma.average(data, axis=0, weights=[[1./4, 3./4], [1., 1./2]])
    Traceback (most recent call last):
        ...
    ValueError: Shape of weights must be consistent
    with shape of a along specified axis.

    >>> avg, sumweights = np.ma.average(x, axis=0, weights=[1, 2, 3],
    ...                                 returned=True)
    >>> avg
    masked_array(data=[2.6666666666666665, 3.6666666666666665],
                 mask=[False, False],
           fill_value=1e+20)

    With ``keepdims=True``, the following result has shape (3, 1).

    >>> np.ma.average(x, axis=1, keepdims=True)
    masked_array(
      data=[[0.5],
            [2.5],
            [4.5]],
      mask=False,
      fill_value=1e+20)
    """
    # 将输入数组 a 转换为 numpy 数组
    a = asarray(a)
    # 获取输入数组 a 的掩码
    m = getmask(a)

    # 如果指定了轴参数 axis，则规范化轴元组
    if axis is not None:
        axis = normalize_axis_tuple(axis, a.ndim, argname="axis")

    # 如果 keepdims 参数未指定，则初始化为默认空字典
    if keepdims is np._NoValue:
        keepdims_kw = {}
    else:
        # 否则，使用指定的 keepdims 参数
        keepdims_kw = {'keepdims': keepdims}

    # 如果未提供权重 weights，则计算简单平均值和计数
    if weights is None:
        # 计算平均值
        avg = a.mean(axis, **keepdims_kw)
        # 计算样本数量并转换为与 avg 相同的数据类型
        scl = avg.dtype.type(a.count(axis))
    else:
        # 否则，将权重数组转换为 numpy 数组
        wgt = asarray(weights)

        # 根据输入数组 a 和权重数组 wgt 的数据类型确定结果数据类型
        if issubclass(a.dtype.type, (np.integer, np.bool)):
            result_dtype = np.result_type(a.dtype, wgt.dtype, 'f8')
        else:
            result_dtype = np.result_type(a.dtype, wgt.dtype)

        # 进行一些基本的健全性检查
        if a.shape != wgt.shape:
            # 如果未指定轴，则引发类型错误
            if axis is None:
                raise TypeError(
                    "Axis must be specified when shapes of a and weights "
                    "differ.")
            # 如果权重数组的形状与指定的轴不一致，则引发值错误
            if wgt.shape != tuple(a.shape[ax] for ax in axis):
                raise ValueError(
                    "Shape of weights must be consistent with "
                    "shape of a along specified axis.")

            # 对权重数组进行广播设置以适应轴
            wgt = wgt.transpose(np.argsort(axis))
            wgt = wgt.reshape(tuple((s if ax in axis else 1)
                                    for ax, s in enumerate(a.shape)))

        # 如果输入数组 a 存在掩码，则在计算权重之前应用掩码
        if m is not nomask:
            wgt = wgt*(~a.mask)
            wgt.mask |= a.mask

        # 计算加权总和，并保持维度
        scl = wgt.sum(axis=axis, dtype=result_dtype, **keepdims_kw)
        # 计算加权平均值
        avg = np.multiply(a, wgt,
                          dtype=result_dtype).sum(axis, **keepdims_kw) / scl

    # 如果指定了 returned 参数
    if returned:
        # 如果加权总和 scl 的形状与平均值 avg 的形状不一致，则进行广播以匹配
        if scl.shape != avg.shape:
            scl = np.broadcast_to(scl, avg.shape).copy()
        # 返回加权平均值 avg 和加权总和 scl
        return avg, scl
    else:
        # 否则，只返回加权平均值 avg
        return avg
# 计算沿指定轴的中位数。

返回数组元素的中位数。

Parameters
----------
a : array_like
    输入数组或可以转换为数组的对象。
axis : int, optional
    计算中位数的轴。默认为None，表示沿数组的扁平化版本计算中位数。
out : ndarray, optional
    可选的输出数组，用于存放结果。它必须具有与期望输出相同的形状和缓冲区长度，但必要时将进行类型转换。
overwrite_input : bool, optional
    如果为True，则允许使用输入数组（a）的内存进行计算。调用中位数将修改输入数组。当不需要保留输入数组的内容时，这将节省内存。处理输入的定义是未定义的，但可能已完全或部分排序。默认为False。注意，如果overwrite_input为True，并且输入尚未是ndarray，则会引发错误。

Returns
-------
median : ndarray
    返回一个新数组，其中包含结果，除非指定了out，在这种情况下返回对out的引用。对于小于float64的整数和浮点数，返回数据类型是float64，否则返回输入数据类型。

See Also
--------
mean

Notes
-----
给定向量V，其中N个非掩码值，V的中位数是V的排序副本Vs的中间值 - 即Vs[(N-1)/2]（当N为奇数时），或{Vs[N/2 - 1] + Vs[N/2]}/2（当N为偶数时）。

Examples
--------
>>> x = np.ma.array(np.arange(8), mask=[0]*4 + [1]*4)
>>> np.ma.median(x)
1.5

>>> x = np.ma.array(np.arange(10).reshape(2, 5), mask=[0]*6 + [1]*4)
>>> np.ma.median(x)
2.5
>>> np.ma.median(x, axis=-1, overwrite_input=True)
masked_array(data=[2.0, 5.0],
             mask=[False, False],
       fill_value=1e+20)

"""
if not hasattr(a, 'mask'):
    # 如果输入数组a没有mask属性，则调用getdata函数获取数据（支持子对象），并计算中位数。
    m = np.median(getdata(a, subok=True), axis=axis,
                  out=out, overwrite_input=overwrite_input,
                  keepdims=keepdims)
    # 如果m是一个数组且维度大于等于1，则返回一个不可变的掩码数组。
    if isinstance(m, np.ndarray) and 1 <= m.ndim:
        return masked_array(m, copy=False)
    else:
        return m

# 否则，调用_ureduce函数，使用_median函数计算中位数。
return _ureduce(a, func=_median, keepdims=keepdims, axis=axis, out=out,
                overwrite_input=overwrite_input)
    # 检查数组 `a` 的数据类型是否是浮点数或复数，如果是，将填充值设为正无穷大
    if np.issubdtype(a.dtype, np.inexact):
        fill_value = np.inf
    else:
        fill_value = None
    
    # 如果允许覆盖输入数组，并且未指定轴向，则将数组扁平化后排序
    # 否则，在指定轴向上对数组进行排序，并将结果赋给 `asorted`
    if overwrite_input:
        if axis is None:
            asorted = a.ravel()
            asorted.sort(fill_value=fill_value)
        else:
            a.sort(axis=axis, fill_value=fill_value)
            asorted = a
    else:
        asorted = sort(a, axis=axis, fill_value=fill_value)

    # 如果未指定轴向，则将轴向设为 0；否则，规范化指定的轴向索引
    if axis is None:
        axis = 0
    else:
        axis = normalize_axis_index(axis, asorted.ndim)

    # 如果指定轴向上的数据长度为 0，则返回空轴向的均值（即 NaN）
    if asorted.shape[axis] == 0:
        indexer = [slice(None)] * asorted.ndim
        indexer[axis] = slice(0, 0)
        indexer = tuple(indexer)
        return np.ma.mean(asorted[indexer], axis=axis, out=out)

    # 如果 `asorted` 是一维数组，则计算中位数
    if asorted.ndim == 1:
        idx, odd = divmod(count(asorted), 2)
        mid = asorted[idx + odd - 1:idx + 1]
        
        # 如果数组数据类型是浮点数或复数且数组大小大于 0，则避免 inf / x = masked 的情况
        if np.issubdtype(asorted.dtype, np.inexact) and asorted.size > 0:
            s = mid.sum(out=out)
            if not odd:
                s = np.true_divide(s, 2., casting='safe', out=out)
            s = np.lib._utils_impl._median_nancheck(asorted, s, axis)
        else:
            s = mid.mean(out=out)
        
        # 如果结果被屏蔽，则根据输入是否包含足够的 minimum_fill_value 或全部值是否被屏蔽来返回最小的填充值
        if np.ma.is_masked(s) and not np.all(asorted.mask):
            return np.ma.minimum_fill_value(asorted)
        return s

    # 计算在指定轴向上的元素个数
    counts = count(asorted, axis=axis, keepdims=True)
    h = counts // 2

    # 如果元素个数是奇数，则设置 odd 为 True
    odd = counts % 2 == 1
    l = np.where(odd, h, h-1)

    # 在指定轴向上连接低位和高位的索引
    lh = np.concatenate([l,h], axis=axis)

    # 获取低位和高位的中位数
    low_high = np.take_along_axis(asorted, lh, axis=axis)

    # 定义一个函数，用于替换掉屏蔽的条目
    def replace_masked(s):
        # 如果 s 是屏蔽的，则用最小的完全填充值替换掉它，除非所有值都被屏蔽
        if np.ma.is_masked(s):
            rep = (~np.all(asorted.mask, axis=axis, keepdims=True)) & s.mask
            s.data[rep] = np.ma.minimum_fill_value(asorted)
            s.mask[rep] = False

    # 替换低位和高位的屏蔽条目
    replace_masked(low_high)

    # 如果数组数据类型是浮点数或复数，则避免 inf / x = masked 的情况
    if np.issubdtype(asorted.dtype, np.inexact):
        s = np.ma.sum(low_high, axis=axis, out=out)
        np.true_divide(s.data, 2., casting='unsafe', out=s.data)

        s = np.lib._utils_impl._median_nancheck(asorted, s, axis)
    else:
        s = np.ma.mean(low_high, axis=axis, out=out)

    # 返回最终的结果
    return s
# 压缩多维数组 `x`，移除包含掩码值的切片。

def compress_nd(x, axis=None):
    """
    压缩多维数组 `x`，移除包含掩码值的切片。

    Parameters
    ----------
    x : array_like, MaskedArray
        要操作的数组。如果不是 MaskedArray 实例（或者没有数组元素被掩盖），则将 `x` 解释为具有 `mask` 设置为 `nomask` 的 MaskedArray。
    axis : tuple of ints or int, optional
        可以用此参数配置要从中压缩切片的维度。
        - 如果 axis 是 int 的元组，则为要从中压缩切片的轴。
        - 如果 axis 是 int，则只有该轴会被压缩切片。
        - 如果 axis 是 None，则选择所有轴。

    Returns
    -------
    compress_array : ndarray
        压缩后的数组。

    Examples
    --------
    >>> arr = [[1, 2], [3, 4]]
    >>> mask = [[0, 1], [0, 0]]
    >>> x = np.ma.array(arr, mask=mask)
    >>> np.ma.compress_nd(x, axis=0)
    array([[3, 4]])
    >>> np.ma.compress_nd(x, axis=1)
    array([[1],
           [3]])
    >>> np.ma.compress_nd(x)
    array([[3]])

    """
    # 将 `x` 转换为数组
    x = asarray(x)
    # 获取掩码
    m = getmask(x)
    
    # 如果 axis 是 None，则设为整数元组
    if axis is None:
        axis = tuple(range(x.ndim))
    else:
        # 根据数组维度规范化 axis 元组
        axis = normalize_axis_tuple(axis, x.ndim)

    # 如果没有任何元素被掩盖，则返回原始数据
    if m is nomask or not m.any():
        return x._data
    # 如果所有元素都被掩盖，则返回空数组
    if m.all():
        return nxarray([])
    
    # 通过布尔索引筛选元素
    data = x._data
    for ax in axis:
        axes = tuple(list(range(ax)) + list(range(ax + 1, x.ndim)))
        data = data[(slice(None),)*ax + (~m.any(axis=axes),)]
    return data




# 压缩二维数组 `x` 中包含掩码值的行和/或列。

def compress_rowcols(x, axis=None):
    """
    压缩二维数组 `x` 中包含掩码值的行和/或列。

    通过 `axis` 参数选择压缩行和/或列的行为。

    - 如果 axis 是 None，则压缩行和列。
    - 如果 axis 是 0，则仅压缩行。
    - 如果 axis 是 1 或 -1，则仅压缩列。

    Parameters
    ----------
    x : array_like, MaskedArray
        要操作的数组。如果不是 MaskedArray 实例（或者没有数组元素被掩盖），则将 `x` 解释为具有 `mask` 设置为 `nomask` 的 MaskedArray。必须是二维数组。
    axis : int, optional
        执行操作的轴。默认为 None。

    Returns
    -------
    compressed_array : ndarray
        压缩后的数组。

    Examples
    --------
    >>> x = np.ma.array(np.arange(9).reshape(3, 3), mask=[[1, 0, 0],
    ...                                                   [1, 0, 0],
    ...                                                   [0, 0, 0]])
    >>> x
    masked_array(
      data=[[--, 1, 2],
            [--, 4, 5],
            [6, 7, 8]],
      mask=[[ True, False, False],
            [ True, False, False],
            [False, False, False]],
      fill_value=999999)

    >>> np.ma.compress_rowcols(x)

    """
    # 将 `x` 转换为数组
    x = asarray(x)
    
    # 如果没有元素被掩盖，则返回原始数据
    if not ma.is_masked(x) or not ma.getmask(x).any():
        return x._data

    # 如果 axis 是 None，则压缩行和列
    if axis is None:
        row_mask = ma.getmaskarray(x).any(axis=1)
        col_mask = ma.getmaskarray(x).any(axis=0)
        return x[~row_mask][:, ~col_mask]._data

    # 如果 axis 是 0，则仅压缩行
    elif axis == 0:
        row_mask = ma.getmaskarray(x).any(axis=1)
        return x[~row_mask]._data

    # 如果 axis 是 1 或 -1，则仅压缩列
    elif axis == 1 or axis == -1:
        col_mask = ma.getmaskarray(x).any(axis=0)
        return x[:, ~col_mask]._data
    # 创建一个二维数组，包含单一的子数组 [[7, 8]]
    array([[7, 8]])
    # 调用 NumPy 的掩码数组函数 compress_rowcols，对输入的数组 x 进行行列压缩，axis=0 表示按行压缩
    >>> np.ma.compress_rowcols(x, 0)
    # 返回压缩后的二维数组，包含一个子数组 [[6, 7, 8]]
    array([[6, 7, 8]])
    # 调用 NumPy 的掩码数组函数 compress_rowcols，对输入的数组 x 进行行列压缩，axis=1 表示按列压缩
    >>> np.ma.compress_rowcols(x, 1)
    # 返回压缩后的二维数组，包含三个子数组：
    # 第一个子数组 [[1, 2]]
    # 第二个子数组 [[4, 5]]
    # 第三个子数组 [[7, 8]]
    array([[1, 2],
           [4, 5],
           [7, 8]])

    """
    # 如果输入数组 x 不是二维的，则抛出未实现错误
    if asarray(x).ndim != 2:
        raise NotImplementedError("compress_rowcols works for 2D arrays only.")
    # 调用 compress_nd 函数对输入数组 x 进行压缩，axis 参数指定压缩的方向（行或列）
    return compress_nd(x, axis=axis)
def compress_rows(a):
    """
    Suppress whole rows of a 2-D array that contain masked values.

    This is equivalent to ``np.ma.compress_rowcols(a, 0)``, see
    `compress_rowcols` for details.

    Parameters
    ----------
    x : array_like, MaskedArray
        The array to operate on. If not a MaskedArray instance (or if no array
        elements are masked), `x` is interpreted as a MaskedArray with
        `mask` set to `nomask`. Must be a 2D array.

    Returns
    -------
    compressed_array : ndarray
        The compressed array.

    See Also
    --------
    compress_rowcols

    Examples
    --------
    >>> a = np.ma.array(np.arange(9).reshape(3, 3), mask=[[1, 0, 0],
    ...                                                   [1, 0, 0],
    ...                                                   [0, 0, 0]])
    >>> np.ma.compress_rows(a)
    array([[6, 7, 8]])
    
    """
    # 将输入数组转换为MaskedArray类型
    a = asarray(a)
    # 检查数组维度，如果不是二维则抛出异常
    if a.ndim != 2:
        raise NotImplementedError("compress_rows works for 2D arrays only.")
    # 调用np.ma.compress_rowcols函数，axis参数设为0，压缩行
    return compress_rowcols(a, 0)


def compress_cols(a):
    """
    Suppress whole columns of a 2-D array that contain masked values.

    This is equivalent to ``np.ma.compress_rowcols(a, 1)``, see
    `compress_rowcols` for details.

    Parameters
    ----------
    x : array_like, MaskedArray
        The array to operate on.  If not a MaskedArray instance (or if no array
        elements are masked), `x` is interpreted as a MaskedArray with
        `mask` set to `nomask`. Must be a 2D array.

    Returns
    -------
    compressed_array : ndarray
        The compressed array.

    See Also
    --------
    compress_rowcols

    Examples
    --------
    >>> a = np.ma.array(np.arange(9).reshape(3, 3), mask=[[1, 0, 0],
    ...                                                   [1, 0, 0],
    ...                                                   [0, 0, 0]])
    >>> np.ma.compress_cols(a)
    array([[1, 2],
           [4, 5],
           [7, 8]])

    """
    # 将输入数组转换为MaskedArray类型
    a = asarray(a)
    # 检查数组维度，如果不是二维则抛出异常
    if a.ndim != 2:
        raise NotImplementedError("compress_cols works for 2D arrays only.")
    # 调用np.ma.compress_rowcols函数，axis参数设为1，压缩列
    return compress_rowcols(a, 1)


def mask_rowcols(a, axis=None):
    """
    Mask rows and/or columns of a 2D array that contain masked values.

    Mask whole rows and/or columns of a 2D array that contain
    masked values.  The masking behavior is selected using the
    `axis` parameter.

      - If `axis` is None, rows *and* columns are masked.
      - If `axis` is 0, only rows are masked.
      - If `axis` is 1 or -1, only columns are masked.

    Parameters
    ----------
    a : array_like, MaskedArray
        The array to mask.  If not a MaskedArray instance (or if no array
        elements are masked), the result is a MaskedArray with `mask` set
        to `nomask` (False). Must be a 2D array.
    axis : int, optional
        Axis along which to perform the operation. If None, applies to a
        flattened version of the array.

    Returns
    -------

    """
    # 将输入数组转换为MaskedArray类型
    a = asarray(a)
    # 如果axis为None，则同时掩码行和列
    if axis is None:
        return masked_array(a, mask=np.ma.mask_or(getmask(a), np.ma.getmaskarray(a).any(axis=0)))
    # 如果axis为0，则只掩码行
    elif axis == 0:
        return masked_array(a, mask=np.ma.getmaskarray(a).any(axis=1))
    # 如果axis为1或-1，则只掩码列
    elif axis == 1 or axis == -1:
        return masked_array(a, mask=np.ma.getmaskarray(a).any(axis=0))
    a : MaskedArray
        输入数组的修改版本，根据 `axis` 参数的值进行屏蔽。

    Raises
    ------
    NotImplementedError
        如果输入数组 `a` 不是二维的。

    See Also
    --------
    mask_rows : 屏蔽包含屏蔽值的二维数组的行。
    mask_cols : 屏蔽包含屏蔽值的二维数组的列。
    masked_where : 根据条件屏蔽数组中的元素。

    Notes
    -----
    此函数会修改输入数组的掩码。

    Examples
    --------
    >>> a = np.zeros((3, 3), dtype=int)
    >>> a[1, 1] = 1
    >>> a
    array([[0, 0, 0],
           [0, 1, 0],
           [0, 0, 0]])
    >>> a = np.ma.masked_equal(a, 1)
    >>> a
    masked_array(
      data=[[0, 0, 0],
            [0, --, 0],
            [0, 0, 0]],
      mask=[[False, False, False],
            [False,  True, False],
            [False, False, False]],
      fill_value=1)
    >>> np.ma.mask_rowcols(a)
    masked_array(
      data=[[0, --, 0],
            [--, --, --],
            [0, --, 0]],
      mask=[[False,  True, False],
            [ True,  True,  True],
            [False,  True, False]],
      fill_value=1)

    """
    # 将输入数组转换为 MaskedArray 类型
    a = array(a, subok=False)
    # 如果数组不是二维的，抛出未实现错误
    if a.ndim != 2:
        raise NotImplementedError("mask_rowcols works for 2D arrays only.")
    # 获取数组的掩码
    m = getmask(a)
    # 如果没有任何元素被屏蔽，则直接返回数组
    if m is nomask or not m.any():
        return a
    # 找出被屏蔽的元素的索引
    maskedval = m.nonzero()
    # 复制数组的掩码以确保不影响原始数据
    a._mask = a._mask.copy()
    # 如果 axis 为假值，则在指定行上屏蔽元素
    if not axis:
        a[np.unique(maskedval[0])] = masked
    # 如果 axis 是 None、1 或 -1，则在指定列上屏蔽元素
    if axis in [None, 1, -1]:
        a[:, np.unique(maskedval[1])] = masked
    return a
# 定义一个函数，用于在二维数组中掩盖包含掩码值的行
def mask_rows(a, axis=np._NoValue):
    """
    Mask rows of a 2D array that contain masked values.

    This function is a shortcut to ``mask_rowcols`` with `axis` equal to 0.

    See Also
    --------
    mask_rowcols : Mask rows and/or columns of a 2D array.
    masked_where : Mask where a condition is met.

    Examples
    --------
    >>> a = np.zeros((3, 3), dtype=int)
    >>> a[1, 1] = 1
    >>> a
    array([[0, 0, 0],
           [0, 1, 0],
           [0, 0, 0]])
    >>> a = np.ma.masked_equal(a, 1)
    >>> a
    masked_array(
      data=[[0, 0, 0],
            [0, --, 0],
            [0, 0, 0]],
      mask=[[False, False, False],
            [False,  True, False],
            [False, False, False]],
      fill_value=1)

    >>> np.ma.mask_rows(a)
    masked_array(
      data=[[0, 0, 0],
            [--, --, --],
            [0, 0, 0]],
      mask=[[False, False, False],
            [ True,  True,  True],
            [False, False, False]],
      fill_value=1)

    """
    # 如果指定了轴参数，发出警告，因为此参数已不再使用
    if axis is not np._NoValue:
        warnings.warn(
            "The axis argument has always been ignored, in future passing it "
            "will raise TypeError", DeprecationWarning, stacklevel=2)
    # 调用mask_rowcols函数，将axis设置为0，返回处理后的结果
    return mask_rowcols(a, 0)


# 定义一个函数，用于在二维数组中掩盖包含掩码值的列
def mask_cols(a, axis=np._NoValue):
    """
    Mask columns of a 2D array that contain masked values.

    This function is a shortcut to ``mask_rowcols`` with `axis` equal to 1.

    See Also
    --------
    mask_rowcols : Mask rows and/or columns of a 2D array.
    masked_where : Mask where a condition is met.

    Examples
    --------
    >>> a = np.zeros((3, 3), dtype=int)
    >>> a[1, 1] = 1
    >>> a
    array([[0, 0, 0],
           [0, 1, 0],
           [0, 0, 0]])
    >>> a = np.ma.masked_equal(a, 1)
    >>> a
    masked_array(
      data=[[0, 0, 0],
            [0, --, 0],
            [0, 0, 0]],
      mask=[[False, False, False],
            [False,  True, False],
            [False, False, False]],
      fill_value=1)
    >>> np.ma.mask_cols(a)
    masked_array(
      data=[[0, --, 0],
            [0, --, 0],
            [0, --, 0]],
      mask=[[False,  True, False],
            [False,  True, False],
            [False,  True, False]],
      fill_value=1)

    """
    # 如果指定了轴参数，发出警告，因为此参数已不再使用
    if axis is not np._NoValue:
        warnings.warn(
            "The axis argument has always been ignored, in future passing it "
            "will raise TypeError", DeprecationWarning, stacklevel=2)
    # 调用mask_rowcols函数，将axis设置为1，返回处理后的结果
    return mask_rowcols(a, 1)
    This function calculates the discrete difference of an array, considering masked values if present, similar to `numpy.ediff1d`.

    Parameters
    ----------
    arr : array_like
        Input array.
    to_begin, to_end : array_like, optional
        Values to prepend or append to the calculated differences.

    Returns
    -------
    ndarray
        The calculated differences.

    Notes
    -----
    This function differs from `numpy.ediff1d` by handling masked values appropriately.

    See Also
    --------
    numpy.ediff1d : Equivalent function for ndarrays.

    Examples
    --------
    >>> arr = np.ma.array([1, 2, 4, 7, 0])
    >>> np.ma.ediff1d(arr)
    masked_array(data=[ 1,  2,  3, -7],
                 mask=False,
           fill_value=999999)
    
    """
    # Convert input array to a masked array, and flatten it
    arr = ma.asanyarray(arr).flat
    # Compute the differences between consecutive elements
    ed = arr[1:] - arr[:-1]
    # Initialize the list of arrays with the computed differences
    arrays = [ed]
    # Insert the 'to_begin' array at the beginning if provided
    if to_begin is not None:
        arrays.insert(0, to_begin)
    # Append the 'to_end' array at the end if provided
    if to_end is not None:
        arrays.append(to_end)
    # If more than one array is present (indicating 'to_begin' or 'to_end' was provided),
    # concatenate all arrays into a single array 'ed'
    if len(arrays) != 1:
        ed = hstack(arrays)
    # Return the computed differences
    return ed
def unique(ar1, return_index=False, return_inverse=False):
    """
    Finds the unique elements of an array.

    Masked values are considered the same element (masked). The output array
    is always a masked array. See `numpy.unique` for more details.

    See Also
    --------
    numpy.unique : Equivalent function for ndarrays.

    Examples
    --------
    >>> a = [1, 2, 1000, 2, 3]
    >>> mask = [0, 0, 1, 0, 0]
    >>> masked_a = np.ma.masked_array(a, mask)
    >>> masked_a
    masked_array(data=[1, 2, --, 2, 3],
                mask=[False, False,  True, False, False],
        fill_value=999999)
    >>> np.ma.unique(masked_a)
    masked_array(data=[1, 2, 3, --],
                mask=[False, False, False,  True],
        fill_value=999999)
    >>> np.ma.unique(masked_a, return_index=True)
    (masked_array(data=[1, 2, 3, --],
                mask=[False, False, False,  True],
        fill_value=999999), array([0, 1, 4, 2]))
    >>> np.ma.unique(masked_a, return_inverse=True)
    (masked_array(data=[1, 2, 3, --],
                mask=[False, False, False,  True],
        fill_value=999999), array([0, 1, 3, 1, 2]))
    >>> np.ma.unique(masked_a, return_index=True, return_inverse=True)
    (masked_array(data=[1, 2, 3, --],
                mask=[False, False, False,  True],
        fill_value=999999), array([0, 1, 4, 2]), array([0, 1, 3, 1, 2]))
    """
    # 使用 numpy 的 unique 函数获取数组 ar1 中的唯一元素
    output = np.unique(ar1,
                       return_index=return_index,
                       return_inverse=return_inverse)
    # 如果输出是一个元组，将第一个元素转换成 MaskedArray 类型
    if isinstance(output, tuple):
        output = list(output)
        output[0] = output[0].view(MaskedArray)
        output = tuple(output)
    else:
        # 否则，将输出直接转换成 MaskedArray 类型
        output = output.view(MaskedArray)
    return output


def intersect1d(ar1, ar2, assume_unique=False):
    """
    Returns the unique elements common to both arrays.

    Masked values are considered equal one to the other.
    The output is always a masked array.

    See `numpy.intersect1d` for more details.

    See Also
    --------
    numpy.intersect1d : Equivalent function for ndarrays.

    Examples
    --------
    >>> x = np.ma.array([1, 3, 3, 3], mask=[0, 0, 0, 1])
    >>> y = np.ma.array([3, 1, 1, 1], mask=[0, 0, 0, 1])
    >>> np.ma.intersect1d(x, y)
    masked_array(data=[1, 3, --],
                 mask=[False, False,  True],
           fill_value=999999)

    """
    if assume_unique:
        # 如果 assume_unique 为 True，直接合并 ar1 和 ar2
        aux = ma.concatenate((ar1, ar2))
    else:
        # 否则，合并 ar1 和 ar2 的唯一元素
        aux = ma.concatenate((unique(ar1), unique(ar2)))
    # 对合并后的数组进行排序
    aux.sort()
    # 返回重复元素组成的 masked array
    return aux[:-1][aux[1:] == aux[:-1]]


def setxor1d(ar1, ar2, assume_unique=False):
    """
    Set exclusive-or of 1-D arrays with unique elements.

    The output is always a masked array. See `numpy.setxor1d` for more details.

    See Also
    --------
    numpy.setxor1d : Equivalent function for ndarrays.

    Examples
    --------
    >>> ar1 = np.ma.array([1, 2, 3, 2, 4])
    """
    # 创建一个新的掩码数组，其中包含整数元素[2, 3, 5, 7, 5]，用于演示目的
    ar2 = np.ma.array([2, 3, 5, 7, 5])
    
    # 计算 ar1 和 ar2 的异或（exclusive-or）运算的结果，返回一个掩码数组
    np.ma.setxor1d(ar1, ar2)
    
    masked_array(data=[1, 4, 5, 7],    # 异或运算后的结果数组，包含不同的元素
                 mask=False,           # 掩码数组的掩码标记，表示所有元素均未被屏蔽
                 fill_value=999999)    # 用于填充未定义值的填充值
    
    """
    if not assume_unique:
        # 如果不假定输入数组已唯一化，则对 ar1 和 ar2 分别进行唯一化处理
        ar1 = unique(ar1)
        ar2 = unique(ar2)
    
    # 将 ar1 和 ar2 沿指定轴（None 表示展平）连接成一个掩码数组
    aux = ma.concatenate((ar1, ar2), axis=None)
    
    # 如果连接后的数组大小为零，则直接返回该数组
    if aux.size == 0:
        return aux
    
    # 对连接后的数组进行排序
    aux.sort()
    
    # 从排序后的数组中生成一个非掩码版本的副本
    auxf = aux.filled()
# 创建一个布尔数组，用于指示数组 auxf 中连续相同元素的位置
flag = ma.concatenate(([True], (auxf[1:] != auxf[:-1]), [True]))

# 创建一个布尔数组，用于指示 flag 中连续相同元素的位置
flag2 = (flag[1:] == flag[:-1])

# 返回数组 aux 中 flag2 为 True 的元素，即连续相同元素的集合
return aux[flag2]
    # 将 ar1 和 ar2 数组按指定轴(axis=None，即展开为一维数组)连接起来，然后返回连接后的唯一值数组
    return unique(ma.concatenate((ar1, ar2), axis=None))
def setdiff1d(ar1, ar2, assume_unique=False):
    """
    Set difference of 1D arrays with unique elements.

    The output is always a masked array. See `numpy.setdiff1d` for more
    details.

    See Also
    --------
    numpy.setdiff1d : Equivalent function for ndarrays.

    Examples
    --------
    >>> x = np.ma.array([1, 2, 3, 4], mask=[0, 1, 0, 1])
    >>> np.ma.setdiff1d(x, [1, 2])
    masked_array(data=[3, --],
                 mask=[False,  True],
           fill_value=999999)

    """
    # 根据 assume_unique 参数决定是否将 ar1 转换为掩码数组
    if assume_unique:
        ar1 = ma.asarray(ar1).ravel()
    else:
        # 将 ar1 和 ar2 转换为唯一值数组
        ar1 = unique(ar1)
        ar2 = unique(ar2)
    # 返回 ar1 中不在 ar2 中的元素，作为掩码数组的形式
    return ar1[in1d(ar1, ar2, assume_unique=True, invert=True)]


###############################################################################
#                                Covariance                                   #
###############################################################################


def _covhelper(x, y=None, rowvar=True, allow_masked=True):
    """
    Private function for the computation of covariance and correlation
    coefficients.

    """
    # 将 x 转换为掩码数组，确保是二维数组且数据类型为 float
    x = ma.array(x, ndmin=2, copy=True, dtype=float)
    xmask = ma.getmaskarray(x)
    # 如果不允许处理掩码数据且 x 中有掩码，则抛出 ValueError 异常
    if not allow_masked and xmask.any():
        raise ValueError("Cannot process masked data.")
    #
    # 如果 x 的第一个维度为 1，则强制 rowvar 为 True
    if x.shape[0] == 1:
        rowvar = True
    # 确保 rowvar 的值为 0 或 1
    rowvar = int(bool(rowvar))
    axis = 1 - rowvar
    if rowvar:
        tup = (slice(None), None)
    else:
        tup = (None, slice(None))
    #
    if y is None:
        # 检查是否可以保证在计算点积之前，(N - ddof) 归一化的整数可以准确表示为单精度
        if x.shape[0] > 2 ** 24 or x.shape[1] > 2 ** 24:
            xnm_dtype = np.float64
        else:
            xnm_dtype = np.float32
        # 创建 xnotmask 数组，用于表示非掩码部分的布尔数组，并指定数据类型
        xnotmask = np.logical_not(xmask).astype(xnm_dtype)
    else:
        # 如果 y 不是数组，则将其转换为二维数组，确保复制内容、数据类型为 float
        y = array(y, copy=False, ndmin=2, dtype=float)
        # 获取 y 的掩码（mask）
        ymask = ma.getmaskarray(y)
        # 如果不允许使用掩码数据，并且 y 中存在掩码
        if not allow_masked and ymask.any():
            # 抛出值错误异常
            raise ValueError("Cannot process masked data.")
        # 如果 x 或 y 中存在掩码
        if xmask.any() or ymask.any():
            # 如果 x 和 y 的形状相同
            if y.shape == x.shape:
                # 定义一个常见的掩码
                common_mask = np.logical_or(xmask, ymask)
                # 如果存在公共掩码
                if common_mask is not nomask:
                    # 将 x 和 y 的掩码设置为公共掩码
                    xmask = x._mask = y._mask = ymask = common_mask
                    # 取消共享掩码
                    x._sharedmask = False
                    y._sharedmask = False
        # 将 y 沿指定轴连接到 x 上
        x = ma.concatenate((x, y), axis)
        # 检查是否可以保证在计算点积之前，(N - ddof) 归一化的整数能够以单精度精确表示
        if x.shape[0] > 2 ** 24 or x.shape[1] > 2 ** 24:
            # 如果 x 的维度超过 2^24，使用双精度浮点数类型
            xnm_dtype = np.float64
        else:
            # 否则使用单精度浮点数类型
            xnm_dtype = np.float32
        # 计算 x 和 y 的合并掩码的补集，并转换为指定类型
        xnotmask = np.logical_not(np.concatenate((xmask, ymask), axis)).astype(
            xnm_dtype
        )
    # 减去 x 指定轴上的均值
    x -= x.mean(axis=rowvar)[tup]
    # 返回 x、xnotmask 和 rowvar
    return (x, xnotmask, rowvar)
# 求取协方差矩阵的估计值
def cov(x, y=None, rowvar=True, bias=False, allow_masked=True, ddof=None):
    """
    Estimate the covariance matrix.

    Except for the handling of missing data this function does the same as
    `numpy.cov`. For more details and examples, see `numpy.cov`.

    By default, masked values are recognized as such. If `x` and `y` have the
    same shape, a common mask is allocated: if ``x[i,j]`` is masked, then
    ``y[i,j]`` will also be masked.
    Setting `allow_masked` to False will raise an exception if values are
    missing in either of the input arrays.

    Parameters
    ----------
    x : array_like
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of `x` represents a variable, and each column a single
        observation of all those variables. Also see `rowvar` below.
    y : array_like, optional
        An additional set of variables and observations. `y` has the same
        shape as `x`.
    rowvar : bool, optional
        If `rowvar` is True (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
    bias : bool, optional
        Default normalization (False) is by ``(N-1)``, where ``N`` is the
        number of observations given (unbiased estimate). If `bias` is True,
        then normalization is by ``N``. This keyword can be overridden by
        the keyword ``ddof`` in numpy versions >= 1.5.
    allow_masked : bool, optional
        If True, masked values are propagated pair-wise: if a value is masked
        in `x`, the corresponding value is masked in `y`.
        If False, raises a `ValueError` exception when some values are missing.
    ddof : {None, int}, optional
        If not ``None`` normalization is by ``(N - ddof)``, where ``N`` is
        the number of observations; this overrides the value implied by
        ``bias``. The default value is ``None``.

        .. versionadded:: 1.5

    Raises
    ------
    ValueError
        Raised if some values are missing and `allow_masked` is False.

    See Also
    --------
    numpy.cov

    Examples
    --------
    >>> x = np.ma.array([[0, 1], [1, 1]], mask=[0, 1, 0, 1])
    >>> y = np.ma.array([[1, 0], [0, 1]], mask=[0, 0, 1, 1])
    >>> np.ma.cov(x, y)
    masked_array(
    data=[[--, --, --, --],
          [--, --, --, --],
          [--, --, --, --],
          [--, --, --, --]],
    mask=[[ True,  True,  True,  True],
          [ True,  True,  True,  True],
          [ True,  True,  True,  True],
          [ True,  True,  True,  True]],
    fill_value=1e+20,
    dtype=float64)
    
    """
    # 检查输入参数
    if ddof is not None and ddof != int(ddof):
        raise ValueError("ddof must be an integer")
    # 设置 ddof
    if ddof is None:
        if bias:
            ddof = 0
        else:
            ddof = 1
    # 调用 _covhelper 函数处理输入参数 x, y, rowvar 和 allow_masked，并解包返回值
    (x, xnotmask, rowvar) = _covhelper(x, y, rowvar, allow_masked)
    # 检查是否按列方向计算协方差
    if not rowvar:
        # 计算除去自由度 ddof 后的中间结果 fact
        fact = np.dot(xnotmask.T, xnotmask) - ddof
        # 创建一个布尔类型的掩码，标记 fact 中小于等于 0 的元素
        mask = np.less_equal(fact, 0, dtype=bool)
        # 忽略除以零或无效操作错误，计算协方差矩阵的数据
        with np.errstate(divide="ignore", invalid="ignore"):
            # 计算填充后的 x 的转置和共轭 x 的乘积，再除以 fact
            data = np.dot(filled(x.T, 0), filled(x.conj(), 0)) / fact
        # 创建一个带有掩码的 MaskedArray 对象，并压缩成一维数组
        result = ma.array(data, mask=mask).squeeze()
    else:
        # 计算除去自由度 ddof 后的中间结果 fact
        fact = np.dot(xnotmask, xnotmask.T) - ddof
        # 创建一个布尔类型的掩码，标记 fact 中小于等于 0 的元素
        mask = np.less_equal(fact, 0, dtype=bool)
        # 忽略除以零或无效操作错误，计算协方差矩阵的数据
        with np.errstate(divide="ignore", invalid="ignore"):
            # 计算填充后的 x 和 x 的转置的共轭乘积，再除以 fact
            data = np.dot(filled(x, 0), filled(x.T.conj(), 0)) / fact
        # 创建一个带有掩码的 MaskedArray 对象，并压缩成一维数组
        result = ma.array(data, mask=mask).squeeze()
    # 返回计算结果
    return result
def corrcoef(x, y=None, rowvar=True, bias=np._NoValue, allow_masked=True,
             ddof=np._NoValue):
    """
    Return Pearson product-moment correlation coefficients.

    Except for the handling of missing data this function does the same as
    `numpy.corrcoef`. For more details and examples, see `numpy.corrcoef`.

    Parameters
    ----------
    x : array_like
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of `x` represents a variable, and each column a single
        observation of all those variables. Also see `rowvar` below.
    y : array_like, optional
        An additional set of variables and observations. `y` has the same
        shape as `x`.
    rowvar : bool, optional
        If `rowvar` is True (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
    bias : _NoValue, optional
        Has no effect, do not use.

        .. deprecated:: 1.10.0
    allow_masked : bool, optional
        If True, masked values are propagated pair-wise: if a value is masked
        in `x`, the corresponding value is masked in `y`.
        If False, raises an exception.  Because `bias` is deprecated, this
        argument needs to be treated as keyword only to avoid a warning.
    ddof : _NoValue, optional
        Has no effect, do not use.

        .. deprecated:: 1.10.0

    See Also
    --------
    numpy.corrcoef : Equivalent function in top-level NumPy module.
    cov : Estimate the covariance matrix.

    Notes
    -----
    This function accepts but discards arguments `bias` and `ddof`.  This is
    for backwards compatibility with previous versions of this function.  These
    arguments had no effect on the return values of the function and can be
    safely ignored in this and previous versions of numpy.

    Examples
    --------
    >>> x = np.ma.array([[0, 1], [1, 1]], mask=[0, 1, 0, 1])
    >>> np.ma.corrcoef(x)
    masked_array(
      data=[[--, --],
            [--, --]],
      mask=[[ True,  True],
            [ True,  True]],
      fill_value=1e+20,
      dtype=float64)

    """
    # 提示信息，警告 bias 和 ddof 已不再起作用，并且已弃用
    msg = 'bias and ddof have no effect and are deprecated'
    if bias is not np._NoValue or ddof is not np._NoValue:
        # 如果 bias 或 ddof 不等于 np._NoValue，则发出警告
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
    # 估计协方差矩阵
    corr = cov(x, y, rowvar, allow_masked=allow_masked)
    # 非掩码版本返回标量的掩码值
    try:
        std = ma.sqrt(ma.diagonal(corr))
    except ValueError:
        # 如果出现 ValueError 异常，则返回掩码常量
        return ma.MaskedConstant()
    # 对相关系数进行标准化
    corr /= ma.multiply.outer(std, std)
    return corr

#####--------------------------------------------------------------------------
#---- --- Concatenation helpers ---
#####--------------------------------------------------------------------------
class MAxisConcatenator(AxisConcatenator):
    """
    Translate slice objects to concatenation along an axis.

    For documentation on usage, see `mr_class`.

    See Also
    --------
    mr_class

    """
    concatenate = staticmethod(concatenate)

    @classmethod
    def makemat(cls, arr):
        # There used to be a view as np.matrix here, but we may eventually
        # deprecate that class. In preparation, we use the unmasked version
        # to construct the matrix (with copy=False for backwards compatibility
        # with the .view)
        # 调用父类的 makemat 方法，使用 arr 的数据构造一个新的数组，保持兼容性
        data = super().makemat(arr.data, copy=False)
        return array(data, mask=arr.mask)

    def __getitem__(self, key):
        # matrix builder syntax, like 'a, b; c, d'
        if isinstance(key, str):
            # 如果 key 是字符串，抛出 MAError 异常，提示不支持对 masked array 的操作
            raise MAError("Unavailable for masked array.")

        # 调用父类的 __getitem__ 方法处理 key，返回处理后的结果
        return super().__getitem__(key)


class mr_class(MAxisConcatenator):
    """
    Translate slice objects to concatenation along the first axis.

    This is the masked array version of `r_`.

    See Also
    --------
    r_

    Examples
    --------
    >>> np.ma.mr_[np.ma.array([1,2,3]), 0, 0, np.ma.array([4,5,6])]
    masked_array(data=[1, 2, 3, ..., 4, 5, 6],
                 mask=False,
           fill_value=999999)

    """
    def __init__(self):
        # 初始化父类 MAxisConcatenator，指定 axis=0
        MAxisConcatenator.__init__(self, 0)

mr_ = mr_class()


#####--------------------------------------------------------------------------
#---- Find unmasked data ---
#####--------------------------------------------------------------------------

def ndenumerate(a, compressed=True):
    """
    Multidimensional index iterator.

    Return an iterator yielding pairs of array coordinates and values,
    skipping elements that are masked. With `compressed=False`,
    `ma.masked` is yielded as the value of masked elements. This
    behavior differs from that of `numpy.ndenumerate`, which yields the
    value of the underlying data array.

    Notes
    -----
    .. versionadded:: 1.23.0

    Parameters
    ----------
    a : array_like
        An array with (possibly) masked elements.
    compressed : bool, optional
        If True (default), masked elements are skipped.

    See Also
    --------
    numpy.ndenumerate : Equivalent function ignoring any mask.

    Examples
    --------
    >>> a = np.ma.arange(9).reshape((3, 3))
    >>> a[1, 0] = np.ma.masked
    >>> a[1, 2] = np.ma.masked
    >>> a[2, 1] = np.ma.masked
    >>> a
    masked_array(
      data=[[0, 1, 2],
            [--, 4, --],
            [6, --, 8]],
      mask=[[False, False, False],
            [ True, False,  True],
            [False,  True, False]],
      fill_value=999999)
    >>> for index, x in np.ma.ndenumerate(a):
    ...     print(index, x)
    (0, 0) 0
    (0, 1) 1
    (0, 2) 2
    (1, 1) 4
    (2, 0) 6
    (2, 2) 8

    >>> for index, x in np.ma.ndenumerate(a, compressed=False):
    ...     print(index, x)
    (0, 0) 0
    (0, 1) 1
    (0, 2) 2
    (1, 0) --
    (1, 1) 4
    (1, 2) --

    """
    # 返回一个迭代器，遍历数组 a 的所有非掩码元素的索引和值
    # 当 compressed=True 时，跳过掩码元素；当 compressed=False 时，以 ma.masked 作为掩码元素的值返回
    pass
    # 使用 NumPy 的 ndenumerate 函数迭代数组 a 中的每个元素及其对应的索引
    for it, mask in zip(np.ndenumerate(a), getmaskarray(a).flat):
        # 如果当前元素没有被遮罩，则返回该元素及其索引
        if not mask:
            yield it
        # 如果当前元素被遮罩，并且未压缩（compressed=False），则返回该元素的索引及遮罩信息
        elif not compressed:
            yield it[0], masked
# 寻找 1-D `MaskedArray` 中第一个和最后一个未被掩码的值的索引。
# 如果所有值都被掩码，则返回 None。
def flatnotmasked_edges(a):
    m = getmask(a)  # 获取数组 a 的掩码
    if m is nomask or not np.any(m):  # 如果没有掩码或者没有任何掩码存在
        return np.array([0, a.size - 1])  # 返回数组中第一个和最后一个索引
    unmasked = np.flatnonzero(~m)  # 找到未被掩码的索引
    if len(unmasked) > 0:  # 如果存在未被掩码的索引
        return unmasked[[0, -1]]  # 返回第一个和最后一个未被掩码的索引
    else:
        return None  # 如果所有值都被掩码，则返回 None


# 找到沿指定轴的数组中第一个和最后一个未被掩码的值的索引。
# 如果所有值都被掩码，则返回 None。否则，返回一个包含第一个和最后一个未被掩码值索引的列表。
def notmasked_edges(a, axis=None):
    a = asarray(a)  # 将输入转换为数组
    if axis is None or a.ndim == 1:  # 如果轴为空或者数组是一维的
        return flatnotmasked_edges(a)  # 调用 flatnotmasked_edges 处理
    m = getmaskarray(a)  # 获取数组 a 的掩码数组
    idx = array(np.indices(a.shape), mask=np.asarray([m] * a.ndim))  # 创建索引数组，考虑掩码
    return [tuple([idx[i].min(axis).compressed() for i in range(a.ndim)]),
            tuple([idx[i].max(axis).compressed() for i in range(a.ndim)])]
    # 获取数组 `a` 的掩码
    m = getmask(a)
    # 如果掩码为 `nomask`，表示没有掩码，返回包含整个数组范围的 slice 对象的列表
    if m is nomask:
        return [slice(0, a.size)]
    # 初始化索引变量 i 和结果列表 result
    i = 0
    result = []
    # 使用 itertools.groupby 对掩码展平后的数组进行分组
    for (k, g) in itertools.groupby(m.ravel()):
        # 计算当前分组的长度
        n = len(list(g))
        # 如果当前分组不被掩码，则将其作为一个 slice 对象添加到结果列表中
        if not k:
            result.append(slice(i, i + n))
        # 更新索引 i
        i += n
    # 返回结果列表，其中包含所有非掩码连续区域的 slice 对象
    return result
# 寻找给定轴向上的连续未屏蔽数据的切片
def notmasked_contiguous(a, axis=None):
    """
    Find contiguous unmasked data in a masked array along the given axis.

    Parameters
    ----------
    a : array_like
        The input array.
    axis : int, optional
        Axis along which to perform the operation.
        If None (default), applies to a flattened version of the array, and this
        is the same as `flatnotmasked_contiguous`.

    Returns
    -------
    endpoints : list
        A list of slices (start and end indexes) of unmasked indexes
        in the array.

        If the input is 2d and axis is specified, the result is a list of lists.

    See Also
    --------
    flatnotmasked_edges, flatnotmasked_contiguous, notmasked_edges
    clump_masked, clump_unmasked

    Notes
    -----
    Only accepts 2-D arrays at most.

    Examples
    --------
    >>> a = np.arange(12).reshape((3, 4))
    >>> mask = np.zeros_like(a)
    >>> mask[1:, :-1] = 1; mask[0, 1] = 1; mask[-1, 0] = 0
    >>> ma = np.ma.array(a, mask=mask)
    >>> ma
    masked_array(
      data=[[0, --, 2, 3],
            [--, --, --, 7],
            [8, --, --, 11]],
      mask=[[False,  True, False, False],
            [ True,  True,  True, False],
            [False,  True,  True, False]],
      fill_value=999999)
    >>> np.array(ma[~ma.mask])
    array([ 0,  2,  3,  7, 8, 11])

    >>> np.ma.notmasked_contiguous(ma)
    [slice(0, 1, None), slice(2, 4, None), slice(7, 9, None), slice(11, 12, None)]

    >>> np.ma.notmasked_contiguous(ma, axis=0)
    [[slice(0, 1, None), slice(2, 3, None)], [], [slice(0, 1, None)], [slice(0, 3, None)]]

    >>> np.ma.notmasked_contiguous(ma, axis=1)
    [[slice(0, 1, None), slice(2, 4, None)], [slice(3, 4, None)], [slice(0, 1, None), slice(3, 4, None)]]

    """
    # 将输入转换为数组
    a = asarray(a)
    # 获取数组的维度
    nd = a.ndim
    # 如果维度大于2，则抛出未实现的错误
    if nd > 2:
        raise NotImplementedError("Currently limited to at most 2D array.")
    # 如果轴为None或数组为1维，则调用flatnotmasked_contiguous函数处理
    if axis is None or nd == 1:
        return flatnotmasked_contiguous(a)
    #
    result = []
    #
    other = (axis + 1) % 2
    idx = [0, 0]
    idx[axis] = slice(None, None)
    #
    for i in range(a.shape[other]):
        idx[other] = i
        result.append(flatnotmasked_contiguous(a[tuple(idx)]))
    return result


def _ezclump(mask):
    """
    Finds the clumps (groups of data with the same values) for a 1D bool array.

    Returns a series of slices.
    """
    # 如果mask的维度大于1，则将其展平为1维数组
    if mask.ndim > 1:
        mask = mask.ravel()
    # 找到不同值处的索引
    idx = (mask[1:] ^ mask[:-1]).nonzero()
    idx = idx[0] + 1

    # 如果数组的第一个元素为True
    if mask[0]:
        # 如果idx为空，则返回整个数组的切片
        if len(idx) == 0:
            return [slice(0, mask.size)]

        # 否则构建切片列表
        r = [slice(0, idx[0])]
        r.extend((slice(left, right)
                  for left, right in zip(idx[1:-1:2], idx[2::2])))
    else:
        # 如果数组的第一个元素为False
        if len(idx) == 0:
            return []

        # 构建切片列表
        r = [slice(left, right) for left, right in zip(idx[:-1:2], idx[1::2])]

    # 如果数组的最后一个元素为True，则添加最后一个切片
    if mask[-1]:
        r.append(slice(idx[-1], mask.size))
    return r


def clump_unmasked(a):
    """
    Finds contiguous groups of unmasked elements in a 1D masked array.

    Returns a list of slices representing the contiguous groups.
    """
    #
    # 获取输入数组 `a` 的掩码（如果存在），否则使用默认的 `nomask`
    mask = getattr(a, '_mask', nomask)
    
    # 如果数组 `a` 没有掩码，直接返回一个包含整个数组范围的切片列表
    if mask is nomask:
        return [slice(0, a.size)]
    
    # 否则，调用 `_ezclump` 函数来计算非掩码部分的连续区域，并返回对应的切片列表
    return _ezclump(~mask)
# 返回一个列表，包含一个一维数组中被掩盖掉的连续区域的切片
def clump_masked(a):
    # 获取数组的掩码
    mask = ma.getmask(a)
    # 如果掩码是空（没有掩盖），则返回空列表
    if mask is nomask:
        return []
    # 调用内部函数 _ezclump 处理掩码数组
    return _ezclump(mask)


###############################################################################
#                              Polynomial fit                                 #
###############################################################################


def vander(x, n=None):
    """
    输入数组中的掩盖值导致生成的行为全为零的矩阵。

    """
    # 使用 np.vander 函数生成 Vandermonde 矩阵
    _vander = np.vander(x, n)
    # 获取输入数组的掩码
    m = getmask(x)
    # 如果掩码不是空，则将对应位置的行置为零
    if m is not nomask:
        _vander[m] = 0
    return _vander

vander.__doc__ = ma.doc_note(np.vander.__doc__, vander.__doc__)


def polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False):
    """
    x 中的任何掩盖值会在 y 中传播，反之亦然。

    """
    # 将输入转换为数组
    x = asarray(x)
    y = asarray(y)

    # 获取 x 的掩码
    m = getmask(x)
    # 如果 y 是一维数组
    if y.ndim == 1:
        # 获取 y 的掩码并将两者合并
        m = mask_or(m, getmask(y))
    # 如果 y 是二维数组
    elif y.ndim == 2:
        # 获取 y 行的掩码并将两者合并
        my = getmask(mask_rows(y))
        if my is not nomask:
            m = mask_or(m, my[:, 0])
    else:
        # 抛出类型错误，y 应该是一维或二维数组
        raise TypeError("Expected a 1D or 2D array for y!")

    # 如果有权重 w
    if w is not None:
        w = asarray(w)
        # 检查权重数组的维度是否正确
        if w.ndim != 1:
            raise TypeError("expected a 1-d array for weights")
        if w.shape[0] != y.shape[0]:
            raise TypeError("expected w and y to have the same length")
        # 合并掩码
        m = mask_or(m, getmask(w))

    # 如果存在掩码
    if m is not nomask:
        # 获取非掩码索引
        not_m = ~m
        # 如果有权重，只使用非掩码部分进行拟合
        if w is not None:
            w = w[not_m]
        return np.polyfit(x[not_m], y[not_m], deg, rcond, full, w, cov)
    else:
        # 没有掩码，直接使用所有数据进行拟合
        return np.polyfit(x, y, deg, rcond, full, w, cov)

polyfit.__doc__ = ma.doc_note(np.polyfit.__doc__, polyfit.__doc__)
```