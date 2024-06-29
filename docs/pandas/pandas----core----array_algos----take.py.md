# `D:\src\scipysrc\pandas\pandas\core\array_algos\take.py`

```
@overload
# 函数重载装饰器，定义了一种特殊的函数签名
def take_nd(
    arr: np.ndarray,
    indexer,
    axis: AxisInt = ...,
    fill_value=...,
    allow_fill: bool = ...,
) -> np.ndarray: ...

@overload
# 函数重载装饰器，定义了另一种特殊的函数签名
def take_nd(
    arr: ExtensionArray,
    indexer,
    axis: AxisInt = ...,
    fill_value=...,
    allow_fill: bool = ...,
) -> ArrayLike: ...


def take_nd(
    arr: ArrayLike,
    indexer,
    axis: AxisInt = 0,
    fill_value=lib.no_default,
    allow_fill: bool = True,
) -> ArrayLike:
    """
    Specialized Cython take which sets NaN values in one pass

    This dispatches to ``take`` defined on ExtensionArrays.

    Note: this function assumes that the indexer is a valid(ated) indexer with
    no out of bound indices.

    Parameters
    ----------
    arr : np.ndarray or ExtensionArray
        输入数组。
    indexer : ndarray
        用于获取元素的一维索引数组，与值为-1的索引对应的子数组将使用fill_value填充。
    axis : int, default 0
        获取元素的轴。
    fill_value : any, default np.nan
        用于替换-1值的填充值。
    allow_fill : bool, default True
        如果为False，则假定索引器不包含-1值，因此不会执行填充操作。这会绕过掩码的计算。如果allow_fill为False且索引器中包含-1，则结果是未定义的。

    Returns
    -------
    subarray : np.ndarray or ExtensionArray
        可能与输入的类型相同，也可能转换为ndarray类型。
    """
    if fill_value is lib.no_default:
        fill_value = na_value_for_dtype(arr.dtype, compat=False)
    elif lib.is_np_dtype(arr.dtype, "mM"):
        dtype, fill_value = maybe_promote(arr.dtype, fill_value)
        if arr.dtype != dtype:
            # EA.take is strict about returning a new object of the same type
            # so for that case cast upfront
            arr = arr.astype(dtype)
    # 如果输入的数组不是 NumPy 的 ndarray 类型
    if not isinstance(arr, np.ndarray):
        # 如果数组的数据类型是 ExtensionArray 类型
        # 这包括 DatetimeArray 和 TimedeltaArray
        if not is_1d_only_ea_dtype(arr.dtype):
            # 将数组强制转换为 NDArrayBackedExtensionArray 类型
            arr = cast("NDArrayBackedExtensionArray", arr)
            # 使用 take 方法从数组中按照索引取值
            return arr.take(
                indexer, fill_value=fill_value, allow_fill=allow_fill, axis=axis
            )
        
        # 如果数组是 DatetimeArray 或 TimedeltaArray，直接使用 take 方法取值
        return arr.take(indexer, fill_value=fill_value, allow_fill=allow_fill)

    # 将非 ndarray 类型的数组转换为 NumPy 的 ndarray 类型
    arr = np.asarray(arr)
    # 调用 _take_nd_ndarray 函数处理 ndarray 类型的数组和索引
    return _take_nd_ndarray(arr, indexer, axis, fill_value, allow_fill)
def _take_nd_ndarray(
    arr: np.ndarray,
    indexer: npt.NDArray[np.intp] | None,
    axis: AxisInt,
    fill_value,
    allow_fill: bool,
) -> np.ndarray:
    # 如果 indexer 为 None，则创建一个索引数组，长度等于 arr 在给定轴上的长度
    if indexer is None:
        indexer = np.arange(arr.shape[axis], dtype=np.intp)
        # 设置 dtype 和 fill_value 为 arr 的数据类型和填充值的类型
        dtype, fill_value = arr.dtype, arr.dtype.type()
    else:
        # 确保 indexer 是平台兼容的整数数组
        indexer = ensure_platform_int(indexer)

    # 预处理索引器和填充值，获取处理后的 dtype、fill_value 和 mask_info
    dtype, fill_value, mask_info = _take_preprocess_indexer_and_fill_value(
        arr, indexer, fill_value, allow_fill
    )

    # 如果 arr 是二维且按列优先存储
    flip_order = False
    if arr.ndim == 2 and arr.flags.f_contiguous:
        flip_order = True

    # 如果需要翻转顺序，则将 arr 转置，并重新计算 axis
    if flip_order:
        arr = arr.T
        axis = arr.ndim - axis - 1

    # 确定输出数组的形状
    out_shape_ = list(arr.shape)
    out_shape_[axis] = len(indexer)
    out_shape = tuple(out_shape_)

    # 根据是否按列优先存储和轴的位置选择不同的存储顺序来创建空的输出数组
    if arr.flags.f_contiguous and axis == arr.ndim - 1:
        # 对于直接从二维 ndarray 初始化的数据框，这是一个可以提升一个数量级的微调
        out = np.empty(out_shape, dtype=dtype, order="F")
    else:
        out = np.empty(out_shape, dtype=dtype)

    # 获取处理索引的函数
    func = _get_take_nd_function(
        arr.ndim, arr.dtype, out.dtype, axis=axis, mask_info=mask_info
    )
    # 使用 func 处理 arr、indexer 和 out，同时填充 fill_value
    func(arr, indexer, out, fill_value)

    # 如果之前翻转了顺序，则重新将输出数组转置回来
    if flip_order:
        out = out.T
    # 返回输出数组
    return out


def take_2d_multi(
    arr: np.ndarray,
    indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]],
    fill_value=np.nan,
) -> np.ndarray:
    """
    Specialized Cython take which sets NaN values in one pass.
    """
    # 这个函数只在 DataFrame._reindex_multi 的一个地方被调用，因此我们知道 indexer 是符合预期的。
    assert indexer is not None
    assert indexer[0] is not None
    assert indexer[1] is not None

    # 将索引数组转换为平台兼容的整数数组
    row_idx, col_idx = indexer
    row_idx = ensure_platform_int(row_idx)
    col_idx = ensure_platform_int(col_idx)
    indexer = row_idx, col_idx
    mask_info = None

    # 根据类型推断是否需要提升（这样做比计算掩码更快）
    dtype, fill_value = maybe_promote(arr.dtype, fill_value)

    # 如果推断的 dtype 与 arr 的 dtype 不同
    if dtype != arr.dtype:
        # 检查是否基于索引需要提升
        row_mask = row_idx == -1
        col_mask = col_idx == -1
        row_needs = row_mask.any()
        col_needs = col_mask.any()
        mask_info = (row_mask, col_mask), (row_needs, col_needs)

        # 如果不需要提升，则降级 dtype，并将 fill_value 设置为虚拟的类型，以避免在 Cython 代码中的转换问题
        if not (row_needs or col_needs):
            dtype, fill_value = arr.dtype, arr.dtype.type()

    # 确保 dtype 能够容纳 arr 的值和 fill_value
    out_shape = len(row_idx), len(col_idx)
    # 创建一个指定形状和数据类型的空数组
    out = np.empty(out_shape, dtype=dtype)

    # 根据输入数组 arr 和输出数组 out 的数据类型，从字典 _take_2d_multi_dict 中获取相应的函数
    func = _take_2d_multi_dict.get((arr.dtype.name, out.dtype.name), None)
    
    # 如果未找到适合的函数，并且输入数组 arr 的数据类型与输出数组 out 的数据类型不同，则再次尝试查找
    if func is None and arr.dtype != out.dtype:
        func = _take_2d_multi_dict.get((out.dtype.name, out.dtype.name), None)
        
        # 如果找到了对应的函数，将其包装成适合输出数据类型的函数
        if func is not None:
            func = _convert_wrapper(func, out.dtype)

    # 如果找到了合适的函数，则调用该函数进行处理
    if func is not None:
        func(arr, indexer, out=out, fill_value=fill_value)
    else:
        # 如果没有找到合适的函数，则调用默认的处理函数 _take_2d_multi_object
        _take_2d_multi_object(
            arr, indexer, out, fill_value=fill_value, mask_info=mask_info
        )

    # 返回处理后的输出数组
    return out
# 使用 functools 模块的 lru_cache 装饰器，对下面的函数进行结果缓存
@functools.lru_cache
def _get_take_nd_function_cached(
    ndim: int, arr_dtype: np.dtype, out_dtype: np.dtype, axis: AxisInt
):
    """
    Part of _get_take_nd_function below that doesn't need `mask_info` and thus
    can be cached (mask_info potentially contains a numpy ndarray which is not
    hashable and thus cannot be used as argument for cached function).
    """
    # 构造元组 tup 用于缓存键值，包括输入和输出数据类型的名称
    tup = (arr_dtype.name, out_dtype.name)
    # 根据维度 ndim 和轴 axis 来选择相应的函数实现
    if ndim == 1:
        func = _take_1d_dict.get(tup, None)
    elif ndim == 2:
        if axis == 0:
            func = _take_2d_axis0_dict.get(tup, None)
        else:
            func = _take_2d_axis1_dict.get(tup, None)
    # 如果找到了对应的函数，直接返回
    if func is not None:
        return func

    # 在此处处理一些特定的数据类型，尝试从 _take_1d_dict 或 _take_2d_axis0_dict/_take_2d_axis1_dict 中获取函数
    # 如果找到了对应的函数，将其转换为 conv_dtype 类型后返回
    tup = (out_dtype.name, out_dtype.name)
    if ndim == 1:
        func = _take_1d_dict.get(tup, None)
    elif ndim == 2:
        if axis == 0:
            func = _take_2d_axis0_dict.get(tup, None)
        else:
            func = _take_2d_axis1_dict.get(tup, None)
    if func is not None:
        func = _convert_wrapper(func, out_dtype)
        return func

    # 如果未找到合适的函数，返回 None
    return None


def _get_take_nd_function(
    ndim: int,
    arr_dtype: np.dtype,
    out_dtype: np.dtype,
    axis: AxisInt = 0,
    mask_info=None,
):
    """
    Get the appropriate "take" implementation for the given dimension, axis
    and dtypes.
    """
    func = None
    # 如果维度不超过 2，调用 _get_take_nd_function_cached 函数获取函数实现
    if ndim <= 2:
        func = _get_take_nd_function_cached(ndim, arr_dtype, out_dtype, axis)

    # 如果 func 为空，则定义一个内部函数 func 处理数组取值操作
    if func is None:

        def func(arr, indexer, out, fill_value=np.nan) -> None:
            # 确保 indexer 是平台整数类型
            indexer = ensure_platform_int(indexer)
            # 调用 _take_nd_object 函数处理数组 arr 的取值操作，可以传递 mask_info
            _take_nd_object(
                arr, indexer, out, axis=axis, fill_value=fill_value, mask_info=mask_info
            )

    # 返回选定的处理函数 func
    return func


def _view_wrapper(f, arr_dtype=None, out_dtype=None, fill_wrap=None):
    """
    Wrap a function f with additional dtype conversion and view operations
    based on optional arguments.
    """
    # 定义一个包装器函数 wrapper，对函数 f 进行视图和数据类型转换操作
    def wrapper(
        arr: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=np.nan
    ) -> None:
        # 如果指定了 arr_dtype，则将 arr 视图化为 arr_dtype 类型
        if arr_dtype is not None:
            arr = arr.view(arr_dtype)
        # 如果指定了 out_dtype，则将 out 视图化为 out_dtype 类型
        if out_dtype is not None:
            out = out.view(out_dtype)
        # 如果 fill_wrap 不为 None，则根据 fill_value 的类型进行转换
        if fill_wrap is not None:
            # FIXME: 如果 fill_value 的类型是日期/时间类型，确保具有匹配的解析选项
            if fill_value.dtype.kind == "m":
                fill_value = fill_value.astype("m8[ns]")
            else:
                fill_value = fill_value.astype("M8[ns]")
            # 使用 fill_wrap 函数对 fill_value 进行包装处理
            fill_value = fill_wrap(fill_value)

        # 调用原始函数 f 处理视图化后的 arr、indexer 和 out，同时传递 fill_value
        f(arr, indexer, out, fill_value=fill_value)

    # 返回定义的 wrapper 包装器函数
    return wrapper


def _convert_wrapper(f, conv_dtype):
    """
    Wrap a function f with dtype conversion based on conv_dtype.
    """
    # 定义一个包装器函数 wrapper，对函数 f 进行 conv_dtype 数据类型转换
    def wrapper(
        arr: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=np.nan
    ) -> None:
        # 调用 f 函数处理数组 arr、indexer 和 out，同时传递 fill_value
        f(arr, indexer, out, fill_value=fill_value)

    # 返回定义的 wrapper 包装器函数
    return wrapper
    ) -> None:
        # 如果 conv_dtype 是 object 类型，则调用 ensure_wrapped_if_datetimelike 函数避免将 dt64/td64 转换为整数
        if conv_dtype == object:
            arr = ensure_wrapped_if_datetimelike(arr)
        # 将 arr 数组转换为指定的 conv_dtype 类型
        arr = arr.astype(conv_dtype)
        # 调用函数 f 处理 arr 数组，使用 indexer 进行索引，将结果输出到 out，使用 fill_value 进行填充
        f(arr, indexer, out, fill_value=fill_value)

    # 返回包装后的函数 wrapper
    return wrapper
# 定义用于一维数组取值操作的函数映射字典
_take_1d_dict = {
    ("int8", "int8"): libalgos.take_1d_int8_int8,  # 当源数组和目标数组类型均为int8时，使用对应的take函数
    ("int8", "int32"): libalgos.take_1d_int8_int32,  # 当源数组为int8，目标数组为int32时，使用对应的take函数
    ("int8", "int64"): libalgos.take_1d_int8_int64,  # 当源数组为int8，目标数组为int64时，使用对应的take函数
    ("int8", "float64"): libalgos.take_1d_int8_float64,  # 当源数组为int8，目标数组为float64时，使用对应的take函数
    ("int16", "int16"): libalgos.take_1d_int16_int16,  # 类似地，定义了各种不同类型组合下的take函数
    ("int16", "int32"): libalgos.take_1d_int16_int32,
    ("int16", "int64"): libalgos.take_1d_int16_int64,
    ("int16", "float64"): libalgos.take_1d_int16_float64,
    ("int32", "int32"): libalgos.take_1d_int32_int32,
    ("int32", "int64"): libalgos.take_1d_int32_int64,
    ("int32", "float64"): libalgos.take_1d_int32_float64,
    ("int64", "int64"): libalgos.take_1d_int64_int64,
    ("uint8", "uint8"): libalgos.take_1d_bool_bool,
    ("uint16", "int64"): libalgos.take_1d_uint16_uint16,
    ("uint32", "int64"): libalgos.take_1d_uint32_uint32,
    ("uint64", "int64"): libalgos.take_1d_uint64_uint64,
    ("int64", "float64"): libalgos.take_1d_int64_float64,
    ("float32", "float32"): libalgos.take_1d_float32_float32,
    ("float32", "float64"): libalgos.take_1d_float32_float64,
    ("float64", "float64"): libalgos.take_1d_float64_float64,
    ("object", "object"): libalgos.take_1d_object_object,
    ("bool", "bool"): _view_wrapper(libalgos.take_1d_bool_bool, np.uint8, np.uint8),  # 当源数组和目标数组均为bool时，使用包装函数_view_wrapper
    ("bool", "object"): _view_wrapper(libalgos.take_1d_bool_object, np.uint8, None),  # 当源数组为bool，目标数组为object时，使用包装函数_view_wrapper
    ("datetime64[ns]", "datetime64[ns]"): _view_wrapper(
        libalgos.take_1d_int64_int64, np.int64, np.int64, np.int64  # 当源数组和目标数组均为datetime64[ns]时，使用包装函数_view_wrapper
    ),
    ("timedelta64[ns]", "timedelta64[ns]"): _view_wrapper(
        libalgos.take_1d_int64_int64, np.int64, np.int64, np.int64  # 当源数组和目标数组均为timedelta64[ns]时，使用包装函数_view_wrapper
    ),
}

# 定义用于二维数组按轴0取值操作的函数映射字典
_take_2d_axis0_dict = {
    ("int8", "int8"): libalgos.take_2d_axis0_int8_int8,  # 当源数组和目标数组类型均为int8时，使用对应的take函数
    ("int8", "int32"): libalgos.take_2d_axis0_int8_int32,  # 当源数组为int8，目标数组为int32时，使用对应的take函数
    ("int8", "int64"): libalgos.take_2d_axis0_int8_int64,  # 当源数组为int8，目标数组为int64时，使用对应的take函数
    ("int8", "float64"): libalgos.take_2d_axis0_int8_float64,  # 当源数组为int8，目标数组为float64时，使用对应的take函数
    ("int16", "int16"): libalgos.take_2d_axis0_int16_int16,  # 类似地，定义了各种不同类型组合下的take函数
    ("int16", "int32"): libalgos.take_2d_axis0_int16_int32,
    ("int16", "int64"): libalgos.take_2d_axis0_int16_int64,
    ("int16", "float64"): libalgos.take_2d_axis0_int16_float64,
    ("int32", "int32"): libalgos.take_2d_axis0_int32_int32,
    ("int32", "int64"): libalgos.take_2d_axis0_int32_int64,
    ("int32", "float64"): libalgos.take_2d_axis0_int32_float64,
    ("int64", "int64"): libalgos.take_2d_axis0_int64_int64,
    ("int64", "float64"): libalgos.take_2d_axis0_int64_float64,
    ("uint8", "uint8"): libalgos.take_2d_axis0_bool_bool,
    ("uint16", "uint16"): libalgos.take_2d_axis0_uint16_uint16,
    ("uint32", "uint32"): libalgos.take_2d_axis0_uint32_uint32,
    ("uint64", "uint64"): libalgos.take_2d_axis0_uint64_uint64,
    ("float32", "float32"): libalgos.take_2d_axis0_float32_float32,
    ("float32", "float64"): libalgos.take_2d_axis0_float32_float64,
    ("float64", "float64"): libalgos.take_2d_axis0_float64_float64,
    ("object", "object"): libalgos.take_2d_axis0_object_object,
}
    # 对于 ("bool", "bool") 类型的元组，使用 libalgos.take_2d_axis0_bool_bool 函数进行包装
    ("bool", "bool"): _view_wrapper(
        libalgos.take_2d_axis0_bool_bool, np.uint8, np.uint8
    ),
    
    # 对于 ("bool", "object") 类型的元组，使用 libalgos.take_2d_axis0_bool_object 函数进行包装
    ("bool", "object"): _view_wrapper(
        libalgos.take_2d_axis0_bool_object, np.uint8, None
    ),
    
    # 对于 ("datetime64[ns]", "datetime64[ns]") 类型的元组，使用 libalgos.take_2d_axis0_int64_int64 函数进行包装
    # 输入和输出的数据类型都是 np.int64，同时使用 np.int64 进行填充和包装
    ("datetime64[ns]", "datetime64[ns]"): _view_wrapper(
        libalgos.take_2d_axis0_int64_int64, np.int64, np.int64, fill_wrap=np.int64
    ),
    
    # 对于 ("timedelta64[ns]", "timedelta64[ns]") 类型的元组，使用 libalgos.take_2d_axis0_int64_int64 函数进行包装
    # 输入和输出的数据类型都是 np.int64，同时使用 np.int64 进行填充和包装
    ("timedelta64[ns]", "timedelta64[ns]"): _view_wrapper(
        libalgos.take_2d_axis0_int64_int64, np.int64, np.int64, fill_wrap=np.int64
    ),
# 定义一个字典 `_take_2d_axis1_dict`，包含多种数据类型组合的处理函数
_take_2d_axis1_dict = {
    ("int8", "int8"): libalgos.take_2d_axis1_int8_int8,
    ("int8", "int32"): libalgos.take_2d_axis1_int8_int32,
    ("int8", "int64"): libalgos.take_2d_axis1_int8_int64,
    ("int8", "float64"): libalgos.take_2d_axis1_int8_float64,
    ("int16", "int16"): libalgos.take_2d_axis1_int16_int16,
    ("int16", "int32"): libalgos.take_2d_axis1_int16_int32,
    ("int16", "int64"): libalgos.take_2d_axis1_int16_int64,
    ("int16", "float64"): libalgos.take_2d_axis1_int16_float64,
    ("int32", "int32"): libalgos.take_2d_axis1_int32_int32,
    ("int32", "int64"): libalgos.take_2d_axis1_int32_int64,
    ("int32", "float64"): libalgos.take_2d_axis1_int32_float64,
    ("int64", "int64"): libalgos.take_2d_axis1_int64_int64,
    ("int64", "float64"): libalgos.take_2d_axis1_int64_float64,
    ("uint8", "uint8"): libalgos.take_2d_axis1_bool_bool,
    ("uint16", "uint16"): libalgos.take_2d_axis1_uint16_uint16,
    ("uint32", "uint32"): libalgos.take_2d_axis1_uint32_uint32,
    ("uint64", "uint64"): libalgos.take_2d_axis1_uint64_uint64,
    ("float32", "float32"): libalgos.take_2d_axis1_float32_float32,
    ("float32", "float64"): libalgos.take_2d_axis1_float32_float64,
    ("float64", "float64"): libalgos.take_2d_axis1_float64_float64,
    ("object", "object"): libalgos.take_2d_axis1_object_object,
    # 处理布尔型数据组合时，使用 `_view_wrapper` 包装函数调用 `libalgos.take_2d_axis1_bool_bool`
    ("bool", "bool"): _view_wrapper(
        libalgos.take_2d_axis1_bool_bool, np.uint8, np.uint8
    ),
    # 处理布尔型和对象型数据组合时，使用 `_view_wrapper` 包装函数调用 `libalgos.take_2d_axis1_bool_object`
    ("bool", "object"): _view_wrapper(
        libalgos.take_2d_axis1_bool_object, np.uint8, None
    ),
    # 处理日期时间型数据组合时，使用 `_view_wrapper` 包装函数调用 `libalgos.take_2d_axis1_int64_int64`
    ("datetime64[ns]", "datetime64[ns]"): _view_wrapper(
        libalgos.take_2d_axis1_int64_int64, np.int64, np.int64, fill_wrap=np.int64
    ),
    # 处理时间间隔型数据组合时，使用 `_view_wrapper` 包装函数调用 `libalgos.take_2d_axis1_int64_int64`
    ("timedelta64[ns]", "timedelta64[ns]"): _view_wrapper(
        libalgos.take_2d_axis1_int64_int64, np.int64, np.int64, fill_wrap=np.int64
    ),
}



# 定义另一个字典 `_take_2d_multi_dict`，包含多种数据类型组合的处理函数
_take_2d_multi_dict = {
    ("int8", "int8"): libalgos.take_2d_multi_int8_int8,
    ("int8", "int32"): libalgos.take_2d_multi_int8_int32,
    ("int8", "int64"): libalgos.take_2d_multi_int8_int64,
    ("int8", "float64"): libalgos.take_2d_multi_int8_float64,
    ("int16", "int16"): libalgos.take_2d_multi_int16_int16,
    ("int16", "int32"): libalgos.take_2d_multi_int16_int32,
    ("int16", "int64"): libalgos.take_2d_multi_int16_int64,
    ("int16", "float64"): libalgos.take_2d_multi_int16_float64,
    ("int32", "int32"): libalgos.take_2d_multi_int32_int32,
    ("int32", "int64"): libalgos.take_2d_multi_int32_int64,
    ("int32", "float64"): libalgos.take_2d_multi_int32_float64,
    ("int64", "int64"): libalgos.take_2d_multi_int64_int64,
    ("int64", "float64"): libalgos.take_2d_multi_int64_float64,
    ("float32", "float32"): libalgos.take_2d_multi_float32_float32,
    ("float32", "float64"): libalgos.take_2d_multi_float32_float64,
    ("float64", "float64"): libalgos.take_2d_multi_float64_float64,
    ("object", "object"): libalgos.take_2d_multi_object_object,
}
    # 对于 ("bool", "bool") 类型，使用 libalgos.take_2d_multi_bool_bool 函数创建视图包装器，
    # 输入类型为 np.uint8，输出类型也为 np.uint8
    ("bool", "bool"): _view_wrapper(
        libalgos.take_2d_multi_bool_bool, np.uint8, np.uint8
    ),
    # 对于 ("bool", "object") 类型，使用 libalgos.take_2d_multi_bool_object 函数创建视图包装器，
    # 输入类型为 np.uint8，输出类型为 None（即没有指定输出类型）
    ("bool", "object"): _view_wrapper(
        libalgos.take_2d_multi_bool_object, np.uint8, None
    ),
    # 对于 ("datetime64[ns]", "datetime64[ns]") 类型，使用 libalgos.take_2d_multi_int64_int64 函数创建视图包装器，
    # 输入类型为 np.int64，输出类型也为 np.int64，同时指定 fill_wrap 参数为 np.int64
    ("datetime64[ns]", "datetime64[ns]"): _view_wrapper(
        libalgos.take_2d_multi_int64_int64, np.int64, np.int64, fill_wrap=np.int64
    ),
    # 对于 ("timedelta64[ns]", "timedelta64[ns]") 类型，使用 libalgos.take_2d_multi_int64_int64 函数创建视图包装器，
    # 输入类型为 np.int64，输出类型也为 np.int64，同时指定 fill_wrap 参数为 np.int64
    ("timedelta64[ns]", "timedelta64[ns]"): _view_wrapper(
        libalgos.take_2d_multi_int64_int64, np.int64, np.int64, fill_wrap=np.int64
    ),
}

# 定义一个函数 _take_nd_object，用于处理一维或多维数组的索引和填充值
def _take_nd_object(
    arr: np.ndarray,
    indexer: npt.NDArray[np.intp],
    out: np.ndarray,
    axis: AxisInt,
    fill_value,  # 填充值，可以是任意类型
    mask_info,   # 表示是否需要进行掩码处理的元组或 None
) -> None:
    # 如果 mask_info 不为 None，则解构 mask_info 元组
    if mask_info is not None:
        mask, needs_masking = mask_info
    else:
        # 否则，根据 indexer 是否为 -1 来生成掩码 mask，并检查是否需要掩码处理
        mask = indexer == -1
        needs_masking = mask.any()
    # 如果 arr 的数据类型不等于 out 的数据类型，则将 arr 转换为 out 的数据类型
    if arr.dtype != out.dtype:
        arr = arr.astype(out.dtype)
    # 如果沿指定轴的 arr 的大小大于 0，则使用索引器 indexer 在指定轴上取值，结果保存到 out 中
    if arr.shape[axis] > 0:
        arr.take(indexer, axis=axis, out=out)
    # 如果需要进行掩码处理，则根据掩码 mask 将 out 的相应部分赋值为 fill_value
    if needs_masking:
        outindexer = [slice(None)] * arr.ndim
        outindexer[axis] = mask
        out[tuple(outindexer)] = fill_value


# 定义一个函数 _take_2d_multi_object，用于处理二维数组的多维索引和填充值
def _take_2d_multi_object(
    arr: np.ndarray,
    indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]],
    out: np.ndarray,
    fill_value,  # 填充值，可以是任意类型
    mask_info,   # 表示是否需要进行掩码处理的元组或 None
) -> None:
    # 这段代码性能不理想，但比抛出异常要好（最好在 Cython 中进行优化以避免进入此处）
    row_idx, col_idx = indexer  # 将索引器解构为行索引和列索引
    # 如果 mask_info 不为 None，则解构 mask_info 元组，分别获取行和列的掩码信息及是否需要掩码处理的布尔值
    if mask_info is not None:
        (row_mask, col_mask), (row_needs, col_needs) = mask_info
    else:
        # 否则，根据行索引和列索引是否为 -1 来生成行和列的掩码，以及是否需要掩码处理的布尔值
        row_mask = row_idx == -1
        col_mask = col_idx == -1
        row_needs = row_mask.any()
        col_needs = col_mask.any()
    # 如果 fill_value 不为 None，则根据需要，将填充值赋给输出数组的行或列
    if fill_value is not None:
        if row_needs:
            out[row_mask, :] = fill_value
        if col_needs:
            out[:, col_mask] = fill_value
    # 使用循环遍历行索引和列索引，将 arr 中对应位置的值复制到 out 中
    for i, u_ in enumerate(row_idx):
        if u_ != -1:
            for j, v in enumerate(col_idx):
                if v != -1:
                    out[i, j] = arr[u_, v]


# 定义一个函数 _take_preprocess_indexer_and_fill_value，用于预处理索引器和填充值
def _take_preprocess_indexer_and_fill_value(
    arr: np.ndarray,
    indexer: npt.NDArray[np.intp],
    fill_value,  # 填充值，可以是任意类型
    allow_fill: bool,
    mask: npt.NDArray[np.bool_] | None = None,
):
    # mask_info 初始化为 None
    mask_info: tuple[np.ndarray | None, bool] | None = None

    # 如果不允许填充，则将 dtype 设置为 arr 的数据类型，填充值设置为 arr 的数据类型的实例
    if not allow_fill:
        dtype, fill_value = arr.dtype, arr.dtype.type()
        mask_info = None, False
    else:
        # 否则，基于类型判断是否需要提升 fill_value 的数据类型
        dtype, fill_value = maybe_promote(arr.dtype, fill_value)
        # 如果提升后的 dtype 不等于 arr 的 dtype
        if dtype != arr.dtype:
            # 如果有 mask，则需要进行掩码处理
            if mask is not None:
                needs_masking = True
            else:
                # 否则，根据 indexer 是否为 -1 来生成掩码 mask，并检查是否需要掩码处理
                mask = indexer == -1
                needs_masking = bool(mask.any())
            # 设置 mask_info 为 mask 和是否需要掩码处理的元组，若不需要掩码处理，则将 dtype 和 fill_value 恢复为 arr 的 dtype
            mask_info = mask, needs_masking
            if not needs_masking:
                dtype, fill_value = arr.dtype, arr.dtype.type()

    # 返回 dtype、fill_value 和 mask_info
    return dtype, fill_value, mask_info
```