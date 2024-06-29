# `D:\src\scipysrc\pandas\pandas\core\construction.py`

```
    """
    Constructor functions intended to be shared by pd.array, Series.__init__,
    and Index.__new__.
    
    These should not depend on core.internals.
    """

    from __future__ import annotations

    from typing import (
        TYPE_CHECKING,
        cast,
        overload,
    )

    import numpy as np
    from numpy import ma

    from pandas._config import using_pyarrow_string_dtype

    from pandas._libs import lib
    from pandas._libs.tslibs import (
        get_supported_dtype,
        is_supported_dtype,
    )

    from pandas.core.dtypes.base import ExtensionDtype
    from pandas.core.dtypes.cast import (
        construct_1d_arraylike_from_scalar,
        construct_1d_object_array_from_listlike,
        maybe_cast_to_datetime,
        maybe_cast_to_integer_array,
        maybe_convert_platform,
        maybe_infer_to_datetimelike,
        maybe_promote,
    )
    from pandas.core.dtypes.common import (
        ensure_object,
        is_list_like,
        is_object_dtype,
        pandas_dtype,
    )
    from pandas.core.dtypes.dtypes import NumpyEADtype
    from pandas.core.dtypes.generic import (
        ABCDataFrame,
        ABCExtensionArray,
        ABCIndex,
        ABCSeries,
    )
    from pandas.core.dtypes.missing import isna

    import pandas.core.common as com

    if TYPE_CHECKING:
        from collections.abc import Sequence

        from pandas._typing import (
            AnyArrayLike,
            ArrayLike,
            Dtype,
            DtypeObj,
            T,
        )

        from pandas import (
            Index,
            Series,
        )
        from pandas.core.arrays import (
            DatetimeArray,
            ExtensionArray,
            TimedeltaArray,
        )


    def array(
        data: Sequence[object] | AnyArrayLike,
        dtype: Dtype | None = None,
        copy: bool = True,
    ) -> ExtensionArray:
        """
        Create an array.

        Parameters
        ----------
        data : Sequence of objects
            The scalars inside `data` should be instances of the
            scalar type for `dtype`. It's expected that `data`
            represents a 1-dimensional array of data.

            When `data` is an Index or Series, the underlying array
            will be extracted from `data`.
        """
    # 定义参数 `dtype`，用于指定数组的数据类型，可以是 NumPy 的 dtype 或通过
    # `pandas.api.extensions.register_extension_dtype` 注册的扩展类型
    dtype : str, np.dtype, or ExtensionDtype, optional
        # 用于数组的数据类型。可以是 NumPy 的 dtype 或通过
        # `pandas.api.extensions.register_extension_dtype` 注册的扩展类型

        # 如果未指定，有两种可能性：
        # 1. 当 `data` 是 Series、Index 或 ExtensionArray 时，数据类型将从数据中获取
        # 2. 否则，pandas 将尝试从数据推断数据类型

        # 注意，当 `data` 是 NumPy 数组时，不会使用 `data.dtype` 推断数组类型。
        # 这是因为 NumPy 不能表示所有可能存储在扩展数组中的数据类型。

        # 当前，pandas 会为以下序列推断扩展数据类型：
        # ============================== =======================================
        # 标量类型                         数组类型
        # ============================== =======================================
        # :class:`pandas.Interval`       :class:`pandas.arrays.IntervalArray`
        # :class:`pandas.Period`         :class:`pandas.arrays.PeriodArray`
        # :class:`datetime.datetime`     :class:`pandas.arrays.DatetimeArray`
        # :class:`datetime.timedelta`    :class:`pandas.arrays.TimedeltaArray`
        # :class:`int`                   :class:`pandas.arrays.IntegerArray`
        # :class:`float`                 :class:`pandas.arrays.FloatingArray`
        # :class:`str`                   :class:`pandas.arrays.StringArray` 或
        #                                :class:`pandas.arrays.ArrowStringArray`
        # :class:`bool`                  :class:`pandas.arrays.BooleanArray`
        # ============================== =======================================

        # 当标量类型是 :class:`str` 时创建的 ExtensionArray 取决于
        # `pd.options.mode.string_storage`，如果未显式给定 dtype。

        # 对于其他情况，将使用 NumPy 的通常推断规则。
    copy : bool, default True
        # 是否复制数据，即使没有必要。根据 `data` 的类型，创建新数组可能需要复制数据，
        # 即使 `copy=False`。

    Returns
    -------
    ExtensionArray
        # 新创建的数组。

    Raises
    ------
    ValueError
        # 当 `data` 不是一维数组时引发异常。

    See Also
    --------
    numpy.array : 构造 NumPy 数组。
    Series : 构造 pandas Series。
    Index : 构造 pandas Index。
    arrays.NumpyExtensionArray : 封装 NumPy 数组的 ExtensionArray。
    Series.array : 提取存储在 Series 中的数组。

    Notes
    -----
    # 省略 `dtype` 参数意味着 pandas 将尝试从数据值中推断出最佳数组类型。
    # 随着 pandas 和第三方库添加新的数组类型，"最佳" 数组类型可能会改变。
    change. We recommend specifying `dtype` to ensure that

# 建议指定 `dtype` 参数，以确保：
# 1. 返回正确的数组数据类型
# 2. 随着 pandas 和第三方库添加新的扩展类型，返回的数组类型不会改变

    1. the correct array type for the data is returned
    2. the returned array type doesn't change as new extension types
       are added by pandas and third-party libraries

# 返回正确的数组类型，以确保数据类型正确，不会因 pandas 和第三方库添加新的扩展类型而改变返回的数组类型


    Additionally, if the underlying memory representation of the returned
    array matters, we recommend specifying the `dtype` as a concrete object
    rather than a string alias or allowing it to be inferred. For example,
    a future version of pandas or a 3rd-party library may include a
    dedicated ExtensionArray for string data. In this event, the following
    would no longer return a :class:`arrays.NumpyExtensionArray` backed by a
    NumPy array.

# 此外，如果返回的数组的底层内存表示很重要，我们建议将 `dtype` 指定为具体对象，而不是字符串别名或允许其推断。例如，未来的 pandas 或第三方库版本可能会包括专用于字符串数据的 ExtensionArray。在这种情况下，以下示例将不再返回由 NumPy 数组支持的 :class:`arrays.NumpyExtensionArray`。


    >>> pd.array(["a", "b"], dtype=str)
    <NumpyExtensionArray>
    ['a', 'b']
    Length: 2, dtype: str32

# 返回一个字符串数组，其中 dtype 为 str 类型的 NumpyExtensionArray


    This would instead return the new ExtensionArray dedicated for string
    data. If you really need the new array to be backed by a  NumPy array,
    specify that in the dtype.

# 相反，此时会返回专用于字符串数据的新 ExtensionArray。如果确实需要新的数组由 NumPy 数组支持，请在 dtype 中指定。


    >>> pd.array(["a", "b"], dtype=np.dtype("<U1"))
    <NumpyExtensionArray>
    ['a', 'b']
    Length: 2, dtype: str32

# 如果需要返回由 NumPy 数组支持的新数组，可以如下指定 dtype。


    Finally, Pandas has arrays that mostly overlap with NumPy

# 最后，Pandas 的数组大部分与 NumPy 重叠


      * :class:`arrays.DatetimeArray`
      * :class:`arrays.TimedeltaArray`

# - :class:`arrays.DatetimeArray`
# - :class:`arrays.TimedeltaArray`


    When data with a ``datetime64[ns]`` or ``timedelta64[ns]`` dtype is
    passed, pandas will always return a ``DatetimeArray`` or ``TimedeltaArray``
    rather than a ``NumpyExtensionArray``. This is for symmetry with the case of
    timezone-aware data, which NumPy does not natively support.

# 当传递 `datetime64[ns]` 或 `timedelta64[ns]` 数据类型时，Pandas 将始终返回 `DatetimeArray` 或 `TimedeltaArray`，而不是 `NumpyExtensionArray`。这是为了与时区感知数据的情况保持对称，而 NumPy 本身不支持时区感知数据。


    >>> pd.array(["2015", "2016"], dtype="datetime64[ns]")
    <DatetimeArray>
    ['2015-01-01 00:00:00', '2016-01-01 00:00:00']
    Length: 2, dtype: datetime64[ns]

# 返回一个 `datetime64[ns]` 类型的 `DatetimeArray`。


    >>> pd.array(["1h", "2h"], dtype="timedelta64[ns]")
    <TimedeltaArray>
    ['0 days 01:00:00', '0 days 02:00:00']
    Length: 2, dtype: timedelta64[ns]

# 返回一个 `timedelta64[ns]` 类型的 `TimedeltaArray`。


    Examples
    --------
    If a dtype is not specified, pandas will infer the best dtype from the values.
    See the description of `dtype` for the types pandas infers for.

# 示例
# 如果没有指定 dtype，则 Pandas 将根据值推断出最佳的 dtype。查看 `dtype` 的描述，了解 Pandas 推断的类型。


    >>> pd.array([1, 2])
    <IntegerArray>
    [1, 2]
    Length: 2, dtype: Int64

# 返回一个整数数组 `IntegerArray`。


    >>> pd.array([1, 2, np.nan])
    <IntegerArray>
    [1, 2, <NA>]
    Length: 3, dtype: Int64

# 返回一个包含 NaN 值的整数数组 `IntegerArray`。


    >>> pd.array([1.1, 2.2])
    <FloatingArray>
    [1.1, 2.2]
    Length: 2, dtype: Float64

# 返回一个浮点数数组 `FloatingArray`。


    >>> pd.array(["a", None, "c"])
    <StringArray>
    ['a', <NA>, 'c']
    Length: 3, dtype: string

# 返回一个字符串数组 `StringArray`，其中包含一个缺失值 `<NA>`。


    >>> with pd.option_context("string_storage", "pyarrow"):
    ...     arr = pd.array(["a", None, "c"])
    >>> arr
    <ArrowStringArray>
    ['a', <NA>, 'c']
    Length: 3, dtype: string

# 在设置 `"string_storage", "pyarrow"` 的上下文中返回一个使用 Arrow 库存储字符串的数组 `ArrowStringArray`。


    >>> pd.array([pd.Period("2000", freq="D"), pd.Period("2000", freq="D")])
    <PeriodArray>
    ['2000-01-01', '2000-01-01']
    Length: 2, dtype: period[D]

# 返回一个 `PeriodArray`，其中包含 `period[D]` 类型的数据。


    You can use the string alias for `dtype`

# 可以使用字符串别名指定 `dtype`。


    >>> pd.array(["a", "b", "a"], dtype="category")
    ['a', 'b', 'a']
    Categories (2, object): ['a', 'b']

# 返回一个分类数据类型 `category` 的数组。


    Or specify the actual dtype

# 或者指定实际的 `dtype`。
    from pandas.core.arrays import (  # 导入所需的 Pandas 扩展数组类型
        BooleanArray,  # 布尔类型数组
        DatetimeArray,  # 日期时间类型数组
        ExtensionArray,  # 扩展数组基类
        FloatingArray,  # 浮点数类型数组
        IntegerArray,  # 整数类型数组
        NumpyExtensionArray,  # NumPy 扩展数组
        TimedeltaArray,  # 时间增量类型数组
    )
    from pandas.core.arrays.string_ import StringDtype  # 导入字符串类型数组

    if lib.is_scalar(data):  # 如果数据是标量
        msg = f"Cannot pass scalar '{data}' to 'pandas.array'."
        raise ValueError(msg)  # 抛出值错误异常，不能将标量传递给 'pandas.array'
    elif isinstance(data, ABCDataFrame):  # 如果数据是 DataFrame 的实例
        raise TypeError("Cannot pass DataFrame to 'pandas.array'")  # 抛出类型错误异常，不能将 DataFrame 传递给 'pandas.array'

    if dtype is None and isinstance(data, (ABCSeries, ABCIndex, ExtensionArray)):
        # 如果 dtype 为空，并且数据是 Series、Index 或者 ExtensionArray 的实例
        # 注意：这里排除了 np.ndarray，将对其进行类型推断
        dtype = data.dtype  # 将数据的 dtype 赋值给 dtype

    data = extract_array(data, extract_numpy=True)  # 从数据中提取数组，提取 NumPy 数组

    # this returns None for not-found dtypes.
    # 如果指定了 dtype，则将其转换为 Pandas 的数据类型
    if dtype is not None:
        dtype = pandas_dtype(dtype)

    if isinstance(data, ExtensionArray) and (dtype is None or data.dtype == dtype):
        # 如果数据是扩展数组，并且 dtype 为空或者数据的 dtype 与指定的 dtype 相同
        # 例如 TimedeltaArray[s]，避免强制转换为 NumpyExtensionArray
        if copy:
            return data.copy()  # 如果需要复制，则返回数据的副本
        return data  # 否则返回原始数据

    if isinstance(dtype, ExtensionDtype):
        # 如果指定的 dtype 是扩展数据类型
        cls = dtype.construct_array_type()  # 构造数组类型
        return cls._from_sequence(data, dtype=dtype, copy=copy)  # 从序列数据创建数组对象，使用指定的 dtype 和复制选项
    # 如果未指定数据类型（dtype）
    if dtype is None:
        # 检查数据是否为 ndarray 类型
        was_ndarray = isinstance(data, np.ndarray)
        # 如果数据不是 ndarray 或者其 dtype 是 object 类型
        if not was_ndarray or data.dtype == object:  # type: ignore[union-attr]
            # 转换对象类型数据，确保所有非数值类型都被转换，并且转换为可空 dtype
            result = lib.maybe_convert_objects(
                ensure_object(data),
                convert_non_numeric=True,
                convert_to_nullable_dtype=True,
                dtype_if_all_nat=None,
            )
            # 确保日期时间类型数据被适当包装
            result = ensure_wrapped_if_datetimelike(result)
            # 如果结果是 ndarray 类型
            if isinstance(result, np.ndarray):
                # 如果结果长度为 0 且原数据不是 ndarray，则返回一个空的浮点型数组
                if len(result) == 0 and not was_ndarray:
                    # 例如：空列表情况下返回浮点型数组
                    return FloatingArray._from_sequence(data, dtype="Float64")
                # 从序列创建 NumpyExtensionArray
                return NumpyExtensionArray._from_sequence(
                    data, dtype=result.dtype, copy=copy
                )
            # 如果结果与原数据相同且需要复制数据
            if result is data and copy:
                return result.copy()
            return result

        # 强制转换数据为 ndarray 类型
        data = cast(np.ndarray, data)
        # 确保日期时间类型数据被适当包装
        result = ensure_wrapped_if_datetimelike(data)
        # 如果结果不是原数据，则强制类型转换为 DatetimeArray 或 TimedeltaArray
        if result is not data:
            result = cast("DatetimeArray | TimedeltaArray", result)
            # 如果需要复制数据且结果与原数据 dtype 相同，则返回复制的结果
            if copy and result.dtype == data.dtype:
                return result.copy()
            return result

        # 如果数据的 dtype 的种类是 "SU"，即字符串类型
        if data.dtype.kind in "SU":
            # 根据字符串类型创建 StringDtype
            dtype = StringDtype()
            # 构造对应的 StringArray/ArrowStringArray 类型
            cls = dtype.construct_array_type()
            return cls._from_sequence(data, dtype=dtype, copy=copy)

        # 如果数据的 dtype 的种类是 "iu"，即整数类型
        elif data.dtype.kind in "iu":
            # 从序列创建 IntegerArray
            return IntegerArray._from_sequence(data, copy=copy)
        
        # 如果数据的 dtype 的种类是 "f"，即浮点数类型
        elif data.dtype.kind == "f":
            # 排除 np.float16，因为 FloatingArray 不支持此类型；使用 NumpyExtensionArray 替代
            if data.dtype == np.float16:
                return NumpyExtensionArray._from_sequence(
                    data, dtype=data.dtype, copy=copy
                )
            # 从序列创建 FloatingArray
            return FloatingArray._from_sequence(data, copy=copy)

        # 如果数据的 dtype 的种类是 "b"，即布尔类型
        elif data.dtype.kind == "b":
            # 从序列创建 BooleanArray，指定 dtype 为 "boolean"
            return BooleanArray._from_sequence(data, dtype="boolean", copy=copy)
        
        # 其他情况，例如复数类型
        else:
            # 使用 NumpyExtensionArray 从序列创建对应类型的数组
            return NumpyExtensionArray._from_sequence(data, dtype=data.dtype, copy=copy)

    # Pandas 会覆盖 NumPy 的行为处理以下类型：
    #   1. datetime64[ns,us,ms,s]
    #   2. timedelta64[ns,us,ms,s]
    # 以确保返回 DatetimeArray 类型的结果
    if lib.is_np_dtype(dtype, "M") and is_supported_dtype(dtype):
        # 从序列创建 DatetimeArray，指定 dtype
        return DatetimeArray._from_sequence(data, dtype=dtype, copy=copy)
    if lib.is_np_dtype(dtype, "m") and is_supported_dtype(dtype):
        # 从序列创建 TimedeltaArray，指定 dtype
        return TimedeltaArray._from_sequence(data, dtype=dtype, copy=copy)
    elif lib.is_np_dtype(dtype, "mM"):
        # 如果 dtype 是 'm' 或 'M'，则抛出值错误异常
        raise ValueError(
            # GH#53817
            # 返回特定的错误消息，指出 datetime64 和 timedelta64 的 dtype 分辨率除了 's', 'ms', 'us', 'ns' 外不再支持
            r"datetime64 and timedelta64 dtype resolutions other than "
            r"'s', 'ms', 'us', and 'ns' are no longer supported."
        )

    # 使用 NumpyExtensionArray 类方法从给定数据序列创建扩展数组对象
    return NumpyExtensionArray._from_sequence(data, dtype=dtype, copy=copy)
# 定义一个不可变的集合 `_typs`，包含了一些字符串，代表了可能的对象类型
_typs = frozenset(
    {
        "index",
        "rangeindex",
        "multiindex",
        "datetimeindex",
        "timedeltaindex",
        "periodindex",
        "categoricalindex",
        "intervalindex",
        "series",
    }
)


# 函数签名装饰器，用于类型提示重载
@overload
def extract_array(
    obj: Series | Index, extract_numpy: bool = ..., extract_range: bool = ...
) -> ArrayLike: ...


# 函数签名装饰器，用于类型提示重载
@overload
def extract_array(
    obj: T, extract_numpy: bool = ..., extract_range: bool = ...
) -> T | ArrayLike: ...


# 从 Series 或 Index 中提取 ndarray 或 ExtensionArray 的函数定义
def extract_array(
    obj: T, extract_numpy: bool = False, extract_range: bool = False
) -> T | ArrayLike:
    """
    Extract the ndarray or ExtensionArray from a Series or Index.

    For all other types, `obj` is just returned as is.

    Parameters
    ----------
    obj : object
        For Series / Index, the underlying ExtensionArray is unboxed.

    extract_numpy : bool, default False
        Whether to extract the ndarray from a NumpyExtensionArray.

    extract_range : bool, default False
        If we have a RangeIndex, return range._values if True
        (which is a materialized integer ndarray), otherwise return unchanged.

    Returns
    -------
    arr : object

    Examples
    --------
    >>> extract_array(pd.Series(["a", "b", "c"], dtype="category"))
    ['a', 'b', 'c']
    Categories (3, object): ['a', 'b', 'c']

    Other objects like lists, arrays, and DataFrames are just passed through.

    >>> extract_array([1, 2, 3])
    [1, 2, 3]

    For an ndarray-backed Series / Index the ndarray is returned.

    >>> extract_array(pd.Series([1, 2, 3]))
    array([1, 2, 3])

    To extract all the way down to the ndarray, pass ``extract_numpy=True``.

    >>> extract_array(pd.Series([1, 2, 3]), extract_numpy=True)
    array([1, 2, 3])
    """
    # 获取 obj 的 _typ 属性的值
    typ = getattr(obj, "_typ", None)
    # 如果 typ 存在于 _typs 集合中
    if typ in _typs:
        # 如果 typ 是 "rangeindex"
        if typ == "rangeindex":
            # 如果 extract_range 为 True，返回 obj._values（假设 obj 有 _values 属性）
            if extract_range:
                return obj._values  # type: ignore[attr-defined]
            # 否则返回 obj 本身
            return obj

        # 对于其它类型的 obj，返回 obj._values（假设 obj 有 _values 属性）
        return obj._values  # type: ignore[attr-defined]

    # 如果 extract_numpy 为 True，并且 typ 是 "npy_extension"
    elif extract_numpy and typ == "npy_extension":
        # 返回 obj 转换成 numpy 数组的结果
        return obj.to_numpy()  # type: ignore[attr-defined]

    # 否则直接返回 obj
    return obj


# 确保将日期时间数组（datetime64 和 timedelta64）包装在 DatetimeArray / TimedeltaArray 中的函数定义
def ensure_wrapped_if_datetimelike(arr):
    """
    Wrap datetime64 and timedelta64 ndarrays in DatetimeArray/TimedeltaArray.
    """
    # 检查 arr 是否为 NumPy 的 ndarray 对象
    if isinstance(arr, np.ndarray):
        # 检查 arr 的数据类型是否为日期时间类型 ('M')
        if arr.dtype.kind == "M":
            # 导入 Pandas 的日期时间数组类
            from pandas.core.arrays import DatetimeArray
            
            # 获取支持的日期时间数据类型
            dtype = get_supported_dtype(arr.dtype)
            # 使用日期时间数组类从序列 arr 创建对象，并指定数据类型
            return DatetimeArray._from_sequence(arr, dtype=dtype)
        
        # 检查 arr 的数据类型是否为时间间隔类型 ('m')
        elif arr.dtype.kind == "m":
            # 导入 Pandas 的时间间隔数组类
            from pandas.core.arrays import TimedeltaArray
            
            # 获取支持的时间间隔数据类型
            dtype = get_supported_dtype(arr.dtype)
            # 使用时间间隔数组类从序列 arr 创建对象，并指定数据类型
            return TimedeltaArray._from_sequence(arr, dtype=dtype)
    
    # 如果不满足以上条件，则直接返回 arr
    return arr
def sanitize_masked_array(data: ma.MaskedArray) -> np.ndarray:
    """
    Convert numpy MaskedArray to ensure mask is softened.
    """
    # 获取数据的掩码数组
    mask = ma.getmaskarray(data)
    # 如果掩码数组中有任何True值
    if mask.any():
        # 获取适当的数据类型和填充值
        dtype, fill_value = maybe_promote(data.dtype, np.nan)
        # 强制类型转换为指定的数据类型
        dtype = cast(np.dtype, dtype)
        # 将数据转换为MaskedArray，并确保数据类型和副本要求
        data = ma.asarray(data.astype(dtype, copy=True))
        # 软化掩码，将硬掩码改为False
        data.soften_mask()  # set hardmask False if it was True
        # 将掩码位置填充为指定的填充值
        data[mask] = fill_value
    else:
        # 如果没有掩码，则创建数据的副本
        data = data.copy()
    # 返回处理后的数据
    return data


def sanitize_array(
    data,
    index: Index | None,
    dtype: DtypeObj | None = None,
    copy: bool = False,
    *,
    allow_2d: bool = False,
) -> ArrayLike:
    """
    Sanitize input data to an ndarray or ExtensionArray, copy if specified,
    coerce to the dtype if specified.

    Parameters
    ----------
    data : Any
    index : Index or None, default None
    dtype : np.dtype, ExtensionDtype, or None, default None
    copy : bool, default False
    allow_2d : bool, default False
        If False, raise if we have a 2D Arraylike.

    Returns
    -------
    np.ndarray or ExtensionArray
    """
    # 保存原始的数据类型
    original_dtype = dtype
    # 如果数据是MaskedArray类型，则调用sanitize_masked_array进行处理
    if isinstance(data, ma.MaskedArray):
        data = sanitize_masked_array(data)

    # 如果dtype是NumpyEADtype类型，则转换为其对应的numpy数据类型
    if isinstance(dtype, NumpyEADtype):
        # 避免最终结果是NumpyExtensionArray类型
        dtype = dtype.numpy_dtype

    # 推断对象是否为对象类型，而不是ABCIndex或ABCSeries的实例
    infer_object = not isinstance(data, (ABCIndex, ABCSeries))

    # 提取ndarray或ExtensionArray，并确保没有NumpyExtensionArray
    data = extract_array(data, extract_numpy=True, extract_range=True)

    # 如果数据是0维的ndarray
    if isinstance(data, np.ndarray) and data.ndim == 0:
        # 如果dtype未指定，则使用数据的数据类型
        if dtype is None:
            dtype = data.dtype
        # 从0维数据中提取项目
        data = lib.item_from_zerodim(data)
    elif isinstance(data, range):
        # 如果数据是range类型，则转换为ndarray
        data = range_to_ndarray(data)
        # 设置copy为False
        copy = False

    # 如果数据不是类列表对象
    if not is_list_like(data):
        # 如果index未指定，则抛出异常
        if index is None:
            raise ValueError("index must be specified when data is not list-like")
        # 如果数据是字符串类型，并且使用了pyarrow的字符串数据类型，并且未指定原始dtype
        if (
            isinstance(data, str)
            and using_pyarrow_string_dtype()
            and original_dtype is None
        ):
            from pandas.core.arrays.string_ import StringDtype

            # 设置dtype为pyarrow_numpy类型的字符串
            dtype = StringDtype("pyarrow_numpy")
        # 根据标量数据构建1维数组
        data = construct_1d_arraylike_from_scalar(data, len(index), dtype)

        # 返回处理后的数据
        return data

    # 如果数据是ExtensionArray类型
    elif isinstance(data, ABCExtensionArray):
        # 此处已确保不是NumpyExtensionArray类型
        # 在修复GH#49309之前，此检查需要在ExtensionDtype检查之前进行
        if dtype is not None:
            # 如果dtype不为空，则转换为指定的dtype类型，并按需复制
            subarr = data.astype(dtype, copy=copy)
        elif copy:
            # 如果copy为True，则复制数据
            subarr = data.copy()
        else:
            # 否则直接使用原始数据
            subarr = data

    # 如果dtype是ExtensionDtype类型
    elif isinstance(dtype, ExtensionDtype):
        # 根据dtype创建一个扩展数组
        _sanitize_non_ordered(data)
        cls = dtype.construct_array_type()
        subarr = cls._from_sequence(data, dtype=dtype, copy=copy)

    # GH#846
    elif isinstance(data, np.ndarray):
        # 如果 data 是 numpy 数组类型
        if isinstance(data, np.matrix):
            # 如果 data 是 numpy 矩阵，转换为普通数组
            data = data.A

        if dtype is None:
            # 如果未指定 dtype
            subarr = data
            if data.dtype == object and infer_object:
                # 如果数据类型是 object 并且需要推断对象类型
                subarr = maybe_infer_to_datetimelike(data)
            elif data.dtype.kind == "U" and using_pyarrow_string_dtype():
                # 如果数据类型是 Unicode 字符串并且使用 PyArrow 字符串类型
                from pandas.core.arrays.string_ import StringDtype

                # 创建 PyArrow 字符串类型的数组
                dtype = StringDtype(storage="pyarrow_numpy")
                subarr = dtype.construct_array_type()._from_sequence(data, dtype=dtype)

            if subarr is data and copy:
                # 如果 subarr 与 data 是同一个对象且需要复制
                subarr = subarr.copy()

        else:
            # 否则尝试按照定义进行类型转换
            subarr = _try_cast(data, dtype, copy)

    elif hasattr(data, "__array__"):
        # 如果 data 具有 __array__ 属性，例如 dask 数组
        # 如果不需要复制，则转换为 numpy 数组
        if not copy:
            data = np.asarray(data)
        else:
            data = np.array(data, copy=copy)
        return sanitize_array(
            data,
            index=index,
            dtype=dtype,
            copy=False,
            allow_2d=allow_2d,
        )

    else:
        # 否则对非有序数据进行处理
        _sanitize_non_ordered(data)
        # 实例化生成器，转换元组和 abc.ValueView
        data = list(data)

        if len(data) == 0 and dtype is None:
            # 如果数据为空且未指定 dtype，默认使用 float64 类型
            subarr = np.array([], dtype=np.float64)

        elif dtype is not None:
            # 否则按照指定的 dtype 进行尝试转换
            subarr = _try_cast(data, dtype, copy)

        else:
            # 否则尝试根据平台进行转换
            subarr = maybe_convert_platform(data)
            if subarr.dtype == object:
                # 如果数据类型是 object 类型
                subarr = cast(np.ndarray, subarr)
                subarr = maybe_infer_to_datetimelike(subarr)

    subarr = _sanitize_ndim(subarr, data, dtype, index, allow_2d=allow_2d)

    if isinstance(subarr, np.ndarray):
        # 确保此时 dtype 为 None 或 subarr.dtype == dtype
        dtype = cast(np.dtype, dtype)
        subarr = _sanitize_str_dtypes(subarr, data, dtype, copy)

    return subarr
# 将 range 对象转换为 numpy 数组
def range_to_ndarray(rng: range) -> np.ndarray:
    """
    Cast a range object to ndarray.
    """
    # GH#30171 perf avoid realizing range as a list in np.array
    try:
        # 尝试使用 int64 类型创建 numpy 数组，处理常规情况
        arr = np.arange(rng.start, rng.stop, rng.step, dtype="int64")
    except OverflowError:
        # GH#30173 处理 int64 范围溢出的情况
        if (rng.start >= 0 and rng.step > 0) or (rng.step < 0 <= rng.stop):
            try:
                # 尝试使用 uint64 类型创建 numpy 数组，处理溢出情况
                arr = np.arange(rng.start, rng.stop, rng.step, dtype="uint64")
            except OverflowError:
                # 如果仍然无法处理，创建一个对象数组来容纳 range 的内容
                arr = construct_1d_object_array_from_listlike(list(rng))
        else:
            # 创建一个对象数组来容纳 range 的内容，如果不满足前述条件
            arr = construct_1d_object_array_from_listlike(list(rng))
    return arr


# 检查数据是否为无序集合，例如 dict_keys，如果是则引发类型错误
def _sanitize_non_ordered(data) -> None:
    """
    Raise only for unordered sets, e.g., not for dict_keys
    """
    if isinstance(data, (set, frozenset)):
        raise TypeError(f"'{type(data).__name__}' type is unordered")


# 确保结果数组是一维数组
def _sanitize_ndim(
    result: ArrayLike,
    data,
    dtype: DtypeObj | None,
    index: Index | None,
    *,
    allow_2d: bool = False,
) -> ArrayLike:
    """
    Ensure we have a 1-dimensional result array.
    """
    if getattr(result, "ndim", 0) == 0:
        # 结果数组应该是多维的，至少有一个维度
        raise ValueError("result should be arraylike with ndim > 0")

    if result.ndim == 1:
        # 我们需要的结果已经是一维数组
        result = _maybe_repeat(result, index)

    elif result.ndim > 1:
        if isinstance(data, np.ndarray):
            if allow_2d:
                # 如果允许二维数组，则直接返回结果
                return result
            # 抛出异常，要求数据必须是一维的，而不是当前的二维数组
            raise ValueError(
                f"Data must be 1-dimensional, got ndarray of shape {data.shape} instead"
            )
        if is_object_dtype(dtype) and isinstance(dtype, ExtensionDtype):
            # 如果数据类型是对象类型且是扩展数据类型，如 NumpyEADtype("O")

            # 使用 asarray_tuplesafe 函数将数据转换为对象数组
            result = com.asarray_tuplesafe(data, dtype=np.dtype("object"))
            # 根据 dtype 构建适当的数组类型
            cls = dtype.construct_array_type()
            result = cls._from_sequence(result, dtype=dtype)
        else:
            # 如果不符合上述条件，则使用 asarray_tuplesafe 函数转换数据类型
            # 错误: "asarray_tuplesafe" 的 "dtype" 参数具有不兼容的类型
            # "Union[dtype[Any], ExtensionDtype, None]"; 期望 "Union[str, dtype[Any], None]"
            result = com.asarray_tuplesafe(data, dtype=dtype)  # type: ignore[arg-type]
    return result


# 确保结果数组的数据类型受 pandas 支持
def _sanitize_str_dtypes(
    result: np.ndarray, data, dtype: np.dtype | None, copy: bool
) -> np.ndarray:
    """
    Ensure we have a dtype that is supported by pandas.
    """

    # 防止混合类型的 Series 被全部强制转换为 NumPy 字符串类型，例如 NaN --> '-1#IND'.
    # 检查 result.dtype.type 是否是 str 类型的子类
    if issubclass(result.dtype.type, str):
        # 标记：GH#16605
        # 如果数据不为空，则将数据转换为指定的 dtype 类型
        # GH#19853: 如果数据是标量，result 已经是最终结果
        if not lib.is_scalar(data):
            # 如果数据中不全是缺失值，则将数据转换为指定的 dtype 类型的 NumPy 数组
            if not np.all(isna(data)):
                data = np.asarray(data, dtype=dtype)
            # 如果不复制数据，则将数据转换为 dtype 为 object 类型的 NumPy 数组
            if not copy:
                result = np.asarray(data, dtype=object)
            else:
                # 否则，将数据转换为 dtype 为 object 类型的 NumPy 数组并复制
                result = np.array(data, dtype=object, copy=copy)
    # 返回处理后的结果对象
    return result
# 如果数组长度为1且指定了预期长度索引，重复数组以匹配索引的长度
def _maybe_repeat(arr: ArrayLike, index: Index | None) -> ArrayLike:
    if index is not None:
        # 检查数组长度是否为1，且索引长度不同，则将数组重复以匹配索引长度
        if 1 == len(arr) != len(index):
            arr = arr.repeat(len(index))
    return arr


# 将输入转换为 numpy ndarray，并可选地转换为指定的 dtype
def _try_cast(
    arr: list | np.ndarray,
    dtype: np.dtype,
    copy: bool,
) -> ArrayLike:
    is_ndarray = isinstance(arr, np.ndarray)

    # 如果 dtype 是 object
    if dtype == object:
        if not is_ndarray:
            # 从类列表对象构造一个一维对象数组
            subarr = construct_1d_object_array_from_listlike(arr)
            return subarr
        # 确保如果是日期时间对象，则包装它并转换为指定的 dtype
        return ensure_wrapped_if_datetimelike(arr).astype(dtype, copy=copy)

    # 如果 dtype 的类型是 Unicode 字符串
    elif dtype.kind == "U":
        # TODO: 需要使用 arr.dtype.kind 在 "mM" 中进行测试用例
        if is_ndarray:
            # 如果是 ndarray，则转换为 ndarray，并处理多维数组为一维
            arr = cast(np.ndarray, arr)
            shape = arr.shape
            if arr.ndim > 1:
                arr = arr.ravel()
        else:
            shape = (len(arr),)
        # 确保是字符串数组，并根据 shape 进行 reshape 处理
        return lib.ensure_string_array(arr, convert_na_value=False, copy=copy).reshape(
            shape
        )

    # 如果 dtype 的类型是日期时间对象
    elif dtype.kind in "mM":
        return maybe_cast_to_datetime(arr, dtype)

    # 如果 dtype 的类型是整数或无符号整数
    elif dtype.kind in "iu":
        # 如果不需要拷贝数据，尝试将数据转换为指定的整数数组类型
        subarr = maybe_cast_to_integer_array(arr, dtype)
    elif not copy:
        # 如果不需要拷贝数据，直接转换为指定的 dtype 的 ndarray
        subarr = np.asarray(arr, dtype=dtype)
    else:
        # 否则，将数据拷贝为指定的 dtype 的 ndarray
        subarr = np.array(arr, dtype=dtype, copy=copy)

    return subarr
```