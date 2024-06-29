# `D:\src\scipysrc\pandas\pandas\core\dtypes\cast.py`

```
"""
Routines for casting.
"""

from __future__ import annotations  # 允许在类型注解中使用类型自身

import datetime as dt  # 导入datetime模块并使用别名dt
import functools  # 导入functools模块
from typing import (  # 导入多个类型提示，包括TYPE_CHECKING, Any, Literal, TypeVar, cast, overload
    TYPE_CHECKING,
    Any,
    Literal,
    TypeVar,
    cast,
    overload,
)
import warnings  # 导入warnings模块

import numpy as np  # 导入numpy并使用别名np

from pandas._config import using_pyarrow_string_dtype  # 从pandas._config中导入using_pyarrow_string_dtype

from pandas._libs import (  # 从pandas._libs中导入多个模块：Interval, Period, lib
    Interval,
    Period,
    lib,
)
from pandas._libs.missing import (  # 从pandas._libs.missing中导入多个对象：NA, NAType, checknull
    NA,
    NAType,
    checknull,
)
from pandas._libs.tslibs import (  # 从pandas._libs.tslibs中导入多个对象：NaT, OutOfBoundsDatetime, OutOfBoundsTimedelta, Timedelta, Timestamp, is_supported_dtype
    NaT,
    OutOfBoundsDatetime,
    OutOfBoundsTimedelta,
    Timedelta,
    Timestamp,
    is_supported_dtype,
)
from pandas._libs.tslibs.timedeltas import array_to_timedelta64  # 从pandas._libs.tslibs.timedeltas中导入array_to_timedelta64

from pandas.errors import (  # 从pandas.errors中导入多个异常类：IntCastingNaNError, LossySetitemError
    IntCastingNaNError,
    LossySetitemError,
)

from pandas.core.dtypes.common import (  # 从pandas.core.dtypes.common中导入多个函数和判断方法
    ensure_int8,
    ensure_int16,
    ensure_int32,
    ensure_int64,
    ensure_object,
    ensure_str,
    is_bool,
    is_complex,
    is_float,
    is_integer,
    is_object_dtype,
    is_scalar,
    is_string_dtype,
    pandas_dtype as pandas_dtype_func,
)
from pandas.core.dtypes.dtypes import (  # 从pandas.core.dtypes.dtypes中导入多个dtype类
    ArrowDtype,
    BaseMaskedDtype,
    CategoricalDtype,
    DatetimeTZDtype,
    ExtensionDtype,
    IntervalDtype,
    PandasExtensionDtype,
    PeriodDtype,
)
from pandas.core.dtypes.generic import (  # 从pandas.core.dtypes.generic中导入多个泛型类
    ABCExtensionArray,
    ABCIndex,
    ABCSeries,
)
from pandas.core.dtypes.inference import is_list_like  # 从pandas.core.dtypes.inference中导入is_list_like函数
from pandas.core.dtypes.missing import (  # 从pandas.core.dtypes.missing中导入多个缺失值处理相关函数
    is_valid_na_for_dtype,
    isna,
    na_value_for_dtype,
    notna,
)

from pandas.io._util import _arrow_dtype_mapping  # 从pandas.io._util中导入_arrow_dtype_mapping

if TYPE_CHECKING:
    from collections.abc import (  # 如果在类型检查模式下，从collections.abc中导入Sequence, Sized
        Sequence,
        Sized,
    )

    from pandas._typing import (  # 从pandas._typing中导入多个类型定义
        ArrayLike,
        Dtype,
        DtypeObj,
        NumpyIndexT,
        Scalar,
        npt,
    )

    from pandas import Index  # 从pandas中导入Index类
    from pandas.core.arrays import (  # 从pandas.core.arrays中导入多个数组类
        Categorical,
        DatetimeArray,
        ExtensionArray,
        IntervalArray,
        PeriodArray,
        TimedeltaArray,
    )


_int8_max = np.iinfo(np.int8).max  # 设置_int8_max为np.int8的最大值
_int16_max = np.iinfo(np.int16).max  # 设置_int16_max为np.int16的最大值
_int32_max = np.iinfo(np.int32).max  # 设置_int32_max为np.int32的最大值

_dtype_obj = np.dtype(object)  # 设置_dtype_obj为np.object的dtype

NumpyArrayT = TypeVar("NumpyArrayT", bound=np.ndarray)  # 创建一个类型变量NumpyArrayT，限制为np.ndarray类型


def maybe_convert_platform(  # 定义函数maybe_convert_platform，参数为values: list | tuple | range | np.ndarray | ExtensionArray，返回类型为ArrayLike
    values: list | tuple | range | np.ndarray | ExtensionArray,
) -> ArrayLike:
    """try to do platform conversion, allow ndarray or list here"""
    arr: ArrayLike  # 声明变量arr为ArrayLike类型

    if isinstance(values, (list, tuple, range)):  # 如果values是list、tuple或range类型的实例
        arr = construct_1d_object_array_from_listlike(values)  # 调用construct_1d_object_array_from_listlike函数，将values转换为一维对象数组
    else:
        # The caller is responsible for ensuring that we have np.ndarray
        #  or ExtensionArray here.
        arr = values  # 否则，arr直接赋值为values，调用者需确保values为np.ndarray或ExtensionArray类型

    if arr.dtype == _dtype_obj:  # 如果arr的dtype为_dtype_obj
        arr = cast(np.ndarray, arr)  # 将arr强制转换为np.ndarray类型
        arr = lib.maybe_convert_objects(arr)  # 调用lib.maybe_convert_objects函数，尝试转换arr中的对象类型数据

    return arr  # 返回arr


def is_nested_object(obj) -> bool:
    """
    return a boolean if we have a nested object, e.g. a Series with 1 or
    more Series elements

    This may not be necessarily be performant.

    """
    # 返回一个布尔值，指示对象obj是否为嵌套对象，例如包含一个或多个Series元素的Series对象
    # 这可能不一定是高效的。
    # 返回一个布尔值，检查对象是否为 ABCSeries 的实例，并且其数据类型是对象类型，并且其数值中至少有一个也是 ABCSeries 的实例。
    return bool(
        isinstance(obj, ABCSeries)  # 检查对象是否为 ABCSeries 的实例
        and is_object_dtype(obj.dtype)  # 检查对象的数据类型是否为对象类型
        and any(isinstance(v, ABCSeries) for v in obj._values)  # 检查对象的数值中是否至少有一个是 ABCSeries 的实例
    )
# 将标量转换为 Timestamp 或 Timedelta，如果标量类似于日期时间且 dtype 不是对象类型
def maybe_box_datetimelike(value: Scalar, dtype: Dtype | None = None) -> Scalar:
    if dtype == _dtype_obj:
        pass  # 如果 dtype 是对象类型，不做任何转换
    elif isinstance(value, (np.datetime64, dt.datetime)):
        value = Timestamp(value)  # 如果 value 是日期时间类型，转换为 Timestamp
    elif isinstance(value, (np.timedelta64, dt.timedelta)):
        value = Timedelta(value)  # 如果 value 是时间间隔类型，转换为 Timedelta
    return value


# 如果传入标量，则将其转换为 Python 原生类型
def maybe_box_native(value: Scalar | None | NAType) -> Scalar | None | NAType:
    if is_float(value):
        value = float(value)  # 如果是浮点数，转换为 float 类型
    elif is_integer(value):
        value = int(value)  # 如果是整数，转换为 int 类型
    elif is_bool(value):
        value = bool(value)  # 如果是布尔值，转换为 bool 类型
    elif isinstance(value, (np.datetime64, np.timedelta64)):
        value = maybe_box_datetimelike(value)  # 如果是日期时间或时间间隔类型，调用 maybe_box_datetimelike 处理
    elif value is NA:
        value = None  # 如果是 NA（缺失值），转换为 None
    return value


# 将 Timedelta 或 Timestamp 转换为 timedelta64 或 datetime64，以设置到 numpy 数组中
def _maybe_unbox_datetimelike(value: Scalar, dtype: DtypeObj) -> Scalar:
    if is_valid_na_for_dtype(value, dtype):
        # 如果 value 是 NaN 且 dtype 允许 NaT，创建 NaT 值
        value = dtype.type("NaT", "ns")
    elif isinstance(value, Timestamp):
        if value.tz is None:
            value = value.to_datetime64()  # 如果是无时区的 Timestamp，转换为 datetime64
        elif not isinstance(dtype, DatetimeTZDtype):
            raise TypeError("Cannot unbox tzaware Timestamp to tznaive dtype")
    elif isinstance(value, Timedelta):
        value = value.to_timedelta64()  # 如果是 Timedelta，转换为 timedelta64

    _disallow_mismatched_datetimelike(value, dtype)  # 检查是否与 dtype 匹配
    return value


# 检查是否禁止不匹配的日期时间类型
def _disallow_mismatched_datetimelike(value, dtype: DtypeObj) -> None:
    vdtype = getattr(value, "dtype", None)
    if vdtype is None:
        return
    elif (vdtype.kind == "m" and dtype.kind == "M") or (
        vdtype.kind == "M" and dtype.kind == "m"
    ):
        raise TypeError(f"Cannot cast {value!r} to {dtype}")


# 用于类型重载的函数签名，用于可能的类型向下转换
@overload
def maybe_downcast_to_dtype(
    result: np.ndarray, dtype: str | np.dtype
) -> np.ndarray: ...


@overload
def maybe_downcast_to_dtype(
    result: ExtensionArray, dtype: str | np.dtype
) -> ArrayLike: ...


# 可能将结果向下转换为指定的 dtype 类型
def maybe_downcast_to_dtype(result: ArrayLike, dtype: str | np.dtype) -> ArrayLike:
    """
    # 尝试将结果转换为指定的数据类型（例如将其转换回布尔型或整型，或者进行从 float64 到 float32 的类型转换）
    """
    如果 result 是 Pandas 的 Series 类型，则将其转换为其底层的 numpy 数组
    if isinstance(result, ABCSeries):
        result = result._values
    设定是否需要对结果进行四舍五入
    do_round = False

    如果 dtype 是字符串类型
    if isinstance(dtype, str):
        如果 dtype 是 "infer"，则根据 result 推断其数据类型
        if dtype == "infer":
            inferred_type = lib.infer_dtype(result, skipna=False)
            根据推断的数据类型设置 dtype
            if inferred_type == "boolean":
                dtype = "bool"
            elif inferred_type == "integer":
                dtype = "int64"
            elif inferred_type == "datetime64":
                dtype = "datetime64[ns]"
            elif inferred_type in ["timedelta", "timedelta64"]:
                dtype = "timedelta64[ns]"
            尝试在此处执行向上转型
            elif inferred_type == "floating":
                dtype = "int64"
                如果 result 的 dtype 类型是 np.number 的子类，则设置 do_round 为 True，表示需要进行四舍五入操作
                    do_round = True
            否则，将 dtype 设置为 "object"
            else:
                dtype = "object"

        将 dtype 转换为 numpy 的数据类型对象
        dtype = np.dtype(dtype)

    如果 dtype 不是 np.dtype 的实例，则抛出类型错误异常
    if not isinstance(dtype, np.dtype):
        强制执行函数签名的注释，抛出类型错误异常
        raise TypeError(dtype)  # pragma: no cover

    调用 maybe_downcast_numeric 函数尝试将 result 转换为指定的 dtype 类型并进行可能的数值降级操作
    converted = maybe_downcast_numeric(result, dtype, do_round)
    如果 converted 不等于 result，则返回转换后的结果
    if converted is not result:
        return converted

    如果 dtype 的 kind 属于 "mM"，并且 result 的 dtype 的 kind 属于 "if"，表示 dtype 是日期时间相关的数据类型
    # GH12821, iNaT is cast to float
    if dtype.kind in "mM" and result.dtype.kind in "if":
        将 result 转换为指定的 dtype 类型
        result = result.astype(dtype)

    如果 dtype 的 kind 为 "m"，并且 result 的 dtype 为 _dtype_obj
    elif dtype.kind == "m" and result.dtype == _dtype_obj:
        将 result 强制转换为 numpy 的 ndarray 类型
        result = cast(np.ndarray, result)
        将 array_to_timedelta64 函数应用于 result 并将结果赋给 result
        result = array_to_timedelta64(result)

    如果 dtype 为 np.dtype("M8[ns]")，并且 result 的 dtype 为 _dtype_obj
    elif dtype == np.dtype("M8[ns]") and result.dtype == _dtype_obj:
        将 result 强制转换为 numpy 的 ndarray 类型
        result = cast(np.ndarray, result)
        调用 maybe_cast_to_datetime 函数将 result 转换为 datetime 类型，并返回转换后的结果
        return np.asarray(maybe_cast_to_datetime(result, dtype=dtype))

    返回处理后的 result 结果
    return result
@overload
def maybe_downcast_numeric(
    result: np.ndarray, dtype: np.dtype, do_round: bool = False
) -> np.ndarray:
    """
    Overloaded function signature for maybe_downcast_numeric with numpy ndarray.

    Parameters
    ----------
    result : np.ndarray
        Input array that may undergo dtype conversion.
    dtype : np.dtype
        Target dtype to which `result` may be converted.
    do_round : bool, optional
        Flag indicating whether rounding should be applied, default is False.

    Returns
    -------
    np.ndarray
        Converted array `result` with dtype `dtype`.
    """

@overload
def maybe_downcast_numeric(
    result: ExtensionArray, dtype: DtypeObj, do_round: bool = False
) -> ArrayLike:
    """
    Overloaded function signature for maybe_downcast_numeric with ExtensionArray.

    Parameters
    ----------
    result : ExtensionArray
        Input array-like object that may undergo dtype conversion.
    dtype : DtypeObj
        Target dtype to which `result` may be converted.
    do_round : bool, optional
        Flag indicating whether rounding should be applied, default is False.

    Returns
    -------
    ArrayLike
        Converted array-like object `result` with dtype `dtype`.
    """

def maybe_downcast_numeric(
    result: ArrayLike, dtype: DtypeObj, do_round: bool = False
) -> ArrayLike:
    """
    Subset of maybe_downcast_to_dtype restricted to numeric dtypes.

    Parameters
    ----------
    result : ArrayLike
        Input array or array-like object that may undergo dtype conversion.
    dtype : DtypeObj
        Target dtype to which `result` may be converted.
    do_round : bool, optional
        Flag indicating whether rounding should be applied, default is False.

    Returns
    -------
    ArrayLike
        Converted array or array-like object `result` with dtype `dtype`.
    """

    if not isinstance(dtype, np.dtype) or not isinstance(result.dtype, np.dtype):
        # e.g. SparseDtype has no itemsize attr
        return result

    def trans(x):
        if do_round:
            return x.round()
        return x

    if dtype.kind == result.dtype.kind:
        # don't allow upcasts here (except if empty)
        if result.dtype.itemsize <= dtype.itemsize and result.size:
            return result

    if dtype.kind in "biu":
        if not result.size:
            # if we don't have any elements, just astype it
            return trans(result).astype(dtype)

        if isinstance(result, np.ndarray):
            element = result.item(0)
        else:
            element = result.iloc[0]
        if not isinstance(element, (np.integer, np.floating, int, float, bool)):
            # a comparable, e.g. a Decimal may slip in here
            return result

        if (
            issubclass(result.dtype.type, (np.object_, np.number))
            and notna(result).all()
        ):
            new_result = trans(result).astype(dtype)
            if new_result.dtype.kind == "O" or result.dtype.kind == "O":
                # np.allclose may raise TypeError on object-dtype
                if (new_result == result).all():
                    return new_result
            else:
                if np.allclose(new_result, result, rtol=0):
                    return new_result

    elif (
        issubclass(dtype.type, np.floating)
        and result.dtype.kind != "b"
        and not is_string_dtype(result.dtype)
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "overflow encountered in cast", RuntimeWarning
            )
            new_result = result.astype(dtype)

        # Adjust tolerances based on floating point size
        size_tols = {4: 5e-4, 8: 5e-8, 16: 5e-16}

        atol = size_tols.get(new_result.dtype.itemsize, 0.0)

        # Check downcast float values are still equal within 7 digits when
        # converting from float64 to float32
        if np.allclose(new_result, result, equal_nan=True, rtol=0.0, atol=atol):
            return new_result
    # 如果 dtype 和 result 的数据类型都是复数类型 "c"
    elif dtype.kind == result.dtype.kind == "c":
        # 将 result 转换为指定的数据类型 dtype
        new_result = result.astype(dtype)

        # 检查转换后的 new_result 是否与原始 result 相等，包括 NaN 值，如果相等则返回 new_result
        if np.array_equal(new_result, result, equal_nan=True):
            # TODO: 使用类似浮点数的容差值来进行比较？
            return new_result

    # 返回原始的 result，如果不符合上述条件（dtype 和 result 不都是 "c" 类型）
    return result
# 如果数组的数据类型位数低于64位整数或浮点数，将其提升为64位
def maybe_upcast_numeric_to_64bit(arr: NumpyIndexT) -> NumpyIndexT:
    dtype = arr.dtype  # 获取数组的数据类型
    if dtype.kind == "i" and dtype != np.int64:  # 如果是整数且不是64位整数
        return arr.astype(np.int64)  # 转换为64位整数类型
    elif dtype.kind == "u" and dtype != np.uint64:  # 如果是无符号整数且不是64位无符号整数
        return arr.astype(np.uint64)  # 转换为64位无符号整数类型
    elif dtype.kind == "f" and dtype != np.float64:  # 如果是浮点数且不是64位浮点数
        return arr.astype(np.float64)  # 转换为64位浮点数类型
    else:
        return arr  # 数据类型已经是64位，返回原数组


# 尝试将点操作的结果转换回原始数据类型（如果适用）
def maybe_cast_pointwise_result(
    result: ArrayLike,
    dtype: DtypeObj,
    numeric_only: bool = False,
    same_dtype: bool = True,
) -> ArrayLike:
    if isinstance(dtype, ExtensionDtype):  # 如果数据类型是扩展数据类型
        cls = dtype.construct_array_type()  # 构造数据类型对应的数组类型
        if same_dtype:
            result = _maybe_cast_to_extension_array(cls, result, dtype=dtype)  # 尝试转换为扩展数组类型
        else:
            result = _maybe_cast_to_extension_array(cls, result)  # 尝试转换为扩展数组类型（不指定数据类型）

    elif (numeric_only and dtype.kind in "iufcb") or not numeric_only:  # 如果要求只转换数值或包括日期时间等非数值类型
        result = maybe_downcast_to_dtype(result, dtype)  # 尝试将结果按指定数据类型转换

    return result  # 返回转换后的结果


# 尝试将对象转换为扩展数组类型，如果失败则返回原对象
def _maybe_cast_to_extension_array(
    cls: type[ExtensionArray], obj: ArrayLike, dtype: ExtensionDtype | None = None
) -> ArrayLike:
    result: ArrayLike

    if dtype is not None:  # 如果指定了数据类型
        try:
            result = cls._from_scalars(obj, dtype=dtype)  # 尝试从标量创建扩展数组对象
        except (TypeError, ValueError):
            return obj  # 发生异常时返回原对象
        return result  # 返回创建的扩展数组对象

    try:
        result = cls._from_sequence(obj, dtype=dtype)  # 尝试从序列创建扩展数组对象
    except Exception:
        # 无法预测下游扩展数组构造函数可能引发的异常
        result = obj  # 发生异常时返回原对象
    return result  # 返回创建的扩展数组对象或原对象


# 确保数据类型能够容纳缺失值（NA）
def ensure_dtype_can_hold_na(dtype: DtypeObj) -> DtypeObj:
    """
    If we have a dtype that cannot hold NA values, find the best match that can.
    """
    # 检查 dtype 是否为 ExtensionDtype 的实例
    if isinstance(dtype, ExtensionDtype):
        # 如果 dtype 能够容纳 NA（缺失值）
        if dtype._can_hold_na:
            # 返回当前的 dtype 对象
            return dtype
        # 如果 dtype 是 IntervalDtype 的实例
        elif isinstance(dtype, IntervalDtype):
            # TODO(GH#45349): 不要特别处理 IntervalDtype，允许
            #  覆盖而不是返回下面的对象。
            # 返回一个 IntervalDtype 对象，使用 np.float64 类型，指定闭合状态
            return IntervalDtype(np.float64, closed=dtype.closed)
        # 返回默认的 dtype 对象
        return _dtype_obj
    # 如果 dtype 的种类是 "b"（布尔类型）
    elif dtype.kind == "b":
        # 返回默认的 dtype 对象
        return _dtype_obj
    # 如果 dtype 的种类在 "iu" 中（整数或无符号整数类型）
    elif dtype.kind in "iu":
        # 返回一个新的 np.float64 类型的 dtype 对象
        return np.dtype(np.float64)
    # 返回原始的 dtype 对象（其他情况）
    return dtype
# 定义规范的 NaN 值字典，用于不同类型的 NaN/NaT 值的规范表示
_canonical_nans = {
    np.datetime64: np.datetime64("NaT", "ns"),  # datetime64 类型的 NaN 表示为 NaT（Not a Time），单位为纳秒
    np.timedelta64: np.timedelta64("NaT", "ns"),  # timedelta64 类型的 NaN 表示为 NaT，单位为纳秒
    type(np.nan): np.nan,  # np.nan 对应的规范表示仍为 np.nan
}

def maybe_promote(dtype: np.dtype, fill_value=np.nan):
    """
    查找可以容纳给定 dtype 和 fill_value 的最小 dtype。

    Parameters
    ----------
    dtype : np.dtype
        数据类型
    fill_value : scalar, default np.nan
        填充值，默认为 np.nan

    Returns
    -------
    dtype
        必要时从 dtype 参数提升后的数据类型。
    fill_value
        必要时从 fill_value 参数提升后的填充值。

    Raises
    ------
    ValueError
        如果 fill_value 是非标量且 dtype 不是 object 类型。
    """
    orig = fill_value
    orig_is_nat = False
    if checknull(fill_value):
        # https://github.com/pandas-dev/pandas/pull/39692#issuecomment-1441051740
        # 避免 NaN/NaT 值（非单例）导致的缓存未命中
        if fill_value is not NA:
            try:
                orig_is_nat = np.isnat(fill_value)
            except TypeError:
                pass

        # 根据 fill_value 的类型从 _canonical_nans 中获取其规范表示
        fill_value = _canonical_nans.get(type(fill_value), fill_value)

    # 为了性能，使用 _maybe_promote 的缓存版本的实际实现
    # 然而，这并不总是有效（对于非可哈希参数），如果需要，我们会回退到实际实现
    try:
        # 错误："_lru_cache_wrapper" 的 "__call__" 的第 3 个参数具有不兼容的类型 "Type[Any]"；期望 "Hashable" [arg-type]
        # 尝试使用缓存的 _maybe_promote_cached 版本（_lru_cache_wrapper 的包装）
        dtype, fill_value = _maybe_promote_cached(
            dtype,
            fill_value,
            type(fill_value),  # type: ignore[arg-type]
        )
    except TypeError:
        # 如果 fill_value 不可哈希（缓存需要）
        dtype, fill_value = _maybe_promote(dtype, fill_value)

    # 如果 dtype 是 _dtype_obj 并且 orig 不为空，或者 orig 是 NaT 并且 orig 的时间数据单位不是 "ns"
    # GH#51592,53497 修复我们潜在的非规范 fill_value
    if (dtype == _dtype_obj and orig is not None) or (
        orig_is_nat and np.datetime_data(orig)[0] != "ns"
    ):
        fill_value = orig  # 恢复原始的可能非规范的 fill_value
    return dtype, fill_value


@functools.lru_cache
def _maybe_promote_cached(dtype, fill_value, fill_value_type):
    """
    _maybe_promote 的缓存版本实现。

    Parameters
    ----------
    dtype
        数据类型
    fill_value
        填充值
    fill_value_type
        填充值的类型

    Returns
    -------
    dtype
        从 _maybe_promote 返回的 dtype
    fill_value
        从 _maybe_promote 返回的 fill_value
    """
    return _maybe_promote(dtype, fill_value)


def _maybe_promote(dtype: np.dtype, fill_value=np.nan):
    """
    函数的实际实现，供 maybe_promote 使用缓存版本。

    Parameters
    ----------
    dtype : np.dtype
        数据类型
    fill_value : scalar, default np.nan
        填充值，默认为 np.nan

    Returns
    -------
    dtype
        从 maybe_promote 返回的 dtype
    fill_value
        从 maybe_promote 返回的 fill_value
    """
    if not is_scalar(fill_value):
        # 对于 object dtype，没有什么可以提升的，用户可以传递几乎任何奇怪的 fill_value
        if dtype != object:
            raise ValueError("fill_value must be a scalar")  # 填充值必须是标量
        dtype = _dtype_obj  # 将 dtype 设置为对象类型
        return dtype, fill_value
    # 检查填充值是否适用于指定的数据类型，且数据类型是数值型
    if is_valid_na_for_dtype(fill_value, dtype) and dtype.kind in "iufcmM":
        # 确保数据类型能够包含缺失值
        dtype = ensure_dtype_can_hold_na(dtype)
        # 获取适用于数据类型的缺失值
        fv = na_value_for_dtype(dtype)
        return dtype, fv

    elif isinstance(dtype, CategoricalDtype):
        # 如果数据类型是分类数据类型
        if fill_value in dtype.categories or isna(fill_value):
            return dtype, fill_value
        else:
            # 确保填充值为对象类型
            return object, ensure_object(fill_value)

    elif isna(fill_value):
        # 如果填充值是缺失值
        dtype = _dtype_obj
        if fill_value is None:
            # 但保留例如 pd.NA
            fill_value = np.nan
        return dtype, fill_value

    # 返回 (数据类型, 填充值) 的元组
    if issubclass(dtype.type, np.datetime64):
        # 从标量推断数据类型和填充值
        inferred, fv = infer_dtype_from_scalar(fill_value)
        if inferred == dtype:
            return dtype, fv

        from pandas.core.arrays import DatetimeArray

        # 创建一个空的 DatetimeArray 对象
        dta = DatetimeArray._from_sequence([], dtype="M8[ns]")
        try:
            # 验证并设置填充值
            fv = dta._validate_setitem_value(fill_value)
            return dta.dtype, fv
        except (ValueError, TypeError):
            return _dtype_obj, fill_value

    elif issubclass(dtype.type, np.timedelta64):
        # 从标量推断数据类型和填充值
        inferred, fv = infer_dtype_from_scalar(fill_value)
        if inferred == dtype:
            return dtype, fv

        elif inferred.kind == "m":
            # 不同的单位，例如传递了 np.timedelta64(24, "h")，但数据类型是 m8[ns]
            # 看看能否无损地将其转换为我们的数据类型
            unit = np.datetime_data(dtype)[0]
            try:
                # 尝试转换为指定单位的 Timedelta 对象
                td = Timedelta(fill_value).as_unit(unit, round_ok=False)
            except OutOfBoundsTimedelta:
                return _dtype_obj, fill_value
            else:
                return dtype, td.asm8

        return _dtype_obj, fill_value

    elif is_float(fill_value):
        if issubclass(dtype.type, np.bool_):
            # 如果数据类型是布尔型，转换为对象型
            dtype = np.dtype(np.object_)

        elif issubclass(dtype.type, np.integer):
            # 如果数据类型是整数型，转换为 float64
            dtype = np.dtype(np.float64)

        elif dtype.kind == "f":
            # 查找最小标量类型，确保它比当前数据类型更小
            mst = np.min_scalar_type(fill_value)
            if mst > dtype:
                # 例如 mst 是 np.float64 而 dtype 是 np.float32
                dtype = mst

        elif dtype.kind == "c":
            # 查找最小标量类型，提升数据类型与 mst 的类型
            mst = np.min_scalar_type(fill_value)
            dtype = np.promote_types(dtype, mst)

    elif is_bool(fill_value):
        if not issubclass(dtype.type, np.bool_):
            # 如果数据类型不是布尔型，转换为对象型
            dtype = np.dtype(np.object_)

    elif is_integer(fill_value):
        if issubclass(dtype.type, np.bool_):
            # 如果数据类型是布尔型，转换为对象型
            dtype = np.dtype(np.object_)

        elif issubclass(dtype.type, np.integer):
            if not np_can_cast_scalar(fill_value, dtype):  # type: ignore[arg-type]
                # 避免溢出，提升数据类型
                mst = np.min_scalar_type(fill_value)
                dtype = np.promote_types(dtype, mst)
                if dtype.kind == "f":
                    # 特殊情况，与 numpy 不一致的地方
                    dtype = np.dtype(np.object_)
    # 如果填充值是复杂类型
    elif is_complex(fill_value):
        # 如果 dtype 是 np.bool_ 的子类
        if issubclass(dtype.type, np.bool_):
            # 将 dtype 转换为 np.object_
            dtype = np.dtype(np.object_)

        # 如果 dtype 是 np.integer 或者 np.floating 的子类
        elif issubclass(dtype.type, (np.integer, np.floating)):
            # 计算填充值的最小标量类型
            mst = np.min_scalar_type(fill_value)
            # 提升 dtype 的类型以匹配最小标量类型
            dtype = np.promote_types(dtype, mst)

        # 如果 dtype 的类型是复数 (complex)
        elif dtype.kind == "c":
            # 计算填充值的最小标量类型
            mst = np.min_scalar_type(fill_value)
            # 如果最小标量类型大于当前 dtype
            if mst > dtype:
                # 例如，mst 是 np.complex128 而 dtype 是 np.complex64
                # 将 dtype 更新为最小标量类型 mst

    else:
        # 如果以上条件都不满足，默认将 dtype 设置为 np.object_
        dtype = np.dtype(np.object_)

    # 如果 dtype 的类型是 bytes 或者 str 的子类
    # 将 dtype 更新为 np.object_
    if issubclass(dtype.type, (bytes, str)):
        dtype = np.dtype(np.object_)

    # 确保 fill_value 的类型与 dtype 匹配
    fill_value = _ensure_dtype_type(fill_value, dtype)
    # 返回更新后的 dtype 和 fill_value
    return dtype, fill_value
def _ensure_dtype_type(value, dtype: np.dtype):
    """
    Ensure that the given value is an instance of the given dtype.

    e.g. if out dtype is np.complex64_, we should have an instance of that
    as opposed to a python complex object.

    Parameters
    ----------
    value : object
        The value to ensure type compatibility.
    dtype : np.dtype
        The NumPy data type to ensure the value conforms to.

    Returns
    -------
    object
        The value cast to the specified dtype.
    """
    # Start with exceptions in which we do _not_ cast to numpy types

    if dtype == _dtype_obj:
        # If the dtype matches _dtype_obj, return the value as is
        return value

    # Note: before we get here we have already excluded isna(value)
    # Cast the value to the specified dtype
    return dtype.type(value)


def infer_dtype_from(val) -> tuple[DtypeObj, Any]:
    """
    Interpret the dtype from a scalar or array.

    Parameters
    ----------
    val : object
        The input value to infer the dtype from.
    """
    if not is_list_like(val):
        # If val is not list-like, infer the dtype from scalar
        return infer_dtype_from_scalar(val)
    # Otherwise, infer the dtype from array
    return infer_dtype_from_array(val)


def infer_dtype_from_scalar(val) -> tuple[DtypeObj, Any]:
    """
    Interpret the dtype from a scalar.

    Parameters
    ----------
    val : object
        The scalar input value to infer the dtype from.
    """
    dtype: DtypeObj = _dtype_obj

    # a 1-element ndarray
    if isinstance(val, np.ndarray):
        # If val is a numpy ndarray
        if val.ndim != 0:
            msg = "invalid ndarray passed to infer_dtype_from_scalar"
            raise ValueError(msg)

        dtype = val.dtype
        val = lib.item_from_zerodim(val)

    elif isinstance(val, str):
        # If val is a string
        # Handle special cases for string dtype inference
        dtype = _dtype_obj
        if using_pyarrow_string_dtype():
            from pandas.core.arrays.string_ import StringDtype

            dtype = StringDtype(storage="pyarrow_numpy")

    elif isinstance(val, (np.datetime64, dt.datetime)):
        # If val is datetime-like
        try:
            val = Timestamp(val)
        except OutOfBoundsDatetime:
            return _dtype_obj, val

        if val is NaT or val.tz is None:
            val = val.to_datetime64()
            dtype = val.dtype
            # TODO: test with datetime(2920, 10, 1) based on test_replace_dtypes
        else:
            dtype = DatetimeTZDtype(unit=val.unit, tz=val.tz)

    elif isinstance(val, (np.timedelta64, dt.timedelta)):
        # If val is timedelta-like
        try:
            val = Timedelta(val)
        except (OutOfBoundsTimedelta, OverflowError):
            dtype = _dtype_obj
        else:
            if val is NaT:
                val = np.timedelta64("NaT", "ns")
            else:
                val = val.asm8
            dtype = val.dtype

    elif is_bool(val):
        # If val is boolean
        dtype = np.dtype(np.bool_)

    elif is_integer(val):
        # If val is integer-like
        if isinstance(val, np.integer):
            dtype = np.dtype(type(val))
        else:
            dtype = np.dtype(np.int64)

        try:
            np.array(val, dtype=dtype)
        except OverflowError:
            dtype = np.array(val).dtype
    # 如果值是浮点数
    elif is_float(val):
        # 如果值是 NumPy 的浮点数类型
        if isinstance(val, np.floating):
            # 使用值的类型来确定 NumPy 的数据类型
            dtype = np.dtype(type(val))
        else:
            # 否则，默认使用 np.float64 数据类型
            dtype = np.dtype(np.float64)

    # 如果值是复数
    elif is_complex(val):
        # 使用 np.complex128 数据类型
        dtype = np.dtype(np.complex128)

    # 如果值是 Period 对象
    if isinstance(val, Period):
        # 使用 PeriodDtype，指定频率为 val.freq
        dtype = PeriodDtype(freq=val.freq)
    # 如果值是 Interval 对象
    elif isinstance(val, Interval):
        # 推断左端点的数据类型
        subtype = infer_dtype_from_scalar(val.left)[0]
        # 使用 IntervalDtype，指定子类型和闭区间属性
        dtype = IntervalDtype(subtype=subtype, closed=val.closed)

    # 返回推断出的数据类型和原始值
    return dtype, val
# 将 datetimelike 键的字典转换为以 Timestamp 键的字典
def dict_compat(d: dict[Scalar, Scalar]) -> dict[Scalar, Scalar]:
    return {maybe_box_datetimelike(key): value for key, value in d.items()}


# 从数组推断 dtype
def infer_dtype_from_array(arr) -> tuple[DtypeObj, ArrayLike]:
    if isinstance(arr, np.ndarray):  # 检查是否为 NumPy 数组
        return arr.dtype, arr

    if not is_list_like(arr):  # 检查是否为类列表对象
        raise TypeError("'arr' must be list-like")

    arr_dtype = getattr(arr, "dtype", None)  # 获取数组的 dtype 属性
    if isinstance(arr_dtype, ExtensionDtype):  # 检查是否为扩展 dtype
        return arr.dtype, arr
    elif isinstance(arr, ABCSeries):  # 检查是否为 pandas 的 Series 类型
        return arr.dtype, np.asarray(arr)

    # 根据数组推断 dtype 类型，不强制使用 numpy 的 coerce with nan's
    inferred = lib.infer_dtype(arr, skipna=False)
    if inferred in ["string", "bytes", "mixed", "mixed-integer"]:  # 检查推断出的类型是否为特定字符串类型
        return (np.dtype(np.object_), arr)

    arr = np.asarray(arr)  # 将输入转换为 NumPy 数组
    return arr.dtype, arr


# 尝试推断对象的 dtype，用于算术操作
def _maybe_infer_dtype_type(element):
    tipo = None
    if hasattr(element, "dtype"):  # 检查对象是否有 dtype 属性
        tipo = element.dtype
    elif is_list_like(element):  # 检查对象是否实现了迭代器协议
        element = np.asarray(element)  # 将对象转换为 NumPy 数组
        tipo = element.dtype
    return tipo


# 将 DataFrame.select_dtypes() 中的字符串类型 dtype 更改为 object
def invalidate_string_dtypes(dtype_set: set[DtypeObj]) -> None:
    non_string_dtypes = dtype_set - {
        np.dtype("S").type,  # 排除字符串类型
        np.dtype("<U").type,  # 排除 unicode 字符串类型
    }
    if non_string_dtypes != dtype_set:
        raise TypeError("string dtypes are not allowed, use 'object' instead")


# 强制将 indexer 输入数组转换为可能的最小 dtype
def coerce_indexer_dtype(indexer, categories) -> np.ndarray:
    """coerce the indexer input array to the smallest dtype possible"""
    length = len(categories)
    # 如果长度小于 _int8_max，调用 ensure_int8 函数处理 indexer，并返回处理结果
    if length < _int8_max:
        return ensure_int8(indexer)
    # 如果长度小于 _int16_max 但大于等于 _int8_max，调用 ensure_int16 函数处理 indexer，并返回处理结果
    elif length < _int16_max:
        return ensure_int16(indexer)
    # 如果长度小于 _int32_max 但大于等于 _int16_max，调用 ensure_int32 函数处理 indexer，并返回处理结果
    elif length < _int32_max:
        return ensure_int32(indexer)
    # 如果长度大于等于 _int32_max，调用 ensure_int64 函数处理 indexer，并返回处理结果
    return ensure_int64(indexer)
def convert_dtypes(
    input_array: ArrayLike,
    convert_string: bool = True,
    convert_integer: bool = True,
    convert_boolean: bool = True,
    convert_floating: bool = True,
    infer_objects: bool = False,
    dtype_backend: Literal["numpy_nullable", "pyarrow"] = "numpy_nullable",
) -> DtypeObj:
    """
    Convert objects to best possible type, and optionally,
    to types supporting ``pd.NA``.

    Parameters
    ----------
    input_array : ExtensionArray or np.ndarray
        输入的数组，可以是扩展数组或者NumPy数组。
    convert_string : bool, default True
        是否将对象数据类型转换为 ``StringDtype()``。
    convert_integer : bool, default True
        是否在可能的情况下将数据类型转换为整数扩展类型。
    convert_boolean : bool, defaults True
        是否将对象数据类型转换为 ``BooleanDtypes()``。
    convert_floating : bool, defaults True
        是否在可能的情况下将数据类型转换为浮点数扩展类型。
        如果 `convert_integer` 也为 True，则优先考虑整数类型，
        如果浮点数可以准确地转换为整数。
    infer_objects : bool, defaults False
        是否还应尝试将对象推断为浮点数/整数。仅在对象数组包含 pd.NA 时生效。
    dtype_backend : {'numpy_nullable', 'pyarrow'}, default 'numpy_nullable'
        应用于结果DataFrame的后端数据类型（仍处于实验阶段）。

        * ``"numpy_nullable"``: 返回支持空值数据类型的 :class:`DataFrame`
          （默认）。
        * ``"pyarrow"``: 返回基于pyarrow的支持空值的 :class:`ArrowDtype`
          DataFrame。

        .. versionadded:: 2.0

    Returns
    -------
    np.dtype, or ExtensionDtype
        返回推断的数据类型或者扩展数据类型对象。
    """
    inferred_dtype: str | DtypeObj

    # 如果任何一个转换标志为True，则执行相应的推断
    if (
        convert_string or convert_integer or convert_boolean or convert_floating
    ):
        # 执行对象类型推断逻辑
        inferred_dtype = input_array.dtype
    else:
        # 如果所有转换标志都为False，则保持输入数组的数据类型
        inferred_dtype = input_array.dtype
    # 如果数据类型后端为 "pyarrow"
    if dtype_backend == "pyarrow":
        # 导入需要的模块和函数
        from pandas.core.arrays.arrow.array import to_pyarrow_type
        from pandas.core.arrays.string_ import StringDtype
        
        # 断言推断出的数据类型不是字符串
        assert not isinstance(inferred_dtype, str)
        
        # 根据条件转换推断的数据类型到 ArrowDtype
        if (
            (convert_integer and inferred_dtype.kind in "iu")
            or (convert_floating and inferred_dtype.kind in "fc")
            or (convert_boolean and inferred_dtype.kind == "b")
            or (convert_string and isinstance(inferred_dtype, StringDtype))
            or (
                inferred_dtype.kind not in "iufcb"
                and not isinstance(inferred_dtype, StringDtype)
            )
        ):
            # 如果推断的数据类型是 PandasExtensionDtype，并且不是 DatetimeTZDtype
            if isinstance(inferred_dtype, PandasExtensionDtype) and not isinstance(
                inferred_dtype, DatetimeTZDtype
            ):
                # 获取其基本数据类型
                base_dtype = inferred_dtype.base
            # 如果推断的数据类型是 BaseMaskedDtype 或 ArrowDtype
            elif isinstance(inferred_dtype, (BaseMaskedDtype, ArrowDtype)):
                # 获取其对应的 numpy 数据类型
                base_dtype = inferred_dtype.numpy_dtype
            # 如果推断的数据类型是 StringDtype
            elif isinstance(inferred_dtype, StringDtype):
                # 将其基本数据类型设置为字符串类型
                base_dtype = np.dtype(str)
            else:
                # 否则使用推断的数据类型本身
                base_dtype = inferred_dtype
            
            # 如果推断的数据类型的基本类型是对象类型，并且输入数组大小大于 0，并且所有值都是缺失值
            if (
                base_dtype.kind == "O"  # type: ignore[union-attr]
                and input_array.size > 0
                and isna(input_array).all()
            ):
                # 导入 pyarrow 库
                import pyarrow as pa
                # 创建一个 pyarrow 的 null 类型
                pa_type = pa.null()
            else:
                # 否则将基本数据类型转换为 pyarrow 的数据类型
                pa_type = to_pyarrow_type(base_dtype)
            
            # 如果成功转换为 pyarrow 的数据类型，则更新推断的数据类型为 ArrowDtype
            if pa_type is not None:
                inferred_dtype = ArrowDtype(pa_type)
    
    # 如果数据类型后端为 "numpy_nullable" 并且推断的数据类型是 ArrowDtype
    elif dtype_backend == "numpy_nullable" and isinstance(inferred_dtype, ArrowDtype):
        # GH 53648，根据映射将推断的数据类型转换为对应的 ArrowDtype
        inferred_dtype = _arrow_dtype_mapping()[inferred_dtype.pyarrow_dtype]
    
    # 返回推断的数据类型，忽略类型检查的错误信息
    return inferred_dtype  # type: ignore[return-value]
# 尝试推断给定值是否为日期时间类对象，并进行可能的转换
def maybe_infer_to_datetimelike(
    value: npt.NDArray[np.object_],
) -> np.ndarray | DatetimeArray | TimedeltaArray | PeriodArray | IntervalArray:
    """
    我们可能有一个类似日期时间的数组（或单个对象），如果没有传递 dtype，则不改变该值，除非我们找到了日期时间/时间差集

    这里相当严格，要求必须有日期时间/时间差，同时可能包含空值或类似字符串的对象

    Parameters
    ----------
    value : np.ndarray[object]
        接受一个 ndarray[object] 类型的值作为输入

    Returns
    -------
    np.ndarray, DatetimeArray, TimedeltaArray, PeriodArray, or IntervalArray
        返回一个可能转换后的数组，可能是日期时间数组、时间差数组、周期数组或间隔数组的一种

    """
    if not isinstance(value, np.ndarray) or value.dtype != object:
        # 调用者需确保仅传递 ndarray[object] 类型的值
        raise TypeError(type(value))  # pragma: no cover
    if value.ndim != 1:
        # 调用者需确保输入值是一维数组
        raise ValueError(value.ndim)  # pragma: no cover

    if not len(value):
        return value

    # error: Incompatible return value type (got "Union[ExtensionArray,
    # ndarray[Any, Any]]", expected "Union[ndarray[Any, Any], DatetimeArray,
    # TimedeltaArray, PeriodArray, IntervalArray]")
    # 调用 lib.maybe_convert_objects 进行可能的对象转换
    return lib.maybe_convert_objects(  # type: ignore[return-value]
        value,
        # 在此处不转换数值类型的 dtypes，因为如果需要，numpy 会自动处理
        convert_numeric=False,
        convert_non_numeric=True,
        dtype_if_all_nat=np.dtype("M8[s]"),
    )


# 尝试将数组或值转换为日期时间类 dtype，将 float nan 转换为 iNaT
def maybe_cast_to_datetime(
    value: np.ndarray | list, dtype: np.dtype
) -> ExtensionArray | np.ndarray:
    """
    尝试将数组或值转换为日期时间类 dtype，同时将 float nan 转换为 iNaT

    调用者需要处理 ExtensionDtype 和非 dt64/td64 的情况。

    Parameters
    ----------
    value : np.ndarray | list
        输入值可以是 ndarray 或列表形式
    dtype : np.dtype
        目标 dtype，必须是日期时间类 dtype

    Returns
    -------
    ExtensionArray or np.ndarray
        返回一个扩展数组或普通 ndarray，具体类型取决于输入值和 dtype

    """
    from pandas.core.arrays.datetimes import DatetimeArray
    from pandas.core.arrays.timedeltas import TimedeltaArray

    assert dtype.kind in "mM"
    if not is_list_like(value):
        raise TypeError("value must be listlike")

    # TODO: _from_sequence would raise ValueError in cases where
    #  _ensure_nanosecond_dtype raises TypeError
    # 调用 _ensure_nanosecond_dtype 确保 dtype 的纳秒精度
    _ensure_nanosecond_dtype(dtype)

    if lib.is_np_dtype(dtype, "m"):
        # 如果 dtype 是时间差 dtype，则使用 TimedeltaArray._from_sequence 进行转换
        res = TimedeltaArray._from_sequence(value, dtype=dtype)
        return res
    else:
        try:
            # 否则尝试使用 DatetimeArray._from_sequence 进行转换
            dta = DatetimeArray._from_sequence(value, dtype=dtype)
        except ValueError as err:
            # 在出现特定 ValueError 时，提供特定的异常信息
            if "cannot supply both a tz and a timezone-naive dtype" in str(err):
                raise ValueError(
                    "Cannot convert timezone-aware data to "
                    "timezone-naive dtype. Use "
                    "pd.Series(values).dt.tz_localize(None) instead."
                ) from err
            raise

        return dta


# 确保将小于纳秒精度的 dtype 转换为纳秒精度
def _ensure_nanosecond_dtype(dtype: DtypeObj) -> None:
    """
    确保将小于纳秒精度的 dtype 转换为纳秒精度
    """
    # 这里没有返回任何内容，只是确保 dtype 的精度满足纳秒级别
    """  # noqa: E501
    msg = (
        f"The '{dtype.name}' dtype has no unit. "
        f"Please pass in '{dtype.name}[ns]' instead."
    )

    # 获取 dtype 的子类型，例如 SparseDtype
    dtype = getattr(dtype, "subtype", dtype)

    if not isinstance(dtype, np.dtype):
        # 如果 dtype 不是 np.dtype 类型，可能是 datetime64tz 类型
        pass

    elif dtype.kind in "mM":
        if not is_supported_dtype(dtype):
            # 在低于纳秒的分辨率下，以纳秒静默替换，以上纳秒的分辨率则抛出异常
            if dtype.name in ["datetime64", "timedelta64"]:
                raise ValueError(msg)
            # TODO: ValueError 还是 TypeError？现有测试
            #  test_constructor_generic_timestamp_bad_frequency 期望 TypeError
            raise TypeError(
                f"dtype={dtype} is not supported. Supported resolutions are 's', "
                "'ms', 'us', and 'ns'"
            )
# TODO: other value-dependent functions to standardize here include
#  Index._find_common_type_compat
# 定义一个函数，用于确定两个对象操作后的结果类型或数据类型
def find_result_type(left_dtype: DtypeObj, right: Any) -> DtypeObj:
    """
    Find the type/dtype for the result of an operation between objects.

    This is similar to find_common_type, but looks at the right object instead
    of just its dtype. This can be useful in particular when the right
    object does not have a `dtype`.

    Parameters
    ----------
    left_dtype : np.dtype or ExtensionDtype
        左侧对象的数据类型或扩展数据类型
    right : Any
        右侧对象

    Returns
    -------
    np.dtype or ExtensionDtype
        操作结果的数据类型或扩展数据类型

    See also
    --------
    find_common_type
    numpy.result_type
    """
    new_dtype: DtypeObj

    # 如果左侧数据类型是 np.dtype，并且其类型为 'iuc' 中的一种，而右侧对象是整数或浮点数
    if (
        isinstance(left_dtype, np.dtype)
        and left_dtype.kind in "iuc"
        and (lib.is_integer(right) or lib.is_float(right))
    ):
        # 例如，当 left_dtype 是 int8 而 right 是 512 时，我们希望得到 np.int16，而 infer_dtype_from(512) 得到的是 np.int64，
        # 这将导致向上转换得太多。
        if lib.is_float(right) and right.is_integer() and left_dtype.kind != "f":
            right = int(right)
        # 在 NEP 50 后，numpy 不会检查 Python 标量
        # TODO: 是否需要为浮点数重新创建 numpy 的检查逻辑（这会导致某些测试失败）
        if isinstance(right, int) and not isinstance(right, np.integer):
            # 默认情况下，这会给出一个无符号类型（如果我们的数是正数）
            # 如果左侧数据类型是有符号的，我们可能不想要这个，因为这可能会导致多出 1 个数据类型位
            # 我们应该检查相应的 int 数据类型（例如 uint64 对应 int64）
            # 是否能容纳这个数
            right_dtype = np.min_scalar_type(right)
            if right == 0:
                # 特殊情况 0
                right = left_dtype
            elif (
                not np.issubdtype(left_dtype, np.unsignedinteger)
                and 0 < right <= np.iinfo(right_dtype).max
            ):
                # 如果左侧数据类型不是无符号整数，检查它是否适合有符号整数数据类型
                right = np.dtype(f"i{right_dtype.itemsize}")
            else:
                right = right_dtype

        # 计算左右操作结果的数据类型
        new_dtype = np.result_type(left_dtype, right)

    # 如果 right 是 left_dtype 可接受的 NA（Not Available，例如 None 或 np.nan）
    elif is_valid_na_for_dtype(right, left_dtype):
        # 例如，IntervalDtype[int] 和 None/np.nan
        new_dtype = ensure_dtype_can_hold_na(left_dtype)

    else:
        # 推断 right 的数据类型，并找到左侧数据类型和推断出的数据类型之间的通用数据类型
        dtype, _ = infer_dtype_from(right)
        new_dtype = find_common_type([left_dtype, dtype])

    return new_dtype


# 更新 find_common_type 的结果以考虑 Categorical 中的 NA
def common_dtype_categorical_compat(
    objs: Sequence[Index | ArrayLike], dtype: DtypeObj
) -> DtypeObj:
    """
    Update the result of find_common_type to account for NAs in a Categorical.

    Parameters
    ----------
    objs : list[np.ndarray | ExtensionArray | Index]
        对象列表，可以是 np.ndarray、ExtensionArray 或 Index
    dtype : np.dtype or ExtensionDtype
        数据类型或扩展数据类型

    Returns
    -------
    np.dtype or ExtensionDtype
        更新后的数据类型或扩展数据类型
    """
    # GH#38240
    # 用于向 find_common_type 的结果中添加对 Categorical 中 NA 的处理
    # TODO: more generally, could do `not can_hold_na(dtype)`
    # 如果 dtype 类型不是整数或无符号整数类型，可能需要更一般化的处理方式
    if lib.is_np_dtype(dtype, "iu"):
        # 遍历所有的对象
        for obj in objs:
            # 确保不意外地允许例如 "categorical" 字符串类型在这里
            obj_dtype = getattr(obj, "dtype", None)
            if isinstance(obj_dtype, CategoricalDtype):
                if isinstance(obj, ABCIndex):
                    # 可能已经缓存了这个检查结果
                    hasnas = obj.hasnans
                else:
                    # 处理分类数据
                    hasnas = cast("Categorical", obj)._hasna

                if hasnas:
                    # 见测试用例 test_union_int_categorical_with_nan
                    # 如果有缺失值，将 dtype 设置为 np.float64 类型
                    dtype = np.dtype(np.float64)
                    break
    # 返回处理后的 dtype
    return dtype
def np_find_common_type(*dtypes: np.dtype) -> np.dtype:
    """
    np.find_common_type implementation pre-1.25 deprecation using np.result_type
    https://github.com/pandas-dev/pandas/pull/49569#issuecomment-1308300065

    Parameters
    ----------
    dtypes : np.dtypes
        可变参数，用于接收一个或多个 NumPy 数据类型对象

    Returns
    -------
    np.dtype
        返回一个 NumPy 数据类型对象，表示找到的公共数据类型
    """
    try:
        # 调用 np.result_type 来获取输入 dtypes 的公共数据类型
        common_dtype = np.result_type(*dtypes)
        if common_dtype.kind in "mMSU":
            # 当公共数据类型的种类在 'mMSU' 中时，由于 NumPy 1.25 中的升级问题，
            # 降级为对象类型 (object)，之前的行为是返回 find_common_type 的结果
            common_dtype = np.dtype("O")

    except TypeError:
        # 如果类型错误，也返回对象类型 (object)
        common_dtype = np.dtype("O")
    return common_dtype


@overload
def find_common_type(types: list[np.dtype]) -> np.dtype: ...


@overload
def find_common_type(types: list[ExtensionDtype]) -> DtypeObj: ...


@overload
def find_common_type(types: list[DtypeObj]) -> DtypeObj: ...


def find_common_type(types):
    """
    Find a common data type among the given dtypes.

    Parameters
    ----------
    types : list of dtypes
        一个包含数据类型的列表

    Returns
    -------
    pandas extension or numpy dtype
        返回一个 Pandas 扩展类型或 NumPy 数据类型对象

    See Also
    --------
    numpy.find_common_type
        参考 NumPy 的 find_common_type 函数

    """
    if not types:
        raise ValueError("no types given")

    first = types[0]

    # workaround for find_common_type([np.dtype('datetime64[ns]')] * 2)
    # => object
    if lib.dtypes_all_equal(list(types)):
        return first

    # get unique types (dict.fromkeys is used as order-preserving set())
    # 获取唯一的数据类型（使用 dict.fromkeys 作为保持顺序的集合）
    types = list(dict.fromkeys(types).keys())

    if any(isinstance(t, ExtensionDtype) for t in types):
        for t in types:
            if isinstance(t, ExtensionDtype):
                res = t._get_common_dtype(types)
                if res is not None:
                    return res
        return np.dtype("object")

    # take lowest unit
    # 取最低的单位
    if all(lib.is_np_dtype(t, "M") for t in types):
        return np.dtype(max(types))
    if all(lib.is_np_dtype(t, "m") for t in types):
        return np.dtype(max(types))

    # don't mix bool / int or float or complex
    # this is different from numpy, which casts bool with float/int as int
    # 不混合 bool / int 或 float 或 complex
    # 这与 NumPy 不同，NumPy 将 bool 与 float/int 合并为 int
    has_bools = any(t.kind == "b" for t in types)
    if has_bools:
        for t in types:
            if t.kind in "iufc":
                return np.dtype("object")

    return np_find_common_type(*types)


def construct_2d_arraylike_from_scalar(
    value: Scalar, length: int, width: int, dtype: np.dtype, copy: bool
) -> np.ndarray:
    shape = (length, width)

    if dtype.kind in "mM":
        # 如果 dtype 的种类是 'm' 或 'M'，则调用 _maybe_box_and_unbox_datetimelike 处理 value
        value = _maybe_box_and_unbox_datetimelike(value, dtype)
    elif dtype == _dtype_obj:
        if isinstance(value, (np.timedelta64, np.datetime64)):
            # 如果 value 是 np.timedelta64 或 np.datetime64 类型，则创建一个对象数组并填充 value
            # 调用 np.array 会将其转换为 pytimedelta/pydatetime
            out = np.empty(shape, dtype=object)
            out.fill(value)
            return out

    # Attempt to coerce to a numpy array
    # 尝试将 value 强制转换为 NumPy 数组
    # 尝试根据给定的 value 和 dtype 创建一个 NumPy 数组 arr
    try:
        # 如果 copy 参数为假（默认情况），将 value 转换为 NumPy 数组
        if not copy:
            arr = np.asarray(value, dtype=dtype)
        # 如果 copy 参数为真，使用 value 创建一个 NumPy 数组，并指定数据类型和复制选项
        else:
            arr = np.array(value, dtype=dtype, copy=copy)
    # 捕获可能的 ValueError 或 TypeError 异常
    except (ValueError, TypeError) as err:
        # 如果出现异常，抛出新的 TypeError 异常，指明不兼容的数据和数据类型错误
        raise TypeError(
            f"DataFrame constructor called with incompatible data and dtype: {err}"
        ) from err

    # 检查创建的数组 arr 的维度是否为 0，即确保它是一个标量值
    if arr.ndim != 0:
        # 如果数组的维度不为 0，抛出 ValueError 异常，表明 DataFrame 构造函数调用不正确
        raise ValueError("DataFrame constructor not properly called!")

    # 返回一个形状为 shape，并填充为 arr 值的 NumPy 数组
    return np.full(shape, arr)
# 从标量值构建一个类似于 np.ndarray 或 pandas 的数组，指定形状和数据类型，并用值填充
def construct_1d_arraylike_from_scalar(
    value: Scalar, length: int, dtype: DtypeObj | None
) -> ArrayLike:
    """
    创建一个指定形状和数据类型的 np.ndarray / pandas 类型数组，用指定的值填充

    Parameters
    ----------
    value : scalar value
        标量值
    length : int
        数组长度
    dtype : pandas_dtype or np.dtype
        pandas 或 np 的数据类型

    Returns
    -------
    np.ndarray / pandas type of length, filled with value
        长度为 length 的 np.ndarray 或 pandas 类型数组，用 value 填充
    """

    if dtype is None:
        try:
            # 尝试推断出标量值的数据类型
            dtype, value = infer_dtype_from_scalar(value)
        except OutOfBoundsDatetime:
            dtype = _dtype_obj

    if isinstance(dtype, ExtensionDtype):
        # 如果 dtype 是 ExtensionDtype 类型
        cls = dtype.construct_array_type()
        seq = [] if length == 0 else [value]
        return cls._from_sequence(seq, dtype=dtype).repeat(length)

    if length and dtype.kind in "iu" and isna(value):
        # 如果长度大于零且数据类型的种类是整数或无符号整数，并且值是 NaN
        # 强制转换为 float64 类型
        dtype = np.dtype("float64")
    elif lib.is_np_dtype(dtype, "US"):
        # 如果数据类型是 Unicode 字符串
        # 需要转换为对象类型，以避免 numpy 将字符串视为标量值
        dtype = np.dtype("object")
        if not isna(value):
            # 确保值是字符串
            value = ensure_str(value)
    elif dtype.kind in "mM":
        # 如果数据类型是日期或时间类型
        value = _maybe_box_and_unbox_datetimelike(value, dtype)

    # 创建一个长度为 length 的空数组，指定数据类型为 dtype
    subarr = np.empty(length, dtype=dtype)
    if length:
        # 如果长度大于零，用 value 填充数组
        # GH 47391: numpy > 1.24 会在整数数据类型中填充 np.nan 时引发错误
        subarr.fill(value)

    return subarr


def _maybe_box_and_unbox_datetimelike(value: Scalar, dtype: DtypeObj):
    # 调用者负责检查 dtype.kind 是否为 "mM"

    if isinstance(value, dt.datetime):
        # 如果值是 datetime 类型，则封装为适当的日期时间类型
        value = maybe_box_datetimelike(value, dtype)

    return _maybe_unbox_datetimelike(value, dtype)


def construct_1d_object_array_from_listlike(values: Sized) -> np.ndarray:
    """
    将任何类似于列表的对象转换为一个对象类型的一维 numpy 数组。

    Parameters
    ----------
    values : any iterable which has a len()
        任何具有长度的可迭代对象

    Raises
    ------
    TypeError
        * 如果 `values` 没有长度

    Returns
    -------
    1-dimensional numpy array of dtype object
        对象类型的一维 numpy 数组
    """
    # numpy 会尝试将嵌套列表解释为更高维度，因此
    # 制作包含类似于列表的 1D 数组有些棘手：
    result = np.empty(len(values), dtype="object")
    result[:] = values
    return result


def maybe_cast_to_integer_array(arr: list | np.ndarray, dtype: np.dtype) -> np.ndarray:
    """
    接受任何数据类型并返回强制转换版本，在数据与整数/无符号整数数据类型不兼容时引发异常。

    Parameters
    ----------
    arr : np.ndarray or list
        要转换的数组。
    dtype : np.dtype
        要将数组转换为的整数数据类型。

    Returns
    -------
    ndarray
        整数或无符号整数数据类型的数组。

    Raises
    ------
    OverflowError : 数据类型与数据不兼容
    """
    ValueError : loss of precision has occurred during casting

    Examples
    --------
    If you try to coerce negative values to unsigned integers, it raises:

    >>> pd.Series([-1], dtype="uint64")
    Traceback (most recent call last):
        ...
    OverflowError: Trying to coerce negative values to unsigned integers

    Also, if you try to coerce float values to integers, it raises:

    >>> maybe_cast_to_integer_array([1, 2, 3.5], dtype=np.dtype("int64"))
    Traceback (most recent call last):
        ...
    ValueError: Trying to coerce float values to integers
    """
    # 断言数据类型的种类是整数或者无符号整数
    assert dtype.kind in "iu"

    try:
        if not isinstance(arr, np.ndarray):
            with warnings.catch_warnings():
                # 忽略警告：NumPy 将不再允许将超出范围的 Python 整数转换
                # 这里我们已经禁止了带负数的 uint 类型（test_constructor_coercion_signed_to_unsigned），因此可以安全地忽略此警告。
                warnings.filterwarnings(
                    "ignore",
                    "NumPy will stop allowing conversion of " "out-of-bound Python int",
                    DeprecationWarning,
                )
                # 将 arr 转换为指定的 dtype 类型的 NumPy 数组
                casted = np.asarray(arr, dtype=dtype)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                # 将 arr 转换为指定的 dtype 类型，copy=False 表示不复制数据
                casted = arr.astype(dtype, copy=False)
    except OverflowError as err:
        # 如果出现 OverflowError，则抛出自定义的错误信息
        raise OverflowError(
            "The elements provided in the data cannot all be "
            f"casted to the dtype {dtype}"
        ) from err

    if isinstance(arr, np.ndarray) and arr.dtype == dtype:
        # 如果 arr 已经是指定的 dtype 类型，则直接返回 casted
        # 避免昂贵的 array_equal 检查
        return casted

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings(
            "ignore", "elementwise comparison failed", FutureWarning
        )
        # 如果 arr 和 casted 在数值上完全相等，则返回 casted
        if np.array_equal(arr, casted):
            return casted

    # 为了进行正确的数据和 dtype 检查，我们进行此转换。
    #
    # 之前没有做这个是因为 NumPy 对 `uint64` 处理不正确。
    arr = np.asarray(arr)

    if np.issubdtype(arr.dtype, str):
        # TODO(numpy-2.0 min): 这种情况会在上面的 OverflowError 中抛出
        if (casted.astype(str) == arr).all():
            return casted
        raise ValueError(f"string values cannot be losslessly cast to {dtype}")

    if dtype.kind == "u" and (arr < 0).any():
        # TODO: 在 NumPy 2.0 之后还可能会出现此情况吗？
        raise OverflowError("Trying to coerce negative values to unsigned integers")

    if arr.dtype.kind == "f":
        if not np.isfinite(arr).all():
            raise IntCastingNaNError(
                "Cannot convert non-finite values (NA or inf) to integer"
            )
        raise ValueError("Trying to coerce float values to integers")
    if arr.dtype == object:
        raise ValueError("Trying to coerce object values to integers")
    # 检查强制转换后的数据类型是否小于原始数组的数据类型
    if casted.dtype < arr.dtype:
        # TODO: 在当前的 NumPy 版本（大于2）中，此分支还可能执行吗？
        # GH#41734 例如 [1, 200, 923442] 并且 dtype="int8" -> 会导致溢出
        raise ValueError(
            f"Values are too large to be losslessly converted to {dtype}. "
            f"To cast anyway, use pd.Series(values).astype({dtype})"
        )

    # 检查数组的数据类型是否属于日期时间类型或时间间隔类型
    if arr.dtype.kind in "mM":
        # 如果尝试从日期时间类型或时间间隔类型的值构造 Series 或 DataFrame，会抛出 TypeError
        raise TypeError(
            f"Constructing a Series or DataFrame from {arr.dtype} values and "
            f"dtype={dtype} is not supported. Use values.view({dtype}) instead."
        )

    # 如果以上两个条件都不满足，抛出 ValueError 表示值无法无损转换为指定的 dtype 类型
    # 虽然当前不知道会出现这种情况，但显式地抛出异常以覆盖所有情况
    raise ValueError(f"values cannot be losslessly cast to {dtype}")
# 检查数组的数据类型
dtype = arr.dtype

# 如果数据类型不是 np.dtype 或者数据类型的种类包含在 "mM" 中
if not isinstance(dtype, np.dtype) or dtype.kind in "mM":
    # 如果数据类型是 PeriodDtype、IntervalDtype、DatetimeTZDtype 或者 np.dtype 的实例
    if isinstance(dtype, (PeriodDtype, IntervalDtype, DatetimeTZDtype, np.dtype)):
        # 尝试将 arr 强制转换为 PeriodArray、DatetimeArray、TimedeltaArray 或 IntervalArray 类型
        arr = cast(
            "PeriodArray | DatetimeArray | TimedeltaArray | IntervalArray", arr
        )
        try:
            # 调用 arr 对象的 _validate_setitem_value 方法，验证 element 是否合法
            arr._validate_setitem_value(element)
            return True  # 如果验证通过，返回 True
        except (ValueError, TypeError):
            return False  # 如果验证失败，返回 False

    # 对于 ExtensionBlock._can_hold_element 方法的保持行为，返回 True
    return True

# 如果数据类型是 np.dtype 类型，调用 np_can_hold_element 函数验证 element 是否可以被无损地存储在数组中
try:
    np_can_hold_element(dtype, element)
    return True  # 如果验证通过，返回 True
except (TypeError, LossySetitemError):
    return False  # 如果验证失败，返回 False
    # 检查 dtype 的种类是否为 'f'，即浮点数
    if dtype.kind == "f":
        # 检查 element 是否为整数或浮点数
        if lib.is_integer(element) or lib.is_float(element):
            # 将 element 强制转换为 dtype 对应的类型
            casted = dtype.type(element)
            # 如果转换后的值为 NaN 或与原始值相等，则返回转换后的值
            if np.isnan(casted) or casted == element:
                return casted
            # 否则抛出 LossySetitemError，例如可能发生溢出情况，见 TestCoercionFloat32
            raise LossySetitemError

        # 如果 tipo 不为 None
        if tipo is not None:
            # TODO: 是否需要进行 itemsize 的检查？
            # 如果 tipo 的种类不是 'i', 'u', 'f' 中的一种，则无法容纳，抛出 LossySetitemError
            if tipo.kind not in "iuf":
                raise LossySetitemError
            # 如果 tipo 不是 np.dtype 类型
            if not isinstance(tipo, np.dtype):
                # 如果 element 存在 NA 值，则抛出 LossySetitemError
                if element._hasna:
                    raise LossySetitemError
                # 否则返回 element
                return element
            # 如果 tipo 的 itemsize 大于 dtype 的 itemsize 或者 tipo 的种类与 dtype 的种类不同
            elif tipo.itemsize > dtype.itemsize or tipo.kind != dtype.kind:
                # 如果 element 是 np.ndarray 类型
                if isinstance(element, np.ndarray):
                    # 将 element 转换为 dtype 类型的数组
                    casted = element.astype(dtype)
                    # 如果转换后的数组与原始数组相等（包括 NaN 值），则返回转换后的数组
                    if np.array_equal(casted, element, equal_nan=True):
                        return casted
                    # 否则抛出 LossySetitemError
                    raise LossySetitemError

            # 返回 element
            return element

        # 如果上述条件均不满足，则抛出 LossySetitemError
        raise LossySetitemError

    # 检查 dtype 的种类是否为 'c'，即复数
    if dtype.kind == "c":
        # 检查 element 是否为整数、复数或浮点数
        if lib.is_integer(element) or lib.is_complex(element) or lib.is_float(element):
            # 如果 element 是 NaN，则返回转换为 dtype 类型后的值
            if np.isnan(element):
                return dtype.type(element)

            # 忽略警告
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                casted = dtype.type(element)
            # 如果转换后的值与原始值相等，则返回转换后的值
            if casted == element:
                return casted
            # 否则抛出 LossySetitemError，例如可能发生溢出情况，见 test_32878_complex_itemsize
            raise LossySetitemError

        # 如果 tipo 不为 None
        if tipo is not None:
            # 如果 tipo 的种类为 'i', 'u', 'f', 'c' 中的一种，则返回 element
            if tipo.kind in "iufc":
                return element
            # 否则抛出 LossySetitemError
            raise LossySetitemError
        # 如果上述条件均不满足，则抛出 LossySetitemError
        raise LossySetitemError

    # 检查 dtype 的种类是否为 'b'，即布尔类型
    if dtype.kind == "b":
        # 如果 tipo 不为 None
        if tipo is not None:
            # 如果 tipo 的种类为 'b'，即布尔类型
            if tipo.kind == "b":
                # 如果 tipo 不是 np.dtype 类型，即布尔数组
                if not isinstance(tipo, np.dtype):
                    # 如果 element 存在 NA 值，则抛出 LossySetitemError
                    if element._hasna:
                        raise LossySetitemError
                # 返回 element
                return element
            # 否则抛出 LossySetitemError
            raise LossySetitemError

        # 如果 element 是布尔类型，则返回 element
        if lib.is_bool(element):
            return element
        # 否则抛出 LossySetitemError
        raise LossySetitemError

    # 检查 dtype 的种类是否为 'S'，即字符串类型
    if dtype.kind == "S":
        # TODO: 需要更有针对性的测试，例如 tests.frame.methods.test_replace，参考 phofl 的 PR
        # 如果 tipo 不为 None
        if tipo is not None:
            # 如果 tipo 的种类为 'S' 并且 tipo 的 itemsize 小于等于 dtype 的 itemsize，则返回 element
            if tipo.kind == "S" and tipo.itemsize <= dtype.itemsize:
                return element
            # 否则抛出 LossySetitemError
            raise LossySetitemError

        # 如果 element 是 bytes 类型并且长度小于等于 dtype 的 itemsize，则返回 element
        if isinstance(element, bytes) and len(element) <= dtype.itemsize:
            return element
        # 否则抛出 LossySetitemError
        raise LossySetitemError
    # 如果 dtype 的类型是 "V"，表示是 NumPy 的 np.void 类型，无法保存任何数据
    if dtype.kind == "V":
        # 抛出自定义的异常 LossySetitemError，表示不支持对 np.void 类型进行赋值操作
        raise LossySetitemError

    # 如果上述条件不满足，则抛出 NotImplementedError，表示该功能尚未实现
    raise NotImplementedError(dtype)
# 检查给定的范围是否可以由指定的 NumPy 数据类型 dtype 所表示
def _dtype_can_hold_range(rng: range, dtype: np.dtype) -> bool:
    """
    _maybe_infer_dtype_type infers to int64 (and float64 for very large endpoints),
    but in many cases a range can be held by a smaller integer dtype.
    Check if this is one of those cases.
    """
    # 如果范围 rng 为空，直接返回 True
    if not len(rng):
        return True
    # 检查范围的起始点和终点是否可以被 dtype 所表示
    return np_can_cast_scalar(rng.start, dtype) and np_can_cast_scalar(rng.stop, dtype)


def np_can_cast_scalar(element: Scalar, dtype: np.dtype) -> bool:
    """
    np.can_cast 的等价实现，用于早于 2-0 版本行为的标量推断

    Parameters
    ----------
    element : Scalar
        要检查的标量值
    dtype : np.dtype
        目标数据类型

    Returns
    -------
    bool
        如果可以将标量 element 转换为 dtype，则返回 True，否则返回 False
    """
    try:
        # 调用 np_can_hold_element 函数来检查是否可以转换 element 到 dtype
        np_can_hold_element(dtype, element)
        return True
    except (LossySetitemError, NotImplementedError):
        return False
```