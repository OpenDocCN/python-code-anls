# `D:\src\scipysrc\pandas\pandas\core\tools\datetimes.py`

```
# 从未来版本导入注解功能
from __future__ import annotations

# 导入 collections 模块的抽象基类
from collections import abc
# 导入 date 模块中的 date 类
from datetime import date
# 导入 functools 模块的 partial 函数
from functools import partial
# 导入 itertools 模块的 islice 函数
from itertools import islice
# 导入 typing 模块中的各种类型提示功能
from typing import (
    TYPE_CHECKING,        # 类型检查标志
    TypedDict,            # 声明类型字典的辅助工具
    Union,                # 合并类型的工具
    cast,                 # 类型强制转换工具
    overload,             # 函数重载的装饰器
)
# 导入警告模块
import warnings

# 导入 NumPy 库，并起别名为 np
import numpy as np

# 从 pandas 库的 _libs 子模块中导入 lib 和 tslib 模块
from pandas._libs import (
    lib,    # 提供底层实用程序的模块
    tslib,  # 时间序列基础库的模块
)
# 从 pandas._libs.tslibs 模块中导入各种时间相关功能
from pandas._libs.tslibs import (
    OutOfBoundsDatetime,     # 超出日期时间范围异常
    Timedelta,               # 时间差类型
    Timestamp,               # 时间戳类型
    astype_overflowsafe,     # 安全类型转换函数
    is_supported_dtype,      # 判断是否支持的数据类型函数
    timezones as libtimezones,  # 时间区别作为 libtimezones
)
# 从 pandas._libs.tslibs.conversion 模块中导入 cast_from_unit_vectorized 函数
from pandas._libs.tslibs.conversion import cast_from_unit_vectorized
# 从 pandas._libs.tslibs.dtypes 模块中导入 NpyDatetimeUnit 类型
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
# 从 pandas._libs.tslibs.parsing 模块中导入 DateParseError 和 guess_datetime_format 函数
from pandas._libs.tslibs.parsing import (
    DateParseError,          # 日期解析错误异常
    guess_datetime_format,   # 猜测日期时间格式的函数
)
# 从 pandas._libs.tslibs.strptime 模块中导入 array_strptime 函数
from pandas._libs.tslibs.strptime import array_strptime
# 从 pandas._typing 模块中导入各种类型别名
from pandas._typing import (
    AnyArrayLike,            # 任意类数组别名
    ArrayLike,               # 类数组别名
    DateTimeErrorChoices,    # 日期时间错误选择别名
)
# 从 pandas.util._exceptions 模块中导入 find_stack_level 函数
from pandas.util._exceptions import find_stack_level

# 从 pandas.core.dtypes.common 模块中导入一些常用的数据类型判断函数
from pandas.core.dtypes.common import (
    ensure_object,           # 确保对象函数
    is_float,                # 是否浮点数判断函数
    is_integer,              # 是否整数判断函数
    is_integer_dtype,        # 是否整数类型判断函数
    is_list_like,            # 是否列表样式判断函数
    is_numeric_dtype,        # 是否数值类型判断函数
)
# 从 pandas.core.dtypes.dtypes 模块中导入 ArrowDtype 和 DatetimeTZDtype 类型
from pandas.core.dtypes.dtypes import (
    ArrowDtype,              # Arrow 数据类型
    DatetimeTZDtype,         # 带时区的日期时间数据类型
)
# 从 pandas.core.dtypes.generic 模块中导入 ABCDataFrame 和 ABCSeries 类型
from pandas.core.dtypes.generic import (
    ABCDataFrame,            # 抽象基类 DataFrame
    ABCSeries,               # 抽象基类 Series
)

# 从 pandas.arrays 模块中导入 DatetimeArray、IntegerArray 和 NumpyExtensionArray 类型
from pandas.arrays import (
    DatetimeArray,           # 日期时间数组类型
    IntegerArray,            # 整数数组类型
    NumpyExtensionArray,     # NumPy 扩展数组类型
)
# 从 pandas.core.algorithms 模块中导入 unique 函数
from pandas.core.algorithms import unique
# 从 pandas.core.arrays 模块中导入 ArrowExtensionArray 类型
from pandas.core.arrays import ArrowExtensionArray
# 从 pandas.core.arrays.base 模块中导入 ExtensionArray 类型
from pandas.core.arrays.base import ExtensionArray
# 从 pandas.core.arrays.datetimes 模块中导入若干时间相关函数
from pandas.core.arrays.datetimes import (
    maybe_convert_dtype,     # 可能转换数据类型函数
    objects_to_datetime64,   # 对象转换为 datetime64 类型函数
    tz_to_dtype,             # 时区转换为数据类型函数
)
# 从 pandas.core.construction 模块中导入 extract_array 函数
from pandas.core.construction import extract_array
# 从 pandas.core.indexes.base 模块中导入 Index 类
from pandas.core.indexes.base import Index
# 从 pandas.core.indexes.datetimes 模块中导入 DatetimeIndex 类
from pandas.core.indexes.datetimes import DatetimeIndex

# 如果是类型检查模式，则继续导入以下内容
if TYPE_CHECKING:
    # 从 collections.abc 模块中导入 Callable 和 Hashable 类型
    from collections.abc import (
        Callable,            # 可调用对象类型
        Hashable,            # 可散列对象类型
    )

    # 从 pandas._libs.tslibs.nattype 模块中导入 NaTType 类型
    from pandas._libs.tslibs.nattype import NaTType
    # 从 pandas._libs.tslibs.timedeltas 模块中导入 UnitChoices 类型
    from pandas._libs.tslibs.timedeltas import UnitChoices

    # 从 pandas 模块中导入 DataFrame 和 Series 类型
    from pandas import (
        DataFrame,           # 数据框架类型
        Series,              # 系列类型
    )

# ---------------------------------------------------------------------
# 在注解中使用的类型

# 定义一个数组可转换类型，可以是列表、元组或任何类数组
ArrayConvertible = Union[list, tuple, AnyArrayLike]
# 定义标量类型，可以是浮点数或字符串
Scalar = Union[float, str]
# 定义日期时间标量类型，可以是标量、date 对象或 np.datetime64 对象
DatetimeScalar = Union[Scalar, date, np.datetime64]

# 定义日期时间标量或可转换为数组类型
DatetimeScalarOrArrayConvertible = Union[DatetimeScalar, ArrayConvertible]

# 定义日期时间字典的参数类型，可以是标量列表、标量元组或任何类数组
DatetimeDictArg = Union[list[Scalar], tuple[Scalar, ...], AnyArrayLike]

# 定义年月日字典类型，继承自 TypedDict，必须包含 year/month/day 键值对
class YearMonthDayDict(TypedDict, total=True):
    year: DatetimeDictArg     # 年份键值对
    month: DatetimeDictArg    # 月份键值对
    day: DatetimeDictArg      # 日份键值对

# 定义完整日期时间字典类型，继承自 YearMonthDayDict，可以包含 hour/minute/second 等键值对
class FulldatetimeDict(YearMonthDayDict, total=False):
    hour: DatetimeDictArg     # 小时键值对
    hours: DatetimeDictArg    # 小时（复数）键值对
    minute: DatetimeDictArg   # 分钟键值对
    minutes: DatetimeDictArg  # 分钟（复数）键值对
    second: DatetimeDictArg   # 秒钟键值对
    seconds: DatetimeDictArg  # 秒钟（复数）键值对
    ms: DatetimeDictArg       # 毫秒键值对
    us: DatetimeDictArg       # 微秒键值对
    ns: DatetimeDictArg       # 纳秒键值对

# 定义可转换为字典的类型，可以是 FulldatetimeDict 或 DataFrame 类型
DictConvertible = Union[FulldatetimeDict, "DataFrame"]
# ---------------------------------------------------------------------
# 猜测数组的日期时间格式，用于给定数组中的第一个非NaN元素，如果无法确定则返回None
def _guess_datetime_format_for_array(arr, dayfirst: bool | None = False) -> str | None:
    # 获取第一个非空元素的索引
    if (first_non_null := tslib.first_non_null(arr)) != -1:
        # 如果第一个非NaN元素是字符串
        if type(first_non_nan_element := arr[first_non_null]) is str:  # noqa: E721
            # 使用猜测的日期时间格式函数，尝试推断日期时间格式
            guessed_format = guess_datetime_format(
                first_non_nan_element, dayfirst=dayfirst
            )
            # 如果成功猜测到格式，则返回该格式
            if guessed_format is not None:
                return guessed_format
            # 如果存在多个非空元素，发出警告说明解析可能不一致
            if tslib.first_non_null(arr[first_non_null + 1 :]) != -1:
                warnings.warn(
                    "Could not infer format, so each element will be parsed "
                    "individually, falling back to `dateutil`. To ensure parsing is "
                    "consistent and as-expected, please specify a format.",
                    UserWarning,
                    stacklevel=find_stack_level(),
                )
    # 如果无法确定格式，则返回None
    return None


# 决定是否进行缓存
def should_cache(
    arg: ArrayConvertible, unique_share: float = 0.7, check_count: int | None = None
) -> bool:
    """
    Decides whether to do caching.

    If the percent of unique elements among `check_count` elements less
    than `unique_share * 100` then we can do caching.

    Parameters
    ----------
    arg: listlike, tuple, 1-d array, Series
        The input data to check for caching.
    unique_share: float, default=0.7, optional
        The threshold for uniqueness share.
    check_count: int, optional
        The number of elements to check for uniqueness.

    Returns
    -------
    do_caching: bool
        True if caching should be done, False otherwise.

    Notes
    -----
    By default for a sequence of less than 50 items in size, we don't do
    caching; for the number of elements less than 5000, we take ten percent of
    all elements to check for a uniqueness share; if the sequence size is more
    than 5000, then we check only the first 500 elements.
    All constants were chosen empirically by.
    """
    do_caching = True

    # 默认情况下
    if check_count is None:
        # 如果数据长度小于启用缓存的阈值，则不进行缓存
        if len(arg) <= start_caching_at:
            return False

        # 如果数据长度小于5000，则检查数据的十分之一
        if len(arg) <= 5000:
            check_count = len(arg) // 10
        else:
            # 否则只检查前500个元素
            check_count = 500
    else:
        # 确保check_count在合理范围内
        assert (
            0 <= check_count <= len(arg)
        ), "check_count must be in next bounds: [0; len(arg)]"
        if check_count == 0:
            return False

    # 确保unique_share在合理范围内
    assert 0 < unique_share < 1, "unique_share must be in next bounds: (0; 1)"

    try:
        # 尝试使用集合检查元素的唯一性，如果元素不可哈希则无法缓存
        unique_elements = set(islice(arg, check_count))
    except TypeError:
        return False
    # 如果唯一元素的数量小于检查数量乘以唯一性阈值，则不进行缓存
    if len(unique_elements) > check_count * unique_share:
        do_caching = False
    return do_caching


def _maybe_cache(
    arg: ArrayConvertible,
    format: str | None,
    cache: bool,
    # ultimate的缓存机制在计算机领域是至关重要的
    # 缓存可以显著提高程序的性能，但在实现时需要考虑一致性和有效性
    # 不同数据和应用场景可能需要不同的缓存策略和清理机制
    # 有效的缓存策略是软件设计中的一项关键技术
    convert_listlike: Callable,
def _box_as_indexlike(
    dt_array: ArrayLike, utc: bool = False, name: Hashable | None = None
) -> Index:
    """
    Properly boxes the ndarray of datetimes to DatetimeIndex
    if it is possible or to generic Index instead

    Parameters
    ----------
    dt_array: 1-d array
        Array of datetimes to be wrapped in an Index.
    utc : bool
        Whether to convert/localize timestamps to UTC.
    name : string, default None
        Name for a resulting index

    Returns
    -------
    result : datetime of converted dates
        - DatetimeIndex if convertible to sole datetime64 type
        - general Index otherwise
    """

    # Check if the dtype of dt_array is datetime64
    if lib.is_np_dtype(dt_array.dtype, "M"):
        # Determine timezone based on utc parameter
        tz = "utc" if utc else None
        # Return a DatetimeIndex with dt_array, optionally with a name
        return DatetimeIndex(dt_array, tz=tz, name=name)
    
    # If dt_array is not datetime64, return a generic Index with dt_array
    return Index(dt_array, name=name, dtype=dt_array.dtype)
    # `name`: 一个可哈希的值或者为 None，表示时区的名称或者时区信息
    # `utc`: 一个布尔值，默认为 False，表示是否将时间解释为 UTC 时间
    # `unit`: 一个字符串或者为 None，表示时间单位，如年、月、日等
    # `errors`: 一个 DateTimeErrorChoices 枚举类型，默认为 "raise"，表示遇到错误时的处理策略
    # `dayfirst`: 一个布尔值或者为 None，表示日期是否以天为先的格式解释
    # `yearfirst`: 一个布尔值或者为 None，表示日期是否以年为先的格式解释
    # `exact`: 一个布尔值，默认为 True，表示是否精确解析输入
    """
    Helper function for to_datetime. Performs the conversions of 1D listlike
    of dates

    Parameters
    ----------
    arg : list, tuple, ndarray, Series, Index
        date to be parsed
    name : object
        None or string for the Index name
    utc : bool
        Whether to convert/localize timestamps to UTC.
    unit : str
        None or string of the frequency of the passed data
    errors : str
        error handling behaviors from to_datetime, 'raise', 'coerce'
    dayfirst : bool
        dayfirst parsing behavior from to_datetime
    yearfirst : bool
        yearfirst parsing behavior from to_datetime
    exact : bool, default True
        exact format matching behavior from to_datetime

    Returns
    -------
    Index-like of parsed dates
    """
    
    # 如果参数 arg 是 list, tuple, ndarray, Series, Index 中的一种，则转换为 numpy 数组
    if isinstance(arg, (list, tuple)):
        arg = np.array(arg, dtype="O")
    elif isinstance(arg, NumpyExtensionArray):
        arg = np.array(arg)

    arg_dtype = getattr(arg, "dtype", None)
    
    # 如果 arg 的 dtype 是 DatetimeTZDtype，则根据 utc 参数处理日期并返回相应的日期索引
    if isinstance(arg_dtype, DatetimeTZDtype):
        if not isinstance(arg, (DatetimeArray, DatetimeIndex)):
            return DatetimeIndex(arg, tz="utc" if utc else None, name=name)
        if utc:
            arg = arg.tz_convert(None).tz_localize("utc")
        return arg
    
    # 如果 arg 的 dtype 是 ArrowDtype 且是 Timestamp 类型，则根据 utc 参数处理日期并返回相应的日期索引或数组
    elif isinstance(arg_dtype, ArrowDtype) and arg_dtype.type is Timestamp:
        if utc:
            # pyarrow 使用 UTC，不是小写的 utc
            if isinstance(arg, Index):
                arg_array = cast(ArrowExtensionArray, arg.array)
                if arg_dtype.pyarrow_dtype.tz is not None:
                    arg_array = arg_array._dt_tz_convert("UTC")
                else:
                    arg_array = arg_array._dt_tz_localize("UTC")
                arg = Index(arg_array)
            else:
                # ArrowExtensionArray
                if arg_dtype.pyarrow_dtype.tz is not None:
                    arg = arg._dt_tz_convert("UTC")
                else:
                    arg = arg._dt_tz_localize("UTC")
        return arg
    
    # 如果 arg 的 dtype 是 numpy datetime 类型 'M'，则根据 utc 参数处理日期并返回相应的日期索引或数组
    elif lib.is_np_dtype(arg_dtype, "M"):
        if not is_supported_dtype(arg_dtype):
            # 转换为最接近的支持的分辨率，如 "s"
            arg = astype_overflowsafe(
                np.asarray(arg),
                np.dtype("M8[s]"),
                is_coerce=errors == "coerce",
            )

        if not isinstance(arg, (DatetimeArray, DatetimeIndex)):
            return DatetimeIndex(arg, tz="utc" if utc else None, name=name)
        elif utc:
            return arg.tz_localize("utc")
        
        return arg
    
    # 如果指定了 unit 参数，则根据 unit 和 utc 参数处理日期并返回相应的日期索引或数组
    elif unit is not None:
        if format is not None:
            raise ValueError("cannot specify both format and unit")
        return _to_datetime_with_unit(arg, unit, name, utc, errors)
    # 如果参数 `arg` 的属性 `ndim` 大于 1，则抛出 TypeError 异常
    elif getattr(arg, "ndim", 1) > 1:
        raise TypeError(
            "arg must be a string, datetime, list, tuple, 1-d array, or Series"
        )

    # 如果传入的 `arg` 是 timedelta64 类型，则发出警告；如果是 PeriodDtype 类型，则抛出异常
    # 注意：这一步必须在单位转换之后进行
    try:
        # 尝试将 `arg` 转换为指定的数据类型，不进行复制，获取时区信息
        arg, _ = maybe_convert_dtype(arg, copy=False, tz=libtimezones.maybe_get_tz(tz))
    except TypeError:
        # 如果转换类型时发生 TypeError 异常
        if errors == "coerce":
            # 如果 `errors` 参数为 "coerce"，则创建一个包含 "NaT" 的 datetime64[ns] 数组，长度与 `arg` 相同，并返回 DatetimeIndex 对象
            npvalues = np.array(["NaT"], dtype="datetime64[ns]").repeat(len(arg))
            return DatetimeIndex(npvalues, name=name)
        # 否则，将异常继续向上抛出
        raise

    # 确保 `arg` 对象是一个对象类型
    arg = ensure_object(arg)

    # 如果未指定 `format` 参数，则根据 `arg` 推测日期时间格式
    if format is None:
        format = _guess_datetime_format_for_array(arg, dayfirst=dayfirst)

    # 如果 `format` 已经推测出来，或者用户没有要求混合格式解析，则调用 _array_strptime_with_fallback 函数解析日期时间数组
    if format is not None and format != "mixed":
        return _array_strptime_with_fallback(arg, name, utc, format, exact, errors)

    # 否则，调用 objects_to_datetime64 函数将 `arg` 转换为 datetime64 数组
    result, tz_parsed = objects_to_datetime64(
        arg,
        dayfirst=dayfirst,
        yearfirst=yearfirst,
        utc=utc,
        errors=errors,
        allow_object=True,
    )

    # 如果解析得到的时区信息 `tz_parsed` 不为 None，则说明结果数组已经是 UTC 时间
    if tz_parsed is not None:
        # 可以简化处理，因为 datetime64 numpy 数组已经是 UTC 时间
        out_unit = np.datetime_data(result.dtype)[0]
        # 根据 `tz_parsed` 和输出单位 `out_unit` 获取对应的 dtype，并创建新的 DatetimeArray 对象
        dtype = tz_to_dtype(tz_parsed, out_unit)
        dt64_values = result.view(f"M8[{dtype.unit}]")
        dta = DatetimeArray._simple_new(dt64_values, dtype=dtype)
        # 返回新创建的 DatetimeIndex 对象
        return DatetimeIndex._simple_new(dta, name=name)

    # 否则，将结果 `_box_as_indexlike` 包装成类似索引的对象，考虑 UTC 时间，然后返回
    return _box_as_indexlike(result, utc=utc, name=name)
def _array_strptime_with_fallback(
    arg,
    name,
    utc: bool,
    fmt: str,
    exact: bool,
    errors: str,
) -> Index:
    """
    Call array_strptime, with fallback behavior depending on 'errors'.
    """
    # 调用 array_strptime 函数，根据 'errors' 参数确定回退行为
    result, tz_out = array_strptime(arg, fmt, exact=exact, errors=errors, utc=utc)
    # 如果 tz_out 不为 None，则处理时区信息
    if tz_out is not None:
        # 获取结果的时间单位
        unit = np.datetime_data(result.dtype)[0]
        # 创建带时区信息的日期时间数据类型
        dtype = DatetimeTZDtype(tz=tz_out, unit=unit)
        # 创建 DatetimeArray 对象
        dta = DatetimeArray._simple_new(result, dtype=dtype)
        # 如果 utc 参数为 True，则转换时区为 UTC
        if utc:
            dta = dta.tz_convert("UTC")
        # 返回带索引名的 Index 对象
        return Index(dta, name=name)
    # 如果 tz_out 为 None 且结果的 dtype 不是 object，并且 utc 参数为 True
    elif result.dtype != object and utc:
        # 获取结果的时间单位
        unit = np.datetime_data(result.dtype)[0]
        # 创建带 UTC 时区信息的日期时间索引对象
        res = Index(result, dtype=f"M8[{unit}, UTC]", name=name)
        return res
    # 否则返回不带时区信息的日期时间索引对象
    return Index(result, dtype=result.dtype, name=name)


def _to_datetime_with_unit(arg, unit, name, utc: bool, errors: str) -> Index:
    """
    to_datetime specalized to the case where a 'unit' is passed.
    """
    # 从数组中提取数据，确保返回一个 ndarray
    arg = extract_array(arg, extract_numpy=True)

    # GH#30050 pass an ndarray to tslib.array_to_datetime
    # because it expects an ndarray argument
    # 如果 arg 是 IntegerArray 类型
    if isinstance(arg, IntegerArray):
        # 将 arg 转换为指定单位的 datetime64 类型
        arr = arg.astype(f"datetime64[{unit}]")
        tz_parsed = None
    else:
        arg = np.asarray(arg)

        # 如果 arg 的 dtype.kind 是 'iu'
        if arg.dtype.kind in "iu":
            # 将 arg 转换为指定单位的 datetime64 类型
            arr = arg.astype(f"datetime64[{unit}]", copy=False)
            try:
                # 尝试将结果转换为 M8[ns] 类型，处理溢出安全性
                arr = astype_overflowsafe(arr, np.dtype("M8[ns]"), copy=False)
            except OutOfBoundsDatetime:
                # 如果溢出且 errors 参数为 "raise"，则抛出异常
                if errors == "raise":
                    raise
                # 否则将 arg 转换为 object 类型，并递归调用 _to_datetime_with_unit 函数
                arg = arg.astype(object)
                return _to_datetime_with_unit(arg, unit, name, utc, errors)
            tz_parsed = None

        # 如果 arg 的 dtype.kind 是 'f'
        elif arg.dtype.kind == "f":
            with np.errstate(over="raise"):
                try:
                    # 尝试将 arg 转换为指定单位的 datetime64 类型
                    arr = cast_from_unit_vectorized(arg, unit=unit)
                except OutOfBoundsDatetime as err:
                    # 如果超出范围且 errors 参数不为 "raise"，则返回转换为 object 类型的结果
                    if errors != "raise":
                        return _to_datetime_with_unit(
                            arg.astype(object), unit, name, utc, errors
                        )
                    # 否则抛出超出范围的异常
                    raise OutOfBoundsDatetime(
                        f"cannot convert input with unit '{unit}'"
                    ) from err

            # 将结果视图类型转换为 M8[ns] 类型
            arr = arr.view("M8[ns]")
            tz_parsed = None
        else:
            # 将 arg 转换为 object 类型，并调用 tslib.array_to_datetime 函数处理
            arg = arg.astype(object, copy=False)
            arr, tz_parsed = tslib.array_to_datetime(
                arg,
                utc=utc,
                errors=errors,
                unit_for_numerics=unit,
                creso=NpyDatetimeUnit.NPY_FR_ns.value,
            )

    # 创建带名称的日期时间索引对象
    result = DatetimeIndex(arr, name=name)
    # 如果 result 不是 DatetimeIndex 类型，则直接返回结果
    if not isinstance(result, DatetimeIndex):
        return result

    # GH#23758: We may still need to localize the result with tz
    # GH#25546: Apply tz_parsed first (from arg), then tz (from caller)
    # result will be naive but in UTC
    # 对结果应用 tz_parsed 参数指定的时区，然后再应用调用者提供的 tz 参数指定的时区
    result = result.tz_localize("UTC").tz_convert(tz_parsed)
    
    # 如果设置了 utc 标志
    if utc:
        # 如果结果的时区为 None，则将其本地化为 UTC
        if result.tz is None:
            result = result.tz_localize("utc")
        else:
            # 否则，将结果转换为 UTC 时区
            result = result.tz_convert("utc")
    
    # 返回处理后的结果
    return result
def _adjust_to_origin(arg, origin, unit):
    """
    Helper function for to_datetime.
    Adjust input argument to the specified origin

    Parameters
    ----------
    arg : list, tuple, ndarray, Series, Index
        date to be adjusted
    origin : 'julian' or Timestamp
        origin offset for the arg
    unit : str
        passed unit from to_datetime, must be 'D'

    Returns
    -------
    ndarray or scalar of adjusted date(s)
    """
    # 如果 origin 是 'julian'
    if origin == "julian":
        # 保存原始参数值
        original = arg
        # 获取时间戳零点的儒略日数并进行计算
        j0 = Timestamp(0).to_julian_date()
        # 如果单位不是 'D'，则引发错误
        if unit != "D":
            raise ValueError("unit must be 'D' for origin='julian'")
        try:
            # 计算参数与 j0 的差值
            arg = arg - j0
        except TypeError as err:
            # 如果类型不匹配，引发特定错误
            raise ValueError(
                "incompatible 'arg' type for given 'origin'='julian'"
            ) from err

        # 预先检查是否超出合理范围
        j_max = Timestamp.max.to_julian_date() - j0
        j_min = Timestamp.min.to_julian_date() - j0
        if np.any(arg > j_max) or np.any(arg < j_min):
            raise OutOfBoundsDatetime(
                f"{original} is Out of Bounds for origin='julian'"
            )
    else:
        # 如果 origin 不是 'julian'
        # 确保 arg 是数值类型
        if not (
            (is_integer(arg) or is_float(arg)) or is_numeric_dtype(np.asarray(arg))
        ):
            raise ValueError(
                f"'{arg}' is not compatible with origin='{origin}'; "
                "it must be numeric with a unit specified"
            )

        # 尝试将 origin 转换为 Timestamp
        try:
            offset = Timestamp(origin, unit=unit)
        except OutOfBoundsDatetime as err:
            raise OutOfBoundsDatetime(f"origin {origin} is Out of Bounds") from err
        except ValueError as err:
            raise ValueError(
                f"origin {origin} cannot be converted to a Timestamp"
            ) from err

        # 检查 offset 是否是时区感知的，如果是则引发错误
        if offset.tz is not None:
            raise ValueError(f"origin offset {offset} must be tz-naive")
        # 计算偏移量
        td_offset = offset - Timestamp(0)

        # 将偏移量转换为与参数单位对应的数值
        ioffset = td_offset // Timedelta(1, unit=unit)

        # 如果 arg 是类似列表的对象且不是 Series、Index 或 ndarray，则转换为 ndarray
        if is_list_like(arg) and not isinstance(arg, (ABCSeries, Index, np.ndarray)):
            arg = np.asarray(arg)
        # 对 arg 应用偏移量
        arg = arg + ioffset
    return arg
    origin=...,  # 初始化变量 origin，具体值待指定
    cache: bool = ...,  # 初始化变量 cache，类型为布尔型，具体值待指定
# 定义一个重载函数签名，将输入的参数转换为DatetimeIndex类型的返回值
@overload
def to_datetime(
    arg: list | tuple | Index | ArrayLike,
    errors: DateTimeErrorChoices = ...,
    dayfirst: bool = ...,
    yearfirst: bool = ...,
    utc: bool = ...,
    format: str | None = ...,
    exact: bool = ...,
    unit: str | None = ...,
    origin=...,
    cache: bool = ...,
) -> DatetimeIndex: ...


# 定义to_datetime函数，将输入参数转换为DatetimeIndex、Series、DatetimeScalar、NaTType或None类型的返回值
def to_datetime(
    arg: DatetimeScalarOrArrayConvertible | DictConvertible,
    errors: DateTimeErrorChoices = "raise",
    dayfirst: bool = False,
    yearfirst: bool = False,
    utc: bool = False,
    format: str | None = None,
    exact: bool | lib.NoDefault = lib.no_default,
    unit: str | None = None,
    origin: str = "unix",
    cache: bool = True,
) -> DatetimeIndex | Series | DatetimeScalar | NaTType | None:
    """
    Convert argument to datetime.

    This function converts a scalar, array-like, :class:`Series` or
    :class:`DataFrame`/dict-like to a pandas datetime object.

    Parameters
    ----------
    arg : int, float, str, datetime, list, tuple, 1-d array, Series, DataFrame/dict-like
        The object to convert to a datetime. If a :class:`DataFrame` is provided, the
        method expects minimally the following columns: :const:`"year"`,
        :const:`"month"`, :const:`"day"`. The column "year"
        must be specified in 4-digit format.
    errors : {'raise', 'coerce'}, default 'raise'
        - If :const:`'raise'`, then invalid parsing will raise an exception.
        - If :const:`'coerce'`, then invalid parsing will be set as :const:`NaT`.
    dayfirst : bool, default False
        Specify a date parse order if `arg` is str or is list-like.
        If :const:`True`, parses dates with the day first, e.g. :const:`"10/11/12"`
        is parsed as :const:`2012-11-10`.

        .. warning::

            ``dayfirst=True`` is not strict, but will prefer to parse
            with day first.

    yearfirst : bool, default False
        Specify a date parse order if `arg` is str or is list-like.

        - If :const:`True` parses dates with the year first, e.g.
          :const:`"10/11/12"` is parsed as :const:`2010-11-12`.
        - If both `dayfirst` and `yearfirst` are :const:`True`, `yearfirst` is
          preceded (same as :mod:`dateutil`).

        .. warning::

            ``yearfirst=True`` is not strict, but will prefer to parse
            with year first.
    utc : bool, default False
        Return UTC DatetimeIndex if True (converting any tz-aware
        datetime.datetime objects as well).
    format : str, default None
        The strftime to parse time, eg "%Y-%m-%d %H:%M:%S.%f".
    exact : bool | lib.NoDefault, default lib.no_default
        If True, require an exact format match.
    unit : str, default None
        The unit of the arg (D,s,ms,us,ns) denote the unit, which is an integer
        index in the array and not the data type
    origin : str, default 'unix'
        Define the reference time for the origin.
    cache : bool, default True
        If True, cache results.
    """
    pass
    # utc 参数控制时区相关的解析、本地化和转换行为，默认为 False
    utc : bool, default False
        控制时区相关的解析、本地化和转换行为。

        - 如果为 :const:`True`，函数始终返回一个时区感知的 UTC 本地化的 :class:`Timestamp`、:class:`Series` 或 :class:`DatetimeIndex`。
          为了达到这个效果，时区无关的输入会被 *本地化* 为 UTC，而时区感知的输入会被 *转换* 为 UTC。

        - 如果为 :const:`False`（默认），输入不会被强制转换为 UTC。
          时区无关的输入保持无关，而时区感知的输入保留其时间偏移。存在混合偏移的限制（通常是夏令时），详细信息请参阅 :ref:`示例 <to_datetime_tz_examples>` 部分。

        参见：pandas 关于 `时区转换和本地化 <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#time-zone-handling>`_ 的一般文档。

    # format 参数指定时间解析的 strftime 格式，例如 :const:`"%d/%m/%Y"`
    format : str, default None
        指定时间解析的 strftime 格式，例如 :const:`"%d/%m/%Y"`。参见 `strftime 文档 <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior>`_ 了解更多信息，尽管注意到 :const:`"%f"` 可以解析到纳秒级别。
        还可以使用以下选项：

        - "ISO8601"，用于解析任何 `ISO8601 <https://en.wikipedia.org/wiki/ISO_8601>`_ 时间字符串（格式不一定完全相同）；
        - "mixed"，用于为每个元素单独推断格式。这是有风险的，您可能应该与 `dayfirst` 一起使用。

        .. 注意::

            如果传递了 :class:`DataFrame`，则 `format` 参数不会生效。

    # exact 参数控制如何使用 `format` 参数进行匹配，默认为 True
    exact : bool, default True
        控制如何使用 `format` 参数进行匹配：

        - 如果为 :const:`True`，要求精确匹配 `format`。
        - 如果为 :const:`False`，允许 `format` 在目标字符串中的任何位置匹配。

        不能与 ``format='ISO8601'`` 或 ``format='mixed'`` 同时使用。

    # unit 参数指定时间单位，例如 'ns' 表示纳秒，默认为 'ns'
    unit : str, default 'ns'
        指定时间单位的参数 (D,s,ms,us,ns)，表示整数或浮点数。这将基于起点进行计算。
        例如，如果 ``unit='ms'`` 且 ``origin='unix'``，则会计算到 Unix 纪元开始的毫秒数。
    origin : scalar, default 'unix'
        定义参考日期。数值将被解析为从该参考日期开始的单位数量（由 `unit` 定义）。
        
        - 如果为 :const:`'unix'`（或 POSIX）时间；origin 被设置为 1970-01-01。
        - 如果为 :const:`'julian'`，unit 必须为 :const:`'D'`，origin 被设置为朱利安日历的起始日期。
          朱利安日数 :const:`0` 被分配给从公元前4713年1月1日中午开始的那一天。
        - 如果可转换为时间戳（Timestamp, dt.datetime, np.datetime64 或日期字符串），origin 被设置为由 origin 标识的时间戳。
        - 如果为浮点数或整数，origin 表示相对于 1970-01-01 的差距（以 `unit` 参数确定的单位）。

    cache : bool, default True
        如果为 :const:`True`，使用唯一的转换日期缓存来应用日期时间转换。
        在解析重复的日期字符串时，特别是带有时区偏移的字符串，可能会显著提高速度。
        仅当存在至少50个值时才使用缓存。存在超出范围的值将使缓存无法使用，并可能减慢解析速度。

    Returns
    -------
    datetime
        如果解析成功。
        返回类型取决于输入（括号中的类型对应于在时区解析失败或时间戳超出范围时的回退）：

        - 标量：:class:`Timestamp`（或 :class:`datetime.datetime`）
        - 类数组：:class:`DatetimeIndex`（或包含 :class:`datetime.datetime` 的 :class:`object` dtype 的 :class:`Series`）
        - Series：:class:`datetime64` dtype 的 :class:`Series`（或包含 :class:`datetime.datetime` 的 :class:`object` dtype 的 :class:`Series`）
        - DataFrame：:class:`datetime64` dtype 的 :class:`Series`（或包含 :class:`datetime.datetime` 的 :class:`object` dtype 的 :class:`Series`）

    Raises
    ------
    ParserError
        当从字符串解析日期失败时。
    ValueError
        当发生其他日期时间转换错误时。例如，当 :class:`DataFrame` 中缺少 'year'、'month'、'day' 列之一时，
        或者在包含混合时区时间偏移的类数组中找到带有时区的 :class:`datetime.datetime`，且 ``utc=False`` 时，
        或者在解析带有混合时区的日期时间时，除非指定了 ``utc=True``。如果解析带有混合时区的日期时间，请指定 ``utc=True``。

    See Also
    --------
    DataFrame.astype : 将参数转换为指定的 dtype。
    to_timedelta : 将参数转换为 timedelta。
    convert_dtypes : 转换 dtypes。

    Notes
    -----
    支持多种输入类型，并导致不同的输出类型：
    # scalars 可以是 int、float、str 或者标准库 datetime 模块或 numpy 中的 datetime 对象。在可能时会转换为 Timestamp 类型，否则转换为 datetime.datetime 类型。None/NaN/null 标量会被转换为 NaT。
    # array-like 可以包含 int、float、str 或 datetime 对象。在可能时会转换为 DatetimeIndex 类型，否则转换为具有 object dtype 的 Index，其中包含 datetime.datetime。None/NaN/null 条目在两种情况下都会被转换为 NaT。
    # Series 在可能时会转换为具有 datetime64 dtype 的 Series 类型，否则转换为具有 object dtype 的 Series，其中包含 datetime.datetime。None/NaN/null 条目在两种情况下都会被转换为 NaT。
    # DataFrame/dict-like 会被转换为具有 datetime64 dtype 的 Series 类型。对于每一行，从组合各个 DataFrame 列中创建一个 datetime。列键可以是常见的缩写，如 ['year', 'month', 'day', 'minute', 'second', 'ms', 'us', 'ns']，或者它们的复数形式。
    
    # 下列原因会导致返回 datetime.datetime 对象（可能在 Index 或具有 object dtype 的 Series 中）而不是正确的 pandas 指定类型（Timestamp、DatetimeIndex 或具有 datetime64 dtype 的 Series）：
    
    # - 当任何输入元素在 Timestamp.min 之前或 Timestamp.max 之后时，参见时间戳限制。
    # - 当 utc=False（默认值）且输入是包含混合本地/UTC时间的 array-like 或 Series，或者是包含混合时间偏移的 aware 时间。注意这种情况经常发生在时区具有夏令时政策的情况下。在这种情况下，您可能希望使用 utc=True。
    
    # 示例
    # 处理各种输入格式
    
    # 从 DataFrame 的多列中组装 datetime。列键可以是常见的缩写，如 ['year', 'month', 'day', 'minute', 'second', 'ms', 'us', 'ns']，或者它们的复数形式。
    # df = pd.DataFrame({"year": [2015, 2016], "month": [2, 3], "day": [4, 5]})
    # pd.to_datetime(df)
    # 0   2015-02-04
    # 1   2016-03-05
    # dtype: datetime64[s]
    
    # 使用 Unix epoch 时间
    # pd.to_datetime(1490195805, unit="s")
    # Timestamp('2017-03-22 15:16:45')
    # pd.to_datetime(1490195805433502912, unit="ns")
    # Timestamp('2017-03-22 15:16:45.433502912')
    
    # 警告：对于 float 参数，可能会发生精度舍入。为了避免意外行为，请使用固定宽度的精确类型。
    # 使用非 Unix 时间戳起点
    
    >>> pd.to_datetime([1, 2, 3], unit="D", origin=pd.Timestamp("1960-01-01"))
    DatetimeIndex(['1960-01-02', '1960-01-03', '1960-01-04'],
                  dtype='datetime64[ns]', freq=None)
    
    # **与 strptime 行为的区别**
    
    :const:`"%f"` 将解析直到纳秒。
    
    >>> pd.to_datetime("2018-10-26 12:00:00.0000000011", format="%Y-%m-%d %H:%M:%S.%f")
    Timestamp('2018-10-26 12:00:00.000000001')
    
    # **无法转换的日期/时间**
    
    传递 ``errors='coerce'`` 将会强制将超出边界的日期转换为 :const:`NaT`，
    同时也会将非日期（或无法解析的日期）转换为 :const:`NaT`。
    
    >>> pd.to_datetime("invalid for Ymd", format="%Y%m%d", errors="coerce")
    NaT
    
    .. _to_datetime_tz_examples:
    
    # **时区和时间偏移**
    
    默认行为（``utc=False``）如下：
    
    - 时区无关的输入被转换为时区无关的 :class:`DatetimeIndex`：
    
    >>> pd.to_datetime(["2018-10-26 12:00:00", "2018-10-26 13:00:15"])
    DatetimeIndex(['2018-10-26 12:00:00', '2018-10-26 13:00:15'],
                  dtype='datetime64[s]', freq=None)
    
    - 带有固定时间偏移的时区感知输入被转换为时区感知的 :class:`DatetimeIndex`：
    
    >>> pd.to_datetime(["2018-10-26 12:00 -0500", "2018-10-26 13:00 -0500"])
    DatetimeIndex(['2018-10-26 12:00:00-05:00', '2018-10-26 13:00:00-05:00'],
                  dtype='datetime64[s, UTC-05:00]', freq=None)
    
    - 然而，带有混合时间偏移的时区感知输入（例如来自带有夏令时的时区，如 Europe/Paris）
      **无法成功转换** 为 :class:`DatetimeIndex`。
      解析混合时区的日期时间将引发 ValueError，除非 ``utc=True``：
    
    >>> pd.to_datetime(
    ...     ["2020-10-25 02:00 +0200", "2020-10-25 04:00 +0100"]
    ... )  # doctest: +SKIP
    ValueError: Mixed timezones detected. Pass utc=True in to_datetime
    or tz='UTC' in DatetimeIndex to convert to a common timezone.
    
    - 若要创建带有混合偏移和 ``object`` dtype 的 :class:`Series`，请使用
      :meth:`Series.apply` 和 :func:`datetime.datetime.strptime`：
    
    >>> import datetime as dt
    >>> ser = pd.Series(["2020-10-25 02:00 +0200", "2020-10-25 04:00 +0100"])
    >>> ser.apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M %z"))
    0    2020-10-25 02:00:00+02:00
    1    2020-10-25 04:00:00+01:00
    dtype: object
    
    - 混合有时区感知和无时区感知的输入也将引发 ValueError，
      除非 ``utc=True``：
    
    >>> from datetime import datetime
    >>> pd.to_datetime(
    ...     ["2020-01-01 01:00:00-01:00", datetime(2020, 1, 1, 3, 0)]
    ... )  # doctest: +SKIP
    ValueError: Mixed timezones detected. Pass utc=True in to_datetime
    or tz='UTC' in DatetimeIndex to convert to a common timezone.
    
    |
    # 如果 `exact` 参数不是 lib.no_default 并且 `format` 参数为 "mixed" 或 "ISO8601"，则抛出 ValueError 异常
    if exact is not lib.no_default and format in {"mixed", "ISO8601"}:
        raise ValueError("Cannot use 'exact' when 'format' is 'mixed' or 'ISO8601'")
    # 如果 `arg` 参数为 None，则直接返回 None
    if arg is None:
        return None

    # 如果 `origin` 参数不是 "unix"，则调用 _adjust_to_origin 函数调整 `arg` 参数
    if origin != "unix":
        arg = _adjust_to_origin(arg, origin, unit)

    # 使用 functools.partial 创建一个部分函数 `_convert_listlike_datetimes`
    # 参数包括 `utc`, `unit`, `dayfirst`, `yearfirst`, `errors`, `exact`
    convert_listlike = partial(
        _convert_listlike_datetimes,
        utc=utc,
        unit=unit,
        dayfirst=dayfirst,
        yearfirst=yearfirst,
        errors=errors,
        exact=exact,
    )
    # 定义结果变量 `result`，可能的类型包括 Timestamp、NaTType、Series 或 Index
    result: Timestamp | NaTType | Series | Index

    # 如果 `arg` 是 Timestamp 类型
    if isinstance(arg, Timestamp):
        result = arg
        # 如果 `utc` 参数为 True，则将 `arg` 的时区转换为 UTC
        if utc:
            if arg.tz is not None:
                result = arg.tz_convert("utc")
            else:
                result = arg.tz_localize("utc")
    # 如果 `arg` 是继承自 ABCSeries 的对象
    elif isinstance(arg, ABCSeries):
        # 尝试缓存转换结果并应用到 `arg` 对象上
        cache_array = _maybe_cache(arg, format, cache, convert_listlike)
        if not cache_array.empty:
            result = arg.map(cache_array)
        else:
            # 否则，直接转换 `arg` 的值并创建新的 Series 对象
            values = convert_listlike(arg._values, format)
            result = arg._constructor(values, index=arg.index, name=arg.name)
    # 如果 `arg` 是 ABCDataFrame 或 abc.MutableMapping 类型的对象
    elif isinstance(arg, (ABCDataFrame, abc.MutableMapping)):
        # 从单位映射中根据参数组装结果对象
        result = _assemble_from_unit_mappings(arg, errors, utc)
    # 如果 `arg` 是 Index 类型的对象
    elif isinstance(arg, Index):
        # 尝试缓存转换结果并转换为 `arg` 对象
        cache_array = _maybe_cache(arg, format, cache, convert_listlike)
        if not cache_array.empty:
            result = _convert_and_box_cache(arg, cache_array, name=arg.name)
        else:
            # 否则，根据参数转换 `arg` 的值，并创建相应的结果对象
            result = convert_listlike(arg, format, name=arg.name)
    elif is_list_like(arg):
        try:
            # 尝试将参数 arg 强制转换为合适的类型：list、tuple、ExtensionArray、np.ndarray、Series 或 Index
            argc = cast(
                Union[list, tuple, ExtensionArray, np.ndarray, "Series", Index], arg
            )
            # 调用 _maybe_cache 函数尝试缓存处理后的结果
            cache_array = _maybe_cache(argc, format, cache, convert_listlike)
        except OutOfBoundsDatetime:
            # 如果尝试缓存导致 OutOfBoundsDatetime 异常，则处理如下：
            if errors == "raise":
                raise  # 如果 errors 设置为 "raise"，则重新抛出异常
            # 否则，继续进行，使用一个空的 Series 对象来作为缓存数组
            from pandas import Series

            cache_array = Series([], dtype=object)  # 只是一个空数组
        if not cache_array.empty:
            # 如果缓存数组不为空，则调用 _convert_and_box_cache 处理结果
            result = _convert_and_box_cache(argc, cache_array)
        else:
            # 如果缓存数组为空，则调用 convert_listlike 处理 argc
            result = convert_listlike(argc, format)
    else:
        # 如果参数 arg 不是 list-like 类型，则将其转换为 np.array([arg])，然后处理第一个元素
        result = convert_listlike(np.array([arg]), format)[0]
        # 如果原始参数是布尔类型并且处理后的结果也是 np.bool_ 类型，则将结果转换为标准的布尔值
        if isinstance(arg, bool) and isinstance(result, np.bool_):
            result = bool(result)  # TODO: 避免这种临时解决方案。

    # 返回处理后的结果，类型为 result 的类型，忽略类型检查的警告
    return result  # type: ignore[return-value]
# mappings for assembling units
# 定义一个字典，用于将时间单位映射为统一格式的字符串表示
_unit_map = {
    "year": "year",  # 年
    "years": "year",  # 年（复数）
    "month": "month",  # 月
    "months": "month",  # 月（复数）
    "day": "day",  # 日
    "days": "day",  # 日（复数）
    "hour": "h",  # 小时
    "hours": "h",  # 小时（复数）
    "minute": "m",  # 分钟
    "minutes": "m",  # 分钟（复数）
    "second": "s",  # 秒
    "seconds": "s",  # 秒（复数）
    "ms": "ms",  # 毫秒
    "millisecond": "ms",  # 毫秒
    "milliseconds": "ms",  # 毫秒（复数）
    "us": "us",  # 微秒
    "microsecond": "us",  # 微秒
    "microseconds": "us",  # 微秒（复数）
    "ns": "ns",  # 纳秒
    "nanosecond": "ns",  # 纳秒
    "nanoseconds": "ns",  # 纳秒（复数）
}


def _assemble_from_unit_mappings(
    arg, errors: DateTimeErrorChoices, utc: bool
) -> Series:
    """
    assemble the unit specified fields from the arg (DataFrame)
    Return a Series for actual parsing

    Parameters
    ----------
    arg : DataFrame
        输入的数据框，包含待解析的时间字段
    errors : {'raise', 'coerce'}, default 'raise'
        - 如果为 'raise'，则在解析错误时抛出异常
        - 如果为 'coerce'，则在解析错误时将结果设为 NaT（Not a Time）
    utc : bool
        是否将时间戳转换或本地化为UTC时间

    Returns
    -------
    Series
        返回一个包含实际解析结果的Series对象
    """
    from pandas import (
        DataFrame,
        to_numeric,
        to_timedelta,
        to_datetime,
    )

    arg = DataFrame(arg)  # 将输入参数转换为DataFrame对象
    if not arg.columns.is_unique:
        raise ValueError("cannot assemble with duplicate keys")
        # 如果列名有重复，抛出值错误异常

    # replace passed unit with _unit_map
    # 使用_unit_map替换传入的时间单位字段
    def f(value):
        if value in _unit_map:
            return _unit_map[value]

        # m is case significant
        # m 大小写敏感
        if value.lower() in _unit_map:
            return _unit_map[value.lower()]

        return value

    unit = {k: f(k) for k in arg.keys()}  # 对输入数据框的列名应用f函数，构建单位映射字典unit
    unit_rev = {v: k for k, v in unit.items()}  # 构建反向映射字典unit_rev

    # we require at least Ymd
    # 至少需要包含年、月、日三个时间单位
    required = ["year", "month", "day"]
    req = set(required) - set(unit_rev.keys())  # 检查是否有必需的时间单位缺失
    if len(req):
        _required = ",".join(sorted(req))
        raise ValueError(
            "to assemble mappings requires at least that "
            f"[year, month, day] be specified: [{_required}] is missing"
        )

    # keys we don't recognize
    # 检查是否有未识别的时间单位
    excess = set(unit_rev.keys()) - set(_unit_map.values())
    if len(excess):
        _excess = ",".join(sorted(excess))
        raise ValueError(
            f"extra keys have been passed to the datetime assemblage: [{_excess}]"
        )

    def coerce(values):
        # we allow coercion to if errors allows
        # 允许根据errors参数进行强制类型转换
        values = to_numeric(values, errors=errors)

        # prevent overflow in case of int8 or int16
        # 防止在int8或int16类型情况下的溢出
        if is_integer_dtype(values.dtype):
            values = values.astype("int64")
        return values

    # assemble values into datetime format
    # 将数值组装成日期时间格式
    values = (
        coerce(arg[unit_rev["year"]]) * 10000
        + coerce(arg[unit_rev["month"]]) * 100
        + coerce(arg[unit_rev["day"]])
    )
    try:
        values = to_datetime(values, format="%Y%m%d", errors=errors, utc=utc)
    except (TypeError, ValueError) as err:
        raise ValueError(f"cannot assemble the datetimes: {err}") from err

    units: list[UnitChoices] = ["h", "m", "s", "ms", "us", "ns"]
    # 定义时间单位的列表，包括小时、分钟、秒、毫秒、微秒和纳秒
    # 遍历 units 列表中的每个单位 u
    for u in units:
        # 从 unit_rev 字典中获取与当前单位 u 对应的值 value
        value = unit_rev.get(u)
        # 如果找到了对应值，并且该值存在于 arg 字典中
        if value is not None and value in arg:
            # 尝试将 arg[value] 强制转换为 timedelta 类型，并添加到 values 中
            try:
                values += to_timedelta(coerce(arg[value]), unit=u, errors=errors)
            # 捕获类型错误或数值错误异常，抛出新的 ValueError 异常
            except (TypeError, ValueError) as err:
                raise ValueError(
                    f"cannot assemble the datetimes [{value}]: {err}"
                ) from err
    # 返回最终计算得到的 values
    return values
# 定义一个模块级别变量 __all__，用于指定在使用 import * 时导入的符号（变量、函数、类等）
__all__ = [
    "DateParseError",   # 导出 DateParseError 符号，可以通过 from 模块名 import * 导入
    "should_cache",     # 导出 should_cache 符号，可以通过 from 模块名 import * 导入
    "to_datetime",      # 导出 to_datetime 符号，可以通过 from 模块名 import * 导入
]
```