# `D:\src\scipysrc\pandas\pandas\core\tools\timedeltas.py`

```
"""
timedelta support tools
"""

# 导入所需的模块和类型定义
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    overload,
)

import numpy as np

from pandas._libs import lib
from pandas._libs.tslibs import (
    NaT,
    NaTType,
)
from pandas._libs.tslibs.timedeltas import (
    Timedelta,
    disallow_ambiguous_unit,
    parse_timedelta_unit,
)

from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas.core.dtypes.generic import (
    ABCIndex,
    ABCSeries,
)

from pandas.core.arrays.timedeltas import sequence_to_td64ns

# 如果是类型检查，导入额外的类型
if TYPE_CHECKING:
    from collections.abc import Hashable
    from datetime import timedelta

    from pandas._libs.tslibs.timedeltas import UnitChoices
    from pandas._typing import (
        ArrayLike,
        DateTimeErrorChoices,
    )

    from pandas import (
        Index,
        Series,
        TimedeltaIndex,
    )


@overload
def to_timedelta(
    arg: str | float | timedelta,
    unit: UnitChoices | None = ...,
    errors: DateTimeErrorChoices = ...,
) -> Timedelta: ...


@overload
def to_timedelta(
    arg: Series,
    unit: UnitChoices | None = ...,
    errors: DateTimeErrorChoices = ...,
) -> Series: ...


@overload
def to_timedelta(
    arg: list | tuple | range | ArrayLike | Index,
    unit: UnitChoices | None = ...,
    errors: DateTimeErrorChoices = ...,
) -> TimedeltaIndex: ...


def to_timedelta(
    arg: str
    | int
    | float
    | timedelta
    | list
    | tuple
    | range
    | ArrayLike
    | Index
    | Series,
    unit: UnitChoices | None = None,
    errors: DateTimeErrorChoices = "raise",
) -> Timedelta | TimedeltaIndex | Series | NaTType | Any:
    """
    Convert argument to timedelta.

    Timedeltas are absolute differences in times, expressed in difference
    units (e.g. days, hours, minutes, seconds). This method converts
    an argument from a recognized timedelta format / value into
    a Timedelta type.

    Parameters
    ----------
    arg : str, timedelta, list-like or Series
        The data to be converted to timedelta.

        .. versionchanged:: 2.0
            Strings with units 'M', 'Y' and 'y' do not represent
            unambiguous timedelta values and will raise an exception.
    """
    """
    unit : str, optional
        用于指定 `arg` 的数值单位，默认为 ``"ns"``。

        可能的取值包括：

        * 'W'
        * 'D' / 'days' / 'day'
        * 'hours' / 'hour' / 'hr' / 'h' / 'H'
        * 'm' / 'minute' / 'min' / 'minutes'
        * 's' / 'seconds' / 'sec' / 'second' / 'S'
        * 'ms' / 'milliseconds' / 'millisecond' / 'milli' / 'millis'
        * 'us' / 'microseconds' / 'microsecond' / 'micro' / 'micros'
        * 'ns' / 'nanoseconds' / 'nano' / 'nanos' / 'nanosecond'

        当 `arg` 包含字符串且 ``errors="raise"`` 时，不应指定单位。

        .. deprecated:: 2.2.0
            单位 'H' 和 'S' 已废弃，将在未来版本中移除。请使用 'h' 和 's'。

    errors : {'raise', 'coerce'}, default 'raise'
        - 如果为 'raise'，则无效的解析将引发异常。
        - 如果为 'coerce'，则无效的解析将设置为 NaT（Not a Time）。

    Returns
    -------
    timedelta
        解析成功时返回。
        返回类型取决于输入：

        - 类似列表：timedelta64 数据类型的 TimedeltaIndex
        - Series：timedelta64 数据类型的 Series
        - 标量：Timedelta

    See Also
    --------
    DataFrame.astype : 将参数转换为指定数据类型。
    to_datetime : 将参数转换为日期时间类型。
    convert_dtypes : 转换数据类型。

    Notes
    -----
    如果精度高于纳秒，字符串输入的持续时间精度将被截断为纳秒。

    Examples
    --------
    将单个字符串解析为 Timedelta：

    >>> pd.to_timedelta("1 days 06:05:01.00003")
    Timedelta('1 days 06:05:01.000030')
    >>> pd.to_timedelta("15.5us")
    Timedelta('0 days 00:00:00.000015500')

    解析列表或数组中的多个字符串：

    >>> pd.to_timedelta(["1 days 06:05:01.00003", "15.5us", "nan"])
    TimedeltaIndex(['1 days 06:05:01.000030', '0 days 00:00:00.000015500', NaT],
                   dtype='timedelta64[ns]', freq=None)

    通过指定 `unit` 关键字参数转换数字：

    >>> pd.to_timedelta(np.arange(5), unit="s")
    TimedeltaIndex(['0 days 00:00:00', '0 days 00:00:01', '0 days 00:00:02',
                    '0 days 00:00:03', '0 days 00:00:04'],
                   dtype='timedelta64[ns]', freq=None)
    >>> pd.to_timedelta(np.arange(5), unit="D")
    TimedeltaIndex(['0 days', '1 days', '2 days', '3 days', '4 days'],
                   dtype='timedelta64[ns]', freq=None)
    """
    # 如果指定了单位，则解析时间单位
    if unit is not None:
        unit = parse_timedelta_unit(unit)
        # 禁止使用含糊不清的单位
        disallow_ambiguous_unit(unit)

    # 如果错误处理选项不在 ("raise", "coerce") 中，则抛出异常
    if errors not in ("raise", "coerce"):
        raise ValueError("errors must be one of 'raise', or 'coerce'.")

    # 如果参数为空，则直接返回参数
    if arg is None:
        return arg
    # 如果参数是 Series 类型，则转换成相应类型并返回
    elif isinstance(arg, ABCSeries):
        values = _convert_listlike(arg._values, unit=unit, errors=errors)
        return arg._constructor(values, index=arg.index, name=arg.name)
    # 如果参数是 ABCIndex 类型的实例，则调用 _convert_listlike 处理，并指定单位和错误处理方式，使用参数的名称作为名称
    elif isinstance(arg, ABCIndex):
        return _convert_listlike(arg, unit=unit, errors=errors, name=arg.name)
    
    # 如果参数是 numpy 数组并且维度为 0，则提取数组标量并进行处理
    elif isinstance(arg, np.ndarray) and arg.ndim == 0:
        # 错误：赋值时类型不兼容（表达式类型为 "object"，变量类型为各种可能的类型）
        arg = lib.item_from_zerodim(arg)  # type: ignore[assignment]  # 将零维数组转换为标量
    # 如果参数是类似列表的对象并且维度为 1，则调用 _convert_listlike 处理，并指定单位和错误处理方式
    elif is_list_like(arg) and getattr(arg, "ndim", 1) == 1:
        return _convert_listlike(arg, unit=unit, errors=errors)
    
    # 如果参数的维度大于 1，则抛出类型错误异常
    elif getattr(arg, "ndim", 1) > 1:
        raise TypeError(
            "arg must be a string, timedelta, list, tuple, 1-d array, or Series"
        )

    # 如果参数是字符串并且 unit 不为 None，则抛出数值错误异常
    if isinstance(arg, str) and unit is not None:
        raise ValueError("unit must not be specified if the input is/contains a str")

    # 否则，参数应为标量值。将其转换为 timedelta 类型后返回
    return _coerce_scalar_to_timedelta_type(arg, unit=unit, errors=errors)
# 将字符串'r'转换为 timedelta 对象
def _coerce_scalar_to_timedelta_type(
    r, unit: UnitChoices | None = "ns", errors: DateTimeErrorChoices = "raise"
) -> Timedelta | NaTType:
    """Convert string 'r' to a timedelta object."""
    result: Timedelta | NaTType

    # 尝试将'r'转换为 timedelta 对象
    try:
        result = Timedelta(r, unit)
    # 如果数值错误，则根据错误处理选项进行处理
    except ValueError:
        if errors == "raise":
            raise
        # 如果错误处理选项为"coerce"，则将结果设置为 NaT
        result = NaT

    # 返回转换后的结果
    return result


# 将对象列表转换为 timedelta 索引对象
def _convert_listlike(
    arg,
    unit: UnitChoices | None = None,
    errors: DateTimeErrorChoices = "raise",
    name: Hashable | None = None,
):
    """Convert a list of objects to a timedelta index object."""
    # 获取 arg 的数据类型
    arg_dtype = getattr(arg, "dtype", None)

    # 如果 arg 是列表或元组，或者其数据类型为 None，则转换为 dtype 为 object 的 NumPy 数组
    if isinstance(arg, (list, tuple)) or arg_dtype is None:
        arg = np.array(arg, dtype=object)
    # 如果 arg 的数据类型是 ArrowDtype，并且其 kind 为 'm'，则直接返回 arg
    elif isinstance(arg_dtype, ArrowDtype) and arg_dtype.kind == "m":
        return arg

    # 将序列转换为 td64ns 类型的数组
    td64arr = sequence_to_td64ns(arg, unit=unit, errors=errors, copy=False)[0]

    # 导入 TimedeltaIndex 类
    from pandas import TimedeltaIndex

    # 使用 td64arr 创建 TimedeltaIndex 对象，指定名称为 name
    value = TimedeltaIndex(td64arr, name=name)
    # 返回创建的 TimedeltaIndex 对象
    return value
```