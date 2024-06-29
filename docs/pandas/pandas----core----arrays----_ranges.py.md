# `D:\src\scipysrc\pandas\pandas\core\arrays\_ranges.py`

```
"""
Helper functions to generate range-like data for DatetimeArray
(and possibly TimedeltaArray/PeriodArray)
"""

# 从未来导入类型检查
from __future__ import annotations

# 导入类型检查相关模块
from typing import TYPE_CHECKING

# 导入 NumPy 库
import numpy as np

# 导入 pandas 库中的内部函数和类
from pandas._libs.lib import i8max
from pandas._libs.tslibs import (
    BaseOffset,
    OutOfBoundsDatetime,
    Timedelta,
    Timestamp,
    iNaT,
)

# 如果是类型检查环境，则导入 npt 类型
if TYPE_CHECKING:
    from pandas._typing import npt


def generate_regular_range(
    start: Timestamp | Timedelta | None,
    end: Timestamp | Timedelta | None,
    periods: int | None,
    freq: BaseOffset,
    unit: str = "ns",
) -> npt.NDArray[np.intp]:
    """
    Generate a range of dates or timestamps with the spans between dates
    described by the given `freq` DateOffset.

    Parameters
    ----------
    start : Timedelta, Timestamp or None
        First point of produced date range.
    end : Timedelta, Timestamp or None
        Last point of produced date range.
    periods : int or None
        Number of periods in produced date range.
    freq : Tick
        Describes space between dates in produced date range.
    unit : str, default "ns"
        The resolution the output is meant to represent.

    Returns
    -------
    ndarray[np.int64]
        Representing the given resolution.
    """
    # 获取 start 的整数表示（如果不为 None）
    istart = start._value if start is not None else None
    # 获取 end 的整数表示（如果不为 None）
    iend = end._value if end is not None else None
    # 检查频率是否为固定频率，若不是则会抛出异常
    freq.nanos  # raises if non-fixed frequency
    # 将 freq 转换为 Timedelta 对象
    td = Timedelta(freq)
    b: int
    e: int
    try:
        # 将 td 转换为指定单位的 Timedelta 对象
        td = td.as_unit(unit, round_ok=False)
    except ValueError as err:
        # 如果转换失败，则抛出错误
        raise ValueError(
            f"freq={freq} is incompatible with unit={unit}. "
            "Use a lower freq or a higher unit instead."
        ) from err
    # 获取步长的整数值
    stride = int(td._value)

    # 根据条件计算起始点和结束点
    if periods is None and istart is not None and iend is not None:
        b = istart
        # 不能简单地使用 e = Timestamp(end) + 1，因为当步长过大时，arange 函数会出错，见 GH10887
        e = b + (iend - b) // stride * stride + stride // 2 + 1
    elif istart is not None and periods is not None:
        b = istart
        e = _generate_range_overflow_safe(b, periods, stride, side="start")
    elif iend is not None and periods is not None:
        e = iend + stride
        b = _generate_range_overflow_safe(e, periods, stride, side="end")
    else:
        # 如果既没有 periods，也没有指定 start 或 end，则抛出错误
        raise ValueError(
            "at least 'start' or 'end' should be specified if a 'period' is given."
        )

    with np.errstate(over="raise"):
        # 如果范围足够大，np.arange 可能会溢出并错误地返回空数组，这里进行异常处理
        try:
            values = np.arange(b, e, stride, dtype=np.int64)
        except FloatingPointError:
            xdr = [b]
            while xdr[-1] != e:
                xdr.append(xdr[-1] + stride)
            values = np.array(xdr[:-1], dtype=np.int64)
    return values


def _generate_range_overflow_safe(
    b: int, periods: int, stride: int, side: str
) -> int:
    """
    Helper function to safely generate a range that avoids overflow issues.

    Parameters
    ----------
    b : int
        Starting point of the range.
    periods : int
        Number of periods in the range.
    stride : int
        Stride between values in the range.
    side : str
        Side of the range to generate ("start" or "end").

    Returns
    -------
    int
        Ending point of the range.
    """
    if side == "start":
        return b + periods * stride
    elif side == "end":
        return b - periods * stride
    else:
        raise ValueError(f"Invalid side: {side}. Must be 'start' or 'end'.")
    endpoint: int, periods: int, stride: int, side: str = "start"


# 定义函数参数：一个整数 endpoint，表示范围的结束点；一个整数 periods，表示周期数；一个整数 stride，表示步长；一个字符串 side，默认为 "start"，表示周期的起始位置
endpoint: int, periods: int, stride: int, side: str = "start"
# Calculate the second endpoint for generating a range safely, ensuring no integer overflow occurs.
# Catch OverflowError and raise it as OutOfBoundsDatetime.

def _generate_range_overflow_safe_signed(
    endpoint: int, periods: int, stride: int, side: str
) -> int:
    """
    A special case for _generate_range_overflow_safe where `periods * stride`
    can be calculated without overflowing int64 bounds.
    """
    assert side in ["start", "end"]

    # Adjust stride if the range endpoint is 'end' to handle negative strides correctly
    if side == "end":
        stride *= -1
    # 设置 NumPy 错误状态以在溢出时引发异常
    with np.errstate(over="raise"):
        # 计算加数，确保使用 int64 类型以处理大整数
        addend = np.int64(periods) * np.int64(stride)
        try:
            # 在不溢出的情况下直接计算结果
            result = np.int64(endpoint) + addend
            if result == iNaT:
                # 如果结果是特殊的 NaT（Not a Time），则抛出 OverflowError
                raise OverflowError
            return int(result)
        except (FloatingPointError, OverflowError):
            # 处理可能的浮点错误或整数溢出异常
            pass

        # 确保 stride 和 endpoint 具有相同的符号，以避免溢出
        assert (stride > 0 and endpoint >= 0) or (stride < 0 and endpoint <= 0)

        if stride > 0:
            # 注意特殊情况：当结果接近实现界限但不超出时
            uresult = np.uint64(endpoint) + np.uint64(addend)
            i64max = np.uint64(i8max)
            assert uresult > i64max
            if uresult <= i64max + np.uint64(stride):
                return int(uresult)

    # 如果无法生成指定范围的时间序列，则抛出 OutOfBoundsDatetime 异常
    raise OutOfBoundsDatetime(
        f"Cannot generate range with {side}={endpoint} and periods={periods}"
    )
```