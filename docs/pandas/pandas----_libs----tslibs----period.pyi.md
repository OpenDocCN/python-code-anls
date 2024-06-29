# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\period.pyi`

```
from datetime import timedelta
from typing import Literal

import numpy as np

from pandas._libs.tslibs.dtypes import PeriodDtypeBase
from pandas._libs.tslibs.nattype import NaTType
from pandas._libs.tslibs.offsets import BaseOffset
from pandas._libs.tslibs.timestamps import Timestamp
from pandas._typing import (
    Frequency,
    npt,
)

# 错误消息常量定义
INVALID_FREQ_ERR_MSG: str
DIFFERENT_FREQ: str

# 自定义异常类，表示频率不兼容的错误
class IncompatibleFrequency(ValueError): ...

# 将 int64 的周期数组转换为 int64 的日期时间数组
def periodarr_to_dt64arr(
    periodarr: npt.NDArray[np.int64],  # const int64_t[:]
    freq: int,
) -> npt.NDArray[np.int64]: ...

# 将周期数组从一种频率转换为另一种频率的数组
def period_asfreq_arr(
    arr: npt.NDArray[np.int64],
    freq1: int,
    freq2: int,
    end: bool,
) -> npt.NDArray[np.int64]: ...

# 从周期数组中提取特定字段的值数组
def get_period_field_arr(
    field: str,
    arr: npt.NDArray[np.int64],  # const int64_t[:]
    freq: int,
) -> npt.NDArray[np.int64]: ...

# 根据序数值数组和频率类型创建周期数组
def from_ordinals(
    values: npt.NDArray[np.int64],  # const int64_t[:]
    freq: timedelta | BaseOffset | str,
) -> npt.NDArray[np.int64]: ...

# 从对象数组中提取周期的序数值数组
def extract_ordinals(
    values: npt.NDArray[np.object_],
    freq: Frequency | int,
) -> npt.NDArray[np.int64]: ...

# 从对象数组中提取频率信息
def extract_freq(
    values: npt.NDArray[np.object_],
) -> BaseOffset: ...

# 将周期数组格式化为字符串数组
def period_array_strftime(
    values: npt.NDArray[np.int64],
    dtype_code: int,
    na_rep,
    date_format: str | None,
) -> npt.NDArray[np.object_]: ...

# 暴露给测试使用的函数，将单个周期转换为另一个周期
def period_asfreq(ordinal: int, freq1: int, freq2: int, end: bool) -> int: ...

# 计算指定日期时间的序数值
def period_ordinal(
    y: int, m: int, d: int, h: int, min: int, s: int, us: int, ps: int, freq: int
) -> int: ...

# 将频率对象转换为对应的数据类型代码
def freq_to_dtype_code(freq: BaseOffset) -> int: ...

# 验证结束别名是否合法
def validate_end_alias(how: str) -> Literal["E", "S"]: ...

# 周期类的混合特性类，包含开始和结束时间的属性和频率匹配验证方法
class PeriodMixin:
    @property
    def end_time(self) -> Timestamp: ...
    @property
    def start_time(self) -> Timestamp: ...
    def _require_matching_freq(self, other: BaseOffset, base: bool = ...) -> None: ...

# 表示具体周期的类，包含序数、频率和数据类型信息
class Period(PeriodMixin):
    ordinal: int  # int64_t
    freq: BaseOffset
    _dtype: PeriodDtypeBase

    # 定义构造函数，支持多种参数类型和返回值
    # 错误: "__new__" must return a class instance (got "Union[Period, NaTType]")
    def __new__(  # type: ignore[misc]
        cls,
        value=...,
        freq: int | str | BaseOffset | None = ...,
        ordinal: int | None = ...,
        year: int | None = ...,
        month: int | None = ...,
        quarter: int | None = ...,
        day: int | None = ...,
        hour: int | None = ...,
        minute: int | None = ...,
        second: int | None = ...,
    ) -> Period | NaTType: ...

    # 类方法，根据给定频率转换为对应的 BaseOffset 对象
    @classmethod
    def _maybe_convert_freq(cls, freq) -> BaseOffset: ...

    # 类方法，根据序数和频率创建周期对象
    @classmethod
    def _from_ordinal(cls, ordinal: int, freq: BaseOffset) -> Period: ...

    # 类方法，返回当前时刻对应的周期对象
    @classmethod
    def now(cls, freq: Frequency) -> Period: ...

    # 格式化周期对象为字符串
    def strftime(self, fmt: str | None) -> str: ...

    # 将周期对象转换为 Timestamp 对象
    def to_timestamp(
        self,
        freq: str | BaseOffset | None = ...,
        how: str = ...,
    ) -> Timestamp: ...

    # 将周期对象的频率转换为指定频率的周期对象
    def asfreq(self, freq: str | BaseOffset, how: str = ...) -> Period: ...
    # 返回当前日期的频率字符串表示形式
    def freqstr(self) -> str: ...

    # 判断当前日期是否为闰年，返回布尔值
    @property
    def is_leap_year(self) -> bool: ...

    # 返回当前日期所在月份的天数
    @property
    def daysinmonth(self) -> int: ...

    # 返回当前日期所在月份的天数
    @property
    def days_in_month(self) -> int: ...

    # 返回当前日期所在季度的年份
    @property
    def qyear(self) -> int: ...

    # 返回当前日期所在季度（1-4）
    @property
    def quarter(self) -> int: ...

    # 返回当前日期在年份中的第几天（1-365或366）
    @property
    def day_of_year(self) -> int: ...

    # 返回当前日期是星期几（0-6，星期一到星期日）
    @property
    def weekday(self) -> int: ...

    # 返回当前日期是星期几（0-6，星期一到星期日）
    @property
    def day_of_week(self) -> int: ...

    # 返回当前日期在年份中的第几周
    @property
    def week(self) -> int: ...

    # 返回当前日期在年份中的第几周
    @property
    def weekofyear(self) -> int: ...

    # 返回当前时间的秒数（0-59）
    @property
    def second(self) -> int: ...

    # 返回当前时间的分钟数（0-59）
    @property
    def minute(self) -> int: ...

    # 返回当前时间的小时数（0-23）
    @property
    def hour(self) -> int: ...

    # 返回当前日期的天数（1-31）
    @property
    def day(self) -> int: ...

    # 返回当前日期的月份（1-12）
    @property
    def month(self) -> int: ...

    # 返回当前日期的年份
    @property
    def year(self) -> int: ...

    # 实现日期之间的减法运算，返回一个 Period 或 BaseOffset 对象
    def __sub__(self, other) -> Period | BaseOffset: ...

    # 实现日期之间的加法运算，返回一个 Period 对象
    def __add__(self, other) -> Period: ...
```