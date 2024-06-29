# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\nattype.pyi`

```
# 导入日期相关的类和函数，使用别名以便在代码中简洁地引用
from datetime import (
    date as date_,
    datetime,
    time as time_,
    timedelta,
    tzinfo as _tzinfo,
)

# 导入类型相关的定义，用于类型提示
from typing import (
    Literal,
    NoReturn,
    TypeAlias,
)

# 导入NumPy库，使用别名np以便在代码中简洁地引用
import numpy as np

# 从pandas库中导入特定的类和类型定义
from pandas._libs.tslibs.period import Period
from pandas._typing import (
    Frequency,
    Self,
    TimestampNonexistent,
)

# 定义NaT类型的别名
NaT: NaTType
# 定义iNaT，表示NaT的整数形式
iNaT: int
# 定义nat_strings，表示NaT的字符串形式的集合
nat_strings: set[str]

# 定义_NaTComparisonTypes类型别名，表示支持与NaT比较的数据类型
_NaTComparisonTypes: TypeAlias = (
    datetime | timedelta | Period | np.datetime64 | np.timedelta64
)

# 定义_NatComparison类，用于比较NaT对象与其他_NaTComparisonTypes对象
class _NatComparison:
    def __call__(self, other: _NaTComparisonTypes) -> bool: ...

# 定义NaTType类，表示"不可用时间"类型
class NaTType:
    _value: np.int64
    @property
    def value(self) -> int: ...
    @property
    def asm8(self) -> np.datetime64: ...
    def to_datetime64(self) -> np.datetime64: ...
    def to_numpy(
        self, dtype: np.dtype | str | None = ..., copy: bool = ...
    ) -> np.datetime64 | np.timedelta64: ...
    @property
    def is_leap_year(self) -> bool: ...
    @property
    def is_month_start(self) -> bool: ...
    @property
    def is_quarter_start(self) -> bool: ...
    @property
    def is_year_start(self) -> bool: ...
    @property
    def is_month_end(self) -> bool: ...
    @property
    def is_quarter_end(self) -> bool: ...
    @property
    def is_year_end(self) -> bool: ...
    @property
    def day_of_year(self) -> float: ...
    @property
    def dayofyear(self) -> float: ...
    @property
    def days_in_month(self) -> float: ...
    @property
    def daysinmonth(self) -> float: ...
    @property
    def day_of_week(self) -> float: ...
    @property
    def dayofweek(self) -> float: ...
    @property
    def week(self) -> float: ...
    @property
    def weekofyear(self) -> float: ...
    @property
    def fold(self) -> int: ...
    def day_name(self) -> float: ...
    def month_name(self) -> float: ...
    def weekday(self) -> float: ...
    def isoweekday(self) -> float: ...
    def isoformat(self, sep: str = ..., timespec: str = ...) -> str: ...
    def strftime(self, format: str) -> NoReturn: ...
    def total_seconds(self) -> float: ...
    def today(self, *args, **kwargs) -> NaTType: ...
    def now(self, *args, **kwargs) -> NaTType: ...
    def to_pydatetime(self) -> NaTType: ...
    def date(self) -> NaTType: ...
    def round(
        self,
        freq: Frequency,
        ambiguous: bool | Literal["raise"] | NaTType = ...,
        nonexistent: TimestampNonexistent = ...,
    ) -> NaTType: ...
    def floor(
        self,
        freq: Frequency,
        ambiguous: bool | Literal["raise"] | NaTType = ...,
        nonexistent: TimestampNonexistent = ...,
    ) -> NaTType: ...
    def ceil(
        self,
        freq: Frequency,
        ambiguous: bool | Literal["raise"] | NaTType = ...,
        nonexistent: TimestampNonexistent = ...,
    ) -> NaTType: ...
    def combine(cls, date: date_, time: time_) -> NoReturn: ...
    @property
    def tzinfo(self) -> None: ...
    @property
    def tz(self) -> None: ...
    # 定义一个方法 tz_convert，用于将日期时间转换到指定时区
    def tz_convert(self, tz: _tzinfo | str | None) -> NaTType: ...

    # 定义一个方法 tz_localize，用于将日期时间本地化到指定时区
    # 参数：
    #   - tz: 目标时区信息，可以是时区对象、时区名称字符串或 None
    #   - ambiguous: 是否处理本地化时存在歧义的情况，可以选择抛出异常或自动处理
    #   - nonexistent: 处理不存在的时间戳情况的策略
    # 返回值：NaTType，表示本地化后的日期时间对象或不存在的时间类型
    def tz_localize(
        self,
        tz: _tzinfo | str | None,
        ambiguous: bool | Literal["raise"] | NaTType = ...,
        nonexistent: TimestampNonexistent = ...,
    ) -> NaTType: ...

    # 定义一个方法 replace，用于替换日期时间对象的各个部分
    # 参数：
    #   - year: 年份
    #   - month: 月份
    #   - day: 日
    #   - hour: 小时
    #   - minute: 分钟
    #   - second: 秒
    #   - microsecond: 微秒
    #   - nanosecond: 纳秒
    #   - tzinfo: 时区信息
    #   - fold: 折叠标志
    # 返回值：NaTType，表示替换后的日期时间对象或不存在的时间类型
    def replace(
        self,
        year: int | None = ...,
        month: int | None = ...,
        day: int | None = ...,
        hour: int | None = ...,
        minute: int | None = ...,
        second: int | None = ...,
        microsecond: int | None = ...,
        nanosecond: int | None = ...,
        tzinfo: _tzinfo | None = ...,
        fold: int | None = ...,
    ) -> NaTType: ...

    # 定义一个属性 year，返回日期时间对象的年份部分
    @property
    def year(self) -> float: ...

    # 定义一个属性 quarter，返回日期时间对象的季度部分
    @property
    def quarter(self) -> float: ...

    # 定义一个属性 month，返回日期时间对象的月份部分
    @property
    def month(self) -> float: ...

    # 定义一个属性 day，返回日期时间对象的日期部分
    @property
    def day(self) -> float: ...

    # 定义一个属性 hour，返回日期时间对象的小时部分
    @property
    def hour(self) -> float: ...

    # 定义一个属性 minute，返回日期时间对象的分钟部分
    @property
    def minute(self) -> float: ...

    # 定义一个属性 second，返回日期时间对象的秒部分
    @property
    def second(self) -> float: ...

    # 定义一个属性 millisecond，返回日期时间对象的毫秒部分
    @property
    def millisecond(self) -> float: ...

    # 定义一个属性 microsecond，返回日期时间对象的微秒部分
    @property
    def microsecond(self) -> float: ...

    # 定义一个属性 nanosecond，返回日期时间对象的纳秒部分
    @property
    def nanosecond(self) -> float: ...

    # 注入 timedelta 的属性
    # 定义一个属性 days，返回时间差对象的天数部分
    @property
    def days(self) -> float: ...

    # 定义一个属性 seconds，返回时间差对象的秒数部分
    @property
    def seconds(self) -> float: ...

    # 定义一个属性 microseconds，返回时间差对象的微秒数部分
    @property
    def microseconds(self) -> float: ...

    # 定义一个属性 nanoseconds，返回时间差对象的纳秒数部分
    @property
    def nanoseconds(self) -> float: ...

    # 注入 Period 的属性
    # 定义一个属性 qyear，返回周期对象的年度季度部分
    @property
    def qyear(self) -> float: ...

    # 定义对象的相等比较方法，判断两个对象是否相等
    def __eq__(self, other: object) -> bool: ...

    # 定义对象的不等比较方法，判断两个对象是否不相等
    def __ne__(self, other: object) -> bool: ...

    # 定义对象的小于比较方法，用于对象之间的小于比较
    __lt__: _NatComparison

    # 定义对象的小于等于比较方法，用于对象之间的小于等于比较
    __le__: _NatComparison

    # 定义对象的大于比较方法，用于对象之间的大于比较
    __gt__: _NatComparison

    # 定义对象的大于等于比较方法，用于对象之间的大于等于比较
    __ge__: _NatComparison

    # 定义对象的减法运算方法，支持对象与时间差或日期时间对象的减法操作
    def __sub__(self, other: Self | timedelta | datetime) -> Self: ...

    # 定义对象的反向减法运算方法，支持时间差或日期时间对象与对象的减法操作
    def __rsub__(self, other: Self | timedelta | datetime) -> Self: ...

    # 定义对象的加法运算方法，支持对象与时间差或日期时间对象的加法操作
    def __add__(self, other: Self | timedelta | datetime) -> Self: ...

    # 定义对象的反向加法运算方法，支持时间差或日期时间对象与对象的加法操作
    def __radd__(self, other: Self | timedelta | datetime) -> Self: ...

    # 定义对象的哈希计算方法，返回对象的哈希值
    def __hash__(self) -> int: ...

    # 定义对象的单位转换方法，将对象转换为指定单位的表示形式
    # 参数：
    #   - unit: 要转换的单位
    #   - round_ok: 是否允许四舍五入
    # 返回值：NaTType，表示转换后的对象或不存在的时间类型
    def as_unit(self, unit: str, round_ok: bool = ...) -> NaTType: ...
```