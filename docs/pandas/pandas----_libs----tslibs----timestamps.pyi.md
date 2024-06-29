# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\timestamps.pyi`

```
# 导入需要的模块和类型别名
from datetime import (
    date as _date,  # 导入日期类型，并命名为 _date
    datetime,       # 导入日期时间类型
    time as _time,  # 导入时间类型，并命名为 _time
    timedelta,      # 导入时间差类型
    tzinfo as _tzinfo,  # 导入时区信息类型，并命名为 _tzinfo
)
from time import struct_time  # 导入 struct_time 类型
from typing import (
    ClassVar,       # 类型变量
    Literal,        # 字面量类型
    TypeAlias,      # 类型别名
    overload,       # 重载装饰器
)

import numpy as np  # 导入 numpy 库，并使用 np 别名

from pandas._libs.tslibs import (
    BaseOffset,     # 基础偏移类型
    NaTType,        # Not-a-Time 类型
    Period,         # 时间段类型
    Tick,           # Tick 类型
    Timedelta,      # 时间差类型
)
from pandas._typing import (
    Self,                   # 自引用类型
    TimestampNonexistent,   # 不存在时间戳类型
)

_TimeZones: TypeAlias = str | _tzinfo | None | int  # 时区类型别名定义

def integer_op_not_supported(obj: object) -> TypeError: ...
    # 定义一个函数，用于处理整数操作不支持的情况，返回 TypeError

class Timestamp(datetime):
    _creso: int
    min: ClassVar[Timestamp]
    max: ClassVar[Timestamp]

    resolution: ClassVar[Timedelta]
    _value: int  # np.int64
    # 定义 _value 属性，表示 np.int64 类型的时间戳值

    # error: "__new__" must return a class instance (got "Union[Timestamp, NaTType]")
    def __new__(  # type: ignore[misc]
        cls: type[Self],
        ts_input: np.integer | float | str | _date | datetime | np.datetime64 = ...,
        year: int | None = ...,
        month: int | None = ...,
        day: int | None = ...,
        hour: int | None = ...,
        minute: int | None = ...,
        second: int | None = ...,
        microsecond: int | None = ...,
        tzinfo: _tzinfo | None = ...,
        *,
        nanosecond: int | None = ...,
        tz: _TimeZones = ...,
        unit: str | int | None = ...,
        fold: int | None = ...,
    ) -> Self | NaTType: ...
    # 定义 __new__ 方法，用于创建 Timestamp 类的新实例，支持多种输入类型，返回 Self 或 NaTType

    @classmethod
    def _from_value_and_reso(
        cls, value: int, reso: int, tz: _TimeZones
    ) -> Timestamp: ...
    # 类方法，根据给定的值、分辨率和时区信息创建 Timestamp 对象实例

    @property
    def value(self) -> int: ...
    # value 属性，返回时间戳的值，类型为 int（np.int64）

    @property
    def year(self) -> int: ...
    # year 属性，返回时间戳的年份

    @property
    def month(self) -> int: ...
    # month 属性，返回时间戳的月份

    @property
    def day(self) -> int: ...
    # day 属性，返回时间戳的日期

    @property
    def hour(self) -> int: ...
    # hour 属性，返回时间戳的小时

    @property
    def minute(self) -> int: ...
    # minute 属性，返回时间戳的分钟

    @property
    def second(self) -> int: ...
    # second 属性，返回时间戳的秒数

    @property
    def microsecond(self) -> int: ...
    # microsecond 属性，返回时间戳的微秒数

    @property
    def nanosecond(self) -> int: ...
    # nanosecond 属性，返回时间戳的纳秒数

    @property
    def tzinfo(self) -> _tzinfo | None: ...
    # tzinfo 属性，返回时间戳的时区信息

    @property
    def tz(self) -> _tzinfo | None: ...
    # tz 属性，返回时间戳的时区信息

    @property
    def fold(self) -> int: ...
    # fold 属性，返回时间戳的 fold 信息

    @classmethod
    def fromtimestamp(cls, ts: float, tz: _TimeZones = ...) -> Self: ...
    # 类方法，根据时间戳创建 Timestamp 对象实例，支持指定时区信息

    @classmethod
    def utcfromtimestamp(cls, ts: float) -> Self: ...
    # 类方法，根据 UTC 时间戳创建 Timestamp 对象实例

    @classmethod
    def today(cls, tz: _TimeZones = ...) -> Self: ...
    # 类方法，返回当前日期时间的 Timestamp 对象实例，支持指定时区信息

    @classmethod
    def fromordinal(
        cls,
        ordinal: int,
        tz: _TimeZones = ...,
    ) -> Self: ...
    # 类方法，根据序数创建 Timestamp 对象实例，支持指定时区信息

    @classmethod
    def now(cls, tz: _TimeZones = ...) -> Self: ...
    # 类方法，返回当前日期时间的 Timestamp 对象实例，支持指定时区信息

    @classmethod
    def utcnow(cls) -> Self: ...
    # 类方法，返回当前 UTC 时间的 Timestamp 对象实例

    # error: Signature of "combine" incompatible with supertype "datetime"
    @classmethod
    def combine(  # type: ignore[override]
        cls, date: _date, time: _time
    ) -> datetime: ...
    # 类方法，组合给定的日期和时间对象，返回 datetime 对象

    @classmethod
    def fromisoformat(cls, date_string: str) -> Self: ...
    # 类方法，从 ISO 格式的日期字符串创建 Timestamp 对象实例

    def strftime(self, format: str) -> str: ...
    # 格式化时间戳对象为指定格式的字符串表示

    def __format__(self, fmt: str) -> str: ...
    # 格式化时间戳对象为指定格式的字符串表示

    def toordinal(self) -> int: ...
    # 将时间戳对象转换为序数表示
    # 返回时间元组表示对象自身的时间信息
    def timetuple(self) -> struct_time: ...

    # 返回对象自身的 UNIX 时间戳
    def timestamp(self) -> float: ...

    # 返回对象自身的 UTC 时间元组
    def utctimetuple(self) -> struct_time: ...

    # 返回对象自身的日期部分
    def date(self) -> _date: ...

    # 返回对象自身的时间部分
    def time(self) -> _time: ...

    # 返回带有时区信息的对象自身的时间部分
    def timetz(self) -> _time: ...

    # 替换对象自身的日期时间部分，支持年、月、日、时、分、秒、微秒、纳秒、时区、折叠状态的设置
    # 注意：此处违反了Liskov Substitution Principle（LSP），因为在datetime.datetime.replace中不存在纳秒，并且后面的位置参数跟随其后
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
        tzinfo: _tzinfo | type[object] | None = ...,
        fold: int | None = ...,
    ) -> Self: ...

    # 将对象自身转换为指定时区的时间
    # 注意：此处违反了LSP，因为datetime.datetime.astimezone具有时区tz的默认值
    def astimezone(self, tz: _TimeZones) -> Self: ...

    # 返回对象自身的字符串表示，按照ISO 8601扩展格式
    def ctime(self) -> str: ...

    # 返回对象自身的 ISO 8601 格式的字符串表示
    # 可选参数 sep 指定日期和时间的分隔符，timespec 指定显示的时间精度
    def isoformat(self, sep: str = ..., timespec: str = ...) -> str: ...

    # 从指定格式的字符串转换为对象自身
    # 注意：实际上，strptime 被禁用，并抛出 NotImplementedError
    def strptime(
        cls,
        date_string: str,
        format: str,
    ) -> Self: ...

    # 返回对象自身的 UTC 偏移量
    def utcoffset(self) -> timedelta | None: ...

    # 返回对象自身的时区名称
    def tzname(self) -> str | None: ...

    # 返回对象自身的 DST（夏令时）偏移量
    def dst(self) -> timedelta | None: ...

    # 判断对象自身是否小于或等于另一个 datetime 对象
    def __le__(self, other: datetime) -> bool: ...

    # 判断对象自身是否小于另一个 datetime 对象
    def __lt__(self, other: datetime) -> bool: ...

    # 判断对象自身是否大于或等于另一个 datetime 对象
    def __ge__(self, other: datetime) -> bool: ...

    # 判断对象自身是否大于另一个 datetime 对象
    def __gt__(self, other: datetime) -> bool: ...

    # 重载加法运算符，支持与 numpy 数组的加法操作
    @overload
    def __add__(self, other: np.ndarray) -> np.ndarray: ...

    # 重载加法运算符，支持与 timedelta、numpy.timedelta64 或 Tick 对象的加法操作
    @overload
    def __add__(self, other: timedelta | np.timedelta64 | Tick) -> Self: ...

    # 支持与 timedelta 对象的右向加法操作
    def __radd__(self, other: timedelta) -> Self: ...

    # 重载减法运算符，支持与 datetime 对象的减法操作
    @overload
    def __sub__(self, other: datetime) -> Timedelta: ...

    # 重载减法运算符，支持与 timedelta、numpy.timedelta64 或 Tick 对象的减法操作
    @overload
    def __sub__(self, other: timedelta | np.timedelta64 | Tick) -> Self: ...

    # 返回对象自身的哈希值
    def __hash__(self) -> int: ...

    # 返回对象自身的星期几，星期一为0，星期日为6
    def weekday(self) -> int: ...

    # 返回对象自身的 ISO 8601 格式的星期几，星期一为1，星期日为7
    def isoweekday(self) -> int: ...

    # 返回对象自身的 ISO 8601 日历元组 (ISO year, ISO week number, ISO weekday)
    # 注意：返回类型为元组 (int, int, int)，与父类 date 中定义的 _IsoCalendarDate 不兼容
    def isocalendar(self) -> tuple[int, int, int]: ...

    # 返回对象自身的年份是否为闰年的布尔值
    @property
    def is_leap_year(self) -> bool: ...

    # 返回对象自身的月初日期的布尔值
    @property
    def is_month_start(self) -> bool: ...

    # 返回对象自身的季初日期的布尔值
    @property
    def is_quarter_start(self) -> bool: ...

    # 返回对象自身的年初日期的布尔值
    @property
    def is_year_start(self) -> bool: ...

    # 返回对象自身的月末日期的布尔值
    @property
    def is_month_end(self) -> bool: ...

    # 返回对象自身的季末日期的布尔值
    @property
    def is_quarter_end(self) -> bool: ...

    # 返回对象自身的年末日期的布尔值
    @property
    def is_year_end(self) -> bool: ...

    # 返回对象自身的 Python datetime 对象表示
    def to_pydatetime(self, warn: bool = ...) -> datetime: ...
    # 返回当前对象的日期时间表示为 np.datetime64 类型
    def to_datetime64(self) -> np.datetime64: ...

    # 将当前对象转换为 Period 对象，可选指定频率 freq
    def to_period(self, freq: BaseOffset | str | None = None) -> Period: ...

    # 返回当前对象的儒略日期表示为 np.float64 类型
    def to_julian_date(self) -> np.float64: ...

    # 返回当前对象的 asm8 表示，即 np.datetime64 类型
    @property
    def asm8(self) -> np.datetime64: ...

    # 将当前对象的时区转换为指定时区 tz，并返回当前对象本身
    def tz_convert(self, tz: _TimeZones) -> Self: ...

    # TODO: 可能返回 NaT，将当前对象本地化到指定时区 tz
    def tz_localize(
        self,
        tz: _TimeZones,
        ambiguous: bool | Literal["raise", "NaT"] = ...,
        nonexistent: TimestampNonexistent = ...,
    ) -> Self: ...

    # 返回一个新的对象，其时间部分已被标准化（例如，时、分、秒等）
    def normalize(self) -> Self: ...

    # TODO: round/floor/ceil 可能返回 NaT，将时间舍入到指定频率 freq
    def round(
        self,
        freq: str,
        ambiguous: bool | Literal["raise", "NaT"] = ...,
        nonexistent: TimestampNonexistent = ...,
    ) -> Self: ...

    # 将时间向下舍入到指定频率 freq
    def floor(
        self,
        freq: str,
        ambiguous: bool | Literal["raise", "NaT"] = ...,
        nonexistent: TimestampNonexistent = ...,
    ) -> Self: ...

    # 将时间向上舍入到指定频率 freq
    def ceil(
        self,
        freq: str,
        ambiguous: bool | Literal["raise", "NaT"] = ...,
        nonexistent: TimestampNonexistent = ...,
    ) -> Self: ...

    # 返回当前日期的星期几名称，可选指定地区 locale
    def day_name(self, locale: str | None = ...) -> str: ...

    # 返回当前日期的月份名称，可选指定地区 locale
    def month_name(self, locale: str | None = ...) -> str: ...

    # 返回当前日期的星期几（0-6 表示周一到周日）
    @property
    def day_of_week(self) -> int: ...

    # 返回当前日期的星期几（0-6 表示周一到周日）
    @property
    def dayofweek(self) -> int: ...

    # 返回当前日期在年份中的第几天（1-365 或 366）
    @property
    def day_of_year(self) -> int: ...

    # 返回当前日期在年份中的第几天（1-365 或 366）
    @property
    def dayofyear(self) -> int: ...

    # 返回当前日期所在季度（1-4）
    @property
    def quarter(self) -> int: ...

    # 返回当前日期所在周是全年的第几周（1-53）
    @property
    def week(self) -> int: ...

    # 将当前对象转换为 numpy.datetime64 类型，可选指定数据类型 dtype 和是否复制 copy
    def to_numpy(
        self, dtype: np.dtype | None = ..., copy: bool = ...
    ) -> np.datetime64: ...

    # 返回当前日期时间对象的字符串表示形式
    @property
    def _date_repr(self) -> str: ...

    # 返回当前月份的天数
    @property
    def days_in_month(self) -> int: ...

    # 返回当前月份的天数
    @property
    def daysinmonth(self) -> int: ...

    # 返回当前日期时间对象的单位（"D" 表示日历天）
    @property
    def unit(self) -> str: ...

    # 将当前对象表示的时间作为指定单位 unit 的 Timestamp 对象返回
    def as_unit(self, unit: str, round_ok: bool = ...) -> Timestamp: ...
```