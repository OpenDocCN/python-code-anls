# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\offsets.pyi`

```
from datetime import (
    datetime,            # 导入 datetime 类，用于处理日期和时间
    time,                # 导入 time 类，用于处理时间
    timedelta,           # 导入 timedelta 类，用于处理时间间隔
)
from typing import (
    Any,                 # 导入 Any 类型，表示任意类型
    Collection,          # 导入 Collection 类型，表示集合类型
    Literal,             # 导入 Literal 类型，用于类型字面值
    TypeVar,             # 导入 TypeVar 类型，用于定义泛型变量
    overload,            # 导入 overload 装饰器，用于多态函数重载
)

import numpy as np         # 导入 NumPy 库，用于数值计算

from pandas._libs.tslibs.nattype import NaTType   # 导入 NaTType 类型，表示不确定的时间类型
from pandas._typing import (
    OffsetCalendar,     # 导入 OffsetCalendar 类型，表示偏移量日历
    Self,               # 导入 Self 类型，表示自身类型
    npt,                # 导入 npt 类型，表示 NumPy 类型
)

_BaseOffsetT = TypeVar("_BaseOffsetT", bound=BaseOffset)   # 定义泛型变量 _BaseOffsetT，约束为 BaseOffset 类型
_DatetimeT = TypeVar("_DatetimeT", bound=datetime)        # 定义泛型变量 _DatetimeT，约束为 datetime 类型
_TimedeltaT = TypeVar("_TimedeltaT", bound=timedelta)     # 定义泛型变量 _TimedeltaT，约束为 timedelta 类型

_relativedelta_kwds: set[str]    # 声明 _relativedelta_kwds 变量为 set[str] 类型，存储字符串集合
prefix_mapping: dict[str, type]   # 声明 prefix_mapping 变量为 dict[str, type] 类型，映射字符串到类型

class ApplyTypeError(TypeError): ...   # 定义 ApplyTypeError 类，继承自 TypeError 类，用于处理应用类型错误

class BaseOffset:
    n: int                 # 类属性 n 表示偏移量的整数值
    normalize: bool        # 类属性 normalize 表示是否标准化的布尔值

    def __init__(self, n: int = ..., normalize: bool = ...) -> None: ...
                           # 构造函数初始化对象，参数 n 表示整数值，normalize 表示布尔值

    def __eq__(self, other) -> bool: ...    # 定义相等运算符重载，判断两个对象是否相等
    def __ne__(self, other) -> bool: ...    # 定义不等运算符重载，判断两个对象是否不相等
    def __hash__(self) -> int: ...          # 定义哈希运算符重载，返回对象的哈希值

    @property
    def kwds(self) -> dict: ...             # 定义属性方法 kwds，返回字典类型

    @property
    def base(self) -> BaseOffset: ...       # 定义属性方法 base，返回 BaseOffset 类型

    @overload
    def __add__(self, other: npt.NDArray[np.object_]) -> npt.NDArray[np.object_]: ...
    @overload
    def __add__(self, other: BaseOffset) -> Self: ...
    @overload
    def __add__(self, other: _DatetimeT) -> _DatetimeT: ...
    @overload
    def __add__(self, other: _TimedeltaT) -> _TimedeltaT: ...
    @overload
    def __radd__(self, other: npt.NDArray[np.object_]) -> npt.NDArray[np.object_]: ...
    @overload
    def __radd__(self, other: BaseOffset) -> Self: ...
    @overload
    def __radd__(self, other: _DatetimeT) -> _DatetimeT: ...
    @overload
    def __radd__(self, other: _TimedeltaT) -> _TimedeltaT: ...
    @overload
    def __radd__(self, other: NaTType) -> NaTType: ...
               # 定义加法运算符重载，支持多种类型的加法操作

    def __sub__(self, other: BaseOffset) -> Self: ...
               # 定义减法运算符重载，返回类型为 Self 的对象

    @overload
    def __rsub__(self, other: npt.NDArray[np.object_]) -> npt.NDArray[np.object_]: ...
    @overload
    def __rsub__(self, other: BaseOffset) -> Self: ...
    @overload
    def __rsub__(self, other: _DatetimeT) -> _DatetimeT: ...
    @overload
    def __rsub__(self, other: _TimedeltaT) -> _TimedeltaT: ...
               # 定义反向减法运算符重载，支持多种类型的反向减法操作

    @overload
    def __mul__(self, other: np.ndarray) -> np.ndarray: ...
    @overload
    def __mul__(self, other: int) -> Self: ...
    @overload
    def __rmul__(self, other: np.ndarray) -> np.ndarray: ...
    @overload
    def __rmul__(self, other: int) -> Self: ...
               # 定义乘法运算符重载，支持多种类型的乘法操作

    def __neg__(self) -> Self: ...   # 定义取负运算符重载，返回类型为 Self 的对象

    def copy(self) -> Self: ...      # 定义复制方法，返回类型为 Self 的对象

    @property
    def name(self) -> str: ...       # 定义属性方法 name，返回字符串类型

    @property
    def rule_code(self) -> str: ...  # 定义属性方法 rule_code，返回字符串类型

    @property
    def freqstr(self) -> str: ...    # 定义属性方法 freqstr，返回字符串类型

    def _apply(self, other): ...     # 定义 _apply 方法，处理应用操作

    def _apply_array(self, dtarr: np.ndarray) -> np.ndarray: ...
               # 定义 _apply_array 方法，处理应用到数组的操作

    def rollback(self, dt: datetime) -> datetime: ...
               # 定义 rollback 方法，返回回滚后的 datetime 对象

    def rollforward(self, dt: datetime) -> datetime: ...
               # 定义 rollforward 方法，返回向前推进后的 datetime 对象

    def is_on_offset(self, dt: datetime) -> bool: ...
               # 定义 is_on_offset 方法，判断给定的 datetime 对象是否在偏移量上

    def __setstate__(self, state) -> None: ...
               # 定义 __setstate__ 方法，反序列化对象状态

    def __getstate__(self): ...
               # 定义 __getstate__ 方法，获取对象状态

    @property
    def nanos(self) -> int: ...      # 定义属性方法 nanos，返回整数类型

def _get_offset(name: str) -> BaseOffset: ...
               # 定义 _get_offset 函数，根据名称获取 BaseOffset 类型的对象

class SingleConstructorOffset(BaseOffset):
    @classmethod
    ...
               # 定义 SingleConstructorOffset 类，继承自 BaseOffset 类，单构造函数偏移量类
    # 定义一个类方法 `_from_name`，这是一个类方法，通常用于根据名称创建实例
    def _from_name(cls, suffix: None = ...) -> Self:
        # 返回一个特定后缀的实例，根据参数 `suffix` 指定的后缀来创建对象
        ...
    
    # 定义一个魔术方法 `__reduce__`，用于支持对象的序列化和反序列化操作
    def __reduce__(self):
        ...
@overload
# 用于类型提示：当 freq 参数为 None 时，返回 None
def to_offset(freq: None, is_period: bool = ...) -> None: ...

@overload
# 用于类型提示：当 freq 参数为 _BaseOffsetT 类型时，返回 _BaseOffsetT 类型的对象
def to_offset(freq: _BaseOffsetT, is_period: bool = ...) -> _BaseOffsetT: ...

@overload
# 用于类型提示：当 freq 参数为 timedelta 或 str 类型时，返回 BaseOffset 类型的对象
def to_offset(freq: timedelta | str, is_period: bool = ...) -> BaseOffset: ...

# Tick 类继承自 SingleConstructorOffset 类
class Tick(SingleConstructorOffset):
    _creso: int  # 声明 _creso 属性为整数类型
    _prefix: str  # 声明 _prefix 属性为字符串类型
    def __init__(self, n: int = ..., normalize: bool = ...) -> None:
        # Tick 类的初始化方法，接受 n（整数类型，默认为省略）、normalize（布尔类型，默认为省略）两个参数
        ...

    @property
    def nanos(self) -> int:
        # 返回整数类型的属性 nanos
        ...

# delta_to_tick 函数接受一个 timedelta 类型的参数，并返回 Tick 类型的对象
def delta_to_tick(delta: timedelta) -> Tick: ...

# Day、Hour、Minute、Second、Milli、Micro、Nano 类都继承自 Tick 类
class Day(Tick): ...
class Hour(Tick): ...
class Minute(Tick): ...
class Second(Tick): ...
class Milli(Tick): ...
class Micro(Tick): ...
class Nano(Tick): ...

# RelativeDeltaOffset 类继承自 BaseOffset 类
class RelativeDeltaOffset(BaseOffset):
    def __init__(self, n: int = ..., normalize: bool = ..., **kwds: Any) -> None:
        # RelativeDeltaOffset 类的初始化方法，接受 n（整数类型，默认为省略）、normalize（布尔类型，默认为省略）、**kwds 参数
        ...

# BusinessMixin 类继承自 SingleConstructorOffset 类
class BusinessMixin(SingleConstructorOffset):
    def __init__(
        self, n: int = ..., normalize: bool = ..., offset: timedelta = ...
    ) -> None:
        # BusinessMixin 类的初始化方法，接受 n（整数类型，默认为省略）、normalize（布尔类型，默认为省略）、offset（timedelta 类型，默认为省略）三个参数
        ...

# BusinessDay 类继承自 BusinessMixin 类
class BusinessDay(BusinessMixin): ...

# BusinessHour 类继承自 BusinessMixin 类
class BusinessHour(BusinessMixin):
    def __init__(
        self,
        n: int = ...,
        normalize: bool = ...,
        start: str | time | Collection[str | time] = ...,
        end: str | time | Collection[str | time] = ...,
        offset: timedelta = ...,
    ) -> None:
        # BusinessHour 类的初始化方法，接受 n（整数类型，默认为省略）、normalize（布尔类型，默认为省略）、start（字符串或 time 对象或其集合，默认为省略）、end（字符串或 time 对象或其集合，默认为省略）、offset（timedelta 类型，默认为省略）五个参数
        ...

# WeekOfMonthMixin 类继承自 SingleConstructorOffset 类
class WeekOfMonthMixin(SingleConstructorOffset):
    def __init__(
        self, n: int = ..., normalize: bool = ..., weekday: int = ...
    ) -> None:
        # WeekOfMonthMixin 类的初始化方法，接受 n（整数类型，默认为省略）、normalize（布尔类型，默认为省略）、weekday（整数类型，默认为省略）三个参数
        ...

# YearOffset 类继承自 SingleConstructorOffset 类
class YearOffset(SingleConstructorOffset):
    def __init__(
        self, n: int = ..., normalize: bool = ..., month: int | None = ...
    ) -> None:
        # YearOffset 类的初始化方法，接受 n（整数类型，默认为省略）、normalize（布尔类型，默认为省略）、month（整数或 None 类型，默认为省略）三个参数
        ...

# BYearEnd、BYearBegin、YearEnd、YearBegin 类都继承自 YearOffset 类
class BYearEnd(YearOffset): ...
class BYearBegin(YearOffset): ...
class YearEnd(YearOffset): ...
class YearBegin(YearOffset): ...

# QuarterOffset 类继承自 SingleConstructorOffset 类
class QuarterOffset(SingleConstructorOffset):
    def __init__(
        self, n: int = ..., normalize: bool = ..., startingMonth: int | None = ...
    ) -> None:
        # QuarterOffset 类的初始化方法，接受 n（整数类型，默认为省略）、normalize（布尔类型，默认为省略）、startingMonth（整数或 None 类型，默认为省略）三个参数
        ...

# BQuarterEnd、BQuarterBegin、QuarterEnd、QuarterBegin 类都继承自 QuarterOffset 类
class BQuarterEnd(QuarterOffset): ...
class BQuarterBegin(QuarterOffset): ...
class QuarterEnd(QuarterOffset): ...
class QuarterBegin(QuarterOffset): ...

# MonthOffset 类继承自 SingleConstructorOffset 类
class MonthOffset(SingleConstructorOffset): ...

# MonthEnd 类继承自 MonthOffset 类
class MonthEnd(MonthOffset): ...

# MonthBegin 类继承自 MonthOffset 类
class MonthBegin(MonthOffset): ...

# BusinessMonthEnd 类继承自 MonthOffset 类
class BusinessMonthEnd(MonthOffset): ...

# BusinessMonthBegin 类继承自 MonthOffset 类
class BusinessMonthBegin(MonthOffset): ...

# SemiMonthOffset 类继承自 SingleConstructorOffset 类
class SemiMonthOffset(SingleConstructorOffset):
    def __init__(
        self, n: int = ..., normalize: bool = ..., day_of_month: int | None = ...
    ) -> None:
        # SemiMonthOffset 类的初始化方法，接受 n（整数类型，默认为省略）、normalize（布尔类型，默认为省略）、day_of_month（整数或 None 类型，默认为省略）三个参数
        ...

# SemiMonthEnd 类继承自 SemiMonthOffset 类
class SemiMonthEnd(SemiMonthOffset): ...

# SemiMonthBegin 类继承自 SemiMonthOffset 类
class SemiMonthBegin(SemiMonthOffset): ...

# Week 类继承自 SingleConstructorOffset 类
class Week(SingleConstructorOffset):
    def __init__(
        self, n: int = ..., normalize: bool = ..., weekday: int | None = ...
    ) -> None:
        # Week 类的初始化方法，接受 n（整数类型，默认为省略）、normalize（布尔类型，默认为省略）、weekday（整数或 None 类型，默认为省略）三个参数
        ...

# WeekOfMonth 类继承自 WeekOfMonthMixin 类
class WeekOfMonth(WeekOfMonthMixin):
    def __init__(
        self, n: int = ..., normalize: bool = ..., week: int = ..., weekday: int = ...
    ) -> None:
        # WeekOfMonth 类的初始化方法，接受 n（整数类型，默认为省略）、normalize（布尔类型，默认为省略）、week（整数，默认为省略）、weekday（整数，默认为省略）四个参数
        ...

# LastWeekOfMonth 类继承自 WeekOfMonthMixin 类
class LastWeekOfMonth(WeekOfMonthMixin):
    def __init__(
        self, n: int = ..., normalize: bool = ..., weekday: int = ...
    ) -> None:
        # LastWeekOfMonth 类的初始化方法，接受 n（整数类型，默认为省略）、normalize（布尔类型，默认为省略）、weekday（整数，默认为省略）三个参数
        ...

# FY5253Mixin 类继承自 SingleConstructorOffset 类
class FY5253Mixin(SingleConstructorOffset):
    # FY5253Mixin 类未定义初始化方法，可能在其他地方定义
    ...
    # 初始化函数，用于创建一个对象实例
    def __init__(
        self,
        n: int = ...,
        normalize: bool = ...,
        weekday: int = ...,
        startingMonth: int = ...,
        variation: Literal["nearest", "last"] = ...,
    ) -> None:
class FY5253(FY5253Mixin): ...
# 定义类 FY5253，继承自 FY5253Mixin

class FY5253Quarter(FY5253Mixin):
    def __init__(
        self,
        n: int = ...,
        normalize: bool = ...,
        weekday: int = ...,
        startingMonth: int = ...,
        qtr_with_extra_week: int = ...,
        variation: Literal["nearest", "last"] = ...,
    ) -> None: ...
    # 定义类 FY5253Quarter，继承自 FY5253Mixin，初始化方法接收多个参数

class Easter(SingleConstructorOffset): ...
# 定义类 Easter，继承自 SingleConstructorOffset

class _CustomBusinessMonth(BusinessMixin):
    def __init__(
        self,
        n: int = ...,
        normalize: bool = ...,
        weekmask: str = ...,
        holidays: list | None = ...,
        calendar: OffsetCalendar | None = ...,
        offset: timedelta = ...,
    ) -> None: ...
    # 定义类 _CustomBusinessMonth，继承自 BusinessMixin，初始化方法接收多个参数

class CustomBusinessDay(BusinessDay):
    def __init__(
        self,
        n: int = ...,
        normalize: bool = ...,
        weekmask: str = ...,
        holidays: list | None = ...,
        calendar: OffsetCalendar | None = ...,
        offset: timedelta = ...,
    ) -> None: ...
    # 定义类 CustomBusinessDay，继承自 BusinessDay，初始化方法接收多个参数

class CustomBusinessHour(BusinessHour):
    def __init__(
        self,
        n: int = ...,
        normalize: bool = ...,
        weekmask: str = ...,
        holidays: list | None = ...,
        calendar: OffsetCalendar | None = ...,
        start: str | time | Collection[str | time] = ...,
        end: str | time | Collection[str | time] = ...,
        offset: timedelta = ...,
    ) -> None: ...
    # 定义类 CustomBusinessHour，继承自 BusinessHour，初始化方法接收多个参数

class CustomBusinessMonthEnd(_CustomBusinessMonth): ...
# 定义类 CustomBusinessMonthEnd，继承自 _CustomBusinessMonth

class CustomBusinessMonthBegin(_CustomBusinessMonth): ...
# 定义类 CustomBusinessMonthBegin，继承自 _CustomBusinessMonth

class OffsetMeta(type): ...
# 定义元类 OffsetMeta

class DateOffset(RelativeDeltaOffset, metaclass=OffsetMeta): ...
# 定义类 DateOffset，继承自 RelativeDeltaOffset，使用 OffsetMeta 作为元类

BDay = BusinessDay
# BDay 是 BusinessDay 的别名

BMonthEnd = BusinessMonthEnd
# BMonthEnd 是 BusinessMonthEnd 的别名

BMonthBegin = BusinessMonthBegin
# BMonthBegin 是 BusinessMonthBegin 的别名

CBMonthEnd = CustomBusinessMonthEnd
# CBMonthEnd 是 CustomBusinessMonthEnd 的别名

CBMonthBegin = CustomBusinessMonthBegin
# CBMonthBegin 是 CustomBusinessMonthBegin 的别名

CDay = CustomBusinessDay
# CDay 是 CustomBusinessDay 的别名

def roll_qtrday(
    other: datetime, n: int, month: int, day_opt: str, modby: int
) -> int: ...
# 定义函数 roll_qtrday，接收 datetime、n、month、day_opt 和 modby 参数，返回 int

INVALID_FREQ_ERR_MSG: Literal["Invalid frequency: {0}"]
# 声明常量 INVALID_FREQ_ERR_MSG，值为文本模板字面量

def shift_months(
    dtindex: npt.NDArray[np.int64],
    months: int,
    day_opt: str | None = ...,
    reso: int = ...,
) -> npt.NDArray[np.int64]: ...
# 定义函数 shift_months，接收 dtindex、months、day_opt 和 reso 参数，返回 npt.NDArray[np.int64]

_offset_map: dict[str, BaseOffset]
# 声明变量 _offset_map，类型为字典，键为 str，值为 BaseOffset 类型的对象
```