# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\timedeltas.pyi`

```
# 导入需要的模块和类
from datetime import timedelta
from typing import (
    ClassVar,
    Literal,
    TypeAlias,
    TypeVar,
    overload,
)

import numpy as np  # 导入NumPy库

from pandas._libs.tslibs import (
    NaTType,
    Tick,
)
from pandas._typing import (
    Frequency,
    Self,
    npt,
)

# 在pandas/_libs/tslibs/timedeltas.pyx文件中，这个类型别名应与timedelta_abbrevs字典中的键保持一致
UnitChoices: TypeAlias = Literal[
    "Y",
    "y",
    "M",
    "W",
    "w",
    "D",
    "d",
    "days",
    "day",
    "hours",
    "hour",
    "hr",
    "h",
    "m",
    "minute",
    "min",
    "minutes",
    "s",
    "seconds",
    "sec",
    "second",
    "ms",
    "milliseconds",
    "millisecond",
    "milli",
    "millis",
    "us",
    "microseconds",
    "microsecond",
    "µs",
    "micro",
    "micros",
    "ns",
    "nanoseconds",
    "nano",
    "nanos",
    "nanosecond",
]

_S = TypeVar("_S", bound=timedelta)  # 泛型变量_S，绑定类型为timedelta

# 下面是一系列函数定义，每个函数都有特定的功能，如下所示：

# 获取舍入时使用的时间单位
def get_unit_for_round(freq, creso: int) -> int: ...

# 防止使用模棱两可的时间单位
def disallow_ambiguous_unit(unit: str | None) -> None: ...

# 将m8values数组中的值转换为Python的timedelta对象数组
def ints_to_pytimedelta(
    m8values: npt.NDArray[np.timedelta64],
    box: bool = ...,
) -> npt.NDArray[np.object_]: ...

# 将数组中的值转换为timedelta64类型的NumPy数组
def array_to_timedelta64(
    values: npt.NDArray[np.object_],
    unit: str | None = ...,
    errors: str = ...,
) -> np.ndarray: ...  # 返回np.ndarray[m8ns]类型的数组

# 解析时间增量单位，返回符合UnitChoices的值
def parse_timedelta_unit(unit: str | None) -> UnitChoices: ...

# 将时间增量转换为纳秒数
def delta_to_nanoseconds(
    delta: np.timedelta64 | timedelta | Tick,
    reso: int = ...,  # NPY_DATETIMEUNIT
    round_ok: bool = ...,
) -> int: ...

# 对象数组的地板除法操作
def floordiv_object_array(
    left: np.ndarray, right: npt.NDArray[np.object_]
) -> np.ndarray: ...

# 对象数组的真除法操作
def truediv_object_array(
    left: np.ndarray, right: npt.NDArray[np.object_]
) -> np.ndarray: ...

# 继承自timedelta的子类Timedelta的定义及其属性和方法如下：

class Timedelta(timedelta):
    _creso: int  # 分辨率
    min: ClassVar[Timedelta]  # 最小时间增量
    max: ClassVar[Timedelta]  # 最大时间增量
    resolution: ClassVar[Timedelta]  # 分辨率
    value: int  # np.int64类型的值
    _value: int  # np.int64类型的值

    # 构造方法，创建一个Timedelta对象，返回类型可以是_S或NaTType类型
    def __new__(  # type: ignore[misc]
        cls: type[_S],
        value=...,
        unit: str | None = ...,
        **kwargs: float | np.integer | np.floating,
    ) -> _S | NaTType: ...

    # 根据value和reso创建Timedelta对象的类方法
    @classmethod
    def _from_value_and_reso(cls, value: np.int64, reso: int) -> Timedelta: ...

    # 获取天数属性
    @property
    def days(self) -> int: ...

    # 获取秒数属性
    @property
    def seconds(self) -> int: ...

    # 获取微秒数属性
    @property
    def microseconds(self) -> int: ...

    # 获取总秒数属性
    def total_seconds(self) -> float: ...

    # 转换为Python的timedelta对象
    def to_pytimedelta(self) -> timedelta: ...

    # 转换为timedelta64类型的NumPy对象
    def to_timedelta64(self) -> np.timedelta64: ...

    # 获取as m8属性，返回np.timedelta64类型
    @property
    def asm8(self) -> np.timedelta64: ...

    # 舍入到指定频率的方法
    def round(self, freq: Frequency) -> Self: ...

    # 向下舍入到指定频率的方法
    def floor(self, freq: Frequency) -> Self: ...

    # 向上舍入到指定频率的方法
    def ceil(self, freq: Frequency) -> Self: ...

    # 获取分辨率字符串的方法
    @property
    def resolution_string(self) -> str: ...

    # 时间增量对象与timedelta对象的加法运算
    def __add__(self, other: timedelta) -> Timedelta: ...
    # 定义魔术方法 __radd__，支持右侧操作数为 timedelta 对象的加法
    def __radd__(self, other: timedelta) -> Timedelta: ...

    # 定义魔术方法 __sub__，支持 timedelta 对象的减法
    def __sub__(self, other: timedelta) -> Timedelta: ...

    # 定义魔术方法 __rsub__，支持右侧操作数为 timedelta 对象的减法
    def __rsub__(self, other: timedelta) -> Timedelta: ...

    # 定义魔术方法 __neg__，支持 timedelta 对象的取负操作
    def __neg__(self) -> Timedelta: ...

    # 定义魔术方法 __pos__，支持 timedelta 对象的取正操作
    def __pos__(self) -> Timedelta: ...

    # 定义魔术方法 __abs__，支持 timedelta 对象的绝对值操作
    def __abs__(self) -> Timedelta: ...

    # 定义魔术方法 __mul__，支持 timedelta 对象与浮点数的乘法
    def __mul__(self, other: float) -> Timedelta: ...

    # 定义魔术方法 __rmul__，支持右侧操作数为浮点数时的乘法
    def __rmul__(self, other: float) -> Timedelta: ...

    # 对 "__floordiv__" 方法进行类型重载，支持多种操作数类型的整除操作
    @overload  # type: ignore[override]
    def __floordiv__(self, other: timedelta) -> int: ...

    @overload
    def __floordiv__(self, other: float) -> Timedelta: ...

    @overload
    def __floordiv__(
        self, other: npt.NDArray[np.timedelta64]
    ) -> npt.NDArray[np.intp]: ...

    @overload
    def __floordiv__(
        self, other: npt.NDArray[np.number]
    ) -> npt.NDArray[np.timedelta64] | Timedelta: ...

    # 定义魔术方法 __rfloordiv__，支持右侧操作数为 timedelta 或字符串的整除操作
    @overload
    def __rfloordiv__(self, other: timedelta | str) -> int: ...

    @overload
    def __rfloordiv__(self, other: None | NaTType) -> NaTType: ...

    @overload
    def __rfloordiv__(self, other: np.ndarray) -> npt.NDArray[np.timedelta64]: ...

    # 定义魔术方法 __truediv__，支持 timedelta 对象与 timedelta 或浮点数的除法
    @overload
    def __truediv__(self, other: timedelta) -> float: ...

    @overload
    def __truediv__(self, other: float) -> Timedelta: ...

    # 定义魔术方法 __mod__，支持 timedelta 对象与 timedelta 的取模操作
    def __mod__(self, other: timedelta) -> Timedelta: ...

    # 定义魔术方法 __divmod__，支持 timedelta 对象与 timedelta 的整除取余操作
    def __divmod__(self, other: timedelta) -> tuple[int, Timedelta]: ...

    # 定义魔术方法 __le__，支持 timedelta 对象的小于等于比较
    def __le__(self, other: timedelta) -> bool: ...

    # 定义魔术方法 __lt__，支持 timedelta 对象的小于比较
    def __lt__(self, other: timedelta) -> bool: ...

    # 定义魔术方法 __ge__，支持 timedelta 对象的大于等于比较
    def __ge__(self, other: timedelta) -> bool: ...

    # 定义魔术方法 __gt__，支持 timedelta 对象的大于比较
    def __gt__(self, other: timedelta) -> bool: ...

    # 定义魔术方法 __hash__，返回 timedelta 对象的哈希值
    def __hash__(self) -> int: ...

    # 返回 timedelta 对象的 ISO 8601 格式字符串表示
    def isoformat(self) -> str: ...

    # 将 timedelta 对象转换为 NumPy timedelta64 类型
    def to_numpy(
        self, dtype: npt.DTypeLike = ..., copy: bool = False
    ) -> np.timedelta64: ...

    # 返回视图，将 timedelta 对象转换为指定 dtype 的对象
    def view(self, dtype: npt.DTypeLike) -> object: ...

    # 返回单位，表示 timedelta 对象的时间单位
    @property
    def unit(self) -> str: ...

    # 将 timedelta 对象转换为指定单位的 Timedelta 对象
    def as_unit(self, unit: str, round_ok: bool = ...) -> Timedelta: ...
```