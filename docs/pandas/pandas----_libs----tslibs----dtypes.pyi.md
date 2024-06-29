# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\dtypes.pyi`

```
# 引入枚举类型 Enum
from enum import Enum

# 导入 pandas._typing 中的 Self 类型
from pandas._typing import Self

# OFFSET_TO_PERIOD_FREQSTR 是一个字典，用于存储偏移量到周期频率字符串的映射关系
OFFSET_TO_PERIOD_FREQSTR: dict[str, str]

# periods_per_day 函数用于计算每天的周期数，reso 参数表示分辨率，默认值由调用方决定
def periods_per_day(reso: int = ...) -> int: ...

# periods_per_second 函数用于计算每秒的周期数，reso 参数表示分辨率
def periods_per_second(reso: int) -> int: ...

# abbrev_to_npy_unit 函数根据缩写返回对应的 numpy 时间单位的整数表示
def abbrev_to_npy_unit(abbrev: str | None) -> int: ...

# PeriodDtypeBase 类定义
class PeriodDtypeBase:
    _dtype_code: int  # 用于存储 PeriodDtypeCode 的整数表示
    _n: int  # 用于存储整数 n

    # 实际上是 __cinit__ 方法，用于创建 PeriodDtypeBase 实例
    def __new__(cls, code: int, n: int) -> Self: ...

    # 返回频率组代码的属性
    @property
    def _freq_group_code(self) -> int: ...

    # 返回分辨率对象的属性
    @property
    def _resolution_obj(self) -> Resolution: ...

    # 获取到时间戳基础的整数表示
    def _get_to_timestamp_base(self) -> int: ...

    # 返回频率字符串的属性
    @property
    def _freqstr(self) -> str: ...

    # 返回对象的哈希值
    def __hash__(self) -> int: ...

    # 判断对象是否类似于 tick（特定时间点）
    def _is_tick_like(self) -> bool: ...

    # 返回 c 表示的属性
    @property
    def _creso(self) -> int: ...

    # 返回 numpy 时间单位的字符串表示属性
    @property
    def _td64_unit(self) -> str: ...

# FreqGroup 枚举类定义
class FreqGroup(Enum):
    FR_ANN: int  # 年度频率的枚举值
    FR_QTR: int  # 季度频率的枚举值
    FR_MTH: int  # 月度频率的枚举值
    FR_WK: int  # 周频率的枚举值
    FR_BUS: int  # 工作日频率的枚举值
    FR_DAY: int  # 每日频率的枚举值
    FR_HR: int  # 小时频率的枚举值
    FR_MIN: int  # 分钟频率的枚举值
    FR_SEC: int  # 秒钟频率的枚举值
    FR_MS: int  # 毫秒频率的枚举值
    FR_US: int  # 微秒频率的枚举值
    FR_NS: int  # 纳秒频率的枚举值
    FR_UND: int  # 未定义频率的枚举值

    # 根据 PeriodDtypeCode 返回对应的 FreqGroup 枚举值的静态方法
    @staticmethod
    def from_period_dtype_code(code: int) -> FreqGroup: ...

# Resolution 枚举类定义
class Resolution(Enum):
    RESO_NS: int  # 纳秒分辨率的枚举值
    RESO_US: int  # 微秒分辨率的枚举值
    RESO_MS: int  # 毫秒分辨率的枚举值
    RESO_SEC: int  # 秒分辨率的枚举值
    RESO_MIN: int  # 分钟分辨率的枚举值
    RESO_HR: int  # 小时分辨率的枚举值
    RESO_DAY: int  # 日分辨率的枚举值
    RESO_MTH: int  # 月分辨率的枚举值
    RESO_QTR: int  # 季度分辨率的枚举值
    RESO_YR: int  # 年分辨率的枚举值

    # 小于比较方法，用于比较两个分辨率对象
    def __lt__(self, other: Resolution) -> bool: ...

    # 大于等于比较方法，用于比较两个分辨率对象
    def __ge__(self, other: Resolution) -> bool: ...

    # 返回属性名的属性
    @property
    def attrname(self) -> str: ...

    # 根据属性名返回分辨率对象的静态方法
    @classmethod
    def from_attrname(cls, attrname: str) -> Resolution: ...

    # 根据频率字符串返回分辨率对象的静态方法
    @classmethod
    def get_reso_from_freqstr(cls, freq: str) -> Resolution: ...

    # 返回属性简称的属性
    @property
    def attr_abbrev(self) -> str: ...

# NpyDatetimeUnit 枚举类定义
class NpyDatetimeUnit(Enum):
    NPY_FR_Y: int  # 年份的 numpy 时间单位的枚举值
    NPY_FR_M: int  # 月份的 numpy 时间单位的枚举值
    NPY_FR_W: int  # 周的 numpy 时间单位的枚举值
    NPY_FR_D: int  # 天的 numpy 时间单位的枚举值
    NPY_FR_h: int  # 小时的 numpy 时间单位的枚举值
    NPY_FR_m: int  # 分钟的 numpy 时间单位的枚举值
    NPY_FR_s: int  # 秒钟的 numpy 时间单位的枚举值
    NPY_FR_ms: int  # 毫秒的 numpy 时间单位的枚举值
    NPY_FR_us: int  # 微秒的 numpy 时间单位的枚举值
    NPY_FR_ns: int  # 纳秒的 numpy 时间单位的枚举值
    NPY_FR_ps: int  # 皮秒的 numpy 时间单位的枚举值
    NPY_FR_fs: int  # 飞秒的 numpy 时间单位的枚举值
    NPY_FR_as: int  # 太秒的 numpy 时间单位的枚举值
    NPY_FR_GENERIC: int  # 通用的 numpy 时间单位的枚举值
```