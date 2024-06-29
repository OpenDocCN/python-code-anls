# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\dtypes.pyx`

```
# period frequency constants corresponding to scikits timeseries
# originals

# 导入枚举类型 Enum 用于定义常量
from enum import Enum
# 导入警告模块，用于发出警告信息
import warnings

# 导入 pandas 库中的异常处理函数 find_stack_level
from pandas.util._exceptions import find_stack_level

# 导入 Cython 中的 C 扩展模块，定义了一些与时间序列相关的常量和函数
from pandas._libs.tslibs.ccalendar cimport c_MONTH_NUMBERS
from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,  # 导入 NumPy 中的日期时间单位
    get_conversion_factor,  # 获取转换因子的函数
    import_pandas_datetime,  # 导入 pandas 的日期时间处理函数
)

# 调用 pandas 的日期时间处理函数，初始化环境
import_pandas_datetime()

# 定义 Cython 类 PeriodDtypeBase
cdef class PeriodDtypeBase:
    """
    Similar to an actual dtype, this contains all of the information
    describing a PeriodDtype in an integer code.
    """
    # 声明只读属性
    #    PeriodDtypeCode _dtype_code
    #    int64_t _n

    # 构造函数，初始化 PeriodDtypeBase 对象
    def __cinit__(self, PeriodDtypeCode code, int64_t n):
        self._dtype_code = code  # 初始化 dtype code
        self._n = n  # 初始化 n 值

    # 定义相等性运算符重载，用于比较两个 PeriodDtypeBase 对象是否相等
    def __eq__(self, other):
        if not isinstance(other, PeriodDtypeBase):
            return False
        if not isinstance(self, PeriodDtypeBase):
            # Cython 语义，这是一种逆向操作
            return False
        return self._dtype_code == other._dtype_code and self._n == other._n

    # 定义哈希函数，返回对象的哈希值
    def __hash__(self) -> int:
        return hash((self._n, self._dtype_code))

    # 定义只读属性 _freq_group_code，返回频率组代码
    @property
    def _freq_group_code(self) -> int:
        # 参见：libperiod.get_freq_group
        return (self._dtype_code // 1000) * 1000

    # 定义只读属性 _resolution_obj，返回分辨率对象
    @property
    def _resolution_obj(self) -> "Resolution":
        fgc = self._freq_group_code
        freq_group = FreqGroup(fgc)  # 创建频率组对象
        abbrev = _period_code_to_abbrev[freq_group.value].split("-")[0]  # 获取简称
        if abbrev == "B":
            return Resolution.RESO_DAY  # 如果简称为 B，返回 RESO_DAY
        attrname = _abbrev_to_attrnames[abbrev]  # 根据简称获取属性名
        return Resolution.from_attrname(attrname)  # 根据属性名获取分辨率对象

    # 定义只读属性 _freqstr，返回频率字符串
    @property
    def _freqstr(self) -> str:
        # 将周期代码转换为简称，将会传递给 Period._maybe_convert_freq 中的 to_offset 方法
        out = _period_code_to_abbrev.get(self._dtype_code)
        if self._n == 1:
            return out
        return str(self._n) + out

    # 定义公共方法 _get_to_timestamp_base，返回用于 to_timestamp 基础的频率代码组
    cpdef int _get_to_timestamp_base(self):
        """
        Return frequency code group used for base of to_timestamp against
        frequency code.

        Return day freq code against longer freq than day.
        Return second freq code against hour between second.

        Returns
        -------
        int
        """
        base = <c_FreqGroup>self._dtype_code
        if base < FR_BUS:
            return FR_DAY
        elif FR_HR <= base <= FR_SEC:
            return FR_SEC
        return base

    # 定义公共方法 _is_tick_like，检查是否类似于 tick
    cpdef bint _is_tick_like(self):
        return self._dtype_code >= PeriodDtypeCode.D

    @property
    # 返回一个字典，根据时间周期类型代码映射到对应的NumPy日期时间单位
    def _creso(self) -> int:
        return {
            PeriodDtypeCode.D: NPY_DATETIMEUNIT.NPY_FR_D,  # 天
            PeriodDtypeCode.H: NPY_DATETIMEUNIT.NPY_FR_h,  # 小时
            PeriodDtypeCode.T: NPY_DATETIMEUNIT.NPY_FR_m,  # 分钟
            PeriodDtypeCode.S: NPY_DATETIMEUNIT.NPY_FR_s,  # 秒
            PeriodDtypeCode.L: NPY_DATETIMEUNIT.NPY_FR_ms,  # 毫秒
            PeriodDtypeCode.U: NPY_DATETIMEUNIT.NPY_FR_us,  # 微秒
            PeriodDtypeCode.N: NPY_DATETIMEUNIT.NPY_FR_ns,  # 纳秒
        }[self._dtype_code]  # 根据对象的时间周期类型代码获取对应的NumPy日期时间单位

    @property
    # 返回与当前时间周期相关的NumPy日期时间单位的缩写字符串
    def _td64_unit(self) -> str:
        return npy_unit_to_abbrev(self._creso)
# 定义一个映射表，将时间频率字符串映射到相应的 PeriodDtypeCode 枚举值

_period_code_map = {
    # 年度频率，以不同的财政年度结束日期为例。
    # 例如，"Y-DEC" 表示以12月为年度结束，对应 PeriodDtypeCode.A_DEC
    "Y-DEC": PeriodDtypeCode.A_DEC,
    "Y-JAN": PeriodDtypeCode.A_JAN,
    "Y-FEB": PeriodDtypeCode.A_FEB,
    "Y-MAR": PeriodDtypeCode.A_MAR,
    "Y-APR": PeriodDtypeCode.A_APR,
    "Y-MAY": PeriodDtypeCode.A_MAY,
    "Y-JUN": PeriodDtypeCode.A_JUN,
    "Y-JUL": PeriodDtypeCode.A_JUL,
    "Y-AUG": PeriodDtypeCode.A_AUG,
    "Y-SEP": PeriodDtypeCode.A_SEP,
    "Y-OCT": PeriodDtypeCode.A_OCT,
    "Y-NOV": PeriodDtypeCode.A_NOV,

    # 季度频率，以不同的财政年度结束日期为例。
    # 例如，"Q-DEC" 表示以12月为季度结束，对应 PeriodDtypeCode.Q_DEC
    "Q-DEC": PeriodDtypeCode.Q_DEC,
    "Q-JAN": PeriodDtypeCode.Q_JAN,
    "Q-FEB": PeriodDtypeCode.Q_FEB,
    "Q-MAR": PeriodDtypeCode.Q_MAR,
    "Q-APR": PeriodDtypeCode.Q_APR,
    "Q-MAY": PeriodDtypeCode.Q_MAY,
    "Q-JUN": PeriodDtypeCode.Q_JUN,
    "Q-JUL": PeriodDtypeCode.Q_JUL,
    "Q-AUG": PeriodDtypeCode.Q_AUG,
    "Q-SEP": PeriodDtypeCode.Q_SEP,
    "Q-OCT": PeriodDtypeCode.Q_OCT,
    "Q-NOV": PeriodDtypeCode.Q_NOV,

    # 月度频率
    "M": PeriodDtypeCode.M,

    # 每周频率，以周的不同结束日期为例。
    "W-SUN": PeriodDtypeCode.W_SUN,
    "W-MON": PeriodDtypeCode.W_MON,
    "W-TUE": PeriodDtypeCode.W_TUE,
    "W-WED": PeriodDtypeCode.W_WED,
    "W-THU": PeriodDtypeCode.W_THU,
    "W-FRI": PeriodDtypeCode.W_FRI,
    "W-SAT": PeriodDtypeCode.W_SAT,

    # 工作日
    "B": PeriodDtypeCode.B,

    # 每日
    "D": PeriodDtypeCode.D,

    # 每小时
    "h": PeriodDtypeCode.H,

    # 每分钟
    "min": PeriodDtypeCode.T,

    # 每秒
    "s": PeriodDtypeCode.S,

    # 每毫秒
    "ms": PeriodDtypeCode.L,

    # 每微秒
    "us": PeriodDtypeCode.U,

    # 每纳秒
    "ns": PeriodDtypeCode.N,
}

# 创建一个从 _period_code_map 中每个键映射到其相应字符串缩写的字典
cdef dict _period_code_to_abbrev = {
    _period_code_map[key]: key for key in _period_code_map
}

# 创建一个集合，包含所有月份名称的缩写形式
cdef set _month_names = set(c_MONTH_NUMBERS.keys())

# 将属性名解析映射到其缩写形式的映射表
cdef dict attrname_to_abbrevs = {
    "year": "Y",                # 将年份映射到简称 "Y"
    "quarter": "Q",             # 将季度映射到简称 "Q"
    "month": "M",               # 将月份映射到简称 "M"
    "day": "D",                 # 将日期映射到简称 "D"
    "hour": "h",                # 将小时映射到简称 "h"
    "minute": "min",            # 将分钟映射到简称 "min"
    "second": "s",              # 将秒数映射到简称 "s"
    "millisecond": "ms",        # 将毫秒映射到简称 "ms"
    "microsecond": "us",        # 将微秒映射到简称 "us"
    "nanosecond": "ns",         # 将纳秒映射到简称 "ns"
}

cdef dict _abbrev_to_attrnames = {v: k for k, v in attrname_to_abbrevs.items()}
# 创建一个从简称到属性名的字典，是 attrname_to_abbrevs 的反向映射

OFFSET_TO_PERIOD_FREQSTR: dict = {
    "WEEKDAY": "D",             # 将工作日偏移映射到频率字符串 "D"
    "EOM": "M",                 # 将月末偏移映射到频率字符串 "M"
    "BME": "M",                 # 将工作日月末偏移映射到频率字符串 "M"
    "SME": "M",                 # 将半月末偏移映射到频率字符串 "M"
    "BMS": "M",                 # 将工作日月初偏移映射到频率字符串 "M"
    "CBME": "M",                # 将自定义工作日月末偏移映射到频率字符串 "M"
    "CBMS": "M",                # 将自定义工作日月初偏移映射到频率字符串 "M"
    "SMS": "M",                 # 将半月初偏移映射到频率字符串 "M"
    "BQS": "Q",                 # 将工作日季末偏移映射到频率字符串 "Q"
    "QS": "Q",                  # 将季末偏移映射到频率字符串 "Q"
    "BQE": "Q",                 # 将工作日季初偏移映射到频率字符串 "Q"
    "BQE-DEC": "Q-DEC",         # 将工作日12月季初偏移映射到频率字符串 "Q-DEC"
    "BQE-JAN": "Q-JAN",         # 将工作日1月季初偏移映射到频率字符串 "Q-JAN"
    "BQE-FEB": "Q-FEB",         # 将工作日2月季初偏移映射到频率字符串 "Q-FEB"
    "BQE-MAR": "Q-MAR",         # 将工作日3月季初偏移映射到频率字符串 "Q-MAR"
    "BQE-APR": "Q-APR",         # 将工作日4月季初偏移映射到频率字符串 "Q-APR"
    "BQE-MAY": "Q-MAY",         # 将工作日5月季初偏移映射到频率字符串 "Q-MAY"
    "BQE-JUN": "Q-JUN",         # 将工作日6月季初偏移映射到频率字符串 "Q-JUN"
    "BQE-JUL": "Q-JUL",         # 将工作日7月季初偏移映射到频率字符串 "Q-JUL"
    "BQE-AUG": "Q-AUG",         # 将工作日8月季初偏移映射到频率字符串 "Q-AUG"
    "BQE-SEP": "Q-SEP",         # 将工作日9月季初偏移映射到频率字符串 "Q-SEP"
    "BQE-OCT": "Q-OCT",         # 将工作日10月季初偏移映射到频率字符串 "Q-OCT"
    "BQE-NOV": "Q-NOV",         # 将工作日11月季初偏移映射到频率字符串 "Q-NOV"
    "MS": "M",                  # 将月初偏移映射到频率字符串 "M"
    "D": "D",                   # 将日偏移映射到频率字符串 "D"
    "B": "B",                   # 将工作日偏移映射到频率字符串 "B"
    "min": "min",               # 将分钟偏移映射到频率字符串 "min"
    "s": "s",                   # 将秒偏移映射到频率字符串 "s"
    "ms": "ms",                 # 将毫秒偏移映射到频率字符串 "ms"
    "us": "us",                 # 将微秒偏移映射到频率字符串 "us"
    "ns": "ns",                 # 将纳秒偏移映射到频率字符串 "ns"
    "h": "h",                   # 将小时偏移映射到频率字符串 "h"
    "QE": "Q",                  # 将季初偏移映射到频率字符串 "Q"
    "QE-DEC": "Q-DEC",          # 将12月季初偏移映射到频率字符串 "Q-DEC"
    "QE-JAN": "Q-JAN",          # 将1月季初偏移映射到频率字符串 "Q-JAN"
    "QE-FEB": "Q-FEB",          # 将2月季初偏移映射到频率字符串 "Q-FEB"
    "QE-MAR": "Q-MAR",          # 将3月季初偏移映射到频率字符串 "Q-MAR"
    "QE-APR": "Q-APR",          # 将4月季初偏移映射到频率字符串 "Q-APR"
    "QE-MAY": "Q-MAY",          # 将5月季初偏移映射到频率字符串 "Q-MAY"
    "QE-JUN": "Q-JUN",          # 将6月季初偏移映射到频率字符串 "Q-JUN"
    "QE-JUL": "Q-JUL",          # 将7月季初偏移映射到频率字符串 "Q-JUL"
    "QE-AUG": "Q-AUG",          # 将8月季初偏移映射到频率字符串 "Q-AUG"
    "QE-SEP": "Q-SEP",          # 将9月季初偏移映射到频率字符串 "Q-SEP"
    "QE-OCT": "Q-OCT",          # 将10月季初偏移映射到频率字符串 "Q-OCT"
    "QE-NOV": "Q-NOV",          # 将11月季初偏移映射到频率字符串 "Q-NOV"
    "YE": "Y",                  # 将年初偏移映射到频率字符串 "Y"
    "YE-DEC": "Y-DEC",          # 将12月年初偏移映射到频率字符串 "Y-DEC"
    "YE-JAN": "Y-JAN",          # 将1月年初偏移映射到频率字符串 "Y-JAN"
    "YE-FEB": "Y-FEB",          # 将2月年初偏移映射到频率字符串 "Y-FEB"
    "YE-MAR": "Y-MAR",          # 将3月年初偏移映射到频率字符串 "Y-MAR"
    "YE-APR": "Y-APR",          # 将4月年初偏移映射到频
    # 定义一个字典，将旧月份缩写映射到新月份缩写
    {
        "BQ-FEB": "BQE-FEB",  # 将旧月份缩写 "BQ-FEB" 映射为新月份缩写 "BQE-FEB"
        "BQ-MAR": "BQE-MAR",  # 将旧月份缩写 "BQ-MAR" 映射为新月份缩写 "BQE-MAR"
        "BQ-APR": "BQE-APR",  # 将旧月份缩写 "BQ-APR" 映射为新月份缩写 "BQE-APR"
        "BQ-MAY": "BQE-MAY",  # 将旧月份缩写 "BQ-MAY" 映射为新月份缩写 "BQE-MAY"
        "BQ-JUN": "BQE-JUN",  # 将旧月份缩写 "BQ-JUN" 映射为新月份缩写 "BQE-JUN"
        "BQ-JUL": "BQE-JUL",  # 将旧月份缩写 "BQ-JUL" 映射为新月份缩写 "BQE-JUL"
        "BQ-AUG": "BQE-AUG",  # 将旧月份缩写 "BQ-AUG" 映射为新月份缩写 "BQE-AUG"
        "BQ-SEP": "BQE-SEP",  # 将旧月份缩写 "BQ-SEP" 映射为新月份缩写 "BQE-SEP"
        "BQ-OCT": "BQE-OCT",  # 将旧月份缩写 "BQ-OCT" 映射为新月份缩写 "BQE-OCT"
        "BQ-NOV": "BQE-NOV",  # 将旧月份缩写 "BQ-NOV" 映射为新月份缩写 "BQE-NOV"
    }
}

# 将周期到偏移频率字符串的映射关系定义为常量字典
PERIOD_TO_OFFSET_FREQSTR = {
    "M": "ME",
    "Q": "QE",
    "Q-DEC": "QE-DEC",
    "Q-JAN": "QE-JAN",
    "Q-FEB": "QE-FEB",
    "Q-MAR": "QE-MAR",
    "Q-APR": "QE-APR",
    "Q-MAY": "QE-MAY",
    "Q-JUN": "QE-JUN",
    "Q-JUL": "QE-JUL",
    "Q-AUG": "QE-AUG",
    "Q-SEP": "QE-SEP",
    "Q-OCT": "QE-OCT",
    "Q-NOV": "QE-NOV",
    "Y": "YE",
    "Y-DEC": "YE-DEC",
    "Y-JAN": "YE-JAN",
    "Y-FEB": "YE-FEB",
    "Y-MAR": "YE-MAR",
    "Y-APR": "YE-APR",
    "Y-MAY": "YE-MAY",
    "Y-JUN": "YE-JUN",
    "Y-JUL": "YE-JUL",
    "Y-AUG": "YE-AUG",
    "Y-SEP": "YE-SEP",
    "Y-OCT": "YE-OCT",
    "Y-NOV": "YE-NOV",
}

# 将偏移到周期频率字符串的映射关系定义为常量字典
cdef dict c_OFFSET_TO_PERIOD_FREQSTR = OFFSET_TO_PERIOD_FREQSTR

# 将过时的分辨率缩写映射到正确的分辨率缩写的常量字典
cdef dict c_DEPR_ABBREVS = {
    "H": "h",
    "BH": "bh",
    "CBH": "cbh",
    "S": "s",
}

# 将过时的时间单位映射到正确的时间单位的常量字典
cdef dict c_DEPR_UNITS = {
    "w": "W",
    "d": "D",
    "H": "h",
    "MIN": "min",
    "S": "s",
    "MS": "ms",
    "US": "us",
    "NS": "ns",
}

# 将过时的周期和偏移频率字符串映射到正确的频率字符串的常量字典
cdef dict c_PERIOD_AND_OFFSET_DEPR_FREQSTR = {
    "w": "W",
    "MIN": "min",
}

class FreqGroup(Enum):
    # Mirrors c_FreqGroup in the .pxd file
    FR_ANN = c_FreqGroup.FR_ANN
    FR_QTR = c_FreqGroup.FR_QTR
    FR_MTH = c_FreqGroup.FR_MTH
    FR_WK = c_FreqGroup.FR_WK
    FR_BUS = c_FreqGroup.FR_BUS
    FR_DAY = c_FreqGroup.FR_DAY
    FR_HR = c_FreqGroup.FR_HR
    FR_MIN = c_FreqGroup.FR_MIN
    FR_SEC = c_FreqGroup.FR_SEC
    FR_MS = c_FreqGroup.FR_MS
    FR_US = c_FreqGroup.FR_US
    FR_NS = c_FreqGroup.FR_NS
    FR_UND = c_FreqGroup.FR_UND  # undefined

    @staticmethod
    def from_period_dtype_code(code: int) -> "FreqGroup":
        # See also: PeriodDtypeBase._freq_group_code
        code = (code // 1000) * 1000
        return FreqGroup(code)

class Resolution(Enum):
    RESO_NS = c_Resolution.RESO_NS
    RESO_US = c_Resolution.RESO_US
    RESO_MS = c_Resolution.RESO_MS
    RESO_SEC = c_Resolution.RESO_SEC
    RESO_MIN = c_Resolution.RESO_MIN
    RESO_HR = c_Resolution.RESO_HR
    RESO_DAY = c_Resolution.RESO_DAY
    RESO_MTH = c_Resolution.RESO_MTH
    RESO_QTR = c_Resolution.RESO_QTR
    RESO_YR = c_Resolution.RESO_YR

    # 定义小于运算符，用于分辨率的比较
    def __lt__(self, other):
        return self.value < other.value

    # 定义大于等于运算符，用于分辨率的比较
    def __ge__(self, other):
        return self.value >= other.value

    # 获取分辨率属性的简称，用于传递给 to_offset 方法
    @property
    def attr_abbrev(self) -> str:
        return attrname_to_abbrevs[self.attrname]

    # 获取与此分辨率对应的日期时间属性名
    @property
    def attrname(self) -> str:
        """
        Return datetime attribute name corresponding to this Resolution.

        Examples
        --------
        >>> Resolution.RESO_SEC.attrname
        'second'
        """
        return _reso_str_map[self.value]

    @classmethod
    def from_attrname(cls, attrname: str) -> "Resolution":
        """
        根据属性名获取分辨率对象。

        参数
        ----
        attrname : str
            属性名字符串，用于获取对应的分辨率代码。

        返回
        ----
        Resolution
            分辨率对象对应的枚举值。

        示例
        --------
        >>> Resolution.from_attrname('second')
        <Resolution.RESO_SEC: 3>

        >>> Resolution.from_attrname('second') == Resolution.RESO_SEC
        True
        """
        return cls(_str_reso_map[attrname])

    @classmethod
    def get_reso_from_freqstr(cls, freq: str) -> "Resolution":
        """
        根据频率字符串获取分辨率代码。

        参数
        ----
        freq : str
            频率字符串，通常由某个 DateOffset 对象的 freqstr 给出。

        返回
        ----
        Resolution
            频率对应的分辨率代码。

        异常
        ----
        KeyError
            如果频率字符串无法在 _abbrev_to_attrnames 中找到对应的属性名，则引发此异常。

        示例
        --------
        >>> Resolution.get_reso_from_freqstr('h')
        <Resolution.RESO_HR: 5>

        >>> Resolution.get_reso_from_freqstr('h') == Resolution.RESO_HR
        True
        """
        cdef:
            str abbrev
        try:
            if freq in c_DEPR_ABBREVS:
                abbrev = c_DEPR_ABBREVS[freq]
                warnings.warn(
                    f"\'{freq}\' is deprecated and will be removed in a future "
                    f"version. Please use \'{abbrev}\' "
                    f"instead of \'{freq}\'.",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
                freq = abbrev
            attr_name = _abbrev_to_attrnames[freq]
        except KeyError:
            # 对于季度和年度分辨率，我们需要去掉一个月份字符串。
            split_freq = freq.split("-")
            if len(split_freq) != 2:
                raise
            if split_freq[1] not in _month_names:
                # 例如，我们需要 "Q-DEC"，而不是 "Q-INVALID"。
                raise
            if split_freq[0] in c_DEPR_ABBREVS:
                abbrev = c_DEPR_ABBREVS[split_freq[0]]
                warnings.warn(
                    f"\'{split_freq[0]}\' is deprecated and will be removed in a "
                    f"future version. Please use \'{abbrev}\' "
                    f"instead of \'{split_freq[0]}\'.",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
                split_freq[0] = abbrev
            attr_name = _abbrev_to_attrnames[split_freq[0]]

        return cls.from_attrname(attr_name)
class NpyDatetimeUnit(Enum):
    """
    Python-space analogue to NPY_DATETIMEUNIT.
    """
    # Define enum members corresponding to NPY_DATETIMEUNIT constants
    NPY_FR_Y = NPY_DATETIMEUNIT.NPY_FR_Y  # Year
    NPY_FR_M = NPY_DATETIMEUNIT.NPY_FR_M  # Month
    NPY_FR_W = NPY_DATETIMEUNIT.NPY_FR_W  # Week
    NPY_FR_D = NPY_DATETIMEUNIT.NPY_FR_D  # Day
    NPY_FR_h = NPY_DATETIMEUNIT.NPY_FR_h  # Hour
    NPY_FR_m = NPY_DATETIMEUNIT.NPY_FR_m  # Minute
    NPY_FR_s = NPY_DATETIMEUNIT.NPY_FR_s  # Second
    NPY_FR_ms = NPY_DATETIMEUNIT.NPY_FR_ms  # Millisecond
    NPY_FR_us = NPY_DATETIMEUNIT.NPY_FR_us  # Microsecond
    NPY_FR_ns = NPY_DATETIMEUNIT.NPY_FR_ns  # Nanosecond
    NPY_FR_ps = NPY_DATETIMEUNIT.NPY_FR_ps  # Picosecond
    NPY_FR_fs = NPY_DATETIMEUNIT.NPY_FR_fs  # Femtosecond
    NPY_FR_as = NPY_DATETIMEUNIT.NPY_FR_as  # Attosecond
    NPY_FR_GENERIC = NPY_DATETIMEUNIT.NPY_FR_GENERIC  # Generic datetime unit


cdef NPY_DATETIMEUNIT get_supported_reso(NPY_DATETIMEUNIT reso):
    """
    Determine the nearest supported resolution if the input resolution is unsupported.

    Args:
        reso: Input datetime resolution unit.

    Returns:
        NPY_DATETIMEUNIT: Nearest supported datetime resolution.
    """
    # If the resolution is generic, return nanoseconds
    if reso == NPY_DATETIMEUNIT.NPY_FR_GENERIC:
        return NPY_DATETIMEUNIT.NPY_FR_ns
    # If resolution is less than seconds, return seconds
    if reso < NPY_DATETIMEUNIT.NPY_FR_s:
        return NPY_DATETIMEUNIT.NPY_FR_s
    # If resolution is greater than nanoseconds, return nanoseconds
    elif reso > NPY_DATETIMEUNIT.NPY_FR_ns:
        return NPY_DATETIMEUNIT.NPY_FR_ns
    return reso


cdef bint is_supported_unit(NPY_DATETIMEUNIT reso):
    """
    Check if the given datetime unit is supported.

    Args:
        reso: Input datetime unit.

    Returns:
        bool: True if supported, False otherwise.
    """
    return (
        reso == NPY_DATETIMEUNIT.NPY_FR_ns
        or reso == NPY_DATETIMEUNIT.NPY_FR_us
        or reso == NPY_DATETIMEUNIT.NPY_FR_ms
        or reso == NPY_DATETIMEUNIT.NPY_FR_s
    )


cdef str npy_unit_to_abbrev(NPY_DATETIMEUNIT unit):
    """
    Convert NPY_DATETIMEUNIT to its corresponding abbreviation.

    Args:
        unit: Input NPY_DATETIMEUNIT.

    Returns:
        str: Corresponding abbreviation.
    
    Raises:
        NotImplementedError: If the input unit is not recognized.
    """
    if unit == NPY_DATETIMEUNIT.NPY_FR_ns or unit == NPY_DATETIMEUNIT.NPY_FR_GENERIC:
        return "ns"  # Nanoseconds
    elif unit == NPY_DATETIMEUNIT.NPY_FR_us:
        return "us"  # Microseconds
    elif unit == NPY_DATETIMEUNIT.NPY_FR_ms:
        return "ms"  # Milliseconds
    elif unit == NPY_DATETIMEUNIT.NPY_FR_s:
        return "s"   # Seconds
    elif unit == NPY_DATETIMEUNIT.NPY_FR_m:
        return "m"   # Minutes
    elif unit == NPY_DATETIMEUNIT.NPY_FR_h:
        return "h"   # Hours
    elif unit == NPY_DATETIMEUNIT.NPY_FR_D:
        return "D"   # Days
    elif unit == NPY_DATETIMEUNIT.NPY_FR_W:
        return "W"   # Weeks
    elif unit == NPY_DATETIMEUNIT.NPY_FR_M:
        return "M"   # Months
    elif unit == NPY_DATETIMEUNIT.NPY_FR_Y:
        return "Y"   # Years
    # Handle not-really-supported units
    elif unit == NPY_DATETIMEUNIT.NPY_FR_ps:
        return "ps"  # Picoseconds
    elif unit == NPY_DATETIMEUNIT.NPY_FR_fs:
        return "fs"  # Femtoseconds
    elif unit == NPY_DATETIMEUNIT.NPY_FR_as:
        return "as"  # Attoseconds
    else:
        raise NotImplementedError(unit)


cpdef NPY_DATETIMEUNIT abbrev_to_npy_unit(str abbrev):
    """
    Convert abbreviation to corresponding NPY_DATETIMEUNIT.

    Args:
        abbrev: Input abbreviation.

    Returns:
        NPY_DATETIMEUNIT: Corresponding NPY_DATETIMEUNIT.
    """
    if abbrev == "Y":
        return NPY_DATETIMEUNIT.NPY_FR_Y  # Year
    elif abbrev == "M":
        return NPY_DATETIMEUNIT.NPY_FR_M  # Month
    elif abbrev == "W":
        return NPY_DATETIMEUNIT.NPY_FR_W  # Week
    elif abbrev == "D" or abbrev == "d":
        return NPY_DATETIMEUNIT.NPY_FR_D  # Day
    elif abbrev == "h":
        # 如果单位缩写是 'h'，返回小时单位对应的 NPY_DATETIMEUNIT 枚举值
        return NPY_DATETIMEUNIT.NPY_FR_h
    elif abbrev == "m":
        # 如果单位缩写是 'm'，返回分钟单位对应的 NPY_DATETIMEUNIT 枚举值
        return NPY_DATETIMEUNIT.NPY_FR_m
    elif abbrev == "s":
        # 如果单位缩写是 's'，返回秒单位对应的 NPY_DATETIMEUNIT 枚举值
        return NPY_DATETIMEUNIT.NPY_FR_s
    elif abbrev == "ms":
        # 如果单位缩写是 'ms'，返回毫秒单位对应的 NPY_DATETIMEUNIT 枚举值
        return NPY_DATETIMEUNIT.NPY_FR_ms
    elif abbrev == "us":
        # 如果单位缩写是 'us'，返回微秒单位对应的 NPY_DATETIMEUNIT 枚举值
        return NPY_DATETIMEUNIT.NPY_FR_us
    elif abbrev == "ns":
        # 如果单位缩写是 'ns'，返回纳秒单位对应的 NPY_DATETIMEUNIT 枚举值
        return NPY_DATETIMEUNIT.NPY_FR_ns
    elif abbrev == "ps":
        # 如果单位缩写是 'ps'，返回皮秒单位对应的 NPY_DATETIMEUNIT 枚举值
        return NPY_DATETIMEUNIT.NPY_FR_ps
    elif abbrev == "fs":
        # 如果单位缩写是 'fs'，返回飞秒单位对应的 NPY_DATETIMEUNIT 枚举值
        return NPY_DATETIMEUNIT.NPY_FR_fs
    elif abbrev == "as":
        # 如果单位缩写是 'as'，返回太秒单位对应的 NPY_DATETIMEUNIT 枚举值
        return NPY_DATETIMEUNIT.NPY_FR_as
    elif abbrev is None:
        # 如果单位缩写是 None，返回通用时间单位对应的 NPY_DATETIMEUNIT 枚举值
        return NPY_DATETIMEUNIT.NPY_FR_GENERIC
    else:
        # 如果单位缩写既不是已知的时间单位也不是 None，则抛出值错误异常
        raise ValueError(f"Unrecognized unit {abbrev}")
# 根据频率代码将其转换为对应的 NPY_DATETIMEUNIT 枚举，以便传递给 npy_datetimestruct_to_datetime 函数
cdef NPY_DATETIMEUNIT freq_group_code_to_npy_unit(int freq) noexcept nogil:
    if freq == FR_MTH:
        return NPY_DATETIMEUNIT.NPY_FR_M   # 如果频率是月，则返回月份的 NPY_DATETIMEUNIT
    elif freq == FR_DAY:
        return NPY_DATETIMEUNIT.NPY_FR_D   # 如果频率是天，则返回天数的 NPY_DATETIMEUNIT
    elif freq == FR_HR:
        return NPY_DATETIMEUNIT.NPY_FR_h   # 如果频率是小时，则返回小时的 NPY_DATETIMEUNIT
    elif freq == FR_MIN:
        return NPY_DATETIMEUNIT.NPY_FR_m   # 如果频率是分钟，则返回分钟的 NPY_DATETIMEUNIT
    elif freq == FR_SEC:
        return NPY_DATETIMEUNIT.NPY_FR_s   # 如果频率是秒，则返回秒的 NPY_DATETIMEUNIT
    elif freq == FR_MS:
        return NPY_DATETIMEUNIT.NPY_FR_ms  # 如果频率是毫秒，则返回毫秒的 NPY_DATETIMEUNIT
    elif freq == FR_US:
        return NPY_DATETIMEUNIT.NPY_FR_us  # 如果频率是微秒，则返回微秒的 NPY_DATETIMEUNIT
    elif freq == FR_NS:
        return NPY_DATETIMEUNIT.NPY_FR_ns  # 如果频率是纳秒，则返回纳秒的 NPY_DATETIMEUNIT
    elif freq == FR_UND:
        return NPY_DATETIMEUNIT.NPY_FR_D   # 默认情况下，返回天数的 NPY_DATETIMEUNIT

# 计算给定时间单位 reso 下，每天有多少个时间段
cpdef int64_t periods_per_day(
    NPY_DATETIMEUNIT reso=NPY_DATETIMEUNIT.NPY_FR_ns
) except? -1:
    """
    计算给定时间单位 reso 下，每天有多少个时间段
    """
    return get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_D, reso)

# 计算给定时间单位 reso 下，每秒有多少个时间段
cpdef int64_t periods_per_second(NPY_DATETIMEUNIT reso) except? -1:
    return get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_s, reso)

# 将字符串表示的时间分辨率映射到其对应的 Resolution 枚举值
cdef dict _reso_str_map = {
    Resolution.RESO_NS.value: "nanosecond",   # 纳秒分辨率映射
    Resolution.RESO_US.value: "microsecond",  # 微秒分辨率映射
    Resolution.RESO_MS.value: "millisecond",  # 毫秒分辨率映射
    Resolution.RESO_SEC.value: "second",       # 秒分辨率映射
    Resolution.RESO_MIN.value: "minute",      # 分钟分辨率映射
    Resolution.RESO_HR.value: "hour",         # 小时分辨率映射
    Resolution.RESO_DAY.value: "day",         # 天分辨率映射
    Resolution.RESO_MTH.value: "month",       # 月份分辨率映射
    Resolution.RESO_QTR.value: "quarter",     # 季度分辨率映射
    Resolution.RESO_YR.value: "year",         # 年份分辨率映射
}

# 将字符串表示的时间分辨率映射到其对应的 Resolution 枚举值
cdef dict _str_reso_map = {v: k for k, v in _reso_str_map.items()}

# 将 NPY_DATETIMEUNIT 枚举值映射到其对应的属性名字符串
cdef dict npy_unit_to_attrname = {
    NPY_DATETIMEUNIT.NPY_FR_Y: "year",          # 年份的属性名映射
    NPY_DATETIMEUNIT.NPY_FR_M: "month",         # 月份的属性名映射
    NPY_DATETIMEUNIT.NPY_FR_D: "day",           # 天数的属性名映射
    NPY_DATETIMEUNIT.NPY_FR_h: "hour",          # 小时的属性名映射
    NPY_DATETIMEUNIT.NPY_FR_m: "minute",        # 分钟的属性名映射
    NPY_DATETIMEUNIT.NPY_FR_s: "second",        # 秒数的属性名映射
    NPY_DATETIMEUNIT.NPY_FR_ms: "millisecond",  # 毫秒的属性名映射
    NPY_DATETIMEUNIT.NPY_FR_us: "microsecond",  # 微秒的属性名映射
    NPY_DATETIMEUNIT.NPY_FR_ns: "nanosecond",   # 纳秒的属性名映射
}

# 将属性名字符串映射到其对应的 NPY_DATETIMEUNIT 枚举值
cdef dict attrname_to_npy_unit = {v: k for k, v in npy_unit_to_attrname.items()}
```