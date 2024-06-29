# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\dtypes.pxd`

```
# 导入必要的模块和类型
from numpy cimport int64_t  # 导入 int64_t 类型定义

from pandas._libs.tslibs.np_datetime cimport NPY_DATETIMEUNIT  # 从 pandas._libs.tslibs.np_datetime 模块中导入 NPY_DATETIMEUNIT 类型


# 定义 C 语言级别的函数 npy_unit_to_abbrev，接受 NPY_DATETIMEUNIT 类型参数，返回 str 类型结果
cdef str npy_unit_to_abbrev(NPY_DATETIMEUNIT unit)
# 定义 Cython 可调用的函数 abbrev_to_npy_unit，接受 str 类型的参数 abbrev，返回 NPY_DATETIMEUNIT 类型结果
cpdef NPY_DATETIMEUNIT abbrev_to_npy_unit(str abbrev)
# 定义 C 语言级别的函数 freq_group_code_to_npy_unit，接受 int 类型的参数 freq，无异常和 GIL（全局解释器锁）保护
cdef NPY_DATETIMEUNIT freq_group_code_to_npy_unit(int freq) noexcept nogil
# 定义 Cython 可调用的函数 periods_per_day，接受 reso 参数，默认为 *，可能抛出异常
cpdef int64_t periods_per_day(NPY_DATETIMEUNIT reso=*) except? -1
# 定义 Cython 可调用的函数 periods_per_second，接受 NPY_DATETIMEUNIT 类型的参数 reso，可能抛出异常
cpdef int64_t periods_per_second(NPY_DATETIMEUNIT reso) except? -1

# 定义 C 语言级别的函数 get_supported_reso，接受 NPY_DATETIMEUNIT 类型的参数 reso
cdef NPY_DATETIMEUNIT get_supported_reso(NPY_DATETIMEUNIT reso)
# 定义 C 语言级别的函数 is_supported_unit，接受 NPY_DATETIMEUNIT 类型的参数 reso
cdef bint is_supported_unit(NPY_DATETIMEUNIT reso)


# 定义一系列全局的 C 语言级别的字典变量
cdef dict c_OFFSET_TO_PERIOD_FREQSTR
cdef dict c_PERIOD_TO_OFFSET_FREQSTR
cdef dict c_OFFSET_RENAMED_FREQSTR
cdef dict c_DEPR_ABBREVS
cdef dict c_DEPR_UNITS
cdef dict c_PERIOD_AND_OFFSET_DEPR_FREQSTR
cdef dict attrname_to_abbrevs
cdef dict npy_unit_to_attrname
cdef dict attrname_to_npy_unit

# 定义 C 语言级别的枚举类型 c_FreqGroup
cdef enum c_FreqGroup:
    # 以下是各种频率组的枚举值，用于表示不同的时间频率
    FR_ANN = 1000
    FR_QTR = 2000
    FR_MTH = 3000
    FR_WK = 4000
    FR_BUS = 5000
    FR_DAY = 6000
    FR_HR = 7000
    FR_MIN = 8000
    FR_SEC = 9000
    FR_MS = 10000
    FR_US = 11000
    FR_NS = 12000
    FR_UND = -10000  # 未定义的频率


# 定义 C 语言级别的枚举类型 c_Resolution
cdef enum c_Resolution:
    # 以下是各种时间分辨率的枚举值，用于表示不同的时间单位
    RESO_NS = 0
    RESO_US = 1
    RESO_MS = 2
    RESO_SEC = 3
    RESO_MIN = 4
    RESO_HR = 5
    RESO_DAY = 6
    RESO_MTH = 7
    RESO_QTR = 8
    RESO_YR = 9


# 定义 C 语言级别的枚举类型 PeriodDtypeCode
cdef enum PeriodDtypeCode:
    # 年度频率与各种财政年度结束的枚举值
    A = 1000      # 默认别名
    A_DEC = 1000  # 年度 - 十二月年度结束
    A_JAN = 1001  # 年度 - 一月年度结束
    A_FEB = 1002  # 年度 - 二月年度结束
    A_MAR = 1003  # 年度 - 三月年度结束
    A_APR = 1004  # 年度 - 四月年度结束
    A_MAY = 1005  # 年度 - 五月年度结束
    A_JUN = 1006  # 年度 - 六月年度结束
    A_JUL = 1007  # 年度 - 七月年度结束
    A_AUG = 1008  # 年度 - 八月年度结束
    A_SEP = 1009  # 年度 - 九月年度结束
    A_OCT = 1010  # 年度 - 十月年度结束
    A_NOV = 1011  # 年度 - 十一月年度结束

    # 季度频率与各种财政年度结束的枚举值
    Q_DEC = 2000    # 季度 - 十二月年度结束
    Q_JAN = 2001    # 季度 - 一月年度结束
    Q_FEB = 2002    # 季度 - 二月年度结束
    Q_MAR = 2003    # 季度 - 三月年度结束
    Q_APR = 2004    # 季度 - 四月年度结束
    Q_MAY = 2005    # 季度 - 五月年度结束
    Q_JUN = 2006    # 季度 - 六月年度结束
    Q_JUL = 2007    # 季度 - 七月年度结束
    Q_AUG = 2008    # 季度 - 八月年度结束
    Q_SEP = 2009    # 季度 - 九月年度结束
    Q_OCT = 2010    # 季度 - 十月年度结束
    Q_NOV = 2011    # 季度 - 十一月年度结束

    M = 3000        # 月度

    W_SUN = 4000    # 每周 - 周日结束
    W_MON = 4001    # 每周 - 周一结束
    W_TUE = 4002    # 每周 - 周二结束
    # 每周的结束时间，以星期几为单位，用于表示每周的不同结束日期
    W_WED = 4003    # Weekly - Wednesday end of week
    W_THU = 4004    # Weekly - Thursday end of week
    W_FRI = 4005    # Weekly - Friday end of week
    W_SAT = 4006    # Weekly - Saturday end of week
    
    # 工作日
    B = 5000        # Business days
    # 每日
    D = 6000        # Daily
    # 每小时
    H = 7000        # Hourly
    # 每分钟
    T = 8000        # Minutely
    # 每秒
    S = 9000        # Secondly
    # 毫秒
    L = 10000       # Millisecondly
    # 微秒
    U = 11000       # Microsecondly
    # 纳秒
    N = 12000       # Nanosecondly
    
    # 未定义的时间单位
    UNDEFINED = -10_000
# 定义一个 Cython 的 cdef 类 PeriodDtypeBase，这个类实现了一些基础的功能
cdef class PeriodDtypeBase:
    # 定义了两个只读成员变量
    cdef readonly:
        # 表示 PeriodDtype 的代码
        PeriodDtypeCode _dtype_code
        # 表示一个 int64_t 类型的整数变量 _n
        int64_t _n

    # 声明一个 Cython 公共方法（cpdef），返回一个整数
    cpdef int _get_to_timestamp_base(self)
    
    # 声明一个 Cython 公共方法（cpdef），返回一个布尔值
    cpdef bint _is_tick_like(self)
```