# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\tzconversion.pxd`

```
# 导入特定模块和函数 cimport 以进行优化的 Cython 扩展
from cpython.datetime cimport tzinfo
from numpy cimport (
    int64_t,          # 导入 int64_t 类型
    intp_t,           # 导入 intp_t 类型
    ndarray,          # 导入 ndarray 类型
)

from pandas._libs.tslibs.np_datetime cimport NPY_DATETIMEUNIT  # 导入 NPY_DATETIMEUNIT 常量


# 定义一个 Cython 的函数，将 UTC 时间转换为特定时区的本地时间
cpdef int64_t tz_convert_from_utc_single(
    int64_t utc_val,       # UTC 时间的整数值
    tzinfo tz,             # 表示时区的 tzinfo 对象
    NPY_DATETIMEUNIT creso=*  # 可选参数，表示时间分辨率
) except? -1
    # 函数体未提供，未能提供详细注释


# 定义一个 Cython 的函数，将本地时间标准化为 UTC 时间
cdef int64_t tz_localize_to_utc_single(
    int64_t val,           # 本地时间的整数值
    tzinfo tz,             # 表示时区的 tzinfo 对象
    object ambiguous=*,    # 可选参数，处理时间模糊性的策略
    object nonexistent=*,  # 可选参数，处理不存在时间的策略
    NPY_DATETIMEUNIT creso=*  # 可选参数，表示时间分辨率
) except? -1
    # 函数体未提供，未能提供详细注释


# 定义一个 Cython 的类 Localizer
cdef class Localizer:
    cdef:
        tzinfo tz                  # 表示时区的 tzinfo 对象
        NPY_DATETIMEUNIT _creso    # 时间的分辨率单位
        bint use_utc, use_fixed, use_tzlocal, use_dst, use_pytz  # 布尔标志位，指示使用的时间解析方式
        ndarray trans              # 用于时间转换的 ndarray 对象
        Py_ssize_t ntrans          # trans 数组的长度
        const int64_t[::1] deltas  # 不可变的 int64_t 类型数组
        int64_t delta              # 单个 int64_t 类型变量
        int64_t* tdata             # 指向 int64_t 类型数据的指针

    # 类中定义的方法，将 UTC 时间转换为本地时间
    cdef int64_t utc_val_to_local_val(
        self,
        int64_t utc_val,           # UTC 时间的整数值
        Py_ssize_t* pos,           # 用于存储位置的指针
        bint* fold=?              # 可选参数，指示是否进行时间折叠
    ) except? -1
        # 方法体未提供，未能提供详细注释
```