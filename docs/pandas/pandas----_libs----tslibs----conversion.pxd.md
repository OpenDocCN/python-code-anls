# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\conversion.pxd`

```
# 导入所需的 C 语言扩展模块和类型声明
from cpython.datetime cimport (
    datetime,    # 导入 datetime 对象
    tzinfo,      # 导入 tzinfo 类型
)
from numpy cimport (
    int32_t,     # 导入 int32_t 类型
    int64_t,     # 导入 int64_t 类型
    ndarray,     # 导入 ndarray 类型
)

# 导入 pandas 库的日期时间相关 C 扩展
from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,    # 导入 NPY_DATETIMEUNIT 枚举类型
    npy_datetimestruct,  # 导入 npy_datetimestruct 结构体
)

# 导入 pandas 库的时间戳相关 C 扩展
from pandas._libs.tslibs.timestamps cimport _Timestamp

# 导入 pandas 库的时区比较函数
from pandas._libs.tslibs.timezones cimport tz_compare


# 定义一个 C 扩展类 _TSObject，包含多个只读属性和方法声明
cdef class _TSObject:
    cdef readonly:
        npy_datetimestruct dts      # npy_datetimestruct 结构体对象
        int64_t value               # numpy dt64 对象
        tzinfo tzinfo               # tzinfo 类型对象
        bint fold                   # 布尔型标志位
        NPY_DATETIMEUNIT creso       # NPY_DATETIMEUNIT 枚举类型

    # 定义一个方法 ensure_reso，接受特定参数，返回 int64_t 类型或者 -1
    cdef int64_t ensure_reso(
        self, NPY_DATETIMEUNIT creso, val=*, bint round_ok=*
    ) except? -1


# 定义一个 C 函数 convert_to_tsobject，接受多个参数并返回 _TSObject 对象
cdef _TSObject convert_to_tsobject(object ts, tzinfo tz, str unit,
                                   bint dayfirst, bint yearfirst,
                                   int32_t nanos=*)

# 定义一个 C 函数 convert_datetime_to_tsobject，接受多个参数并返回 _TSObject 对象
cdef _TSObject convert_datetime_to_tsobject(datetime ts, tzinfo tz,
                                            int32_t nanos=*,
                                            NPY_DATETIMEUNIT reso=*)

# 定义一个 C 函数 convert_str_to_tsobject，接受多个参数并返回 _TSObject 对象
cdef _TSObject convert_str_to_tsobject(str ts, tzinfo tz,
                                       bint dayfirst=*,
                                       bint yearfirst=*)

# 定义一个 C 函数 get_datetime64_nanos，接受 object 和 NPY_DATETIMEUNIT 两个参数，返回 int64_t 或者 -1
cdef int64_t get_datetime64_nanos(object val, NPY_DATETIMEUNIT reso) except? -1

# 定义一个 Cython 可公开的函数，localize_pydatetime，接受 datetime 和 tzinfo 对象，返回 datetime 对象
cpdef datetime localize_pydatetime(datetime dt, tzinfo tz)

# 定义一个 C 函数 cast_from_unit，接受 object 和 str 两个参数，返回 int64_t 或者 -1
cdef int64_t cast_from_unit(object ts, str unit, NPY_DATETIMEUNIT out_reso=*) except? -1

# 定义一个 C 函数 precision_from_unit，接受两个 NPY_DATETIMEUNIT 类型参数，返回一个元组 (int64_t, int)
cdef (int64_t, int) precision_from_unit(
    NPY_DATETIMEUNIT in_reso, NPY_DATETIMEUNIT out_reso=*
)

# 定义一个 C 函数 maybe_localize_tso，接受 _TSObject 对象、tzinfo 对象和 NPY_DATETIMEUNIT 类型参数
cdef maybe_localize_tso(_TSObject obj, tzinfo tz, NPY_DATETIMEUNIT reso)

# 定义一个 C 函数 parse_pydatetime，接受 datetime 对象、npy_datetimestruct 结构体指针和 NPY_DATETIMEUNIT 类型参数，返回 int64_t 或者 -1
cdef int64_t parse_pydatetime(
    datetime val,
    npy_datetimestruct *dts,
    NPY_DATETIMEUNIT creso,
) except? -1
```