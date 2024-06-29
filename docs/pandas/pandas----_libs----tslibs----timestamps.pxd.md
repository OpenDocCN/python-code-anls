# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\timestamps.pxd`

```
# 从cpython.datetime中导入datetime和tzinfo，用于日期时间处理
from cpython.datetime cimport (
    datetime,
    tzinfo,
)

# 从numpy中导入int64_t，用于处理64位整数
from numpy cimport int64_t

# 从pandas._libs.tslibs.base中导入ABCTimestamp，用于时间戳基类
from pandas._libs.tslibs.base cimport ABCTimestamp

# 从pandas._libs.tslibs.np_datetime中导入NPY_DATETIMEUNIT和npy_datetimestruct，
# 用于处理numpy的日期时间单位和日期时间结构
from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,
    npy_datetimestruct,
)

# 从pandas._libs.tslibs.offsets中导入BaseOffset，用于时间偏移基类
from pandas._libs.tslibs.offsets cimport BaseOffset


# 定义Cython函数_create_timestamp_from_ts，用于创建时间戳
cdef _Timestamp create_timestamp_from_ts(int64_t value,
                                         npy_datetimestruct dts,
                                         tzinfo tz,
                                         bint fold,
                                         NPY_DATETIMEUNIT reso=*)

# 定义Cython类_Timestamp，继承自ABCTimestamp，表示时间戳对象
cdef class _Timestamp(ABCTimestamp):
    # 声明只读属性：_value（整型值）、nanosecond（纳秒）、year（年份）、_creso（时间单位）
    cdef readonly:
        int64_t _value, nanosecond, year
        NPY_DATETIMEUNIT _creso

    # 声明Cython方法 _get_start_end_field，获取起始和结束字段值
    cdef bint _get_start_end_field(self, str field, freq)
    
    # 声明Cython方法 _get_date_name_field，获取日期名称字段值
    cdef _get_date_name_field(self, str field, object locale)
    
    # 声明Cython方法 _maybe_convert_value_to_local，将值可能转换为本地时间的方法
    cdef int64_t _maybe_convert_value_to_local(self) except? -1
    
    # 声明Cython方法 _can_compare，判断是否可以比较时间戳对象
    cdef bint _can_compare(self, datetime other)
    
    # 声明Cython公共方法 to_datetime64，转换为datetime64类型
    cpdef to_datetime64(self)
    
    # 声明Cython公共方法 to_pydatetime，转换为Python的datetime对象
    cpdef datetime to_pydatetime(_Timestamp self, bint warn=*)
    
    # 声明Cython方法 _compare_outside_nanorange，比较不在范围内的时间戳对象
    cdef bint _compare_outside_nanorange(_Timestamp self, datetime other,
                                         int op) except -1
    
    # 声明Cython方法 _compare_mismatched_resos，比较不匹配的时间戳分辨率
    cdef bint _compare_mismatched_resos(_Timestamp self, _Timestamp other, int op)
    
    # 声明Cython方法 _as_creso，将时间戳对象转换为指定时间单位的方法
    cdef _Timestamp _as_creso(_Timestamp self, NPY_DATETIMEUNIT creso, bint round_ok=*)
```