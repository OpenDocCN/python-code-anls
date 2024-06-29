# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\np_datetime.pxd`

```
# 导入Cython版本的numpy模块
cimport numpy as cnp
# 从Cython版本的datetime模块中导入特定内容
from cpython.datetime cimport (
    date,
    datetime,
)
# 从Cython版本的numpy模块中导入特定内容
from numpy cimport (
    int32_t,
    int64_t,
    npy_datetime,
    npy_timedelta,
)

# 在此注释以下内容的导入可以直接从numpy中导入
# TODO(cython3): most of these can be cimported directly from numpy

# 从numpy/ndarraytypes.h头文件中导入结构体np_datetimestruct
cdef extern from "numpy/ndarraytypes.h":
    ctypedef struct npy_datetimestruct:
        int64_t year
        int32_t month, day, hour, min, sec, us, ps, as

    # 定义枚举类型NPY_DATETIMEUNIT，表示日期时间单位
    ctypedef enum NPY_DATETIMEUNIT:
        NPY_FR_Y
        NPY_FR_M
        NPY_FR_W
        NPY_FR_D
        NPY_FR_B
        NPY_FR_h
        NPY_FR_m
        NPY_FR_s
        NPY_FR_ms
        NPY_FR_us
        NPY_FR_ns
        NPY_FR_ps
        NPY_FR_fs
        NPY_FR_as
        NPY_FR_GENERIC

    # 定义NPY_DATETIME_NAT为int64_t类型，表示日期时间的无效值
    int64_t NPY_DATETIME_NAT  # elswhere we call this NPY_NAT


# 从pandas/datetime/pd_datetime.h头文件中导入结构体pandas_timedeltastruct及相关函数
cdef extern from "pandas/datetime/pd_datetime.h":
    ctypedef struct pandas_timedeltastruct:
        int64_t days
        int32_t hrs, min, sec, ms, us, ns, seconds, microseconds, nanoseconds

    # 将numpy日期时间转换为np_datetimestruct结构体
    void pandas_datetime_to_datetimestruct(npy_datetime val,
                                           NPY_DATETIMEUNIT fr,
                                           npy_datetimestruct *result) nogil

    # 将np_datetimestruct结构体转换为numpy日期时间
    npy_datetime npy_datetimestruct_to_datetime(NPY_DATETIMEUNIT fr,
                                                npy_datetimestruct *d) except? -1 nogil

    # 将pandas时间差转换为pandas_timedeltastruct结构体
    void pandas_timedelta_to_timedeltastruct(npy_timedelta val,
                                             NPY_DATETIMEUNIT fr,
                                             pandas_timedeltastruct *result
                                             ) nogil

    # 导入Pandas日期时间相关函数
    void PandasDateTime_IMPORT()

    # 定义枚举类型FormatRequirement，表示格式要求
    ctypedef enum FormatRequirement:
        PARTIAL_MATCH
        EXACT_MATCH
        INFER_FORMAT

# 必须在使用PandasDateTime CAPI函数之前调用此函数
cdef inline void import_pandas_datetime() noexcept:
    PandasDateTime_IMPORT

# 比较两个int64_t类型数值的大小
cdef bint cmp_scalar(int64_t lhs, int64_t rhs, int op) except -1

# 将np_datetimestruct结构体转换为ISO格式的字符串
cdef str dts_to_iso_string(npy_datetimestruct *dts)

# 检查np_datetimestruct结构体是否超出范围
cdef check_dts_bounds(npy_datetimestruct *dts, NPY_DATETIMEUNIT unit=?)

# 将Python datetime转换为dt64整数
cdef int64_t pydatetime_to_dt64(
    datetime val, npy_datetimestruct *dts, NPY_DATETIMEUNIT reso=?
) except? -1

# 将Python datetime转换为np_datetimestruct结构体
cdef void pydatetime_to_dtstruct(datetime dt, npy_datetimestruct *dts) noexcept

# 将Python date转换为dt64整数
cdef int64_t pydate_to_dt64(
    date val, npy_datetimestruct *dts, NPY_DATETIMEUNIT reso=?
) except? -1

# 将Python date转换为np_datetimestruct结构体
cdef void pydate_to_dtstruct(date val, npy_datetimestruct *dts) noexcept

# 从dtype中获取NPY_DATETIMEUNIT类型
cdef NPY_DATETIMEUNIT get_datetime64_unit(object obj) noexcept nogil

# 将字符串转换为np_datetimestruct结构体
cdef int string_to_dts(
    str val,
    npy_datetimestruct* dts,
    NPY_DATETIMEUNIT* out_bestunit,
    int* out_local,
    int* out_tzoffset,
    bint want_exc,
    format: str | None = *,
    bint exact = *
) except? -1

# 从dtype中获取NPY_DATETIMEUNIT类型
cdef NPY_DATETIMEUNIT get_unit_from_dtype(cnp.dtype dtype)

# 定义Cython公共API函数astype_overflowsafe，用于安全转换数据类型
cpdef cnp.ndarray astype_overflowsafe(
    cnp.ndarray values,  # ndarray[datetime64[anyunit]]
    cnp.dtype dtype,  # ndarray[datetime64[anyunit]]
    bint copy=*,
    bint round_ok=*,
    bint is_coerce=*,
)
# 定义一个 Cython 的函数，用于获取日期时间单位之间的转换系数
cdef int64_t get_conversion_factor(
    NPY_DATETIMEUNIT from_unit,
    NPY_DATETIMEUNIT to_unit,
) except? -1

# 定义一个 Cython 的函数，用于比较两个日期时间结构体的大小
cdef bint cmp_dtstructs(npy_datetimestruct* left, npy_datetimestruct* right, int op)

# 定义一个 Cython 的函数，用于获取特定分辨率下的实现边界
cdef get_implementation_bounds(
    NPY_DATETIMEUNIT reso, npy_datetimestruct *lower, npy_datetimestruct *upper
)

# 定义一个 Cython 的函数，用于将一个日期时间值从一个分辨率转换到另一个分辨率，并可选择是否进行四舍五入
cdef int64_t convert_reso(
    int64_t value,
    NPY_DATETIMEUNIT from_reso,
    NPY_DATETIMEUNIT to_reso,
    bint round_ok,
) except? -1

# 定义一个 Cython 的函数，用于在 NumPy 数组上执行溢出安全的加法操作，并返回一个新的数组
cpdef cnp.ndarray add_overflowsafe(cnp.ndarray left, cnp.ndarray right)
```