# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\np_datetime.pyx`

```
# 导入 Cython 扩展模块
cimport cython

# 从 CPython datetime 模块中导入指定的 C 扩展 API
from cpython.datetime cimport (
    PyDateTime_CheckExact,
    PyDateTime_DATE_GET_HOUR,
    PyDateTime_DATE_GET_MICROSECOND,
    PyDateTime_DATE_GET_MINUTE,
    PyDateTime_DATE_GET_SECOND,
    PyDateTime_GET_DAY,
    PyDateTime_GET_MONTH,
    PyDateTime_GET_YEAR,
    import_datetime,
)

# 从 CPython object 模块中导入指定的 C 扩展 API
from cpython.object cimport (
    Py_EQ,
    Py_GE,
    Py_GT,
    Py_LE,
    Py_LT,
    Py_NE,
)

# 从 CPython unicode 模块中导入 PyUnicode_AsUTF8AndSize 函数
from cpython.unicode cimport PyUnicode_AsUTF8AndSize

# 导入 libc 库中的 INT64_MAX 常量
from libc.stdint cimport INT64_MAX

# 调用 import_datetime() 函数，导入日期时间相关扩展
import_datetime()

# 导入 PandasDateTime_IMPORT
PandasDateTime_IMPORT

# 导入 Python 内置的 operator 模块
import operator

# 导入 NumPy 模块，并且在 Cython 中导入相应的 C 扩展 API
import numpy as np
cimport numpy as cnp

# 调用 cnp.import_array() 导入 NumPy 数组接口
cnp.import_array()

# 从 NumPy 中导入指定的 C 扩展 API
from numpy cimport (
    PyArray_DatetimeMetaData,
    PyDatetimeScalarObject,
    int64_t,
    ndarray,
    uint8_t,
)

# 从 pandas._libs.tslibs.dtypes 模块中导入指定的 C 扩展 API
from pandas._libs.tslibs.dtypes cimport (
    get_supported_reso,
    is_supported_unit,
    npy_unit_to_abbrev,
    npy_unit_to_attrname,
)

# 使用 extern from 导入 pandas/datetime/pd_datetime.h 头文件中的 C 函数和结构体定义
cdef extern from "pandas/datetime/pd_datetime.h":
    int cmp_npy_datetimestruct(npy_datetimestruct *a,
                               npy_datetimestruct *b)

    # 导入指定的 datetime 结构体定义，不同精度版本的最小值和最大值
    npy_datetimestruct _NS_MIN_DTS, _NS_MAX_DTS
    npy_datetimestruct _US_MIN_DTS, _US_MAX_DTS
    npy_datetimestruct _MS_MIN_DTS, _MS_MAX_DTS
    npy_datetimestruct _S_MIN_DTS, _S_MAX_DTS
    npy_datetimestruct _M_MIN_DTS, _M_MAX_DTS

    # 获取 dtype 的日期时间元数据的 C 函数声明
    PyArray_DatetimeMetaData get_datetime_metadata_from_dtype(cnp.PyArray_Descr *dtype)

    # 解析 ISO 8601 格式日期时间字符串的 C 函数声明
    int parse_iso_8601_datetime(const char *str, int len, int want_exc,
                                npy_datetimestruct *out,
                                NPY_DATETIMEUNIT *out_bestunit,
                                int *out_local, int *out_tzoffset,
                                const char *format, int format_len,
                                FormatRequirement exact)

# ----------------------------------------------------------------------
# numpy 对象检查

# 获取 numpy datetime64 对象的单位部分
cdef NPY_DATETIMEUNIT get_datetime64_unit(object obj) noexcept nogil:
    """
    返回 numpy datetime64 对象的单位部分。
    """
    return <NPY_DATETIMEUNIT>(<PyDatetimeScalarObject*>obj).obmeta.base

# 从 dtype 中获取单位信息
cdef NPY_DATETIMEUNIT get_unit_from_dtype(cnp.dtype dtype):
    # 注意：调用者需要确保这是 datetime64 或 timedelta64 dtype，否则可能导致段错误
    cdef:
        cnp.PyArray_Descr* descr = <cnp.PyArray_Descr*>dtype
        PyArray_DatetimeMetaData meta
    meta = get_datetime_metadata_from_dtype(descr)
    return meta.base

# 对外提供的 Python 函数，用于从 dtype 中获取单位信息
def py_get_unit_from_dtype(dtype):
    """
    用于测试 get_unit_from_dtype 函数；会向 .so 文件中添加 896 字节。
    """
    return get_unit_from_dtype(dtype)

# 获取支持的 dtype
def get_supported_dtype(dtype: cnp.dtype) -> cnp.dtype:
    reso = get_unit_from_dtype(dtype)
    new_reso = get_supported_reso(reso)
    new_unit = npy_unit_to_abbrev(new_reso)

    # 这里访问 dtype.kind 会不正确地返回 "" 而不是 "m"/"M"，所以我们检查 type_num 代替
    # 检查给定数据类型是否为 NumPy 的日期时间类型
    if dtype.type_num == cnp.NPY_DATETIME:
        # 如果是日期时间类型，创建一个新的日期时间 dtype，使用指定的单位
        new_dtype = np.dtype(f"M8[{new_unit}]")
    else:
        # 如果不是日期时间类型，创建一个新的时间戳 dtype，使用指定的单位
        new_dtype = np.dtype(f"m8[{new_unit}]")
    # 返回新创建的 dtype
    return new_dtype
# 检查给定的数据类型是否为支持的 datetime64 或 timedelta64 类型
def is_supported_dtype(dtype: cnp.dtype) -> bool:
    # 如果数据类型不是 datetime64 或 timedelta64，则抛出 ValueError 异常
    if dtype.type_num not in [cnp.NPY_DATETIME, cnp.NPY_TIMEDELTA]:
        raise ValueError("is_unitless dtype must be datetime64 or timedelta64")
    # 获取数据类型的单位
    cdef:
        NPY_DATETIMEUNIT unit = get_unit_from_dtype(dtype)
    # 返回该单位是否被支持的布尔值
    return is_supported_unit(unit)


def is_unitless(dtype: cnp.dtype) -> bool:
    """
    检查 datetime64 或 timedelta64 数据类型是否没有附加的单位。
    """
    # 如果数据类型不是 datetime64 或 timedelta64，则抛出 ValueError 异常
    if dtype.type_num not in [cnp.NPY_DATETIME, cnp.NPY_TIMEDELTA]:
        raise ValueError("is_unitless dtype must be datetime64 or timedelta64")
    # 获取数据类型的单位
    cdef:
        NPY_DATETIMEUNIT unit = get_unit_from_dtype(dtype)

    # 返回单位是否为通用时间单位的布尔值
    return unit == NPY_DATETIMEUNIT.NPY_FR_GENERIC


# ----------------------------------------------------------------------
# 比较函数


cdef bint cmp_dtstructs(
    npy_datetimestruct* left, npy_datetimestruct* right, int op
):
    cdef:
        int cmp_res

    # 比较两个 npy_datetimestruct 结构体
    cmp_res = cmp_npy_datetimestruct(left, right)
    # 根据操作符返回比较结果的布尔值
    if op == Py_EQ:
        return cmp_res == 0
    if op == Py_NE:
        return cmp_res != 0
    if op == Py_GT:
        return cmp_res == 1
    if op == Py_LT:
        return cmp_res == -1
    if op == Py_GE:
        return cmp_res == 1 or cmp_res == 0
    else:
        # 即 op == Py_LE
        return cmp_res == -1 or cmp_res == 0


cdef bint cmp_scalar(int64_t lhs, int64_t rhs, int op) except -1:
    """
    cmp_scalar 是 PyObject_RichCompare 的性能更高的版本，
    专门用于 int64_t 类型的参数。
    """
    # 根据操作符比较两个 int64_t 类型的数值
    if op == Py_EQ:
        return lhs == rhs
    elif op == Py_NE:
        return lhs != rhs
    elif op == Py_LT:
        return lhs < rhs
    elif op == Py_LE:
        return lhs <= rhs
    elif op == Py_GT:
        return lhs > rhs
    elif op == Py_GE:
        return lhs >= rhs


class OutOfBoundsDatetime(ValueError):
    """
    当日期时间超出可表示范围时引发的异常。

    示例
    --------
    >>> pd.to_datetime("08335394550")
    Traceback (most recent call last):
    OutOfBoundsDatetime: Parsing "08335394550" to datetime overflows,
    at position 0
    """
    pass


class OutOfBoundsTimedelta(ValueError):
    """
    当遇到不能表示的 timedelta 值时引发的异常。

    表示应该在 timedelta64[ns] 范围内。

    示例
    --------
    >>> pd.date_range(start="1/1/1700", freq="B", periods=100000)
    Traceback (most recent call last):
    OutOfBoundsTimedelta: Cannot cast 139999 days 00:00:00
    to unit='ns' without overflow.
    """
    # Timedelta 的异常类似于 OutOfBoundsDatetime
    pass


cdef get_implementation_bounds(
    NPY_DATETIMEUNIT reso,
    npy_datetimestruct *lower,
    npy_datetimestruct *upper,
):
    # 根据单位设置 npy_datetimestruct 结构体的上下界
    if reso == NPY_FR_ns:
        upper[0] = _NS_MAX_DTS
        lower[0] = _NS_MIN_DTS
    elif reso == NPY_FR_us:
        upper[0] = _US_MAX_DTS
        lower[0] = _US_MIN_DTS
    elif reso == NPY_FR_ms:
        upper[0] = _MS_MAX_DTS
        lower[0] = _MS_MIN_DTS
    # 如果时间分辨率为秒（NPY_FR_s），设置上限为_S_MAX_DTS，下限为_S_MIN_DTS
    elif reso == NPY_FR_s:
        upper[0] = _S_MAX_DTS
        lower[0] = _S_MIN_DTS
    # 如果时间分辨率为分钟（NPY_FR_m），设置上限为_M_MAX_DTS，下限为_M_MIN_DTS
    elif reso == NPY_FR_m:
        upper[0] = _M_MAX_DTS
        lower[0] = _M_MIN_DTS
    else:
        # 如果时间分辨率不在已知范围内，抛出未实现的错误，并附带当前分辨率信息
        raise NotImplementedError(reso)
# 定义一个 Cython 函数，将给定的 npy_datetimestruct 结构体转换为 ISO 格式的字符串表示
cdef str dts_to_iso_string(npy_datetimestruct *dts):
    return (f"{dts.year}-{dts.month:02d}-{dts.day:02d} "
            f"{dts.hour:02d}:{dts.min:02d}:{dts.sec:02d}")


# 定义一个 Cython 函数，检查给定的 npy_datetimestruct 结构体是否超出了指定单位的表示范围
cdef check_dts_bounds(npy_datetimestruct *dts, NPY_DATETIMEUNIT unit=NPY_FR_ns):
    """Raises OutOfBoundsDatetime if the given date is outside the range that
    can be represented by nanosecond-resolution 64-bit integers."""
    cdef:
        bint error = False  # 错误标志，初始化为 False
        npy_datetimestruct cmp_upper, cmp_lower  # 比较用的上界和下界结构体

    # 获取指定单位下的实现范围的上界和下界
    get_implementation_bounds(unit, &cmp_lower, &cmp_upper)

    # 检查给定的 dts 是否超出范围，设置错误标志
    if cmp_npy_datetimestruct(dts, &cmp_lower) == -1:
        error = True
    elif cmp_npy_datetimestruct(dts, &cmp_upper) == 1:
        error = True

    # 如果有错误，构造错误信息并抛出异常
    if error:
        fmt = dts_to_iso_string(dts)  # 将超出范围的日期时间结构体转换为 ISO 格式字符串
        attrname = npy_unit_to_attrname[unit]  # 获取单位对应的属性名称
        raise OutOfBoundsDatetime(f"Out of bounds {attrname} timestamp: {fmt}")


# ----------------------------------------------------------------------
# Conversion


# 仅用于测试时暴露的函数，将 int64_t 型的 td64 转换为 tdstruct 结构体，单位为 unit
def py_td64_to_tdstruct(int64_t td64, NPY_DATETIMEUNIT unit):
    cdef:
        pandas_timedeltastruct tds  # Pandas 时间差结构体
    pandas_timedelta_to_timedeltastruct(td64, unit, &tds)
    return tds  # 以字典形式返回给 Python


# 将 Python 的 datetime 对象转换为 npy_datetimestruct 结构体
cdef void pydatetime_to_dtstruct(datetime dt, npy_datetimestruct *dts) noexcept:
    if PyDateTime_CheckExact(dt):  # 如果是精确的 datetime 对象
        dts.year = PyDateTime_GET_YEAR(dt)  # 获取年份
    else:
        # 对于 Timestamp，我们使用 dt.year 而不是 PyDateTime_GET_YEAR，因为我们重写了年份
        # 使得 PyDateTime_GET_YEAR 不正确。
        dts.year = dt.year  # 获取年份

    dts.month = PyDateTime_GET_MONTH(dt)  # 获取月份
    dts.day = PyDateTime_GET_DAY(dt)  # 获取日
    dts.hour = PyDateTime_DATE_GET_HOUR(dt)  # 获取小时
    dts.min = PyDateTime_DATE_GET_MINUTE(dt)  # 获取分钟
    dts.sec = PyDateTime_DATE_GET_SECOND(dt)  # 获取秒
    dts.us = PyDateTime_DATE_GET_MICROSECOND(dt)  # 获取微秒
    dts.ps = dts.as = 0  # 初始化皮秒和太秒为 0


# 将 Python 的 datetime 对象转换为 int64_t 型的时间戳，单位为 reso（默认为纳秒）
cdef int64_t pydatetime_to_dt64(datetime val,
                                npy_datetimestruct *dts,
                                NPY_DATETIMEUNIT reso=NPY_FR_ns) except? -1:
    """
    Note we are assuming that the datetime object is timezone-naive.
    """
    cdef int64_t result
    pydatetime_to_dtstruct(val, dts)  # 调用函数将 datetime 对象转换为结构体

    try:
        result = npy_datetimestruct_to_datetime(reso, dts)  # 转换结构体为时间戳
    except OverflowError as err:
        attrname = npy_unit_to_attrname[reso]  # 获取单位对应的属性名称
        raise OutOfBoundsDatetime(
            f"Out of bounds {attrname} timestamp: {val}"
        ) from err

    return result  # 返回时间戳


# 将 Python 的 date 对象转换为 npy_datetimestruct 结构体
cdef void pydate_to_dtstruct(date val, npy_datetimestruct *dts) noexcept:
    dts.year = PyDateTime_GET_YEAR(val)  # 获取年份
    dts.month = PyDateTime_GET_MONTH(val)  # 获取月份
    dts.day = PyDateTime_GET_DAY(val)  # 获取日
    dts.hour = dts.min = dts.sec = dts.us = 0  # 小时、分钟、秒、微秒初始化为 0
    dts.ps = dts.as = 0  # 皮秒和太秒初始化为 0
    return


# 将 Python 的 date 对象转换为 int64_t 型的时间戳，单位为 reso（默认为纳秒）
cdef int64_t pydate_to_dt64(
    date val, npy_datetimestruct *dts, NPY_DATETIMEUNIT reso=NPY_FR_ns
) except? -1:
    cdef int64_t result
    pydate_to_dtstruct(val, dts)  # 调用函数将 date 对象转换为结构体

    try:
        result = npy_datetimestruct_to_datetime(reso, dts)  # 转换结构体为时间戳
    # 如果发生 OverflowError 异常，则执行以下代码块
    except OverflowError as err:
        # 从 npy_unit_to_attrname 字典中获取 reso 对应的属性名
        attrname = npy_unit_to_attrname[reso]
        # 抛出 OutOfBoundsDatetime 异常，包含详细信息和原始异常 err
        raise OutOfBoundsDatetime(f"Out of bounds {attrname} timestamp: {val}") from err

    # 返回 result 变量的值作为函数的结果
    return result
cdef int string_to_dts(
    str val,                                    # val 参数：输入的字符串值
    npy_datetimestruct* dts,                    # dts 参数：日期时间结构体指针
    NPY_DATETIMEUNIT* out_bestunit,             # out_bestunit 参数：最佳时间单位指针
    int* out_local,                             # out_local 参数：本地时间标志指针
    int* out_tzoffset,                          # out_tzoffset 参数：时区偏移指针
    bint want_exc,                              # want_exc 参数：是否期望异常标志
    format: str | None=None,                    # format 参数：日期时间格式字符串或 None
    bint exact=True,                            # exact 参数：是否精确匹配格式标志
) except? -1:                                   # 异常处理规则：返回 -1 表示异常

    cdef:
        Py_ssize_t length                        # length 变量：val 字符串的长度
        const char* buf                          # buf 变量：val 字符串的UTF-8 编码
        Py_ssize_t format_length                 # format_length 变量：format 字符串的长度
        const char* format_buf                   # format_buf 变量：format 字符串的UTF-8 编码
        FormatRequirement format_requirement     # format_requirement 变量：格式需求枚举值

    buf = PyUnicode_AsUTF8AndSize(val, &length)  # 将 val 转换为 UTF-8 编码，并获取长度
    if format is None:
        format_buf = b""                         # 若 format 为 None，则 format_buf 置空
        format_length = 0                       # format_length 设为 0
        format_requirement = INFER_FORMAT       # format_requirement 设为 INFER_FORMAT
    else:
        format_buf = PyUnicode_AsUTF8AndSize(format, &format_length)  # 否则获取 format 的 UTF-8 编码及长度
        format_requirement = <FormatRequirement>exact  # 将 exact 转换为 FormatRequirement 枚举值
    return parse_iso_8601_datetime(buf, length, want_exc,
                                   dts, out_bestunit, out_local, out_tzoffset,
                                   format_buf, format_length,
                                   format_requirement)  # 调用解析 ISO 8601 格式日期时间的函数


cpdef ndarray astype_overflowsafe(
    ndarray values,                            # values 参数：输入的 ndarray
    cnp.dtype dtype,                           # dtype 参数：目标数据类型
    bint copy=True,                            # copy 参数：是否复制数据标志，默认为 True
    bint round_ok=True,                        # round_ok 参数：是否允许舍入标志，默认为 True
    bint is_coerce=False,                      # is_coerce 参数：是否强制转换标志，默认为 False
):
    """
    Convert an ndarray with datetime64[X] to datetime64[Y]
    or timedelta64[X] to timedelta64[Y],
    raising on overflow.
    """
    if values.descr.type_num == dtype.type_num == cnp.NPY_DATETIME:
        # i.e. dtype.kind == "M"
        dtype_name = "datetime64"               # 确定数据类型名称为 datetime64
    elif values.descr.type_num == dtype.type_num == cnp.NPY_TIMEDELTA:
        # i.e. dtype.kind == "m"
        dtype_name = "timedelta64"              # 确定数据类型名称为 timedelta64
    else:
        raise TypeError(
            "astype_overflowsafe values.dtype and dtype must be either "
            "both-datetime64 or both-timedelta64."
        )                                       # 抛出类型错误异常，要求值和目标类型必须匹配

    cdef:
        NPY_DATETIMEUNIT from_unit = get_unit_from_dtype(values.dtype)  # 获取 values 的时间单位
        NPY_DATETIMEUNIT to_unit = get_unit_from_dtype(dtype)           # 获取目标 dtype 的时间单位

    if from_unit == NPY_DATETIMEUNIT.NPY_FR_GENERIC:
        raise TypeError(f"{dtype_name} values must have a unit specified")  # 如果 values 单位为通用型，则抛出类型错误异常

    if to_unit == NPY_DATETIMEUNIT.NPY_FR_GENERIC:
        raise ValueError(
            f"{dtype_name} dtype must have a unit specified"
        )                                       # 如果目标单位为通用型，则抛出值错误异常

    if from_unit == to_unit:
        if copy:
            return values.copy()                # 如果单位相同且允许复制，则返回 values 的复制
        return values                           # 否则直接返回 values

    elif from_unit > to_unit:
        iresult2 = _astype_overflowsafe_to_smaller_unit(
            values.view("i8"), from_unit, to_unit, round_ok=round_ok
        )                                       # 若单位从大到小转换，则调用特定函数处理
        return iresult2.view(dtype)             # 返回转换后的结果视图

    if (<object>values).dtype.byteorder == ">":
        values = values.astype(values.dtype.newbyteorder("<"))  # 若 values 的字节顺序为大端，则转换为小端
    cdef:
        # 将 values 视图转换为 int64 类型的 ndarray
        ndarray i8values = values.view("i8")

        # 创建一个形状与 values 相同，数据类型为 int64 的空数组
        # 相当于 result = np.empty((<object>values).shape, dtype="i8")
        ndarray iresult = cnp.PyArray_EMPTY(
            values.ndim, values.shape, cnp.NPY_INT64, 0
        )

        # 创建一个多迭代器，用于同时迭代 iresult 和 i8values
        cnp.broadcast mi = cnp.PyArray_MultiIterNew2(iresult, i8values)

        # 设置迭代器中使用的索引变量 i 和 N 的大小为 values 的大小
        Py_ssize_t i, N = values.size

        # 定义整数变量 value 和 new_value，以及 npy_datetimestruct 结构体 dts
        int64_t value, new_value
        npy_datetimestruct dts

        # 如果 dtype 的类型编号是 NPY_TIMEDELTA，则设置 is_td 为 True
        bint is_td = dtype.type_num == cnp.NPY_TIMEDELTA

    for i in range(N):
        # 从多迭代器 mi 中获取当前迭代的 values[i] 的值
        # 相当于 item = values[i]
        value = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

        if value == NPY_DATETIME_NAT:
            # 如果 value 是 NPY_DATETIME_NAT，则 new_value 也设置为 NPY_DATETIME_NAT
            new_value = NPY_DATETIME_NAT
        else:
            # 否则，将 pandas 的日期时间值转换为 datetimestruct 结构体
            pandas_datetime_to_datetimestruct(value, from_unit, &dts)

            try:
                # 检查转换后的 datetimestruct 是否在合理的范围内
                check_dts_bounds(&dts, to_unit)
            except OutOfBoundsDatetime as err:
                # 处理日期时间超出范围的异常
                if is_coerce:
                    # 如果允许强制转换，则设置 new_value 为 NPY_DATETIME_NAT
                    new_value = NPY_DATETIME_NAT
                elif is_td:
                    # 如果目标类型是 timedelta，并且不允许强制转换，则抛出 timedelta 超出范围的异常
                    from_abbrev = np.datetime_data(values.dtype)[0]
                    np_val = np.timedelta64(value, from_abbrev)
                    msg = (
                        "Cannot convert {np_val} to {dtype} without overflow"
                        .format(np_val=str(np_val), dtype=str(dtype))
                    )
                    raise OutOfBoundsTimedelta(msg) from err
                else:
                    # 其他情况直接抛出异常
                    raise
            else:
                # 如果没有异常，则将 datetimestruct 转换为目标单位的 datetime 并赋给 new_value
                new_value = npy_datetimestruct_to_datetime(to_unit, &dts)

        # 将 new_value 赋给 iresult 的当前迭代位置 i
        # 相当于 iresult[i] = new_value
        (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = new_value

        # 移动多迭代器 mi 到下一个迭代位置
        cnp.PyArray_MultiIter_NEXT(mi)

    # 返回 iresult 的视图，视图的数据类型为 dtype
    return iresult.view(dtype)
# TODO(numpy#16352): 尝试将此修复反馈到 NumPy
def compare_mismatched_resolutions(ndarray left, ndarray right, op):
    """
    Overflow-safe comparison of timedelta64/datetime64 with mismatched resolutions.

    >>> left = np.array([500], dtype="M8[Y]")
    >>> right = np.array([0], dtype="M8[ns]")
    >>> left < right  # <- wrong!
    array([ True])
    """

    if left.dtype.kind != right.dtype.kind or left.dtype.kind not in "mM":
        raise ValueError("left and right must both be timedelta64 or both datetime64")

    cdef:
        int op_code = op_to_op_code(op)  # 获取操作符对应的操作码
        NPY_DATETIMEUNIT left_unit = get_unit_from_dtype(left.dtype)  # 获取左边数组的时间单位
        NPY_DATETIMEUNIT right_unit = get_unit_from_dtype(right.dtype)  # 获取右边数组的时间单位

        # 创建一个布尔型的空数组，形状与 left 相同
        ndarray result = cnp.PyArray_EMPTY(
            left.ndim, left.shape, cnp.NPY_BOOL, 0
        )

        # 将 left 和 right 数组视图转换为 64 位整数数组
        ndarray lvalues = left.view("i8")
        ndarray rvalues = right.view("i8")

        # 创建一个多迭代器，用于并行迭代 result, lvalues, rvalues
        cnp.broadcast mi = cnp.PyArray_MultiIterNew3(result, lvalues, rvalues)
        int64_t lval, rval
        bint res_value

        Py_ssize_t i, N = left.size  # 获取 left 数组的大小
        npy_datetimestruct ldts, rdts  # 创建两个日期时间结构体 ldts 和 rdts

    for i in range(N):
        # 获取 lvalues[i] 的整数值
        lval = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

        # 获取 rvalues[i] 的整数值
        rval = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 2))[0]

        if lval == NPY_DATETIME_NAT or rval == NPY_DATETIME_NAT:
            res_value = op_code == Py_NE  # 如果 lval 或 rval 是 NaT，则 res_value 为 op_code 是否等于 Py_NE

        else:
            # 将 lval 转换为日期时间结构体 ldts
            pandas_datetime_to_datetimestruct(lval, left_unit, &ldts)
            # 将 rval 转换为日期时间结构体 rdts
            pandas_datetime_to_datetimestruct(rval, right_unit, &rdts)

            # 比较 ldts 和 rdts 结构体的结果，结果存储在 res_value 中
            res_value = cmp_dtstructs(&ldts, &rdts, op_code)

        # 将 res_value 存储到 result[i] 中
        (<uint8_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = res_value

        cnp.PyArray_MultiIter_NEXT(mi)  # 迭代器移动到下一个元素

    return result  # 返回比较结果的数组


cdef int op_to_op_code(op):
    # 根据操作符 op 返回对应的操作码
    if op is operator.eq:
        return Py_EQ
    if op is operator.ne:
        return Py_NE
    if op is operator.le:
        return Py_LE
    if op is operator.lt:
        return Py_LT
    if op is operator.ge:
        return Py_GE
    if op is operator.gt:
        return Py_GT


cdef ndarray _astype_overflowsafe_to_smaller_unit(
    ndarray i8values,
    NPY_DATETIMEUNIT from_unit,
    NPY_DATETIMEUNIT to_unit,
    bint round_ok,
):
    """
    Overflow-safe conversion for cases with from_unit > to_unit, e.g. ns->us.
    In addition for checking for overflows (which can occur near the lower
    implementation bound, see numpy#22346), this checks for truncation,
    e.g. 1500ns->1us.
    """
    # 例如 test_astype_ns_to_ms_near_bounds 是一个 round_ok=True 的情况，这里进行了溢出安全的转换
    # 定义变量 i 和 N，分别表示迭代器和数组 i8values 的大小
    cdef:
        Py_ssize_t i, N = i8values.size

        # 创建一个空的 NumPy 数组 iresult，其形状与 i8values 相同，数据类型为 int64
        ndarray iresult = cnp.PyArray_EMPTY(
            i8values.ndim, i8values.shape, cnp.NPY_INT64, 0
        )
        
        # 创建一个多迭代器 mi，用于同时迭代 iresult 和 i8values
        cnp.broadcast mi = cnp.PyArray_MultiIterNew2(iresult, i8values)

        # 获取从 from_unit 到 to_unit 的转换因子
        int64_t mult = get_conversion_factor(to_unit, from_unit)
        int64_t value, mod

    # 循环遍历 i8values 数组
    for i in range(N):
        # 获取当前迭代位置的值
        value = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

        # 如果当前值为 NPY_DATETIME_NAT，则新值也为 NPY_DATETIME_NAT
        if value == NPY_DATETIME_NAT:
            new_value = NPY_DATETIME_NAT
        else:
            # 否则，对当前值进行除法运算，并计算余数
            new_value, mod = divmod(value, mult)
            # 如果不允许四舍五入且余数不为 0，则抛出数值错误
            if not round_ok and mod != 0:
                from_abbrev = npy_unit_to_abbrev(from_unit)
                to_abbrev = npy_unit_to_abbrev(to_unit)
                raise ValueError(
                    f"Cannot losslessly cast '{value} {from_abbrev}' to {to_abbrev}"
                )

        # 将计算得到的新值赋给 iresult 数组的当前迭代位置
        (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = new_value

        # 将迭代器 mi 推进到下一个位置
        cnp.PyArray_MultiIter_NEXT(mi)

    # 返回填充后的 iresult 数组
    return iresult
cdef int64_t get_conversion_factor(
    NPY_DATETIMEUNIT from_unit,
    NPY_DATETIMEUNIT to_unit
) except? -1:
    """
    Find the factor by which we need to multiply to convert from from_unit to to_unit.
    """
    cdef int64_t value, overflow_limit, factor
    # 如果起始单位或目标单位为通用单位，则抛出异常
    if (
        from_unit == NPY_DATETIMEUNIT.NPY_FR_GENERIC
        or to_unit == NPY_DATETIMEUNIT.NPY_FR_GENERIC
    ):
        raise ValueError("unit-less resolutions are not supported")
    # 如果起始单位大于目标单位，抛出异常
    if from_unit > to_unit:
        raise ValueError("from_unit must be <= to_unit")

    # 如果起始单位与目标单位相同，返回乘数 1
    if from_unit == to_unit:
        return 1

    # 根据起始单位确定乘数因子
    if from_unit == NPY_DATETIMEUNIT.NPY_FR_W:
        value = get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_D, to_unit)
        factor = 7
    elif from_unit == NPY_DATETIMEUNIT.NPY_FR_D:
        value = get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_h, to_unit)
        factor = 24
    elif from_unit == NPY_DATETIMEUNIT.NPY_FR_h:
        value = get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_m, to_unit)
        factor = 60
    elif from_unit == NPY_DATETIMEUNIT.NPY_FR_m:
        value = get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_s, to_unit)
        factor = 60
    elif from_unit == NPY_DATETIMEUNIT.NPY_FR_s:
        value = get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_ms, to_unit)
        factor = 1000
    elif from_unit == NPY_DATETIMEUNIT.NPY_FR_ms:
        value = get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_us, to_unit)
        factor = 1000
    elif from_unit == NPY_DATETIMEUNIT.NPY_FR_us:
        value = get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_ns, to_unit)
        factor = 1000
    elif from_unit == NPY_DATETIMEUNIT.NPY_FR_ns:
        value = get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_ps, to_unit)
        factor = 1000
    elif from_unit == NPY_DATETIMEUNIT.NPY_FR_ps:
        value = get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_fs, to_unit)
        factor = 1000
    elif from_unit == NPY_DATETIMEUNIT.NPY_FR_fs:
        value = get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_as, to_unit)
        factor = 1000
    else:
        raise ValueError("Converting from M or Y units is not supported.")

    # 计算溢出限制
    overflow_limit = INT64_MAX // factor
    # 如果结果超出溢出限制，抛出溢出异常
    if value > overflow_limit or value < -overflow_limit:
        raise OverflowError("result would overflow")

    # 返回乘积结果
    return factor * value


cdef int64_t convert_reso(
    int64_t value,
    NPY_DATETIMEUNIT from_reso,
    NPY_DATETIMEUNIT to_reso,
    bint round_ok,
) except? -1:
    cdef:
        int64_t res_value, mult, div, mod, overflow_limit

    # 如果起始单位与目标单位相同，直接返回值
    if from_reso == to_reso:
        return value

    # 如果目标单位小于起始单位，进行单位转换
    elif to_reso < from_reso:
        # 获取从目标单位到起始单位的转换乘数
        mult = get_conversion_factor(to_reso, from_reso)
        # 进行除法和取模操作
        div, mod = divmod(value, mult)
        # 如果取模结果大于 0 且不允许舍入，则抛出异常
        if mod > 0 and not round_ok:
            raise ValueError("Cannot losslessly convert units")

        # 返回转换后的值（总是向下舍入）
        res_value = div
    # 如果源时间分辨率或目标时间分辨率是年或月
    elif (
        from_reso == NPY_FR_Y
        or from_reso == NPY_FR_M
        or to_reso == NPY_FR_Y
        or to_reso == NPY_FR_M
    ):
        # 通过乘法转换并不完全正确，因为月份或年的秒数并不是固定的。
        res_value = _convert_reso_with_dtstruct(value, from_reso, to_reso)

    else:
        # 例如，从纳秒到微秒的转换，存在溢出的风险，但没有丢失精度的风险。
        # 获取从源时间分辨率到目标时间分辨率的乘法转换因子
        mult = get_conversion_factor(from_reso, to_reso)
        # 计算溢出的限制，防止溢出
        overflow_limit = INT64_MAX // mult
        # 如果值超过溢出限制或小于负的溢出限制，则抛出 OverflowError 异常
        # 注意：调用者负责将其重新引发为 OutOfBoundsTimedelta 异常
        raise OverflowError("result would overflow") if value > overflow_limit or value < -overflow_limit else None

        # 计算转换后的值
        res_value = value * mult

    # 返回转换后的结果值
    return res_value
cdef int64_t _convert_reso_with_dtstruct(
    int64_t value,
    NPY_DATETIMEUNIT from_unit,
    NPY_DATETIMEUNIT to_unit,
) except? -1:
    cdef:
        npy_datetimestruct dts  # 定义一个结构体变量用于存储日期时间信息
        int64_t result  # 定义一个变量用于存储转换后的日期时间值

    pandas_datetime_to_datetimestruct(value, from_unit, &dts)  # 调用函数将 Pandas 的日期时间值转换为结构体
    try:
        result = npy_datetimestruct_to_datetime(to_unit, &dts)  # 尝试将结构体中的日期时间值按指定单位转换为 int64_t 类型
    except OverflowError as err:
        raise OutOfBoundsDatetime from err  # 如果转换过程中发生溢出错误，则抛出自定义的日期时间超出范围异常

    return result  # 返回转换后的日期时间值


@cython.overflowcheck(True)
cpdef cnp.ndarray add_overflowsafe(cnp.ndarray left, cnp.ndarray right):
    """
    Overflow-safe addition for datetime64/timedelta64 dtypes.

    `right` may either be zero-dim or of the same shape as `left`.
    """
    cdef:
        Py_ssize_t N = left.size  # 获取左操作数数组的元素数量
        int64_t lval, rval, res_value  # 定义变量用于存储左操作数、右操作数和结果值
        ndarray iresult = cnp.PyArray_EMPTY(
            left.ndim, left.shape, cnp.NPY_INT64, 0
        )  # 创建一个与左操作数形状相同的 int64 类型的空数组
        cnp.broadcast mi = cnp.PyArray_MultiIterNew3(iresult, left, right)  # 创建广播迭代器以同时迭代三个数组

    # Note: doing this try/except outside the loop improves performance over
    #  doing it inside the loop.
    try:
        for i in range(N):
            # Analogous to: lval = lvalues[i]
            lval = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]  # 获取左操作数数组的当前元素值

            # Analogous to: rval = rvalues[i]
            rval = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 2))[0]  # 获取右操作数数组的当前元素值

            if lval == NPY_DATETIME_NAT or rval == NPY_DATETIME_NAT:
                res_value = NPY_DATETIME_NAT  # 如果任一操作数为 NAT 值，则结果值为 NAT
            else:
                res_value = lval + rval  # 否则，执行整数加法运算

            # Analogous to: result[i] = res_value
            (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = res_value  # 将计算结果存入结果数组的当前位置

            cnp.PyArray_MultiIter_NEXT(mi)  # 移动迭代器到下一个元素位置
    except OverflowError as err:
        raise OverflowError("Overflow in int64 addition") from err  # 如果发生溢出错误，则抛出整数加法溢出异常

    return iresult  # 返回包含结果值的数组
```