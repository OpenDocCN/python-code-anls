# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\conversion.pyx`

```
cimport cython  # 导入 Cython 扩展模块

import numpy as np  # 导入 NumPy 库

cimport numpy as cnp  # 使用 Cython 导入 NumPy 扩展

from libc.math cimport log10  # 导入 C 标准库中的 log10 函数
from numpy cimport (  # 使用 Cython 导入 NumPy 类型
    float64_t,
    int32_t,
    int64_t,
)

cnp.import_array()  # 导入 NumPy C-API 的数组接口

# 标准库 datetime 的相关导入
from datetime import timezone  # 导入时区相关的模块

from cpython.datetime cimport (  # 使用 Cython 导入 CPython datetime 模块的特定部分
    PyDate_Check,
    PyDateTime_Check,
    datetime,
    import_datetime,
    time,
    timedelta,
    tzinfo,
)

import_datetime()  # 调用 datetime 模块的 import_datetime 函数

from pandas._libs.missing cimport checknull_with_nat_and_na  # 使用 Cython 导入 pandas 中的缺失值检查函数
from pandas._libs.tslibs.dtypes cimport (  # 使用 Cython 导入 pandas 中的数据类型相关函数
    abbrev_to_npy_unit,
    get_supported_reso,
    npy_unit_to_attrname,
    periods_per_second,
)
from pandas._libs.tslibs.np_datetime cimport (  # 使用 Cython 导入 pandas 中的日期时间处理相关函数
    NPY_DATETIMEUNIT,
    NPY_FR_ns,
    NPY_FR_us,
    astype_overflowsafe,
    check_dts_bounds,
    convert_reso,
    dts_to_iso_string,
    get_conversion_factor,
    get_datetime64_unit,
    get_implementation_bounds,
    import_pandas_datetime,
    npy_datetime,
    npy_datetimestruct,
    npy_datetimestruct_to_datetime,
    pandas_datetime_to_datetimestruct,
    pydatetime_to_dt64,
    pydatetime_to_dtstruct,
    string_to_dts,
)

import_pandas_datetime()  # 调用 pandas_datetime 函数

from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime  # 导入 pandas 中的日期时间异常类

from pandas._libs.tslibs.nattype cimport (  # 使用 Cython 导入 pandas 中的缺失值类型相关函数
    NPY_NAT,
    c_nat_strings as nat_strings,
)
from pandas._libs.tslibs.parsing cimport parse_datetime_string  # 使用 Cython 导入 pandas 中的日期时间解析函数
from pandas._libs.tslibs.timestamps cimport _Timestamp  # 使用 Cython 导入 pandas 中的时间戳类
from pandas._libs.tslibs.timezones cimport (  # 使用 Cython 导入 pandas 中的时区相关函数
    get_utcoffset,
    is_utc,
)
from pandas._libs.tslibs.tzconversion cimport (  # 使用 Cython 导入 pandas 中的时区转换相关函数
    Localizer,
    tz_localize_to_utc_single,
)
from pandas._libs.tslibs.util cimport (  # 使用 Cython 导入 pandas 中的通用工具函数
    is_float_object,
    is_integer_object,
    is_nan,
)

# ----------------------------------------------------------------------
# 常量

DT64NS_DTYPE = np.dtype("M8[ns]")  # 定义一个 NumPy 数据类型，表示纳秒精度的日期时间
TD64NS_DTYPE = np.dtype("m8[ns]")  # 定义一个 NumPy 数据类型，表示纳秒精度的时间增量


# ----------------------------------------------------------------------
# 单位转换辅助函数

@cython.boundscheck(False)  # 禁用边界检查优化
@cython.wraparound(False)  # 禁用负索引处理优化
@cython.overflowcheck(True)  # 启用溢出检查优化
def cast_from_unit_vectorized(
    ndarray values,  # 输入的 NumPy 数组 values
    str unit,  # 输入的单位字符串 unit
    str out_unit="ns",  # 输出的单位字符串，默认为纳秒
):
    """
    Vectorized analogue to cast_from_unit.
    """
    cdef:
        int64_t m  # 整型变量 m
        int p  # 整型变量 p
        NPY_DATETIMEUNIT in_reso, out_reso  # NumPy 日期时间单位枚举类型变量 in_reso 和 out_reso
        Py_ssize_t i  # Python 原生大小类型变量 i

    assert values.dtype.kind == "f"  # 断言输入数组 values 的数据类型为浮点数

    if unit in "YM":
        if not (((values % 1) == 0) | np.isnan(values)).all():
            # 如果值不是整数或者包含 NaN，则抛出 ValueError 异常
            raise ValueError(
                f"Conversion of non-round float with unit={unit} "
                "is ambiguous"
            )

        # 使用 np.datetime64 转换单位，避免奇怪的结果，例如 "Y" 和 150 可能得到 2120-01-01 09:00:00
        values = values.astype(f"M8[{unit}]")  # 将值转换为指定单位的 np.datetime64 类型
        dtype = np.dtype(f"M8[{out_unit}]")  # 定义输出的 NumPy 数据类型
        return astype_overflowsafe(values, dtype=dtype, copy=False).view("i8")  # 返回转换后的整型视图
    # 将输入单位转换为对应的 numpy 单位表示
    in_reso = abbrev_to_npy_unit(unit)
    # 将输出单位转换为对应的 numpy 单位表示
    out_reso = abbrev_to_npy_unit(out_unit)
    # 根据输入和输出单位计算精度信息
    m, p = precision_from_unit(in_reso, out_reso)

    # 声明 C 类型变量：base 为 int64_t 类型的 ndarray，out 为 int64_t 类型的 ndarray，frac 为 float64_t 类型的 ndarray
    cdef:
        ndarray[int64_t] base, out
        ndarray[float64_t] frac
        tuple shape = (<object>values).shape

    # 创建一个形状与 values 相同的空 int64_t 类型的 ndarray，并赋值给 out
    out = np.empty(shape, dtype="i8")
    # 创建一个形状与 values 相同的空 int64_t 类型的 ndarray，并赋值给 base
    base = np.empty(shape, dtype="i8")
    # 创建一个形状与 values 相同的空 float64_t 类型的 ndarray，并赋值给 frac
    frac = np.empty(shape, dtype="f8")

    # 遍历 values 数组
    for i in range(len(values)):
        # 如果 values[i] 是 NaN，则将 base[i] 设为 NPY_NAT
        if is_nan(values[i]):
            base[i] = NPY_NAT
        else:
            # 否则，将 base[i] 设为 values[i] 的整数部分，frac[i] 设为 values[i] 的小数部分
            base[i] = <int64_t>values[i]
            frac[i] = values[i] - base[i]

    # 如果有指定精度 p，则将 frac 数组中的每个元素四舍五入到 p 位小数
    if p:
        frac = np.round(frac, p)

    # 尝试计算输出数组 out 的每个元素
    try:
        for i in range(len(values)):
            # 如果 base[i] 是 NPY_NAT，则将 out[i] 设为 NPY_NAT
            if base[i] == NPY_NAT:
                out[i] = NPY_NAT
            else:
                # 否则，将 out[i] 计算为 base[i] 和 frac[i] 经过 m 倍缩放后的结果
                out[i] = <int64_t>(base[i] * m) + <int64_t>(frac[i] * m)
    except (OverflowError, FloatingPointError) as err:
        # 如果出现 OverflowError 或 FloatingPointError，则抛出 OutOfBoundsDatetime 异常
        # FloatingPointError 可能由于 float 类型且设置了 np.errstate(over="raise") 导致
        raise OutOfBoundsDatetime(
            f"cannot convert input {values[i]} with the unit '{unit}'"
        ) from err
    # 返回计算结果数组 out
    return out
cdef int64_t cast_from_unit(
    object ts,
    str unit,
    NPY_DATETIMEUNIT out_reso=NPY_FR_ns
) except? -1:
    """
    将给定单位的时间戳转换为纳秒，并进行舍入以保留指定精度的小数部分。

    Parameters
    ----------
    ts : int, float, or None
        输入的时间戳，可以是整数、浮点数或者 None
    unit : str
        时间单位的字符串表示
    out_reso : NPY_DATETIMEUNIT, optional
        输出结果的时间分辨率，默认为纳秒

    Returns
    -------
    int64_t
        转换后的时间戳，以纳秒为单位
    """
    cdef:
        int64_t m  # 时间单位的倍数
        int p  # 要保留的小数点位数
        NPY_DATETIMEUNIT in_reso  # 输入时间单位的 numpy 表示

    if unit in ["Y", "M"]:
        if is_float_object(ts) and not ts.is_integer():
            # 如果输入的是浮点数且不是整数，抛出异常
            raise ValueError(
                f"Conversion of non-round float with unit={unit} "
                "is ambiguous"
            )
        if is_float_object(ts):
            ts = int(ts)  # 将浮点数转换为整数，以避免不明确的结果
        dt64obj = np.datetime64(ts, unit)  # 创建 np.datetime64 对象
        return get_datetime64_nanos(dt64obj, out_reso)  # 获取 np.datetime64 对象的纳秒表示

    in_reso = abbrev_to_npy_unit(unit)  # 将单位缩写转换为 numpy 时间单位
    if out_reso < in_reso and in_reso != NPY_DATETIMEUNIT.NPY_FR_GENERIC:
        # 如果输出分辨率小于输入分辨率并且输入分辨率不是通用类型，则进行向下取整操作
        m, _ = precision_from_unit(out_reso, in_reso)
        return (<int64_t>ts) // m

    m, p = precision_from_unit(in_reso, out_reso)

    try:
        base = <int64_t>ts  # 将时间戳转换为整数
    except OverflowError as err:
        raise OutOfBoundsDatetime(
            f"cannot convert input {ts} with the unit '{unit}'"
        ) from err

    frac = ts - base  # 计算时间戳的小数部分
    if p:
        frac = round(frac, p)  # 对小数部分进行四舍五入，保留指定位数的小数

    try:
        return <int64_t>(base * m) + <int64_t>(frac * m)  # 计算最终的纳秒表示结果
    except OverflowError as err:
        raise OutOfBoundsDatetime(
            f"cannot convert input {ts} with the unit '{unit}'"
        ) from err


cdef (int64_t, int) precision_from_unit(
    NPY_DATETIMEUNIT in_reso,
    NPY_DATETIMEUNIT out_reso=NPY_DATETIMEUNIT.NPY_FR_ns,
):
    """
    返回指定时间单位的纳秒表示及其小数部分的精度。

    Notes
    -----
    调用者需要确保将 "ns" 的默认值用于 None 的位置。
    """
    cdef:
        int64_t m  # 时间单位的纳秒倍数
        int64_t multiplier  # 乘数
        int p  # 小数点后的位数

    if in_reso == NPY_DATETIMEUNIT.NPY_FR_GENERIC:
        in_reso = NPY_DATETIMEUNIT.NPY_FR_ns
    if in_reso == NPY_DATETIMEUNIT.NPY_FR_Y:
        # 每 400 年有 97 个闰年，平均每年多出 97/400=0.2425 天
        # 通过 3600*24*365.2425=31556952 计算出每年的秒数
        multiplier = periods_per_second(out_reso)
        m = multiplier * 31556952  # 计算每年的纳秒数
    elif in_reso == NPY_DATETIMEUNIT.NPY_FR_M:
        # 如果输入的时间单位是分钟（NPY_FR_M），则计算对应的乘数。
        # 2629746 是从“年”单位的秒数除以 12 得到的。
        multiplier = periods_per_second(out_reso)
        m = multiplier * 2629746
    else:
        # 注意：如果 get_conversion_factor 抛出异常，
        # 异常不会传播，而是会得到一个关于忽略异常的警告。
        # 参考：https://github.com/pandas-dev/pandas/pull/51483#discussion_r1115198951
        m = get_conversion_factor(in_reso, out_reso)

    p = <int>log10(m)  # 计算 'm' 的位数减一
    return m, p
cdef int64_t get_datetime64_nanos(object val, NPY_DATETIMEUNIT reso) except? -1:
    """
    从 np.datetime64 对象中提取值和单位，然后按需将值转换为纳秒。
    """
    cdef:
        npy_datetimestruct dts          # 定义 npy_datetimestruct 结构体变量 dts
        NPY_DATETIMEUNIT unit           # 定义 NPY_DATETIMEUNIT 枚举类型变量 unit
        npy_datetime ival               # 定义 npy_datetime 类型变量 ival

    # 调用 cnp 模块的函数获取 np.datetime64 对象的值
    ival = cnp.get_datetime64_value(val)
    # 如果值为 NPY_NAT（不可用的时间戳），直接返回 NPY_NAT
    if ival == NPY_NAT:
        return NPY_NAT

    # 获取 np.datetime64 对象的单位
    unit = get_datetime64_unit(val)

    # 如果单位不等于指定的 reso 单位
    if unit != reso:
        # 将 ival 转换为 npy_datetimestruct 结构体 dts
        pandas_datetime_to_datetimestruct(ival, unit, &dts)
        try:
            # 尝试将 dts 转换为指定单位 reso 的 npy_datetime 值
            ival = npy_datetimestruct_to_datetime(reso, &dts)
        except OverflowError as err:
            # 如果转换过程中出现溢出错误，抛出 OutOfBoundsDatetime 异常
            attrname = npy_unit_to_attrname[reso]
            raise OutOfBoundsDatetime(
                f"Out of bounds {attrname} timestamp: {val}"
            ) from err

    # 返回转换后的 npy_datetime 值
    return ival


# ----------------------------------------------------------------------
# _TSObject Conversion

# lightweight C object to hold datetime & int64 pair
cdef class _TSObject:
    # cdef:
    #    npy_datetimestruct dts      # npy_datetimestruct 结构体
    #    int64_t value               # numpy dt64 值
    #    tzinfo tzinfo              # 时区信息
    #    bint fold                  # 折叠标志
    #    NPY_DATETIMEUNIT creso      # 时间单位

    def __cinit__(self):
        # GH 25057. As per PEP 495, set fold to 0 by default
        self.fold = 0                 # 初始化折叠标志为 0（按 PEP 495 的建议）
        self.creso = NPY_FR_ns        # 默认时间单位为纳秒

    cdef int64_t ensure_reso(
        self, NPY_DATETIMEUNIT creso, val=None, bint round_ok=False
    ) except? -1:
        # 如果当前时间单位不等于传入的 creso 单位
        if self.creso != creso:
            try:
                # 调用 convert_reso 函数将 self.value 从当前单位 self.creso 转换为 creso 单位
                self.value = convert_reso(
                    self.value, self.creso, creso, round_ok=round_ok
                )
            except OverflowError as err:
                # 如果转换过程中出现溢出错误，根据传入的 val 抛出 OutOfBoundsDatetime 异常
                if val is not None:
                    attrname = npy_unit_to_attrname[creso]
                    raise OutOfBoundsDatetime(
                        f"Out of bounds {attrname} timestamp: {val}"
                    ) from err
                raise OutOfBoundsDatetime from err

            # 更新对象的时间单位为 creso
            self.creso = creso
        # 返回转换后的 self.value 值
        return self.value


cdef _TSObject convert_to_tsobject(object ts, tzinfo tz, str unit,
                                   bint dayfirst, bint yearfirst, int32_t nanos=0):
    """
    从以下对象中提取 datetime 和 int64 值：
        - np.int64（单位提供可能的修饰符）
        - np.datetime64
        - float（单位提供可能的修饰符）
        - python int 或 long 对象（单位提供可能的修饰符）
        - iso8601 字符串对象
        - python datetime 对象
        - 另一个时间戳对象

    Raises
    ------
    OutOfBoundsDatetime: 无法在实现边界内转换 ts
    """
    cdef:
        _TSObject obj               # 定义 _TSObject 类型变量 obj
        NPY_DATETIMEUNIT reso       # 定义 NPY_DATETIMEUNIT 枚举类型变量 reso

    # 初始化 _TSObject 类对象
    obj = _TSObject()

    # 如果 ts 是字符串，调用 convert_str_to_tsobject 函数处理并返回结果
    if isinstance(ts, str):
        return convert_str_to_tsobject(ts, tz, dayfirst, yearfirst)

    # 检查 ts 是否为 null 值（NPY_NAT 或 NA），设置 obj.value 为 NPY_NAT
    if checknull_with_nat_and_na(ts):
        obj.value = NPY_NAT
    # 检查时间戳是否为 pandas 的 datetime64 对象
    elif cnp.is_datetime64_object(ts):
        # 根据时间戳获取支持的时间分辨率
        reso = get_supported_reso(get_datetime64_unit(ts))
        # 将时间分辨率设置到对象的 creso 属性中
        obj.creso = reso
        # 获取时间戳的纳秒表示值
        obj.value = get_datetime64_nanos(ts, reso)
        # 如果获取的值不是 NaT（Not a Time），则将其转换为日期时间结构体
        if obj.value != NPY_NAT:
            pandas_datetime_to_datetimestruct(obj.value, reso, &obj.dts)
            # 如果指定了时区，则将时间戳本地化为 UTC 时间
            if tz is not None:
                # GH#24559, GH#42288 我们将 np.datetime64 对象视为 *墙上* 时间
                obj.value = tz_localize_to_utc_single(
                    obj.value, tz, ambiguous="raise", nonexistent=None, creso=reso
                )
    
    # 检查时间戳是否为整数对象
    elif is_integer_object(ts):
        try:
            # 尝试将时间戳强制转换为 int64 类型
            ts = <int64_t>ts
        except OverflowError:
            # GH#26651 抛出 OutOfBoundsDatetime 异常，说明时间戳超出范围
            raise OutOfBoundsDatetime(f"Out of bounds nanosecond timestamp {ts}")
        # 如果时间戳为 NaT，则将对象的值设置为 NaT
        if ts == NPY_NAT:
            obj.value = NPY_NAT
        else:
            # 如果未指定单位，则默认为 "ns"
            if unit is None:
                unit = "ns"
            # 将输入单位缩写转换为 NumPy 支持的单位
            in_reso = abbrev_to_npy_unit(unit)
            # 获取支持的时间分辨率
            reso = get_supported_reso(in_reso)
            # 根据指定单位将时间戳转换为指定的时间分辨率
            ts = cast_from_unit(ts, unit, reso)
            # 将转换后的时间戳设置为对象的值
            obj.value = ts
            # 将时间戳转换为日期时间结构体
            pandas_datetime_to_datetimestruct(ts, reso, &obj.dts)
    
    # 检查时间戳是否为浮点数对象
    elif is_float_object(ts):
        # 如果时间戳为 NaN 或 NaT，则将对象的值设置为 NaT
        if ts != ts or ts == NPY_NAT:
            obj.value = NPY_NAT
        else:
            # 根据指定单位将时间戳转换为 NumPy 支持的单位
            ts = cast_from_unit(ts, unit)
            # 将转换后的时间戳设置为对象的值
            obj.value = ts
            # 将时间戳转换为日期时间结构体，使用纳秒分辨率
            pandas_datetime_to_datetimestruct(ts, NPY_FR_ns, &obj.dts)
    
    # 检查时间戳是否为 Python 的 datetime 对象
    elif PyDateTime_Check(ts):
        # 根据纳秒数判断时间戳是否为 0
        if nanos == 0:
            # 如果时间戳是 _Timestamp 类型的实例，则获取其对应的时间分辨率
            if isinstance(ts, _Timestamp):
                reso = (<_Timestamp>ts)._creso
            else:
                # 否则，默认设置时间分辨率为微秒级别
                # TODO: 如果用户显式传递 nanos=0 会怎么样？
                reso = NPY_FR_us
        else:
            # 否则，时间分辨率为纳秒级别
            reso = NPY_FR_ns
        # 将 Python 的 datetime 对象转换为时间戳对象并返回
        return convert_datetime_to_tsobject(ts, tz, nanos, reso=reso)
    
    # 检查时间戳是否为 Python 的 date 对象
    elif PyDate_Check(ts):
        # 将 date 对象的时间部分设为零点，并转换为 datetime 对象
        ts = datetime.combine(ts, time())
        # 将 datetime 对象转换为时间戳对象，并使用秒级别的时间分辨率
        return convert_datetime_to_tsobject(
            ts, tz, nanos=0, reso=NPY_DATETIMEUNIT.NPY_FR_s
        )
    
    else:
        # 如果时间戳不属于以上任何一种类型，则抛出类型错误异常
        from .period import Period
        if isinstance(ts, Period):
            raise ValueError("Cannot convert Period to Timestamp "
                             "unambiguously. Use to_timestamp")
        raise TypeError(f"Cannot convert input [{ts}] of type {type(ts)} to "
                        f"Timestamp")
    
    # 可能根据时区本地化时间戳对象的值
    maybe_localize_tso(obj, tz, obj.creso)
    # 返回处理后的时间戳对象
    return obj
def maybe_localize_tso(_TSObject obj, tzinfo tz, NPY_DATETIMEUNIT reso):
    # 如果时区不为空，调用 _localize_tso 方法进行时区本地化处理
    if tz is not None:
        _localize_tso(obj, tz, reso)

    # 如果 obj 的值不是 NaT（Not a Time），则调用 check_dts_bounds 检查日期时间结构的边界
    if obj.value != NPY_NAT:
        # check_overflows 需要在 _localize_tso 之后运行
        check_dts_bounds(&obj.dts, reso)
        check_overflows(obj, reso)


cdef _TSObject convert_datetime_to_tsobject(
    datetime ts,
    tzinfo tz,
    int32_t nanos=0,
    NPY_DATETIMEUNIT reso=NPY_FR_ns,
):
    """
    Convert a datetime (or Timestamp) input `ts`, along with optional timezone
    object `tz` to a _TSObject.

    The optional argument `nanos` allows for cases where datetime input
    needs to be supplemented with higher-precision information.

    Parameters
    ----------
    ts : datetime or Timestamp
        Value to be converted to _TSObject
    tz : tzinfo or None
        timezone for the timezone-aware output
    nanos : int32_t, default is 0
        nanoseconds supplement the precision of the datetime input ts
    reso : NPY_DATETIMEUNIT, default NPY_FR_ns

    Returns
    -------
    obj : _TSObject
    """
    cdef:
        _TSObject obj = _TSObject()  # 创建 _TSObject 对象
        int64_t pps  # 声明一个 int64_t 类型的变量 pps，用于存储每秒的周期数

    obj.creso = reso  # 设置 obj 的时间单位精度
    obj.fold = ts.fold  # 设置 obj 的 fold 属性为 ts 的 fold 属性值
    if tz is not None:
        if ts.tzinfo is not None:
            # 如果 ts 已经有 tzinfo 信息，则将其转换为指定的 tz 时区
            ts = ts.astimezone(tz)
            pydatetime_to_dtstruct(ts, &obj.dts)  # 将 Python datetime 转换为 C 结构体
            obj.tzinfo = ts.tzinfo  # 设置 obj 的时区信息为 ts 的时区信息
        elif not is_utc(tz):
            ts = _localize_pydatetime(ts, tz)  # 在非 UTC 时区情况下，将 ts 本地化为 tz 时区
            pydatetime_to_dtstruct(ts, &obj.dts)
            obj.tzinfo = ts.tzinfo
        else:
            # 对于 UTC 时区，直接将 ts 转换为 datetime 结构体
            pydatetime_to_dtstruct(ts, &obj.dts)
            obj.tzinfo = tz  # 设置 obj 的时区为 UTC
    else:
        pydatetime_to_dtstruct(ts, &obj.dts)  # 将 ts 转换为 datetime 结构体
        obj.tzinfo = ts.tzinfo  # 设置 obj 的时区为 ts 的时区

    if isinstance(ts, _Timestamp):
        obj.dts.ps = ts.nanosecond * 1000  # 如果 ts 是 _Timestamp 类型，则设置 obj 的 picoseconds

    if nanos:
        obj.dts.ps = nanos * 1000  # 如果有 nanos 参数，设置 obj 的 picoseconds

    try:
        obj.value = npy_datetimestruct_to_datetime(reso, &obj.dts)  # 将 datetime 结构体转换为 numpy datetime
    except OverflowError as err:
        attrname = npy_unit_to_attrname[reso]
        raise OutOfBoundsDatetime(f"Out of bounds {attrname} timestamp") from err

    if obj.tzinfo is not None and not is_utc(obj.tzinfo):
        offset = get_utcoffset(obj.tzinfo, ts)  # 获取时区偏移量
        pps = periods_per_second(reso)  # 获取每秒的周期数
        obj.value -= int(offset.total_seconds() * pps)  # 调整 obj 的值以考虑时区偏移

    check_overflows(obj, reso)  # 检查是否发生溢出
    return obj


cdef _adjust_tsobject_tz_using_offset(_TSObject obj, tzinfo tz):
    """
    Convert a datetimestruct `obj.dts`, with an attached tzinfo to a new
    user-provided tz.

    Parameters
    ----------
    obj : _TSObject
    tz : tzinfo
        timezone for the timezone-aware output.
    """
    cdef:
        datetime dt  # 声明一个 datetime 变量 dt
        Py_ssize_t pos  # 声明一个 Py_ssize_t 类型的变量 pos
        int64_t ps = obj.dts.ps  # 获取 obj 的 picoseconds
        Localizer info = Localizer(tz, obj.creso)  # 创建一个 Localizer 对象，用于时区本地化

    # 从偏移调整后的 obj.value 推断 fold 属性
    # 参见 PEP 495 https://www.python.org/dev/peps/pep-0495/#the-fold-attribute
    if info.use_utc:
        pass  # 如果使用 UTC，则跳过处理
    elif info.use_tzlocal:
        # 如果info.use_tzlocal为True，则使用本地时区转换obj.value的UTC值
        info.utc_val_to_local_val(obj.value, &pos, &obj.fold)
    elif info.use_dst and not info.use_pytz:
        # 如果info.use_dst为True且info.use_pytz为False（即使用dateutil），则同样使用本地时区转换obj.value的UTC值
        # 例如，使用dateutil库进行时区转换
        info.utc_val_to_local_val(obj.value, &pos, &obj.fold)

    # 使用与PyDateTime相同的转换器来创建datetime对象
    dt = datetime(obj.dts.year, obj.dts.month, obj.dts.day,
                  obj.dts.hour, obj.dts.min, obj.dts.sec,
                  obj.dts.us, obj.tzinfo, fold=obj.fold)

    # 后续步骤与convert_datetime_to_tsobject中的双时区路径类似，
    # 但避免重新计算obj.value
    # 将datetime对象转换为指定时区tz的时间
    dt = dt.astimezone(tz)
    # 将转换后的datetime对象的日期时间信息填充到obj.dts结构体中
    pydatetime_to_dtstruct(dt, &obj.dts)
    # 更新obj对象的时区信息为转换后datetime对象的时区信息
    obj.tzinfo = dt.tzinfo
    # 检查obj.dts的边界情况，根据obj.creso进行检查
    check_dts_bounds(&obj.dts, obj.creso)
    # 检查obj对象是否存在溢出情况，根据obj.creso进行检查
    check_overflows(obj, obj.creso)
# 定义函数 convert_str_to_tsobject，将输入字符串 `ts` 与可选的时区对象 `tz` 转换为 _TSObject 类型对象

cdef _TSObject convert_str_to_tsobject(str ts, tzinfo tz,
                                       bint dayfirst=False,
                                       bint yearfirst=False):
    """
    Convert a string input `ts`, along with optional timezone object`tz`
    to a _TSObject.

    The optional arguments `dayfirst` and `yearfirst` are passed to the
    dateutil parser.

    Parameters
    ----------
    ts : str
        Value to be converted to _TSObject
    tz : tzinfo or None
        timezone for the timezone-aware output
    dayfirst : bool, default False
        When parsing an ambiguous date string, interpret e.g. "3/4/1975" as
        April 3, as opposed to the standard US interpretation March 4.
    yearfirst : bool, default False
        When parsing an ambiguous date string, interpret e.g. "01/05/09"
        as "May 9, 2001", as opposed to the default "Jan 5, 2009"

    Returns
    -------
    obj : _TSObject
    """
    
    # 定义 C 语言级别的变量
    cdef:
        npy_datetimestruct dts              # NumPy datetime 结构体
        int out_local = 0, out_tzoffset = 0, string_to_dts_failed  # 整型变量声明
        datetime dt                         # Python datetime 对象
        int64_t ival, nanos = 0             # 64 位整数声明，nanos 初始化为 0
        NPY_DATETIMEUNIT out_bestunit, reso # NumPy datetime 单位类型声明
        _TSObject obj                       # 自定义 _TSObject 对象

    # 如果输入字符串长度为 0 或者在 nat_strings 中，则返回空的 _TSObject 对象
    if len(ts) == 0 or ts in nat_strings:
        obj = _TSObject()
        obj.value = NPY_NAT
        obj.tzinfo = tz
        return obj
    # 如果输入字符串是 "now"，则返回当前时刻的 _TSObject 对象
    elif ts == "now":
        # 为了避免进入 np_datetime_strings 函数，直接获取当前时刻的 datetime 对象
        dt = datetime.now(tz)
        return convert_datetime_to_tsobject(dt, tz, nanos=0, reso=NPY_FR_us)
    # 如果输入字符串是 "today"，则返回当前日期的 _TSObject 对象
    elif ts == "today":
        # 为了避免进入 np_datetime_strings 函数，直接获取当前日期的 datetime 对象，并进行时区替换
        dt = datetime.now(tz)
        # 相当于 datetime.today().replace(tzinfo=tz)
        return convert_datetime_to_tsobject(dt, tz, nanos=0, reso=NPY_FR_us)
    # 如果不是第一个条件下的情况，则调用 string_to_dts 函数进行日期时间字符串到结构化时间的转换
    else:
        # 调用 string_to_dts 函数，将字符串转换为结构化时间，返回是否失败的标志
        string_to_dts_failed = string_to_dts(
            ts, &dts, &out_bestunit, &out_local,
            &out_tzoffset, False
        )
        # 如果转换成功
        if not string_to_dts_failed:
            # 根据最佳时间单位获取支持的分辨率
            reso = get_supported_reso(out_bestunit)
            # 检查结构化时间是否在有效范围内
            check_dts_bounds(&dts, reso)
            # 创建一个 _TSObject 实例
            obj = _TSObject()
            obj.dts = dts  # 将结构化时间赋给 obj 实例的 dts 属性
            obj.creso = reso  # 将分辨率赋给 obj 实例的 creso 属性
            # 将结构化时间转换为 datetime 对象
            ival = npy_datetimestruct_to_datetime(reso, &dts)

            # 如果是本地时间
            if out_local == 1:
                # 根据时区偏移创建时区对象
                obj.tzinfo = timezone(timedelta(minutes=out_tzoffset))
                # 将时间本地化到 UTC，并根据参数调整行为
                obj.value = tz_localize_to_utc_single(
                    ival, obj.tzinfo, ambiguous="raise", nonexistent=None, creso=reso
                )
                # 如果未提供时区信息，则检查溢出情况并返回 obj
                if tz is None:
                    check_overflows(obj, reso)
                    return obj
                # 根据偏移调整 obj 的时区信息
                _adjust_tsobject_tz_using_offset(obj, tz)
                return obj
            else:
                # 如果提供了时区信息，则根据时区本地化时间到 UTC
                if tz is not None:
                    ival = tz_localize_to_utc_single(
                        ival, tz, ambiguous="raise", nonexistent=None, creso=reso
                    )
                # 将时间赋给 obj 实例的 value 属性
                obj.value = ival
                # 可能根据时区信息本地化 _TSObject 实例
                maybe_localize_tso(obj, tz, obj.creso)
                return obj

        # 如果 string_to_dts 函数转换失败，则尝试解析日期时间字符串
        dt = parse_datetime_string(
            ts,
            dayfirst=dayfirst,
            yearfirst=yearfirst,
            out_bestunit=&out_bestunit,
            nanos=&nanos,
        )
        # 根据最佳时间单位获取支持的分辨率
        reso = get_supported_reso(out_bestunit)
        # 将解析后的 datetime 对象转换为 TSObject 对象并返回
        return convert_datetime_to_tsobject(dt, tz, nanos=nanos, reso=reso)
# ----------------------------------------------------------------------
# Localization

cdef void _localize_tso(_TSObject obj, tzinfo tz, NPY_DATETIMEUNIT reso) noexcept:
    """
    Given the UTC nanosecond timestamp in obj.value, find the wall-clock
    representation of that timestamp in the given timezone.

    Parameters
    ----------
    obj : _TSObject
        An object containing the UTC nanosecond timestamp to be localized.
    tz : tzinfo
        Timezone information to localize the timestamp to.
    reso : NPY_DATETIMEUNIT
        Resolution of the datetime unit.

    Returns
    -------
    None

    Notes
    -----
    Sets obj.tzinfo inplace, alters obj.dts inplace.
    """
    cdef:
        int64_t local_val  # Localized value of the timestamp in the target timezone
        Py_ssize_t outpos = -1  # Output position for pytz path in localization
        Localizer info = Localizer(tz, reso)  # Object to handle localization information

    assert obj.tzinfo is None  # Ensure obj.tzinfo is initially None

    if info.use_utc:
        pass  # No conversion needed if already in UTC
    elif obj.value == NPY_NAT:
        pass  # No conversion needed if value is NaT (Not a Time)
    else:
        # Convert UTC value to local value using provided timezone information
        local_val = info.utc_val_to_local_val(obj.value, &outpos, &obj.fold)

        if info.use_pytz:
            # If pytz was used, adjust timezone info using transition data
            tz = tz._tzinfos[tz._transition_info[outpos]]

        # Convert localized value to datetimestruct
        pandas_datetime_to_datetimestruct(local_val, reso, &obj.dts)

    obj.tzinfo = tz  # Set the timezone information inplace in obj


cdef datetime _localize_pydatetime(datetime dt, tzinfo tz):
    """
    Take a datetime/Timestamp in UTC and localizes to timezone tz.

    NB: Unlike the public version, this treats datetime and Timestamp objects
        identically, i.e. discards nanos from Timestamps.
        It also assumes that the `tz` input is not None.
    """
    try:
        # Localize datetime object using pytz library (if available)
        # Note: This may not correctly handle `fold` attribute
        return tz.localize(dt, is_dst=None)
    except AttributeError:
        # Fallback: Replace timezone information if AttributeError occurs
        return dt.replace(tzinfo=tz)


cpdef inline datetime localize_pydatetime(datetime dt, tzinfo tz):
    """
    Take a datetime/Timestamp in UTC and localizes to timezone tz.

    Parameters
    ----------
    dt : datetime
        Datetime object to be localized.
    tz : tzinfo
        Timezone information to localize the datetime to.

    Returns
    -------
    datetime
        Localized datetime object.
    """
    dt : datetime or Timestamp
    tz : tzinfo or None

# 定义函数的参数：
# - `dt`：可以是 `datetime` 或 `Timestamp` 对象。
# - `tz`：可以是 `tzinfo` 对象或 `None`。


    Returns
    -------
    localized : datetime or Timestamp

# 返回值：
# - `localized`：经过时区本地化处理后的 `datetime` 或 `Timestamp` 对象。


    """
    如果 `tz` 为 `None`，直接返回 `dt`。
    如果 `dt` 是 `_Timestamp` 类的实例，调用其 `tz_localize(tz)` 方法进行时区本地化处理后返回。
    否则，调用 `_localize_pydatetime(dt, tz)` 处理 `dt` 对象并返回结果。
    """
    if tz is None:
        return dt
    elif isinstance(dt, _Timestamp):
        return dt.tz_localize(tz)
    return _localize_pydatetime(dt, tz)
cdef int64_t parse_pydatetime(
    datetime val,
    npy_datetimestruct *dts,
    NPY_DATETIMEUNIT creso,
) except? -1:
    """
    Convert pydatetime to datetime64.

    Parameters
    ----------
    val : datetime
        Element being processed.
    dts : *npy_datetimestruct
        Needed to use in pydatetime_to_dt64, which writes to it.
    creso : NPY_DATETIMEUNIT
        Resolution to store the the result.

    Raises
    ------
    OutOfBoundsDatetime
        Raised if the datetime value is out of the supported range.
    """
    cdef:
        _TSObject _ts   # 定义一个 _TSObject 类型的变量 _ts
        int64_t result  # 定义一个 int64_t 类型的变量 result

    # 如果 val 带有时区信息
    if val.tzinfo is not None:
        # 将 val 转换为 _TSObject 对象，nanos 设为 0，使用指定的分辨率 creso
        _ts = convert_datetime_to_tsobject(val, None, nanos=0, reso=creso)
        # 将 _ts 对象的值赋给 result
        result = _ts.value
    else:
        # 如果 val 是 _Timestamp 类型的实例
        if isinstance(val, _Timestamp):
            # 将 val 强制转换为 _Timestamp 类型，然后使用给定的分辨率 creso 获取其值
            result = (<_Timestamp>val)._as_creso(creso, round_ok=True)._value
        else:
            # 将 pydatetime 转换为 datetime64，并将结果存入 dts 所指向的结构体中
            result = pydatetime_to_dt64(val, dts, reso=creso)
    
    # 返回处理后的结果
    return result
```