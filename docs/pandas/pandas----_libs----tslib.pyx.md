# `D:\src\scipysrc\pandas\pandas\_libs\tslib.pyx`

```
# 导入Cython模块
cimport cython

# 导入Python标准库中的timezone类
from datetime import timezone

# 导入Cython中datetime模块中的特定函数和类型定义
from cpython.datetime cimport (
    PyDate_Check,
    PyDateTime_Check,
    import_datetime,
    timedelta,
    tzinfo,
)
# 导入CPython对象定义
from cpython.object cimport PyObject

# 调用C API中的datetime模块
import_datetime()

# 导入Cython封装的numpy模块
cimport numpy as cnp

# 从numpy C API中导入特定类型和对象
from numpy cimport (
    int64_t,
    ndarray,
)

# 导入标准的numpy模块
import numpy as np

# 调用numpy的import_array函数
cnp.import_array()

# 导入pandas私有库中的tslibs.dtypes模块
from pandas._libs.tslibs.dtypes cimport (
    get_supported_reso,
    npy_unit_to_abbrev,
)

# 导入pandas私有库中的np_datetime模块
from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,
    NPY_FR_ns,
    get_datetime64_unit,
    import_pandas_datetime,
    npy_datetimestruct,
    npy_datetimestruct_to_datetime,
    pandas_datetime_to_datetimestruct,
    pydate_to_dt64,
    string_to_dts,
)

# 调用pandas私有库中的import_pandas_datetime函数
import_pandas_datetime()

# 导入pandas私有库中的tslibs.strptime模块
from pandas._libs.tslibs.strptime cimport (
    DatetimeParseState,
    parse_today_now,
)

# 导入pandas私有库中的util模块
from pandas._libs.util cimport (
    is_float_object,
    is_integer_object,
)

# 导入pandas私有库中的tslibs.np_datetime模块中的特定异常
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime

# 导入pandas私有库中的tslibs.conversion模块中的特定函数和类型
from pandas._libs.tslibs.conversion cimport (
    _TSObject,
    cast_from_unit,
    convert_str_to_tsobject,
    convert_to_tsobject,
    get_datetime64_nanos,
    parse_pydatetime,
)

# 导入pandas私有库中的tslibs.dtypes模块中的特定函数和类型
from pandas._libs.tslibs.dtypes cimport (
    get_supported_reso,
    npy_unit_to_abbrev,
)

# 导入pandas私有库中的tslibs.nattype模块中的特定常量和字符串
from pandas._libs.tslibs.nattype cimport (
    NPY_NAT,
    c_nat_strings as nat_strings,
)

# 导入pandas私有库中的tslibs.timestamps模块中的特定类和函数
from pandas._libs.tslibs.timestamps cimport _Timestamp

# 导入pandas私有库中的tslibs模块中的特定类和函数
from pandas._libs.tslibs import (
    Resolution,
    get_resolution,
)

# 导入pandas私有库中的tslibs.timestamps模块中的Timestamp类
from pandas._libs.tslibs.timestamps import Timestamp

# 导入pandas私有库中的missing模块中的特定函数
from pandas._libs.missing cimport checknull_with_nat_and_na

# 导入pandas私有库中的tslibs.tzconversion模块中的特定函数
from pandas._libs.tslibs.tzconversion cimport tz_localize_to_utc_single


# 定义一个函数，用于解析ISO8601格式的时间字符串为Timestamp对象
def _test_parse_iso8601(ts: str):
    """
    TESTING ONLY: Parse string into Timestamp using iso8601 parser. Used
    only for testing, actual construction uses `convert_str_to_tsobject`
    """
    # 定义一个_TSObject对象
    cdef:
        _TSObject obj
        int out_local = 0, out_tzoffset = 0
        NPY_DATETIMEUNIT out_bestunit

    # 初始化_TSObject对象
    obj = _TSObject()

    # 调用string_to_dts函数解析时间字符串，填充_TSObject对象的dts属性
    string_to_dts(ts, &obj.dts, &out_bestunit, &out_local, &out_tzoffset, True)
    try:
        # 尝试将npy_datetimestruct转换为Python datetime对象
        obj.value = npy_datetimestruct_to_datetime(NPY_FR_ns, &obj.dts)
    except OverflowError as err:
        # 若转换失败，抛出OutOfBoundsDatetime异常
        raise OutOfBoundsDatetime(f"Out of bounds nanosecond timestamp: {ts}") from err
    if out_local == 1:
        # 若时间字符串包含本地时区信息，则创建时区对象并调整时间为UTC时间
        obj.tzinfo = timezone(timedelta(minutes=out_tzoffset))
        obj.value = tz_localize_to_utc_single(obj.value, obj.tzinfo)
        return Timestamp(obj.value, tz=obj.tzinfo)
    else:
        # 若时间字符串不包含本地时区信息，直接返回Timestamp对象
        return Timestamp(obj.value)


# 定义一个函数，用于将ndarray中的datetime数据格式化为字符串数组
@cython.wraparound(False)
@cython.boundscheck(False)
def format_array_from_datetime(
    ndarray values,
    tzinfo tz=None,
    str format=None,
    na_rep: str | float = "NaT",
    NPY_DATETIMEUNIT reso=NPY_FR_ns,
) -> np.ndarray:
    """
    return a np object array of the string formatted values

    Parameters
    ----------
    values : ndarray[int64_t], arbitrary ndim
    """
    # 返回一个包含格式化字符串值的np object数组
    pass
    # tz : tzinfo or None, default None
    # format : str or None, default None
    #       a strftime capable string
    # na_rep : optional, default is None
    #       a nat format
    # reso : NPY_DATETIMEUNIT, default NPY_FR_ns

    Returns
    -------
    np.ndarray[object]
    """
    cdef:
        # 定义变量
        int64_t val, ns, N = values.size
        # 控制微秒、毫秒、纳秒的显示
        bint show_ms = False, show_us = False, show_ns = False
        # 控制基本格式和日期格式的显示
        bint basic_format = False, basic_format_day = False
        _Timestamp ts
        # 存储结果的对象
        object res
        # 存储日期时间结构
        npy_datetimestruct dts

        # Note that `result` (and thus `result_flat`) is C-order and
        #  `it` iterates C-order as well, so the iteration matches
        #  See discussion at
        #  github.com/pandas-dev/pandas/pull/46886#discussion_r860261305
        # 创建一个空的对象数组，用于存储结果
        ndarray result = cnp.PyArray_EMPTY(values.ndim, values.shape, cnp.NPY_OBJECT, 0)
        # 将结果数组扁平化，但不产生副本
        object[::1] res_flat = result.ravel()     # should NOT be a copy
        # 创建一个扁平迭代器，以便按C顺序迭代数组
        cnp.flatiter it = cnp.PyArray_IterNew(values)

    if tz is None:
        # 如果没有时区信息

        # if we don't have a format nor tz, then choose
        # a format based on precision
        # 如果没有指定格式，则根据精度选择格式
        basic_format = format is None
        if basic_format:
            # 获取精度对象
            reso_obj = get_resolution(values, tz=tz, reso=reso)
            # 根据精度对象设置显示微秒、毫秒或纳秒
            show_ns = reso_obj == Resolution.RESO_NS
            show_us = reso_obj == Resolution.RESO_US
            show_ms = reso_obj == Resolution.RESO_MS

        elif format == "%Y-%m-%d %H:%M:%S":
            # 与默认格式相同，但精度硬编码为秒
            basic_format = True
            show_ns = show_us = show_ms = False

        elif format == "%Y-%m-%d %H:%M:%S.%f":
            # 与默认格式相同，但精度硬编码为微秒
            basic_format = show_us = True
            show_ns = show_ms = False

        elif format == "%Y-%m-%d":
            # 日期的默认格式
            basic_format_day = True

    assert not (basic_format_day and basic_format)
    for i in range(N):
        # 获取当前迭代位置的值
        val = (<int64_t*>cnp.PyArray_ITER_DATA(it))[0]

        if val == NPY_NAT:
            # 如果值为NPY_NAT，使用指定的替代字符串
            res = na_rep
        elif basic_format_day:
            # 如果使用基本日期格式，将 pandas datetime 转换为日期时间结构
            pandas_datetime_to_datetimestruct(val, reso, &dts)
            # 构建日期字符串，格式为"年-月-日"
            res = f"{dts.year}-{dts.month:02d}-{dts.day:02d}"
        elif basic_format:
            # 如果使用基本日期时间格式，将 pandas datetime 转换为日期时间结构
            pandas_datetime_to_datetimestruct(val, reso, &dts)
            # 构建日期时间字符串，格式为"年-月-日 时:分:秒"
            res = (f"{dts.year}-{dts.month:02d}-{dts.day:02d} "
                   f"{dts.hour:02d}:{dts.min:02d}:{dts.sec:02d}")
            if show_ns:
                # 如果需要显示纳秒，添加纳秒部分
                ns = dts.ps // 1000
                res += f".{ns + dts.us * 1000:09d}"
            elif show_us:
                # 如果需要显示微秒，添加微秒部分
                res += f".{dts.us:06d}"
            elif show_ms:
                # 如果需要显示毫秒，添加毫秒部分
                res += f".{dts.us // 1000:03d}"
        else:
            # 否则，使用 Timestamp 的值和分辨率生成 Timestamp 对象
            ts = Timestamp._from_value_and_reso(val, reso=reso, tz=tz)
            if format is None:
                # 如果格式为None，则使用 ISO 格式返回字符串表示
                res = str(ts)
            else:
                try:
                    # 尝试使用指定的格式化字符串格式化时间戳
                    res = ts.strftime(format)
                except ValueError:
                    # 如果格式化失败，同样使用 ISO 格式返回字符串表示
                    res = str(ts)

        # 将结果存储到结果数组中
        res_flat[i] = res

        # 移动到下一个迭代位置
        cnp.PyArray_ITER_NEXT(it)

    # 返回最终结果数组
    return result
@cython.wraparound(False)
@cython.boundscheck(False)
def first_non_null(values: ndarray) -> int:
    """Find position of first non-null value, return -1 if there isn't one."""
    # 声明并初始化变量n为values数组的长度
    cdef:
        Py_ssize_t n = len(values)
        # 声明Py_ssize_t类型的变量i
        Py_ssize_t i
    # 遍历数组values中的元素
    for i in range(n):
        # 获取values中索引为i的值赋给变量val
        val = values[i]
        # 检查val是否为空值，如果是则继续下一个循环
        if checknull_with_nat_and_na(val):
            continue
        # 检查val是否为字符串类型，且长度为0或者在特定字符串集合中
        if (
            isinstance(val, str)
            and
            (len(val) == 0 or val in nat_strings or val in ("now", "today"))
        ):
            continue
        # 如果以上条件均不满足，则返回当前索引i作为第一个非空值的位置
        return i
    else:
        # 若数组中所有值均为空，则返回-1
        return -1


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef array_to_datetime(
    ndarray values,  # object dtype, arbitrary ndim
    str errors="raise",
    bint dayfirst=False,
    bint yearfirst=False,
    bint utc=False,
    NPY_DATETIMEUNIT creso=NPY_DATETIMEUNIT.NPY_FR_GENERIC,
    str unit_for_numerics=None,
):
    """
    Converts a 1D array of date-like values to a numpy array of either:
        1) datetime64[ns] data
        2) datetime.datetime objects, if OutOfBoundsDatetime or TypeError
           is encountered

    Also returns a fixed-offset tzinfo object if an array of strings with the same
    timezone offset is passed and utc=True is not passed. Otherwise, None
    is returned

    Handles datetime.date, datetime.datetime, np.datetime64 objects, numeric,
    strings

    Parameters
    ----------
    values : ndarray of object
        date-like objects to convert
    errors : str, default 'raise'
        error behavior when parsing
    dayfirst : bool, default False
        dayfirst parsing behavior when encountering datetime strings
    yearfirst : bool, default False
        yearfirst parsing behavior when encountering datetime strings
    utc : bool, default False
        indicator whether the dates should be UTC
    creso : NPY_DATETIMEUNIT, default NPY_FR_GENERIC
        If NPY_FR_GENERIC, conduct inference.
    unit_for_numerics : str, default "ns"

    Returns
    -------
    np.ndarray
        May be datetime64[creso_unit] or object dtype
    tzinfo or None
    """
    # 声明并初始化变量n为values数组的大小
    cdef:
        Py_ssize_t i, n = values.size
        # 声明val为对象类型的变量
        object val
        # 声明iresult为int64_t类型的数组
        ndarray[int64_t] iresult
        # 声明dts为npy_datetimestruct类型的变量
        npy_datetimestruct dts
        # 声明并初始化变量utc_convert为utc的布尔值
        bint utc_convert = bool(utc)
        # 声明并初始化变量is_raise为errors是否等于"raise"的布尔值
        bint is_raise = errors == "raise"
        # 声明并初始化变量is_coerce为errors是否等于"coerce"的布尔值
        bint is_coerce = errors == "coerce"
        # 声明并初始化变量tsobj为_TSObject对象
        _TSObject tsobj
        # 声明tz和tz_out为tzinfo类型的变量，默认为None
        tzinfo tz, tz_out = None
        # 声明it为values的迭代器
        cnp.flatiter it = cnp.PyArray_IterNew(values)
        # 声明item_reso为NPY_DATETIMEUNIT类型的变量
        NPY_DATETIMEUNIT item_reso
        # 声明并初始化infer_reso为creso是否为NPY_DATETIMEUNIT.NPY_FR_GENERIC的布尔值
        bint infer_reso = creso == NPY_DATETIMEUNIT.NPY_FR_GENERIC
        # 声明并初始化state为DatetimeParseState对象，使用creso作为参数
        DatetimeParseState state = DatetimeParseState(creso)
        # 声明并初始化abbrev为空字符串
        str abbrev

    # 确保错误处理条件满足is_raise或is_coerce至少一个为True
    assert is_raise or is_coerce

    # 根据infer_reso的值设置abbrev的值为"ns"或者根据creso获取对应的时间单位缩写
    if infer_reso:
        abbrev = "ns"
    else:
        abbrev = npy_unit_to_abbrev(creso)

    # 如果unit_for_numerics不为None，则确保只有creso或unit_for_numerics其中一个被传递
    if unit_for_numerics is not None:
        assert creso == NPY_FR_ns
    else:
        unit_for_numerics = abbrev
    # 创建一个空的 NumPy 数组，形状与输入对象 values 相同，数据类型为指定的日期时间类型
    result = np.empty((<object>values).shape, dtype=f"M8[{abbrev}]")
    # 将 result 视图转换为 int64 类型的数组，并展平为一维数组
    iresult = result.view("i8").ravel()

    # 检查并处理混合输入时区的情况，返回处理后的时区信息 tz_out
    tz_out = state.check_for_mixed_inputs(tz_out, utc)

    # 如果需要推断时间分辨率
    if infer_reso:
        # 如果先前遇到过时间分辨率的变化
        if state.creso_ever_changed:
            # 遇到了分辨率不匹配的情况，需要使用正确的分辨率重新解析
            return array_to_datetime(
                values,
                errors=errors,
                yearfirst=yearfirst,
                dayfirst=dayfirst,
                utc=utc,
                creso=state.creso,
            )
        # 如果分辨率为通用的 NPY_FR_GENERIC
        elif state.creso == NPY_DATETIMEUNIT.NPY_FR_GENERIC:
            # 即未遇到非 NaT 的情况，使用默认的 "s" 分辨率，确保 NaT 的插入和连接操作不会提升单位
            result = iresult.view("M8[s]").reshape(result.shape)
        else:
            # 否则使用遇到的单一分辨率，避免进行第二次遍历
            abbrev = npy_unit_to_abbrev(state.creso)
            result = iresult.view(f"M8[{abbrev}]").reshape(result.shape)
    
    # 返回处理后的结果数组 result 和处理后的时区信息 tz_out
    return result, tz_out
# 将数组转换为带有时区信息的日期时间数组
def array_to_datetime_with_tz(
    # 输入参数：包含日期时间值的数组，时区信息，是否日优先，是否年优先，日期时间单位
    ndarray values, tzinfo tz, bint dayfirst, bint yearfirst, NPY_DATETIMEUNIT creso
):
    """
    Vectorized analogue to pd.Timestamp(value, tz=tz)

    values has object-dtype, unrestricted ndim.

    Major differences between this and array_to_datetime with utc=True
        - np.datetime64 objects are treated as _wall_ times.
        - tznaive datetimes are treated as _wall_ times.
    """
    cdef:
        # 创建空的 int64 数组，用于存储结果
        ndarray result = cnp.PyArray_EMPTY(values.ndim, values.shape, cnp.NPY_INT64, 0)
        # 创建多迭代器，用于同时迭代结果数组和输入数组
        cnp.broadcast mi = cnp.PyArray_MultiIterNew2(result, values)
        # 初始化变量 i 和 n，分别表示迭代器和值数组的大小
        Py_ssize_t i, n = values.size
        # 声明变量 item 为对象类型
        object item
        # 声明变量 ival 为 int64 类型
        int64_t ival
        # 声明变量 tsobj 为 _TSObject 类型
        _TSObject tsobj
        # 声明变量 infer_reso 为布尔类型，表示是否推断日期时间单位
        bint infer_reso = creso == NPY_DATETIMEUNIT.NPY_FR_GENERIC
        # 声明变量 state 为 DatetimeParseState 类型，使用给定的日期时间单位初始化
        DatetimeParseState state = DatetimeParseState(creso)
        # 声明变量 abbrev 为字符串类型，表示日期时间单位的缩写

    if infer_reso:
        # 如果需要推断日期时间单位，将 abbrev 设置为 "ns" 表示纳秒
        abbrev = "ns"
    else:
        # 否则调用函数 npy_unit_to_abbrev 获取日期时间单位的缩写
        abbrev = npy_unit_to_abbrev(creso)

    for i in range(n):
        # 从输入数组中获取当前项的对象
        item = <object>(<PyObject**>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

        if checknull_with_nat_and_na(item):
            # 检查当前项是否为 null，如果是，则设置 ival 为 NPY_NAT
            ival = NPY_NAT

        else:
            # 否则将当前项转换为 tsobj 对象，包括时区信息和日期时间单位等参数
            tsobj = convert_to_tsobject(
                item,
                tz=tz,
                unit=abbrev,
                dayfirst=dayfirst,
                yearfirst=yearfirst,
                nanos=0,
            )
            if tsobj.value != NPY_NAT:
                # 如果转换成功，则更新日期时间单位的状态
                state.update_creso(tsobj.creso)
                if infer_reso:
                    # 如果正在推断日期时间单位，则更新 creso
                    creso = state.creso
                # 确保 tsobj 的日期时间单位符合预期
                tsobj.ensure_reso(creso, item, round_ok=True)
            # 获取 tsobj 的数值表示赋给 ival
            ival = tsobj.value

        # 将 ival 赋值给结果数组中的当前索引位置
        (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = ival

        # 移动多迭代器到下一个位置
        cnp.PyArray_MultiIter_NEXT(mi)

    if infer_reso:
        if state.creso_ever_changed:
            # 如果推断日期时间单位过程中发生了变化，需要使用正确的单位重新解析
            return array_to_datetime_with_tz(
                values, tz=tz, dayfirst=dayfirst, yearfirst=yearfirst, creso=creso
            )
        elif creso == NPY_DATETIMEUNIT.NPY_FR_GENERIC:
            # 否则，如果没有遇到非 NaT 的日期时间值，默认使用 "s" 作为单位，确保不会上升单位
            result = result.view("M8[s]")
        else:
            # 否则，使用遇到的单一日期时间单位，避免进行第二次解析
            abbrev = npy_unit_to_abbrev(creso)
            result = result.view(f"M8[{abbrev}]")
    else:
        # 如果不推断日期时间单位，则根据给定的 creso 设置结果数组的单位
        abbrev = npy_unit_to_abbrev(creso)
        result = result.view(f"M8[{abbrev}]")
    # 返回结果数组
    return result
```