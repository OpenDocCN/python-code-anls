# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\vectorized.pyx`

```
cimport cython
cimport numpy as cnp
from cpython.datetime cimport (
    date,               # 导入日期类
    datetime,           # 导入日期时间类
    time,               # 导入时间类
    tzinfo,             # 导入时区信息类
)
from numpy cimport (
    int64_t,            # 导入64位整数类型
    ndarray,            # 导入多维数组类型
)

cnp.import_array()      # 导入numpy C API

from .dtypes import Resolution  # 从自定义模块中导入Resolution类型

from .dtypes cimport (
    c_Resolution,       # 从自定义模块中导入C定义的Resolution类型
    periods_per_day,    # 从自定义模块中导入每天周期数
)
from .nattype cimport (
    NPY_NAT,            # 从自定义模块中导入缺失值标志
    c_NaT as NaT,       # 从自定义模块中导入缺失值别名
)
from .np_datetime cimport (
    NPY_DATETIMEUNIT,           # 从自定义模块中导入日期时间单位
    NPY_FR_ns,                  # 从自定义模块中导入纳秒单位
    import_pandas_datetime,     # 从自定义模块中导入导入pandas日期时间函数
    npy_datetimestruct,         # 从自定义模块中导入日期时间结构
    pandas_datetime_to_datetimestruct,  # 从自定义模块中导入将pandas日期时间转换为日期时间结构的函数
)

import_pandas_datetime()       # 导入pandas日期时间

from .period cimport get_period_ordinal   # 从自定义模块中导入获取周期序数的函数
from .timestamps cimport create_timestamp_from_ts  # 从自定义模块中导入根据时间戳创建时间戳的函数
from .timezones cimport is_utc   # 从自定义模块中导入检查是否为UTC时间的函数
from .tzconversion cimport Localizer  # 从自定义模块中导入时区本地化类


@cython.boundscheck(False)
@cython.wraparound(False)
def tz_convert_from_utc(ndarray stamps, tzinfo tz, NPY_DATETIMEUNIT reso=NPY_FR_ns):
    # stamps is int64_t, arbitrary ndim
    """
    Convert the values (in i8) from UTC to tz

    Parameters
    ----------
    stamps : ndarray[int64]
        包含时间戳的整数数组
    tz : tzinfo
        目标时区信息
    reso : NPY_DATETIMEUNIT, optional
        日期时间精度，默认为纳秒

    Returns
    -------
    ndarray[int64]
        转换后的时间戳数组
    """
    if tz is None or is_utc(tz) or stamps.size == 0:
        # Much faster than going through the "standard" pattern below;
        #  do this before initializing Localizer.
        return stamps.copy()

    cdef:
        Localizer info = Localizer(tz, creso=reso)  # 使用给定的时区信息创建本地化对象
        int64_t utc_val, local_val  # 定义UTC时间和本地时间的变量
        Py_ssize_t pos, i, n = stamps.size  # 获取时间戳数组的大小

        ndarray result   # 定义结果数组
        cnp.broadcast mi  # 定义多迭代器对象

    result = cnp.PyArray_EMPTY(stamps.ndim, stamps.shape, cnp.NPY_INT64, 0)  # 创建一个空的int64类型数组
    mi = cnp.PyArray_MultiIterNew2(result, stamps)  # 使用结果数组和时间戳数组创建多迭代器对象

    for i in range(n):
        # Analogous to: utc_val = stamps[i]
        utc_val = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]  # 获取时间戳数组中当前索引处的UTC时间值

        if utc_val == NPY_NAT:
            local_val = NPY_NAT  # 如果UTC时间值为缺失值标志，则本地时间值也为缺失值标志
        else:
            local_val = info.utc_val_to_local_val(utc_val, &pos)  # 否则，使用本地化对象将UTC时间转换为本地时间

        # Analogous to: result[i] = local_val
        (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = local_val  # 将计算得到的本地时间值写入结果数组的对应位置

        cnp.PyArray_MultiIter_NEXT(mi)  # 移动多迭代器对象到下一个位置

    return result


# -------------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
def ints_to_pydatetime(
    ndarray stamps,
    tzinfo tz=None,
    str box="datetime",
    NPY_DATETIMEUNIT reso=NPY_FR_ns,
) -> ndarray:
    # stamps is int64, arbitrary ndim
    """
    Convert an i8 repr to an ndarray of datetimes, date, time or Timestamp.

    Parameters
    ----------
    stamps : array of i8
        包含时间戳的整数数组
    tz : str, optional
        目标时区字符串，可选
    box : {'datetime', 'timestamp', 'date', 'time'}, default 'datetime'
        转换为的目标类型：
        * 如果是 'datetime'，转换为 datetime.datetime
        * 如果是 'timestamp'，转换为 pandas.Timestamp
        * 如果是 'date'，转换为 datetime.date
        * 如果是 'time'，转换为 datetime.time
    reso : NPY_DATETIMEUNIT, optional
        日期时间精度，默认为纳秒

    Returns
    -------
    ndarray[object] of type specified by box
        根据box参数指定类型的对象数组
    """
    cdef:
        // 创建一个本地化器对象 `info`，使用指定的时区 `tz` 和分辨率 `reso`
        Localizer info = Localizer(tz, creso=reso)
        // 定义整型变量 `utc_val` 和 `local_val`，以及迭代器 `i` 和数组大小 `n`
        int64_t utc_val, local_val
        Py_ssize_t i, n = stamps.size
        // 初始化 `pos` 为未使用状态，避免未初始化警告
        Py_ssize_t pos = -1  // unused, avoid not-initialized warning

        // 定义日期时间结构 `dts` 和时区信息 `new_tz`
        npy_datetimestruct dts
        tzinfo new_tz
        // 标志变量，指示是否使用日期、时间戳或Python日期时间对象
        bint use_date = False, use_ts = False, use_pydt = False
        // 结果值对象 `res_val` 和折叠标志 `fold`
        object res_val
        bint fold = 0

        // 注意：`result` 和 `result_flat` 是按 C 顺序排列的，与 `it` 的迭代方式相匹配
        // 详细讨论请参见：
        // github.com/pandas-dev/pandas/pull/46886#discussion_r860261305
        // 创建空的对象数组 `result`，形状与 `stamps` 相同
        ndarray result = cnp.PyArray_EMPTY(stamps.ndim, stamps.shape, cnp.NPY_OBJECT, 0)
        // 将 `result` 展平为一维数组 `res_flat`，但不会复制数据
        object[::1] res_flat = result.ravel()     // should NOT be a copy
        // 创建按 C 顺序迭代的迭代器 `it`
        cnp.flatiter it = cnp.PyArray_IterNew(stamps)

    // 根据 `box` 的值设置相应的使用标志
    if box == "date":
        // 当转换为日期时，要求 `tz` 必须为 None
        assert (tz is None), "tz should be None when converting to date"
        use_date = True
    elif box == "timestamp":
        use_ts = True
    elif box == "datetime":
        use_pydt = True
    elif box != "time":
        // 若 `box` 不是 'datetime', 'date', 'time' 或 'timestamp'，抛出值错误
        raise ValueError(
            "box must be one of 'datetime', 'date', 'time' or 'timestamp'"
        )

    // 遍历 `stamps` 数组的每个元素
    for i in range(n):
        // 获取当前元素 `stamps[i]`，相当于 `utc_val = stamps[i]`
        utc_val = (<int64_t*>cnp.PyArray_ITER_DATA(it))[0]

        // 初始化 `new_tz` 为指定的时区 `tz`
        new_tz = tz

        // 如果当前时间戳为 `NPY_NAT`，则结果值为 NaT
        if utc_val == NPY_NAT:
            res_val = <object>NaT
        else:
            // 将 UTC 时间转换为本地时间 `local_val`，并更新 `pos` 和 `fold`
            local_val = info.utc_val_to_local_val(utc_val, &pos, &fold)
            // 如果使用了 `pytz`，找到正确的 `pytz` 时区表示
            if info.use_pytz:
                new_tz = tz._tzinfos[tz._transition_info[pos]]

            // 将 Pandas 的日期时间结构转换为 `dts` 结构
            pandas_datetime_to_datetimestruct(local_val, reso, &dts)

            // 根据使用情况创建相应的时间戳、日期时间对象或日期对象
            if use_ts:
                res_val = create_timestamp_from_ts(
                    utc_val, dts, new_tz, fold, reso=reso
                )
            elif use_pydt:
                res_val = datetime(
                    dts.year, dts.month, dts.day, dts.hour, dts.min, dts.sec, dts.us,
                    new_tz, fold=fold,
                )
            elif use_date:
                res_val = date(dts.year, dts.month, dts.day)
            else:
                res_val = time(dts.hour, dts.min, dts.sec, dts.us, new_tz, fold=fold)

        // 将计算得到的结果值 `res_val` 存储到结果数组 `result` 的对应位置 `i`
        // 注意：由于 `result` 是已知的 C 顺序数组，可以直接使用索引存储结果，而不需要使用 PyArray_MultiIter_DATA
        res_flat[i] = res_val

        // 移动 `it` 迭代器到下一个元素
        cnp.PyArray_ITER_NEXT(it)

    // 返回填充好的结果数组 `result`
    return result
# -------------------------------------------------------------------------
cdef c_Resolution _reso_stamp(npy_datetimestruct *dts):
    # 根据给定的时间结构体dts，确定时间分辨率并返回相应的枚举值
    if dts.ps != 0:
        return c_Resolution.RESO_NS  # 纳秒级分辨率
    elif dts.us != 0:
        if dts.us % 1000 == 0:
            return c_Resolution.RESO_MS  # 毫秒级分辨率
        return c_Resolution.RESO_US  # 微秒级分辨率
    elif dts.sec != 0:
        return c_Resolution.RESO_SEC  # 秒级分辨率
    elif dts.min != 0:
        return c_Resolution.RESO_MIN  # 分钟级分辨率
    elif dts.hour != 0:
        return c_Resolution.RESO_HR  # 小时级分辨率
    return c_Resolution.RESO_DAY  # 天级分辨率

@cython.wraparound(False)
@cython.boundscheck(False)
def get_resolution(
    ndarray stamps, tzinfo tz=None, NPY_DATETIMEUNIT reso=NPY_FR_ns
) -> Resolution:
    # 获取时间戳数组的最低分辨率
    # stamps是int64类型的数组，可能是任意维度
    cdef:
        Localizer info = Localizer(tz, creso=reso)  # 创建本地化器对象
        int64_t utc_val, local_val  # 声明UTC和本地时间戳变量
        Py_ssize_t i, n = stamps.size  # 获取数组大小
        Py_ssize_t pos = -1  # 未使用的变量，避免未初始化警告
        cnp.flatiter it = cnp.PyArray_IterNew(stamps)  # 创建NumPy数组迭代器

        npy_datetimestruct dts  # 创建时间结构体对象
        c_Resolution pd_reso = c_Resolution.RESO_DAY, curr_reso  # 初始化pandas和当前分辨率

    for i in range(n):
        # 从数组中获取UTC时间戳，并进行处理
        utc_val = (<int64_t*>cnp.PyArray_ITER_DATA(it))[0]

        if utc_val == NPY_NAT:
            pass  # 如果时间戳为NPY_NAT，则跳过处理
        else:
            local_val = info.utc_val_to_local_val(utc_val, &pos)  # 转换UTC时间戳到本地时间戳
            pandas_datetime_to_datetimestruct(local_val, reso, &dts)  # 将Pandas日期时间转换为时间结构体
            curr_reso = _reso_stamp(&dts)  # 获取当前时间结构体的分辨率
            if curr_reso < pd_reso:
                pd_reso = curr_reso  # 更新最低分辨率

        cnp.PyArray_ITER_NEXT(it)  # 移动到数组的下一个元素

    return Resolution(pd_reso)  # 返回最低分辨率对象

# -------------------------------------------------------------------------

@cython.cdivision(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cpdef ndarray normalize_i8_timestamps(ndarray stamps, tzinfo tz, NPY_DATETIMEUNIT reso):
    """
    对给定数组中的每个（纳秒级）时区感知时间戳进行归一化处理，
    将其向下舍入到当天的开始（即午夜）。
    这里的午夜是特定时区‘tz’的午夜。

    Parameters
    ----------
    stamps : int64 ndarray
        包含时间戳的数组
    tz : tzinfo or None
        时区信息或None
    reso : NPY_DATETIMEUNIT
        时间戳的单位

    Returns
    -------
    result : int64 ndarray
        被归一化为纳秒级时间戳的数组
    """
    cdef:
        Localizer info = Localizer(tz, creso=reso)  # 创建本地化器对象
        int64_t utc_val, local_val, res_val  # 声明UTC时间戳、本地时间戳和结果值变量
        Py_ssize_t i, n = stamps.size  # 获取数组大小
        Py_ssize_t pos = -1  # 未使用的变量，避免未初始化警告

        ndarray result = cnp.PyArray_EMPTY(stamps.ndim, stamps.shape, cnp.NPY_INT64, 0)  # 创建空的结果数组
        cnp.broadcast mi = cnp.PyArray_MultiIterNew2(result, stamps)  # 创建广播迭代器
        int64_t ppd = periods_per_day(reso)  # 计算每天的时间段数
    # 遍历范围为n的循环
    for i in range(n):
        # 从cnp.PyArray_MultiIter_DATA(mi, 1)中获取第i个元素，赋值给utc_val
        utc_val = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

        # 如果utc_val等于NPY_NAT，则res_val也为NPY_NAT
        if utc_val == NPY_NAT:
            res_val = NPY_NAT
        else:
            # 调用info.utc_val_to_local_val方法，将utc_val转换为本地时间local_val
            local_val = info.utc_val_to_local_val(utc_val, &pos)
            # 计算res_val为local_val减去local_val与ppd的余数
            res_val = local_val - (local_val % ppd)

        # 将res_val赋值给cnp.PyArray_MultiIter_DATA(mi, 0)的第i个元素
        (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = res_val

        # 移动到下一个迭代器位置
        cnp.PyArray_MultiIter_NEXT(mi)

    # 返回结果数组result
    return result
@cython.wraparound(False)
@cython.boundscheck(False)
def is_date_array_normalized(ndarray stamps, tzinfo tz, NPY_DATETIMEUNIT reso) -> bool:
    # 禁止 Cython 对数组的负数索引包装和边界检查
    # stamps 是 int64_t 类型的数组，任意维度
    """
    检查所有给定的（纳秒级）时间戳是否已经标准化为午夜，即小时 == 分钟 == 秒 == 0。
    如果可选的时区 `tz` 不是 None，则表示该时区的午夜时间。

    Parameters
    ----------
    stamps : int64 ndarray
        时间戳数组
    tz : tzinfo or None
        时区信息或者 None
    reso : NPY_DATETIMEUNIT
        时间单位信息

    Returns
    -------
    is_normalized : bool
        如果所有时间戳都已标准化则返回 True
    """
    cdef:
        # 创建 Localizer 对象，用于处理时区和时间分辨率
        Localizer info = Localizer(tz, creso=reso)
        # 声明变量并初始化，用于迭代 stamps 数组
        int64_t utc_val, local_val
        Py_ssize_t i, n = stamps.size
        Py_ssize_t pos = -1  # 未使用，避免未初始化的警告
        cnp.flatiter it = cnp.PyArray_IterNew(stamps)
        # 计算每天的周期数
        int64_t ppd = periods_per_day(reso)

    for i in range(n):
        # 类似于：utc_val = stamps[i]
        utc_val = (<int64_t*>cnp.PyArray_ITER_DATA(it))[0]

        # 将 UTC 时间转换为本地时间
        local_val = info.utc_val_to_local_val(utc_val, &pos)

        # 检查本地时间是否标准化
        if local_val % ppd != 0:
            return False

        # 移动到 stamps 数组的下一个元素
        cnp.PyArray_ITER_NEXT(it)

    return True


# -------------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
def dt64arr_to_periodarr(
    ndarray stamps, int freq, tzinfo tz, NPY_DATETIMEUNIT reso=NPY_FR_ns
):
    # stamps 是 int64_t 类型的数组，任意维度
    cdef:
        # 创建 Localizer 对象，用于处理时区和时间分辨率
        Localizer info = Localizer(tz, creso=reso)
        Py_ssize_t i, n = stamps.size
        Py_ssize_t pos = -1  # 未使用，避免未初始化的警告
        int64_t utc_val, local_val, res_val

        # 用于保存结果的数组，与 stamps 的维度和形状相同
        ndarray result = cnp.PyArray_EMPTY(stamps.ndim, stamps.shape, cnp.NPY_INT64, 0)
        # 创建 MultiIterator 以便同时迭代 result 和 stamps 两个数组
        cnp.broadcast mi = cnp.PyArray_MultiIterNew2(result, stamps)

    for i in range(n):
        # 类似于：utc_val = stamps[i]
        utc_val = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

        if utc_val == NPY_NAT:
            res_val = NPY_NAT
        else:
            # 将 UTC 时间转换为本地时间
            local_val = info.utc_val_to_local_val(utc_val, &pos)
            # 将 pandas datetime 转换为 datetimestruct
            pandas_datetime_to_datetimestruct(local_val, reso, &dts)
            # 获取时间段的序数
            res_val = get_period_ordinal(&dts, freq)

        # 类似于：result[i] = res_val
        (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = res_val

        # 移动到 result 和 stamps 数组的下一个元素
        cnp.PyArray_MultiIter_NEXT(mi)

    return result
```