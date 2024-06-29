# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\tzconversion.pyx`

```
"""
timezone conversion
"""
# 导入Cython模块
cimport cython
# 从Cython模块导入datetime相关C API
from cpython.datetime cimport (
    PyDelta_Check,
    datetime,
    datetime_new,
    import_datetime,
    timedelta,
    tzinfo,
)
# 调用import_datetime函数，初始化Python的datetime模块
import_datetime()

# 导入NumPy库
import numpy as np
# 导入pytz库
import pytz

# 从Cython导入NumPy相关的C API
cimport numpy as cnp
from numpy cimport (
    int64_t,
    intp_t,
    ndarray,
    uint8_t,
)
# 调用cnp.import_array()函数，初始化NumPy的C API
cnp.import_array()

# 从pandas._libs.tslibs.dtypes导入特定数据类型
from pandas._libs.tslibs.dtypes cimport (
    periods_per_day,
    periods_per_second,
)
# 从pandas._libs.tslibs.nattype导入NPY_NAT
from pandas._libs.tslibs.nattype cimport NPY_NAT
# 从pandas._libs.tslibs.np_datetime导入日期时间相关的C API
from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,
    import_pandas_datetime,
    npy_datetimestruct,
    pandas_datetime_to_datetimestruct,
    pydatetime_to_dt64,
)
# 调用import_pandas_datetime()函数，初始化pandas的日期时间处理模块
import_pandas_datetime()

# 从pandas._libs.tslibs.timezones导入时区相关的C API
from pandas._libs.tslibs.timezones cimport (
    get_dst_info,
    is_fixed_offset,
    is_tzlocal,
    is_utc,
    is_zoneinfo,
    utc_stdlib,
)

# 定义常量数组_deltas_placeholder
cdef const int64_t[::1] _deltas_placeholder = np.array([], dtype=np.int64)

# 定义Cython类Localizer，标记为freelist大小为16，final类型
@cython.freelist(16)
@cython.final
cdef class Localizer:
    # 定义类成员变量，使用Cython的cdef关键字声明类型
    # cdef:
    #    tzinfo tz
    #    NPY_DATETIMEUNIT _creso
    #    bint use_utc, use_fixed, use_tzlocal, use_dst, use_pytz
    #    ndarray trans
    #    Py_ssize_t ntrans
    #    const int64_t[::1] deltas
    #    int64_t delta
    #    int64_t* tdata

    # 标记为未初始化检查，禁用边界检查
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    # 初始化函数，设置时区信息和时间单位
    def __cinit__(self, tzinfo tz, NPY_DATETIMEUNIT creso):
        self.tz = tz  # 设置时区信息
        self._creso = creso  # 设置时间单位
        self.use_utc = self.use_tzlocal = self.use_fixed = False  # 初始化标志位，默认为False
        self.use_dst = self.use_pytz = False  # 初始化标志位，默认为False
        self.ntrans = -1  # 转换点数量的占位符
        self.delta = -1  # 时间增量的占位符
        self.deltas = _deltas_placeholder  # 时间增量的占位符
        self.tdata = NULL  # 时间数据的占位符

        # 根据时区类型设置标志位
        if is_utc(tz) or tz is None:
            self.use_utc = True
        elif is_tzlocal(tz) or is_zoneinfo(tz):
            self.use_tzlocal = True
        else:
            # 获取夏令时信息
            trans, deltas, typ = get_dst_info(tz)

            # 根据时间单位调整夏令时信息的精度
            if creso != NPY_DATETIMEUNIT.NPY_FR_ns:
                # 使用地板除法假设 trans 和 deltas 是整数秒数
                if creso == NPY_DATETIMEUNIT.NPY_FR_us:
                    trans = np.array(trans) // 1_000
                    deltas = np.array(deltas) // 1_000
                elif creso == NPY_DATETIMEUNIT.NPY_FR_ms:
                    trans = np.array(trans) // 1_000_000
                    deltas = np.array(deltas) // 1_000_000
                elif creso == NPY_DATETIMEUNIT.NPY_FR_s:
                    trans = np.array(trans) // 1_000_000_000
                    deltas = np.array(deltas) // 1_000_000_000
                else:
                    raise NotImplementedError(creso)

            # 设置夏令时相关属性
            self.trans = trans
            self.ntrans = self.trans.shape[0]  # 转换点数量
            self.deltas = deltas

            # 根据 typ 设置使用静态/固定时间或者动态夏令时
            if typ != "pytz" and typ != "dateutil":
                self.use_fixed = True
                self.delta = deltas[0]  # 固定时间增量
            else:
                self.use_dst = True
                if typ == "pytz":
                    self.use_pytz = True
                self.tdata = <int64_t*>cnp.PyArray_DATA(trans)  # 动态夏令时数据

    # 定义函数，将 UTC 时间值转换为本地时间值
    @cython.boundscheck(False)
    cdef int64_t utc_val_to_local_val(
        self, int64_t utc_val, Py_ssize_t* pos, bint* fold=NULL
    ) except? -1:
        if self.use_utc:
            return utc_val  # 如果使用 UTC，直接返回 UTC 时间值
        elif self.use_tzlocal:
            # 使用 tzinfo API 将 UTC 时间值本地化
            return utc_val + _tz_localize_using_tzinfo_api(
                utc_val, self.tz, to_utc=False, creso=self._creso, fold=fold
            )
        elif self.use_fixed:
            return utc_val + self.delta  # 如果使用固定时间增量，加上固定增量
        else:
            # 使用二分查找找到合适的转换点索引
            pos[0] = bisect_right_i8(self.tdata, utc_val, self.ntrans) - 1
            if fold is not NULL:
                # 推断 dateutil 的 fold 属性
                fold[0] = _infer_dateutil_fold(
                    utc_val, self.trans, self.deltas, pos[0]
                )
            # 根据转换点索引调整时间值
            return utc_val + self.deltas[pos[0]]
cdef int64_t tz_localize_to_utc_single(
    int64_t val,
    tzinfo tz,
    object ambiguous=None,
    object nonexistent=None,
    NPY_DATETIMEUNIT creso=NPY_DATETIMEUNIT.NPY_FR_ns,
) except? -1:
    """See tz_localize_to_utc.__doc__"""
    cdef:
        int64_t delta  # 用于存储时间偏移量的变量
        int64_t[::1] deltas  # 存储多个时间偏移量的数组

    if val == NPY_NAT:
        return val  # 如果输入的时间值为 NaT，则直接返回 NaT

    elif is_utc(tz) or tz is None:
        return val  # 如果时区为 UTC 或者为 None，则直接返回输入的时间值

    elif is_tzlocal(tz):
        return val - _tz_localize_using_tzinfo_api(val, tz, to_utc=True, creso=creso)
        # 如果时区为本地时区，调用 _tz_localize_using_tzinfo_api 函数进行本地化，并返回调整后的时间值

    elif is_fixed_offset(tz):
        _, deltas, _ = get_dst_info(tz)
        delta = deltas[0]
        # 获取固定偏移量时区的时间偏移量

        # 根据 creso 的精度调整时间偏移量
        if creso != NPY_DATETIMEUNIT.NPY_FR_ns:
            if creso == NPY_DATETIMEUNIT.NPY_FR_us:
                delta = delta // 1000
            elif creso == NPY_DATETIMEUNIT.NPY_FR_ms:
                delta = delta // 1_000_000
            elif creso == NPY_DATETIMEUNIT.NPY_FR_s:
                delta = delta // 1_000_000_000

        return val - delta  # 根据计算得到的时间偏移量进行时间值的调整

    else:
        return tz_localize_to_utc(
            np.array([val], dtype="i8"),
            tz,
            ambiguous=ambiguous,
            nonexistent=nonexistent,
            creso=creso,
        )[0]
        # 对于其他类型的时区，调用 tz_localize_to_utc 函数进行处理，并返回处理后的第一个元素


@cython.boundscheck(False)
@cython.wraparound(False)
def tz_localize_to_utc(
    ndarray[int64_t] vals,
    tzinfo tz,
    object ambiguous=None,
    object nonexistent=None,
    NPY_DATETIMEUNIT creso=NPY_DATETIMEUNIT.NPY_FR_ns,
):
    """
    Localize tzinfo-naive i8 to given time zone (using pytz). If
    there are ambiguities in the values, raise AmbiguousTimeError.

    Parameters
    ----------
    vals : ndarray[int64_t]
        待本地化的时间值数组
    tz : tzinfo or None
        目标时区信息或 None
    ambiguous : str, bool, or arraylike
        处理时间重复时的行为参数
    nonexistent : {None, "NaT", "shift_forward", "shift_backward", "raise", \
timedelta-like}
        处理不存在时间的行为参数
    creso : NPY_DATETIMEUNIT, default NPY_FR_ns
        时间精度单位

    Returns
    -------
    localized : ndarray[int64_t]
        本地化后的时间值数组
    """

    if tz is None or is_utc(tz) or vals.size == 0:
        # 如果目标时区为 None 或者为 UTC，或者输入的时间值数组为空，则直接返回原始数组
        return vals.copy()
    cdef:
        # 定义一个 C 扩展类型的变量 ambiguous_array，用于存储布尔类型的数组
        ndarray[uint8_t, cast=True] ambiguous_array
        # 使用 Python 中的 ssize_t 类型定义变量 i 和 n，vals 数组的行数
        Py_ssize_t i, n = vals.shape[0]
        # 定义变量 delta_idx_offset 和 delta_idx，均为 Py_ssize_t 类型
        Py_ssize_t delta_idx_offset, delta_idx
        # 定义整型变量 v, left, right, val, new_local, remaining_mins
        int64_t v, left, right, val, new_local, remaining_mins
        # 定义整型变量 first_delta 和 delta，初始化为 0
        int64_t first_delta, delta
        # 初始化 shift_delta 为整型变量，值为 0
        int64_t shift_delta = 0
        # 定义整型数组 result_a, result_b, dst_hours
        ndarray[int64_t] result_a, result_b, dst_hours
        # 定义一维整型数组 result
        int64_t[::1] result
        # 布尔类型变量 is_zi 初始化为 False
        bint is_zi = False
        # 布尔类型变量 infer_dst, is_dst, fill 初始化为 False
        bint infer_dst = False, is_dst = False, fill = False
        # 布尔类型变量 shift_forward, shift_backward, fill_nonexist 初始化为 False
        bint shift_forward = False, shift_backward = False
        bint fill_nonexist = False
        # 字符串类型变量 stamp 初始化为空字符串
        str stamp
        # 创建 Localizer 对象 info，传入参数 tz 和 creso
        Localizer info = Localizer(tz, creso=creso)
        # 计算 periods per hour 和 periods per second，分别赋值给 pph 和 pps
        int64_t pph = periods_per_day(creso) // 24
        int64_t pps = periods_per_second(creso)
        # 创建 npy_datetimestruct 结构体变量 dts

    # DstTzInfo.localize 的向量化版本

    # 禁止误报的编译器警告
    # 初始化 ambiguous_array 为空的布尔类型数组
    ambiguous_array = np.empty(0, dtype=bool)
    # 检查 ambiguous 是否为字符串类型
    if isinstance(ambiguous, str):
        # 如果 ambiguous 为 "infer"，设置 infer_dst 为 True
        if ambiguous == "infer":
            infer_dst = True
        # 如果 ambiguous 为 "NaT"，设置 fill 为 True
        elif ambiguous == "NaT":
            fill = True
    # 如果 ambiguous 是布尔类型
    elif isinstance(ambiguous, bool):
        is_dst = True
        # 如果 ambiguous 为 True，创建长度与 vals 相同的全为 True 的布尔数组
        if ambiguous:
            ambiguous_array = np.ones(len(vals), dtype=bool)
        # 如果 ambiguous 为 False，创建长度与 vals 相同的全为 False 的布尔数组
        else:
            ambiguous_array = np.zeros(len(vals), dtype=bool)
    # 如果 ambiguous 是可迭代对象
    elif hasattr(ambiguous, "__iter__"):
        is_dst = True
        # 检查 ambiguous 的长度与 vals 是否相等，否则抛出 ValueError 异常
        if len(ambiguous) != len(vals):
            raise ValueError("Length of ambiguous bool-array must be "
                             "the same size as vals")

        # 将 ambiguous 转换为布尔类型的数组 ambiguous_array
        ambiguous_array = np.asarray(ambiguous, dtype=bool)

    # 处理 nonexistent 参数
    if nonexistent == "NaT":
        fill_nonexist = True
    elif nonexistent == "shift_forward":
        shift_forward = True
    elif nonexistent == "shift_backward":
        shift_backward = True
    elif PyDelta_Check(nonexistent):
        # 如果 nonexistent 是 PyDelta 对象，转换为纳秒单位的时间差并赋值给 shift_delta
        from .timedeltas import delta_to_nanoseconds
        shift_delta = delta_to_nanoseconds(nonexistent, reso=creso)
    elif nonexistent not in ("raise", None):
        # 如果 nonexistent 不在指定的可选值中，抛出 ValueError 异常
        msg = ("nonexistent must be one of {'NaT', 'raise', 'shift_forward', "
               "shift_backwards} or a timedelta object")
        raise ValueError(msg)

    # 创建一个空的 int64 类型的数组 result，与 vals 的维度和形状相同
    result = cnp.PyArray_EMPTY(vals.ndim, vals.shape, cnp.NPY_INT64, 0)

    # 如果 info.use_tzlocal 为 True 且 tz 不是 zoneinfo
    if info.use_tzlocal and not is_zoneinfo(tz):
        # 遍历 vals 数组的每个元素
        for i in range(n):
            v = vals[i]
            # 如果 v 是 NPY_NAT（Not a Time），将 result[i] 设为 NPY_NAT
            if v == NPY_NAT:
                result[i] = NPY_NAT
            else:
                # 否则，使用 _tz_localize_using_tzinfo_api 函数进行本地化处理
                result[i] = v - _tz_localize_using_tzinfo_api(
                    v, tz, to_utc=True, creso=creso
                )
        # 返回 result 数组的基础对象，即 ndarray
        return result.base

    # 如果 info.use_fixed 为 True
    elif info.use_fixed:
        delta = info.delta
        # 遍历 vals 数组的每个元素
        for i in range(n):
            v = vals[i]
            # 如果 v 是 NPY_NAT（Not a Time），将 result[i] 设为 NPY_NAT
            if v == NPY_NAT:
                result[i] = NPY_NAT
            else:
                # 否则，将 result[i] 设为 v 减去固定偏移量 delta
                result[i] = v - delta
        # 返回 result 数组的基础对象，即 ndarray
        return result.base

    # 确定每个日期是否位于 DST 转换左侧（存储在 result_a 中）或右侧（存储在 result_b 中）
    # 如果时区信息表明是一个区域信息时区
    if is_zoneinfo(tz):
        # 标记为区域信息时区
        is_zi = True
        # 调用区域信息时区的函数获取 UTC 时间边界
        result_a, result_b = _get_utc_bounds_zoneinfo(
            vals, tz, creso=creso
        )
    else:
        # 调用非区域信息时区的函数获取 UTC 时间边界
        result_a, result_b = _get_utc_bounds(
            vals, info.tdata, info.ntrans, info.deltas, creso=creso
        )

    # 禁止错误的编译器警告
    dst_hours = np.empty(0, dtype=np.int64)
    if infer_dst:
        # 如果需要推断 DST，调用函数获取 DST 时间段
        dst_hours = _get_dst_hours(vals, result_a, result_b, creso=creso)

    # 预先计算 delta_idx_offset，用于处理可能不存在的路径
    # 根据目标时区的 UTC 偏移量决定 delta_idx 的偏移量
    # TODO: 对于区域信息时区，delta_idx_offset 和 info.deltas 是必需的，
    # 但并不适用于所有时区。将前者设置为 0 并对后者进行长度检查可以避免未定义行为，
    # 但这可能需要进行更大的重构。
    delta_idx_offset = 0
    if len(info.deltas):
        first_delta = info.deltas[0]
        if (shift_forward or shift_delta > 0) and first_delta > 0:
            delta_idx_offset = 1
        elif (shift_backward or shift_delta < 0) and first_delta < 0:
            delta_idx_offset = 1

    return result.base  # 返回底层的 ndarray 数据
cdef Py_ssize_t bisect_right_i8(
    const int64_t *data,
    int64_t val,
    Py_ssize_t n
) noexcept:
    # 调用者负责检查 n > 0
    # 这段代码与 ndarray.searchsorted 实现中的 local_search_right 非常相似。

    cdef:
        Py_ssize_t pivot, left = 0, right = n

    # 边界情况处理
    if val > data[n - 1]:
        return n

    # 调用者负责确保 'val >= data[0]'。这是因为 'data' 是从 get_dst_info 中获取的，
    # 其中 data[0] *总是* NPY_NAT+1。如果这个条件发生改变，我们需要恢复以下已禁用的检查。
    # if val < data[0]:
    #    return 0

    while left < right:
        pivot = left + (right - left) // 2

        if data[pivot] <= val:
            left = pivot + 1
        else:
            right = pivot

    return left


cdef str _render_tstamp(int64_t val, NPY_DATETIMEUNIT creso):
    """ 辅助函数，用于呈现异常消息 """
    from pandas._libs.tslibs.timestamps import Timestamp
    ts = Timestamp._from_value_and_reso(val, creso, None)
    return str(ts)


cdef _get_utc_bounds(
    ndarray[int64_t] vals,
    const int64_t* tdata,
    Py_ssize_t ntrans,
    const int64_t[::1] deltas,
    NPY_DATETIMEUNIT creso,
):
    # 确定每个日期位于夏令时转换左侧（存储在 result_a 中）还是右侧（存储在 result_b 中）

    cdef:
        ndarray[int64_t] result_a, result_b
        Py_ssize_t i, n = vals.size
        int64_t val, v_left, v_right
        Py_ssize_t isl, isr, pos_left, pos_right
        int64_t ppd = periods_per_day(creso)

    result_a = cnp.PyArray_EMPTY(vals.ndim, vals.shape, cnp.NPY_INT64, 0)
    result_b = cnp.PyArray_EMPTY(vals.ndim, vals.shape, cnp.NPY_INT64, 0)

    for i in range(n):
        # 此循环类似于 pytz 的 DstTZInfo.localize 方法中的 "Find the two best possibilities" 块。
        result_a[i] = NPY_NAT
        result_b[i] = NPY_NAT

        val = vals[i]
        if val == NPY_NAT:
            continue

        # TODO: 注意 val-ppd 可能会溢出
        isl = bisect_right_i8(tdata, val - ppd, ntrans) - 1
        if isl < 0:
            isl = 0

        v_left = val - deltas[isl]
        pos_left = bisect_right_i8(tdata, v_left, ntrans) - 1
        # 时间戳位于夏令时转换左侧
        if v_left + deltas[pos_left] == val:
            result_a[i] = v_left

        # TODO: 注意 val+ppd 可能会溢出
        isr = bisect_right_i8(tdata, val + ppd, ntrans) - 1
        if isr < 0:
            isr = 0

        v_right = val - deltas[isr]
        pos_right = bisect_right_i8(tdata, v_right, ntrans) - 1
        # 时间戳位于夏令时转换右侧
        if v_right + deltas[pos_right] == val:
            result_b[i] = v_right

    return result_a, result_b


cdef _get_utc_bounds_zoneinfo(ndarray vals, tz, NPY_DATETIMEUNIT creso):
    """
    # 对于 'vals' 中的每个时间戳，分别计算在 fold=0 和 fold=1 情况下对应的 UTC 时间。
    # 在无歧义的情况下，这两者应该是相同的。

    # 参数说明：
    # vals: ndarray[int64_t]，包含时间戳的数组
    # tz: ZoneInfo，时区信息对象
    # creso: NPY_DATETIMEUNIT，时间单位枚举值

    # 返回值：
    # 两个 ndarray[int64_t]，分别对应 fold=0 和 fold=1 情况下的结果
    """
    cdef:
        Py_ssize_t i, n = vals.size  # 定义循环变量和数组大小
        npy_datetimestruct dts  # 定义 numpy 的日期时间结构
        datetime dt, rt, left, right, aware, as_utc  # 定义 datetime 对象和其他辅助变量
        int64_t val, pps = periods_per_second(creso)  # 定义时间单位内的周期数和时间戳变量
        ndarray result_a, result_b  # 定义两个结果数组

    # 创建两个空的 int64 类型的 ndarray 数组，与 vals 的维度和形状相同
    result_a = cnp.PyArray_EMPTY(vals.ndim, vals.shape, cnp.NPY_INT64, 0)
    result_b = cnp.PyArray_EMPTY(vals.ndim, vals.shape, cnp.NPY_INT64, 0)

    # 遍历 vals 中的每个时间戳
    for i in range(n):
        val = vals[i]  # 获取当前时间戳值
        if val == NPY_NAT:
            result_a[i] = NPY_NAT  # 如果时间戳为 NPY_NAT，结果数组相应位置也为 NPY_NAT
            result_b[i] = NPY_NAT
            continue  # 继续下一个循环

        # 将 pandas 的 datetime 转换为 numpy 的 datetimestruct 结构
        pandas_datetime_to_datetimestruct(val, creso, &dts)
        
        # 计算额外的时间，用于补充后续处理过程中丢失的纳秒等精度
        extra = (dts.ps // 1000) * (pps // 1_000_000_000)

        # 创建 datetime 对象，基于 dts 结构中的日期和时间信息
        dt = datetime_new(dts.year, dts.month, dts.day, dts.hour,
                          dts.min, dts.sec, dts.us, None)

        # 将 datetime 对象转换为时区感知时间 aware
        aware = dt.replace(tzinfo=tz)
        
        # 将 aware 时间转换为 UTC 时间
        as_utc = aware.astimezone(utc_stdlib)
        
        # 将 UTC 时间再转换回指定时区的时间 rt
        rt = as_utc.astimezone(tz)
        
        # 检查 aware 和 rt 是否相同，如果不同，说明 aware 可能不存在
        if aware != rt:
            # 如果 aware 和 rt 不同，将结果设置为 NPY_NAT
            result_a[i] = NPY_NAT
        else:
            # 否则，将 as_utc 转换为不带时区信息的本地时间 left
            left = as_utc.replace(tzinfo=None)
            # 将 left 转换为 dt64 格式的时间并加上额外的时间精度
            result_a[i] = pydatetime_to_dt64(left, &dts, creso) + extra

        # 创建 fold=1 的 aware 时间
        aware = dt.replace(fold=1, tzinfo=tz)
        # 将 fold=1 的 aware 时间转换为 UTC 时间
        as_utc = aware.astimezone(utc_stdlib)
        # 将 UTC 时间再转换回指定时区的时间 rt
        rt = as_utc.astimezone(tz)
        # 检查 aware 和 rt 是否相同，如果不同，说明 aware 可能不存在
        if aware != rt:
            # 如果 aware 和 rt 不同，将结果设置为 NPY_NAT
            result_b[i] = NPY_NAT
        else:
            # 否则，将 as_utc 转换为不带时区信息的本地时间 right
            right = as_utc.replace(tzinfo=None)
            # 将 right 转换为 dt64 格式的时间并加上额外的时间精度
            result_b[i] = pydatetime_to_dt64(right, &dts, creso) + extra

    # 返回计算得到的两个结果数组
    return result_a, result_b
# 在 Cython 中禁用边界检查，提升性能
@cython.boundscheck(False)
cdef ndarray[int64_t] _get_dst_hours(
    # vals 是一个整型数组，用于存储时间戳；creso 仅在此处可能用于生成异常消息
    const int64_t[:] vals,  # 输入参数：时间戳数组
    ndarray[int64_t] result_a,  # 输入参数：结果数组 A
    ndarray[int64_t] result_b,  # 输入参数：结果数组 B
    NPY_DATETIMEUNIT creso,  # 输入参数：时间单位
):
    cdef:
        Py_ssize_t i, n = vals.shape[0]  # 使用 vals 的长度作为循环次数 n
        ndarray[uint8_t, cast=True] mismatch  # 声明一个无符号 8 位整型数组 mismatch
        ndarray[int64_t] delta, dst_hours  # 声明两个 64 位整型数组 delta 和 dst_hours
        ndarray[intp_t] switch_idxs, trans_idx, grp, a_idx, b_idx, one_diff  # 声明多个整型数组
        # TODO: Can uncomment when numpy >=2 is the minimum
        # tuple trans_grp  # 声明一个元组变量 trans_grp
        intp_t switch_idx  # 声明一个整型变量 switch_idx
        int64_t left, right  # 声明两个 64 位整型变量 left 和 right

    # 创建一个和 result_a 相同形状的空数组，用于存储 DST 相关小时数
    dst_hours = cnp.PyArray_EMPTY(result_a.ndim, result_a.shape, cnp.NPY_INT64, 0)
    dst_hours[:] = NPY_NAT  # 将 dst_hours 数组初始化为 NPY_NAT

    # 创建一个和 result_a 相同形状的布尔类型数组 mismatch，初始化为全零
    mismatch = cnp.PyArray_ZEROS(result_a.ndim, result_a.shape, cnp.NPY_BOOL, 0)

    # 遍历 vals 数组
    for i in range(n):
        left = result_a[i]  # 获取 result_a 的第 i 个元素
        right = result_b[i]  # 获取 result_b 的第 i 个元素

        # 获取模糊时间（即 result_a != result_b 且它们都不是 NPY_NAT 的小时数）
        if left != right and left != NPY_NAT and right != NPY_NAT:
            mismatch[i] = 1  # 将 mismatch 数组相应位置标记为 1

    # 找出 mismatch 数组中非零元素的索引
    trans_idx = mismatch.nonzero()[0]

    # 如果 trans_idx 的大小为 1，则抛出 AmbiguousTimeError 异常
    if trans_idx.size == 1:
        # 查看 test_tz_localize_to_utc_ambiguous_infer 测试中的时间戳
        stamp = _render_tstamp(vals[trans_idx[0]], creso=creso)
        raise pytz.AmbiguousTimeError(
            f"Cannot infer dst time from {stamp} as there "
            "are no repeated times"
        )

    # 将数组分割成连续的块（其中索引之间的差为 1），用于检查单个年份中是否存在模糊转换
    # 如果 trans_idx 的大小大于 0，则进入条件判断
    if trans_idx.size > 0:
        # 找到 trans_idx 中相邻元素差值不为 1 的位置，并加一得到索引数组
        one_diff = np.where(np.diff(trans_idx) != 1)[0] + 1
        # 将 trans_idx 按照 one_diff 的索引分割成多个子数组
        trans_grp = np.array_split(trans_idx, one_diff)

        # 遍历 trans_grp 中的每个子数组，检查每天的小时是否有负的增量（表明重复的小时），如果没有，则无法推断切换
        for grp in trans_grp:

            # 计算结果数组 result_a 中 grp 子数组的差分
            delta = np.diff(result_a[grp])
            # 如果 grp 的大小为 1 或者所有的增量都大于 0，则抛出 AmbiguousTimeError 异常
            if grp.size == 1 or np.all(delta > 0):
                # 生成时间戳并抛出 AmbiguousTimeError 异常
                stamp = _render_tstamp(vals[grp[0]], creso=creso)
                raise pytz.AmbiguousTimeError(stamp)

            # 找到增量小于等于 0 的索引位置，并存储在 switch_idxs 中
            switch_idxs = (delta <= 0).nonzero()[0]
            # 如果 switch_idxs 中的元素个数大于 1，则抛出 AmbiguousTimeError 异常
            if switch_idxs.size > 1:
                raise pytz.AmbiguousTimeError(
                    f"There are {switch_idxs.size} dst switches when "
                    "there should only be 1."
                )

            # 取第一个 switch_idx 的值加一，作为切换点的索引
            switch_idx = switch_idxs[0] + 1
            # 将 grp 分为两部分 a_idx 和 b_idx
            a_idx = grp[:switch_idx]
            b_idx = grp[switch_idx:]
            # 更新 dst_hours 中 grp 对应位置的值为组合的结果数组
            dst_hours[grp] = np.hstack((result_a[a_idx], result_b[b_idx]))

    # 返回更新后的 dst_hours 字典
    return dst_hours
# ----------------------------------------------------------------------
# Timezone Conversion

# 定义一个Cython函数，将UTC时间转换为指定时区的本地时间
cpdef int64_t tz_convert_from_utc_single(
    int64_t utc_val, tzinfo tz, NPY_DATETIMEUNIT creso=NPY_DATETIMEUNIT.NPY_FR_ns
) except? -1:
    """
    Convert the val (in i8) from UTC to tz

    This is a single value version of tz_convert_from_utc.

    Parameters
    ----------
    utc_val : int64
        UTC时间值，以int64表示
    tz : tzinfo
        目标时区的tzinfo对象
    creso : NPY_DATETIMEUNIT, default NPY_FR_ns
        结果的时间精度，默认为纳秒

    Returns
    -------
    converted: int64
        转换后的本地时间值，以int64表示
    """
    # 创建Localizer对象，用于执行时区转换
    cdef:
        Localizer info = Localizer(tz, creso=creso)
        Py_ssize_t pos

    # 注意：调用者需确保utc_val不等于NPY_NAT（不是有效的时间戳）
    # 调用Localizer对象的方法执行UTC时间到本地时间的转换
    return info.utc_val_to_local_val(utc_val, &pos)


# OSError可能由tzlocal在Windows上在1970-01-01附近抛出
# 请参考https://github.com/pandas-dev/pandas/pull/37591#issuecomment-720628241
# 定义一个Cython函数，使用datetime/tzinfo API将通用时区时间转换为UTC时间或反向转换
cdef int64_t _tz_localize_using_tzinfo_api(
    int64_t val,
    tzinfo tz,
    bint to_utc=True,
    NPY_DATETIMEUNIT creso=NPY_DATETIMEUNIT.NPY_FR_ns,
    bint* fold=NULL,
) except? -1:
    """
    Convert the i8 representation of a datetime from a general-case timezone to
    UTC, or vice-versa using the datetime/tzinfo API.

    Private, not intended for use outside of tslibs.tzconversion.

    Parameters
    ----------
    val : int64_t
        表示日期时间的int64_t类型的值
    tz : tzinfo
        目标时区的tzinfo对象
    to_utc : bint
        如果为True，则从通用时区转换为UTC；如果为False，则反向转换
    creso : NPY_DATETIMEUNIT
        结果的时间精度
    fold : bint*, default NULL
        指向fold的指针：调整后日期时间是否处于fold状态
        仅在to_utc=False时传递

    Returns
    -------
    delta : int64_t
        从UTC转换时要添加的值，转换到UTC时要减去的值

    Notes
    -----
    通过指针设置fold值
    """
    cdef:
        npy_datetimestruct dts
        datetime dt
        int64_t delta
        timedelta td
        int64_t pps = periods_per_second(creso)

    # 将int64_t类型的时间值转换为npy_datetimestruct结构体
    pandas_datetime_to_datetimestruct(val, creso, &dts)

    # 如果不是转换为UTC，则将时间结构体转换为datetime对象
    if not to_utc:
        # 如果val是UTC时间戳，则将其转换为本地墙壁时间
        dt = _astimezone(dts, tz)

        # 如果fold不为NULL，则设置fold值
        if fold is not NULL:
            # 注意：fold仅在to_utc=False时传递
            fold[0] = dt.fold
    else:
        # 否则，创建一个datetime对象表示UTC时间
        dt = datetime_new(dts.year, dts.month, dts.day, dts.hour,
                          dts.min, dts.sec, dts.us, None)

    # 计算本地时间或UTC时间与UTC的偏移量
    td = tz.utcoffset(dt)
    delta = int(td.total_seconds() * pps)
    return delta


# 定义一个Cython函数，优化实现时区转换功能
cdef datetime _astimezone(npy_datetimestruct dts, tzinfo tz):
    """
    Optimized equivalent to:

    dt = datetime(dts.year, dts.month, dts.day, dts.hour,
                  dts.min, dts.sec, dts.us, utc_stdlib)
    dt = dt.astimezone(tz)

    Derived from the datetime.astimezone implementation at
    https://github.com/python/cpython/blob/main/Modules/_datetimemodule.c#L6187

    NB: we are assuming tz is not None.

    Parameters
    ----------
    dts : npy_datetimestruct
        表示日期时间的npy_datetimestruct结构体
    tz : tzinfo
        目标时区的tzinfo对象
    """
    # 使用优化的方法将dts表示的时间转换为目标时区tz表示的本地时间
    dt = datetime(dts.year, dts.month, dts.day, dts.hour,
                  dts.min, dts.sec, dts.us, utc_stdlib)
    dt = dt.astimezone(tz)
    return dt
    """
    cdef:
        datetime result  # 声明一个变量 result，类型为 datetime

    result = datetime_new(dts.year, dts.month, dts.day, dts.hour,
                          dts.min, dts.sec, dts.us, tz)
    # 调用 datetime_new 函数创建一个新的 datetime 对象，使用给定的日期时间信息和时区 tz

    return tz.fromutc(result)
    # 将 result 对象转换为给定时区 tz 的本地时间并返回
    ```
# NB: relies on dateutil internals, subject to change.
# 使用了 dateutil 的内部实现，可能会随时更改。

@cython.boundscheck(False)
@cython.wraparound(False)
# 关闭 Cython 的边界检查和负索引检查

cdef bint _infer_dateutil_fold(
    int64_t value,
    const int64_t[::1] trans,
    const int64_t[::1] deltas,
    Py_ssize_t pos,
):
    """
    Infer _TSObject fold property from value by assuming 0 and then setting
    to 1 if necessary.

    Parameters
    ----------
    value : int64_t
        传入的整数值，用于推断 fold 属性。
    trans : ndarray[int64_t]
        包含自纪元以来的纳秒偏移转换点的数组。
    deltas : int64_t[:]
        对应于 trans 中转换点的偏移数组。
    pos : Py_ssize_t
        在考虑 fold 属性之前最后一个转换点的位置。

    Returns
    -------
    bint
        由于夏令时的存在，当从夏季时间切换到冬季时间时，墙上钟时间可能发生两次重叠；
        fold 描述了 datetime 对象是对应第一次（0）还是第二次（1）发生重叠的时间。

    References
    ----------
    .. [1] "PEP 495 - Local Time Disambiguation"
           https://www.python.org/dev/peps/pep-0495/#the-fold-attribute
    """
    cdef:
        bint fold = 0
        int64_t fold_delta

    if pos > 0:
        # 计算当前位置前一个转换点与当前转换点之间的偏移量
        fold_delta = deltas[pos - 1] - deltas[pos]
        # 如果 value 减去 fold_delta 小于当前转换点的时间，则设置 fold 为 1
        if value - fold_delta < trans[pos]:
            fold = 1

    return fold
```