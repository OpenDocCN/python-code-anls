# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\timedeltas.pyx`

```
# 导入 collections 模块，用于命名元组 Components
import collections
# 导入 warnings 模块，用于警告处理
import warnings

# 导入 pandas.util._exceptions 模块中的 find_stack_level 函数
from pandas.util._exceptions import find_stack_level

# 导入 Cython 扩展模块 cimport
cimport cython
# 从 cpython.object 中导入多个 C 类型常量和函数
from cpython.object cimport (
    Py_EQ,
    Py_GE,
    Py_GT,
    Py_LE,
    Py_LT,
    Py_NE,
    PyObject,
    PyObject_RichCompare,
)

# 导入 NumPy 库，使用 np 别名
import numpy as np

# 导入 Cython 扩展模块 cimport
cimport numpy as cnp
# 从 numpy cimport 导入 int64_t 和 ndarray 类型
from numpy cimport (
    int64_t,
    ndarray,
)

# 调用 cnp 模块的 import_array 函数
cnp.import_array()

# 导入 Cython 扩展模块 cimport
from cpython.datetime cimport (
    PyDateTime_Check,
    PyDelta_Check,
    import_datetime,
    timedelta,
)

# 执行 datetime 相关的初始化操作
import_datetime()

# 导入 pandas._libs.tslibs.util 模块中的 util 别名
cimport pandas._libs.tslibs.util as util
# 从 pandas._libs.missing 模块中 cimport checknull_with_nat_and_na 函数
from pandas._libs.missing cimport checknull_with_nat_and_na
# 从 pandas._libs.tslibs.base 模块中 cimport ABCTimestamp 类型
from pandas._libs.tslibs.base cimport ABCTimestamp
# 从 pandas._libs.tslibs.conversion 模块中 cimport 多个函数
from pandas._libs.tslibs.conversion cimport (
    cast_from_unit,
    precision_from_unit,
)
# 从 pandas._libs.tslibs.dtypes 模块中 cimport 多个常量和函数
from pandas._libs.tslibs.dtypes cimport (
    c_DEPR_UNITS,
    get_supported_reso,
    is_supported_unit,
    npy_unit_to_abbrev,
)
# 从 pandas._libs.tslibs.nattype 模块中 cimport 多个常量和函数
from pandas._libs.tslibs.nattype cimport (
    NPY_NAT,
    c_NaT as NaT,
    c_nat_strings as nat_strings,
    checknull_with_nat,
)
# 从 pandas._libs.tslibs.np_datetime 模块中 cimport 多个常量和函数
from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,
    NPY_FR_ns,
    cmp_dtstructs,
    cmp_scalar,
    convert_reso,
    get_datetime64_unit,
    get_unit_from_dtype,
    import_pandas_datetime,
    npy_datetimestruct,
    pandas_datetime_to_datetimestruct,
    pandas_timedelta_to_timedeltastruct,
    pandas_timedeltastruct,
)

# 执行 pandas_datetime 相关的初始化操作
import_pandas_datetime()

# 从 pandas._libs.tslibs.np_datetime 模块中 import 出 OutOfBoundsDatetime 和 OutOfBoundsTimedelta 类
from pandas._libs.tslibs.np_datetime import (
    OutOfBoundsDatetime,
    OutOfBoundsTimedelta,
)

# 从 pandas._libs.tslibs.offsets 模块中 cimport is_tick_object 函数
from pandas._libs.tslibs.offsets cimport is_tick_object
# 从 pandas._libs.tslibs.util 模块中 cimport 多个函数和常量
from pandas._libs.tslibs.util cimport (
    is_array,
    is_float_object,
    is_integer_object,
)

# 从 pandas._libs.tslibs.fields 模块中 cimport RoundTo 和 round_nsint64 函数
from pandas._libs.tslibs.fields import (
    RoundTo,
    round_nsint64,
)

# ----------------------------------------------------------------------
# 常量定义

# 定义名为 Components 的命名元组，包含多个时间组件字段
Components = collections.namedtuple(
    "Components",
    [
        "days",
        "hours",
        "minutes",
        "seconds",
        "milliseconds",
        "microseconds",
        "nanoseconds",
    ],
)

# 定义 timedelta 缩写字典，与 pandas/_libs/tslibs/timedeltas.pyi 中的 UnitChoices 保持一致
cdef dict timedelta_abbrevs = {
    "Y": "Y",
    "y": "Y",
    "M": "M",
    "W": "W",
    "w": "W",
    "D": "D",
    "d": "D",
    "days": "D",
    "day": "D",
    "hours": "h",
    "hour": "h",
    "hr": "h",
    "h": "h",
    "m": "m",
    "minute": "m",
    "min": "m",
    "minutes": "m",
    "s": "s",
    "seconds": "s",
    "sec": "s",
    "second": "s",
    "ms": "ms",
    "milliseconds": "ms",
    "millisecond": "ms",
    "milli": "ms",
    "millis": "ms",
    "us": "us",
    "microseconds": "us",
    "microsecond": "us",
    "µs": "us",
    "micro": "us",
    "micros": "us",
    "ns": "ns",
    "nanoseconds": "ns",
    "nano": "ns",
    "nanos": "ns",
    "nanosecond": "ns",
}

# 定义占位对象 _no_input
_no_input = object()

# ----------------------------------------------------------------------
# API

# 使用 Cython 的 boundscheck(False) 修饰符禁用边界检查
@cython.boundscheck(False)
# 使用 Cython 的 wraparound(False) 修饰符禁用负数索引检查
@cython.wraparound(False)
def ints_to_pytimedelta(ndarray m8values, box=False):
    """
    convert an i8 repr to an ndarray of timedelta or Timedelta (if box ==
    True)

    Parameters
    ----------
    arr : ndarray[timedelta64]
    box : bool, default False

    Returns
    -------
    result : ndarray[object]
        array of Timedelta or timedeltas objects
    """
    # 获取数据类型的时间单位
    cdef:
        NPY_DATETIMEUNIT reso = get_unit_from_dtype(m8values.dtype)
        # 获取数组大小和迭代变量
        Py_ssize_t i, n = m8values.size
        int64_t value
        object res_val

        # Note that `result` (and thus `result_flat`) is C-order and
        #  `it` iterates C-order as well, so the iteration matches
        #  See discussion at
        #  github.com/pandas-dev/pandas/pull/46886#discussion_r860261305
        # 创建一个空的对象数组来存储结果，保持与输入数组的形状和维度一致
        ndarray result = cnp.PyArray_EMPTY(
            m8values.ndim, m8values.shape, cnp.NPY_OBJECT, 0
        )
        # 将结果数组展平成一维对象数组，但不会复制数据
        object[::1] res_flat = result.ravel()     # should NOT be a copy

        # 将输入数组视图转换为'i8'类型，即'int64'类型的视图
        ndarray arr = m8values.view("i8")
        # 创建一个扁平迭代器，用于按顺序访问输入数组的元素
        cnp.flatiter it = cnp.PyArray_IterNew(arr)

    for i in range(n):
        # 类似于：value = arr[i]
        # 从迭代器中获取当前元素的值
        value = (<int64_t*>cnp.PyArray_ITER_DATA(it))[0]

        if value == NPY_NAT:
            # 如果值为缺失值，则结果为NaT（Not a Time）
            res_val = <object>NaT
        else:
            if box:
                # 如果box为True，则封装为Timedelta对象
                res_val = _timedelta_from_value_and_reso(Timedelta, value, reso=reso)
            elif reso == NPY_DATETIMEUNIT.NPY_FR_ns:
                # 根据纳秒单位创建timedelta对象
                res_val = timedelta(microseconds=int(value) / 1000)
            elif reso == NPY_DATETIMEUNIT.NPY_FR_us:
                # 根据微秒单位创建timedelta对象
                res_val = timedelta(microseconds=value)
            elif reso == NPY_DATETIMEUNIT.NPY_FR_ms:
                # 根据毫秒单位创建timedelta对象
                res_val = timedelta(milliseconds=value)
            elif reso == NPY_DATETIMEUNIT.NPY_FR_s:
                # 根据秒单位创建timedelta对象
                res_val = timedelta(seconds=value)
            elif reso == NPY_DATETIMEUNIT.NPY_FR_m:
                # 根据分钟单位创建timedelta对象
                res_val = timedelta(minutes=value)
            elif reso == NPY_DATETIMEUNIT.NPY_FR_h:
                # 根据小时单位创建timedelta对象
                res_val = timedelta(hours=value)
            elif reso == NPY_DATETIMEUNIT.NPY_FR_D:
                # 根据天单位创建timedelta对象
                res_val = timedelta(days=value)
            elif reso == NPY_DATETIMEUNIT.NPY_FR_W:
                # 根据周单位创建timedelta对象
                res_val = timedelta(weeks=value)
            else:
                # 如果单位不在已定义的范围内，则抛出未实现错误
                raise NotImplementedError(reso)

        # 注意：我们可以直接通过索引访问result，而不是使用PyArray_MultiIter_DATA
        #  这是因为result是已知的C连续数组，并且是PyArray_MultiIterNew2的第一个参数。
        #  对于对象dtype，通常的模式似乎无法使用。
        #  参见讨论：
        #  github.com/pandas-dev/pandas/pull/46886#discussion_r860261305
        # 将计算得到的值放入结果数组的相应位置
        res_flat[i] = res_val

        # 移动到迭代器的下一个元素
        cnp.PyArray_ITER_NEXT(it)

    # 返回结果数组
    return result
# Note: this will raise on timedelta64 with Y or M unit

cdef:
    # 定义变量，用于存储时间单位和整数值
    NPY_DATETIMEUNIT in_reso
    int64_t n

if is_tick_object(delta):
    # 如果 delta 是 Tick 对象，则获取其 n 属性和 _creso 属性
    n = delta.n
    in_reso = delta._creso

elif isinstance(delta, _Timedelta):
    # 如果 delta 是 _Timedelta 类型的对象，则获取其 _value 属性和 _creso 属性
    n = delta._value
    in_reso = delta._creso

elif cnp.is_timedelta64_object(delta):
    # 如果 delta 是 timedelta64 对象，则获取其单位并检查是否为 Y 或 M 单位
    in_reso = get_datetime64_unit(delta)
    if in_reso == NPY_DATETIMEUNIT.NPY_FR_Y or in_reso == NPY_DATETIMEUNIT.NPY_FR_M:
        # 如果单位为年或月，则抛出值错误
        raise ValueError(
            "delta_to_nanoseconds does not support Y or M units, "
            "as their duration in nanoseconds is ambiguous."
        )
    n = cnp.get_timedelta64_value(delta)

elif PyDelta_Check(delta):
    # 如果 delta 是 Python 的 datetime.timedelta 对象
    in_reso = NPY_DATETIMEUNIT.NPY_FR_us
    try:
        # 计算 delta 的总微秒数，并转换为纳秒
        n = (
            delta.days * 24 * 3600 * 1_000_000
            + delta.seconds * 1_000_000
            + delta.microseconds
            )
    except OverflowError as err:
        # 捕获溢出错误并抛出自定义的 OutOfBoundsTimedelta 异常
        raise OutOfBoundsTimedelta(*err.args) from err

else:
    # 如果 delta 不是上述任何类型，则抛出类型错误
    raise TypeError(type(delta))

try:
    # 调用 convert_reso 函数进行单位转换，并返回结果
    return convert_reso(n, in_reso, reso, round_ok=round_ok)
except (OutOfBoundsDatetime, OverflowError) as err:
    # 捕获转换过程中可能抛出的 OutOfBoundsDatetime 或 OverflowError 异常
    # 提示转换为目标单位时可能会溢出
    unit_str = npy_unit_to_abbrev(reso)
    raise OutOfBoundsTimedelta(
        f"Cannot cast {str(delta)} to unit={unit_str} without overflow."
    ) from err


@cython.overflowcheck(True)
cdef object ensure_td64ns(object ts):
    """
    Overflow-safe implementation of td64.astype("m8[ns]")

    Parameters
    ----------
    ts : np.timedelta64

    Returns
    -------
    np.timedelta64[ns]
    """
    cdef:
        # 定义变量，用于存储时间单位和整数值
        NPY_DATETIMEUNIT td64_unit
        int64_t td64_value, mult

    # 获取 td64 对象的时间单位
    td64_unit = get_datetime64_unit(ts)
    if (
        td64_unit != NPY_DATETIMEUNIT.NPY_FR_ns
        and td64_unit != NPY_DATETIMEUNIT.NPY_FR_GENERIC
    ):
        # 如果单位不是纳秒或通用单位，则进行精度调整和溢出检查
        td64_value = cnp.get_timedelta64_value(ts)

        mult = precision_from_unit(td64_unit)[0]
        try:
            # 尝试将 td64_value 乘以倍数 mult
            # 注意：此处不能使用 *= ，这是因为 Cython 的一个问题
            td64_value = td64_value * mult
        except OverflowError as err:
            # 捕获溢出错误并抛出自定义的 OutOfBoundsTimedelta 异常
            raise OutOfBoundsTimedelta(ts) from err

        # 返回经过溢出安全处理后的 np.timedelta64[ns] 对象
        return np.timedelta64(td64_value, "ns")

    # 如果单位已经是纳秒，则直接返回原始 ts 对象
    return ts


cdef convert_to_timedelta64(object ts, str unit):
    """
    Convert an incoming object to a timedelta64 if possible.
    Before calling, unit must be standardized to avoid repeated unit conversion

    Handle these types of objects:
        - timedelta/Timedelta
        - timedelta64
        - an offset
        - np.int64 (with unit providing a possible modifier)
        - None/NaT

    Return an ns based int64
    """
    # 调用 checknull_with_nat_and_na 函数检查是否为 None 或 NaT
    if checknull_with_nat_and_na(ts):
        # 如果是 None 或 NaT，则返回特定的 np.timedelta64[ns] 对象
        return np.timedelta64(NPY_NAT, "ns")
    # 如果输入的时间戳已经是 _Timedelta 对象
    elif isinstance(ts, _Timedelta):
        # 已经是正确的格式，无需处理
        if ts._creso != NPY_FR_ns:
            # 将时间戳转换为纳秒单位的 _Timedelta 对象
            ts = ts.as_unit("ns").asm8
        else:
            # 将时间戳转换为纳秒精度的 timedelta64 对象
            ts = np.timedelta64(ts._value, "ns")
    
    # 如果输入的时间戳是 timedelta64 对象
    elif cnp.is_timedelta64_object(ts):
        # 确保时间戳以纳秒为单位
        ts = ensure_td64ns(ts)
    
    # 如果输入的时间戳是整数对象
    elif is_integer_object(ts):
        # 如果是 NaT（Not-a-Time），返回纳秒单位的 NaT
        if ts == NPY_NAT:
            return np.timedelta64(NPY_NAT, "ns")
        else:
            # 尝试从给定的单位转换时间戳
            ts = _maybe_cast_from_unit(ts, unit)
    
    # 如果输入的时间戳是浮点数对象
    elif is_float_object(ts):
        # 尝试从给定的单位转换时间戳
        ts = _maybe_cast_from_unit(ts, unit)
    
    # 如果输入的时间戳是字符串对象
    elif isinstance(ts, str):
        # 如果字符串以 "P" 或 "-P" 开头，解析 ISO 格式时间字符串
        if (len(ts) > 0 and ts[0] == "P") or (len(ts) > 1 and ts[:2] == "-P"):
            ts = parse_iso_format_string(ts)
        else:
            # 解析普通时间间隔格式的时间字符串
            ts = parse_timedelta_string(ts)
        # 转换为纳秒单位的 timedelta64 对象
        ts = np.timedelta64(ts, "ns")
    
    # 如果输入的时间戳是 tick 对象
    elif is_tick_object(ts):
        # 转换为纳秒单位的 timedelta64 对象
        ts = np.timedelta64(ts.nanos, "ns")
    
    # 如果时间戳是 PyDelta 类型的对象
    if PyDelta_Check(ts):
        # 将 PyDelta 类型对象转换为纳秒单位的 timedelta64 对象
        ts = np.timedelta64(delta_to_nanoseconds(ts), "ns")
    # 如果时间戳不是 timedelta64 对象，则抛出类型错误
    elif not cnp.is_timedelta64_object(ts):
        raise TypeError(f"Invalid type for timedelta scalar: {type(ts)}")
    
    # 将时间戳转换为纳秒单位的 timedelta64 类型，并返回结果
    return ts.astype("timedelta64[ns]")
# 定义一个Cython函数，用于根据单位将时间序列转换为timedelta64格式的数组
@cython.boundscheck(False)
@cython.wraparound(False)
def array_to_timedelta64(
    ndarray values, str unit=None, str errors="raise"
) -> ndarray:
    # values是对象类型的ndarray，可能是二维的
    """
    Convert an ndarray to an array of timedeltas. If errors == 'coerce',
    coerce non-convertible objects to NaT. Otherwise, raise.

    Returns
    -------
    np.ndarray[timedelta64ns]
    """
    # 调用者需负责检查
    assert unit not in ["Y", "y", "M"]

    # 定义Cython变量
    cdef:
        Py_ssize_t i, n = values.size
        ndarray result = np.empty((<object>values).shape, dtype="m8[ns]")
        object item
        int64_t ival
        cnp.broadcast mi = cnp.PyArray_MultiIterNew2(result, values)
        cnp.flatiter it

    # 如果values不是对象类型，抛出TypeError
    if values.descr.type_num != cnp.NPY_OBJECT:
        raise TypeError("array_to_timedelta64 'values' must have object dtype")

    # 如果errors不在{'ignore', 'raise', 'coerce'}中，抛出ValueError
    if errors not in {"ignore", "raise", "coerce"}:
        raise ValueError("errors must be one of {'ignore', 'raise', or 'coerce'}")

    # 如果unit不为None且errors不是'coerce'，则进行以下检查
    if unit is not None and errors != "coerce":
        it = cnp.PyArray_IterNew(values)
        for i in range(n):
            # 类似于: item = values[i]
            item = cnp.PyArray_GETITEM(values, cnp.PyArray_ITER_DATA(it))
            if isinstance(item, str):
                raise ValueError(
                    "unit must not be specified if the input contains a str"
                )
            cnp.PyArray_ITER_NEXT(it)

    # 通常情况下，所有元素都是字符串，这时我们可以使用快速路径
    # 如果快速路径失败，我们将尝试不同的转换方式，并在此处进行错误处理
    try:
        for i in range(n):
            # 类似于: item = values[i]
            item = <object>(<PyObject**>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

            # 调用快速路径函数进行转换
            ival = _item_to_timedelta64_fastpath(item)

            # 类似于: iresult[i] = ival
            (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = ival

            cnp.PyArray_MultiIter_NEXT(mi)

    except (TypeError, ValueError):
        # 如果出现TypeError或ValueError，重置MultiIter
        cnp.PyArray_MultiIter_RESET(mi)

        # 解析时间增量单位，如果unit为None则默认为'ns'
        parsed_unit = parse_timedelta_unit(unit or "ns")
        for i in range(n):
            item = <object>(<PyObject**>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

            # 调用通用的_timedelta64转换函数，并处理错误
            ival = _item_to_timedelta64(item, parsed_unit, errors)

            # 类似于: iresult[i] = ival
            (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = ival

            cnp.PyArray_MultiIter_NEXT(mi)

    # 返回转换后的结果数组
    return result


# 定义一个Cython函数，用于快速转换对象为timedelta64格式的整数表示
cdef int64_t _item_to_timedelta64_fastpath(object item) except? -1:
    """
    See array_to_timedelta64.
    """
    # 如果 item 是 NaT（Not a Time），则执行以下逻辑
    if item is NaT:
        # 在快速路径中允许此检查，因为 NaT 是一个 C 对象，
        # 所以这是一个廉价的检查
        return NPY_NAT
    # 如果 item 不是 NaT，则执行以下逻辑
    else:
        # 调用函数 parse_timedelta_string 来解析时间间隔字符串 item
        return parse_timedelta_string(item)
cdef int64_t _item_to_timedelta64(
    object item,
    str parsed_unit,
    str errors
) except? -1:
    """
    See array_to_timedelta64.

    Convert a generic item to int64_t representation of timedelta.
    Handles errors based on specified error handling strategy.
    """

    try:
        # Convert the item to timedelta64 using parsed unit
        return cnp.get_timedelta64_value(convert_to_timedelta64(item, parsed_unit))
    except ValueError as err:
        if errors == "coerce":
            # Return NaT (Not a Time) if coercion is specified
            return NPY_NAT
        elif "unit abbreviation w/o a number" in str(err):
            # Raise a more specific ValueError with a customized message
            msg = f"Could not convert '{item}' to NumPy timedelta"
            raise ValueError(msg) from err
        else:
            # Re-raise other ValueErrors that are not handled explicitly
            raise


@cython.cpow(True)
cdef int64_t parse_timedelta_string(str ts) except? -1:
    """
    Parse a regular format timedelta string into int64_t representation in nanoseconds.
    Raise ValueError on invalid parse.

    Parses the string 'ts' to extract and convert time duration into nanoseconds.
    """

    cdef:
        str c
        bint neg = 0, have_dot = 0, have_value = 0, have_hhmmss = 0
        str current_unit = None
        int64_t result = 0, m = 0, r
        list number = [], frac = [], unit = []

    # neg : tracks if we have a leading negative for the value
    # have_dot : tracks if we are processing a dot (either post hhmmss or inside an expression)
    # have_value : track if we have at least 1 leading unit
    # have_hhmmss : tracks if we have a regular format hh:mm:ss

    if len(ts) == 0 or ts in nat_strings:
        # Return NaT if the input string is empty or in the list of natural strings
        return NPY_NAT
    # 遍历时间字符串中的每个字符
    for c in ts:

        # 跳过空格和逗号
        if c == " " or c == ",":
            pass

        # 忽略正号
        elif c == "+":
            pass

        # 处理负号
        elif c == "-":

            # 如果已经有负号、已经有数值或者已经有 hh:mm:ss 格式，则抛出异常
            if neg or have_value or have_hhmmss:
                raise ValueError("only leading negative signs are allowed")

            neg = 1

        # 处理数字字符 (ascii codes)
        elif ord(c) >= 48 and ord(c) <= 57:

            # 如果已经有小数点
            if have_dot:

                # 如果已经有单位，则当前是一个小数部分，添加到 frac 列表
                if len(unit):
                    number.append(c)
                    have_dot = 0
                else:
                    frac.append(c)

            # 如果没有单位，则当前是整数部分
            elif not len(unit):
                number.append(c)

            # 如果有单位，则根据当前的数值、小数部分和单位计算时间增量
            else:
                r = timedelta_from_spec(number, frac, unit)
                unit, number, frac = [], [c], []

                # 将计算得到的时间增量按照当前是否负数进行加减
                result += timedelta_as_neg(r, neg)

        # 处理 hh:mm:ss 格式的时间
        elif c == ":":

            # 如果已经有数值，则重置负号标志
            if have_value:
                neg = 0

            # 如果已经有数值，则当前处于 hh:mm:ss 格式
            if len(number):
                if current_unit is None:
                    current_unit = "h"
                    m = 1000000000 * 3600
                elif current_unit == "h":
                    current_unit = "m"
                    m = 1000000000 * 60
                elif current_unit == "m":
                    current_unit = "s"
                    m = 1000000000
                r = <int64_t>int("".join(number)) * m
                result += timedelta_as_neg(r, neg)
                have_hhmmss = 1
            else:
                raise ValueError(f"expecting hh:mm:ss format, received: {ts}")

            unit, number = [], []

        # 处理小数点
        elif c == ".":

            # 如果已经有数值且当前有单位
            if len(number) and current_unit is not None:

                # 如果当前单位不是分钟，则格式不符合预期
                if current_unit != "m":
                    raise ValueError("expected hh:mm:ss format before .")
                m = 1000000000
                r = <int64_t>int("".join(number)) * m
                result += timedelta_as_neg(r, neg)
                have_value = 1
                unit, number, frac = [], [], []

            have_dot = 1

        # 处理时间单位
        else:
            unit.append(c)
            have_value = 1
            have_dot = 0

    # 处理存在小数点但没有单位的情况
    if have_dot and len(unit):
        r = timedelta_from_spec(number, frac, unit)
        result += timedelta_as_neg(r, neg)

    # 处理存在小数点的正常格式，如 hh:mm:ss.fffffff
    # 如果存在小数点的情况
    elif have_dot:
        # 如果小数部分或整数部分长度不为零且单位长度为零，并且当前单位为None，则抛出数值错误异常
        if ((len(number) or len(frac)) and not len(unit)
                and current_unit is None):
            raise ValueError("no units specified")

        # 根据小数部分的长度确定乘数m的值，以转换为纳秒
        if len(frac) > 0 and len(frac) <= 3:
            m = 10**(3 - len(frac)) * 1000 * 1000
        elif len(frac) > 3 and len(frac) <= 6:
            m = 10**(6 - len(frac)) * 1000
        elif len(frac) > 6 and len(frac) <= 9:
            m = 10**(9 - len(frac))
        else:
            m = 1
            frac = frac[:9]
        
        # 计算纳秒级别的时间间隔，并添加到结果中
        r = <int64_t>int("".join(frac)) * m
        result += timedelta_as_neg(r, neg)

    # 正常格式处理
    # 此时必须有秒数（因此单位仍然是'm'）
    elif current_unit is not None:
        # 如果当前单位不是'm'，则抛出数值错误异常
        if current_unit != "m":
            raise ValueError("expected hh:mm:ss format")
        
        # 每个小时的纳秒数
        m = 1000000000
        r = <int64_t>int("".join(number)) * m
        result += timedelta_as_neg(r, neg)

    # 最后一个单位缩写
    elif len(unit):
        # 如果存在数字，则计算时间间隔
        if len(number):
            r = timedelta_from_spec(number, frac, unit)
            result += timedelta_as_neg(r, neg)
        else:
            # 单位缩写没有数字时抛出数值错误异常
            raise ValueError("unit abbreviation w/o a number")

    # 只有符号，没有数字
    elif len(number) == 0:
        # 没有数字却有符号时抛出数值错误异常
        raise ValueError("symbols w/o a number")

    # 将作为纳秒处理
    # 但仅在没有其他情况下
    else:
        # 如果有多余的值存在，则抛出数值错误异常
        if have_value:
            raise ValueError("have leftover units")
        # 如果存在数字，则计算时间间隔为纳秒
        if len(number):
            r = timedelta_from_spec(number, frac, "ns")
            result += timedelta_as_neg(r, neg)

    # 返回计算结果
    return result
# 定义一个 Cython 函数，将时间差值转换为负数（如果需要的话）
cdef int64_t timedelta_as_neg(int64_t value, bint neg):
    """
    Parameters
    ----------
    value : int64_t
        时间差值
    neg : bool
        是否为负数

    Returns
    -------
    int64_t
        负数或正数的时间差值
    """
    if neg:
        return -value
    return value


# 定义一个 Cython 函数，从给定的数、小数和单位解析时间差值
cdef timedelta_from_spec(object number, object frac, object unit):
    """
    Parameters
    ----------
    number : object
        数字列表
    frac : object
        小数列表
    unit : object
        单位字符列表

    Returns
    -------
    timedelta
        解析后的时间差值

    Raises
    ------
    ValueError
        如果单位为 'M', 'Y' 或 'y'，因其无法唯一表示时间差值且不受支持
    """
    cdef:
        str n

    unit = "".join(unit)  # 将单位字符列表合并为字符串
    if unit in ["M", "Y", "y"]:
        raise ValueError(
            "Units 'M', 'Y' and 'y' do not represent unambiguous timedelta "
            "values and are not supported."
        )

    unit = parse_timedelta_unit(unit)  # 解析时间差值的单位

    n = "".join(number) + "." + "".join(frac)  # 合并数字和小数部分为一个字符串
    return cast_from_unit(float(n), unit)  # 将字符串表示的数值转换为指定单位的时间差值


# 解析时间差值的单位字符串，返回规范化后的单位字符串
cpdef inline str parse_timedelta_unit(str unit):
    """
    Parameters
    ----------
    unit : str or None
        单位字符串或 None

    Returns
    -------
    str
        规范化后的单位字符串

    Raises
    ------
    ValueError
        如果输入的单位字符串无法解析
    """
    if unit is None:
        return "ns"
    elif unit == "M":
        return unit
    elif unit in c_DEPR_UNITS:
        warnings.warn(
            f"\'{unit}\' is deprecated and will be removed in a "
            f"future version. Please use \'{c_DEPR_UNITS.get(unit)}\' "
            f"instead of \'{unit}\'.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        unit = c_DEPR_UNITS[unit]  # 替换已废弃的单位字符串
    try:
        return timedelta_abbrevs[unit.lower()]  # 返回单位字符串的规范化形式
    except KeyError:
        raise ValueError(f"invalid unit abbreviation: {unit}")


# ----------------------------------------------------------------------
# 时间差值操作工具函数

# 检查是否与给定对象兼容进行操作，返回 True 或 False
cdef bint _validate_ops_compat(other):
    """
    Parameters
    ----------
    other : object
        待检查的对象

    Returns
    -------
    bint
        True 表示兼容，False 表示不兼容
    """
    if checknull_with_nat(other):
        return True
    elif is_any_td_scalar(other):
        return True
    elif isinstance(other, str):
        return True
    return False


# 定义一个操作符方法的辅助函数，返回新的时间差值对象
def _op_unary_method(func, name):
    """
    Parameters
    ----------
    func : function
        执行操作的函数
    name : str
        方法名称

    Returns
    -------
    function
        执行操作的函数并返回新的时间差值对象
    """
    def f(self):
        new_value = func(self._value)
        return _timedelta_from_value_and_reso(Timedelta, new_value, self._creso)
    f.__name__ = name
    return f


# 定义一个二元操作符方法的辅助函数，用于与时间差值或时间差值数组进行操作
def _binary_op_method_timedeltalike(op, name):
    """
    Parameters
    ----------
    op : operation
        执行的操作
    name : str
        方法名称

    Notes
    -----
    定义一个二元操作，仅在另一个参数为时间差值或时间差值数组时有效
    """
    # define a binary operation that only works if the other argument is
    # timedelta like or an array of timedeltalike
    # 定义一个方法 `f`，接受参数 `other`
    def f(self, other):
        # 如果 `other` 是 `NaT`，返回 `NaT`
        if other is NaT:
            return NaT

        # 如果 `other` 是 `datetime64` 对象或者是 `PyDateTime` 对象但不是 `ABCTimestamp` 的实例
        elif cnp.is_datetime64_object(other) or (
            PyDateTime_Check(other) and not isinstance(other, ABCTimestamp)
        ):
            # 这种情况处理的是一个特定的 `datetime` 对象，但不是 `Timestamp`，因为 `Timestamp` 的情况在下面的 `_validate_ops_compat` 返回 `False` 后处理
            from pandas._libs.tslibs.timestamps import Timestamp
            return op(self, Timestamp(other))
            # 我们隐含地要求规范行为由 `Timestamp` 方法定义。

        # 如果 `other` 是数组
        elif is_array(other):
            # 如果 `other` 是零维数组
            if other.ndim == 0:
                # 参见：`item_from_zerodim`
                # 从零维数组中获取项目
                item = cnp.PyArray_ToScalar(cnp.PyArray_DATA(other), other)
                # 递归调用 `f`，处理 `self` 和 `item`
                return f(self, item)

            # 如果 `other` 的数据类型种类是 `mM`（日期时间类型）
            elif other.dtype.kind in "mM":
                # 对 `self` 调用 `to_timedelta64` 方法，然后与 `other` 进行操作
                return op(self.to_timedelta64(), other)
            # 如果 `other` 的数据类型种类是 `O`（对象类型）
            elif other.dtype.kind == "O":
                # 对 `other` 中的每个元素 `x`，对 `self` 和 `x` 进行操作，然后返回结果的数组
                return np.array([op(self, x) for x in other])
            else:
                # 否则返回 `NotImplemented`
                return NotImplemented

        # 如果 `other` 与 `self` 的操作不兼容
        elif not _validate_ops_compat(other):
            # 包括任何非 Cython 类的情况
            return NotImplemented

        # 尝试将 `other` 转换为 `Timedelta` 对象
        try:
            other = Timedelta(other)
        except ValueError:
            # 如果解析为 `Timedelta` 失败，返回 `NotImplemented`
            return NotImplemented

        # 如果 `other` 是 `NaT`
        if other is NaT:
            # 例如，如果原始的 `other` 是 `timedelta64('NaT')`
            return NaT

        # 匹配 numpy 的行为，将值转换为更高的分辨率。与 numpy 不同的是，我们在转换期间会引发异常而不是静默溢出。
        # 如果 `self` 的分辨率 `_creso` 小于 `other` 的分辨率 `_creso`
        if self._creso < other._creso:
            # 将 `self` 转换为 `_creso` 更高的分辨率
            self = (<_Timedelta>self)._as_creso(other._creso, round_ok=True)
        # 如果 `self` 的分辨率 `_creso` 大于 `other` 的分辨率 `_creso`
        elif self._creso > other._creso:
            # 将 `other` 转换为 `_creso` 更高的分辨率
            other = (<_Timedelta>other)._as_creso(self._creso, round_ok=True)

        # 对 `self` 和 `other` 的值 `_value` 执行操作 `op`
        res = op(self._value, other._value)
        # 如果结果为 `NPY_NAT`
        if res == NPY_NAT:
            # 例如，`test_implementation_limits` 中的情况
            # TODO: 在 `op` 中更一般地可以进行溢出检查？
            return NaT

        # 根据值 `res` 和分辨率 `_creso` 创建 `Timedelta` 对象并返回
        return _timedelta_from_value_and_reso(Timedelta, res, reso=self._creso)

    # 设置方法 `f` 的名称为 `name`
    f.__name__ = name
    # 返回方法 `f`
    return f
# ----------------------------------------------------------------------
# Timedelta Construction

# 拒绝不明确的时间单位，如 'Y', 'y', 'M'
cpdef disallow_ambiguous_unit(unit):
    if unit in {"Y", "y", "M"}:
        raise ValueError(
            "Units 'M', 'Y', and 'y' are no longer supported, as they do not "
            "represent unambiguous timedelta values durations."
        )

# 定义一个Cython函数，解析ISO 8601格式的时间字符串并返回精确到纳秒的整数值
cdef int64_t parse_iso_format_string(str ts) except? -1:
    """
    从ISO 8601持续时间格式字符串的匹配对象中提取和清理相应的值

    Parameters
    ----------
    ts: str
        ISO 8601持续时间格式的字符串

    Returns
    -------
    ns: int64_t
        匹配的ISO 8601持续时间的纳秒精度

    Raises
    ------
    ValueError
        如果无法解析``ts``
    """

    cdef:
        unicode c  # Unicode字符
        int64_t result = 0, r  # 结果值和中间结果值
        int p = 0, sign = 1  # 指针和符号
        object dec_unit = "ms", err_msg  # 小数单位和错误消息对象
        bint have_dot = 0, have_value = 0, neg = 0  # 是否有小数点、是否有值、是否为负数
        list number = [], unit = []  # 数字和单位列表

    err_msg = f"Invalid ISO 8601 Duration format - {ts}"  # 错误消息，说明格式无效

    if ts[0] == "-":
        sign = -1
        ts = ts[1:]
    for c in ts:
        # 遍历字符串 ts 中的每个字符 c
        # 如果字符 c 是数字 (ascii 码范围内)
        if 48 <= ord(c) <= 57:
            # 标记已有数值
            have_value = 1
            # 如果有小数点
            if have_dot:
                # 如果小数点后的单位是 "ns" 且位置 p 是 3，则将单位添加到 unit 列表中，并根据当前单位调整 dec_unit
                if p == 3 and dec_unit != "ns":
                    unit.append(dec_unit)
                    if dec_unit == "ms":
                        dec_unit = "us"
                    elif dec_unit == "us":
                        dec_unit = "ns"
                    p = 0
                p += 1

            # 如果 unit 列表为空，则将当前数字字符 c 添加到 number 列表中
            if not len(unit):
                number.append(c)
            else:
                # 根据 number 和 unit 调用 timedelta_from_spec 函数生成 timedelta 对象 r，并将其作为负数添加到 result 中
                r = timedelta_from_spec(number, "0", unit)
                result += timedelta_as_neg(r, neg)

                neg = 0
                unit, number = [], [c]
        else:
            # 处理非数字字符 c
            if c == "P" or c == "T":
                pass  # 忽略标记字符 P 和 T
            elif c == "-":
                # 处理负号字符
                if neg or have_value:
                    raise ValueError(err_msg)
                else:
                    neg = 1
            elif c == "+":
                pass  # 忽略加号字符
            elif c in ["W", "D", "H", "M"]:
                # 处理时间单位字符
                if c in ["H", "M"] and len(number) > 2:
                    raise ValueError(err_msg)
                if c in ["M", "H"]:
                    c = c.replace("M", "min").replace("H", "h")
                unit.append(c)
                # 根据 number 和 unit 调用 timedelta_from_spec 函数生成 timedelta 对象 r，并将其作为负数添加到 result 中
                r = timedelta_from_spec(number, "0", unit)
                result += timedelta_as_neg(r, neg)

                neg = 0
                unit, number = [], []
            elif c == ".":
                # 处理小数点字符
                # 如果 number 列表中有内容，则根据 number 和 "s" 调用 timedelta_from_spec 函数生成 timedelta 对象 r，并将其作为负数添加到 result 中
                if len(number):
                    r = timedelta_from_spec(number, "0", "s")
                    result += timedelta_as_neg(r, neg)
                    unit, number = [], []
                have_dot = 1
            elif c == "S":
                # 处理字符 "S"
                if have_dot:  # 处理毫秒、微秒或纳秒
                    if not len(number) or p > 3:
                        raise ValueError(err_msg)
                    # 根据需求将 number 列表填充至 3 位数
                    pad = 3 - p
                    while pad > 0:
                        number.append("0")
                        pad -= 1

                    # 根据 number 和 dec_unit 调用 timedelta_from_spec 函数生成 timedelta 对象 r，并将其作为负数添加到 result 中
                    r = timedelta_from_spec(number, "0", dec_unit)
                    result += timedelta_as_neg(r, neg)
                else:  # 处理秒数
                    # 根据 number 和 "s" 调用 timedelta_from_spec 函数生成 timedelta 对象 r，并将其作为负数添加到 result 中
                    r = timedelta_from_spec(number, "0", "s")
                    result += timedelta_as_neg(r, neg)
            else:
                # 如果字符 c 无法识别，则抛出 ValueError 异常
                raise ValueError(err_msg)

    if not have_value:
        # 如果没有解析出任何值，则抛出 ValueError 异常
        raise ValueError(err_msg)

    # 返回带有符号的 result 结果
    return sign * result
cdef _to_py_int_float(v):
    # 如果输入的值是整数对象，则转换为整数并返回
    if is_integer_object(v):
        return int(v)
    # 如果输入的值是浮点数对象，则转换为浮点数并返回
    elif is_float_object(v):
        return float(v)
    # 如果输入的值不是整数或浮点数对象，则抛出类型错误异常
    raise TypeError(f"Invalid type {type(v)}. Must be int or float.")


def _timedelta_unpickle(value, reso):
    # 从值和分辨率创建一个 Timedelta 对象并返回
    return _timedelta_from_value_and_reso(Timedelta, value, reso)


cdef _timedelta_from_value_and_reso(cls, int64_t value, NPY_DATETIMEUNIT reso):
    # 如果值为 NPY_NAT，则断言失败
    assert value != NPY_NAT
    # 根据不同的时间分辨率创建一个 _Timedelta 对象
    if reso == NPY_FR_ns:
        td_base = _Timedelta.__new__(cls, microseconds=int(value) // 1000)
    elif reso == NPY_DATETIMEUNIT.NPY_FR_us:
        td_base = _Timedelta.__new__(cls, microseconds=int(value))
    elif reso == NPY_DATETIMEUNIT.NPY_FR_ms:
        td_base = _Timedelta.__new__(cls, milliseconds=0)
    elif reso == NPY_DATETIMEUNIT.NPY_FR_s:
        td_base = _Timedelta.__new__(cls, seconds=0)
    # 其他分辨率未启用，但在这里可以潜在地实现
    else:
        raise NotImplementedError(
            "Only resolutions 's', 'ms', 'us', 'ns' are supported."
        )

    # 设置 _Timedelta 对象的值、标志和分辨率
    td_base._value = value
    td_base._is_populated = 0
    td_base._creso = reso
    return td_base


class MinMaxReso:
    """
    We need to define min/max/resolution on both the Timedelta _instance_
    and Timedelta class.  On an instance, these depend on the object's _reso.
    On the class, we default to the values we would get with nanosecond _reso.
    """
    def __init__(self, name):
        self._name = name

    def __get__(self, obj, type=None):
        # 如果属性名为 "min"，则返回 np.int64 的最小值加1
        if self._name == "min":
            val = np.iinfo(np.int64).min + 1
        # 如果属性名为 "max"，则返回 np.int64 的最大值
        elif self._name == "max":
            val = np.iinfo(np.int64).max
        else:
            assert self._name == "resolution"
            val = 1

        if obj is None:
            # 如果 obj 为 None，表示这是类级别的调用，返回一个 Timedelta 对象
            # 默认使用纳秒分辨率
            return Timedelta(val)
        else:
            # 如果 obj 不为 None，返回根据 obj 的分辨率创建的 Timedelta 对象
            return Timedelta._from_value_and_reso(val, obj._creso)

    def __set__(self, obj, value):
        # 不允许设置属性值，抛出 AttributeError 异常
        raise AttributeError(f"{self._name} is not settable.")
# timedeltas that we need to do object instantiation in python. This will
# serve as a C extension type that shadows the Python class, where we do any
# heavy lifting.
cdef class _Timedelta(timedelta):
    # cdef readonly:
    #    int64_t value      # nanoseconds
    #    bint _is_populated  # are my components populated
    #    int64_t _d, _h, _m, _s, _ms, _us, _ns
    #    NPY_DATETIMEUNIT _reso

    # higher than np.ndarray and np.matrix
    __array_priority__ = 100
    # Define constants representing minimum, maximum, and resolution values for timedelta objects
    min = MinMaxReso("min")
    max = MinMaxReso("max")
    resolution = MinMaxReso("resolution")

    @property
    def value(self):
        """
        Return the value of Timedelta object in nanoseconds.

        Return the total seconds, milliseconds and microseconds
        of the timedelta as nanoseconds.

        Returns
        -------
        int

        See Also
        --------
        Timedelta.unit : Return the unit of Timedelta object.

        Examples
        --------
        >>> pd.Timedelta(1, "us").value
        1000
        """
        try:
            # Convert the timedelta value to nanoseconds using a conversion function
            return convert_reso(self._value, self._creso, NPY_FR_ns, False)
        except OverflowError:
            # Raise an OverflowError if the conversion to nanoseconds would cause overflow
            raise OverflowError(
                "Cannot convert Timedelta to nanoseconds without overflow. "
                "Use `.asm8.view('i8')` to cast represent Timedelta in its own "
                f"unit (here, {self.unit})."
            )

    @property
    def _unit(self) -> str:
        """
        The abbreviation associated with self._creso.
        """
        # Return the unit abbreviation associated with the current resolution
        return npy_unit_to_abbrev(self._creso)

    @property
    def days(self) -> int:  # TODO(cython3): make cdef property
        """
        Returns the days of the timedelta.

        The `days` attribute of a `pandas.Timedelta` object provides the number
        of days represented by the `Timedelta`. This is useful for extracting
        the day component from a `Timedelta` that may also include hours, minutes,
        seconds, and smaller time units. This attribute simplifies the process
        of working with durations where only the day component is of interest.

        Returns
        -------
        int

        See Also
        --------
        Timedelta.seconds : Returns the seconds component of the timedelta.
        Timedelta.microseconds : Returns the microseconds component of the timedelta.
        Timedelta.total_seconds : Returns the total duration in seconds.

        Examples
        --------
        >>> td = pd.Timedelta(1, "d")
        >>> td.days
        1

        >>> td = pd.Timedelta('4 min 3 us 42 ns')
        >>> td.days
        0
        """
        # Ensure that the timedelta components are populated and return the days component
        # of the timedelta object
        self._ensure_components()
        return self._d

    @property
    def seconds(self) -> int:  # TODO(cython3): make cdef property
        """
        Return the total hours, minutes, and seconds of the timedelta as seconds.

        Timedelta.seconds = hours * 3600 + minutes * 60 + seconds.

        Returns
        -------
        int
            Number of seconds.

        See Also
        --------
        Timedelta.components : Return all attributes with assigned values
            (i.e. days, hours, minutes, seconds, milliseconds, microseconds,
            nanoseconds).
        Timedelta.total_seconds : Express the Timedelta as total number of seconds.

        Examples
        --------
        **Using string input**

        >>> td = pd.Timedelta('1 days 2 min 3 us 42 ns')
        >>> td.seconds
        120

        **Using integer input**

        >>> td = pd.Timedelta(42, unit='s')
        >>> td.seconds
        42
        """
        # NB: using the python C-API PyDateTime_DELTA_GET_SECONDS will fail
        #  (or be incorrect)
        # Ensure that the components of the timedelta are up to date
        self._ensure_components()
        # Calculate total seconds from hours, minutes, and seconds components
        return self._h * 3600 + self._m * 60 + self._s

    @property
    def microseconds(self) -> int:  # TODO(cython3): make cdef property
        # NB: using the python C-API PyDateTime_DELTA_GET_MICROSECONDS will fail
        #  (or be incorrect)
        """
        Return the number of microseconds (n), where 0 <= n < 1 millisecond.

        Timedelta.microseconds = milliseconds * 1000 + microseconds.

        Returns
        -------
        int
            Number of microseconds.

        See Also
        --------
        Timedelta.components : Return all attributes with assigned values
            (i.e. days, hours, minutes, seconds, milliseconds, microseconds,
            nanoseconds).

        Examples
        --------
        **Using string input**

        >>> td = pd.Timedelta('1 days 2 min 3 us')

        >>> td.microseconds
        3

        **Using integer input**

        >>> td = pd.Timedelta(42, unit='us')
        >>> td.microseconds
        42
        """
        # Ensure that the components of the timedelta are up to date
        self._ensure_components()
        # Calculate total microseconds from milliseconds and microseconds components
        return self._ms * 1000 + self._us

    def total_seconds(self) -> float:
        """
        Total seconds in the duration.

        Examples
        --------
        >>> td = pd.Timedelta('1min')
        >>> td
        Timedelta('0 days 00:01:00')
        >>> td.total_seconds()
        60.0
        """
        # We need to override bc we overrode days/seconds/microseconds
        # TODO: add nanos/1e9?
        # Calculate total seconds including days, seconds, and microseconds
        return self.days * 24 * 3600 + self.seconds + self.microseconds / 1_000_000

    @property
    # 返回 Timedelta 对象的单位字符串表示，默认为纳秒 ('ns')
    def unit(self) -> str:
        """
        Return the unit of Timedelta object.

        The unit of Timedelta object is nanosecond, i.e., 'ns' by default.

        Returns
        -------
        str

        See Also
        --------
        Timedelta.value : Return the value of Timedelta object in nanoseconds.
        Timedelta.as_unit : Convert the underlying int64 representation to
            the given unit.

        Examples
        --------
        >>> td = pd.Timedelta(42, unit='us')
        'ns'
        """
        return npy_unit_to_abbrev(self._creso)

    # 定义 Timedelta 对象的哈希函数
    def __hash__(_Timedelta self):
        # 如果 Timedelta 对象的分辨率是纳秒，则返回其值的哈希码
        if self._has_ns():
            # 注意：这里不满足不变性条件
            # td1 == td2 \\Rightarrow hash(td1) == hash(td2)
            # 如果 td1 和 td2 的 _resos 不同，timedelta64 也有这种不满足不变性的行为。
            # 参见 GH#44504
            return hash(self._value)
        # 如果 Timedelta 对象在 Python timedelta 能处理的范围内，并且分辨率是纳秒或微秒
        elif self._is_in_pytimedelta_bounds() and (
            self._creso == NPY_FR_ns or self._creso == NPY_DATETIMEUNIT.NPY_FR_us
        ):
            # 如果可以委托给 timedelta.__hash__ 处理，则这样做，因为这样可以确保哈希码不受 _reso 影响。
            # 只有纳秒和微秒可以委托，因为在这两个分辨率下，我们在 _timedelta_from_value_and_reso 中使用正确的输入调用 _Timedelta.__new__;
            # 因此 timedelta.__hash__ 会是正确的。
            return timedelta.__hash__(self)
        else:
            # 我们希望确保两个等效的 Timedelta 对象具有相同的哈希码。因此，我们尝试将分辨率降低到下一个较低的级别。
            try:
                # 尝试将对象转换为较低分辨率
                obj = (<_Timedelta>self)._as_creso(<NPY_DATETIMEUNIT>(self._creso + 1))
            except OutOfBoundsTimedelta:
                # 转换失败，则返回当前对象的值的哈希码
                # 超出范围的 Timedelta 不需要处理哈希码
                return hash(self._value)
            else:
                # 如果转换成功，则返回转换后对象的哈希码
                return hash(obj)
    # 定义一个特殊方法 __richcmp__，用于比较两个 _Timedelta 对象或与其他对象的关系
    def __richcmp__(_Timedelta self, object other, int op):
        # 声明一个 _Timedelta 类型的变量 ots
        cdef:
            _Timedelta ots

        # 如果 other 是 _Timedelta 类型的实例，则直接赋值给 ots
        if isinstance(other, _Timedelta):
            ots = other
        # 如果 other 是任何 timedelta 标量类型的实例
        elif is_any_td_scalar(other):
            try:
                # 尝试将 other 转换为 Timedelta 对象
                ots = Timedelta(other)
            except OutOfBoundsTimedelta as err:
                # 如果 other 超出了 Timedelta 的范围
                # GH#49021 pytimedelta.max 溢出
                if not PyDelta_Check(other):
                    # TODO: 处理这种情况
                    raise
                # 构建当前对象和 other 对象的时间元组
                ltup = (self.days, self.seconds, self.microseconds, self.nanoseconds)
                rtup = (other.days, other.seconds, other.microseconds, 0)
                # 根据操作符 op 执行比较操作
                if op == Py_EQ:
                    return ltup == rtup
                elif op == Py_NE:
                    return ltup != rtup
                elif op == Py_LT:
                    return ltup < rtup
                elif op == Py_LE:
                    return ltup <= rtup
                elif op == Py_GT:
                    return ltup > rtup
                elif op == Py_GE:
                    return ltup >= rtup

        # 如果 other 是 NaT（Not a Time），则执行不等于操作
        elif other is NaT:
            return op == Py_NE

        # 如果 other 是数组类型
        elif util.is_array(other):
            # 如果数组元素的类型是时间类型 "m"
            if other.dtype.kind == "m":
                # 调用 PyObject_RichCompare 函数进行比较
                return PyObject_RichCompare(self.asm8, other, op)
            # 如果数组元素的类型是对象类型 "O"
            elif other.dtype.kind == "O":
                # 对数组中的每个元素执行元素级操作
                return np.array(
                    [PyObject_RichCompare(self, x, op) for x in other],
                    dtype=bool,
                )
            # 如果操作是等于操作
            if op == Py_EQ:
                return np.zeros(other.shape, dtype=bool)
            # 如果操作是不等于操作
            elif op == Py_NE:
                return np.ones(other.shape, dtype=bool)
            # 返回 NotImplemented，交由其他对象处理 TypeError
            return NotImplemented

        else:
            # 返回 NotImplemented，交由其他对象处理比较
            return NotImplemented

        # 如果当前对象和 ots 对象的 _creso 属性相等，则执行标量比较
        if self._creso == ots._creso:
            return cmp_scalar(self._value, ots._value, op)
        # 否则执行不同分辨率的比较处理
        return self._compare_mismatched_resos(ots, op)

    # TODO: 与 Timestamp 共享重用
    # 定义一个方法 _compare_mismatched_resos，用于处理不同分辨率的 Timedelta 对象比较
    cdef bint _compare_mismatched_resos(self, _Timedelta other, op):
        # 不能简单地分派给 numpy，因为它们会静默溢出并出错
        cdef:
            npy_datetimestruct dts_self
            npy_datetimestruct dts_other

        # 调用 pandas_datetime_to_datetimestruct 函数处理当前对象和 other 对象的时间结构
        pandas_datetime_to_datetimestruct(self._value, self._creso, &dts_self)
        pandas_datetime_to_datetimestruct(other._value, other._creso, &dts_other)
        # 调用 cmp_dtstructs 函数比较两个时间结构对象
        return cmp_dtstructs(&dts_self,  &dts_other, op)

    # 定义一个方法 _has_ns，用于判断当前对象是否包含纳秒精度
    cdef bint _has_ns(self):
        # 如果当前对象的分辨率是 NPY_FR_ns，则判断值是否包含非零纳秒部分
        if self._creso == NPY_FR_ns:
            return self._value % 1000 != 0
        # 如果当前对象的分辨率小于 NPY_FR_ns，则认为不包含纳秒精度
        elif self._creso < NPY_FR_ns:
            # 即秒、毫秒、微秒等情况
            return False
        else:
            # 抛出 NotImplementedError 异常，表示当前分辨率未实现处理方法
            raise NotImplementedError(self._creso)
    # 检查当前对象的时间间隔是否在 datetime.timedelta 的范围内
    cdef bint _is_in_pytimedelta_bounds(self):
        """
        Check if we are within the bounds of datetime.timedelta.
        """
        # 确保时间间隔组件已经计算
        self._ensure_components()
        # 返回时间间隔是否在 -999999999 到 999999999 范围内的布尔值
        return -999999999 <= self._d and self._d <= 999999999

    cdef _ensure_components(_Timedelta self):
        """
        compute the components
        """
        # 如果时间间隔已经被计算过，则直接返回
        if self._is_populated:
            return

        cdef:
            pandas_timedeltastruct tds

        # 将 pandas Timedelta 转换为时间间隔结构体
        pandas_timedelta_to_timedeltastruct(self._value, self._creso, &tds)
        # 将计算得到的时间间隔组件赋值给对象的各个属性
        self._d = tds.days
        self._h = tds.hrs
        self._m = tds.min
        self._s = tds.sec
        self._ms = tds.ms
        self._us = tds.us
        self._ns = tds.ns
        self._seconds = tds.seconds
        self._microseconds = tds.microseconds

        # 将标志置为已经计算过
        self._is_populated = 1

    cpdef timedelta to_pytimedelta(_Timedelta self):
        """
        Convert a pandas Timedelta object into a python ``datetime.timedelta`` object.

        Timedelta objects are internally saved as numpy datetime64[ns] dtype.
        Use to_pytimedelta() to convert to object dtype.

        Returns
        -------
        datetime.timedelta or numpy.array of datetime.timedelta

        See Also
        --------
        to_timedelta : Convert argument to Timedelta type.

        Notes
        -----
        Any nanosecond resolution will be lost.

        Examples
        --------
        >>> td = pd.Timedelta('3D')
        >>> td
        Timedelta('3 days 00:00:00')
        >>> td.to_pytimedelta()
        datetime.timedelta(days=3)
        """
        # 如果时间间隔的分辨率为纳秒
        if self._creso == NPY_FR_ns:
            # 将微秒和剩余部分转换为微秒，并进行四舍五入
            us, remainder = divmod(self._value, 1000)
            if remainder >= 500:
                us += 1
            # 返回以微秒为单位的时间间隔对象
            return timedelta(microseconds=us)

        # 否则，确保时间间隔的组件已经计算，并返回相应的时间间隔对象
        self._ensure_components()
        return timedelta(
            days=self._d, seconds=self._seconds, microseconds=self._microseconds
        )

    def to_timedelta64(self) -> np.timedelta64:
        """
        Return a numpy.timedelta64 object with 'ns' precision.

        Examples
        --------
        >>> td = pd.Timedelta('3D')
        >>> td
        Timedelta('3 days 00:00:00')
        >>> td.to_timedelta64()
        numpy.timedelta64(259200000000000,'ns')
        """
        cdef:
            str abbrev = npy_unit_to_abbrev(self._creso)
        # TODO: 使用时间间隔分辨率直接创建一个 np.timedelta64 对象的方法，而非获取单位缩写的方式？
        # 返回一个以当前时间间隔值和单位缩写为参数的 numpy.timedelta64 对象
        return np.timedelta64(self._value, abbrev)
    @property
    def components(self):
        """
        Return a components namedtuple-like.

        Examples
        --------
        >>> td = pd.Timedelta('2 day 4 min 3 us 42 ns')
        >>> td.components
        Components(days=2, hours=0, minutes=4, seconds=0, milliseconds=0,
            microseconds=3, nanoseconds=42)
        """
        # Ensure components are computed if not already done
        self._ensure_components()
        # Return a namedtuple-like Components object with timedelta components
        return Components(self._d, self._h, self._m, self._s,
                          self._ms, self._us, self._ns)

    @property
    def asm8(self) -> np.timedelta64:
        """
        Return a numpy timedelta64 array scalar view.

        Provides access to the array scalar view (i.e. a combination of the
        value and the units) associated with the numpy.timedelta64().view(),
        including a 64-bit integer representation of the timedelta in
        nanoseconds (Python int compatible).

        Returns
        -------
        numpy timedelta64 array scalar view
            Array scalar view of the timedelta in nanoseconds.

        Examples
        --------
        >>> td = pd.Timedelta('1 days 2 min 3 us 42 ns')
        >>> td.asm8
        numpy.timedelta64(86520000003042,'ns')

        >>> td = pd.Timedelta('2 min 3 s')
        >>> td.asm8
        numpy.timedelta64(123000000000,'ns')

        >>> td = pd.Timedelta('3 ms 5 us')
        >>> td.asm8
        numpy.timedelta64(3005000,'ns')

        >>> td = pd.Timedelta(42, unit='ns')
        >>> td.asm8
        numpy.timedelta64(42,'ns')
        """
        # Convert Timedelta to numpy timedelta64 representation
        return self.to_timedelta64()
    def resolution_string(self) -> str:
        """
        Return a string representing the lowest timedelta resolution.

        Each timedelta has a defined resolution that represents the lowest OR
        most granular level of precision. Each level of resolution is
        represented by a short string as defined below:

        Resolution:     Return value

        * Days:         'D'
        * Hours:        'h'
        * Minutes:      'min'
        * Seconds:      's'
        * Milliseconds: 'ms'
        * Microseconds: 'us'
        * Nanoseconds:  'ns'

        Returns
        -------
        str
            Timedelta resolution.

        Examples
        --------
        >>> td = pd.Timedelta('1 days 2 min 3 us 42 ns')
        >>> td.resolution_string
        'ns'

        >>> td = pd.Timedelta('1 days 2 min 3 us')
        >>> td.resolution_string
        'us'

        >>> td = pd.Timedelta('2 min 3 s')
        >>> td.resolution_string
        's'

        >>> td = pd.Timedelta(36, unit='us')
        >>> td.resolution_string
        'us'
        """
        # Ensure that internal components are computed
        self._ensure_components()
        
        # Determine the timedelta resolution based on the smallest component
        if self._ns:
            return "ns"
        elif self._us:
            return "us"
        elif self._ms:
            return "ms"
        elif self._s:
            return "s"
        elif self._m:
            return "min"
        elif self._h:
            return "h"
        else:
            return "D"

    @property
    def nanoseconds(self):
        """
        Return the number of nanoseconds (n), where 0 <= n < 1 microsecond.

        Returns
        -------
        int
            Number of nanoseconds.

        See Also
        --------
        Timedelta.components : Return all attributes with assigned values
            (i.e. days, hours, minutes, seconds, milliseconds, microseconds,
            nanoseconds).

        Examples
        --------
        **Using string input**

        >>> td = pd.Timedelta('1 days 2 min 3 us 42 ns')

        >>> td.nanoseconds
        42

        **Using integer input**

        >>> td = pd.Timedelta(42, unit='ns')
        >>> td.nanoseconds
        42
        """
        # Ensure that internal components are computed
        self._ensure_components()
        
        # Return the number of nanoseconds
        return self._ns
    def _repr_base(self, format=None) -> str:
        """
        根据指定的格式生成时间差的字符串表示形式。

        Parameters
        ----------
        format : None|all|sub_day|long
            格式化选项，控制生成的字符串的内容和格式。

        Returns
        -------
        converted : string of a Timedelta
            生成的时间差字符串表示形式。
        """
        cdef:
            str sign, fmt  # 定义字符串变量 sign 和 fmt
            dict comp_dict  # 定义字典变量 comp_dict
            object subs  # 定义对象变量 subs

        self._ensure_components()  # 确保已经计算出时间差的各个组成部分

        if self._d < 0:
            sign = " +"  # 如果时间差为负数，设置符号为 "+"
        else:
            sign = " "  # 如果时间差为非负数，设置符号为空格

        if format == "all":
            # 如果格式为 "all"，定义详细的时间差格式
            fmt = ("{days} days{sign}{hours:02}:{minutes:02}:{seconds:02}."
                   "{milliseconds:03}{microseconds:03}{nanoseconds:03}")
        else:
            # 否则，如果有不完整的天数或时间单位（时、分、秒等）
            subs = (self._h or self._m or self._s or
                    self._ms or self._us or self._ns)

            if self._ms or self._us or self._ns:
                # 根据存在的毫秒、微秒或纳秒添加到秒的格式化字符串
                seconds_fmt = "{seconds:02}.{milliseconds:03}{microseconds:03}"
                if self._ns:
                    # 如果存在纳秒，额外添加纳秒的格式化
                    seconds_fmt += "{nanoseconds:03}"
            else:
                seconds_fmt = "{seconds:02}"

            if format == "sub_day" and not self._d:
                # 如果格式为 "sub_day" 且不存在完整的天数，则只显示时、分、秒
                fmt = "{hours:02}:{minutes:02}:" + seconds_fmt
            elif subs or format == "long":
                # 如果存在不完整的天数或指定为 "long" 格式，显示完整的时间差信息
                fmt = "{days} days{sign}{hours:02}:{minutes:02}:" + seconds_fmt
            else:
                fmt = "{days} days"  # 否则，只显示天数

        comp_dict = self.components._asdict()  # 将时间差的各个组成部分转换为字典形式
        comp_dict["sign"] = sign  # 将符号添加到组成部分字典中

        return fmt.format(**comp_dict)  # 使用格式化字符串生成时间差的字符串表示形式

    def __repr__(self) -> str:
        """
        返回时间差对象的详细字符串表示形式。

        Returns
        -------
        str
            包含时间差详细信息的字符串。
        """
        repr_based = self._repr_base(format="long")  # 使用 "long" 格式生成详细的时间差字符串
        return f"Timedelta('{repr_based}')"  # 返回包含时间差详细信息的字符串形式

    def __str__(self) -> str:
        """
        返回时间差对象的字符串表示形式。

        Returns
        -------
        str
            时间差的字符串表示形式。
        """
        return self._repr_base(format="long")  # 直接返回使用 "long" 格式生成的时间差字符串

    def __bool__(self) -> bool:
        """
        判断时间差对象是否为真（非零）。

        Returns
        -------
        bool
            如果时间差对象不等于零，返回 True；否则返回 False。
        """
        return self._value != 0  # 返回时间差对象的值是否不等于零的布尔值
    def isoformat(self) -> str:
        """
        Format the Timedelta as ISO 8601 Duration.

        ``P[n]Y[n]M[n]DT[n]H[n]M[n]S``, where the ``[n]`` s are replaced by the
        values. See https://en.wikipedia.org/wiki/ISO_8601#Durations.

        Returns
        -------
        str
            ISO 8601 formatted duration string representing the Timedelta.

        See Also
        --------
        Timestamp.isoformat : Function is used to convert the given
            Timestamp object into the ISO format.

        Notes
        -----
        The longest component is days, whose value may be larger than
        365.
        Every component is always included, even if its value is 0.
        Pandas uses nanosecond precision, so up to 9 decimal places may
        be included in the seconds component.
        Trailing 0's are removed from the seconds component after the decimal.
        We do not 0 pad components, so it's `...T5H...`, not `...T05H...`

        Examples
        --------
        >>> td = pd.Timedelta(days=6, minutes=50, seconds=3,
        ...                   milliseconds=10, microseconds=10, nanoseconds=12)

        >>> td.isoformat()
        'P6DT0H50M3.010010012S'
        >>> pd.Timedelta(hours=1, seconds=10).isoformat()
        'P0DT1H0M10S'
        >>> pd.Timedelta(days=500.5).isoformat()
        'P500DT12H0M0S'
        """
        # 获取时间差对象的各个时间组件
        components = self.components
        # 构建表示秒数的字符串，包括毫秒、微秒和纳秒
        seconds = (f"{components.seconds}."
                   f"{components.milliseconds:0>3}"
                   f"{components.microseconds:0>3}"
                   f"{components.nanoseconds:0>3}")
        # 去除秒数字符串末尾的不必要的0和小数点
        seconds = seconds.rstrip("0").rstrip(".")
        # 构建最终的 ISO 8601 时间差格式字符串
        tpl = (f"P{components.days}DT{components.hours}"
               f"H{components.minutes}M{seconds}S")
        return tpl

    # ----------------------------------------------------------------
    # Constructors

    @classmethod
    def _from_value_and_reso(cls, int64_t value, NPY_DATETIMEUNIT reso):
        """
        Class method for creating Timedelta object from value and resolution.

        Parameters
        ----------
        value : int64_t
            Integer value representing the timedelta.
        reso : NPY_DATETIMEUNIT
            Resolution unit for the timedelta.

        Returns
        -------
        Timedelta
            A Timedelta object created from the provided value and resolution.
        """
        # 用于测试目的，暴露为类方法
        return _timedelta_from_value_and_reso(cls, value, reso)
    def as_unit(self, str unit, bint round_ok=True):
        """
        Convert the underlying int64 representation to the given unit.

        Parameters
        ----------
        unit : {"ns", "us", "ms", "s"}
            The unit to which the timedelta should be converted.
        round_ok : bool, default True
            If False and rounding is necessary, raise an exception.

        Returns
        -------
        Timedelta
            A new Timedelta object representing the converted value.

        See Also
        --------
        Timedelta : Represents a duration, the difference between two dates or times.
        to_timedelta : Convert argument to timedelta.
        Timedelta.asm8 : Return a numpy timedelta64 array scalar view.

        Examples
        --------
        >>> td = pd.Timedelta('1001ms')
        >>> td
        Timedelta('0 days 00:00:01.001000')
        >>> td.as_unit('s')
        Timedelta('0 days 00:00:01')
        """
        # Define the data type corresponding to the specified unit
        dtype = np.dtype(f"m8[{unit}]")
        # Get the resolution corresponding to the dtype
        reso = get_unit_from_dtype(dtype)
        # Convert the timedelta to the specified resolution
        return self._as_creso(reso, round_ok=round_ok)

    @cython.cdivision(False)
    cdef _Timedelta _as_creso(self, NPY_DATETIMEUNIT reso, bint round_ok=True):
        """
        Convert the timedelta to a given resolution.

        Parameters
        ----------
        reso : NPY_DATETIMEUNIT
            The resolution to which the timedelta should be converted.
        round_ok : bint, default True
            If False and rounding is required, raise an exception.

        Returns
        -------
        _Timedelta
            A new _Timedelta object with the converted value.

        Raises
        ------
        OutOfBoundsTimedelta
            If the conversion leads to overflow.

        Notes
        -----
        This function is used for internal conversion purposes.

        See Also
        --------
        _maybe_cast_to_matching_resos : Adjusts timedelta resolution to match another timedelta.

        """
        cdef:
            int64_t value

        # If the resolution matches the current resolution, return self
        if reso == self._creso:
            return self

        try:
            # Convert the internal value to the new resolution
            value = convert_reso(self._value, self._creso, reso, round_ok=round_ok)
        except OverflowError as err:
            unit = npy_unit_to_abbrev(reso)
            raise OutOfBoundsTimedelta(
                f"Cannot cast {self} to unit='{unit}' without overflow."
            ) from err

        # Return a new _Timedelta object with the converted value and resolution
        return type(self)._from_value_and_reso(value, reso=reso)

    cpdef _maybe_cast_to_matching_resos(self, _Timedelta other):
        """
        Adjusts the resolution of self and other to match the higher resolution, raising on overflow.

        Parameters
        ----------
        other : _Timedelta
            Another _Timedelta object whose resolution might need adjustment.

        Returns
        -------
        _Timedelta
            Adjusted self timedelta.
        _Timedelta
            Adjusted other timedelta.
        """
        # If self resolution is higher than other, adjust other's resolution
        if self._creso > other._creso:
            other = other._as_creso(self._creso)
        # If self resolution is lower than other, adjust self's resolution
        elif self._creso < other._creso:
            self = self._as_creso(other._creso)
        return self, other
# Python前端到C扩展类型 _Timedelta 的接口
# 这充当timedelta64的包装盒

class Timedelta(_Timedelta):
    """
    表示持续时间，即两个日期或时间之间的差异。

    Timedelta 是 pandas 中等价于 Python 的 ``datetime.timedelta``，
    在大多数情况下可以互换使用。

    Parameters
    ----------
    value : Timedelta, timedelta, np.timedelta64, str, or int
        输入值。
    unit : str, 默认 'ns'
        如果输入是整数，表示输入的单位。

        可能的值有：

        * 'W', 或 'D'
        * 'days', 或 'day'
        * 'hours', 'hour', 'hr', 或 'h'
        * 'minutes', 'minute', 'min', 或 'm'
        * 'seconds', 'second', 'sec', 或 's'
        * 'milliseconds', 'millisecond', 'millis', 'milli', 或 'ms'
        * 'microseconds', 'microsecond', 'micros', 'micro', 或 'us'
        * 'nanoseconds', 'nanosecond', 'nanos', 'nano', 或 'ns'.

        .. deprecated:: 3.0.0

            允许 `w`, `d`, `MIN`, `MS`, `US` 和 `NS` 作为单位的值已弃用，
            推荐使用 `W`, `D`, `min`, `ms`, `us` 和 `ns`。

    **kwargs
        可用的kwargs：{days, seconds, microseconds,
        milliseconds, minutes, hours, weeks}。
        用于与 datetime.timedelta 兼容的构造值。
        将 numpy 的整数和浮点数强制转换为 Python 的整数和浮点数。

    See Also
    --------
    Timestamp : 表示时间戳。
    TimedeltaIndex : 不可变的 timedelta64 数据索引。
    DateOffset : 用于日期范围的标准日期增量。
    to_timedelta : 将参数转换为 timedelta。
    datetime.timedelta : datetime 模块中表示持续时间的类。
    numpy.timedelta64 : 与 NumPy 兼容的持续时间。

    Notes
    -----
    构造函数可以接受 value 和 unit 两者的值，也可以使用上述 kwargs 中的任意一个。
    在初始化过程中必须使用其中之一。

    ``.value`` 属性始终以纳秒（ns）为单位。

    如果精度高于纳秒，持续时间的精度将被截断为纳秒。

    Examples
    --------
    使用 value 和 unit 初始化 Timedelta 对象的示例：

    >>> td = pd.Timedelta(1, "d")
    >>> td
    Timedelta('1 days 00:00:00')

    使用 kwargs 初始化 Timedelta 对象的示例：

    >>> td2 = pd.Timedelta(days=1)
    >>> td2
    Timedelta('1 days 00:00:00')

    我们可以看到两种方式都得到了相同的结果。
    """

    _req_any_kwargs_new = {"weeks", "days", "hours", "minutes", "seconds",
                           "milliseconds", "microseconds", "nanoseconds"}

    def __setstate__(self, state):
        if len(state) == 1:
            # 旧版 pickle，只支持纳秒
            value = state[0]
            reso = NPY_FR_ns
        else:
            value, reso = state
        self._value= value
        self._creso = reso
    def __reduce__(self):
        # 定义对象状态为当前值和精度
        object_state = self._value, self._creso
        # 返回一个元组，包含自定义的反序列化函数和对象状态
        return (_timedelta_unpickle, object_state)

    @cython.cdivision(True)
    def _round(self, freq, mode):
        """
        使用 Cython 加速的方法，将 Timedelta 对象舍入到指定的频率。

        Parameters
        ----------
        freq : str
            表示舍入分辨率的频率字符串，与类构造函数:class:`~pandas.Timedelta`使用相同的单位。
        mode : int
            舍入模式，定义舍入的行为。

        Returns
        -------
        Timedelta
            返回舍入到给定频率的新 Timedelta 对象。

        Raises
        ------
        OutOfBoundsTimedelta
            如果舍入操作导致溢出，则引发此异常。
        """
        cdef:
            int64_t result, unit
            ndarray[int64_t] arr

        # 根据频率获取舍入的单位
        unit = get_unit_for_round(freq, self._creso)

        # 创建包含当前值的 NumPy 数组
        arr = np.array([self._value], dtype="i8")
        try:
            # 调用 round_nsint64 函数进行舍入操作
            result = round_nsint64(arr, mode, unit)[0]
        except OverflowError as err:
            # 如果溢出则抛出 OutOfBoundsTimedelta 异常
            raise OutOfBoundsTimedelta(
                f"Cannot round {self} to freq={freq} without overflow"
            ) from err
        # 根据舍入结果创建新的 Timedelta 对象并返回
        return Timedelta._from_value_and_reso(result, self._creso)

    def round(self, freq):
        """
        将 Timedelta 对象舍入到指定的分辨率。

        Parameters
        ----------
        freq : str
            表示舍入分辨率的频率字符串，与类构造函数:class:`~pandas.Timedelta`使用相同的单位。

        Returns
        -------
        Timedelta
            返回舍入到给定频率的新 Timedelta 对象。

        Raises
        ------
        ValueError
            如果无法将频率转换为有效的单位时引发异常。

        Examples
        --------
        >>> td = pd.Timedelta('1001ms')
        >>> td
        Timedelta('0 days 00:00:01.001000')
        >>> td.round('s')
        Timedelta('0 days 00:00:01')
        """
        return self._round(freq, RoundTo.NEAREST_HALF_EVEN)

    def floor(self, freq):
        """
        返回一个按指定分辨率向下取整的新 Timedelta 对象。

        Parameters
        ----------
        freq : str
            表示取整分辨率的频率字符串，与类构造函数:class:`~pandas.Timedelta`使用相同的单位。

        Examples
        --------
        >>> td = pd.Timedelta('1001ms')
        >>> td
        Timedelta('0 days 00:00:01.001000')
        >>> td.floor('s')
        Timedelta('0 days 00:00:01')
        """
        return self._round(freq, RoundTo.MINUS_INFTY)

    def ceil(self, freq):
        """
        返回一个按指定分辨率向上取整的新 Timedelta 对象。

        Parameters
        ----------
        freq : str
            表示取整分辨率的频率字符串，与类构造函数:class:`~pandas.Timedelta`使用相同的单位。

        Examples
        --------
        >>> td = pd.Timedelta('1001ms')
        >>> td
        Timedelta('0 days 00:00:01.001000')
        >>> td.ceil('s')
        Timedelta('0 days 00:00:02')
        """
        return self._round(freq, RoundTo.PLUS_INFTY)

    # ----------------------------------------------------------------
    # Arithmetic Methods
    # TODO: Can some of these be defined in the cython class?

    __neg__ = _op_unary_method(lambda x: -x, "__neg__")
    __pos__ = _op_unary_method(lambda x: x, "__pos__")
    __abs__ = _op_unary_method(lambda x: abs(x), "__abs__")

    __add__ = _binary_op_method_timedeltalike(lambda x, y: x + y, "__add__")
    # 定义 __radd__ 方法，通过 _binary_op_method_timedeltalike 包装，实现时间增加操作
    __radd__ = _binary_op_method_timedeltalike(lambda x, y: x + y, "__radd__")

    # 定义 __sub__ 方法，通过 _binary_op_method_timedeltalike 包装，实现时间减法操作
    __sub__ = _binary_op_method_timedeltalike(lambda x, y: x - y, "__sub__")

    # 定义 __rsub__ 方法，通过 _binary_op_method_timedeltalike 包装，实现反向时间减法操作
    __rsub__ = _binary_op_method_timedeltalike(lambda x, y: y - x, "__rsub__")

    # 定义 __mul__ 方法，用于实现时间乘法操作
    def __mul__(self, other):
        # 若 other 是整数或浮点数对象
        if is_integer_object(other) or is_float_object(other):
            # 若 other 是 NaN，则返回 NaT
            if util.is_nan(other):
                # np.nan * timedelta -> np.timedelta64("NaT"), 在这种情况下返回 NaT
                return NaT
            
            # 根据 other 值和当前时间增量对象的值创建新的时间增量对象
            return _timedelta_from_value_and_reso(
                Timedelta,
                <int64_t>(other * self._value),  # 计算乘积
                reso=self._creso,  # 使用当前时间增量对象的分辨率
            )

        # 若 other 是数组对象
        elif is_array(other):
            if other.ndim == 0:
                # 参考：item_from_zerodim
                # 将数组中的零维数据转换为标量，并递归调用当前对象的 __mul__ 方法
                item = cnp.PyArray_ToScalar(cnp.PyArray_DATA(other), other)
                return self.__mul__(item)
            # 对数组对象执行时间乘法操作，调用当前对象的 to_timedelta64 方法
            return other * self.to_timedelta64()

        # 其他情况下返回 NotImplemented，表示不支持的操作
        return NotImplemented

    # 定义 __rmul__ 方法，与 __mul__ 方法相同
    __rmul__ = __mul__

    # 定义 __truediv__ 方法，用于实现时间除法操作
    def __truediv__(self, other):
        # 若 other 应转换为时间增量对象
        if _should_cast_to_timedelta(other):
            # 将 other 解释为时间增量对象 Timedelta
            other = Timedelta(other)
            # 若 other 是 NaT，则返回 np.nan
            if other is NaT:
                return np.nan
            # 若 other 的分辨率与当前对象不同，尝试将两者匹配
            if other._creso != self._creso:
                self, other = self._maybe_cast_to_matching_resos(other)
            # 返回当前对象值除以 other 对象值的浮点数结果
            return self._value / float(other._value)

        # 若 other 是整数或浮点数对象
        elif is_integer_object(other) or is_float_object(other):
            # 若 other 是 NaN，则返回 NaT
            if util.is_nan(other):
                return NaT
            # 根据 other 值执行当前对象值除以 other 的操作，并返回结果
            return Timedelta._from_value_and_reso(
                <int64_t>(self._value / other),  # 执行除法操作
                self._creso  # 使用当前对象的分辨率
            )

        # 若 other 是数组对象
        elif is_array(other):
            if other.ndim == 0:
                # 参考：item_from_zerodim
                # 将数组中的零维数据转换为标量，并递归调用当前对象的 __truediv__ 方法
                item = cnp.PyArray_ToScalar(cnp.PyArray_DATA(other), other)
                return self.__truediv__(item)
            # 对数组执行当前对象的 to_timedelta64 方法，再执行除法操作
            return self.to_timedelta64() / other

        # 其他情况下返回 NotImplemented，表示不支持的操作
        return NotImplemented
    def __rtruediv__(self, other):
        # 如果需要将 `other` 转换为 Timedelta 对象，则进行处理
        if _should_cast_to_timedelta(other):
            # 将 NaT 解释为 timedelta64("NaT")
            other = Timedelta(other)
            # 如果 `other` 是 NaT，则返回 np.nan
            if other is NaT:
                return np.nan
            # 如果当前对象的分辨率不同于 `other` 的分辨率，则尝试进行匹配
            if self._creso != other._creso:
                self, other = self._maybe_cast_to_matching_resos(other)
            # 返回 other 的值除以当前对象的值（浮点数）
            return float(other._value) / self._value

        # 如果 `other` 是数组类型
        elif is_array(other):
            # 如果 `other` 是零维数组，使用其标量值来调用当前对象的 __rtruediv__ 方法
            if other.ndim == 0:
                # 参考：item_from_zerodim
                item = cnp.PyArray_ToScalar(cnp.PyArray_DATA(other), other)
                return self.__rtruediv__(item)
            # 如果 `other` 的元素类型是对象类型
            elif other.dtype.kind == "O":
                # GH#31869
                # 对 `other` 中的每个元素调用当前对象的 __rtruediv__ 方法，并返回结果数组
                return np.array([x / self for x in other])

            # TODO: 如果 `other` 的元素类型是日期时间类型（"m"），并且类型不同于当前对象的 asm8 类型
            # 则应当禁止该操作，以保持与标量行为的一致性；这需要一个废弃周期（或更改标量行为）。
            # 返回 `other` 除以当前对象的时间增量值（to_timedelta64 方法返回）
            return other / self.to_timedelta64()

        # 如果以上条件都不满足，则返回 Not Implemented
        return NotImplemented
    # numpy 不为 timedelta64 数据类型实现 floordiv 操作，因此我们无法简单地推迟处理
    if _should_cast_to_timedelta(other):
        # 将 NaT 解释为 timedelta64("NaT")
        other = Timedelta(other)
        # 如果 other 是 NaT，则返回 np.nan
        if other is NaT:
            return np.nan
        # 如果 self 和 other 的分辨率不匹配，则尝试进行类型转换以匹配分辨率
        if self._creso != other._creso:
            self, other = self._maybe_cast_to_matching_resos(other)
        # 执行整数除法操作并返回结果
        return self._value // other._value

    elif is_integer_object(other) or is_float_object(other):
        # 如果 other 是整数或浮点数对象
        if util.is_nan(other):
            return NaT
        # 根据 NEP 50，希望 NumPy 数值标量表现像 Python 标量
        if isinstance(other, cnp.integer):
            other = int(other)
        if isinstance(other, cnp.floating):
            other = float(other)
        # 根据当前类型的值和分辨率创建新的对象并执行整数除法操作
        return type(self)._from_value_and_reso(self._value // other, self._creso)

    elif is_array(other):
        # 如果 other 是数组
        if other.ndim == 0:
            # 对于零维数组，获取其标量值并递归调用 __floordiv__
            item = cnp.PyArray_ToScalar(cnp.PyArray_DATA(other), other)
            return self.__floordiv__(item)

        if other.dtype.kind == "m":
            # 如果 other 的数据类型是 timedelta-like
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    "invalid value encountered in floor_divide",
                    RuntimeWarning
                )
                # 执行时间戳除法操作
                result = self.asm8 // other
            mask = other.view("i8") == NPY_NAT
            if mask.any():
                # 与 numpy 不同，将结果中的 NaN 替换为 np.nan
                result = result.astype("f8")
                result[mask] = np.nan
            return result

        elif other.dtype.kind in "iuf":
            # 如果 other 的数据类型是整数、无符号整数或浮点数
            if other.ndim == 0:
                # 如果 other 是零维数组，执行整数除法并返回结果
                return self // other.item()
            else:
                # 否则，将 self 转换为 timedelta64 类型再执行除法操作
                return self.to_timedelta64() // other

        # 若 other 的数据类型无法处理，则抛出类型错误异常
        raise TypeError(f"Invalid dtype {other.dtype} for __floordiv__")

    # 若无法处理给定的 other 类型，则返回 NotImplemented
    return NotImplemented
    # 定义特殊方法 __rfloordiv__，处理右操作数为 timedelta64 类型时的整除运算
    def __rfloordiv__(self, other):
        # numpy 不支持 timedelta64 类型的 floordiv 操作，因此需要特殊处理
        if _should_cast_to_timedelta(other):
            # 将 NaT 解释为 timedelta64("NaT")
            other = Timedelta(other)
            # 如果 other 是 NaT，则返回 NaN
            if other is NaT:
                return np.nan
            # 如果 self 和 other 的分辨率不同，则尝试将它们转换为匹配的分辨率
            if self._creso != other._creso:
                self, other = self._maybe_cast_to_matching_resos(other)
            # 返回整除后的值
            return other._value // self._value

        elif is_array(other):
            if other.ndim == 0:
                # 对于零维数组，转换为标量处理，并递归调用 __rfloordiv__
                item = cnp.PyArray_ToScalar(cnp.PyArray_DATA(other), other)
                return self.__rfloordiv__(item)

            if other.dtype.kind == "m":
                # 如果 other 是时间增量类型，进行整除操作
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        "invalid value encountered in floor_divide",
                        RuntimeWarning
                    )
                    result = other // self.asm8
                # 处理 NaT 值，将其转换为 NaN
                mask = other.view("i8") == NPY_NAT
                if mask.any():
                    # 与 numpy 不同的是，将结果转换为 float64，并将 mask 对应位置设置为 NaN
                    result = result.astype("f8")
                    result[mask] = np.nan
                return result

            # 不允许整数数组和 Timedelta 进行整除，抛出类型错误
            raise TypeError(f"Invalid dtype {other.dtype} for __floordiv__")

        # 如果其他情况均不适用，则返回 NotImplemented
        return NotImplemented

    # 定义特殊方法 __mod__，实现模运算，暂时采用简单的实现方式
    def __mod__(self, other):
        # 返回模运算的余数
        return self.__divmod__(other)[1]

    # 定义特殊方法 __rmod__，实现右操作数的模运算，暂时采用简单的实现方式
    def __rmod__(self, other):
        # 返回模运算的余数
        return self.__rdivmod__(other)[1]

    # 定义特殊方法 __divmod__，实现除法和取模的组合运算，暂时采用简单的实现方式
    def __divmod__(self, other):
        # 计算除法结果
        div = self // other
        # 计算模运算的余数
        return div, self - div * other

    # 定义特殊方法 __rdivmod__，实现右操作数的除法和取模的组合运算，暂时采用简单的实现方式
    def __rdivmod__(self, other):
        # 计算除法结果
        div = other // self
        # 计算模运算的余数
        return div, other - div * self
# 使用 Cython 定义一个函数，用于计算两个 ndarray 对象的真除运算，结果以对象数组形式返回
def truediv_object_array(ndarray left, ndarray right):
    cdef:
        # 创建一个对象类型的 ndarray，用于存储计算结果，与 left 数组形状相同
        ndarray[object] result = np.empty((<object>left).shape, dtype=object)
        object td64  # 如果能声明 timedelta64 的话真实上是 timedelta64
        object obj, res_value  # 定义对象和结果值变量
        _Timedelta td  # 定义一个 _Timedelta 类型的变量
        Py_ssize_t i  # Python ssize_t 类型的循环索引变量

    for i in range(len(left)):
        td64 = left[i]  # 从 left 数组获取 timedelta64 类型的值
        obj = right[i]   # 从 right 数组获取对应索引的对象值

        if cnp.get_timedelta64_value(td64) == NPY_NAT:
            # 如果 td64 是 NaT，则将其视为 timedelta64 NaT
            if _should_cast_to_timedelta(obj):
                res_value = np.nan  # 如果 obj 应转换为 timedelta，则结果为 NaN
            else:
                # 如果 obj 是数值，则让 numpy 处理除法，否则 numpy 会引发异常
                res_value = td64 / obj
        else:
            td = Timedelta(td64)  # 将 td64 转换为 Timedelta 对象
            res_value = td / obj   # 执行 Timedelta 对象与 obj 的除法运算

        result[i] = res_value  # 将计算结果赋值给 result 数组的当前索引位置

    return result  # 返回存储计算结果的对象数组


# 使用 Cython 定义一个函数，用于计算两个 ndarray 对象的整除运算，结果以对象数组形式返回
def floordiv_object_array(ndarray left, ndarray right):
    cdef:
        # 创建一个对象类型的 ndarray，用于存储计算结果，与 left 数组形状相同
        ndarray[object] result = np.empty((<object>left).shape, dtype=object)
        object td64  # 如果能声明 timedelta64 的话真实上是 timedelta64
        object obj, res_value  # 定义对象和结果值变量
        _Timedelta td  # 定义一个 _Timedelta 类型的变量
        Py_ssize_t i  # Python ssize_t 类型的循环索引变量

    for i in range(len(left)):
        td64 = left[i]  # 从 left 数组获取 timedelta64 类型的值
        obj = right[i]   # 从 right 数组获取对应索引的对象值

        if cnp.get_timedelta64_value(td64) == NPY_NAT:
            # 如果 td64 是 NaT，则将其视为 timedelta64 NaT
            if _should_cast_to_timedelta(obj):
                res_value = np.nan  # 如果 obj 应转换为 timedelta，则结果为 NaN
            else:
                # 如果 obj 是数值，则让 numpy 处理整除，否则 numpy 会引发异常
                res_value = td64 // obj
        else:
            td = Timedelta(td64)  # 将 td64 转换为 Timedelta 对象
            res_value = td // obj   # 执行 Timedelta 对象与 obj 的整除运算

        result[i] = res_value  # 将计算结果赋值给 result 数组的当前索引位置

    return result  # 返回存储计算结果的对象数组


# 使用 Cython 定义一个函数，用于检查对象是否为任何时间增量类型的标量
def is_any_td_scalar(object obj):
    """
    Cython 中相当于 `isinstance(obj, (timedelta, np.timedelta64, Tick))` 的函数

    Parameters
    ----------
    obj : object

    Returns
    -------
    bool
    """
    return (
        PyDelta_Check(obj) or cnp.is_timedelta64_object(obj) or is_tick_object(obj)
    )


# 使用 Cython 定义一个函数，判断是否应将对象视为 Timedelta 以便进行二进制运算
def _should_cast_to_timedelta(object obj):
    """
    是否应将此对象视为 Timedelta 用于二进制操作的函数
    """
    return (
        is_any_td_scalar(obj) or obj is None or obj is NaT or isinstance(obj, str)
    )


# 使用 Cython 定义一个函数，获取用于舍入的单位
cpdef int64_t get_unit_for_round(freq, NPY_DATETIMEUNIT creso) except? -1:
    from pandas._libs.tslibs.offsets import to_offset

    freq = to_offset(freq)
    freq.nanos  # 对非固定频率进行检查时会引发异常
    return delta_to_nanoseconds(freq, creso)
```