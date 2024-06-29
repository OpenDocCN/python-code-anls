# `D:\src\scipysrc\pandas\pandas\tests\scalar\timedelta\test_constructors.py`

```
# 从 datetime 模块中导入 timedelta 类
# 从 itertools 模块中导入 product 函数
from datetime import timedelta
from itertools import product

# 导入 numpy 库，并用 np 别名引用
import numpy as np

# 导入 pytest 测试框架
import pytest

# 从 pandas._libs.tslibs 中导入 OutOfBoundsTimedelta 类
from pandas._libs.tslibs import OutOfBoundsTimedelta
# 从 pandas._libs.tslibs.dtypes 中导入 NpyDatetimeUnit 类
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit

# 从 pandas 库中导入多个类和函数
from pandas import (
    Index,          # 导入 Index 类
    NaT,            # 导入 NaT 对象，表示缺失的时间戳值
    Timedelta,      # 导入 Timedelta 类
    TimedeltaIndex, # 导入 TimedeltaIndex 类
    offsets,        # 导入 offsets 模块
    to_timedelta,   # 导入 to_timedelta 函数
)

# 导入 pandas._testing 库，并用 tm 别名引用
import pandas._testing as tm


# 定义测试类 TestTimedeltaConstructorUnitKeyword
class TestTimedeltaConstructorUnitKeyword:
    # 使用 pytest.mark.parametrize 装饰器指定参数化测试，unit 参数取值为 ["Y", "y", "M"]
    @pytest.mark.parametrize("unit", ["Y", "y", "M"])
    # 定义测试方法 test_unit_m_y_raises，接受 unit 参数
    def test_unit_m_y_raises(self, unit):
        # 设置错误消息
        msg = "Units 'M', 'Y', and 'y' are no longer supported"

        # 使用 pytest.raises 断言捕获 ValueError 异常，并验证异常消息匹配 msg
        with pytest.raises(ValueError, match=msg):
            Timedelta(10, unit)  # 创建 Timedelta 对象时使用单位参数 unit

        with pytest.raises(ValueError, match=msg):
            to_timedelta(10, unit)  # 使用 to_timedelta 函数时使用单位参数 unit

        with pytest.raises(ValueError, match=msg):
            to_timedelta([1, 2], unit)  # 使用 to_timedelta 函数时使用单位参数 unit 的列表形式

    # 使用 pytest.mark.parametrize 装饰器指定参数化测试，unit 和 unit_depr 参数从元组列表中取值
    @pytest.mark.parametrize(
        "unit,unit_depr",
        [
            ("W", "w"),     # 单位 "W" 对应的过期单位为 "w"
            ("D", "d"),     # 单位 "D" 对应的过期单位为 "d"
            ("min", "MIN"), # 单位 "min" 对应的过期单位为 "MIN"
            ("s", "S"),     # 单位 "s" 对应的过期单位为 "S"
            ("h", "H"),     # 单位 "h" 对应的过期单位为 "H"
            ("ms", "MS"),   # 单位 "ms" 对应的过期单位为 "MS"
            ("us", "US"),   # 单位 "us" 对应的过期单位为 "US"
        ],
    )
    # 定义测试方法 test_unit_deprecated，接受 unit 和 unit_depr 参数
    def test_unit_deprecated(self, unit, unit_depr):
        # 设置过期警告消息
        msg = f"'{unit_depr}' is deprecated and will be removed in a future version."

        # 创建预期的 Timedelta 对象，指定单位为 unit
        expected = Timedelta(1, unit=unit)

        # 使用 tm.assert_produces_warning 断言产生 FutureWarning 警告，并验证警告消息匹配 msg
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # 创建 Timedelta 对象时使用过期单位 unit_depr
            result = Timedelta(1, unit=unit_depr)

        # 使用 tm.assert_equal 函数断言 result 和 expected 相等
        tm.assert_equal(result, expected)
    @pytest.mark.parametrize(  # 使用 pytest 模块的 parametrize 装饰器，允许多组参数化测试数据
        "unit, np_unit",  # 定义参数的名称
        [("W", "W")]  # 包括一个基础测试参数组
        + [(value, "D") for value in ["D", "days", "day", "Days", "Day"]]  # 创建一组时间单位为天的测试参数
        + [  # 创建一组时间单位为分钟的测试参数
            (value, "m")
            for value in [
                "m",
                "minute",
                "min",
                "minutes",
                "Minute",
                "Min",
                "Minutes",
            ]
        ]
        + [  # 创建一组时间单位为秒的测试参数
            (value, "s")
            for value in [
                "s",
                "seconds",
                "sec",
                "second",
                "Seconds",
                "Sec",
                "Second",
            ]
        ]
        + [  # 创建一组时间单位为毫秒的测试参数
            (value, "ms")
            for value in [
                "ms",
                "milliseconds",
                "millisecond",
                "milli",
                "millis",
                "Milliseconds",
                "Millisecond",
                "Milli",
                "Millis",
            ]
        ]
        + [  # 创建一组时间单位为微秒的测试参数
            (value, "us")
            for value in [
                "us",
                "microseconds",
                "microsecond",
                "micro",
                "micros",
                "Microseconds",
                "Microsecond",
                "Micro",
                "Micros",
            ]
        ]
        + [  # 创建一组时间单位为纳秒的测试参数
            (value, "ns")
            for value in [
                "ns",
                "nanoseconds",
                "nanosecond",
                "nano",
                "nanos",
                "Nanoseconds",
                "Nanosecond",
                "Nano",
                "Nanos",
            ]
        ],
    )
    @pytest.mark.parametrize("wrapper", [np.array, list, Index])  # 参数化测试的第二个参数，指定包装器函数

    def test_unit_parser(self, unit, np_unit, wrapper):  # 定义测试方法，接收参数 unit, np_unit 和 wrapper
        # validate all units, GH 6855, GH 21762  # 验证所有时间单位的正确性，参考 GitHub issue 6855 和 21762
        # array-likes  # 数组样式的测试数据

        expected = TimedeltaIndex(  # 创建预期结果，使用 TimedeltaIndex 类
            [np.timedelta64(i, np_unit) for i in np.arange(5).tolist()],  # 生成一个包含五个时间差的 TimedeltaIndex 对象
            dtype="m8[ns]",  # 指定数据类型为 'm8[ns]'
        )

        # TODO(2.0): the desired output dtype may have non-nano resolution
        # TODO(2.0): 期望的输出数据类型可能具有非纳秒分辨率

        result = to_timedelta(wrapper(range(5)), unit=unit)  # 调用 to_timedelta 函数生成结果
        tm.assert_index_equal(result, expected)  # 使用 tm.assert_index_equal 检查结果是否与预期相同

        str_repr = [f"{x}{unit}" for x in np.arange(5)]  # 生成字符串表示形式的列表
        result = to_timedelta(wrapper(str_repr))  # 使用字符串表示形式调用 to_timedelta 函数生成结果
        tm.assert_index_equal(result, expected)  # 使用 tm.assert_index_equal 检查结果是否与预期相同

        result = to_timedelta(wrapper(str_repr))  # 再次使用字符串表示形式调用 to_timedelta 函数生成结果
        tm.assert_index_equal(result, expected)  # 使用 tm.assert_index_equal 检查结果是否与预期相同

        # scalar  # 标量的测试数据

        expected = Timedelta(np.timedelta64(2, np_unit).astype("timedelta64[ns]"))  # 创建标量的预期结果
        result = to_timedelta(2, unit=unit)  # 调用 to_timedelta 函数生成结果
        assert result == expected  # 断言结果与预期相同

        result = Timedelta(2, unit=unit)  # 创建 Timedelta 对象
        assert result == expected  # 断言结果与预期相同

        result = to_timedelta(f"2{unit}")  # 使用字符串表示形式调用 to_timedelta 函数生成结果
        assert result == expected  # 断言结果与预期相同

        result = Timedelta(f"2{unit}")  # 创建 Timedelta 对象
        assert result == expected  # 断言结果与预期相同
    # 定义测试方法，验证当单位参数异常时抛出 ValueError 异常
    def test_unit_T_L_N_U_raises(self, unit):
        # 生成异常消息，指明单位参数异常
        msg = f"invalid unit abbreviation: {unit}"
        
        # 使用 pytest 断言捕获 ValueError 异常，验证 Timedelta 对象创建时的异常情况
        with pytest.raises(ValueError, match=msg):
            Timedelta(1, unit=unit)

        # 使用 pytest 断言捕获 ValueError 异常，验证 to_timedelta 函数在单位参数异常时的异常情况
        with pytest.raises(ValueError, match=msg):
            to_timedelta(10, unit)

        # 使用 pytest 断言捕获 ValueError 异常，验证 to_timedelta 函数在单位参数异常时的异常情况（对列表参数的处理）
        with pytest.raises(ValueError, match=msg):
            to_timedelta([1, 2], unit)
def test_construct_from_kwargs_overflow():
    # 测试超出范围的时间间隔构造函数
    # GH#55503
    msg = "seconds=86400000000000000000, milliseconds=0, microseconds=0, nanoseconds=0"
    # 使用 pytest 的断言来检测是否会抛出 OutOfBoundsTimedelta 异常，并匹配特定消息
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        Timedelta(days=10**6)
    
    msg = "seconds=60000000000000000000, milliseconds=0, microseconds=0, nanoseconds=0"
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        Timedelta(minutes=10**9)


def test_construct_with_weeks_unit_overflow():
    # 测试以周为单位构造时间间隔是否会溢出
    # GH#47268 don't silently wrap around
    with pytest.raises(OutOfBoundsTimedelta, match="without overflow"):
        Timedelta(1000000000000000000, unit="W")

    with pytest.raises(OutOfBoundsTimedelta, match="without overflow"):
        Timedelta(1000000000000000000.0, unit="W")


def test_construct_from_td64_with_unit():
    # 测试从 np.timedelta64 对象构造 Timedelta 对象时的行为
    # 忽略单位，因为它可能导致静默溢出，导致不正确的结果，而在非溢出情况下，单位是无关紧要的 GH#46827
    obj = np.timedelta64(123456789000000000, "h")

    with pytest.raises(OutOfBoundsTimedelta, match="123456789000000000 hours"):
        Timedelta(obj, unit="ps")

    with pytest.raises(OutOfBoundsTimedelta, match="123456789000000000 hours"):
        Timedelta(obj, unit="ns")

    with pytest.raises(OutOfBoundsTimedelta, match="123456789000000000 hours"):
        Timedelta(obj)


def test_from_td64_retain_resolution():
    # 测试从 np.timedelta64 对象构造 Timedelta 对象时是否保留分辨率
    # 保留毫秒级分辨率的情况
    obj = np.timedelta64(12345, "ms")

    td = Timedelta(obj)
    assert td._value == obj.view("i8")
    assert td._creso == NpyDatetimeUnit.NPY_FR_ms.value

    # 转换为最接近支持的分辨率的情况
    obj2 = np.timedelta64(1234, "D")
    td2 = Timedelta(obj2)
    assert td2._creso == NpyDatetimeUnit.NPY_FR_s.value
    assert td2 == obj2
    assert td2.days == 1234

    # 如果不支持非纳秒分辨率则会溢出的情况
    obj3 = np.timedelta64(1000000000000000000, "us")
    td3 = Timedelta(obj3)
    assert td3.total_seconds() == 1000000000000
    assert td3._creso == NpyDatetimeUnit.NPY_FR_us.value


def test_from_pytimedelta_us_reso():
    # 测试从 Python datetime.timedelta 对象构造 Timedelta 对象时的行为
    # pytimedelta 具有微秒级分辨率，因此 Timedelta(pytd) 继承这一特性
    td = timedelta(days=4, minutes=3)
    result = Timedelta(td)
    assert result.to_pytimedelta() == td
    assert result._creso == NpyDatetimeUnit.NPY_FR_us.value


def test_from_tick_reso():
    # 测试从 pandas 的时间偏移对象构造 Timedelta 对象时的分辨率
    tick = offsets.Nano()
    assert Timedelta(tick)._creso == NpyDatetimeUnit.NPY_FR_ns.value

    tick = offsets.Micro()
    assert Timedelta(tick)._creso == NpyDatetimeUnit.NPY_FR_us.value

    tick = offsets.Milli()
    assert Timedelta(tick)._creso == NpyDatetimeUnit.NPY_FR_ms.value

    tick = offsets.Second()
    assert Timedelta(tick)._creso == NpyDatetimeUnit.NPY_FR_s.value

    # 超过秒钟的偏移会被转换为最接近的支持分辨率：秒
    tick = offsets.Minute()
    assert Timedelta(tick)._creso == NpyDatetimeUnit.NPY_FR_s.value

    tick = offsets.Hour()
    # 确保给定的时间增量 tick 对象的 _creso 属性等于 NpyDatetimeUnit.NPY_FR_s.value
    assert Timedelta(tick)._creso == NpyDatetimeUnit.NPY_FR_s.value
    
    # 创建一个表示一天时间跨度的 offsets.Day() 对象，并确保其对应的时间增量 tick 对象的 _creso 属性也等于 NpyDatetimeUnit.NPY_FR_s.value
    tick = offsets.Day()
    assert Timedelta(tick)._creso == NpyDatetimeUnit.NPY_FR_s.value
def test_construction():
    # 创建一个期待的时间增量，用于后续断言比较
    expected = np.timedelta64(10, "D").astype("m8[ns]").view("i8")
    # 断言不同方式创建的 Timedelta 对象的值是否等于预期值
    assert Timedelta(10, unit="D")._value == expected
    assert Timedelta(10.0, unit="D")._value == expected
    assert Timedelta("10 days")._value == expected
    assert Timedelta(days=10)._value == expected
    assert Timedelta(days=10.0)._value == expected

    # 更新期待的时间增量，增加10秒
    expected += np.timedelta64(10, "s").astype("m8[ns]").view("i8")
    # 断言不同方式创建的 Timedelta 对象的值是否等于更新后的期待值
    assert Timedelta("10 days 00:00:10")._value == expected
    assert Timedelta(days=10, seconds=10)._value == expected
    assert Timedelta(days=10, milliseconds=10 * 1000)._value == expected
    assert Timedelta(days=10, microseconds=10 * 1000 * 1000)._value == expected

    # 精度四舍五入的情况
    assert Timedelta(82739999850000)._value == 82739999850000
    assert "0 days 22:58:59.999850" in str(Timedelta(82739999850000))
    assert Timedelta(123072001000000)._value == 123072001000000
    assert "1 days 10:11:12.001" in str(Timedelta(123072001000000))

    # 字符串转换测试，包括带或不带前导零的情况
    # GH#9570
    assert Timedelta("0:00:00") == timedelta(hours=0)
    assert Timedelta("00:00:00") == timedelta(hours=0)
    assert Timedelta("-1:00:00") == -timedelta(hours=1)
    assert Timedelta("-01:00:00") == -timedelta(hours=1)

    # 更多字符串和缩写测试
    # GH#8190
    assert Timedelta("1 h") == timedelta(hours=1)
    assert Timedelta("1 hour") == timedelta(hours=1)
    assert Timedelta("1 hr") == timedelta(hours=1)
    assert Timedelta("1 hours") == timedelta(hours=1)
    assert Timedelta("-1 hours") == -timedelta(hours=1)
    assert Timedelta("1 m") == timedelta(minutes=1)
    assert Timedelta("1.5 m") == timedelta(seconds=90)
    assert Timedelta("1 minute") == timedelta(minutes=1)
    assert Timedelta("1 minutes") == timedelta(minutes=1)
    assert Timedelta("1 s") == timedelta(seconds=1)
    assert Timedelta("1 second") == timedelta(seconds=1)
    assert Timedelta("1 seconds") == timedelta(seconds=1)
    assert Timedelta("1 ms") == timedelta(milliseconds=1)
    assert Timedelta("1 milli") == timedelta(milliseconds=1)
    assert Timedelta("1 millisecond") == timedelta(milliseconds=1)
    assert Timedelta("1 us") == timedelta(microseconds=1)
    assert Timedelta("1 µs") == timedelta(microseconds=1)
    assert Timedelta("1 micros") == timedelta(microseconds=1)
    assert Timedelta("1 microsecond") == timedelta(microseconds=1)
    assert Timedelta("1.5 microsecond") == Timedelta("00:00:00.000001500")
    assert Timedelta("1 ns") == Timedelta("00:00:00.000000001")
    assert Timedelta("1 nano") == Timedelta("00:00:00.000000001")
    assert Timedelta("1 nanosecond") == Timedelta("00:00:00.000000001")

    # 组合测试
    assert Timedelta("10 days 1 hour") == timedelta(days=10, hours=1)
    assert Timedelta("10 days 1 h") == timedelta(days=10, hours=1)
    assert Timedelta("10 days 1 h 1m 1s") == timedelta(
        days=10, hours=1, minutes=1, seconds=1
    )
    # 使用 Timedelta 类的构造函数创建一个 Timedelta 对象，验证其等于负数的 timedelta 对象
    assert Timedelta("-10 days 1 h 1m 1s") == -timedelta(
        days=10, hours=1, minutes=1, seconds=1
    )
    # 同上，验证带有微秒的情况
    assert Timedelta("-10 days 1 h 1m 1s 3us") == -timedelta(
        days=10, hours=1, minutes=1, seconds=1, microseconds=3
    )
    # 验证分钟带小数的情况
    assert Timedelta("-10 days 1 h 1.5m 1s 3us") == -timedelta(
        days=10, hours=1, minutes=1, seconds=31, microseconds=3
    )

    # 测试输入 "-10 days -1 h 1.5m 1s 3us"，应该引发 ValueError 异常，只允许负号出现在天数的部分
    msg = "only leading negative signs are allowed"
    with pytest.raises(ValueError, match=msg):
        Timedelta("-10 days -1 h 1.5m 1s 3us")

    # 测试输入 "10 days -1 h 1.5m 1s 3us"，应该引发 ValueError 异常，只允许负号出现在天数的部分
    with pytest.raises(ValueError, match=msg):
        Timedelta("10 days -1 h 1.5m 1s 3us")

    # 测试输入 "3.1415"，应该引发 ValueError 异常，因为没有指定时间单位
    msg = "no units specified"
    with pytest.raises(ValueError, match=msg):
        Timedelta("3.1415")

    # 测试输入空参数构造 Timedelta 对象，应该引发 ValueError 异常
    msg = "cannot construct a Timedelta"
    with pytest.raises(ValueError, match=msg):
        Timedelta()

    # 测试输入 "foo"，应该引发 ValueError 异常，因为时间单位不合法
    msg = "unit abbreviation w/o a number"
    with pytest.raises(ValueError, match=msg):
        Timedelta("foo")

    # 测试输入 day=10 的参数构造 Timedelta 对象，应该引发 ValueError 异常，因为关键字参数不合法
    msg = (
        "cannot construct a Timedelta from "
        "the passed arguments, allowed keywords are "
    )
    with pytest.raises(ValueError, match=msg):
        Timedelta(day=10)

    # 测试使用浮点数作为参数构造 Timedelta 对象，验证结果与 numpy 运算相符
    expected = np.timedelta64(10, "s").astype("m8[ns]").view("i8") + np.timedelta64(
        500, "ms"
    ).astype("m8[ns]").view("i8")
    assert Timedelta(10.5, unit="s")._value == expected

    # 测试使用 offsets.Hour(2) 构造 Timedelta 对象，验证结果与 to_timedelta 函数的返回相等
    assert to_timedelta(offsets.Hour(2)) == Timedelta(hours=2)
    assert Timedelta(offsets.Hour(2)) == Timedelta(hours=2)

    # 测试使用 offsets.Second(2) 构造 Timedelta 对象，验证结果与预期相等
    assert Timedelta(offsets.Second(2)) == Timedelta(seconds=2)

    # GH#11995: 测试用例，验证创建 Timedelta 对象与预期相等
    expected = Timedelta("1h")
    result = Timedelta("1h")
    assert result == expected
    assert to_timedelta(offsets.Hour(2)) == Timedelta("0 days, 02:00:00")

    # 测试输入 "foo bar"，应该引发 ValueError 异常，因为时间单位不合法
    msg = "unit abbreviation w/o a number"
    with pytest.raises(ValueError, match=msg):
        Timedelta("foo bar")
@pytest.mark.parametrize(
    "item",
    list(
        {
            "days": "D",
            "seconds": "s",
            "microseconds": "us",
            "milliseconds": "ms",
            "minutes": "m",
            "hours": "h",
            "weeks": "W",
        }.items()
    ),
)
@pytest.mark.parametrize(
    "npdtype", [np.int64, np.int32, np.int16, np.float64, np.float32, np.float16]
)
def test_td_construction_with_np_dtypes(npdtype, item):
    # GH#8757: test construction with np dtypes
    # 为了测试使用 numpy 数据类型构建 Timedelta 对象
    pykwarg, npkwarg = item
    # 根据参数 item 解构出 Python 和 numpy 的关键字参数
    expected = np.timedelta64(1, npkwarg).astype("m8[ns]").view("i8")
    # 创建一个预期的 np.timedelta64 对象，并转换为 "m8[ns]" 类型的视图
    assert Timedelta(**{pykwarg: npdtype(1)})._value == expected
    # 断言 Timedelta 对象的内部值与预期的值相等


@pytest.mark.parametrize(
    "val",
    [
        "1s",
        "-1s",
        "1us",
        "-1us",
        "1 day",
        "-1 day",
        "-23:59:59.999999",
        "-1 days +23:59:59.999999",
        "-1ns",
        "1ns",
        "-23:59:59.999999999",
    ],
)
def test_td_from_repr_roundtrip(val):
    # round-trip both for string and value
    # 测试字符串和值之间的双向转换
    td = Timedelta(val)
    # 使用给定的字符串创建 Timedelta 对象
    assert Timedelta(td._value) == td
    # 断言 Timedelta 对象的内部值与原始对象相等

    assert Timedelta(str(td)) == td
    # 断言 Timedelta 对象的字符串表示与原始对象相等
    assert Timedelta(td._repr_base(format="all")) == td
    # 断言 Timedelta 对象的基础表示与原始对象相等
    assert Timedelta(td._repr_base()) == td
    # 断言 Timedelta 对象的基础表示与原始对象相等


def test_overflow_on_construction():
    # GH#3374
    # 测试 Timedelta 构造函数的溢出情况
    value = Timedelta("1day")._value * 20169940
    # 计算 Timedelta 对象的内部值乘以一个大数
    msg = "Cannot cast 1742682816000000000000 from ns to 'ns' without overflow"
    # 定义预期的错误消息
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        # 捕获 OutOfBoundsTimedelta 异常并验证错误消息
        Timedelta(value)

    # xref GH#17637
    # 引用 GH#17637
    msg = "Cannot cast 139993 from D to 'ns' without overflow"
    # 定义另一个预期的错误消息
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        # 捕获 OutOfBoundsTimedelta 异常并验证错误消息
        Timedelta(7 * 19999, unit="D")

    # used to overflow before non-ns support
    # 在不支持 ns 的情况下，以前会溢出
    td = Timedelta(timedelta(days=13 * 19999))
    # 使用 datetime.timedelta 创建 Timedelta 对象
    assert td._creso == NpyDatetimeUnit.NPY_FR_us.value
    # 断言 Timedelta 对象的 _creso 属性与预期的值相等
    assert td.days == 13 * 19999
    # 断言 Timedelta 对象的 days 属性与预期的值相等


@pytest.mark.parametrize(
    "val, unit",
    [
        (15251, "W"),  # 1
        (106752, "D"),  # change from previous:
        (2562048, "h"),  # 0 hours
        (153722868, "m"),  # 13 minutes
        (9223372037, "s"),  # 44 seconds
    ],
)
def test_construction_out_of_bounds_td64ns(val, unit):
    # TODO: parametrize over units just above/below the implementation bounds
    #  once GH#38964 is resolved
    # 在 GH#38964 解决之后，通过不同单位进行参数化测试

    # Timedelta.max is just under 106752 days
    # Timedelta.max 稍低于 106752 天
    td64 = np.timedelta64(val, unit)
    # 使用给定的值和单位创建 np.timedelta64 对象
    assert td64.astype("m8[ns]").view("i8") < 0  # i.e. naive astype will be wrong
    # 断言转换为 "m8[ns]" 类型后的 np.timedelta64 对象视图小于 0，即原始的转换方式不正确

    td = Timedelta(td64)
    # 使用 np.timedelta64 创建 Timedelta 对象
    if unit != "M":
        # with unit="M" the conversion to "s" is poorly defined
        #  (and numpy issues DeprecationWarning)
        # 当单位为 "M" 时，转换为 "s" 是不明确的，并且 numpy 会发出 DeprecationWarning
        assert td.asm8 == td64
        # 断言 Timedelta 对象的 asm8 属性与原始 np.timedelta64 对象相等
    assert td.asm8.dtype == "m8[s]"
    # 断言 Timedelta 对象的 asm8 属性的数据类型为 "m8[s]"
    msg = r"Cannot cast 1067\d\d days .* to unit='ns' without overflow"
    # 定义预期的错误消息模式
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        # 捕获 OutOfBoundsTimedelta 异常并验证错误消息
        td.as_unit("ns")

    # But just back in bounds and we are OK
    # 但是如果在允许范围内，则正常运行
    assert Timedelta(td64 - 1) == td64 - 1
    # 断言 Timedelta 对象减去 1 后与预期值相等

    td64 *= -1
    # 将 np.timedelta64 对象乘以 -1
    assert td64.astype("m8[ns]").view("i8") > 0  # i.e. naive astype will be wrong
    # 断言转换为 "m8[ns]" 类型后的 np.timedelta64 对象视图大于 0，即原始的转换方式不正确
    # 使用 Timedelta 类创建一个时间间隔对象 td2，传入 td64 变量作为参数
    td2 = Timedelta(td64)
    # 设置错误消息的正则表达式模式，用于匹配特定的错误信息
    msg = r"Cannot cast -1067\d\d days .* to unit='ns' without overflow"
    # 使用 pytest.raises 上下文管理器检测是否抛出 OutOfBoundsTimedelta 异常，并匹配特定的错误消息
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        # 调用 td2 对象的 as_unit 方法，尝试将时间单位转换为纳秒 ('ns')
        td2.as_unit("ns")

    # 断言语句：验证 Timedelta 对象增加 1 后是否等于 td64 增加 1 后的值
    # 如果条件为真，则表示程序在边界内操作正常
    assert Timedelta(td64 + 1) == td64 + 1
@pytest.mark.parametrize(
    "val, unit",
    [  # 参数化测试，提供不同的值和单位组合
        (15251 * 10**9, "W"),    # 测试超出范围的时间增量，单位为周
        (106752 * 10**9, "D"),   # 测试超出范围的时间增量，单位为天
        (2562048 * 10**9, "h"),  # 测试超出范围的时间增量，单位为小时
        (153722868 * 10**9, "m"),# 测试超出范围的时间增量，单位为分钟
    ],
)
def test_construction_out_of_bounds_td64s(val, unit):
    td64 = np.timedelta64(val, unit)  # 创建 numpy.timedelta64 对象
    with pytest.raises(OutOfBoundsTimedelta, match=str(td64)):
        Timedelta(td64)  # 期望此处抛出 OutOfBoundsTimedelta 异常

    # 但是如果回到有效范围内就应该是OK的
    assert Timedelta(td64 - 10**9) == td64 - 10**9


@pytest.mark.parametrize(
    "fmt,exp",
    [  # 参数化测试，提供不同的 ISO 8601 时间间隔字符串和预期 Timedelta 对象
        (
            "P6DT0H50M3.010010012S",
            Timedelta(
                days=6,
                minutes=50,
                seconds=3,
                milliseconds=10,
                microseconds=10,
                nanoseconds=12,
            ),
        ),
        (
            "P-6DT0H50M3.010010012S",
            Timedelta(
                days=-6,
                minutes=50,
                seconds=3,
                milliseconds=10,
                microseconds=10,
                nanoseconds=12,
            ),
        ),
        ("P4DT12H30M5S", Timedelta(days=4, hours=12, minutes=30, seconds=5)),
        ("P0DT0H0M0.000000123S", Timedelta(nanoseconds=123)),
        ("P0DT0H0M0.00001S", Timedelta(microseconds=10)),
        ("P0DT0H0M0.001S", Timedelta(milliseconds=1)),
        ("P0DT0H1M0S", Timedelta(minutes=1)),
        ("P1DT25H61M61S", Timedelta(days=1, hours=25, minutes=61, seconds=61)),
        ("PT1S", Timedelta(seconds=1)),
        ("PT0S", Timedelta(seconds=0)),
        ("P1WT0S", Timedelta(days=7, seconds=0)),
        ("P1D", Timedelta(days=1)),
        ("P1DT1H", Timedelta(days=1, hours=1)),
        ("P1W", Timedelta(days=7)),
        ("PT300S", Timedelta(seconds=300)),
        ("P1DT0H0M00000000000S", Timedelta(days=1)),
        ("PT-6H3M", Timedelta(hours=-6, minutes=3)),
        ("-PT6H3M", Timedelta(hours=-6, minutes=-3)),
        ("-PT-6H+3M", Timedelta(hours=6, minutes=-3)),
    ],
)
def test_iso_constructor(fmt, exp):
    assert Timedelta(fmt) == exp  # 测试通过 ISO 8601 字符串构造 Timedelta 对象


@pytest.mark.parametrize(
    "fmt",
    [  # 参数化测试，提供不同的无效 ISO 8601 时间间隔字符串
        "PPPPPPPPPPPP",
        "PDTHMS",
        "P0DT999H999M999S",
        "P1DT0H0M0.0000000000000S",
        "P1DT0H0M0.S",
        "P",
        "-P",
    ],
)
def test_iso_constructor_raises(fmt):
    msg = f"Invalid ISO 8601 Duration format - {fmt}"  # 预期的错误消息
    with pytest.raises(ValueError, match=msg):
        Timedelta(fmt)  # 期望此处抛出 ValueError 异常


@pytest.mark.parametrize(
    "constructed_td, conversion",
    [
        # 创建一个元组，包含 Timedelta 对象和对应的描述字符串 "100ns"
        (Timedelta(nanoseconds=100), "100ns"),
        (
            # 创建一个 Timedelta 对象，表示一段时间跨度，包括天、小时、分钟、周、秒、毫秒、微秒和纳秒
            Timedelta(
                days=1,
                hours=1,
                minutes=1,
                weeks=1,
                seconds=1,
                milliseconds=1,
                microseconds=1,
                nanoseconds=1,
            ),
            # 对应的描述字符串 "694861001001001"
            694861001001001,
        ),
        # 创建一个元组，包含 Timedelta 对象表示微秒和纳秒相加的结果，以及对应的描述字符串 "1us1ns"
        (Timedelta(microseconds=1) + Timedelta(nanoseconds=1), "1us1ns"),
        # 创建一个元组，包含 Timedelta 对象表示微秒和纳秒相减的结果，以及对应的描述字符串 "999ns"
        (Timedelta(microseconds=1) - Timedelta(nanoseconds=1), "999ns"),
        # 创建一个元组，包含 Timedelta 对象表示微秒加上五倍纳秒的负值的结果，以及对应的描述字符串 "990ns"
        (Timedelta(microseconds=1) + 5 * Timedelta(nanoseconds=-2), "990ns"),
    ],
# GH#9273
assert constructed_td == Timedelta(conversion)

# 验证构造函数是否正确处理纳秒时间差值
def test_td_constructor_on_nanoseconds(constructed_td, conversion):
    assert constructed_td == Timedelta(conversion)


# 使用 pytest 来测试 Timedelta 构造函数是否能正确处理类型错误
def test_td_constructor_value_error():
    msg = "Invalid type <class 'str'>. Must be int or float."
    with pytest.raises(TypeError, match=msg):
        Timedelta(nanoseconds="abc")


# 测试 Timedelta 构造函数是否能保持身份不变
# 对于 GitHub 上的问题 #30543 进行验证
def test_timedelta_constructor_identity():
    expected = Timedelta(np.timedelta64(1, "s"))
    result = Timedelta(expected)
    assert result is expected


# 测试当同时传递 Timedelta 输入和 timedelta 关键字参数时是否能正确引发异常
# 对于 GitHub 上的问题 GH#48898 进行验证
def test_timedelta_pass_td_and_kwargs_raises():
    td = Timedelta(days=1)
    msg = (
        "Cannot pass both a Timedelta input and timedelta keyword arguments, "
        r"got \['days'\]"
    )
    with pytest.raises(ValueError, match=msg):
        Timedelta(td, days=2)


# 使用 pytest 的参数化测试来验证处理带单位的字符串时是否正确引发异常
def test_string_with_unit(constructor, value, unit):
    with pytest.raises(ValueError, match="unit must not be specified"):
        constructor(value, unit=unit)


# 使用 pytest 的参数化测试来验证处理不带数字的字符串时是否正确引发异常
# 对于 GitHub 上的问题 GH39710 进行验证
def test_string_without_numbers(value):
    msg = (
        "symbols w/o a number"
        if value != "--"
        else "only leading negative signs are allowed"
    )
    with pytest.raises(ValueError, match=msg):
        Timedelta(value)


# 测试处理 np.timedelta64("NaT", "h") 是否返回 NaT
# 对于 GitHub 上的问题 GH#48898 进行验证
def test_timedelta_new_npnat():
    nat = np.timedelta64("NaT", "h")
    assert Timedelta(nat) is NaT


# 测试 Timedelta 的子类是否被正确识别
# 对于 GitHub 上的问题 GH#49579 进行验证
def test_subclass_respected():
    class MyCustomTimedelta(Timedelta):
        pass

    td = MyCustomTimedelta("1 minute")
    assert isinstance(td, MyCustomTimedelta)


# 测试 Timedelta 处理非纳秒单位时的值
# 对于 GitHub 上的问题 https://github.com/pandas-dev/pandas/issues/49076 进行验证
def test_non_nano_value():
    result = Timedelta(10, unit="D").as_unit("s").value
    assert result == 864000000000000

    # 测试超出纳秒边界时是否正确引发 OverflowError
    msg = (
        r"Cannot convert Timedelta to nanoseconds without overflow. "
        r"Use `.asm8.view\('i8'\)` to cast represent Timedelta in its "
        r"own unit \(here, s\).$"
    )
    td = Timedelta(1_000, "D").as_unit("s") * 1_000
    with pytest.raises(OverflowError, match=msg):
        td.value

    # 验证建议的解决方法是否有效
    result = td.asm8.view("i8")
    assert result == 86400000000
```