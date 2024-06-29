# `D:\src\scipysrc\pandas\pandas\tests\scalar\timedelta\test_timedelta.py`

```
"""test the scalar Timedelta"""

# 导入 timedelta 类用于时间间隔计算
from datetime import timedelta
# 导入 sys 模块，通常用于访问与 Python 解释器相关的变量和函数
import sys

# 导入 Hypothesis 库中的 given 和 strategies 模块
from hypothesis import (
    given,
    strategies as st,
)
# 导入 NumPy 库
import numpy as np
# 导入 pytest 库，用于编写和运行测试用例
import pytest

# 导入 pandas 库中的 C 扩展库
from pandas._libs import lib
# 导入 pandas 库中处理缺失值的 NA 对象
from pandas._libs.missing import NA
# 导入 pandas 库中时间序列相关的 NaT 和 iNaT 对象
from pandas._libs.tslibs import (
    NaT,
    iNaT,
)
# 导入 pandas 库中 tslibs 模块中的 NpyDatetimeUnit 类型
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
# 导入 pandas 库中的 OutOfBoundsTimedelta 异常类
from pandas.errors import OutOfBoundsTimedelta

# 导入 pandas 库中 Timedelta 类和 to_timedelta 函数
from pandas import (
    Timedelta,
    to_timedelta,
)
# 导入 pandas 库中的测试模块
import pandas._testing as tm


class TestNonNano:
    # 使用 pytest 的 fixture 功能，参数化测试单位字符串
    @pytest.fixture(params=["s", "ms", "us"])
    def unit_str(self, request):
        return request.param

    # 使用 pytest 的 fixture 功能，根据单位字符串获取对应的 NpyDatetimeUnit 值
    @pytest.fixture
    def unit(self, unit_str):
        # 7, 8, 9 分别对应秒、毫秒和微秒
        attr = f"NPY_FR_{unit_str}"
        return getattr(NpyDatetimeUnit, attr).value

    # 使用 pytest 的 fixture 功能，根据单位值生成相应的超出范围的时间值
    @pytest.fixture
    def val(self, unit):
        # 微秒值，刚好超出纳秒范围
        us = 9223372800000000
        if unit == NpyDatetimeUnit.NPY_FR_us.value:
            value = us
        elif unit == NpyDatetimeUnit.NPY_FR_ms.value:
            value = us // 1000
        else:
            value = us // 1_000_000
        return value

    # 使用 pytest 的 fixture 功能，根据单位值和时间值生成 Timedelta 对象
    @pytest.fixture
    def td(self, unit, val):
        return Timedelta._from_value_and_reso(val, unit)

    # 测试 Timedelta._from_value_and_reso 方法的正确性
    def test_from_value_and_reso(self, unit, val):
        # 测试 fixture 是否返回了预期的结果
        td = Timedelta._from_value_and_reso(val, unit)
        assert td._value == val
        assert td._creso == unit
        assert td.days == 106752

    # 测试 Timedelta 对象的一元操作（非纳秒单位）
    def test_unary_non_nano(self, td, unit):
        assert abs(td)._creso == unit
        assert (-td)._creso == unit
        assert (+td)._creso == unit

    # 测试 Timedelta 对象的减法操作是否保留了单位
    def test_sub_preserves_reso(self, td, unit):
        res = td - td
        expected = Timedelta._from_value_and_reso(0, unit)
        assert res == expected
        assert res._creso == unit

    # 测试 Timedelta 对象的乘法操作是否保留了单位
    def test_mul_preserves_reso(self, td, unit):
        # td fixture 的值始终远离实现边界，因此加倍不会导致溢出风险
        res = td * 2
        assert res._value == td._value * 2
        assert res._creso == unit

    # 测试比较不同单位 Timedelta 对象的比较操作
    def test_cmp_cross_reso(self, td):
        # NumPy 由于静默溢出而得出错误的结果
        other = Timedelta(days=106751, unit="ns")
        assert other < td
        assert td > other
        assert not other == td
        assert td != other

    # 测试 Timedelta 对象转换为 Python 原生 timedelta 对象的操作
    def test_to_pytimedelta(self, td):
        res = td.to_pytimedelta()
        expected = timedelta(days=106752)
        assert type(res) is timedelta
        assert res == expected
    # 测试将 Timedelta 对象转换为 timedelta64 类型
    def test_to_timedelta64(self, td, unit):
        # 遍历需要测试的转换结果：to_timedelta64(), to_numpy(), asm8
        for res in [td.to_timedelta64(), td.to_numpy(), td.asm8]:
            # 断言结果是 np.timedelta64 类型
            assert isinstance(res, np.timedelta64)
            # 断言结果的视图为整数，与 Timedelta 对象的值相同
            assert res.view("i8") == td._value
            # 根据不同的单位断言 dtype 的正确性
            if unit == NpyDatetimeUnit.NPY_FR_s.value:
                assert res.dtype == "m8[s]"
            elif unit == NpyDatetimeUnit.NPY_FR_ms.value:
                assert res.dtype == "m8[ms]"
            elif unit == NpyDatetimeUnit.NPY_FR_us.value:
                assert res.dtype == "m8[us]"

    # 测试 Timedelta 对象与类似 Timedelta 的对象之间的真除操作
    def test_truediv_timedeltalike(self, td):
        # 断言 Timedelta 对象与自身的真除为 1
        assert td / td == 1
        # 断言 (2.5 * td) 与 td 的真除结果为 2.5
        assert (2.5 * td) / td == 2.5

        # 创建另一个 Timedelta 对象 other
        other = Timedelta(td._value)
        # 准备错误消息
        msg = "Cannot cast 106752 days 00:00:00 to unit='ns' without overflow."
        # 使用 pytest 检查是否抛出了预期的异常 OutOfBoundsTimedelta，并匹配消息
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            td / other

        # 因为 Timedelta(other.to_pytimedelta()) 具有微秒分辨率，所以不需要完全转换为纳秒
        # 因此这个除法操作成功
        res = other.to_pytimedelta() / td
        expected = other.to_pytimedelta() / td.to_pytimedelta()
        assert res == expected

        # 如果没有溢出，我们将转换为更高的分辨率
        left = Timedelta._from_value_and_reso(50, NpyDatetimeUnit.NPY_FR_us.value)
        right = Timedelta._from_value_and_reso(50, NpyDatetimeUnit.NPY_FR_ms.value)
        result = left / right
        assert result == 0.001

        result = right / left
        assert result == 1000

    # 测试 Timedelta 对象与数值类型之间的真除操作
    def test_truediv_numeric(self, td):
        # 断言 Timedelta 对象与 np.nan 的真除结果为 NaT
        assert td / np.nan is NaT

        # 断言 Timedelta 对象与整数 2 的真除结果
        res = td / 2
        assert res._value == td._value / 2
        assert res._creso == td._creso

        # 断言 Timedelta 对象与浮点数 2.0 的真除结果
        res = td / 2.0
        assert res._value == td._value / 2
        assert res._creso == td._creso

    # 测试当 Timedelta 对象与 NA 类型不能进行真除操作时的异常情况
    def test_truediv_na_type_not_supported(self, td):
        # 准备错误消息：Timedelta 与 NAType 之间不支持真除操作
        msg_td_floordiv_na = (
            r"unsupported operand type\(s\) for /: 'Timedelta' and 'NAType'"
        )
        # 使用 pytest 检查是否抛出了预期的异常 TypeError，并匹配消息
        with pytest.raises(TypeError, match=msg_td_floordiv_na):
            td / NA

        # 准备错误消息：NAType 与 Timedelta 之间不支持真除操作
        msg_na_floordiv_td = (
            r"unsupported operand type\(s\) for /: 'NAType' and 'Timedelta'"
        )
        # 使用 pytest 检查是否抛出了预期的异常 TypeError，并匹配消息
        with pytest.raises(TypeError, match=msg_na_floordiv_td):
            NA / td
    # 测试用例：测试类似时间增量的整除运算
    def test_floordiv_timedeltalike(self, td):
        # 断言时间增量整除自身等于1
        assert td // td == 1
        # 断言2.5倍时间增量整除时间增量等于2
        assert (2.5 * td) // td == 2

        # 创建另一个时间增量对象
        other = Timedelta(td._value)
        # 设置错误消息
        msg = "Cannot cast 106752 days 00:00:00 to unit='ns' without overflow"
        # 使用 pytest 检查是否引发 OutOfBoundsTimedelta 异常，并匹配错误消息
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            td // other

        # 由于 Timedelta(other.to_pytimedelta()) 具有微秒分辨率，
        # 因此整除操作不需要完全转换为纳秒级别，所以成功
        res = other.to_pytimedelta() // td
        assert res == 0

        # 如果没有溢出，我们将转换到更高的分辨率
        left = Timedelta._from_value_and_reso(50050, NpyDatetimeUnit.NPY_FR_us.value)
        right = Timedelta._from_value_and_reso(50, NpyDatetimeUnit.NPY_FR_ms.value)
        result = left // right
        assert result == 1
        result = right // left
        assert result == 0

    # 测试用例：测试数值类型的整除运算
    def test_floordiv_numeric(self, td):
        # 断言时间增量整除 NaN 结果为 NaT
        assert td // np.nan is NaT

        # 对整数进行时间增量整除操作
        res = td // 2
        assert res._value == td._value // 2
        assert res._creso == td._creso

        # 对浮点数进行时间增量整除操作
        res = td // 2.0
        assert res._value == td._value // 2
        assert res._creso == td._creso

        # 断言时间增量整除 numpy 数组中的 NaN 结果为 NaT
        assert td // np.array(np.nan) is NaT

        # 对 numpy 数组中的整数进行时间增量整除操作
        res = td // np.array(2)
        assert res._value == td._value // 2
        assert res._creso == td._creso

        # 对 numpy 数组中的浮点数进行时间增量整除操作
        res = td // np.array(2.0)
        assert res._value == td._value // 2
        assert res._creso == td._creso

    # 测试用例：测试不支持的 NA 类型的整除运算
    def test_floordiv_na_type_not_supported(self, td):
        # 设置错误消息：时间增量与 NA 类型整除不支持
        msg_td_floordiv_na = (
            r"unsupported operand type\(s\) for //: 'Timedelta' and 'NAType'"
        )
        # 使用 pytest 检查是否引发 TypeError 异常，并匹配错误消息
        with pytest.raises(TypeError, match=msg_td_floordiv_na):
            td // NA

        # 设置错误消息：NA 类型与时间增量整除不支持
        msg_na_floordiv_td = (
            r"unsupported operand type\(s\) for //: 'NAType' and 'Timedelta'"
        )
        # 使用 pytest 检查是否引发 TypeError 异常，并匹配错误消息
        with pytest.raises(TypeError, match=msg_na_floordiv_td):
            NA // td
    def test_addsub_mismatched_reso(self, td):
        # 需要将 td 转换为微秒单位，因为 td 超出了 ns 的范围，否则会引发 OverflowError
        other = Timedelta(days=1).as_unit("us")

        # 计算 td + other
        result = td + other
        # 断言结果的分辨率与 other 相同
        assert result._creso == other._creso
        # 断言结果的天数是 td 的天数加一
        assert result.days == td.days + 1

        # 计算 other + td
        result = other + td
        # 断言结果的分辨率与 other 相同
        assert result._creso == other._creso
        # 断言结果的天数是 td 的天数加一
        assert result.days == td.days + 1

        # 计算 td - other
        result = td - other
        # 断言结果的分辨率与 other 相同
        assert result._creso == other._creso
        # 断言结果的天数是 td 的天数减一
        assert result.days == td.days - 1

        # 计算 other - td
        result = other - td
        # 断言结果的分辨率与 other 相同
        assert result._creso == other._creso
        # 断言结果的天数是一减去 td 的天数
        assert result.days == 1 - td.days

        # 创建另一个 Timedelta 对象 other2
        other2 = Timedelta(500)
        # 准备错误信息
        msg = "Cannot cast 106752 days 00:00:00 to unit='ns' without overflow"
        # 断言在进行运算时会引发 OutOfBoundsTimedelta 异常，匹配错误信息 msg
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            td + other2
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            other2 + td
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            td - other2
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            other2 - td
def test_timedelta_class_min_max_resolution():
    # 当从类访问时（而不是实例），Timedelta.min 默认单位为纳秒
    assert Timedelta.min == Timedelta(NaT._value + 1)
    # 确保 Timedelta.min 的单位为纳秒
    assert Timedelta.min._creso == NpyDatetimeUnit.NPY_FR_ns.value

    # Timedelta.max 设置为 np.int64 的最大值
    assert Timedelta.max == Timedelta(np.iinfo(np.int64).max)
    # 确保 Timedelta.max 的单位为纳秒
    assert Timedelta.max._creso == NpyDatetimeUnit.NPY_FR_ns.value

    # Timedelta.resolution 设置为 1 单位时间间隔
    assert Timedelta.resolution == Timedelta(1)
    # 确保 Timedelta.resolution 的单位为纳秒
    assert Timedelta.resolution._creso == NpyDatetimeUnit.NPY_FR_ns.value


class TestTimedeltaUnaryOps:
    def test_invert(self):
        # 创建一个 Timedelta 对象 td，表示 10 天
        td = Timedelta(10, unit="D")

        # 测试在 td 上应用 ~ 运算符会抛出 TypeError 异常
        msg = "bad operand type for unary ~"
        with pytest.raises(TypeError, match=msg):
            ~td

        # 使用 pytimedelta 尝试再次应用 ~ 运算符，预期同样会抛出 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            ~(td.to_pytimedelta())

        # 使用 timedelta64 尝试再次应用 ~ 运算符，预期会抛出 ufunc 'invert' not supported 异常
        umsg = "ufunc 'invert' not supported for the input types"
        with pytest.raises(TypeError, match=umsg):
            ~(td.to_timedelta64())

    def test_unary_ops(self):
        # 创建一个 Timedelta 对象 td，表示 10 天
        td = Timedelta(10, unit="D")

        # 测试负号操作符 __neg__
        assert -td == Timedelta(-10, unit="D")
        assert -td == Timedelta("-10D")
        # 测试正号操作符 __pos__
        assert +td == Timedelta(10, unit="D")

        # 测试绝对值操作符 __abs__
        assert abs(td) == td
        assert abs(-td) == td
        assert abs(-td) == Timedelta("10D")


class TestTimedeltas:
    @pytest.mark.parametrize(
        "unit, value, expected",
        [
            ("us", 9.999, 9999),
            ("ms", 9.999999, 9999999),
            ("s", 9.999999999, 9999999999),
        ],
    )
    def test_rounding_on_int_unit_construction(self, unit, value, expected):
        # GH 12690：测试在整数单位构造中的四舍五入
        result = Timedelta(value, unit=unit)
        assert result._value == expected
        result = Timedelta(str(value) + unit)
        assert result._value == expected

    def test_total_seconds_scalar(self):
        # 查看 gh-10939：测试 total_seconds() 方法
        rng = Timedelta("1 days, 10:11:12.100123456")
        # 预期的秒数总和，包括毫秒和纳秒
        expt = 1 * 86400 + 10 * 3600 + 11 * 60 + 12 + 100123456.0 / 1e9
        tm.assert_almost_equal(rng.total_seconds(), expt)

        # 测试当 Timedelta 对象包含 NaN 时的情况
        rng = Timedelta(np.nan)
        assert np.isnan(rng.total_seconds())

    def test_conversion(self):
        for td in [Timedelta(10, unit="D"), Timedelta("1 days, 10:11:12.012345")]:
            # 测试转换为 pytimedelta
            pydt = td.to_pytimedelta()
            assert td == Timedelta(pydt)
            assert td == pydt
            # pytimedelta 是 timedelta 的实例，但不是 Timedelta 的实例
            assert isinstance(pydt, timedelta) and not isinstance(pydt, Timedelta)

            # 测试转换为 timedelta64
            assert td == np.timedelta64(td._value, "ns")
            td64 = td.to_timedelta64()

            assert td64 == np.timedelta64(td._value, "ns")
            assert td == td64

            assert isinstance(td64, np.timedelta64)

        # 这个例子不相等，因为无法精确转换（由于纳秒级别的差异）
        td = Timedelta("1 days, 10:11:12.012345678")
        assert td != td.to_pytimedelta()
    # 定义测试方法 test_fields，用于测试 timedelta 相关功能
    def test_fields(self):
        # 定义辅助函数 check，用于断言值是否为整数
        def check(value):
            # 断言 value 是整数类型
            assert isinstance(value, int)

        # 使用 to_timedelta 函数创建时间间隔 rng，表示1天10小时11分12秒
        rng = to_timedelta("1 days, 10:11:12")
        # 断言时间间隔 rng 的天数为1
        assert rng.days == 1
        # 断言时间间隔 rng 的秒数为10小时11分12秒转换成秒的结果
        assert rng.seconds == 10 * 3600 + 11 * 60 + 12
        # 断言时间间隔 rng 的微秒数为0
        assert rng.microseconds == 0
        # 断言时间间隔 rng 的纳秒数为0
        assert rng.nanoseconds == 0

        # 使用 pytest 模块验证对不存在属性的访问会抛出 AttributeError 异常，并匹配指定消息
        msg = "'Timedelta' object has no attribute '{}'"
        with pytest.raises(AttributeError, match=msg.format("hours")):
            rng.hours
        with pytest.raises(AttributeError, match=msg.format("minutes")):
            rng.minutes
        with pytest.raises(AttributeError, match=msg.format("milliseconds")):
            rng.milliseconds

        # 对 rng 的天数、秒数、微秒数、纳秒数应用 check 函数验证其为整数
        check(rng.days)
        check(rng.seconds)
        check(rng.microseconds)
        check(rng.nanoseconds)

        # 创建 Timedelta 对象 td，表示-1天10小时11分12秒
        td = Timedelta("-1 days, 10:11:12")
        # 断言 td 的绝对值为 Timedelta("13:48:48")
        assert abs(td) == Timedelta("13:48:48")
        # 断言 td 的字符串表示为 "-1 days +10:11:12"
        assert str(td) == "-1 days +10:11:12"
        # 断言 -td 的值为 Timedelta("0 days 13:48:48")
        assert -td == Timedelta("0 days 13:48:48")
        # 断言 -Timedelta("-1 days, 10:11:12")._value 的值为 49728000000000
        assert -Timedelta("-1 days, 10:11:12")._value == 49728000000000
        # 断言 Timedelta("-1 days, 10:11:12")._value 的值为 -49728000000000
        assert Timedelta("-1 days, 10:11:12")._value == -49728000000000

        # 使用 to_timedelta 函数创建时间间隔 rng，表示-1天10小时11分12.100123456秒
        rng = to_timedelta("-1 days, 10:11:12.100123456")
        # 断言时间间隔 rng 的天数为-1
        assert rng.days == -1
        # 断言时间间隔 rng 的秒数为10小时11分12秒转换成秒的结果
        assert rng.seconds == 10 * 3600 + 11 * 60 + 12
        # 断言时间间隔 rng 的微秒数为100毫秒123微秒
        assert rng.microseconds == 100 * 1000 + 123
        # 断言时间间隔 rng 的纳秒数为456
        assert rng.nanoseconds == 456
        # 再次使用 pytest 模块验证对不存在属性的访问会抛出 AttributeError 异常，并匹配指定消息
        with pytest.raises(AttributeError, match=msg.format("hours")):
            rng.hours
        with pytest.raises(AttributeError, match=msg.format("minutes")):
            rng.minutes
        with pytest.raises(AttributeError, match=msg.format("milliseconds")):
            rng.milliseconds

        # 使用 to_timedelta 函数创建时间间隔 tup，表示-1微秒
        tup = to_timedelta(-1, "us").components
        # 断言时间间隔 tup 的天数为-1
        assert tup.days == -1
        # 断言时间间隔 tup 的小时数为23
        assert tup.hours == 23
        # 断言时间间隔 tup 的分钟数为59
        assert tup.minutes == 59
        # 断言时间间隔 tup 的秒数为59
        assert tup.seconds == 59
        # 断言时间间隔 tup 的毫秒数为999
        assert tup.milliseconds == 999
        # 断言时间间隔 tup 的微秒数为999
        assert tup.microseconds == 999
        # 断言时间间隔 tup 的纳秒数为0
        assert tup.nanoseconds == 0

        # 对 tup 的各个时间单位应用 check 函数验证其为整数
        check(tup.days)
        check(tup.hours)
        check(tup.minutes)
        check(tup.seconds)
        check(tup.milliseconds)
        check(tup.microseconds)
        check(tup.nanoseconds)

        # 使用 to_timedelta 函数创建时间间隔 tup，表示-1天1微秒
        tup = Timedelta("-1 days 1 us").components
        # 断言时间间隔 tup 的天数为-2
        assert tup.days == -2
        # 断言时间间隔 tup 的小时数为23
        assert tup.hours == 23
        # 断言时间间隔 tup 的分钟数为59
        assert tup.minutes == 59
        # 断言时间间隔 tup 的秒数为59
        assert tup.seconds == 59
        # 断言时间间隔 tup 的毫秒数为999
        assert tup.milliseconds == 999
        # 断言时间间隔 tup 的微秒数为999
        assert tup.microseconds == 999
        # 断言时间间隔 tup 的纳秒数为0
        assert tup.nanoseconds == 0

    # TODO: this is a test of to_timedelta string parsing
    # TODO: 这是对 to_timedelta 函数解析字符串的测试
    def test_iso_conversion(self):
        # GH #21877
        # GH #21877
        expected = Timedelta(1, unit="s")
        # 断言将字符串 "P0DT0H0M1S" 转换为时间间隔对象，结果与 expected 相等
        assert to_timedelta("P0DT0H0M1S") == expected

    # TODO: this is a test of to_timedelta returning NaT
    # TODO: 这是对 to_timedelta 返回 NaT 的测试
    # 测试将字符串"nat"转换为numpy的时间增量，然后转换为numpy数组
    def test_nat_converters(self):
        result = to_timedelta("nat").to_numpy()
        # 断言结果的数据类型的种类是时间增量
        assert result.dtype.kind == "M"
        # 断言结果转换为int64类型后与iNaT相等
        assert result.astype("int64") == iNaT

        # 测试将字符串"nan"转换为numpy的时间增量，然后转换为numpy数组
        result = to_timedelta("nan").to_numpy()
        # 断言结果的数据类型的种类是时间增量
        assert result.dtype.kind == "M"
        # 断言结果转换为int64类型后与iNaT相等
        assert result.astype("int64") == iNaT

    # 测试不同单位下的时间增量与numpy.timedelta64的转换
    def test_numeric_conversions(self):
        # 断言时间增量0与np.timedelta64(0, "ns")相等
        assert Timedelta(0) == np.timedelta64(0, "ns")
        # 断言时间增量10与np.timedelta64(10, "ns")相等
        assert Timedelta(10) == np.timedelta64(10, "ns")
        # 断言时间增量10秒与np.timedelta64(10, "ns")相等
        assert Timedelta(10, unit="ns") == np.timedelta64(10, "ns")

        # 断言时间增量10微秒与np.timedelta64(10, "us")相等
        assert Timedelta(10, unit="us") == np.timedelta64(10, "us")
        # 断言时间增量10毫秒与np.timedelta64(10, "ms")相等
        assert Timedelta(10, unit="ms") == np.timedelta64(10, "ms")
        # 断言时间增量10秒与np.timedelta64(10, "s")相等
        assert Timedelta(10, unit="s") == np.timedelta64(10, "s")
        # 断言时间增量10天与np.timedelta64(10, "D")相等
        assert Timedelta(10, unit="D") == np.timedelta64(10, "D")

    # 测试将datetime.timedelta转换为时间增量，并确保转换后的精度为纳秒
    def test_timedelta_conversions(self):
        # 断言将1秒的datetime.timedelta转换为np.timedelta64(1, "s")后，再转换为"m8[ns]"类型相等
        assert Timedelta(timedelta(seconds=1)) == np.timedelta64(1, "s").astype(
            "m8[ns]"
        )
        # 断言将1微秒的datetime.timedelta转换为np.timedelta64(1, "us")后，再转换为"m8[ns]"类型相等
        assert Timedelta(timedelta(microseconds=1)) == np.timedelta64(1, "us").astype(
            "m8[ns]"
        )
        # 断言将1天的datetime.timedelta转换为np.timedelta64(1, "D")后，再转换为"m8[ns]"类型相等
        assert Timedelta(timedelta(days=1)) == np.timedelta64(1, "D").astype("m8[ns]")

    # 测试.to_numpy()方法的别名，用于标量的情况
    def test_to_numpy_alias(self):
        # GH 24653: 验证将时间增量字符串"10m7s"转换为numpy.timedelta64，并用.to_numpy()进行比较
        td = Timedelta("10m7s")
        assert td.to_timedelta64() == td.to_numpy()

        # GH#44460: 验证.to_numpy()方法在指定dtype和copy参数时抛出值错误的异常
        msg = "dtype and copy arguments are ignored"
        with pytest.raises(ValueError, match=msg):
            td.to_numpy("m8[s]")
        with pytest.raises(ValueError, match=msg):
            td.to_numpy(copy=True)

    # 测试时间增量对象的身份确认
    def test_identity(self):
        # 创建一个时间增量对象
        td = Timedelta(10, unit="D")
        # 断言该对象是Timedelta类的实例
        assert isinstance(td, Timedelta)
        # 断言该对象是datetime.timedelta类的实例
        assert isinstance(td, timedelta)
    # 定义测试函数 test_short_format_converters，用于测试时间差字符串转换功能
    def test_short_format_converters(self):
        # 定义内部函数 conv，将输入的时间差转换为纳秒精度
        def conv(v):
            return v.astype("m8[ns]")

        # 断言 Timedelta("10") 等于 np.timedelta64(10, "ns")
        assert Timedelta("10") == np.timedelta64(10, "ns")
        # 断言 Timedelta("10ns") 等于 np.timedelta64(10, "ns")
        assert Timedelta("10ns") == np.timedelta64(10, "ns")
        # 断言 Timedelta("100") 等于 np.timedelta64(100, "ns")
        assert Timedelta("100") == np.timedelta64(100, "ns")
        # 断言 Timedelta("100ns") 等于 np.timedelta64(100, "ns")
        assert Timedelta("100ns") == np.timedelta64(100, "ns")

        # 断言 Timedelta("1000") 等于 np.timedelta64(1000, "ns")
        assert Timedelta("1000") == np.timedelta64(1000, "ns")
        # 断言 Timedelta("1000ns") 等于 np.timedelta64(1000, "ns")
        assert Timedelta("1000ns") == np.timedelta64(1000, "ns")

        # 发出警告消息 "'NS' is deprecated and will be removed in a future version."
        msg = "'NS' is deprecated and will be removed in a future version."
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # 断言 Timedelta("1000NS") 等于 np.timedelta64(1000, "ns")
            assert Timedelta("1000NS") == np.timedelta64(1000, "ns")

        # 断言 Timedelta("10us") 等于 np.timedelta64(10000, "ns")
        assert Timedelta("10us") == np.timedelta64(10000, "ns")
        # 断言 Timedelta("100us") 等于 np.timedelta64(100000, "ns")
        assert Timedelta("100us") == np.timedelta64(100000, "ns")
        # 断言 Timedelta("1000us") 等于 np.timedelta64(1000000, "ns")
        assert Timedelta("1000us") == np.timedelta64(1000000, "ns")
        # 断言 Timedelta("1000Us") 等于 np.timedelta64(1000000, "ns")
        assert Timedelta("1000Us") == np.timedelta64(1000000, "ns")
        # 断言 Timedelta("1000uS") 等于 np.timedelta64(1000000, "ns")
        assert Timedelta("1000uS") == np.timedelta64(1000000, "ns")

        # 断言 Timedelta("1ms") 等于 np.timedelta64(1000000, "ns")
        assert Timedelta("1ms") == np.timedelta64(1000000, "ns")
        # 断言 Timedelta("10ms") 等于 np.timedelta64(10000000, "ns")
        assert Timedelta("10ms") == np.timedelta64(10000000, "ns")
        # 断言 Timedelta("100ms") 等于 np.timedelta64(100000000, "ns")
        assert Timedelta("100ms") == np.timedelta64(100000000, "ns")
        # 断言 Timedelta("1000ms") 等于 np.timedelta64(1000000000, "ns")
        assert Timedelta("1000ms") == np.timedelta64(1000000000, "ns")

        # 断言 Timedelta("-1s") 等于 -np.timedelta64(1000000000, "ns")
        assert Timedelta("-1s") == -np.timedelta64(1000000000, "ns")
        # 断言 Timedelta("1s") 等于 np.timedelta64(1000000000, "ns")
        assert Timedelta("1s") == np.timedelta64(1000000000, "ns")
        # 断言 Timedelta("10s") 等于 np.timedelta64(10000000000, "ns")
        assert Timedelta("10s") == np.timedelta64(10000000000, "ns")
        # 断言 Timedelta("100s") 等于 np.timedelta64(100000000000, "ns")
        assert Timedelta("100s") == np.timedelta64(100000000000, "ns")
        # 断言 Timedelta("1000s") 等于 np.timedelta64(1000000000000, "ns")
        assert Timedelta("1000s") == np.timedelta64(1000000000000, "ns")

        # 发出警告消息 "'d' is deprecated and will be removed in a future version."
        msg = "'d' is deprecated and will be removed in a future version."
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # 断言 Timedelta("1d") 等于 conv(np.timedelta64(1, "D"))
            assert Timedelta("1d") == conv(np.timedelta64(1, "D"))
        # 断言 Timedelta("-1D") 等于 -conv(np.timedelta64(1, "D"))
        assert Timedelta("-1D") == -conv(np.timedelta64(1, "D"))
        # 断言 Timedelta("1D") 等于 conv(np.timedelta64(1, "D"))
        assert Timedelta("1D") == conv(np.timedelta64(1, "D"))
        # 断言 Timedelta("10D") 等于 conv(np.timedelta64(10, "D"))
        assert Timedelta("10D") == conv(np.timedelta64(10, "D"))
        # 断言 Timedelta("100D") 等于 conv(np.timedelta64(100, "D"))
        assert Timedelta("100D") == conv(np.timedelta64(100, "D"))
        # 断言 Timedelta("1000D") 等于 conv(np.timedelta64(1000, "D"))
        assert Timedelta("1000D") == conv(np.timedelta64(1000, "D"))
        # 断言 Timedelta("10000D") 等于 conv(np.timedelta64(10000, "D"))
        assert Timedelta("10000D") == conv(np.timedelta64(10000, "D"))

        # 断言 Timedelta(" 10000D ") 等于 conv(np.timedelta64(10000, "D"))
        assert Timedelta(" 10000D ") == conv(np.timedelta64(10000, "D"))
        # 断言 Timedelta(" - 10000D ") 等于 -conv(np.timedelta64(10000, "D"))
        assert Timedelta(" - 10000D ") == -conv(np.timedelta64(10000, "D"))

        # 无效输入，预期抛出 ValueError 异常，匹配消息 "invalid unit abbreviation"
        msg = "invalid unit abbreviation"
        with pytest.raises(ValueError, match=msg):
            Timedelta("1foo")
        # 无效输入，预期抛出 ValueError 异常，匹配消息 "unit abbreviation w/o a number"
        msg = "unit abbreviation w/o a number"
        with pytest.raises(ValueError, match=msg):
            Timedelta("foo")
    # 定义一个测试方法，用于测试完整格式的转换器功能
    def test_full_format_converters(self):
        # 定义一个内部函数 conv，用于将输入值转换为纳秒精度的时间增量
        def conv(v):
            return v.astype("m8[ns]")

        # 创建一个表示一天时间增量的 timedelta 对象
        d1 = np.timedelta64(1, "D")

        # 断言：将 timedelta 对象 d1 转换为 Timedelta 对象并比较结果
        assert Timedelta("1days") == conv(d1)
        # 断言：将 timedelta 对象 d1 转换为 Timedelta 对象并比较结果，带逗号
        assert Timedelta("1days,") == conv(d1)
        # 断言：将 -d1 转换为 Timedelta 对象并比较结果，带负号
        assert Timedelta("- 1days,") == -conv(d1)

        # 断言：将表示 1 秒的 timedelta 对象转换为 Timedelta 对象并比较结果
        assert Timedelta("00:00:01") == conv(np.timedelta64(1, "s"))
        # 断言：将表示 6 小时 1 秒的 timedelta 对象转换为 Timedelta 对象并比较结果
        assert Timedelta("06:00:01") == conv(np.timedelta64(6 * 3600 + 1, "s"))
        # 断言：将表示 6 小时 1 秒的 timedelta 对象转换为 Timedelta 对象并比较结果（带整数秒）
        assert Timedelta("06:00:01.0") == conv(np.timedelta64(6 * 3600 + 1, "s"))
        # 断言：将表示 6 小时 1.01 秒的 timedelta 对象转换为 Timedelta 对象并比较结果（带毫秒）
        assert Timedelta("06:00:01.01") == conv(
            np.timedelta64(1000 * (6 * 3600 + 1) + 10, "ms")
        )

        # 断言：将 -d1 + 1 秒 转换为 Timedelta 对象并比较结果，带负号
        assert Timedelta("- 1days, 00:00:01") == conv(-d1 + np.timedelta64(1, "s"))
        # 断言：将 d1 + 6 小时 1 秒 转换为 Timedelta 对象并比较结果
        assert Timedelta("1days, 06:00:01") == conv(
            d1 + np.timedelta64(6 * 3600 + 1, "s")
        )
        # 断言：将 d1 + 6 小时 1.01 秒 转换为 Timedelta 对象并比较结果（带毫秒）
        assert Timedelta("1days, 06:00:01.01") == conv(
            d1 + np.timedelta64(1000 * (6 * 3600 + 1) + 10, "ms")
        )

        # 无效断言：检查抛出 ValueError 异常，其消息与指定的 msg 匹配
        msg = "have leftover units"
        with pytest.raises(ValueError, match=msg):
            Timedelta("- 1days, 00")

    # 定义一个测试方法，用于测试 Timedelta 对象的 pickle 序列化和反序列化功能
    def test_pickle(self):
        # 创建一个 Timedelta 对象 v
        v = Timedelta("1 days 10:11:12.0123456")
        # 使用 pandas 内部方法 round_trip_pickle 进行对象 v 的 pickle 序列化和反序列化
        v_p = tm.round_trip_pickle(v)
        # 断言：验证反序列化后的对象 v_p 是否与原对象 v 相等
        assert v == v_p

    # 定义一个测试方法，用于验证 Timedelta 对象的哈希值相等性
    def test_timedelta_hash_equality(self):
        # GH 11129

        # 创建一个表示 1 天的 Timedelta 对象 v 和一个 Python 的 timedelta 对象 td
        v = Timedelta(1, "D")
        td = timedelta(days=1)
        # 断言：验证 Timedelta 对象 v 和 timedelta 对象 td 的哈希值是否相等
        assert hash(v) == hash(td)

        # 创建一个字典 d，将 timedelta 对象 td 作为键，值为 2
        d = {td: 2}
        # 断言：验证 Timedelta 对象 v 作为键是否能在字典 d 中找到相应的值
        assert d[v] == 2

        # 创建一个列表 tds，其中包含 20 个 Timedelta 对象，分别表示从 0 到 19 天加 1 秒的时间增量
        tds = [Timedelta(seconds=1) + Timedelta(days=n) for n in range(20)]
        # 断言：验证列表 tds 中所有 Timedelta 对象的哈希值与其对应的 Python timedelta 对象的哈希值是否相等
        assert all(hash(td) == hash(td.to_pytimedelta()) for td in tds)

        # Python 的 timedelta 对象忽略纳秒精度
        # 创建一个表示 1 纳秒的 Timedelta 对象 ns_td
        ns_td = Timedelta(1, "ns")
        # 断言：验证 Timedelta 对象 ns_td 的哈希值与其对应的 Python timedelta 对象的哈希值是否不相等
        assert hash(ns_td) != hash(ns_td.to_pytimedelta())

    # 使用 pytest 的装饰器标记，跳过 UBSan 测试，并标记为预期的失败测试
    @pytest.mark.skip_ubsan
    @pytest.mark.xfail(
        reason="pd.Timedelta violates the Python hash invariant (GH#44504).",
    )
    # 使用 hypothesis 的 given 装饰器定义一个测试方法，用于验证哈希值相等性不变性
    @given(
        st.integers(
            min_value=(-sys.maxsize - 1) // 500,
            max_value=sys.maxsize // 500,
        )
    )
    def test_hash_equality_invariance(self, half_microseconds: int) -> None:
        # GH#44504

        # 计算半微秒数对应的纳秒数
        nanoseconds = half_microseconds * 500

        # 创建一个表示 nanoseconds 纳秒的 Timedelta 对象 pandas_timedelta
        pandas_timedelta = Timedelta(nanoseconds)
        # 创建一个表示 nanoseconds 纳秒的 numpy timedelta 对象 numpy_timedelta
        numpy_timedelta = np.timedelta64(nanoseconds)

        # 根据 Python 文档中的定义，验证可哈希对象在相等时具有相同的哈希值
        assert pandas_timedelta != numpy_timedelta or hash(pandas_timedelta) == hash(
            numpy_timedelta
        )
    def test_implementation_limits(self):
        # 创建Timedelta对象，使用Timedelta的最小值和最大值
        min_td = Timedelta(Timedelta.min)
        max_td = Timedelta(Timedelta.max)

        # GH 12727
        # timedelta的限制对应于int64的边界
        assert min_td._value == iNaT + 1
        assert max_td._value == lib.i8max

        # 在下限之外，溢出之前的NAT
        assert (min_td - Timedelta(1, "ns")) is NaT

        # 使用pytest检查OverflowError异常，确保溢出
        msg = "int too (large|big) to convert"
        with pytest.raises(OverflowError, match=msg):
            min_td - Timedelta(2, "ns")

        with pytest.raises(OverflowError, match=msg):
            max_td + Timedelta(1, "ns")

        # 使用内部纳秒值进行相同的测试
        td = Timedelta(min_td._value - 1, "ns")
        assert td is NaT

        # 使用pytest检查OutOfBoundsTimedelta异常，确保溢出
        msg = "Cannot cast -9223372036854775809 from ns to 'ns' without overflow"
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            Timedelta(min_td._value - 2, "ns")

        msg = "Cannot cast 9223372036854775808 from ns to 'ns' without overflow"
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            Timedelta(max_td._value + 1, "ns")

    def test_total_seconds_precision(self):
        # GH 19458
        # 检查total_seconds方法的精度
        assert Timedelta("30s").total_seconds() == 30.0
        assert Timedelta("0").total_seconds() == 0.0
        assert Timedelta("-2s").total_seconds() == -2.0
        assert Timedelta("5.324s").total_seconds() == 5.324
        assert (Timedelta("30s").total_seconds() - 30.0) < 1e-20
        assert (30.0 - Timedelta("30s").total_seconds()) < 1e-20

    def test_resolution_string(self):
        # GH 19458
        # 检查resolution_string属性返回的时间分辨率字符串
        assert Timedelta(days=1).resolution_string == "D"
        assert Timedelta(days=1, hours=6).resolution_string == "h"
        assert Timedelta(days=1, minutes=6).resolution_string == "min"
        assert Timedelta(days=1, seconds=6).resolution_string == "s"
        assert Timedelta(days=1, milliseconds=6).resolution_string == "ms"
        assert Timedelta(days=1, microseconds=6).resolution_string == "us"
        assert Timedelta(days=1, nanoseconds=6).resolution_string == "ns"

    def test_resolution_deprecated(self):
        # GH#21344
        # 检查resolution属性的行为
        td = Timedelta(days=4, hours=3)
        result = td.resolution
        assert result == Timedelta(nanoseconds=1)

        # 检查类属性上的resolution属性，模仿标准库timedelta的行为
        result = Timedelta.resolution
        assert result == Timedelta(nanoseconds=1)

    @pytest.mark.parametrize(
        "unit,unit_depr",
        [
            ("W", "w"),
            ("D", "d"),
            ("min", "MIN"),
            ("s", "S"),
            ("h", "H"),
            ("ms", "MS"),
            ("us", "US"),
        ],
    )
    # 定义一个测试方法，用于测试单位是否已经被弃用
    def test_unit_deprecated(self, unit, unit_depr):
        # GH#59051
        # 构建弃用警告的消息，提示用户该单位在未来版本中将被移除
        msg = f"'{unit_depr}' is deprecated and will be removed in a future version."

        # 使用断言确保在创建 Timedelta 对象时会产生 FutureWarning 警告，并且警告消息与预期相匹配
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # 创建 Timedelta 对象，使用已弃用的单位
            result = Timedelta(1, unit_depr)
        
        # 断言结果对象与指定单位的 Timedelta 对象相等
        assert result == Timedelta(1, unit)
# 使用 pytest 的 pytest.mark.parametrize 装饰器进行参数化测试
@pytest.mark.parametrize(
    "value, expected",
    [
        # 测试 Timedelta 对象是否被视为 True
        (Timedelta("10s"), True),
        (Timedelta("-10s"), True),
        (Timedelta(10, unit="ns"), True),
        # 测试 Timedelta 对象是否被视为 False
        (Timedelta(0, unit="ns"), False),
        (Timedelta(-10, unit="ns"), True),
        # 测试 Timedelta(None) 和 NaT 是否被视为 True
        (Timedelta(None), True),
        (NaT, True),
    ],
)
def test_truthiness(value, expected):
    # 验证 Timedelta 对象的布尔真值是否与预期一致
    assert bool(value) is expected


def test_timedelta_attribute_precision():
    # GH 31354：测试 Timedelta 对象的精度问题
    td = Timedelta(1552211999999999872, unit="ns")
    # 计算 Timedelta 对象的总纳秒数
    result = td.days * 86400
    result += td.seconds
    result *= 1000000
    result += td.microseconds
    result *= 1000
    result += td.nanoseconds
    expected = td._value
    # 验证计算出的总纳秒数是否与预期相等
    assert result == expected


def test_to_pytimedelta_large_values():
    # 测试 Timedelta 对象转换为 Python 内置 timedelta 对象的大数值情况
    td = Timedelta(1152921504609987375, unit="ns")
    # 执行 Timedelta 对象到 Python 内置 timedelta 对象的转换
    result = td.to_pytimedelta()
    # 预期的 Python 内置 timedelta 对象
    expected = timedelta(days=13343, seconds=86304, microseconds=609987)
    # 验证转换结果是否与预期相等
    assert result == expected
```