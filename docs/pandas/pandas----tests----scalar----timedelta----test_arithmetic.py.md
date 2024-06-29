# `D:\src\scipysrc\pandas\pandas\tests\scalar\timedelta\test_arithmetic.py`

```
"""
Tests for scalar Timedelta arithmetic ops
"""

# 引入需要的模块和类
from datetime import (
    datetime,
    timedelta,
)
import operator

import numpy as np
import pytest

# 引入 Pandas 库及相关模块和异常类
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
    NaT,
    Timedelta,
    Timestamp,
    offsets,
)
import pandas._testing as tm
from pandas.core import ops


class TestTimedeltaAdditionSubtraction:
    """
    Tests for Timedelta methods:

        __add__, __radd__,
        __sub__, __rsub__
    """

    @pytest.mark.parametrize(
        "ten_seconds",
        [
            Timedelta(10, unit="s"),
            timedelta(seconds=10),
            np.timedelta64(10, "s"),
            np.timedelta64(10000000000, "ns"),
            offsets.Second(10),
        ],
    )
    def test_td_add_sub_ten_seconds(self, ten_seconds):
        # GH#6808
        # 定义基准时间和预期结果
        base = Timestamp("20130101 09:01:12.123456")
        expected_add = Timestamp("20130101 09:01:22.123456")
        expected_sub = Timestamp("20130101 09:01:02.123456")

        # 测试 Timedelta 的加法操作
        result = base + ten_seconds
        assert result == expected_add

        # 测试 Timedelta 的减法操作
        result = base - ten_seconds
        assert result == expected_sub

    @pytest.mark.parametrize(
        "one_day_ten_secs",
        [
            Timedelta("1 day, 00:00:10"),
            Timedelta("1 days, 00:00:10"),
            timedelta(days=1, seconds=10),
            np.timedelta64(1, "D") + np.timedelta64(10, "s"),
            offsets.Day() + offsets.Second(10),
        ],
    )
    def test_td_add_sub_one_day_ten_seconds(self, one_day_ten_secs):
        # GH#6808
        # 定义基准时间和预期结果
        base = Timestamp("20130102 09:01:12.123456")
        expected_add = Timestamp("20130103 09:01:22.123456")
        expected_sub = Timestamp("20130101 09:01:02.123456")

        # 测试 Timedelta 的加法操作
        result = base + one_day_ten_secs
        assert result == expected_add

        # 测试 Timedelta 的减法操作
        result = base - one_day_ten_secs
        assert result == expected_sub

    @pytest.mark.parametrize("op", [operator.add, ops.radd])
    def test_td_add_datetimelike_scalar(self, op):
        # GH#19738
        # 创建 Timedelta 对象
        td = Timedelta(10, unit="D")

        # 测试 Timedelta 与 datetime 相加的操作
        result = op(td, datetime(2016, 1, 1))
        if op is operator.add:
            # 当使用加法操作时，Timedelta.__radd__ 不会被调用，
            # 因此返回的是 datetime 而不是 Timestamp
            assert isinstance(result, Timestamp)
        assert result == Timestamp(2016, 1, 11)

        # 测试 Timedelta 与 Timestamp 相加的操作
        result = op(td, Timestamp("2018-01-12 18:09"))
        assert isinstance(result, Timestamp)
        assert result == Timestamp("2018-01-22 18:09")

        # 测试 Timedelta 与 np.datetime64 相加的操作
        result = op(td, np.datetime64("2018-01-12"))
        assert isinstance(result, Timestamp)
        assert result == Timestamp("2018-01-22")

        # 测试 Timedelta 与 NaT（Not a Time）相加的操作
        result = op(td, NaT)
        assert result is NaT
    # 测试当时间增量超出范围时是否引发异常
    def test_td_add_timestamp_overflow(self):
        # 创建起始时间戳为公元1700年1月1日的时间戳对象，并以纳秒为单位表示
        ts = Timestamp("1700-01-01").as_unit("ns")
        # 设置异常消息内容
        msg = "Cannot cast 259987 from D to 'ns' without overflow."
        # 使用 pytest 检查是否引发 OutOfBoundsTimedelta 异常，并匹配预期的异常消息
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            # 尝试将时间增量设定为 13 * 19999 天并加到时间戳上
            ts + Timedelta(13 * 19999, unit="D")

        # 设置第二个异常消息内容
        msg = "Cannot cast 259987 days 00:00:00 to unit='ns' without overflow"
        # 再次使用 pytest 检查是否引发 OutOfBoundsTimedelta 异常，并匹配预期的异常消息
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            # 尝试将时间增量设定为 13 * 19999 天并加到时间戳上，使用 Python 的 timedelta 对象
            ts + timedelta(days=13 * 19999)

    # 使用参数化测试来测试时间增量与时间增量的加法操作
    @pytest.mark.parametrize("op", [operator.add, ops.radd])
    def test_td_add_td(self, op):
        # 创建一个时间增量对象，表示 10 天
        td = Timedelta(10, unit="D")

        # 使用 op 参数（加法操作符或其反向版本）对两个时间增量进行加法操作
        result = op(td, Timedelta(days=10))
        # 断言结果是 Timedelta 类型，并且值为 20 天
        assert isinstance(result, Timedelta)
        assert result == Timedelta(days=20)

    # 使用参数化测试来测试时间增量与 Python 的 timedelta 对象的加法操作
    @pytest.mark.parametrize("op", [operator.add, ops.radd])
    def test_td_add_pytimedelta(self, op):
        # 创建一个时间增量对象，表示 10 天
        td = Timedelta(10, unit="D")
        # 使用 op 参数（加法操作符或其反向版本）对时间增量与 Python 的 timedelta 对象进行加法操作
        result = op(td, timedelta(days=9))
        # 断言结果是 Timedelta 类型，并且值为 19 天
        assert isinstance(result, Timedelta)
        assert result == Timedelta(days=19)

    # 使用参数化测试来测试时间增量与 numpy 的 timedelta64 对象的加法操作
    @pytest.mark.parametrize("op", [operator.add, ops.radd])
    def test_td_add_timedelta64(self, op):
        # 创建一个时间增量对象，表示 10 天
        td = Timedelta(10, unit="D")
        # 使用 op 参数（加法操作符或其反向版本）对时间增量与 numpy 的 timedelta64 对象进行加法操作
        result = op(td, np.timedelta64(-4, "D"))
        # 断言结果是 Timedelta 类型，并且值为 6 天
        assert isinstance(result, Timedelta)
        assert result == Timedelta(days=6)

    # 使用参数化测试来测试时间增量与偏移量对象的加法操作
    @pytest.mark.parametrize("op", [operator.add, ops.radd])
    def test_td_add_offset(self, op):
        # 创建一个时间增量对象，表示 10 天
        td = Timedelta(10, unit="D")

        # 使用 op 参数（加法操作符或其反向版本）对时间增量与偏移量对象（offsets.Hour(6) 表示6小时的偏移量）进行加法操作
        result = op(td, offsets.Hour(6))
        # 断言结果是 Timedelta 类型，并且值为 10 天 6 小时
        assert isinstance(result, Timedelta)
        assert result == Timedelta(days=10, hours=6)

    # 测试时间增量与自身的减法操作
    def test_td_sub_td(self):
        # 创建一个时间增量对象，表示 10 天
        td = Timedelta(10, unit="D")
        # 创建一个预期的时间增量对象，表示 0 纳秒
        expected = Timedelta(0, unit="ns")
        # 进行时间增量与自身的减法操作
        result = td - td
        # 断言结果是 Timedelta 类型，并且值为预期的 0 纳秒
        assert isinstance(result, Timedelta)
        assert result == expected

    # 测试时间增量与 Python 的 timedelta 对象的减法操作
    def test_td_sub_pytimedelta(self):
        # 创建一个时间增量对象，表示 10 天
        td = Timedelta(10, unit="D")
        # 创建一个预期的时间增量对象，表示 0 纳秒
        expected = Timedelta(0, unit="ns")

        # 进行时间增量与 Python 的 timedelta 对象的减法操作
        result = td - td.to_pytimedelta()
        # 断言结果是 Timedelta 类型，并且值为预期的 0 纳秒
        assert isinstance(result, Timedelta)
        assert result == expected

        # 进行 Python 的 timedelta 对象与时间增量的减法操作
        result = td.to_pytimedelta() - td
        # 断言结果是 Timedelta 类型，并且值为预期的 0 纳秒
        assert isinstance(result, Timedelta)
        assert result == expected

    # 测试时间增量与 numpy 的 timedelta64 对象的减法操作
    def test_td_sub_timedelta64(self):
        # 创建一个时间增量对象，表示 10 天
        td = Timedelta(10, unit="D")
        # 创建一个预期的时间增量对象，表示 0 纳秒
        expected = Timedelta(0, unit="ns")

        # 进行时间增量与 numpy 的 timedelta64 对象的减法操作
        result = td - td.to_timedelta64()
        # 断言结果是 Timedelta 类型，并且值为预期的 0 纳秒
        assert isinstance(result, Timedelta)
        assert result == expected

        # 进行 numpy 的 timedelta64 对象与时间增量的减法操作
        result = td.to_timedelta64() - td
        # 断言结果是 Timedelta 类型，并且值为预期的 0 纳秒
        assert isinstance(result, Timedelta)
        assert result == expected

    # 测试时间增量与 NaT 的减法操作
    def test_td_sub_nat(self):
        # 在这个上下文中，pd.NaT 被视为类似时间增量的对象
        td = Timedelta(10, unit="D")
        # 进行时间增量与 NaT 的减法操作
        result = td - NaT
        # 断言结果是 NaT
        assert result is NaT

    # 测试时间增量与 numpy 的 timedelta64 对象 NaT 的减法操作
    def test_td_sub_td64_nat(self):
        # 创建一个时间增量对象，表示 10 天
        td = Timedelta(10, unit="D")
        # 创建一个 numpy 的 timedelta64 对象，表示 NaT
        td_nat = np.timedelta64("NaT")

        # 进行时间增量与 numpy 的 timedelta64 对象 NaT 的减法操作
        result = td - td_nat
        # 断言结果是 NaT
        assert result is NaT

        # 进行 numpy 的 timedelta64 对象 NaT 与时间增量的减法操作
        result = td_nat - td
        # 断言结果是 NaT
        assert result is NaT
    # 测试 Timedelta 对象的减法操作，减去一个小时的时间偏移量
    def test_td_sub_offset(self):
        td = Timedelta(10, unit="D")
        result = td - offsets.Hour(1)
        # 断言结果是 Timedelta 类型
        assert isinstance(result, Timedelta)
        # 断言结果等于减去 1 小时后的 Timedelta 对象，相当于 239 小时
        assert result == Timedelta(239, unit="h")

    # 测试 Timedelta 对象与各种类型的数字相加和相减，预期会抛出 TypeError 异常
    def test_td_add_sub_numeric_raises(self):
        td = Timedelta(10, unit="D")
        msg = "unsupported operand type"
        for other in [2, 2.0, np.int64(2), np.float64(2)]:
            # 测试相加操作是否会引发 TypeError 异常，匹配异常信息
            with pytest.raises(TypeError, match=msg):
                td + other
            # 测试反向相加操作是否会引发 TypeError 异常，匹配异常信息
            with pytest.raises(TypeError, match=msg):
                other + td
            # 测试相减操作是否会引发 TypeError 异常，匹配异常信息
            with pytest.raises(TypeError, match=msg):
                td - other
            # 测试反向相减操作是否会引发 TypeError 异常，匹配异常信息
            with pytest.raises(TypeError, match=msg):
                other - td

    # 测试 Timedelta 对象与整型 NumPy 数组的加法和减法操作，预期会抛出 TypeError 异常
    def test_td_add_sub_int_ndarray(self):
        td = Timedelta("1 day")
        other = np.array([1])

        msg = r"unsupported operand type\(s\) for \+: 'Timedelta' and 'int'"
        # 测试 Timedelta 对象与整型 NumPy 数组相加操作是否会引发 TypeError 异常，匹配异常信息
        with pytest.raises(TypeError, match=msg):
            td + np.array([1])

        msg = "|".join(
            [
                (
                    r"unsupported operand type\(s\) for \+: 'numpy.ndarray' "
                    "and 'Timedelta'"
                ),
                # 这条消息说明“请不要依赖此错误；它可能不会在所有 Python 实现中给出”
                "Concatenation operation is not implemented for NumPy arrays",
            ]
        )
        # 测试整型 NumPy 数组与 Timedelta 对象相加操作是否会引发 TypeError 异常，匹配异常信息
        with pytest.raises(TypeError, match=msg):
            other + td
        msg = r"unsupported operand type\(s\) for -: 'Timedelta' and 'int'"
        # 测试 Timedelta 对象与整型 NumPy 数组相减操作是否会引发 TypeError 异常，匹配异常信息
        with pytest.raises(TypeError, match=msg):
            td - other
        msg = r"unsupported operand type\(s\) for -: 'numpy.ndarray' and 'Timedelta'"
        # 测试整型 NumPy 数组与 Timedelta 对象相减操作是否会引发 TypeError 异常，匹配异常信息
        with pytest.raises(TypeError, match=msg):
            other - td

    # 测试 Timedelta 对象右向减法操作，从 NaT（Not a Time）减去一个 Timedelta 对象
    def test_td_rsub_nat(self):
        td = Timedelta(10, unit="D")
        result = NaT - td
        # 断言结果是 NaT（Not a Time）
        assert result is NaT

        result = np.datetime64("NaT") - td
        # 断言结果是 NaT（Not a Time）
        assert result is NaT

    # 测试 Timedelta 对象右向减法操作，减去一个时间偏移量对象
    def test_td_rsub_offset(self):
        result = offsets.Hour(1) - Timedelta(10, unit="D")
        # 断言结果是 Timedelta 类型
        assert isinstance(result, Timedelta)
        # 断言结果等于减去 10 天后的 Timedelta 对象，相当于 -239 小时
        assert result == Timedelta(-239, unit="h")

    # 测试 Timedelta 对象与类似时间间隔的对象（Timedelta-like object）和 dtype 数组的减法操作
    def test_td_sub_timedeltalike_object_dtype_array(self):
        # GH#21980
        arr = np.array([Timestamp("20130101 9:01"), Timestamp("20121230 9:02")])
        exp = np.array([Timestamp("20121231 9:01"), Timestamp("20121229 9:02")])
        res = arr - Timedelta("1D")
        # 使用 pytest 断言库比较 NumPy 数组 res 和 exp 是否相等
        tm.assert_numpy_array_equal(res, exp)
    def test_td_sub_mixed_most_timedeltalike_object_dtype_array(self):
        # 测试函数：对混合数据类型的时间增量数组进行减法操作
        # GH#21980：GitHub issue编号
        now = Timestamp("2021-11-09 09:54:00")  # 当前时间戳设定为2021年11月9日 09:54:00
        arr = np.array([now, Timedelta("1D"), np.timedelta64(2, "h")])  # 创建包含时间戳和时间增量的NumPy数组
        exp = np.array(
            [
                now - Timedelta("1D"),  # 计算当前时间减去一天的时间戳
                Timedelta("0D"),  # 时间增量为0天
                np.timedelta64(2, "h") - Timedelta("1D"),  # 计算2小时减去一天的时间增量
            ]
        )
        res = arr - Timedelta("1D")  # 对数组中的每个元素减去一天的时间增量
        tm.assert_numpy_array_equal(res, exp)  # 断言：验证结果数组与期望数组是否相等

    def test_td_rsub_mixed_most_timedeltalike_object_dtype_array(self):
        # 测试函数：对混合数据类型的时间增量数组进行反向减法操作
        # GH#21980：GitHub issue编号
        now = Timestamp("2021-11-09 09:54:00")  # 当前时间戳设定为2021年11月9日 09:54:00
        arr = np.array([now, Timedelta("1D"), np.timedelta64(2, "h")])  # 创建包含时间戳和时间增量的NumPy数组
        msg = r"unsupported operand type\(s\) for \-: 'Timedelta' and 'Timestamp'"
        with pytest.raises(TypeError, match=msg):
            Timedelta("1D") - arr  # 尝试执行时间增量减去数组的操作，预期引发类型错误异常

    @pytest.mark.parametrize("op", [operator.add, ops.radd])
    def test_td_add_timedeltalike_object_dtype_array(self, op):
        # 测试函数：对时间增量对象数据类型数组进行加法操作
        # GH#21980：GitHub issue编号
        arr = np.array([Timestamp("20130101 9:01"), Timestamp("20121230 9:02")])  # 创建包含时间戳的NumPy数组
        exp = np.array([Timestamp("20130102 9:01"), Timestamp("20121231 9:02")])  # 预期的结果数组
        res = op(arr, Timedelta("1D"))  # 使用给定的加法操作符对数组中的每个元素加上一天的时间增量
        tm.assert_numpy_array_equal(res, exp)  # 断言：验证结果数组与期望数组是否相等

    @pytest.mark.parametrize("op", [operator.add, ops.radd])
    def test_td_add_mixed_timedeltalike_object_dtype_array(self, op):
        # 测试函数：对混合数据类型的时间增量对象数据类型数组进行加法操作
        # GH#21980：GitHub issue编号
        now = Timestamp("2021-11-09 09:54:00")  # 当前时间戳设定为2021年11月9日 09:54:00
        arr = np.array([now, Timedelta("1D")])  # 创建包含时间戳和一天时间增量的NumPy数组
        exp = np.array([now + Timedelta("1D"), Timedelta("2D")])  # 预期的结果数组
        res = op(arr, Timedelta("1D"))  # 使用给定的加法操作符对数组中的每个元素加上一天的时间增量
        tm.assert_numpy_array_equal(res, exp)  # 断言：验证结果数组与期望数组是否相等

    def test_td_add_sub_td64_ndarray(self):
        # 测试函数：对Timedelta对象和td64类型的NumPy数组进行加减法操作
        td = Timedelta("1 day")  # 创建一天的时间增量对象

        other = np.array([td.to_timedelta64()])  # 将时间增量对象转换为td64类型的NumPy数组
        expected = np.array([Timedelta("2 Days").to_timedelta64()])  # 预期的结果数组

        result = td + other  # 对td和other数组执行加法操作
        tm.assert_numpy_array_equal(result, expected)  # 断言：验证结果数组与期望数组是否相等
        result = other + td  # 对other数组和td执行加法操作
        tm.assert_numpy_array_equal(result, expected)  # 断言：验证结果数组与期望数组是否相等

        result = td - other  # 对td和other数组执行减法操作
        tm.assert_numpy_array_equal(result, expected * 0)  # 断言：验证结果数组是否与期望的零数组相等
        result = other - td  # 对other数组和td执行减法操作
        tm.assert_numpy_array_equal(result, expected * 0)  # 断言：验证结果数组是否与期望的零数组相等

    def test_td_add_sub_dt64_ndarray(self):
        # 测试函数：对Timedelta对象和dt64类型的NumPy数组进行加减法操作
        td = Timedelta("1 day")  # 创建一天的时间增量对象
        other = np.array(["2000-01-01"], dtype="M8[ns]")  # 创建包含日期字符串的dt64类型的NumPy数组

        expected = np.array(["2000-01-02"], dtype="M8[ns]")  # 预期的结果数组
        tm.assert_numpy_array_equal(td + other, expected)  # 断言：验证结果数组与期望数组是否相等
        tm.assert_numpy_array_equal(other + td, expected)  # 断言：验证结果数组与期望数组是否相等

        expected = np.array(["1999-12-31"], dtype="M8[ns]")  # 预期的结果数组
        tm.assert_numpy_array_equal(-td + other, expected)  # 断言：验证结果数组与期望数组是否相等
        tm.assert_numpy_array_equal(other - td, expected)  # 断言：验证结果数组与期望数组是否相等
    # 定义测试方法，用于测试 Timedelta 对象与 NumPy 数组的加减运算
    def test_td_add_sub_ndarray_0d(self):
        # 创建 Timedelta 对象 td，表示一天的时间间隔
        td = Timedelta("1 day")
        # 将 td 的 asm8 属性转换为 NumPy 数组，并赋给 other 变量
        other = np.array(td.asm8)

        # 测试 Timedelta 对象 td 与 NumPy 数组 other 的加法运算
        result = td + other
        # 断言结果 result 是 Timedelta 类型
        assert isinstance(result, Timedelta)
        # 断言结果 result 等于 td 乘以 2
        assert result == 2 * td

        # 测试 NumPy 数组 other 与 Timedelta 对象 td 的加法运算（顺序颠倒）
        result = other + td
        # 断言结果 result 是 Timedelta 类型
        assert isinstance(result, Timedelta)
        # 断言结果 result 等于 td 乘以 2
        assert result == 2 * td

        # 测试 NumPy 数组 other 与 Timedelta 对象 td 的减法运算
        result = other - td
        # 断言结果 result 是 Timedelta 类型
        assert isinstance(result, Timedelta)
        # 断言结果 result 等于 td 乘以 0
        assert result == 0 * td

        # 测试 Timedelta 对象 td 与 NumPy 数组 other 的减法运算（顺序颠倒）
        result = td - other
        # 断言结果 result 是 Timedelta 类型
        assert isinstance(result, Timedelta)
        # 断言结果 result 等于 td 乘以 0
        assert result == 0 * td
class TestTimedeltaMultiplicationDivision:
    """
    Tests for Timedelta methods:

        __mul__, __rmul__,
        __div__, __rdiv__,
        __truediv__, __rtruediv__,
        __floordiv__, __rfloordiv__,
        __mod__, __rmod__,
        __divmod__, __rdivmod__
    """

    # ---------------------------------------------------------------
    # Timedelta.__mul__, __rmul__

    @pytest.mark.parametrize(
        "td_nat", [NaT, np.timedelta64("NaT", "ns"), np.timedelta64("NaT")]
    )
    @pytest.mark.parametrize("op", [operator.mul, ops.rmul])
    def test_td_mul_nat(self, op, td_nat):
        # GH#19819
        # 创建一个 Timedelta 对象，表示 10 天
        td = Timedelta(10, unit="D")
        # 指定类型字符串，用于错误消息匹配
        typs = "|".join(["numpy.timedelta64", "NaTType", "Timedelta"])
        msg = "|".join(
            [
                rf"unsupported operand type\(s\) for \*: '{typs}' and '{typs}'",
                r"ufunc '?multiply'? cannot use operands with types",
            ]
        )
        # 确保在执行操作时抛出预期的 TypeError 异常，并匹配错误消息
        with pytest.raises(TypeError, match=msg):
            op(td, td_nat)

    @pytest.mark.parametrize("nan", [np.nan, np.float64("NaN"), float("nan")])
    @pytest.mark.parametrize("op", [operator.mul, ops.rmul])
    def test_td_mul_nan(self, op, nan):
        # np.float64('NaN') 具有 'dtype' 属性，避免将其误认为是数组
        td = Timedelta(10, unit="D")
        # 执行操作，期望结果为 NaT
        result = op(td, nan)
        assert result is NaT

    @pytest.mark.parametrize("op", [operator.mul, ops.rmul])
    def test_td_mul_scalar(self, op):
        # GH#19738
        # 创建一个 Timedelta 对象，表示 3 分钟
        td = Timedelta(minutes=3)

        # 测试乘法运算
        result = op(td, 2)
        assert result == Timedelta(minutes=6)

        result = op(td, 1.5)
        assert result == Timedelta(minutes=4, seconds=30)

        assert op(td, np.nan) is NaT

        # 测试负数乘法
        assert op(-1, td)._value == -1 * td._value
        assert op(-1.0, td)._value == -1.0 * td._value

        msg = "unsupported operand type"
        with pytest.raises(TypeError, match=msg):
            # timedelta * datetime is gibberish
            op(td, Timestamp(2016, 1, 2))

        with pytest.raises(TypeError, match=msg):
            # invalid multiply with another timedelta
            op(td, td)

    def test_td_mul_numeric_ndarray(self):
        # 创建一个 Timedelta 对象，表示 1 天
        td = Timedelta("1 day")
        other = np.array([2])
        expected = np.array([Timedelta("2 Days").to_timedelta64()])

        # 测试 Timedelta 与数值数组的乘法运算
        result = td * other
        tm.assert_numpy_array_equal(result, expected)

        result = other * td
        tm.assert_numpy_array_equal(result, expected)

    def test_td_mul_numeric_ndarray_0d(self):
        # 创建一个 Timedelta 对象，表示 1 天
        td = Timedelta("1 day")
        other = np.array(2, dtype=np.int64)
        assert other.ndim == 0
        expected = Timedelta("2 days")

        # 测试 Timedelta 与零维数值数组的乘法运算
        res = td * other
        assert type(res) is Timedelta
        assert res == expected

        res = other * td
        assert type(res) is Timedelta
        assert res == expected
    # 定义一个测试函数，用于测试Timedelta对象与非有效ndarray的乘法操作是否引发TypeError异常
    def test_td_mul_td64_ndarray_invalid(self):
        # 创建一个Timedelta对象表示1天
        td = Timedelta("1 day")
        # 创建一个包含Timedelta对象"2 Days"对应的timedelta64数组
        other = np.array([Timedelta("2 Days").to_timedelta64()])

        # 定义错误消息，用于匹配pytest.raises抛出的TypeError异常
        msg = (
            "ufunc '?multiply'? cannot use operands with types "
            rf"dtype\('{tm.ENDIAN}m8\[ns\]'\) and dtype\('{tm.ENDIAN}m8\[ns\]'\)"
        )

        # 断言乘法操作td * other会引发TypeError异常，并且异常消息与msg匹配
        with pytest.raises(TypeError, match=msg):
            td * other
        # 断言乘法操作other * td也会引发TypeError异常，并且异常消息与msg匹配
        with pytest.raises(TypeError, match=msg):
            other * td

    # ---------------------------------------------------------------
    # Timedelta.__div__, __truediv__

    # 定义一个测试函数，用于测试Timedelta对象与timedelta-like标量的除法操作
    def test_td_div_timedeltalike_scalar(self):
        # 创建一个Timedelta对象表示10天
        td = Timedelta(10, unit="D")

        # 进行除法操作，计算结果与预期结果进行比较
        result = td / offsets.Hour(1)
        assert result == 240

        assert td / td == 1  # 断言Timedelta对象与自身的除法结果为1
        assert td / np.timedelta64(60, "h") == 4  # 断言Timedelta对象与60小时的除法结果为4

        # 断言Timedelta对象与NaT的除法结果为NaN
        assert np.isnan(td / NaT)

    # 定义一个测试函数，用于测试Timedelta对象与非纳秒单位的timedelta64的除法操作
    def test_td_div_td64_non_nano(self):
        # 创建一个Timedelta对象表示"1 days 2 hours 3 ns"
        td = Timedelta("1 days 2 hours 3 ns")

        # 进行truediv操作，计算结果与预期结果进行比较
        result = td / np.timedelta64(1, "D")
        assert result == td._value / (86400 * 10**9)
        result = td / np.timedelta64(1, "s")
        assert result == td._value / 10**9
        result = td / np.timedelta64(1, "ns")
        assert result == td._value

        # 进行floordiv操作，计算结果与预期结果进行比较
        result = td // np.timedelta64(1, "D")
        assert result == 1
        result = td // np.timedelta64(1, "s")
        assert result == 93600
        result = td // np.timedelta64(1, "ns")
        assert result == td._value

    # 定义一个测试函数，用于测试Timedelta对象与数值标量的除法操作
    def test_td_div_numeric_scalar(self):
        # 创建一个Timedelta对象表示10天
        td = Timedelta(10, unit="D")

        # 进行除法操作，计算结果与预期结果进行比较
        result = td / 2
        assert isinstance(result, Timedelta)
        assert result == Timedelta(days=5)

        result = td / 5
        assert isinstance(result, Timedelta)
        assert result == Timedelta(days=2)

    # 使用pytest.mark.parametrize标记的参数化测试，测试Timedelta对象与NaN的除法操作
    @pytest.mark.parametrize(
        "nan",
        [
            np.nan,
            np.float64("NaN"),
            float("nan"),
        ],
    )
    def test_td_div_nan(self, nan):
        # 创建一个Timedelta对象表示10天
        td = Timedelta(10, unit="D")

        # 进行除法操作，结果应为NaT
        result = td / nan
        assert result is NaT

        # 进行floordiv操作，结果应为NaT
        result = td // nan
        assert result is NaT

    # 定义一个测试函数，用于测试Timedelta对象与timedelta64数组的除法操作
    def test_td_div_td64_ndarray(self):
        # 创建一个Timedelta对象表示1天
        td = Timedelta("1 day")

        # 创建一个包含Timedelta对象"2 Days"对应的timedelta64数组
        other = np.array([Timedelta("2 Days").to_timedelta64()])
        # 创建一个预期结果数组
        expected = np.array([0.5])

        # 进行除法操作，结果与预期结果进行比较
        result = td / other
        tm.assert_numpy_array_equal(result, expected)

        # 进行除法操作，结果与预期结果进行比较
        result = other / td
        tm.assert_numpy_array_equal(result, expected * 4)

    # 定义一个测试函数，用于测试Timedelta对象与0维ndarray的除法操作
    def test_td_div_ndarray_0d(self):
        # 创建一个Timedelta对象表示1天
        td = Timedelta("1 day")

        # 创建一个包含数值1的0维ndarray
        other = np.array(1)
        # 进行除法操作，计算结果与预期结果进行比较
        res = td / other
        assert isinstance(res, Timedelta)
        assert res == td

    # ---------------------------------------------------------------
    # Timedelta.__rdiv__
    # 定义测试函数，用于测试Timedelta对象的右除运算方法
    def test_td_rdiv_timedeltalike_scalar(self):
        # GH#19738
        # 创建一个Timedelta对象，表示10天
        td = Timedelta(10, unit="D")
        # 计算一个时间偏移为1小时除以td的结果
        result = offsets.Hour(1) / td
        # 断言计算结果应该等于1除以240.0，即0.004166666666666667
        assert result == 1 / 240.0

        # 断言numpy中的60小时时间偏移除以td的结果应该等于0.25
        assert np.timedelta64(60, "h") / td == 0.25

    # 定义测试函数，测试Timedelta对象的右除运算方法处理None的情况
    def test_td_rdiv_na_scalar(self):
        # GH#31869 None会被转换为NaT
        # 创建一个Timedelta对象，表示10天
        td = Timedelta(10, unit="D")

        # 计算NaT除以td的结果，预期结果应该是NaN
        result = NaT / td
        assert np.isnan(result)

        # 计算None除以td的结果，预期结果应该是NaN
        result = None / td
        assert np.isnan(result)

        # 计算numpy中的NaT除以td的结果，预期结果应该是NaN
        result = np.timedelta64("NaT") / td
        assert np.isnan(result)

        # 使用pytest断言，计算numpy中的NaT的datetime64类型除以td会引发TypeError异常，异常信息中应包含特定字符串
        msg = r"unsupported operand type\(s\) for /: 'numpy.datetime64' and 'Timedelta'"
        with pytest.raises(TypeError, match=msg):
            np.datetime64("NaT") / td

        # 使用pytest断言，计算NaN除以td会引发TypeError异常，异常信息中应包含特定字符串
        msg = r"unsupported operand type\(s\) for /: 'float' and 'Timedelta'"
        with pytest.raises(TypeError, match=msg):
            np.nan / td

    # 定义测试函数，测试Timedelta对象的右除运算方法处理numpy数组的情况
    def test_td_rdiv_ndarray(self):
        # 创建一个Timedelta对象，表示10天
        td = Timedelta(10, unit="D")

        # 创建一个包含td对象的numpy数组，数据类型为object
        arr = np.array([td], dtype=object)
        # 计算数组arr中的每个元素除以td的结果
        result = arr / td
        # 创建一个预期的numpy数组，包含计算结果1，数据类型为np.float64
        expected = np.array([1], dtype=np.float64)
        # 使用tm.assert_numpy_array_equal方法断言结果数组与预期数组相等
        tm.assert_numpy_array_equal(result, expected)

        # 创建一个包含None的numpy数组
        arr = np.array([None])
        # 计算数组arr中的每个元素除以td的结果，预期结果应该是包含NaN的数组
        result = arr / td
        expected = np.array([np.nan])
        tm.assert_numpy_array_equal(result, expected)

        # 创建一个包含NaN的对象数组
        arr = np.array([np.nan], dtype=object)
        # 使用pytest断言，计算数组arr中的元素除以td会引发TypeError异常，异常信息中应包含特定字符串
        msg = r"unsupported operand type\(s\) for /: 'float' and 'Timedelta'"
        with pytest.raises(TypeError, match=msg):
            arr / td

        # 创建一个包含NaN的np.float64类型数组
        arr = np.array([np.nan], dtype=np.float64)
        # 使用pytest断言，计算数组arr中的元素除以td会引发TypeError异常，异常信息中应包含特定字符串
        msg = "cannot use operands with types dtype"
        with pytest.raises(TypeError, match=msg):
            arr / td

    # 定义测试函数，测试Timedelta对象的右除运算方法处理0维数组的情况
    def test_td_rdiv_ndarray_0d(self):
        # 创建一个Timedelta对象，表示10天
        td = Timedelta(10, unit="D")

        # 创建一个包含td的0维数组，并将其转换为标量
        arr = np.array(td.asm8)
        # 断言标量arr除以td的结果应该等于1
        assert arr / td == 1

    # ---------------------------------------------------------------
    # Timedelta.__floordiv__

    # 定义测试函数，测试Timedelta对象的整除运算方法处理时间偏移量的情况
    def test_td_floordiv_timedeltalike_scalar(self):
        # GH#18846
        # 创建一个Timedelta对象，表示3小时4分钟
        td = Timedelta(hours=3, minutes=4)
        # 创建一个时间偏移量对象，表示3小时3分钟
        scalar = Timedelta(hours=3, minutes=3)

        # 断言td除以scalar的整数部分应该等于1
        assert td // scalar == 1
        # 断言-td除以scalar转换为Python timedelta后的整数部分应该等于-2
        assert -td // scalar.to_pytimedelta() == -2
        # 断言2倍td除以scalar转换为timedelta64后的整数部分应该等于2
        assert (2 * td) // scalar.to_timedelta64() == 2

    # 定义测试函数，测试Timedelta对象的整除运算方法处理空值的情况
    def test_td_floordiv_null_scalar(self):
        # GH#18846
        # 创建一个Timedelta对象，表示3小时4分钟
        td = Timedelta(hours=3, minutes=4)

        # 断言td除以NaN应该返回NaT
        assert td // np.nan is NaT
        # 断言td除以NaT应该返回NaN
        assert np.isnan(td // NaT)
        # 断言td除以NaT类型的timedelta64应该返回NaN
        assert np.isnan(td // np.timedelta64("NaT"))

    # 定义测试函数，测试Timedelta对象的整除运算方法处理时间偏移量的情况
    def test_td_floordiv_offsets(self):
        # GH#19738
        # 创建一个Timedelta对象，表示3小时4分钟
        td = Timedelta(hours=3, minutes=4)
        # 断言td除以1小时的时间偏移量应该返回3
        assert td // offsets.Hour(1) == 3
        # 断言td除以2分钟的时间偏移量应该返回92
        assert td // offsets.Minute(2) == 92
    def test_td_floordiv_invalid_scalar(self):
        # GH#18846
        # 创建一个时间增量对象，表示3小时4分钟
        td = Timedelta(hours=3, minutes=4)

        # 定义错误消息，用于匹配异常信息
        msg = "|".join(
            [
                r"Invalid dtype datetime64\[D\] for __floordiv__",
                "'dtype' is an invalid keyword argument for this function",
                "this function got an unexpected keyword argument 'dtype'",
                r"ufunc '?floor_divide'? cannot use operands with types",
            ]
        )
        # 断言当执行 // 操作时会抛出 TypeError 异常，并且异常消息需要匹配定义的 msg
        with pytest.raises(TypeError, match=msg):
            td // np.datetime64("2016-01-01", dtype="datetime64[us]")

    def test_td_floordiv_numeric_scalar(self):
        # GH#18846
        # 创建一个时间增量对象，表示3小时4分钟
        td = Timedelta(hours=3, minutes=4)

        # 预期的结果，表示1小时32分钟
        expected = Timedelta(hours=1, minutes=32)
        # 断言时间增量对象 td 除以不同类型的标量值得到的结果符合预期
        assert td // 2 == expected
        assert td // 2.0 == expected
        assert td // np.float64(2.0) == expected
        assert td // np.int32(2.0) == expected
        assert td // np.uint8(2.0) == expected

    def test_td_floordiv_timedeltalike_array(self):
        # GH#18846
        # 创建一个时间增量对象，表示3小时4分钟
        td = Timedelta(hours=3, minutes=4)
        # 创建一个时间增量对象，表示3小时3分钟
        scalar = Timedelta(hours=3, minutes=3)

        # 对于类数组的其他时间增量对象，验证 td 除以 np.array(scalar.to_timedelta64()) 的结果为1
        assert td // np.array(scalar.to_timedelta64()) == 1

        # 验证 (3 * td) // np.array([scalar.to_timedelta64()]) 的结果与预期结果一致
        res = (3 * td) // np.array([scalar.to_timedelta64()])
        expected = np.array([3], dtype=np.int64)
        tm.assert_numpy_array_equal(res, expected)

        # 验证 (10 * td) // np.array([scalar.to_timedelta64(), np.timedelta64("NaT")]) 的结果与预期结果一致
        res = (10 * td) // np.array([scalar.to_timedelta64(), np.timedelta64("NaT")])
        expected = np.array([10, np.nan])
        tm.assert_numpy_array_equal(res, expected)

    def test_td_floordiv_numeric_series(self):
        # GH#18846
        # 创建一个时间增量对象，表示3小时4分钟
        td = Timedelta(hours=3, minutes=4)
        # 创建一个包含一个整数的 Pandas Series 对象
        ser = pd.Series([1], dtype=np.int64)
        # 执行 td 除以 ser 的操作，验证结果的数据类型为 'm' (时间增量)
        res = td // ser
        assert res.dtype.kind == "m"

    # ---------------------------------------------------------------
    # Timedelta.__rfloordiv__

    def test_td_rfloordiv_timedeltalike_scalar(self):
        # GH#18846
        # 创建一个时间增量对象，表示3小时3分钟
        td = Timedelta(hours=3, minutes=3)
        # 创建一个时间增量对象，表示3小时4分钟
        scalar = Timedelta(hours=3, minutes=4)

        # 验证 td.__rfloordiv__(scalar) 的结果为1
        assert td.__rfloordiv__(scalar) == 1
        # 验证 (-td).__rfloordiv__(scalar.to_pytimedelta()) 的结果为-2
        assert (-td).__rfloordiv__(scalar.to_pytimedelta()) == -2
        # 验证 (2 * td).__rfloordiv__(scalar.to_timedelta64()) 的结果为0
        assert (2 * td).__rfloordiv__(scalar.to_timedelta64()) == 0

    def test_td_rfloordiv_null_scalar(self):
        # GH#18846
        # 创建一个时间增量对象，表示3小时3分钟
        td = Timedelta(hours=3, minutes=3)

        # 验证 td.__rfloordiv__(NaT) 的结果为 NaN
        assert np.isnan(td.__rfloordiv__(NaT))
        # 验证 td.__rfloordiv__(np.timedelta64("NaT")) 的结果为 NaN
        assert np.isnan(td.__rfloordiv__(np.timedelta64("NaT")))

    def test_td_rfloordiv_offsets(self):
        # GH#19738
        # 验证 offsets.Hour(1) 除以 Timedelta(minutes=25) 的结果为2
        assert offsets.Hour(1) // Timedelta(minutes=25) == 2
    # 定义测试方法：测试在右除时处理无效标量情况
    def test_td_rfloordiv_invalid_scalar(self):
        # GH#18846：GitHub问题跟踪编号
        td = Timedelta(hours=3, minutes=3)  # 创建一个时间增量对象，表示3小时3分钟

        dt64 = np.datetime64("2016-01-01", "us")  # 创建一个微秒精度的numpy.datetime64对象

        # 断言右除操作不支持
        assert td.__rfloordiv__(dt64) is NotImplemented

        # 定义错误消息
        msg = (
            r"unsupported operand type\(s\) for //: 'numpy.datetime64' and 'Timedelta'"
        )
        # 使用pytest断言检测是否引发了预期的TypeError异常，并匹配指定消息
        with pytest.raises(TypeError, match=msg):
            dt64 // td

    # 定义测试方法：测试在右除时处理数值标量情况
    def test_td_rfloordiv_numeric_scalar(self):
        # GH#18846：GitHub问题跟踪编号
        td = Timedelta(hours=3, minutes=3)  # 创建一个时间增量对象，表示3小时3分钟

        # 检查右除操作是否不支持各种数值类型
        assert td.__rfloordiv__(np.nan) is NotImplemented
        assert td.__rfloordiv__(3.5) is NotImplemented
        assert td.__rfloordiv__(2) is NotImplemented
        assert td.__rfloordiv__(np.float64(2.0)) is NotImplemented
        assert td.__rfloordiv__(np.uint8(9)) is NotImplemented
        assert td.__rfloordiv__(np.int32(2.0)) is NotImplemented

        # 定义错误消息模式
        msg = r"unsupported operand type\(s\) for //: '.*' and 'Timedelta'"
        # 使用pytest断言检测是否引发了预期的TypeError异常，并匹配指定消息
        with pytest.raises(TypeError, match=msg):
            np.float64(2.0) // td
        with pytest.raises(TypeError, match=msg):
            np.uint8(9) // td
        with pytest.raises(TypeError, match=msg):
            # 对于已弃用的问题GH#19761，强制执行GH#29797
            np.int32(2.0) // td

    # 定义测试方法：测试在右除时处理类似时间增量的数组情况
    def test_td_rfloordiv_timedeltalike_array(self):
        # GH#18846：GitHub问题跟踪编号
        td = Timedelta(hours=3, minutes=3)  # 创建一个时间增量对象，表示3小时3分钟
        scalar = Timedelta(hours=3, minutes=4)  # 创建另一个时间增量对象，表示3小时4分钟

        # 对数组样式的其他对象执行右除操作
        assert td.__rfloordiv__(np.array(scalar.to_timedelta64())) == 1

        # 进行预期结果的数组比较
        res = td.__rfloordiv__(np.array([(3 * scalar).to_timedelta64()]))
        expected = np.array([3], dtype=np.int64)
        tm.assert_numpy_array_equal(res, expected)

        # 创建包含时间增量和NaT值的numpy数组，并执行右除操作
        arr = np.array([(10 * scalar).to_timedelta64(), np.timedelta64("NaT")])
        res = td.__rfloordiv__(arr)
        expected = np.array([10, np.nan])
        tm.assert_numpy_array_equal(res, expected)

    # 定义测试方法：测试在右除时处理整数数组情况
    def test_td_rfloordiv_intarray(self):
        # 对于已弃用的问题GH#19761，强制执行GH#29797
        ints = np.array([1349654400, 1349740800, 1349827200, 1349913600]) * 10**9

        # 定义错误消息
        msg = "Invalid dtype"
        # 使用pytest断言检测是否引发了预期的TypeError异常，并匹配指定消息
        with pytest.raises(TypeError, match=msg):
            ints // Timedelta(1, unit="s")

    # 定义测试方法：测试在右除时处理数值序列情况
    def test_td_rfloordiv_numeric_series(self):
        # GH#18846：GitHub问题跟踪编号
        td = Timedelta(hours=3, minutes=3)  # 创建一个时间增量对象，表示3小时3分钟
        ser = pd.Series([1], dtype=np.int64)  # 创建一个包含一个整数的Pandas Series对象
        res = td.__rfloordiv__(ser)  # 尝试执行右除操作

        # 断言右除操作返回不支持
        assert res is NotImplemented

        # 定义错误消息
        msg = "Invalid dtype"
        # 使用pytest断言检测是否引发了预期的TypeError异常，并匹配指定消息
        with pytest.raises(TypeError, match=msg):
            # 对于已弃用的问题GH#19761，强制执行GH#29797
            ser // td

    # ----------------------------------------------------------------
    # Timedelta.__mod__, __rmod__
    # 定义测试函数，验证 Timedelta 对象的模运算特性
    def test_mod_timedeltalike(self):
        # GH#19365
        # 创建一个 Timedelta 对象，表示37小时
        td = Timedelta(hours=37)

        # 对 Timedelta 对象进行模运算，以小时为单位
        result = td % Timedelta(hours=6)
        assert isinstance(result, Timedelta)
        assert result == Timedelta(hours=1)

        # 对 Timedelta 对象进行模运算，以分钟为单位
        result = td % timedelta(minutes=60)
        assert isinstance(result, Timedelta)
        assert result == Timedelta(0)

        # 对 Timedelta 对象进行模运算，使用 NaT (Not a Time) 对象
        result = td % NaT
        assert result is NaT

    # 定义测试函数，验证 Timedelta 对象与 numpy timedelta64 的模运算
    def test_mod_timedelta64_nat(self):
        # GH#19365
        # 创建一个 Timedelta 对象，表示37小时
        td = Timedelta(hours=37)

        # 对 Timedelta 对象与指定的 NaT (Not a Time) 进行模运算
        result = td % np.timedelta64("NaT", "ns")
        assert result is NaT

    # 定义测试函数，验证 Timedelta 对象与 numpy timedelta64 的模运算
    def test_mod_timedelta64(self):
        # GH#19365
        # 创建一个 Timedelta 对象，表示37小时
        td = Timedelta(hours=37)

        # 对 Timedelta 对象与指定的2小时的 numpy timedelta64 进行模运算
        result = td % np.timedelta64(2, "h")
        assert isinstance(result, Timedelta)
        assert result == Timedelta(hours=1)

    # 定义测试函数，验证 Timedelta 对象与 pandas offsets 的模运算
    def test_mod_offset(self):
        # GH#19365
        # 创建一个 Timedelta 对象，表示37小时
        td = Timedelta(hours=37)

        # 对 Timedelta 对象与偏移量为5小时的 offsets.Hour 进行模运算
        result = td % offsets.Hour(5)
        assert isinstance(result, Timedelta)
        assert result == Timedelta(hours=2)

    # 定义测试函数，验证 Timedelta 对象与数值类型的模运算
    def test_mod_numeric(self):
        # GH#19365
        # 创建一个 Timedelta 对象，表示37小时
        td = Timedelta(hours=37)

        # 对 Timedelta 对象与整数2进行模运算
        result = td % 2
        assert isinstance(result, Timedelta)
        assert result == Timedelta(0)

        # 对 Timedelta 对象与浮点数1e12进行模运算
        result = td % 1e12
        assert isinstance(result, Timedelta)
        assert result == Timedelta(minutes=3, seconds=20)

        # 对 Timedelta 对象与整数1e12进行模运算
        result = td % int(1e12)
        assert isinstance(result, Timedelta)
        assert result == Timedelta(minutes=3, seconds=20)

    # 定义测试函数，验证 Timedelta 对象与非法操作数的模运算
    def test_mod_invalid(self):
        # GH#19365
        # 创建一个 Timedelta 对象，表示37小时
        td = Timedelta(hours=37)
        msg = "unsupported operand type"

        # 验证 Timedelta 对象与 Timestamp 对象进行模运算抛出异常
        with pytest.raises(TypeError, match=msg):
            td % Timestamp("2018-01-22")

        # 验证 Timedelta 对象与空列表进行模运算抛出异常
        with pytest.raises(TypeError, match=msg):
            td % []

    # 定义测试函数，验证 timedelta 对象与 Timedelta 对象的模运算
    def test_rmod_pytimedelta(self):
        # GH#19365
        # 创建一个 Timedelta 对象，表示3分钟
        td = Timedelta(minutes=3)

        # 对 Python 的 timedelta 对象与 Timedelta 对象进行模运算
        result = timedelta(minutes=4) % td
        assert isinstance(result, Timedelta)
        assert result == Timedelta(minutes=1)

    # 定义测试函数，验证 numpy timedelta64 与 Timedelta 对象的模运算
    def test_rmod_timedelta64(self):
        # GH#19365
        # 创建一个 Timedelta 对象，表示3分钟
        td = Timedelta(minutes=3)

        # 对 numpy timedelta64 对象与 Timedelta 对象进行模运算
        result = np.timedelta64(5, "m") % td
        assert isinstance(result, Timedelta)
        assert result == Timedelta(minutes=2)

    # 定义测试函数，验证非法操作数与 Timedelta 对象的模运算
    def test_rmod_invalid(self):
        # GH#19365
        # 创建一个 Timedelta 对象，表示3分钟
        td = Timedelta(minutes=3)
        msg = "unsupported operand"

        # 验证 Timestamp 对象与 Timedelta 对象进行模运算抛出异常
        with pytest.raises(TypeError, match=msg):
            Timestamp("2018-01-22") % td

        # 验证整数与 Timedelta 对象进行模运算抛出异常
        with pytest.raises(TypeError, match=msg):
            15 % td

        # 验证浮点数与 Timedelta 对象进行模运算抛出异常
        with pytest.raises(TypeError, match=msg):
            16.0 % td

        msg = "Invalid dtype int"
        # 验证 numpy 数组与 Timedelta 对象进行模运算抛出异常
        with pytest.raises(TypeError, match=msg):
            np.array([22, 24]) % td
    def test_divmod_numeric(self):
        # GH#19365
        # 创建一个时间增量对象，表示2天6小时
        td = Timedelta(days=2, hours=6)

        # 使用 divmod 函数计算 td 除以 53 * 3600 * 1e9 的结果
        result = divmod(td, 53 * 3600 * 1e9)
        # 断言结果的商为1纳秒的时间增量对象
        assert result[0] == Timedelta(1, unit="ns")
        # 断言结果的余数为时间增量对象
        assert isinstance(result[1], Timedelta)
        # 断言余数为1小时的时间增量对象
        assert result[1] == Timedelta(hours=1)

        # 断言结果非空
        assert result

        # 使用 divmod 函数计算 td 除以 NaN 的结果
        result = divmod(td, np.nan)
        # 断言结果的商为 NaT（Not a Time）
        assert result[0] is NaT
        # 断言结果的余数为 NaT
        assert result[1] is NaT

    def test_divmod(self):
        # GH#19365
        # 创建一个时间增量对象，表示2天6小时
        td = Timedelta(days=2, hours=6)

        # 使用 divmod 函数计算 td 除以 1天的时间增量对象的结果
        result = divmod(td, timedelta(days=1))
        # 断言结果的商为2
        assert result[0] == 2
        # 断言结果的余数为时间增量对象，表示6小时
        assert isinstance(result[1], Timedelta)
        assert result[1] == Timedelta(hours=6)

        # 使用 divmod 函数计算 td 除以 54 的结果
        result = divmod(td, 54)
        # 断言结果的商为1小时的时间增量对象
        assert result[0] == Timedelta(hours=1)
        # 断言结果的余数为时间增量对象，表示0纳秒
        assert isinstance(result[1], Timedelta)
        assert result[1] == Timedelta(0)

        # 使用 divmod 函数计算 td 除以 NaT 的结果
        result = divmod(td, NaT)
        # 断言结果的商为 NaN
        assert np.isnan(result[0])
        # 断言结果的余数为 NaT
        assert result[1] is NaT

    def test_divmod_offset(self):
        # GH#19365
        # 创建一个时间增量对象，表示2天6小时
        td = Timedelta(days=2, hours=6)

        # 使用 divmod 函数计算 td 除以 偏移量为-4小时 的结果
        result = divmod(td, offsets.Hour(-4))
        # 断言结果的商为-14
        assert result[0] == -14
        # 断言结果的余数为时间增量对象，表示-2小时
        assert isinstance(result[1], Timedelta)
        assert result[1] == Timedelta(hours=-2)

    def test_divmod_invalid(self):
        # GH#19365
        # 创建一个时间增量对象，表示2天6小时
        td = Timedelta(days=2, hours=6)

        # 准备一个错误信息字符串，用于异常断言
        msg = r"unsupported operand type\(s\) for //: 'Timedelta' and 'Timestamp'"
        
        # 使用 pytest 的异常断言，验证在除法操作中，Timedelta 对象与 Timestamp 对象会引发 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            divmod(td, Timestamp("2018-01-22"))

    def test_rdivmod_pytimedelta(self):
        # GH#19365
        # 使用 divmod 函数计算 2天6小时 除以 1天 的结果
        result = divmod(timedelta(days=2, hours=6), Timedelta(days=1))
        # 断言结果的商为2
        assert result[0] == 2
        # 断言结果的余数为时间增量对象，表示6小时
        assert isinstance(result[1], Timedelta)
        assert result[1] == Timedelta(hours=6)

    def test_rdivmod_offset(self):
        # 使用 divmod 函数计算 54小时 除以 偏移量为-4小时 的结果
        result = divmod(offsets.Hour(54), Timedelta(hours=-4))
        # 断言结果的商为-14
        assert result[0] == -14
        # 断言结果的余数为时间增量对象，表示-2小时
        assert isinstance(result[1], Timedelta)
        assert result[1] == Timedelta(hours=-2)

    def test_rdivmod_invalid(self):
        # GH#19365
        # 创建一个时间增量对象，表示3分钟
        td = Timedelta(minutes=3)
        # 准备一个错误信息字符串，用于异常断言
        msg = "unsupported operand type"

        # 使用 pytest 的异常断言，验证在除法操作中，Timestamp 对象与 Timedelta 对象会引发 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            divmod(Timestamp("2018-01-22"), td)

        # 使用 pytest 的异常断言，验证在除法操作中，整数 15 与 Timedelta 对象会引发 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            divmod(15, td)

        # 使用 pytest 的异常断言，验证在除法操作中，浮点数 16.0 与 Timedelta 对象会引发 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            divmod(16.0, td)

        # 准备一个错误信息字符串，用于异常断言
        msg = "Invalid dtype int"
        
        # 使用 pytest 的异常断言，验证在除法操作中，NumPy 整型数组与 Timedelta 对象会引发 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            divmod(np.array([22, 24]), td)
    # 定义一个测试方法，用于测试时间增量和类时间增量数组的操作
    def test_td_op_timedelta_timedeltalike_array(self, op, arr):
        # 将输入的数组转换为 NumPy 数组
        arr = np.array(arr)
        # 定义错误消息，用于检测是否会引发特定类型的 TypeError 异常
        msg = "unsupported operand type|cannot use operands with types"
        # 使用 pytest 框架来验证是否会抛出预期的 TypeError 异常，同时匹配指定的错误消息
        with pytest.raises(TypeError, match=msg):
            # 调用被测试的操作函数 op，并传入数组 arr 和一个时间增量 Timedelta("1D")
            op(arr, Timedelta("1D"))
class TestTimedeltaComparison:
    @pytest.mark.skip_ubsan
    def test_compare_pytimedelta_bounds(self):
        # GH#49021 don't overflow on comparison with very large pytimedeltas
        # 测试不会在与非常大的 pytimedelta 比较时溢出

        # 循环处理 "ns" 和 "us" 单位
        for unit in ["ns", "us"]:
            # 获取 Timedelta 的最大值，并按指定单位转换为最大值
            tdmax = Timedelta.max.as_unit(unit).max
            # 获取 Timedelta 的最小值，并按指定单位转换为最小值
            tdmin = Timedelta.min.as_unit(unit).min

            # 断言 Timedelta 的最大值小于 Python 标准库中 timedelta 的最大值
            assert tdmax < timedelta.max
            # 断言 Timedelta 的最大值小于或等于 Python 标准库中 timedelta 的最大值
            assert tdmax <= timedelta.max
            # 断言 Timedelta 的最大值不大于 Python 标准库中 timedelta 的最大值
            assert not tdmax > timedelta.max
            # 断言 Timedelta 的最大值不大于或等于 Python 标准库中 timedelta 的最大值
            assert not tdmax >= timedelta.max
            # 断言 Timedelta 的最大值不等于 Python 标准库中 timedelta 的最大值
            assert tdmax != timedelta.max
            # 断言 Timedelta 的最大值不等于 Python 标准库中 timedelta 的最大值
            assert not tdmax == timedelta.max

            # 断言 Timedelta 的最小值大于 Python 标准库中 timedelta 的最小值
            assert tdmin > timedelta.min
            # 断言 Timedelta 的最小值大于或等于 Python 标准库中 timedelta 的最小值
            assert tdmin >= timedelta.min
            # 断言 Timedelta 的最小值不小于 Python 标准库中 timedelta 的最小值
            assert not tdmin < timedelta.min
            # 断言 Timedelta 的最小值不小于或等于 Python 标准库中 timedelta 的最小值
            assert not tdmin <= timedelta.min
            # 断言 Timedelta 的最小值不等于 Python 标准库中 timedelta 的最小值
            assert tdmin != timedelta.min
            # 断言 Timedelta 的最小值不等于 Python 标准库中 timedelta 的最小值
            assert not tdmin == timedelta.min

        # 处理 "ms" 和 "s" 单位的情况，超过了 pytimedelta 的范围
        for unit in ["ms", "s"]:
            # 获取 Timedelta 的最大值，并按指定单位转换为最大值
            tdmax = Timedelta.max.as_unit(unit).max
            # 获取 Timedelta 的最小值，并按指定单位转换为最小值
            tdmin = Timedelta.min.as_unit(unit).min

            # 断言 Timedelta 的最大值大于 Python 标准库中 timedelta 的最大值
            assert tdmax > timedelta.max
            # 断言 Timedelta 的最大值大于或等于 Python 标准库中 timedelta 的最大值
            assert tdmax >= timedelta.max
            # 断言 Timedelta 的最大值不小于 Python 标准库中 timedelta 的最大值
            assert not tdmax < timedelta.max
            # 断言 Timedelta 的最大值不小于或等于 Python 标准库中 timedelta 的最大值
            assert not tdmax <= timedelta.max
            # 断言 Timedelta 的最大值不等于 Python 标准库中 timedelta 的最大值
            assert tdmax != timedelta.max
            # 断言 Timedelta 的最大值不等于 Python 标准库中 timedelta 的最大值
            assert not tdmax == timedelta.max

            # 断言 Timedelta 的最小值小于 Python 标准库中 timedelta 的最小值
            assert tdmin < timedelta.min
            # 断言 Timedelta 的最小值小于或等于 Python 标准库中 timedelta 的最小值
            assert tdmin <= timedelta.min
            # 断言 Timedelta 的最小值不大于 Python 标准库中 timedelta 的最小值
            assert not tdmin > timedelta.min
            # 断言 Timedelta 的最小值不大于或等于 Python 标准库中 timedelta 的最小值
            assert not tdmin >= timedelta.min
            # 断言 Timedelta 的最小值不等于 Python 标准库中 timedelta 的最小值
            assert tdmin != timedelta.min
            # 断言 Timedelta 的最小值不等于 Python 标准库中 timedelta 的最小值
            assert not tdmin == timedelta.min

    def test_compare_pytimedelta_bounds2(self):
        # a pytimedelta outside the microsecond bounds
        # 一个超出微秒范围的 pytimedelta
        pytd = timedelta(days=999999999, seconds=86399)
        # NB: np.timedelta64(td, "s"") incorrectly overflows
        # 注意：np.timedelta64(td, "s"") 会不正确地溢出
        td64 = np.timedelta64(pytd.days, "D") + np.timedelta64(pytd.seconds, "s")
        # 创建 Timedelta 对象
        td = Timedelta(td64)
        # 断言 Timedelta 对象的天数等于给定的 pytimedelta 的天数
        assert td.days == pytd.days
        # 断言 Timedelta 对象的秒数等于给定的 pytimedelta 的秒数
        assert td.seconds == pytd.seconds

        # 断言 Timedelta 对象等于给定的 pytimedelta
        assert td == pytd
        # 断言 Timedelta 对象不不等于给定的 pytimedelta
        assert not td != pytd
        # 断言 Timedelta 对象不小于给定的 pytimedelta
        assert not td < pytd
        # 断言 Timedelta 对象不大于给定的 pytimedelta
        assert not td > pytd
        # 断言 Timedelta 对象小于或等于给定的 pytimedelta
        assert td <= pytd
        # 断言 Timedelta 对象大于或等于给定的 pytimedelta
        assert td >= pytd

        # 对 Timedelta 对象减去一秒后进行断言
        td2 = td - Timedelta(seconds=1).as_unit("s")
        # 断言减去一秒后的 Timedelta 对象不等于给定的 pytimedelta
        assert td2 != pytd
        # 断言减去一秒后的 Timedelta 对象不等于给定的 pytimedelta
        assert not td2 == pytd
        # 断言减去一秒后的 Timedelta 对象小于给定的 pytimedelta
        assert td2 < pytd
        # 断言减去一秒后的 Timedelta 对象小于或等于给定的 pytimedelta
        assert td2 <= pytd
        # 断言减去一秒后的 Timedelta 对象不大于给定的 pytimedelta
        assert not td2 > pytd
        # 断言减去一秒后的 Timedelta 对象不大于或等于给定的 pytimedelta
        assert not td2 >= pytd

    def test_compare_tick(self, tick_classes):
        # a tick outside the microsecond bounds
        # 一个超出微秒范围的 tick
        cls = tick_classes

        # 创建一个 tick 类的实例 off
        off = cls(4)
        # 获取 off 对象的 _as_pd_timedelta 属性，应为 Timedelta 类型
        td = off._as_pd_timedelta
        # 断言 td 是 Timedelta 类型的实例
        assert isinstance(td, Timedelta)

        # 断言 Timedelta 对象 td 等于 off 对象
        assert td == off
        # 断言 Timedelta 对象 td 不不等于 off 对象
        assert not td != off
        # 断言 Timedelta 对象 td 小于或等于 off 对象
        assert td <= off
        # 断言 Timedelta 对象 td 大于或等于 off 对象
        assert td >= off
        # 断言 Timedelta 对象 td 不小于 off 对象
        assert not td < off
        # 断言 Timedelta 对象 td 不大于 off 对象
        assert not td > off

        # 断言 Timedelta 对象 td 不等于 2 倍的 off 对象
        assert not td ==
    def test_comparison_object_array(self):
        # 模拟 GH#15183 的情况

        # 创建 Timedelta 对象表示 "2 days"
        td = Timedelta("2 days")
        # 创建另一个 Timedelta 对象表示 "3 hours"
        other = Timedelta("3 hours")

        # 创建包含 Timedelta 对象的 NumPy 数组，数据类型为 object
        arr = np.array([other, td], dtype=object)
        # 比较数组中的每个元素是否等于 td，并返回布尔值数组
        res = arr == td
        # 期望的比较结果，是一个布尔值数组
        expected = np.array([False, True], dtype=bool)
        # 断言所有比较结果与期望结果相等
        assert (res == expected).all()

        # 处理二维数组的情况
        arr = np.array([[other, td], [td, other]], dtype=object)
        # 比较数组中的每个元素是否不等于 td
        res = arr != td
        # 期望的比较结果，是一个布尔值数组
        expected = np.array([[True, False], [False, True]], dtype=bool)
        # 断言比较结果的形状与期望结果相同
        assert res.shape == expected.shape
        # 断言所有比较结果与期望结果相等
        assert (res == expected).all()

    def test_compare_timedelta_ndarray(self):
        # 处理 GH#11835 的情况

        # 创建 Timedelta 对象列表
        periods = [Timedelta("0 days 01:00:00"), Timedelta("0 days 01:00:00")]
        # 创建包含 Timedelta 对象的 NumPy 数组
        arr = np.array(periods)
        # 比较数组的第一个元素是否大于数组中的每个元素，并返回布尔值数组
        result = arr[0] > arr
        # 期望的比较结果，是一个布尔值数组
        expected = np.array([False, False])
        # 使用测试模块中的方法断言 NumPy 数组是否相等
        tm.assert_numpy_array_equal(result, expected)

    def test_compare_td64_ndarray(self):
        # 处理 GG#33441 的情况

        # 创建一个 timedelta64 类型的 NumPy 数组，范围为 [0, 4]
        arr = np.arange(5).astype("timedelta64[ns]")
        # 创建一个 Timedelta 对象，表示数组中第二个元素
        td = Timedelta(arr[1])

        # 期望的比较结果，是一个布尔值数组
        expected = np.array([False, True, False, False, False], dtype=bool)

        # 比较 Timedelta 对象与数组中的每个元素是否相等，并返回布尔值数组
        result = td == arr
        # 使用测试模块中的方法断言 NumPy 数组是否相等
        tm.assert_numpy_array_equal(result, expected)

        # 比较数组中的每个元素是否与 Timedelta 对象不相等，并返回布尔值数组
        result = arr != td
        # 使用测试模块中的方法断言 NumPy 数组是否相等
        tm.assert_numpy_array_equal(result, ~expected)

    def test_compare_custom_object(self):
        """
        确保 Timedelta 对象上的不支持操作返回 NotImplemented，
        并委托给其他操作数（GH#20829）。
        """

        class CustomClass:
            def __init__(self, cmp_result=None) -> None:
                self.cmp_result = cmp_result

            def generic_result(self):
                if self.cmp_result is None:
                    return NotImplemented
                else:
                    return self.cmp_result

            def __eq__(self, other):
                return self.generic_result()

            def __gt__(self, other):
                return self.generic_result()

        # 创建 Timedelta 对象表示 "1s"
        t = Timedelta("1s")

        # 断言 Timedelta 对象与不同类型的操作数不相等
        assert t != "string"
        assert t != 1
        assert t != CustomClass()
        assert t != CustomClass(cmp_result=False)

        # 断言 Timedelta 对象小于具有不同比较结果的 CustomClass 实例
        assert t < CustomClass(cmp_result=True)
        assert not t < CustomClass(cmp_result=False)

        # 断言 Timedelta 对象等于具有相同比较结果的 CustomClass 实例
        assert t == CustomClass(cmp_result=True)

    @pytest.mark.parametrize("val", ["string", 1])
    def test_compare_unknown_type(self, val):
        # 处理 GH#20829 的情况

        # 创建 Timedelta 对象表示 "1s"
        t = Timedelta("1s")
        # 预期的错误消息
        msg = "not supported between instances of 'Timedelta' and '(int|str)'"
        
        # 使用 pytest 的断言来验证异常是否被正确抛出
        with pytest.raises(TypeError, match=msg):
            t >= val
        with pytest.raises(TypeError, match=msg):
            t > val
        with pytest.raises(TypeError, match=msg):
            t <= val
        with pytest.raises(TypeError, match=msg):
            t < val
def test_ops_notimplemented():
    # 定义一个空的类 Other
    class Other:
        pass

    # 创建 Other 类的实例对象 other
    other = Other()

    # 创建 Timedelta 对象 td，表示时间间隔为 "1 day"
    td = Timedelta("1 day")

    # 断言以下操作不支持（返回 NotImplemented）
    assert td.__add__(other) is NotImplemented
    assert td.__sub__(other) is NotImplemented
    assert td.__truediv__(other) is NotImplemented
    assert td.__mul__(other) is NotImplemented
    assert td.__floordiv__(other) is NotImplemented


def test_ops_error_str():
    # GH#13624，标识 GitHub 上的问题编号

    # 创建 Timedelta 对象 td，表示时间间隔为 "1 day"
    td = Timedelta("1 day")

    # 迭代器，left 和 right 分别为 td 和 "a"，"a" 和 td
    for left, right in [(td, "a"), ("a", td)]:
        # 构建异常消息，用于检查错误类型 TypeError 和具体错误信息
        msg = "|".join(
            [
                "unsupported operand type",
                r'can only concatenate str \(not "Timedelta"\) to str',
                "must be str, not Timedelta",
            ]
        )
        # 使用 pytest 断言捕获特定的 TypeError 异常，并匹配消息内容
        with pytest.raises(TypeError, match=msg):
            left + right

        # 构建异常消息，用于检查错误类型 TypeError
        msg = "not supported between instances of"
        # 使用 pytest 断言捕获特定的 TypeError 异常，并匹配消息内容
        with pytest.raises(TypeError, match=msg):
            left > right

        # 断言 left 不等于 right
        assert not left == right
        # 断言 left 不等于 right（不相等）
        assert left != right
```