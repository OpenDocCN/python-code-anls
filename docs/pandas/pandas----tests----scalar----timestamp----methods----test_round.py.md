# `D:\src\scipysrc\pandas\pandas\tests\scalar\timestamp\methods\test_round.py`

```
from hypothesis import (
    given,
    strategies as st,
)
import numpy as np
import pytest
import pytz

from pandas._libs import lib
from pandas._libs.tslibs import (
    NaT,
    OutOfBoundsDatetime,
    Timedelta,
    Timestamp,
    iNaT,
    to_offset,
)
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG

import pandas._testing as tm

class TestTimestampRound:
    def test_round_division_by_zero_raises(self):
        ts = Timestamp("2016-01-01")

        msg = "Division by zero in rounding"
        # 检查是否在进行0除法时引发 ValueError 异常，匹配特定的错误消息
        with pytest.raises(ValueError, match=msg):
            ts.round("0ns")

    @pytest.mark.parametrize(
        "timestamp, freq, expected",
        [
            ("20130101 09:10:11", "D", "20130101"),
            ("20130101 19:10:11", "D", "20130102"),
            ("20130201 12:00:00", "D", "20130202"),
            ("20130104 12:00:00", "D", "20130105"),
            ("2000-01-05 05:09:15.13", "D", "2000-01-05 00:00:00"),
            ("2000-01-05 05:09:15.13", "h", "2000-01-05 05:00:00"),
            ("2000-01-05 05:09:15.13", "s", "2000-01-05 05:09:15"),
        ],
    )
    def test_round_frequencies(self, timestamp, freq, expected):
        dt = Timestamp(timestamp)
        # 对给定的时间戳进行特定频率的舍入，比较结果与期望的时间戳是否相等
        result = dt.round(freq)
        expected = Timestamp(expected)
        assert result == expected

    def test_round_tzaware(self):
        # 测试带有时区信息的时间戳舍入操作
        dt = Timestamp("20130101 09:10:11", tz="US/Eastern")
        result = dt.round("D")
        expected = Timestamp("20130101", tz="US/Eastern")
        assert result == expected

        dt = Timestamp("20130101 09:10:11", tz="US/Eastern")
        result = dt.round("s")
        assert result == dt

    def test_round_30min(self):
        # 测试30分钟内的时间戳舍入
        dt = Timestamp("20130104 12:32:00")
        result = dt.round("30Min")
        expected = Timestamp("20130104 12:30:00")
        assert result == expected

    def test_round_subsecond(self):
        # 测试毫秒及微秒级别的时间戳舍入
        # GH#14440 & GH#15578
        result = Timestamp("2016-10-17 12:00:00.0015").round("ms")
        expected = Timestamp("2016-10-17 12:00:00.002000")
        assert result == expected

        result = Timestamp("2016-10-17 12:00:00.00149").round("ms")
        expected = Timestamp("2016-10-17 12:00:00.001000")
        assert result == expected

        ts = Timestamp("2016-10-17 12:00:00.0015")
        for freq in ["us", "ns"]:
            # 对微秒和纳秒级别的时间戳进行自我舍入比较
            assert ts == ts.round(freq)

        result = Timestamp("2016-10-17 12:00:00.001501031").round("10ns")
        expected = Timestamp("2016-10-17 12:00:00.001501030")
        assert result == expected

    def test_round_nonstandard_freq(self):
        # 测试非标准频率下的时间戳舍入，同时禁止产生警告
        with tm.assert_produces_warning(False):
            Timestamp("2016-10-17 12:00:00.001501031").round("1010ns")

    def test_round_invalid_arg(self):
        # 测试使用无效参数进行时间戳舍入时是否引发 ValueError 异常
        stamp = Timestamp("2000-01-05 05:09:15.13")
        with pytest.raises(ValueError, match=INVALID_FREQ_ERR_MSG):
            stamp.round("foo")
    @pytest.mark.parametrize(
        "test_input, rounder, freq, expected",
        [
            ("2117-01-01 00:00:45", "floor", "15s", "2117-01-01 00:00:45"),
            ("2117-01-01 00:00:45", "ceil", "15s", "2117-01-01 00:00:45"),
            (
                "2117-01-01 00:00:45.000000012",
                "floor",
                "10ns",
                "2117-01-01 00:00:45.000000010",
            ),
            (
                "1823-01-01 00:00:01.000000012",
                "ceil",
                "10ns",
                "1823-01-01 00:00:01.000000020",
            ),
            ("1823-01-01 00:00:01", "floor", "1s", "1823-01-01 00:00:01"),
            ("1823-01-01 00:00:01", "ceil", "1s", "1823-01-01 00:00:01"),
            ("NaT", "floor", "1s", "NaT"),
            ("NaT", "ceil", "1s", "NaT"),
        ],
    )
    def test_ceil_floor_edge(self, test_input, rounder, freq, expected):
        # 参数化测试用例，测试时间戳的向上取整和向下取整方法在边缘情况下的行为
        dt = Timestamp(test_input)
        func = getattr(dt, rounder)  # 获取对应的取整方法
        result = func(freq)  # 调用取整方法

        if dt is NaT:  # 如果时间戳为 NaT（Not a Time），则预期结果也应为 NaT
            assert result is NaT
        else:
            expected = Timestamp(expected)  # 转换预期结果为时间戳对象
            assert result == expected  # 断言取整后的结果与预期结果相符

    @pytest.mark.parametrize(
        "test_input, freq, expected",
        [
            ("2018-01-01 00:02:06", "2s", "2018-01-01 00:02:06"),
            ("2018-01-01 00:02:00", "2min", "2018-01-01 00:02:00"),
            ("2018-01-01 00:04:00", "4min", "2018-01-01 00:04:00"),
            ("2018-01-01 00:15:00", "15min", "2018-01-01 00:15:00"),
            ("2018-01-01 00:20:00", "20min", "2018-01-01 00:20:00"),
            ("2018-01-01 03:00:00", "3h", "2018-01-01 03:00:00"),
        ],
    )
    @pytest.mark.parametrize("rounder", ["ceil", "floor", "round"])
    def test_round_minute_freq(self, test_input, freq, expected, rounder):
        # 参数化测试用例，测试时间戳的取整方法在不同频率下的行为

        dt = Timestamp(test_input)
        expected = Timestamp(expected)
        func = getattr(dt, rounder)  # 获取对应的取整方法（ceil、floor 或 round）
        result = func(freq)  # 调用取整方法

        assert result == expected  # 断言取整后的结果与预期结果相符

    def test_ceil(self, unit):
        # 测试时间戳的向上取整方法

        dt = Timestamp("20130101 09:10:11").as_unit(unit)  # 将时间戳转换为指定单位
        result = dt.ceil("D")  # 对日期进行向上取整到天
        expected = Timestamp("20130102")  # 预期的向上取整结果为第二天的日期
        assert result == expected  # 断言取整后的结果与预期结果相符
        assert result._creso == dt._creso  # 断言取整后的结果的单位保持不变

    def test_floor(self, unit):
        # 测试时间戳的向下取整方法

        dt = Timestamp("20130101 09:10:11").as_unit(unit)  # 将时间戳转换为指定单位
        result = dt.floor("D")  # 对日期进行向下取整到天
        expected = Timestamp("20130101")  # 预期的向下取整结果为当天的日期
        assert result == expected  # 断言取整后的结果与预期结果相符
        assert result._creso == dt._creso  # 断言取整后的结果的单位保持不变

    @pytest.mark.parametrize("method", ["ceil", "round", "floor"])
    def test_round_dst_border_ambiguous(self, method, unit):
        # 定义一个测试方法，用于测试处理模糊时间边界情况
        # 创建一个带有时区信息的时间戳对象，初始时间为UTC时间的"2017-10-29 00:00:00"，然后转换为"Europe/Madrid"时区
        ts = Timestamp("2017-10-29 00:00:00", tz="UTC").tz_convert("Europe/Madrid")
        # 将时间戳对象按指定单位重新调整
        ts = ts.as_unit(unit)

        # 调用指定方法处理时间戳对象，处理时考虑模糊时间情况
        result = getattr(ts, method)("h", ambiguous=True)
        # 断言结果与原时间戳对象相同
        assert result == ts
        # 断言结果的内部属性 _creso 等于指定单位的值
        assert result._creso == getattr(NpyDatetimeUnit, f"NPY_FR_{unit}").value

        # 以不模糊的方式调用指定方法处理时间戳对象
        result = getattr(ts, method)("h", ambiguous=False)
        # 预期结果为UTC时间的"2017-10-29 01:00:00"，转换为"Europe/Madrid"时区后的时间戳对象
        expected = Timestamp("2017-10-29 01:00:00", tz="UTC").tz_convert(
            "Europe/Madrid"
        )
        assert result == expected
        assert result._creso == getattr(NpyDatetimeUnit, f"NPY_FR_{unit}").value

        # 以 NaT（Not a Time）方式调用指定方法处理时间戳对象
        result = getattr(ts, method)("h", ambiguous="NaT")
        assert result is NaT

        # 使用 pytest 框架验证当处理时间戳对象时，会引发 pytz.AmbiguousTimeError 异常
        msg = "Cannot infer dst time"
        with pytest.raises(pytz.AmbiguousTimeError, match=msg):
            getattr(ts, method)("h", ambiguous="raise")

    @pytest.mark.parametrize(
        "method, ts_str, freq",
        [
            ["ceil", "2018-03-11 01:59:00-0600", "5min"],
            ["round", "2018-03-11 01:59:00-0600", "5min"],
            ["floor", "2018-03-11 03:01:00-0500", "2h"],
        ],
    )
    def test_round_dst_border_nonexistent(self, method, ts_str, freq, unit):
        # 定义一个测试方法，用于测试处理不存在时间边界情况
        # 根据给定的时间字符串创建一个带有时区信息的时间戳对象，并按指定单位重新调整
        ts = Timestamp(ts_str, tz="America/Chicago").as_unit(unit)

        # 调用指定方法处理时间戳对象，处理时考虑不存在时间情况
        result = getattr(ts, method)(freq, nonexistent="shift_forward")
        # 预期结果为"2018-03-11 03:00:00"的时间戳对象，时区为"America/Chicago"
        expected = Timestamp("2018-03-11 03:00:00", tz="America/Chicago")
        assert result == expected
        assert result._creso == getattr(NpyDatetimeUnit, f"NPY_FR_{unit}").value

        # 以 NaT 方式调用指定方法处理时间戳对象
        result = getattr(ts, method)(freq, nonexistent="NaT")
        assert result is NaT

        # 使用 pytest 框架验证当处理时间戳对象时，会引发 pytz.NonExistentTimeError 异常
        msg = "2018-03-11 02:00:00"
        with pytest.raises(pytz.NonExistentTimeError, match=msg):
            getattr(ts, method)(freq, nonexistent="raise")

    @pytest.mark.parametrize(
        "timestamp",
        [
            "2018-01-01 0:0:0.124999360",
            "2018-01-01 0:0:0.125000367",
            "2018-01-01 0:0:0.125500",
            "2018-01-01 0:0:0.126500",
            "2018-01-01 12:00:00",
            "2019-01-01 12:00:00",
        ],
    )
    @pytest.mark.parametrize(
        "freq",
        [
            "2ns",
            "3ns",
            "4ns",
            "5ns",
            "6ns",
            "7ns",
            "250ns",
            "500ns",
            "750ns",
            "1us",
            "19us",
            "250us",
            "500us",
            "750us",
            "1s",
            "2s",
            "3s",
            "1D",
        ],
    )
    # 定义一个测试方法，用于验证对 int64 精度的所有舍入模式的准确性
    def test_round_int64(self, timestamp, freq):
        # 将输入的时间戳转换为 Timestamp 对象，并以纳秒为单位处理
        dt = Timestamp(timestamp).as_unit("ns")
        # 根据频率获取时间单位的纳秒数
        unit = to_offset(freq).nanos

        # 测试向下取整
        result = dt.floor(freq)
        # 断言结果是频率的整数倍
        assert result._value % unit == 0, f"floor not a {freq} multiple"
        # 断言误差在一个单位之内
        assert 0 <= dt._value - result._value < unit, "floor error"

        # 测试向上取整
        result = dt.ceil(freq)
        # 断言结果是频率的整数倍
        assert result._value % unit == 0, f"ceil not a {freq} multiple"
        # 断言误差在一个单位之内
        assert 0 <= result._value - dt._value < unit, "ceil error"

        # 测试四舍五入
        result = dt.round(freq)
        # 断言结果是频率的整数倍
        assert result._value % unit == 0, f"round not a {freq} multiple"
        # 断言误差在半个单位之内
        assert abs(result._value - dt._value) <= unit // 2, "round error"
        # 如果单位是偶数且误差恰好为半个单位，则进行偶数舍入
        if unit % 2 == 0 and abs(result._value - dt._value) == unit // 2:
            assert result._value // unit % 2 == 0, "round half to even error"

    # 测试 Timestamp 类的取整实现边界条件
    def test_round_implementation_bounds(self):
        # 检查最小时间戳的向上取整结果
        result = Timestamp.min.ceil("s")
        expected = Timestamp(1677, 9, 21, 0, 12, 44)
        assert result == expected

        # 检查最大时间戳的向下取整结果
        result = Timestamp.max.floor("s")
        expected = Timestamp.max - Timedelta(854775807)
        assert result == expected

        # 检查对最小时间戳进行向下取整会超出边界的情况
        msg = "Cannot round 1677-09-21 00:12:43.145224193 to freq=<Second>"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp.min.floor("s")

        # 检查对最小时间戳进行四舍五入会超出边界的情况
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp.min.round("s")

        # 检查对最大时间戳进行向上取整会超出边界的情况
        msg = "Cannot round 2262-04-11 23:47:16.854775807 to freq=<Second>"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp.max.ceil("s")

        # 检查对最大时间戳进行四舍五入会超出边界的情况
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp.max.round("s")

    @given(val=st.integers(iNaT + 1, lib.i8max))
    @pytest.mark.parametrize(
        "method", [Timestamp.round, Timestamp.floor, Timestamp.ceil]
    )
    def test_round_sanity(self, val, method):
        # 定义测试方法的起始点，接收参数 val 和 method
        cls = Timestamp
        # 引用 Timestamp 类作为 cls

        err_cls = OutOfBoundsDatetime
        # 引用 OutOfBoundsDatetime 类作为 err_cls

        val = np.int64(val)
        # 将 val 转换为 NumPy 的 int64 类型

        ts = cls(val)
        # 使用 cls 创建 Timestamp 对象 ts

        def checker(ts, nanos, unit):
            # 定义内部函数 checker，接收 Timestamp 对象 ts，时间单位 nanos，单位名称 unit

            # 首先检查是否会在应该抛出异常的情况下引发异常
            if nanos == 1:
                # 如果时间单位是 1 纳秒，不进行额外处理，直接通过
                pass
            else:
                # 计算除法和取模结果
                div, mod = divmod(ts._value, nanos)
                # 计算与时间单位 nanos 的差异
                diff = int(nanos - mod)
                # 计算下界 lb 和上界 ub
                lb = ts._value - mod
                assert lb <= ts._value  # 确保没有 Python 整数溢出
                ub = ts._value + diff
                assert ub > ts._value  # 确保没有 Python 整数溢出

                msg = "without overflow"
                if mod == 0:
                    # 如果 mod 等于 0，则不应该引发异常
                    pass
                elif method is cls.ceil:
                    # 如果 method 是 cls.ceil
                    if ub > cls.max._value:
                        # 如果上界 ub 超过了 Timestamp 类的最大值
                        with pytest.raises(err_cls, match=msg):
                            method(ts, unit)
                        return
                elif method is cls.floor:
                    # 如果 method 是 cls.floor
                    if lb < cls.min._value:
                        # 如果下界 lb 小于 Timestamp 类的最小值
                        with pytest.raises(err_cls, match=msg):
                            method(ts, unit)
                        return
                elif mod >= diff:
                    # 如果 mod 大于等于 diff
                    if ub > cls.max._value:
                        # 如果上界 ub 超过了 Timestamp 类的最大值
                        with pytest.raises(err_cls, match=msg):
                            method(ts, unit)
                        return
                elif lb < cls.min._value:
                    # 如果下界 lb 小于 Timestamp 类的最小值
                    with pytest.raises(err_cls, match=msg):
                        method(ts, unit)
                    return

            # 调用 method 方法计算结果 res
            res = method(ts, unit)

            # 计算结果 res 与 ts 的时间差 td
            td = res - ts
            # 计算时间差的绝对值
            diff = abs(td._value)
            # 确保时间差小于 nanos
            assert diff < nanos
            # 确保 res 的值可以被 nanos 整除
            assert res._value % nanos == 0

            if method is cls.round:
                # 如果 method 是 cls.round
                assert diff <= nanos / 2
            elif method is cls.floor:
                # 如果 method 是 cls.floor
                assert res <= ts
            elif method is cls.ceil:
                # 如果 method 是 cls.ceil
                assert res >= ts

        # 依次使用不同的时间单位调用 checker 函数进行测试
        nanos = 1
        checker(ts, nanos, "ns")

        nanos = 1000
        checker(ts, nanos, "us")

        nanos = 1_000_000
        checker(ts, nanos, "ms")

        nanos = 1_000_000_000
        checker(ts, nanos, "s")

        nanos = 60 * 1_000_000_000
        checker(ts, nanos, "min")

        nanos = 60 * 60 * 1_000_000_000
        checker(ts, nanos, "h")

        nanos = 24 * 60 * 60 * 1_000_000_000
        checker(ts, nanos, "D")
```