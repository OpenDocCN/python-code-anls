# `D:\src\scipysrc\pandas\pandas\tests\scalar\timestamp\test_arithmetic.py`

```
from datetime import (
    datetime,  # 导入 datetime 模块中的 datetime 类
    timedelta,  # 导入 datetime 模块中的 timedelta 类
    timezone,  # 导入 datetime 模块中的 timezone 类
)

from dateutil.tz import gettz  # 导入 dateutil.tz 模块中的 gettz 函数
import numpy as np  # 导入 numpy 库，并用 np 作为别名
import pytest  # 导入 pytest 库

from pandas._libs.tslibs import (
    OutOfBoundsDatetime,  # 从 pandas._libs.tslibs 中导入 OutOfBoundsDatetime 异常类
    OutOfBoundsTimedelta,  # 从 pandas._libs.tslibs 中导入 OutOfBoundsTimedelta 异常类
    Timedelta,  # 从 pandas._libs.tslibs 中导入 Timedelta 类
    Timestamp,  # 从 pandas._libs.tslibs 中导入 Timestamp 类
    offsets,  # 从 pandas._libs.tslibs 中导入 offsets 模块
    to_offset,  # 从 pandas._libs.tslibs 中导入 to_offset 函数
)

import pandas._testing as tm  # 导入 pandas._testing 库，并用 tm 作为别名


class TestTimestampArithmetic:
    def test_overflow_offset(self):
        # no overflow expected
        # 没有预期的溢出情况

        stamp = Timestamp("2000/1/1")  # 创建 Timestamp 对象 stamp，表示日期 "2000/1/1"
        offset_no_overflow = to_offset("D") * 100  # 创建表示 100 天的 Timedelta 偏移量对象

        expected = Timestamp("2000/04/10")  # 预期的 Timestamp 结果对象
        assert stamp + offset_no_overflow == expected  # 断言表达式，验证加法操作的结果是否符合预期

        assert offset_no_overflow + stamp == expected  # 断言表达式，验证反向加法操作的结果是否符合预期

        expected = Timestamp("1999/09/23")  # 更新预期的 Timestamp 结果对象
        assert stamp - offset_no_overflow == expected  # 断言表达式，验证减法操作的结果是否符合预期

    def test_overflow_offset_raises(self):
        # xref https://github.com/statsmodels/statsmodels/issues/3374
        # ends up multiplying really large numbers which overflow
        # 引用链接，详见 https://github.com/statsmodels/statsmodels/issues/3374
        # 乘以非常大的数字导致溢出

        stamp = Timestamp("2017-01-13 00:00:00").as_unit("ns")  # 创建 Timestamp 对象 stamp，表示日期 "2017-01-13 00:00:00"，单位为纳秒
        offset_overflow = 20169940 * offsets.Day(1)  # 创建一个偏移量，乘以一个大数值，预期会导致溢出
        lmsg2 = r"Cannot cast -?20169940 days \+?00:00:00 to unit='ns' without overflow"  # 预期的异常消息模式

        with pytest.raises(OutOfBoundsTimedelta, match=lmsg2):  # 使用 pytest 检查是否引发了 OutOfBoundsTimedelta 异常，并匹配异常消息模式 lmsg2
            stamp + offset_overflow

        with pytest.raises(OutOfBoundsTimedelta, match=lmsg2):  # 使用 pytest 检查反向加法操作是否引发了 OutOfBoundsTimedelta 异常，并匹配异常消息模式 lmsg2
            offset_overflow + stamp

        with pytest.raises(OutOfBoundsTimedelta, match=lmsg2):  # 使用 pytest 检查减法操作是否引发了 OutOfBoundsTimedelta 异常，并匹配异常消息模式 lmsg2
            stamp - offset_overflow

        # xref https://github.com/pandas-dev/pandas/issues/14080
        # used to crash, so check for proper overflow exception
        # 引用链接，详见 https://github.com/pandas-dev/pandas/issues/14080
        # 曾经崩溃，因此检查正确的溢出异常情况

        stamp = Timestamp("2000/1/1").as_unit("ns")  # 创建 Timestamp 对象 stamp，表示日期 "2000/1/1"，单位为纳秒
        offset_overflow = to_offset("D") * 100**5  # 创建一个偏移量，乘以 100 的 5 次方，预期会导致溢出

        lmsg3 = (
            r"Cannot cast -?10000000000 days \+?00:00:00 to unit='ns' without overflow"
        )  # 预期的异常消息模式
        with pytest.raises(OutOfBoundsTimedelta, match=lmsg3):  # 使用 pytest 检查是否引发了 OutOfBoundsTimedelta 异常，并匹配异常消息模式 lmsg3
            stamp + offset_overflow

        with pytest.raises(OutOfBoundsTimedelta, match=lmsg3):  # 使用 pytest 检查反向加法操作是否引发了 OutOfBoundsTimedelta 异常，并匹配异常消息模式 lmsg3
            offset_overflow + stamp

        with pytest.raises(OutOfBoundsTimedelta, match=lmsg3):  # 使用 pytest 检查减法操作是否引发了 OutOfBoundsTimedelta 异常，并匹配异常消息模式 lmsg3
            stamp - offset_overflow

    def test_overflow_timestamp_raises(self):
        # https://github.com/pandas-dev/pandas/issues/31774
        msg = "Result is too large"
        a = Timestamp("2101-01-01 00:00:00").as_unit("ns")  # 创建 Timestamp 对象 a，表示日期 "2101-01-01 00:00:00"，单位为纳秒
        b = Timestamp("1688-01-01 00:00:00").as_unit("ns")  # 创建 Timestamp 对象 b，表示日期 "1688-01-01 00:00:00"，单位为纳秒

        with pytest.raises(OutOfBoundsDatetime, match=msg):  # 使用 pytest 检查是否引发了 OutOfBoundsDatetime 异常，并匹配异常消息 msg
            a - b

        # but we're OK for timestamp and datetime.datetime
        # 但是对于 timestamp 和 datetime.datetime 我们没有问题
        assert (a - b.to_pydatetime()) == (a.to_pydatetime() - b)

    def test_delta_preserve_nanos(self):
        val = Timestamp(1337299200000000123)  # 创建 Timestamp 对象 val，表示纳秒精度的时间戳
        result = val + timedelta(1)  # 计算 val 加一天的结果
        assert result.nanosecond == val.nanosecond  # 断言表达式，验证结果的纳秒部分与原始时间戳的纳秒部分相等
    def test_rsub_dtscalars(self, tz_naive_fixture):
        # 测试 datetime64 减去 Timestamp 的操作，见 GH#28286
        td = Timedelta(1235345642000)  # 创建一个 Timedelta 对象，表示一段时间
        ts = Timestamp("2021-01-01", tz=tz_naive_fixture)  # 创建一个带有时区信息的 Timestamp 对象
        other = ts + td  # 计算 ts 加上 td 后的结果

        assert other - ts == td  # 断言 ts 与 other 相减的结果应该等于 td
        assert other.to_pydatetime() - ts == td  # 断言 ts 转换为 Python 的 datetime 后与 other 相减的结果应该等于 td
        if tz_naive_fixture is None:
            assert other.to_datetime64() - ts == td  # 如果 tz_naive_fixture 是 None，断言 ts 转换为 datetime64 后与 other 相减的结果应该等于 td
        else:
            msg = "Cannot subtract tz-naive and tz-aware datetime-like objects"
            with pytest.raises(TypeError, match=msg):  # 断言尝试计算 tz-naive 和 tz-aware datetime-like 对象的减法会引发 TypeError 异常
                other.to_datetime64() - ts

    def test_timestamp_sub_datetime(self):
        dt = datetime(2013, 10, 12)  # 创建一个普通的 Python datetime 对象
        ts = Timestamp(datetime(2013, 10, 13))  # 创建一个 Timestamp 对象，使用 Python datetime 对象初始化
        assert (ts - dt).days == 1  # 断言 Timestamp 减去 datetime 的结果应该是 1 天
        assert (dt - ts).days == -1  # 断言 datetime 减去 Timestamp 的结果应该是 -1 天

    def test_subtract_tzaware_datetime(self):
        t1 = Timestamp("2020-10-22T22:00:00+00:00")  # 创建一个带时区信息的 Timestamp 对象
        t2 = datetime(2020, 10, 22, 22, tzinfo=timezone.utc)  # 创建一个带有 UTC 时区信息的 Python datetime 对象

        result = t1 - t2  # 计算 t1 减去 t2 的结果

        assert isinstance(result, Timedelta)  # 断言结果是 Timedelta 对象
        assert result == Timedelta("0 days")  # 断言结果应该是 0 天的 Timedelta 对象

    def test_subtract_timestamp_from_different_timezone(self):
        t1 = Timestamp("20130101").tz_localize("US/Eastern")  # 创建一个带有美东时区信息的 Timestamp 对象
        t2 = Timestamp("20130101").tz_localize("CET")  # 创建一个带有 CET 时区信息的 Timestamp 对象

        result = t1 - t2  # 计算 t1 减去 t2 的结果

        assert isinstance(result, Timedelta)  # 断言结果是 Timedelta 对象
        assert result == Timedelta("0 days 06:00:00")  # 断言结果应该是 6 小时的 Timedelta 对象

    def test_subtracting_involving_datetime_with_different_tz(self):
        t1 = datetime(2013, 1, 1, tzinfo=timezone(timedelta(hours=-5)))  # 创建一个带有 -5 小时时差的 Python datetime 对象
        t2 = Timestamp("20130101").tz_localize("CET")  # 创建一个带有 CET 时区信息的 Timestamp 对象

        result = t1 - t2  # 计算 t1 减去 t2 的结果

        assert isinstance(result, Timedelta)  # 断言结果是 Timedelta 对象
        assert result == Timedelta("0 days 06:00:00")  # 断言结果应该是 6 小时的 Timedelta 对象

        result = t2 - t1  # 计算 t2 减去 t1 的结果
        assert isinstance(result, Timedelta)  # 断言结果是 Timedelta 对象
        assert result == Timedelta("-1 days +18:00:00")  # 断言结果应该是 -1 天 18 小时的 Timedelta 对象

    def test_subtracting_different_timezones(self, tz_aware_fixture):
        t_raw = Timestamp("20130101")  # 创建一个原始的 Timestamp 对象
        t_UTC = t_raw.tz_localize("UTC")  # 将原始的 Timestamp 对象设定为 UTC 时区
        t_diff = t_UTC.tz_convert(tz_aware_fixture) + Timedelta("0 days 05:00:00")  # 将 t_UTC 转换为指定时区后再加上 5 小时的 Timedelta

        result = t_diff - t_UTC  # 计算 t_diff 减去 t_UTC 的结果

        assert isinstance(result, Timedelta)  # 断言结果是 Timedelta 对象
        assert result == Timedelta("0 days 05:00:00")  # 断言结果应该是 5 小时的 Timedelta 对象
    def test_addition_subtraction_types(self):
        # Assert on the types resulting from Timestamp +/- various date/time
        # objects
        
        # 创建一个 datetime 对象 dt，表示日期为 2014 年 3 月 4 日
        dt = datetime(2014, 3, 4)
        # 创建一个 timedelta 对象 td，表示时间间隔为 1 秒
        td = timedelta(seconds=1)
        # 创建一个 Timestamp 对象 ts，以 datetime 对象 dt 初始化
        ts = Timestamp(dt)

        # 设置错误消息
        msg = "Addition/subtraction of integers"
        
        # 使用 pytest 检查以下表达式是否引发 TypeError，并匹配指定的错误消息
        with pytest.raises(TypeError, match=msg):
            # GH#22535 add/sub with integers is deprecated
            ts + 1
        with pytest.raises(TypeError, match=msg):
            ts - 1

        # Timestamp 对象不支持与 datetime 相加，但支持相减，并返回 timedelta
        # 更多测试在 tseries/base/tests/test_base.py 中
        assert type(ts - dt) == Timedelta
        assert type(ts + td) == Timestamp
        assert type(ts - td) == Timestamp

        # Timestamp 不支持与 datetime64 相加或相减，因此未进行测试（可能会引发错误？）
        td64 = np.timedelta64(1, "D")
        assert type(ts + td64) == Timestamp
        assert type(ts - td64) == Timestamp

    @pytest.mark.parametrize(
        "td", [Timedelta(hours=3), np.timedelta64(3, "h"), timedelta(hours=3)]
    )
    def test_radd_tdscalar(self, td, fixed_now_ts):
        # GH#24775 timedelta64+Timestamp should not raise
        ts = fixed_now_ts
        assert td + ts == ts + td

    @pytest.mark.parametrize(
        "other,expected_difference",
        [
            (np.timedelta64(-123, "ns"), -123),
            (np.timedelta64(1234567898, "ns"), 1234567898),
            (np.timedelta64(-123, "us"), -123000),
            (np.timedelta64(-123, "ms"), -123000000),
        ],
    )
    def test_timestamp_add_timedelta64_unit(self, other, expected_difference):
        now = datetime.now(timezone.utc)
        # 用当前时间创建 Timestamp 对象 ts，并将其转换为纳秒单位
        ts = Timestamp(now).as_unit("ns")
        result = ts + other
        # 计算结果的值与 ts 值之间的差异
        valdiff = result._value - ts._value
        assert valdiff == expected_difference

        ts2 = Timestamp(now)
        assert ts2 + other == result

    @pytest.mark.parametrize(
        "ts",
        [
            Timestamp("1776-07-04"),
            Timestamp("1776-07-04", tz="UTC"),
        ],
    )
    @pytest.mark.parametrize(
        "other",
        [
            1,
            np.int64(1),
            np.array([1, 2], dtype=np.int32),
            np.array([3, 4], dtype=np.uint64),
        ],
    )
    def test_add_int_with_freq(self, ts, other):
        msg = "Addition/subtraction of integers and integer-arrays"
        
        # 使用 pytest 检查以下表达式是否引发 TypeError，并匹配指定的错误消息
        with pytest.raises(TypeError, match=msg):
            ts + other
        with pytest.raises(TypeError, match=msg):
            other + ts

        with pytest.raises(TypeError, match=msg):
            ts - other

        # 检查是否引发 TypeError，匹配指定的错误消息
        msg = "unsupported operand type"
        with pytest.raises(TypeError, match=msg):
            other - ts

    @pytest.mark.parametrize("shape", [(6,), (2, 3)])
    # 测试函数：test_addsub_m8ndarray
    def test_addsub_m8ndarray(self, shape):
        # GH#33296
        # 创建一个时间戳对象，代表 "2020-04-04 15:45"，单位为纳秒
        ts = Timestamp("2020-04-04 15:45").as_unit("ns")
        # 创建一个 numpy 数组，包含 0 到 5 的整数，类型为 "m8[h]"，并按照给定形状重塑
        other = np.arange(6).astype("m8[h]").reshape(shape)

        # 执行时间戳与数组的加法操作
        result = ts + other

        # 生成预期的时间戳数组列表，包含时间戳加上从 0 到 5 小时的时间增量
        ex_stamps = [ts + Timedelta(hours=n) for n in range(6)]
        # 将预期结果转换为 numpy 数组，类型为 "M8[ns]"，并按照给定形状重塑
        expected = np.array([x.asm8 for x in ex_stamps], dtype="M8[ns]").reshape(shape)
        # 断言 numpy 数组的相等性
        tm.assert_numpy_array_equal(result, expected)

        # 执行数组与时间戳的加法操作
        result = other + ts
        # 断言 numpy 数组的相等性
        tm.assert_numpy_array_equal(result, expected)

        # 执行时间戳与数组的减法操作
        result = ts - other
        # 生成预期的时间戳数组列表，包含时间戳减去从 0 到 5 小时的时间增量
        ex_stamps = [ts - Timedelta(hours=n) for n in range(6)]
        # 将预期结果转换为 numpy 数组，类型为 "M8[ns]"，并按照给定形状重塑
        expected = np.array([x.asm8 for x in ex_stamps], dtype="M8[ns]").reshape(shape)
        # 断言 numpy 数组的相等性
        tm.assert_numpy_array_equal(result, expected)

        # 创建一个正则表达式字符串，用于匹配错误消息
        msg = r"unsupported operand type\(s\) for -: 'numpy.ndarray' and 'Timestamp'"
        # 使用 pytest 断言捕获到预期的 TypeError 异常，并匹配错误消息
        with pytest.raises(TypeError, match=msg):
            # 执行不支持的操作：数组与时间戳的减法操作
            other - ts

    # 测试函数：test_addsub_m8ndarray_tzaware
    @pytest.mark.parametrize("shape", [(6,), (2, 3)])
    def test_addsub_m8ndarray_tzaware(self, shape):
        # GH#33296
        # 创建一个时区感知的时间戳对象，代表 "2020-04-04 15:45"，时区为 "US/Pacific"
        ts = Timestamp("2020-04-04 15:45", tz="US/Pacific")

        # 创建一个 numpy 数组，包含 0 到 5 的整数，类型为 "m8[h]"，并按照给定形状重塑
        other = np.arange(6).astype("m8[h]").reshape(shape)

        # 执行时间戳与数组的加法操作
        result = ts + other

        # 生成预期的时间戳数组列表，包含时间戳加上从 0 到 5 小时的时间增量
        ex_stamps = [ts + Timedelta(hours=n) for n in range(6)]
        # 将预期结果转换为 numpy 数组，按照给定形状重塑
        expected = np.array(ex_stamps).reshape(shape)
        # 断言 numpy 数组的相等性
        tm.assert_numpy_array_equal(result, expected)

        # 执行数组与时间戳的加法操作
        result = other + ts
        # 断言 numpy 数组的相等性
        tm.assert_numpy_array_equal(result, expected)

        # 执行时间戳与数组的减法操作
        result = ts - other
        # 生成预期的时间戳数组列表，包含时间戳减去从 0 到 5 小时的时间增量
        ex_stamps = [ts - Timedelta(hours=n) for n in range(6)]
        # 将预期结果转换为 numpy 数组，按照给定形状重塑
        expected = np.array(ex_stamps).reshape(shape)
        # 断言 numpy 数组的相等性
        tm.assert_numpy_array_equal(result, expected)

        # 创建一个正则表达式字符串，用于匹配错误消息
        msg = r"unsupported operand type\(s\) for -: 'numpy.ndarray' and 'Timestamp'"
        # 使用 pytest 断言捕获到预期的 TypeError 异常，并匹配错误消息
        with pytest.raises(TypeError, match=msg):
            # 执行不支持的操作：数组与时间戳的减法操作
            other - ts

    # 测试函数：test_subtract_different_utc_objects
    def test_subtract_different_utc_objects(self, utc_fixture, utc_fixture2):
        # GH 32619
        # 创建一个 datetime 对象，代表 "2021-01-01"
        dt = datetime(2021, 1, 1)
        # 创建两个时区感知的时间戳对象，使用不同的时区
        ts1 = Timestamp(dt, tz=utc_fixture)
        ts2 = Timestamp(dt, tz=utc_fixture2)
        # 计算两个时间戳对象的差值
        result = ts1 - ts2
        # 创建一个期待的时间增量对象，代表 0
        expected = Timedelta(0)
        # 断言计算结果与期待结果相等
        assert result == expected

    # 测试函数：test_timestamp_add_timedelta_push_over_dst_boundary
    @pytest.mark.parametrize(
        "tz",
        [
            "pytz/US/Eastern",
            gettz("US/Eastern"),
            "US/Eastern",
            "dateutil/US/Eastern",
        ],
    )
    def test_timestamp_add_timedelta_push_over_dst_boundary(self, tz):
        # GH#1389
        # 如果时区是以 "pytz/" 开头的字符串，则导入 pytz 模块并使用正确的时区对象
        if isinstance(tz, str) and tz.startswith("pytz/"):
            pytz = pytest.importorskip("pytz")
            tz = pytz.timezone(tz.removeprefix("pytz/"))
        # 创建一个时间戳对象，代表 "2012-03-10 22:00"，使用给定的时区
        stamp = Timestamp("3/10/2012 22:00", tz=tz)

        # 执行时间戳与时间增量的加法操作，增加 6 小时
        result = stamp + timedelta(hours=6)

        # 创建一个期待的时间戳对象，代表 "2012-03-11 05:00"，使用相同的时区
        expected = Timestamp("3/11/2012 05:00", tz=tz)

        # 断言计算结果与期待结果相等
        assert result == expected
class SubDatetime(datetime):
    pass

# 定义了一个名为 `SubDatetime` 的子类，继承自 `datetime` 类。


@pytest.mark.parametrize(
    "lh,rh",
    [
        (SubDatetime(2000, 1, 1), Timedelta(hours=1)),
        (Timedelta(hours=1), SubDatetime(2000, 1, 1)),
    ],
)

# 使用 `pytest.mark.parametrize` 装饰器标记的测试函数，定义了参数化测试，测试参数包括两组元组：
# 第一组：(`SubDatetime(2000, 1, 1)`, `Timedelta(hours=1)`)
# 第二组：(`Timedelta(hours=1)`, `SubDatetime(2000, 1, 1)`)


def test_dt_subclass_add_timedelta(lh, rh):
    # GH#25851
    # ensure that subclassed datetime works for
    # Timedelta operations
    result = lh + rh
    expected = SubDatetime(2000, 1, 1, 1)
    assert result == expected

# 测试函数 `test_dt_subclass_add_timedelta`，接收参数 `lh` 和 `rh`，分别表示左手边和右手边的值。
# 通过 `lh + rh` 执行了日期时间与时间增量的加法操作。
# `assert result == expected` 断言确保加法操作的结果 `result` 等于预期的 `expected` 值 `SubDatetime(2000, 1, 1, 1)`。
```