# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\methods\test_round.py`

```
import pytest  # 导入 pytest 库

from pandas._libs.tslibs import to_offset  # 导入 pandas 内部模块 tslibs 的 to_offset 函数
from pandas._libs.tslibs.offsets import INVALID_FREQ_ERR_MSG  # 导入 pandas 内部模块 tslibs.offsets 中的 INVALID_FREQ_ERR_MSG 变量

from pandas import (  # 从 pandas 库导入以下对象：
    DatetimeIndex,  # DatetimeIndex 类
    Timestamp,  # Timestamp 类
    date_range,  # date_range 函数
)
import pandas._testing as tm  # 导入 pandas 内部测试模块 pandas._testing，并重命名为 tm


class TestDatetimeIndexRound:  # 定义测试类 TestDatetimeIndexRound

    def test_round_daily(self):  # 定义测试方法 test_round_daily
        dti = date_range("20130101 09:10:11", periods=5)  # 创建一个包含5个日期时间的 DatetimeIndex 对象，起始于 "20130101 09:10:11"
        result = dti.round("D")  # 对 DatetimeIndex 进行按日（"D"）舍入
        expected = date_range("20130101", periods=5)  # 创建一个期望的 DatetimeIndex，起始于 "20130101"
        tm.assert_index_equal(result, expected)  # 断言 result 与 expected 相等

        dti = dti.tz_localize("UTC").tz_convert("US/Eastern")  # 将 DatetimeIndex 对象转换为 UTC 时区，再转换为 US/Eastern 时区
        result = dti.round("D")  # 再次按日（"D"）舍入
        expected = date_range("20130101", periods=5).tz_localize("US/Eastern")  # 创建一个带有 US/Eastern 时区信息的期望 DatetimeIndex
        tm.assert_index_equal(result, expected)  # 断言 result 与 expected 相等

        result = dti.round("s")  # 对 DatetimeIndex 进行按秒（"s"）舍入
        tm.assert_index_equal(result, dti)  # 断言舍入后的结果与原 DatetimeIndex 相等

    @pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，对下面的测试方法进行多次调用
        "freq, error_msg",  # 参数名称
        [  # 参数组合列表
            ("YE", "<YearEnd: month=12> is a non-fixed frequency"),  # 参数组合1
            ("ME", "<MonthEnd> is a non-fixed frequency"),  # 参数组合2
            ("foobar", "Invalid frequency: foobar"),  # 参数组合3
        ],
    )
    def test_round_invalid(self, freq, error_msg):  # 参数化的测试方法，接受 freq 和 error_msg 参数
        dti = date_range("20130101 09:10:11", periods=5)  # 创建一个 DatetimeIndex 对象
        dti = dti.tz_localize("UTC").tz_convert("US/Eastern")  # 将 DatetimeIndex 对象转换为 UTC 时区，再转换为 US/Eastern 时区
        with pytest.raises(ValueError, match=error_msg):  # 使用 pytest 断言抛出 ValueError 异常，并匹配给定的错误消息
            dti.round(freq)  # 对 DatetimeIndex 进行舍入操作

    def test_round(self, tz_naive_fixture, unit):  # 测试方法，接受 tz_naive_fixture 和 unit 两个参数
        tz = tz_naive_fixture  # 设置时区
        rng = date_range(start="2016-01-01", periods=5, freq="30Min", tz=tz, unit=unit)  # 创建一个 DatetimeIndex，起始于 "2016-01-01"，包含5个时间点，频率为每30分钟
        elt = rng[1]  # 获取索引为1的元素

        expected_rng = DatetimeIndex(  # 创建一个期望的 DatetimeIndex 对象
            [
                Timestamp("2016-01-01 00:00:00", tz=tz),  # 第一个时间点
                Timestamp("2016-01-01 00:00:00", tz=tz),  # 第二个时间点
                Timestamp("2016-01-01 01:00:00", tz=tz),  # 第三个时间点
                Timestamp("2016-01-01 02:00:00", tz=tz),  # 第四个时间点
                Timestamp("2016-01-01 02:00:00", tz=tz),  # 第五个时间点
            ]
        ).as_unit(unit)  # 将期望的 DatetimeIndex 转换为指定单位

        expected_elt = expected_rng[1]  # 获取期望的 DatetimeIndex 中索引为1的元素

        result = rng.round(freq="h")  # 对 DatetimeIndex 进行按小时（"h"）舍入
        tm.assert_index_equal(result, expected_rng)  # 断言舍入后的结果与期望的 DatetimeIndex 相等
        assert elt.round(freq="h") == expected_elt  # 断言索引为1的元素舍入后与期望的元素相等

        msg = INVALID_FREQ_ERR_MSG  # 获取无效频率的错误消息
        with pytest.raises(ValueError, match=msg):  # 使用 pytest 断言抛出 ValueError 异常，并匹配给定的错误消息
            rng.round(freq="foo")  # 对 DatetimeIndex 进行使用无效频率进行舍入操作
        with pytest.raises(ValueError, match=msg):  # 使用 pytest 断言抛出 ValueError 异常，并匹配给定的错误消息
            elt.round(freq="foo")  # 对索引为1的元素使用无效频率进行舍入操作

        msg = "<MonthEnd> is a non-fixed frequency"  # 获取特定错误消息
        with pytest.raises(ValueError, match=msg):  # 使用 pytest 断言抛出 ValueError 异常，并匹配给定的错误消息
            rng.round(freq="ME")  # 对 DatetimeIndex 进行使用非固定频率进行舍入操作
        with pytest.raises(ValueError, match=msg):  # 使用 pytest 断言抛出 ValueError 异常，并匹配给定的错误消息
            elt.round(freq="ME")  # 对索引为1的元素使用非固定频率进行舍入操作

    def test_round2(self, tz_naive_fixture):  # 测试方法，接受 tz_naive_fixture 参数
        tz = tz_naive_fixture  # 设置时区
        # GH#14440 & GH#15578
        index = DatetimeIndex(["2016-10-17 12:00:00.0015"], tz=tz).as_unit("ns")  # 创建一个带有纳秒单位的 DatetimeIndex
        result = index.round("ms")  # 对 DatetimeIndex 进行按毫秒（"ms"）舍入
        expected = DatetimeIndex(["2016-10-17 12:00:00.002000"], tz=tz).as_unit("ns")  # 创建一个期望的 DatetimeIndex
        tm.assert_index_equal(result, expected)  # 断言舍入后的结果与期望的 DatetimeIndex 相等

        for freq in ["us", "ns"]:  # 遍历频率列表
            tm.assert_index_equal(index, index.round(freq))  # 断言使用不同的频率对 DatetimeIndex 进行舍入后结果与原 DatetimeIndex 相等
    @pytest.mark.parametrize(
        "test_input, rounder, freq, expected",
        [  # 参数化测试输入和期望输出
            (["2117-01-01 00:00:45"], "floor", "15s", ["2117-01-01 00:00:45"]),
            (["2117-01-01 00:00:45"], "ceil", "15s", ["2117-01-01 00:00:45"]),
            (
                ["2117-01-01 00:00:45.000000012"],
                "floor",
                "10ns",
                ["2117-01-01 00:00:45.000000010"],
            ),
            (
                ["1823-01-01 00:00:01.000000012"],
                "ceil",
                "10ns",
                ["1823-01-01 00:00:01.000000020"],
            ),
            (["1823-01-01 00:00:01"], "floor", "1s", ["1823-01-01 00:00:01"]),
            (["1823-01-01 00:00:01"], "ceil", "1s", ["1823-01-01 00:00:01"]),
            (["2018-01-01 00:15:00"], "ceil", "15min", ["2018-01-01 00:15:00"]),
            (["2018-01-01 00:15:00"], "floor", "15min", ["2018-01-01 00:15:00"]),
            (["1823-01-01 03:00:00"], "ceil", "3h", ["1823-01-01 03:00:00"]),
            (["1823-01-01 03:00:00"], "floor", "3h", ["1823-01-01 03:00:00"]),
            (
                ("NaT", "1823-01-01 00:00:01"),
                "floor",
                "1s",
                ("NaT", "1823-01-01 00:00:01"),
            ),
            (
                ("NaT", "1823-01-01 00:00:01"),
                "ceil",
                "1s",
                ("NaT", "1823-01-01 00:00:01"),
            ),
        ],
    )



        # 定义测试方法，用于测试日期时间索引的舍入行为
        def test_round3(self, tz_naive_fixture):
            # 使用提供的时区创建日期时间索引对象
            tz = tz_naive_fixture
            # 创建包含单个时间戳的日期时间索引，指定纳秒为单位
            index = DatetimeIndex(["2016-10-17 12:00:00.00149"], tz=tz).as_unit("ns")
            # 对索引进行毫秒级舍入操作
            result = index.round("ms")
            # 创建期望的日期时间索引，指定纳秒为单位
            expected = DatetimeIndex(["2016-10-17 12:00:00.001000"], tz=tz).as_unit("ns")
            # 断言索引是否相等
            tm.assert_index_equal(result, expected)

        # 定义测试方法，测试日期时间索引的舍入行为
        def test_round4(self, tz_naive_fixture):
            # 创建包含单个时间戳的日期时间索引，指定纳秒为单位
            index = DatetimeIndex(["2016-10-17 12:00:00.001501031"], dtype="M8[ns]")
            # 对索引进行10纳秒级舍入操作
            result = index.round("10ns")
            # 创建期望的日期时间索引，指定纳秒为单位
            expected = DatetimeIndex(["2016-10-17 12:00:00.001501030"], dtype="M8[ns]")
            # 断言索引是否相等
            tm.assert_index_equal(result, expected)

            # 创建时间戳字符串
            ts = "2016-10-17 12:00:00.001501031"
            # 创建日期时间索引，指定纳秒为单位
            dti = DatetimeIndex([ts], dtype="M8[ns]")
            # 禁用警告，并对索引进行1010纳秒级舍入操作
            with tm.assert_produces_warning(False):
                dti.round("1010ns")

        # 定义测试方法，测试不进行舍入的情况
        def test_no_rounding_occurs(self, tz_naive_fixture):
            # GH 21262
            # 使用提供的时区创建日期时间范围
            tz = tz_naive_fixture
            rng = date_range(start="2016-01-01", periods=5, freq="2Min", tz=tz)

            # 创建期望的日期时间索引，指定纳秒为单位
            expected_rng = DatetimeIndex(
                [
                    Timestamp("2016-01-01 00:00:00", tz=tz),
                    Timestamp("2016-01-01 00:02:00", tz=tz),
                    Timestamp("2016-01-01 00:04:00", tz=tz),
                    Timestamp("2016-01-01 00:06:00", tz=tz),
                    Timestamp("2016-01-01 00:08:00", tz=tz),
                ]
            ).as_unit("ns")

            # 对日期时间范围进行2分钟级舍入操作
            result = rng.round(freq="2min")
            # 断言索引是否相等
            tm.assert_index_equal(result, expected_rng)
    # 定义测试函数，用于测试 DatetimeIndex 对象的 ceil、floor 和 round 方法的边界情况
    def test_ceil_floor_edge(self, test_input, rounder, freq, expected):
        # 创建 DatetimeIndex 对象，包含测试输入的日期时间列表
        dt = DatetimeIndex(list(test_input))
        # 根据字符串形式的方法名获取对应方法对象
        func = getattr(dt, rounder)
        # 调用指定频率下的 ceil、floor 或 round 方法，返回处理后的结果
        result = func(freq)
        # 创建预期的 DatetimeIndex 对象，包含期望的结果日期时间列表
        expected = DatetimeIndex(list(expected))
        # 使用 assert 断言两个 DatetimeIndex 对象是否相等
        assert expected.equals(result)

    # 使用 pytest 参数化装饰器声明多组测试参数
    @pytest.mark.parametrize(
        "start, index_freq, periods",
        [("2018-01-01", "12h", 25), ("2018-01-01 0:0:0.124999", "1ns", 1000)],
    )
    @pytest.mark.parametrize(
        "round_freq",
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
            "12h",
            "1D",
        ],
    )
    # 定义测试函数，测试在给定起始日期时间、频率和周期下的 round 方法的边界情况
    def test_round_int64(self, start, index_freq, periods, round_freq):
        # 创建 DatetimeIndex 对象，包含指定起始日期时间、频率和周期的日期时间列表
        dt = date_range(start=start, freq=index_freq, periods=periods)
        # 计算 round 的单位，将 round_freq 转换为纳秒
        unit = to_offset(round_freq).nanos

        # 测试 floor 方法
        result = dt.floor(round_freq)
        # 计算日期时间差值
        diff = dt.asi8 - result.asi8
        # 计算结果的模与单位的余数
        mod = result.asi8 % unit
        # 使用 assert 断言 mod 中所有元素是否为 0，即 floor 结果是否是 round_freq 的倍数
        assert (mod == 0).all(), f"floor not a {round_freq} multiple"
        # 使用 assert 断言 diff 中所有元素是否在 [0, unit) 范围内，检测 floor 方法的误差
        assert (0 <= diff).all() and (diff < unit).all(), "floor error"

        # 测试 ceil 方法
        result = dt.ceil(round_freq)
        # 计算日期时间差值
        diff = result.asi8 - dt.asi8
        # 计算结果的模与单位的余数
        mod = result.asi8 % unit
        # 使用 assert 断言 mod 中所有元素是否为 0，即 ceil 结果是否是 round_freq 的倍数
        assert (mod == 0).all(), f"ceil not a {round_freq} multiple"
        # 使用 assert 断言 diff 中所有元素是否在 [0, unit) 范围内，检测 ceil 方法的误差
        assert (0 <= diff).all() and (diff < unit).all(), "ceil error"

        # 测试 round 方法
        result = dt.round(round_freq)
        # 计算结果与原始日期时间的绝对差值
        diff = abs(result.asi8 - dt.asi8)
        # 计算结果的模与单位的余数
        mod = result.asi8 % unit
        # 使用 assert 断言 mod 中所有元素是否为 0，即 round 结果是否是 round_freq 的倍数
        assert (mod == 0).all(), f"round not a {round_freq} multiple"
        # 使用 assert 断言 diff 中所有元素是否在 [0, unit/2) 范围内，检测 round 方法的误差
        assert (diff <= unit // 2).all(), "round error"
        # 如果 unit 是偶数，使用 assert 断言 round 后的结果是否满足 "round half to even" 的条件
        if unit % 2 == 0:
            assert (
                result.asi8[diff == unit // 2] % 2 == 0
            ).all(), "round half to even error"
```