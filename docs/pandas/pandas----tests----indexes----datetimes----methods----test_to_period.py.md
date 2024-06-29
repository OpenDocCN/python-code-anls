# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\methods\test_to_period.py`

```
from datetime import timezone  # 导入时区相关模块

import dateutil.tz  # 导入日期时间相关模块
from dateutil.tz import tzlocal  # 导入本地时区相关模块
import pytest  # 导入 pytest 测试框架

from pandas._libs.tslibs.ccalendar import MONTHS  # 导入月份常量
from pandas._libs.tslibs.offsets import MonthEnd  # 导入月末偏移量
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG  # 导入频率错误消息常量

from pandas import (  # 导入 pandas 库中的多个对象
    DatetimeIndex,  # 日期时间索引对象
    Period,  # 时期对象
    PeriodIndex,  # 时期索引对象
    Timestamp,  # 时间戳对象
    date_range,  # 创建日期范围的函数
    period_range,  # 创建时期范围的函数
)
import pandas._testing as tm  # 导入 pandas 测试模块

class TestToPeriod:
    def test_dti_to_period(self):
        dti = date_range(start="1/1/2005", end="12/1/2005", freq="ME")  # 创建月末频率的日期时间索引对象
        pi1 = dti.to_period()  # 将日期时间索引转换为时期索引
        pi2 = dti.to_period(freq="D")  # 将日期时间索引转换为日频率的时期索引
        pi3 = dti.to_period(freq="3D")  # 将日期时间索引转换为每3天的时期索引

        assert pi1[0] == Period("Jan 2005", freq="M")  # 断言第一个时期对象的频率和值
        assert pi2[0] == Period("1/31/2005", freq="D")  # 断言第一个时期对象的频率和值
        assert pi3[0] == Period("1/31/2005", freq="3D")  # 断言第一个时期对象的频率和值

        assert pi1[-1] == Period("Nov 2005", freq="M")  # 断言最后一个时期对象的频率和值
        assert pi2[-1] == Period("11/30/2005", freq="D")  # 断言最后一个时期对象的频率和值
        assert pi3[-1], Period("11/30/2005", freq="3D")  # 断言最后一个时期对象的频率和值

        tm.assert_index_equal(pi1, period_range("1/1/2005", "11/1/2005", freq="M"))  # 使用测试模块断言索引相等
        tm.assert_index_equal(
            pi2, period_range("1/1/2005", "11/1/2005", freq="M").asfreq("D")
        )  # 使用测试模块断言索引相等
        tm.assert_index_equal(
            pi3, period_range("1/1/2005", "11/1/2005", freq="M").asfreq("3D")
        )  # 使用测试模块断言索引相等

    @pytest.mark.parametrize("month", MONTHS)
    def test_to_period_quarterly(self, month):
        # 确保可以正常执行周期转换
        freq = f"Q-{month}"
        rng = period_range("1989Q3", "1991Q3", freq=freq)  # 创建指定频率的时期范围对象
        stamps = rng.to_timestamp()  # 将时期转换为时间戳
        result = stamps.to_period(freq)  # 将时间戳转换回时期
        tm.assert_index_equal(rng, result)  # 使用测试模块断言索引相等

    @pytest.mark.parametrize("off", ["BQE", "QS", "BQS"])
    def test_to_period_quarterlyish(self, off):
        rng = date_range("01-Jan-2012", periods=8, freq=off)  # 创建指定频率的日期范围对象
        prng = rng.to_period()  # 将日期范围转换为时期范围
        assert prng.freq == "QE-DEC"  # 断言时期范围对象的频率

    @pytest.mark.parametrize("off", ["BYE", "YS", "BYS"])
    def test_to_period_annualish(self, off):
        rng = date_range("01-Jan-2012", periods=8, freq=off)  # 创建指定频率的日期范围对象
        prng = rng.to_period()  # 将日期范围转换为时期范围
        assert prng.freq == "YE-DEC"  # 断言时期范围对象的频率

    def test_to_period_monthish(self):
        offsets = ["MS", "BME"]
        for off in offsets:
            rng = date_range("01-Jan-2012", periods=8, freq=off)  # 创建指定频率的日期范围对象
            prng = rng.to_period()  # 将日期范围转换为时期范围
            assert prng.freqstr == "M"  # 断言时期范围对象的频率字符串

        rng = date_range("01-Jan-2012", periods=8, freq="ME")  # 创建指定频率的日期范围对象
        prng = rng.to_period()  # 将日期范围转换为时期范围
        assert prng.freqstr == "M"  # 断言时期范围对象的频率字符串

        with pytest.raises(ValueError, match=INVALID_FREQ_ERR_MSG):
            date_range("01-Jan-2012", periods=8, freq="EOM")  # 使用 pytest 断言抛出特定异常消息

    @pytest.mark.parametrize(
        "freq_offset, freq_period",
        [
            ("2ME", "2M"),  # 参数化测试用例，定义频率偏移和对应的时期频率
            (MonthEnd(2), MonthEnd(2)),  # 参数化测试用例，定义月末偏移对象和对应的时期频率对象
        ],
    )
    def test_dti_to_period_2monthish(self, freq_offset, freq_period):
        # 使用给定的频率偏移和频率周期创建日期范围
        dti = date_range("2020-01-01", periods=3, freq=freq_offset)
        # 将日期时间索引转换为周期索引
        pi = dti.to_period()

        # 断言周期索引与给定频率周期的期间范围相等
        tm.assert_index_equal(pi, period_range("2020-01", "2020-05", freq=freq_period))

    @pytest.mark.parametrize(
        "freq", ["2ME", "1me", "2QE", "2QE-SEP", "1YE", "ye", "2YE-MAR"]
    )
    def test_to_period_frequency_M_Q_Y_raises(self, freq):
        # 准备错误消息，用于测试无效频率的异常抛出
        msg = f"Invalid frequency: {freq}"

        # 创建一个日期范围，频率为 "ME" (MonthEnd)
        rng = date_range("01-Jan-2012", periods=8, freq="ME")
        # 使用 pytest 断言预期抛出 ValueError 异常，且错误消息匹配 msg
        with pytest.raises(ValueError, match=msg):
            rng.to_period(freq)

    def test_to_period_infer(self):
        # 创建一个日期范围，包括时区信息
        rng = date_range(
            start="2019-12-22 06:40:00+00:00",
            end="2019-12-22 08:45:00+00:00",
            freq="5min",
        )

        # 使用 pytest 断言预期抛出 UserWarning 警告，警告信息匹配 "drop timezone info"
        with tm.assert_produces_warning(UserWarning, match="drop timezone info"):
            # 将日期时间索引转换为周期索引，指定频率为 "5min"
            pi1 = rng.to_period("5min")

        with tm.assert_produces_warning(UserWarning, match="drop timezone info"):
            # 将日期时间索引转换为周期索引，自动推断频率
            pi2 = rng.to_period()

        # 断言两个周期索引对象相等
        tm.assert_index_equal(pi1, pi2)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_period_dt64_round_trip(self):
        # 创建一个工作日频率的日期范围
        dti = date_range("1/1/2000", "1/7/2002", freq="B")
        # 将日期时间索引转换为周期索引
        pi = dti.to_period()
        # 断言周期索引再转回时间戳索引后与原始日期时间索引相等
        tm.assert_index_equal(pi.to_timestamp(), dti)

        # 创建一个工作日频率的日期范围
        dti = date_range("1/1/2000", "1/7/2002", freq="B")
        # 将日期时间索引转换为指定频率为 "h" 的周期索引
        pi = dti.to_period(freq="h")
        # 断言周期索引再转回时间戳索引后与原始日期时间索引相等
        tm.assert_index_equal(pi.to_timestamp(), dti)

    def test_to_period_millisecond(self):
        # 创建一个包含毫秒级别时间戳的日期时间索引
        index = DatetimeIndex(
            [
                Timestamp("2007-01-01 10:11:12.123456Z"),
                Timestamp("2007-01-01 10:11:13.789123Z"),
            ]
        )

        # 使用 pytest 断言预期抛出 UserWarning 警告，警告信息匹配 "drop timezone info"
        with tm.assert_produces_warning(UserWarning, match="drop timezone info"):
            # 将日期时间索引转换为周期索引，频率为 "ms" (毫秒)
            period = index.to_period(freq="ms")
        # 断言周期索引长度为 2
        assert 2 == len(period)
        # 断言周期索引第一个元素与预期的周期对象相等
        assert period[0] == Period("2007-01-01 10:11:12.123Z", "ms")
        # 断言周期索引第二个元素与预期的周期对象相等
        assert period[1] == Period("2007-01-01 10:11:13.789Z", "ms")

    def test_to_period_microsecond(self):
        # 创建一个包含微秒级别时间戳的日期时间索引
        index = DatetimeIndex(
            [
                Timestamp("2007-01-01 10:11:12.123456Z"),
                Timestamp("2007-01-01 10:11:13.789123Z"),
            ]
        )

        # 使用 pytest 断言预期抛出 UserWarning 警告，警告信息匹配 "drop timezone info"
        with tm.assert_produces_warning(UserWarning, match="drop timezone info"):
            # 将日期时间索引转换为周期索引，频率为 "us" (微秒)
            period = index.to_period(freq="us")
        # 断言周期索引长度为 2
        assert 2 == len(period)
        # 断言周期索引第一个元素与预期的周期对象相等
        assert period[0] == Period("2007-01-01 10:11:12.123456Z", "us")
        # 断言周期索引第二个元素与预期的周期对象相等
        assert period[1] == Period("2007-01-01 10:11:13.789123Z", "us")

    @pytest.mark.parametrize(
        "tz",
        [
            "US/Eastern",
            timezone.utc,
            tzlocal(),
            "dateutil/US/Eastern",
            dateutil.tz.tzutc(),
        ],
    )
    # 使用给定时区创建一个日期范围对象 `ts`，从 "2000-01-01" 到 "2000-02-01"
    ts = date_range("1/1/2000", "2/1/2000", tz=tz)

    # 断言应产生 `UserWarning` 警告，其中包含字符串 "drop timezone info"
    with tm.assert_produces_warning(UserWarning, match="drop timezone info"):
        # 将日期时间索引 `ts` 转换为周期（Period），并获取第一个周期的结果
        result = ts.to_period()[0]
        # 预期结果是将 `ts[0]` 的日期时间对象转换为与 `ts.freq` 相对应的周期对象
        expected = ts[0].to_period(ts.freq)

    # 断言 `result` 等于 `expected`
    assert result == expected

    # 创建预期的日期范围对象，不带时区信息的周期对象
    expected = date_range("1/1/2000", "2/1/2000").to_period()

    # 再次断言应产生 `UserWarning` 警告，其中包含字符串 "drop timezone info"
    with tm.assert_produces_warning(UserWarning, match="drop timezone info"):
        # 将日期时间索引 `ts` 转换为周期（Period），使用 `ts.freq` 作为频率
        result = ts.to_period(ts.freq)

    # 断言 `result` 等于 `expected`
    tm.assert_index_equal(result, expected)

# 使用参数化测试，测试时区参数 `tz` 分别为 "Etc/GMT-1" 和 "Etc/GMT+1"
@pytest.mark.parametrize("tz", ["Etc/GMT-1", "Etc/GMT+1"])
def test_to_period_tz_utc_offset_consistency(self, tz):
    # 创建日期范围对象 `ts`，起始和结束日期为 "2000-01-01" 到 "2000-02-01"，使用给定时区 `tz`
    ts = date_range("1/1/2000", "2/1/2000", tz="Etc/GMT-1")
    # 断言应产生 `UserWarning` 警告，其中包含字符串 "drop timezone info"
    with tm.assert_produces_warning(UserWarning, match="drop timezone info"):
        # 将日期时间索引 `ts` 转换为周期（Period），并获取第一个周期的结果
        result = ts.to_period()[0]
        # 预期结果是将 `ts[0]` 的日期时间对象转换为与 `ts.freq` 相对应的周期对象
        expected = ts[0].to_period(ts.freq)
        # 断言 `result` 等于 `expected`
        assert result == expected

def test_to_period_nofreq(self):
    # 创建一个日期时间索引对象 `idx`，包含日期 "2000-01-01", "2000-01-02", "2000-01-04"
    idx = DatetimeIndex(["2000-01-01", "2000-01-02", "2000-01-04"])
    # 定义异常消息 "You must pass a freq argument as current index has none."，预期会引发 ValueError 异常
    msg = "You must pass a freq argument as current index has none."
    # 使用 pytest 断言应该抛出 ValueError 异常，并且异常消息匹配 `msg`
    with pytest.raises(ValueError, match=msg):
        # 调用 `idx.to_period()` 方法
        idx.to_period()

    # 创建一个日期时间索引对象 `idx`，包含日期 "2000-01-01", "2000-01-02", "2000-01-03"，推断频率为 "D"
    idx = DatetimeIndex(["2000-01-01", "2000-01-02", "2000-01-03"], freq="infer")
    # 断言 `idx.freqstr` 等于 "D"
    assert idx.freqstr == "D"
    # 创建预期的周期索引对象 `expected`，频率为 "D"
    expected = PeriodIndex(["2000-01-01", "2000-01-02", "2000-01-03"], freq="D")
    # 使用 `tm.assert_index_equal` 断言 `idx.to_period()` 的结果等于 `expected`
    tm.assert_index_equal(idx.to_period(), expected)

    # 创建一个日期时间索引对象 `idx`，包含日期 "2000-01-01", "2000-01-02", "2000-01-03"
    idx = DatetimeIndex(["2000-01-01", "2000-01-02", "2000-01-03"])
    # 断言 `idx.freqstr` 是 None
    assert idx.freqstr is None
    # 使用 `tm.assert_index_equal` 断言 `idx.to_period()` 的结果等于 `expected`
    tm.assert_index_equal(idx.to_period(), expected)

# 使用参数化测试，测试频率参数 `freq` 分别为 "2BME", "SME-15", "2BMS"
@pytest.mark.parametrize("freq", ["2BME", "SME-15", "2BMS"])
def test_to_period_offsets_not_supported(self, freq):
    # 创建消息字符串 `msg`，包含不支持的频率信息
    msg = "|".join(
        [
            f"Invalid frequency: {freq}",
            f"{freq} is not supported as period frequency",
        ]
    )
    # 创建日期范围对象 `ts`，起始日期 "1/1/2012"，包含 4 个周期，使用给定的频率 `freq`
    ts = date_range("1/1/2012", periods=4, freq=freq)
    # 使用 pytest 断言应该抛出 ValueError 异常，并且异常消息匹配 `msg`
    with pytest.raises(ValueError, match=msg):
        # 调用 `ts.to_period()` 方法
        ts.to_period()
```