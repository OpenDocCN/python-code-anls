# `D:\src\scipysrc\pandas\pandas\tests\resample\test_period_index.py`

```
# 导入必要的库和模块
from datetime import (
    datetime,    # 导入 datetime 对象
    timezone,    # 导入 timezone 对象
)
import re    # 导入正则表达式模块
import warnings    # 导入警告处理模块
import zoneinfo    # 导入时区信息模块

import dateutil    # 导入日期工具模块
import numpy as np    # 导入 NumPy 库
import pytest    # 导入 pytest 测试框架

from pandas._libs.tslibs.ccalendar import (
    DAYS,    # 导入工作日列表
    MONTHS,    # 导入月份列表
)
from pandas._libs.tslibs.period import IncompatibleFrequency    # 导入异常类 IncompatibleFrequency
from pandas.errors import InvalidIndexError    # 导入索引错误类 InvalidIndexError

import pandas as pd    # 导入 Pandas 库
from pandas import (
    DataFrame,    # 导入 DataFrame 类
    Series,    # 导入 Series 类
    Timestamp,    # 导入时间戳类 Timestamp
)
import pandas._testing as tm    # 导入 Pandas 测试工具模块
from pandas.core.indexes.datetimes import date_range    # 导入日期范围生成函数 date_range
from pandas.core.indexes.period import (
    Period,    # 导入周期类 Period
    PeriodIndex,    # 导入周期索引类 PeriodIndex
    period_range,    # 导入周期范围生成函数 period_range
)
from pandas.core.resample import _get_period_range_edges    # 导入获取周期范围边界函数 _get_period_range_edges

from pandas.tseries import offsets    # 导入时间偏移量模块 offsets

# 使用 pytest 标记，忽略特定警告
pytestmark = pytest.mark.filterwarnings(
    "ignore:Resampling with a PeriodIndex is deprecated:FutureWarning"
)


@pytest.fixture
def simple_period_range_series():
    """
    用于测试目的的周期范围索引和随机数据的 Series。
    """

    def _simple_period_range_series(start, end, freq="D"):
        with warnings.catch_warnings():
            # 忽略 Period[B] 的弃用警告
            msg = "|".join(["Period with BDay freq", r"PeriodDtype\[B\] is deprecated"])
            warnings.filterwarnings(
                "ignore",
                msg,
                category=FutureWarning,
            )
            # 生成指定频率的周期范围
            rng = period_range(start, end, freq=freq)
        return Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

    return _simple_period_range_series


class TestPeriodIndex:
    @pytest.mark.parametrize("freq", ["2D", "1h", "2h"])
    def test_asfreq(self, frame_or_series, freq):
        # GH 12884, 15944

        # 使用 frame_or_series 创建对象，周期范围从 "2020-01-01" 开始，共 5 个周期
        obj = frame_or_series(range(5), index=period_range("2020-01-01", periods=5))

        # 将周期索引转换为时间戳后重新采样，并作为期望值
        expected = obj.to_timestamp().resample(freq).asfreq()
        result = obj.to_timestamp().resample(freq).asfreq()
        tm.assert_almost_equal(result, expected)

        # 计算开始和结束时间戳，生成新索引，并作为期望值
        start = obj.index[0].to_timestamp(how="start")
        end = (obj.index[-1] + obj.index.freq).to_timestamp(how="start")
        new_index = date_range(start=start, end=end, freq=freq, inclusive="left")
        expected = obj.to_timestamp().reindex(new_index).to_period(freq)

        # 对象重新采样并作为期望值
        result = obj.resample(freq).asfreq()
        tm.assert_almost_equal(result, expected)

        # 重新采样到时间戳，然后转换为周期，并作为期望值
        result = obj.resample(freq).asfreq().to_timestamp().to_period()
        tm.assert_almost_equal(result, expected)
    def test_asfreq_fill_value(self):
        # 测试在重采样过程中使用填充值，解决问题 3715

        # 创建一个日期周期范围索引，从 2005 年 1 月 1 日到 2005 年 1 月 10 日，频率为每天
        index = period_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="D")
        # 创建一个 Series 对象，索引为上述日期周期范围的日期，数据为索引的位置
        s = Series(range(len(index)), index=index)
        # 生成一个新的日期范围索引，从第一个索引日期的开始时间戳到最后一个索引日期的开始时间戳，频率为每小时
        new_index = date_range(
            s.index[0].to_timestamp(how="start"),
            (s.index[-1]).to_timestamp(how="start"),
            freq="1h",
        )
        # 创建预期的结果，将 Series 转换为时间戳后，根据新索引重新索引，并用填充值 4.0 填充缺失值
        expected = s.to_timestamp().reindex(new_index, fill_value=4.0)
        # 对 Series 进行时间戳转换后，按每小时频率重采样，并使用填充值 4.0
        result = s.to_timestamp().resample("1h").asfreq(fill_value=4.0)
        # 断言两个 Series 对象是否相等
        tm.assert_series_equal(result, expected)

        # 将 Series 转为 DataFrame，列名为"value"
        frame = s.to_frame("value")
        # 生成一个新的日期范围索引，从第一个索引日期的开始时间戳到最后一个索引日期的开始时间戳，频率为每小时
        new_index = date_range(
            frame.index[0].to_timestamp(how="start"),
            (frame.index[-1]).to_timestamp(how="start"),
            freq="1h",
        )
        # 创建预期的结果，将 DataFrame 转换为时间戳后，根据新索引重新索引，并用填充值 3.0 填充缺失值
        expected = frame.to_timestamp().reindex(new_index, fill_value=3.0)
        # 对 DataFrame 进行时间戳转换后，按每小时频率重采样，并使用填充值 3.0
        result = frame.to_timestamp().resample("1h").asfreq(fill_value=3.0)
        # 断言两个 DataFrame 对象是否相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("freq", ["h", "12h", "2D", "W"])
    @pytest.mark.parametrize("kwargs", [{"on": "date"}, {"level": "d"}])
    def test_selection(self, freq, kwargs):
        # 这是一个 bug，这些应该被实现
        # GitHub 14008
        # 创建一个日期周期范围索引，从 2005 年 1 月 1 日到 2005 年 1 月 10 日，频率为每天
        index = period_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="D")
        # 创建一个 numpy 数组，长度与索引相同，类型为 int64
        rng = np.arange(len(index), dtype=np.int64)
        # 创建一个 DataFrame 对象，包含列 "date" 和 "a"，使用 MultiIndex 构造多层索引
        df = DataFrame(
            {"date": index, "a": rng},
            index=pd.MultiIndex.from_arrays([rng, index], names=["v", "d"]),
        )
        # 设置异常消息内容
        msg = (
            "Resampling from level= or on= selection with a PeriodIndex is "
            r"not currently supported, use \.set_index\(\.\.\.\) to "
            "explicitly set index"
        )
        # 使用 pytest 断言抛出 NotImplementedError 异常，并匹配特定的消息
        with pytest.raises(NotImplementedError, match=msg):
            df.resample(freq, **kwargs)

    @pytest.mark.parametrize("month", MONTHS)
    @pytest.mark.parametrize("meth", ["ffill", "bfill"])
    @pytest.mark.parametrize("conv", ["start", "end"])
    @pytest.mark.parametrize(
        ("offset", "period"), [("D", "D"), ("B", "B"), ("ME", "M"), ("QE", "Q")]
    )
    def test_annual_upsample_cases(
        self, offset, period, conv, meth, month, simple_period_range_series
    ):
        # 创建一个简单的时间周期范围 Series 对象，从 1990 年 1 月 1 日到 1991 年 12 月 31 日，频率为每年减少月份
        ts = simple_period_range_series("1/1/1990", "12/31/1991", freq=f"Y-{month}")
        # 如果 period 为 "B"，则生成 FutureWarning 警告
        warn = FutureWarning if period == "B" else None
        # 设置警告消息内容
        msg = r"PeriodDtype\[B\] is deprecated"
        if warn is None:
            msg = "Resampling with a PeriodIndex is deprecated"
            warn = FutureWarning
        # 使用 tm.assert_produces_warning 断言是否生成特定警告，并匹配消息
        with tm.assert_produces_warning(warn, match=msg):
            # 对时间序列进行 period 频率的重采样，根据 conv 参数指定的约定方式
            result = getattr(ts.resample(period, convention=conv), meth)()
            # 将结果转换为时间戳，并根据 offset 和 meth 参数重新对齐，并转换为周期
            expected = result.to_timestamp(period, how=conv)
            expected = expected.asfreq(offset, meth).to_period()
        # 断言两个 Series 对象是否相等
        tm.assert_series_equal(result, expected)
    # 定义一个测试函数，用于测试基本的降采样操作
    def test_basic_downsample(self, simple_period_range_series):
        # 创建一个简单的时间序列，从"1/1/1990"到"6/30/1995"，频率为每月
        ts = simple_period_range_series("1/1/1990", "6/30/1995", freq="M")
        # 对时间序列进行年度降采样，并计算每年的均值
        result = ts.resample("Y-DEC").mean()

        # 创建预期的结果，按年度对时间序列进行分组并计算均值
        expected = ts.groupby(ts.index.year).mean()
        # 调整预期结果的索引为指定的年度区间
        expected.index = period_range("1/1/1990", "6/30/1995", freq="Y-DEC")
        # 断言结果是否与预期相等
        tm.assert_series_equal(result, expected)

        # 通过断言确保两次调用的结果相等
        tm.assert_series_equal(ts.resample("Y-DEC").mean(), result)
        tm.assert_series_equal(ts.resample("Y").mean(), result)

    # 使用参数化测试标记，定义测试不兼容的子周期规则
    @pytest.mark.parametrize(
        "rule,expected_error_msg",
        [
            ("Y-DEC", "<YearEnd: month=12>"),
            ("Q-MAR", "<QuarterEnd: startingMonth=3>"),
            ("M", "<MonthEnd>"),
            ("w-thu", "<Week: weekday=3>"),
        ],
    )
    def test_not_subperiod(self, simple_period_range_series, rule, expected_error_msg):
        # 创建一个简单的时间序列，从"1/1/1990"到"6/30/1995"，频率为每周的周三
        ts = simple_period_range_series("1/1/1990", "6/30/1995", freq="w-wed")
        # 构造错误消息，说明这些周期规则不能被重新采样到期望的周期
        msg = (
            "Frequency <Week: weekday=2> cannot be resampled to "
            f"{expected_error_msg}, as they are not sub or super periods"
        )
        # 使用断言检查是否抛出预期的异常，并匹配错误消息
        with pytest.raises(IncompatibleFrequency, match=msg):
            ts.resample(rule).mean()

    # 使用参数化测试标记，定义测试基本的升采样操作
    @pytest.mark.parametrize("freq", ["D", "2D"])
    def test_basic_upsample(self, freq, simple_period_range_series):
        # 创建一个简单的时间序列，从"1/1/1990"到"6/30/1995"，频率为每月
        ts = simple_period_range_series("1/1/1990", "6/30/1995", freq="M")
        # 对时间序列进行年度降采样，并计算每年的均值
        result = ts.resample("Y-DEC").mean()

        # 构造警告消息，指出Series.resample中的'convention'关键字已被弃用
        msg = "The 'convention' keyword in Series.resample is deprecated"
        # 使用断言确保产生未来警告，并匹配警告消息
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # 对结果进行进一步的升采样，并使用前向填充
            resampled = result.resample(freq, convention="end").ffill()
        # 将结果转换为时间戳，并按指定频率前向填充，转换为周期
        expected = result.to_timestamp(freq, how="end")
        expected = expected.asfreq(freq, "ffill").to_period(freq)
        # 断言结果是否与预期相等
        tm.assert_series_equal(resampled, expected)

    # 定义测试函数，测试带有限制条件的升采样操作
    def test_upsample_with_limit(self):
        # 创建一个具有随机正态分布值的时间序列，从"1/1/2000"开始，5个周期，频率为每年
        rng = period_range("1/1/2000", periods=5, freq="Y")
        ts = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)

        # 构造警告消息，指出Series.resample中的'convention'关键字已被弃用
        msg = "The 'convention' keyword in Series.resample is deprecated"
        # 使用断言确保产生未来警告，并匹配警告消息
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # 对时间序列进行月度升采样，并使用结束时间的前向填充，设置填充限制为2
            result = ts.resample("M", convention="end").ffill(limit=2)
        # 将原始时间序列按月重新采样，使用前向填充，设置填充限制为2
        expected = ts.asfreq("M").reindex(result.index, method="ffill", limit=2)
        # 断言结果是否与预期相等
        tm.assert_series_equal(result, expected)

    # 定义测试函数，测试年度升采样操作
    def test_annual_upsample(self, simple_period_range_series):
        # 创建一个简单的时间序列，从"1/1/1990"到"12/31/1995"，频率为每年
        ts = simple_period_range_series("1/1/1990", "12/31/1995", freq="Y-DEC")
        # 创建一个DataFrame，包含时间序列"a"
        df = DataFrame({"a": ts})
        # 对DataFrame进行日度重新采样，并使用前向填充
        rdf = df.resample("D").ffill()
        # 构造预期结果，对时间序列"a"进行日度重新采样，并使用前向填充
        exp = df["a"].resample("D").ffill()
        # 断言结果是否与预期相等
        tm.assert_series_equal(rdf["a"], exp)
    # 定义一个测试方法，用于测试年度数据上采样到月度数据的功能
    def test_annual_upsample2(self):
        # 创建一个年度的时间范围，从"2000"到"2003"，频率为每年12月结束
        rng = period_range("2000", "2003", freq="Y-DEC")
        # 创建一个时间序列，包含值[1, 2, 3, 4]，索引为上面创建的时间范围
        ts = Series([1, 2, 3, 4], index=rng)

        # 对时间序列进行月度重采样，并使用前向填充法填充缺失值
        result = ts.resample("M").ffill()
        # 创建一个期间索引，从"2000-01"到"2003-12"，频率为每月
        ex_index = period_range("2000-01", "2003-12", freq="M")

        # 使用起始时间戳的方式将时间序列转换为每月频率，然后根据扩展索引重建，并使用前向填充法填充缺失值
        expected = ts.asfreq("M", how="start").reindex(ex_index, method="ffill")
        # 断言两个时间序列是否相等
        tm.assert_series_equal(result, expected)

    # 使用pytest的参数化装饰器，为月份和约定参数化
    @pytest.mark.parametrize("month", MONTHS)
    @pytest.mark.parametrize("convention", ["start", "end"])
    # 使用pytest的参数化装饰器，为偏移和周期参数化
    @pytest.mark.parametrize(
        ("offset", "period"), [("D", "D"), ("B", "B"), ("ME", "M")]
    )
    # 定义一个测试方法，测试季度数据上采样的功能
    def test_quarterly_upsample(
        self, month, offset, period, convention, simple_period_range_series
    ):
        # 根据指定的月份创建简单的期间范围时间序列
        freq = f"Q-{month}"
        ts = simple_period_range_series("1/1/1990", "12/31/1995", freq=freq)
        # 如果周期为"B"，则发出FutureWarning警告
        warn = FutureWarning if period == "B" else None
        msg = r"PeriodDtype\[B\] is deprecated"
        if warn is None:
            # 否则，发出Resampling with a PeriodIndex is deprecated警告
            msg = "Resampling with a PeriodIndex is deprecated"
            warn = FutureWarning
        # 使用pytest的上下文管理器检查警告是否被正确触发
        with tm.assert_produces_warning(warn, match=msg):
            # 对时间序列进行偏移重采样，并使用前向填充法填充缺失值
            result = ts.resample(period, convention=convention).ffill()
            # 将结果转换为时间戳，并按照指定的偏移和前向填充法重采样
            expected = result.to_timestamp(period, how=convention)
            expected = expected.asfreq(offset, "ffill").to_period()
        # 断言两个时间序列是否相等
        tm.assert_series_equal(result, expected)

    # 使用pytest的参数化装饰器，为目标频率和约定参数化
    @pytest.mark.parametrize("target", ["D", "B"])
    @pytest.mark.parametrize("convention", ["start", "end"])
    # 定义一个测试方法，测试月度数据上采样的功能
    def test_monthly_upsample(self, target, convention, simple_period_range_series):
        # 创建一个月度频率的简单期间范围时间序列
        ts = simple_period_range_series("1/1/1990", "12/31/1995", freq="M")

        # 如果目标频率为"D"，则警告为None；否则发出FutureWarning警告
        warn = None if target == "D" else FutureWarning
        msg = r"PeriodDtype\[B\] is deprecated"
        if warn is None:
            # 否则，发出Resampling with a PeriodIndex is deprecated警告
            msg = "Resampling with a PeriodIndex is deprecated"
            warn = FutureWarning
        # 使用pytest的上下文管理器检查警告是否被正确触发
        with tm.assert_produces_warning(warn, match=msg):
            # 对时间序列进行目标频率重采样，并使用前向填充法填充缺失值
            result = ts.resample(target, convention=convention).ffill()
            # 将结果转换为时间戳，并按照指定的目标频率和前向填充法重采样
            expected = result.to_timestamp(target, how=convention)
            expected = expected.asfreq(target, "ffill").to_period()
        # 断言两个时间序列是否相等
        tm.assert_series_equal(result, expected)

    # 定义一个测试方法，测试基本的重采样功能
    def test_resample_basic(self):
        # 创建一个包含100个浮点数的时间序列，索引为每秒的时间范围，从"20130101"开始，共100秒
        s = Series(
            range(100),
            index=date_range("20130101", freq="s", periods=100, name="idx"),
            dtype="float",
        )
        # 将索引为10到30的值设为NaN
        s[10:30] = np.nan
        # 创建一个包含两个分钟周期的期间索引
        index = PeriodIndex(
            [Period("2013-01-01 00:00", "min"), Period("2013-01-01 00:01", "min")],
            name="idx",
        )
        # 创建预期的结果，计算每分钟的均值
        expected = Series([34.5, 79.5], index=index)
        result = s.to_period().resample("min").mean()
        # 断言两个时间序列是否相等
        tm.assert_series_equal(result, expected)
        result2 = s.resample("min").mean().to_period()
        # 断言两个时间序列是否相等
        tm.assert_series_equal(result2, expected)

    # 使用pytest的参数化装饰器，为频率和预期值参数化
    @pytest.mark.parametrize(
        "freq,expected_vals", [("M", [31, 29, 31, 9]), ("2M", [31 + 29, 31 + 9])]
    )
    # 定义一个测试函数，测试时间序列重采样后计数是否正确
    def test_resample_count(self, freq, expected_vals):
        # GH12774: 测试用例标识符
        series = Series(1, index=period_range(start="2000", periods=100))
        # 对时间序列进行重采样，并计算每个时间段的计数
        result = series.resample(freq).count()
        # 根据预期值创建时间序列索引
        expected_index = period_range(
            start="2000", freq=freq, periods=len(expected_vals)
        )
        # 创建预期的时间序列
        expected = Series(expected_vals, index=expected_index)
        # 使用测试框架检查计算结果与预期结果是否相等
        tm.assert_series_equal(result, expected)

    # 定义一个测试函数，测试时间序列重采样是否保持相同频率
    def test_resample_same_freq(self, resample_method):
        # GH12770: 测试用例标识符
        series = Series(range(3), index=period_range(start="2000", periods=3, freq="M"))
        # 将时间序列按月重采样，并使用指定的重采样方法处理
        result = getattr(series.resample("M"), resample_method)()
        # 预期结果与原时间序列相同
        expected = series
        # 使用测试框架检查计算结果与预期结果是否相等
        tm.assert_series_equal(result, expected)

    # 定义一个测试函数，测试不兼容的时间序列重采样频率是否引发异常
    def test_resample_incompat_freq(self):
        # 错误消息内容
        msg = (
            "Frequency <MonthEnd> cannot be resampled to <Week: weekday=6>, "
            "as they are not sub or super periods"
        )
        # 创建一个月频率的时间索引
        pi = period_range(start="2000", periods=3, freq="M")
        # 创建一个包含数据的时间序列
        ser = Series(range(3), index=pi)
        # 对时间序列进行周频率的重采样
        rs = ser.resample("W")
        # 使用测试框架检查重采样时是否引发了指定异常
        with pytest.raises(IncompatibleFrequency, match=msg):
            # TODO: 应该在重采样调用时引发异常，而不是在均值计算时
            rs.mean()

    @pytest.mark.parametrize(
        "tz",
        [
            zoneinfo.ZoneInfo("America/Los_Angeles"),
            dateutil.tz.gettz("America/Los_Angeles"),
        ],
    )
    # 定义一个测试函数，测试在本地时区进行时间序列重采样的情况
    def test_with_local_timezone(self, tz):
        # GH-5430: 测试用例标识符
        local_timezone = tz

        # 创建起始时间和结束时间，带有时区信息（UTC）
        start = datetime(
            year=2013, month=11, day=1, hour=0, minute=0, tzinfo=timezone.utc
        )
        end = datetime(
            year=2013, month=11, day=2, hour=0, minute=0, tzinfo=timezone.utc
        )

        # 创建一个包含小时频率索引的时间序列
        index = date_range(start, end, freq="h", name="idx")

        # 创建时间序列，每个时间点的值为1
        series = Series(1, index=index)
        # 将时间序列转换为指定本地时区
        series = series.tz_convert(local_timezone)
        # 设置警告消息内容
        msg = "Converting to PeriodArray/Index representation will drop timezone"
        # 测试重采样后计算结果是否引发指定警告
        with tm.assert_produces_warning(UserWarning, match=msg):
            result = series.resample("D").mean().to_period()

        # 创建预期的时间序列
        # 使用本地时区转换后的时间，将索引移回一天，从UTC到Pacific时区
        expected_index = (
            period_range(start=start, end=end, freq="D", name="idx") - offsets.Day()
        )
        expected = Series(1.0, index=expected_index)
        # 使用测试框架检查计算结果与预期结果是否相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "tz",
        [
            zoneinfo.ZoneInfo("America/Los_Angeles"),
            dateutil.tz.gettz("America/Los_Angeles"),
        ],
    )
    def test_resample_with_tz(self, tz, unit):
        # GH 13238
        # 创建一个带有时区和单位的日期时间索引，每小时频率，从 "2017-01-01" 开始，48个时间点
        dti = date_range("2017-01-01", periods=48, freq="h", tz=tz, unit=unit)
        # 创建一个 Series 对象，值为2，索引为上面创建的日期时间索引
        ser = Series(2, index=dti)
        # 对该 Series 进行重采样，将频率从小时 ("h") 转换为天 ("D")，并计算每天的均值
        result = ser.resample("D").mean()
        # 期望的日期时间索引，使用指定的时区和频率单位
        exp_dti = pd.DatetimeIndex(
            ["2017-01-01", "2017-01-02"], tz=tz, freq="D"
        ).as_unit(unit)
        # 创建一个期望的 Series 对象，每个日期时间点的值都为2.0
        expected = Series(
            2.0,
            index=exp_dti,
        )
        # 使用测试工具比较结果 Series 和期望的 Series 是否相等
        tm.assert_series_equal(result, expected)

    def test_resample_nonexistent_time_bin_edge(self):
        # GH 19375
        # 创建一个日期时间索引，从 "2017-03-12" 开始，到 "2017-03-12 1:45:00"，15分钟频率
        index = date_range("2017-03-12", "2017-03-12 1:45:00", freq="15min")
        # 创建一个 Series 对象，所有值为0，索引为上述的日期时间索引
        s = Series(np.zeros(len(index)), index=index)
        # 将 Series 设置时区为 "US/Pacific"
        expected = s.tz_localize("US/Pacific")
        # 将索引转换为指定的频率 "900s"（15分钟），并进行均值重采样
        expected.index = pd.DatetimeIndex(expected.index, freq="900s")
        result = expected.resample("900s").mean()
        # 使用测试工具比较结果 Series 和期望的 Series 是否相等
        tm.assert_series_equal(result, expected)

    def test_resample_nonexistent_time_bin_edge2(self):
        # GH 23742
        # 创建一个从 "2017-10-10" 到 "2017-10-20" 的日期时间索引，1小时频率
        index = date_range(start="2017-10-10", end="2017-10-20", freq="1h")
        # 将索引设置为 "UTC" 时区，然后转换为 "America/Sao_Paulo" 时区
        index = index.tz_localize("UTC").tz_convert("America/Sao_Paulo")
        # 创建一个 DataFrame 对象，索引为上述的日期时间索引，数据为索引的长度
        df = DataFrame(data=list(range(len(index))), index=index)
        # 对 DataFrame 进行按天 ("1D") 分组，计算每天的计数
        result = df.groupby(pd.Grouper(freq="1D")).count()
        # 期望的日期时间索引，从 "2017-10-09" 到 "2017-10-20"，每天频率，指定了时区和非存在时间的处理方式
        expected = date_range(
            start="2017-10-09",
            end="2017-10-20",
            freq="D",
            tz="America/Sao_Paulo",
            nonexistent="shift_forward",
            inclusive="left",
        )
        # 使用测试工具比较结果 DataFrame 的索引和期望的日期时间索引是否相等
        tm.assert_index_equal(result.index, expected)

    def test_resample_ambiguous_time_bin_edge(self):
        # GH 10117
        # 创建一个从 "2014-10-25 22:00:00" 到 "2014-10-26 00:30:00" 的日期时间索引，30分钟频率，时区为 "Europe/London"
        idx = date_range(
            "2014-10-25 22:00:00",
            "2014-10-26 00:30:00",
            freq="30min",
            tz="Europe/London",
        )
        # 创建一个 Series 对象，所有值为0，索引为上述的日期时间索引
        expected = Series(np.zeros(len(idx)), index=idx)
        # 对 Series 进行均值重采样，频率为 "30min"
        result = expected.resample("30min").mean()
        # 使用测试工具比较结果 Series 和期望的 Series 是否相等
        tm.assert_series_equal(result, expected)

    def test_fill_method_and_how_upsample(self):
        # GH2073
        # 创建一个 Series 对象，索引为从 "2010-01-01" 开始，9个时间点，四半期频率
        s = Series(
            np.arange(9, dtype="int64"),
            index=date_range("2010-01-01", periods=9, freq="QE"),
        )
        # 对 Series 进行均值向前填充 ("ffill") 的月末重采样 ("ME")
        last = s.resample("ME").ffill()
        # 连续两次对 Series 进行均值向前填充和最后值采样，结果转换为 int64 类型
        both = s.resample("ME").ffill().resample("ME").last().astype("int64")
        # 使用测试工具比较最后的 Series 和连续两次重采样后的 Series 是否相等
        tm.assert_series_equal(last, both)

    @pytest.mark.parametrize("day", DAYS)
    @pytest.mark.parametrize("target", ["D", "B"])
    @pytest.mark.parametrize("convention", ["start", "end"])
    def test_weekly_upsample(self, day, target, convention, simple_period_range_series):
        # 构建频率字符串，例如 "W-WED"
        freq = f"W-{day}"
        # 使用给定的频率创建时间序列
        ts = simple_period_range_series("1/1/1990", "12/31/1995", freq=freq)

        # 初始化警告信息为 None，如果目标不是 "D"，则设置为 FutureWarning
        warn = None if target == "D" else FutureWarning
        # 初始化警告消息，用于匹配未来警告
        msg = r"PeriodDtype\[B\] is deprecated"
        if warn is None:
            # 如果警告为 None，则更新警告消息
            msg = "Resampling with a PeriodIndex is deprecated"
            # 设置警告类型为 FutureWarning
            warn = FutureWarning
        # 断言操作过程中产生的警告与预期的消息匹配
        with tm.assert_produces_warning(warn, match=msg):
            # 对时间序列进行重新取样，并向前填充缺失值
            result = ts.resample(target, convention=convention).ffill()
            # 将结果转换为时间戳并按指定方式重新取样
            expected = result.to_timestamp(target, how=convention)
            # 将时间戳数据按照指定频率重新取样并向前填充缺失值，并转换为周期数据
            expected = expected.asfreq(target, "ffill").to_period()
        # 断言结果和预期相等
        tm.assert_series_equal(result, expected)

    def test_resample_to_timestamps(self, simple_period_range_series):
        # 使用简单的时间范围创建时间序列
        ts = simple_period_range_series("1/1/1990", "12/31/1995", freq="M")

        # 对时间序列进行年末重新取样，并计算平均值后转换为时间戳
        result = ts.resample("Y-DEC").mean().to_timestamp()
        # 对时间序列进行年末重新取样，计算平均值后以开始日期为准转换为时间戳
        expected = ts.resample("Y-DEC").mean().to_timestamp(how="start")
        # 断言结果和预期相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("month", MONTHS)
    def test_resample_to_quarterly(self, simple_period_range_series, month):
        # 使用简单的时间范围创建时间序列，以指定的年和月频率
        ts = simple_period_range_series("1990", "1992", freq=f"Y-{month}")
        # 对时间序列进行季度重新取样，并向前填充缺失值
        quar_ts = ts.resample(f"Q-{month}").ffill()

        # 将时间序列索引转换为时间戳，以开始日期为准
        stamps = ts.to_timestamp("D", how="start")
        # 创建指定频率的日期范围
        qdates = period_range(
            ts.index[0].asfreq("D", "start"),
            ts.index[-1].asfreq("D", "end"),
            freq=f"Q-{month}",
        )

        # 根据时间戳重新索引预期值，并使用前向填充方法
        expected = stamps.reindex(qdates.to_timestamp("D", "s"), method="ffill")
        expected.index = qdates

        # 断言季度重新取样的结果和预期相等
        tm.assert_series_equal(quar_ts, expected)

    @pytest.mark.parametrize("how", ["start", "end"])
    def test_resample_to_quarterly_start_end(self, simple_period_range_series, how):
        # 使用简单的时间范围创建时间序列，以指定的年和月频率（六月为结束）
        ts = simple_period_range_series("1990", "1992", freq="Y-JUN")
        # 设置警告消息，用于匹配未来警告
        msg = "The 'convention' keyword in Series.resample is deprecated"
        # 断言操作过程中产生的警告与预期的消息匹配
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # 对时间序列进行三月季度重新取样，并向前填充缺失值
            result = ts.resample("Q-MAR", convention=how).ffill()
        # 根据指定方式重新取样时间序列，并使用前向填充方法
        expected = ts.asfreq("Q-MAR", how=how)
        expected = expected.reindex(result.index, method="ffill")

        # 断言结果和预期相等
        tm.assert_series_equal(result, expected)

    def test_resample_fill_missing(self):
        # 创建包含四个年份的周期索引
        rng = PeriodIndex([2000, 2005, 2007, 2009], freq="Y")

        # 创建具有随机标准正态分布数据的系列
        s = Series(np.random.default_rng(2).standard_normal(4), index=rng)

        # 将周期索引转换为时间戳
        stamps = s.to_timestamp()
        # 对时间序列进行年度重新取样，并向前填充缺失值
        filled = s.resample("Y").ffill()
        # 根据年度结束日期重新取样时间戳，并向前填充缺失值，再转换为周期数据
        expected = stamps.resample("YE").ffill().to_period("Y")
        # 断言填充后的结果和预期相等
        tm.assert_series_equal(filled, expected)
    # 定义一个测试方法，用于验证在存在重复时间索引时无法填充缺失值
    def test_cant_fill_missing_dups(self):
        # 创建一个时间索引对象，包含多个重复的年份
        rng = PeriodIndex([2000, 2005, 2005, 2007, 2007], freq="Y")
        # 创建一个随机数据的时间序列，使用标准正态分布生成数据
        s = Series(np.random.default_rng(2).standard_normal(5), index=rng)
        # 当尝试对年度频率重新采样为年度频率时，使用前向填充方法，期望触发异常
        msg = "Reindexing only valid with uniquely valued Index objects"
        with pytest.raises(InvalidIndexError, match=msg):
            s.resample("Y").ffill()

    # 定义一个测试方法，验证时间序列的5分钟重采样操作
    def test_resample_5minute(self):
        # 创建一个分钟频率的时间索引范围
        rng = period_range("1/1/2000", "1/5/2000", freq="min")
        # 创建一个随机数据的时间序列，使用标准正态分布生成数据，并指定索引为rng
        ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
        # 期望结果：将时间序列转换为时间戳后，按5分钟频率重采样并计算均值
        expected = ts.to_timestamp().resample("5min").mean()
        # 实际结果：直接对时间序列按5分钟频率重采样并计算均值，再转换回时间戳
        result = ts.resample("5min").mean().to_timestamp()
        # 使用测试框架验证结果是否相等
        tm.assert_series_equal(result, expected)

        # 期望结果：将时间序列按5分钟频率重采样并计算均值，然后转换为5分钟周期
        expected = expected.to_period("5min")
        # 实际结果：直接对时间序列按5分钟频率重采样并计算均值
        result = ts.resample("5min").mean()
        # 使用测试框架验证结果是否相等
        tm.assert_series_equal(result, expected)
        # 实际结果：对时间序列按5分钟频率重采样并计算均值，然后转换为时间戳，再转换为周期
        result = ts.resample("5min").mean().to_timestamp().to_period()
        # 使用测试框架验证结果是否相等
        tm.assert_series_equal(result, expected)

    # 定义一个测试方法，验证按工作日重采样时的上采样操作
    def test_upsample_daily_business_daily(self, simple_period_range_series):
        # 创建一个简单的时间序列，时间范围为2000年1月1日至2000年2月1日，频率为工作日
        ts = simple_period_range_series("1/1/2000", "2/1/2000", freq="B")

        # 期望结果：将时间序列按日频率重采样，并将缺失的日期填充为NaN
        result = ts.resample("D").asfreq()
        # 期望结果：将时间序列按日频率重采样，并重新索引到指定日期范围
        expected = ts.asfreq("D").reindex(period_range("1/3/2000", "2/1/2000"))
        # 使用测试框架验证结果是否相等
        tm.assert_series_equal(result, expected)

        # 创建一个简单的时间序列，时间范围为2000年1月1日至2000年2月1日
        ts = simple_period_range_series("1/1/2000", "2/1/2000")
        # 期望警告消息：Series.resample 中 'convention' 关键字已废弃
        msg = "The 'convention' keyword in Series.resample is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # 期望结果：将时间序列按小时频率重采样，并使用'start'方法填充缺失值
            result = ts.resample("h", convention="s").asfreq()
        # 期望结果：将时间序列按小时频率重采样，并重新索引到指定日期范围
        exp_rng = period_range("1/1/2000", "2/1/2000 23:00", freq="h")
        expected = ts.asfreq("h", how="s").reindex(exp_rng)
        # 使用测试框架验证结果是否相等
        tm.assert_series_equal(result, expected)

    # 定义一个测试方法，验证在不规则时间序列上的稀疏重采样操作
    def test_resample_irregular_sparse(self):
        # 创建一个不规则时间索引，从2012年1月1日开始，5分钟频率，共1000个时间点
        dr = date_range(start="1/1/2012", freq="5min", periods=1000)
        # 创建一个值全为100的时间序列，索引为dr
        s = Series(np.array(100), index=dr)
        # 从时间序列中选取部分数据
        subset = s[:"2012-01-04 06:55"]

        # 期望结果：将子序列按10分钟频率重采样，并应用长度函数（即计算每个时间段的数据点数）
        result = subset.resample("10min").apply(len)
        # 期望结果：将原始序列按10分钟频率重采样，并应用长度函数，然后选取结果的对应部分
        expected = s.resample("10min").apply(len).loc[result.index]
        # 使用测试框架验证结果是否相等
        tm.assert_series_equal(result, expected)

    # 定义一个测试方法，验证在周频率上的稀疏重采样操作，所有数据为NaN
    def test_resample_weekly_all_na(self):
        # 创建一个每周三的日期索引，从2000年1月1日开始，共10个时间点
        rng = date_range("1/1/2000", periods=10, freq="W-WED")
        # 创建一个随机数据的时间序列，使用标准正态分布生成数据，并指定索引为rng
        ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

        # 期望结果：将时间序列按每周四的频率重采样，并将所有值填充为NaN
        result = ts.resample("W-THU").asfreq()

        # 断言：检查结果序列中所有值是否都为NaN
        assert result.isna().all()

        # 期望结果：将时间序列按每周四的频率重采样，并前向填充缺失值，然后去掉最后一个时间点
        result = ts.resample("W-THU").asfreq().ffill()[:-1]
        # 期望结果：将时间序列按每周四的频率重采样，并前向填充缺失值
        expected = ts.asfreq("W-THU").ffill()
        # 使用测试框架验证结果是否相等
        tm.assert_series_equal(result, expected)
    # 定义一个测试方法，用于测试时区本地化后的时间序列重采样
    def test_resample_tz_localized(self, unit):
        # 创建一个日期范围，从"2012-4-13"到"2012-5-1"，使用给定的时间单位
        dr = date_range(start="2012-4-13", end="2012-5-1", unit=unit)
        # 根据日期范围创建一个序列，索引为日期范围，值为序列索引
        ts = Series(range(len(dr)), index=dr)

        # 将序列转换为UTC时区
        ts_utc = ts.tz_localize("UTC")
        # 将UTC时区的序列转换为"America/Los_Angeles"时区
        ts_local = ts_utc.tz_convert("America/Los_Angeles")

        # 对本地化时区后的序列进行周重采样，求均值
        result = ts_local.resample("W").mean()

        # 创建本地化时区的序列的一个深拷贝
        ts_local_naive = ts_local.copy()
        # 清除深拷贝序列的时区信息
        ts_local_naive.index = ts_local_naive.index.tz_localize(None)

        # 对清除时区信息的序列进行周重采样，求均值，并重新设定时区为"America/Los_Angeles"
        exp = ts_local_naive.resample("W").mean().tz_localize("America/Los_Angeles")
        # 将重采样后的索引转换为DatetimeIndex，频率为每周
        exp.index = pd.DatetimeIndex(exp.index, freq="W")

        # 使用测试工具验证结果序列与期望序列是否相等
        tm.assert_series_equal(result, exp)

        # 证实工作正常
        result = ts_local.resample("D").mean()

    # 定义另一个测试方法，用于测试另一种时区本地化后的时间序列重采样情况
    def test_resample_tz_localized2(self):
        # 创建一个日期范围，从"2001-09-20 15:59"到"2001-09-20 16:00"，频率为每分钟，时区为"Australia/Sydney"
        idx = date_range(
            "2001-09-20 15:59", "2001-09-20 16:00", freq="min", tz="Australia/Sydney"
        )
        # 创建一个序列，包含索引和数据
        s = Series([1, 2], index=idx)

        # 对序列进行每日重采样，求均值，设置关闭右侧区间，标签为右侧，生成结果
        result = s.resample("D", closed="right", label="right").mean()
        # 创建期望的序列，仅包含一个值1.5，索引为"2001-09-21"，时区为"Australia/Sydney"
        ex_index = date_range("2001-09-21", periods=1, freq="D", tz="Australia/Sydney")
        expected = Series([1.5], index=ex_index)

        # 使用测试工具验证结果序列与期望序列是否相等
        tm.assert_series_equal(result, expected)

        # 为了充分考虑
        # 在转换为PeriodArray/Index表示时会丢失时区的警告消息
        msg = "Converting to PeriodArray/Index representation will drop timezone "
        # 使用测试工具验证警告是否产生
        with tm.assert_produces_warning(UserWarning, match=msg):
            # 对序列进行每日重采样，求均值，然后转换为Period类型
            result = s.resample("D").mean().to_period()
        # 创建期望的序列，仅包含一个值1.5，索引为"2001-09-20"，频率为每日
        ex_index = period_range("2001-09-20", periods=1, freq="D")
        expected = Series([1.5], index=ex_index)
        
        # 使用测试工具验证结果序列与期望序列是否相等
        tm.assert_series_equal(result, expected)

    # 定义第三个测试方法，用于测试不同偏移量且不保留时区信息的情况
    def test_resample_tz_localized3(self):
        # 创建一个从"1/1/2011"开始的日期范围，包含20000个时间点，频率为每小时，时区为"EST"
        rng = date_range("1/1/2011", periods=20000, freq="h")
        # 将日期范围设置为"EST"时区
        rng = rng.tz_localize("EST")
        # 创建一个数据帧，索引为日期范围
        ts = DataFrame(index=rng)
        # 向数据帧添加两列数据，第一列为标准正态分布的随机数，第二列为累积随机数
        ts["first"] = np.random.default_rng(2).standard_normal(len(rng))
        ts["second"] = np.cumsum(np.random.default_rng(2).standard_normal(len(rng)))

        # 创建期望的数据帧，包含"first"列的年度总和和"second"列的年度均值
        expected = DataFrame(
            {
                "first": ts.resample("YE").sum()["first"],
                "second": ts.resample("YE").mean()["second"],
            },
            columns=["first", "second"],
        )
        # 对数据帧进行年度重采样，分别计算"first"列的总和和"second"列的均值，并重新索引列
        result = (
            ts.resample("YE")
            .agg({"first": "sum", "second": "mean"})
            .reindex(columns=["first", "second"])
        )

        # 使用测试工具验证结果数据帧与期望数据帧是否相等
        tm.assert_frame_equal(result, expected)
    def test_closed_left_corner(self):
        # #1465
        # 创建一个包含21个随机标准正态分布值的Series对象，索引为从"1/1/2012 9:30"开始，每分钟一个数据点，共21个数据点
        s = Series(
            np.random.default_rng(2).standard_normal(21),
            index=date_range(start="1/1/2012 9:30", freq="1min", periods=21),
        )
        # 将第一个数据点设为NaN
        s.iloc[0] = np.nan

        # 对Series对象进行10分钟的重采样，使用左闭右开区间，标签设为右边界，计算每个区间的均值
        result = s.resample("10min", closed="left", label="right").mean()
        # 期望的重采样结果，去除第一个数据点后，使用左闭右开区间，标签设为右边界，计算每个区间的均值
        exp = s[1:].resample("10min", closed="left", label="right").mean()
        # 断言结果与期望一致
        tm.assert_series_equal(result, exp)

        # 对Series对象进行10分钟的重采样，使用左闭右开区间，标签设为左边界，计算每个区间的均值
        result = s.resample("10min", closed="left", label="left").mean()
        # 期望的重采样结果，去除第一个数据点后，使用左闭右开区间，标签设为左边界，计算每个区间的均值
        exp = s[1:].resample("10min", closed="left", label="left").mean()

        # 期望的索引，从"1/1/2012 9:30"开始，每10分钟一个数据点，共3个数据点
        ex_index = date_range(start="1/1/2012 9:30", freq="10min", periods=3)

        # 断言重采样后的索引与期望的索引一致
        tm.assert_index_equal(result.index, ex_index)
        # 断言结果与期望一致
        tm.assert_series_equal(result, exp)

    def test_quarterly_resampling(self):
        # 创建一个包含10个整数的Series对象，索引为每季度的时间段，频率为每年以12月结束的季度
        rng = period_range("2000Q1", periods=10, freq="Q-DEC")
        ts = Series(np.arange(10), index=rng)

        # 对Series对象进行年度重采样，计算每年的均值
        result = ts.resample("Y").mean()
        # 将Series对象转换为时间戳，并进行年度重采样，计算每年的均值，再转换回季度频率
        exp = ts.to_timestamp().resample("YE").mean().to_period()
        # 断言结果与期望一致
        tm.assert_series_equal(result, exp)

    def test_resample_weekly_bug_1726(self):
        # 创建一个日期范围从"8/6/2012"到"8/26/2012"的DatetimeIndex对象，频率为每天
        ind = date_range(start="8/6/2012", end="8/26/2012", freq="D")
        n = len(ind)
        # 创建一个DataFrame对象，包含从0到n-1的整数，每列重复5次，列名为["open", "high", "low", "close", "vol"]，索引为ind
        data = [[x] * 5 for x in range(n)]
        df = DataFrame(data, columns=["open", "high", "low", "close", "vol"], index=ind)

        # 对DataFrame对象进行周重采样，使用左闭右开区间，标签设为左边界，取每个区间的第一个值
        # 这里的"it works!"是注释，不需要翻译
        df.resample("W-MON", closed="left", label="left").first()

    def test_resample_with_dst_time_change(self):
        # GH 15549
        # 创建一个带有时区信息的DatetimeIndex对象，包含两个时间戳
        index = (
            pd.DatetimeIndex([1457537600000000000, 1458059600000000000])
            .tz_localize("UTC")
            .tz_convert("America/Chicago")
        )
        # 创建一个包含两个整数的DataFrame对象，索引为index
        df = DataFrame([1, 2], index=index)
        # 对DataFrame对象进行12小时重采样，使用右闭左开区间，标签设为右边界，取每个区间的最后一个非空值并向前填充缺失值
        result = df.resample("12h", closed="right", label="right").last().ffill()

        # 期望的索引值列表，转换为时区"America/Chicago"并设定频率为12小时
        expected_index_values = [
            "2016-03-09 12:00:00-06:00",
            "2016-03-10 00:00:00-06:00",
            "2016-03-10 12:00:00-06:00",
            "2016-03-11 00:00:00-06:00",
            "2016-03-11 12:00:00-06:00",
            "2016-03-12 00:00:00-06:00",
            "2016-03-12 12:00:00-06:00",
            "2016-03-13 00:00:00-06:00",
            "2016-03-13 13:00:00-05:00",
            "2016-03-14 01:00:00-05:00",
            "2016-03-14 13:00:00-05:00",
            "2016-03-15 01:00:00-05:00",
            "2016-03-15 13:00:00-05:00",
        ]
        # 转换为时区"America/Chicago"并设定频率为12小时的DatetimeIndex对象
        index = (
            pd.to_datetime(expected_index_values, utc=True)
            .tz_convert("America/Chicago")
            .as_unit(index.unit)
        )
        index = pd.DatetimeIndex(index, freq="12h")
        # 创建一个DataFrame对象，包含期望的值列表，索引为index
        expected = DataFrame(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
            index=index,
        )
        # 断言结果与期望一致
        tm.assert_frame_equal(result, expected)
    def test_resample_bms_2752(self):
        # GH2753
        # 创建一个时间序列，索引为2000年1月1日到2000年2月1日的工作日日期范围，数据类型为float64
        timeseries = Series(
            index=pd.bdate_range("20000101", "20000201"), dtype=np.float64
        )
        # 对时间序列进行“BMS”频率重新取样，计算每月第一个工作日的平均值
        res1 = timeseries.resample("BMS").mean()
        # 连续对时间序列进行两次重新取样，首先从“BMS”到“B”，然后再从“B”到日频，计算各段时间的平均值
        res2 = timeseries.resample("BMS").mean().resample("B").mean()
        # 断言：第一个重新取样结果的第一个索引日期应为2000年1月3日
        assert res1.index[0] == Timestamp("20000103")
        # 断言：两次重新取样结果的第一个索引日期应相同
        assert res1.index[0] == res2.index[0]

    @pytest.mark.xfail(reason="Commented out for more than 3 years. Should this work?")
    def test_monthly_convention_span(self):
        # 创建一个时间段索引，从2000年1月开始，周期为3个月，频率为“ME”（月末）
        rng = period_range("2000-01", periods=3, freq="ME")
        # 创建一个Series，索引为rng，数据为0到2
        ts = Series(np.arange(3), index=rng)

        # 通过重新取样和填充方式获得预期的Series对象
        # 首先创建期望的时间索引范围，从2000年1月1日到2000年3月31日，频率为日
        exp_index = period_range("2000-01-01", "2000-03-31", freq="D")
        # 使用"end"方法将时间序列按日频率对齐，然后重新索引为期望的时间范围，并使用后向填充的方式填充缺失值
        expected = ts.asfreq("D", how="end").reindex(exp_index)
        expected = expected.fillna(method="bfill")

        # 对时间序列进行日频率重新取样，计算各日的平均值
        result = ts.resample("D").mean()

        # 断言：重新取样结果应与预期的结果相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "from_freq, to_freq", [("D", "ME"), ("QE", "YE"), ("ME", "QE"), ("D", "W")]
    )
    def test_default_right_closed_label(self, from_freq, to_freq):
        # 创建一个日期范围索引，从"8/15/2012"开始，包含100个日期，频率由from_freq指定
        idx = date_range(start="8/15/2012", periods=100, freq=from_freq)
        # 创建一个DataFrame，形状为(len(idx), 2)，数据服从标准正态分布
        df = DataFrame(np.random.default_rng(2).standard_normal((len(idx), 2)), idx)

        # 对DataFrame进行目标频率to_freq的重新取样，计算各段时间的平均值
        resampled = df.resample(to_freq).mean()
        # 断言：重新取样结果应与使用右闭合和标签为右边的方式取样的结果相等
        tm.assert_frame_equal(
            resampled, df.resample(to_freq, closed="right", label="right").mean()
        )

    @pytest.mark.parametrize(
        "from_freq, to_freq",
        [("D", "MS"), ("QE", "YS"), ("ME", "QS"), ("h", "D"), ("min", "h")],
    )
    def test_default_left_closed_label(self, from_freq, to_freq):
        # 创建一个日期范围索引，从"8/15/2012"开始，包含100个日期，频率由from_freq指定
        idx = date_range(start="8/15/2012", periods=100, freq=from_freq)
        # 创建一个DataFrame，形状为(len(idx), 2)，数据服从标准正态分布
        df = DataFrame(np.random.default_rng(2).standard_normal((len(idx), 2)), idx)

        # 对DataFrame进行目标频率to_freq的重新取样，计算各段时间的平均值
        resampled = df.resample(to_freq).mean()
        # 断言：重新取样结果应与使用左闭合和标签为左边的方式取样的结果相等
        tm.assert_frame_equal(
            resampled, df.resample(to_freq, closed="left", label="left").mean()
        )

    def test_all_values_single_bin(self):
        # GH#2070
        # 创建一个时间段索引，从"2012-01-01"到"2012-12-31"，频率为月末
        index = period_range(start="2012-01-01", end="2012-12-31", freq="M")
        # 创建一个Series，索引为index，数据为服从标准正态分布的随机数
        ser = Series(np.random.default_rng(2).standard_normal(len(index)), index=index)

        # 对Series进行年度频率的重新取样，计算每年的平均值
        result = ser.resample("Y").mean()
        # 断言：重新取样结果的第一个值应与原始Series的平均值接近
        tm.assert_almost_equal(result.iloc[0], ser.mean())

    def test_evenly_divisible_with_no_extra_bins(self):
        # GH#4076
        # 当频率可以整除时，有时会有额外的箱子

        # 创建一个DataFrame，形状为(9, 3)，数据服从标准正态分布，索引为从"2000-1-1"开始的9天日期
        df = DataFrame(
            np.random.default_rng(2).standard_normal((9, 3)),
            index=date_range("2000-1-1", periods=9),
        )
        # 对DataFrame进行5天的频率重新取样，计算各段时间的平均值
        result = df.resample("5D").mean()
        # 创建预期的DataFrame，由前5行和后4行的均值构成
        expected = pd.concat([df.iloc[0:5].mean(), df.iloc[5:].mean()], axis=1).T
        # 设定预期DataFrame的索引为特定的日期索引，频率为5天
        expected.index = pd.DatetimeIndex(
            [Timestamp("2000-1-1"), Timestamp("2000-1-6")], dtype="M8[ns]", freq="5D"
        )
        # 断言：重新取样结果应与预期的结果相等
        tm.assert_frame_equal(result, expected)
    # 定义一个测试函数，用于测试数据的分组和聚合操作，验证预期的输出是否与实际一致
    def test_evenly_divisible_with_no_extra_bins2(self):
        # 创建一个日期范围对象，从指定日期开始，包含28个时间点
        index = date_range(start="2001-5-4", periods=28)
        # 创建一个包含两组数据的 DataFrame：
        # 第一组包含 REST_KEY 为 1 的数据重复28次，每行包括多个字段和相应数值
        # 第二组包含 REST_KEY 为 2 的数据重复28次，每行包括多个字段和相应数值
        df = DataFrame(
            [
                {
                    "REST_KEY": 1,
                    "DLY_TRN_QT": 80,
                    "DLY_SLS_AMT": 90,
                    "COOP_DLY_TRN_QT": 30,
                    "COOP_DLY_SLS_AMT": 20,
                }
            ]
            * 28
            + [
                {
                    "REST_KEY": 2,
                    "DLY_TRN_QT": 70,
                    "DLY_SLS_AMT": 10,
                    "COOP_DLY_TRN_QT": 50,
                    "COOP_DLY_SLS_AMT": 20,
                }
            ]
            * 28,
            index=index.append(index),
        ).sort_index()

        # 重新定义日期范围，以7天为周期，创建一个预期的 DataFrame
        index = date_range("2001-5-4", periods=4, freq="7D")
        expected = DataFrame(
            [
                {
                    "REST_KEY": 14,
                    "DLY_TRN_QT": 14,
                    "DLY_SLS_AMT": 14,
                    "COOP_DLY_TRN_QT": 14,
                    "COOP_DLY_SLS_AMT": 14,
                }
            ]
            * 4,
            index=index,
        )
        # 对原始数据进行按7天周期的重采样，并将结果与预期的 DataFrame 进行比较
        result = df.resample("7D").count()
        tm.assert_frame_equal(result, expected)

        # 创建另一个预期的 DataFrame，以7天为周期，包含对应列的总和
        expected = DataFrame(
            [
                {
                    "REST_KEY": 21,
                    "DLY_TRN_QT": 1050,
                    "DLY_SLS_AMT": 700,
                    "COOP_DLY_TRN_QT": 560,
                    "COOP_DLY_SLS_AMT": 280,
                }
            ]
            * 4,
            index=index,
        )
        # 对原始数据进行按7天周期的重采样，并将结果与预期的 DataFrame 进行比较
        result = df.resample("7D").sum()
        tm.assert_frame_equal(result, expected)

    # 使用 pytest 的参数化装饰器，定义一个测试函数，测试上采样和 OHLC 数据的处理
    @pytest.mark.parametrize("freq, period_mult", [("h", 24), ("12h", 2)])
    def test_upsampling_ohlc(self, freq, period_mult):
        # 创建一个日期范围对象，从指定日期开始，包含10个时间点
        pi = period_range(start="2000", freq="D", periods=10)
        # 创建一个序列，包含一系列数字，索引为日期范围对象 pi
        s = Series(range(len(pi)), index=pi)
        # 对序列进行时间戳转换，并按指定频率进行重采样，计算 OHLC 数据，最后转换为周期数据
        expected = s.to_timestamp().resample(freq).ohlc().to_period(freq)

        # 时间戳的重采样不包括最后一个原始周期的所有子周期，因此需要相应扩展：
        # 创建一个新的日期范围对象，包含按指定频率扩展后的时间点
        new_index = period_range(start="2000", freq=freq, periods=period_mult * len(pi))
        # 根据新的日期范围对象重新索引预期的 DataFrame
        expected = expected.reindex(new_index)
        # 对序列进行指定频率的重采样，并计算 OHLC 数据，最后转换为时间戳再转为周期
        result = s.resample(freq).ohlc()
        tm.assert_frame_equal(result, expected)

        # 对序列进行指定频率的重采样，计算 OHLC 数据，最后转换为时间戳再转为周期，并与预期的 DataFrame 进行比较
        result = s.resample(freq).ohlc().to_timestamp().to_period()
        tm.assert_frame_equal(result, expected)
    @pytest.mark.parametrize(
        "periods, values",
        [
            (
                [
                    pd.NaT,  # 第一个周期值为 NaT (Not a Time)
                    "1970-01-01 00:00:00",  # 第二个周期值为具体的时间戳字符串
                    pd.NaT,  # 第三个周期值为 NaT
                    "1970-01-01 00:00:02",  # 第四个周期值为具体的时间戳字符串
                    "1970-01-01 00:00:03",  # 第五个周期值为具体的时间戳字符串
                ],
                [2, 3, 5, 7, 11],  # 对应的值列表
            ),
            (
                [
                    pd.NaT,  # 第一个周期值为 NaT
                    pd.NaT,  # 第二个周期值为 NaT
                    "1970-01-01 00:00:00",  # 第三个周期值为具体的时间戳字符串
                    pd.NaT,  # 第四个周期值为 NaT
                    pd.NaT,  # 第五个周期值为 NaT
                    pd.NaT,  # 第六个周期值为 NaT
                    "1970-01-01 00:00:02",  # 第七个周期值为具体的时间戳字符串
                    "1970-01-01 00:00:03",  # 第八个周期值为具体的时间戳字符串
                    pd.NaT,  # 第九个周期值为 NaT
                    pd.NaT,  # 第十个周期值为 NaT
                ],
                [1, 2, 3, 5, 6, 8, 7, 11, 12, 13],  # 对应的值列表
            ),
        ],
    )
    @pytest.mark.parametrize(
        "freq, expected_values",
        [
            ("1s", [3, np.nan, 7, 11]),  # 使用频率 "1s"，期望得到的值列表
            ("2s", [3, (7 + 11) / 2]),  # 使用频率 "2s"，期望得到的值列表
            ("3s", [(3 + 7) / 2, 11]),  # 使用频率 "3s"，期望得到的值列表
        ],
    )
    def test_resample_with_nat(self, periods, values, freq, expected_values):
        # GH 13224
        index = PeriodIndex(periods, freq="s")  # 使用给定周期和频率创建 PeriodIndex 对象
        frame = DataFrame(values, index=index)  # 使用给定值和索引创建 DataFrame 对象

        expected_index = period_range(
            "1970-01-01 00:00:00", periods=len(expected_values), freq=freq
        )  # 创建预期的时间周期范围索引
        expected = DataFrame(expected_values, index=expected_index)  # 使用预期值和索引创建 DataFrame 对象
        msg = "Resampling with a PeriodIndex is deprecated"  # 设置警告信息文本
        with tm.assert_produces_warning(FutureWarning, match=msg):
            rs = frame.resample(freq)  # 对 DataFrame 进行重新采样
        result = rs.mean()  # 计算重新采样后的均值
        tm.assert_frame_equal(result, expected)  # 断言重新采样结果与预期的 DataFrame 相等

    def test_resample_with_only_nat(self):
        # GH 13224
        pi = PeriodIndex([pd.NaT] * 3, freq="s")  # 创建只包含 NaT 值的 PeriodIndex 对象
        frame = DataFrame([2, 3, 5], index=pi, columns=["a"])  # 使用给定值和索引创建 DataFrame 对象
        expected_index = PeriodIndex(data=[], freq=pi.freq)  # 创建空的预期时间周期范围索引
        expected = DataFrame(index=expected_index, columns=["a"], dtype="float64")  # 创建预期的空 DataFrame 对象
        result = frame.resample("1s").mean()  # 对 DataFrame 进行 "1s" 频率的重新采样并计算均值
        tm.assert_frame_equal(result, expected)  # 断言重新采样结果与预期的 DataFrame 相等
    @pytest.mark.parametrize(
        "start,end,start_freq,end_freq,offset",
        [
            ("19910905", "19910909 03:00", "h", "24h", "10h"),
            ("19910905", "19910909 12:00", "h", "24h", "10h"),
            ("19910905", "19910909 23:00", "h", "24h", "10h"),
            ("19910905 10:00", "19910909", "h", "24h", "10h"),
            ("19910905 10:00", "19910909 10:00", "h", "24h", "10h"),
            ("19910905", "19910909 10:00", "h", "24h", "10h"),
            ("19910905 12:00", "19910909", "h", "24h", "10h"),
            ("19910905 12:00", "19910909 03:00", "h", "24h", "10h"),
            ("19910905 12:00", "19910909 12:00", "h", "24h", "10h"),
            ("19910905 12:00", "19910909 12:00", "h", "24h", "34h"),
            ("19910905 12:00", "19910909 12:00", "h", "17h", "10h"),
            ("19910905 12:00", "19910909 12:00", "h", "17h", "3h"),
            ("19910905", "19910913 06:00", "2h", "24h", "10h"),
            ("19910905", "19910905 01:39", "Min", "5Min", "3Min"),
            ("19910905", "19910905 03:18", "2Min", "5Min", "3Min"),
        ],
    )
    # 使用 pytest 的参数化装饰器，定义了多组测试参数，对 test_resample_with_offset 方法进行参数化测试
    def test_resample_with_offset(self, start, end, start_freq, end_freq, offset):
        # GH 23882 & 31809
        # 创建 PeriodIndex 对象 pi，根据指定的起始时间、结束时间和频率 start_freq
        pi = period_range(start, end, freq=start_freq)
        # 创建一个 Series 对象 ser，以 pi 作为索引，数值为对应索引位置的数组
        ser = Series(np.arange(len(pi)), index=pi)
        # 设置警告消息内容
        msg = "Resampling with a PeriodIndex is deprecated"
        # 使用 assert_produces_warning 上下文管理器检查是否产生 FutureWarning 警告，并匹配消息 msg
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # 对 ser 应用 resample 方法，以 end_freq 为结束频率，使用给定的 offset
            rs = ser.resample(end_freq, offset=offset)
        # 计算 resample 后的结果的均值
        result = rs.mean()
        # 将结果转换为指定频率 end_freq 的时间戳
        result = result.to_timestamp(end_freq)
        # 将 ser 转换为时间戳后再次进行 resample，以验证预期结果
        expected = ser.to_timestamp().resample(end_freq, offset=offset).mean()
        # 使用 assert_series_equal 检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

    def test_resample_with_offset_month(self):
        # GH 23882 & 31809
        # 创建 PeriodIndex 对象 pi，指定时间范围和频率 "h"
        pi = period_range("19910905 12:00", "19910909 1:00", freq="h")
        # 创建一个 Series 对象 ser，以 pi 作为索引，数值为对应索引位置的数组
        ser = Series(np.arange(len(pi)), index=pi)
        # 设置警告消息内容
        msg = "Resampling with a PeriodIndex is deprecated"
        # 使用 assert_produces_warning 上下文管理器检查是否产生 FutureWarning 警告，并匹配消息 msg
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # 对 ser 应用 resample 方法，以 "M" 为结束频率，使用 "3h" 的 offset
            rs = ser.resample("M", offset="3h")
        # 计算 resample 后的结果的均值
        result = rs.mean()
        # 将结果转换为 "M" 频率的时间戳
        result = result.to_timestamp("M")
        # 将 ser 转换为时间戳后再次进行 resample，以验证预期结果
        expected = ser.to_timestamp().resample("ME", offset="3h").mean()
        # 重置 expected 的索引频率为 None，用于比较非刻度的特征 (GH 33815)
        expected.index = expected.index._with_freq(None)
        # 使用 assert_series_equal 检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)
    @pytest.mark.parametrize(
        "first,last,freq,freq_to_offset,exp_first,exp_last",
        [
            # 参数化测试用例，定义多组输入和期望输出
            ("19910905", "19920406", "D", "D", "19910905", "19920406"),  # 每日频率的日期范围
            ("19910905 00:00", "19920406 06:00", "D", "D", "19910905", "19920406"),  # 含时间的每日频率日期范围
            (
                "19910905 06:00",
                "19920406 06:00",
                "h",
                "h",
                "19910905 06:00",
                "19920406 06:00",
            ),  # 每小时频率的日期范围
            ("19910906", "19920406", "M", "ME", "1991-09", "1992-04"),  # 月末频率的日期范围
            ("19910831", "19920430", "M", "ME", "1991-08", "1992-04"),  # 月末频率的日期范围
            ("1991-08", "1992-04", "M", "ME", "1991-08", "1992-04"),  # 月末频率的日期范围
        ],
    )
    def test_get_period_range_edges(
        self, first, last, freq, freq_to_offset, exp_first, exp_last
    ):
        first = Period(first)  # 将起始日期字符串转换为Period对象
        last = Period(last)  # 将结束日期字符串转换为Period对象

        exp_first = Period(exp_first, freq=freq)  # 用指定频率创建期间对象
        exp_last = Period(exp_last, freq=freq)  # 用指定频率创建期间对象

        freq = pd.tseries.frequencies.to_offset(freq_to_offset)  # 将频率字符串转换为时间偏移对象
        result = _get_period_range_edges(first, last, freq)  # 调用函数获取期间范围边界
        expected = (exp_first, exp_last)  # 设置预期输出
        assert result == expected  # 断言结果与预期相符

    def test_sum_min_count(self):
        # GH 19974
        index = date_range(start="2018", freq="ME", periods=6)  # 创建日期范围，以月末为频率
        data = np.ones(6)  # 创建包含六个1的数组
        data[3:6] = np.nan  # 将索引3到5的元素设为NaN
        s = Series(data, index).to_period()  # 将数据和索引转换为期间序列
        msg = "Resampling with a PeriodIndex is deprecated"  # 设置警告消息内容
        with tm.assert_produces_warning(FutureWarning, match=msg):  # 检查警告是否被触发
            rs = s.resample("Q")  # 对期间序列进行重新采样到季度频率
        result = rs.sum(min_count=1)  # 计算重新采样后的和，至少有一个非NaN值
        expected = Series(
            [3.0, np.nan], index=PeriodIndex(["2018Q1", "2018Q2"], freq="Q-DEC")
        )  # 设置预期的结果序列
        tm.assert_series_equal(result, expected)  # 断言结果序列与预期序列相等

    def test_resample_t_l_deprecated(self):
        # GH#52536
        msg_t = "Invalid frequency: T"  # 设置时间频率无效的警告消息
        msg_l = "Invalid frequency: L"  # 设置毫秒频率无效的警告消息

        with pytest.raises(ValueError, match=msg_l):  # 检查是否触发毫秒频率无效的值错误
            period_range(
                "2020-01-01 00:00:00 00:00", "2020-01-01 00:00:00 00:01", freq="L"
            )  # 创建毫秒级的期间范围
        rng_l = period_range(
            "2020-01-01 00:00:00 00:00", "2020-01-01 00:00:00 00:01", freq="ms"
        )  # 创建毫秒级的期间范围
        ser = Series(np.arange(len(rng_l)), index=rng_l)  # 创建毫秒级的期间序列

        with pytest.raises(ValueError, match=msg_t):  # 检查是否触发时间频率无效的值错误
            ser.resample("T").mean()  # 对期间序列进行时间重新采样求均值

    @pytest.mark.parametrize(
        "freq, freq_depr, freq_depr_res",
        [
            ("2Q", "2q", "2y"),  # 季度频率及其在不同版本中的替代
            ("2M", "2m", "2q"),  # 月末频率及其在不同版本中的替代
        ],
    )
    # 测试在指定的频率下是否会引发异常
    def test_resample_lowercase_frequency_raises(self, freq, freq_depr, freq_depr_res):
        # 准备错误消息，用于匹配抛出的异常信息
        msg = f"Invalid frequency: {freq_depr}"
        # 使用 pytest 检查是否会抛出 ValueError 异常，并匹配特定消息
        with pytest.raises(ValueError, match=msg):
            period_range("2020-01-01", "2020-08-01", freq=freq_depr)

        # 准备错误消息，用于匹配抛出的异常信息
        msg = f"Invalid frequency: {freq_depr_res}"
        # 创建时间序列范围并检查是否会抛出 ValueError 异常，并匹配特定消息
        rng = period_range("2020-01-01", "2020-08-01", freq=freq)
        # 创建序列并检查是否会抛出 ValueError 异常，计算重采样的均值
        ser = Series(np.arange(len(rng)), index=rng)
        with pytest.raises(ValueError, match=msg):
            ser.resample(freq_depr_res).mean()

    @pytest.mark.parametrize(
        "offset",
        [
            offsets.MonthBegin(),
            offsets.BYearBegin(2),
            offsets.BusinessHour(2),
        ],
    )
    # 测试在无效的时间偏移量下是否会引发异常
    def test_asfreq_invalid_period_offset(self, offset, frame_or_series):
        # 准备错误消息的正则表达式，用于匹配抛出的异常信息
        msg = re.escape(f"{offset} is not supported as period frequency")

        # 创建数据帧或者序列，并检查是否会抛出 ValueError 异常，匹配特定消息
        obj = frame_or_series(range(5), index=period_range("2020-01-01", periods=5))
        with pytest.raises(ValueError, match=msg):
            obj.asfreq(freq=offset)
@pytest.mark.parametrize(
    "freq",
    [
        ("2ME"),
        ("2QE"),
        ("2QE-FEB"),
        ("2YE"),
        ("2YE-MAR"),
        ("2me"),
        ("2qe"),
        ("2ye-mar"),
    ],
)
# 定义一个参数化测试函数，用于测试 resample 方法在不同频率下抛出 ValueError 的情况
def test_resample_frequency_ME_QE_YE_raises(frame_or_series, freq):
    # GH#9586
    # 构造错误消息，指出不支持的频率
    msg = f"{freq[1:]} is not supported as period frequency"

    # 创建一个测试对象，使用指定范围和周期范围的索引
    obj = frame_or_series(range(5), index=period_range("2020-01-01", periods=5))
    # 更新错误消息，指明无效的频率
    msg = f"Invalid frequency: {freq}"
    # 使用 pytest 断言抛出 ValueError，并匹配特定的错误消息
    with pytest.raises(ValueError, match=msg):
        obj.resample(freq)


# 测试处理 PeriodIndex 的边界情况
def test_corner_cases_period(simple_period_range_series):
    # miscellaneous test coverage
    # 创建一个长度为零的 PeriodIndex 时间序列
    len0pts = simple_period_range_series("2007-01", "2010-05", freq="M")[:0]
    # 期望产生 FutureWarning 警告，匹配给定的消息
    msg = "Resampling with a PeriodIndex is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        # 对长度为零的时间序列进行重新采样，期望返回空结果
        result = len0pts.resample("Y-DEC").mean()
    # 断言结果的长度为零
    assert len(result) == 0


@pytest.mark.parametrize("freq", ["2BME", "2CBME", "2SME", "2BQE-FEB", "2BYE-MAR"])
# 测试不支持的频率下 resample 方法是否会抛出 ValueError
def test_resample_frequency_invalid_freq(frame_or_series, freq):
    # GH#9586
    # 构造错误消息，指明无效的频率
    msg = f"Invalid frequency: {freq}"

    # 创建一个测试对象，使用指定范围和周期范围的索引
    obj = frame_or_series(range(5), index=period_range("2020-01-01", periods=5))
    # 使用 pytest 断言抛出 ValueError，并匹配特定的错误消息
    with pytest.raises(ValueError, match=msg):
        obj.resample(freq)
```