# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_asof.py`

```
import numpy as np
import pytest

from pandas._libs.tslibs import IncompatibleFrequency

from pandas import (
    DatetimeIndex,
    PeriodIndex,
    Series,
    Timestamp,
    date_range,
    isna,
    notna,
    offsets,
    period_range,
)
import pandas._testing as tm


class TestSeriesAsof:
    def test_asof_nanosecond_index_access(self):
        ts = Timestamp("20130101").as_unit("ns")._value
        # 创建包含多个 nanosecond 时间戳的 DatetimeIndex
        dti = DatetimeIndex([ts + 50 + i for i in range(100)])
        # 创建一个 Series，具有随机标准正态分布数据，索引为 dti
        ser = Series(np.random.default_rng(2).standard_normal(100), index=dti)

        # 检查 DatetimeIndex 的分辨率为 "nanosecond"
        assert dti.resolution == "nanosecond"

        # 通过 asof 方法获取第一个时间戳对应的值
        first_value = ser.asof(ser.index[0])

        # 这个断言确保以前错误地被标记为 "day"，现在应该是 "nanosecond"
        assert dti.resolution == "nanosecond"

        # 这个断言检查第一个值是否与特定的时间戳匹配
        assert first_value == ser["2013-01-01 00:00:00.000000050"]

        # 预期的时间戳，以 numpy 的 datetime64 格式表示
        expected_ts = np.datetime64("2013-01-01 00:00:00.000000050", "ns")
        # 再次确认第一个值是否与预期的时间戳匹配
        assert first_value == ser[Timestamp(expected_ts)]

    def test_basic(self):
        # 创建一个长度为 N 的日期范围，频率为 53 秒
        N = 50
        rng = date_range("1/1/1990", periods=N, freq="53s")
        # 创建一个 Series，包含随机标准正态分布数据，索引为 rng
        ts = Series(np.random.default_rng(2).standard_normal(N), index=rng)
        # 将索引为 15 到 30 的位置设置为 NaN
        ts.iloc[15:30] = np.nan
        # 创建一个长度为 N*3 的日期范围，频率为 25 秒
        dates = date_range("1/1/1990", periods=N * 3, freq="25s")

        # 使用 asof 方法获取 dates 对应位置的值
        result = ts.asof(dates)
        # 检查结果中是否所有的值都不是 NaN
        assert notna(result).all()

        # 获取 ts 索引为 14 的时间戳
        lb = ts.index[14]
        # 获取 ts 索引为 30 的时间戳
        ub = ts.index[30]

        # 再次使用 asof 方法获取 dates 对应位置的值（传入一个列表）
        result = ts.asof(list(dates))
        # 检查结果中是否所有的值都不是 NaN
        assert notna(result).all()

        # 创建一个掩码，用于选择 lb <= result.index < ub 的结果
        mask = (result.index >= lb) & (result.index < ub)
        # 从 result 中选择符合掩码的子集 rs
        rs = result[mask]
        # 检查 rs 中的所有值是否等于 ts[lb] 的值
        assert (rs == ts[lb]).all()

        # 获取 result 中第一个大于等于 ub 的时间戳对应的值
        val = result[result.index[result.index >= ub][0]]
        # 检查 ts[ub] 是否等于 val
        assert ts[ub] == val

    def test_scalar(self):
        N = 30
        # 创建一个长度为 N 的日期范围，频率为 53 秒
        rng = date_range("1/1/1990", periods=N, freq="53s")
        # 将索引类型明确为 float，避免设置 NaN 时的隐式类型转换
        ts = Series(np.arange(N), index=rng, dtype="float")
        # 将索引为 5 到 10 的位置设置为 NaN
        ts.iloc[5:10] = np.nan
        # 将索引为 15 到 20 的位置设置为 NaN
        ts.iloc[15:20] = np.nan

        # 获取 ts 索引为 7 的时间戳对应的 asof 值
        val1 = ts.asof(ts.index[7])
        # 检查 val1 是否等于 ts.iloc[4]
        assert val1 == ts.iloc[4]

        # 获取 ts 索引为 19 的时间戳对应的 asof 值
        val2 = ts.asof(ts.index[19])
        # 检查 val2 是否等于 ts.iloc[14]
        assert val2 == ts.iloc[14]

        # 可以接受字符串作为参数
        val1 = ts.asof(str(ts.index[7]))
        # 检查 val1 是否等于 ts.iloc[4]
        assert val1 == ts.iloc[4]

        # 在存在的时间戳中查找特定的 asof 值
        result = ts.asof(ts.index[3])
        # 检查 result 是否等于 ts.iloc[3]
        assert result == ts.iloc[3]

        # 当不存在 as of 值时
        d = ts.index[0] - offsets.BDay()
        # 检查 ts.asof(d) 是否返回 NaN
        assert np.isnan(ts.asof(d))
    def test_with_nan(self):
        # basic asof test
        # 创建一个日期范围，从 "1/1/2000" 到 "1/2/2000"，频率为每4小时
        rng = date_range("1/1/2000", "1/2/2000", freq="4h")
        # 创建一个序列，序列的值为该范围的索引，索引为 rng
        s = Series(np.arange(len(rng)), index=rng)
        # 对序列进行重采样，频率为每2小时，采用均值聚合
        r = s.resample("2h").mean()

        # 获取重采样后的序列在其索引位置的 asof 值
        result = r.asof(r.index)
        # 期望的序列，包含特定索引和对应的值
        expected = Series(
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6.0],
            index=date_range("1/1/2000", "1/2/2000", freq="2h"),
        )
        # 断言结果序列与期望序列相等
        tm.assert_series_equal(result, expected)

        # 将第3到第5个位置的值设为 NaN
        r.iloc[3:5] = np.nan
        # 获取更新后的 asof 值
        result = r.asof(r.index)
        # 更新后的期望序列
        expected = Series(
            [0, 0, 1, 1, 1, 1, 3, 3, 4, 4, 5, 5, 6.0],
            index=date_range("1/1/2000", "1/2/2000", freq="2h"),
        )
        # 断言结果序列与期望序列相等
        tm.assert_series_equal(result, expected)

        # 将倒数第3到倒数第1个位置的值设为 NaN
        r.iloc[-3:] = np.nan
        # 获取更新后的 asof 值
        result = r.asof(r.index)
        # 更新后的期望序列
        expected = Series(
            [0, 0, 1, 1, 1, 1, 3, 3, 4, 4, 4, 4, 4.0],
            index=date_range("1/1/2000", "1/2/2000", freq="2h"),
        )
        # 断言结果序列与期望序列相等
        tm.assert_series_equal(result, expected)

    def test_periodindex(self):
        # array or list or dates
        # 创建一个包含50个周期的周期范围，从 "1/1/1990" 开始，频率为每小时
        N = 50
        rng = period_range("1/1/1990", periods=N, freq="h")
        # 创建一个包含50个随机正态分布值的序列，索引为 rng
        ts = Series(np.random.default_rng(2).standard_normal(N), index=rng)
        # 将第15到第30个位置的值设为 NaN
        ts.iloc[15:30] = np.nan
        # 创建一个日期范围，包含N*3个日期，频率为每37分钟
        dates = date_range("1/1/1990", periods=N * 3, freq="37min")

        # 获取序列在指定日期处的 asof 值
        result = ts.asof(dates)
        # 断言结果中没有 NaN 值
        assert notna(result).all()
        # 设置 lb 为索引第14个位置的值，ub 为索引第30个位置的值
        lb = ts.index[14]
        ub = ts.index[30]

        # 将结果转换为周期索引
        pix = PeriodIndex(result.index.values, freq="h")
        # 创建一个布尔掩码，选择落在 lb 到 ub 之间的结果
        mask = (pix >= lb) & (pix < ub)
        # 从结果中选择符合掩码的子序列
        rs = result[mask]
        # 断言结果子序列中的值等于 lb 处的值
        assert (rs == ts[lb]).all()

        # 将第5到第10个位置的值设为 NaN
        ts.iloc[5:10] = np.nan
        # 将第15到第20个位置的值设为 NaN
        ts.iloc[15:20] = np.nan

        # 获取 ts 索引第7个位置处的 asof 值
        val1 = ts.asof(ts.index[7])
        # 获取 ts 索引第19个位置处的 asof 值
        val2 = ts.asof(ts.index[19])

        # 断言第7个位置处的 asof 值等于索引第4个位置处的值
        assert val1 == ts.iloc[4]
        # 断言第19个位置处的 asof 值等于索引第14个位置处的值
        assert val2 == ts.iloc[14]

        # 接受字符串输入，获取 ts 索引第7个位置处的 asof 值
        val1 = ts.asof(str(ts.index[7]))
        # 断言第7个位置处的 asof 值等于索引第4个位置处的值
        assert val1 == ts.iloc[4]

        # 获取 ts 索引第3个位置处的 asof 值
        assert ts.asof(ts.index[3]) == ts.iloc[3]

        # 测试没有 as of 值的情况
        # 计算 ts 索引第0个位置的时间戳之前一个工作日的时间戳
        d = ts.index[0].to_timestamp() - offsets.BDay()
        # 断言 ts 在该时间戳处没有 as of 值
        assert isna(ts.asof(d))

        # 测试频率不匹配的情况
        msg = "Input has different freq"
        with pytest.raises(IncompatibleFrequency, match=msg):
            # 尝试使用不同频率的周期来调用 asof 方法
            ts.asof(rng.asfreq("D"))
    def test_errors(self):
        s = Series(
            [1, 2, 3],
            index=[Timestamp("20130101"), Timestamp("20130103"), Timestamp("20130102")],
        )

        # non-monotonic
        # 检查索引是否单调递增
        assert not s.index.is_monotonic_increasing
        # 使用 pytest 检查是否会抛出 ValueError 异常，并匹配特定错误信息
        with pytest.raises(ValueError, match="requires a sorted index"):
            s.asof(s.index[0])

        # subset with Series
        # 创建一个时间范围和随机数据的 Series 对象
        N = 10
        rng = date_range("1/1/1990", periods=N, freq="53s")
        s = Series(np.random.default_rng(2).standard_normal(N), index=rng)
        # 使用 pytest 检查是否会抛出 ValueError 异常，并匹配特定错误信息
        with pytest.raises(ValueError, match="not valid for Series"):
            s.asof(s.index[0], subset="foo")

    def test_all_nans(self):
        # GH 15713
        # series is all nans
        # 创建一个所有元素为 NaN 的 Series，用于测试

        # testing non-default indexes
        # 创建时间范围和相应的所有元素为 NaN 的 Series 对象，然后调用 asof 方法
        N = 50
        rng = date_range("1/1/1990", periods=N, freq="53s")

        dates = date_range("1/1/1990", periods=N * 3, freq="25s")
        # 使用 asof 方法获取结果并与期望结果进行比较
        result = Series(np.nan, index=rng).asof(dates)
        expected = Series(np.nan, index=dates)
        tm.assert_series_equal(result, expected)

        # testing scalar input
        # 创建所有元素为 NaN 的 Series，然后调用 asof 方法
        date = date_range("1/1/1990", periods=N * 3, freq="25s")[0]
        # 检查返回结果是否为 NaN
        result = Series(np.nan, index=rng).asof(date)
        assert isna(result)

        # test name is propagated
        # 创建所有元素为 NaN 的 Series，指定名称为 "test"，然后调用 asof 方法
        result = Series(np.nan, index=[1, 2, 3, 4], name="test").asof([4, 5])
        expected = Series(np.nan, index=[4, 5], name="test")
        # 检查返回结果与期望结果是否相等
        tm.assert_series_equal(result, expected)
```