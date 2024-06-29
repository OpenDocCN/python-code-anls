# `D:\src\scipysrc\pandas\pandas\tests\indexes\period\methods\test_to_timestamp.py`

```
    # 导入必要的模块和库
    from datetime import datetime

    import numpy as np  # 导入NumPy库，用于处理数值数据
    import pytest  # 导入pytest模块，用于编写和运行测试

    from pandas import (  # 从Pandas库中导入多个模块和类
        DatetimeIndex,  # 时间索引类
        NaT,  # Not-a-Time，表示缺失日期时间
        PeriodIndex,  # 时间段索引类
        Timedelta,  # 时间差类型
        Timestamp,  # 时间戳类型
        date_range,  # 生成日期范围
        period_range,  # 生成时间段范围
    )
    import pandas._testing as tm  # 导入Pandas内部测试模块

    class TestToTimestamp:
        def test_to_timestamp_non_contiguous(self):
            # GH#44100
            # 创建一个日期范围，从'2021-10-18'开始，持续9天，每天一条记录
            dti = date_range("2021-10-18", periods=9, freq="D")
            # 将日期索引转换为时间段索引
            pi = dti.to_period()

            # 获取时间段索引的每隔两个值转换为时间戳
            result = pi[::2].to_timestamp()
            expected = dti[::2]
            tm.assert_index_equal(result, expected)

            # 使用底层数据获取每隔两个值转换为时间戳
            result = pi._data[::2].to_timestamp()
            expected = dti._data[::2]
            # TODO: can we get the freq to round-trip?
            tm.assert_datetime_array_equal(result, expected, check_freq=False)

            # 获取倒序的时间段索引转换为时间戳
            result = pi[::-1].to_timestamp()
            expected = dti[::-1]
            tm.assert_index_equal(result, expected)

            # 使用底层数据获取倒序的时间段索引转换为时间戳
            result = pi._data[::-1].to_timestamp()
            expected = dti._data[::-1]
            tm.assert_datetime_array_equal(result, expected, check_freq=False)

            # 获取每隔两个值后倒序的时间段索引转换为时间戳
            result = pi[::2][::-1].to_timestamp()
            expected = dti[::2][::-1]
            tm.assert_index_equal(result, expected)

            # 使用底层数据获取每隔两个值后倒序的时间段索引转换为时间戳
            result = pi._data[::2][::-1].to_timestamp()
            expected = dti._data[::2][::-1]
            tm.assert_datetime_array_equal(result, expected, check_freq=False)

        def test_to_timestamp_freq(self):
            # 创建一个时间段范围，从'2017'开始，持续12个时间段，每年一条记录，年度结束在12月
            idx = period_range("2017", periods=12, freq="Y-DEC")
            # 将时间段索引转换为时间戳索引
            result = idx.to_timestamp()
            # 生成一个日期范围，从'2017'开始，持续12个时间点，年度开始在1月
            expected = date_range("2017", periods=12, freq="YS-JAN")
            tm.assert_index_equal(result, expected)

        def test_to_timestamp_pi_nat(self):
            # GH#7228
            # 创建一个时间段索引，包含字符串"NaT"、"2011-01"、"2011-02"，频率为每月，名称为'idx'
            index = PeriodIndex(["NaT", "2011-01", "2011-02"], freq="M", name="idx")

            # 将时间段索引转换为时间戳索引，频率为每天
            result = index.to_timestamp("D")
            # 生成一个日期时间索引，包含NaT、2011-01-01、2011-02-01，数据类型为'M8[ns]'，名称为'idx'
            expected = DatetimeIndex(
                [NaT, datetime(2011, 1, 1), datetime(2011, 2, 1)],
                dtype="M8[ns]",
                name="idx",
            )
            tm.assert_index_equal(result, expected)
            # 断言结果索引的名称为'idx'
            assert result.name == "idx"

            # 将时间戳索引转换回原始的时间段索引，频率为每月
            result2 = result.to_period(freq="M")
            tm.assert_index_equal(result2, index)
            assert result2.name == "idx"

            # 将时间戳索引转换为频率为每3个月的时间段索引
            result3 = result.to_period(freq="3M")
            exp = PeriodIndex(["NaT", "2011-01", "2011-02"], freq="3M", name="idx")
            tm.assert_index_equal(result3, exp)
            # 断言结果索引的频率字符串为'3M'
            assert result3.freqstr == "3M"

            # 使用负数频率尝试将时间戳索引转换为时间段索引，预期抛出值错误异常
            msg = "Frequency must be positive, because it represents span: -2Y"
            with pytest.raises(ValueError, match=msg):
                result.to_period(freq="-2Y")

        def test_to_timestamp_preserve_name(self):
            # 创建一个年度频率的时间段索引，从'2001年1月1日'到'2009年12月1日'，名称为'foo'
            index = period_range(freq="Y", start="1/1/2001", end="12/1/2009", name="foo")
            # 断言索引的名称为'foo'
            assert index.name == "foo"

            # 将时间段索引转换为时间戳索引，频率为每天
            conv = index.to_timestamp("D")
            # 断言转换后的时间戳索引的名称仍为'foo'
            assert conv.name == "foo"
    # 定义测试方法，用于测试修复季度时间戳转换的 bug
    def test_to_timestamp_quarterly_bug(self):
        # 创建包含1960年到1999年的年份数组，每年重复四次
        years = np.arange(1960, 2000).repeat(4)
        # 创建包含1到4的季度数组，总共重复40次
        quarters = np.tile(list(range(1, 5)), 40)

        # 使用年份和季度数组创建周期索引对象
        pindex = PeriodIndex.from_fields(year=years, quarter=quarters)

        # 将周期索引对象转换为时间戳索引对象，每季度最后一天作为时间戳
        stamps = pindex.to_timestamp("D", "end")
        # 创建期望的时间戳索引对象，与 pindex 中的每个周期对象对应
        expected = DatetimeIndex([x.to_timestamp("D", "end") for x in pindex])

        # 断言生成的时间戳索引对象与期望的时间戳索引对象相等
        tm.assert_index_equal(stamps, expected)
        # 断言生成的时间戳索引对象的频率与期望的时间戳索引对象的频率相等
        assert stamps.freq == expected.freq

    # 定义测试方法，用于测试多种频率下的周期索引转换为时间戳
    def test_to_timestamp_pi_mult(self):
        # 创建包含日期字符串和 NaT（Not a Time）的周期索引对象，频率为每两个月
        idx = PeriodIndex(["2011-01", "NaT", "2011-02"], freq="2M", name="idx")

        # 将周期索引对象转换为时间戳索引对象，默认使用每个周期的第一天作为时间戳
        result = idx.to_timestamp()
        # 创建期望的时间戳索引对象，与 idx 中的每个周期对象对应
        expected = DatetimeIndex(
            ["2011-01-01", "NaT", "2011-02-01"], dtype="M8[ns]", name="idx"
        )
        # 断言生成的时间戳索引对象与期望的时间戳索引对象相等
        tm.assert_index_equal(result, expected)

        # 将周期索引对象转换为时间戳索引对象，使用每个周期的最后一天作为时间戳
        result = idx.to_timestamp(how="E")
        # 创建期望的时间戳索引对象，与 idx 中的每个周期对象对应
        expected = DatetimeIndex(
            ["2011-02-28", "NaT", "2011-03-31"], dtype="M8[ns]", name="idx"
        )
        # 将期望的时间戳索引对象调整为每个周期的最后一毫秒
        expected = expected + Timedelta(1, "D") - Timedelta(1, "ns")
        # 断言生成的时间戳索引对象与期望的时间戳索引对象相等
        tm.assert_index_equal(result, expected)

    # 定义测试方法，用于测试复合频率下的周期索引转换为时间戳
    def test_to_timestamp_pi_combined(self):
        # 创建从2011年开始，共两个周期，频率为每天每小时的周期索引对象
        idx = period_range(start="2011", periods=2, freq="1D1h", name="idx")

        # 将周期索引对象转换为时间戳索引对象，默认使用每个周期的第一天第零点作为时间戳
        result = idx.to_timestamp()
        # 创建期望的时间戳索引对象，与 idx 中的每个周期对象对应
        expected = DatetimeIndex(
            ["2011-01-01 00:00", "2011-01-02 01:00"], dtype="M8[ns]", name="idx"
        )
        # 断言生成的时间戳索引对象与期望的时间戳索引对象相等
        tm.assert_index_equal(result, expected)

        # 将周期索引对象转换为时间戳索引对象，使用每个周期的最后一小时的最后一秒作为时间戳
        result = idx.to_timestamp(how="E")
        # 创建期望的时间戳索引对象，与 idx 中的每个周期对象对应
        expected = DatetimeIndex(
            ["2011-01-02 00:59:59", "2011-01-03 01:59:59"], name="idx", dtype="M8[ns]"
        )
        # 将期望的时间戳索引对象调整为每个周期的最后一秒
        expected = expected + Timedelta(1, "s") - Timedelta(1, "ns")
        # 断言生成的时间戳索引对象与期望的时间戳索引对象相等
        tm.assert_index_equal(result, expected)

        # 将周期索引对象转换为时间戳索引对象，使用每小时作为频率，默认使用每个周期的第一小时作为时间戳
        result = idx.to_timestamp(how="E", freq="h")
        # 创建期望的时间戳索引对象，与 idx 中的每个周期对象对应
        expected = DatetimeIndex(
            ["2011-01-02 00:00", "2011-01-03 01:00"], dtype="M8[ns]", name="idx"
        )
        # 将期望的时间戳索引对象调整为每个周期的最后一小时的最后一秒
        expected = expected + Timedelta(1, "h") - Timedelta(1, "ns")
        # 断言生成的时间戳索引对象与期望的时间戳索引对象相等
        tm.assert_index_equal(result, expected)

    # 定义测试方法，用于测试从日期周期范围创建时间戳索引对象
    def test_to_timestamp_1703(self):
        # 创建从"1/1/2012"开始的日期周期范围，共四个周期，频率为每天
        index = period_range("1/1/2012", periods=4, freq="D")

        # 将周期索引对象转换为时间戳索引对象，默认使用每个周期的第一天作为时间戳
        result = index.to_timestamp()
        # 断言生成的时间戳索引对象的第一个时间戳与预期的时间戳相等
        assert result[0] == Timestamp("1/1/2012")
```