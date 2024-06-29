# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_at_time.py`

```
    from datetime import (
        time,        # 导入datetime模块中的time类，用于处理特定时间
        timezone,    # 导入datetime模块中的timezone类，用于处理时区信息
    )
    import zoneinfo  # 导入zoneinfo模块，用于处理时区信息

    import numpy as np  # 导入NumPy库，用于数值计算
    import pytest  # 导入pytest库，用于编写和运行测试用例

    from pandas._libs.tslibs import timezones  # 导入pandas库中用于处理时间序列的模块

    from pandas import (
        DataFrame,     # 从pandas库中导入DataFrame类，用于处理二维数据
        date_range,    # 从pandas库中导入date_range函数，用于生成日期范围
    )
    import pandas._testing as tm  # 导入pandas库中用于测试的模块


    class TestAtTime:
        @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
        def test_localized_at_time(self, tzstr, frame_or_series):
            tz = timezones.maybe_get_tz(tzstr)  # 获取时区信息

            rng = date_range("4/16/2012", "5/1/2012", freq="h")  # 生成日期范围
            ts = frame_or_series(
                np.random.default_rng(2).standard_normal(len(rng)), index=rng
            )  # 使用随机数生成时间序列或数据帧

            ts_local = ts.tz_localize(tzstr)  # 将时间序列本地化为指定时区

            result = ts_local.at_time(time(10, 0))  # 获取指定时间点的数据
            expected = ts.at_time(time(10, 0)).tz_localize(tzstr)  # 获取预期的指定时间点数据
            tm.assert_equal(result, expected)  # 断言结果与预期相等
            assert timezones.tz_compare(result.index.tz, tz)  # 断言结果的时区与预期时区一致

        def test_at_time(self, frame_or_series):
            rng = date_range("1/1/2000", "1/5/2000", freq="5min")  # 生成日期范围
            ts = DataFrame(
                np.random.default_rng(2).standard_normal((len(rng), 2)), index=rng
            )  # 使用随机数生成具有时间索引的数据帧
            ts = tm.get_obj(ts, frame_or_series)  # 获取测试对象

            rs = ts.at_time(rng[1])  # 获取指定时间点的数据
            assert (rs.index.hour == rng[1].hour).all()  # 断言结果中的小时与预期一致
            assert (rs.index.minute == rng[1].minute).all()  # 断言结果中的分钟与预期一致
            assert (rs.index.second == rng[1].second).all()  # 断言结果中的秒数与预期一致

            result = ts.at_time("9:30")  # 获取指定时间的数据
            expected = ts.at_time(time(9, 30))  # 获取预期的指定时间数据
            tm.assert_equal(result, expected)  # 断言结果与预期相等

        def test_at_time_midnight(self, frame_or_series):
            # midnight, everything
            rng = date_range("1/1/2000", "1/31/2000")  # 生成日期范围
            ts = DataFrame(
                np.random.default_rng(2).standard_normal((len(rng), 3)), index=rng
            )  # 使用随机数生成具有时间索引的数据帧
            ts = tm.get_obj(ts, frame_or_series)  # 获取测试对象

            result = ts.at_time(time(0, 0))  # 获取午夜时刻的数据
            tm.assert_equal(result, ts)  # 断言结果与原始数据帧相等

        def test_at_time_nonexistent(self, frame_or_series):
            # time doesn't exist
            rng = date_range("1/1/2012", freq="23Min", periods=384)  # 生成日期范围
            ts = DataFrame(np.random.default_rng(2).standard_normal(len(rng)), rng)  # 使用随机数生成时间序列
            ts = tm.get_obj(ts, frame_or_series)  # 获取测试对象
            rs = ts.at_time("16:00")  # 尝试获取不存在的时间点的数据
            assert len(rs) == 0  # 断言结果为空

        @pytest.mark.parametrize(
            "hour", ["1:00", "1:00AM", time(1), time(1, tzinfo=timezone.utc)]
        )
        def test_at_time_errors(self, hour):
            # GH#24043
            dti = date_range("2018", periods=3, freq="h")  # 生成日期范围
            df = DataFrame(list(range(len(dti))), index=dti)  # 使用列表生成数据帧
            if getattr(hour, "tzinfo", None) is None:
                result = df.at_time(hour)  # 获取指定时间点的数据
                expected = df.iloc[1:2]  # 获取预期的数据帧切片
                tm.assert_frame_equal(result, expected)  # 断言结果与预期相等
            else:
                with pytest.raises(ValueError, match="Index must be timezone"):
                    df.at_time(hour)  # 尝试使用带有时区信息的时间点，预期抛出值错误异常
    def test_at_time_tz(self):
        # GH#24043
        # 创建一个日期范围，每小时一次，从2018年开始，时区为US/Pacific
        dti = date_range("2018", periods=3, freq="h", tz="US/Pacific")
        # 创建一个DataFrame，以日期范围作为索引，数据为递增数列
        df = DataFrame(list(range(len(dti))), index=dti)
        # 从DataFrame中选择特定时间的数据，时间为04:00 US/Eastern时区
        result = df.at_time(time(4, tzinfo=zoneinfo.ZoneInfo("US/Eastern")))
        # 期望的结果是DataFrame中第二行的数据
        expected = df.iloc[1:2]
        # 断言两个DataFrame是否相等
        tm.assert_frame_equal(result, expected)

    def test_at_time_raises(self, frame_or_series):
        # GH#20725
        # 创建一个DataFrame对象
        obj = DataFrame([[1, 2, 3], [4, 5, 6]])
        # 获取适当的对象（DataFrame或Series）
        obj = tm.get_obj(obj, frame_or_series)
        # 预期的错误信息
        msg = "Index must be DatetimeIndex"
        # 使用pytest断言引发特定类型和消息的异常
        with pytest.raises(TypeError, match=msg):  # index is not a DatetimeIndex
            obj.at_time("00:00")

    def test_at_time_axis(self, axis):
        # issue 8839
        # 创建一个日期范围，从2000年1月1日到2000年1月2日，每5分钟一次
        rng = date_range("1/1/2000", "1/2/2000", freq="5min")
        # 创建一个随机数填充的DataFrame，行和列的索引都是日期范围rng
        ts = DataFrame(np.random.default_rng(2).standard_normal((len(rng), len(rng))))
        ts.index, ts.columns = rng, rng

        # 选取满足条件的日期时间索引
        indices = rng[(rng.hour == 9) & (rng.minute == 30) & (rng.second == 0)]

        # 根据指定的轴（行或列）选择数据
        if axis in ["index", 0]:
            expected = ts.loc[indices, :]
        elif axis in ["columns", 1]:
            expected = ts.loc[:, indices]

        # 根据特定的时间选择数据
        result = ts.at_time("9:30", axis=axis)

        # 如果不清除频率信息，结果的频率将是1440分钟，而期望的是5分钟
        result.index = result.index._with_freq(None)
        expected.index = expected.index._with_freq(None)
        # 断言两个DataFrame是否相等
        tm.assert_frame_equal(result, expected)

    def test_at_time_datetimeindex(self):
        # 创建一个日期范围，每30分钟一次，从2012年1月1日开始到2012年1月5日
        index = date_range("2012-01-01", "2012-01-05", freq="30min")
        # 创建一个随机数填充的DataFrame，行索引为日期范围index，列数为5
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(index), 5)), index=index
        )
        # 指定的时间键
        akey = time(12, 0, 0)
        # 选择符合指定时间的数据
        result = df.at_time(akey)
        # 期望的结果是具有相同时间的行
        expected = df.loc[akey]
        # 第二个期望结果是具有特定时间索引的行
        expected2 = df.iloc[[24, 72, 120, 168]]
        # 断言两个DataFrame是否相等
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result, expected2)
        # 断言结果DataFrame的长度为4
        assert len(result) == 4
```