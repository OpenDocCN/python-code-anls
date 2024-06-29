# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\methods\test_tz_convert.py`

```
    from datetime import datetime
    从 datetime 模块导入 datetime 类

    import dateutil.tz
    导入 dateutil.tz 模块

    from dateutil.tz import gettz
    从 dateutil.tz 模块导入 gettz 函数

    import numpy as np
    导入 numpy 库，并使用 np 别名

    import pytest
    导入 pytest 库

    from pandas._libs.tslibs import timezones
    从 pandas 库中导入 _libs.tslibs.timezones 模块

    from pandas import (
        DatetimeIndex,
        Index,
        NaT,
        Timestamp,
        date_range,
        offsets,
    )
    从 pandas 库中导入多个类和函数：DatetimeIndex, Index, NaT, Timestamp, date_range, offsets

    import pandas._testing as tm
    导入 pandas._testing 模块并使用 tm 别名

    class TestTZConvert:
        定义一个名为 TestTZConvert 的测试类

        def test_tz_convert_nat(self):
            定义一个测试方法 test_tz_convert_nat，测试时区转换处理 NaT 的情况

            # GH#5546
            使用 GH#5546 作为注释

            dates = [NaT]
            创建一个包含 NaT（Not a Time）的日期列表

            idx = DatetimeIndex(dates)
            创建 DatetimeIndex 对象 idx，使用 dates 初始化

            idx = idx.tz_localize("US/Pacific")
            将 idx 对象本地化为 "US/Pacific" 时区

            tm.assert_index_equal(idx, DatetimeIndex(dates, tz="US/Pacific"))
            使用 tm.assert_index_equal 检查 idx 是否与指定时区的 DatetimeIndex 相等

            idx = idx.tz_convert("US/Eastern")
            将 idx 对象转换为 "US/Eastern" 时区

            tm.assert_index_equal(idx, DatetimeIndex(dates, tz="US/Eastern"))
            使用 tm.assert_index_equal 检查 idx 是否与指定时区的 DatetimeIndex 相等

            idx = idx.tz_convert("UTC")
            将 idx 对象转换为 "UTC" 时区

            tm.assert_index_equal(idx, DatetimeIndex(dates, tz="UTC"))
            使用 tm.assert_index_equal 检查 idx 是否与指定时区的 DatetimeIndex 相等

            dates = ["2010-12-01 00:00", "2010-12-02 00:00", NaT]
            创建包含日期和 NaT 的日期列表 dates

            idx = DatetimeIndex(dates)
            创建 DatetimeIndex 对象 idx，使用 dates 初始化

            idx = idx.tz_localize("US/Pacific")
            将 idx 对象本地化为 "US/Pacific" 时区

            tm.assert_index_equal(idx, DatetimeIndex(dates, tz="US/Pacific"))
            使用 tm.assert_index_equal 检查 idx 是否与指定时区的 DatetimeIndex 相等

            idx = idx.tz_convert("US/Eastern")
            将 idx 对象转换为 "US/Eastern" 时区

            expected = ["2010-12-01 03:00", "2010-12-02 03:00", NaT]
            创建预期的日期列表 expected，考虑到时区转换后的差异

            tm.assert_index_equal(idx, DatetimeIndex(expected, tz="US/Eastern"))
            使用 tm.assert_index_equal 检查 idx 是否与预期的 DatetimeIndex 相等

            idx = idx + offsets.Hour(5)
            将 idx 对象的时间增加 5 小时

            expected = ["2010-12-01 08:00", "2010-12-02 08:00", NaT]
            创建预期的日期列表 expected，考虑到时间增加后的差异

            tm.assert_index_equal(idx, DatetimeIndex(expected, tz="US/Eastern"))
            使用 tm.assert_index_equal 检查 idx 是否与预期的 DatetimeIndex 相等

            idx = idx.tz_convert("US/Pacific")
            将 idx 对象转换为 "US/Pacific" 时区

            expected = ["2010-12-01 05:00", "2010-12-02 05:00", NaT]
            创建预期的日期列表 expected，考虑到时区转换后的差异

            tm.assert_index_equal(idx, DatetimeIndex(expected, tz="US/Pacific"))
            使用 tm.assert_index_equal 检查 idx 是否与预期的 DatetimeIndex 相等

            idx = idx + np.timedelta64(3, "h")
            将 idx 对象的时间增加 3 小时

            expected = ["2010-12-01 08:00", "2010-12-02 08:00", NaT]
            创建预期的日期列表 expected，考虑到时间增加后的差异

            tm.assert_index_equal(idx, DatetimeIndex(expected, tz="US/Pacific"))
            使用 tm.assert_index_equal 检查 idx 是否与预期的 DatetimeIndex 相等

            idx = idx.tz_convert("US/Eastern")
            将 idx 对象转换为 "US/Eastern" 时区

            expected = ["2010-12-01 11:00", "2010-12-02 11:00", NaT]
            创建预期的日期列表 expected，考虑到时区转换后的差异

            tm.assert_index_equal(idx, DatetimeIndex(expected, tz="US/Eastern"))

        @pytest.mark.parametrize("prefix", ["", "dateutil/"])
        使用 pytest 的 parametrize 标记，为 prefix 参数设置两种值

        def test_dti_tz_convert_compat_timestamp(self, prefix):
            定义一个测试方法 test_dti_tz_convert_compat_timestamp，测试 DatetimeIndex 的时区转换和兼容性处理

            strdates = ["1/1/2012", "3/1/2012", "4/1/2012"]
            创建一个字符串日期列表 strdates

            idx = DatetimeIndex(strdates, tz=prefix + "US/Eastern")
            创建 DatetimeIndex 对象 idx，使用 strdates 和指定时区初始化

            conv = idx[0].tz_convert(prefix + "US/Pacific")
            将 idx 中第一个元素的时区转换为 "US/Pacific"，并赋值给 conv

            expected = idx.tz_convert(prefix + "US/Pacific")[0]
            从 idx 中转换所有元素的时区为 "US/Pacific"，并获取第一个元素作为预期结果

            assert conv == expected
            使用断言检查 conv 是否等于 expected
    # 定义一个测试方法，用于测试时区转换中小时数溢出的问题，这是对 GitHub issue #13306 的回归测试

    # 第一组测试：按时间排序，从 US/Eastern 时区转换到 UTC 时区
    ts = ["2008-05-12 09:50:00", "2008-12-12 09:50:35", "2009-05-12 09:50:32"]
    # 创建一个 DatetimeIndex 对象，将时间戳列表转换为本地化为 US/Eastern 时区的日期时间索引
    tt = DatetimeIndex(ts).tz_localize("US/Eastern")
    # 将本地化的时区转换为 UTC 时区
    ut = tt.tz_convert("UTC")
    # 期望的结果是一个包含小时数的索引对象，应为 [13, 14, 13]
    expected = Index([13, 14, 13], dtype=np.int32)
    # 使用 pytest 模块的 assert_index_equal 方法验证实际结果和期望结果是否相符
    tm.assert_index_equal(ut.hour, expected)

    # 第二组测试：按时间排序，从 UTC 时区转换到 US/Eastern 时区
    ts = ["2008-05-12 13:50:00", "2008-12-12 14:50:35", "2009-05-12 13:50:32"]
    # 创建一个 DatetimeIndex 对象，将时间戳列表转换为本地化为 UTC 时区的日期时间索引
    tt = DatetimeIndex(ts).tz_localize("UTC")
    # 将本地化的时区转换为 US/Eastern 时区
    ut = tt.tz_convert("US/Eastern")
    # 期望的结果是一个包含小时数的索引对象，应为 [9, 9, 9]
    expected = Index([9, 9, 9], dtype=np.int32)
    # 使用 pytest 模块的 assert_index_equal 方法验证实际结果和期望结果是否相符
    tm.assert_index_equal(ut.hour, expected)

    # 第三组测试：按时间不排序，从 US/Eastern 时区转换到 UTC 时区
    ts = ["2008-05-12 09:50:00", "2008-12-12 09:50:35", "2008-05-12 09:50:32"]
    # 创建一个 DatetimeIndex 对象，将时间戳列表转换为本地化为 US/Eastern 时区的日期时间索引
    tt = DatetimeIndex(ts).tz_localize("US/Eastern")
    # 将本地化的时区转换为 UTC 时区
    ut = tt.tz_convert("UTC")
    # 期望的结果是一个包含小时数的索引对象，应为 [13, 14, 13]
    expected = Index([13, 14, 13], dtype=np.int32)
    # 使用 pytest 模块的 assert_index_equal 方法验证实际结果和期望结果是否相符
    tm.assert_index_equal(ut.hour, expected)

    # 第四组测试：按时间不排序，从 UTC 时区转换到 US/Eastern 时区
    ts = ["2008-05-12 13:50:00", "2008-12-12 14:50:35", "2008-05-12 13:50:32"]
    # 创建一个 DatetimeIndex 对象，将时间戳列表转换为本地化为 UTC 时区的日期时间索引
    tt = DatetimeIndex(ts).tz_localize("UTC")
    # 将本地化的时区转换为 US/Eastern 时区
    ut = tt.tz_convert("US/Eastern")
    # 期望的结果是一个包含小时数的索引对象，应为 [9, 9, 9]
    expected = Index([9, 9, 9], dtype=np.int32)
    # 使用 pytest 模块的 assert_index_equal 方法验证实际结果和期望结果是否相符
    tm.assert_index_equal(ut.hour, expected)
    def test_dti_tz_convert_hour_overflow_dst_timestamps(self, tz):
        # Regression test for GH#13306
        
        # sorted case US/Eastern -> UTC
        # 创建时间戳列表，使用给定时区进行时区转换
        ts = [
            Timestamp("2008-05-12 09:50:00", tz=tz),
            Timestamp("2008-12-12 09:50:35", tz=tz),
            Timestamp("2009-05-12 09:50:32", tz=tz),
        ]
        # 将时间戳列表转换为日期时间索引
        tt = DatetimeIndex(ts)
        # 将日期时间索引转换为UTC时区
        ut = tt.tz_convert("UTC")
        # 预期结果为整数索引数组
        expected = Index([13, 14, 13], dtype=np.int32)
        # 断言索引相等
        tm.assert_index_equal(ut.hour, expected)

        # sorted case UTC -> US/Eastern
        # 创建时间戳列表，从UTC时区转换为给定时区
        ts = [
            Timestamp("2008-05-12 13:50:00", tz="UTC"),
            Timestamp("2008-12-12 14:50:35", tz="UTC"),
            Timestamp("2009-05-12 13:50:32", tz="UTC"),
        ]
        # 将时间戳列表转换为日期时间索引
        tt = DatetimeIndex(ts)
        # 将日期时间索引转换为指定时区
        ut = tt.tz_convert("US/Eastern")
        # 预期结果为整数索引数组
        expected = Index([9, 9, 9], dtype=np.int32)
        # 断言索引相等
        tm.assert_index_equal(ut.hour, expected)

        # unsorted case US/Eastern -> UTC
        # 创建未排序的时间戳列表，使用给定时区进行时区转换
        ts = [
            Timestamp("2008-05-12 09:50:00", tz=tz),
            Timestamp("2008-12-12 09:50:35", tz=tz),
            Timestamp("2008-05-12 09:50:32", tz=tz),
        ]
        # 将时间戳列表转换为日期时间索引
        tt = DatetimeIndex(ts)
        # 将日期时间索引转换为UTC时区
        ut = tt.tz_convert("UTC")
        # 预期结果为整数索引数组
        expected = Index([13, 14, 13], dtype=np.int32)
        # 断言索引相等
        tm.assert_index_equal(ut.hour, expected)

        # unsorted case UTC -> US/Eastern
        # 创建未排序的时间戳列表，从UTC时区转换为给定时区
        ts = [
            Timestamp("2008-05-12 13:50:00", tz="UTC"),
            Timestamp("2008-12-12 14:50:35", tz="UTC"),
            Timestamp("2008-05-12 13:50:32", tz="UTC"),
        ]
        # 将时间戳列表转换为日期时间索引
        tt = DatetimeIndex(ts)
        # 将日期时间索引转换为指定时区
        ut = tt.tz_convert("US/Eastern")
        # 预期结果为整数索引数组
        expected = Index([9, 9, 9], dtype=np.int32)
        # 断言索引相等
        tm.assert_index_equal(ut.hour, expected)

    @pytest.mark.parametrize("freq, n", [("h", 1), ("min", 60), ("s", 3600)])
    def test_dti_tz_convert_trans_pos_plus_1__bug(self, freq, n):
        # Regression test for tslib.tz_convert(vals, tz1, tz2).
        # See GH#4496 for details.
        # 创建日期范围，指定频率和时区
        idx = date_range(datetime(2011, 3, 26, 23), datetime(2011, 3, 27, 1), freq=freq)
        # 将日期范围本地化为UTC时区
        idx = idx.tz_localize("UTC")
        # 将日期范围转换为指定时区
        idx = idx.tz_convert("Europe/Moscow")

        # 创建预期结果，重复数组并计算元素
        expected = np.repeat(np.array([3, 4, 5]), np.array([n, n, 1]))
        # 断言索引相等
        tm.assert_index_equal(idx.hour, Index(expected, dtype=np.int32))
    # 定义一个测试方法，用于测试时区转换和夏令时处理
    def test_dti_tz_convert_dst(self):
        # 遍历不同频率和重复次数的组合
        for freq, n in [("h", 1), ("min", 60), ("s", 3600)]:
            # 开始夏令时测试
            # 创建日期范围对象，从 "2014-03-08 23:00" 到 "2014-03-09 09:00"，使用指定频率和UTC时区
            idx = date_range(
                "2014-03-08 23:00", "2014-03-09 09:00", freq=freq, tz="UTC"
            )
            # 将日期范围对象转换为 "US/Eastern" 时区
            idx = idx.tz_convert("US/Eastern")
            # 生成预期结果，重复数组中包含小时数，分别对应不同的重复次数n
            expected = np.repeat(
                np.array([18, 19, 20, 21, 22, 23, 0, 1, 3, 4, 5]),
                np.array([n, n, n, n, n, n, n, n, n, n, 1]),
            )
            # 断言索引的小时部分与预期结果相等
            tm.assert_index_equal(idx.hour, Index(expected, dtype=np.int32))

            # 反向测试夏令时结束情况
            # 创建日期范围对象，从 "2014-03-08 18:00" 到 "2014-03-09 05:00"，使用指定频率和 "US/Eastern" 时区
            idx = date_range(
                "2014-03-08 18:00", "2014-03-09 05:00", freq=freq, tz="US/Eastern"
            )
            # 将日期范围对象转换为UTC时区
            idx = idx.tz_convert("UTC")
            # 生成预期结果，重复数组中包含小时数，分别对应不同的重复次数n
            expected = np.repeat(
                np.array([23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                np.array([n, n, n, n, n, n, n, n, n, n, 1]),
            )
            # 断言索引的小时部分与预期结果相等
            tm.assert_index_equal(idx.hour, Index(expected, dtype=np.int32))

            # 结束夏令时测试
            # 创建日期范围对象，从 "2014-11-01 23:00" 到 "2014-11-02 09:00"，使用指定频率和UTC时区
            idx = date_range(
                "2014-11-01 23:00", "2014-11-02 09:00", freq=freq, tz="UTC"
            )
            # 将日期范围对象转换为 "US/Eastern" 时区
            idx = idx.tz_convert("US/Eastern")
            # 生成预期结果，重复数组中包含小时数，分别对应不同的重复次数n
            expected = np.repeat(
                np.array([19, 20, 21, 22, 23, 0, 1, 1, 2, 3, 4]),
                np.array([n, n, n, n, n, n, n, n, n, n, 1]),
            )
            # 断言索引的小时部分与预期结果相等
            tm.assert_index_equal(idx.hour, Index(expected, dtype=np.int32))

            # 反向测试夏令时结束情况
            # 创建日期范围对象，从 "2014-11-01 18:00" 到 "2014-11-02 05:00"，使用指定频率和 "US/Eastern" 时区
            idx = date_range(
                "2014-11-01 18:00", "2014-11-02 05:00", freq=freq, tz="US/Eastern"
            )
            # 将日期范围对象转换为UTC时区
            idx = idx.tz_convert("UTC")
            # 生成预期结果，重复数组中包含小时数，分别对应不同的重复次数n
            expected = np.repeat(
                np.array([22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                np.array([n, n, n, n, n, n, n, n, n, n, n, n, 1]),
            )
            # 断言索引的小时部分与预期结果相等
            tm.assert_index_equal(idx.hour, Index(expected, dtype=np.int32))

        # 每日频率测试
        # 开始夏令时测试
        # 创建日期范围对象，从 "2014-03-08 00:00" 到 "2014-03-09 00:00"，使用每日频率和UTC时区
        idx = date_range("2014-03-08 00:00", "2014-03-09 00:00", freq="D", tz="UTC")
        # 将日期范围对象转换为 "US/Eastern" 时区
        idx = idx.tz_convert("US/Eastern")
        # 断言索引的小时部分与预期结果相等，预期结果是 [19, 19]
        tm.assert_index_equal(idx.hour, Index([19, 19], dtype=np.int32))

        # 反向测试夏令时结束情况
        # 创建日期范围对象，从 "2014-03-08 00:00" 到 "2014-03-09 00:00"，使用每日频率和 "US/Eastern" 时区
        idx = date_range(
            "2014-03-08 00:00", "2014-03-09 00:00", freq="D", tz="US/Eastern"
        )
        # 将日期范围对象转换为UTC时区
        idx = idx.tz_convert("UTC")
        # 断言索引的小时部分与预期结果相等，预期结果是 [5, 5]
        tm.assert_index_equal(idx.hour, Index([5, 5], dtype=np.int32))

        # 结束夏令时测试
        # 创建日期范围对象，从 "2014-11-01 00:00" 到 "2014-11-02 00:00"，使用每日频率和UTC时区
        idx = date_range("2014-11-01 00:00", "2014-11-02 00:00", freq="D", tz="UTC")
        # 将日期范围对象转换为 "US/Eastern" 时区
        idx = idx.tz_convert("US/Eastern")
        # 断言索引的小时部分与预期结果相等，预期结果是 [20, 20]
        tm.assert_index_equal(idx.hour, Index([20, 20], dtype=np.int32))

        # 反向测试夏令时结束情况
        # 创建日期范围对象，从 "2014-11-01 00:00" 到 "2014-11-02 000:00"，使用每日频率和 "US/Eastern" 时区
        idx = date_range(
            "2014-11-01 00:00", "2014-11-02 000:00", freq="D", tz="US/Eastern"
        )
        # 将日期范围对象转换为UTC时区
        idx = idx.tz_convert("UTC")
        # 断言索引的小时部分与预期结果相等，预期结果是 [4, 4]
        tm.assert_index_equal(idx.hour, Index([4, 4], dtype=np.int32))
    # 测试时区转换的往返功能
    def test_tz_convert_roundtrip(self, tz_aware_fixture):
        # 获取测试用的时区感知对象
        tz = tz_aware_fixture
        
        # 创建不同时区的日期范围索引和预期结果
        idx1 = date_range(start="2014-01-01", end="2014-12-31", freq="ME", tz="UTC")
        exp1 = date_range(start="2014-01-01", end="2014-12-31", freq="ME")

        idx2 = date_range(start="2014-01-01", end="2014-12-31", freq="D", tz="UTC")
        exp2 = date_range(start="2014-01-01", end="2014-12-31", freq="D")

        idx3 = date_range(start="2014-01-01", end="2014-03-01", freq="h", tz="UTC")
        exp3 = date_range(start="2014-01-01", end="2014-03-01", freq="h")

        idx4 = date_range(start="2014-08-01", end="2014-10-31", freq="min", tz="UTC")
        exp4 = date_range(start="2014-08-01", end="2014-10-31", freq="min")

        # 遍历不同时区的日期范围索引和其预期结果
        for idx, expected in [(idx1, exp1), (idx2, exp2), (idx3, exp3), (idx4, exp4)]:
            # 将索引进行时区转换
            converted = idx.tz_convert(tz)
            # 将转换后的索引再转回原始时区
            reset = converted.tz_convert(None)
            # 断言转换后的索引与预期结果相等
            tm.assert_index_equal(reset, expected)
            # 断言重置后的索引不再包含时区信息
            assert reset.tzinfo is None
            # 对转换后的索引再次转换为UTC时区，然后去除本地化信息
            expected = converted.tz_convert("UTC").tz_localize(None)
            expected = expected._with_freq("infer")
            # 断言重置后的索引与重新转换的预期结果相等
            tm.assert_index_equal(reset, expected)

    # 测试时区转换对内部对象不产生影响的情况
    def test_dti_tz_convert_tzlocal(self):
        # GH#13583
        # 创建带有UTC时区的日期范围对象
        dti = date_range(start="2001-01-01", end="2001-03-01", tz="UTC")
        # 将UTC时区转换为本地时区
        dti2 = dti.tz_convert(dateutil.tz.tzlocal())
        # 断言转换后的日期范围对象的整数表示相等
        tm.assert_numpy_array_equal(dti2.asi8, dti.asi8)

        # 创建带有本地时区的日期范围对象
        dti = date_range(start="2001-01-01", end="2001-03-01", tz=dateutil.tz.tzlocal())
        # 将本地时区转换为无时区
        dti2 = dti.tz_convert(None)
        # 断言转换后的日期范围对象的整数表示相等
        tm.assert_numpy_array_equal(dti2.asi8, dti.asi8)

    # 使用参数化测试多个时区字符串进行测试
    @pytest.mark.parametrize(
        "tz",
        [
            "US/Eastern",
            "dateutil/US/Eastern",
            "pytz/US/Eastern",
            gettz("US/Eastern"),
        ],
    )
    def test_dti_tz_convert_utc_to_local_no_modify(self, tz):
        # 如果时区是以"pytz/"开头的字符串，则导入pytz模块
        if isinstance(tz, str) and tz.startswith("pytz/"):
            pytz = pytest.importorskip("pytz")
            tz = pytz.timezone(tz.removeprefix("pytz/"))
        # 创建UTC时区下的日期范围对象
        rng = date_range("3/11/2012", "3/12/2012", freq="h", tz="utc")
        # 将日期范围对象转换为指定时区的日期范围对象
        rng_eastern = rng.tz_convert(tz)

        # 断言转换后的日期范围对象的整数表示相等
        tm.assert_numpy_array_equal(rng.asi8, rng_eastern.asi8)

        # 断言转换后的时区与预期时区相同
        assert timezones.tz_compare(rng_eastern.tz, timezones.maybe_get_tz(tz))

    # 测试时区转换在逆序操作时的情况
    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_tz_convert_unsorted(self, tzstr):
        # 创建UTC时区下的日期范围对象
        dr = date_range("2012-03-09", freq="h", periods=100, tz="utc")
        # 将日期范围对象转换为指定时区的日期范围对象
        dr = dr.tz_convert(tzstr)

        # 获取逆序后的小时数
        result = dr[::-1].hour
        # 获取逆序后的预期小时数
        exp = dr.hour[::-1]
        # 断言结果与预期相近
        tm.assert_almost_equal(result, exp)
```