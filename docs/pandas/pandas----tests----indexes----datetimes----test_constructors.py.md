# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\test_constructors.py`

```
    # 导入必要的模块和类
    from __future__ import annotations

    from datetime import (
        datetime,                 # 导入 datetime 类
        timedelta,                # 导入 timedelta 类
        timezone,                 # 导入 timezone 类
    )
    from functools import partial  # 导入 partial 函数
    from operator import attrgetter  # 导入 attrgetter 函数
    import zoneinfo  # 导入 zoneinfo 模块

    import dateutil  # 导入 dateutil 库
    import dateutil.tz  # 导入 dateutil.tz 模块
    from dateutil.tz import gettz  # 导入 gettz 函数
    import numpy as np  # 导入 numpy 库
    import pytest  # 导入 pytest 库
    import pytz  # 导入 pytz 库

    from pandas._libs.tslibs import (
        astype_overflowsafe,    # 导入 astype_overflowsafe 函数
        timezones,              # 导入 timezones 对象
    )

    import pandas as pd  # 导入 pandas 库
    from pandas import (
        DatetimeIndex,          # 导入 DatetimeIndex 类
        Index,                  # 导入 Index 类
        Timestamp,              # 导入 Timestamp 类
        date_range,             # 导入 date_range 函数
        offsets,                # 导入 offsets 模块
        to_datetime,            # 导入 to_datetime 函数
    )
    import pandas._testing as tm  # 导入 pandas 测试工具模块
    from pandas.core.arrays import period_array  # 导入 period_array 类

    class TestDatetimeIndex:
        def test_from_dt64_unsupported_unit(self):
            # GH#49292 测试从 np.datetime64 类型创建不支持单位的 DatetimeIndex
            val = np.datetime64(1, "D")
            result = DatetimeIndex([val], tz="US/Pacific")

            expected = DatetimeIndex([val.astype("M8[s]")], tz="US/Pacific")
            tm.assert_index_equal(result, expected)

        def test_explicit_tz_none(self):
            # GH#48659 测试当传入数据是时区感知的时候，使用 tz=None 的情况
            dti = date_range("2016-01-01", periods=10, tz="UTC")

            msg = "Passed data is timezone-aware, incompatible with 'tz=None'"
            with pytest.raises(ValueError, match=msg):
                DatetimeIndex(dti, tz=None)

            with pytest.raises(ValueError, match=msg):
                DatetimeIndex(np.array(dti), tz=None)

            msg = "Cannot pass both a timezone-aware dtype and tz=None"
            with pytest.raises(ValueError, match=msg):
                DatetimeIndex([], dtype="M8[ns, UTC]", tz=None)

        def test_freq_validation_with_nat(self):
            # GH#11587 测试在包含 NaT 的情况下，频率验证的错误消息
            msg = (
                "Inferred frequency None from passed values does not conform "
                "to passed frequency D"
            )
            with pytest.raises(ValueError, match=msg):
                DatetimeIndex([pd.NaT, Timestamp("2011-01-01")], freq="D")
            with pytest.raises(ValueError, match=msg):
                DatetimeIndex([pd.NaT, Timestamp("2011-01-01")._value], freq="D")

        # TODO: better place for tests shared by DTI/TDI?
        @pytest.mark.parametrize(
            "index",
            [
                date_range("2016-01-01", periods=5, tz="US/Pacific"),
                pd.timedelta_range("1 Day", periods=5),
            ],
        )
        def test_shallow_copy_inherits_array_freq(self, index):
            # 如果我们将 DTA/TDA 传递给 shallow_copy 而不指定频率，
            # 我们应该继承数组的频率，而不是自己的频率。
            array = index._data

            arr = array[[0, 3, 2, 4, 1]]
            assert arr.freq is None

            result = index._shallow_copy(arr)
            assert result.freq is None
    def test_categorical_preserves_tz(self):
        # GH#18664 retain tz when going DTI-->Categorical-->DTI
        # 创建一个DatetimeIndex对象，包括不同的时间戳和时区为"US/Eastern"
        dti = DatetimeIndex(
            [pd.NaT, "2015-01-01", "1999-04-06 15:14:13", "2015-01-01"], tz="US/Eastern"
        )

        for dtobj in [dti, dti._data]:
            # works for DatetimeIndex or DatetimeArray

            # 使用DatetimeIndex或DatetimeArray创建CategoricalIndex对象
            ci = pd.CategoricalIndex(dtobj)
            # 使用DatetimeIndex或DatetimeArray创建Categorical对象
            carr = pd.Categorical(dtobj)
            # 将DatetimeIndex或DatetimeArray转换为Series对象
            cser = pd.Series(ci)

            for obj in [ci, carr, cser]:
                # 使用对象创建新的DatetimeIndex对象
                result = DatetimeIndex(obj)
                # 断言新创建的DatetimeIndex对象与原始的DatetimeIndex对象相等
                tm.assert_index_equal(result, dti)

    def test_dti_with_period_data_raises(self):
        # GH#23675
        # 创建一个PeriodIndex对象，包含周期字符串"2016Q1"和"2016Q2"，频率为季度
        data = pd.PeriodIndex(["2016Q1", "2016Q2"], freq="Q")

        # 使用pytest断言引发TypeError异常，消息为"PeriodDtype data is invalid"
        with pytest.raises(TypeError, match="PeriodDtype data is invalid"):
            # 尝试使用PeriodIndex对象创建DatetimeIndex对象
            DatetimeIndex(data)

        with pytest.raises(TypeError, match="PeriodDtype data is invalid"):
            # 尝试使用to_datetime函数将PeriodIndex对象转换为DatetimeIndex对象
            to_datetime(data)

        with pytest.raises(TypeError, match="PeriodDtype data is invalid"):
            # 尝试使用period_array函数将PeriodIndex对象转换为DatetimeIndex对象
            DatetimeIndex(period_array(data))

        with pytest.raises(TypeError, match="PeriodDtype data is invalid"):
            # 尝试使用to_datetime函数将period_array转换为DatetimeIndex对象
            to_datetime(period_array(data))

    def test_dti_with_timedelta64_data_raises(self):
        # GH#23675 deprecated, enforrced in GH#29794
        # 创建一个包含单个值的numpy数组，数据类型为"timedelta64[ns]"
        data = np.array([0], dtype="m8[ns]")
        # 设置期望的错误消息
        msg = r"timedelta64\[ns\] cannot be converted to datetime64"
        # 使用pytest断言引发TypeError异常，消息为设置的msg变量值
        with pytest.raises(TypeError, match=msg):
            # 尝试使用numpy数组创建DatetimeIndex对象
            DatetimeIndex(data)

        with pytest.raises(TypeError, match=msg):
            # 尝试使用to_datetime函数将numpy数组转换为DatetimeIndex对象
            to_datetime(data)

        with pytest.raises(TypeError, match=msg):
            # 尝试使用pd.TimedeltaIndex对象创建DatetimeIndex对象
            DatetimeIndex(pd.TimedeltaIndex(data))

        with pytest.raises(TypeError, match=msg):
            # 尝试使用to_datetime函数将pd.TimedeltaIndex对象转换为DatetimeIndex对象
            to_datetime(pd.TimedeltaIndex(data))

    def test_constructor_from_sparse_array(self):
        # https://github.com/pandas-dev/pandas/issues/35843
        # 创建一个包含两个Timestamp对象的SparseArray对象
        values = [
            Timestamp("2012-05-01T01:00:00.000000"),
            Timestamp("2016-05-01T01:00:00.000000"),
        ]
        arr = pd.arrays.SparseArray(values)
        # 使用SparseArray对象创建Index对象
        result = Index(arr)
        # 断言result对象的类型是Index，并且其数据类型与arr对象相同
        assert type(result) is Index
        assert result.dtype == arr.dtype

    def test_construction_caching(self):
        # https://github.com/pandas-dev/pandas/issues/35843
        # 创建一个DataFrame对象，包含不同日期范围和时区为"US/Eastern"的日期范围
        df = pd.DataFrame(
            {
                "dt": date_range("20130101", periods=3),
                "dttz": date_range(
                    "20130101", periods=3, tz=zoneinfo.ZoneInfo("US/Eastern")
                ),
                "dt_with_null": [
                    Timestamp("20130101"),
                    pd.NaT,
                    Timestamp("20130103"),
                ],
                "dtns": date_range("20130101", periods=3, freq="ns"),
            }
        )
        # 断言DataFrame对象的dttz列的时区关键字为"US/Eastern"
        assert df.dttz.dtype.tz.key == "US/Eastern"

    @pytest.mark.parametrize(
        "kwargs",
        [{"tz": "dtype.tz"}, {"dtype": "dtype"}, {"dtype": "dtype", "tz": "dtype.tz"}],
    )
    # 使用参数 kwargs 和 tz_aware_fixture 进行测试构造函数的替代路径
    def test_construction_with_alt(self, kwargs, tz_aware_fixture):
        # 设置时区
        tz = tz_aware_fixture
        # 创建一个日期范围，每小时频率，使用给定的时区
        i = date_range("20130101", periods=5, freq="h", tz=tz)
        # 将 kwargs 中指定的键值对映射为属性值，并重新赋值给 kwargs
        kwargs = {key: attrgetter(val)(i) for key, val in kwargs.items()}
        # 使用给定的 kwargs 构造一个 DatetimeIndex 对象
        result = DatetimeIndex(i, **kwargs)
        # 断言构造的结果与原始日期范围相等
        tm.assert_index_equal(i, result)

    # 使用参数化测试来测试具有备用时区本地化的构造函数
    @pytest.mark.parametrize(
        "kwargs",
        [{"tz": "dtype.tz"}, {"dtype": "dtype"}, {"dtype": "dtype", "tz": "dtype.tz"}],
    )
    def test_construction_with_alt_tz_localize(self, kwargs, tz_aware_fixture):
        # 设置时区
        tz = tz_aware_fixture
        # 创建一个日期范围，每小时频率，使用给定的时区
        i = date_range("20130101", periods=5, freq="h", tz=tz)
        # 将日期范围的频率设置为空
        i = i._with_freq(None)
        # 将 kwargs 中指定的键值对映射为属性值，并重新赋值给 kwargs
        kwargs = {key: attrgetter(val)(i) for key, val in kwargs.items()}

        # 如果 kwargs 中包含 'tz' 键
        if "tz" in kwargs:
            # 创建一个以 UTC 时区为基础的日期索引，并转换为 kwargs 中指定的时区
            result = DatetimeIndex(i.asi8, tz="UTC").tz_convert(kwargs["tz"])

            # 构造预期的日期索引对象
            expected = DatetimeIndex(i, **kwargs)
            # 断言结果与预期相等
            tm.assert_index_equal(result, expected)

        # 将日期本地化到指定的时区
        i2 = DatetimeIndex(i.tz_localize(None).asi8, tz="UTC")
        expected = i.tz_localize(None).tz_localize("UTC")
        # 断言结果与预期相等
        tm.assert_index_equal(i2, expected)

        # 时区/数据类型不兼容情况的处理
        msg = "cannot supply both a tz and a dtype with a tz"
        # 使用 pytest 断言检查是否引发 ValueError 异常，并检查异常消息是否匹配
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(
                i.tz_localize(None).asi8,
                dtype=i.dtype,
                tz=zoneinfo.ZoneInfo("US/Hawaii"),
            )

    # 测试使用基础构造函数创建 DatetimeIndex 对象
    def test_construction_base_constructor(self):
        # 创建日期时间戳数组
        arr = [Timestamp("2011-01-01"), pd.NaT, Timestamp("2011-01-03")]
        # 断言索引对象与 DatetimeIndex 对象相等
        tm.assert_index_equal(Index(arr), DatetimeIndex(arr))
        # 断言索引对象与使用 numpy 数组创建的 DatetimeIndex 对象相等
        tm.assert_index_equal(Index(np.array(arr)), DatetimeIndex(np.array(arr)))

        # 创建包含 NaN 值的日期时间戳数组
        arr = [np.nan, pd.NaT, Timestamp("2011-01-03")]
        # 断言索引对象与 DatetimeIndex 对象相等
        tm.assert_index_equal(Index(arr), DatetimeIndex(arr))
        # 断言索引对象与使用 numpy 数组创建的 DatetimeIndex 对象相等
        tm.assert_index_equal(Index(np.array(arr)), DatetimeIndex(np.array(arr)))

    # 测试在超出范围时构造 DatetimeIndex 对象
    def test_construction_outofbounds(self):
        # 创建日期时间对象列表，包含超出当前支持的日期范围的日期
        dates = [
            datetime(3000, 1, 1),
            datetime(4000, 1, 1),
            datetime(5000, 1, 1),
            datetime(6000, 1, 1),
        ]
        # 期望的日期时间索引对象，使用 'M8[us]' 数据类型
        exp = Index(dates, dtype="M8[us]")
        # 创建日期时间索引对象
        res = Index(dates)
        # 断言索引对象相等
        tm.assert_index_equal(res, exp)

        # 创建 DatetimeIndex 对象，超出范围的日期时间会被截断处理
        DatetimeIndex(dates)

    # 使用参数化测试来测试超出支持日期范围的日期时间构造
    @pytest.mark.parametrize("data", [["1400-01-01"], [datetime(1400, 1, 1)]])
    def test_dti_date_out_of_range(self, data):
        # GH#1475
        # 创建 DatetimeIndex 对象，用于测试超出支持日期范围的日期时间
        DatetimeIndex(data)

    # 测试使用 ndarray 构造 DatetimeIndex 对象
    def test_construction_with_ndarray(self):
        # 创建日期时间对象列表
        dates = [datetime(2013, 10, 7), datetime(2013, 10, 8), datetime(2013, 10, 9)]
        # 创建日期时间索引对象，并使用 BDay 频率
        data = DatetimeIndex(dates, freq=offsets.BDay()).values
        # 使用 ndarray 创建 DatetimeIndex 对象，并指定 BDay 频率
        result = DatetimeIndex(data, freq=offsets.BDay())
        # 期望的日期时间索引对象，指定数据类型为 'M8[us]'，频率为 'B'
        expected = DatetimeIndex(
            ["2013-10-07", "2013-10-08", "2013-10-09"], dtype="M8[us]", freq="B"
        )
        # 断言结果与预期相等
        tm.assert_index_equal(result, expected)
    # 测试函数：验证整数值和时区被解释为 UTC
    def test_integer_values_and_tz_interpreted_as_utc(self):
        # 创建一个 np.datetime64 对象，表示 "2000-01-01 00:00:00" 的纳秒精度时间戳
        val = np.datetime64("2000-01-01 00:00:00", "ns")
        # 创建一个包含该时间戳视图的 numpy 数组
        values = np.array([val.view("i8")])

        # 使用 DatetimeIndex 类初始化对象，指定时区为 "US/Central"
        result = DatetimeIndex(values).tz_localize("US/Central")

        # 期望的 DatetimeIndex 对象，包含一个日期时间索引 "2000-01-01T00:00:00"，数据类型为 "M8[ns, US/Central]"
        expected = DatetimeIndex(["2000-01-01T00:00:00"], dtype="M8[ns, US/Central]")
        # 断言两个 DatetimeIndex 对象是否相等
        tm.assert_index_equal(result, expected)

        # 但是 UTC 时区并未被弃用
        # 断言在没有警告的情况下执行以下代码块
        with tm.assert_produces_warning(None):
            # 使用 DatetimeIndex 类初始化对象，指定时区为 "UTC"
            result = DatetimeIndex(values, tz="UTC")
        # 期望的 DatetimeIndex 对象，包含一个日期时间索引 "2000-01-01T00:00:00"，数据类型为 "M8[ns, UTC]"
        expected = DatetimeIndex(["2000-01-01T00:00:00"], dtype="M8[ns, UTC]")
        # 断言两个 DatetimeIndex 对象是否相等
        tm.assert_index_equal(result, expected)

    # 测试函数：测试 DatetimeIndex 的构造覆盖范围
    def test_constructor_coverage(self):
        # 消息字符串，表明 DatetimeIndex 对象必须使用一个集合来调用
        msg = r"DatetimeIndex\(\.\.\.\) must be called with a collection"
        # 断言在调用 DatetimeIndex 类初始化时会抛出 TypeError 异常，并且异常消息匹配 msg
        with pytest.raises(TypeError, match=msg):
            DatetimeIndex("1/1/2000")

        # 生成器表达式
        gen = (datetime(2000, 1, 1) + timedelta(i) for i in range(10))
        # 使用 DatetimeIndex 类初始化对象，传入生成器对象 gen
        result = DatetimeIndex(gen)
        # 期望的 DatetimeIndex 对象，包含从 "2000-01-01" 开始的 10 个日期时间索引
        expected = DatetimeIndex(
            [datetime(2000, 1, 1) + timedelta(i) for i in range(10)]
        )
        # 断言两个 DatetimeIndex 对象是否相等
        tm.assert_index_equal(result, expected)

        # NumPy 字符串数组
        strings = np.array(["2000-01-01", "2000-01-02", "2000-01-03"])
        # 使用 DatetimeIndex 类初始化对象，传入字符串数组 strings
        result = DatetimeIndex(strings)
        # 期望的 DatetimeIndex 对象，使用字符串数组 strings 转换为对象，数据类型为 "O"
        expected = DatetimeIndex(strings.astype("O"))
        # 断言两个 DatetimeIndex 对象是否相等
        tm.assert_index_equal(result, expected)

        # 从整数类型创建 DatetimeIndex 对象，先转换为纳秒单位，再转换为秒单位
        from_ints = DatetimeIndex(expected.as_unit("ns").asi8).as_unit("s")
        # 断言两个 DatetimeIndex 对象是否相等
        tm.assert_index_equal(from_ints, expected)

        # 包含 NaT 的字符串数组
        strings = np.array(["2000-01-01", "2000-01-02", "NaT"])
        # 使用 DatetimeIndex 类初始化对象，传入字符串数组 strings
        result = DatetimeIndex(strings)
        # 期望的 DatetimeIndex 对象，使用字符串数组 strings 转换为对象，数据类型为 "O"
        expected = DatetimeIndex(strings.astype("O"))
        # 断言两个 DatetimeIndex 对象是否相等
        tm.assert_index_equal(result, expected)

        # 非符合预期的测试
        # 消息字符串，表明从传入的值推断的频率 None 与传入的频率 "D" 不一致
        msg = (
            "Inferred frequency None from passed values does not conform "
            "to passed frequency D"
        )
        # 断言在调用 DatetimeIndex 类初始化时会抛出 ValueError 异常，并且异常消息匹配 msg
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(["2000-01-01", "2000-01-02", "2000-01-04"], freq="D")
    def test_constructor_datetime64_tzformat(self, freq):
        # see GH#6572: ISO 8601 format results in stdlib timezone object
        # 使用 ISO 8601 格式的日期范围创建时间索引对象
        idx = date_range(
            "2013-01-01T00:00:00-05:00", "2016-01-01T23:59:59-05:00", freq=freq
        )
        # 预期结果使用特定时区偏移
        expected = date_range(
            "2013-01-01T00:00:00",
            "2016-01-01T23:59:59",
            freq=freq,
            tz=timezone(timedelta(minutes=-300)),
        )
        # 断言确保索引对象与预期结果相等
        tm.assert_index_equal(idx, expected)
        
        # Unable to use `US/Eastern` because of DST
        # 由于夏令时问题，无法使用 `US/Eastern` 时区
        expected_i8 = date_range(
            "2013-01-01T00:00:00", "2016-01-01T23:59:59", freq=freq, tz="America/Lima"
        )
        # 断言确保 NumPy 数组表示的索引与预期相等
        tm.assert_numpy_array_equal(idx.asi8, expected_i8.asi8)

        idx = date_range(
            "2013-01-01T00:00:00+09:00", "2016-01-01T23:59:59+09:00", freq=freq
        )
        # 使用特定时区偏移创建时间索引对象
        expected = date_range(
            "2013-01-01T00:00:00",
            "2016-01-01T23:59:59",
            freq=freq,
            tz=timezone(timedelta(minutes=540)),
        )
        # 断言确保索引对象与预期结果相等
        tm.assert_index_equal(idx, expected)
        
        expected_i8 = date_range(
            "2013-01-01T00:00:00", "2016-01-01T23:59:59", freq=freq, tz="Asia/Tokyo"
        )
        # 断言确保 NumPy 数组表示的索引与预期相等
        tm.assert_numpy_array_equal(idx.asi8, expected_i8.asi8)

        # Non ISO 8601 format results in dateutil.tz.tzoffset
        # 使用非 ISO 8601 格式的日期范围，结果使用 dateutil.tz.tzoffset 表示
        idx = date_range("2013/1/1 0:00:00-5:00", "2016/1/1 23:59:59-5:00", freq=freq)
        # 预期结果使用特定时区偏移
        expected = date_range(
            "2013-01-01T00:00:00",
            "2016-01-01T23:59:59",
            freq=freq,
            tz=timezone(timedelta(minutes=-300)),
        )
        # 断言确保索引对象与预期结果相等
        tm.assert_index_equal(idx, expected)
        
        # Unable to use `US/Eastern` because of DST
        # 由于夏令时问题，无法使用 `US/Eastern` 时区
        expected_i8 = date_range(
            "2013-01-01T00:00:00", "2016-01-01T23:59:59", freq=freq, tz="America/Lima"
        )
        # 断言确保 NumPy 数组表示的索引与预期相等
        tm.assert_numpy_array_equal(idx.asi8, expected_i8.asi8)

        idx = date_range("2013/1/1 0:00:00+9:00", "2016/1/1 23:59:59+09:00", freq=freq)
        # 使用特定时区偏移创建时间索引对象
        expected = date_range(
            "2013-01-01T00:00:00",
            "2016-01-01T23:59:59",
            freq=freq,
            tz=timezone(timedelta(minutes=540)),
        )
        # 断言确保索引对象与预期结果相等
        tm.assert_index_equal(idx, expected)
        
        expected_i8 = date_range(
            "2013-01-01T00:00:00", "2016-01-01T23:59:59", freq=freq, tz="Asia/Tokyo"
        )
        # 断言确保 NumPy 数组表示的索引与预期相等
        tm.assert_numpy_array_equal(idx.asi8, expected_i8.asi8)

    def test_constructor_dtype(self):
        # passing a dtype with a tz should localize
        # 使用带有时区信息的 dtype 应该进行本地化处理
        idx = DatetimeIndex(
            ["2013-01-01", "2013-01-02"], dtype="datetime64[ns, US/Eastern]"
        )
        expected = (
            DatetimeIndex(["2013-01-01", "2013-01-02"])
            .as_unit("ns")
            .tz_localize("US/Eastern")
        )
        # 断言确保索引对象与预期结果相等
        tm.assert_index_equal(idx, expected)

        idx = DatetimeIndex(["2013-01-01", "2013-01-02"], tz="US/Eastern").as_unit("ns")
        # 断言确保索引对象与预期结果相等
        tm.assert_index_equal(idx, expected)
    def test_constructor_dtype_tz_mismatch_raises(self):
        # 创建一个包含日期时间索引的对象，指定了时区为 US/Eastern
        idx = DatetimeIndex(
            ["2013-01-01", "2013-01-02"], dtype="datetime64[ns, US/Eastern]"
        )

        # 设置错误消息，当尝试使用不匹配的时区或未指定时区时抛出 ValueError
        msg = (
            "cannot supply both a tz and a timezone-naive dtype "
            r"\(i\.e\. datetime64\[ns\]\)"
        )
        with pytest.raises(ValueError, match=msg):
            # 尝试创建一个新的日期时间索引对象，但其数据类型缺少时区信息
            DatetimeIndex(idx, dtype="datetime64[ns]")

        # 尝试将已经指定了时区的日期时间索引对象设置为不同的时区，预期会抛出 TypeError
        msg = "data is already tz-aware US/Eastern, unable to set specified tz: CET"
        with pytest.raises(TypeError, match=msg):
            # 尝试设置已经指定了时区的日期时间索引对象为另一个时区
            DatetimeIndex(idx, dtype="datetime64[ns, CET]")

        # 尝试在设置时区的同时使用另一个带有时区的数据类型，预期会抛出 ValueError
        msg = "cannot supply both a tz and a dtype with a tz"
        with pytest.raises(ValueError, match=msg):
            # 尝试同时指定日期时间索引对象的时区和数据类型的时区信息
            DatetimeIndex(idx, tz="CET", dtype="datetime64[ns, US/Eastern]")

        # 创建一个新的日期时间索引对象，验证其与原始索引的一致性
        result = DatetimeIndex(idx, dtype="datetime64[ns, US/Eastern]")
        tm.assert_index_equal(idx, result)

    @pytest.mark.parametrize("dtype", [object, np.int32, np.int64])
    def test_constructor_invalid_dtype_raises(self, dtype):
        # GH 23986
        # 设置错误消息，用于检测当传入非法的数据类型时是否会抛出 ValueError
        msg = "Unexpected value for 'dtype'"
        with pytest.raises(ValueError, match=msg):
            # 尝试使用非法的数据类型来创建日期时间索引对象
            DatetimeIndex([1, 2], dtype=dtype)

    def test_000constructor_resolution(self):
        # 2252
        # 创建一个时间戳对象，表示特定的时间点
        t1 = Timestamp((1352934390 * 1000000000) + 1000000 + 1000 + 1)
        # 使用时间戳创建一个日期时间索引对象
        idx = DatetimeIndex([t1])

        # 断言索引对象的纳秒部分与时间戳的纳秒部分相等
        assert idx.nanosecond[0] == t1.nanosecond

    def test_disallow_setting_tz(self):
        # GH 3746
        # 创建一个带有时区信息为 UTC 的日期时间索引对象
        dti = DatetimeIndex(["2010"], tz="UTC")
        # 设置错误消息，用于检测是否能直接设置时区属性是否会抛出 AttributeError
        msg = "Cannot directly set timezone"
        with pytest.raises(AttributeError, match=msg):
            # 尝试直接设置日期时间索引对象的时区属性为另一个时区
            dti.tz = zoneinfo.ZoneInfo("US/Pacific")

    @pytest.mark.parametrize(
        "tz",
        [
            None,
            "America/Los_Angeles",
            pytz.timezone("America/Los_Angeles"),
            Timestamp("2000", tz="America/Los_Angeles").tz,
        ],
    )
    def test_constructor_start_end_with_tz(self, tz):
        # GH 18595
        # 创建起始和结束时间戳，均带有时区信息为 America/Los_Angeles
        start = Timestamp("2013-01-01 06:00:00", tz="America/Los_Angeles")
        end = Timestamp("2013-01-02 06:00:00", tz="America/Los_Angeles")
        # 使用起始和结束时间戳创建一个日期范围，带有给定的时区信息
        result = date_range(freq="D", start=start, end=end, tz=tz)
        # 创建一个预期的日期时间索引对象，验证其与结果的一致性
        expected = DatetimeIndex(
            ["2013-01-01 06:00:00", "2013-01-02 06:00:00"],
            dtype="M8[ns, America/Los_Angeles]",
            freq="D",
        )
        tm.assert_index_equal(result, expected)
        # 断言 pytz 时区对象是否与结果索引对象的时区属性相同
        assert pytz.timezone("America/Los_Angeles") is result.tz

    @pytest.mark.parametrize("tz", ["US/Pacific", "US/Eastern", "Asia/Tokyo"])
    # 测试用例：使用非标准化的时区进行构造函数测试
    def test_constructor_with_non_normalized_pytz(self, tz):
        # GH 18595
        # 导入 pytest 模块，如果不存在则跳过测试
        pytz = pytest.importorskip("pytz")
        # 使用给定的时区字符串创建 pytz 的时区对象
        tz_in = pytz.timezone(tz)
        # 使用指定的时区创建一个 Timestamp 对象，并获取其非规范化的时区信息
        non_norm_tz = Timestamp("2010", tz=tz_in).tz
        # 使用 DatetimeIndex 构造函数创建一个时间索引对象
        result = DatetimeIndex(["2010"], tz=non_norm_tz)
        # 断言结果中的时区与原始时区对象相同
        assert pytz.timezone(tz) is result.tz

    # 测试用例：在夏令时附近创建 Timestamp 的 DatetimeIndex
    def test_constructor_timestamp_near_dst(self):
        # GH 20854
        # 创建两个带有夏令时的 Timestamp 对象
        ts = [
            Timestamp("2016-10-30 03:00:00+0300", tz="Europe/Helsinki"),
            Timestamp("2016-10-30 03:00:00+0200", tz="Europe/Helsinki"),
        ]
        # 使用 DatetimeIndex 构造函数创建时间索引，并将单位设置为纳秒
        result = DatetimeIndex(ts).as_unit("ns")
        # 创建预期的 DatetimeIndex 对象，将 Timestamp 对象转换为 Python datetime 对象
        expected = DatetimeIndex(
            [ts[0].to_pydatetime(), ts[1].to_pydatetime()]
        ).as_unit("ns")
        # 使用测试工具方法比较两个时间索引对象是否相等
        tm.assert_index_equal(result, expected)

    # 参数化测试：使用不同的类和函数对时区和数据类型进行构造函数测试
    @pytest.mark.parametrize("klass", [Index, DatetimeIndex])
    @pytest.mark.parametrize("box", [np.array, partial(np.array, dtype=object), list])
    @pytest.mark.parametrize(
        "tz, dtype",
        [("US/Pacific", "datetime64[ns, US/Pacific]"), (None, "datetime64[ns]")],
    )
    def test_constructor_with_int_tz(self, klass, box, tz, dtype):
        # GH 20997, 20964
        # 创建一个带有指定时区的 Timestamp 对象，并将单位设置为纳秒
        ts = Timestamp("2018-01-01", tz=tz).as_unit("ns")
        # 使用指定的类和数据类型创建一个时间索引对象
        result = klass(box([ts._value]), dtype=dtype)
        # 创建预期的时间索引对象
        expected = klass([ts])
        # 断言结果与预期对象相等
        assert result == expected

    # 测试用例：通过替换带有夏令时的 Timestamp 对象来创建时间索引
    def test_construction_from_replaced_timestamps_with_dst(self):
        # GH 18785
        # 创建一个日期范围对象，其中包含带有夏令时的 Timestamp 对象
        index = date_range(
            Timestamp(2000, 12, 31),
            Timestamp(2005, 12, 31),
            freq="YE-DEC",
            tz="Australia/Melbourne",
        )
        # 使用列表推导式替换每个 Timestamp 对象的月份和日期，创建一个时间索引对象
        result = DatetimeIndex([x.replace(month=6, day=1) for x in index])
        # 创建预期的时间索引对象，设置时区为 "Australia/Melbourne"，并将单位设置为纳秒
        expected = DatetimeIndex(
            [
                "2000-06-01 00:00:00",
                "2001-06-01 00:00:00",
                "2002-06-01 00:00:00",
                "2003-06-01 00:00:00",
                "2004-06-01 00:00:00",
                "2005-06-01 00:00:00",
            ],
            tz="Australia/Melbourne",
        ).as_unit("ns")
        # 使用测试工具方法比较两个时间索引对象是否相等
        tm.assert_index_equal(result, expected)

    # 测试用例：使用时区和时区感知的 DatetimeIndex 构造函数
    def test_construction_with_tz_and_tz_aware_dti(self):
        # GH 23579
        # 创建一个带有时区信息的日期范围对象
        dti = date_range("2016-01-01", periods=3, tz="US/Central")
        # 准备异常消息字符串
        msg = "data is already tz-aware US/Central, unable to set specified tz"
        # 断言在设置时指定不同的时区时会引发 TypeError 异常，且异常消息符合预期
        with pytest.raises(TypeError, match=msg):
            DatetimeIndex(dti, tz="Asia/Tokyo")

    # 测试用例：使用 tzlocal 和 NaT 创建 DatetimeIndex
    def test_construction_with_nat_and_tzlocal(self):
        # 获取本地时区对象
        tz = dateutil.tz.tzlocal()
        # 使用指定时间戳和时区对象创建 DatetimeIndex，单位设置为纳秒
        result = DatetimeIndex(["2018", "NaT"], tz=tz).as_unit("ns")
        # 创建预期的 DatetimeIndex 对象，其中包含一个带有时区信息的 Timestamp 对象和 NaT
        expected = DatetimeIndex([Timestamp("2018", tz=tz), pd.NaT]).as_unit("ns")
        # 使用测试工具方法比较两个时间索引对象是否相等
        tm.assert_index_equal(result, expected)
    def test_constructor_with_ambiguous_keyword_arg(self):
        # GH 35297
        # 定义预期的日期时间索引，指定了时区和模糊性
        expected = DatetimeIndex(
            ["2020-11-01 01:00:00", "2020-11-02 01:00:00"],
            dtype="datetime64[ns, America/New_York]",
            freq="D",
            ambiguous=False,
        )

        # 在开始时间中使用模糊性关键字
        timezone = "America/New_York"
        start = Timestamp(year=2020, month=11, day=1, hour=1).tz_localize(
            timezone, ambiguous=False
        )
        # 创建日期范围，指定开始时间和周期数，并且明确模糊性
        result = date_range(start=start, periods=2, ambiguous=False)
        tm.assert_index_equal(result, expected)

        # 在结束时间中使用模糊性关键字
        end = Timestamp(year=2020, month=11, day=2, hour=1).tz_localize(
            timezone, ambiguous=False
        )
        # 创建日期范围，指定结束时间和周期数，并且明确模糊性
        result = date_range(end=end, periods=2, ambiguous=False)
        tm.assert_index_equal(result, expected)

    def test_constructor_with_nonexistent_keyword_arg(self, warsaw):
        # GH 35297
        # 使用华沙时区
        timezone = warsaw

        # 在开始时间中使用不存在的关键字
        start = Timestamp("2015-03-29 02:30:00").tz_localize(
            timezone, nonexistent="shift_forward"
        )
        # 创建日期范围，指定开始时间和周期数，频率为每小时
        result = date_range(start=start, periods=2, freq="h")
        # 定义预期的日期时间索引，使用指定时区
        expected = DatetimeIndex(
            [
                Timestamp("2015-03-29 03:00:00+02:00", tz=timezone),
                Timestamp("2015-03-29 04:00:00+02:00", tz=timezone),
            ]
        ).as_unit("ns")

        tm.assert_index_equal(result, expected)

        # 在结束时间中使用不存在的关键字
        end = start
        # 创建日期范围，指定结束时间和周期数，频率为每小时
        result = date_range(end=end, periods=2, freq="h")
        # 定义预期的日期时间索引，使用指定时区
        expected = DatetimeIndex(
            [
                Timestamp("2015-03-29 01:00:00+01:00", tz=timezone),
                Timestamp("2015-03-29 03:00:00+02:00", tz=timezone),
            ]
        ).as_unit("ns")

        tm.assert_index_equal(result, expected)

    def test_constructor_no_precision_raises(self):
        # GH-24753, GH-24739
        # 测试在没有精度的情况下是否会触发异常

        msg = "with no precision is not allowed"
        # 使用 pytest 断言是否会抛出 ValueError 异常并包含特定消息
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(["2000"], dtype="datetime64")

        msg = "The 'datetime64' dtype has no unit. Please pass in"
        # 使用 pytest 断言是否会抛出 ValueError 异常并包含特定消息
        with pytest.raises(ValueError, match=msg):
            Index(["2000"], dtype="datetime64")

    def test_constructor_wrong_precision_raises(self):
        # 测试在错误的精度设置下是否能正确转换成合适的类型
        dti = DatetimeIndex(["2000"], dtype="datetime64[us]")
        assert dti.dtype == "M8[us]"
        assert dti[0] == Timestamp(2000, 1, 1)

    def test_index_constructor_with_numpy_object_array_and_timestamp_tz_with_nan(self):
        # GH 27011
        # 创建索引，包含具有时区的时间戳和 NaN
        result = Index(np.array([Timestamp("2019", tz="UTC"), np.nan], dtype=object))
        # 定义预期的日期时间索引，其中包括一个时区设定的时间戳和一个 NaT（Not a Time）
        expected = DatetimeIndex([Timestamp("2019", tz="UTC"), pd.NaT])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "tz", [zoneinfo.ZoneInfo("US/Eastern"), gettz("US/Eastern")]
    )
    # 定义一个测试方法，测试从具有时区信息的日期时间生成 DatetimeIndex
    def test_dti_from_tzaware_datetime(self, tz):
        # 创建一个包含一个带有时区信息的 datetime 对象的列表
        d = [datetime(2012, 8, 19, tzinfo=tz)]
        
        # 使用 DatetimeIndex 构造函数创建索引对象
        index = DatetimeIndex(d)
        # 断言索引对象的时区与给定的时区相同
        assert timezones.tz_compare(index.tz, tz)

    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    # 定义一个参数化测试方法，测试不同的 DatetimeIndex 构造方式与时区
    def test_dti_tz_constructors(self, tzstr):
        """Test different DatetimeIndex constructions with timezone
        Follow-up of GH#4229
        """
        # 创建包含日期时间字符串的数组
        arr = ["11/10/2005 08:00:00", "11/10/2005 09:00:00"]

        # 使用 to_datetime 函数将日期时间字符串转换为 DatetimeIndex，并添加时区信息
        idx1 = to_datetime(arr).tz_localize(tzstr)
        
        # 使用 date_range 函数创建具有时区信息的 DatetimeIndex
        idx2 = date_range(
            start="2005-11-10 08:00:00", freq="h", periods=2, tz=tzstr, unit="s"
        )
        # 将 idx2 的频率设置为 None，与其他索引对象一致
        idx2 = idx2._with_freq(None)
        
        # 使用 DatetimeIndex 构造函数创建索引对象，并将时间单位设置为秒
        idx3 = DatetimeIndex(arr, tz=tzstr).as_unit("s")
        
        # 使用 DatetimeIndex 构造函数创建索引对象，传入 NumPy 数组，并将时间单位设置为秒
        idx4 = DatetimeIndex(np.array(arr), tz=tzstr).as_unit("s")
        
        # 断言不同索引对象的相等性
        tm.assert_index_equal(idx1, idx2)
        tm.assert_index_equal(idx1, idx3)
        tm.assert_index_equal(idx1, idx4)

    # 定义一个测试方法，测试 DatetimeIndex 构造的幂等性
    def test_dti_construction_idempotent(self, unit):
        # 使用 date_range 函数创建带有频率和时区信息的日期范围
        rng = date_range(
            "03/12/2012 00:00", periods=10, freq="W-FRI", tz="US/Eastern", unit=unit
        )
        # 使用 DatetimeIndex 构造函数创建另一个具有相同数据和时区的索引对象
        rng2 = DatetimeIndex(data=rng, tz="US/Eastern")
        # 断言两个索引对象相等
        tm.assert_index_equal(rng, rng2)

    @pytest.mark.parametrize("prefix", ["", "dateutil/"])
    # 定义一个参数化测试方法，测试静态 tzinfo 参数化时区的 DatetimeIndex 构造方式
    def test_dti_constructor_static_tzinfo(self, prefix):
        # 创建只含有一个具有时区信息的 datetime 对象的 DatetimeIndex
        index = DatetimeIndex([datetime(2012, 1, 1)], tz=prefix + "EST")
        # 访问索引对象的小时属性
        index.hour
        # 访问索引对象的第一个元素
        index[0]

    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    # 定义一个参数化测试方法，测试将 datetime 列表转换为 DatetimeIndex 的功能
    def test_dti_convert_datetime_list(self, tzstr):
        # 创建具有时区信息的日期范围对象
        dr = date_range("2012-06-02", periods=10, tz=tzstr, name="foo")
        # 使用 DatetimeIndex 构造函数创建另一个具有相同数据的索引对象，并设置频率为天
        dr2 = DatetimeIndex(list(dr), name="foo", freq="D")
        # 断言两个索引对象相等
        tm.assert_index_equal(dr, dr2)

    @pytest.mark.parametrize(
        "tz",
        [
            "pytz/US/Eastern",
            gettz("US/Eastern"),
        ],
    )
    @pytest.mark.parametrize("use_str", [True, False])
    @pytest.mark.parametrize("box_cls", [Timestamp, DatetimeIndex])
    # 其他参数化测试方法，不涉及具体的代码块注释
    # 测试函数，用于验证在不同情况下，DTI（DatetimeIndex）构造函数和Timestamp构造函数的行为是否一致
    def test_dti_ambiguous_matches_timestamp(self, tz, use_str, box_cls, request):
        # 检查是否需要移除"pytz/"前缀的时区字符串，并转换为对应的时区对象
        if isinstance(tz, str) and tz.startswith("pytz/"):
            tz = pytz.timezone(tz.removeprefix("pytz/"))
        
        # 设定一个日期时间字符串
        dtstr = "2013-11-03 01:59:59.999999"
        item = dtstr
        
        # 如果不使用字符串形式，则将日期时间字符串转换为Python datetime对象
        if not use_str:
            item = Timestamp(dtstr).to_pydatetime()
        
        # 如果box_cls不是Timestamp类，则将item转换为列表
        if box_cls is not Timestamp:
            item = [item]
        
        # 如果不使用字符串形式，并且tz是dateutil.tz.tzfile类型的时区对象
        if not use_str and isinstance(tz, dateutil.tz.tzfile):
            # FIXME: 这里的Timestamp构造函数行为与其他情况不同，因为使用dateutil/zoneinfo时，
            #  我们隐含地得到fold=0。这里的抛出异常不是重点，重点是保持行为在各种情况下的一致性。
            mark = pytest.mark.xfail(reason="We implicitly get fold=0.")
            request.applymarker(mark)
        
        # 断言在给定时区和日期时间字符串情况下会抛出pytz.AmbiguousTimeError异常
        with pytest.raises(pytz.AmbiguousTimeError, match=dtstr):
            box_cls(item, tz=tz)

    # 参数化测试函数，测试DTI构造函数在非纳秒精度情况下的行为
    @pytest.mark.parametrize("tz", [None, "UTC", "US/Pacific"])
    def test_dti_constructor_with_non_nano_dtype(self, tz):
        # GH#55756, GH#54620
        
        # 创建一个Timestamp对象
        ts = Timestamp("2999-01-01")
        
        # 根据tz的值确定dtype的具体格式
        dtype = "M8[us]"
        if tz is not None:
            dtype = f"M8[us, {tz}]"
        
        # 创建一个包含不同类型值的列表
        vals = [ts, "2999-01-02 03:04:05.678910", 2500]
        
        # 使用vals创建一个DatetimeIndex对象
        result = DatetimeIndex(vals, dtype=dtype)
        
        # 解释：2500被解释为微秒，与如果从vals[:2]和vals[2:]创建DatetimeIndexes并连接结果一致。
        
        # 创建一个包含时间点的列表
        pointwise = [
            vals[0].tz_localize(tz),
            Timestamp(vals[1], tz=tz),
            to_datetime(vals[2], unit="us", utc=True).tz_convert(tz),
        ]
        
        # 将时间点转换为微秒单位的numpy.datetime64类型值，并组成数组
        exp_vals = [x.as_unit("us").asm8 for x in pointwise]
        exp_arr = np.array(exp_vals, dtype="M8[us]")
        
        # 创建预期的DatetimeIndex对象
        expected = DatetimeIndex(exp_arr, dtype="M8[us]")
        
        # 如果tz不为None，则将expected对象本地化到UTC，并转换为tz时区
        if tz is not None:
            expected = expected.tz_localize("UTC").tz_convert(tz)
        
        # 断言result与expected对象相等
        tm.assert_index_equal(result, expected)
        
        # 使用包含对象的vals数组创建DatetimeIndex对象
        result2 = DatetimeIndex(np.array(vals, dtype=object), dtype=dtype)
        
        # 断言result2与expected对象相等
        tm.assert_index_equal(result2, expected)
    def test_dti_constructor_with_non_nano_now_today(self, request):
        # 测试DatetimeIndex构造函数处理非纳秒精度的now和today方法
        # GH#55756
        # 获取当前时间戳
        now = Timestamp.now()
        # 获取今天的时间戳
        today = Timestamp.today()
        # 使用DatetimeIndex构造函数创建时间索引对象，指定dtype为"M8[s]"
        result = DatetimeIndex(["now", "today"], dtype="M8[s]")
        # 断言结果的dtype为"M8[s]"
        assert result.dtype == "M8[s]"

        # 计算索引0的时间差
        diff0 = result[0] - now.as_unit("s")
        # 计算索引1的时间差
        diff1 = result[1] - today.as_unit("s")
        # 断言diff1大于等于零
        assert diff1 >= pd.Timedelta(0), f"The difference is {diff0}"
        # 断言diff0大于等于零
        assert diff0 >= pd.Timedelta(0), f"The difference is {diff0}"

        # 由于结果可能不完全匹配[now, today]，因此我们将使用容差来测试。
        # （由于四舍五入的原因，它们可能会完全匹配）
        # GH 57535
        # 将xfail标记应用于测试，理由是结果可能不完全匹配[now, today]
        request.applymarker(
            pytest.mark.xfail(
                reason="result may not exactly match [now, today]", strict=False
            )
        )
        # 设置容差为1秒
        tolerance = pd.Timedelta(seconds=1)
        # 断言diff0小于容差
        assert diff0 < tolerance, f"The difference is {diff0}"
        # 断言diff1小于容差
        assert diff1 < tolerance, f"The difference is {diff0}"

    def test_dti_constructor_object_float_matches_float_dtype(self):
        # 测试DatetimeIndex构造函数处理对象路径与浮点数dtype匹配的情况
        # GH#55780
        # 创建一个包含整数和NaN的NumPy数组，dtype为np.float64
        arr = np.array([0, np.nan], dtype=np.float64)
        # 将arr转换为对象类型的数组
        arr2 = arr.astype(object)

        # 使用DatetimeIndex构造函数创建时间索引对象，指定时区为CET
        dti1 = DatetimeIndex(arr, tz="CET")
        dti2 = DatetimeIndex(arr2, tz="CET")
        # 断言两个时间索引对象相等
        tm.assert_index_equal(dti1, dti2)

    @pytest.mark.parametrize("dtype", ["M8[us]", "M8[us, US/Pacific]"])
    def test_dti_constructor_with_dtype_object_int_matches_int_dtype(self, dtype):
        # 测试DatetimeIndex构造函数处理dtype为对象类型时与整数dtype匹配的情况
        # 通过对象路径应该与非对象路径匹配

        # 创建一个包含整数的NumPy数组
        vals1 = np.arange(5, dtype="i8") * 1000
        # 将第一个元素设置为NaT的值
        vals1[0] = pd.NaT.value

        # 将vals1转换为np.float64类型的数组
        vals2 = vals1.astype(np.float64)
        # 将第一个元素设置为NaN
        vals2[0] = np.nan

        # 将vals1转换为对象类型的数组
        vals3 = vals1.astype(object)
        # 将lib.infer_dtype(vals3)的结果从"integer"更改为pd.NaT，以便通过array_to_datetime在_sequence_to_dt64中进行处理
        vals3[0] = pd.NaT

        # 将vals2转换为对象类型的数组
        vals4 = vals2.astype(object)

        # 使用DatetimeIndex构造函数创建时间索引对象，指定dtype为参数传递进来的值
        res1 = DatetimeIndex(vals1, dtype=dtype)
        res2 = DatetimeIndex(vals2, dtype=dtype)
        res3 = DatetimeIndex(vals3, dtype=dtype)
        res4 = DatetimeIndex(vals4, dtype=dtype)

        # 创建一个预期的DatetimeIndex对象，使用vals1.view("M8[us]")作为基础
        expected = DatetimeIndex(vals1.view("M8[us]"))
        # 如果res1有时区信息，则将预期对象本地化为UTC时区，并转换为res1的时区
        if res1.tz is not None:
            expected = expected.tz_localize("UTC").tz_convert(res1.tz)
        # 断言四个结果与预期对象相等
        tm.assert_index_equal(res1, expected)
        tm.assert_index_equal(res2, expected)
        tm.assert_index_equal(res3, expected)
        tm.assert_index_equal(res4, expected)
class TestTimeSeries:
    # 测试日期时间索引构造函数，确保保留日期时间索引的频率信息
    def test_dti_constructor_preserve_dti_freq(self):
        # 创建一个日期范围，频率为每5分钟
        rng = date_range("1/1/2000", "1/2/2000", freq="5min")

        # 使用日期时间索引对象构造函数构建另一个日期时间索引对象
        rng2 = DatetimeIndex(rng)
        # 断言两个日期时间索引对象的频率相同
        assert rng.freq == rng2.freq

    # 测试显式指定频率为 None 的情况
    def test_explicit_none_freq(self):
        # 显式传递 freq=None 时保持其值不变
        rng = date_range("1/1/2000", "1/2/2000", freq="5min")

        # 使用 freq=None 构造日期时间索引对象
        result = DatetimeIndex(rng, freq=None)
        # 断言结果日期时间索引对象的频率为 None
        assert result.freq is None

        # 使用数组的数据和 freq=None 构造日期时间索引对象
        result = DatetimeIndex(rng._data, freq=None)
        # 断言结果日期时间索引对象的频率为 None
        assert result.freq is None

    # 测试日期时间索引构造函数与小整数的情况
    def test_dti_constructor_small_int(self, any_int_numpy_dtype):
        # 见 GitHub 问题 #13721
        # 期望的日期时间索引对象
        exp = DatetimeIndex(
            [
                "1970-01-01 00:00:00.00000000",
                "1970-01-01 00:00:00.00000001",
                "1970-01-01 00:00:00.00000002",
            ]
        )

        # 创建一个小整数类型的 NumPy 数组
        arr = np.array([0, 10, 20], dtype=any_int_numpy_dtype)
        # 断言构造的日期时间索引对象与期望的日期时间索引对象相等
        tm.assert_index_equal(DatetimeIndex(arr), exp)

    # 测试字符串形式的日内时间构造函数
    def test_ctor_str_intraday(self):
        # 使用字符串构造日期时间索引对象，包含秒信息
        rng = DatetimeIndex(["1-1-2000 00:00:01"])
        # 断言日期时间索引对象的第一个元素的秒数为 1
        assert rng[0].second == 1

    # 测试索引类型转换为 datetime64 其它单位的情况
    def test_index_cast_datetime64_other_units(self):
        # 创建一个 int64 类型的数组，并转换为 'M8[D]' 类型
        arr = np.arange(0, 100, 10, dtype=np.int64).view("M8[D]")
        # 使用数组构造索引对象
        idx = Index(arr)

        # 断言索引对象的值经过转换后与给定的 dtype=np.dtype("M8[ns]") 的结果一致
        assert (idx.values == astype_overflowsafe(arr, dtype=np.dtype("M8[ns]"))).all()

    # 测试构造函数中的 int64 类型数组的情况（不复制数据）
    def test_constructor_int64_nocopy(self):
        # GitHub 问题 #1624
        # 创建一个从 0 到 999 的 int64 类型的数组
        arr = np.arange(1000, dtype=np.int64)
        # 使用数组构造日期时间索引对象
        index = DatetimeIndex(arr)

        # 修改数组的部分数据为 -1，并断言索引对象相应的数据也为 -1
        arr[50:100] = -1
        assert (index.asi8[50:100] == -1).all()

        # 创建一个新的从 0 到 999 的 int64 类型的数组（复制数据）
        arr = np.arange(1000, dtype=np.int64)
        # 使用数组和 copy=True 构造日期时间索引对象
        index = DatetimeIndex(arr, copy=True)

        # 修改数组的部分数据为 -1，并断言索引对象相应的数据不为 -1
        assert (index.asi8[50:100] != -1).all()

    # 使用 pytest 的参数化测试，根据不同的频率重建日期时间索引对象
    @pytest.mark.parametrize(
        "freq",
        ["ME", "QE", "YE", "D", "B", "bh", "min", "s", "ms", "us", "h", "ns", "C"],
    )
    def test_from_freq_recreate_from_data(self, freq):
        # 创建原始的日期范围对象，指定频率和周期数
        org = date_range(start="2001/02/01 09:00", freq=freq, periods=1)
        # 使用原始的日期范围对象和指定的频率构造日期时间索引对象
        idx = DatetimeIndex(org, freq=freq)
        # 断言重建的日期时间索引对象与原始对象相等
        tm.assert_index_equal(idx, org)

        # 创建带时区信息的原始日期范围对象，指定频率、时区和周期数
        org = date_range(
            start="2001/02/01 09:00", freq=freq, tz="US/Pacific", periods=1
        )
        # 使用带时区信息的原始日期范围对象、指定的频率和时区构造日期时间索引对象
        idx = DatetimeIndex(org, freq=freq, tz="US/Pacific")
        # 断言重建的日期时间索引对象与原始对象相等
        tm.assert_index_equal(idx, org)
    # 定义测试方法，用于测试 DatetimeIndex 构造函数中的各种情况
    def test_datetimeindex_constructor_misc(self):
        # 准备日期字符串数组
        arr = ["1/1/2005", "1/2/2005", "Jn 3, 2005", "2005-01-04"]
        # 准备异常匹配模式字符串
        msg = r"(\(')?Unknown datetime string format(:', 'Jn 3, 2005'\))?"
        # 使用 pytest 检查是否抛出 ValueError 异常，并匹配预期的异常消息
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(arr)

        # 创建 DatetimeIndex 对象，从日期字符串数组 arr 中
        arr = ["1/1/2005", "1/2/2005", "1/3/2005", "2005-01-04"]
        idx1 = DatetimeIndex(arr)

        # 创建 DatetimeIndex 对象，包括 datetime 对象和日期字符串
        arr = [datetime(2005, 1, 1), "1/2/2005", "1/3/2005", "2005-01-04"]
        idx2 = DatetimeIndex(arr)

        # 创建 DatetimeIndex 对象，包括 Timestamp 对象和日期字符串
        arr = [Timestamp(datetime(2005, 1, 1)), "1/2/2005", "1/3/2005", "2005-01-04"]
        idx3 = DatetimeIndex(arr)

        # 创建 DatetimeIndex 对象，使用 numpy 数组作为输入
        arr = np.array(["1/1/2005", "1/2/2005", "1/3/2005", "2005-01-04"], dtype="O")
        idx4 = DatetimeIndex(arr)

        # 使用 dayfirst 参数创建 DatetimeIndex 对象，比较两个索引对象是否相等
        idx5 = DatetimeIndex(["12/05/2007", "25/01/2008"], dayfirst=True)
        idx6 = DatetimeIndex(
            ["2007/05/12", "2008/01/25"], dayfirst=False, yearfirst=True
        )
        tm.assert_index_equal(idx5, idx6)

        # 对于其他 DatetimeIndex 对象，检查它们的值是否相等
        for other in [idx2, idx3, idx4]:
            assert (idx1.values == other.values).all()

    # 定义测试方法，验证带有时区、dayfirst 和 yearfirst 参数的 DatetimeIndex 构造函数
    def test_dti_constructor_object_dtype_dayfirst_yearfirst_with_tz(self):
        # 测试 GH#55813 情况
        val = "5/10/16"

        # 创建指定时区的 Timestamp 对象
        dfirst = Timestamp(2016, 10, 5, tz="US/Pacific")
        yfirst = Timestamp(2005, 10, 16, tz="US/Pacific")

        # 使用 dayfirst 参数创建 DatetimeIndex 对象，并验证与预期的索引对象相等
        result1 = DatetimeIndex([val], tz="US/Pacific", dayfirst=True)
        expected1 = DatetimeIndex([dfirst]).as_unit("s")
        tm.assert_index_equal(result1, expected1)

        # 使用 yearfirst 参数创建 DatetimeIndex 对象，并验证与预期的索引对象相等
        result2 = DatetimeIndex([val], tz="US/Pacific", yearfirst=True)
        expected2 = DatetimeIndex([yfirst]).as_unit("s")
        tm.assert_index_equal(result2, expected2)
```