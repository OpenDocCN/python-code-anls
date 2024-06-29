# `D:\src\scipysrc\pandas\pandas\tests\series\accessors\test_dt_accessor.py`

```
import calendar  # 导入日历模块
from datetime import (  # 从 datetime 模块导入以下类和函数
    date,  # 日期类
    datetime,  # 日期时间类
    time,  # 时间类
)
import locale  # 导入 locale 模块，用于处理特定地域的格式化
import unicodedata  # 导入 unicodedata 模块，用于Unicode字符数据的处理

import numpy as np  # 导入 NumPy 库并重命名为 np
import pytest  # 导入 pytest 测试框架
import pytz  # 导入 pytz 用于处理时区

from pandas._libs.tslibs.timezones import maybe_get_tz  # 从 pandas 库导入 maybe_get_tz 函数

from pandas.core.dtypes.common import (  # 从 pandas 核心数据类型导入以下函数
    is_integer_dtype,  # 判断是否为整数类型
    is_list_like,  # 判断是否为列表类型
)

import pandas as pd  # 导入 pandas 库并重命名为 pd
from pandas import (  # 从 pandas 库导入以下类和函数
    DataFrame,  # 数据框类
    DatetimeIndex,  # 日期时间索引类
    Index,  # 索引类
    Period,  # 时期类
    PeriodIndex,  # 时期索引类
    Series,  # 序列类
    TimedeltaIndex,  # 时间增量索引类
    date_range,  # 日期范围生成函数
    period_range,  # 时期范围生成函数
    timedelta_range,  # 时间增量范围生成函数
)
import pandas._testing as tm  # 导入 pandas 测试模块，并重命名为 tm
from pandas.core.arrays import (  # 从 pandas 核心数组导入以下类
    DatetimeArray,  # 日期时间数组类
    PeriodArray,  # 时期数组类
    TimedeltaArray,  # 时间增量数组类
)

ok_for_period = PeriodArray._datetimelike_ops  # 检查时期数组是否支持日期时间操作
ok_for_period_methods = ["strftime", "to_timestamp", "asfreq"]  # 支持时期方法列表
ok_for_dt = DatetimeArray._datetimelike_ops  # 检查日期时间数组是否支持日期时间操作
ok_for_dt_methods = [  # 支持日期时间方法列表
    "to_period", "to_pydatetime", "tz_localize", "tz_convert", 
    "normalize", "strftime", "round", "floor", "ceil", 
    "day_name", "month_name", "isocalendar", "as_unit",
]
ok_for_td = TimedeltaArray._datetimelike_ops  # 检查时间增量数组是否支持日期时间操作
ok_for_td_methods = [  # 支持时间增量方法列表
    "components", "to_pytimedelta", "total_seconds", "round", 
    "floor", "ceil", "as_unit",
]


def get_dir(ser):
    # 获取序列的 .dt 命名空间访问器中不以下划线开头的属性列表
    results = [r for r in ser.dt.__dir__() if not r.startswith("_")]
    return sorted(set(results))


class TestSeriesDatetimeValues:
    def _compare(self, ser, name):
        # 测试 .dt 命名空间访问器
        # GH 7207, 11128

        def get_expected(ser, prop):
            # 获取预期的属性值
            result = getattr(Index(ser._values), prop)
            if isinstance(result, np.ndarray):
                if is_integer_dtype(result):
                    result = result.astype("int64")
            elif not is_list_like(result) or isinstance(result, DataFrame):
                return result
            return Series(result, index=ser.index, name=ser.name)

        left = getattr(ser.dt, name)  # 获取左侧的属性值
        right = get_expected(ser, name)  # 获取右侧的预期属性值
        if not (is_list_like(left) and is_list_like(right)):
            assert left == right  # 如果左右值不是列表类型，则直接比较
        elif isinstance(left, DataFrame):
            tm.assert_frame_equal(left, right)  # 如果左侧为数据框，则使用 pandas 测试模块比较
        else:
            tm.assert_series_equal(left, right)  # 否则，使用 pandas 测试模块比较序列


    @pytest.mark.parametrize("freq", ["D", "s", "ms"])
    def test_dt_namespace_accessor_datetime64(self, freq):
        # GH#7207, GH#11128
        # test .dt namespace accessor

        # 创建一个日期范围，从"20130101"开始，包含5个时间点，以给定频率freq
        dti = date_range("20130101", periods=5, freq=freq)
        
        # 创建一个Series对象，将日期范围dti作为数据，名称为"xxx"
        ser = Series(dti, name="xxx")

        # 遍历允许用于.dt访问器的属性列表ok_for_dt
        for prop in ok_for_dt:
            # 对于非"freq"属性，进行比较测试
            if prop != "freq":
                self._compare(ser, prop)

        # 遍历允许用于.dt访问器的方法列表ok_for_dt_methods，并调用每个方法
        for prop in ok_for_dt_methods:
            getattr(ser.dt, prop)

        # 将Series对象中的日期时间转换为Python内置的datetime对象的Series
        result = ser.dt.to_pydatetime()
        
        # 断言结果是一个Series对象
        assert isinstance(result, Series)
        
        # 断言结果的数据类型为object
        assert result.dtype == object

        # 将Series对象中的日期时间本地化为"US/Eastern"时区
        result = ser.dt.tz_localize("US/Eastern")
        
        # 期望值：将Series对象的值转换为DatetimeIndex，并在"US/Eastern"时区本地化
        exp_values = DatetimeIndex(ser.values).tz_localize("US/Eastern")
        expected = Series(exp_values, index=ser.index, name="xxx")
        
        # 断言结果与期望值的Series对象相等
        tm.assert_series_equal(result, expected)

        # 获取本地化后的结果的时区属性
        tz_result = result.dt.tz
        
        # 断言时区的字符串表示为"US/Eastern"
        assert str(tz_result) == "US/Eastern"
        
        # 获取Series对象中的频率属性
        freq_result = ser.dt.freq
        
        # 断言频率与推断出的频率一致
        assert freq_result == DatetimeIndex(ser.values, freq="infer").freq

        # 将日期时间先本地化为"UTC"时区，再转换为"US/Eastern"时区
        result = ser.dt.tz_localize("UTC").dt.tz_convert("US/Eastern")
        
        # 期望值：先在"UTC"时区本地化，再转换为"US/Eastern"时区
        exp_values = (
            DatetimeIndex(ser.values).tz_localize("UTC").tz_convert("US/Eastern")
        )
        expected = Series(exp_values, index=ser.index, name="xxx")
        
        # 断言结果与期望值的Series对象相等
        tm.assert_series_equal(result, expected)

    def test_dt_namespace_accessor_datetime64tz(self):
        # GH#7207, GH#11128
        # test .dt namespace accessor

        # 创建一个带时区的日期范围，从"20130101"开始，包含5个时间点，时区为"US/Eastern"
        dti = date_range("20130101", periods=5, tz="US/Eastern")
        
        # 创建一个Series对象，将带时区的日期范围dti作为数据，名称为"xxx"
        ser = Series(dti, name="xxx")
        
        # 遍历允许用于.dt访问器的属性列表ok_for_dt
        for prop in ok_for_dt:
            # 对于非"freq"属性，进行比较测试
            if prop != "freq":
                self._compare(ser, prop)

        # 遍历允许用于.dt访问器的方法列表ok_for_dt_methods，并调用每个方法
        for prop in ok_for_dt_methods:
            getattr(ser.dt, prop)

        # 将Series对象中的日期时间转换为Python内置的datetime对象的Series
        result = ser.dt.to_pydatetime()
        
        # 断言结果是一个Series对象
        assert isinstance(result, Series)
        
        # 断言结果的数据类型为object
        assert result.dtype == object

        # 将Series对象中的日期时间从"US/Eastern"时区转换为"CET"时区
        result = ser.dt.tz_convert("CET")
        
        # 期望值：将Series对象的值在"CET"时区中进行转换
        expected = Series(ser._values.tz_convert("CET"), index=ser.index, name="xxx")
        
        # 断言结果与期望值的Series对象相等
        tm.assert_series_equal(result, expected)

        # 获取本地化后的结果的时区属性
        tz_result = result.dt.tz
        
        # 断言时区的字符串表示为"CET"
        assert str(tz_result) == "CET"
        
        # 获取Series对象中的频率属性
        freq_result = ser.dt.freq
        
        # 断言频率与推断出的频率一致
        assert freq_result == DatetimeIndex(ser.values, freq="infer").freq
    def test_dt_namespace_accessor_timedelta(self):
        # GH#7207, GH#11128
        # test .dt namespace accessor

        # timedelta index
        cases = [
            Series(
                timedelta_range("1 day", periods=5), index=list("abcde"), name="xxx"
            ),  # 创建一个Series对象，包含timedelta_range生成的时间差序列，以及指定的索引和名称
            Series(timedelta_range("1 day 01:23:45", periods=5, freq="s"), name="xxx"),  # 创建一个Series对象，包含秒级频率的时间差序列，指定名称
            Series(
                timedelta_range("2 days 01:23:45.012345", periods=5, freq="ms"),
                name="xxx",
            ),  # 创建一个Series对象，包含毫秒级频率的时间差序列，指定名称
        ]
        for ser in cases:
            for prop in ok_for_td:
                # we test freq below
                if prop != "freq":
                    self._compare(ser, prop)  # 调用自定义方法_compare，比较Series对象和属性prop

            for prop in ok_for_td_methods:
                getattr(ser.dt, prop)  # 获取Series对象的dt属性下的方法prop

            result = ser.dt.components  # 获取Series对象的dt属性下的components属性，返回一个DataFrame对象
            assert isinstance(result, DataFrame)  # 断言result是DataFrame类型
            tm.assert_index_equal(result.index, ser.index)  # 使用tm.assert_index_equal断言result的索引与ser的索引相等

            msg = "The behavior of TimedeltaProperties.to_pytimedelta is deprecated"
            with tm.assert_produces_warning(FutureWarning, match=msg):
                result = ser.dt.to_pytimedelta()  # 获取Series对象的dt属性下的to_pytimedelta方法的结果，返回一个np.ndarray对象
            assert isinstance(result, np.ndarray)  # 断言result是np.ndarray类型
            assert result.dtype == object  # 断言result的数据类型是object

            result = ser.dt.total_seconds()  # 获取Series对象的dt属性下的total_seconds方法的结果，返回一个Series对象
            assert isinstance(result, Series)  # 断言result是Series类型
            assert result.dtype == "float64"  # 断言result的数据类型是float64

            freq_result = ser.dt.freq  # 获取Series对象的dt属性下的freq属性的值
            assert freq_result == TimedeltaIndex(ser.values, freq="infer").freq  # 断言freq_result等于根据ser的值和推断频率创建的TimedeltaIndex对象的频率

    def test_dt_namespace_accessor_period(self):
        # GH#7207, GH#11128
        # test .dt namespace accessor

        # periodindex
        pi = period_range("20130101", periods=5, freq="D")  # 创建一个PeriodIndex对象，包含指定日期范围和频率
        ser = Series(pi, name="xxx")  # 创建一个Series对象，使用pi作为数据，指定名称为'xxx'

        for prop in ok_for_period:
            # we test freq below
            if prop != "freq":
                self._compare(ser, prop)  # 调用自定义方法_compare，比较Series对象和属性prop

        for prop in ok_for_period_methods:
            getattr(ser.dt, prop)  # 获取Series对象的dt属性下的方法prop

        freq_result = ser.dt.freq  # 获取Series对象的dt属性下的freq属性的值
        assert freq_result == PeriodIndex(ser.values).freq  # 断言freq_result等于根据ser的值创建的PeriodIndex对象的频率

    def test_dt_namespace_accessor_index_and_values(self):
        # both
        index = date_range("20130101", periods=3, freq="D")  # 创建一个日期范围的DatetimeIndex对象，包含指定日期范围和频率
        dti = date_range("20140204", periods=3, freq="s")  # 创建一个日期范围的DatetimeIndex对象，包含指定日期范围和秒级频率
        ser = Series(dti, index=index, name="xxx")  # 创建一个Series对象，使用dti作为数据，index作为索引，指定名称为'xxx'
        exp = Series(
            np.array([2014, 2014, 2014], dtype="int32"), index=index, name="xxx"
        )  # 创建一个预期的Series对象，包含指定的数据数组和索引，指定名称为'xxx'
        tm.assert_series_equal(ser.dt.year, exp)  # 使用tm.assert_series_equal断言ser的dt属性下的year属性与exp相等

        exp = Series(np.array([2, 2, 2], dtype="int32"), index=index, name="xxx")  # 创建一个预期的Series对象，包含指定的数据数组和索引，指定名称为'xxx'
        tm.assert_series_equal(ser.dt.month, exp)  # 使用tm.assert_series_equal断言ser的dt属性下的month属性与exp相等

        exp = Series(np.array([0, 1, 2], dtype="int32"), index=index, name="xxx")  # 创建一个预期的Series对象，包含指定的数据数组和索引，指定名称为'xxx'
        tm.assert_series_equal(ser.dt.second, exp)  # 使用tm.assert_series_equal断言ser的dt属性下的second属性与exp相等

        exp = Series([ser.iloc[0]] * 3, index=index, name="xxx")  # 创建一个预期的Series对象，包含指定的数据数组和索引，指定名称为'xxx'
        tm.assert_series_equal(ser.dt.normalize(), exp)  # 使用tm.assert_series_equal断言ser的dt属性下的normalize方法的结果与exp相等
    def test_dt_accessor_limited_display_api(self):
        # tznaive
        # 创建一个包含5个日期的Series，频率为每天一次，未指定时区
        ser = Series(date_range("20130101", periods=5, freq="D"), name="xxx")
        # 调用get_dir函数，返回结果应接近于ok_for_dt和ok_for_dt_methods的并集的排序集合
        results = get_dir(ser)
        tm.assert_almost_equal(results, sorted(set(ok_for_dt + ok_for_dt_methods)))

        # tzaware
        # 创建一个包含分钟频率日期范围的Series，从"2015-01-01"到"2016-01-01"，指定时区为UTC，再转换为"America/Chicago"时区
        ser = Series(date_range("2015-01-01", "2016-01-01", freq="min"), name="xxx")
        ser = ser.dt.tz_localize("UTC").dt.tz_convert("America/Chicago")
        # 调用get_dir函数，返回结果应接近于ok_for_dt和ok_for_dt_methods的并集的排序集合
        results = get_dir(ser)
        tm.assert_almost_equal(results, sorted(set(ok_for_dt + ok_for_dt_methods)))

        # Period
        # 创建一个包含5个Period对象的Series，频率为每天一次，名称为"xxx"
        idx = period_range("20130101", periods=5, freq="D", name="xxx")
        ser = Series(idx)
        # 调用get_dir函数，返回结果应接近于ok_for_period和ok_for_period_methods的并集的排序集合
        results = get_dir(ser)
        tm.assert_almost_equal(
            results, sorted(set(ok_for_period + ok_for_period_methods))
        )

    def test_dt_accessor_ambiguous_freq_conversions(self):
        # GH#11295
        # 在转换时产生歧义时间错误
        ser = Series(date_range("2015-01-01", "2016-01-01", freq="min"), name="xxx")
        ser = ser.dt.tz_localize("UTC").dt.tz_convert("America/Chicago")

        # 期望的数值范围，包含时区信息"UTC"，并转换为"America/Chicago"时区
        exp_values = date_range(
            "2015-01-01", "2016-01-01", freq="min", tz="UTC"
        ).tz_convert("America/Chicago")
        # tz_localize上述未保留频率信息
        exp_values = exp_values._with_freq(None)
        # 创建一个包含期望数值的Series，名称为"xxx"
        expected = Series(exp_values, name="xxx")
        tm.assert_series_equal(ser, expected)

    def test_dt_accessor_not_writeable(self):
        # no setting allowed
        # 创建一个包含5个日期的Series，频率为每天一次，名称为"xxx"
        ser = Series(date_range("20130101", periods=5, freq="D"), name="xxx")
        # 试图修改dt.hour属性时，引发值错误异常，异常信息包含"modifications"
        with pytest.raises(ValueError, match="modifications"):
            ser.dt.hour = 5

        # trying to set a copy
        # 尝试设置一个副本时，引发连锁赋值错误异常
        with tm.raises_chained_assignment_error():
            ser.dt.hour[0] = 5

    @pytest.mark.parametrize(
        "method, dates",
        [
            ["round", ["2012-01-02", "2012-01-02", "2012-01-01"]],
            ["floor", ["2012-01-01", "2012-01-01", "2012-01-01"]],
            ["ceil", ["2012-01-02", "2012-01-02", "2012-01-02"]],
        ],
    )
    def test_dt_round(self, method, dates):
        # round
        # 创建一个包含三个日期时间对象的Series，名称为"xxx"
        ser = Series(
            pd.to_datetime(
                ["2012-01-01 13:00:00", "2012-01-01 12:01:00", "2012-01-01 08:00:00"]
            ),
            name="xxx",
        )
        # 调用ser.dt对象的method方法，传入"D"作为参数，返回结果
        result = getattr(ser.dt, method)("D")
        # 创建一个包含期望日期时间对象的Series，名称为"xxx"
        expected = Series(pd.to_datetime(dates), name="xxx")
        tm.assert_series_equal(result, expected)
    def test_dt_round_tz(self):
        # 创建一个包含日期时间的序列
        ser = Series(
            pd.to_datetime(
                ["2012-01-01 13:00:00", "2012-01-01 12:01:00", "2012-01-01 08:00:00"]
            ),
            name="xxx",
        )
        # 将序列中的日期时间转换为UTC时区，然后转换为US/Eastern时区，并进行日期舍入处理
        result = ser.dt.tz_localize("UTC").dt.tz_convert("US/Eastern").dt.round("D")

        # 生成预期结果的日期时间序列，转换为US/Eastern时区
        exp_values = pd.to_datetime(
            ["2012-01-01", "2012-01-01", "2012-01-01"]
        ).tz_localize("US/Eastern")
        expected = Series(exp_values, name="xxx")
        # 断言结果序列与预期序列相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("method", ["ceil", "round", "floor"])
    def test_dt_round_tz_ambiguous(self, method):
        # 测试处理“fall back”情况下的日期时间舍入
        df1 = DataFrame(
            [
                pd.to_datetime("2017-10-29 02:00:00+02:00", utc=True),
                pd.to_datetime("2017-10-29 02:00:00+01:00", utc=True),
                pd.to_datetime("2017-10-29 03:00:00+01:00", utc=True),
            ],
            columns=["date"],
        )
        # 将日期时间列转换为Europe/Madrid时区
        df1["date"] = df1["date"].dt.tz_convert("Europe/Madrid")
        
        # 使用getattr方法调用指定的舍入方法（ceil/round/floor），处理模糊时间（ambiguous="infer"）
        result = getattr(df1.date.dt, method)("h", ambiguous="infer")
        expected = df1["date"]
        tm.assert_series_equal(result, expected)

        # 使用bool数组指定ambiguous参数
        result = getattr(df1.date.dt, method)("h", ambiguous=[True, False, False])
        tm.assert_series_equal(result, expected)

        # 处理NaT情况的日期时间舍入
        result = getattr(df1.date.dt, method)("h", ambiguous="NaT")
        expected = df1["date"].copy()
        expected.iloc[0:2] = pd.NaT
        tm.assert_series_equal(result, expected)

        # 检查在raise模式下处理ambiguous时间时是否引发异常
        with tm.external_error_raised(pytz.AmbiguousTimeError):
            getattr(df1.date.dt, method)("h", ambiguous="raise")

    @pytest.mark.parametrize(
        "method, ts_str, freq",
        [
            ["ceil", "2018-03-11 01:59:00-0600", "5min"],
            ["round", "2018-03-11 01:59:00-0600", "5min"],
            ["floor", "2018-03-11 03:01:00-0500", "2h"],
        ],
    )
    def test_dt_round_tz_nonexistent(self, method, ts_str, freq):
        # 测试处理“spring forward”情况下的日期时间舍入
        ser = Series([pd.Timestamp(ts_str, tz="America/Chicago")])
        # 使用getattr方法调用指定的舍入方法（ceil/round/floor），处理不存在时间（nonexistent="shift_forward"）
        result = getattr(ser.dt, method)(freq, nonexistent="shift_forward")
        expected = Series([pd.Timestamp("2018-03-11 03:00:00", tz="America/Chicago")])
        tm.assert_series_equal(result, expected)

        # 处理NaT情况的日期时间舍入
        result = getattr(ser.dt, method)(freq, nonexistent="NaT")
        expected = Series([pd.NaT]).dt.tz_localize(result.dt.tz)
        tm.assert_series_equal(result, expected)

        # 检查在raise模式下处理nonexistent时间时是否引发异常
        with pytest.raises(pytz.NonExistentTimeError, match="2018-03-11 02:00:00"):
            getattr(ser.dt, method)(freq, nonexistent="raise")

    @pytest.mark.parametrize("freq", ["ns", "us", "1000us"])
    # 定义测试方法，用于测试 datetime 数据的 round 方法在更高分辨率无操作的情况下
    def test_dt_round_nonnano_higher_resolution_no_op(self, freq):
        # GH 52761: GitHub issue reference
        # 创建包含三个 datetime 字符串的 Series 对象，数据类型为 datetime64 毫秒级
        ser = Series(
            ["2020-05-31 08:00:00", "2000-12-31 04:00:05", "1800-03-14 07:30:20"],
            dtype="datetime64[ms]",
        )
        # 复制原始 Series 对象作为预期结果
        expected = ser.copy()
        # 对 Series 对象中的 datetime 数据执行 round 方法
        result = ser.dt.round(freq)
        # 使用测试工具检查结果与预期是否相等
        tm.assert_series_equal(result, expected)

        # 使用 numpy 的 shares_memory 方法检查两个数组是否共享内存
        assert not np.shares_memory(ser.array._ndarray, result.array._ndarray)

    # 定义测试方法，测试 dt 命名空间访问器在分类数据上的行为
    def test_dt_namespace_accessor_categorical(self):
        # GH 19468: GitHub issue reference
        # 创建 DateTimeIndex，重复其中的日期数据两次
        dti = DatetimeIndex(["20171111", "20181212"]).repeat(2)
        # 使用 pandas 的 Categorical 将 DateTimeIndex 转换为分类类型的 Series
        ser = Series(pd.Categorical(dti), name="foo")
        # 使用 dt.year 访问器获取年份
        result = ser.dt.year
        # 创建预期结果，包含期望的年份数据
        expected = Series([2017, 2017, 2018, 2018], dtype="int32", name="foo")
        # 使用测试工具检查结果与预期是否相等
        tm.assert_series_equal(result, expected)

    # 定义测试方法，测试 dt.tz_localize 方法在分类数据上的行为
    def test_dt_tz_localize_categorical(self, tz_aware_fixture):
        # GH 27952: GitHub issue reference
        # 获取时区信息
        tz = tz_aware_fixture
        # 创建包含三个日期字符串的 Series，数据类型为 datetime64 纳秒级
        datetimes = Series(
            ["2019-01-01", "2019-01-01", "2019-01-02"], dtype="datetime64[ns]"
        )
        # 将日期数据转换为分类数据类型
        categorical = datetimes.astype("category")
        # 使用 dt.tz_localize 方法将分类数据本地化到指定时区
        result = categorical.dt.tz_localize(tz)
        # 使用 datetime 数据的 dt.tz_localize 方法进行对比，创建预期结果
        expected = datetimes.dt.tz_localize(tz)
        # 使用测试工具检查结果与预期是否相等
        tm.assert_series_equal(result, expected)

    # 定义测试方法，测试 dt.tz_convert 方法在分类数据上的行为
    def test_dt_tz_convert_categorical(self, tz_aware_fixture):
        # GH 27952: GitHub issue reference
        # 获取时区信息
        tz = tz_aware_fixture
        # 创建包含三个日期字符串的 Series，数据类型为 datetime64 带 MET 时区信息
        datetimes = Series(
            ["2019-01-01", "2019-01-01", "2019-01-02"], dtype="datetime64[ns, MET]"
        )
        # 将日期数据转换为分类数据类型
        categorical = datetimes.astype("category")
        # 使用 dt.tz_convert 方法将分类数据转换到指定时区
        result = categorical.dt.tz_convert(tz)
        # 使用 datetime 数据的 dt.tz_convert 方法进行对比，创建预期结果
        expected = datetimes.dt.tz_convert(tz)
        # 使用测试工具检查结果与预期是否相等
        tm.assert_series_equal(result, expected)

    # 定义测试方法，测试 dt 的其他访问器在分类数据上的行为
    @pytest.mark.parametrize("accessor", ["year", "month", "day"])
    def test_dt_other_accessors_categorical(self, accessor):
        # GH 27952: GitHub issue reference
        # 创建包含三个日期字符串的 Series，数据类型为 datetime64 纳秒级
        datetimes = Series(
            ["2018-01-01", "2018-01-01", "2019-01-02"], dtype="datetime64[ns]"
        )
        # 将日期数据转换为分类数据类型
        categorical = datetimes.astype("category")
        # 使用 getattr 获取指定访问器的结果
        result = getattr(categorical.dt, accessor)
        # 使用 getattr 获取 datetime 数据对应访问器的预期结果
        expected = getattr(datetimes.dt, accessor)
        # 使用测试工具检查结果与预期是否相等
        tm.assert_series_equal(result, expected)

    # 定义测试方法，验证 dt 访问器无法添加新属性的行为
    def test_dt_accessor_no_new_attributes(self):
        # https://github.com/pandas-dev/pandas/issues/10673: GitHub issue reference
        # 创建包含日期范围的 Series
        ser = Series(date_range("20130101", periods=5, freq="D"))
        # 使用 pytest 的 raises 方法检查是否抛出预期的 AttributeError 异常
        with pytest.raises(AttributeError, match="You cannot add any new attribute"):
            ser.dt.xlabel = "a"

    # 标记注释为错误信息，说明发生了错误 "Unsupported operand types for + ("List[None]" and "List[str])"
    @pytest.mark.parametrize(
        "time_locale",
        [None] + tm.get_locales(),  # type: ignore[operator]
    )
    # 定义一个测试方法，用于测试日期时间访问器中的名称访问功能，传入时间区域参数 time_locale
    def test_dt_accessor_datetime_name_accessors(self, time_locale):
        # 如果 time_locale 为 None
        if time_locale is None:
            # 若时间区域为 None，则期望返回英文的星期几和月份名称
            expected_days = [
                "Monday",        # 期望星期一
                "Tuesday",       # 期望星期二
                "Wednesday",     # 期望星期三
                "Thursday",      # 期望星期四
                "Friday",        # 期望星期五
                "Saturday",      # 期望星期六
                "Sunday",        # 期望星期日
            ]
            expected_months = [
                "January",       # 期望一月
                "February",      # 期望二月
                "March",         # 期望三月
                "April",         # 期望四月
                "May",           # 期望五月
                "June",          # 期望六月
                "July",          # 期望七月
                "August",        # 期望八月
                "September",     # 期望九月
                "October",       # 期望十月
                "November",      # 期望十一月
                "December",      # 期望十二月
            ]
        else:
            # 否则，设置时间区域并获取当前系统的星期几和月份名称
            with tm.set_locale(time_locale, locale.LC_TIME):
                expected_days = calendar.day_name[:]       # 获取当前语言环境下的星期几列表
                expected_months = calendar.month_name[1:]  # 获取当前语言环境下的月份名称列表（从1开始）

        # 创建一个包含一年日期范围的 Series 对象
        ser = Series(date_range(freq="D", start=datetime(1998, 1, 1), periods=365))
        
        # 英文星期几列表（固定值）
        english_days = [
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
        ]
        
        # 遍历指定范围内的日期，对比本地化后的星期几名称和英文名称
        for day, name, eng_name in zip(range(4, 11), expected_days, english_days):
            name = name.capitalize()   # 首字母大写处理
            # 断言本地化后的星期几名称与期望名称相同
            assert ser.dt.day_name(locale=time_locale)[day] == name
            # 断言英文的星期几名称与固定值相同
            assert ser.dt.day_name(locale=None)[day] == eng_name
        
        # 将空值（NaT）添加到 Series 中
        ser = pd.concat([ser, Series([pd.NaT])])
        # 断言最后一个日期的星期几本地化结果为 NaN
        assert np.isnan(ser.dt.day_name(locale=time_locale).iloc[-1])

        # 创建一个包含每月最后一天日期范围的 Series 对象
        ser = Series(date_range(freq="M", start="2012", end="2013"))
        # 获取本地化后的月份名称结果
        result = ser.dt.month_name(locale=time_locale)
        # 期望的本地化后的月份名称列表
        expected = Series([month.capitalize() for month in expected_months])

        # 处理因已知问题而进行的工作区调整
        result = result.str.normalize("NFD")
        expected = expected.str.normalize("NFD")

        # 断言 Series 对象中本地化后的月份名称与期望值相同
        tm.assert_series_equal(result, expected)

        # 遍历日期序列，比较本地化后的月份名称与期望值
        for s_date, expected in zip(ser, expected_months):
            result = s_date.month_name(locale=time_locale)   # 获取日期的本地化月份名称
            expected = expected.capitalize()                 # 首字母大写处理

            # 标准化处理结果和期望值以解决 Unicode 兼容性问题
            result = unicodedata.normalize("NFD", result)
            expected = unicodedata.normalize("NFD", expected)

            # 断言处理结果与期望值相同
            assert result == expected
        
        # 将空值（NaT）添加到 Series 中
        ser = pd.concat([ser, Series([pd.NaT])])
        # 断言最后一个日期的本地化后的月份名称为 NaN
        assert np.isnan(ser.dt.month_name(locale=time_locale).iloc[-1])
    # 测试 strftime 方法的功能
    def test_strftime(self):
        # 创建一个日期序列，从"20130101"开始，包括5个周期
        ser = Series(date_range("20130101", periods=5))
        # 对日期序列进行格式化为"%Y/%m/%d"格式的操作
        result = ser.dt.strftime("%Y/%m/%d")
        # 创建一个预期的日期序列，每个日期按"%Y/%m/%d"格式
        expected = Series(
            ["2013/01/01", "2013/01/02", "2013/01/03", "2013/01/04", "2013/01/05"]
        )
        # 断言两个序列是否相等
        tm.assert_series_equal(result, expected)

        # 创建一个日期时间序列，从"2015-02-03 11:22:33.4567"开始，包括5个周期
        ser = Series(date_range("2015-02-03 11:22:33.4567", periods=5))
        # 对日期时间序列进行格式化为"%Y/%m/%d %H-%M-%S"格式的操作
        result = ser.dt.strftime("%Y/%m/%d %H-%M-%S")
        # 创建一个预期的日期时间序列，每个日期时间按"%Y/%m/%d %H-%M-%S"格式
        expected = Series(
            [
                "2015/02/03 11-22-33",
                "2015/02/04 11-22-33",
                "2015/02/05 11-22-33",
                "2015/02/06 11-22-33",
                "2015/02/07 11-22-33",
            ]
        )
        # 断言两个序列是否相等
        tm.assert_series_equal(result, expected)

        # 创建一个周期范围的序列，从"20130101"开始，包括5个周期
        ser = Series(period_range("20130101", periods=5))
        # 对周期范围序列进行格式化为"%Y/%m/%d"格式的操作
        result = ser.dt.strftime("%Y/%m/%d")
        # 创建一个预期的日期序列，每个日期按"%Y/%m/%d"格式
        expected = Series(
            ["2013/01/01", "2013/01/02", "2013/01/03", "2013/01/04", "2013/01/05"]
        )
        # 断言两个序列是否相等
        tm.assert_series_equal(result, expected)

        # 创建一个秒频率的周期范围序列，从"2015-02-03 11:22:33.4567"开始，包括5个周期
        ser = Series(period_range("2015-02-03 11:22:33.4567", periods=5, freq="s"))
        # 对秒频率的周期范围序列进行格式化为"%Y/%m/%d %H-%M-%S"格式的操作
        result = ser.dt.strftime("%Y/%m/%d %H-%M-%S")
        # 创建一个预期的日期时间序列，每个日期时间按"%Y/%m/%d %H-%M-%S"格式
        expected = Series(
            [
                "2015/02/03 11-22-33",
                "2015/02/03 11-22-34",
                "2015/02/03 11-22-35",
                "2015/02/03 11-22-36",
                "2015/02/03 11-22-37",
            ]
        )
        # 断言两个序列是否相等
        tm.assert_series_equal(result, expected)

    # 测试 strftime 方法处理日期时间序列中的 NaT 值和日期范围
    def test_strftime_dt64_days(self):
        # 创建一个日期序列，从"20130101"开始，包括5个周期
        ser = Series(date_range("20130101", periods=5))
        # 将第一个元素设为 NaT（Not a Time），并对日期序列进行格式化为"%Y/%m/%d"格式的操作
        ser.iloc[0] = pd.NaT
        result = ser.dt.strftime("%Y/%m/%d")
        # 创建一个预期的日期序列，第一个元素为 np.nan，其余按"%Y/%m/%d"格式
        expected = Series(
            [np.nan, "2013/01/02", "2013/01/03", "2013/01/04", "2013/01/05"]
        )
        # 断言两个序列是否相等
        tm.assert_series_equal(result, expected)

        # 创建一个日期范围索引，从"20150301"开始，包括5个周期，并对其进行"%Y/%m/%d"格式化操作
        datetime_index = date_range("20150301", periods=5)
        result = datetime_index.strftime("%Y/%m/%d")
        # 创建一个预期的日期索引，每个日期按"%Y/%m/%d"格式
        expected = Index(
            ["2015/03/01", "2015/03/02", "2015/03/03", "2015/03/04", "2015/03/05"],
            dtype=np.object_,
        )
        # 根据 Python 版本可能为 S10 或 U10 类型
        # 断言两个索引是否相等
        tm.assert_index_equal(result, expected)

    # 测试 strftime 方法处理周期范围序列的功能
    def test_strftime_period_days(self, using_infer_string):
        # 创建一个周期范围的索引，从"20150301"开始，包括5个周期，并对其进行"%Y/%m/%d"格式化操作
        period_index = period_range("20150301", periods=5)
        result = period_index.strftime("%Y/%m/%d")
        # 创建一个预期的日期索引，每个日期按"%Y/%m/%d"格式
        expected = Index(
            ["2015/03/01", "2015/03/02", "2015/03/03", "2015/03/04", "2015/03/05"],
            dtype="=U10",
        )
        # 如果 using_infer_string 为真，则将预期结果转换为指定格式
        if using_infer_string:
            expected = expected.astype("string[pyarrow_numpy]")
        # 断言两个索引是否相等
        tm.assert_index_equal(result, expected)

    # 测试 strftime 方法处理微秒分辨率的日期时间序列
    def test_strftime_dt64_microsecond_resolution(self):
        # 创建一个包含日期时间的序列，精确到微秒级别
        ser = Series([datetime(2013, 1, 1, 2, 32, 59), datetime(2013, 1, 2, 14, 32, 1)])
        # 对日期时间序列进行格式化为"%Y-%m-%d %H:%M:%S"格式的操作
        result = ser.dt.strftime("%Y-%m-%d %H:%M:%S")
        # 创建一个预期的日期时间序列，每个日期时间按"%Y-%m-%d %H:%M:%S"格式
        expected = Series(["2013-01-01 02:32:59", "2013-01-02 14:32:01"])
        # 断言两个序列是否相等
        tm.assert_series_equal(result, expected)
    def test_strftime_period_hours(self):
        # 创建一个包含4个小时频率时间段的序列
        ser = Series(period_range("20130101", periods=4, freq="h"))
        # 对时间序列进行格式化为指定格式的字符串操作
        result = ser.dt.strftime("%Y/%m/%d %H:%M:%S")
        # 期望的结果序列，包含格式化后的时间字符串
        expected = Series(
            [
                "2013/01/01 00:00:00",
                "2013/01/01 01:00:00",
                "2013/01/01 02:00:00",
                "2013/01/01 03:00:00",
            ]
        )
        # 使用测试工具比较结果序列和期望序列，确保它们相等
        tm.assert_series_equal(result, expected)

    def test_strftime_period_minutes(self):
        # 创建一个包含4个毫秒频率时间段的序列
        ser = Series(period_range("20130101", periods=4, freq="ms"))
        # 对时间序列进行格式化为指定格式的字符串操作，包含毫秒
        result = ser.dt.strftime("%Y/%m/%d %H:%M:%S.%l")
        # 期望的结果序列，包含格式化后带毫秒的时间字符串
        expected = Series(
            [
                "2013/01/01 00:00:00.000",
                "2013/01/01 00:00:00.001",
                "2013/01/01 00:00:00.002",
                "2013/01/01 00:00:00.003",
            ]
        )
        # 使用测试工具比较结果序列和期望序列，确保它们相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "data",
        [
            DatetimeIndex(["2019-01-01", pd.NaT]),  # 创建包含日期和NaT（Not a Time）的日期时间索引
            PeriodIndex(["2019-01-01", pd.NaT], dtype="period[D]"),  # 创建包含日期和NaT的周期索引
        ],
    )
    def test_strftime_nat(self, data):
        # GH 29578
        # 使用给定的数据创建序列
        ser = Series(data)
        # 对时间序列进行格式化为指定格式的字符串操作
        result = ser.dt.strftime("%Y-%m-%d")
        # 期望的结果序列，包含格式化后的日期字符串和NaN值
        expected = Series(["2019-01-01", np.nan])
        # 使用测试工具比较结果序列和期望序列，确保它们相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "data", [DatetimeIndex([pd.NaT]), PeriodIndex([pd.NaT], dtype="period[D]")]
    )
    def test_strftime_all_nat(self, data):
        # https://github.com/pandas-dev/pandas/issues/45858
        # 使用给定的数据创建序列
        ser = Series(data)
        # 检查警告是否被触发
        with tm.assert_produces_warning(None):
            # 对时间序列进行格式化为指定格式的字符串操作
            result = ser.dt.strftime("%Y-%m-%d")
        # 期望的结果序列，包含NaN值
        expected = Series([np.nan], dtype=object)
        # 使用测试工具比较结果序列和期望序列，确保它们相等
        tm.assert_series_equal(result, expected)

    def test_valid_dt_with_missing_values(self):
        # GH 8689
        # 创建一个包含5个日期频率时间段的序列
        ser = Series(date_range("20130101", periods=5, freq="D"))
        # 将第3个位置的值设置为NaT（缺失值）
        ser.iloc[2] = pd.NaT

        # 遍历多个时间序列属性，检查对应位置的值
        for attr in ["microsecond", "nanosecond", "second", "minute", "hour", "day"]:
            # 获取时间序列中指定属性的期望值
            expected = getattr(ser.dt, attr).copy()
            expected.iloc[2] = np.nan
            # 获取时间序列中指定属性的实际值
            result = getattr(ser.dt, attr)
            # 使用测试工具比较结果序列和期望序列，确保它们相等
            tm.assert_series_equal(result, expected)

        # 获取时间序列的日期部分
        result = ser.dt.date
        # 期望的结果序列，包含日期对象和NaN值
        expected = Series(
            [
                date(2013, 1, 1),
                date(2013, 1, 2),
                pd.NaT,
                date(2013, 1, 4),
                date(2013, 1, 5),
            ],
            dtype="object",
        )
        # 使用测试工具比较结果序列和期望序列，确保它们相等
        tm.assert_series_equal(result, expected)

        # 获取时间序列的时间部分
        result = ser.dt.time
        # 期望的结果序列，包含时间对象和NaN值
        expected = Series([time(0), time(0), pd.NaT, time(0), time(0)], dtype="object")
        # 使用测试工具比较结果序列和期望序列，确保它们相等
        tm.assert_series_equal(result, expected)
    def test_dt_accessor_api(self):
        # 导入需要的类和函数，GH 9322
        from pandas.core.indexes.accessors import (
            CombinedDatetimelikeProperties,
            DatetimeProperties,
        )

        # 断言Series的.dt属性是CombinedDatetimelikeProperties类
        assert Series.dt is CombinedDatetimelikeProperties

        # 创建一个包含日期范围的Series，验证其.dt属性是DatetimeProperties类的实例
        ser = Series(date_range("2000-01-01", periods=3))
        assert isinstance(ser.dt, DatetimeProperties)

    @pytest.mark.parametrize(
        "data",
        [
            np.arange(5),
            list("abcde"),
            np.random.default_rng(2).standard_normal(5),
        ],
    )
    def test_dt_accessor_invalid(self, data):
        # GH#9322 检查具有不正确dtype的Series不具有.dt属性
        ser = Series(data)
        with pytest.raises(AttributeError, match="only use .dt accessor"):
            ser.dt
        assert not hasattr(ser, "dt")

    def test_dt_accessor_updates_on_inplace(self):
        # 创建一个包含日期范围的Series
        ser = Series(date_range("2018-01-01", periods=10))
        
        # 修改Series中的一个元素为None，并用指定日期填充
        ser[2] = None
        return_value = ser.fillna(pd.Timestamp("2018-01-01"), inplace=True)
        
        # 断言填充操作的返回值为None
        assert return_value is None
        
        # 获取Series中日期的.dt.date属性，断言修改后的日期与原来的相同
        result = ser.dt.date
        assert result[0] == result[2]

    def test_date_tz(self):
        # GH11757 创建一个带有时区信息的DatetimeIndex
        rng = DatetimeIndex(
            ["2014-04-04 23:56", "2014-07-18 21:24", "2015-11-22 22:14"],
            tz="US/Eastern",
        )
        ser = Series(rng)
        
        # 创建预期的日期Series，并使用tm.assert_series_equal进行断言
        expected = Series([date(2014, 4, 4), date(2014, 7, 18), date(2015, 11, 22)])
        tm.assert_series_equal(ser.dt.date, expected)
        
        # 使用lambda函数获取每个日期的日期部分，再次使用tm.assert_series_equal进行断言
        tm.assert_series_equal(ser.apply(lambda x: x.date()), expected)

    def test_dt_timetz_accessor(self, tz_naive_fixture):
        # GH21358 创建一个带有时区信息的DatetimeIndex
        tz = maybe_get_tz(tz_naive_fixture)

        dtindex = DatetimeIndex(
            ["2014-04-04 23:56", "2014-07-18 21:24", "2015-11-22 22:14"], tz=tz
        )
        ser = Series(dtindex)
        
        # 创建预期的时间Series，并使用tm.assert_series_equal进行断言
        expected = Series(
            [time(23, 56, tzinfo=tz), time(21, 24, tzinfo=tz), time(22, 14, tzinfo=tz)]
        )
        result = ser.dt.timetz
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "input_series, expected_output",
        [
            [["2020-01-01"], [[2020, 1, 3]]],
            [[pd.NaT], [[np.nan, np.nan, np.nan]]],
            [["2019-12-31", "2019-12-29"], [[2020, 1, 2], [2019, 52, 7]]],
            [["2010-01-01", pd.NaT], [[2009, 53, 5], [np.nan, np.nan, np.nan]]],
            # see GH#36032
            [["2016-01-08", "2016-01-04"], [[2016, 1, 5], [2016, 1, 1]]],
            [["2016-01-07", "2016-01-01"], [[2016, 1, 4], [2015, 53, 5]]],
        ],
    )
    def test_isocalendar(self, input_series, expected_output):
        # 将输入Series转换为datetime，并获取其.isocalendar()结果
        result = pd.to_datetime(Series(input_series)).dt.isocalendar()
        
        # 创建预期的DataFrame，并使用tm.assert_frame_equal进行断言
        expected_frame = DataFrame(
            expected_output, columns=["year", "week", "day"], dtype="UInt32"
        )
        tm.assert_frame_equal(result, expected_frame)
    `
        # 定义一个名为 test_hour_index 的测试方法，测试时间序列的小时索引功能
        def test_hour_index(self):
            # 创建一个时间序列 dt_series，包含从 "2021-01-01" 开始的5个小时频率时间点，指定索引为 [2, 6, 7, 8, 11]，数据类型为类别型
            dt_series = Series(
                date_range(start="2021-01-01", periods=5, freq="h"),
                index=[2, 6, 7, 8, 11],
                dtype="category",
            )
            # 获取时间序列 dt_series 中每个时间点的小时数，结果为 result
            result = dt_series.dt.hour
            # 创建一个预期的时间序列 expected，包含小时数的整数数组 [0, 1, 2, 3, 4]，数据类型为 int32，指定相同的索引 [2, 6, 7, 8, 11]
            expected = Series(
                [0, 1, 2, 3, 4],
                dtype="int32",
                index=[2, 6, 7, 8, 11],
            )
            # 使用测试框架中的断言方法，比较 result 和 expected 是否相等
            tm.assert_series_equal(result, expected)
class TestSeriesPeriodValuesDtAccessor:
    @pytest.mark.parametrize(
        "input_vals",
        [
            [Period("2016-01", freq="M"), Period("2016-02", freq="M")],
            [Period("2016-01-01", freq="D"), Period("2016-01-02", freq="D")],
            [
                Period("2016-01-01 00:00:00", freq="h"),
                Period("2016-01-01 01:00:00", freq="h"),
            ],
            [
                Period("2016-01-01 00:00:00", freq="M"),
                Period("2016-01-01 00:01:00", freq="M"),
            ],
            [
                Period("2016-01-01 00:00:00", freq="s"),
                Period("2016-01-01 00:00:01", freq="s"),
            ],
        ],
    )
    def test_end_time_timevalues(self, input_vals):
        """
        GH#17157
        检查使用 dt 访问器在 Series 上时，Period 的时间部分是否被 end_time 调整。
        """
        input_vals = PeriodArray._from_sequence(np.asarray(input_vals))

        ser = Series(input_vals)
        result = ser.dt.end_time
        expected = ser.apply(lambda x: x.end_time)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("input_vals", [("2001"), ("NaT")])
    def test_to_period(self, input_vals):
        """
        GH#21205
        将 datetime64[ns] 类型的 Series 转换为 Period[D] 类型的 Series。
        """
        expected = Series([input_vals], dtype="Period[D]")
        result = Series([input_vals], dtype="datetime64[ns]").dt.to_period("D")
        tm.assert_series_equal(result, expected)


def test_normalize_pre_epoch_dates():
    """
    GH: 36294
    将 Series 中的日期时间转换为标准日期格式，去除时间部分。
    """
    ser = pd.to_datetime(Series(["1969-01-01 09:00:00", "2016-01-01 09:00:00"]))
    result = ser.dt.normalize()
    expected = pd.to_datetime(Series(["1969-01-01", "2016-01-01"]))
    tm.assert_series_equal(result, expected)


def test_day_attribute_non_nano_beyond_int32():
    """
    GH 52386
    计算 timedelta64[s] 类型的 Series 中每个值的天数，并返回为整数型 Series。
    """
    data = np.array(
        [
            136457654736252,
            134736784364431,
            245345345545332,
            223432411,
            2343241,
            3634548734,
            23234,
        ],
        dtype="timedelta64[s]",
    )
    ser = Series(data)
    result = ser.dt.days
    expected = Series([1579371003, 1559453522, 2839645203, 2586, 27, 42066, 0])
    tm.assert_series_equal(result, expected)
```