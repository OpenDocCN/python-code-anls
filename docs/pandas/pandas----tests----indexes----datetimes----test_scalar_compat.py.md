# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\test_scalar_compat.py`

```
"""
Tests for DatetimeIndex methods behaving like their Timestamp counterparts
"""

import calendar  # 导入日历模块
from datetime import (  # 从 datetime 模块导入以下对象
    date,  # 日期对象
    datetime,  # 日期时间对象
    time,  # 时间对象
)
import locale  # 导入本地化模块
import unicodedata  # 导入Unicode数据模块

from hypothesis import given  # 从 hypothesis 模块导入 given 函数
import hypothesis.strategies as st  # 导入 hypothesis.strategies 模块并重命名为 st
import numpy as np  # 导入 NumPy 库并重命名为 np
import pytest  # 导入 pytest 测试框架

from pandas._libs.tslibs import timezones  # 从 pandas 库的 _libs.tslibs 模块导入 timezones

from pandas import (  # 导入 pandas 库中以下对象
    DatetimeIndex,  # 时间索引对象
    Index,  # 索引对象
    NaT,  # 不可用时间对象
    Timestamp,  # 时间戳对象
    date_range,  # 日期范围生成函数
    offsets,  # 时间偏移量
)
import pandas._testing as tm  # 导入 pandas 库中的 _testing 模块并重命名为 tm
from pandas.core.arrays import DatetimeArray  # 从 pandas 库的 core.arrays 模块导入 DatetimeArray 类


class TestDatetimeIndexOps:  # 定义测试类 TestDatetimeIndexOps
    def test_dti_no_millisecond_field(self):  # 定义测试方法 test_dti_no_millisecond_field
        msg = "type object 'DatetimeIndex' has no attribute 'millisecond'"  # 设置错误消息
        with pytest.raises(AttributeError, match=msg):  # 使用 pytest 的断言捕获预期的 AttributeError 异常
            DatetimeIndex.millisecond

        msg = "'DatetimeIndex' object has no attribute 'millisecond'"  # 设置另一个错误消息
        with pytest.raises(AttributeError, match=msg):  # 使用 pytest 的断言捕获预期的 AttributeError 异常
            DatetimeIndex([]).millisecond

    def test_dti_time(self):  # 定义测试方法 test_dti_time
        rng = date_range("1/1/2000", freq="12min", periods=10)  # 创建一个日期范围对象 rng
        result = Index(rng).time  # 获取日期范围对象的时间属性
        expected = [t.time() for t in rng]  # 生成预期的时间列表
        assert (result == expected).all()  # 使用断言检查结果是否符合预期

    def test_dti_date(self):  # 定义测试方法 test_dti_date
        rng = date_range("1/1/2000", freq="12h", periods=10)  # 创建一个日期范围对象 rng
        result = Index(rng).date  # 获取日期范围对象的日期属性
        expected = [t.date() for t in rng]  # 生成预期的日期列表
        assert (result == expected).all()  # 使用断言检查结果是否符合预期

    @pytest.mark.parametrize(  # 使用 pytest.mark.parametrize 装饰器定义参数化测试
        "dtype",  # 参数名
        [None, "datetime64[ns, CET]", "datetime64[ns, EST]", "datetime64[ns, UTC]"],  # 参数值列表
    )
    def test_dti_date2(self, dtype):  # 定义参数化测试方法 test_dti_date2
        # Regression test for GH#21230
        expected = np.array([date(2018, 6, 4), NaT])  # 生成预期的 NumPy 数组

        index = DatetimeIndex(["2018-06-04 10:00:00", NaT], dtype=dtype)  # 创建日期时间索引对象
        result = index.date  # 获取日期部分

        tm.assert_numpy_array_equal(result, expected)  # 使用测试工具中的方法检查结果与预期是否相等

    @pytest.mark.parametrize(  # 使用 pytest.mark.parametrize 装饰器定义参数化测试
        "dtype",  # 参数名
        [None, "datetime64[ns, CET]", "datetime64[ns, EST]", "datetime64[ns, UTC]"],  # 参数值列表
    )
    def test_dti_time2(self, dtype):  # 定义参数化测试方法 test_dti_time2
        # Regression test for GH#21267
        expected = np.array([time(10, 20, 30), NaT])  # 生成预期的 NumPy 数组

        index = DatetimeIndex(["2018-06-04 10:20:30", NaT], dtype=dtype)  # 创建日期时间索引对象
        result = index.time  # 获取时间部分

        tm.assert_numpy_array_equal(result, expected)  # 使用测试工具中的方法检查结果与预期是否相等

    def test_dti_timetz(self, tz_naive_fixture):  # 定义测试方法 test_dti_timetz
        # GH#21358
        tz = timezones.maybe_get_tz(tz_naive_fixture)  # 获取时区信息

        expected = np.array([time(10, 20, 30, tzinfo=tz), NaT])  # 生成预期的 NumPy 数组

        index = DatetimeIndex(["2018-06-04 10:20:30", NaT], tz=tz)  # 创建带时区信息的日期时间索引对象
        result = index.timetz  # 获取带时区的时间部分

        tm.assert_numpy_array_equal(result, expected)  # 使用测试工具中的方法检查结果与预期是否相等

    @pytest.mark.parametrize(  # 使用 pytest.mark.parametrize 装饰器定义参数化测试
        "field",  # 参数名
        [  # 参数值列表
            "dayofweek",
            "day_of_week",
            "dayofyear",
            "day_of_year",
            "quarter",
            "days_in_month",
            "is_month_start",
            "is_month_end",
            "is_quarter_start",
            "is_quarter_end",
            "is_year_start",
            "is_year_end",
        ],
    )
    # 测试日期时间索引的时间戳字段
    def test_dti_timestamp_fields(self, field):
        # 创建一个日期范围索引，从"2020-01-01"开始，持续10个时间段
        idx = date_range("2020-01-01", periods=10)
        # 获取索引中指定字段（如quarter和week）的最后一个值作为期望值
        expected = getattr(idx, field)[-1]

        # 创建一个Timestamp对象，以索引中的最后一个时间戳为参数，获取指定字段的值
        result = getattr(Timestamp(idx[-1]), field)
        # 断言获取的结果与期望值相等
        assert result == expected

    # 测试日期时间索引的纳秒字段
    def test_dti_nanosecond(self):
        # 创建一个包含0到9的DatetimeIndex对象
        dti = DatetimeIndex(np.arange(10))
        # 创建一个期望的索引对象，包含0到9的整数值
        expected = Index(np.arange(10, dtype=np.int32))

        # 断言DatetimeIndex对象的纳秒字段与期望的索引对象相等
        tm.assert_index_equal(dti.nanosecond, expected)

    # 使用pytest参数化装饰器，测试带有时区的小时字段
    @pytest.mark.parametrize("prefix", ["", "dateutil/"])
    def test_dti_hour_tzaware(self, prefix):
        # 创建包含字符串日期的DatetimeIndex对象，指定时区为prefix + "US/Eastern"
        strdates = ["1/1/2012", "3/1/2012", "4/1/2012"]
        rng = DatetimeIndex(strdates, tz=prefix + "US/Eastern")
        # 断言索引中所有的小时数都为0
        assert (rng.hour == 0).all()

        # 创建一个日期范围索引，从"2011-10-02 00:00"开始，频率为每小时，持续10个时间段，
        # 指定时区为prefix + "America/Atikokan"
        dr = date_range(
            "2011-10-02 00:00", freq="h", periods=10, tz=prefix + "America/Atikokan"
        )

        # 创建一个期望的索引对象，包含0到9的整数值
        expected = Index(np.arange(10, dtype=np.int32))
        # 断言日期范围索引对象的小时字段与期望的索引对象相等
        tm.assert_index_equal(dr.hour, expected)

    # GH#12806
    # 错误: 不支持的操作类型 + ("List[None]" 和 "List[str]")
    @pytest.mark.parametrize(
        "time_locale",
        [None] + tm.get_locales(),  # type: ignore[operator]
    )
    def test_day_name_month_name(self, time_locale):
        # 测试星期一到星期日和一月到十二月的名称，按照顺序进行测试
        if time_locale is None:
            # 如果 time_locale 是 None，则 day_name 和 month_name 应返回英文属性
            expected_days = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            expected_months = [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ]
        else:
            # 设置时间区域为 time_locale，并使用 LC_TIME 区域
            with tm.set_locale(time_locale, locale.LC_TIME):
                expected_days = calendar.day_name[:]
                expected_months = calendar.month_name[1:]

        # GH#11128
        # 创建一个日期范围对象 dti，从 1998 年 1 月 1 日开始，周期为 365 天
        dti = date_range(freq="D", start=datetime(1998, 1, 1), periods=365)
        english_days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        # 对于指定范围内的日期，进行测试
        for day, name, eng_name in zip(range(4, 11), expected_days, english_days):
            # 将名称首字母大写
            name = name.capitalize()
            # 断言指定日期的本地化名称与预期的名称相符
            assert dti.day_name(locale=time_locale)[day] == name
            # 断言指定日期的英文名称与预期的英文名称相符
            assert dti.day_name(locale=None)[day] == eng_name
            # 创建一个时间戳对象 ts，日期为 2016 年 4 月 day 日
            ts = Timestamp(datetime(2016, 4, day))
            # 断言时间戳对象的本地化日期名称与预期的名称相符
            assert ts.day_name(locale=time_locale) == name
        # 将 NaT（不是一个时间）添加到日期范围对象 dti 中
        dti = dti.append(DatetimeIndex([NaT]))
        # 断言最后一个日期的本地化日期名称为 NaN
        assert np.isnan(dti.day_name(locale=time_locale)[-1])
        # 创建一个时间戳对象 ts，日期为 NaT
        ts = Timestamp(NaT)
        # 断言时间戳对象的本地化日期名称为 NaN
        assert np.isnan(ts.day_name(locale=time_locale))

        # GH#12805
        # 创建一个日期范围对象 dti，频率为每月末，开始日期为 2012 年，结束日期为 2013 年
        dti = date_range(freq="ME", start="2012", end="2013")
        # 获取本地化的月份名称结果
        result = dti.month_name(locale=time_locale)
        # 创建预期的月份名称索引，首字母大写
        expected = Index([month.capitalize() for month in expected_months])

        # 解决不同规范化方案的问题 GH#22342
        # 对结果进行 Unicode 标准化，使用 NFD 形式
        result = result.str.normalize("NFD")
        expected = expected.str.normalize("NFD")

        # 断言两个索引对象相等
        tm.assert_index_equal(result, expected)

        # 对于日期范围对象中的每个项，断言本地化的月份名称与预期的名称相符
        for item, expected in zip(dti, expected_months):
            # 获取当前项的本地化月份名称
            result = item.month_name(locale=time_locale)
            # 将预期的月份名称首字母大写
            expected = expected.capitalize()

            # 对结果和预期进行 Unicode 标准化，使用 NFD 形式
            result = unicodedata.normalize("NFD", result)
            expected = unicodedata.normalize("NFD", result)

            # 断言结果与预期相等
            assert result == expected
        # 将 NaT 添加到日期范围对象 dti 中
        dti = dti.append(DatetimeIndex([NaT]))
        # 断言最后一个日期的本地化月份名称为 NaN
        assert np.isnan(dti.month_name(locale=time_locale)[-1])
    # 定义一个测试方法，用于测试 DatetimeIndex 和其 Timestamp 元素的周数访问器行为
    def test_dti_week(self):
        # GH#6538: 检查 DatetimeIndex 和其 Timestamp 元素在接近新年时带有时区的 weekofyear 访问器返回相同结果
        # 创建日期字符串列表
        dates = ["2013/12/29", "2013/12/30", "2013/12/31"]
        # 使用时区 Europe/Brussels 创建 DatetimeIndex 对象
        dates = DatetimeIndex(dates, tz="Europe/Brussels")
        # 期望的周数列表
        expected = [52, 1, 1]
        # 断言调用 isocalendar() 方法后返回的周数列表与期望列表相同
        assert dates.isocalendar().week.tolist() == expected
        # 断言通过列表推导式获取每个日期对象的 weekofyear 属性列表与期望列表相同
        assert [d.weekofyear for d in dates] == expected

    # 使用 pytest 的 parametrize 装饰器，传递不同的时区参数来测试
    @pytest.mark.parametrize("tz", [None, "US/Eastern"])
    # 定义一个测试函数，用于测试日期时间索引对象的字段
    def test_dti_fields(self, tz):
        # GH#13303: 对应 GitHub issue #13303
        
        # 创建一个包含365天日期范围的日期时间索引对象
        dti = date_range(freq="D", start=datetime(1998, 1, 1), periods=365, tz=tz)
        
        # 断言索引对象的年份属性第一个元素为1998
        assert dti.year[0] == 1998
        # 断言索引对象的月份属性第一个元素为1（即1月）
        assert dti.month[0] == 1
        # 断言索引对象的日期属性第一个元素为1号
        assert dti.day[0] == 1
        # 断言索引对象的小时属性第一个元素为0
        assert dti.hour[0] == 0
        # 断言索引对象的分钟属性第一个元素为0
        assert dti.minute[0] == 0
        # 断言索引对象的秒属性第一个元素为0
        assert dti.second[0] == 0
        # 断言索引对象的微秒属性第一个元素为0
        assert dti.microsecond[0] == 0
        # 断言索引对象的星期几属性第一个元素为3（星期三）
        assert dti.dayofweek[0] == 3

        # 断言索引对象的一年中第几天属性第一个元素为1
        assert dti.dayofyear[0] == 1
        # 断言索引对象的一年中第121天的一年中第几天属性为121
        assert dti.dayofyear[120] == 121

        # 断言索引对象的 ISO 日历周数第一个元素为1
        assert dti.isocalendar().week.iloc[0] == 1
        # 断言索引对象的 ISO 日历周数第121天的 ISO 日历周数为18
        assert dti.isocalendar().week.iloc[120] == 18

        # 断言索引对象的季度属性第一个元素为1
        assert dti.quarter[0] == 1
        # 断言索引对象的第121天的季度属性为2
        assert dti.quarter[120] == 2

        # 断言索引对象的每月天数属性第一个元素为31
        assert dti.days_in_month[0] == 31
        # 断言索引对象的第91天的每月天数属性为30
        assert dti.days_in_month[90] == 30

        # 断言索引对象的每月起始属性第一个元素为True
        assert dti.is_month_start[0]
        # 断言索引对象的第二天的每月起始属性为False
        assert not dti.is_month_start[1]
        # 断言索引对象的第32天的每月起始属性为True
        assert dti.is_month_start[31]
        # 断言索引对象的每季度起始属性第一个元素为True
        assert dti.is_quarter_start[0]
        # 断言索引对象的第91天的每季度起始属性为True
        assert dti.is_quarter_start[90]
        # 断言索引对象的每年起始属性第一个元素为True
        assert dti.is_year_start[0]
        # 断言索引对象的最后一天不是每年起始属性
        assert not dti.is_year_start[364]
        # 断言索引对象的第一天不是每月结束属性
        assert not dti.is_month_end[0]
        # 断言索引对象的第31天是每月结束属性
        assert dti.is_month_end[30]
        # 断言索引对象的第32天不是每月结束属性
        assert not dti.is_month_end[31]
        # 断言索引对象的最后一天是每月结束属性
        assert dti.is_month_end[364]
        # 断言索引对象的第一天不是每季度结束属性
        assert not dti.is_quarter_end[0]
        # 断言索引对象的第31天不是每季度结束属性
        assert not dti.is_quarter_end[30]
        # 断言索引对象的第90天是每季度结束属性
        assert dti.is_quarter_end[89]
        # 断言索引对象的最后一天是每季度结束属性
        assert dti.is_quarter_end[364]
        # 断言索引对象的第一天不是每年结束属性
        assert not dti.is_year_end[0]
        # 断言索引对象的最后一天是每年结束属性
        assert dti.is_year_end[364]

        # 断言索引对象的年份属性长度为365
        assert len(dti.year) == 365
        # 断言索引对象的月份属性长度为365
        assert len(dti.month) == 365
        # 断言索引对象的日期属性长度为365
        assert len(dti.day) == 365
        # 断言索引对象的小时属性长度为365
        assert len(dti.hour) == 365
        # 断言索引对象的分钟属性长度为365
        assert len(dti.minute) == 365
        # 断言索引对象的秒属性长度为365
        assert len(dti.second) == 365
        # 断言索引对象的微秒属性长度为365
        assert len(dti.microsecond) == 365
        # 断言索引对象的星期几属性长度为365
        assert len(dti.dayofweek) == 365
        # 断言索引对象的一年中第几天属性长度为365
        assert len(dti.dayofyear) == 365
        # 断言索引对象的 ISO 日历属性长度为365
        assert len(dti.isocalendar()) == 365
        # 断言索引对象的季度属性长度为365
        assert len(dti.quarter) == 365
        # 断言索引对象的每月起始属性长度为365
        assert len(dti.is_month_start) == 365
        # 断言索引对象的每月结束属性长度为365
        assert len(dti.is_month_end) == 365
        # 断言索引对象的每季度起始属性长度为365
        assert len(dti.is_quarter_start) == 365
        # 断言索引对象的每季度结束属性长度为365
        assert len(dti.is_quarter_end) == 365
        # 断言索引对象的每年起始属性长度为365
        assert len(dti.is_year_start) == 365
        # 断言索引对象的每年结束属性长度为365
        assert len(dti.is_year_end) == 365

        # 将索引对象的名称设为"name"
        dti.name = "name"

        # 针对非布尔访问器 -> 返回索引对象
        for accessor in DatetimeArray._field_ops:
            res = getattr(dti, accessor)
            # 断言结果的长度为365
            assert len(res) == 365
            # 断言结果是 Index 类型
            assert isinstance(res, Index)
            # 断言结果的名称为"name"
            assert res.name == "name"

        # 针对布尔访问器 -> 返回数组
        for accessor in DatetimeArray._bool_ops:
            res = getattr(dti, accessor)
            # 断言结果的长度为365
            assert len(res) == 365
            # 断言结果是 numpy 数组
            assert isinstance(res, np.ndarray)

        # 测试布尔索引
        res = dti[dti.is_quarter_start]
        exp = dti[[0, 90, 181, 273]]
        # 断言索引结果与预期结果相等
        tm.assert_index_equal(res, exp)
        res = dti[dti.is_leap_year]
        exp = DatetimeIndex([], freq="D", tz=dti.tz, name="name").as_unit("ns")
        # 断言索引结果与预期结果相等
        tm.assert_index_equal(res, exp)
    def test_dti_is_year_quarter_start(self):
        # 创建一个时间范围，频率为每季度末的二月，从1998年1月1日开始，共4个周期
        dti = date_range(freq="BQE-FEB", start=datetime(1998, 1, 1), periods=4)

        # 断言：确保在生成的时间范围中没有季度开始日期
        assert sum(dti.is_quarter_start) == 0
        # 断言：确保在生成的时间范围中有4个季度结束日期
        assert sum(dti.is_quarter_end) == 4
        # 断言：确保在生成的时间范围中没有年初日期
        assert sum(dti.is_year_start) == 0
        # 断言：确保在生成的时间范围中有1个年末日期
        assert sum(dti.is_year_end) == 1

    def test_dti_is_month_start(self):
        # 创建一个日期时间索引，包含三个日期
        dti = DatetimeIndex(["2000-01-01", "2000-01-02", "2000-01-03"])

        # 断言：确保第一个日期是月初
        assert dti.is_month_start[0] == 1

    def test_dti_is_month_start_custom(self):
        # 确保对于自定义工作日，访问is_start/end访问器会引发ValueError异常
        bday_egypt = offsets.CustomBusinessDay(weekmask="Sun Mon Tue Wed Thu")
        # 创建一个自定义工作日频率的时间范围，从2013年4月30日开始，共5个周期
        dti = date_range(datetime(2013, 4, 30), periods=5, freq=bday_egypt)
        msg = "Custom business days is not supported by is_month_start"
        # 使用pytest断言确保访问is_month_start属性时会抛出预期的ValueError异常
        with pytest.raises(ValueError, match=msg):
            dti.is_month_start

    @pytest.mark.parametrize(
        "timestamp, freq, periods, expected_values",
        [
            ("2017-12-01", "MS", 3, np.array([False, True, False])),
            ("2017-12-01", "QS", 3, np.array([True, False, False])),
            ("2017-12-01", "YS", 3, np.array([True, True, True])),
        ],
    )
    def test_dti_dr_is_year_start(self, timestamp, freq, periods, expected_values):
        # GH57377
        # 测试确保日期范围生成器在特定频率下生成的结果与期望值一致
        result = date_range(timestamp, freq=freq, periods=periods).is_year_start
        tm.assert_numpy_array_equal(result, expected_values)

    @pytest.mark.parametrize(
        "timestamp, freq, periods, expected_values",
        [
            ("2017-12-01", "ME", 3, np.array([True, False, False])),
            ("2017-12-01", "QE", 3, np.array([True, False, False])),
            ("2017-12-01", "YE", 3, np.array([True, True, True])),
        ],
    )
    def test_dti_dr_is_year_end(self, timestamp, freq, periods, expected_values):
        # GH57377
        # 测试确保日期范围生成器在特定频率下生成的结果与期望值一致
        result = date_range(timestamp, freq=freq, periods=periods).is_year_end
        tm.assert_numpy_array_equal(result, expected_values)

    @pytest.mark.parametrize(
        "timestamp, freq, periods, expected_values",
        [
            ("2017-12-01", "MS", 3, np.array([False, True, False])),
            ("2017-12-01", "QS", 3, np.array([True, True, True])),
            ("2017-12-01", "YS", 3, np.array([True, True, True])),
        ],
    )
    def test_dti_dr_is_quarter_start(self, timestamp, freq, periods, expected_values):
        # GH57377
        # 测试确保日期范围生成器在特定频率下生成的结果与期望值一致
        result = date_range(timestamp, freq=freq, periods=periods).is_quarter_start
        tm.assert_numpy_array_equal(result, expected_values)

    @pytest.mark.parametrize(
        "timestamp, freq, periods, expected_values",
        [
            ("2017-12-01", "ME", 3, np.array([True, False, False])),
            ("2017-12-01", "QE", 3, np.array([True, True, True])),
            ("2017-12-01", "YE", 3, np.array([True, True, True])),
        ],
    )
    def test_dti_dr_is_quarter_end(self, timestamp, freq, periods, expected_values):
        # GH57377
        # 调用日期范围生成函数，生成时间序列并检查是否季度结束
        result = date_range(timestamp, freq=freq, periods=periods).is_quarter_end
        tm.assert_numpy_array_equal(result, expected_values)

    @pytest.mark.parametrize(
        "timestamp, freq, periods, expected_values",
        [
            ("2017-12-01", "MS", 3, np.array([True, True, True])),
            ("2017-12-01", "QS", 3, np.array([True, True, True])),
            ("2017-12-01", "YS", 3, np.array([True, True, True])),
        ],
    )
    def test_dti_dr_is_month_start(self, timestamp, freq, periods, expected_values):
        # GH57377
        # 调用日期范围生成函数，生成时间序列并检查是否月初
        result = date_range(timestamp, freq=freq, periods=periods).is_month_start
        tm.assert_numpy_array_equal(result, expected_values)

    @pytest.mark.parametrize(
        "timestamp, freq, periods, expected_values",
        [
            ("2017-12-01", "ME", 3, np.array([True, True, True])),
            ("2017-12-01", "QE", 3, np.array([True, True, True])),
            ("2017-12-01", "YE", 3, np.array([True, True, True])),
        ],
    )
    def test_dti_dr_is_month_end(self, timestamp, freq, periods, expected_values):
        # GH57377
        # 调用日期范围生成函数，生成时间序列并检查是否月末
        result = date_range(timestamp, freq=freq, periods=periods).is_month_end
        tm.assert_numpy_array_equal(result, expected_values)

    def test_dti_is_year_quarter_start_doubledigit_freq(self):
        # GH#58523
        # 创建时间范围对象，检查是否年初（对于两位数频率）
        dr = date_range("2017-01-01", periods=2, freq="10YS")
        assert all(dr.is_year_start)

        # 创建时间范围对象，检查是否季度初（对于两位数频率）
        dr = date_range("2017-01-01", periods=2, freq="10QS")
        assert all(dr.is_quarter_start)

    def test_dti_is_year_start_freq_custom_business_day_with_digit(self):
        # GH#58664
        # 创建时间范围对象，使用自定义工作日频率，验证是否引发值错误异常
        dr = date_range("2020-01-01", periods=2, freq="2C")
        msg = "Custom business days is not supported by is_year_start"
        with pytest.raises(ValueError, match=msg):
            dr.is_year_start

    @pytest.mark.parametrize("freq", ["3BMS", offsets.BusinessMonthBegin(3)])
    def test_dti_is_year_quarter_start_freq_business_month_begin(self, freq):
        # GH#58729
        # 创建时间范围对象，使用不同的频率，检查每个日期是否年初
        dr = date_range("2020-01-01", periods=5, freq=freq)
        result = [x.is_year_start for x in dr]
        assert result == [True, False, False, False, True]

        # 创建时间范围对象，使用不同的频率，检查每个日期是否季度初
        dr = date_range("2020-01-01", periods=4, freq=freq)
        result = [x.is_quarter_start for x in dr]
        assert all(dr.is_quarter_start)
# 使用 pytest 的 given 装饰器，定义测试参数，包括日期时间范围、整数范围和频率选项
@given(
    dt=st.datetimes(min_value=datetime(1960, 1, 1), max_value=datetime(1980, 1, 1)),
    n=st.integers(min_value=1, max_value=10),
    freq=st.sampled_from(["MS", "QS", "YS"]),
)
# 使用 pytest 的 mark 标记，将此测试标记为慢速执行的测试用例
@pytest.mark.slow
# 定义测试函数，对给定的频率、日期时间和整数参数进行测试
def test_against_scalar_parametric(freq, dt, n):
    # 添加链接注释，指向相关的 GitHub 问题页面
    # https://github.com/pandas-dev/pandas/issues/49606
    # 将整数 n 和频率 freq 结合为一个字符串，更新频率参数
    freq = f"{n}{freq}"
    # 创建日期范围，以给定的日期时间为起点，生成 3 个时间点，使用组合后的频率参数
    d = date_range(dt, periods=3, freq=freq)
    # 获取日期范围中每个日期是否为年初的布尔列表
    result = list(d.is_year_start)
    # 生成预期的每个日期是否为年初的布尔列表
    expected = [x.is_year_start for x in d]
    # 断言实际结果与预期结果是否一致
    assert result == expected
```