# `D:\src\scipysrc\pandas\pandas\tests\scalar\timestamp\test_constructors.py`

```
import calendar  # 导入日历模块
from datetime import (  # 从 datetime 模块中导入以下类和函数
    date,  # 日期类
    datetime,  # 日期时间类
    timedelta,  # 时间间隔类
    timezone,  # 时区类
)
import zoneinfo  # 导入 zoneinfo 模块，用于时区信息

import dateutil.tz  # 导入 dateutil 库的时区功能
from dateutil.tz import (  # 从 dateutil.tz 模块中导入以下类和函数
    gettz,  # 获取时区函数
    tzoffset,  # 创建固定偏移量时区对象的函数
    tzutc,  # UTC 时区对象
)
import numpy as np  # 导入 NumPy 库并使用 np 别名
import pytest  # 导入 pytest 测试框架

import pytz  # 导入 pytz 库，用于处理时区

from pandas._libs.tslibs.dtypes import NpyDatetimeUnit  # 从 pandas 库中导入 NpyDatetimeUnit 类
from pandas.errors import OutOfBoundsDatetime  # 从 pandas 库中导入 OutOfBoundsDatetime 异常类

from pandas import (  # 从 pandas 库中导入以下类和函数
    NA,  # 表示缺失值
    NaT,  # 表示缺失的时间戳
    Period,  # 表示时间段
    Timedelta,  # 表示时间间隔
    Timestamp,  # 表示时间戳
)
import pandas._testing as tm  # 导入 pandas 测试相关的模块

# 定义测试类 TestTimestampConstructorUnitKeyword
class TestTimestampConstructorUnitKeyword:

    @pytest.mark.parametrize("typ", [int, float])
    def test_constructor_int_float_with_YM_unit(self, typ):
        # GH#47266 避免在 cast_from_unit 中的类型转换
        val = typ(150)

        ts = Timestamp(val, unit="Y")  # 使用年单位创建时间戳对象
        expected = Timestamp("2120-01-01")  # 期望的时间戳对象为 2120-01-01
        assert ts == expected  # 断言实际得到的时间戳与期望的时间戳相等

        ts = Timestamp(val, unit="M")  # 使用月单位创建时间戳对象
        expected = Timestamp("1982-07-01")  # 期望的时间戳对象为 1982-07-01
        assert ts == expected  # 断言实际得到的时间戳与期望的时间戳相等

    @pytest.mark.parametrize("typ", [int, float])
    def test_construct_from_int_float_with_unit_out_of_bound_raises(self, typ):
        # GH#50870 确保我们得到 OutOfBoundsDatetime 而不是 OverflowError
        val = typ(150000000000000)

        msg = f"cannot convert input {val} with the unit 'D'"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp(val, unit="D")  # 使用天单位创建时间戳对象，预期引发 OutOfBoundsDatetime 异常

    def test_constructor_float_not_round_with_YM_unit_raises(self):
        # GH#47267 避免在 cast_from_unit 中的类型转换

        msg = "Conversion of non-round float with unit=[MY] is ambiguous"
        with pytest.raises(ValueError, match=msg):
            Timestamp(150.5, unit="Y")  # 使用年单位创建非整数时间戳，预期引发 ValueError 异常

        with pytest.raises(ValueError, match=msg):
            Timestamp(150.5, unit="M")  # 使用月单位创建非整数时间戳，预期引发 ValueError 异常
    # 使用 pytest.mark.parametrize 装饰器为 test_construct_with_unit 方法添加多组参数化测试数据
    @pytest.mark.parametrize(
        "value, check_kwargs",
        [
            # 参数化测试数据，时间戳为 946688461000000000，不带单位
            [946688461000000000, {}],
            # 参数化测试数据，时间戳为 946688461000000000 微秒，传入单位为 "us"
            [946688461000000000 / 1000, {"unit": "us"}],
            # 参数化测试数据，时间戳为 946688461000000000 毫秒，传入单位为 "ms"
            [946688461000000000 / 1_000_000, {"unit": "ms"}],
            # 参数化测试数据，时间戳为 946688461000000000 秒，传入单位为 "s"
            [946688461000000000 / 1_000_000_000, {"unit": "s"}],
            # 参数化测试数据，时间戳为 10957 天，传入单位为 "D"，小时为 0
            [10957, {"unit": "D", "h": 0}],
            # 参数化测试数据，时间戳为 946688461000000000.5 秒，传入单位为 "s"，微秒为 499964
            [
                (946688461000000000 + 500000) / 1000000000,
                {"unit": "s", "us": 499, "ns": 964},
            ],
            # 参数化测试数据，时间戳为 946688461500000000 秒，传入单位为 "s"，微秒为 500000
            [
                (946688461000000000 + 500000000) / 1000000000,
                {"unit": "s", "us": 500000},
            ],
            # 参数化测试数据，时间戳为 946688461000.5 毫秒，传入单位为 "ms"，微秒为 500
            [(946688461000000000 + 500000) / 1000, {"unit": "ms", "us": 500}],
            # 参数化测试数据，时间戳为 946688461000.5 微秒，传入单位为 "us"，微秒为 500
            [(946688461000000000 + 500000) / 1000, {"unit": "us", "us": 500}],
            # 参数化测试数据，时间戳为 946688461500.0 毫秒，传入单位为 "ms"，微秒为 500000
            [(946688461000000000 + 500000000) / 1000000, {"unit": "ms", "us": 500000}],
            # 参数化测试数据，时间戳为 946688461005.0 微秒，传入单位为 "us"，微秒为 5
            [946688461000000000 / 1000.0 + 5, {"unit": "us", "us": 5}],
            # 参数化测试数据，时间戳为 946688461005.0 微秒，传入单位为 "us"，微秒为 5000
            [946688461000000000 / 1000.0 + 5000, {"unit": "us", "us": 5000}],
            # 参数化测试数据，时间戳为 946688461000.5 毫秒，传入单位为 "ms"，微秒为 500
            [946688461000000000 / 1000000.0 + 0.5, {"unit": "ms", "us": 500}],
            # 参数化测试数据，时间戳为 946688461000.005 毫秒，传入单位为 "ms"，微秒为 5，纳秒为 5
            [946688461000000000 / 1000000.0 + 0.005, {"unit": "ms", "us": 5, "ns": 5}],
            # 参数化测试数据，时间戳为 946688461000000000.5 秒，传入单位为 "s"，微秒为 500000
            [946688461000000000 / 1000000000.0 + 0.5, {"unit": "s", "us": 500000}],
            # 参数化测试数据，时间戳为 10957.5 天，传入单位为 "D"，小时为 12
            [10957 + 0.5, {"unit": "D", "h": 12}],
        ],
    )
    # 定义测试方法 test_construct_with_unit，用于测试 Timestamp 类的构造函数
    def test_construct_with_unit(self, value, check_kwargs):
        # 定义内部函数 check，用于验证 Timestamp 对象的属性是否符合预期
        def check(value, unit=None, h=1, s=1, us=0, ns=0):
            # 创建 Timestamp 对象，传入时间戳值和单位参数
            stamp = Timestamp(value, unit=unit)
            # 断言 Timestamp 对象的年份为 2000 年
            assert stamp.year == 2000
            # 断言 Timestamp 对象的月份为 1 月
            assert stamp.month == 1
            # 断言 Timestamp 对象的日期为 1 日
            assert stamp.day == 1
            # 断言 Timestamp 对象的小时数与传入参数 h 一致
            assert stamp.hour == h
            # 如果单位不是 "D"，则继续验证分钟、秒钟和微秒
            if unit != "D":
                assert stamp.minute == 1
                assert stamp.second == s
                assert stamp.microsecond == us
            else:
                # 如果单位是 "D"，则分钟、秒钟和微秒应为 0
                assert stamp.minute == 0
                assert stamp.second == 0
                assert stamp.microsecond == 0
            # 断言 Timestamp 对象的纳秒数为传入参数 ns
            assert stamp.nanosecond == ns

        # 调用 check 函数，传入参数进行验证
        check(value, **check_kwargs)
class TestTimestampConstructorFoldKeyword:
    # 测试时间戳构造函数中 fold 参数的异常情况
    def test_timestamp_constructor_invalid_fold_raise(self):
        # 测试 GH#25057
        # fold 参数的有效值仅限于 [None, 0, 1]
        msg = "Valid values for the fold argument are None, 0, or 1."
        # 使用 pytest 来确保引发 ValueError 异常，并匹配特定的错误消息
        with pytest.raises(ValueError, match=msg):
            Timestamp(123, fold=2)

    # 测试时间戳构造函数中使用 pytz 时 fold 参数的异常情况
    def test_timestamp_constructor_pytz_fold_raise(self):
        # 测试 GH#25057
        # pytz 不支持 fold。检查是否在使用 pytz 时引发异常
        pytz = pytest.importorskip("pytz")
        msg = "pytz timezones do not support fold. Please use dateutil timezones."
        tz = pytz.timezone("Europe/London")
        # 使用 pytest 来确保引发 ValueError 异常，并匹配特定的错误消息
        with pytest.raises(ValueError, match=msg):
            Timestamp(datetime(2019, 10, 27, 0, 30, 0, 0), tz=tz, fold=0)

    # 参数化测试，测试时间戳构造函数中 fold 冲突的情况
    @pytest.mark.parametrize("fold", [0, 1])
    @pytest.mark.parametrize(
        "ts_input",
        [
            1572136200000000000,
            1572136200000000000.0,
            np.datetime64(1572136200000000000, "ns"),
            "2019-10-27 01:30:00+01:00",
            datetime(2019, 10, 27, 0, 30, 0, 0, tzinfo=timezone.utc),
        ],
    )
    def test_timestamp_constructor_fold_conflict(self, ts_input, fold):
        # 测试 GH#25057
        # 检查在 fold 冲突时是否引发异常
        msg = (
            "Cannot pass fold with possibly unambiguous input: int, float, "
            "numpy.datetime64, str, or timezone-aware datetime-like. "
            "Pass naive datetime-like or build Timestamp from components."
        )
        # 使用 pytest 来确保引发 ValueError 异常，并匹配特定的错误消息
        with pytest.raises(ValueError, match=msg):
            Timestamp(ts_input=ts_input, fold=fold)

    # 参数化测试，测试时间戳构造函数中保留 fold 的情况
    @pytest.mark.parametrize("tz", ["dateutil/Europe/London", None])
    @pytest.mark.parametrize("fold", [0, 1])
    def test_timestamp_constructor_retain_fold(self, tz, fold):
        # 测试 GH#25057
        # 检查是否正确保留 fold
        ts = Timestamp(year=2019, month=10, day=27, hour=1, minute=30, tz=tz, fold=fold)
        result = ts.fold
        expected = fold
        # 使用 assert 断言来验证结果是否符合预期
        assert result == expected

    # 参数化测试，根据值推断 fold 的情况
    @pytest.mark.parametrize(
        "tz",
        [
            "dateutil/Europe/London",
            zoneinfo.ZoneInfo("Europe/London"),
        ],
    )
    @pytest.mark.parametrize(
        "ts_input,fold_out",
        [
            (1572136200000000000, 0),
            (1572139800000000000, 1),
            ("2019-10-27 01:30:00+01:00", 0),
            ("2019-10-27 01:30:00+00:00", 1),
            (datetime(2019, 10, 27, 1, 30, 0, 0, fold=0), 0),
            (datetime(2019, 10, 27, 1, 30, 0, 0, fold=1), 1),
        ],
    )
    def test_timestamp_constructor_infer_fold_from_value(self, tz, ts_input, fold_out):
        # 测试 GH#25057
        # 检查是否根据时间戳或字符串正确推断 fold
        ts = Timestamp(ts_input, tz=tz)
        result = ts.fold
        expected = fold_out
        # 使用 assert 断言来验证结果是否符合预期
        assert result == expected
    @pytest.mark.parametrize("tz", ["dateutil/Europe/London"])
    @pytest.mark.parametrize(
        "fold,value_out",
        [
            (0, 1572136200000000),  # 定义测试参数：折叠值为0时的预期输出时间戳
            (1, 1572139800000000),  # 定义测试参数：折叠值为1时的预期输出时间戳
        ],
    )
    def test_timestamp_constructor_adjust_value_for_fold(self, tz, fold, value_out):
        # Test for GH#25057
        # Check that we adjust value for fold correctly
        # based on timestamps since utc

        # 创建一个输入的时间戳对象，表示2019年10月27日1点30分
        ts_input = datetime(2019, 10, 27, 1, 30)
        
        # 使用给定的时区（tz）和折叠值（fold）创建时间戳对象（Timestamp对象）
        ts = Timestamp(ts_input, tz=tz, fold=fold)
        
        # 获取时间戳对象内部的时间值（即存储的时间戳）
        result = ts._value
        
        # 定义预期的时间戳值
        expected = value_out
        
        # 断言实际的时间戳值与预期值相等
        assert result == expected
    # 定义一个测试类，用于测试时间戳构造函数的位置参数和关键字参数的支持
    class TestTimestampConstructorPositionalAndKeywordSupport:
        
        # 测试位置参数构造函数
        
        def test_constructor_positional(self):
            # 测试位置参数构造函数对缺少day参数时的异常情况
            msg = "'NoneType' object cannot be interpreted as an integer"
            with pytest.raises(TypeError, match=msg):
                Timestamp(2000, 1)
    
            # 测试位置参数构造函数对月份超出范围时的异常情况
            msg = "month must be in 1..12"
            with pytest.raises(ValueError, match=msg):
                Timestamp(2000, 0, 1)
            with pytest.raises(ValueError, match=msg):
                Timestamp(2000, 13, 1)
    
            # 测试位置参数构造函数对日期超出范围时的异常情况
            msg = "day is out of range for month"
            with pytest.raises(ValueError, match=msg):
                Timestamp(2000, 1, 0)
            with pytest.raises(ValueError, match=msg):
                Timestamp(2000, 1, 32)
    
            # 测试位置参数构造函数是否正确处理日期表示的字符串
            # 详见gh-11630
            assert repr(Timestamp(2015, 11, 12)) == repr(Timestamp("20151112"))
            assert repr(Timestamp(2015, 11, 12, 1, 2, 3, 999999)) == repr(
                Timestamp("2015-11-12 01:02:03.999999")
            )
    
        # 测试关键字参数构造函数
        
        def test_constructor_keyword(self):
            # 测试关键字参数构造函数对缺少day参数时的异常情况
            msg = "function missing required argument 'day'|Required argument 'day'"
            with pytest.raises(TypeError, match=msg):
                Timestamp(year=2000, month=1)
    
            # 测试关键字参数构造函数对月份超出范围时的异常情况
            msg = "month must be in 1..12"
            with pytest.raises(ValueError, match=msg):
                Timestamp(year=2000, month=0, day=1)
            with pytest.raises(ValueError, match=msg):
                Timestamp(year=2000, month=13, day=1)
    
            # 测试关键字参数构造函数对日期超出范围时的异常情况
            msg = "day is out of range for month"
            with pytest.raises(ValueError, match=msg):
                Timestamp(year=2000, month=1, day=0)
            with pytest.raises(ValueError, match=msg):
                Timestamp(year=2000, month=1, day=32)
    
            # 测试关键字参数构造函数是否正确处理日期表示的字符串
            assert repr(Timestamp(year=2015, month=11, day=12)) == repr(
                Timestamp("20151112")
            )
    
            assert repr(
                Timestamp(
                    year=2015,
                    month=11,
                    day=12,
                    hour=1,
                    minute=2,
                    second=3,
                    microsecond=999999,
                )
            ) == repr(Timestamp("2015-11-12 01:02:03.999999"))
    
        # 测试带有无效日期关键字参数输入的情况
        
        @pytest.mark.parametrize(
            "arg",
            [
                "year",
                "month",
                "day",
                "hour",
                "minute",
                "second",
                "microsecond",
                "nanosecond",
            ],
        )
        def test_invalid_date_kwarg_with_string_input(self, arg):
            kwarg = {arg: 1}
            # 测试是否能够捕获试图将日期属性作为关键字参数传递的异常情况
            msg = "Cannot pass a date attribute keyword argument"
            with pytest.raises(ValueError, match=msg):
                Timestamp("2010-10-10 12:59:59.999999999", **kwarg)
    
        # 测试带有多种有效和无效关键字参数组合的情况
        
        @pytest.mark.parametrize("kwargs", [{}, {"year": 2020}, {"year": 2020, "month": 1}])
    # 定义一个测试函数，用于测试在缺少关键字参数时构造 Timestamp 对象是否会引发异常
    def test_constructor_missing_keyword(self, kwargs):
        # GH#31200

        # 定义两个可能的错误消息模式，用于匹配 TypeError 异常信息
        msg1 = r"function missing required argument '(year|month|day)' \(pos [123]\)"
        msg2 = r"Required argument '(year|month|day)' \(pos [123]\) not found"
        msg = "|".join([msg1, msg2])

        # 使用 pytest.raises 来捕获 TypeError 异常，并验证是否匹配预期的错误消息模式
        with pytest.raises(TypeError, match=msg):
            Timestamp(**kwargs)

    # 定义一个测试函数，用于测试带时区信息的位置参数构造 Timestamp 对象的正确性
    def test_constructor_positional_with_tzinfo(self):
        # GH#31929

        # 创建一个带有时区信息的 Timestamp 对象
        ts = Timestamp(2020, 12, 31, tzinfo=timezone.utc)
        # 创建预期的 Timestamp 对象，带有相同的日期和时区信息
        expected = Timestamp("2020-12-31", tzinfo=timezone.utc)
        # 断言实际创建的 Timestamp 对象与预期的对象相等
        assert ts == expected

    # 使用 pytest.mark.parametrize 注解来多次调用同一个测试函数，测试不同的参数组合
    @pytest.mark.parametrize("kwd", ["nanosecond", "microsecond", "second", "minute"])
    # 定义一个测试函数，测试带时区信息的位置参数和关键字参数混合构造 Timestamp 对象的正确性
    def test_constructor_positional_keyword_mixed_with_tzinfo(self, kwd, request):
        # TODO: if we passed microsecond with a keyword we would mess up
        #  xref GH#45307
        if kwd != "nanosecond":
            # 如果参数不是 "nanosecond"，则标记为预期失败，因为在 2.0 版本后，只有 "nanosecond" 是关键字参数
            mark = pytest.mark.xfail(reason="GH#45307")
            request.applymarker(mark)

        # 准备关键字参数字典，包含一个指定单位的时间参数
        kwargs = {kwd: 4}
        # 创建带有时区信息和指定时间参数的 Timestamp 对象
        ts = Timestamp(2020, 12, 31, tzinfo=timezone.utc, **kwargs)

        # 准备 Timedelta 对象的关键字参数字典，用于计算预期的 Timestamp
        td_kwargs = {kwd + "s": 4}
        # 创建一个 Timedelta 对象，用于与 Timestamp 对象相加
        td = Timedelta(**td_kwargs)
        # 创建预期的 Timestamp 对象，为初始日期加上 Timedelta
        expected = Timestamp("2020-12-31", tz=timezone.utc) + td
        # 断言实际创建的 Timestamp 对象与预期的对象相等
        assert ts == expected
class TestTimestampClassMethodConstructors:
    # Timestamp constructors other than __new__

    def test_utcnow_deprecated(self):
        # 测试方法：test_utcnow_deprecated
        # GH#56680
        # 设置警告消息
        msg = "Timestamp.utcnow is deprecated"
        # 断言在调用 Timestamp.utcnow() 时会产生 FutureWarning 警告，并匹配消息内容
        with tm.assert_produces_warning(FutureWarning, match=msg):
            Timestamp.utcnow()

    def test_utcfromtimestamp_deprecated(self):
        # 测试方法：test_utcfromtimestamp_deprecated
        # GH#56680
        # 设置警告消息
        msg = "Timestamp.utcfromtimestamp is deprecated"
        # 断言在调用 Timestamp.utcfromtimestamp(43) 时会产生 FutureWarning 警告，并匹配消息内容
        with tm.assert_produces_warning(FutureWarning, match=msg):
            Timestamp.utcfromtimestamp(43)

    def test_constructor_strptime(self):
        # 测试方法：test_constructor_strptime
        # GH#25016
        # 测试对 Timestamp.strptime 的支持
        # 设置时间格式和时间戳
        fmt = "%Y%m%d-%H%M%S-%f%z"
        ts = "20190129-235348-000001+0000"
        # 设置未实现的错误消息
        msg = r"Timestamp.strptime\(\) is not implemented"
        # 断言调用 Timestamp.strptime(ts, fmt) 会抛出 NotImplementedError，并匹配消息内容
        with pytest.raises(NotImplementedError, match=msg):
            Timestamp.strptime(ts, fmt)

    def test_constructor_fromisocalendar(self):
        # 测试方法：test_constructor_fromisocalendar
        # GH#30395
        # 预期的时间戳
        expected_timestamp = Timestamp("2000-01-03 00:00:00")
        # 使用标准库创建的预期时间
        expected_stdlib = datetime.fromisocalendar(2000, 1, 1)
        # 调用 Timestamp.fromisocalendar(2000, 1, 1) 获取结果
        result = Timestamp.fromisocalendar(2000, 1, 1)
        # 断言结果与预期时间戳相等
        assert result == expected_timestamp
        # 断言结果与标准库创建的预期时间相等
        assert result == expected_stdlib
        # 断言结果是 Timestamp 类的实例
        assert isinstance(result, Timestamp)

    def test_constructor_fromordinal(self):
        # 测试方法：test_constructor_fromordinal
        base = datetime(2000, 1, 1)

        # 使用 fromordinal 创建时间戳并进行断言
        ts = Timestamp.fromordinal(base.toordinal())
        assert base == ts
        assert base.toordinal() == ts.toordinal()

        # 使用带有时区信息的 fromordinal 创建时间戳并进行断言
        ts = Timestamp.fromordinal(base.toordinal(), tz="US/Eastern")
        assert Timestamp("2000-01-01", tz="US/Eastern") == ts
        assert base.toordinal() == ts.toordinal()

        # GH#3042
        dt = datetime(2011, 4, 16, 0, 0)
        # 使用 fromordinal 创建时间戳并进行断言
        ts = Timestamp.fromordinal(dt.toordinal())
        assert ts.to_pydatetime() == dt

        # 使用带有时区信息的 fromordinal 创建时间戳并进行断言
        stamp = Timestamp("2011-4-16", tz="US/Eastern")
        dt_tz = stamp.to_pydatetime()
        ts = Timestamp.fromordinal(dt_tz.toordinal(), tz="US/Eastern")
        assert ts.to_pydatetime() == dt_tz

    def test_now(self):
        # 测试方法：test_now
        # GH#9000
        # 从字符串创建时间戳
        ts_from_string = Timestamp("now")
        # 使用 now() 方法创建时间戳
        ts_from_method = Timestamp.now()
        # 获取当前系统时间
        ts_datetime = datetime.now()

        # 从带时区信息的字符串创建时间戳
        ts_from_string_tz = Timestamp("now", tz="US/Eastern")
        # 使用带时区信息的 now() 方法创建时间戳
        ts_from_method_tz = Timestamp.now(tz="US/Eastern")

        # 检查两个时间戳之间的时间差小于1秒（任意小）
        delta = Timedelta(seconds=1)
        assert abs(ts_from_method - ts_from_string) < delta
        assert abs(ts_datetime - ts_from_method) < delta
        assert abs(ts_from_method_tz - ts_from_string_tz) < delta
        assert (
            abs(
                ts_from_string_tz.tz_localize(None)
                - ts_from_method_tz.tz_localize(None)
            )
            < delta
        )
    # 定义一个测试方法，用于测试时间戳相关功能
    def test_today(self):
        # 从字符串 "today" 创建时间戳对象
        ts_from_string = Timestamp("today")
        # 调用静态方法获取当前时间戳对象
        ts_from_method = Timestamp.today()
        # 获取当前日期时间对象
        ts_datetime = datetime.today()

        # 使用时区 "US/Eastern" 从字符串 "today" 创建时间戳对象
        ts_from_string_tz = Timestamp("today", tz="US/Eastern")
        # 使用时区 "US/Eastern" 调用静态方法获取当前时间戳对象
        ts_from_method_tz = Timestamp.today(tz="US/Eastern")

        # 检查时间戳之间的时间差是否小于 1 秒（任意设置的阈值）
        delta = Timedelta(seconds=1)
        assert abs(ts_from_method - ts_from_string) < delta
        assert abs(ts_datetime - ts_from_method) < delta
        assert abs(ts_from_method_tz - ts_from_string_tz) < delta
        # 检查使用不含时区信息的本地化时间戳之间的时间差是否小于 1 秒
        assert (
            abs(
                ts_from_string_tz.tz_localize(None)
                - ts_from_method_tz.tz_localize(None)
            )
            < delta
        )
class TestTimestampResolutionInference:
    def test_construct_from_time_unit(self):
        # 创建一个时间戳对象，仅包含时间部分，没有日期
        ts = Timestamp("01:01:01.111")
        assert ts.unit == "ms"

    def test_constructor_str_infer_reso(self):
        # 非ISO8601路径

        # 使用_parse_delimited_date解析日期字符串
        ts = Timestamp("01/30/2023")
        assert ts.unit == "s"

        # 使用_parse_dateabbr_string解析日期字符串
        ts = Timestamp("2015Q1")
        assert ts.unit == "s"

        # 使用dateutil_parse解析日期字符串
        ts = Timestamp("2016-01-01 1:30:01 PM")
        assert ts.unit == "s"

        # 解析包含日期和时间的字符串，推断时间戳的分辨率为毫秒
        ts = Timestamp("2016 June 3 15:25:01.345")
        assert ts.unit == "ms"

        # 解析未来日期，推断时间戳的分辨率为秒
        ts = Timestamp("300-01-01")
        assert ts.unit == "s"

        # 解析未来日期和时间，推断时间戳的分辨率为毫秒
        ts = Timestamp("300 June 1:30:01.300")
        assert ts.unit == "ms"

        # 使用dateutil解析ISO格式的日期字符串，保留尾部零
        ts = Timestamp("01-01-2013T00:00:00.000000000+0000")
        assert ts.unit == "ns"

        # 使用dateutil解析日期和时间，推断时间戳的分辨率为微秒
        ts = Timestamp("2016/01/02 03:04:05.001000 UTC")
        assert ts.unit == "us"

        # 解析日期字符串，保留纳秒级精度
        ts = Timestamp("01-01-2013T00:00:00.000000002100+0000")
        assert ts == Timestamp("01-01-2013T00:00:00.000000002+0000")
        assert ts.unit == "ns"

        # 通过ISO8601路径解析具有时区偏移的日期字符串，推断时间戳的分辨率为秒
        ts = Timestamp("2020-01-01 00:00+00:00")
        assert ts.unit == "s"

        # 通过ISO8601路径解析具有时区偏移的日期字符串，推断时间戳的分辨率为秒
        ts = Timestamp("2020-01-01 00+00:00")
        assert ts.unit == "s"

    @pytest.mark.parametrize("method", ["now", "today"])
    def test_now_today_unit(self, method):
        # GH#55879
        # 调用Timestamp类的now或today方法，检查返回的时间戳单位为微秒
        ts_from_method = getattr(Timestamp, method)()
        ts_from_string = Timestamp(method)
        assert ts_from_method.unit == ts_from_string.unit == "us"


class TestTimestampConstructors:
    def test_weekday_but_no_day_raises(self):
        # GH#52659
        # 测试如果解析带有星期几但没有日期信息的字符串，是否会引发ValueError异常
        msg = "Parsing datetimes with weekday but no day information is not supported"
        with pytest.raises(ValueError, match=msg):
            Timestamp("2023 Sept Thu")

    def test_construct_from_string_invalid_raises(self):
        # dateutil（奇怪地）将"200622-12-31"解析为datetime(2022, 6, 20, 12, 0, tzinfo=tzoffset(None, -111600)
        # 这不仅解析错误，还会导致str(ts)抛出ValueError。在构造函数中确保抛出异常。
        # 参见test_to_datetime_malformed_raise用于类似to_datetime的测试
        with pytest.raises(ValueError, match="gives an invalid tzoffset"):
            Timestamp("200622-12-31")
    # 测试函数：从 ISO8601 格式带有偏移量和不同分辨率的字符串构造 Timestamp 对象
    def test_constructor_from_iso8601_str_with_offset_reso(self):
        # 创建 Timestamp 对象，传入带有时区偏移的日期时间字符串
        ts = Timestamp("2016-01-01 04:05:06-01:00")
        # 断言 Timestamp 对象的时间单位为秒
        assert ts.unit == "s"

        # 创建 Timestamp 对象，传入带有毫秒级精度和时区偏移的日期时间字符串
        ts = Timestamp("2016-01-01 04:05:06.000-01:00")
        # 断言 Timestamp 对象的时间单位为毫秒
        assert ts.unit == "ms"

        # 创建 Timestamp 对象，传入带有微秒级精度和时区偏移的日期时间字符串
        ts = Timestamp("2016-01-01 04:05:06.000000-01:00")
        # 断言 Timestamp 对象的时间单位为微秒
        assert ts.unit == "us"

        # 创建 Timestamp 对象，传入带有纳秒级精度和时区偏移的日期时间字符串
        ts = Timestamp("2016-01-01 04:05:06.000000001-01:00")
        # 断言 Timestamp 对象的时间单位为纳秒
        assert ts.unit == "ns"

    # 测试函数：从日期对象构造 Timestamp 对象，精度为秒
    def test_constructor_from_date_second_reso(self):
        # 创建日期对象
        obj = date(2012, 9, 1)
        # 从日期对象构造 Timestamp 对象
        ts = Timestamp(obj)
        # 断言 Timestamp 对象的时间单位为秒
        assert ts.unit == "s"

    # 测试函数：从带有时区信息的 np.datetime64 对象构造 Timestamp 对象
    def test_constructor_datetime64_with_tz(self):
        # 创建 np.datetime64 对象，表示特定时间
        dt = np.datetime64("1970-01-01 05:00:00")
        # 时区字符串
        tzstr = "UTC+05:00"

        # 创建带有时区信息的 Timestamp 对象
        ts = Timestamp(dt, tz=tzstr)

        # 通过 Timestamp 对象方法设置同样的时区信息
        alt = Timestamp(dt).tz_localize(tzstr)
        # 断言两个 Timestamp 对象相等
        assert ts == alt
        # 断言 Timestamp 对象的小时属性为 5
        assert ts.hour == 5
    # 定义测试类中的构造函数测试方法
    def test_constructor(self):
        # 设定基准字符串和对应的 datetime 对象，以及预期的纳秒时间戳
        base_str = "2014-07-01 09:00"
        base_dt = datetime(2014, 7, 1, 9)
        base_expected = 1_404_205_200_000_000_000

        # 确认基准表示正确
        assert calendar.timegm(base_dt.timetuple()) * 1_000_000_000 == base_expected

        # 设定测试用例列表，每个元组包含输入字符串、datetime 对象和预期的纳秒时间戳
        tests = [
            (base_str, base_dt, base_expected),
            (
                "2014-07-01 10:00",
                datetime(2014, 7, 1, 10),
                base_expected + 3600 * 1_000_000_000,
            ),
            (
                "2014-07-01 09:00:00.000008000",
                datetime(2014, 7, 1, 9, 0, 0, 8),
                base_expected + 8000,
            ),
            (
                "2014-07-01 09:00:00.000000005",
                Timestamp("2014-07-01 09:00:00.000000005"),
                base_expected + 5,
            ),
        ]

        # 设定时区列表，每个元组包含时区对象或字符串以及预期的偏移量
        timezones = [
            (None, 0),
            ("UTC", 0),
            (timezone.utc, 0),
            ("Asia/Tokyo", 9),
            ("US/Eastern", -4),
            ("dateutil/US/Pacific", -7),
            (timezone(timedelta(hours=-3)), -3),
            (dateutil.tz.tzoffset(None, 18000), 5),
        ]

        # 迭代所有测试用例
        for date_str, date_obj, expected in tests:
            # 对于每个日期字符串或 datetime 对象，生成 Timestamp 对象，并验证其纳秒值是否符合预期
            for result in [Timestamp(date_str), Timestamp(date_obj)]:
                result = result.as_unit("ns")  # 测试在非纳秒单位下的原始写法
                # 仅针对时间字符串进行断言
                assert result.as_unit("ns")._value == expected

                # 重新创建不应影响内部值
                result = Timestamp(result)
                assert result.as_unit("ns")._value == expected

            # 对于带有时区的情况
            for tz, offset in timezones:
                # 对于每个日期字符串或 datetime 对象，生成带有时区的 Timestamp 对象，并验证其纳秒值是否符合预期的时区调整后值
                for result in [Timestamp(date_str, tz=tz), Timestamp(date_obj, tz=tz)]:
                    result = result.as_unit(
                        "ns"
                    )  # 测试在非纳秒单位下的原始写法
                    expected_tz = expected - offset * 3600 * 1_000_000_000
                    assert result.as_unit("ns")._value == expected_tz

                    # 应保持时区
                    result = Timestamp(result)
                    assert result.as_unit("ns")._value == expected_tz

                    # 应转换为 UTC
                    if tz is not None:
                        result = Timestamp(result).tz_convert("UTC")
                    else:
                        result = Timestamp(result, tz="UTC")
                    expected_utc = expected - offset * 3600 * 1_000_000_000
                    assert result.as_unit("ns")._value == expected_utc

    # 定义测试类中的无效构造函数参数测试方法
    def test_constructor_invalid(self):
        # 设置错误消息并使用 pytest 的 assertRaises 验证错误类型和消息
        msg = "Cannot convert input"
        with pytest.raises(TypeError, match=msg):
            Timestamp(slice(2))
        msg = "Cannot convert Period"
        with pytest.raises(ValueError, match=msg):
            Timestamp(Period("1000-01-01"))
    def test_constructor_invalid_tz(self):
        # GH#17690
        # 准备错误消息，指出 'tzinfo' 参数类型错误
        msg = (
            "Argument 'tzinfo' has incorrect type "
            r"\(expected datetime.tzinfo, got str\)"
        )
        # 断言在创建 Timestamp 对象时抛出 TypeError，并检查错误消息是否匹配
        with pytest.raises(TypeError, match=msg):
            Timestamp("2017-10-22", tzinfo="US/Eastern")

        # 准备错误消息，指出传递了多个时区相关参数
        msg = "at most one of"
        # 断言在创建 Timestamp 对象时抛出 ValueError，并检查错误消息是否匹配
        with pytest.raises(ValueError, match=msg):
            Timestamp("2017-10-22", tzinfo=timezone.utc, tz="UTC")

        # 准备错误消息，指出在传递日期字符串时，不能同时传递日期属性关键字参数
        msg = "Cannot pass a date attribute keyword argument when passing a date string"
        # 断言在创建 Timestamp 对象时抛出 ValueError，并检查错误消息是否匹配
        with pytest.raises(ValueError, match=msg):
            # GH#5168
            # 处理用户试图将 tz 作为参数而不是关键字参数的情况，会被解释为 `year`
            Timestamp("2012-01-01", "US/Pacific")

    def test_constructor_tz_or_tzinfo(self):
        # GH#17943, GH#17690, GH#5168
        # 准备多个 Timestamp 对象的列表，每个对象使用不同的时区相关参数进行创建
        stamps = [
            Timestamp(year=2017, month=10, day=22, tz="UTC"),
            Timestamp(year=2017, month=10, day=22, tzinfo=timezone.utc),
            Timestamp(year=2017, month=10, day=22, tz=timezone.utc),
            Timestamp(datetime(2017, 10, 22), tzinfo=timezone.utc),
            Timestamp(datetime(2017, 10, 22), tz="UTC"),
            Timestamp(datetime(2017, 10, 22), tz=timezone.utc),
        ]
        # 断言所有的 Timestamp 对象都与列表中的第一个对象相等
        assert all(ts == stamps[0] for ts in stamps)

    @pytest.mark.parametrize(
        "result",
        [
            # 准备多个带有纳秒参数的 Timestamp 对象
            Timestamp(datetime(2000, 1, 2, 3, 4, 5, 6), nanosecond=1),
            Timestamp(
                year=2000,
                month=1,
                day=2,
                hour=3,
                minute=4,
                second=5,
                microsecond=6,
                nanosecond=1,
            ),
            Timestamp(
                year=2000,
                month=1,
                day=2,
                hour=3,
                minute=4,
                second=5,
                microsecond=6,
                nanosecond=1,
                tz="UTC",
            ),
            Timestamp(2000, 1, 2, 3, 4, 5, 6, None, nanosecond=1),
            Timestamp(2000, 1, 2, 3, 4, 5, 6, tz=pytz.UTC, nanosecond=1),
        ],
    )
    def test_constructor_nanosecond(self, result):
        # GH 18898
        # 从 2.0 版本开始（GH 49416），不应接受纳秒参数作为位置参数
        # 准备预期的 Timestamp 对象，加上 Timedelta 后应与 result 相等
        expected = Timestamp(datetime(2000, 1, 2, 3, 4, 5, 6), tz=result.tz)
        expected = expected + Timedelta(nanoseconds=1)
        # 断言 result 与 expected 相等
        assert result == expected

    @pytest.mark.parametrize("z", ["Z0", "Z00"])
    def test_constructor_invalid_Z0_isostring(self, z):
        # GH 8910
        # 准备错误消息，指出无法解析未知的日期时间字符串格式
        msg = f"Unknown datetime string format, unable to parse: 2014-11-02 01:00{z}"
        # 断言在创建 Timestamp 对象时抛出 ValueError，并检查错误消息是否匹配
        with pytest.raises(ValueError, match=msg):
            Timestamp(f"2014-11-02 01:00{z}")
    def test_out_of_bounds_integer_value(self):
        # GH#26651 检查是否引发 OutOfBoundsDatetime 而不是 OverflowError
        # 构造一个超出边界的消息字符串
        msg = str(Timestamp.max._value * 2)
        # 使用 pytest 检查是否引发 OutOfBoundsDatetime，并匹配特定消息
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp(Timestamp.max._value * 2)
        # 构造另一个超出边界的消息字符串
        msg = str(Timestamp.min._value * 2)
        # 使用 pytest 检查是否引发 OutOfBoundsDatetime，并匹配特定消息
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp(Timestamp.min._value * 2)

    def test_out_of_bounds_value(self):
        # 定义一个微秒级的 timedelta
        one_us = np.timedelta64(1).astype("timedelta64[us]")

        # 根据定义，我们不能在 [ns] 范围外，因此将 datetime64 转换为 [us] 以便越界
        min_ts_us = np.datetime64(Timestamp.min).astype("M8[us]") + one_us
        max_ts_us = np.datetime64(Timestamp.max).astype("M8[us]")

        # 对于最小和最大的 datetime 没有错误
        Timestamp(min_ts_us)
        Timestamp(max_ts_us)

        # 在支持非纳秒级之前，我们曾经在这些情况下引发错误
        us_val = NpyDatetimeUnit.NPY_FR_us.value
        assert Timestamp(min_ts_us - one_us)._creso == us_val
        assert Timestamp(max_ts_us + one_us)._creso == us_val

        # https://github.com/numpy/numpy/issues/22346 解释了为什么我们无法像上面的分钟分辨率一样构造

        # too_low 和 too_high 是 M8[s] 范围之外的值
        too_low = np.datetime64("-292277022657-01-27T08:29", "m")
        too_high = np.datetime64("292277026596-12-04T15:31", "m")

        msg = "Out of bounds"
        # 少于最小值一微秒是一个错误
        with pytest.raises(ValueError, match=msg):
            Timestamp(too_low)

        # 多于最大值一微秒是一个错误
        with pytest.raises(ValueError, match=msg):
            Timestamp(too_high)

    def test_out_of_bounds_string(self):
        # 构造一个错误消息的正则表达式模式
        msg = "Cannot cast .* to unit='ns' without overflow"
        # 使用 pytest 检查是否引发 ValueError，并匹配特定消息模式
        with pytest.raises(ValueError, match=msg):
            Timestamp("1676-01-01").as_unit("ns")
        with pytest.raises(ValueError, match=msg):
            Timestamp("2263-01-01").as_unit("ns")

        # 创建 Timestamp 对象并断言其单位为秒
        ts = Timestamp("2263-01-01")
        assert ts.unit == "s"

        # 创建 Timestamp 对象并断言其单位为秒
        ts = Timestamp("1676-01-01")
        assert ts.unit == "s"

    def test_barely_out_of_bounds(self):
        # GH#19529
        # GH#19382 接近边界但删除纳秒会得到一个在边界内的 datetime
        # 构造一个超出边界的消息字符串
        msg = "Out of bounds nanosecond timestamp: 2262-04-11 23:47:16"
        # 使用 pytest 检查是否引发 OutOfBoundsDatetime，并匹配特定消息
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp("2262-04-11 23:47:16.854775808")

    @pytest.mark.skip_ubsan
    def test_bounds_with_different_units(self):
        # 定义超出时间戳边界的日期字符串
        out_of_bounds_dates = ("1677-09-21", "2262-04-12")

        # 定义时间单位
        time_units = ("D", "h", "m", "s", "ms", "us")

        # 遍历超出边界的日期字符串和时间单位
        for date_string in out_of_bounds_dates:
            for unit in time_units:
                # 创建 numpy 的 datetime64 对象
                dt64 = np.datetime64(date_string, unit)
                # 创建 pandas 的 Timestamp 对象
                ts = Timestamp(dt64)
                if unit in ["s", "ms", "us"]:
                    # 如果单位是秒、毫秒或微秒，则保留输入单位
                    assert ts._value == dt64.view("i8")
                else:
                    # 否则选择最接近的支持单位
                    assert ts._creso == NpyDatetimeUnit.NPY_FR_s.value

        # 处理更极端的情况，无法适应秒的精度
        info = np.iinfo(np.int64)
        msg = "Out of bounds second timestamp:"
        for value in [info.min + 1, info.max]:
            for unit in ["D", "h", "m"]:
                # 创建 numpy 的 datetime64 对象
                dt64 = np.datetime64(value, unit)
                # 使用 pytest 断言引发 OutOfBoundsDatetime 异常
                with pytest.raises(OutOfBoundsDatetime, match=msg):
                    Timestamp(dt64)

        # 定义在时间戳边界内的日期字符串
        in_bounds_dates = ("1677-09-23", "2262-04-11")

        # 遍历在时间戳边界内的日期字符串和时间单位
        for date_string in in_bounds_dates:
            for unit in time_units:
                # 创建 numpy 的 datetime64 对象
                dt64 = np.datetime64(date_string, unit)
                # 创建 pandas 的 Timestamp 对象
                Timestamp(dt64)

    @pytest.mark.parametrize("arg", ["001-01-01", "0001-01-01"])
    def test_out_of_bounds_string_consistency(self, arg):
        # GH 15829
        # 定义异常消息
        msg = "Cannot cast 0001-01-01 00:00:00 to unit='ns' without overflow"
        # 使用 pytest 断言引发 OutOfBoundsDatetime 异常，并匹配异常消息
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp(arg).as_unit("ns")

        # 创建 Timestamp 对象
        ts = Timestamp(arg)
        # 断言 Timestamp 对象的单位为秒，并且年、月、日均为1
        assert ts.unit == "s"
        assert ts.year == ts.month == ts.day == 1

    def test_min_valid(self):
        # Ensure that Timestamp.min is a valid Timestamp
        # 确保 Timestamp.min 是一个有效的 Timestamp 对象
        Timestamp(Timestamp.min)

    def test_max_valid(self):
        # Ensure that Timestamp.max is a valid Timestamp
        # 确保 Timestamp.max 是一个有效的 Timestamp 对象
        Timestamp(Timestamp.max)

    @pytest.mark.parametrize("offset", ["+0300", "+0200"])
    def test_construct_timestamp_near_dst(self, offset):
        # GH 20854
        # 创建预期的 Timestamp 对象，带有指定时区偏移量
        expected = Timestamp(f"2016-10-30 03:00:00{offset}", tz="Europe/Helsinki")
        # 转换 Timestamp 对象的时区
        result = Timestamp(expected).tz_convert("Europe/Helsinki")
        # 使用断言验证结果与预期相等
        assert result == expected

    @pytest.mark.parametrize(
        "arg", ["2013/01/01 00:00:00+09:00", "2013-01-01 00:00:00+09:00"]
    )
    def test_construct_with_different_string_format(self, arg):
        # GH 12064
        # 创建 Timestamp 对象，使用不同的日期时间字符串格式
        result = Timestamp(arg)
        # 创建预期的 Timestamp 对象，带有指定时区偏移量
        expected = Timestamp(datetime(2013, 1, 1), tz=timezone(timedelta(hours=9)))
        # 使用断言验证结果与预期相等
        assert result == expected

    @pytest.mark.parametrize("box", [datetime, Timestamp])
    # 测试在传入带有 tzinfo 参数的 datetime 输入时是否会引发异常
    def test_raise_tz_and_tzinfo_in_datetime_input(self, box):
        # 定义一个包含年月日和时区信息的字典
        kwargs = {"year": 2018, "month": 1, "day": 1, "tzinfo": timezone.utc}
        # 异常消息
        msg = "Cannot pass a datetime or Timestamp"
        # 测试使用 box 函数创建 Timestamp 对象时是否会引发 ValueError 异常，并匹配指定消息
        with pytest.raises(ValueError, match=msg):
            Timestamp(box(**kwargs), tz="US/Pacific")
        # 再次测试，确保使用 tzinfo 参数也会引发相同的异常
        with pytest.raises(ValueError, match=msg):
            Timestamp(box(**kwargs), tzinfo=pytz.timezone("US/Pacific"))

    # 测试不将 dateutil utc 转换为默认的 utc
    def test_dont_convert_dateutil_utc_to_default_utc(self):
        # 创建一个带有 datetime 参数的 Timestamp 对象，并指定时区为 tzutc()
        result = Timestamp(datetime(2018, 1, 1), tz=tzutc())
        # 创建另一个 Timestamp 对象，指定 datetime 参数，并使用 tz_localize 方法指定时区为 tzutc()
        expected = Timestamp(datetime(2018, 1, 1)).tz_localize(tzutc())
        # 断言两个 Timestamp 对象相等
        assert result == expected

    # 测试 Timestamp 构造函数能否处理继承自 datetime 的子类
    def test_constructor_subclassed_datetime(self):
        # 创建一个名为 SubDatetime 的 datetime 子类
        class SubDatetime(datetime):
            pass

        # 使用 SubDatetime 类创建一个实例
        data = SubDatetime(2000, 1, 1)
        # 使用 Timestamp 构造函数创建 Timestamp 对象
        result = Timestamp(data)
        # 创建一个预期的 Timestamp 对象
        expected = Timestamp(2000, 1, 1)
        # 断言两个 Timestamp 对象相等
        assert result == expected

    # 测试 Timestamp 构造函数在使用 tz="utc" 参数时的行为
    def test_timestamp_constructor_tz_utc(self):
        # 创建一个使用 "utc" 时区字符串的 Timestamp 对象
        utc_stamp = Timestamp("3/11/2012 05:00", tz="utc")
        # 断言该 Timestamp 对象的时区信息为 timezone.utc
        assert utc_stamp.tzinfo is timezone.utc
        # 断言该 Timestamp 对象的小时为 5

        # 创建一个 Timestamp 对象，并使用 tz_localize 方法指定时区为 "utc"
        utc_stamp = Timestamp("3/11/2012 05:00").tz_localize("utc")
        # 断言该 Timestamp 对象的小时为 5

    # 测试 Timestamp 对象转换为 datetime 对象时的行为，使用 tzoffset 作为时区信息
    def test_timestamp_to_datetime_tzoffset(self):
        # 创建一个具有指定 tzinfo 参数的 Timestamp 对象
        tzinfo = tzoffset(None, 7200)
        expected = Timestamp("3/11/2012 04:00", tz=tzinfo)
        # 将 Timestamp 对象转换为普通的 datetime 对象
        result = Timestamp(expected.to_pydatetime())
        # 断言预期的 Timestamp 对象与转换后的结果对象相等
        assert expected == result
    def test_timestamp_constructor_near_dst_boundary(self):
        # 测试接近夏令时边界的时间戳构造函数行为
        # GH#11481 & GH#15777
        # 解决了使用 tz_convert_from_utc_single 而不是 tz_localize_to_utc 时
        # 对简单字符串时间戳的错误本地化问题

        for tz in ["Europe/Brussels", "Europe/Prague"]:
            # 使用指定时区 tz 创建时间戳 result
            result = Timestamp("2015-10-25 01:00", tz=tz)
            # 创建预期的本地化时间戳 expected
            expected = Timestamp("2015-10-25 01:00").tz_localize(tz)
            # 断言 result 和 expected 相等
            assert result == expected

            # 预期抛出异常的消息
            msg = "Cannot infer dst time from 2015-10-25 02:00:00"
            # 使用 pytest 检查是否抛出了指定异常
            with pytest.raises(pytz.AmbiguousTimeError, match=msg):
                Timestamp("2015-10-25 02:00", tz=tz)

        # 创建一个特定时区下的时间戳 result
        result = Timestamp("2017-03-26 01:00", tz="Europe/Paris")
        # 创建预期的本地化时间戳 expected
        expected = Timestamp("2017-03-26 01:00").tz_localize("Europe/Paris")
        # 断言 result 和 expected 相等
        assert result == expected

        # 预期抛出异常的消息
        msg = "2017-03-26 02:00"
        # 使用 pytest 检查是否抛出了指定异常
        with pytest.raises(pytz.NonExistentTimeError, match=msg):
            Timestamp("2017-03-26 02:00", tz="Europe/Paris")

        # GH#11708
        # 创建一个简单的时间戳 naive
        naive = Timestamp("2015-11-18 10:00:00")
        # 将 naive 本地化为 UTC，然后转换为指定时区 result
        result = naive.tz_localize("UTC").tz_convert("Asia/Kolkata")
        # 创建预期的本地化时间戳 expected
        expected = Timestamp("2015-11-18 15:30:00+0530", tz="Asia/Kolkata")
        # 断言 result 和 expected 相等
        assert result == expected

        # GH#15823
        # 创建一个特定时区下的时间戳 result
        result = Timestamp("2017-03-26 00:00", tz="Europe/Paris")
        # 创建预期的本地化时间戳 expected
        expected = Timestamp("2017-03-26 00:00:00+0100", tz="Europe/Paris")
        # 断言 result 和 expected 相等
        assert result == expected

        # 创建一个特定时区下的时间戳 result
        result = Timestamp("2017-03-26 01:00", tz="Europe/Paris")
        # 创建预期的本地化时间戳 expected
        expected = Timestamp("2017-03-26 01:00:00+0100", tz="Europe/Paris")
        # 断言 result 和 expected 相等
        assert result == expected

        # 预期抛出异常的消息
        msg = "2017-03-26 02:00"
        # 使用 pytest 检查是否抛出了指定异常
        with pytest.raises(pytz.NonExistentTimeError, match=msg):
            Timestamp("2017-03-26 02:00", tz="Europe/Paris")

        # 创建一个特定时区下的时间戳 result
        result = Timestamp("2017-03-26 02:00:00+0100", tz="Europe/Paris")
        # 从 result 获取纳秒单位的时间戳，转换为 naive 时间戳
        naive = Timestamp(result.as_unit("ns")._value)
        # 将 naive 时间戳本地化为 UTC，然后转换为指定时区 result
        expected = naive.tz_localize("UTC").tz_convert("Europe/Paris")
        # 断言 result 和 expected 相等
        assert result == expected

        # 创建一个特定时区下的时间戳 result
        result = Timestamp("2017-03-26 03:00", tz="Europe/Paris")
        # 创建预期的本地化时间戳 expected
        expected = Timestamp("2017-03-26 03:00:00+0200", tz="Europe/Paris")
        # 断言 result 和 expected 相等
        assert result == expected

    @pytest.mark.parametrize(
        "tz",
        [
            "pytz/US/Eastern",
            gettz("US/Eastern"),
            "US/Eastern",
            "dateutil/US/Eastern",
        ],
    )
    def test_timestamp_constructed_by_date_and_tz(self, tz):
        # 测试根据日期和时区构造 Timestamp 的行为
        # GH#2993, Timestamp 无法正确地通过 datetime.date 和时区构造

        # 如果 tz 是以 "pytz/" 开头的字符串，导入 pytest 的 pytz 并跳过该测试
        if isinstance(tz, str) and tz.startswith("pytz/"):
            pytz = pytest.importorskip("pytz")
            tz = pytz.timezone(tz.removeprefix("pytz/"))
        # 使用指定日期和时区构造时间戳 result
        result = Timestamp(date(2012, 3, 11), tz=tz)

        # 创建预期的时间戳 expected
        expected = Timestamp("3/11/2012", tz=tz)
        # 断言 result 的小时与 expected 的小时相等
        assert result.hour == expected.hour
        # 断言 result 和 expected 相等
        assert result == expected
    def test_explicit_tz_none(self):
        # 定义测试方法：test_explicit_tz_none，测试针对时区为 None 的情况
        # GH#48688：GitHub 上的 issue 编号 48688，此测试解决相关问题
        msg = "Passed data is timezone-aware, incompatible with 'tz=None'"
        # 使用 pytest 断言，验证传入时区为 None 时抛出 ValueError 异常，并匹配特定错误信息
        with pytest.raises(ValueError, match=msg):
            # 创建 Timestamp 对象，传入具有时区信息的 datetime 对象，但同时指定 tz=None，应当抛出异常
            Timestamp(datetime(2022, 1, 1, tzinfo=timezone.utc), tz=None)

        with pytest.raises(ValueError, match=msg):
            # 创建 Timestamp 对象，传入带有时区信息的字符串，但同时指定 tz=None，应当抛出异常
            Timestamp("2022-01-01 00:00:00", tzinfo=timezone.utc, tz=None)

        with pytest.raises(ValueError, match=msg):
            # 创建 Timestamp 对象，传入带有时区信息的字符串（带有偏移量），但同时指定 tz=None，应当抛出异常
            Timestamp("2022-01-01 00:00:00-0400", tz=None)
def test_constructor_ambiguous_dst():
    # GH 24329
    # 确保调用 Timestamp 构造函数
    # 在模糊时间戳上不改变 Timestamp.value
    ts = Timestamp(1382835600000000000, tz="dateutil/Europe/London")
    expected = ts._value
    result = Timestamp(ts)._value
    assert result == expected


@pytest.mark.parametrize("epoch", [1552211999999999872, 1552211999999999999])
def test_constructor_before_dst_switch(epoch):
    # GH 31043
    # 确保在 DST 切换前调用 Timestamp 构造函数
    # 不会导致不存在的时间或值的变化
    ts = Timestamp(epoch, tz="dateutil/America/Los_Angeles")
    result = ts.tz.dst(ts)
    expected = timedelta(seconds=0)
    assert Timestamp(ts)._value == epoch
    assert result == expected


def test_timestamp_constructor_identity():
    # Test for #30543
    # 测试 Timestamp 构造函数的身份特性
    expected = Timestamp("2017-01-01T12")
    result = Timestamp(expected)
    assert result is expected


@pytest.mark.parametrize("nano", [-1, 1000])
def test_timestamp_nano_range(nano):
    # GH 48255
    # 使用 pytest 确保纳秒在有效范围内
    with pytest.raises(ValueError, match="nanosecond must be in 0..999"):
        Timestamp(year=2022, month=1, day=1, nanosecond=nano)


def test_non_nano_value():
    # https://github.com/pandas-dev/pandas/issues/49076
    # 检查非纳秒值的 Timestamp 行为
    result = Timestamp("1800-01-01", unit="s").value
    # `.value` 显示的是纳秒，尽管单位是 's'
    assert result == -5364662400000000000

    # 超出纳秒范围的 `.value` 会抛出信息详尽的错误信息
    msg = (
        r"Cannot convert Timestamp to nanoseconds without overflow. "
        r"Use `.asm8.view\('i8'\)` to cast represent Timestamp in its "
        r"own unit \(here, s\).$"
    )
    ts = Timestamp("0300-01-01")
    with pytest.raises(OverflowError, match=msg):
        ts.value
    # 检查建议的解决方法确实有效
    result = ts.asm8.view("i8")
    assert result == -52700112000


@pytest.mark.parametrize("na_value", [None, np.nan, np.datetime64("NaT"), NaT, NA])
def test_timestamp_constructor_na_value(na_value):
    # GH45481
    # 测试 Timestamp 构造函数对于 NA 值的处理
    result = Timestamp(na_value)
    expected = NaT
    assert result is expected
```