# `D:\src\scipysrc\pandas\pandas\tests\scalar\period\test_period.py`

```
# 导入所需的模块和类
from datetime import (
    date,
    datetime,
    timedelta,
)
import re  # 导入正则表达式模块

import numpy as np  # 导入 NumPy 库
import pytest  # 导入 pytest 测试框架

# 导入 pandas 库中的一些特定模块和类
from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.ccalendar import (
    DAYS,
    MONTHS,
)
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas._libs.tslibs.parsing import DateParseError
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG

# 导入 pandas 中的一些核心类和函数
from pandas import (
    NaT,
    Period,
    Timedelta,
    Timestamp,
    offsets,
)
import pandas._testing as tm  # 导入 pandas 测试模块

bday_msg = "Period with BDay freq is deprecated"  # 定义一个变量 bday_msg

# 定义一个测试类 TestPeriodDisallowedFreqs，用于测试不支持的频率
class TestPeriodDisallowedFreqs:
    
    # 使用 pytest.mark.parametrize 标记的测试方法，测试不支持的频率
    @pytest.mark.parametrize(
        "freq, freq_msg",
        [
            (offsets.BYearBegin(), "BYearBegin"),
            (offsets.YearBegin(2), "YearBegin"),
            (offsets.QuarterBegin(startingMonth=12), "QuarterBegin"),
            (offsets.BusinessMonthEnd(2), "BusinessMonthEnd"),
        ],
    )
    def test_offsets_not_supported(self, freq, freq_msg):
        # GH#55785
        # 构造匹配的异常消息，以验证是否抛出预期的 ValueError 异常
        msg = re.escape(f"{freq} is not supported as period frequency")
        with pytest.raises(ValueError, match=msg):
            Period(year=2014, freq=freq)

    # 测试自定义工作日频率是否引发异常
    def test_custom_business_day_freq_raises(self):
        # GH#52534
        msg = "C is not supported as period frequency"
        with pytest.raises(ValueError, match=msg):
            Period("2023-04-10", freq="C")
        msg = f"{offsets.CustomBusinessDay().base} is not supported as period frequency"
        with pytest.raises(ValueError, match=msg):
            Period("2023-04-10", freq=offsets.CustomBusinessDay())

    # 测试无效频率错误消息
    def test_invalid_frequency_error_message(self):
        msg = "WOM-1MON is not supported as period frequency"
        with pytest.raises(ValueError, match=msg):
            Period("2012-01-02", freq="WOM-1MON")

    # 测试无效周期频率错误消息
    def test_invalid_frequency_period_error_message(self):
        msg = "Invalid frequency: ME"
        with pytest.raises(ValueError, match=msg):
            Period("2012-01-02", freq="ME")


# 定义一个测试类 TestPeriodConstruction，用于测试周期对象的构造
class TestPeriodConstruction:
    
    # 测试从 np.datetime64('NaT') 引发异常的情况
    def test_from_td64nat_raises(self):
        # GH#44507
        td = NaT.to_numpy("m8[ns]")

        # 构造匹配的异常消息，以验证是否抛出预期的 ValueError 异常
        msg = "Value must be Period, string, integer, or datetime"
        with pytest.raises(ValueError, match=msg):
            Period(td)

        with pytest.raises(ValueError, match=msg):
            Period(td, freq="D")
    def test_construction(self):
        # 创建两个 Period 对象，指定不同的频率，但指向相同时间点，断言它们相等
        i1 = Period("1/1/2005", freq="M")
        i2 = Period("Jan 2005")

        assert i1 == i2

        # GH#54105 - Period 可以混淆地使用小写频率参数进行实例化
        # TODO: 在将来传递小写频率参数时引发错误
        i1 = Period("2005", freq="Y")
        i2 = Period("2005")

        assert i1 == i2

        # 创建另一个 Period 对象，指定不同的频率，断言它们不相等
        i4 = Period("2005", freq="M")
        assert i1 != i4

        # 使用 now 方法创建 Period 对象，指定频率为季度（Q）
        i1 = Period.now(freq="Q")
        i2 = Period(datetime.now(), freq="Q")

        assert i1 == i2

        # 作为测试有时将频率作为关键字参数传递
        # https://github.com/pandas-dev/pandas/issues/53369
        i1 = Period.now(freq="D")
        i2 = Period(datetime.now(), freq="D")
        i3 = Period.now(offsets.Day())

        assert i1 == i2
        assert i1 == i3

        # 使用过时的频率 'min' 创建 Period 对象，并验证警告消息
        i1 = Period("1982", freq="min")
        msg = "'MIN' is deprecated and will be removed in a future version."
        with tm.assert_produces_warning(FutureWarning, match=msg):
            i2 = Period("1982", freq="MIN")
        assert i1 == i2

        # 使用年、月、日等参数创建 Period 对象，并断言相等
        i1 = Period(year=2005, month=3, day=1, freq="D")
        i2 = Period("3/1/2005", freq="D")
        assert i1 == i2

        # 使用小写频率 'd' 创建 Period 对象，并断言相等
        i3 = Period(year=2005, month=3, day=1, freq="d")
        assert i1 == i3

        # 使用带有毫秒的时间字符串创建 Period 对象，并断言相等
        i1 = Period("2007-01-01 09:00:00.001")
        expected = Period(datetime(2007, 1, 1, 9, 0, 0, 1000), freq="ms")
        assert i1 == expected

        expected = Period("2007-01-01 09:00:00.001", freq="ms")
        assert i1 == expected

        # 使用带有微秒的时间字符串创建 Period 对象，并断言相等
        i1 = Period("2007-01-01 09:00:00.00101")
        expected = Period(datetime(2007, 1, 1, 9, 0, 0, 1010), freq="us")
        assert i1 == expected

        expected = Period("2007-01-01 09:00:00.00101", freq="us")
        assert i1 == expected

        # 测试缺少 ordinal 参数时的 ValueError 异常
        msg = "Must supply freq for ordinal value"
        with pytest.raises(ValueError, match=msg):
            Period(ordinal=200701)

        # 测试传递无效频率 'X' 时的 ValueError 异常
        msg = "Invalid frequency: X"
        with pytest.raises(ValueError, match=msg):
            Period("2007-1-1", freq="X")
    def test_construction_bday(self):
        # Biz day construction, roll forward if non-weekday
        # 测试工作日的构造，如果日期非工作日则向前滚动
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            # 创建周期对象 i1 和 i2，频率为工作日
            i1 = Period("3/10/12", freq="B")
            i2 = Period("3/10/12", freq="D")
            # 将 i2 转换为工作日频率，并与 i1 进行比较
            assert i1 == i2.asfreq("B")
            i2 = Period("3/11/12", freq="D")
            assert i1 == i2.asfreq("B")
            i2 = Period("3/12/12", freq="D")
            assert i1 == i2.asfreq("B")

            # 创建周期对象 i3，频率为小写的工作日
            i3 = Period("3/10/12", freq="b")
            assert i1 == i3

            # 重新赋值 i1 和 i2，确保它们相等
            i1 = Period(year=2012, month=3, day=10, freq="B")
            i2 = Period("3/12/12", freq="B")
            assert i1 == i2

    def test_construction_quarter(self):
        # 创建季度周期对象 i1 和 i2
        i1 = Period(year=2005, quarter=1, freq="Q")
        i2 = Period("1/1/2005", freq="Q")
        assert i1 == i2

        # 创建季度周期对象 i1 和 i2，确保它们相等
        i1 = Period(year=2005, quarter=3, freq="Q")
        i2 = Period("9/1/2005", freq="Q")
        assert i1 == i2

        # 创建多个表示相同季度的周期对象，确保它们相等
        i1 = Period("2005Q1")
        i2 = Period(year=2005, quarter=1, freq="Q")
        i3 = Period("2005q1")
        assert i1 == i2
        assert i1 == i3

        i1 = Period("05Q1")
        assert i1 == i2
        lower = Period("05q1")
        assert i1 == lower

        i1 = Period("1Q2005")
        assert i1 == i2
        lower = Period("1q2005")
        assert i1 == lower

        i1 = Period("1Q05")
        assert i1 == i2
        lower = Period("1q05")
        assert i1 == lower

        i1 = Period("4Q1984")
        assert i1.year == 1984
        lower = Period("4q1984")
        assert i1 == lower

    def test_construction_month(self):
        # 创建月度周期对象 i1 和 expected，确保它们相等
        expected = Period("2007-01", freq="M")
        i1 = Period("200701", freq="M")
        assert i1 == expected

        i1 = Period("200701", freq="M")
        assert i1 == expected

        i1 = Period(200701, freq="M")
        assert i1 == expected

        # 使用 ordinal 参数创建周期对象 i1，验证其年份
        i1 = Period(ordinal=200701, freq="M")
        assert i1.year == 18695

        # 创建不同类型的日期时间对象，并确保它们与 i1 相等
        i1 = Period(datetime(2007, 1, 1), freq="M")
        i2 = Period("200701", freq="M")
        assert i1 == i2

        i1 = Period(date(2007, 1, 1), freq="M")
        i2 = Period(datetime(2007, 1, 1), freq="M")
        i3 = Period(np.datetime64("2007-01-01"), freq="M")
        i4 = Period("2007-01-01 00:00:00", freq="M")
        i5 = Period("2007-01-01 00:00:00.000", freq="M")
        assert i1 == i2
        assert i1 == i3
        assert i1 == i4
        assert i1 == i5
    # 测试 Period 类的构造函数中的偏移量
    def test_period_constructor_offsets(self):
        # 断言使用 offsets.MonthEnd() 频率创建的 Period 对象与使用 "M" 频率创建的 Period 对象相等
        assert Period("1/1/2005", freq=offsets.MonthEnd()) == Period(
            "1/1/2005", freq="M"
        )
        # 断言使用 offsets.YearEnd() 频率创建的 Period 对象与使用 "Y" 频率创建的 Period 对象相等
        assert Period("2005", freq=offsets.YearEnd()) == Period("2005", freq="Y")
        # 断言使用 offsets.MonthEnd() 频率创建的 Period 对象与使用 "M" 频率创建的 Period 对象相等
        assert Period("2005", freq=offsets.MonthEnd()) == Period("2005", freq="M")
        # 断言使用 offsets.BusinessDay() 频率创建的 Period 对象与使用 "B" 频率创建的 Period 对象相等，并产生 FutureWarning 警告
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            assert Period("3/10/12", freq=offsets.BusinessDay()) == Period(
                "3/10/12", freq="B"
            )
        # 断言使用 offsets.Day() 频率创建的 Period 对象与使用 "D" 频率创建的 Period 对象相等
        assert Period("3/10/12", freq=offsets.Day()) == Period("3/10/12", freq="D")

        # 断言使用 offsets.QuarterEnd(startingMonth=12) 频率创建的 Period 对象与使用 "Q" 频率创建的 Period 对象相等
        assert Period(
            year=2005, quarter=1, freq=offsets.QuarterEnd(startingMonth=12)
        ) == Period(year=2005, quarter=1, freq="Q")
        assert Period(
            year=2005, quarter=2, freq=offsets.QuarterEnd(startingMonth=12)
        ) == Period(year=2005, quarter=2, freq="Q")

        # 断言使用 offsets.Day() 频率创建的 Period 对象与使用 "D" 频率创建的 Period 对象相等
        assert Period(year=2005, month=3, day=1, freq=offsets.Day()) == Period(
            year=2005, month=3, day=1, freq="D"
        )
        # 断言使用 offsets.BDay() 频率创建的 Period 对象与使用 "B" 频率创建的 Period 对象相等，并产生 FutureWarning 警告
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            assert Period(year=2012, month=3, day=10, freq=offsets.BDay()) == Period(
                year=2012, month=3, day=10, freq="B"
            )

        # 断言使用 offsets.Day(3) 频率创建的 Period 对象与指定日期相等
        expected = Period("2005-03-01", freq="3D")
        assert Period(year=2005, month=3, day=1, freq=offsets.Day(3)) == expected
        assert Period(year=2005, month=3, day=1, freq="3D") == expected

        # 断言使用 offsets.BDay(3) 频率创建的 Period 对象与指定日期相等，并产生 FutureWarning 警告
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            assert Period(year=2012, month=3, day=10, freq=offsets.BDay(3)) == Period(
                year=2012, month=3, day=10, freq="3B"
            )

        # 断言使用 offsets.MonthEnd() 频率创建的 Period 对象与指定日期相等
        assert Period(200701, freq=offsets.MonthEnd()) == Period(200701, freq="M")

        # 创建两个 Period 对象，比较它们的相等性和年份
        i1 = Period(ordinal=200701, freq=offsets.MonthEnd())
        i2 = Period(ordinal=200701, freq="M")
        assert i1 == i2
        assert i1.year == 18695
        assert i2.year == 18695

        # 创建两个 Period 对象，比较它们的相等性
        i1 = Period(datetime(2007, 1, 1), freq="M")
        i2 = Period("200701", freq="M")
        assert i1 == i2

        # 创建多个不同类型的日期对象，比较它们的相等性
        i1 = Period(date(2007, 1, 1), freq="M")
        i2 = Period(datetime(2007, 1, 1), freq="M")
        i3 = Period(np.datetime64("2007-01-01"), freq="M")
        i4 = Period("2007-01-01 00:00:00", freq="M")
        i5 = Period("2007-01-01 00:00:00.000", freq="M")
        assert i1 == i2
        assert i1 == i3
        assert i1 == i4
        assert i1 == i5

        # 创建一个 Period 对象，比较它与预期的 Period 对象的相等性
        i1 = Period("2007-01-01 09:00:00.001")
        expected = Period(datetime(2007, 1, 1, 9, 0, 0, 1000), freq="ms")
        assert i1 == expected

        # 创建一个 Period 对象，比较它与预期的 Period 对象的相等性
        expected = Period("2007-01-01 09:00:00.001", freq="ms")
        assert i1 == expected

        # 创建一个 Period 对象，比较它与预期的 Period 对象的相等性
        i1 = Period("2007-01-01 09:00:00.00101")
        expected = Period(datetime(2007, 1, 1, 9, 0, 0, 1010), freq="us")
        assert i1 == expected

        # 创建一个 Period 对象，比较它与预期的 Period 对象的相等性
        expected = Period("2007-01-01 09:00:00.00101", freq="us")
        assert i1 == expected
    def test_invalid_arguments(self):
        # 检查传入 Period 对象构造函数的无效参数情况

        msg = "Must supply freq for datetime value"
        # 预期抛出 ValueError 异常，且异常消息应包含指定的错误信息
        with pytest.raises(ValueError, match=msg):
            Period(datetime.now())
        # 同上，但是传入的是 datetime 对象的日期部分
        with pytest.raises(ValueError, match=msg):
            Period(datetime.now().date())

        msg = "Value must be Period, string, integer, or datetime"
        # 预期抛出 ValueError 异常，且异常消息应包含指定的错误信息
        with pytest.raises(ValueError, match=msg):
            Period(1.6, freq="D")
        msg = "Ordinal must be an integer"
        # 预期抛出 ValueError 异常，且异常消息应包含指定的错误信息
        with pytest.raises(ValueError, match=msg):
            Period(ordinal=1.6, freq="D")
        msg = "Only value or ordinal but not both should be given but not both"
        # 预期抛出 ValueError 异常，且异常消息应包含指定的错误信息
        with pytest.raises(ValueError, match=msg):
            Period(ordinal=2, value=1, freq="D")

        msg = "If value is None, freq cannot be None"
        # 预期抛出 ValueError 异常，且异常消息应包含指定的错误信息
        with pytest.raises(ValueError, match=msg):
            Period(month=1)

        msg = '^Given date string "-2000" not likely a datetime$'
        # 预期抛出 ValueError 异常，且异常消息应包含指定的错误信息
        with pytest.raises(ValueError, match=msg):
            Period("-2000", "Y")
        msg = "day is out of range for month"
        # 预期抛出 DateParseError 异常，且异常消息应包含指定的错误信息
        with pytest.raises(DateParseError, match=msg):
            Period("0", "Y")
        msg = "Unknown datetime string format, unable to parse"
        # 预期抛出 DateParseError 异常，且异常消息应包含指定的错误信息
        with pytest.raises(DateParseError, match=msg):
            Period("1/1/-2000", "Y")

    def test_constructor_corner(self):
        # 测试 Period 对象构造函数的边界情况

        expected = Period("2007-01", freq="2M")
        # 断言构造的 Period 对象与预期的对象相等
        assert Period(year=2007, month=1, freq="2M") == expected

        # 断言构造的 Period 对象是 NaT（Not a Time）
        assert Period(None) is NaT

        p = Period("2007-01-01", freq="D")

        result = Period(p, freq="Y")
        exp = Period("2007", freq="Y")
        # 断言构造的 Period 对象与预期的对象相等
        assert result == exp

    def test_constructor_infer_freq(self):
        # 测试 Period 对象构造函数自动推断频率的情况

        p = Period("2007-01-01")
        # 断言构造的 Period 对象的频率是 "D"（天）
        assert p.freq == "D"

        p = Period("2007-01-01 07")
        # 断言构造的 Period 对象的频率是 "h"（小时）
        assert p.freq == "h"

        p = Period("2007-01-01 07:10")
        # 断言构造的 Period 对象的频率是 "min"（分钟）
        assert p.freq == "min"

        p = Period("2007-01-01 07:10:15")
        # 断言构造的 Period 对象的频率是 "s"（秒）
        assert p.freq == "s"

        p = Period("2007-01-01 07:10:15.123")
        # 断言构造的 Period 对象的频率是 "ms"（毫秒）
        assert p.freq == "ms"

        # 尽管小数部分都是零，但因为有6位小数，所以推断频率是 "us"（微秒）
        p = Period("2007-01-01 07:10:15.123000")
        assert p.freq == "us"

        p = Period("2007-01-01 07:10:15.123400")
        # 断言构造的 Period 对象的频率是 "us"（微秒）
        assert p.freq == "us"

    def test_multiples(self):
        # 测试 Period 对象的算术操作及频率处理

        result1 = Period("1989", freq="2Y")
        result2 = Period("1989", freq="Y")
        # 断言两个 Period 对象的 ordinal 属性相同
        assert result1.ordinal == result2.ordinal
        # 断言两个 Period 对象的 freqstr 属性分别为 "2Y-DEC" 和 "Y-DEC"
        assert result1.freqstr == "2Y-DEC"
        assert result2.freqstr == "Y-DEC"
        # 断言两个 Period 对象的 freq 属性分别为 offsets.YearEnd(2) 和 offsets.YearEnd()
        assert result1.freq == offsets.YearEnd(2)
        assert result2.freq == offsets.YearEnd()

        # 断言 Period 对象与整数的加法操作
        assert (result1 + 1).ordinal == result1.ordinal + 2
        assert (1 + result1).ordinal == result1.ordinal + 2
        # 断言 Period 对象与整数的减法操作
        assert (result1 - 1).ordinal == result2.ordinal - 2
        assert (-1 + result1).ordinal == result2.ordinal - 2

    @pytest.mark.parametrize("month", MONTHS)
    def test_period_cons_quarterly(self, month):
        # 在 scikits.timeseries 中存在的 bug
        # 根据给定的月份构造频率字符串
        freq = f"Q-{month}"
        # 创建一个期间对象，表示1989年第三季度，使用给定的频率
        exp = Period("1989Q3", freq=freq)
        # 断言字符串 "1989Q3" 在期间对象的字符串表示中
        assert "1989Q3" in str(exp)
        # 将期间对象转换为时间戳，精确到日，取结束日期
        stamp = exp.to_timestamp("D", how="end")
        # 根据时间戳创建新的期间对象，使用相同的频率
        p = Period(stamp, freq=freq)
        # 断言新创建的期间对象与期望的期间对象相等
        assert p == exp

        # 将期间对象转换为时间戳，精确到3天后的结束日期
        stamp = exp.to_timestamp("3D", how="end")
        # 根据时间戳创建新的期间对象，使用相同的频率
        p = Period(stamp, freq=freq)
        # 断言新创建的期间对象与期望的期间对象相等
        assert p == exp

    @pytest.mark.parametrize("month", MONTHS)
    def test_period_cons_annual(self, month):
        # 在 scikits.timeseries 中存在的 bug
        # 根据给定的月份构造频率字符串
        freq = f"Y-{month}"
        # 创建一个期间对象，表示1989年，使用给定的频率
        exp = Period("1989", freq=freq)
        # 将期间对象转换为时间戳，精确到日，加30天
        stamp = exp.to_timestamp("D", how="end") + timedelta(days=30)
        # 根据时间戳创建新的期间对象，使用相同的频率
        p = Period(stamp, freq=freq)

        # 断言新创建的期间对象与期望的期间对象相差1天
        assert p == exp + 1
        # 断言新创建的对象是期间对象的实例
        assert isinstance(p, Period)

    @pytest.mark.parametrize("day", DAYS)
    @pytest.mark.parametrize("num", range(10, 17))
    def test_period_cons_weekly(self, num, day):
        # 根据给定的日期构造日期字符串
        daystr = f"2011-02-{num}"
        # 根据给定的星期几构造频率字符串
        freq = f"W-{day}"

        # 创建一个期间对象，表示给定日期的特定星期几，使用给定的频率
        result = Period(daystr, freq=freq)
        # 创建一个期间对象，表示给定日期的具体日期，然后按照给定频率重新取样
        expected = Period(daystr, freq="D").asfreq(freq)
        # 断言两个期间对象相等
        assert result == expected
        # 断言结果对象是期间对象的实例
        assert isinstance(result, Period)

    def test_parse_week_str_roundstrip(self):
        # GH#50803
        # 创建一个期间对象，表示给定日期范围，自动识别频率为每周的星期日为结束日
        per = Period("2017-01-23/2017-01-29")
        # 断言期间对象的频率字符串为 "W-SUN"
        assert per.freq.freqstr == "W-SUN"

        # 创建一个期间对象，表示给定日期范围，自动识别频率为每周的星期一为结束日
        per = Period("2017-01-24/2017-01-30")
        # 断言期间对象的频率字符串为 "W-MON"
        assert per.freq.freqstr == "W-MON"

        # 构造错误信息
        msg = "Could not parse as weekly-freq Period"
        # 使用 pytest 的断言捕获异常，匹配错误信息
        with pytest.raises(ValueError, match=msg):
            # 创建一个期间对象，表示错误的日期范围，跨越不到6天
            Period("2016-01-23/2017-01-29")

    def test_period_from_ordinal(self):
        # 创建一个期间对象，表示给定年月，使用月份为频率
        p = Period("2011-01", freq="M")
        # 根据序数值和频率创建期间对象
        res = Period._from_ordinal(p.ordinal, freq=p.freq)
        # 断言两个期间对象相等
        assert p == res
        # 断言结果对象是期间对象的实例
        assert isinstance(res, Period)

    @pytest.mark.parametrize("freq", ["Y", "M", "D", "h"])
    def test_construct_from_nat_string_and_freq(self, freq):
        # 创建一个期间对象，表示 "NaT"（Not a Time），使用给定的频率
        per = Period("NaT", freq=freq)
        # 断言期间对象是 NaT
        assert per is NaT

        # 创建一个期间对象，表示 "NaT"，使用给定的频率
        per = Period("NaT", freq="2" + freq)
        # 断言期间对象是 NaT
        assert per is NaT

        # 创建一个期间对象，表示 "NaT"，使用给定的频率
        per = Period("NaT", freq="3" + freq)
        # 断言期间对象是 NaT
        assert per is NaT

    def test_period_cons_nat(self):
        # 创建一个期间对象，表示 NaT（Not a Time），使用给定的频率
        p = Period("nat", freq="W-SUN")
        # 断言期间对象是 NaT
        assert p is NaT

        # 创建一个期间对象，表示 NaT，使用给定的频率
        p = Period(iNaT, freq="D")
        # 断言期间对象是 NaT
        assert p is NaT

        # 创建一个期间对象，表示 NaT，使用给定的频率
        p = Period(iNaT, freq="3D")
        # 断言期间对象是 NaT
        assert p is NaT

        # 创建一个期间对象，表示 NaT，使用给定的频率
        p = Period(iNaT, freq="1D1h")
        # 断言期间对象是 NaT
        assert p is NaT

        # 创建一个期间对象，表示 "NaT"（Not a Time）
        p = Period("NaT")
        # 断言期间对象是 NaT
        assert p is NaT

        # 创建一个期间对象，表示 NaT
        p = Period(iNaT)
        # 断言期间对象是 NaT
        assert p is NaT
    # 测试 Period 类的一些常见操作和属性
    def test_period_cons_mult(self):
        # 创建两个 Period 对象，分别表示不同频率的时间段，断言它们的序数相同
        p1 = Period("2011-01", freq="3M")
        p2 = Period("2011-01", freq="M")
        assert p1.ordinal == p2.ordinal

        # 断言 p1 的频率为每三个月的月末
        assert p1.freq == offsets.MonthEnd(3)
        assert p1.freqstr == "3M"

        # 断言 p2 的频率为每月月末
        assert p2.freq == offsets.MonthEnd()
        assert p2.freqstr == "M"

        # 对 p1 进行加法操作，断言结果的序数与 p2 加三个月的序数相同
        result = p1 + 1
        assert result.ordinal == (p2 + 3).ordinal

        # 断言加法结果的频率与 p1 相同，并且频率字符串为 "3M"
        assert result.freq == p1.freq
        assert result.freqstr == "3M"

        # 对 p1 进行减法操作，断言结果的序数与 p2 减三个月的序数相同
        result = p1 - 1
        assert result.ordinal == (p2 - 3).ordinal

        # 断言减法结果的频率与 p1 相同，并且频率字符串为 "3M"
        assert result.freq == p1.freq
        assert result.freqstr == "3M"

        # 测试异常情况：尝试创建一个频率为负数的 Period 对象，断言会抛出 ValueError 异常
        msg = "Frequency must be positive, because it represents span: -3M"
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="-3M")

        # 测试异常情况：尝试创建一个频率为零的 Period 对象，断言会抛出 ValueError 异常
        msg = "Frequency must be positive, because it represents span: 0M"
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="0M")
    # 定义一个测试方法，用于测试 Period 对象的组合频率情况
    def test_period_cons_combined(self):
        # 准备测试数据，包含不同的 Period 对象和频率组合
        p = [
            (
                Period("2011-01", freq="1D1h"),    # 创建一个日期时间段，每天1小时
                Period("2011-01", freq="1h1D"),    # 创建一个日期时间段，每小时1天
                Period("2011-01", freq="h"),       # 创建一个每小时的日期时间段
            ),
            (
                Period(ordinal=1, freq="1D1h"),    # 创建一个序号为1的日期时间段，每天1小时
                Period(ordinal=1, freq="1h1D"),    # 创建一个序号为1的日期时间段，每小时1天
                Period(ordinal=1, freq="h"),       # 创建一个序号为1的每小时的日期时间段
            ),
        ]

        # 遍历测试数据
        for p1, p2, p3 in p:
            # 断言各个 Period 对象的序号相同
            assert p1.ordinal == p3.ordinal
            assert p2.ordinal == p3.ordinal

            # 断言 p1 的频率为 25 小时
            assert p1.freq == offsets.Hour(25)
            assert p1.freqstr == "25h"

            # 断言 p2 的频率为 25 小时
            assert p2.freq == offsets.Hour(25)
            assert p2.freqstr == "25h"

            # 断言 p3 的频率为默认的每小时
            assert p3.freq == offsets.Hour()
            assert p3.freqstr == "h"

            # 对 p1 和 p2 分别进行加法操作
            result = p1 + 1
            assert result.ordinal == (p3 + 25).ordinal
            assert result.freq == p1.freq
            assert result.freqstr == "25h"

            result = p2 + 1
            assert result.ordinal == (p3 + 25).ordinal
            assert result.freq == p2.freq
            assert result.freqstr == "25h"

            # 对 p1 和 p2 分别进行减法操作
            result = p1 - 1
            assert result.ordinal == (p3 - 25).ordinal
            assert result.freq == p1.freq
            assert result.freqstr == "25h"

            result = p2 - 1
            assert result.ordinal == (p3 - 25).ordinal
            assert result.freq == p2.freq
            assert result.freqstr == "25h"

        # 测试异常情况：频率为负数的情况
        msg = "Frequency must be positive, because it represents span: -25h"
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="-1D1h")
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="-1h1D")
        with pytest.raises(ValueError, match=msg):
            Period(ordinal=1, freq="-1D1h")
        with pytest.raises(ValueError, match=msg):
            Period(ordinal=1, freq="-1h1D")

        # 测试异常情况：频率为零的情况
        msg = "Frequency must be positive, because it represents span: 0D"
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="0D0h")
        with pytest.raises(ValueError, match=msg):
            Period(ordinal=1, freq="0D0h")

        # 测试异常情况：组合了无效的频率
        msg = "Invalid frequency: 1W1D"
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="1W1D")
        msg = "Invalid frequency: 1D1W"
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="1D1W")
    @pytest.mark.parametrize(
        "sec_float, expected",
        [  # 参数化测试用例，测试不同的秒数字符串和期望的结果
            (".000000001", 1),            # 测试秒数 ".000000001" 对应的期望结果为 1
            (".000000999", 999),          # 测试秒数 ".000000999" 对应的期望结果为 999
            (".123456789", 789),          # 测试秒数 ".123456789" 对应的期望结果为 789
            (".999999999", 999),          # 测试秒数 ".999999999" 对应的期望结果为 999
            (".999999000", 0),            # 测试秒数 ".999999000" 对应的期望结果为 0
            # 测试飞秒、阿托秒和皮秒等较小的时间单位会像时间戳一样被丢弃
            (".999999001123", 1),         # 测试秒数 ".999999001123" 对应的期望结果为 1
            (".999999001123456", 1),      # 测试秒数 ".999999001123456" 对应的期望结果为 1
            (".999999001123456789", 1),   # 测试秒数 ".999999001123456789" 对应的期望结果为 1
        ],
    )
    def test_period_constructor_nanosecond(self, day, hour, sec_float, expected):
        # GH 34621
        # 测试时间段构造函数，验证秒数的浮点表示对应的纳秒部分是否符合预期

        assert Period(day + hour + sec_float).start_time.nanosecond == expected
        # 断言：构造一个时间段对象，验证其起始时间的纳秒部分是否等于期望的值

    @pytest.mark.parametrize("hour", range(24))
    def test_period_large_ordinal(self, hour):
        # Issue #36430
        # 测试大序数的时间段，验证在最大时间戳以上的整数溢出情况
        p = Period(ordinal=2562048 + hour, freq="1h")
        # 创建一个时间段对象，其序数为 2562048 + hour，频率为每小时一次

        assert p.hour == hour
        # 断言：验证该时间段对象的小时部分是否等于输入的小时值
class TestPeriodMethods:
    def test_round_trip(self):
        # 创建一个 Period 对象，表示时间周期为 "2000Q1"
        p = Period("2000Q1")
        # 使用自定义的方法 round_trip_pickle 序列化和反序列化 Period 对象 p
        new_p = tm.round_trip_pickle(p)
        # 断言新对象 new_p 应与原始对象 p 相等
        assert new_p == p

    def test_hash(self):
        # 断言两个相同时间点的 Period 对象具有相同的哈希值
        assert hash(Period("2011-01", freq="M")) == hash(Period("2011-01", freq="M"))

        # 断言具有不同时间精度的 Period 对象有不同的哈希值
        assert hash(Period("2011-01-01", freq="D")) != hash(Period("2011-01", freq="M"))

        # 断言具有不同频率的 Period 对象有不同的哈希值
        assert hash(Period("2011-01", freq="3M")) != hash(Period("2011-01", freq="2M"))

        # 断言不同时间点的同频率 Period 对象有不同的哈希值
        assert hash(Period("2011-01", freq="M")) != hash(Period("2011-02", freq="M"))

    # --------------------------------------------------------------
    # to_timestamp

    def test_to_timestamp_mult(self):
        # 创建一个频率为月的 Period 对象，表示 "2011-01"
        p = Period("2011-01", freq="M")
        # 断言将 Period 对象 p 转换为 Timestamp，按 'S' 模式应返回 Timestamp("2011-01-01")
        assert p.to_timestamp(how="S") == Timestamp("2011-01-01")
        # 期望结果是 Timestamp("2011-02-01")，减去 1 纳秒，因为 'E' 模式返回最后一刻该周期的时间戳
        expected = Timestamp("2011-02-01") - Timedelta(1, "ns")
        assert p.to_timestamp(how="E") == expected

        # 创建一个频率为季度的 Period 对象，表示 "2011Q1"
        p = Period("2011-01", freq="3M")
        # 断言将 Period 对象 p 转换为 Timestamp，按 'S' 模式应返回 Timestamp("2011-01-01")
        assert p.to_timestamp(how="S") == Timestamp("2011-01-01")
        # 期望结果是 Timestamp("2011-04-01")，减去 1 纳秒，因为 'E' 模式返回最后一刻该周期的时间戳
        expected = Timestamp("2011-04-01") - Timedelta(1, "ns")
        assert p.to_timestamp(how="E") == expected

    @pytest.mark.filterwarnings(
        "ignore:Period with BDay freq is deprecated:FutureWarning"
    )
    # 定义测试方法，验证 Period 对象的 to_timestamp 方法的行为
    def test_to_timestamp(self):
        # 创建一个频率为年的 Period 对象，从 1982 年开始
        p = Period("1982", freq="Y")
        # 获取起始时间戳，使用 "S" 选项将 Period 转换为 Timestamp
        start_ts = p.to_timestamp(how="S")
        # 别名列表，用于验证多个别名与 "S" 选项的等效性
        aliases = ["s", "StarT", "BEGIn"]
        for a in aliases:
            # 断言起始时间戳等于使用不同别名进行转换得到的结果
            assert start_ts == p.to_timestamp("D", how=a)
            # 断言频率为 "3D" 时的起始时间戳也应该等同于 "S" 选项
            assert start_ts == p.to_timestamp("3D", how=a)

        # 获取结束时间戳，使用 "E" 选项将 Period 转换为 Timestamp
        end_ts = p.to_timestamp(how="E")
        # 别名列表，用于验证多个别名与 "E" 选项的等效性
        aliases = ["e", "end", "FINIsH"]
        for a in aliases:
            # 断言结束时间戳等于使用不同别名进行转换得到的结果
            assert end_ts == p.to_timestamp("D", how=a)
            assert end_ts == p.to_timestamp("3D", how=a)

        # 频率列表，用于测试不同频率下的转换行为
        from_lst = ["Y", "Q", "M", "W", "B", "D", "h", "Min", "s"]

        # 内部函数 _ex，根据不同频率计算结束时间
        def _ex(p):
            if p.freq == "B":
                return p.start_time + Timedelta(days=1, nanoseconds=-1)
            return Timestamp((p + p.freq).start_time._value - 1)

        # 遍历频率列表，验证 Period 对象到 Timestamp 的转换是否可逆
        for fcode in from_lst:
            p = Period("1982", freq=fcode)
            # 将 Timestamp 转回 Period，验证是否得到原始 Period 对象
            result = p.to_timestamp().to_period(fcode)
            assert result == p

            # 验证起始时间戳是否与 Period 的起始时间一致
            assert p.start_time == p.to_timestamp(how="S")

            # 验证结束时间戳是否与预期计算的结束时间一致
            assert p.end_time == _ex(p)

        # 验证频率为年的 Period 对象转换为不同时间单位的结束时间戳
        p = Period("1985", freq="Y")
        result = p.to_timestamp("h", how="end")
        expected = Timestamp(1986, 1, 1) - Timedelta(1, "ns")
        assert result == expected
        result = p.to_timestamp("3h", how="end")
        assert result == expected

        result = p.to_timestamp("min", how="end")
        assert result == expected
        result = p.to_timestamp("2min", how="end")
        assert result == expected

        result = p.to_timestamp(how="end")
        assert result == expected

        # 频率为年的 Period 对象转换为不同时间单位的起始时间戳
        expected = datetime(1985, 1, 1)
        result = p.to_timestamp("h", how="start")
        assert result == expected
        result = p.to_timestamp("min", how="start")
        assert result == expected
        result = p.to_timestamp("s", how="start")
        assert result == expected
        result = p.to_timestamp("3h", how="start")
        assert result == expected
        result = p.to_timestamp("5s", how="start")
        assert result == expected

    # 测试 Period 对象的 to_timestamp 方法在工作日频率下的行为
    def test_to_timestamp_business_end(self):
        # 使用断言来验证未来警告是否会产生
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            # 创建一个频率为工作日的 Period 对象，起始日期为 1990 年 1 月 5 日（星期五）
            per = Period("1990-01-05", "B")
            # 将工作日 Period 对象转换为 Timestamp，使用 "E" 选项表示结束时间
            result = per.to_timestamp("B", how="E")

        # 预期的结束时间戳，应为工作日结束后的前一纳秒
        expected = Timestamp("1990-01-06") - Timedelta(nanoseconds=1)
        assert result == expected
    # 使用 pytest 的 parametrize 装饰器，定义多组参数化测试数据
    @pytest.mark.parametrize(
        "ts, expected",
        [
            ("1970-01-01 00:00:00", 0),  # 测试时间戳为 "1970-01-01 00:00:00" 时的期望结果为 0
            ("1970-01-01 00:00:00.000001", 1),  # 测试时间戳微秒为 1
            ("1970-01-01 00:00:00.00001", 10),  # 测试时间戳微秒为 10
            ("1970-01-01 00:00:00.499", 499000),  # 测试时间戳微秒为 499000
            ("1999-12-31 23:59:59.999", 999000),  # 测试时间戳微秒为 999000
            ("1999-12-31 23:59:59.999999", 999999),  # 测试时间戳微秒为 999999
            ("2050-12-31 23:59:59.5", 500000),  # 测试时间戳微秒为 500000
            ("2050-12-31 23:59:59.500001", 500001),  # 测试时间戳微秒为 500001
            ("2050-12-31 23:59:59.123456", 123456),  # 测试时间戳微秒为 123456
        ],
    )
    @pytest.mark.parametrize("freq", [None, "us", "ns"])  # 参数化频率参数为 None, "us", "ns"
    def test_to_timestamp_microsecond(self, ts, expected, freq):
        # GH 24444：GitHub 上的 issue 编号，测试 Period 对象的 to_timestamp 方法返回微秒部分
        result = Period(ts).to_timestamp(freq=freq).microsecond
        assert result == expected  # 断言结果与期望值相符

    # --------------------------------------------------------------
    # Rendering: __repr__, strftime, etc

    @pytest.mark.parametrize(
        "str_ts,freq,str_res,str_freq",
        (
            ("Jan-2000", None, "2000-01", "M"),  # 测试频率为月份的字符串时间戳
            ("2000-12-15", None, "2000-12-15", "D"),  # 测试频率为日期的字符串时间戳
            ("2000-12-15 13:45:26.123456789", "ns", "2000-12-15 13:45:26.123456789", "ns"),  # 测试频率为纳秒的时间戳
            ("2000-12-15 13:45:26.123456789", "us", "2000-12-15 13:45:26.123456", "us"),  # 测试频率为微秒的时间戳
            ("2000-12-15 13:45:26.123456", None, "2000-12-15 13:45:26.123456", "us"),  # 测试默认频率为微秒的时间戳
            ("2000-12-15 13:45:26.123456789", "ms", "2000-12-15 13:45:26.123", "ms"),  # 测试频率为毫秒的时间戳
            ("2000-12-15 13:45:26.123", None, "2000-12-15 13:45:26.123", "ms"),  # 测试默认频率为毫秒的时间戳
            ("2000-12-15 13:45:26", "s", "2000-12-15 13:45:26", "s"),  # 测试频率为秒的时间戳
            ("2000-12-15 13:45:26", "min", "2000-12-15 13:45", "min"),  # 测试频率为分钟的时间戳
            ("2000-12-15 13:45:26", "h", "2000-12-15 13:00", "h"),  # 测试频率为小时的时间戳
            ("2000-12-15", "Y", "2000", "Y-DEC"),  # 测试频率为年的时间戳
            ("2000-12-15", "Q", "2000Q4", "Q-DEC"),  # 测试频率为季度的时间戳
            ("2000-12-15", "M", "2000-12", "M"),  # 测试频率为月份的时间戳
            ("2000-12-15", "W", "2000-12-11/2000-12-17", "W-SUN"),  # 测试频率为周的时间戳
            ("2000-12-15", "D", "2000-12-15", "D"),  # 测试频率为天的时间戳
            ("2000-12-15", "B", "2000-12-15", "B"),  # 测试频率为工作日的时间戳
        ),
    )
    @pytest.mark.filterwarnings(
        "ignore:Period with BDay freq is deprecated:FutureWarning"
    )
    def test_repr(self, str_ts, freq, str_res, str_freq):
        p = Period(str_ts, freq=freq)  # 创建 Period 对象
        assert str(p) == str_res  # 断言 Period 对象的字符串表示与期望结果相符
        assert repr(p) == f"Period('{str_res}', '{str_freq}')"  # 断言 Period 对象的 repr 表示与期望结果相符

    def test_repr_nat(self):
        p = Period("nat", freq="M")  # 创建一个 NaT (Not a Time) 的 Period 对象
        assert repr(NaT) in repr(p)  # 断言 NaT 在 Period 对象的 repr 中出现

    def test_strftime(self):
        # GH#3363：GitHub 上的 issue 编号，测试 Period 对象的 strftime 方法
        p = Period("2000-1-1 12:34:12", freq="s")  # 创建 Period 对象
        res = p.strftime("%Y-%m-%d %H:%M:%S")  # 格式化时间戳
        assert res == "2000-01-01 12:34:12"  # 断言格式化后的结果与期望相符
        assert isinstance(res, str)  # 断言结果是一个字符串
    """Test properties such as year, month, weekday, etc...."""

    @pytest.mark.parametrize("freq", ["Y", "M", "D", "h"])
    # 使用pytest的参数化装饰器，设置测试频率参数freq为年、月、日、小时
    def test_is_leap_year(self, freq):
        # GH 13727
        # 创建一个时间段对象p，起始时间为 "2000-01-01 00:00:00"，频率为参数freq指定的频率
        p = Period("2000-01-01 00:00:00", freq=freq)
        # 断言p对象表示的年份是否为闰年
        assert p.is_leap_year
        # 断言p对象的is_leap_year属性为布尔类型
        assert isinstance(p.is_leap_year, bool)

        # 同样的测试用例，验证非闰年和其他年份的情况
        p = Period("1999-01-01 00:00:00", freq=freq)
        assert not p.is_leap_year

        p = Period("2004-01-01 00:00:00", freq=freq)
        assert p.is_leap_year

        p = Period("2100-01-01 00:00:00", freq=freq)
        assert not p.is_leap_year

    # 测试季度负序数情况
    def test_quarterly_negative_ordinals(self):
        # 创建季度时间段对象p，序数为-1，频率为"Q-DEC"
        p = Period(ordinal=-1, freq="Q-DEC")
        assert p.year == 1969  # 断言p对象的年份为1969
        assert p.quarter == 4   # 断言p对象的季度为4
        assert isinstance(p, Period)  # 断言p对象是Period类型的实例

        p = Period(ordinal=-2, freq="Q-DEC")
        assert p.year == 1969
        assert p.quarter == 3
        assert isinstance(p, Period)

        # 使用月频率测试负二序数情况
        p = Period(ordinal=-2, freq="M")
        assert p.year == 1969
        assert p.month == 11
        assert isinstance(p, Period)

    # 测试频率字符串设置
    def test_freq_str(self):
        # 创建一个频率为"Min"的Period对象i1
        i1 = Period("1982", freq="Min")
        # 断言i1对象的频率为offsets.Minute()对象
        assert i1.freq == offsets.Minute()
        # 断言i1对象的频率字符串为"min"
        assert i1.freqstr == "min"

    # 测试过时频率处理
    @pytest.mark.filterwarnings(
        "ignore:Period with BDay freq is deprecated:FutureWarning"
    )
    def test_period_deprecated_freq(self):
        # 定义频率与其可能的别名对应关系
        cases = {
            "M": ["MTH", "MONTH", "MONTHLY", "Mth", "month", "monthly"],
            "B": ["BUS", "BUSINESS", "BUSINESSLY", "WEEKDAY", "bus"],
            "D": ["DAY", "DLY", "DAILY", "Day", "Dly", "Daily"],
            "h": ["HR", "HOUR", "HRLY", "HOURLY", "hr", "Hour", "HRly"],
            "min": ["minute", "MINUTE", "MINUTELY", "minutely"],
            "s": ["sec", "SEC", "SECOND", "SECONDLY", "second"],
            "ms": ["MILLISECOND", "MILLISECONDLY", "millisecond"],
            "us": ["MICROSECOND", "MICROSECONDLY", "microsecond"],
            "ns": ["NANOSECOND", "NANOSECONDLY", "nanosecond"],
        }

        # 获取频率不支持的错误消息
        msg = INVALID_FREQ_ERR_MSG
        # 遍历每个频率及其可能的别名
        for exp, freqs in cases.items():
            for freq in freqs:
                # 断言使用不支持的频率会抛出ValueError异常，且错误消息与msg匹配
                with pytest.raises(ValueError, match=msg):
                    Period("2016-03-01 09:00", freq=freq)
                with pytest.raises(ValueError, match=msg):
                    Period(ordinal=1, freq=freq)

            # 验证支持的频率别名仍然有效
            p1 = Period("2016-03-01 09:00", freq=exp)
            p2 = Period(ordinal=1, freq=exp)
            assert isinstance(p1, Period)
            assert isinstance(p2, Period)

    # 静态方法，用于构造Period对象
    @staticmethod
    def _period_constructor(bound, offset):
        # 使用给定的边界和偏移量创建Period对象，频率为"us"
        return Period(
            year=bound.year,
            month=bound.month,
            day=bound.day,
            hour=bound.hour,
            minute=bound.minute,
            second=bound.second + offset,
            freq="us",
        )

    # 参数化测试，测试边界和偏移量对于Period对象的影响
    @pytest.mark.parametrize("bound, offset", [(Timestamp.min, -1), (Timestamp.max, 1)])
    # 使用 pytest 的参数化装饰器，为每个测试提供不同的输入参数 period_property
    @pytest.mark.parametrize("period_property", ["start_time", "end_time"])
    # 定义测试函数，用于测试时间范围的上下界
    def test_outer_bounds_start_and_end_time(self, bound, offset, period_property):
        # GH #13346：引用 GitHub issue 编号以便跟踪相关问题
        # 使用 _period_constructor 方法构建时间段对象 period
        period = TestPeriodProperties._period_constructor(bound, offset)
        # 断言期望抛出 OutOfBoundsDatetime 异常，并匹配给定的异常消息
        with pytest.raises(OutOfBoundsDatetime, match="Out of bounds nanosecond"):
            # 获取期间对象 period 的 period_property 属性
            getattr(period, period_property)

    # 使用 pytest 的参数化装饰器，为每个测试提供不同的输入参数 bound, offset 和 period_property
    @pytest.mark.parametrize("bound, offset", [(Timestamp.min, -1), (Timestamp.max, 1)])
    @pytest.mark.parametrize("period_property", ["start_time", "end_time"])
    # 定义测试函数，用于测试时间范围的内部界限
    def test_inner_bounds_start_and_end_time(self, bound, offset, period_property):
        # GH #13346：引用 GitHub issue 编号以便跟踪相关问题
        # 使用 _period_constructor 方法构建时间段对象 period，将 offset 取反作为构造参数
        period = TestPeriodProperties._period_constructor(bound, -offset)
        # 计算期望值，四舍五入到秒，并断言 period_property 属性的值与期望值相等
        expected = period.to_timestamp().round(freq="s")
        assert getattr(period, period_property).round(freq="s") == expected
        # 计算期望值，向下取整到秒，并断言 period_property 属性的值与期望值相等
        expected = (bound - offset * Timedelta(1, unit="s")).floor("s")
        assert getattr(period, period_property).floor("s") == expected

    # 定义测试开始时间的函数
    def test_start_time(self):
        # 频率列表，包括年、季度、月、日、小时、分钟和秒
        freq_lst = ["Y", "Q", "M", "D", "h", "min", "s"]
        # xp 为预期的开始时间，设置为 2012 年 1 月 1 日
        xp = datetime(2012, 1, 1)
        # 遍历频率列表中的每个频率 f
        for f in freq_lst:
            # 创建周期对象 p，频率为当前 f，起始年份为 "2012"
            p = Period("2012", freq=f)
            # 断言周期对象 p 的 start_time 属性等于预期的开始时间 xp
            assert p.start_time == xp
        # 验证是否会产生 FutureWarning 警告，并匹配指定的警告消息 bday_msg
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            # 断言以工作日频率 "B" 创建的周期对象的 start_time 属性等于预期的日期时间对象
            assert Period("2012", freq="B").start_time == datetime(2012, 1, 2)
        # 断言以周频率 "W" 创建的周期对象的 start_time 属性等于预期的日期时间对象
        assert Period("2012", freq="W").start_time == datetime(2011, 12, 26)

    # 定义测试结束时间的函数
    def test_end_time(self):
        # 创建年度频率为 "Y" 的周期对象 p
        p = Period("2012", freq="Y")

        # 定义内部函数 _ex，用于计算与给定参数 args 相对应的时间戳对象
        def _ex(*args):
            return Timestamp(Timestamp(datetime(*args)).as_unit("ns")._value - 1)

        # 设置 xp 为预期的结束时间，使用内部函数 _ex 计算年度频率 "Y" 的结束时间
        xp = _ex(2013, 1, 1)
        # 断言周期对象 p 的 end_time 属性等于预期的结束时间 xp
        assert xp == p.end_time

        # 创建季度频率为 "Q" 的周期对象 p
        p = Period("2012", freq="Q")
        # 计算预期的结束时间 xp，使用内部函数 _ex 计算季度频率 "Q" 的结束时间
        xp = _ex(2012, 4, 1)
        # 断言周期对象 p 的 end_time 属性等于预期的结束时间 xp
        assert xp == p.end_time

        # 创建月度频率为 "M" 的周期对象 p
        p = Period("2012", freq="M")
        # 计算预期的结束时间 xp，使用内部函数 _ex 计算月度频率 "M" 的结束时间
        xp = _ex(2012, 2, 1)
        # 断言周期对象 p 的 end_time 属性等于预期的结束时间 xp
        assert xp == p.end_time

        # 创建日频率为 "D" 的周期对象 p
        p = Period("2012", freq="D")
        # 计算预期的结束时间 xp，使用内部函数 _ex 计算日频率 "D" 的结束时间
        xp = _ex(2012, 1, 2)
        # 断言周期对象 p 的 end_time 属性等于预期的结束时间 xp
        assert xp == p.end_time

        # 创建小时频率为 "h" 的周期对象 p
        p = Period("2012", freq="h")
        # 计算预期的结束时间 xp，使用内部函数 _ex 计算小时频率 "h" 的结束时间
        xp = _ex(2012, 1, 1, 1)
        # 断言周期对象 p 的 end_time 属性等于预期的结束时间 xp
        assert xp == p.end_time

        # 验证是否会产生 FutureWarning 警告，并匹配指定的警告消息 bday_msg
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            # 创建工作日频率 "B" 的周期对象 p
            p = Period("2012", freq="B")
            # 计算预期的结束时间 xp，使用内部函数 _ex 计算工作日频率 "B" 的结束时间
            xp = _ex(2012, 1, 3)
            # 断言周期对象 p 的 end_time 属性等于预期的结束时间 xp
            assert xp == p.end_time

        # 创建周频率 "W" 的周期对象 p
        p = Period("2012", freq="W")
        # 计算预期的结束时间 xp，使用内部函数 _ex 计算周频率 "W" 的结束时间
        xp = _ex(2012, 1, 2)
        # 断言周期对象 p 的 end_time 属性等于预期的结束时间 xp
        assert xp == p.end_time

        # 测试 GH 11738
        # 创建周期频率为 "15D" 的周期对象 p
        p = Period("2012", freq="15D")
        # 计算预期的结束时间 xp，使用内部函数 _ex 计算周期频率 "15D" 的结束时间
        xp = _ex(2012, 1, 16)
        # 断言周期对象 p 的 end_time 属性等于预期的结束时间 xp
        assert xp == p.end_time

        # 创建周期频率为 "1D1h" 的周期对象 p
        p = Period("2012", freq="1D1h")
        # 计算预期的结束时间 xp，使用内部函数 _ex 计算周期频率 "1D1h" 的结束时间
        xp = _ex(2012, 1, 2, 1)
        # 断言周期对象 p 的 end_time 属性等于预期的结束时间 xp
        assert xp == p.end_time

        # 创建周期频率为 "1h1D" 的周期对象 p
        p = Period("2012", freq="1h1D")
        # 计算预期的结束时间 xp，使用内部函数 _ex 计算周期频率 "1h1D" 的结束时间
        xp = _ex(2012, 1, 2, 1)
        # 断言周期对象 p
    def test_end_time_business_friday(self):
        # GH#34449
        # 使用断言来验证未来警告是否会被触发，匹配指定的警告消息
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            # 创建一个时间段对象，以"1990-01-05"为起始日期，频率为工作日
            per = Period("1990-01-05", "B")
            # 获取时间段的结束时间
            result = per.end_time

        # 预期的结束时间为比"1990-01-06"早一纳秒的时间戳
        expected = Timestamp("1990-01-06") - Timedelta(nanoseconds=1)
        # 断言实际结果与预期结果相等
        assert result == expected

    def test_anchor_week_end_time(self):
        def _ex(*args):
            # 返回一个时间戳对象，其时间是输入日期时间的前一纳秒
            return Timestamp(Timestamp(datetime(*args)).as_unit("ns")._value - 1)

        # 创建一个周期对象，从"2013-1-1"到"2013-1-6"的周末
        p = Period("2013-1-1", "W-SAT")
        # 预期的结束时间为"2013-1-6"的前一纳秒时间戳
        xp = _ex(2013, 1, 6)
        # 断言周期对象的结束时间与预期的结束时间相等
        assert p.end_time == xp

    def test_properties_annually(self):
        # 在具有年度频率的周期上测试属性
        a_date = Period(freq="Y", year=2007)
        # 断言周期对象的年份为2007
        assert a_date.year == 2007

    def test_properties_quarterly(self):
        # 在具有季度频率的周期上测试属性
        qedec_date = Period(freq="Q-DEC", year=2007, quarter=1)
        qejan_date = Period(freq="Q-JAN", year=2007, quarter=1)
        qejun_date = Period(freq="Q-JUN", year=2007, quarter=1)
        #
        for x in range(3):
            for qd in (qedec_date, qejan_date, qejun_date):
                # 断言周期对象的年份为2007
                assert (qd + x).qyear == 2007
                # 断言周期对象的季度与迭代次数相关
                assert (qd + x).quarter == x + 1

    def test_properties_monthly(self):
        # 在具有月度频率的周期上测试属性
        m_date = Period(freq="M", year=2007, month=1)
        for x in range(11):
            m_ival_x = m_date + x
            # 断言周期对象的年份为2007
            assert m_ival_x.year == 2007
            if 1 <= x + 1 <= 3:
                # 断言周期对象的季度为1，根据月份确定
                assert m_ival_x.quarter == 1
            elif 4 <= x + 1 <= 6:
                assert m_ival_x.quarter == 2
            elif 7 <= x + 1 <= 9:
                assert m_ival_x.quarter == 3
            elif 10 <= x + 1 <= 12:
                assert m_ival_x.quarter == 4
            # 断言周期对象的月份与迭代次数相关
            assert m_ival_x.month == x + 1

    def test_properties_weekly(self):
        # 在具有周频率的周期上测试属性
        w_date = Period(freq="W", year=2007, month=1, day=7)
        #
        assert w_date.year == 2007
        assert w_date.quarter == 1
        assert w_date.month == 1
        assert w_date.week == 1
        assert (w_date - 1).week == 52
        assert w_date.days_in_month == 31
        assert Period(freq="W", year=2012, month=2, day=1).days_in_month == 29

    def test_properties_weekly_legacy(self):
        # 在具有周频率的周期上测试属性
        w_date = Period(freq="W", year=2007, month=1, day=7)
        assert w_date.year == 2007
        assert w_date.quarter == 1
        assert w_date.month == 1
        assert w_date.week == 1
        assert (w_date - 1).week == 52
        assert w_date.days_in_month == 31

        # 创建预期的周期对象，频率为周，年份为2012，月份为2，日为1
        exp = Period(freq="W", year=2012, month=2, day=1)
        assert exp.days_in_month == 29

        # 定义一个无效频率错误消息
        msg = INVALID_FREQ_ERR_MSG
        # 使用断言来检测是否会引发指定的值错误异常，匹配特定的错误消息
        with pytest.raises(ValueError, match=msg):
            Period(freq="WK", year=2007, month=1, day=7)
    def test_properties_daily(self):
        # Test properties on Periods with daily frequency.
        # 使用断言检查是否会产生未来警告，警告信息需要匹配 bday_msg
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            # 创建一个频率为工作日（Business Day）的 Period 对象，日期为 2007 年 1 月 1 日
            b_date = Period(freq="B", year=2007, month=1, day=1)
        #
        # 断言 Period 对象的属性值
        assert b_date.year == 2007
        assert b_date.quarter == 1
        assert b_date.month == 1
        assert b_date.day == 1
        assert b_date.weekday == 0  # 星期一，Python 中星期一是 0
        assert b_date.dayofyear == 1  # 一年中的第一天
        assert b_date.days_in_month == 31  # 1 月有 31 天
        # 再次检查是否会产生未来警告，警告信息需要匹配 bday_msg
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            assert Period(freq="B", year=2012, month=2, day=1).days_in_month == 29  # 2012 年 2 月有 29 天

        # 创建一个频率为每日（Daily）的 Period 对象，日期为 2007 年 1 月 1 日
        d_date = Period(freq="D", year=2007, month=1, day=1)

        # 断言 Period 对象的属性值
        assert d_date.year == 2007
        assert d_date.quarter == 1
        assert d_date.month == 1
        assert d_date.day == 1
        assert d_date.weekday == 0  # 星期一
        assert d_date.dayofyear == 1  # 一年中的第一天
        assert d_date.days_in_month == 31  # 1 月有 31 天
        assert Period(freq="D", year=2012, month=2, day=1).days_in_month == 29  # 2012 年 2 月有 29 天

    def test_properties_hourly(self):
        # Test properties on Periods with hourly frequency.
        # 创建一个每小时（Hourly）频率的 Period 对象，日期为 2007 年 1 月 1 日，小时为 0
        h_date1 = Period(freq="h", year=2007, month=1, day=1, hour=0)
        # 创建一个每 2 小时（Every 2 hours）频率的 Period 对象，日期为 2007 年 1 月 1 日，小时为 0
        h_date2 = Period(freq="2h", year=2007, month=1, day=1, hour=0)

        # 遍历 h_date1 和 h_date2
        for h_date in [h_date1, h_date2]:
            # 断言 Period 对象的属性值
            assert h_date.year == 2007
            assert h_date.quarter == 1
            assert h_date.month == 1
            assert h_date.day == 1
            assert h_date.weekday == 0  # 星期一
            assert h_date.dayofyear == 1  # 一年中的第一天
            assert h_date.hour == 0  # 小时为 0
            assert h_date.days_in_month == 31  # 1 月有 31 天
            assert (
                Period(freq="h", year=2012, month=2, day=1, hour=0).days_in_month == 29
            )  # 2012 年 2 月有 29 天

    def test_properties_minutely(self):
        # Test properties on Periods with minutely frequency.
        # 创建一个每分钟（Minutely）频率的 Period 对象，日期为 2007 年 1 月 1 日，小时和分钟都为 0
        t_date = Period(freq="Min", year=2007, month=1, day=1, hour=0, minute=0)
        #
        # 断言 Period 对象的属性值
        assert t_date.quarter == 1
        assert t_date.month == 1
        assert t_date.day == 1
        assert t_date.weekday == 0  # 星期一
        assert t_date.dayofyear == 1  # 一年中的第一天
        assert t_date.hour == 0  # 小时为 0
        assert t_date.minute == 0  # 分钟为 0
        assert t_date.days_in_month == 31  # 1 月有 31 天
        assert (
            Period(freq="D", year=2012, month=2, day=1, hour=0, minute=0).days_in_month
            == 29
        )  # 2012 年 2 月有 29 天
    def test_properties_secondly(self):
        # 定义一个测试函数，测试带有每分钟频率的时间段的属性

        # 创建一个 Period 对象，表示2007年1月1日 00:00:00
        s_date = Period(
            freq="Min", year=2007, month=1, day=1, hour=0, minute=0, second=0
        )
        
        # 断言各个属性的值是否符合预期
        assert s_date.year == 2007
        assert s_date.quarter == 1
        assert s_date.month == 1
        assert s_date.day == 1
        assert s_date.weekday == 0
        assert s_date.dayofyear == 1
        assert s_date.hour == 0
        assert s_date.minute == 0
        assert s_date.second == 0
        assert s_date.days_in_month == 31
        
        # 创建另一个 Period 对象，检查2012年2月1日 00:00:00的月份天数是否为29天
        assert (
            Period(
                freq="Min", year=2012, month=2, day=1, hour=0, minute=0, second=0
            ).days_in_month
            == 29
        )
class TestPeriodComparisons:
    # 定义测试类 TestPeriodComparisons，用于比较 Period 对象的功能

    def test_sort_periods(self):
        # 测试排序期间对象的功能
        jan = Period("2000-01", "M")
        # 创建一个月度期间对象 jan，表示2000年1月
        feb = Period("2000-02", "M")
        # 创建一个月度期间对象 feb，表示2000年2月
        mar = Period("2000-03", "M")
        # 创建一个月度期间对象 mar，表示2000年3月
        periods = [mar, jan, feb]
        # 创建包含上述期间对象的列表 periods
        correctPeriods = [jan, feb, mar]
        # 创建已排序的期间对象列表 correctPeriods
        assert sorted(periods) == correctPeriods
        # 断言排序后的 periods 列表与 correctPeriods 列表相等


def test_period_immutable():
    # 定义测试函数 test_period_immutable，用于测试 Period 对象的不可变性
    # see gh-17116
    # 参考 GitHub issue 17116

    msg = "not writable"
    # 设置错误信息提示字符串为 "not writable"

    per = Period("2014Q1")
    # 创建一个季度期间对象 per，表示2014年第1季度
    with pytest.raises(AttributeError, match=msg):
        # 使用 pytest 的断言来检查是否抛出 AttributeError 异常，并匹配错误信息
        per.ordinal = 14

    freq = per.freq
    # 获取 per 对象的频率属性值，并赋给 freq 变量
    with pytest.raises(AttributeError, match=msg):
        # 使用 pytest 的断言来检查是否抛出 AttributeError 异常，并匹配错误信息
        per.freq = 2 * freq


def test_small_year_parsing():
    # 定义测试函数 test_small_year_parsing，用于测试解析小年份的功能

    per1 = Period("0001-01-07", "D")
    # 创建一个每日频率的期间对象 per1，表示公元1年1月7日
    assert per1.year == 1
    # 断言 per1 对象的年份属性为 1
    assert per1.day == 7
    # 断言 per1 对象的日期属性为 7


def test_negone_ordinals():
    # 定义测试函数 test_negone_ordinals，用于测试负数序数的功能

    freqs = ["Y", "M", "Q", "D", "h", "min", "s"]
    # 创建包含不同频率的字符串列表 freqs

    period = Period(ordinal=-1, freq="D")
    # 创建一个每日频率的期间对象 period，其序数为 -1
    for freq in freqs:
        # 遍历 freqs 列表中的每个频率字符串
        repr(period.asfreq(freq))

    for freq in freqs:
        # 遍历 freqs 列表中的每个频率字符串
        period = Period(ordinal=-1, freq=freq)
        # 使用不同频率创建序数为 -1 的期间对象 period
        repr(period)
        # 调用 repr 方法显示期间对象的表示形式
        assert period.year == 1969
        # 断言 period 对象的年份属性为 1969

    with tm.assert_produces_warning(FutureWarning, match=bday_msg):
        # 使用 tm.assert_produces_warning 检查是否会产生 FutureWarning 警告，并匹配 bday_msg
        period = Period(ordinal=-1, freq="B")
    repr(period)
    # 调用 repr 方法显示期间对象的表示形式
    period = Period(ordinal=-1, freq="W")
    # 创建一个每周频率的序数为 -1 的期间对象 period
    repr(period)
    # 调用 repr 方法显示期间对象的表示形式
```