# `D:\src\scipysrc\pandas\pandas\tests\tseries\offsets\test_week.py`

```
"""
Tests for the following offsets:
- Week
- WeekOfMonth
- LastWeekOfMonth
"""

# 导入必要的模块和类
from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
)

import pytest

from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
    Day,
    LastWeekOfMonth,
    Week,
    WeekOfMonth,
)

from pandas.tests.tseries.offsets.common import (
    WeekDay,
    assert_is_on_offset,
    assert_offset_equal,
)

# 定义测试类 TestWeek
class TestWeek:
    # 测试 Week 类的 repr 方法
    def test_repr(self):
        assert repr(Week(weekday=0)) == "<Week: weekday=0>"
        assert repr(Week(n=-1, weekday=0)) == "<-1 * Week: weekday=0>"
        assert repr(Week(n=-2, weekday=0)) == "<-2 * Weeks: weekday=0>"

    # 测试 Week 类在边缘情况下的行为
    def test_corner(self):
        # 测试在给定无效星期几时是否引发 ValueError 异常
        with pytest.raises(ValueError, match="Day must be"):
            Week(weekday=7)

        with pytest.raises(ValueError, match="Day must be"):
            Week(weekday=-1)

    offset_cases = []

    # 添加测试用例：非工作周情况下的 Week 偏移
    offset_cases.append(
        (
            Week(),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 8),
                datetime(2008, 1, 4): datetime(2008, 1, 11),
                datetime(2008, 1, 5): datetime(2008, 1, 12),
                datetime(2008, 1, 6): datetime(2008, 1, 13),
                datetime(2008, 1, 7): datetime(2008, 1, 14),
            },
        )
    )

    # 添加测试用例：星期一情况下的 Week 偏移
    offset_cases.append(
        (
            Week(weekday=0),
            {
                datetime(2007, 12, 31): datetime(2008, 1, 7),
                datetime(2008, 1, 4): datetime(2008, 1, 7),
                datetime(2008, 1, 5): datetime(2008, 1, 7),
                datetime(2008, 1, 6): datetime(2008, 1, 7),
                datetime(2008, 1, 7): datetime(2008, 1, 14),
            },
        )
    )

    # 添加测试用例：n=0 时向前滚动的星期一情况
    offset_cases.append(
        (
            Week(0, weekday=0),
            {
                datetime(2007, 12, 31): datetime(2007, 12, 31),
                datetime(2008, 1, 4): datetime(2008, 1, 7),
                datetime(2008, 1, 5): datetime(2008, 1, 7),
                datetime(2008, 1, 6): datetime(2008, 1, 7),
                datetime(2008, 1, 7): datetime(2008, 1, 7),
            },
        )
    )

    # 添加测试用例：n=-2 时向前滚动的星期二情况
    offset_cases.append(
        (
            Week(-2, weekday=1),
            {
                datetime(2010, 4, 6): datetime(2010, 3, 23),
                datetime(2010, 4, 8): datetime(2010, 3, 30),
                datetime(2010, 4, 5): datetime(2010, 3, 23),
            },
        )
    )

    # 使用 @pytest.mark.parametrize 注入参数化测试用例
    @pytest.mark.parametrize("case", offset_cases)
    # 测试 Week 偏移的功能是否按预期工作
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    # 参数化测试：遍历所有星期几来测试 Week 偏移
    @pytest.mark.parametrize("weekday", range(7))
    def test_is_on_offset(self, weekday):
        # 创建一个 Week 对象，设置其 weekday 属性为参数 weekday
        offset = Week(weekday=weekday)

        # 遍历日期范围为 1 到 7
        for day in range(1, 8):
            # 创建 datetime 对象，表示 2008 年 1 月的每一天
            date = datetime(2008, 1, day)
            # 计算预期值，用于检查日期是否在偏移量之上
            expected = day % 7 == weekday
        
        # 调用 assert_is_on_offset 函数，验证偏移量计算是否正确
        assert_is_on_offset(offset, date, expected)

    @pytest.mark.parametrize(
        "n,date",
        [
            # 参数化测试，提供不同的 n 和 date 组合
            (2, "1862-01-13 09:03:34.873477378+0210"),
            (-2, "1856-10-24 16:18:36.556360110-0717"),
        ],
    )
    def test_is_on_offset_weekday_none(self, n, date):
        # GH 18510 Week with weekday = None, normalize = False
        # 创建 Week 对象，设置其 n 属性为参数 n，weekday 属性为 None
        offset = Week(n=n, weekday=None)
        # 创建 Timestamp 对象，表示给定的日期和时区
        ts = Timestamp(date, tz="Africa/Lusaka")
        # 使用 offset.is_on_offset 方法快速计算是否在偏移量上
        fast = offset.is_on_offset(ts)
        # 使用数学计算检查偏移量是否正确
        slow = (ts + offset) - offset == ts
        # 断言快速计算结果和慢速计算结果应相等
        assert fast == slow

    def test_week_add_invalid(self):
        # Week with weekday should raise TypeError and _not_ AttributeError
        # when adding invalid offset
        # 创建 Week 对象，设置其 weekday 属性为 1
        offset = Week(weekday=1)
        # 创建 Day 对象
        other = Day()
        # 使用 pytest.raises 断言，验证当添加无效偏移量时会引发 TypeError 错误
        with pytest.raises(TypeError, match="Cannot add"):
            offset + other
class TestWeekOfMonth:
    # WeekOfMonth 类的单元测试类

    def test_constructor(self):
        # 测试构造函数的异常情况，确保抛出特定类型的异常
        with pytest.raises(ValueError, match="^Week"):
            WeekOfMonth(n=1, week=4, weekday=0)

        with pytest.raises(ValueError, match="^Week"):
            WeekOfMonth(n=1, week=-1, weekday=0)

        with pytest.raises(ValueError, match="^Day"):
            WeekOfMonth(n=1, week=0, weekday=-1)

        with pytest.raises(ValueError, match="^Day"):
            WeekOfMonth(n=1, week=0, weekday=-7)

    def test_repr(self):
        # 测试对象的字符串表示形式是否符合预期
        assert (
            repr(WeekOfMonth(weekday=1, week=2)) == "<WeekOfMonth: week=2, weekday=1>"
        )
    def test_offset(self):
        date1 = datetime(2011, 1, 4)  # 定义日期变量date1，表示每月的第一个星期二
        date2 = datetime(2011, 1, 11)  # 定义日期变量date2，表示每月的第二个星期二
        date3 = datetime(2011, 1, 18)  # 定义日期变量date3，表示每月的第三个星期二
        date4 = datetime(2011, 1, 25)  # 定义日期变量date4，表示每月的第四个星期二

        # 设置测试用例列表，每个元组包含 (n, week, weekday, dt, expected)
        # 其中n表示偏移量，week表示星期，weekday表示星期几，dt表示基准日期，expected表示预期结果日期
        test_cases = [
            (-2, 2, 1, date1, datetime(2010, 11, 16)),  # 对date1进行偏移，预期结果是2010年11月16日
            (-2, 2, 1, date2, datetime(2010, 11, 16)),  # 对date2进行偏移，预期结果是2010年11月16日
            (-2, 2, 1, date3, datetime(2010, 11, 16)),  # 对date3进行偏移，预期结果是2010年11月16日
            (-2, 2, 1, date4, datetime(2010, 12, 21)),  # 对date4进行偏移，预期结果是2010年12月21日
            (-1, 2, 1, date1, datetime(2010, 12, 21)),  # 对date1进行偏移，预期结果是2010年12月21日
            (-1, 2, 1, date2, datetime(2010, 12, 21)),  # 对date2进行偏移，预期结果是2010年12月21日
            (-1, 2, 1, date3, datetime(2010, 12, 21)),  # 对date3进行偏移，预期结果是2010年12月21日
            (-1, 2, 1, date4, datetime(2011, 1, 18)),   # 对date4进行偏移，预期结果是2011年1月18日
            (0, 0, 1, date1, datetime(2011, 1, 4)),     # 对date1进行偏移，预期结果是2011年1月4日
            (0, 0, 1, date2, datetime(2011, 2, 1)),     # 对date2进行偏移，预期结果是2011年2月1日
            (0, 0, 1, date3, datetime(2011, 2, 1)),     # 对date3进行偏移，预期结果是2011年2月1日
            (0, 0, 1, date4, datetime(2011, 2, 1)),     # 对date4进行偏移，预期结果是2011年2月1日
            (0, 1, 1, date1, datetime(2011, 1, 11)),    # 对date1进行偏移，预期结果是2011年1月11日
            (0, 1, 1, date2, datetime(2011, 1, 11)),    # 对date2进行偏移，预期结果是2011年1月11日
            (0, 1, 1, date3, datetime(2011, 2, 8)),     # 对date3进行偏移，预期结果是2011年2月8日
            (0, 1, 1, date4, datetime(2011, 2, 8)),     # 对date4进行偏移，预期结果是2011年2月8日
            (0, 0, 1, date1, datetime(2011, 1, 4)),     # 对date1进行偏移，预期结果是2011年1月4日
            (0, 1, 1, date2, datetime(2011, 1, 11)),    # 对date2进行偏移，预期结果是2011年1月11日
            (0, 2, 1, date3, datetime(2011, 1, 18)),    # 对date3进行偏移，预期结果是2011年1月18日
            (0, 3, 1, date4, datetime(2011, 1, 25)),    # 对date4进行偏移，预期结果是2011年1月25日
            (1, 0, 0, date1, datetime(2011, 2, 7)),     # 对date1进行偏移，预期结果是2011年2月7日
            (1, 0, 0, date2, datetime(2011, 2, 7)),     # 对date2进行偏移，预期结果是2011年2月7日
            (1, 0, 0, date3, datetime(2011, 2, 7)),     # 对date3进行偏移，预期结果是2011年2月7日
            (1, 0, 0, date4, datetime(2011, 2, 7)),     # 对date4进行偏移，预期结果是2011年2月7日
            (1, 0, 1, date1, datetime(2011, 2, 1)),     # 对date1进行偏移，预期结果是2011年2月1日
            (1, 0, 1, date2, datetime(2011, 2, 1)),     # 对date2进行偏移，预期结果是2011年2月1日
            (1, 0, 1, date3, datetime(2011, 2, 1)),     # 对date3进行偏移，预期结果是2011年2月1日
            (1, 0, 1, date4, datetime(2011, 2, 1)),     # 对date4进行偏移，预期结果是2011年2月1日
            (1, 0, 2, date1, datetime(2011, 1, 5)),     # 对date1进行偏移，预期结果是2011年1月5日
            (1, 0, 2, date2, datetime(2011, 2, 2)),     # 对date2进行偏移，预期结果是2011年2月2日
            (1, 0, 2, date3, datetime(2011, 2, 2)),     # 对date3进行偏移，预期结果是2011年2月2日
            (1, 0, 2, date4, datetime(2011, 2, 2)),     # 对date4进行偏移，预期结果是2011年2月2日
            (1, 2, 1, date1, datetime(2011, 1, 18)),    # 对date1进行偏移，预期结果是2011年1月18日
            (1, 2, 1, date2, datetime(2011, 1, 18)),    # 对date2进行偏移，预期结果是2011年1月18日
            (1, 2, 1, date3, datetime(2011, 2, 15)),    # 对date3进行偏移，预期结果是2011年2月15日
            (1, 2, 1, date4, datetime(2011, 2, 15)),    # 对date4进行偏移，预期结果是2011年2月15日
            (2, 2, 1, date1, datetime(2011, 2, 15)),    # 对date1进行偏移，预期结果是2011年2月15日
            (2, 2, 1, date2, datetime(2011, 2, 15)),    # 对date2进行偏移，预期结果是2011年2月15日
            (2, 2, 1, date3, datetime(2011, 3, 15)),    # 对date3进行偏移，预期结果是2011年3月15日
            (2, 2, 1, date4, datetime(2011, 3, 15)),    # 对date4
    # 定义包含多个测试用例的列表，每个元组表示一个测试情况
    on_offset_cases = [
        (0, 0, datetime(2011, 2, 7), True),    # 第一个测试情况：周数为0，星期数为0，日期为2011年2月7日，期望结果为True
        (0, 0, datetime(2011, 2, 6), False),   # 第二个测试情况：周数为0，星期数为0，日期为2011年2月6日，期望结果为False
        (0, 0, datetime(2011, 2, 14), False),  # 第三个测试情况：周数为0，星期数为0，日期为2011年2月14日，期望结果为False
        (1, 0, datetime(2011, 2, 14), True),   # 第四个测试情况：周数为1，星期数为0，日期为2011年2月14日，期望结果为True
        (0, 1, datetime(2011, 2, 1), True),    # 第五个测试情况：周数为0，星期数为1，日期为2011年2月1日，期望结果为True
        (0, 1, datetime(2011, 2, 8), False),   # 第六个测试情况：周数为0，星期数为1，日期为2011年2月8日，期望结果为False
    ]

    # 使用 pytest 的 parametrize 装饰器，对 test_is_on_offset 方法进行参数化测试
    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        # 从参数中解包出周数、星期数、日期和期望结果
        week, weekday, dt, expected = case
        # 创建 WeekOfMonth 对象，表示给定的周数和星期数的偏移量
        offset = WeekOfMonth(week=week, weekday=weekday)
        # 断言调用 is_on_offset 方法的结果是否等于期望结果
        assert offset.is_on_offset(dt) == expected

    # 使用 pytest 的 parametrize 装饰器，对 test_is_on_offset_nanoseconds 方法进行参数化测试
    @pytest.mark.parametrize(
        "n,week,date,tz",
        [
            (2, 2, "1916-05-15 01:14:49.583410462+0422", "Asia/Qyzylorda"),
            (-3, 1, "1980-12-08 03:38:52.878321185+0500", "Asia/Oral"),
        ],
    )
    def test_is_on_offset_nanoseconds(self, n, week, date, tz):
        # GH 18864
        # 确保纳秒不会影响 is_on_offset 方法（以及 apply 方法）
        # 创建 WeekOfMonth 对象，表示给定的周数、星期数和时区的偏移量
        offset = WeekOfMonth(n=n, week=week, weekday=0)
        # 创建 Timestamp 对象，表示给定的日期和时区的时间戳
        ts = Timestamp(date, tz=tz)
        # 快速计算偏移后的时间戳是否在偏移后保持不变
        fast = offset.is_on_offset(ts)
        # 慢速计算：先加上偏移量，再减去偏移量，结果是否与原时间戳相同
        slow = (ts + offset) - offset == ts
        # 断言快速计算结果与慢速计算结果是否相等
        assert fast == slow
class TestLastWeekOfMonth:
    # 测试构造函数，验证参数 n 为 0 时会引发 ValueError 异常
    def test_constructor(self):
        with pytest.raises(ValueError, match="^N cannot be 0"):
            LastWeekOfMonth(n=0, weekday=1)

        # 测试构造函数，验证参数 weekday 为负数时会引发 ValueError 异常
        with pytest.raises(ValueError, match="^Day"):
            LastWeekOfMonth(n=1, weekday=-1)

        # 测试构造函数，验证参数 weekday 超出 1 到 7 的范围时会引发 ValueError 异常
        with pytest.raises(ValueError, match="^Day"):
            LastWeekOfMonth(n=1, weekday=7)

    # 测试 offset 方法
    def test_offset(self):
        # 设置星期六的日期
        last_sat = datetime(2013, 8, 31)
        next_sat = datetime(2013, 9, 28)
        offset_sat = LastWeekOfMonth(n=1, weekday=5)

        # 测试上一天的结果是否符合预期
        one_day_before = last_sat + timedelta(days=-1)
        assert one_day_before + offset_sat == last_sat

        # 测试下一天的结果是否符合预期
        one_day_after = last_sat + timedelta(days=+1)
        assert one_day_after + offset_sat == next_sat

        # 测试同一天的结果是否符合预期
        assert last_sat + offset_sat == next_sat

        # 设置星期四的日期
        offset_thur = LastWeekOfMonth(n=1, weekday=3)
        last_thurs = datetime(2013, 1, 31)
        next_thurs = datetime(2013, 2, 28)

        # 测试上一天的结果是否符合预期
        one_day_before = last_thurs + timedelta(days=-1)
        assert one_day_before + offset_thur == last_thurs

        # 测试下一天的结果是否符合预期
        one_day_after = last_thurs + timedelta(days=+1)
        assert one_day_after + offset_thur == next_thurs

        # 测试同一天的结果是否符合预期
        assert last_thurs + offset_thur == next_thurs

        # 测试提前三天的结果是否符合预期
        three_before = last_thurs + timedelta(days=-3)
        assert three_before + offset_thur == last_thurs

        # 测试延后两天的结果是否符合预期
        two_after = last_thurs + timedelta(days=+2)
        assert two_after + offset_thur == next_thurs

        # 设置星期日的偏移量
        offset_sunday = LastWeekOfMonth(n=1, weekday=WeekDay.SUN)
        assert datetime(2013, 7, 31) + offset_sunday == datetime(2013, 8, 25)

    # 测试特定日期是否在偏移量上
    on_offset_cases = [
        (WeekDay.SUN, datetime(2013, 1, 27), True),
        (WeekDay.SAT, datetime(2013, 3, 30), True),
        (WeekDay.MON, datetime(2013, 2, 18), False),  # 不是最后一个星期一
        (WeekDay.SUN, datetime(2013, 2, 25), False),  # 不是星期日
        (WeekDay.MON, datetime(2013, 2, 25), True),
        (WeekDay.SAT, datetime(2013, 11, 30), True),
        (WeekDay.SAT, datetime(2006, 8, 26), True),
        (WeekDay.SAT, datetime(2007, 8, 25), True),
        (WeekDay.SAT, datetime(2008, 8, 30), True),
        (WeekDay.SAT, datetime(2009, 8, 29), True),
        (WeekDay.SAT, datetime(2010, 8, 28), True),
        (WeekDay.SAT, datetime(2011, 8, 27), True),
        (WeekDay.SAT, datetime(2019, 8, 31), True),
    ]

    # 使用参数化测试，验证 is_on_offset 方法的预期行为
    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        weekday, dt, expected = case
        offset = LastWeekOfMonth(weekday=weekday)
        assert offset.is_on_offset(dt) == expected

    # 更多的参数化测试案例
    @pytest.mark.parametrize(
        "n,weekday,date,tz",
        [
            (4, 6, "1917-05-27 20:55:27.084284178+0200", "Europe/Warsaw"),
            (-4, 5, "2005-08-27 05:01:42.799392561-0500", "America/Rainy_River"),
        ],
    )
    # 测试函数，验证 LastWeekOfMonth 在给定条件下的偏移量计算是否正确
    def test_last_week_of_month_on_offset(self, n, weekday, date, tz):
        # 创建 LastWeekOfMonth 对象，指定偏移量的条件
        offset = LastWeekOfMonth(n=n, weekday=weekday)
        # 创建时间戳对象，使用指定的日期和时区
        ts = Timestamp(date, tz=tz)
        # 使用慢速计算方法验证偏移量是否正确
        slow = (ts + offset) - offset == ts
        # 使用快速计算方法验证偏移量是否正确
        fast = offset.is_on_offset(ts)
        # 断言快速计算结果与慢速计算结果是否一致
        assert fast == slow

    # 测试函数，验证 LastWeekOfMonth 对象的字符串表示形式是否正确
    def test_repr(self):
        # 断言创建的 LastWeekOfMonth 对象的 repr 方法返回的字符串是否符合预期格式
        assert (
            repr(LastWeekOfMonth(n=2, weekday=1)) == "<2 * LastWeekOfMonths: weekday=1>"
        )
```