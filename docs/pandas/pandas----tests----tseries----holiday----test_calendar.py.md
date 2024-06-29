# `D:\src\scipysrc\pandas\pandas\tests\tseries\holiday\test_calendar.py`

```
# 导入datetime模块中的datetime类，用于处理日期和时间
from datetime import datetime

# 导入pytest模块，用于编写和运行测试
import pytest

# 从pandas库中导入所需的模块
from pandas import (
    DatetimeIndex,  # 日期时间索引
    offsets,  # 偏移量
    to_datetime,  # 将对象转换为日期时间类型
)

# 导入pandas内部测试模块
import pandas._testing as tm

# 从pandas.tseries.holiday模块中导入以下类和函数
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,  # 抽象节假日日历类
    Holiday,  # 节假日类
    Timestamp,  # 时间戳类
    USFederalHolidayCalendar,  # 美国联邦节假日日历类
    USLaborDay,  # 美国劳动节
    USThanksgivingDay,  # 美国感恩节
    get_calendar,  # 获取节假日日历对象的函数
)

# 使用pytest的装饰器@parametrize，参数化测试函数test_calendar
@pytest.mark.parametrize(
    "transform", [lambda x: x, lambda x: x.strftime("%Y-%m-%d"), lambda x: Timestamp(x)]
)
def test_calendar(transform):
    # 定义起始日期和结束日期
    start_date = datetime(2012, 1, 1)
    end_date = datetime(2012, 12, 31)

    # 创建USFederalHolidayCalendar对象
    calendar = USFederalHolidayCalendar()

    # 获取转换后的起始日期到结束日期之间的节假日
    holidays = calendar.holidays(transform(start_date), transform(end_date))

    # 预期的节假日日期列表
    expected = [
        datetime(2012, 1, 2),
        datetime(2012, 1, 16),
        datetime(2012, 2, 20),
        datetime(2012, 5, 28),
        datetime(2012, 7, 4),
        datetime(2012, 9, 3),
        datetime(2012, 10, 8),
        datetime(2012, 11, 12),
        datetime(2012, 11, 22),
        datetime(2012, 12, 25),
    ]

    # 断言节假日列表是否与预期一致
    assert list(holidays.to_pydatetime()) == expected


# 定义测试函数test_calendar_caching
def test_calendar_caching():
    # 测试日历缓存功能，参见GitHub issue gh-9552

    # 定义TestCalendar类，继承自AbstractHolidayCalendar
    class TestCalendar(AbstractHolidayCalendar):
        def __init__(self, name=None, rules=None) -> None:
            super().__init__(name=name, rules=rules)

    # 创建TestCalendar对象jan1和jan2，分别包含不同的节假日规则
    jan1 = TestCalendar(rules=[Holiday("jan1", year=2015, month=1, day=1)])
    jan2 = TestCalendar(rules=[Holiday("jan2", year=2015, month=1, day=2)])

    # 获取1月1日的节假日，验证是否影响1月2日的结果
    expected = DatetimeIndex(["01-Jan-2015"]).as_unit("us")
    tm.assert_index_equal(jan1.holidays(), expected)

    # 获取1月2日的节假日
    expected2 = DatetimeIndex(["02-Jan-2015"]).as_unit("us")
    tm.assert_index_equal(jan2.holidays(), expected2)


# 定义测试函数test_calendar_observance_dates
def test_calendar_observance_dates():
    # 测试观察日期功能，参见GitHub issue gh-11477

    # 获取USFederalHolidayCalendar对象
    us_fed_cal = get_calendar("USFederalHolidayCalendar")

    # 获取2015年7月3日到2015年7月3日之间的节假日
    holidays0 = us_fed_cal.holidays(
        datetime(2015, 7, 3), datetime(2015, 7, 3)
    )

    # 获取2015年7月3日到2015年7月6日之间的节假日
    holidays1 = us_fed_cal.holidays(
        datetime(2015, 7, 3), datetime(2015, 7, 6)
    )

    # 再次获取2015年7月3日到2015年7月3日之间的节假日
    holidays2 = us_fed_cal.holidays(
        datetime(2015, 7, 3), datetime(2015, 7, 3)
    )

    # 断言节假日结果是否相等
    tm.assert_index_equal(holidays0, holidays1)
    tm.assert_index_equal(holidays0, holidays2)


# 定义测试函数test_rule_from_name
def test_rule_from_name():
    # 测试通过名称获取节假日规则的功能

    # 获取USFederalHolidayCalendar对象
    us_fed_cal = get_calendar("USFederalHolidayCalendar")

    # 断言获取"Thanksgiving Day"规则是否等于USThanksgivingDay类
    assert us_fed_cal.rule_from_name("Thanksgiving Day") == USThanksgivingDay


# 定义测试函数test_calendar_2031
def test_calendar_2031():
    # 测试2031年的节假日计算，参见GitHub issue gh-27790

    # 创建testCalendar类，继承自AbstractHolidayCalendar
    class testCalendar(AbstractHolidayCalendar):
        rules = [USLaborDay]  # 包含USLaborDay规则
    # 创建一个名为cal的testCalendar对象
    cal = testCalendar()
    # 使用自定义的工作日偏移量，使用cal作为工作日历
    workDay = offsets.CustomBusinessDay(calendar=cal)
    # 将字符串"2031-08-30"转换为datetime对象
    Sat_before_Labor_Day_2031 = to_datetime("2031-08-30")
    # 计算Sat_before_Labor_Day_2031日期后0个工作日的日期
    next_working_day = Sat_before_Labor_Day_2031 + 0 * workDay
    # 断言下一个工作日为"2031-09-02"
    assert next_working_day == to_datetime("2031-09-02")
def test_no_holidays_calendar():
    # Test for issue #31415

    # 定义一个名为 NoHolidaysCalendar 的类，继承自 AbstractHolidayCalendar
    class NoHolidaysCalendar(AbstractHolidayCalendar):
        pass

    # 创建 NoHolidaysCalendar 类的实例
    cal = NoHolidaysCalendar()

    # 使用 cal 对象调用 holidays 方法，获取指定日期范围内的假期
    holidays = cal.holidays(Timestamp("01-Jan-2020"), Timestamp("01-Jan-2021"))

    # 创建一个空的 DatetimeIndex 对象，用于与假期列表比较，类型为 DatetimeIndex
    empty_index = DatetimeIndex([])  # Type is DatetimeIndex since return_name=False

    # 使用 tm 模块的 assert_index_equal 函数比较假期列表和空的日期索引对象
    tm.assert_index_equal(holidays, empty_index)
```