# `D:\src\scipysrc\pandas\pandas\tests\tseries\holiday\test_holiday.py`

```
# 从datetime模块中导入datetime和timezone类
from datetime import (
    datetime,
    timezone,
)

# 导入pytest模块，用于单元测试
import pytest

# 从pandas库中导入DatetimeIndex和Series类
from pandas import (
    DatetimeIndex,
    Series,
)

# 导入pandas库中的测试工具模块
import pandas._testing as tm

# 从pandas库的tseries.holiday模块中导入多个类和函数
from pandas.tseries.holiday import (
    MO,
    SA,
    AbstractHolidayCalendar,
    DateOffset,
    EasterMonday,
    GoodFriday,
    Holiday,
    HolidayCalendarFactory,
    Timestamp,
    USColumbusDay,
    USFederalHolidayCalendar,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    get_calendar,
    next_monday,
)

# 使用pytest的参数化装饰器，为test_holiday_dates函数多次运行测试用例
@pytest.mark.parametrize(
    "holiday,start_date,end_date,expected",
    ],
)
def test_holiday_dates(holiday, start_date, end_date, expected):
    # 断言：验证holiday对象生成的日期列表是否与预期列表相同
    assert list(holiday.dates(start_date, end_date)) == expected

    # 断言：验证时区信息是否被保留在生成的日期列表中
    assert list(
        holiday.dates(
            Timestamp(start_date, tz=timezone.utc), Timestamp(end_date, tz=timezone.utc)
        )
    ) == [dt.replace(tzinfo=timezone.utc) for dt in expected]

# 下面的代码块未提供完整，需要补全代码才能进行注释
@pytest.mark.parametrize(
    "holiday,start,expected",
    [
        # 创建一个包含元组的列表，每个元组包含节日名称、日期或日期时间对象以及一个时间戳列表
        (USMemorialDay, datetime(2015, 7, 1), []),  # Memorial Day 的日期和时间戳列表为空
        (USMemorialDay, "2015-05-25", [Timestamp("2015-05-25")]),  # Memorial Day 的时间戳为 2015-05-25
        (USLaborDay, datetime(2015, 7, 1), []),  # Labor Day 的日期和时间戳列表为空
        (USLaborDay, "2015-09-07", [Timestamp("2015-09-07")]),  # Labor Day 的时间戳为 2015-09-07
        (USColumbusDay, datetime(2015, 7, 1), []),  # Columbus Day 的日期和时间戳列表为空
        (USColumbusDay, "2015-10-12", [Timestamp("2015-10-12")]),  # Columbus Day 的时间戳为 2015-10-12
        (USThanksgivingDay, datetime(2015, 7, 1), []),  # Thanksgiving Day 的日期和时间戳列表为空
        (USThanksgivingDay, "2015-11-26", [Timestamp("2015-11-26")]),  # Thanksgiving Day 的时间戳为 2015-11-26
        (USMartinLutherKingJr, datetime(2015, 7, 1), []),  # Martin Luther King Jr. Day 的日期和时间戳列表为空
        (USMartinLutherKingJr, "2015-01-19", [Timestamp("2015-01-19")]),  # Martin Luther King Jr. Day 的时间戳为 2015-01-19
        (USPresidentsDay, datetime(2015, 7, 1), []),  # Presidents Day 的日期和时间戳列表为空
        (USPresidentsDay, "2015-02-16", [Timestamp("2015-02-16")]),  # Presidents Day 的时间戳为 2015-02-16
        (GoodFriday, datetime(2015, 7, 1), []),  # Good Friday 的日期和时间戳列表为空
        (GoodFriday, "2015-04-03", [Timestamp("2015-04-03")]),  # Good Friday 的时间戳为 2015-04-03
        (EasterMonday, "2015-04-06", [Timestamp("2015-04-06")]),  # Easter Monday 的时间戳为 2015-04-06
        (EasterMonday, datetime(2015, 7, 1), []),  # Easter Monday 的日期和时间戳列表为空
        (EasterMonday, "2015-04-05", []),  # 对于 Easter Monday 的日期 2015-04-05，时间戳为空
        ("New Year's Day", "2015-01-01", [Timestamp("2015-01-01")]),  # 新年日的时间戳为 2015-01-01
        ("New Year's Day", "2010-12-31", [Timestamp("2010-12-31")]),  # 新年日的时间戳为 2010-12-31
        ("New Year's Day", datetime(2015, 7, 1), []),  # 新年日的日期和时间戳列表为空
        ("New Year's Day", "2011-01-01", []),  # 对于新年日的日期 2011-01-01，时间戳为空
        ("Independence Day", "2015-07-03", [Timestamp("2015-07-03")]),  # 独立日的时间戳为 2015-07-03
        ("Independence Day", datetime(2015, 7, 1), []),  # 独立日的日期和时间戳列表为空
        ("Independence Day", "2015-07-04", []),  # 对于独立日的日期 2015-07-04，时间戳为空
        ("Veterans Day", "2012-11-12", [Timestamp("2012-11-12")]),  # 退伍军人节的时间戳为 2012-11-12
        ("Veterans Day", datetime(2015, 7, 1), []),  # 退伍军人节的日期和时间戳列表为空
        ("Veterans Day", "2012-11-11", []),  # 对于退伍军人节的日期 2012-11-11，时间戳为空
        ("Christmas Day", "2011-12-26", [Timestamp("2011-12-26")]),  # 圣诞节的时间戳为 2011-12-26
        ("Christmas Day", datetime(2015, 7, 1), []),  # 圣诞节的日期和时间戳列表为空
        ("Christmas Day", "2011-12-25", []),  # 对于圣诞节的日期 2011-12-25，时间戳为空
        ("Juneteenth National Independence Day", "2020-06-19", []),  # 端午节国庆日的时间戳为空
        ("Juneteenth National Independence Day", "2021-06-18", [Timestamp("2021-06-18")]),  # 端午节国庆日的时间戳为 2021-06-18
        ("Juneteenth National Independence Day", "2022-06-19", []),  # 端午节国庆日的时间戳为空
        ("Juneteenth National Independence Day", "2022-06-20", [Timestamp("2022-06-20")]),  # 端午节国庆日的时间戳为 2022-06-20
    ],
# 检查节日在给定日期范围内的行为，验证修复情况
def test_holidays_within_dates(holiday, start, expected):
    # 见 issue gh-11477
    #
    # 修复节日的行为，使得 holiday.dates 返回在开始/结束日期之外的日期，
    # 或者不能应用观察规则，因为节日不在原始日期范围内（例如，7/4/2015 -> 7/3/2015）。

    # 如果 holiday 是字符串，转换为对应的日历对象
    if isinstance(holiday, str):
        calendar = get_calendar("USFederalHolidayCalendar")
        holiday = calendar.rule_from_name(holiday)

    # 断言节日在指定日期范围内的日期列表是否符合预期
    assert list(holiday.dates(start, start)) == expected

    # 验证时区信息是否保留
    assert list(
        holiday.dates(
            Timestamp(start, tz=timezone.utc), Timestamp(start, tz=timezone.utc)
        )
    ) == [dt.replace(tzinfo=timezone.utc) for dt in expected]


@pytest.mark.parametrize(
    "transform", [lambda x: x.strftime("%Y-%m-%d"), lambda x: Timestamp(x)]
)
# 测试不同类型的参数转换
def test_argument_types(transform):
    start_date = datetime(2011, 1, 1)
    end_date = datetime(2020, 12, 31)

    # 获取美国感恩节节日在给定日期范围内的日期列表
    holidays = USThanksgivingDay.dates(start_date, end_date)
    # 使用指定的转换函数获取节日在给定日期范围内的日期列表
    holidays2 = USThanksgivingDay.dates(transform(start_date), transform(end_date))
    # 断言两个日期列表是否相等
    tm.assert_index_equal(holidays, holidays2)


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("One-Time", {"year": 2012, "month": 5, "day": 28}),
        (
            "Range",
            {
                "month": 5,
                "day": 28,
                "start_date": datetime(2012, 1, 1),
                "end_date": datetime(2012, 12, 31),
                "offset": DateOffset(weekday=MO(1)),
            },
        ),
    ],
)
# 测试特殊的节日日期计算
def test_special_holidays(name, kwargs):
    base_date = [datetime(2012, 5, 28)]
    # 创建指定名称和参数的节日对象
    holiday = Holiday(name, **kwargs)

    start_date = datetime(2011, 1, 1)
    end_date = datetime(2020, 12, 31)

    # 断言特定节日在给定日期范围内的日期列表是否符合预期
    assert base_date == holiday.dates(start_date, end_date)


# 测试获取日历对象的方法
def test_get_calendar():
    # 定义一个测试用的日历类
    class TestCalendar(AbstractHolidayCalendar):
        rules = []

    # 获取指定名称的日历对象
    calendar = get_calendar("TestCalendar")
    # 断言返回的日历对象类型是否正确
    assert TestCalendar == type(calendar)


# 测试节日日历工厂方法
def test_factory():
    # 创建两个节日对象
    class_1 = HolidayCalendarFactory(
        "MemorialDay", AbstractHolidayCalendar, USMemorialDay
    )
    class_2 = HolidayCalendarFactory(
        "Thanksgiving", AbstractHolidayCalendar, USThanksgivingDay
    )
    # 组合两个节日对象
    class_3 = HolidayCalendarFactory("Combined", class_1, class_2)

    # 断言各个节日对象规则列表的长度是否符合预期
    assert len(class_1.rules) == 1
    assert len(class_2.rules) == 1
    assert len(class_3.rules) == 2


# 测试同时使用偏移和观察规则时的错误情况
def test_both_offset_observance_raises():
    # 见 issue gh-10217
    msg = "Cannot use both offset and observance"
    # 断言使用偏移和观察规则时会抛出 NotImplementedError 错误，并包含指定错误消息
    with pytest.raises(NotImplementedError, match=msg):
        Holiday(
            "Cyber Monday",
            month=11,
            day=1,
            offset=[DateOffset(weekday=SA(4))],
            observance=next_monday,
        )


# 测试禁止使用偏移的偏移列表的情况
def test_list_of_list_of_offsets_raises():
    # 见 issue gh-29049
    # 测试禁止使用偏移的偏移列表的情况
    # 创建名为 holiday1 的节日对象，表示美国感恩节
    holiday1 = Holiday(
        "Holiday1",
        month=USThanksgivingDay.month,  # 设置月份为感恩节的月份
        day=USThanksgivingDay.day,      # 设置日期为感恩节的日期
        offset=[USThanksgivingDay.offset, DateOffset(1)],  # 设置偏移量，包括感恩节和一个日期偏移
    )
    # 准备异常消息，用于检查偏移量的类型是否支持
    msg = "Only BaseOffsets and flat lists of them are supported for offset."
    # 使用 pytest 检查是否会引发 ValueError 异常，并验证异常消息
    with pytest.raises(ValueError, match=msg):
        # 创建名为 Holiday2 的节日对象，与 holiday1 具有相同的月份和日期
        # 但是偏移量包含 holiday1 的偏移量和一个额外的 DateOffset(3)，这应该触发异常
        Holiday(
            "Holiday2",
            month=holiday1.month,
            day=holiday1.day,
            offset=[holiday1.offset, DateOffset(3)],
        )
def test_half_open_interval_with_observance():
    # Prompted by GH 49075
    # Check for holidays that have a half-open date interval where
    # they have either a start_date or end_date defined along
    # with a defined observance pattern to make sure that the return type
    # for Holiday.dates() remains consistent before & after the year that
    # marks the 'edge' of the half-open date interval.

    # 创建一个名为 'Arbitrary Holiday - start 2022-03-14' 的节日对象，设置起始日期为2022年3月14日
    holiday_1 = Holiday(
        "Arbitrary Holiday - start 2022-03-14",
        start_date=datetime(2022, 3, 14),
        month=3,
        day=14,
        observance=next_monday,
    )

    # 创建一个名为 'Arbitrary Holiday 2 - end 2022-03-20' 的节日对象，设置结束日期为2022年3月20日
    holiday_2 = Holiday(
        "Arbitrary Holiday 2 - end 2022-03-20",
        end_date=datetime(2022, 3, 20),
        month=3,
        day=20,
        observance=next_monday,
    )

    # 定义一个测试用的节日日历类 TestHolidayCalendar，包含特定的节日规则
    class TestHolidayCalendar(AbstractHolidayCalendar):
        rules = [
            USMartinLutherKingJr,
            holiday_1,
            holiday_2,
            USLaborDay,
        ]

    # 设置开始日期为 '2022-08-01'
    start = Timestamp("2022-08-01")
    # 设置结束日期为 '2022-08-31'
    end = Timestamp("2022-08-31")
    # 设置年份偏移量为5年
    year_offset = DateOffset(years=5)
    # 定义预期结果为空的 DatetimeIndex
    expected_results = DatetimeIndex([], dtype="datetime64[ns]", freq=None)
    # 创建一个 TestHolidayCalendar 的实例
    test_cal = TestHolidayCalendar()

    # 计算日期区间为 start - year_offset 到 end - year_offset 的假日
    date_interval_low = test_cal.holidays(start - year_offset, end - year_offset)
    # 计算日期区间为 start 到 end 的假日
    date_window_edge = test_cal.holidays(start, end)
    # 计算日期区间为 start + year_offset 到 end + year_offset 的假日
    date_interval_high = test_cal.holidays(start + year_offset, end + year_offset)

    # 断言各日期区间计算结果与预期的空结果相等
    tm.assert_index_equal(date_interval_low, expected_results)
    tm.assert_index_equal(date_window_edge, expected_results)
    tm.assert_index_equal(date_interval_high, expected_results)


def test_holidays_with_timezone_specified_but_no_occurences():
    # GH 54580
    # _apply_rule() in holiday.py was silently dropping timezones if you passed it
    # an empty list of holiday dates that had timezone information

    # 设置起始日期为 '2018-01-01'，时区为 'America/Chicago'
    start_date = Timestamp("2018-01-01", tz="America/Chicago")
    # 设置结束日期为 '2018-01-11'，时区为 'America/Chicago'
    end_date = Timestamp("2018-01-11", tz="America/Chicago")
    # 调用 USFederalHolidayCalendar().holidays 方法计算指定日期范围内的节日，返回节日名称
    test_case = USFederalHolidayCalendar().holidays(
        start_date, end_date, return_name=True
    )
    # 创建一个预期结果 Series，包含一个索引为 start_date 的元素 'New Year's Day'
    expected_results = Series("New Year's Day", index=[start_date])
    # 将预期结果的索引单位转换为 'ns'
    expected_results.index = expected_results.index.as_unit("ns")

    # 断言计算结果与预期结果相等
    tm.assert_equal(test_case, expected_results)
```