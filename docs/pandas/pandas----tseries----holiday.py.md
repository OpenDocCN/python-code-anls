# `D:\src\scipysrc\pandas\pandas\tseries\holiday.py`

```
from __future__ import annotations
# 导入未来版本的注解支持

from datetime import (
    datetime,  # 导入 datetime 类
    timedelta,  # 导入 timedelta 类
)
from typing import TYPE_CHECKING  # 导入类型检查标记
import warnings  # 导入警告模块

from dateutil.relativedelta import (  # 导入相对日期模块
    FR,  # 星期五的相对日期
    MO,  # 星期一的相对日期
    SA,  # 星期六的相对日期
    SU,  # 星期日的相对日期
    TH,  # 星期四的相对日期
    TU,  # 星期二的相对日期
    WE,  # 星期三的相对日期
)
import numpy as np  # 导入 NumPy 模块

from pandas._libs.tslibs.offsets import BaseOffset  # 导入 pandas 基本偏移量类
from pandas.errors import PerformanceWarning  # 导入性能警告类

from pandas import (  # 导入 pandas 中的多个类和函数
    DateOffset,  # 日期偏移类
    DatetimeIndex,  # 日期时间索引类
    Series,  # 系列类
    Timestamp,  # 时间戳类
    concat,  # 连接函数
    date_range,  # 生成日期范围函数
)

from pandas.tseries.offsets import (  # 导入 pandas 时间序列偏移类
    Day,  # 天偏移类
    Easter,  # 复活节偏移类
)

if TYPE_CHECKING:
    from collections.abc import Callable  # 如果是类型检查模式，导入可调用对象类

def next_monday(dt: datetime) -> datetime:
    """
    If holiday falls on Saturday, use following Monday instead;
    if holiday falls on Sunday, use Monday instead
    """
    if dt.weekday() == 5:  # 如果日期是星期六
        return dt + timedelta(2)  # 返回下周一
    elif dt.weekday() == 6:  # 如果日期是星期日
        return dt + timedelta(1)  # 返回下周一
    return dt  # 否则返回原日期

def next_monday_or_tuesday(dt: datetime) -> datetime:
    """
    For second holiday of two adjacent ones!
    If holiday falls on Saturday, use following Monday instead;
    if holiday falls on Sunday or Monday, use following Tuesday instead
    (because Monday is already taken by adjacent holiday on the day before)
    """
    dow = dt.weekday()  # 获取日期的星期几
    if dow in (5, 6):  # 如果日期是星期六或星期日
        return dt + timedelta(2)  # 返回下周二
    if dow == 0:  # 如果日期是星期一
        return dt + timedelta(1)  # 返回下周二
    return dt  # 否则返回原日期

def previous_friday(dt: datetime) -> datetime:
    """
    If holiday falls on Saturday or Sunday, use previous Friday instead.
    """
    if dt.weekday() == 5:  # 如果日期是星期六
        return dt - timedelta(1)  # 返回上周五
    elif dt.weekday() == 6:  # 如果日期是星期日
        return dt - timedelta(2)  # 返回上周五
    return dt  # 否则返回原日期

def sunday_to_monday(dt: datetime) -> datetime:
    """
    If holiday falls on Sunday, use day thereafter (Monday) instead.
    """
    if dt.weekday() == 6:  # 如果日期是星期日
        return dt + timedelta(1)  # 返回下周一
    return dt  # 否则返回原日期

def weekend_to_monday(dt: datetime) -> datetime:
    """
    If holiday falls on Sunday or Saturday,
    use day thereafter (Monday) instead.
    Needed for holidays such as Christmas observation in Europe
    """
    if dt.weekday() == 6:  # 如果日期是星期日
        return dt + timedelta(1)  # 返回下周一
    elif dt.weekday() == 5:  # 如果日期是星期六
        return dt + timedelta(2)  # 返回下周一
    return dt  # 否则返回原日期

def nearest_workday(dt: datetime) -> datetime:
    """
    If holiday falls on Saturday, use day before (Friday) instead;
    if holiday falls on Sunday, use day thereafter (Monday) instead.
    """
    if dt.weekday() == 5:  # 如果日期是星期六
        return dt - timedelta(1)  # 返回上周五
    elif dt.weekday() == 6:  # 如果日期是星期日
        return dt + timedelta(1)  # 返回下周一
    return dt  # 否则返回原日期

def next_workday(dt: datetime) -> datetime:
    """
    returns next workday used for observances
    """
    dt += timedelta(days=1)  # 增加一天
    while dt.weekday() > 4:  # 如果是周末（星期五、六）
        dt += timedelta(days=1)  # 继续增加一天
    return dt  # 返回下一个工作日的日期

def previous_workday(dt: datetime) -> datetime:
    """
    returns previous workday used for observances
    """
    dt -= timedelta(days=1)  # 减少一天
    while dt.weekday() > 4:  # 如果是周末（星期五、六）
        dt -= timedelta(days=1)  # 继续减少一天
    return dt  # 返回前一个工作日的日期
    return dt
# 返回最近工作日之前的上一个工作日
def before_nearest_workday(dt: datetime) -> datetime:
    return previous_workday(nearest_workday(dt))


# 返回最近工作日之后的下一个工作日
# 适用于圣诞节或连续多个节假日
def after_nearest_workday(dt: datetime) -> datetime:
    return next_workday(nearest_workday(dt))


class Holiday:
    """
    表示一个假期的类，包括开始/结束日期和遵守的规则。
    """

    start_date: Timestamp | None
    end_date: Timestamp | None
    days_of_week: tuple[int, ...] | None

    def __init__(
        self,
        name: str,
        year=None,
        month=None,
        day=None,
        offset: BaseOffset | list[BaseOffset] | None = None,
        observance: Callable | None = None,
        start_date=None,
        end_date=None,
        days_of_week: tuple | None = None,
    ):
        """
        初始化假期对象。
        参数：
        - name: 假期名称
        - year, month, day: 假期的年、月、日
        - offset: 偏移量或偏移列表，指定日期的特定偏移量
        - observance: 观察规则，定义假期如何被观察
        - start_date, end_date: 假期的开始和结束日期
        - days_of_week: 在假期期间重复的工作日列表
        """
        ...

    def __repr__(self) -> str:
        """
        返回对象的字符串表示形式。
        """
        info = ""
        if self.year is not None:
            info += f"year={self.year}, "
        info += f"month={self.month}, day={self.day}, "

        if self.offset is not None:
            info += f"offset={self.offset}"

        if self.observance is not None:
            info += f"observance={self.observance}"

        repr = f"Holiday: {self.name} ({info})"
        return repr

    def dates(
        self, start_date, end_date, return_name: bool = False
    ):
        """
        返回给定日期范围内的假期日期列表。
        参数：
        - start_date: 开始日期
        - end_date: 结束日期
        - return_name: 如果为True，返回假期名称；默认为False
        """
        ...
    ) -> Series | DatetimeIndex:
        """
        Calculate holidays observed between start date and end date

        Parameters
        ----------
        start_date : starting date, datetime-like, optional
        end_date : ending date, datetime-like, optional
        return_name : bool, optional, default=False
            If True, return a series that has dates and holiday names.
            False will only return dates.

        Returns
        -------
        Series or DatetimeIndex
            Series if return_name is True
        """
        # 将开始日期和结束日期转换为 Timestamp 对象
        start_date = Timestamp(start_date)
        end_date = Timestamp(end_date)

        # 设置过滤起始日期和结束日期的初始值
        filter_start_date = start_date
        filter_end_date = end_date

        # 如果设置了特定的年份，返回该年的指定日期作为节假日
        if self.year is not None:
            # 创建一个特定日期的 Timestamp 对象
            dt = Timestamp(datetime(self.year, self.month, self.day))
            # 将该日期放入 DatetimeIndex 中
            dti = DatetimeIndex([dt])
            # 如果需要返回节假日名称，返回一个带有名称的 Series
            if return_name:
                return Series(self.name, index=dti)
            else:
                # 否则，只返回 DatetimeIndex
                return dti

        # 根据设定的规则计算起始日期和结束日期之间的所有参考日期
        dates = self._reference_dates(start_date, end_date)
        # 应用节假日规则，获取节假日日期
        holiday_dates = self._apply_rule(dates)

        # 如果限制了节假日只能在特定星期几，则进行过滤
        if self.days_of_week is not None:
            holiday_dates = holiday_dates[
                np.isin(
                    # 错误： "DatetimeIndex" 对象没有 "dayofweek" 属性
                    holiday_dates.dayofweek,  # type: ignore[attr-defined]
                    self.days_of_week,
                ).ravel()
            ]

        # 如果设置了起始日期的限制，取最大值
        if self.start_date is not None:
            filter_start_date = max(
                self.start_date.tz_localize(filter_start_date.tz), filter_start_date
            )
        # 如果设置了结束日期的限制，取最小值
        if self.end_date is not None:
            filter_end_date = min(
                self.end_date.tz_localize(filter_end_date.tz), filter_end_date
            )

        # 根据过滤后的起始日期和结束日期，进一步筛选节假日日期
        holiday_dates = holiday_dates[
            (holiday_dates >= filter_start_date) & (holiday_dates <= filter_end_date)
        ]

        # 如果需要返回节假日名称，返回一个带有名称的 Series
        if return_name:
            return Series(self.name, index=holiday_dates)
        # 否则，只返回节假日日期
        return holiday_dates
    def dates(self) -> DatetimeIndex:
        """
        Get reference dates for the holiday.

        Return reference dates for the holiday also returning the year
        prior to the start_date and year following the end_date.  This ensures
        that any offsets to be applied will yield the holidays within
        the passed in dates.
        """
        # 如果有指定起始日期，则将起始日期本地化为与 self.start_date 相同的时区
        if self.start_date is not None:
            start_date = self.start_date.tz_localize(start_date.tz)

        # 如果有指定结束日期，则将结束日期本地化为与 self.start_date 相同的时区
        if self.end_date is not None:
            end_date = self.end_date.tz_localize(start_date.tz)

        # 定义一个年份偏移量对象，增加/减少1年
        year_offset = DateOffset(years=1)

        # 计算参考起始日期，即当前月份和日的前一年
        reference_start_date = Timestamp(
            datetime(start_date.year - 1, self.month, self.day)
        )

        # 计算参考结束日期，即当前月份和日的后一年
        reference_end_date = Timestamp(
            datetime(end_date.year + 1, self.month, self.day)
        )

        # 生成一个日期范围，从参考起始日期到参考结束日期，按照年份偏移量增加
        # 使用 start_date 的时区
        dates = date_range(
            start=reference_start_date,
            end=reference_end_date,
            freq=year_offset,
            tz=start_date.tz,
        )

        return dates

    def _apply_rule(self, dates: DatetimeIndex) -> DatetimeIndex:
        """
        Apply the given offset/observance to a DatetimeIndex of dates.

        Parameters
        ----------
        dates : DatetimeIndex
            Dates to apply the given offset/observance rule

        Returns
        -------
        Dates with rules applied
        """
        # 如果传入的日期集合为空，则直接返回其副本
        if dates.empty:
            return dates.copy()

        # 如果定义了 observance 函数，则对每个日期应用该函数
        if self.observance is not None:
            return dates.map(lambda d: self.observance(d))

        # 如果定义了 offset，则依次对每个日期应用偏移量
        if self.offset is not None:
            if not isinstance(self.offset, list):
                offsets = [self.offset]
            else:
                offsets = self.offset
            for offset in offsets:
                # 如果正在添加一个非向量化的值，忽略性能警告
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", PerformanceWarning)
                    dates += offset
        return dates
# 创建一个空字典，用于存储节假日日历类的注册信息
holiday_calendars = {}

# 注册节假日日历类到全局字典 holiday_calendars 中
def register(cls) -> None:
    try:
        name = cls.name
    except AttributeError:
        name = cls.__name__
    holiday_calendars[name] = cls

# 根据名称从 holiday_calendars 字典中获取并返回对应的节假日日历类的实例
def get_calendar(name: str) -> AbstractHolidayCalendar:
    """
    Return an instance of a calendar based on its name.

    Parameters
    ----------
    name : str
        Calendar name to return an instance of
    """
    return holiday_calendars[name]()

# 定义一个元类 HolidayCalendarMetaClass，用于自动注册每个子类到 holiday_calendars 字典中
class HolidayCalendarMetaClass(type):
    def __new__(cls, clsname: str, bases, attrs):
        calendar_class = super().__new__(cls, clsname, bases, attrs)
        register(calendar_class)
        return calendar_class

# 定义一个抽象节假日日历类 AbstractHolidayCalendar，使用 HolidayCalendarMetaClass 元类
class AbstractHolidayCalendar(metaclass=HolidayCalendarMetaClass):
    """
    Abstract interface to create holidays following certain rules.
    """

    # 节假日规则列表，默认为空
    rules: list[Holiday] = []
    # 起始日期，默认为 1970 年 1 月 1 日的 Timestamp 对象
    start_date = Timestamp(datetime(1970, 1, 1))
    # 结束日期，默认为 2200 年 12 月 31 日的 Timestamp 对象
    end_date = Timestamp(datetime(2200, 12, 31))
    # 缓存，初始为 None
    _cache = None

    # 初始化方法，用于设置节假日日历的名称和规则
    def __init__(self, name: str = "", rules=None) -> None:
        """
        Initializes holiday object with a given set a rules.  Normally
        classes just have the rules defined within them.

        Parameters
        ----------
        name : str
            Name of the holiday calendar, defaults to class name
        rules : array of Holiday objects
            A set of rules used to create the holidays.
        """
        super().__init__()
        # 如果未提供名称，则使用类名作为名称
        if not name:
            name = type(self).__name__
        self.name = name

        # 如果提供了规则，则使用提供的规则列表
        if rules is not None:
            self.rules = rules

    # 根据规则名称查找并返回对应的节假日规则对象，如果找不到则返回 None
    def rule_from_name(self, name: str) -> Holiday | None:
        for rule in self.rules:
            if rule.name == name:
                return rule

        return None
    def holidays(self, start=None, end=None, return_name: bool = False):
        """
        Returns a curve with holidays between start_date and end_date

        Parameters
        ----------
        start : starting date, datetime-like, optional
            The start date for querying holidays. If not provided, defaults to the calendar's start date.
        end : ending date, datetime-like, optional
            The end date for querying holidays. If not provided, defaults to the calendar's end date.
        return_name : bool, optional
            If True, return a series that has dates and holiday names.
            False will only return a DatetimeIndex of dates.

        Returns
        -------
        DatetimeIndex of holidays
        """
        if self.rules is None:
            raise Exception(
                f"Holiday Calendar {self.name} does not have any rules specified"
            )

        if start is None:
            start = AbstractHolidayCalendar.start_date

        if end is None:
            end = AbstractHolidayCalendar.end_date

        start = Timestamp(start)
        end = Timestamp(end)

        # If we don't have a cache or the dates are outside the prior cache, we
        # get them again
        if self._cache is None or start < self._cache[0] or end > self._cache[1]:
            # Retrieve precomputed holidays from each rule in self.rules
            pre_holidays = [
                rule.dates(start, end, return_name=True) for rule in self.rules
            ]
            if pre_holidays:
                # Concatenate the list of precomputed holidays into a single series
                holidays = concat(pre_holidays)  # type: ignore[arg-type]
            else:
                # If no holidays are found, create an empty Series
                holidays = Series(index=DatetimeIndex([]), dtype=object)  # type: ignore[assignment]

            # Update the cache with the retrieved holidays
            self._cache = (start, end, holidays.sort_index())

        # Extract holidays within the specified start and end dates
        holidays = self._cache[2]
        holidays = holidays[start:end]

        # Return either the holiday dates or dates along with their names based on return_name flag
        if return_name:
            return holidays
        else:
            return holidays.index
    def merge_class(base, other):
        """
        合并假期日历。基础日历将优先于其他日历。合并将根据每个假期的名称进行。

        Parameters
        ----------
        base : AbstractHolidayCalendar
          实例/子类或假期对象数组
        other : AbstractHolidayCalendar
          实例/子类或假期对象数组
        """
        try:
            other = other.rules  # 尝试获取其他日历的规则
        except AttributeError:
            pass

        if not isinstance(other, list):
            other = [other]  # 如果其他不是列表，则转换为列表
        other_holidays = {holiday.name: holiday for holiday in other}  # 创建其他日历假期名称到假期对象的映射

        try:
            base = base.rules  # 尝试获取基础日历的规则
        except AttributeError:
            pass

        if not isinstance(base, list):
            base = [base]  # 如果基础不是列表，则转换为列表
        base_holidays = {holiday.name: holiday for holiday in base}  # 创建基础日历假期名称到假期对象的映射

        other_holidays.update(base_holidays)  # 合并基础和其他日历的假期映射
        return list(other_holidays.values())  # 返回合并后的假期对象列表

    def merge(self, other, inplace: bool = False):
        """
        合并假期日历。调用者的类规则优先。合并将根据每个假期的名称进行。

        Parameters
        ----------
        other : holiday calendar
          假期日历对象
        inplace : bool (default=False)
            如果为True，则将规则设置为假期，否则返回假期数组
        """
        holidays = self.merge_class(self, other)  # 调用merge_class方法合并假期日历
        if inplace:
            self.rules = holidays  # 如果inplace为True，则设置调用者的规则为合并后的假期
        else:
            return holidays  # 否则返回合并后的假期数组
USMemorialDay = Holiday(
    "Memorial Day", month=5, day=31, offset=DateOffset(weekday=MO(-1))
)
# 创建 Memorial Day 的节日对象，月份为 5，日期为 31 日（如果是周一则返回这一周的上一个周一）

USLaborDay = Holiday("Labor Day", month=9, day=1, offset=DateOffset(weekday=MO(1)))
# 创建 Labor Day 的节日对象，月份为 9，日期为 1 日（如果是周一则返回这一周的下一个周一）

USColumbusDay = Holiday(
    "Columbus Day", month=10, day=1, offset=DateOffset(weekday=MO(2))
)
# 创建 Columbus Day 的节日对象，月份为 10，日期为 1 日（如果是周一则返回这一周的下一个周二）

USThanksgivingDay = Holiday(
    "Thanksgiving Day", month=11, day=1, offset=DateOffset(weekday=TH(4))
)
# 创建 Thanksgiving Day 的节日对象，月份为 11，日期为 1 日（如果是周四则返回这一周的第四个周四）

USMartinLutherKingJr = Holiday(
    "Birthday of Martin Luther King, Jr.",
    start_date=datetime(1986, 1, 1),
    month=1,
    day=1,
    offset=DateOffset(weekday=MO(3)),
)
# 创建 Martin Luther King Jr. 生日的节日对象，起始日期为 1986 年 1 月 1 日，月份为 1，日期为 1 日（如果是周一则返回这一周的第三个周一）

USPresidentsDay = Holiday(
    "Washington's Birthday", month=2, day=1, offset=DateOffset(weekday=MO(3))
)
# 创建 Presidents' Day 的节日对象，月份为 2，日期为 1 日（如果是周一则返回这一周的第三个周一）

GoodFriday = Holiday("Good Friday", month=1, day=1, offset=[Easter(), Day(-2)])
# 创建 Good Friday 的节日对象，月份为 1，日期为 1 日，日期偏移为复活节前两天

EasterMonday = Holiday("Easter Monday", month=1, day=1, offset=[Easter(), Day(1)])
# 创建 Easter Monday 的节日对象，月份为 1，日期为 1 日，日期偏移为复活节后一天

class USFederalHolidayCalendar(AbstractHolidayCalendar):
    """
    US Federal Government Holiday Calendar based on rules specified by:
    https://www.opm.gov/policy-data-oversight/pay-leave/federal-holidays/
    """

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        USMemorialDay,
        Holiday(
            "Juneteenth National Independence Day",
            month=6,
            day=19,
            start_date="2021-06-18",
            observance=nearest_workday,
        ),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USColumbusDay,
        Holiday("Veterans Day", month=11, day=11, observance=nearest_workday),
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]
    # 定义了美国联邦政府的节假日日历，遵循 https://www.opm.gov/policy-data-oversight/pay-leave/federal-holidays/ 上的规则

def HolidayCalendarFactory(name: str, base, other, base_class=AbstractHolidayCalendar):
    rules = AbstractHolidayCalendar.merge_class(base, other)
    # 将基本和其他规则合并成一个规则列表
    calendar_class = type(name, (base_class,), {"rules": rules, "name": name})
    # 创建一个新的节假日日历类，继承自 base_class，使用合并后的规则列表和给定的名称
    return calendar_class

__all__ = [
    "after_nearest_workday",
    "before_nearest_workday",
    "FR",
    "get_calendar",
    "HolidayCalendarFactory",
    "MO",
    "nearest_workday",
    "next_monday",
    "next_monday_or_tuesday",
    "next_workday",
    "previous_friday",
    "previous_workday",
    "register",
    "SA",
    "SU",
    "sunday_to_monday",
    "TH",
    "TU",
    "WE",
    "weekend_to_monday",
]
# 模块导出的公共接口列表，包括函数和常量等
```