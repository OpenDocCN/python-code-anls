# `D:\src\scipysrc\pandas\pandas\tests\tseries\holiday\test_observance.py`

```
# 导入需要的模块和函数
from datetime import datetime

# 导入 pytest 模块，用于测试
import pytest

# 从 pandas 库中导入日期处理相关的函数
from pandas.tseries.holiday import (
    after_nearest_workday,
    before_nearest_workday,
    nearest_workday,
    next_monday,
    next_monday_or_tuesday,
    next_workday,
    previous_friday,
    previous_workday,
    sunday_to_monday,
    weekend_to_monday,
)

# 定义一些预先设置好的日期常量
_WEDNESDAY = datetime(2014, 4, 9)
_THURSDAY = datetime(2014, 4, 10)
_FRIDAY = datetime(2014, 4, 11)
_SATURDAY = datetime(2014, 4, 12)
_SUNDAY = datetime(2014, 4, 13)
_MONDAY = datetime(2014, 4, 14)
_TUESDAY = datetime(2014, 4, 15)
_NEXT_WEDNESDAY = datetime(2014, 4, 16)


# 使用 pytest.mark.parametrize 装饰器定义测试函数，测试 next_monday 函数
@pytest.mark.parametrize("day", [_SATURDAY, _SUNDAY])
def test_next_monday(day):
    assert next_monday(day) == _MONDAY


# 使用 pytest.mark.parametrize 装饰器定义测试函数，测试 next_monday_or_tuesday 函数
@pytest.mark.parametrize(
    "day,expected", [(_SATURDAY, _MONDAY), (_SUNDAY, _TUESDAY), (_MONDAY, _TUESDAY)]
)
def test_next_monday_or_tuesday(day, expected):
    assert next_monday_or_tuesday(day) == expected


# 使用 pytest.mark.parametrize 装饰器定义测试函数，测试 previous_friday 函数
@pytest.mark.parametrize("day", [_SATURDAY, _SUNDAY])
def test_previous_friday(day):
    assert previous_friday(day) == _FRIDAY


# 测试 sunday_to_monday 函数
def test_sunday_to_monday():
    assert sunday_to_monday(_SUNDAY) == _MONDAY


# 使用 pytest.mark.parametrize 装饰器定义测试函数，测试 nearest_workday 函数
@pytest.mark.parametrize(
    "day,expected", [(_SATURDAY, _FRIDAY), (_SUNDAY, _MONDAY), (_MONDAY, _MONDAY)]
)
def test_nearest_workday(day, expected):
    assert nearest_workday(day) == expected


# 使用 pytest.mark.parametrize 装饰器定义测试函数，测试 weekend_to_monday 函数
@pytest.mark.parametrize(
    "day,expected", [(_SATURDAY, _MONDAY), (_SUNDAY, _MONDAY), (_MONDAY, _MONDAY)]
)
def test_weekend_to_monday(day, expected):
    assert weekend_to_monday(day) == expected


# 使用 pytest.mark.parametrize 装饰器定义测试函数，测试 next_workday 函数
@pytest.mark.parametrize(
    "day,expected",
    [
        (_WEDNESDAY, _THURSDAY),
        (_THURSDAY, _FRIDAY),
        (_SATURDAY, _MONDAY),
        (_SUNDAY, _MONDAY),
        (_MONDAY, _TUESDAY),
        (_TUESDAY, _NEXT_WEDNESDAY),  # WED is same week as TUE
    ],
)
def test_next_workday(day, expected):
    assert next_workday(day) == expected


# 使用 pytest.mark.parametrize 装饰器定义测试函数，测试 previous_workday 函数
@pytest.mark.parametrize(
    "day,expected", [(_SATURDAY, _FRIDAY), (_SUNDAY, _FRIDAY), (_TUESDAY, _MONDAY)]
)
def test_previous_workday(day, expected):
    assert previous_workday(day) == expected


# 使用 pytest.mark.parametrize 装饰器定义测试函数，测试 before_nearest_workday 函数
@pytest.mark.parametrize(
    "day,expected",
    [
        (_THURSDAY, _WEDNESDAY),
        (_FRIDAY, _THURSDAY),
        (_SATURDAY, _THURSDAY),
        (_SUNDAY, _FRIDAY),
        (_MONDAY, _FRIDAY),  # last week Friday
        (_TUESDAY, _MONDAY),
        (_NEXT_WEDNESDAY, _TUESDAY),  # WED is same week as TUE
    ],
)
def test_before_nearest_workday(day, expected):
    assert before_nearest_workday(day) == expected


# 使用 pytest.mark.parametrize 装饰器定义测试函数，测试 after_nearest_workday 函数
@pytest.mark.parametrize(
    "day,expected", [(_SATURDAY, _MONDAY), (_SUNDAY, _TUESDAY), (_FRIDAY, _MONDAY)]
)
def test_after_nearest_workday(day, expected):
    assert after_nearest_workday(day) == expected
```