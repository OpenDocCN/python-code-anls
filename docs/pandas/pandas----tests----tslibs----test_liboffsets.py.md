# `D:\src\scipysrc\pandas\pandas\tests\tslibs\test_liboffsets.py`

```
"""
Tests for helper functions in the cython tslibs.offsets
"""

# 从 datetime 模块中导入 datetime 类
from datetime import datetime

# 导入 pytest 模块
import pytest

# 从 pandas._libs.tslibs.ccalendar 模块中导入指定函数
from pandas._libs.tslibs.ccalendar import (
    get_firstbday,
    get_lastbday,
)

# 导入 pandas._libs.tslibs.offsets 模块
import pandas._libs.tslibs.offsets as liboffsets

# 从 pandas._libs.tslibs.offsets 模块中导入 roll_qtrday 函数
from pandas._libs.tslibs.offsets import roll_qtrday

# 从 pandas 模块中导入 Timestamp 类
from pandas import Timestamp


# 定义 pytest fixture，返回测试用例参数中的不同选项
@pytest.fixture(params=["start", "end", "business_start", "business_end"])
def day_opt(request):
    return request.param


# 使用 pytest.mark.parametrize 装饰器，定义测试函数 test_get_last_bday 的参数化测试
@pytest.mark.parametrize(
    "dt,exp_week_day,exp_last_day",
    [
        (datetime(2017, 11, 30), 3, 30),  # Business day.
        (datetime(1993, 10, 31), 6, 29),  # Non-business day.
    ],
)
# 测试获取上一个工作日函数 get_lastbday
def test_get_last_bday(dt, exp_week_day, exp_last_day):
    assert dt.weekday() == exp_week_day
    assert get_lastbday(dt.year, dt.month) == exp_last_day


# 使用 pytest.mark.parametrize 装饰器，定义测试函数 test_get_first_bday 的参数化测试
@pytest.mark.parametrize(
    "dt,exp_week_day,exp_first_day",
    [
        (datetime(2017, 4, 1), 5, 3),  # Non-weekday.
        (datetime(1993, 10, 1), 4, 1),  # Business day.
    ],
)
# 测试获取下一个工作日函数 get_firstbday
def test_get_first_bday(dt, exp_week_day, exp_first_day):
    assert dt.weekday() == exp_week_day
    assert get_firstbday(dt.year, dt.month) == exp_first_day


# 使用 pytest.mark.parametrize 装饰器，定义测试函数 test_shift_month_dt 的参数化测试
@pytest.mark.parametrize(
    "months,day_opt,expected",
    [
        (0, 15, datetime(2017, 11, 15)),
        (0, None, datetime(2017, 11, 30)),
        (1, "start", datetime(2017, 12, 1)),
        (-145, "end", datetime(2005, 10, 31)),
        (0, "business_end", datetime(2017, 11, 30)),
        (0, "business_start", datetime(2017, 11, 1)),
    ],
)
# 测试偏移月份函数 shift_month
def test_shift_month_dt(months, day_opt, expected):
    dt = datetime(2017, 11, 30)
    assert liboffsets.shift_month(dt, months, day_opt=day_opt) == expected


# 使用 pytest.mark.parametrize 装饰器，定义测试函数 test_shift_month_ts 的参数化测试
@pytest.mark.parametrize(
    "months,day_opt,expected",
    [
        (1, "start", Timestamp("1929-06-01")),
        (-3, "end", Timestamp("1929-02-28")),
        (25, None, Timestamp("1931-06-5")),
        (-1, 31, Timestamp("1929-04-30")),
    ],
)
# 测试偏移时间戳月份函数 shift_month
def test_shift_month_ts(months, day_opt, expected):
    ts = Timestamp("1929-05-05")
    assert liboffsets.shift_month(ts, months, day_opt=day_opt) == expected


# 测试偏移月份函数 shift_month 在输入错误参数时是否抛出 ValueError 异常
def test_shift_month_error():
    dt = datetime(2017, 11, 15)
    day_opt = "this should raise"

    with pytest.raises(ValueError, match=day_opt):
        liboffsets.shift_month(dt, 3, day_opt=day_opt)


# 使用 pytest.mark.parametrize 装饰器，定义测试函数 test_roll_qtrday_year 的参数化测试
@pytest.mark.parametrize(
    "other,expected",
    [
        # Before March 1.
        (datetime(2017, 2, 10), {2: 1, -7: -7, 0: 0}),
        # After March 1.
        (Timestamp("2014-03-15", tz="US/Eastern"), {2: 2, -7: -6, 0: 1}),
    ],
)
# 使用 pytest.mark.parametrize 装饰器，进一步参数化测试函数 test_roll_qtrday_year
@pytest.mark.parametrize("n", [2, -7, 0])
# 测试季度日滚动函数 roll_qtrday
def test_roll_qtrday_year(other, expected, n):
    month = 3
    day_opt = "start"  # `other` will be compared to March 1.

    assert roll_qtrday(other, n, month, day_opt, modby=12) == expected[n]
    [
        # 6月30日之前的日期，创建一个元组，包含日期对象和相应的字典
        (datetime(1999, 6, 29), {5: 4, -7: -7, 0: 0}),
        # 6月30日之后的时间戳，创建一个元组，包含时间戳对象和相应的字典
        (Timestamp(2072, 8, 24, 6, 17, 18), {5: 5, -7: -6, 0: 1}),
    ],
@pytest.mark.parametrize("n", [5, -7, 0])
# 使用 pytest 的 parametrize 装饰器，定义了参数化测试，参数 n 分别为 5、-7、0
def test_roll_qtrday_year2(other, expected, n):
    # 设定月份为 6
    month = 6
    # day_opt 设置为 "end"，用于将 `other` 与 6 月 30 日进行比较

    # 断言调用 roll_qtrday 函数，传入参数 other, n, month, day_opt 和 modby=12，检查返回值是否等于 expected[n]
    assert roll_qtrday(other, n, month, day_opt, modby=12) == expected[n]


def test_get_day_of_month_error():
    # get_day_of_month 没有直接暴露在外部接口中
    # 我们通过 roll_qtrday 函数来测试它
    dt = datetime(2017, 11, 15)
    day_opt = "foo"

    # 使用 pytest.raises 检查 ValueError 异常是否被抛出，并且匹配异常消息为 day_opt 的值
    with pytest.raises(ValueError, match=day_opt):
        # 为了触发异常情况，需要满足 month == dt.month 且 n > 0
        roll_qtrday(dt, n=3, month=11, day_opt=day_opt, modby=12)


@pytest.mark.parametrize("month", [3, 5])
# 使用 pytest 的 parametrize 装饰器，定义了参数化测试，参数 month 分别为 3、5
@pytest.mark.parametrize("n", [4, -3])
# 使用 pytest 的 parametrize 装饰器，定义了参数化测试，参数 n 分别为 4、-3
def test_roll_qtr_day_not_mod_unequal(day_opt, month, n):
    # 预期结果字典
    expected = {3: {-3: -2, 4: 4}, 5: {-3: -3, 4: 3}}

    # 设定 other 为 Timestamp 对象，日期为 2072 年 10 月 1 日，星期六
    other = Timestamp(2072, 10, 1, 6, 17, 18)
    
    # 断言调用 roll_qtrday 函数，传入参数 other, n, month, day_opt 和 modby=3，检查返回值是否等于预期的值
    assert roll_qtrday(other, n, month, day_opt, modby=3) == expected[month][n]


@pytest.mark.parametrize(
    "other,month,exp_dict",
    [
        # 星期一
        (datetime(1999, 5, 31), 2, {-1: {"start": 0, "business_start": 0}}),
        # 星期六
        (
            Timestamp(2072, 10, 1, 6, 17, 18),
            4,
            {2: {"end": 1, "business_end": 1, "business_start": 1}},
        ),
        # 第一个工作日
        (
            Timestamp(2072, 10, 3, 6, 17, 18),
            4,
            {2: {"end": 1, "business_end": 1}, -1: {"start": 0}},
        ),
    ],
)
# 使用 pytest 的 parametrize 装饰器，定义了多个参数化测试场景，对 other, month, exp_dict 进行参数化
@pytest.mark.parametrize("n", [2, -1])
# 使用 pytest 的 parametrize 装饰器，定义了参数化测试，参数 n 分别为 2、-1
def test_roll_qtr_day_mod_equal(other, month, exp_dict, n, day_opt):
    # 预期结果
    expected = exp_dict.get(n, {}).get(day_opt, n)
    
    # 断言调用 roll_qtrday 函数，传入参数 other, n, month, day_opt 和 modby=3，检查返回值是否等于预期的值
    assert roll_qtrday(other, n, month, day_opt, modby=3) == expected


@pytest.mark.parametrize(
    "n,expected", [(42, {29: 42, 1: 42, 31: 41}), (-4, {29: -4, 1: -3, 31: -4})]
)
# 使用 pytest 的 parametrize 装饰器，定义了参数化测试，参数 n 分别为 42、-4，expected 是一个字典列表
@pytest.mark.parametrize("compare", [29, 1, 31])
# 使用 pytest 的 parametrize 装饰器，定义了参数化测试，参数 compare 分别为 29、1、31
def test_roll_convention(n, expected, compare):
    # 断言调用 liboffsets.roll_convention 函数，传入参数 29, n, compare，检查返回值是否等于 expected[compare]
    assert liboffsets.roll_convention(29, n, compare) == expected[compare]
```