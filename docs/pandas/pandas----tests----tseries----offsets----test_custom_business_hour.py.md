# `D:\src\scipysrc\pandas\pandas\tests\tseries\offsets\test_custom_business_hour.py`

```
"""
Tests for offsets.CustomBusinessHour
"""

# 从 __future__ 模块导入 annotations 特性，使得函数能够支持类型注解
from __future__ import annotations

# 导入 datetime 模块中的 datetime 和 time 类
from datetime import (
    datetime,
    time as dt_time,
)

# 导入 numpy 库，并将其重命名为 np
import numpy as np

# 导入 pytest 库，用于编写和运行测试用例
import pytest

# 从 pandas._libs.tslibs 中导入 Timestamp 类
from pandas._libs.tslibs import Timestamp

# 从 pandas._libs.tslibs.offsets 中导入 BusinessHour 和 CustomBusinessHour 类
from pandas._libs.tslibs.offsets import (
    BusinessHour,
    CustomBusinessHour,
    Nano,
)

# 从 pandas.tests.tseries.offsets.common 中导入 assert_offset_equal 函数
from pandas.tests.tseries.offsets.common import assert_offset_equal

# 从 pandas.tseries.holiday 中导入 USFederalHolidayCalendar 类
from pandas.tseries.holiday import USFederalHolidayCalendar

# 定义一个列表 holidays，包含了三个假日的日期
holidays = ["2014-06-27", datetime(2014, 6, 30), np.datetime64("2014-07-02")]

# 定义一个 pytest 的 fixture，返回一个 datetime 对象
@pytest.fixture
def dt():
    return datetime(2014, 7, 1, 10, 00)

# 定义一个 pytest 的 fixture，返回 CustomBusinessHour 类
@pytest.fixture
def _offset():
    return CustomBusinessHour

# 定义一个 pytest 的 fixture，返回自定义的 CustomBusinessHour 对象，设置 weekmask
@pytest.fixture
def offset1():
    return CustomBusinessHour(weekmask="Tue Wed Thu Fri")

# 定义一个 pytest 的 fixture，返回自定义的 CustomBusinessHour 对象，设置 holidays
@pytest.fixture
def offset2():
    return CustomBusinessHour(holidays=holidays)

# 定义一个测试类 TestCustomBusinessHour
class TestCustomBusinessHour:
    
    # 测试 CustomBusinessHour 构造函数的错误情况
    def test_constructor_errors(self):
        msg = "time data must be specified only with hour and minute"
        with pytest.raises(ValueError, match=msg):
            CustomBusinessHour(start=dt_time(11, 0, 5))
        msg = "time data must match '%H:%M' format"
        with pytest.raises(ValueError, match=msg):
            CustomBusinessHour(start="AAA")
        msg = "time data must match '%H:%M' format"
        with pytest.raises(ValueError, match=msg):
            CustomBusinessHour(start="14:00:05")

    # 测试 normalize 参数不同导致的 __eq__ 方法返回结果不同的情况
    def test_different_normalize_equals(self, _offset):
        offset = _offset()
        offset2 = _offset(normalize=True)
        assert offset != offset2

    # 测试 CustomBusinessHour 的字符串表示形式
    def test_repr(self, offset1, offset2):
        assert repr(offset1) == "<CustomBusinessHour: cbh=09:00-17:00>"
        assert repr(offset2) == "<CustomBusinessHour: cbh=09:00-17:00>"

    # 测试 CustomBusinessHour 与时间偏移的加法操作
    def test_with_offset(self, dt):
        expected = Timestamp("2014-07-01 13:00")

        assert dt + CustomBusinessHour() * 3 == expected
        assert dt + CustomBusinessHour(n=3) == expected

    # 测试 CustomBusinessHour 的相等性比较
    def test_eq(self, offset1, offset2):
        for offset in [offset1, offset2]:
            assert offset == offset

        assert CustomBusinessHour() != CustomBusinessHour(-1)
        assert CustomBusinessHour(start="09:00") == CustomBusinessHour()
        assert CustomBusinessHour(start="09:00") != CustomBusinessHour(start="09:01")
        assert CustomBusinessHour(start="09:00", end="17:00") != CustomBusinessHour(
            start="17:00", end="09:01"
        )

        assert CustomBusinessHour(weekmask="Tue Wed Thu Fri") != CustomBusinessHour(
            weekmask="Mon Tue Wed Thu Fri"
        )
        assert CustomBusinessHour(holidays=["2014-06-27"]) != CustomBusinessHour(
            holidays=["2014-06-28"]
        )
    # 测试哈希函数，确保对同一对象的哈希值相同
    def test_hash(self, offset1, offset2):
        assert hash(offset1) == hash(offset1)
        assert hash(offset2) == hash(offset2)

    # 测试日期时间对象的加法操作
    def test_add_dateime(self, dt, offset1, offset2):
        # 确保偏移量加上日期时间对象后的结果与预期的日期时间相同
        assert offset1 + dt == datetime(2014, 7, 1, 11)
        assert offset2 + dt == datetime(2014, 7, 1, 11)

    # 测试回滚功能1
    def testRollback1(self, dt, offset1, offset2):
        # 确保偏移量对象能够正确回滚到给定的日期时间
        assert offset1.rollback(dt) == dt
        assert offset2.rollback(dt) == dt

        d = datetime(2014, 7, 1, 0)

        # 2014年7月1日是星期二，回滚到6月27日下午5点的日期时间
        assert offset1.rollback(d) == datetime(2014, 6, 27, 17)

        # 2014年6月30日和6月27日都是节假日，回滚到6月26日下午5点的日期时间
        assert offset2.rollback(d) == datetime(2014, 6, 26, 17)

    # 测试回滚功能2
    def testRollback2(self, _offset):
        # 确保自定义偏移量对象能够正确回滚到给定的日期时间
        assert _offset(-3).rollback(datetime(2014, 7, 5, 15, 0)) == datetime(
            2014, 7, 4, 17, 0
        )

    # 测试前进功能1
    def testRollforward1(self, dt, offset1, offset2):
        # 确保偏移量对象能够正确前进到给定的日期时间
        assert offset1.rollforward(dt) == dt
        assert offset2.rollforward(dt) == dt

        d = datetime(2014, 7, 1, 0)
        
        # 2014年7月1日前进到上午9点的日期时间
        assert offset1.rollforward(d) == datetime(2014, 7, 1, 9)
        assert offset2.rollforward(d) == datetime(2014, 7, 1, 9)

    # 测试前进功能2
    def testRollforward2(self, _offset):
        # 确保自定义偏移量对象能够正确前进到给定的日期时间
        assert _offset(-3).rollforward(datetime(2014, 7, 5, 16, 0)) == datetime(
            2014, 7, 7, 9
        )

    # 测试日期对象的回滚和前进
    def test_roll_date_object(self):
        offset = BusinessHour()

        dt = datetime(2014, 7, 6, 15, 0)

        # 确保偏移量对象能够正确回滚到给定的日期时间
        result = offset.rollback(dt)
        assert result == datetime(2014, 7, 4, 17)

        # 确保偏移量对象能够正确前进到给定的日期时间
        result = offset.rollforward(dt)
        assert result == datetime(2014, 7, 7, 9)
    # 定义了三个不同的 CustomBusinessHour 实例及其对应的测试用例字典
    normalize_cases = [
        (
            CustomBusinessHour(normalize=True, holidays=holidays),
            {
                datetime(2014, 7, 1, 8): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 3),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 3),
                datetime(2014, 7, 1, 23): datetime(2014, 7, 3),
                datetime(2014, 7, 1, 0): datetime(2014, 7, 1),
                datetime(2014, 7, 4, 15): datetime(2014, 7, 4),
                datetime(2014, 7, 4, 15, 59): datetime(2014, 7, 4),
                datetime(2014, 7, 4, 16, 30): datetime(2014, 7, 7),
                datetime(2014, 7, 5, 23): datetime(2014, 7, 7),
                datetime(2014, 7, 6, 10): datetime(2014, 7, 7),
            },
        ),
        (
            CustomBusinessHour(-1, normalize=True, holidays=holidays),
            {
                datetime(2014, 7, 1, 8): datetime(2014, 6, 26),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 10): datetime(2014, 6, 26),
                datetime(2014, 7, 1, 0): datetime(2014, 6, 26),
                datetime(2014, 7, 7, 10): datetime(2014, 7, 4),
                datetime(2014, 7, 7, 10, 1): datetime(2014, 7, 7),
                datetime(2014, 7, 5, 23): datetime(2014, 7, 4),
                datetime(2014, 7, 6, 10): datetime(2014, 7, 4),
            },
        ),
        (
            CustomBusinessHour(
                1, normalize=True, start="17:00", end="04:00", holidays=holidays
            ),
            {
                datetime(2014, 7, 1, 8): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 23): datetime(2014, 7, 2),
                datetime(2014, 7, 2, 2): datetime(2014, 7, 2),
                datetime(2014, 7, 2, 3): datetime(2014, 7, 3),
                datetime(2014, 7, 4, 23): datetime(2014, 7, 5),
                datetime(2014, 7, 5, 2): datetime(2014, 7, 5),
                datetime(2014, 7, 7, 2): datetime(2014, 7, 7),
                datetime(2014, 7, 7, 17): datetime(2014, 7, 7),
            },
        ),
    ]

    # 使用 pytest 的 parametrize 装饰器，传入 normalize_cases 数组作为参数
    @pytest.mark.parametrize("norm_cases", normalize_cases)
    # 定义测试方法 test_normalize，使用传入的 norm_cases 作为参数
    def test_normalize(self, norm_cases):
        # 将参数 norm_cases 解构为 offset 和 cases 两部分
        offset, cases = norm_cases
        # 遍历 cases 字典中的每个键值对
        for dt, expected in cases.items():
            # 断言 CustomBusinessHour 实例 offset 的 _apply 方法对 dt 的应用结果等于预期的 expected
            assert offset._apply(dt) == expected

    # 使用 pytest 的 parametrize 装饰器，传入 dt 和 expected 数组作为参数
    @pytest.mark.parametrize(
        "dt, expected",
        [
            [datetime(2014, 7, 1, 9), False],
            [datetime(2014, 7, 1, 10), True],
            [datetime(2014, 7, 1, 15), True],
            [datetime(2014, 7, 1, 15, 1), False],
            [datetime(2014, 7, 5, 12), False],
            [datetime(2014, 7, 6, 12), False],
        ],
    )
    # 定义测试方法，用于验证给定的时间偏移是否正确
    def test_is_on_offset(self, dt, expected):
        # 创建自定义的工作小时对象，设置起始时间为 "10:00" 到结束时间 "15:00"，并包含假期列表
        offset = CustomBusinessHour(start="10:00", end="15:00", holidays=holidays)
        # 断言给定日期时间是否符合预期的时间偏移
        assert offset.is_on_offset(dt) == expected

    # 定义多组测试案例，每组案例包含自定义工作小时对象及其对应的日期时间偏移映射
    apply_cases = [
        (
            CustomBusinessHour(holidays=holidays),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 1, 12),
                datetime(2014, 7, 1, 13): datetime(2014, 7, 1, 14),
                datetime(2014, 7, 1, 15): datetime(2014, 7, 1, 16),
                datetime(2014, 7, 1, 19): datetime(2014, 7, 3, 10),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 3, 9),
                datetime(2014, 7, 1, 16, 30, 15): datetime(2014, 7, 3, 9, 30, 15),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 3, 10),
                datetime(2014, 7, 2, 11): datetime(2014, 7, 3, 10),
                # 非工作小时
                datetime(2014, 7, 2, 8): datetime(2014, 7, 3, 10),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 3, 10),
                datetime(2014, 7, 2, 23): datetime(2014, 7, 3, 10),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 3, 10),
                # 星期六
                datetime(2014, 7, 5, 15): datetime(2014, 7, 7, 10),
                datetime(2014, 7, 4, 17): datetime(2014, 7, 7, 10),
                datetime(2014, 7, 4, 16, 30): datetime(2014, 7, 7, 9, 30),
                datetime(2014, 7, 4, 16, 30, 30): datetime(2014, 7, 7, 9, 30, 30),
            },
        ),
        (
            CustomBusinessHour(4, holidays=holidays),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 1, 15),
                datetime(2014, 7, 1, 13): datetime(2014, 7, 3, 9),
                datetime(2014, 7, 1, 15): datetime(2014, 7, 3, 11),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 3, 12),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 2, 11): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 2, 23): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 7, 13),
                datetime(2014, 7, 4, 17): datetime(2014, 7, 7, 13),
                datetime(2014, 7, 4, 16, 30): datetime(2014, 7, 7, 12, 30),
                datetime(2014, 7, 4, 16, 30, 30): datetime(2014, 7, 7, 12, 30, 30),
            },
        ),
    ]

    # 使用参数化测试，循环执行每组测试案例，并验证偏移后的日期时间是否符合预期
    @pytest.mark.parametrize("apply_case", apply_cases)
    def test_apply(self, apply_case):
        # 从测试案例中获取偏移对象及其对应的日期时间偏移映射
        offset, cases = apply_case
        # 遍历每个日期时间偏移映射，断言偏移后的日期时间是否与预期相等
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)
    nano_cases = [
        (
            CustomBusinessHour(holidays=holidays),  # 创建一个自定义的工作时间对象，指定了假日
            {
                Timestamp("2014-07-01 15:00") + Nano(5): Timestamp("2014-07-01 16:00")
                + Nano(5),  # 设置特定日期时间的偏移量和预期的结果时间
                Timestamp("2014-07-01 16:00") + Nano(5): Timestamp("2014-07-03 09:00")
                + Nano(5),  # 设置另一个特定日期时间的偏移量和预期的结果时间
                Timestamp("2014-07-01 16:00") - Nano(5): Timestamp("2014-07-01 17:00")
                - Nano(5),  # 设置另一个特定日期时间的偏移量和预期的结果时间
            },
        ),
        (
            CustomBusinessHour(-1, holidays=holidays),  # 创建一个自定义的工作时间对象，指定了负偏移和假日
            {
                Timestamp("2014-07-01 15:00") + Nano(5): Timestamp("2014-07-01 14:00")
                + Nano(5),  # 设置特定日期时间的偏移量和预期的结果时间
                Timestamp("2014-07-01 10:00") + Nano(5): Timestamp("2014-07-01 09:00")
                + Nano(5),  # 设置另一个特定日期时间的偏移量和预期的结果时间
                Timestamp("2014-07-01 10:00") - Nano(5): Timestamp("2014-06-26 17:00")
                - Nano(5),  # 设置另一个特定日期时间的偏移量和预期的结果时间
            },
        ),
    ]

    @pytest.mark.parametrize("nano_case", nano_cases)
    def test_apply_nanoseconds(self, nano_case):
        offset, cases = nano_case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    def test_us_federal_holiday_with_datetime(self):
        # GH 16867
        bhour_us = CustomBusinessHour(calendar=USFederalHolidayCalendar())  # 创建一个自定义的工作时间对象，使用了美国联邦假日日历
        t0 = datetime(2014, 1, 17, 15)  # 定义一个特定的日期时间
        result = t0 + bhour_us * 8  # 计算工作时间偏移量的结果
        expected = Timestamp("2014-01-21 15:00:00")  # 预期的结果时间戳
        assert result == expected  # 断言计算结果与预期结果相等
@pytest.mark.parametrize(
    "weekmask, expected_time, mult",
    [
        ["Mon Tue Wed Thu Fri Sat", "2018-11-10 09:00:00", 10],  # 参数化测试数据1：周掩码为工作日，期望时间为2018年11月10日09:00:00，倍数为10
        ["Tue Wed Thu Fri Sat", "2018-11-13 08:00:00", 18],     # 参数化测试数据2：周掩码去除周一，期望时间为2018年11月13日08:00:00，倍数为18
    ],
)
def test_custom_businesshour_weekmask_and_holidays(weekmask, expected_time, mult):
    # GH 23542
    # 标记：GH 23542，指示此测试解决了GitHub上的问题编号23542
    
    holidays = ["2018-11-09"]  # 定义假期列表，包含2018年11月9日
    
    # 创建自定义工作小时对象bh，设定工作时间从08:00到17:00，使用给定的周掩码和假期列表
    bh = CustomBusinessHour(
        start="08:00", end="17:00", weekmask=weekmask, holidays=holidays
    )
    
    # 计算预期结果时间，起始时间为2018年11月8日08:00，加上倍数乘以自定义工作小时对象bh的结果
    result = Timestamp("2018-11-08 08:00") + mult * bh
    
    expected = Timestamp(expected_time)  # 预期结果为给定的日期时间字符串转换为Timestamp对象
    
    assert result == expected  # 断言计算结果等于预期结果
```