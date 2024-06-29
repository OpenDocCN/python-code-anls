# `D:\src\scipysrc\pandas\pandas\tests\tseries\offsets\test_business_year.py`

```
"""
Tests for the following offsets:
- BYearBegin
- BYearEnd
"""

# 导入必要的模块
from __future__ import annotations
from datetime import datetime
import pytest
from pandas.tests.tseries.offsets.common import (
    assert_is_on_offset,
    assert_offset_equal,
)
from pandas.tseries.offsets import (
    BYearBegin,
    BYearEnd,
)

# 测试类 TestBYearBegin
class TestBYearBegin:
    # 测试月份输入错误的情况
    def test_misspecified(self):
        msg = "Month must go from 1 to 12"
        with pytest.raises(ValueError, match=msg):
            BYearBegin(month=13)
        with pytest.raises(ValueError, match=msg):
            BYearEnd(month=13)

    # 定义不同情况下的偏移量和预期结果
    offset_cases = []
    offset_cases.append(
        (
            BYearBegin(),
            {
                datetime(2008, 1, 1): datetime(2009, 1, 1),
                datetime(2008, 6, 30): datetime(2009, 1, 1),
                datetime(2008, 12, 31): datetime(2009, 1, 1),
                datetime(2011, 1, 1): datetime(2011, 1, 3),
                datetime(2011, 1, 3): datetime(2012, 1, 2),
                datetime(2005, 12, 30): datetime(2006, 1, 2),
                datetime(2005, 12, 31): datetime(2006, 1, 2),
            },
        )
    )

    offset_cases.append(
        (
            BYearBegin(0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 1),
                datetime(2008, 6, 30): datetime(2009, 1, 1),
                datetime(2008, 12, 31): datetime(2009, 1, 1),
                datetime(2005, 12, 30): datetime(2006, 1, 2),
                datetime(2005, 12, 31): datetime(2006, 1, 2),
            },
        )
    )

    offset_cases.append(
        (
            BYearBegin(-1),
            {
                datetime(2007, 1, 1): datetime(2006, 1, 2),
                datetime(2009, 1, 4): datetime(2009, 1, 1),
                datetime(2009, 1, 1): datetime(2008, 1, 1),
                datetime(2008, 6, 30): datetime(2008, 1, 1),
                datetime(2008, 12, 31): datetime(2008, 1, 1),
                datetime(2006, 12, 29): datetime(2006, 1, 2),
                datetime(2006, 12, 30): datetime(2006, 1, 2),
                datetime(2006, 1, 1): datetime(2005, 1, 3),
            },
        )
    )

    offset_cases.append(
        (
            BYearBegin(-2),
            {
                datetime(2007, 1, 1): datetime(2005, 1, 3),
                datetime(2007, 6, 30): datetime(2006, 1, 2),
                datetime(2008, 12, 31): datetime(2007, 1, 1),
            },
        )
    )

    # 参数化测试用例
    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

# 测试类 TestBYearEnd
class TestBYearEnd:
    offset_cases = []
    offset_cases.append(
        (
            BYearEnd(),
            {
                datetime(2008, 1, 1): datetime(2008, 12, 31),
                datetime(2008, 6, 30): datetime(2008, 12, 31),
                datetime(2008, 12, 31): datetime(2009, 12, 31),
                datetime(2005, 12, 30): datetime(2006, 12, 29),
                datetime(2005, 12, 31): datetime(2006, 12, 29),
            },
        )
    )

# 将 BYearEnd 实例化后作为第一个元素，与一个包含不同日期与对应偏移日期的字典组成元组，添加到 offset_cases 列表中。


    offset_cases.append(
        (
            BYearEnd(0),
            {
                datetime(2008, 1, 1): datetime(2008, 12, 31),
                datetime(2008, 6, 30): datetime(2008, 12, 31),
                datetime(2008, 12, 31): datetime(2008, 12, 31),
                datetime(2005, 12, 31): datetime(2006, 12, 29),
            },
        )
    )

# 将 BYearEnd(0) 实例化后作为第一个元素，与一个包含不同日期与对应偏移日期的字典组成元组，添加到 offset_cases 列表中。


    offset_cases.append(
        (
            BYearEnd(-1),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 29),
                datetime(2008, 6, 30): datetime(2007, 12, 31),
                datetime(2008, 12, 31): datetime(2007, 12, 31),
                datetime(2006, 12, 29): datetime(2005, 12, 30),
                datetime(2006, 12, 30): datetime(2006, 12, 29),
                datetime(2007, 1, 1): datetime(2006, 12, 29),
            },
        )
    )

# 将 BYearEnd(-1) 实例化后作为第一个元素，与一个包含不同日期与对应偏移日期的字典组成元组，添加到 offset_cases 列表中。


    offset_cases.append(
        (
            BYearEnd(-2),
            {
                datetime(2007, 1, 1): datetime(2005, 12, 30),
                datetime(2008, 6, 30): datetime(2006, 12, 29),
                datetime(2008, 12, 31): datetime(2006, 12, 29),
            },
        )
    )

# 将 BYearEnd(-2) 实例化后作为第一个元素，与一个包含不同日期与对应偏移日期的字典组成元组，添加到 offset_cases 列表中。


    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

# 使用 pytest 的 parametrize 装饰器为 test_offset 函数添加参数化测试。对于每个 case，从元组中解包 offset 和 cases，然后对 cases 中的每个日期与期望日期进行断言比较。


    on_offset_cases = [
        (BYearEnd(), datetime(2007, 12, 31), True),
        (BYearEnd(), datetime(2008, 1, 1), False),
        (BYearEnd(), datetime(2006, 12, 31), False),
        (BYearEnd(), datetime(2006, 12, 29), True),
    ]

# 创建一个包含多个元组的列表 on_offset_cases，每个元组包含 BYearEnd 实例、一个日期和一个布尔值，用于测试 assert_is_on_offset 函数。


    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)

# 使用 pytest 的 parametrize 装饰器为 test_is_on_offset 函数添加参数化测试。对于每个 case，从元组中解包 offset、dt 和 expected，然后测试 assert_is_on_offset 函数的输出是否符合预期。
# 定义一个测试类 TestBYearEndLagged，用于测试 BYearEnd 类的行为
class TestBYearEndLagged:

    # 测试当传入错误的月份时是否会引发 ValueError 异常
    def test_bad_month_fail(self):
        msg = "Month must go from 1 to 12"
        
        # 断言当传入月份为 13 时会引发 ValueError 异常，并且异常消息匹配特定信息
        with pytest.raises(ValueError, match=msg):
            BYearEnd(month=13)
        
        # 断言当传入月份为 0 时会引发 ValueError 异常，并且异常消息匹配特定信息
        with pytest.raises(ValueError, match=msg):
            BYearEnd(month=0)

    # 定义一个空列表 offset_cases，用于存储不同的测试案例
    offset_cases = []

    # 向 offset_cases 列表添加一个测试案例元组
    offset_cases.append(
        (
            BYearEnd(month=6),  # 创建一个 BYearEnd 对象，设定月份为 6
            {
                datetime(2008, 1, 1): datetime(2008, 6, 30),  # 预期将 2008 年 1 月 1 日偏移至 2008 年 6 月 30 日
                datetime(2007, 6, 30): datetime(2008, 6, 30),  # 预期将 2007 年 6 月 30 日偏移至 2008 年 6 月 30 日
            },
        )
    )

    # 向 offset_cases 列表添加另一个测试案例元组
    offset_cases.append(
        (
            BYearEnd(n=-1, month=6),  # 创建一个 BYearEnd 对象，设定月份为 6，年份偏移量为 -1
            {
                datetime(2008, 1, 1): datetime(2007, 6, 29),  # 预期将 2008 年 1 月 1 日偏移至 2007 年 6 月 29 日
                datetime(2007, 6, 30): datetime(2007, 6, 29),  # 预期将 2007 年 6 月 30 日偏移至 2007 年 6 月 29 日
            },
        )
    )

    # 使用 pytest 的参数化装饰器，依次执行 offset_cases 列表中的每个测试案例
    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        
        # 遍历 cases 字典，对每个基准日期进行偏移测试
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    # 测试 BYearEnd 类的 rollforward 和 rollback 方法
    def test_roll(self):
        offset = BYearEnd(month=6)  # 创建一个 BYearEnd 对象，设定月份为 6
        date = datetime(2009, 11, 30)  # 创建一个日期对象，表示 2009 年 11 月 30 日

        # 断言 rollforward 方法将日期对象向前滚动到下一个偏移日期
        assert offset.rollforward(date) == datetime(2010, 6, 30)
        
        # 断言 rollback 方法将日期对象向后滚动到上一个偏移日期
        assert offset.rollback(date) == datetime(2009, 6, 30)

    # 定义一个列表 on_offset_cases，用于存储测试 BYearEnd 类的 is_on_offset 方法的案例
    on_offset_cases = [
        (BYearEnd(month=2), datetime(2007, 2, 28), True),   # 创建 BYearEnd 对象，设定月份为 2，检查日期 2007 年 2 月 28 日是否处于偏移日期上
        (BYearEnd(month=6), datetime(2007, 6, 30), False),  # 创建 BYearEnd 对象，设定月份为 6，检查日期 2007 年 6 月 30 日是否处于偏移日期上
    ]

    # 使用 pytest 的参数化装饰器，依次执行 on_offset_cases 列表中的每个测试案例
    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        
        # 断言 is_on_offset 方法返回的结果与预期值一致
        assert_is_on_offset(offset, dt, expected)
```