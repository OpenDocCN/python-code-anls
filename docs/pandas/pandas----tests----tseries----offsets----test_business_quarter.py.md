# `D:\src\scipysrc\pandas\pandas\tests\tseries\offsets\test_business_quarter.py`

```
"""
Tests for the following offsets:
- BQuarterBegin
- BQuarterEnd
"""

# 引入必要的模块和库
from __future__ import annotations  # 使用 future 特性来支持注解
from datetime import datetime  # 引入 datetime 模块

import pytest  # 引入 pytest 测试框架

# 从 pandas 的测试模块中导入通用的时间序列偏移测试函数
from pandas.tests.tseries.offsets.common import (
    assert_is_on_offset,
    assert_offset_equal,
)

# 从 pandas 的时间序列偏移模块中导入所需的偏移类
from pandas.tseries.offsets import (
    BQuarterBegin,
    BQuarterEnd,
)


def test_quarterly_dont_normalize():
    # 定义一个日期时间对象
    date = datetime(2012, 3, 31, 5, 30)

    # 定义一个偏移类的元组
    offsets = (BQuarterEnd, BQuarterBegin)

    # 遍历偏移类元组
    for klass in offsets:
        # 计算日期时间对象加上当前偏移类的结果
        result = date + klass()
        # 断言结果的时间部分与原始日期时间对象的时间部分相同
        assert result.time() == date.time()


@pytest.mark.parametrize("offset", [BQuarterBegin(), BQuarterEnd()])
def test_on_offset(offset):
    # 生成一组日期时间对象列表
    dates = [
        datetime(2016, m, d)
        for m in [10, 11, 12]
        for d in [1, 2, 3, 28, 29, 30, 31]
        if not (m == 11 and d == 31)
    ]
    
    # 遍历日期时间对象列表
    for date in dates:
        # 调用偏移对象的 is_on_offset 方法检查日期是否在偏移上
        res = offset.is_on_offset(date)
        # 使用慢速版本检查日期是否在偏移上
        slow_version = date == (date + offset) - offset
        # 断言结果与慢速版本的一致性
        assert res == slow_version


class TestBQuarterBegin:
    def test_repr(self):
        # 测试偏移对象的字符串表示形式是否正确
        expected = "<BusinessQuarterBegin: startingMonth=3>"
        assert repr(BQuarterBegin()) == expected
        expected = "<BusinessQuarterBegin: startingMonth=3>"
        assert repr(BQuarterBegin(startingMonth=3)) == expected
        expected = "<BusinessQuarterBegin: startingMonth=1>"
        assert repr(BQuarterBegin(startingMonth=1)) == expected

    def test_offset_corner_case(self):
        # 测试特殊情况的偏移计算
        offset = BQuarterBegin(n=-1, startingMonth=1)
        assert datetime(2007, 4, 3) + offset == datetime(2007, 4, 2)

    # 定义偏移案例列表
    offset_cases = []
    offset_cases.append(
        (
            BQuarterBegin(startingMonth=1),  # 使用指定的开始月份创建偏移对象
            {
                datetime(2008, 1, 1): datetime(2008, 4, 1),
                datetime(2008, 1, 31): datetime(2008, 4, 1),
                datetime(2008, 2, 15): datetime(2008, 4, 1),
                datetime(2008, 2, 29): datetime(2008, 4, 1),
                datetime(2008, 3, 15): datetime(2008, 4, 1),
                datetime(2008, 3, 31): datetime(2008, 4, 1),
                datetime(2008, 4, 15): datetime(2008, 7, 1),
                datetime(2007, 3, 15): datetime(2007, 4, 2),
                datetime(2007, 2, 28): datetime(2007, 4, 2),
                datetime(2007, 1, 1): datetime(2007, 4, 2),
                datetime(2007, 4, 15): datetime(2007, 7, 2),
                datetime(2007, 7, 1): datetime(2007, 7, 2),
                datetime(2007, 4, 1): datetime(2007, 4, 2),
                datetime(2007, 4, 2): datetime(2007, 7, 2),
                datetime(2008, 4, 30): datetime(2008, 7, 1),
            },
        )
    )
    offset_cases.append(
        (  # 向 offset_cases 列表添加一个元组
            BQuarterBegin(startingMonth=2),  # 使用 BQuarterBegin 类创建一个实例，指定 startingMonth 参数为 2
            {  # 添加一个字典，表示日期映射关系
                datetime(2008, 1, 1): datetime(2008, 2, 1),  # 将 2008 年 1 月 1 日映射到 2008 年 2 月 1 日
                datetime(2008, 1, 31): datetime(2008, 2, 1),  # 将 2008 年 1 月 31 日映射到 2008 年 2 月 1 日
                datetime(2008, 1, 15): datetime(2008, 2, 1),  # 将 2008 年 1 月 15 日映射到 2008 年 2 月 1 日
                datetime(2008, 2, 29): datetime(2008, 5, 1),  # 将 2008 年 2 月 29 日映射到 2008 年 5 月 1 日
                datetime(2008, 3, 15): datetime(2008, 5, 1),  # 将 2008 年 3 月 15 日映射到 2008 年 5 月 1 日
                datetime(2008, 3, 31): datetime(2008, 5, 1),  # 将 2008 年 3 月 31 日映射到 2008 年 5 月 1 日
                datetime(2008, 4, 15): datetime(2008, 5, 1),  # 将 2008 年 4 月 15 日映射到 2008 年 5 月 1 日
                datetime(2008, 8, 15): datetime(2008, 11, 3),  # 将 2008 年 8 月 15 日映射到 2008 年 11 月 3 日
                datetime(2008, 9, 15): datetime(2008, 11, 3),  # 将 2008 年 9 月 15 日映射到 2008 年 11 月 3 日
                datetime(2008, 11, 1): datetime(2008, 11, 3),  # 将 2008 年 11 月 1 日映射到 2008 年 11 月 3 日
                datetime(2008, 4, 30): datetime(2008, 5, 1),  # 将 2008 年 4 月 30 日映射到 2008 年 5 月 1 日
            },
        )
    )

    offset_cases.append(
        (  # 向 offset_cases 列表添加一个元组
            BQuarterBegin(startingMonth=1, n=0),  # 使用 BQuarterBegin 类创建一个实例，指定 startingMonth 参数为 1，n 参数为 0
            {  # 添加一个字典，表示日期映射关系
                datetime(2008, 1, 1): datetime(2008, 1, 1),  # 将 2008 年 1 月 1 日映射到 2008 年 1 月 1 日
                datetime(2007, 12, 31): datetime(2008, 1, 1),  # 将 2007 年 12 月 31 日映射到 2008 年 1 月 1 日
                datetime(2008, 2, 15): datetime(2008, 4, 1),  # 将 2008 年 2 月 15 日映射到 2008 年 4 月 1 日
                datetime(2008, 2, 29): datetime(2008, 4, 1),  # 将 2008 年 2 月 29 日映射到 2008 年 4 月 1 日
                datetime(2008, 1, 15): datetime(2008, 4, 1),  # 将 2008 年 1 月 15 日映射到 2008 年 4 月 1 日
                datetime(2008, 2, 27): datetime(2008, 4, 1),  # 将 2008 年 2 月 27 日映射到 2008 年 4 月 1 日
                datetime(2008, 3, 15): datetime(2008, 4, 1),  # 将 2008 年 3 月 15 日映射到 2008 年 4 月 1 日
                datetime(2007, 4, 1): datetime(2007, 4, 2),  # 将 2007 年 4 月 1 日映射到 2007 年 4 月 2 日
                datetime(2007, 4, 2): datetime(2007, 4, 2),  # 将 2007 年 4 月 2 日映射到 2007 年 4 月 2 日
                datetime(2007, 7, 1): datetime(2007, 7, 2),  # 将 2007 年 7 月 1 日映射到 2007 年 7 月 2 日
                datetime(2007, 4, 15): datetime(2007, 7, 2),  # 将 2007 年 4 月 15 日映射到 2007 年 7 月 2 日
                datetime(2007, 7, 2): datetime(2007, 7, 2),  # 将 2007 年 7 月 2 日映射到 2007 年 7 月 2 日
            },
        )
    )

    offset_cases.append(
        (  # 向 offset_cases 列表添加一个元组
            BQuarterBegin(startingMonth=1, n=-1),  # 使用 BQuarterBegin 类创建一个实例，指定 startingMonth 参数为 1，n 参数为 -1
            {  # 添加一个字典，表示日期映射关系
                datetime(2008, 1, 1): datetime(2007, 10, 1),  # 将 2008 年 1 月 1 日映射到 2007 年 10 月 1 日
                datetime(2008, 1, 31): datetime(2008, 1, 1),  # 将 2008 年 1 月 31 日映射到 2008 年 1 月 1 日
                datetime(2008, 2, 15): datetime(2008, 1, 1),  # 将 2008 年 2 月 15 日映射到 2008 年 1 月 1 日
                datetime(2008, 2, 29): datetime(2008, 1, 1),  # 将 2008 年 2 月 29 日映射到 2008 年 1 月 1 日
                datetime(2008, 3, 15): datetime(2008, 1, 1),  # 将 2008 年 3 月 15 日映射到 2008 年 1 月 1 日
                datetime(2008, 3, 31): datetime(2008, 1, 1),  # 将 2008 年 3 月 31 日映射到 2008 年 1 月 1 日
                datetime(2008, 4, 15): datetime(2008, 4, 1),  # 将 2008 年 4 月 15 日映射到 2008 年 4 月 1 日
                datetime(2007, 7, 3): datetime(2007, 7, 2),  # 将 2007 年 7 月 3 日映射到 2007 年 7 月 2 日
                datetime(2007, 4, 3): datetime(2007, 4, 2),  # 将 2007 年
    offset_cases.append(
        (
            BQuarterBegin(startingMonth=1, n=2),  # 创建一个 BQuarterBegin 对象，表示从每季度开始的日期偏移，从1月开始，偏移2次
            {
                datetime(2008, 1, 1): datetime(2008, 7, 1),    # 2008年1月1日偏移后为2008年7月1日
                datetime(2008, 1, 15): datetime(2008, 7, 1),   # 2008年1月15日偏移后为2008年7月1日
                datetime(2008, 2, 29): datetime(2008, 7, 1),   # 2008年2月29日偏移后为2008年7月1日（因为是闰年2月，不会影响结果）
                datetime(2008, 3, 15): datetime(2008, 7, 1),   # 2008年3月15日偏移后为2008年7月1日
                datetime(2007, 3, 31): datetime(2007, 7, 2),   # 2007年3月31日偏移后为2007年7月2日
                datetime(2007, 4, 15): datetime(2007, 10, 1),  # 2007年4月15日偏移后为2007年10月1日
                datetime(2008, 4, 30): datetime(2008, 10, 1),  # 2008年4月30日偏移后为2008年10月1日
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)
class TestBQuarterEnd:
    # 测试 repr 方法是否返回预期的字符串表示形式
    def test_repr(self):
        expected = "<BusinessQuarterEnd: startingMonth=3>"
        # 断言默认设置下的 repr 结果是否符合预期
        assert repr(BQuarterEnd()) == expected
        expected = "<BusinessQuarterEnd: startingMonth=3>"
        # 断言设置起始月为3时的 repr 结果是否符合预期
        assert repr(BQuarterEnd(startingMonth=3)) == expected
        expected = "<BusinessQuarterEnd: startingMonth=1>"
        # 断言设置起始月为1时的 repr 结果是否符合预期
        assert repr(BQuarterEnd(startingMonth=1)) == expected

    # 测试在边界情况下的日期偏移计算
    def test_offset_corner_case(self):
        # 创建起始月为1，向前偏移一次的 BQuarterEnd 偏移对象
        offset = BQuarterEnd(n=-1, startingMonth=1)
        # 断言2010年1月31日加上偏移量后是否等于2010年1月29日
        assert datetime(2010, 1, 31) + offset == datetime(2010, 1, 29)

    offset_cases = []
    # 添加测试用例：起始月为1时的日期偏移测试
    offset_cases.append(
        (
            BQuarterEnd(startingMonth=1),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 4, 30),
                datetime(2008, 2, 15): datetime(2008, 4, 30),
                datetime(2008, 2, 29): datetime(2008, 4, 30),
                datetime(2008, 3, 15): datetime(2008, 4, 30),
                datetime(2008, 3, 31): datetime(2008, 4, 30),
                datetime(2008, 4, 15): datetime(2008, 4, 30),
                datetime(2008, 4, 30): datetime(2008, 7, 31),
            },
        )
    )

    # 添加测试用例：起始月为2时的日期偏移测试
    offset_cases.append(
        (
            BQuarterEnd(startingMonth=2),
            {
                datetime(2008, 1, 1): datetime(2008, 2, 29),
                datetime(2008, 1, 31): datetime(2008, 2, 29),
                datetime(2008, 2, 15): datetime(2008, 2, 29),
                datetime(2008, 2, 29): datetime(2008, 5, 30),
                datetime(2008, 3, 15): datetime(2008, 5, 30),
                datetime(2008, 3, 31): datetime(2008, 5, 30),
                datetime(2008, 4, 15): datetime(2008, 5, 30),
                datetime(2008, 4, 30): datetime(2008, 5, 30),
            },
        )
    )

    # 添加测试用例：起始月为1，无偏移的日期偏移测试
    offset_cases.append(
        (
            BQuarterEnd(startingMonth=1, n=0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 1, 31),
                datetime(2008, 2, 15): datetime(2008, 4, 30),
                datetime(2008, 2, 29): datetime(2008, 4, 30),
                datetime(2008, 3, 15): datetime(2008, 4, 30),
                datetime(2008, 3, 31): datetime(2008, 4, 30),
                datetime(2008, 4, 15): datetime(2008, 4, 30),
                datetime(2008, 4, 30): datetime(2008, 4, 30),
            },
        )
    )
    offset_cases.append(
        (
            BQuarterEnd(startingMonth=1, n=-1),
            {
                datetime(2008, 1, 1): datetime(2007, 10, 31),
                datetime(2008, 1, 31): datetime(2007, 10, 31),
                datetime(2008, 2, 15): datetime(2008, 1, 31),
                datetime(2008, 2, 29): datetime(2008, 1, 31),
                datetime(2008, 3, 15): datetime(2008, 1, 31),
                datetime(2008, 3, 31): datetime(2008, 1, 31),
                datetime(2008, 4, 15): datetime(2008, 1, 31),
                datetime(2008, 4, 30): datetime(2008, 1, 31),
            },
        )
    )

将一个包含偏移对象和测试用例的元组添加到`offset_cases`列表中，测试用例包含了基准日期和预期结果的字典。


    offset_cases.append(
        (
            BQuarterEnd(startingMonth=1, n=2),
            {
                datetime(2008, 1, 31): datetime(2008, 7, 31),
                datetime(2008, 2, 15): datetime(2008, 7, 31),
                datetime(2008, 2, 29): datetime(2008, 7, 31),
                datetime(2008, 3, 15): datetime(2008, 7, 31),
                datetime(2008, 3, 31): datetime(2008, 7, 31),
                datetime(2008, 4, 15): datetime(2008, 7, 31),
                datetime(2008, 4, 30): datetime(2008, 10, 31),
            },
        )
    )

将另一个包含不同偏移对象和测试用例的元组添加到`offset_cases`列表中，测试用例也包含了基准日期和预期结果的字典。


    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

定义一个使用`offset_cases`中测试用例的参数化测试函数`test_offset`，其中每个测试用例被解包为`offset`和`cases`，然后对每个测试用例中的基准日期和预期结果进行断言验证。
    # 定义测试用例列表，每个元素包含一个偏移对象、日期时间和预期布尔值
    on_offset_cases = [
        (BQuarterEnd(1, startingMonth=1), datetime(2008, 1, 31), True),   # 测试从一月份开始，日期为2008年1月31日，预期结果为True
        (BQuarterEnd(1, startingMonth=1), datetime(2007, 12, 31), False),  # 测试从一月份开始，日期为2007年12月31日，预期结果为False
        (BQuarterEnd(1, startingMonth=1), datetime(2008, 2, 29), False),  # 测试从一月份开始，日期为2008年2月29日（闰年），预期结果为False
        (BQuarterEnd(1, startingMonth=1), datetime(2007, 3, 30), False),  # 测试从一月份开始，日期为2007年3月30日，预期结果为False
        (BQuarterEnd(1, startingMonth=1), datetime(2007, 3, 31), False),  # 测试从一月份开始，日期为2007年3月31日，预期结果为False
        (BQuarterEnd(1, startingMonth=1), datetime(2008, 4, 30), True),   # 测试从一月份开始，日期为2008年4月30日，预期结果为True
        (BQuarterEnd(1, startingMonth=1), datetime(2008, 5, 30), False),  # 测试从一月份开始，日期为2008年5月30日，预期结果为False
        (BQuarterEnd(1, startingMonth=1), datetime(2007, 6, 29), False),  # 测试从一月份开始，日期为2007年6月29日，预期结果为False
        (BQuarterEnd(1, startingMonth=1), datetime(2007, 6, 30), False),  # 测试从一月份开始，日期为2007年6月30日，预期结果为False
        (BQuarterEnd(1, startingMonth=2), datetime(2008, 1, 31), False),  # 测试从二月份开始，日期为2008年1月31日，预期结果为False
        (BQuarterEnd(1, startingMonth=2), datetime(2007, 12, 31), False),  # 测试从二月份开始，日期为2007年12月31日，预期结果为False
        (BQuarterEnd(1, startingMonth=2), datetime(2008, 2, 29), True),   # 测试从二月份开始，日期为2008年2月29日（闰年），预期结果为True
        (BQuarterEnd(1, startingMonth=2), datetime(2007, 3, 30), False),  # 测试从二月份开始，日期为2007年3月30日，预期结果为False
        (BQuarterEnd(1, startingMonth=2), datetime(2007, 3, 31), False),  # 测试从二月份开始，日期为2007年3月31日，预期结果为False
        (BQuarterEnd(1, startingMonth=2), datetime(2008, 4, 30), False),  # 测试从二月份开始，日期为2008年4月30日，预期结果为False
        (BQuarterEnd(1, startingMonth=2), datetime(2008, 5, 30), True),   # 测试从二月份开始，日期为2008年5月30日，预期结果为True
        (BQuarterEnd(1, startingMonth=2), datetime(2007, 6, 29), False),  # 测试从二月份开始，日期为2007年6月29日，预期结果为False
        (BQuarterEnd(1, startingMonth=2), datetime(2007, 6, 30), False),  # 测试从二月份开始，日期为2007年6月30日，预期结果为False
        (BQuarterEnd(1, startingMonth=3), datetime(2008, 1, 31), False),  # 测试从三月份开始，日期为2008年1月31日，预期结果为False
        (BQuarterEnd(1, startingMonth=3), datetime(2007, 12, 31), True),   # 测试从三月份开始，日期为2007年12月31日，预期结果为True
        (BQuarterEnd(1, startingMonth=3), datetime(2008, 2, 29), False),  # 测试从三月份开始，日期为2008年2月29日（闰年），预期结果为False
        (BQuarterEnd(1, startingMonth=3), datetime(2007, 3, 30), True),   # 测试从三月份开始，日期为2007年3月30日，预期结果为True
        (BQuarterEnd(1, startingMonth=3), datetime(2007, 3, 31), False),  # 测试从三月份开始，日期为2007年3月31日，预期结果为False
        (BQuarterEnd(1, startingMonth=3), datetime(2008, 4, 30), False),  # 测试从三月份开始，日期为2008年4月30日，预期结果为False
        (BQuarterEnd(1, startingMonth=3), datetime(2008, 5, 30), False),  # 测试从三月份开始，日期为2008年5月30日，预期结果为False
        (BQuarterEnd(1, startingMonth=3), datetime(2007, 6, 29), True),   # 测试从三月份开始，日期为2007年6月29日，预期结果为True
        (BQuarterEnd(1, startingMonth=3), datetime(2007, 6, 30), False),  # 测试从三月份开始，日期为2007年6月30日，预期结果为False
    ]
    
    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)
```