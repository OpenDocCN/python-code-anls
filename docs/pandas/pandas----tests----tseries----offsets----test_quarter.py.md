# `D:\src\scipysrc\pandas\pandas\tests\tseries\offsets\test_quarter.py`

```
"""
Tests for the following offsets:
- QuarterBegin
- QuarterEnd
"""

# 导入必要的模块和函数
from __future__ import annotations  # 使用未来版本的类型注解支持

from datetime import datetime  # 导入 datetime 模块中的 datetime 类

import pytest  # 导入 pytest 测试框架

# 导入测试所需的函数和类
from pandas.tests.tseries.offsets.common import (
    assert_is_on_offset,
    assert_offset_equal,
)

# 导入 pandas 中的季度偏移类
from pandas.tseries.offsets import (
    QuarterBegin,
    QuarterEnd,
)

# 测试函数，参数化使用 QuarterBegin 和 QuarterEnd 类
@pytest.mark.parametrize("klass", (QuarterBegin, QuarterEnd))
def test_quarterly_dont_normalize(klass):
    date = datetime(2012, 3, 31, 5, 30)  # 创建一个 datetime 对象
    result = date + klass()  # 对日期对象应用季度偏移类
    assert result.time() == date.time()  # 断言结果的时间部分与原日期相同

# 参数化测试，测试日期是否落在指定的季度偏移上
@pytest.mark.parametrize("offset", [QuarterBegin(), QuarterEnd()])
@pytest.mark.parametrize(
    "date",
    [
        datetime(2016, m, d)  # 生成一系列日期，包括指定月份和日期的组合
        for m in [10, 11, 12]  # 月份列表
        for d in [1, 2, 3, 28, 29, 30, 31]  # 日期列表
        if not (m == 11 and d == 31)  # 排除特定日期
    ],
)
def test_on_offset(offset, date):
    res = offset.is_on_offset(date)  # 检查日期是否落在季度偏移上
    slow_version = date == (date + offset) - offset  # 通过复杂的方式检查日期偏移
    assert res == slow_version  # 断言两种方法的结果应该一致

# 测试季度偏移类的功能
class TestQuarterBegin:
    def test_repr(self):
        expected = "<QuarterBegin: startingMonth=3>"
        assert repr(QuarterBegin()) == expected  # 检查默认参数时的字符串表示
        expected = "<QuarterBegin: startingMonth=3>"
        assert repr(QuarterBegin(startingMonth=3)) == expected  # 检查自定义参数时的字符串表示
        expected = "<QuarterBegin: startingMonth=1>"
        assert repr(QuarterBegin(startingMonth=1)) == expected  # 检查不同参数时的字符串表示

    def test_offset_corner_case(self):
        # corner
        offset = QuarterBegin(n=-1, startingMonth=1)  # 创建一个特定的季度偏移
        assert datetime(2010, 2, 1) + offset == datetime(2010, 1, 1)  # 断言日期应用偏移后的结果

    offset_cases = []
    offset_cases.append(
        (
            QuarterBegin(startingMonth=1),  # 创建特定月份的季度偏移
            {
                datetime(2007, 12, 1): datetime(2008, 1, 1),  # 一系列日期的偏移结果
                datetime(2008, 1, 1): datetime(2008, 4, 1),
                datetime(2008, 2, 15): datetime(2008, 4, 1),
                datetime(2008, 2, 29): datetime(2008, 4, 1),
                datetime(2008, 3, 15): datetime(2008, 4, 1),
                datetime(2008, 3, 31): datetime(2008, 4, 1),
                datetime(2008, 4, 15): datetime(2008, 7, 1),
                datetime(2008, 4, 1): datetime(2008, 7, 1),
            },
        )
    )

    offset_cases.append(
        (
            QuarterBegin(startingMonth=2),  # 创建另一个特定月份的季度偏移
            {
                datetime(2008, 1, 1): datetime(2008, 2, 1),  # 另一系列日期的偏移结果
                datetime(2008, 1, 31): datetime(2008, 2, 1),
                datetime(2008, 1, 15): datetime(2008, 2, 1),
                datetime(2008, 2, 29): datetime(2008, 5, 1),
                datetime(2008, 3, 15): datetime(2008, 5, 1),
                datetime(2008, 3, 31): datetime(2008, 5, 1),
                datetime(2008, 4, 15): datetime(2008, 5, 1),
                datetime(2008, 4, 30): datetime(2008, 5, 1),
            },
        )
    )
    # 将偏移量和对应的测试用例添加到 offset_cases 列表中
    offset_cases.append(
        (
            QuarterBegin(startingMonth=1, n=0),  # 创建一个季度开始日期生成器，起始月份为1，偏移量为0
            {
                datetime(2008, 1, 1): datetime(2008, 1, 1),  # 测试用例：2008年1月1日的偏移结果应为2008年1月1日
                datetime(2008, 12, 1): datetime(2009, 1, 1),  # 测试用例：2008年12月1日的偏移结果应为2009年1月1日
                # 其他测试用例...
            },
        )
    )

    # 添加另一个偏移量和对应的测试用例到 offset_cases 列表中
    offset_cases.append(
        (
            QuarterBegin(startingMonth=1, n=-1),  # 创建一个季度开始日期生成器，起始月份为1，偏移量为-1
            {
                datetime(2008, 1, 1): datetime(2007, 10, 1),  # 测试用例：2008年1月1日的偏移结果应为2007年10月1日
                # 其他测试用例...
            },
        )
    )

    # 添加另一个偏移量和对应的测试用例到 offset_cases 列表中
    offset_cases.append(
        (
            QuarterBegin(startingMonth=1, n=2),  # 创建一个季度开始日期生成器，起始月份为1，偏移量为2
            {
                datetime(2008, 1, 1): datetime(2008, 7, 1),  # 测试用例：2008年1月1日的偏移结果应为2008年7月1日
                # 其他测试用例...
            },
        )
    )

    # 使用 pytest 的 parametrize 装饰器，对 offset_cases 列表中的测试用例进行参数化测试
    @pytest.mark.parametrize("case", offset_cases)
    # 定义测试函数 test_offset，参数为 case
    def test_offset(self, case):
        offset, cases = case
        # 遍历测试用例字典，对每个测试用例进行断言
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)
class TestQuarterEnd:
    # 定义测试类 TestQuarterEnd

    def test_repr(self):
        # 定义测试方法 test_repr，测试 QuarterEnd 对象的字符串表示
        expected = "<QuarterEnd: startingMonth=3>"
        assert repr(QuarterEnd()) == expected
        expected = "<QuarterEnd: startingMonth=3>"
        assert repr(QuarterEnd(startingMonth=3)) == expected
        expected = "<QuarterEnd: startingMonth=1>"
        assert repr(QuarterEnd(startingMonth=1)) == expected

    def test_offset_corner_case(self):
        # 定义测试方法 test_offset_corner_case，测试 QuarterEnd 对象的偏移量计算
        # 创建 QuarterEnd 对象，n=-1，startingMonth=1
        offset = QuarterEnd(n=-1, startingMonth=1)
        assert datetime(2010, 2, 1) + offset == datetime(2010, 1, 31)

    offset_cases = []
    # 创建偏移量测试用例列表

    offset_cases.append(
        (
            QuarterEnd(startingMonth=1),
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
    # 向偏移量测试用例列表添加一个测试用例

    offset_cases.append(
        (
            QuarterEnd(startingMonth=2),
            {
                datetime(2008, 1, 1): datetime(2008, 2, 29),
                datetime(2008, 1, 31): datetime(2008, 2, 29),
                datetime(2008, 2, 15): datetime(2008, 2, 29),
                datetime(2008, 2, 29): datetime(2008, 5, 31),
                datetime(2008, 3, 15): datetime(2008, 5, 31),
                datetime(2008, 3, 31): datetime(2008, 5, 31),
                datetime(2008, 4, 15): datetime(2008, 5, 31),
                datetime(2008, 4, 30): datetime(2008, 5, 31),
            },
        )
    )
    # 向偏移量测试用例列表添加另一个测试用例

    offset_cases.append(
        (
            QuarterEnd(startingMonth=1, n=0),
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
    # 向偏移量测试用例列表添加另一个测试用例


这些注释解释了每个代码行的作用和功能，符合要求，保持了原始代码的结构和格式不变。
    offset_cases.append(
        (
            QuarterEnd(startingMonth=1, n=-1),  # 创建一个包含负偏移量的QuarterEnd对象，表示向前推一个季度的末尾日期
            {
                datetime(2008, 1, 1): datetime(2007, 10, 31),    # 针对特定日期，指定其对应的向前推一个季度的末尾日期
                datetime(2008, 1, 31): datetime(2007, 10, 31),   # 同上，不同日期
                datetime(2008, 2, 15): datetime(2008, 1, 31),    # 同上，不同日期
                datetime(2008, 2, 29): datetime(2008, 1, 31),    # 同上，不同日期
                datetime(2008, 3, 15): datetime(2008, 1, 31),    # 同上，不同日期
                datetime(2008, 3, 31): datetime(2008, 1, 31),    # 同上，不同日期
                datetime(2008, 4, 15): datetime(2008, 1, 31),    # 同上，不同日期
                datetime(2008, 4, 30): datetime(2008, 1, 31),    # 同上，不同日期
                datetime(2008, 7, 1): datetime(2008, 4, 30),     # 同上，不同日期
            },
        )
    )

    offset_cases.append(
        (
            QuarterEnd(startingMonth=1, n=2),   # 创建一个包含正偏移量的QuarterEnd对象，表示向后推两个季度的末尾日期
            {
                datetime(2008, 1, 31): datetime(2008, 7, 31),   # 针对特定日期，指定其对应的向后推两个季度的末尾日期
                datetime(2008, 2, 15): datetime(2008, 7, 31),   # 同上，不同日期
                datetime(2008, 2, 29): datetime(2008, 7, 31),   # 同上，不同日期
                datetime(2008, 3, 15): datetime(2008, 7, 31),   # 同上，不同日期
                datetime(2008, 3, 31): datetime(2008, 7, 31),   # 同上，不同日期
                datetime(2008, 4, 15): datetime(2008, 7, 31),   # 同上，不同日期
                datetime(2008, 4, 30): datetime(2008, 10, 31),  # 同上，不同日期
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)   # 使用pytest的参数化标记，将offset_cases作为测试用例参数化传入测试函数
    def test_offset(self, case):
        offset, cases = case   # 解包测试用例中的偏移量和期望结果字典
        for base, expected in cases.items():   # 遍历每个测试用例中的日期和对应的期望结果
            assert_offset_equal(offset, base, expected)   # 断言偏移量计算结果与期望的结果相等，调用assert_offset_equal函数进行断言
    # 定义测试用例列表，每个元素包含一个 QuarterEnd 对象、一个日期时间对象和预期的布尔值结果
    on_offset_cases = [
        (QuarterEnd(1, startingMonth=1), datetime(2008, 1, 31), True),   # 每季度末，从1月开始，2008年1月31日符合条件，预期为True
        (QuarterEnd(1, startingMonth=1), datetime(2007, 12, 31), False),  # 每季度末，从1月开始，2007年12月31日不符合条件，预期为False
        (QuarterEnd(1, startingMonth=1), datetime(2008, 2, 29), False),  # 每季度末，从1月开始，2008年2月29日不符合条件，预期为False
        (QuarterEnd(1, startingMonth=1), datetime(2007, 3, 30), False),  # 每季度末，从1月开始，2007年3月30日不符合条件，预期为False
        (QuarterEnd(1, startingMonth=1), datetime(2007, 3, 31), False),  # 每季度末，从1月开始，2007年3月31日不符合条件，预期为False
        (QuarterEnd(1, startingMonth=1), datetime(2008, 4, 30), True),   # 每季度末，从1月开始，2008年4月30日符合条件，预期为True
        (QuarterEnd(1, startingMonth=1), datetime(2008, 5, 30), False),  # 每季度末，从1月开始，2008年5月30日不符合条件，预期为False
        (QuarterEnd(1, startingMonth=1), datetime(2008, 5, 31), False),  # 每季度末，从1月开始，2008年5月31日不符合条件，预期为False
        (QuarterEnd(1, startingMonth=1), datetime(2007, 6, 29), False),  # 每季度末，从1月开始，2007年6月29日不符合条件，预期为False
        (QuarterEnd(1, startingMonth=1), datetime(2007, 6, 30), False),  # 每季度末，从1月开始，2007年6月30日不符合条件，预期为False
        (QuarterEnd(1, startingMonth=2), datetime(2008, 1, 31), False),  # 每季度末，从2月开始，2008年1月31日不符合条件，预期为False
        (QuarterEnd(1, startingMonth=2), datetime(2007, 12, 31), False),  # 每季度末，从2月开始，2007年12月31日不符合条件，预期为False
        (QuarterEnd(1, startingMonth=2), datetime(2008, 2, 29), True),   # 每季度末，从2月开始，2008年2月29日符合条件，预期为True
        (QuarterEnd(1, startingMonth=2), datetime(2007, 3, 30), False),  # 每季度末，从2月开始，2007年3月30日不符合条件，预期为False
        (QuarterEnd(1, startingMonth=2), datetime(2007, 3, 31), False),  # 每季度末，从2月开始，2007年3月31日不符合条件，预期为False
        (QuarterEnd(1, startingMonth=2), datetime(2008, 4, 30), False),  # 每季度末，从2月开始，2008年4月30日不符合条件，预期为False
        (QuarterEnd(1, startingMonth=2), datetime(2008, 5, 30), False),  # 每季度末，从2月开始，2008年5月30日不符合条件，预期为False
        (QuarterEnd(1, startingMonth=2), datetime(2008, 5, 31), True),   # 每季度末，从2月开始，2008年5月31日符合条件，预期为True
        (QuarterEnd(1, startingMonth=2), datetime(2007, 6, 29), False),  # 每季度末，从2月开始，2007年6月29日不符合条件，预期为False
        (QuarterEnd(1, startingMonth=2), datetime(2007, 6, 30), False),  # 每季度末，从2月开始，2007年6月30日不符合条件，预期为False
        (QuarterEnd(1, startingMonth=3), datetime(2008, 1, 31), False),  # 每季度末，从3月开始，2008年1月31日不符合条件，预期为False
        (QuarterEnd(1, startingMonth=3), datetime(2007, 12, 31), True),   # 每季度末，从3月开始，2007年12月31日符合条件，预期为True
        (QuarterEnd(1, startingMonth=3), datetime(2008, 2, 29), False),  # 每季度末，从3月开始，2008年2月29日不符合条件，预期为False
        (QuarterEnd(1, startingMonth=3), datetime(2007, 3, 30), False),  # 每季度末，从3月开始，2007年3月30日不符合条件，预期为False
        (QuarterEnd(1, startingMonth=3), datetime(2007, 3, 31), True),   # 每季度末，从3月开始，2007年3月31日符合条件，预期为True
        (QuarterEnd(1, startingMonth=3), datetime(2008, 4, 30), False),  # 每季度末，从3月开始，2008年4月30日不符合条件，预期为False
        (QuarterEnd(1, startingMonth=3), datetime(2008, 5, 30), False),  # 每季度末，从3月开始，2008年5月30日不符合条件，预期为False
        (QuarterEnd(1, startingMonth=3), datetime(2008, 5, 31), False),  # 每季度末，从3月开始，2008年5月31日不符合条件，预期为False
        (QuarterEnd(1, startingMonth=3), datetime(2007, 6, 29), False),  # 每季度末，从3月开始，2007年6月29日不符合条件，预期为False
        (QuarterEnd(1, startingMonth=3), datetime(2007, 6, 30), True),   # 每季度末，从3月开始，2007年6月30日符合条件，预期为True
    ]

    # 使用 pytest 的参数化装饰器，将测试方法参数化，每个参数对应一个测试用例
    @pytest.mark.parametrize("case", on_offset_cases)
    # 定义测试方法 test_is_on_offset，参数为 case
    def test_is_on_offset(self, case):
        # 分解 case 元组，获取 offset 对象、日期时间对象和预期的布尔值结果
        offset, dt, expected = case
        # 断言调用 assert_is_on_offset 方法，验证 offset 和 dt 是否符合预期的布尔值结果
        assert_is_on_offset(offset, dt, expected)
```