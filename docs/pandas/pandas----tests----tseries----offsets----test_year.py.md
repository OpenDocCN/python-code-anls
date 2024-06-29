# `D:\src\scipysrc\pandas\pandas\tests\tseries\offsets\test_year.py`

```
"""
Tests for the following offsets:
- YearBegin
- YearEnd
"""

# 导入必要的库
from __future__ import annotations  # 支持类型提示

from datetime import datetime  # 导入日期时间模块

import numpy as np  # 导入 numpy 库
import pytest  # 导入 pytest 测试框架

from pandas import Timestamp  # 导入 pandas 的 Timestamp 类
from pandas.tests.tseries.offsets.common import (  # 导入共用的测试函数
    assert_is_on_offset,
    assert_offset_equal,
)

from pandas.tseries.offsets import (  # 导入时间序列偏移量
    YearBegin,
    YearEnd,
)


class TestYearBegin:
    # 测试当月份超出范围时抛出异常
    def test_misspecified(self):
        with pytest.raises(ValueError, match="Month must go from 1 to 12"):
            YearBegin(month=13)

    offset_cases = []  # 初始化测试用例列表

    # 添加测试用例：默认偏移量情况
    offset_cases.append(
        (
            YearBegin(),
            {
                datetime(2008, 1, 1): datetime(2009, 1, 1),
                datetime(2008, 6, 30): datetime(2009, 1, 1),
                datetime(2008, 12, 31): datetime(2009, 1, 1),
                datetime(2005, 12, 30): datetime(2006, 1, 1),
                datetime(2005, 12, 31): datetime(2006, 1, 1),
            },
        )
    )

    # 添加测试用例：指定从年初偏移0年
    offset_cases.append(
        (
            YearBegin(0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 1),
                datetime(2008, 6, 30): datetime(2009, 1, 1),
                datetime(2008, 12, 31): datetime(2009, 1, 1),
                datetime(2005, 12, 30): datetime(2006, 1, 1),
                datetime(2005, 12, 31): datetime(2006, 1, 1),
            },
        )
    )

    # 添加测试用例：指定从年初偏移3年
    offset_cases.append(
        (
            YearBegin(3),
            {
                datetime(2008, 1, 1): datetime(2011, 1, 1),
                datetime(2008, 6, 30): datetime(2011, 1, 1),
                datetime(2008, 12, 31): datetime(2011, 1, 1),
                datetime(2005, 12, 30): datetime(2008, 1, 1),
                datetime(2005, 12, 31): datetime(2008, 1, 1),
            },
        )
    )

    # 添加测试用例：指定从年初偏移-1年
    offset_cases.append(
        (
            YearBegin(-1),
            {
                datetime(2007, 1, 1): datetime(2006, 1, 1),
                datetime(2007, 1, 15): datetime(2007, 1, 1),
                datetime(2008, 6, 30): datetime(2008, 1, 1),
                datetime(2008, 12, 31): datetime(2008, 1, 1),
                datetime(2006, 12, 29): datetime(2006, 1, 1),
                datetime(2006, 12, 30): datetime(2006, 1, 1),
                datetime(2007, 1, 1): datetime(2006, 1, 1),
            },
        )
    )

    # 添加测试用例：指定从年初偏移-2年
    offset_cases.append(
        (
            YearBegin(-2),
            {
                datetime(2007, 1, 1): datetime(2005, 1, 1),
                datetime(2008, 6, 30): datetime(2007, 1, 1),
                datetime(2008, 12, 31): datetime(2007, 1, 1),
            },
        )
    )
    offset_cases.append(
        (
            YearBegin(month=4),  # 添加一个包含 YearBegin 对象和日期映射的元组到 offset_cases 列表中
            {
                datetime(2007, 4, 1): datetime(2008, 4, 1),  # 将指定日期映射到未来一年的4月1日
                datetime(2007, 4, 15): datetime(2008, 4, 1),  # 将指定日期映射到未来一年的4月1日
                datetime(2007, 3, 1): datetime(2007, 4, 1),  # 将指定日期映射到同年的4月1日
                datetime(2007, 12, 15): datetime(2008, 4, 1),  # 将指定日期映射到未来一年的4月1日
                datetime(2012, 1, 31): datetime(2012, 4, 1),  # 将指定日期映射到同年的4月1日
            },
        )
    )

    offset_cases.append(
        (
            YearBegin(0, month=4),  # 添加一个包含 YearBegin 对象和日期映射的元组到 offset_cases 列表中
            {
                datetime(2007, 4, 1): datetime(2007, 4, 1),  # 将指定日期映射到同年的4月1日
                datetime(2007, 3, 1): datetime(2007, 4, 1),  # 将指定日期映射到同年的4月1日
                datetime(2007, 12, 15): datetime(2008, 4, 1),  # 将指定日期映射到未来一年的4月1日
                datetime(2012, 1, 31): datetime(2012, 4, 1),  # 将指定日期映射到同年的4月1日
            },
        )
    )

    offset_cases.append(
        (
            YearBegin(4, month=4),  # 添加一个包含 YearBegin 对象和日期映射的元组到 offset_cases 列表中
            {
                datetime(2007, 4, 1): datetime(2011, 4, 1),  # 将指定日期映射到四年后的4月1日
                datetime(2007, 4, 15): datetime(2011, 4, 1),  # 将指定日期映射到四年后的4月1日
                datetime(2007, 3, 1): datetime(2010, 4, 1),  # 将指定日期映射到三年后的4月1日
                datetime(2007, 12, 15): datetime(2011, 4, 1),  # 将指定日期映射到四年后的4月1日
                datetime(2012, 1, 31): datetime(2015, 4, 1),  # 将指定日期映射到三年后的4月1日
            },
        )
    )

    offset_cases.append(
        (
            YearBegin(-1, month=4),  # 添加一个包含 YearBegin 对象和日期映射的元组到 offset_cases 列表中
            {
                datetime(2007, 4, 1): datetime(2006, 4, 1),  # 将指定日期映射到前一年的4月1日
                datetime(2007, 3, 1): datetime(2006, 4, 1),  # 将指定日期映射到前一年的4月1日
                datetime(2007, 12, 15): datetime(2007, 4, 1),  # 将指定日期映射到同年的4月1日
                datetime(2012, 1, 31): datetime(2011, 4, 1),  # 将指定日期映射到前一年的4月1日
            },
        )
    )

    offset_cases.append(
        (
            YearBegin(-3, month=4),  # 添加一个包含 YearBegin 对象和日期映射的元组到 offset_cases 列表中
            {
                datetime(2007, 4, 1): datetime(2004, 4, 1),  # 将指定日期映射到前三年的4月1日
                datetime(2007, 3, 1): datetime(2004, 4, 1),  # 将指定日期映射到前三年的4月1日
                datetime(2007, 12, 15): datetime(2005, 4, 1),  # 将指定日期映射到前两年的4月1日
                datetime(2012, 1, 31): datetime(2009, 4, 1),  # 将指定日期映射到前三年的4月1日
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):  # 声明一个测试函数，用于测试年初偏移量的计算
        offset, cases = case  # 解包测试用例
        for base, expected in cases.items():  # 遍历测试用例中的日期映射
            assert_offset_equal(offset, base, expected)  # 断言年初偏移量计算的准确性

    on_offset_cases = [
        (YearBegin(), datetime(2007, 1, 3), False),  # 声明包含年初对象、日期和预期结果的元组到 on_offset_cases 列表中
        (YearBegin(), datetime(2008, 1, 1), True),  # 声明包含年初对象、日期和预期结果的元组到 on_offset_cases 列表中
        (YearBegin(), datetime(2006, 12, 31), False),  # 声明包含年初对象、日期和预期结果的元组到 on_offset_cases 列表中
        (YearBegin(), datetime(2006, 1, 2), False),  # 声明包含年初对象、日期和预期结果的元组到 on_offset_cases 列表中
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):  # 声明一个测试函数，用于测试是否为年初的计算
        offset, dt, expected = case  # 解包测试用例
        assert_is_on_offset(offset, dt, expected)  # 断言是否为年初的计算的准确性
class TestYearEnd:
    # 定义测试方法，验证当月数超出合法范围时是否触发值错误
    def test_misspecified(self):
        # 使用 pytest 断言异常，验证传入的月份超出合法范围时是否引发 ValueError 异常
        with pytest.raises(ValueError, match="Month must go from 1 to 12"):
            YearEnd(month=13)

    # 初始化偏移量测试用例列表
    offset_cases = []
    # 添加测试用例：默认 YearEnd 实例的偏移结果与预期映射
    offset_cases.append(
        (
            YearEnd(),
            {
                datetime(2008, 1, 1): datetime(2008, 12, 31),
                datetime(2008, 6, 30): datetime(2008, 12, 31),
                datetime(2008, 12, 31): datetime(2009, 12, 31),
                datetime(2005, 12, 30): datetime(2005, 12, 31),
                datetime(2005, 12, 31): datetime(2006, 12, 31),
            },
        )
    )

    # 添加测试用例：指定 YearEnd(0) 的偏移结果与预期映射
    offset_cases.append(
        (
            YearEnd(0),
            {
                datetime(2008, 1, 1): datetime(2008, 12, 31),
                datetime(2008, 6, 30): datetime(2008, 12, 31),
                datetime(2008, 12, 31): datetime(2008, 12, 31),
                datetime(2005, 12, 30): datetime(2005, 12, 31),
            },
        )
    )

    # 添加测试用例：指定 YearEnd(-1) 的偏移结果与预期映射
    offset_cases.append(
        (
            YearEnd(-1),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 31),
                datetime(2008, 6, 30): datetime(2007, 12, 31),
                datetime(2008, 12, 31): datetime(2007, 12, 31),
                datetime(2006, 12, 29): datetime(2005, 12, 31),
                datetime(2006, 12, 30): datetime(2005, 12, 31),
                datetime(2007, 1, 1): datetime(2006, 12, 31),
            },
        )
    )

    # 添加测试用例：指定 YearEnd(-2) 的偏移结果与预期映射
    offset_cases.append(
        (
            YearEnd(-2),
            {
                datetime(2007, 1, 1): datetime(2005, 12, 31),
                datetime(2008, 6, 30): datetime(2006, 12, 31),
                datetime(2008, 12, 31): datetime(2006, 12, 31),
            },
        )
    )

    # 使用 pytest 的参数化装饰器，迭代测试偏移量用例
    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        # 解包测试用例中的偏移对象和预期结果映射
        offset, cases = case
        # 迭代测试用例，使用自定义的断言函数验证偏移结果与预期是否一致
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    # 初始化偏移日期匹配测试用例列表
    on_offset_cases = [
        (YearEnd(), datetime(2007, 12, 31), True),
        (YearEnd(), datetime(2008, 1, 1), False),
        (YearEnd(), datetime(2006, 12, 31), True),
        (YearEnd(), datetime(2006, 12, 29), False),
    ]

    # 使用 pytest 的参数化装饰器，迭代测试偏移日期匹配用例
    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        # 解包测试用例中的偏移对象、日期和预期结果
        offset, dt, expected = case
        # 使用自定义的断言函数验证偏移日期匹配结果与预期是否一致
        assert_is_on_offset(offset, dt, expected)


class TestYearEndDiffMonth:
    # 初始化偏移量测试用例列表
    offset_cases = []
    # 添加测试用例：指定不同月份的 YearEnd 实例的偏移结果与预期映射
    offset_cases.append(
        (
            YearEnd(month=3),
            {
                datetime(2008, 1, 1): datetime(2008, 3, 31),
                datetime(2008, 2, 15): datetime(2008, 3, 31),
                datetime(2008, 3, 31): datetime(2009, 3, 31),
                datetime(2008, 3, 30): datetime(2008, 3, 31),
                datetime(2005, 3, 31): datetime(2006, 3, 31),
                datetime(2006, 7, 30): datetime(2007, 3, 31),
            },
        )
    )
    offset_cases.append(
        (
            YearEnd(0, month=3),  # 添加一个以3月为结束月份的年底偏移量，起始日期为当前年度
            {
                datetime(2008, 1, 1): datetime(2008, 3, 31),  # 对应日期的年度结束日期为当年3月31日
                datetime(2008, 2, 28): datetime(2008, 3, 31),  # 对应日期的年度结束日期为当年3月31日
                datetime(2008, 3, 31): datetime(2008, 3, 31),  # 对应日期的年度结束日期为当年3月31日
                datetime(2005, 3, 30): datetime(2005, 3, 31),  # 对应日期的年度结束日期为2005年3月31日
            },
        )
    )

    offset_cases.append(
        (
            YearEnd(-1, month=3),  # 添加一个以3月为结束月份的年底偏移量，向前偏移一年
            {
                datetime(2007, 1, 1): datetime(2006, 3, 31),  # 对应日期的年度结束日期为2006年3月31日
                datetime(2008, 2, 28): datetime(2007, 3, 31),  # 对应日期的年度结束日期为2007年3月31日
                datetime(2008, 3, 31): datetime(2007, 3, 31),  # 对应日期的年度结束日期为2007年3月31日
                datetime(2006, 3, 29): datetime(2005, 3, 31),  # 对应日期的年度结束日期为2005年3月31日
                datetime(2006, 3, 30): datetime(2005, 3, 31),  # 对应日期的年度结束日期为2005年3月31日
                datetime(2007, 3, 1): datetime(2006, 3, 31),   # 对应日期的年度结束日期为2006年3月31日
            },
        )
    )

    offset_cases.append(
        (
            YearEnd(-2, month=3),  # 添加一个以3月为结束月份的年底偏移量，向前偏移两年
            {
                datetime(2007, 1, 1): datetime(2005, 3, 31),  # 对应日期的年度结束日期为2005年3月31日
                datetime(2008, 6, 30): datetime(2007, 3, 31),  # 对应日期的年度结束日期为2007年3月31日
                datetime(2008, 3, 31): datetime(2006, 3, 31),  # 对应日期的年度结束日期为2006年3月31日
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    on_offset_cases = [
        (YearEnd(month=3), datetime(2007, 3, 31), True),    # 检查特定日期是否为3月年底
        (YearEnd(month=3), datetime(2008, 1, 1), False),   # 检查特定日期是否为3月年底
        (YearEnd(month=3), datetime(2006, 3, 31), True),    # 检查特定日期是否为3月年底
        (YearEnd(month=3), datetime(2006, 3, 29), False),   # 检查特定日期是否为3月年底
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)
def test_add_out_of_pydatetime_range():
    # 创建一个Timestamp对象，表示公元前20000年12月31日的日期时间
    ts = Timestamp(np.datetime64("-20000-12-31"))
    # 创建一个YearEnd对象，用于表示年末的时间偏移量
    off = YearEnd()

    # 将Timestamp对象ts和YearEnd对象off相加，得到结果Timestamp对象result
    result = ts + off

    # 断言结果result的年份应该是-19999或1973年
    assert result.year in (-19999, 1973)
    # 断言结果result的月份应该是12月
    assert result.month == 12
    # 断言结果result的日期应该是31日
    assert result.day == 31
```