# `D:\src\scipysrc\pandas\pandas\tests\tseries\offsets\test_business_month.py`

```
"""
Tests for the following offsets:
- BMonthBegin
- BMonthEnd
"""

# 导入必要的库和模块
from __future__ import annotations  # 允许在类型提示中使用当前类名

from datetime import datetime  # 导入 datetime 类

import pytest  # 导入 pytest 测试框架

import pandas as pd  # 导入 pandas 库
from pandas.tests.tseries.offsets.common import (  # 导入用于测试的共同功能
    assert_is_on_offset,
    assert_offset_equal,
)

from pandas.tseries.offsets import (  # 导入 pandas 中的时间偏移类
    BMonthBegin,
    BMonthEnd,
)

# 定义测试函数 test_apply_index，使用 pytest.mark.parametrize 进行参数化测试
@pytest.mark.parametrize("n", [-2, 1])
@pytest.mark.parametrize(
    "cls",
    [
        BMonthBegin,
        BMonthEnd,
    ],
)
def test_apply_index(cls, n):
    offset = cls(n=n)  # 创建指定类和偏移量的实例
    rng = pd.date_range(start="1/1/2000", periods=100000, freq="min")  # 创建日期范围
    ser = pd.Series(rng)  # 使用日期范围创建 Series

    res = rng + offset  # 对日期范围应用偏移量
    assert res.freq is None  # 验证频率未保留
    assert res[0] == rng[0] + offset  # 验证第一个元素应用偏移后的结果
    assert res[-1] == rng[-1] + offset  # 验证最后一个元素应用偏移后的结果
    res2 = ser + offset  # 对 Series 应用偏移量
    # apply_index 仅适用于索引，不适用于 Series，因此不需要 res2_v2
    assert res2.iloc[0] == ser.iloc[0] + offset  # 验证 Series 的第一个元素应用偏移后的结果
    assert res2.iloc[-1] == ser.iloc[-1] + offset  # 验证 Series 的最后一个元素应用偏移后的结果

# 定义测试类 TestBMonthBegin
class TestBMonthBegin:
    def test_offsets_compare_equal(self):
        # 根本原因是 #456
        offset1 = BMonthBegin()  # 创建 BMonthBegin 实例
        offset2 = BMonthBegin()  # 创建另一个 BMonthBegin 实例
        assert not offset1 != offset2  # 断言两个实例相等

    # 定义偏移量测试用例列表
    offset_cases = []
    offset_cases.append(
        (
            BMonthBegin(),  # 创建 BMonthBegin 实例
            {  # 定义日期到偏移后日期的映射字典
                datetime(2008, 1, 1): datetime(2008, 2, 1),
                datetime(2008, 1, 31): datetime(2008, 2, 1),
                datetime(2006, 12, 29): datetime(2007, 1, 1),
                datetime(2006, 12, 31): datetime(2007, 1, 1),
                datetime(2006, 9, 1): datetime(2006, 10, 2),
                datetime(2007, 1, 1): datetime(2007, 2, 1),
                datetime(2006, 12, 1): datetime(2007, 1, 1),
            },
        )
    )

    offset_cases.append(
        (
            BMonthBegin(0),  # 创建指定偏移量的 BMonthBegin 实例
            {  # 定义日期到偏移后日期的映射字典
                datetime(2008, 1, 1): datetime(2008, 1, 1),
                datetime(2006, 10, 2): datetime(2006, 10, 2),
                datetime(2008, 1, 31): datetime(2008, 2, 1),
                datetime(2006, 12, 29): datetime(2007, 1, 1),
                datetime(2006, 12, 31): datetime(2007, 1, 1),
                datetime(2006, 9, 15): datetime(2006, 10, 2),
            },
        )
    )

    offset_cases.append(
        (
            BMonthBegin(2),  # 创建指定偏移量的 BMonthBegin 实例
            {  # 定义日期到偏移后日期的映射字典
                datetime(2008, 1, 1): datetime(2008, 3, 3),
                datetime(2008, 1, 15): datetime(2008, 3, 3),
                datetime(2006, 12, 29): datetime(2007, 2, 1),
                datetime(2006, 12, 31): datetime(2007, 2, 1),
                datetime(2007, 1, 1): datetime(2007, 3, 1),
                datetime(2006, 11, 1): datetime(2007, 1, 1),
            },
        )
    )
    offset_cases.append(
        (
            BMonthBegin(-1),  # 创建一个 BMonthBegin 对象，以每月的第一个工作日为基准，向前偏移一个月
            {
                datetime(2007, 1, 1): datetime(2006, 12, 1),   # 测试用例：2007年1月1日的偏移结果应为2006年12月1日
                datetime(2008, 6, 30): datetime(2008, 6, 2),  # 测试用例：2008年6月30日的偏移结果应为2008年6月2日
                datetime(2008, 6, 1): datetime(2008, 5, 1),   # 测试用例：2008年6月1日的偏移结果应为2008年5月1日
                datetime(2008, 3, 10): datetime(2008, 3, 3),  # 测试用例：2008年3月10日的偏移结果应为2008年3月3日
                datetime(2008, 12, 31): datetime(2008, 12, 1), # 测试用例：2008年12月31日的偏移结果应为2008年12月1日
                datetime(2006, 12, 29): datetime(2006, 12, 1), # 测试用例：2006年12月29日的偏移结果应为2006年12月1日
                datetime(2006, 12, 30): datetime(2006, 12, 1), # 测试用例：2006年12月30日的偏移结果应为2006年12月1日
                datetime(2007, 1, 1): datetime(2006, 12, 1),   # 测试用例：2007年1月1日的偏移结果应为2006年12月1日（重复的测试用例）
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)  # 断言每个测试用例的偏移结果与期望结果是否相等

    on_offset_cases = [
        (BMonthBegin(), datetime(2007, 12, 31), False),   # 测试用例：2007年12月31日是否为每月的第一个工作日
        (BMonthBegin(), datetime(2008, 1, 1), True),      # 测试用例：2008年1月1日是否为每月的第一个工作日
        (BMonthBegin(), datetime(2001, 4, 2), True),      # 测试用例：2001年4月2日是否为每月的第一个工作日
        (BMonthBegin(), datetime(2008, 3, 3), True),      # 测试用例：2008年3月3日是否为每月的第一个工作日
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)  # 断言每个测试用例给定日期是否为每月的第一个工作日，检查是否符合期望结果
class TestBMonthEnd:
    # BMonthEnd 测试类

    def test_normalize(self):
        # 测试日期时间的规范化处理
        dt = datetime(2007, 1, 1, 3)

        # 计算应用偏移后的结果
        result = dt + BMonthEnd(normalize=True)
        # 期望的结果是将小时置为0后再应用月末偏移
        expected = dt.replace(hour=0) + BMonthEnd()
        # 断言结果与期望相等
        assert result == expected

    def test_offsets_compare_equal(self):
        # 偏移量相等性比较的测试
        # 引起问题 #456 的根本原因
        offset1 = BMonthEnd()
        offset2 = BMonthEnd()
        assert not offset1 != offset2

    offset_cases = []
    offset_cases.append(
        (
            BMonthEnd(),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 2, 29),
                datetime(2006, 12, 29): datetime(2007, 1, 31),
                datetime(2006, 12, 31): datetime(2007, 1, 31),
                datetime(2007, 1, 1): datetime(2007, 1, 31),
                datetime(2006, 12, 1): datetime(2006, 12, 29),
            },
        )
    )

    offset_cases.append(
        (
            BMonthEnd(0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 1, 31),
                datetime(2006, 12, 29): datetime(2006, 12, 29),
                datetime(2006, 12, 31): datetime(2007, 1, 31),
                datetime(2007, 1, 1): datetime(2007, 1, 31),
            },
        )
    )

    offset_cases.append(
        (
            BMonthEnd(2),
            {
                datetime(2008, 1, 1): datetime(2008, 2, 29),
                datetime(2008, 1, 31): datetime(2008, 3, 31),
                datetime(2006, 12, 29): datetime(2007, 2, 28),
                datetime(2006, 12, 31): datetime(2007, 2, 28),
                datetime(2007, 1, 1): datetime(2007, 2, 28),
                datetime(2006, 11, 1): datetime(2006, 12, 29),
            },
        )
    )

    offset_cases.append(
        (
            BMonthEnd(-1),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 29),
                datetime(2008, 6, 30): datetime(2008, 5, 30),
                datetime(2008, 12, 31): datetime(2008, 11, 28),
                datetime(2006, 12, 29): datetime(2006, 11, 30),
                datetime(2006, 12, 30): datetime(2006, 12, 29),
                datetime(2007, 1, 1): datetime(2006, 12, 29),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        # 测试不同偏移量的日期处理
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    on_offset_cases = [
        (BMonthEnd(), datetime(2007, 12, 31), True),
        (BMonthEnd(), datetime(2008, 1, 1), False),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        # 测试日期是否处于偏移后的月末
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)
```