# `D:\src\scipysrc\pandas\pandas\tests\tseries\offsets\common.py`

```
"""
Assertion helpers and base class for offsets tests
"""

# 导入未来的注释语法支持，使得可以在类定义中使用 annotations
from __future__ import annotations


# 定义一个函数，用于断言偏移量计算的准确性
def assert_offset_equal(offset, base, expected):
    # 计算实际偏移量，基于提供的偏移量和基础值
    actual = offset + base
    # 对调顺序后再次计算实际偏移量
    actual_swapped = base + offset
    # 调用偏移量对象的 _apply 方法计算实际应用后的值
    actual_apply = offset._apply(base)
    try:
        # 断言实际偏移量与期望值相等
        assert actual == expected
        # 断言对调顺序后的实际偏移量与期望值相等
        assert actual_swapped == expected
        # 断言应用偏移量后的实际值与期望值相等
        assert actual_apply == expected
    except AssertionError as err:
        # 如果有断言错误，抛出详细的错误信息
        raise AssertionError(
            f"\nExpected: {expected}\nActual: {actual}\nFor Offset: {offset})"
            f"\nAt Date: {base}"
        ) from err


# 定义一个函数，用于断言特定日期是否在给定偏移量上
def assert_is_on_offset(offset, date, expected):
    # 调用偏移量对象的 is_on_offset 方法检查日期是否在偏移量上
    actual = offset.is_on_offset(date)
    assert actual == expected, (
        f"\nExpected: {expected}\nActual: {actual}\nFor Offset: {offset})"
        f"\nAt Date: {date}"
    )


# 定义一个类表示工作日，使用常量表示每个工作日
class WeekDay:
    MON = 0  # 星期一
    TUE = 1  # 星期二
    WED = 2  # 星期三
    THU = 3  # 星期四
    FRI = 4  # 星期五
    SAT = 5  # 星期六
    SUN = 6  # 星期天
```