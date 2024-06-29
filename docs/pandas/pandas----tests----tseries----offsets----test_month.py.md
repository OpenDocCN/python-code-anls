# `D:\src\scipysrc\pandas\pandas\tests\tseries\offsets\test_month.py`

```
"""
Tests for the following offsets:
- SemiMonthBegin
- SemiMonthEnd
- MonthBegin
- MonthEnd
"""

# 引入必要的库和模块
from __future__ import annotations  # 允许在注解中使用类型提示

from datetime import datetime  # 导入日期时间处理模块

import pytest  # 导入 pytest 测试框架

# 导入 pandas 相关模块和类
from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
    MonthBegin,
    MonthEnd,
    SemiMonthBegin,
    SemiMonthEnd,
)

from pandas import (  # 导入 pandas 中的类和模块
    DatetimeIndex,
    Series,
    _testing as tm,
)
from pandas.tests.tseries.offsets.common import (  # 导入测试所需的辅助函数
    assert_is_on_offset,
    assert_offset_equal,
)


class TestSemiMonthEnd:
    def test_offset_whole_year(self):
        # 测试整年的半月末偏移
        dates = (
            datetime(2007, 12, 31),
            datetime(2008, 1, 15),
            datetime(2008, 1, 31),
            datetime(2008, 2, 15),
            datetime(2008, 2, 29),
            datetime(2008, 3, 15),
            datetime(2008, 3, 31),
            datetime(2008, 4, 15),
            datetime(2008, 4, 30),
            datetime(2008, 5, 15),
            datetime(2008, 5, 31),
            datetime(2008, 6, 15),
            datetime(2008, 6, 30),
            datetime(2008, 7, 15),
            datetime(2008, 7, 31),
            datetime(2008, 8, 15),
            datetime(2008, 8, 31),
            datetime(2008, 9, 15),
            datetime(2008, 9, 30),
            datetime(2008, 10, 15),
            datetime(2008, 10, 31),
            datetime(2008, 11, 15),
            datetime(2008, 11, 30),
            datetime(2008, 12, 15),
            datetime(2008, 12, 31),
        )

        for base, exp_date in zip(dates[:-1], dates[1:]):
            assert_offset_equal(SemiMonthEnd(), base, exp_date)

        # 确保 .apply_index 的行为符合预期
        shift = DatetimeIndex(dates[:-1])
        with tm.assert_produces_warning(None):
            # GH#22535 检查我们在将整数数组添加到 PeriodIndex 时不会收到 FutureWarning
            result = SemiMonthEnd() + shift

        exp = DatetimeIndex(dates[1:])
        tm.assert_index_equal(result, exp)

    offset_cases = []
    offset_cases.append(
        (
            SemiMonthEnd(),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 15),
                datetime(2008, 1, 15): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 2, 15),
                datetime(2006, 12, 14): datetime(2006, 12, 15),
                datetime(2006, 12, 29): datetime(2006, 12, 31),
                datetime(2006, 12, 31): datetime(2007, 1, 15),
                datetime(2007, 1, 1): datetime(2007, 1, 15),
                datetime(2006, 12, 1): datetime(2006, 12, 15),
                datetime(2006, 12, 15): datetime(2006, 12, 31),
            },
        )
    )
    # 将偏移日期和对应的结果添加到 offset_cases 列表中
    offset_cases.append(
        (
            # 创建 SemiMonthEnd 对象，设置 day_of_month 参数为 20
            SemiMonthEnd(day_of_month=20),
            {
                # 以下是输入日期和对应的预期结果的字典
                datetime(2008, 1, 1): datetime(2008, 1, 20),
                datetime(2008, 1, 15): datetime(2008, 1, 20),
                datetime(2008, 1, 21): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 2, 20),
                datetime(2006, 12, 14): datetime(2006, 12, 20),
                datetime(2006, 12, 29): datetime(2006, 12, 31),
                datetime(2006, 12, 31): datetime(2007, 1, 20),
                datetime(2007, 1, 1): datetime(2007, 1, 20),
                datetime(2006, 12, 1): datetime(2006, 12, 20),
                datetime(2006, 12, 15): datetime(2006, 12, 20),
            },
        )
    )
    
    offset_cases.append(
        (
            # 创建 SemiMonthEnd 对象，设置 day_of_month 参数为 0
            SemiMonthEnd(0),
            {
                # 以下是输入日期和对应的预期结果的字典
                datetime(2008, 1, 1): datetime(2008, 1, 15),
                datetime(2008, 1, 16): datetime(2008, 1, 31),
                datetime(2008, 1, 15): datetime(2008, 1, 15),
                datetime(2008, 1, 31): datetime(2008, 1, 31),
                datetime(2006, 12, 29): datetime(2006, 12, 31),
                datetime(2006, 12, 31): datetime(2006, 12, 31),
                datetime(2007, 1, 1): datetime(2007, 1, 15),
            },
        )
    )
    
    offset_cases.append(
        (
            # 创建 SemiMonthEnd 对象，设置 day_of_month 参数为 16
            SemiMonthEnd(0, day_of_month=16),
            {
                # 以下是输入日期和对应的预期结果的字典
                datetime(2008, 1, 1): datetime(2008, 1, 16),
                datetime(2008, 1, 16): datetime(2008, 1, 16),
                datetime(2008, 1, 15): datetime(2008, 1, 16),
                datetime(2008, 1, 31): datetime(2008, 1, 31),
                datetime(2006, 12, 29): datetime(2006, 12, 31),
                datetime(2006, 12, 31): datetime(2006, 12, 31),
                datetime(2007, 1, 1): datetime(2007, 1, 16),
            },
        )
    )
    
    offset_cases.append(
        (
            # 创建 SemiMonthEnd 对象，设置 day_of_month 参数为 2
            SemiMonthEnd(2),
            {
                # 以下是输入日期和对应的预期结果的字典
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 2, 29),
                datetime(2006, 12, 29): datetime(2007, 1, 15),
                datetime(2006, 12, 31): datetime(2007, 1, 31),
                datetime(2007, 1, 1): datetime(2007, 1, 31),
                datetime(2007, 1, 16): datetime(2007, 2, 15),
                datetime(2006, 11, 1): datetime(2006, 11, 30),
            },
        )
    )
    
    offset_cases.append(
        (
            # 创建 SemiMonthEnd 对象，设置 day_of_month 参数为 -1
            SemiMonthEnd(-1),
            {
                # 以下是输入日期和对应的预期结果的字典
                datetime(2007, 1, 1): datetime(2006, 12, 31),
                datetime(2008, 6, 30): datetime(2008, 6, 15),
                datetime(2008, 12, 31): datetime(2008, 12, 15),
                datetime(2006, 12, 29): datetime(2006, 12, 15),
                datetime(2006, 12, 30): datetime(2006, 12, 15),
                datetime(2007, 1, 1): datetime(2006, 12, 31),
            },
        )
    )
    # 将元组 (SemiMonthEnd(-1, day_of_month=4), {...}) 添加到 offset_cases 列表中
    offset_cases.append(
        (
            SemiMonthEnd(-1, day_of_month=4),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 31),
                datetime(2007, 1, 4): datetime(2006, 12, 31),
                datetime(2008, 6, 30): datetime(2008, 6, 4),
                datetime(2008, 12, 31): datetime(2008, 12, 4),
                datetime(2006, 12, 5): datetime(2006, 12, 4),
                datetime(2006, 12, 30): datetime(2006, 12, 4),
                datetime(2007, 1, 1): datetime(2006, 12, 31),
            },
        )
    )

    # 将元组 (SemiMonthEnd(-2), {...}) 添加到 offset_cases 列表中
    offset_cases.append(
        (
            SemiMonthEnd(-2),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 15),
                datetime(2008, 6, 30): datetime(2008, 5, 31),
                datetime(2008, 3, 15): datetime(2008, 2, 15),
                datetime(2008, 12, 31): datetime(2008, 11, 30),
                datetime(2006, 12, 29): datetime(2006, 11, 30),
                datetime(2006, 12, 14): datetime(2006, 11, 15),
                datetime(2007, 1, 1): datetime(2006, 12, 15),
            },
        )
    )

    # 使用 pytest 的 parametrize 标记，为 test_offset 方法参数化测试用例
    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        # 遍历 cases 字典中的每对键值对，执行断言函数 assert_offset_equal 进行测试
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    # 使用 pytest 的 parametrize 标记，为 test_apply_index 方法参数化测试用例
    @pytest.mark.parametrize("case", offset_cases)
    def test_apply_index(self, case):
        # 从 case 元组中获取 offset 和 cases
        offset, cases = case
        # 创建一个 DatetimeIndex 对象，包含 cases 字典的键（日期时间）
        shift = DatetimeIndex(cases.keys())
        # 创建一个 DatetimeIndex 对象，包含 cases 字典的值（预期的日期时间）
        exp = DatetimeIndex(cases.values())

        # 使用 assert_produces_warning 上下文管理器确保不产生警告信息
        with tm.assert_produces_warning(None):
            # 将 offset 加到 shift 上，得到 result
            result = offset + shift
        # 断言 result 和 exp 相等
        tm.assert_index_equal(result, exp)

    # 创建一个包含元组 (datetime 对象, True/False) 的列表 on_offset_cases
    on_offset_cases = [
        (datetime(2007, 12, 31), True),
        (datetime(2007, 12, 15), True),
        (datetime(2007, 12, 14), False),
        (datetime(2007, 12, 1), False),
        (datetime(2008, 2, 29), True),
    ]

    # 使用 pytest 的 parametrize 标记，为 test_is_on_offset 方法参数化测试用例
    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        dt, expected = case
        # 调用 assert_is_on_offset 函数进行断言测试
        assert_is_on_offset(SemiMonthEnd(), dt, expected)

    # 使用 pytest 的 parametrize 标记，为 klass 参数化测试用例列表
    @pytest.mark.parametrize("klass", [Series, DatetimeIndex])
    def test_vectorized_offset_addition(self, klass):
        # 创建一个包含两个时间戳的实例，带有时区信息，用于测试
        shift = klass(
            [
                Timestamp("2000-01-15 00:15:00", tz="US/Central"),
                Timestamp("2000-02-15", tz="US/Central"),
            ],
            name="a",
        )

        # 确保不会因为将整数数组加到PeriodIndex而收到FutureWarning警告
        with tm.assert_produces_warning(None):
            # 对实例进行半月结束日的偏移计算，检查结果
            result = shift + SemiMonthEnd()
            result2 = SemiMonthEnd() + shift

        # 预期的结果实例，包含两个根据半月结束日偏移后的时间戳
        exp = klass(
            [
                Timestamp("2000-01-31 00:15:00", tz="US/Central"),
                Timestamp("2000-02-29", tz="US/Central"),
            ],
            name="a",
        )
        # 检查计算结果与预期结果是否一致
        tm.assert_equal(result, exp)
        tm.assert_equal(result2, exp)

        # 修改时间戳实例为另一组值，用于第二组测试
        shift = klass(
            [
                Timestamp("2000-01-01 00:15:00", tz="US/Central"),
                Timestamp("2000-02-01", tz="US/Central"),
            ],
            name="a",
        )

        # 确保不会因为将整数数组加到PeriodIndex而收到FutureWarning警告
        with tm.assert_produces_warning(None):
            # 对实例进行半月结束日的偏移计算，检查结果
            result = shift + SemiMonthEnd()
            result2 = SemiMonthEnd() + shift

        # 预期的结果实例，包含两个根据半月结束日偏移后的时间戳
        exp = klass(
            [
                Timestamp("2000-01-15 00:15:00", tz="US/Central"),
                Timestamp("2000-02-15", tz="US/Central"),
            ],
            name="a",
        )
        # 检查计算结果与预期结果是否一致
        tm.assert_equal(result, exp)
        tm.assert_equal(result2, exp)
class TestSemiMonthBegin:
    # 定义测试类 TestSemiMonthBegin，用于测试半月初偏移量

    def test_offset_whole_year(self):
        # 测试整年的偏移量

        dates = (
            datetime(2007, 12, 15),
            datetime(2008, 1, 1),
            datetime(2008, 1, 15),
            datetime(2008, 2, 1),
            datetime(2008, 2, 15),
            datetime(2008, 3, 1),
            datetime(2008, 3, 15),
            datetime(2008, 4, 1),
            datetime(2008, 4, 15),
            datetime(2008, 5, 1),
            datetime(2008, 5, 15),
            datetime(2008, 6, 1),
            datetime(2008, 6, 15),
            datetime(2008, 7, 1),
            datetime(2008, 7, 15),
            datetime(2008, 8, 1),
            datetime(2008, 8, 15),
            datetime(2008, 9, 1),
            datetime(2008, 9, 15),
            datetime(2008, 10, 1),
            datetime(2008, 10, 15),
            datetime(2008, 11, 1),
            datetime(2008, 11, 15),
            datetime(2008, 12, 1),
            datetime(2008, 12, 15),
        )

        for base, exp_date in zip(dates[:-1], dates[1:]):
            # 对于每对日期，验证半月初偏移量计算的准确性
            assert_offset_equal(SemiMonthBegin(), base, exp_date)

        # ensure .apply_index works as expected
        shift = DatetimeIndex(dates[:-1])
        with tm.assert_produces_warning(None):
            # GH#22535 check that we don't get a FutureWarning from adding
            # an integer array to PeriodIndex
            # 检查我们不会因为将整数数组添加到 PeriodIndex 而收到 FutureWarning
            result = SemiMonthBegin() + shift

        exp = DatetimeIndex(dates[1:])
        tm.assert_index_equal(result, exp)

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        # 参数化测试半月初偏移量的不同情况

        offset, cases = case
        for base, expected in cases.items():
            # 验证每种情况下半月初偏移量的正确性
            assert_offset_equal(offset, base, expected)

    @pytest.mark.parametrize("case", offset_cases)
    def test_apply_index(self, case):
        # 测试将半月初偏移量应用于索引的效果

        offset, cases = case
        shift = DatetimeIndex(cases.keys())

        with tm.assert_produces_warning(None):
            # GH#22535 check that we don't get a FutureWarning from adding
            # an integer array to PeriodIndex
            # 检查我们不会因为将整数数组添加到 PeriodIndex 而收到 FutureWarning
            result = offset + shift

        exp = DatetimeIndex(cases.values())
        tm.assert_index_equal(result, exp)

    on_offset_cases = [
        (datetime(2007, 12, 1), True),
        (datetime(2007, 12, 15), True),
        (datetime(2007, 12, 14), False),
        (datetime(2007, 12, 31), False),
        (datetime(2008, 2, 15), True),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        # 测试是否给定日期处于半月初偏移量

        dt, expected = case
        assert_is_on_offset(SemiMonthBegin(), dt, expected)

    @pytest.mark.parametrize("klass", [Series, DatetimeIndex])
    # 定义测试函数，测试向量化偏移加法的行为
    def test_vectorized_offset_addition(self, klass):
        # 创建一个 klass 对象，其中包含两个带时区信息的 Timestamp 对象，作为偏移量
        shift = klass(
            [
                Timestamp("2000-01-15 00:15:00", tz="US/Central"),
                Timestamp("2000-02-15", tz="US/Central"),
            ],
            name="a",
        )
        # 使用上下文管理器确保不会收到任何警告
        with tm.assert_produces_warning(None):
            # GH#22535 检查是否向 PeriodIndex 添加整数数组会触发 FutureWarning
            result = shift + SemiMonthBegin()
            result2 = SemiMonthBegin() + shift

        # 预期的结果，是将 Timestamp 对象加上半月开始日期后得到的新 Timestamp 对象
        exp = klass(
            [
                Timestamp("2000-02-01 00:15:00", tz="US/Central"),
                Timestamp("2000-03-01", tz="US/Central"),
            ],
            name="a",
        )
        # 断言 result 和 exp 相等
        tm.assert_equal(result, exp)
        tm.assert_equal(result2, exp)

        # 重新设置 shift 变量，包含另一组 Timestamp 对象
        shift = klass(
            [
                Timestamp("2000-01-01 00:15:00", tz="US/Central"),
                Timestamp("2000-02-01", tz="US/Central"),
            ],
            name="a",
        )
        with tm.assert_produces_warning(None):
            # 再次检查是否向 PeriodIndex 添加整数数组会触发 FutureWarning
            result = shift + SemiMonthBegin()
            result2 = SemiMonthBegin() + shift

        # 预期的结果，是将 Timestamp 对象加上半月开始日期后得到的新 Timestamp 对象
        exp = klass(
            [
                Timestamp("2000-01-15 00:15:00", tz="US/Central"),
                Timestamp("2000-02-15", tz="US/Central"),
            ],
            name="a",
        )
        # 断言 result 和 exp 相等
        tm.assert_equal(result, exp)
        tm.assert_equal(result2, exp)
class TestMonthBegin:
    offset_cases = []
    # offset_cases 列表用于存储不同测试用例，每个元素是一个元组，包含一个 MonthBegin 对象和一个预期输出的字典

    # 添加第一个测试用例到 offset_cases 列表中
    offset_cases.append(
        (
            MonthBegin(),
            {
                datetime(2008, 1, 31): datetime(2008, 2, 1),
                datetime(2008, 2, 1): datetime(2008, 3, 1),
                datetime(2006, 12, 31): datetime(2007, 1, 1),
                datetime(2006, 12, 1): datetime(2007, 1, 1),
                datetime(2007, 1, 31): datetime(2007, 2, 1),
            },
        )
    )

    # 添加第二个测试用例到 offset_cases 列表中
    offset_cases.append(
        (
            MonthBegin(0),
            {
                datetime(2008, 1, 31): datetime(2008, 2, 1),
                datetime(2008, 1, 1): datetime(2008, 1, 1),
                datetime(2006, 12, 3): datetime(2007, 1, 1),
                datetime(2007, 1, 31): datetime(2007, 2, 1),
            },
        )
    )

    # 添加第三个测试用例到 offset_cases 列表中
    offset_cases.append(
        (
            MonthBegin(2),
            {
                datetime(2008, 2, 29): datetime(2008, 4, 1),
                datetime(2008, 1, 31): datetime(2008, 3, 1),
                datetime(2006, 12, 31): datetime(2007, 2, 1),
                datetime(2007, 12, 28): datetime(2008, 2, 1),
                datetime(2007, 1, 1): datetime(2007, 3, 1),
                datetime(2006, 11, 1): datetime(2007, 1, 1),
            },
        )
    )

    # 添加第四个测试用例到 offset_cases 列表中
    offset_cases.append(
        (
            MonthBegin(-1),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 1),
                datetime(2008, 5, 31): datetime(2008, 5, 1),
                datetime(2008, 12, 31): datetime(2008, 12, 1),
                datetime(2006, 12, 29): datetime(2006, 12, 1),
                datetime(2006, 1, 2): datetime(2006, 1, 1),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    # 使用 pytest 的参数化测试标记，参数为 offset_cases 列表中的每个测试用例
    def test_offset(self, case):
        offset, cases = case
        # 分解元组 case，得到 offset 和 cases

        # 遍历 cases 字典中的每个测试数据
        for base, expected in cases.items():
            # 断言调用 assert_offset_equal 函数，验证 offset 对象的效果
            assert_offset_equal(offset, base, expected)


class TestMonthEnd:
    def test_day_of_month(self):
        dt = datetime(2007, 1, 1)
        offset = MonthEnd()

        result = dt + offset
        assert result == Timestamp(2007, 1, 31)

        result = result + offset
        assert result == Timestamp(2007, 2, 28)

    def test_normalize(self):
        dt = datetime(2007, 1, 1, 3)

        result = dt + MonthEnd(normalize=True)
        expected = dt.replace(hour=0) + MonthEnd()
        assert result == expected

    offset_cases = []
    offset_cases.append(
        (
            MonthEnd(),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 2, 29),
                datetime(2006, 12, 29): datetime(2006, 12, 31),
                datetime(2006, 12, 31): datetime(2007, 1, 31),
                datetime(2007, 1, 1): datetime(2007, 1, 31),
                datetime(2006, 12, 1): datetime(2006, 12, 31),
            },
        )
    )

添加月末偏移案例到列表 `offset_cases`，每个案例包含一个 `MonthEnd()` 对象和一个包含日期映射的字典。


    offset_cases.append(
        (
            MonthEnd(0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 1, 31),
                datetime(2006, 12, 29): datetime(2006, 12, 31),
                datetime(2006, 12, 31): datetime(2006, 12, 31),
                datetime(2007, 1, 1): datetime(2007, 1, 31),
            },
        )
    )

添加月末偏移为零的案例到 `offset_cases` 列表，测试月末偏移为零的情况。


    offset_cases.append(
        (
            MonthEnd(2),
            {
                datetime(2008, 1, 1): datetime(2008, 2, 29),
                datetime(2008, 1, 31): datetime(2008, 3, 31),
                datetime(2006, 12, 29): datetime(2007, 1, 31),
                datetime(2006, 12, 31): datetime(2007, 2, 28),
                datetime(2007, 1, 1): datetime(2007, 2, 28),
                datetime(2006, 11, 1): datetime(2006, 12, 31),
            },
        )
    )

添加月末偏移为两个月的案例到 `offset_cases` 列表，测试月末偏移为正数的情况。


    offset_cases.append(
        (
            MonthEnd(-1),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 31),
                datetime(2008, 6, 30): datetime(2008, 5, 31),
                datetime(2008, 12, 31): datetime(2008, 11, 30),
                datetime(2006, 12, 29): datetime(2006, 11, 30),
                datetime(2006, 12, 30): datetime(2006, 11, 30),
                datetime(2007, 1, 1): datetime(2006, 12, 31),
            },
        )
    )

添加月末偏移为负一月的案例到 `offset_cases` 列表，测试月末偏移为负数的情况。


    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

定义测试函数 `test_offset`，使用 `pytest` 的参数化装饰器，依次对 `offset_cases` 列表中的每个案例进行测试，验证偏移操作的正确性。


    on_offset_cases = [
        (MonthEnd(), datetime(2007, 12, 31), True),
        (MonthEnd(), datetime(2008, 1, 1), False),
    ]

定义了一个包含日期和预期结果的列表 `on_offset_cases`，用于测试 `assert_is_on_offset` 函数的特定情况。


    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)

定义测试函数 `test_is_on_offset`，使用 `pytest` 的参数化装饰器，对 `on_offset_cases` 列表中的每个案例进行测试，验证 `assert_is_on_offset` 函数的正确性。
```