# `D:\src\scipysrc\pandas\pandas\tests\indexes\timedeltas\test_setops.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

import pandas as pd  # 导入 Pandas 库，用于数据处理和分析
from pandas import (  # 从 Pandas 中导入特定模块和函数
    Index,  # 索引对象
    TimedeltaIndex,  # 时间增量索引对象
    timedelta_range,  # 生成时间增量范围
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块

from pandas.tseries.offsets import Hour  # 从 Pandas 时间序列模块中导入小时偏移量对象


class TestTimedeltaIndex:
    def test_union(self):
        i1 = timedelta_range("1day", periods=5)  # 创建时间增量范围对象 i1
        i2 = timedelta_range("3day", periods=5)  # 创建时间增量范围对象 i2
        result = i1.union(i2)  # 对 i1 和 i2 执行并集操作，得到结果对象 result
        expected = timedelta_range("1day", periods=7)  # 预期的时间增量范围对象 expected
        tm.assert_index_equal(result, expected)  # 使用测试模块验证 result 与 expected 相等性

        i1 = Index(np.arange(0, 20, 2, dtype=np.int64))  # 创建整数索引对象 i1
        i2 = timedelta_range(start="1 day", periods=10, freq="D")  # 创建时间增量范围对象 i2
        i1.union(i2)  # 对 i1 和 i2 执行并集操作，这里运行成功
        i2.union(i1)  # 对 i2 和 i1 执行并集操作，这里会因为 "AttributeError: can't set attribute" 失败

    def test_union_sort_false(self):
        tdi = timedelta_range("1day", periods=5)  # 创建时间增量范围对象 tdi

        left = tdi[3:]  # 切片操作，left 包含 tdi 的后三个元素
        right = tdi[:3]  # 切片操作，right 包含 tdi 的前三个元素

        # 检查我们是否在测试期望的代码路径
        assert left._can_fast_union(right)  # 断言 left 可以快速合并 right

        result = left.union(right)  # 对 left 和 right 执行并集操作，得到结果对象 result
        tm.assert_index_equal(result, tdi)  # 使用测试模块验证 result 与 tdi 相等性

        result = left.union(right, sort=False)  # 对 left 和 right 执行并集操作，不进行排序
        expected = TimedeltaIndex(["4 Days", "5 Days", "1 Days", "2 Day", "3 Days"])  # 预期的时间增量索引对象 expected
        tm.assert_index_equal(result, expected)  # 使用测试模块验证 result 与 expected 相等性

    def test_union_coverage(self):
        # GH#59051
        msg = "'d' is deprecated and will be removed in a future version."  # 警告消息内容
        with tm.assert_produces_warning(FutureWarning, match=msg):  # 验证警告消息是否符合预期
            idx = TimedeltaIndex(["3d", "1d", "2d"])  # 创建时间增量索引对象 idx
        ordered = TimedeltaIndex(idx.sort_values(), freq="infer")  # 对 idx 排序并创建具有推断频率的时间增量索引对象 ordered
        result = ordered.union(idx)  # 对 ordered 和 idx 执行并集操作，得到结果对象 result
        tm.assert_index_equal(result, ordered)  # 使用测试模块验证 result 与 ordered 相等性

        result = ordered[:0].union(ordered)  # 对 ordered 的空切片和 ordered 执行并集操作，得到结果对象 result
        tm.assert_index_equal(result, ordered)  # 使用测试模块验证 result 与 ordered 相等性
        assert result.freq == ordered.freq  # 验证 result 的频率与 ordered 的频率相等

    def test_union_bug_1730(self):
        rng_a = timedelta_range("1 day", periods=4, freq="3h")  # 创建时间增量范围对象 rng_a
        rng_b = timedelta_range("1 day", periods=4, freq="4h")  # 创建时间增量范围对象 rng_b

        result = rng_a.union(rng_b)  # 对 rng_a 和 rng_b 执行并集操作，得到结果对象 result
        exp = TimedeltaIndex(sorted(set(rng_a) | set(rng_b)))  # 预期的时间增量索引对象 exp
        tm.assert_index_equal(result, exp)  # 使用测试模块验证 result 与 exp 相等性

    def test_union_bug_1745(self):
        left = TimedeltaIndex(["1 day 15:19:49.695000"])  # 创建时间增量索引对象 left
        right = TimedeltaIndex(  # 创建时间增量索引对象 right
            ["2 day 13:04:21.322000", "1 day 15:27:24.873000", "1 day 15:31:05.350000"]
        )

        result = left.union(right)  # 对 left 和 right 执行并集操作，得到结果对象 result
        exp = TimedeltaIndex(sorted(set(left) | set(right)))  # 预期的时间增量索引对象 exp
        tm.assert_index_equal(result, exp)  # 使用测试模块验证 result 与 exp 相等性

    def test_union_bug_4564(self):
        left = timedelta_range("1 day", "30D")  # 创建时间增量范围对象 left
        right = left + pd.offsets.Minute(15)  # 将 left 的每个元素增加 15 分钟得到 right

        result = left.union(right)  # 对 left 和 right 执行并集操作，得到结果对象 result
        exp = TimedeltaIndex(sorted(set(left) | set(right)))  # 预期的时间增量索引对象 exp
        tm.assert_index_equal(result, exp)  # 使用测试模块验证 result 与 exp 相等性
    def test_union_freq_infer(self):
        # 当合并两个 TimedeltaIndexes 时，即使参数没有频率，也会推断出一个频率。这与 DatetimeIndex 的行为相匹配。
        tdi = timedelta_range("1 Day", periods=5)  # 创建一个 Timedelta 索引，每个间隔为1天，共5个周期
        left = tdi[[0, 1, 3, 4]]  # 左侧选择部分索引值
        right = tdi[[2, 3, 1]]  # 右侧选择部分索引值

        assert left.freq is None  # 断言左侧索引的频率为None
        assert right.freq is None  # 断言右侧索引的频率为None

        result = left.union(right)  # 对左右两侧索引进行合并操作
        tm.assert_index_equal(result, tdi)  # 使用测试工具断言合并结果与原始索引相等
        assert result.freq == "D"  # 断言合并后的结果频率为"天"

    def test_intersection_bug_1708(self):
        index_1 = timedelta_range("1 day", periods=4, freq="h")  # 创建一个 Timedelta 索引，每个间隔为1天，共4个周期，频率为每小时
        index_2 = index_1 + pd.offsets.Hour(5)  # 将 index_1 的每个元素增加5小时

        result = index_1.intersection(index_2)  # 计算两个索引的交集
        assert len(result) == 0  # 断言交集长度为0

        index_1 = timedelta_range("1 day", periods=4, freq="h")  # 重新创建 Timedelta 索引，每个间隔为1天，共4个周期，频率为每小时
        index_2 = index_1 + pd.offsets.Hour(1)  # 将 index_1 的每个元素增加1小时

        result = index_1.intersection(index_2)  # 计算两个索引的交集
        expected = timedelta_range("1 day 01:00:00", periods=3, freq="h")  # 创建预期的交集索引，频率为每小时
        tm.assert_index_equal(result, expected)  # 使用测试工具断言计算的交集与预期的交集相等
        assert result.freq == expected.freq  # 断言交集的频率与预期的频率相等

    def test_intersection_equal(self, sort):
        # GH 24471 测试交集结果在给定 sort 关键字的情况下的行为
        # 对于相等的索引，交集应该返回原始索引，不受 sort 影响
        first = timedelta_range("1 day", periods=4, freq="h")  # 创建第一个 Timedelta 索引，每个间隔为1天，共4个周期，频率为每小时
        second = timedelta_range("1 day", periods=4, freq="h")  # 创建第二个 Timedelta 索引，每个间隔为1天，共4个周期，频率为每小时
        intersect = first.intersection(second, sort=sort)  # 计算两个索引的交集，根据 sort 参数排序结果
        if sort is None:
            tm.assert_index_equal(intersect, second.sort_values())  # 如果 sort 为 None，断言交集等于按值排序后的第二个索引
        tm.assert_index_equal(intersect, second)  # 断言交集等于第二个索引

        # 边界情况
        inter = first.intersection(first, sort=sort)  # 计算一个索引与自身的交集
        assert inter is first  # 断言交集结果与第一个索引对象相同

    @pytest.mark.parametrize("period_1, period_2", [(0, 4), (4, 0)])
    def test_intersection_zero_length(self, period_1, period_2, sort):
        # GH 24471 测试不重叠情况下交集的长度应该为零
        index_1 = timedelta_range("1 day", periods=period_1, freq="h")  # 创建第一个 Timedelta 索引，根据参数确定周期数和频率
        index_2 = timedelta_range("1 day", periods=period_2, freq="h")  # 创建第二个 Timedelta 索引，根据参数确定周期数和频率
        expected = timedelta_range("1 day", periods=0, freq="h")  # 创建预期的零长度交集索引，频率为每小时
        result = index_1.intersection(index_2, sort=sort)  # 计算两个索引的交集，根据 sort 参数排序结果
        tm.assert_index_equal(result, expected)  # 使用测试工具断言计算的交集与预期的交集相等

    def test_zero_length_input_index(self, sort):
        # GH 24966 测试零长度索引的交集
        index_1 = timedelta_range("1 day", periods=0, freq="h")  # 创建零长度的 Timedelta 索引，频率为每小时
        index_2 = timedelta_range("1 day", periods=3, freq="h")  # 创建另一个 Timedelta 索引，每个间隔为1天，共3个周期，频率为每小时
        result = index_1.intersection(index_2, sort=sort)  # 计算两个索引的交集，根据 sort 参数排序结果
        assert index_1 is not result  # 断言结果不是第一个索引的引用
        assert index_2 is not result  # 断言结果不是第二个索引的引用
        tm.assert_copy(result, index_1)  # 使用测试工具断言结果是第一个索引的副本
    @pytest.mark.parametrize(
        "rng, expected",
        # 定义参数化测试的参数范围和期望结果
        [
            (
                timedelta_range("1 day", periods=5, freq="h", name="idx"),
                timedelta_range("1 day", periods=4, freq="h", name="idx"),
            ),
            # 如果目标具有相同的名称，则保留该名称
            (
                timedelta_range("1 day", periods=5, freq="h", name="other"),
                timedelta_range("1 day", periods=4, freq="h", name=None),
            ),
            # 如果不存在重叠部分，则返回空的时间增量索引
            (
                timedelta_range("1 day", periods=10, freq="h", name="idx")[5:],
                TimedeltaIndex([], freq="h", name="idx"),
            ),
        ],
    )
    def test_intersection(self, rng, expected, sort):
        # GH 4690 (with tz)
        # 创建基准时间增量索引对象
        base = timedelta_range("1 day", periods=4, freq="h", name="idx")
        # 执行交集操作，并获取结果
        result = base.intersection(rng, sort=sort)
        if sort is None:
            # 如果未指定排序，对期望结果进行排序
            expected = expected.sort_values()
        # 断言结果与期望结果相等
        tm.assert_index_equal(result, expected)
        # 断言结果的名称与期望结果相同
        assert result.name == expected.name
        # 断言结果的频率与期望结果相同
        assert result.freq == expected.freq

    @pytest.mark.parametrize(
        "rng, expected",
        # 部分交集的情况
        [
            (
                TimedeltaIndex(["5 hour", "2 hour", "4 hour", "9 hour"], name="idx"),
                TimedeltaIndex(["2 hour", "4 hour"], name="idx"),
            ),
            # 重新排序后的部分交集
            (
                TimedeltaIndex(["2 hour", "5 hour", "5 hour", "1 hour"], name="other"),
                TimedeltaIndex(["1 hour", "2 hour"], name=None),
            ),
            # 倒序索引
            (
                TimedeltaIndex(["1 hour", "2 hour", "4 hour", "3 hour"], name="idx")[
                    ::-1
                ],
                TimedeltaIndex(["1 hour", "2 hour", "4 hour", "3 hour"], name="idx"),
            ),
        ],
    )
    def test_intersection_non_monotonic(self, rng, expected, sort):
        # 24471 非单调
        # 创建基准时间增量索引对象
        base = TimedeltaIndex(["1 hour", "2 hour", "4 hour", "3 hour"], name="idx")
        # 执行交集操作，并获取结果
        result = base.intersection(rng, sort=sort)
        if sort is None:
            # 如果未指定排序，对期望结果进行排序
            expected = expected.sort_values()
        # 断言结果与期望结果相等
        tm.assert_index_equal(result, expected)
        # 断言结果的名称与期望结果相同
        assert result.name == expected.name

        # 如果倒序相等，频率仍然相同
        if all(base == rng[::-1]) and sort is None:
            assert isinstance(result.freq, Hour)
        else:
            assert result.freq is None
class TestTimedeltaIndexDifference:
    # 定义测试类 TestTimedeltaIndexDifference，用于测试 TimedeltaIndex 的差异操作

    def test_difference_freq(self, sort):
        # 测试方法：test_difference_freq
        # 参数 sort 用于指定是否排序结果

        # 创建一个时间增量索引，从 "0 days" 到 "5 days"，频率为每天一次
        index = timedelta_range("0 days", "5 days", freq="D")

        # 创建另一个时间增量索引，从 "1 days" 到 "4 days"，频率为每天一次
        other = timedelta_range("1 days", "4 days", freq="D")

        # 创建预期的时间增量索引对象，包含 "0 days" 和 "5 days"，频率为 None
        expected = TimedeltaIndex(["0 days", "5 days"], freq=None)

        # 对时间增量索引进行差异操作，根据参数 sort 是否排序结果
        idx_diff = index.difference(other, sort)

        # 断言 idx_diff 和 expected 相等
        tm.assert_index_equal(idx_diff, expected)

        # 断言 idx_diff 的频率属性与 expected 相等
        tm.assert_attr_equal("freq", idx_diff, expected)

        # 当差异是原始范围的连续子集时，保留频率
        other = timedelta_range("2 days", "5 days", freq="D")

        # 再次进行差异操作
        idx_diff = index.difference(other, sort)

        # 创建预期的时间增量索引对象，包含 "0 days" 和 "1 days"，频率为每天一次
        expected = TimedeltaIndex(["0 days", "1 days"], freq="D")

        # 断言 idx_diff 和 expected 相等
        tm.assert_index_equal(idx_diff, expected)

    def test_difference_sort(self, sort):
        # 测试方法：test_difference_sort
        # 参数 sort 用于指定是否排序结果

        # 创建一个时间增量索引，包含 ["5 days", "3 days", "2 days", "4 days", "1 days", "0 days"]
        index = TimedeltaIndex(
            ["5 days", "3 days", "2 days", "4 days", "1 days", "0 days"]
        )

        # 创建另一个时间增量索引，从 "1 days" 到 "4 days"，频率为每天一次
        other = timedelta_range("1 days", "4 days", freq="D")

        # 进行差异操作
        idx_diff = index.difference(other, sort)

        # 创建预期的时间增量索引对象，包含 ["5 days", "0 days"]，频率为 None
        expected = TimedeltaIndex(["5 days", "0 days"], freq=None)

        # 如果 sort 是 None，则对预期结果进行排序
        if sort is None:
            expected = expected.sort_values()

        # 断言 idx_diff 和 expected 相等
        tm.assert_index_equal(idx_diff, expected)

        # 创建另一个时间增量索引，从 "2 days" 到 "5 days"，频率为每天一次
        other = timedelta_range("2 days", "5 days", freq="D")

        # 再次进行差异操作
        idx_diff = index.difference(other, sort)

        # 创建预期的时间增量索引对象，包含 ["1 days", "0 days"]，频率为 None
        expected = TimedeltaIndex(["1 days", "0 days"], freq=None)

        # 如果 sort 是 None，则对预期结果进行排序
        if sort is None:
            expected = expected.sort_values()

        # 断言 idx_diff 和 expected 相等
        tm.assert_index_equal(idx_diff, expected)
```