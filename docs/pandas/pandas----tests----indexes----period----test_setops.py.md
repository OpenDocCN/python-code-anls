# `D:\src\scipysrc\pandas\pandas\tests\indexes\period\test_setops.py`

```
# 导入 NumPy 库并使用别名 np
import numpy as np
# 导入 pytest 库
import pytest

# 导入 Pandas 库并从中导入特定模块和函数
import pandas as pd
from pandas import (
    PeriodIndex,
    date_range,
    period_range,
)
# 导入 Pandas 的测试模块
import pandas._testing as tm


# 定义一个函数 _permute，用于对传入的对象进行随机排列
def _permute(obj):
    return obj.take(np.random.default_rng(2).permutation(len(obj)))


# 定义测试类 TestPeriodIndex
class TestPeriodIndex:
    # 定义测试方法 test_union_misc，接受一个名为 sort 的参数
    def test_union_misc(self, sort):
        # 创建一个包含日期范围的 PeriodIndex 对象
        index = period_range("1/1/2000", "1/20/2000", freq="D")

        # 执行 union 操作，将前面部分与后面部分合并，并根据 sort 参数进行排序
        result = index[:-5].union(index[10:], sort=sort)
        # 断言结果与原始索引相等
        tm.assert_index_equal(result, index)

        # 对排列后的 index 进行 union 操作，再次根据 sort 参数进行排序
        result = _permute(index[:-5]).union(_permute(index[10:]), sort=sort)
        # 根据 sort 参数决定是否排序结果，然后与原始索引比较
        if sort is False:
            tm.assert_index_equal(result.sort_values(), index)
        else:
            tm.assert_index_equal(result, index)

        # 创建两个不同频率的 PeriodIndex 对象，执行 union 操作
        index = period_range("1/1/2000", "1/20/2000", freq="D")
        index2 = period_range("1/1/2000", "1/20/2000", freq="W-WED")
        # 期望的结果是两个索引对象转换为 object 类型后执行 union 操作的结果
        result = index.union(index2, sort=sort)
        expected = index.astype(object).union(index2.astype(object), sort=sort)
        tm.assert_index_equal(result, expected)

    # 定义测试方法 test_intersection，接受一个名为 sort 的参数
    def test_intersection(self, sort):
        # 创建一个包含日期范围的 PeriodIndex 对象
        index = period_range("1/1/2000", "1/20/2000", freq="D")

        # 执行 intersection 操作，找出两个索引对象的交集，并根据 sort 参数进行排序
        result = index[:-5].intersection(index[10:], sort=sort)
        # 断言结果与预期的交集部分相等
        tm.assert_index_equal(result, index[10:-5])

        # 对排列后的两个 index 对象执行 intersection 操作，再次根据 sort 参数进行排序
        left = _permute(index[:-5])
        right = _permute(index[10:])
        result = left.intersection(right, sort=sort)
        # 根据 sort 参数决定是否排序结果，然后与预期的交集部分比较
        if sort is False:
            tm.assert_index_equal(result.sort_values(), index[10:-5])
        else:
            tm.assert_index_equal(result, index[10:-5])

        # 创建两个不同频率的 PeriodIndex 对象，执行 intersection 操作
        index = period_range("1/1/2000", "1/20/2000", freq="D")
        index2 = period_range("1/1/2000", "1/20/2000", freq="W-WED")
        # 期望的结果是一个空的 object 类型的索引
        result = index.intersection(index2, sort=sort)
        expected = pd.Index([], dtype=object)
        tm.assert_index_equal(result, expected)

        # 创建另一个不同频率的 PeriodIndex 对象，执行 intersection 操作
        index3 = period_range("1/1/2000", "1/20/2000", freq="2D")
        result = index.intersection(index3, sort=sort)
        # 期望的结果是一个空的 object 类型的索引
        tm.assert_index_equal(result, expected)
    # 定义测试函数，用于测试交集情况，接受一个排序参数 sort
    def test_intersection_cases(self, sort):
        # 创建基础时间周期范围，从 "6/1/2000" 到 "6/30/2000"，每天频率为 "D"，指定名称为 "idx"
        base = period_range("6/1/2000", "6/30/2000", freq="D", name="idx")

        # 如果目标时间周期与基础时间周期有相同的名称，则保留该名称
        rng2 = period_range("5/15/2000", "6/20/2000", freq="D", name="idx")
        expected2 = period_range("6/1/2000", "6/20/2000", freq="D", name="idx")

        # 如果目标时间周期的名称不同，则将其重置为默认名称
        rng3 = period_range("5/15/2000", "6/20/2000", freq="D", name="other")
        expected3 = period_range("6/1/2000", "6/20/2000", freq="D", name=None)

        # 创建另一个时间周期范围 rng4，从 "7/1/2000" 到 "7/31/2000"，每天频率为 "D"，指定名称为 "idx"
        rng4 = period_range("7/1/2000", "7/31/2000", freq="D", name="idx")
        expected4 = PeriodIndex([], name="idx", freq="D")

        # 遍历上述定义的时间周期范围和期望结果的列表
        for rng, expected in [
            (rng2, expected2),
            (rng3, expected3),
            (rng4, expected4),
        ]:
            # 计算基础时间周期范围与目标时间周期范围的交集，并根据 sort 参数进行排序
            result = base.intersection(rng, sort=sort)
            # 断言交集结果与期望结果相等
            tm.assert_index_equal(result, expected)
            # 断言交集结果的名称与期望结果的名称相同
            assert result.name == expected.name
            # 断言交集结果的频率与期望结果的频率相同
            assert result.freq == expected.freq

        # 创建非单调递增的时间周期索引 base
        base = PeriodIndex(
            ["2011-01-05", "2011-01-04", "2011-01-02", "2011-01-03"],
            freq="D",
            name="idx",
        )

        # 创建另一个非单调递增的时间周期索引 rng2
        rng2 = PeriodIndex(
            ["2011-01-04", "2011-01-02", "2011-02-02", "2011-02-03"],
            freq="D",
            name="idx",
        )
        expected2 = PeriodIndex(["2011-01-04", "2011-01-02"], freq="D", name="idx")

        # 创建具有不同名称的时间周期索引 rng3
        rng3 = PeriodIndex(
            ["2011-01-04", "2011-01-02", "2011-02-02", "2011-02-03"],
            freq="D",
            name="other",
        )
        expected3 = PeriodIndex(["2011-01-04", "2011-01-02"], freq="D", name=None)

        # 创建另一个时间周期范围 rng4，从 "7/1/2000" 到 "7/31/2000"，每天频率为 "D"，指定名称为 "idx"
        rng4 = period_range("7/1/2000", "7/31/2000", freq="D", name="idx")
        expected4 = PeriodIndex([], freq="D", name="idx")

        # 遍历上述定义的时间周期范围和期望结果的列表
        for rng, expected in [
            (rng2, expected2),
            (rng3, expected3),
            (rng4, expected4),
        ]:
            # 计算基础时间周期索引与目标时间周期索引的交集，并根据 sort 参数进行排序（如果 sort 不为 None）
            result = base.intersection(rng, sort=sort)
            # 如果 sort 参数为 None，则对期望结果进行排序
            if sort is None:
                expected = expected.sort_values()
            # 断言交集结果与期望结果相等
            tm.assert_index_equal(result, expected)
            # 断言交集结果的名称与期望结果的名称相同
            assert result.name == expected.name
            # 断言交集结果的频率为 "D"
            assert result.freq == "D"

        # 创建一个日期范围 rng，从 "6/1/2000" 到 "6/15/2000"，每分钟频率为 "min"
        rng = date_range("6/1/2000", "6/15/2000", freq="min")
        # 计算空交集，即两个时间范围的交集结果为空
        result = rng[0:0].intersection(rng)
        # 断言交集结果的长度为 0
        assert len(result) == 0

        # 再次计算空交集，这次交换顺序
        result = rng.intersection(rng[0:0])
        # 断言交集结果的长度为 0
        assert len(result) == 0
    # 定义一个测试方法，用于测试时间段索引的差异操作
    def test_difference(self, sort):
        # 准备测试数据集1
        period_rng = ["1/3/2000", "1/2/2000", "1/1/2000", "1/5/2000", "1/4/2000"]
        # 创建第一个时间段索引对象
        rng1 = PeriodIndex(period_rng, freq="D")
        # 创建另一个时间段索引对象
        other1 = period_range("1/6/2000", freq="D", periods=5)
        # 期望的结果为第一个时间段索引对象rng1

        # 准备测试数据集2
        rng2 = PeriodIndex(period_rng, freq="D")
        # 创建另一个时间段索引对象
        other2 = period_range("1/4/2000", freq="D", periods=5)
        # 期望的结果为包含部分元素的时间段索引对象

        # 准备测试数据集3
        rng3 = PeriodIndex(period_rng, freq="D")
        # 创建一个空的时间段索引对象
        other3 = PeriodIndex([], freq="D")
        # 期望的结果为第一个时间段索引对象rng3

        # 准备测试数据集4
        period_rng = [
            "2000-01-01 10:00",
            "2000-01-01 09:00",
            "2000-01-01 12:00",
            "2000-01-01 11:00",
            "2000-01-01 13:00",
        ]
        # 创建第四个时间段索引对象
        rng4 = PeriodIndex(period_rng, freq="h")
        # 创建另一个时间段索引对象
        other4 = period_range("2000-01-02 09:00", freq="h", periods=5)
        # 期望的结果为第四个时间段索引对象rng4

        # 准备测试数据集5
        rng5 = PeriodIndex(
            ["2000-01-01 09:03", "2000-01-01 09:01", "2000-01-01 09:05"], freq="min"
        )
        # 创建另一个时间段索引对象
        other5 = PeriodIndex(["2000-01-01 09:01", "2000-01-01 09:05"], freq="min")
        # 期望的结果为包含单个元素的时间段索引对象

        # 准备测试数据集6
        period_rng = [
            "2000-02-01",
            "2000-01-01",
            "2000-06-01",
            "2000-07-01",
            "2000-05-01",
            "2000-03-01",
            "2000-04-01",
        ]
        # 创建第六个时间段索引对象
        rng6 = PeriodIndex(period_rng, freq="M")
        # 创建另一个时间段索引对象
        other6 = period_range("2000-04-01", freq="M", periods=7)
        # 期望的结果为包含部分元素的时间段索引对象

        # 准备测试数据集7
        period_rng = ["2003", "2007", "2006", "2005", "2004"]
        # 创建第七个时间段索引对象
        rng7 = PeriodIndex(period_rng, freq="Y")
        # 创建另一个时间段索引对象
        other7 = period_range("1998-01-01", freq="Y", periods=8)
        # 期望的结果为包含部分元素的时间段索引对象

        # 遍历所有测试数据集，进行差异操作并验证结果
        for rng, other, expected in [
            (rng1, other1, expected1),
            (rng2, other2, expected2),
            (rng3, other3, expected3),
            (rng4, other4, expected4),
            (rng5, other5, expected5),
            (rng6, other6, expected6),
            (rng7, other7, expected7),
        ]:
            # 执行时间段索引对象的差异操作
            result_difference = rng.difference(other, sort=sort)
            # 如果不需要排序且other非空，根据GH#24959的建议，对期望结果进行排序
            if sort is None and len(other):
                expected = expected.sort_values()
            # 断言差异操作的结果与期望结果相等
            tm.assert_index_equal(result_difference, expected)
    def test_difference_freq(self, sort):
        # GH14323: difference of Period MUST preserve frequency
        # but the ability to union results must be preserved

        # 创建一个日期范围索引，频率为每日
        index = period_range("20160920", "20160925", freq="D")

        # 创建另一个日期范围索引，频率为每日
        other = period_range("20160921", "20160924", freq="D")
        # 预期的索引结果
        expected = PeriodIndex(["20160920", "20160925"], freq="D")
        # 求两个索引的差集，并验证结果与预期是否相等
        idx_diff = index.difference(other, sort)
        tm.assert_index_equal(idx_diff, expected)
        # 验证差集的频率属性与预期一致
        tm.assert_attr_equal("freq", idx_diff, expected)

        # 创建另一个日期范围索引，频率为每日
        other = period_range("20160922", "20160925", freq="D")
        # 求两个索引的差集，并验证结果与预期是否相等
        idx_diff = index.difference(other, sort)
        expected = PeriodIndex(["20160920", "20160921"], freq="D")
        tm.assert_index_equal(idx_diff, expected)

    def test_intersection_equal_duplicates(self):
        # GH#38302
        # 创建一个两个元素的日期范围索引
        idx = period_range("2011-01-01", periods=2)
        # 将索引重复一次，形成包含重复元素的索引
        idx_dup = idx.append(idx)
        # 求索引与自身的交集，并验证结果与原索引是否相等
        result = idx_dup.intersection(idx_dup)
        tm.assert_index_equal(result, idx)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_union_duplicates(self):
        # GH#36289
        # 创建一个两个元素的日期范围索引
        idx = period_range("2011-01-01", periods=2)
        # 将索引重复一次，形成包含重复元素的索引
        idx_dup = idx.append(idx)

        # 创建另一个两个元素的日期范围索引
        idx2 = period_range("2011-01-02", periods=2)
        # 将第二个索引也重复一次，形成包含重复元素的索引
        idx2_dup = idx2.append(idx2)
        # 求两个重复索引的并集，并验证结果与预期是否相等
        result = idx_dup.union(idx2_dup)

        # 预期的并集结果
        expected = PeriodIndex(
            [
                "2011-01-01",
                "2011-01-01",
                "2011-01-02",
                "2011-01-02",
                "2011-01-03",
                "2011-01-03",
            ],
            freq="D",
        )
        tm.assert_index_equal(result, expected)
```