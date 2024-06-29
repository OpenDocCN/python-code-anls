# `D:\src\scipysrc\pandas\pandas\tests\indexes\interval\test_setops.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于编写和运行测试用例

from pandas import (  # 从 Pandas 库中导入以下模块
    Index,  # 数据结构，用于索引标签
    IntervalIndex,  # 数据结构，用于区间索引
    Timestamp,  # 数据结构，表示时间戳
    interval_range,  # 函数，用于创建区间范围
)
import pandas._testing as tm  # 导入 Pandas 测试模块，用于编写测试断言

def monotonic_index(start, end, dtype="int64", closed="right"):
    # 创建一个单调递增的 IntervalIndex 对象
    return IntervalIndex.from_breaks(np.arange(start, end, dtype=dtype), closed=closed)

def empty_index(dtype="int64", closed="right"):
    # 创建一个空的 IntervalIndex 对象
    return IntervalIndex(np.array([], dtype=dtype), closed=closed)

class TestIntervalIndex:
    def test_union(self, closed, sort):
        # 测试 IntervalIndex 对象的 union 方法
        
        # 创建两个单调递增的 IntervalIndex 对象
        index = monotonic_index(0, 11, closed=closed)
        other = monotonic_index(5, 13, closed=closed)

        # 预期的合并结果 IntervalIndex 对象
        expected = monotonic_index(0, 13, closed=closed)
        
        # 执行 union 操作，得到合并后的结果
        result = index[::-1].union(other, sort=sort)
        
        # 根据 sort 参数判断是否需要排序结果
        if sort in (None, True):
            tm.assert_index_equal(result, expected)
        else:
            tm.assert_index_equal(result.sort_values(), expected)

        # 再次执行 union 操作，交换参数位置，得到相同的预期结果
        result = other[::-1].union(index, sort=sort)
        if sort in (None, True):
            tm.assert_index_equal(result, expected)
        else:
            tm.assert_index_equal(result.sort_values(), expected)

        # 测试 IntervalIndex 对象自身与自身的 union 操作
        tm.assert_index_equal(index.union(index, sort=sort), index)
        # 测试 IntervalIndex 对象与其部分子集的 union 操作
        tm.assert_index_equal(index.union(index[:1], sort=sort), index)

    def test_union_empty_result(self, closed, sort):
        # 测试 union 方法在结果为空时的行为
        
        # 创建一个空的 IntervalIndex 对象
        index = empty_index(dtype="int64", closed=closed)
        
        # 执行自身与自身的 union 操作，预期结果为原始对象
        result = index.union(index, sort=sort)
        tm.assert_index_equal(result, index)

        # 创建一个空的不同 dtype 的 IntervalIndex 对象
        other = empty_index(dtype="float64", closed=closed)
        
        # 执行与不同 dtype 的空对象的 union 操作，预期结果为另一个空对象
        result = index.union(other, sort=sort)
        expected = other
        tm.assert_index_equal(result, expected)

        # 再次执行与自身的 union 操作，预期结果仍为另一个空对象
        other = index.union(index, sort=sort)
        tm.assert_index_equal(result, expected)

        # 创建一个空的不同 dtype 的 IntervalIndex 对象
        other = empty_index(dtype="uint64", closed=closed)
        
        # 执行与不同 dtype 的空对象的 union 操作，预期结果为之前的空对象
        result = index.union(other, sort=sort)
        tm.assert_index_equal(result, expected)

        # 执行与不同 dtype 的空对象的 union 操作，预期结果为之前的空对象
        result = other.union(index, sort=sort)
        tm.assert_index_equal(result, expected)
    # 定义测试方法，用于测试索引的交集操作
    def test_intersection(self, closed, sort):
        # 创建闭区间索引，范围是[0, 11]，根据参数确定是否包含端点
        index = monotonic_index(0, 11, closed=closed)
        # 创建另一个闭区间索引，范围是[5, 13]，根据参数确定是否包含端点
        other = monotonic_index(5, 13, closed=closed)

        # 创建期望的闭区间索引，范围是[5, 11]，根据参数确定是否包含端点
        expected = monotonic_index(5, 11, closed=closed)
        # 计算索引的逆序与另一个索引的交集，根据参数确定是否排序结果
        result = index[::-1].intersection(other, sort=sort)
        # 根据排序参数检查结果是否与期望相等
        if sort in (None, True):
            tm.assert_index_equal(result, expected)
        else:
            tm.assert_index_equal(result.sort_values(), expected)

        # 交换索引顺序并重复上述操作
        result = other[::-1].intersection(index, sort=sort)
        if sort in (None, True):
            tm.assert_index_equal(result, expected)
        else:
            tm.assert_index_equal(result.sort_values(), expected)

        # 检查索引与自身的交集，根据参数确定是否排序结果
        tm.assert_index_equal(index.intersection(index, sort=sort), index)

        # GH 26225: 嵌套区间
        # 创建包含元组的区间索引，再计算与另一个索引的交集
        index = IntervalIndex.from_tuples([(1, 2), (1, 3), (1, 4), (0, 2)])
        other = IntervalIndex.from_tuples([(1, 2), (1, 3)])
        expected = IntervalIndex.from_tuples([(1, 2), (1, 3)])
        result = index.intersection(other)
        tm.assert_index_equal(result, expected)

        # GH 26225
        index = IntervalIndex.from_tuples([(0, 3), (0, 2)])
        other = IntervalIndex.from_tuples([(0, 2), (1, 3)])
        expected = IntervalIndex.from_tuples([(0, 2)])
        result = index.intersection(other)
        tm.assert_index_equal(result, expected)

        # GH 26225: 包含重复的 NaN 元素
        index = IntervalIndex([np.nan, np.nan])
        other = IntervalIndex([np.nan])
        expected = IntervalIndex([np.nan])
        result = index.intersection(other)
        tm.assert_index_equal(result, expected)

    # 测试空结果的交集情况
    def test_intersection_empty_result(self, closed, sort):
        index = monotonic_index(0, 11, closed=closed)

        # GH 19101: 空结果，但数据类型相同
        other = monotonic_index(300, 314, closed=closed)
        expected = empty_index(dtype="int64", closed=closed)
        result = index.intersection(other, sort=sort)
        tm.assert_index_equal(result, expected)

        # GH 19101: 空结果，不同数值类型 -> 公共类型为 float64
        other = monotonic_index(300, 314, dtype="float64", closed=closed)
        result = index.intersection(other, sort=sort)
        expected = other[:0]
        tm.assert_index_equal(result, expected)

        other = monotonic_index(300, 314, dtype="uint64", closed=closed)
        result = index.intersection(other, sort=sort)
        tm.assert_index_equal(result, expected)

    # 测试重复元素的交集情况
    def test_intersection_duplicates(self):
        # GH#38743
        index = IntervalIndex.from_tuples([(1, 2), (1, 2), (2, 3), (3, 4)])
        other = IntervalIndex.from_tuples([(1, 2), (2, 3)])
        expected = IntervalIndex.from_tuples([(1, 2), (2, 3)])
        result = index.intersection(other)
        tm.assert_index_equal(result, expected)
    # 定义一个测试方法，用于测试 IntervalIndex 的 difference 操作
    def test_difference(self, closed, sort):
        # 创建一个 IntervalIndex 对象，从给定的左右边界数组创建，指定是否闭合
        index = IntervalIndex.from_arrays([1, 0, 3, 2], [1, 2, 3, 4], closed=closed)
        # 对该 IntervalIndex 对象执行 difference 操作，将结果保存到 result 中
        result = index.difference(index[:1], sort=sort)
        # 创建预期的 IntervalIndex 对象，去掉第一个元素
        expected = index[1:]
        # 如果 sort 参数为 None，则对预期结果进行排序
        if sort is None:
            expected = expected.sort_values()
        # 使用 assert_index_equal 函数断言 result 和 expected 是否相等
        tm.assert_index_equal(result, expected)

        # GH 19101: 空结果，但是数据类型相同
        # 对同一个 IntervalIndex 对象执行 difference 操作，预期结果是一个空的索引
        result = index.difference(index, sort=sort)
        expected = empty_index(dtype="int64", closed=closed)
        tm.assert_index_equal(result, expected)

        # GH 19101: 空结果，但是数据类型不同
        # 创建另一个 IntervalIndex 对象 other，从 index 的左边界转换为 float64 类型
        other = IntervalIndex.from_arrays(
            index.left.astype("float64"), index.right, closed=closed
        )
        # 对 index 和 other 执行 difference 操作，预期结果是一个空的索引
        result = index.difference(other, sort=sort)
        tm.assert_index_equal(result, expected)

    # 定义一个测试方法，用于测试 IntervalIndex 的 symmetric_difference 操作
    def test_symmetric_difference(self, closed, sort):
        # 创建一个单调递增的 IntervalIndex 对象
        index = monotonic_index(0, 11, closed=closed)
        # 对 index[1:] 和 index[:-1] 执行 symmetric_difference 操作，保存结果到 result 中
        result = index[1:].symmetric_difference(index[:-1], sort=sort)
        # 创建预期的 IntervalIndex 对象，包含 index 的第一个和最后一个元素
        expected = IntervalIndex([index[0], index[-1]])
        # 如果 sort 参数是 None 或 True，则断言 result 和 expected 是否相等
        if sort in (None, True):
            tm.assert_index_equal(result, expected)
        # 否则，对 result 进行排序后再进行断言
        else:
            tm.assert_index_equal(result.sort_values(), expected)

        # GH 19101: 空结果，但是数据类型相同
        # 对同一个 IntervalIndex 对象执行 symmetric_difference 操作，预期结果是一个空的索引
        result = index.symmetric_difference(index, sort=sort)
        expected = empty_index(dtype="int64", closed=closed)
        # 根据 sort 参数的值来断言 result 和 expected 是否相等或经过排序后相等
        if sort in (None, True):
            tm.assert_index_equal(result, expected)
        else:
            tm.assert_index_equal(result.sort_values(), expected)

        # GH 19101: 空结果，但是数据类型不同
        # 创建另一个 IntervalIndex 对象 other，从 index 的左边界转换为 float64 类型
        other = IntervalIndex.from_arrays(
            index.left.astype("float64"), index.right, closed=closed
        )
        # 对 index 和 other 执行 symmetric_difference 操作，预期结果是一个空的索引
        result = index.symmetric_difference(other, sort=sort)
        expected = empty_index(dtype="float64", closed=closed)
        tm.assert_index_equal(result, expected)

    # 使用 pytest 标记来忽略 RuntimeWarning，并对多个操作名称进行参数化测试
    @pytest.mark.filterwarnings("ignore:'<' not supported between:RuntimeWarning")
    @pytest.mark.parametrize(
        "op_name", ["union", "intersection", "difference", "symmetric_difference"]
    )
    # 定义一个测试方法，用于测试不兼容类型的情况
    def test_set_incompatible_types(self, closed, op_name, sort):
        # 生成一个单调递增的整数索引
        index = monotonic_index(0, 11, closed=closed)
        # 获取索引对象的指定操作方法，如 union、intersection 等
        set_op = getattr(index, op_name)

        # TODO: 标准化非联合集合操作的返回类型（self vs other）
        # 当操作名称为 "difference" 时，预期结果为原索引对象
        if op_name == "difference":
            expected = index
        else:
            # 将索引对象转换为对象类型后执行指定的集合操作，传入另一个索引对象
            expected = getattr(index.astype("O"), op_name)(Index([1, 2, 3]))
        # 执行集合操作，传入另一个索引对象和排序方式
        result = set_op(Index([1, 2, 3]), sort=sort)
        # 断言操作结果与预期结果相等
        tm.assert_index_equal(result, expected)

        # 对于混合的 closed 值 -> 强制转换为对象类型
        for other_closed in {"right", "left", "both", "neither"} - {closed}:
            # 生成一个具有不同 closed 值的单调递增索引对象
            other = monotonic_index(0, 11, closed=other_closed)
            # 将当前索引对象转换为对象类型后执行指定的集合操作，传入另一个索引对象
            expected = getattr(index.astype(object), op_name)(other, sort=sort)
            if op_name == "difference":
                expected = index
            # 执行集合操作，传入另一个索引对象和排序方式
            result = set_op(other, sort=sort)
            # 断言操作结果与预期结果相等
            tm.assert_index_equal(result, expected)

        # 处理不兼容的 dtype（GH 19016） -> 强制转换为对象类型
        # 生成一个时间间隔范围索引对象
        other = interval_range(Timestamp("20180101"), periods=9, closed=closed)
        # 将当前索引对象转换为对象类型后执行指定的集合操作，传入时间间隔范围索引对象
        expected = getattr(index.astype(object), op_name)(other, sort=sort)
        if op_name == "difference":
            expected = index
        # 执行集合操作，传入时间间隔范围索引对象和排序方式
        result = set_op(other, sort=sort)
        # 断言操作结果与预期结果相等
        tm.assert_index_equal(result, expected)
```