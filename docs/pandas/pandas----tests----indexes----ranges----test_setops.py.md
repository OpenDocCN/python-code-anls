# `D:\src\scipysrc\pandas\pandas\tests\indexes\ranges\test_setops.py`

```
# 导入必要的模块和类
from datetime import (
    datetime,
    timedelta,
)

from hypothesis import (
    assume,
    given,
    strategies as st,
)
import numpy as np  # 导入 NumPy 库
import pytest  # 导入 Pytest 库

from pandas import (
    Index,
    RangeIndex,
)
import pandas._testing as tm  # 导入 Pandas 测试模块


class TestRangeIndexSetOps:
    @pytest.mark.parametrize("dtype", [None, "int64", "uint64"])
    def test_intersection_mismatched_dtype(self, dtype):
        # 检查是否将类型转换为 float64 而不是 object
        index = RangeIndex(start=0, stop=20, step=2, name="foo")  # 创建 RangeIndex 对象
        index = Index(index, dtype=dtype)  # 使用指定类型创建 Index 对象

        flt = index.astype(np.float64)  # 将 Index 对象转换为 float64 类型

        # 由于 index.equals(flt)，所以我们通过快速路径并得到 RangeIndex 返回
        result = index.intersection(flt)  # 计算 index 和 flt 的交集
        tm.assert_index_equal(result, index, exact=True)  # 断言结果与 index 相等，要求精确匹配

        result = flt.intersection(index)  # 计算 flt 和 index 的交集
        tm.assert_index_equal(result, flt, exact=True)  # 断言结果与 flt 相等，要求精确匹配

        # 非空且不相等
        result = index.intersection(flt[1:])  # 计算 index 和 flt[1:] 的交集
        tm.assert_index_equal(result, flt[1:], exact=True)  # 断言结果与 flt[1:] 相等，要求精确匹配

        result = flt[1:].intersection(index)  # 计算 flt[1:] 和 index 的交集
        tm.assert_index_equal(result, flt[1:], exact=True)  # 断言结果与 flt[1:] 相等，要求精确匹配

        # 空的其他集合
        result = index.intersection(flt[:0])  # 计算 index 和 flt[:0] 的交集
        tm.assert_index_equal(result, flt[:0], exact=True)  # 断言结果与 flt[:0] 相等，要求精确匹配

        result = flt[:0].intersection(index)  # 计算 flt[:0] 和 index 的交集
        tm.assert_index_equal(result, flt[:0], exact=True)  # 断言结果与 flt[:0] 相等，要求精确匹配

    def test_intersection_empty(self, sort, names):
        # 在空交集上保持名称
        index = RangeIndex(start=0, stop=20, step=2, name=names[0])  # 创建 RangeIndex 对象

        # 空的其他集合
        result = index.intersection(index[:0].rename(names[1]), sort=sort)  # 计算 index 和空的 index[:0] 的交集，保持名称和排序
        tm.assert_index_equal(result, index[:0].rename(names[2]), exact=True)  # 断言结果与空的重命名后的 index[:0] 相等，要求精确匹配

        # 空的自身集合
        result = index[:0].intersection(index.rename(names[1]), sort=sort)  # 计算空的 index[:0] 和 index 的交集，保持名称和排序
        tm.assert_index_equal(result, index[:0].rename(names[2]), exact=True)  # 断言结果与空的重命名后的 index[:0] 相等，要求精确匹配
    # 定义一个测试方法，用于测试索引的交集操作，接受一个排序参数
    def test_intersection(self, sort):
        # 创建一个从0到20步长为2的整数索引对象
        index = RangeIndex(start=0, stop=20, step=2)
        # 创建一个包含从1到5的整数索引对象
        other = Index(np.arange(1, 6))
        # 对当前索引对象和另一个索引对象执行交集操作，返回结果索引
        result = index.intersection(other, sort=sort)
        # 期望的结果是将两个索引对象值的交集排序后形成的索引对象
        expected = Index(np.sort(np.intersect1d(index.values, other.values)))
        # 断言结果索引与期望索引相等
        tm.assert_index_equal(result, expected)

        # 对另一个索引对象和当前索引对象执行交集操作，返回结果索引
        result = other.intersection(index, sort=sort)
        # 期望的结果同样是将两个索引对象值的交集排序后形成的索引对象
        expected = Index(
            np.sort(np.asarray(np.intersect1d(index.values, other.values)))
        )
        # 断言结果索引与期望索引相等
        tm.assert_index_equal(result, expected)

        # 创建一个从1到6的整数范围索引对象
        other = RangeIndex(1, 6)
        # 对当前索引对象和另一个索引对象执行交集操作，返回结果索引
        result = index.intersection(other, sort=sort)
        # 期望的结果是将两个索引对象值的交集排序后形成的索引对象
        expected = Index(np.sort(np.intersect1d(index.values, other.values)))
        # 断言结果索引与期望索引相等，要求精确匹配
        tm.assert_index_equal(result, expected, exact="equiv")

        # 创建一个从5到0步长为-1的整数范围索引对象
        other = RangeIndex(5, 0, -1)
        # 对当前索引对象和另一个索引对象执行交集操作，返回结果索引
        result = index.intersection(other, sort=sort)
        # 期望的结果是将两个索引对象值的交集排序后形成的索引对象
        expected = Index(np.sort(np.intersect1d(index.values, other.values)))
        # 断言结果索引与期望索引相等，要求精确匹配
        tm.assert_index_equal(result, expected, exact="equiv")

        # 对另一个索引对象和当前索引对象执行交集操作，返回结果索引
        result = other.intersection(index, sort=sort)
        # 断言结果索引与期望索引相等，要求精确匹配
        tm.assert_index_equal(result, expected, exact="equiv")

        # 创建一个从10到-2步长为-2的整数范围索引对象
        first = RangeIndex(10, -2, -2)
        # 创建一个从5到-4步长为-1的整数范围索引对象
        other = RangeIndex(5, -4, -1)
        # 期望的结果是从4到-2步长为-2的整数范围索引对象
        expected = RangeIndex(start=4, stop=-2, step=-2)
        # 对当前索引对象和另一个索引对象执行交集操作，返回结果索引
        result = first.intersection(other, sort=sort)
        # 断言结果索引与期望索引相等
        tm.assert_index_equal(result, expected)

        # 对另一个索引对象和当前索引对象执行交集操作，返回结果索引
        result = other.intersection(first, sort=sort)
        # 断言结果索引与期望索引相等
        tm.assert_index_equal(result, expected)

        # 创建一个从5开始名为"foo"的整数范围索引对象
        index = RangeIndex(5, name="foo")

        # 创建一个从5到10步长为1名为"foo"的整数范围索引对象
        other = RangeIndex(5, 10, 1, name="foo")
        # 对当前索引对象和另一个索引对象执行交集操作，返回结果索引
        result = index.intersection(other, sort=sort)
        # 期望的结果是一个空的整数范围索引对象，名为"foo"
        expected = RangeIndex(0, 0, 1, name="foo")
        # 断言结果索引与期望索引相等
        tm.assert_index_equal(result, expected)

        # 创建一个从-1到-5步长为-1的整数范围索引对象
        other = RangeIndex(-1, -5, -1)
        # 对当前索引对象和另一个索引对象执行交集操作，返回结果索引
        result = index.intersection(other, sort=sort)
        # 期望的结果是一个空的整数范围索引对象
        expected = RangeIndex(0, 0, 1)
        # 断言结果索引与期望索引相等
        tm.assert_index_equal(result, expected)

        # 创建一个从0到0步长为1的整数范围索引对象
        other = RangeIndex(0, 0, 1)
        # 对当前索引对象和另一个索引对象执行交集操作，返回结果索引
        result = index.intersection(other, sort=sort)
        # 期望的结果是一个空的整数范围索引对象
        expected = RangeIndex(0, 0, 1)
        # 断言结果索引与期望索引相等
        tm.assert_index_equal(result, expected)

        # 对另一个索引对象和当前索引对象执行交集操作，返回结果索引
        result = other.intersection(index, sort=sort)
        # 断言结果索引与期望索引相等
        tm.assert_index_equal(result, expected)

    # 定义一个测试方法，用于测试基于起始值和最大公约数的非重叠值的交集操作，接受排序参数和名称参数
    def test_intersection_non_overlapping_gcd(self, sort, names):
        # 创建一个从1开始，到10步长为2的整数范围索引对象，名为names[0]
        index = RangeIndex(1, 10, 2, name=names[0])
        # 创建一个从0开始，到10步长为4的整数范围索引对象，名为names[1]
        other = RangeIndex(0, 10, 4, name=names[1])
        # 对当前索引对象和另一个索引对象执行交集操作，返回结果索引
        result = index.intersection(other, sort=sort)
        # 期望的结果是一个空的整数范围索引对象，名为names[2]
        expected = RangeIndex(0, 0, 1, name=names[2])
        # 断言结果索引与期望索引相等
        tm.assert_index_equal(result, expected)
    def test_union_noncomparable(self, sort):
        # corner case, Index with non-int64 dtype
        # 创建一个从 0 到 20，步长为 2 的 RangeIndex 对象
        index = RangeIndex(start=0, stop=20, step=2)
        # 创建一个包含当前日期时间加上一些时间增量的 Index 对象，dtype 为 object
        other = Index([datetime.now() + timedelta(i) for i in range(4)], dtype=object)
        # 对 index 和 other 进行 union 操作，返回结果存储在 result 中，sort 参数指定排序方式
        result = index.union(other, sort=sort)
        # 创建一个期望的 Index 对象，包含 index 和 other 合并后的结果
        expected = Index(np.concatenate((index, other)))
        # 断言 result 和 expected 相等
        tm.assert_index_equal(result, expected)

        # 对 other 和 index 进行 union 操作，返回结果存储在 result 中，sort 参数指定排序方式
        result = other.union(index, sort=sort)
        # 创建一个期望的 Index 对象，包含 other 和 index 合并后的结果
        expected = Index(np.concatenate((other, index)))
        # 断言 result 和 expected 相等
        tm.assert_index_equal(result, expected)

    def test_union_sorted(self, idx1, idx2, expected_sorted, expected_notsorted):
        # 对 idx1 和 idx2 进行 union 操作，sort=None 保持排序，返回结果存储在 res1 中
        res1 = idx1.union(idx2, sort=None)
        # 断言 res1 和 expected_sorted 相等
        tm.assert_index_equal(res1, expected_sorted, exact=True)

        # 对 idx1 和 idx2 进行 union 操作，sort=False 不排序，返回结果存储在 res1 中
        res1 = idx1.union(idx2, sort=False)
        # 断言 res1 和 expected_notsorted 相等
        tm.assert_index_equal(res1, expected_notsorted, exact=True)

        # 对 idx2 和 idx1 进行 union 操作，sort=None 保持排序，返回结果存储在 res2 中
        res2 = idx2.union(idx1, sort=None)
        # 创建一个新的 Index 对象，名为 idx1.name，包含 idx1 的值，并对其进行 union 操作，sort=None 保持排序，返回结果存储在 res3 中
        res3 = Index(idx1._values, name=idx1.name).union(idx2, sort=None)
        # 断言 res2 和 expected_sorted 相等
        tm.assert_index_equal(res2, expected_sorted, exact=True)
        # 断言 res3 和 expected_sorted 相等（使用 equiv 模式）
        tm.assert_index_equal(res3, expected_sorted, exact="equiv")

    def test_union_same_step_misaligned(self):
        # GH#44019
        # 创建一个步长为 4 的 RangeIndex 对象，范围从 0 到 20
        left = RangeIndex(range(0, 20, 4))
        # 创建一个步长为 4 的 RangeIndex 对象，范围从 1 到 21
        right = RangeIndex(range(1, 21, 4))

        # 对 left 和 right 进行 union 操作，返回结果存储在 result 中
        result = left.union(right)
        # 创建一个期望的 Index 对象，包含 left 和 right 合并后的结果
        expected = Index([0, 1, 4, 5, 8, 9, 12, 13, 16, 17])
        # 断言 result 和 expected 相等
        tm.assert_index_equal(result, expected, exact=True)

    def test_difference(self):
        # GH#12034 Cases where we operate against another RangeIndex and may
        #  get back another RangeIndex
        # 创建一个从 1 到 10 的 RangeIndex 对象，命名为 "foo"
        obj = RangeIndex.from_range(range(1, 10), name="foo")

        # 对 obj 自身进行 difference 操作，返回结果存储在 result 中
        result = obj.difference(obj)
        # 创建一个空的 RangeIndex 对象，命名为 "foo"
        expected = RangeIndex.from_range(range(0), name="foo")
        # 断言 result 和 expected 相等
        tm.assert_index_equal(result, expected, exact=True)

        # 对 obj 和一个重命名为 "bar" 的空 RangeIndex 对象进行 difference 操作，返回结果存储在 result 中
        result = obj.difference(expected.rename("bar"))
        # 断言 result 和 obj 名称为 None 的 Index 对象相等
        tm.assert_index_equal(result, obj.rename(None), exact=True)

        # 对 obj 和 obj[:3] 进行 difference 操作，返回结果存储在 result 中
        result = obj.difference(obj[:3])
        # 断言 result 和 obj[3:] 相等
        tm.assert_index_equal(result, obj[3:], exact=True)

        # 对 obj 和 obj[-3:] 进行 difference 操作，返回结果存储在 result 中
        result = obj.difference(obj[-3:])
        # 断言 result 和 obj[:-3] 相等
        tm.assert_index_equal(result, obj[:-3], exact=True)

        # 对 obj[::-1] 和 obj[-3:] 进行 difference 操作，sort=None，返回结果存储在 result 中
        result = obj[::-1].difference(obj[-3:])
        # 断言 result 和 obj[:-3] 相等
        tm.assert_index_equal(result, obj[:-3], exact=True)

        # 对 obj[::-1] 和 obj[-3:] 进行 difference 操作，sort=False，返回结果存储在 result 中
        result = obj[::-1].difference(obj[-3:], sort=False)
        # 断言 result 和 obj[:-3][::-1] 相等
        tm.assert_index_equal(result, obj[:-3][::-1], exact=True)

        # 对 obj[::-1] 和 obj[-3:][::-1] 进行 difference 操作，返回结果存储在 result 中
        result = obj[::-1].difference(obj[-3:][::-1])
        # 断言 result 和 obj[:-3] 相等
        tm.assert_index_equal(result, obj[:-3], exact=True)

        # 对 obj[::-1] 和 obj[-3:][::-1] 进行 difference 操作，sort=False，返回结果存储在 result 中
        result = obj[::-1].difference(obj[-3:][::-1], sort=False)
        # 断言 result 和 obj[:-3][::-1] 相等
        tm.assert_index_equal(result, obj[:-3][::-1], exact=True)

        # 对 obj 和 obj[2:6] 进行 difference 操作，返回结果存储在 result 中
        result = obj.difference(obj[2:6])
        # 创建一个期望的 Index 对象，包含差集的结果
        expected = Index([1, 2, 7, 8, 9], name="foo")
        # 断言 result 和 expected 相等
        tm.assert_index_equal(result, expected, exact=True)
    def test_difference_sort(self):
        # GH#44085 ensure we respect the sort keyword
        
        # 创建一个逆序的 Index 对象
        idx = Index(range(4))[::-1]
        # 创建一个包含单个元素的 Index 对象
        other = Index(range(3, 4))

        # 对 idx 和 other 进行差集运算，按照默认的排序方式
        result = idx.difference(other)
        # 创建预期的 Index 对象
        expected = Index(range(3))
        # 断言两个 Index 对象相等
        tm.assert_index_equal(result, expected, exact=True)

        # 对 idx 和 other 进行差集运算，禁用排序功能
        result = idx.difference(other, sort=False)
        # 将预期的 Index 对象逆序
        expected = expected[::-1]
        # 断言两个 Index 对象相等
        tm.assert_index_equal(result, expected, exact=True)

        # 处理交集为空的情况
        other = range(10, 12)
        # 对 idx 和 other 进行差集运算，不排序
        result = idx.difference(other, sort=None)
        # 将预期的 Index 对象逆序
        expected = idx[::-1]
        # 断言两个 Index 对象相等
        tm.assert_index_equal(result, expected, exact=True)

    def test_difference_mismatched_step(self):
        # 创建一个 RangeIndex 对象
        obj = RangeIndex.from_range(range(1, 10), name="foo")

        # 对 obj 和 obj[::2] 进行差集运算
        result = obj.difference(obj[::2])
        # 创建预期的 Index 对象
        expected = obj[1::2]
        # 断言两个 Index 对象相等
        tm.assert_index_equal(result, expected, exact=True)

        # 对 obj 的逆序和 obj[::2] 进行差集运算，禁用排序
        result = obj[::-1].difference(obj[::2], sort=False)
        # 将预期的 Index 对象逆序
        tm.assert_index_equal(result, expected[::-1], exact=True)

        # 对 obj 和 obj[1::2] 进行差集运算
        result = obj.difference(obj[1::2])
        # 创建预期的 Index 对象
        expected = obj[::2]
        # 断言两个 Index 对象相等
        tm.assert_index_equal(result, expected, exact=True)

        # 对 obj 的逆序和 obj[1::2] 进行差集运算，禁用排序
        result = obj[::-1].difference(obj[1::2], sort=False)
        # 将预期的 Index 对象逆序
        tm.assert_index_equal(result, expected[::-1], exact=True)

    def test_difference_interior_overlap_endpoints_preserved(self):
        # 创建左右两个 RangeIndex 对象
        left = RangeIndex(range(4))
        right = RangeIndex(range(1, 3))

        # 对 left 和 right 进行差集运算
        result = left.difference(right)
        # 创建预期的 RangeIndex 对象
        expected = RangeIndex(0, 4, 3)
        # 断言两个 RangeIndex 对象相等
        assert expected.tolist() == [0, 3]
        tm.assert_index_equal(result, expected, exact=True)

    def test_difference_endpoints_overlap_interior_preserved(self):
        # 创建左右两个 RangeIndex 对象
        left = RangeIndex(-8, 20, 7)
        right = RangeIndex(13, -9, -3)

        # 对 left 和 right 进行差集运算
        result = left.difference(right)
        # 创建预期的 RangeIndex 对象
        expected = RangeIndex(-1, 13, 7)
        # 断言两个 RangeIndex 对象相等
        assert expected.tolist() == [-1, 6]
        tm.assert_index_equal(result, expected, exact=True)

    def test_difference_interior_non_preserving(self):
        # 创建一个 Index 对象
        idx = Index(range(10))

        # 创建一个包含单个元素的 Index 对象
        other = idx[3:4]
        # 对 idx 和 other 进行差集运算
        result = idx.difference(other)
        # 创建预期的 Index 对象
        expected = Index([0, 1, 2, 4, 5, 6, 7, 8, 9])
        # 断言两个 Index 对象相等
        tm.assert_index_equal(result, expected, exact=True)

        # 创建一个步长大于2的 Index 对象
        other = idx[::3]
        # 对 idx 和 other 进行差集运算
        result = idx.difference(other)
        # 创建预期的 Index 对象
        expected = Index([1, 2, 4, 5, 7, 8])
        # 断言两个 Index 对象相等
        tm.assert_index_equal(result, expected, exact=True)

        # 创建一个包含单个元素的 Index 对象
        other = idx[:10:2]
        # 对 idx 和 other 进行差集运算
        result = obj.difference(other)
        # 创建预期的 Index 对象
        expected = Index([1, 3, 5, 7, 9] + list(range(10, 20)))
        # 断言两个 Index 对象相等
        tm.assert_index_equal(result, expected, exact=True)

        # 创建一个不重合的 Index 对象
        other = idx[1:11:2]
        # 对 idx 和 other 进行差集运算
        result = obj.difference(other)
        # 创建预期的 Index 对象
        expected = Index([0, 2, 4, 6, 8, 10] + list(range(11, 20)))
        # 断言两个 Index 对象相等
        tm.assert_index_equal(result, expected, exact=True)
    # 定义一个测试函数，用于测试 RangeIndex 的对称差集操作
    def test_symmetric_difference(self):
        # 创建一个名为 "foo" 的 RangeIndex 对象，包含从 1 到 9 的整数范围
        left = RangeIndex.from_range(range(1, 10), name="foo")

        # 对同一个 RangeIndex 对象执行对称差集操作，预期结果是空的 RangeIndex 对象
        result = left.symmetric_difference(left)
        expected = RangeIndex.from_range(range(0), name="foo")
        # 断言两个索引对象相等
        tm.assert_index_equal(result, expected)

        # 对 left 对象和重命名为 "bar" 的 expected 对象执行对称差集操作
        result = left.symmetric_difference(expected.rename("bar"))
        # 预期结果是 left 对象，名称为空
        tm.assert_index_equal(result, left.rename(None))

        # 对 left 对象的子集进行对称差集操作，预期结果包含 [1, 2, 8, 9] 这些元素
        result = left[:-2].symmetric_difference(left[2:])
        expected = Index([1, 2, 8, 9], name="foo")
        # 断言两个索引对象相等，要求精确匹配
        tm.assert_index_equal(result, expected, exact=True)

        # 创建一个从 10 到 14 的 RangeIndex 对象
        right = RangeIndex.from_range(range(10, 15))

        # 对 left 和 right 对象执行对称差集操作，预期结果包含从 1 到 14 的整数范围
        result = left.symmetric_difference(right)
        expected = RangeIndex.from_range(range(1, 15))
        # 断言两个索引对象相等
        tm.assert_index_equal(result, expected)

        # 对 left 和 right 的子集进行对称差集操作，预期结果包含 [1, 2, 3, ..., 14] 这些元素
        result = left.symmetric_difference(right[1:])
        expected = Index([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14])
        # 断言两个索引对象相等，要求精确匹配
        tm.assert_index_equal(result, expected, exact=True)
# 确保索引对象是 RangeIndex，或者无法被表示为 RangeIndex
def assert_range_or_not_is_rangelike(index):
    # 如果索引不是 RangeIndex 并且长度大于0，则进行下面的操作
    if not isinstance(index, RangeIndex) and len(index) > 0:
        # 计算索引的相邻元素之间的差异
        diff = index[:-1] - index[1:]
        # 断言所有差异不相等
        assert not (diff == diff[0]).all()


@given(
    st.integers(-20, 20),
    st.integers(-20, 20),
    st.integers(-20, 20),
    st.integers(-20, 20),
    st.integers(-20, 20),
    st.integers(-20, 20),
)
# 定义测试函数，验证：
# a) 确保匹配 Index[int64].difference
# b) 尽可能返回 RangeIndex
def test_range_difference(start1, stop1, step1, start2, stop2, step2):
    # 假设步长不为0，以确保 RangeIndex 能够正确创建
    assume(step1 != 0)
    assume(step2 != 0)

    # 创建左侧和右侧的 RangeIndex 对象
    left = RangeIndex(start1, stop1, step1)
    right = RangeIndex(start2, stop2, step2)

    # 计算 left 和 right 的差集，不进行排序
    result = left.difference(right, sort=None)
    # 断言结果是 RangeIndex 或者不符合 RangeIndex 的特征
    assert_range_or_not_is_rangelike(result)

    # 将 RangeIndex 转换为 Index[int64] 对象
    left_int64 = Index(left.to_numpy())
    right_int64 = Index(right.to_numpy())

    # 使用 int64 类型的索引对象计算差集，不排序
    alt = left_int64.difference(right_int64, sort=None)
    # 断言两种计算方式得到的结果相等
    tm.assert_index_equal(result, alt, exact="equiv")

    # 再次计算 left 和 right 的差集，但这次不排序
    result = left.difference(right, sort=False)
    # 断言结果是 RangeIndex 或者不符合 RangeIndex 的特征
    assert_range_or_not_is_rangelike(result)

    # 使用 int64 类型的索引对象计算差集，不排序
    alt = left_int64.difference(right_int64, sort=False)
    # 断言两种计算方式得到的结果相等
    tm.assert_index_equal(result, alt, exact="equiv")
```