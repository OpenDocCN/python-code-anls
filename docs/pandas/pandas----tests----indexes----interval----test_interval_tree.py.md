# `D:\src\scipysrc\pandas\pandas\tests\indexes\interval\test_interval_tree.py`

```
from itertools import permutations

import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

from pandas._libs.interval import IntervalTree  # 导入 IntervalTree 类，用于处理区间
from pandas.compat import IS64  # 导入 IS64 变量，用于判断是否为64位系统

import pandas._testing as tm  # 导入 pandas 测试工具模块，用于测试辅助功能


def skipif_32bit(param):
    """
    Skip parameters in a parametrize on 32bit systems. Specifically used
    here to skip leaf_size parameters related to GH 23440.
    """
    marks = pytest.mark.skipif(not IS64, reason="GH 23440: int type mismatch on 32bit")
    return pytest.param(param, marks=marks)


@pytest.fixture(params=[skipif_32bit(1), skipif_32bit(2), 10])
def leaf_size(request):
    """
    Fixture to specify IntervalTree leaf_size parameter; to be used with the
    tree fixture.
    """
    return request.param


@pytest.fixture(
    params=[
        np.arange(5, dtype="int64"),
        np.arange(5, dtype="uint64"),
        np.arange(5, dtype="float64"),
        np.array([0, 1, 2, 3, 4, np.nan], dtype="float64"),
    ]
)
def tree(request, leaf_size):
    left = request.param
    return IntervalTree(left, left + 2, leaf_size=leaf_size)


class TestIntervalTree:
    def test_get_indexer(self, tree):
        result = tree.get_indexer(np.array([1.0, 5.5, 6.5]))
        expected = np.array([0, 4, -1], dtype="intp")
        tm.assert_numpy_array_equal(result, expected)

        with pytest.raises(
            KeyError, match="'indexer does not intersect a unique set of intervals'"
        ):
            tree.get_indexer(np.array([3.0]))

    @pytest.mark.parametrize(
        "dtype, target_value, target_dtype",
        [("int64", 2**63 + 1, "uint64"), ("uint64", -1, "int64")],
    )
    def test_get_indexer_overflow(self, dtype, target_value, target_dtype):
        left, right = np.array([0, 1], dtype=dtype), np.array([1, 2], dtype=dtype)
        tree = IntervalTree(left, right)

        result = tree.get_indexer(np.array([target_value], dtype=target_dtype))
        expected = np.array([-1], dtype="intp")
        tm.assert_numpy_array_equal(result, expected)

    def test_get_indexer_non_unique(self, tree):
        indexer, missing = tree.get_indexer_non_unique(np.array([1.0, 2.0, 6.5]))

        result = indexer[:1]
        expected = np.array([0], dtype="intp")
        tm.assert_numpy_array_equal(result, expected)

        result = np.sort(indexer[1:3])
        expected = np.array([0, 1], dtype="intp")
        tm.assert_numpy_array_equal(result, expected)

        result = np.sort(indexer[3:])
        expected = np.array([-1], dtype="intp")
        tm.assert_numpy_array_equal(result, expected)

        result = missing
        expected = np.array([2], dtype="intp")
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "dtype, target_value, target_dtype",
        [("int64", 2**63 + 1, "uint64"), ("uint64", -1, "int64")],
    )
    # 定义一个测试函数，用于测试在存在重叠区间的情况下，获取索引器的行为
    def test_get_indexer_non_unique_overflow(self, dtype, target_value, target_dtype):
        # 创建两个数组 left 和 right，分别包含两个元素 0 和 2，数据类型为参数 dtype 指定的类型
        left, right = np.array([0, 2], dtype=dtype), np.array([1, 3], dtype=dtype)
        # 使用 IntervalTree 类构造一个树结构，表示区间 [0, 1) 和 [2, 3)
        tree = IntervalTree(left, right)
        # 创建一个包含单个目标值的数组 target，数据类型为 target_dtype 指定的类型
        target = np.array([target_value], dtype=target_dtype)

        # 调用 IntervalTree 对象的 get_indexer_non_unique 方法，获取目标值在树中的索引器和缺失值
        result_indexer, result_missing = tree.get_indexer_non_unique(target)
        # 创建一个预期的索引器数组，包含一个值 -1，数据类型为 "intp"
        expected_indexer = np.array([-1], dtype="intp")
        # 使用 pytest 断言验证获取的索引器与预期索引器相等
        tm.assert_numpy_array_equal(result_indexer, expected_indexer)

        # 创建一个预期的缺失值数组，包含一个值 0，数据类型为 "intp"
        expected_missing = np.array([0], dtype="intp")
        # 使用 pytest 断言验证获取的缺失值与预期缺失值相等
        tm.assert_numpy_array_equal(result_missing, expected_missing)

    # 使用 pytest 的参数化装饰器，依次传入 "int64", "float64", "uint64" 三种数据类型进行测试
    @pytest.mark.parametrize("dtype", ["int64", "float64", "uint64"])
    # 定义一个测试函数，测试在存在重复区间的情况下的行为
    def test_duplicates(self, dtype):
        # 创建一个包含三个元素 0 的数组 left，数据类型为参数 dtype 指定的类型
        left = np.array([0, 0, 0], dtype=dtype)
        # 使用 IntervalTree 类构造一个树结构，表示三个区间 [0, 1)
        tree = IntervalTree(left, left + 1)

        # 使用 pytest 的上下文管理器，验证在抛出 KeyError 异常时的行为
        with pytest.raises(
            KeyError, match="'indexer does not intersect a unique set of intervals'"
        ):
            # 调用 IntervalTree 对象的 get_indexer 方法，传入目标值数组 [0.5]，预期抛出异常
            tree.get_indexer(np.array([0.5]))

        # 调用 IntervalTree 对象的 get_indexer_non_unique 方法，传入目标值数组 [0.5]
        indexer, missing = tree.get_indexer_non_unique(np.array([0.5]))
        # 对获取的索引器数组进行排序
        result = np.sort(indexer)
        # 创建一个预期的索引器数组，包含三个值 [0, 1, 2]，数据类型为 "intp"
        expected = np.array([0, 1, 2], dtype="intp")
        # 使用 pytest 断言验证获取的排序后的索引器与预期索引器相等
        tm.assert_numpy_array_equal(result, expected)

        # 将获取的缺失值赋给 result
        result = missing
        # 创建一个预期的空数组，数据类型为 "intp"
        expected = np.array([], dtype="intp")
        # 使用 pytest 断言验证获取的缺失值与预期缺失值相等
        tm.assert_numpy_array_equal(result, expected)

    # 使用 pytest 的参数化装饰器，传入 leaf_size 参数进行测试
    @pytest.mark.parametrize(
        "leaf_size", [skipif_32bit(1), skipif_32bit(10), skipif_32bit(100), 10000]
    )
    # 定义一个测试函数，测试在不同的闭区间条件下，获取索引器的行为
    def test_get_indexer_closed(self, closed, leaf_size):
        # 创建一个包含 1000 个元素的浮点数数组 x
        x = np.arange(1000, dtype="float64")
        # 创建一个包含 1000 个元素的整数数组 found，并将 x 转换为整数类型
        found = x.astype("intp")
        # 创建一个包含 1000 个元素的整数数组 not_found，每个元素为 -1
        not_found = (-1 * np.ones(1000)).astype("intp")

        # 使用 IntervalTree 类构造一个树结构，表示区间 [x, x+0.5)
        tree = IntervalTree(x, x + 0.5, closed=closed, leaf_size=leaf_size)
        # 使用 pytest 断言验证获取的索引器与预期索引器相等
        tm.assert_numpy_array_equal(found, tree.get_indexer(x + 0.25))

        # 根据树的闭区间属性确定预期值
        expected = found if tree.closed_left else not_found
        # 使用 pytest 断言验证获取的索引器与预期索引器相等
        tm.assert_numpy_array_equal(expected, tree.get_indexer(x + 0.0))

        # 根据树的闭区间属性确定预期值
        expected = found if tree.closed_right else not_found
        # 使用 pytest 断言验证获取的索引器与预期索引器相等
        tm.assert_numpy_array_equal(expected, tree.get_indexer(x + 0.5))

    # 使用 pytest 的参数化装饰器，传入 left, right, expected 参数进行测试
    @pytest.mark.parametrize(
        "left, right, expected",
        [
            (np.array([0, 1, 4], dtype="int64"), np.array([2, 3, 5]), True),
            (np.array([0, 1, 2], dtype="int64"), np.array([5, 4, 3]), True),
            (np.array([0, 1, np.nan]), np.array([5, 4, np.nan]), True),
            (np.array([0, 2, 4], dtype="int64"), np.array([1, 3, 5]), False),
            (np.array([0, 2, np.nan]), np.array([1, 3, np.nan]), False),
        ],
    )
    # 使用 pytest 的参数化装饰器，传入 order 参数，测试区间重叠判断的行为
    @pytest.mark.parametrize("order", (list(x) for x in permutations(range(3))))
    # 定义一个测试函数，测试 IntervalTree 对象的 is_overlapping 属性
    def test_is_overlapping(self, closed, order, left, right, expected):
        # 创建一个 IntervalTree 对象，传入 left 和 right 数组，以及闭区间属性 closed
        tree = IntervalTree(left[order], right[order], closed=closed)
        # 获取树的 is_overlapping 属性，表示区间是否存在重叠
        result = tree.is_overlapping
        # 使用普通断言验证获取的结果与预期值相等
        assert result is expected

    # 使用 pytest 的参数化装饰器，传入 order 参数，测试区间重叠判断的行为
    @pytest.mark.parametrize("order", (list(x) for x in permutations(range(3))))
    def test_is_overlapping_endpoints(self, closed, order):
        """shared endpoints are marked as overlapping"""
        # GH 23309
        # 创建左右端点数组，使用指定的数据类型 int64
        left, right = np.arange(3, dtype="int64"), np.arange(1, 4)
        # 使用给定的端点数组创建区间树对象
        tree = IntervalTree(left[order], right[order], closed=closed)
        # 获取区间树对象的是否重叠属性
        result = tree.is_overlapping
        # 根据闭合方式确定预期的重叠结果
        expected = closed == "both"
        # 断言实际结果与预期结果相同
        assert result is expected

    @pytest.mark.parametrize(
        "left, right",
        [
            (np.array([], dtype="int64"), np.array([], dtype="int64")),
            (np.array([0], dtype="int64"), np.array([1], dtype="int64")),
            (np.array([np.nan]), np.array([np.nan])),
            (np.array([np.nan] * 3), np.array([np.nan] * 3)),
        ],
    )
    def test_is_overlapping_trivial(self, closed, left, right):
        # GH 23309
        # 使用给定的左右端点数组创建区间树对象
        tree = IntervalTree(left, right, closed=closed)
        # 断言区间树对象的是否重叠属性为 False
        assert tree.is_overlapping is False

    @pytest.mark.skipif(not IS64, reason="GH 23440")
    def test_construction_overflow(self):
        # GH 25485
        # 创建左右端点数组，左端点为0到100，右端点为int64类型的最大值
        left, right = np.arange(101, dtype="int64"), [np.iinfo(np.int64).max] * 101
        # 使用左右端点数组创建区间树对象
        tree = IntervalTree(left, right)

        # 计算根节点的中位数
        result = tree.root.pivot
        # 预期中位数为左右端点中位数的平均值
        expected = (50 + np.iinfo(np.int64).max) / 2
        # 断言实际结果与预期结果相同
        assert result == expected

    @pytest.mark.xfail(not IS64, reason="GH 23440")
    @pytest.mark.parametrize(
        "left, right, expected",
        [
            ([-np.inf, 1.0], [1.0, 2.0], 0.0),
            ([-np.inf, -2.0], [-2.0, -1.0], -2.0),
            ([-2.0, -1.0], [-1.0, np.inf], 0.0),
            ([1.0, 2.0], [2.0, np.inf], 2.0),
        ],
    )
    def test_inf_bound_infinite_recursion(self, left, right, expected):
        # GH 46658
        # 创建左右端点数组，每个元素重复101次
        tree = IntervalTree(left * 101, right * 101)

        # 获取根节点的中位数
        result = tree.root.pivot
        # 断言实际结果与预期结果相同
        assert result == expected
```