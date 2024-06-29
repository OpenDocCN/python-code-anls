# `D:\src\scipysrc\pandas\pandas\tests\indexes\numeric\test_join.py`

```
import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于编写和运行测试

import pandas._testing as tm  # 导入pandas内部测试模块作为tm别名
from pandas.core.indexes.api import Index  # 从pandas核心索引API中导入Index类


class TestJoinInt64Index:
    def test_join_non_unique(self):
        left = Index([4, 4, 3, 3])  # 创建一个Index对象left，包含重复元素

        joined, lidx, ridx = left.join(left, return_indexers=True)
        # 对left进行自身的join操作，返回joined（合并后的索引）、lidx（左侧索引）、ridx（右侧索引）

        exp_joined = Index([4, 4, 4, 4, 3, 3, 3, 3])  # 期望的合并后的索引
        tm.assert_index_equal(joined, exp_joined)  # 断言joined与exp_joined相等，用于测试

        exp_lidx = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.intp)
        tm.assert_numpy_array_equal(lidx, exp_lidx)  # 断言lidx与exp_lidx的NumPy数组相等，用于测试

        exp_ridx = np.array([0, 1, 0, 1, 2, 3, 2, 3], dtype=np.intp)
        tm.assert_numpy_array_equal(ridx, exp_ridx)  # 断言ridx与exp_ridx的NumPy数组相等，用于测试

    def test_join_inner(self):
        index = Index(range(0, 20, 2), dtype=np.int64, name="lhs")  # 创建一个名为"lhs"的Index对象
        other = Index([7, 12, 25, 1, 2, 5], dtype=np.int64, name="rhs")  # 创建一个名为"rhs"的Index对象
        other_mono = Index([1, 2, 5, 7, 12, 25], dtype=np.int64, name="rhs")  # 创建另一个名为"rhs"的Index对象

        # not monotonic
        res, lidx, ridx = index.join(other, how="inner", return_indexers=True)
        # 对index和other进行inner join操作，返回res（合并后的索引）、lidx（左侧索引）、ridx（右侧索引）

        eres = Index([2, 12], dtype=np.int64, name="lhs")  # 期望的合并后的索引
        elidx = np.array([1, 6], dtype=np.intp)  # 期望的左侧索引
        eridx = np.array([4, 1], dtype=np.intp)  # 期望的右侧索引

        assert isinstance(res, Index) and res.dtype == np.int64
        tm.assert_index_equal(res, eres)  # 断言res与eres相等，用于测试
        tm.assert_numpy_array_equal(lidx, elidx)  # 断言lidx与elidx的NumPy数组相等，用于测试
        tm.assert_numpy_array_equal(ridx, eridx)  # 断言ridx与eridx的NumPy数组相等，用于测试

        # monotonic
        res, lidx, ridx = index.join(other_mono, how="inner", return_indexers=True)
        # 对index和other_mono进行inner join操作，返回res（合并后的索引）、lidx（左侧索引）、ridx（右侧索引）

        res2 = index.intersection(other_mono).set_names(["lhs"])
        tm.assert_index_equal(res, res2)  # 断言res与res2相等，用于测试

        elidx = np.array([1, 6], dtype=np.intp)  # 期望的左侧索引
        eridx = np.array([1, 4], dtype=np.intp)  # 期望的右侧索引
        assert isinstance(res, Index) and res.dtype == np.int64
        tm.assert_index_equal(res, eres)  # 断言res与eres相等，用于测试
        tm.assert_numpy_array_equal(lidx, elidx)  # 断言lidx与elidx的NumPy数组相等，用于测试
        tm.assert_numpy_array_equal(ridx, eridx)  # 断言ridx与eridx的NumPy数组相等，用于测试
    # 定义一个测试方法，用于测试 Index 类的左连接功能
    def test_join_left(self):
        # 创建一个索引对象 index，包含从 0 到 18 的偶数，数据类型为 np.int64，名称为 "lhs"
        index = Index(range(0, 20, 2), dtype=np.int64, name="lhs")
        # 创建另一个索引对象 other，包含指定的整数，数据类型为 np.int64，名称为 "rhs"
        other = Index([7, 12, 25, 1, 2, 5], dtype=np.int64, name="rhs")
        # 创建第三个索引对象 other_mono，数据类型与 other 相同，但元素顺序不同
        other_mono = Index([1, 2, 5, 7, 12, 25], dtype=np.int64, name="rhs")

        # 非单调情况下的左连接
        # 调用 index 对象的 join 方法，与 other 索引进行左连接，返回结果 res、左索引 lidx、右索引 ridx
        res, lidx, ridx = index.join(other, how="left", return_indexers=True)
        # 预期的结果索引对象为 index
        eres = index
        # 预期的右索引 ridx 为特定的 numpy 数组 eridx
        eridx = np.array([-1, 4, -1, -1, -1, -1, 1, -1, -1, -1], dtype=np.intp)

        # 断言 res 是 Index 类型且数据类型为 np.int64
        assert isinstance(res, Index) and res.dtype == np.int64
        # 断言 res 与 eres 相等
        tm.assert_index_equal(res, eres)
        # 断言 lidx 为 None
        assert lidx is None
        # 断言 ridx 与 eridx 相等
        tm.assert_numpy_array_equal(ridx, eridx)

        # 单调情况下的左连接
        # 调用 index 对象的 join 方法，与 other_mono 索引进行左连接，返回结果 res、左索引 lidx、右索引 ridx
        res, lidx, ridx = index.join(other_mono, how="left", return_indexers=True)
        # 更新预期的右索引 ridx 为特定的 numpy 数组 eridx
        eridx = np.array([-1, 1, -1, -1, -1, -1, 4, -1, -1, -1], dtype=np.intp)
        # 断言 res 是 Index 类型且数据类型为 np.int64
        assert isinstance(res, Index) and res.dtype == np.int64
        # 断言 res 与 eres 相等
        tm.assert_index_equal(res, eres)
        # 断言 lidx 为 None
        assert lidx is None
        # 断言 ridx 与 eridx 相等
        tm.assert_numpy_array_equal(ridx, eridx)

        # 非唯一情况下的左连接
        # 创建一个非唯一的索引对象 idx，数据包含指定的整数，名称为 "rhs"
        idx = Index([1, 1, 2, 5], name="rhs")
        # 创建另一个索引对象 idx2，包含指定的整数，名称为 "lhs"
        idx2 = Index([1, 2, 5, 7, 9], name="lhs")
        # 调用 idx2 对象的 join 方法，与 idx 索引进行左连接，返回结果 res、左索引 lidx、右索引 ridx
        res, lidx, ridx = idx2.join(idx, how="left", return_indexers=True)
        # 预期的结果索引对象为 eres
        eres = Index([1, 1, 2, 5, 7, 9], name="lhs")  # 1 在 idx2 中出现两次，因此应出现两次
        # 预期的右索引 ridx 为特定的 numpy 数组 eridx
        eridx = np.array([0, 1, 2, 3, -1, -1], dtype=np.intp)
        # 预期的左索引 lidx 为特定的 numpy 数组 elidx
        elidx = np.array([0, 0, 1, 2, 3, 4], dtype=np.intp)
        # 断言 res 与 eres 相等
        tm.assert_index_equal(res, eres)
        # 断言 lidx 与 elidx 相等
        tm.assert_numpy_array_equal(lidx, elidx)
        # 断言 ridx 与 eridx 相等
        tm.assert_numpy_array_equal(ridx, eridx)
    def test_join_right(self):
        # 创建一个包含偶数的索引对象，名称为"lhs"
        index = Index(range(0, 20, 2), dtype=np.int64, name="lhs")
        # 创建一个包含整数的索引对象，名称为"rhs"
        other = Index([7, 12, 25, 1, 2, 5], dtype=np.int64, name="rhs")
        # 创建一个包含整数的单调递增索引对象，名称为"rhs"
        other_mono = Index([1, 2, 5, 7, 12, 25], dtype=np.int64, name="rhs")

        # 非单调递增情况下的右连接
        res, lidx, ridx = index.join(other, how="right", return_indexers=True)
        eres = other
        elidx = np.array([-1, 6, -1, -1, 1, -1], dtype=np.intp)

        # 断言：other 是 Index 类型且数据类型为 np.int64
        assert isinstance(other, Index) and other.dtype == np.int64
        # 断言：res 和 eres 索引对象相等
        tm.assert_index_equal(res, eres)
        # 断言：lidx 和 elidx 数组相等
        tm.assert_numpy_array_equal(lidx, elidx)
        # 断言：ridx 为 None
        assert ridx is None

        # 单调递增情况下的右连接
        res, lidx, ridx = index.join(other_mono, how="right", return_indexers=True)
        eres = other_mono
        elidx = np.array([-1, 1, -1, -1, 6, -1], dtype=np.intp)
        # 断言：other 是 Index 类型且数据类型为 np.int64
        assert isinstance(other, Index) and other.dtype == np.int64
        # 断言：res 和 eres 索引对象相等
        tm.assert_index_equal(res, eres)
        # 断言：lidx 和 elidx 数组相等
        tm.assert_numpy_array_equal(lidx, elidx)
        # 断言：ridx 为 None
        assert ridx is None

        # 非唯一值情况下的右连接
        idx = Index([1, 1, 2, 5], name="lhs")
        idx2 = Index([1, 2, 5, 7, 9], name="rhs")
        res, lidx, ridx = idx.join(idx2, how="right", return_indexers=True)
        eres = Index([1, 1, 2, 5, 7, 9], name="rhs")  # 1 在 idx2 中出现了两次，因此应该是 x2
        elidx = np.array([0, 1, 2, 3, -1, -1], dtype=np.intp)
        eridx = np.array([0, 0, 1, 2, 3, 4], dtype=np.intp)
        # 断言：res 和 eres 索引对象相等
        tm.assert_index_equal(res, eres)
        # 断言：lidx 和 elidx 数组相等
        tm.assert_numpy_array_equal(lidx, elidx)
        # 断言：ridx 和 eridx 数组相等
        tm.assert_numpy_array_equal(ridx, eridx)

    def test_join_non_int_index(self):
        index = Index(range(0, 20, 2), dtype=np.int64)
        other = Index([3, 6, 7, 8, 10], dtype=object)

        # 外连接
        outer = index.join(other, how="outer")
        outer2 = other.join(index, how="outer")
        expected = Index([0, 2, 3, 4, 6, 7, 8, 10, 12, 14, 16, 18])
        # 断言：outer 和 outer2 索引对象相等
        tm.assert_index_equal(outer, outer2)
        # 断言：outer 和 expected 索引对象相等
        tm.assert_index_equal(outer, expected)

        # 内连接
        inner = index.join(other, how="inner")
        inner2 = other.join(index, how="inner")
        expected = Index([6, 8, 10])
        # 断言：inner 和 inner2 索引对象相等
        tm.assert_index_equal(inner, inner2)
        # 断言：inner 和 expected 索引对象相等
        tm.assert_index_equal(inner, expected)

        # 左连接
        left = index.join(other, how="left")
        # 断言：left 和 index 转换为 object 类型后的索引对象相等
        tm.assert_index_equal(left, index.astype(object))

        left2 = other.join(index, how="left")
        # 断言：left2 和 other 索引对象相等
        tm.assert_index_equal(left2, other)

        # 右连接
        right = index.join(other, how="right")
        # 断言：right 和 other 索引对象相等
        tm.assert_index_equal(right, other)

        right2 = other.join(index, how="right")
        # 断言：right2 和 index 转换为 object 类型后的索引对象相等
        tm.assert_index_equal(right2, index.astype(object))
    # 定义测试方法，用于测试 Index 对象的 join 方法中的 "outer" 模式
    def test_join_outer(self):
        # 创建一个 Index 对象，包含从 0 到 18 的偶数，数据类型为 np.int64
        index = Index(range(0, 20, 2), dtype=np.int64)
        # 创建另一个 Index 对象，包含指定整数，数据类型为 np.int64
        other = Index([7, 12, 25, 1, 2, 5], dtype=np.int64)
        # 创建第三个 Index 对象，包含与 other 相同整数，但已排序，数据类型为 np.int64
        other_mono = Index([1, 2, 5, 7, 12, 25], dtype=np.int64)

        # 在非单调情况下，使用 "outer" 模式进行 join 操作，返回结果和索引器
        res, lidx, ridx = index.join(other, how="outer", return_indexers=True)
        # 对比不含索引器的结果
        noidx_res = index.join(other, how="outer")
        # 确保结果 res 与不含索引器的结果 noidx_res 相等
        tm.assert_index_equal(res, noidx_res)

        # 预期的结果 Index 对象，包含合并后的整数序列，数据类型为 np.int64
        eres = Index([0, 1, 2, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 25], dtype=np.int64)
        # 预期的左索引器数组，指示每个元素在 index 中的位置，-1 表示未匹配
        elidx = np.array([0, -1, 1, 2, -1, 3, -1, 4, 5, 6, 7, 8, 9, -1], dtype=np.intp)
        # 预期的右索引器数组，指示每个元素在 other 中的位置，-1 表示未匹配
        eridx = np.array([-1, 3, 4, -1, 5, -1, 0, -1, -1, 1, -1, -1, -1, 2], dtype=np.intp)

        # 确保结果 res 是 Index 对象且数据类型为 np.int64
        assert isinstance(res, Index) and res.dtype == np.int64
        # 确保结果 res 与预期的 eres 相等
        tm.assert_index_equal(res, eres)
        # 确保左索引器 lidx 与预期的 elidx 相等
        tm.assert_numpy_array_equal(lidx, elidx)
        # 确保右索引器 ridx 与预期的 eridx 相等

        # 在单调情况下，使用 "outer" 模式进行 join 操作，返回结果和索引器
        res, lidx, ridx = index.join(other_mono, how="outer", return_indexers=True)
        # 对比不含索引器的结果
        noidx_res = index.join(other_mono, how="outer")
        # 确保结果 res 与不含索引器的结果 noidx_res 相等
        tm.assert_index_equal(res, noidx_res)

        # 更新预期的左索引器数组，指示每个元素在 index 中的位置，-1 表示未匹配
        elidx = np.array([0, -1, 1, 2, -1, 3, -1, 4, 5, 6, 7, 8, 9, -1], dtype=np.intp)
        # 更新预期的右索引器数组，指示每个元素在 other_mono 中的位置，-1 表示未匹配
        eridx = np.array([-1, 0, 1, -1, 2, -1, 3, -1, -1, 4, -1, -1, -1, 5], dtype=np.intp)

        # 确保结果 res 是 Index 对象且数据类型为 np.int64
        assert isinstance(res, Index) and res.dtype == np.int64
        # 确保结果 res 与预期的 eres 相等
        tm.assert_index_equal(res, eres)
        # 确保左索引器 lidx 与预期的 elidx 相等
        tm.assert_numpy_array_equal(lidx, elidx)
        # 确保右索引器 ridx 与预期的 eridx 相等
class TestJoinUInt64Index:
    @pytest.fixture
    def index_large(self):
        # 在 TestUInt64Index 中使用的大值，不需要与 int64/float64 兼容
        large = [2**63, 2**63 + 10, 2**63 + 15, 2**63 + 20, 2**63 + 25]
        return Index(large, dtype=np.uint64)

    def test_join_inner(self, index_large):
        # 创建另一个索引对象 other，包含一组 uint64 类型的值
        other = Index(2**63 + np.array([7, 12, 25, 1, 2, 10], dtype="uint64"))
        # 创建另一个单调递增的索引对象 other_mono，包含一组 uint64 类型的值
        other_mono = Index(2**63 + np.array([1, 2, 7, 10, 12, 25], dtype="uint64"))

        # 对 index_large 和 other 执行 inner join 操作，返回结果 res 和对应的索引数组 lidx 和 ridx
        res, lidx, ridx = index_large.join(other, how="inner", return_indexers=True)

        # 由于结果未排序，为了比较，需要对结果进行排序
        ind = res.argsort()
        res = res.take(ind)
        lidx = lidx.take(ind)
        ridx = ridx.take(ind)

        # 预期的结果对象 eres，以及对应的索引数组 elidx 和 eridx
        eres = Index(2**63 + np.array([10, 25], dtype="uint64"))
        elidx = np.array([1, 4], dtype=np.intp)
        eridx = np.array([5, 2], dtype=np.intp)

        # 断言结果 res 是 Index 对象且其数据类型为 np.uint64
        assert isinstance(res, Index) and res.dtype == np.uint64
        # 使用 pytest 的 assert_index_equal 检查 res 和 eres 是否相等
        tm.assert_index_equal(res, eres)
        # 使用 pytest 的 assert_numpy_array_equal 检查 lidx 和 elidx 是否相等
        tm.assert_numpy_array_equal(lidx, elidx)
        # 使用 pytest 的 assert_numpy_array_equal 检查 ridx 和 eridx 是否相等
        tm.assert_numpy_array_equal(ridx, eridx)

        # 对 index_large 和 other_mono 执行 inner join 操作，返回结果 res 和对应的索引数组 lidx 和 ridx
        res, lidx, ridx = index_large.join(
            other_mono, how="inner", return_indexers=True
        )

        # 使用 intersection 方法计算 index_large 和 other_mono 的交集，结果存储在 res2 中
        res2 = index_large.intersection(other_mono)
        # 使用 pytest 的 assert_index_equal 检查 res 和 res2 是否相等
        tm.assert_index_equal(res, res2)

        # 预期的结果对象 eres，以及对应的索引数组 elidx 和 eridx
        elidx = np.array([1, 4], dtype=np.intp)
        eridx = np.array([3, 5], dtype=np.intp)

        # 断言结果 res 是 Index 对象且其数据类型为 np.uint64
        assert isinstance(res, Index) and res.dtype == np.uint64
        # 使用 pytest 的 assert_index_equal 检查 res 和 eres 是否相等
        tm.assert_index_equal(res, eres)
        # 使用 pytest 的 assert_numpy_array_equal 检查 lidx 和 elidx 是否相等
        tm.assert_numpy_array_equal(lidx, elidx)
        # 使用 pytest 的 assert_numpy_array_equal 检查 ridx 和 eridx 是否相等
        tm.assert_numpy_array_equal(ridx, eridx)
    # 定义测试方法，用于测试索引对象与较大索引的左连接操作
    def test_join_left(self, index_large):
        # 创建另一个索引对象 `other`，其值比 `index_large` 中的值大
        other = Index(2**63 + np.array([7, 12, 25, 1, 2, 10], dtype="uint64"))
        # 创建另一个单调递增的索引对象 `other_mono`
        other_mono = Index(2**63 + np.array([1, 2, 7, 10, 12, 25], dtype="uint64"))

        # 在非单调递增情况下执行左连接操作，返回连接结果、左索引和右索引
        res, lidx, ridx = index_large.join(other, how="left", return_indexers=True)
        # 预期的连接结果为 `index_large`
        eres = index_large
        # 预期的右索引应为指定的数组 `eridx`
        eridx = np.array([-1, 5, -1, -1, 2], dtype=np.intp)

        # 断言连接结果为索引对象且数据类型为 `np.uint64`
        assert isinstance(res, Index) and res.dtype == np.uint64
        # 断言连接后的索引对象与预期结果相等
        tm.assert_index_equal(res, eres)
        # 左索引预期为 `None`
        assert lidx is None
        # 断言右索引与预期的 `eridx` 数组相等
        tm.assert_numpy_array_equal(ridx, eridx)

        # 在单调递增情况下执行左连接操作，返回连接结果、左索引和右索引
        res, lidx, ridx = index_large.join(other_mono, how="left", return_indexers=True)
        # 单调递增情况下预期的右索引为指定的数组 `eridx`
        eridx = np.array([-1, 3, -1, -1, 5], dtype=np.intp)

        # 断言连接结果为索引对象且数据类型为 `np.uint64`
        assert isinstance(res, Index) and res.dtype == np.uint64
        # 断言连接后的索引对象与预期结果相等
        tm.assert_index_equal(res, eres)
        # 左索引预期为 `None`
        assert lidx is None
        # 断言右索引与预期的 `eridx` 数组相等
        tm.assert_numpy_array_equal(ridx, eridx)

        # 在非唯一值情况下执行左连接操作，返回连接结果、左索引和右索引
        idx = Index(2**63 + np.array([1, 1, 2, 5], dtype="uint64"))
        idx2 = Index(2**63 + np.array([1, 2, 5, 7, 9], dtype="uint64"))
        res, lidx, ridx = idx2.join(idx, how="left", return_indexers=True)

        # 预期连接结果为 `idx2` 与 `idx` 的合并结果
        eres = Index(2**63 + np.array([1, 1, 2, 5, 7, 9], dtype="uint64"))
        # 预期的右索引应为指定的数组 `eridx`
        eridx = np.array([0, 1, 2, 3, -1, -1], dtype=np.intp)
        # 预期的左索引应为指定的数组 `elidx`
        elidx = np.array([0, 0, 1, 2, 3, 4], dtype=np.intp)

        # 断言连接结果与预期的 `eres` 索引对象相等
        tm.assert_index_equal(res, eres)
        # 断言左索引与预期的 `elidx` 数组相等
        tm.assert_numpy_array_equal(lidx, elidx)
        # 断言右索引与预期的 `eridx` 数组相等
        tm.assert_numpy_array_equal(ridx, eridx)
    # 定义测试方法，测试索引与大索引的右连接操作
    def test_join_right(self, index_large):
        # 创建一个包含超出普通整数范围的索引对象
        other = Index(2**63 + np.array([7, 12, 25, 1, 2, 10], dtype="uint64"))
        # 创建另一个单调递增的索引对象
        other_mono = Index(2**63 + np.array([1, 2, 7, 10, 12, 25], dtype="uint64"))

        # 执行右连接操作，返回连接结果、左索引和右索引
        res, lidx, ridx = index_large.join(other, how="right", return_indexers=True)
        # 预期的右连接结果
        eres = other
        # 预期的左索引
        elidx = np.array([-1, -1, 4, -1, -1, 1], dtype=np.intp)

        # 使用断言验证左索引
        tm.assert_numpy_array_equal(lidx, elidx)
        # 验证 other 是 Index 类型且数据类型为 np.uint64
        assert isinstance(other, Index) and other.dtype == np.uint64
        # 使用断言验证连接结果
        tm.assert_index_equal(res, eres)
        # 验证右索引为 None
        assert ridx is None

        # 执行单调递增索引的右连接操作
        res, lidx, ridx = index_large.join(
            other_mono, how="right", return_indexers=True
        )
        # 预期的右连接结果
        eres = other_mono
        # 预期的左索引
        elidx = np.array([-1, -1, -1, 1, -1, 4], dtype=np.intp)

        # 验证 other 是 Index 类型且数据类型为 np.uint64
        assert isinstance(other, Index) and other.dtype == np.uint64
        # 使用断言验证左索引
        tm.assert_numpy_array_equal(lidx, elidx)
        # 使用断言验证连接结果
        tm.assert_index_equal(res, eres)
        # 验证右索引为 None
        assert ridx is None

        # 创建包含非唯一值的索引对象
        idx = Index(2**63 + np.array([1, 1, 2, 5], dtype="uint64"))
        idx2 = Index(2**63 + np.array([1, 2, 5, 7, 9], dtype="uint64"))
        # 执行右连接操作，返回连接结果、左索引和右索引
        res, lidx, ridx = idx.join(idx2, how="right", return_indexers=True)

        # 预期的右连接结果
        eres = Index(2**63 + np.array([1, 1, 2, 5, 7, 9], dtype="uint64"))
        # 预期的左索引
        elidx = np.array([0, 1, 2, 3, -1, -1], dtype=np.intp)
        # 预期的右索引
        eridx = np.array([0, 0, 1, 2, 3, 4], dtype=np.intp)

        # 使用断言验证连接结果
        tm.assert_index_equal(res, eres)
        # 使用断言验证左索引
        tm.assert_numpy_array_equal(lidx, elidx)
        # 使用断言验证右索引
        tm.assert_numpy_array_equal(ridx, eridx)

    # 测试非整数类型索引的连接操作
    def test_join_non_int_index(self, index_large):
        # 创建包含对象类型数据的索引对象
        other = Index(2**63 + np.array([1, 5, 7, 10, 20], dtype="uint64"), dtype=object)

        # 执行外连接操作
        outer = index_large.join(other, how="outer")
        outer2 = other.join(index_large, how="outer")
        # 预期的外连接结果
        expected = Index(2**63 + np.array([0, 1, 5, 7, 10, 15, 20, 25], dtype="uint64"))
        # 使用断言验证外连接结果
        tm.assert_index_equal(outer, outer2)
        tm.assert_index_equal(outer, expected)

        # 执行内连接操作
        inner = index_large.join(other, how="inner")
        inner2 = other.join(index_large, how="inner")
        # 预期的内连接结果
        expected = Index(2**63 + np.array([10, 20], dtype="uint64"))
        # 使用断言验证内连接结果
        tm.assert_index_equal(inner, inner2)
        tm.assert_index_equal(inner, expected)

        # 执行左连接操作
        left = index_large.join(other, how="left")
        # 使用断言验证左连接结果
        tm.assert_index_equal(left, index_large.astype(object))

        left2 = other.join(index_large, how="left")
        # 使用断言验证左连接结果
        tm.assert_index_equal(left2, other)

        # 执行右连接操作
        right = index_large.join(other, how="right")
        # 使用断言验证右连接结果
        tm.assert_index_equal(right, other)

        right2 = other.join(index_large, how="right")
        # 使用断言验证右连接结果
        tm.assert_index_equal(right2, index_large.astype(object))
    # 定义一个测试函数，用于测试索引对象的外连接操作
    def test_join_outer(self, index_large):
        # 创建一个包含特定整数的索引对象 other
        other = Index(2**63 + np.array([7, 12, 25, 1, 2, 10], dtype="uint64"))
        # 创建一个单调递增的索引对象 other_mono
        other_mono = Index(2**63 + np.array([1, 2, 7, 10, 12, 25], dtype="uint64"))

        # 执行索引对象 index_large 和 other 的外连接操作，返回连接后的结果、左索引和右索引
        res, lidx, ridx = index_large.join(other, how="outer", return_indexers=True)
        # 执行索引对象 index_large 和 other 的外连接操作，返回连接后的结果（无索引器）
        noidx_res = index_large.join(other, how="outer")
        # 断言两个结果相等
        tm.assert_index_equal(res, noidx_res)

        # 期望的连接结果索引对象 eres
        eres = Index(2**63 + np.array([0, 1, 2, 7, 10, 12, 15, 20, 25], dtype="uint64"))
        # 期望的左索引数组 elidx
        elidx = np.array([0, -1, -1, -1, 1, -1, 2, 3, 4], dtype=np.intp)
        # 期望的右索引数组 eridx
        eridx = np.array([-1, 3, 4, 0, 5, 1, -1, -1, 2], dtype=np.intp)

        # 断言 res 是 Index 对象且其数据类型为 np.uint64
        assert isinstance(res, Index) and res.dtype == np.uint64
        # 断言 res 和 eres 相等
        tm.assert_index_equal(res, eres)
        # 断言 lidx 和 elidx 数组相等
        tm.assert_numpy_array_equal(lidx, elidx)
        # 断言 ridx 和 eridx 数组相等
        tm.assert_numpy_array_equal(ridx, eridx)

        # 执行索引对象 index_large 和 other_mono 的外连接操作，返回连接后的结果、左索引和右索引
        res, lidx, ridx = index_large.join(
            other_mono, how="outer", return_indexers=True
        )
        # 执行索引对象 index_large 和 other_mono 的外连接操作，返回连接后的结果（无索引器）
        noidx_res = index_large.join(other_mono, how="outer")
        # 断言两个结果相等
        tm.assert_index_equal(res, noidx_res)

        # 重新定义期望的左索引数组 elidx
        elidx = np.array([0, -1, -1, -1, 1, -1, 2, 3, 4], dtype=np.intp)
        # 重新定义期望的右索引数组 eridx
        eridx = np.array([-1, 0, 1, 2, 3, 4, -1, -1, 5], dtype=np.intp)

        # 断言 res 是 Index 对象且其数据类型为 np.uint64
        assert isinstance(res, Index) and res.dtype == np.uint64
        # 断言 res 和 eres 相等
        tm.assert_index_equal(res, eres)
        # 断言 lidx 和 elidx 数组相等
        tm.assert_numpy_array_equal(lidx, elidx)
        # 断言 ridx 和 eridx 数组相等
        tm.assert_numpy_array_equal(ridx, eridx)
```