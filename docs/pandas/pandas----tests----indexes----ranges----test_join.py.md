# `D:\src\scipysrc\pandas\pandas\tests\indexes\ranges\test_join.py`

```
import numpy as np  # 导入 NumPy 库，用于处理数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试

from pandas import (  # 从 pandas 库中导入 Index 和 RangeIndex 类
    Index,
    RangeIndex,
)
import pandas._testing as tm  # 导入 pandas 测试模块作为 tm

class TestJoin:
    def test_join_outer(self):
        # 使用 RangeIndex 创建索引对象 index，起始于 0，结束于 20，步长为 2
        index = RangeIndex(start=0, stop=20, step=2)
        # 创建另一个索引对象 other，包含从 25 到 14 的整数，dtype 为 np.int64
        other = Index(np.arange(25, 14, -1, dtype=np.int64))

        # 进行 outer 连接，并返回连接结果 res，以及左右索引的数组 lidx 和 ridx
        res, lidx, ridx = index.join(other, how="outer", return_indexers=True)
        # 对比不包含索引的结果
        noidx_res = index.join(other, how="outer")
        tm.assert_index_equal(res, noidx_res)  # 使用测试模块验证结果的一致性

        # 预期的结果索引对象 eres
        eres = Index(
            [0, 2, 4, 6, 8, 10, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        )
        # 预期的左右索引数组 elidx 和 eridx
        elidx = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, -1, 8, -1, 9, -1, -1, -1, -1, -1, -1, -1],
            dtype=np.intp,
        )
        eridx = np.array(
            [-1, -1, -1, -1, -1, -1, -1, -1, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            dtype=np.intp,
        )

        # 断言结果 res 的类型为 Index 类型且数据类型为 np.int64
        assert isinstance(res, Index) and res.dtype == np.dtype(np.int64)
        # 断言 res 不是 RangeIndex 类型
        assert not isinstance(res, RangeIndex)
        tm.assert_index_equal(res, eres, exact=True)  # 使用测试模块验证索引对象的一致性
        tm.assert_numpy_array_equal(lidx, elidx)  # 使用测试模块验证左索引数组的一致性
        tm.assert_numpy_array_equal(ridx, eridx)  # 使用测试模块验证右索引数组的一致性

        # 使用 RangeIndex 进行 outer 连接
        other = RangeIndex(25, 14, -1)

        res, lidx, ridx = index.join(other, how="outer", return_indexers=True)
        noidx_res = index.join(other, how="outer")
        tm.assert_index_equal(res, noidx_res)  # 使用测试模块验证结果的一致性

        assert isinstance(res, Index) and res.dtype == np.int64
        assert not isinstance(res, RangeIndex)
        tm.assert_index_equal(res, eres)  # 使用测试模块验证索引对象的一致性
        tm.assert_numpy_array_equal(lidx, elidx)  # 使用测试模块验证左索引数组的一致性
        tm.assert_numpy_array_equal(ridx, eridx)  # 使用测试模块验证右索引数组的一致性

    def test_join_inner(self):
        # 使用 RangeIndex 创建索引对象 index，起始于 0，结束于 20，步长为 2
        index = RangeIndex(start=0, stop=20, step=2)
        # 创建另一个索引对象 other，包含从 25 到 14 的整数，dtype 为 np.int64
        other = Index(np.arange(25, 14, -1, dtype=np.int64))

        # 进行 inner 连接，并返回连接结果 res，以及左右索引的数组 lidx 和 ridx
        res, lidx, ridx = index.join(other, how="inner", return_indexers=True)

        # 由于结果可能无序，因此对结果进行排序以进行比较
        ind = res.argsort()
        res = res.take(ind)
        lidx = lidx.take(ind)
        ridx = ridx.take(ind)

        # 预期的结果索引对象 eres
        eres = Index([16, 18])
        # 预期的左右索引数组 elidx 和 eridx
        elidx = np.array([8, 9], dtype=np.intp)
        eridx = np.array([9, 7], dtype=np.intp)

        # 断言结果 res 的类型为 Index 类型且数据类型为 np.int64
        assert isinstance(res, Index) and res.dtype == np.int64
        tm.assert_index_equal(res, eres)  # 使用测试模块验证索引对象的一致性
        tm.assert_numpy_array_equal(lidx, elidx)  # 使用测试模块验证左索引数组的一致性
        tm.assert_numpy_array_equal(ridx, eridx)  # 使用测试模块验证右索引数组的一致性

        # 使用 RangeIndex 进行 inner 连接
        other = RangeIndex(25, 14, -1)

        res, lidx, ridx = index.join(other, how="inner", return_indexers=True)

        assert isinstance(res, RangeIndex)
        tm.assert_index_equal(res, eres, exact="equiv")  # 使用测试模块验证索引对象的一致性
        tm.assert_numpy_array_equal(lidx, elidx)  # 使用测试模块验证左索引数组的一致性
        tm.assert_numpy_array_equal(ridx, eridx)  # 使用测试模块验证右索引数组的一致性
    def test_join_left(self):
        # 使用 RangeIndex 创建索引对象，起始值为 0，终止值为 20，步长为 2
        index = RangeIndex(start=0, stop=20, step=2)
        # 使用给定的 np 数组创建 Index 对象作为其他索引
        other = Index(np.arange(25, 14, -1, dtype=np.int64))

        # 进行左连接操作，返回结果 res、左侧索引 lidx、右侧索引 ridx，并返回索引数组
        res, lidx, ridx = index.join(other, how="left", return_indexers=True)
        # 预期的结果为 index 自身
        eres = index
        # 预期的右侧索引数组为指定的 np 数组
        eridx = np.array([-1, -1, -1, -1, -1, -1, -1, -1, 9, 7], dtype=np.intp)

        # 断言结果 res 是 RangeIndex 类型
        assert isinstance(res, RangeIndex)
        # 断言 res 与预期结果 eres 相等
        tm.assert_index_equal(res, eres)
        # 断言左侧索引 lidx 为 None
        assert lidx is None
        # 断言右侧索引 ridx 与预期结果 eridx 相等
        tm.assert_numpy_array_equal(ridx, eridx)

        # 使用 RangeIndex 进行左连接操作
        other = Index(np.arange(25, 14, -1, dtype=np.int64))

        # 重新进行左连接操作
        res, lidx, ridx = index.join(other, how="left", return_indexers=True)

        # 断言结果 res 是 RangeIndex 类型
        assert isinstance(res, RangeIndex)
        # 断言 res 与预期结果 eres 相等
        tm.assert_index_equal(res, eres)
        # 断言左侧索引 lidx 为 None
        assert lidx is None
        # 断言右侧索引 ridx 与预期结果 eridx 相等
        tm.assert_numpy_array_equal(ridx, eridx)

    def test_join_right(self):
        # 使用 RangeIndex 创建索引对象，起始值为 0，终止值为 20，步长为 2
        index = RangeIndex(start=0, stop=20, step=2)
        # 使用给定的 np 数组创建 Index 对象作为其他索引
        other = Index(np.arange(25, 14, -1, dtype=np.int64))

        # 进行右连接操作，返回结果 res、左侧索引 lidx、右侧索引 ridx，并返回索引数组
        res, lidx, ridx = index.join(other, how="right", return_indexers=True)
        # 预期的结果为 other 自身
        eres = other
        # 预期的左侧索引数组为指定的 np 数组
        elidx = np.array([-1, -1, -1, -1, -1, -1, -1, 9, -1, 8, -1], dtype=np.intp)

        # 断言 other 是 Index 类型且数据类型为 np.int64
        assert isinstance(other, Index) and other.dtype == np.int64
        # 断言 res 与预期结果 eres 相等
        tm.assert_index_equal(res, eres)
        # 断言左侧索引 lidx 与预期结果 elidx 相等
        tm.assert_numpy_array_equal(lidx, elidx)
        # 断言右侧索引 ridx 为 None
        assert ridx is None

        # 使用 RangeIndex 进行右连接操作
        other = RangeIndex(25, 14, -1)

        # 重新进行右连接操作
        res, lidx, ridx = index.join(other, how="right", return_indexers=True)
        # 预期的结果为 other 自身

        # 断言 other 是 RangeIndex 类型
        assert isinstance(other, RangeIndex)
        # 断言 res 与预期结果 eres 相等
        tm.assert_index_equal(res, eres)
        # 断言左侧索引 lidx 与预期结果 elidx 相等
        tm.assert_numpy_array_equal(lidx, elidx)
        # 断言右侧索引 ridx 为 None

    def test_join_non_int_index(self):
        # 使用 RangeIndex 创建索引对象，起始值为 0，终止值为 20，步长为 2
        index = RangeIndex(start=0, stop=20, step=2)
        # 使用给定的数组创建 Index 对象作为其他索引，数据类型为 object
        other = Index([3, 6, 7, 8, 10], dtype=object)

        # 执行外连接操作，返回结果 outer 和 outer2
        outer = index.join(other, how="outer")
        outer2 = other.join(index, how="outer")
        # 预期的结果为指定的 Index 对象
        expected = Index([0, 2, 3, 4, 6, 7, 8, 10, 12, 14, 16, 18])
        # 断言 outer 与 outer2 相等
        tm.assert_index_equal(outer, outer2)
        # 断言 outer 与预期结果 expected 相等
        tm.assert_index_equal(outer, expected)

        # 执行内连接操作，返回结果 inner 和 inner2
        inner = index.join(other, how="inner")
        inner2 = other.join(index, how="inner")
        # 预期的结果为指定的 Index 对象
        expected = Index([6, 8, 10])
        # 断言 inner 与 inner2 相等
        tm.assert_index_equal(inner, inner2)
        # 断言 inner 与预期结果 expected 相等
        tm.assert_index_equal(inner, expected)

        # 执行左连接操作，返回结果 left
        left = index.join(other, how="left")
        # 断言 left 与 index 转换为 object 类型后的索引相等
        tm.assert_index_equal(left, index.astype(object))

        # 执行左连接操作，返回结果 left2
        left2 = other.join(index, how="left")
        # 断言 left2 与 other 相等
        tm.assert_index_equal(left2, other)

        # 执行右连接操作，返回结果 right
        right = index.join(other, how="right")
        # 断言 right 与 other 相等
        tm.assert_index_equal(right, other)

        # 执行右连接操作，返回结果 right2
        right2 = other.join(index, how="right")
        # 断言 right2 与 index 转换为 object 类型后的索引相等
        tm.assert_index_equal(right2, index.astype(object))
    # 定义一个测试方法，用于测试非唯一索引的连接操作
    def test_join_non_unique(self):
        # 创建一个起始于0、终止于20、步长为2的RangeIndex对象
        index = RangeIndex(start=0, stop=20, step=2)
        # 创建另一个Index对象，包含重复元素
        other = Index([4, 4, 3, 3])

        # 执行索引对象index与other的连接操作，并返回连接结果、左索引、右索引
        res, lidx, ridx = index.join(other, return_indexers=True)

        # 期望的连接结果，一个包含合并后索引值的Index对象
        eres = Index([0, 2, 4, 4, 6, 8, 10, 12, 14, 16, 18])
        # 期望的左索引数组，对应于结果索引中每个元素在原始index中的位置
        elidx = np.array([0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.intp)
        # 期望的右索引数组，对应于结果索引中每个元素在other中的位置
        eridx = np.array([-1, -1, 0, 1, -1, -1, -1, -1, -1, -1, -1], dtype=np.intp)

        # 使用测试工具tm断言连接后的索引res与期望结果eres相等
        tm.assert_index_equal(res, eres)
        # 使用测试工具tm断言左索引lidx与期望左索引elidx的numpy数组相等
        tm.assert_numpy_array_equal(lidx, elidx)
        # 使用测试工具tm断言右索引ridx与期望右索引eridx的numpy数组相等
        tm.assert_numpy_array_equal(ridx, eridx)

    # 定义一个测试方法，用于测试索引对象与自身的连接操作
    def test_join_self(self, join_type):
        # 创建一个起始于0、终止于20、步长为2的RangeIndex对象
        index = RangeIndex(start=0, stop=20, step=2)
        # 执行索引对象index与自身的连接操作，根据指定的连接类型
        joined = index.join(index, how=join_type)
        # 使用断言验证索引对象index与joined对象是同一个对象
        assert index is joined
# 使用 pytest 提供的 parametrize 装饰器，为测试函数 test_join_preserves_rangeindex 添加多组参数化测试数据
@pytest.mark.parametrize(
    "left, right, expected, expected_lidx, expected_ridx, how",
    [
        [RangeIndex(2), RangeIndex(3), RangeIndex(2), None, [0, 1], "left"],
        [RangeIndex(2), RangeIndex(2), RangeIndex(2), None, None, "left"],
        [RangeIndex(2), RangeIndex(20, 22), RangeIndex(2), None, [-1, -1], "left"],
        [RangeIndex(2), RangeIndex(3), RangeIndex(3), [0, 1, -1], None, "right"],
        [RangeIndex(2), RangeIndex(2), RangeIndex(2), None, None, "right"],
        [
            RangeIndex(2),
            RangeIndex(20, 22),
            RangeIndex(20, 22),
            [-1, -1],
            None,
            "right",
        ],
        [RangeIndex(2), RangeIndex(3), RangeIndex(2), [0, 1], [0, 1], "inner"],
        [RangeIndex(2), RangeIndex(2), RangeIndex(2), None, None, "inner"],
        [RangeIndex(2), RangeIndex(1, 3), RangeIndex(1, 2), [1], [0], "inner"],
        [RangeIndex(2), RangeIndex(3), RangeIndex(3), [0, 1, -1], [0, 1, 2], "outer"],
        [RangeIndex(2), RangeIndex(2), RangeIndex(2), None, None, "outer"],
        [
            RangeIndex(2),
            RangeIndex(2, 4),
            RangeIndex(4),
            [0, 1, -1, -1],
            [-1, -1, 0, 1],
            "outer",
        ],
        [RangeIndex(2), RangeIndex(0), RangeIndex(2), None, [-1, -1], "left"],
        [RangeIndex(2), RangeIndex(0), RangeIndex(0), [], None, "right"],
        [RangeIndex(2), RangeIndex(0), RangeIndex(0), [], None, "inner"],
        [RangeIndex(2), RangeIndex(0), RangeIndex(2), None, [-1, -1], "outer"],
    ],
)
# 对 right_type 参数进行参数化，分别传入 RangeIndex 和自定义的匿名函数 lambda x: Index(list(x), dtype=x.dtype)
@pytest.mark.parametrize(
    "right_type", [RangeIndex, lambda x: Index(list(x), dtype=x.dtype)]
)
# 定义测试函数 test_join_preserves_rangeindex，接受多个参数用于参数化测试
def test_join_preserves_rangeindex(
    left, right, expected, expected_lidx, expected_ridx, how, right_type
):
    # 调用 left 对象的 join 方法，将 right 转换为 right_type 类型，指定连接方式 how，并请求返回索引信息
    result, lidx, ridx = left.join(right_type(right), how=how, return_indexers=True)
    # 使用 pandas.testing 模块的 assert_index_equal 函数，确保 result 与 expected 相等，要求精确匹配
    tm.assert_index_equal(result, expected, exact=True)

    # 判断 expected_lidx 是否为 None
    if expected_lidx is None:
        # 如果是，断言 lidx 也为 expected_lidx
        assert lidx is expected_lidx
    else:
        # 否则，将 expected_lidx 转换为 np.intp 类型的 numpy 数组
        exp_lidx = np.array(expected_lidx, dtype=np.intp)
        # 使用 pandas.testing 模块的 assert_numpy_array_equal 函数，确保 lidx 与 exp_lidx 相等
        tm.assert_numpy_array_equal(lidx, exp_lidx)

    # 判断 expected_ridx 是否为 None
    if expected_ridx is None:
        # 如果是，断言 ridx 也为 expected_ridx
        assert ridx is expected_ridx
    else:
        # 否则，将 expected_ridx 转换为 np.intp 类型的 numpy 数组
        exp_ridx = np.array(expected_ridx, dtype=np.intp)
        # 使用 pandas.testing 模块的 assert_numpy_array_equal 函数，确保 ridx 与 exp_ridx 相等
        tm.assert_numpy_array_equal(ridx, exp_ridx)
```