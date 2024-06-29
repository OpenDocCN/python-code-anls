# `D:\src\scipysrc\pandas\pandas\tests\libs\test_join.py`

```
# 导入 NumPy 库，简写为 np
import numpy as np
# 导入 pytest 测试框架
import pytest

# 导入 pandas 库中的 join 模块
from pandas._libs import join as libjoin
# 导入 pandas 库中 join 模块中的特定函数
from pandas._libs.join import (
    inner_join,
    left_outer_join,
)

# 导入 pandas 测试工具模块
import pandas._testing as tm

# 定义测试类 TestIndexer
class TestIndexer:
    # 使用 pytest 的参数化装饰器，传递参数 dtype 进行测试
    @pytest.mark.parametrize(
        "dtype", ["int32", "int64", "float32", "float64", "object"]
    )
    # 测试函数，测试外连接索引器
    def test_outer_join_indexer(self, dtype):
        # 获取外连接索引器函数
        indexer = libjoin.outer_join_indexer

        # 创建不同类型的 NumPy 数组作为测试数据
        left = np.arange(3, dtype=dtype)
        right = np.arange(2, 5, dtype=dtype)
        empty = np.array([], dtype=dtype)

        # 调用索引器函数进行计算，获取结果和索引
        result, lindexer, rindexer = indexer(left, right)

        # 断言结果的类型为 NumPy 数组
        assert isinstance(result, np.ndarray)
        assert isinstance(lindexer, np.ndarray)
        assert isinstance(rindexer, np.ndarray)

        # 使用测试工具函数检查结果是否与预期一致
        tm.assert_numpy_array_equal(result, np.arange(5, dtype=dtype))
        exp = np.array([0, 1, 2, -1, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(lindexer, exp)
        exp = np.array([-1, -1, 0, 1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(rindexer, exp)

        # 使用空数组进行测试
        result, lindexer, rindexer = indexer(empty, right)
        tm.assert_numpy_array_equal(result, right)
        exp = np.array([-1, -1, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(lindexer, exp)
        exp = np.array([0, 1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(rindexer, exp)

        # 再次使用空数组进行测试
        result, lindexer, rindexer = indexer(left, empty)
        tm.assert_numpy_array_equal(result, left)
        exp = np.array([0, 1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(lindexer, exp)
        exp = np.array([-1, -1, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(rindexer, exp)

    # 测试 Cython 实现的左外连接函数
    def test_cython_left_outer_join(self):
        # 创建示例数据数组
        left = np.array([0, 1, 2, 1, 2, 0, 0, 1, 2, 3, 3], dtype=np.intp)
        right = np.array([1, 1, 0, 4, 2, 2, 1], dtype=np.intp)
        max_group = 5

        # 调用 Cython 实现的左外连接函数
        ls, rs = left_outer_join(left, right, max_group)

        # 使用 mergesort 对左右数组进行排序
        exp_ls = left.argsort(kind="mergesort")
        exp_rs = right.argsort(kind="mergesort")

        # 预期的左右索引数组
        exp_li = np.array([0, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8, 8, 9, 10])
        exp_ri = np.array(
            [0, 0, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 4, 5, 4, 5, -1, -1]
        )

        # 根据预期的索引获取排序后的结果数组
        exp_ls = exp_ls.take(exp_li)
        exp_ls[exp_li == -1] = -1
        exp_rs = exp_rs.take(exp_ri)
        exp_rs[exp_ri == -1] = -1

        # 使用测试工具函数检查结果是否与预期一致
        tm.assert_numpy_array_equal(ls, exp_ls, check_dtype=False)
        tm.assert_numpy_array_equal(rs, exp_rs, check_dtype=False)
    # 定义一个测试函数，用于测试左外连接函数的功能
    def test_cython_right_outer_join(self):
        # 创建一个 NumPy 数组 left，表示左表的键
        left = np.array([0, 1, 2, 1, 2, 0, 0, 1, 2, 3,
# 定义一个测试函数，用于测试 left_join_indexer_unique 函数的行为
def test_left_join_indexer_unique(writable):
    # 创建一个包含整数的 NumPy 数组 a
    a = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    # 创建另一个包含整数的 NumPy 数组 b
    b = np.array([2, 2, 3, 4, 4], dtype=np.int64)
    # 设置数组 a 和 b 可写，GH#37312, GH#37264 是相关的 GitHub 问题号
    a.setflags(write=writable)
    b.setflags(write=writable)

    # 调用 libjoin 左连接索引器唯一值函数，将 b 作为主表，a 作为次表
    result = libjoin.left_join_indexer_unique(b, a)
    # 创建预期结果数组 expected，包含指定的整数值
    expected = np.array([1, 1, 2, 3, 3], dtype=np.intp)
    # 使用测试框架中的函数验证 result 是否等于 expected
    tm.assert_numpy_array_equal(result, expected)


# 定义一个测试函数，用于测试 left_outer_join 函数的特定 bug
def test_left_outer_join_bug():
    # 创建一个包含整数的 NumPy 数组 left
    left = np.array(
        [
            0,
            1,
            # 中间省略了大量整数值
            2,
        ],
        dtype=np.intp,
    )

    # 创建一个包含整数的 NumPy 数组 right
    right = np.array([3, 1], dtype=np.intp)
    # 设置最大分组数为 4
    max_groups = 4

    # 调用 libjoin 左外连接函数，获取左右索引 lidx 和 ridx
    lidx, ridx = libjoin.left_outer_join(left, right, max_groups, sort=False)

    # 创建预期的左索引数组 exp_lidx，包含从 0 到 left 数组长度的整数
    exp_lidx = np.arange(len(left), dtype=np.intp)
    # 创建预期的右索引数组 exp_ridx，初始化为 -1 的整数数组，与 left 数组长度相同
    exp_ridx = -np.ones(len(left), dtype=np.intp)

    # 根据 left 数组中的特定值设置 exp_ridx 中的对应索引值
    exp_ridx[left == 1] = 1
    exp_ridx[left == 3] = 0

    # 使用测试框架中的函数验证 lidx 和 exp_lidx 是否相等
    tm.assert_numpy_array_equal(lidx, exp_lidx)
    # 使用测试框架中的函数验证 ridx 和 exp_ridx 是否相等
    tm.assert_numpy_array_equal(ridx, exp_ridx)


# 定义一个测试函数，用于测试 inner_join_indexer 函数的行为
def test_inner_join_indexer():
    # 创建一个包含整数的 NumPy 数组 a
    a = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    # 创建一个包含整数的 NumPy 数组 b
    b = np.array([0, 3, 5, 7, 9], dtype=np.int64)

    # 调用 libjoin 内连接索引器函数，获取索引 index，以及连接后的 ares 和 bres
    index, ares, bres = libjoin.inner_join_indexer(a, b)

    # 创建预期的索引数组 index_exp，包含指定的整数值
    index_exp = np.array([3, 5], dtype=np.int64)
    # 使用测试框架中的函数验证 index 和 index_exp 是否接近相等
    tm.assert_almost_equal(index, index_exp)

    # 创建预期的数组 aexp 和 bexp，包含指定的整数值
    aexp = np.array([2, 4], dtype=np.intp)
    bexp = np.array([1, 2], dtype=np.intp)
    # 使用测试框架中的函数验证 ares 和 aexp 是否接近相等
    tm.assert_almost_equal(ares, aexp)
    # 使用测试框架中的函数验证 bres 和 bexp 是否接近相等
    tm.assert_almost_equal(bres, bexp)

    # 创建一个包含单个整数值的 NumPy 数组 a
    a = np.array([5], dtype=np.int64)
    # 创建一个包含单个整数值的 NumPy 数组 b
    b = np.array([5], dtype=np.int64)

    # 再次调用 libjoin 内连接索引器函数，获取索引 index，以及连接后的 ares 和 bres
    index, ares, bres = libjoin.inner_join_indexer(a, b)
    # 使用测试框架中的函数来断言两个 NumPy 数组是否相等，一个是索引数组 index，期望值为 [5]，数据类型为 np.int64
    tm.assert_numpy_array_equal(index, np.array([5], dtype=np.int64))
    # 使用测试框架中的函数来断言两个 NumPy 数组是否相等，一个是数组 ares，期望值为 [0]，数据类型为 np.intp
    tm.assert_numpy_array_equal(ares, np.array([0], dtype=np.intp))
    # 使用测试框架中的函数来断言两个 NumPy 数组是否相等，一个是数组 bres，期望值为 [0]，数据类型为 np.intp
    tm.assert_numpy_array_equal(bres, np.array([0], dtype=np.intp))
# 测试外连接索引器函数
def test_outer_join_indexer():
    # 创建两个NumPy数组a和b
    a = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    b = np.array([0, 3, 5, 7, 9], dtype=np.int64)

    # 调用外连接索引器函数，返回索引、a的结果和b的结果
    index, ares, bres = libjoin.outer_join_indexer(a, b)

    # 期望的索引结果
    index_exp = np.array([0, 1, 2, 3, 4, 5, 7, 9], dtype=np.int64)
    # 断言索引结果接近期望结果
    tm.assert_almost_equal(index, index_exp)

    # 期望的a的结果和b的结果
    aexp = np.array([-1, 0, 1, 2, 3, 4, -1, -1], dtype=np.intp)
    bexp = np.array([0, -1, -1, 1, -1, 2, 3, 4], dtype=np.intp)
    # 断言a的结果和b的结果接近期望结果
    tm.assert_almost_equal(ares, aexp)
    tm.assert_almost_equal(bres, bexp)

    # 修改a和b的值
    a = np.array([5], dtype=np.int64)
    b = np.array([5], dtype=np.int64)

    # 再次调用外连接索引器函数，返回索引、a的结果和b的结果
    index, ares, bres = libjoin.outer_join_indexer(a, b)
    # 断言索引结果等于期望的结果数组
    tm.assert_numpy_array_equal(index, np.array([5], dtype=np.int64))
    tm.assert_numpy_array_equal(ares, np.array([0], dtype=np.intp))
    tm.assert_numpy_array_equal(bres, np.array([0], dtype=np.intp))


# 测试左连接索引器函数
def test_left_join_indexer():
    # 创建两个NumPy数组a和b
    a = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    b = np.array([0, 3, 5, 7, 9], dtype=np.int64)

    # 调用左连接索引器函数，返回索引、a的结果和b的结果
    index, ares, bres = libjoin.left_join_indexer(a, b)

    # 断言索引结果等于a
    tm.assert_almost_equal(index, a)

    # 期望的a的结果和b的结果
    aexp = np.array([0, 1, 2, 3, 4], dtype=np.intp)
    bexp = np.array([-1, -1, 1, -1, 2], dtype=np.intp)
    # 断言a的结果和b的结果接近期望结果
    tm.assert_almost_equal(ares, aexp)
    tm.assert_almost_equal(bres, bexp)

    # 修改a和b的值
    a = np.array([5], dtype=np.int64)
    b = np.array([5], dtype=np.int64)

    # 再次调用左连接索引器函数，返回索引、a的结果和b的结果
    index, ares, bres = libjoin.left_join_indexer(a, b)
    # 断言索引结果等于期望的结果数组
    tm.assert_numpy_array_equal(index, np.array([5], dtype=np.int64))
    tm.assert_numpy_array_equal(ares, np.array([0], dtype=np.intp))
    tm.assert_numpy_array_equal(bres, np.array([0], dtype=np.intp))


# 测试左连接索引器函数2
def test_left_join_indexer2():
    # 创建两个NumPy数组idx和idx2
    idx = np.array([1, 1, 2, 5], dtype=np.int64)
    idx2 = np.array([1, 2, 5, 7, 9], dtype=np.int64)

    # 调用左连接索引器函数，返回结果、左侧索引和右侧索引
    res, lidx, ridx = libjoin.left_join_indexer(idx2, idx)

    # 期望的结果数组
    exp_res = np.array([1, 1, 2, 5, 7, 9], dtype=np.int64)
    # 断言结果数组接近期望结果
    tm.assert_almost_equal(res, exp_res)

    # 期望的左侧索引和右侧索引
    exp_lidx = np.array([0, 0, 1, 2, 3, 4], dtype=np.intp)
    exp_ridx = np.array([0, 1, 2, 3, -1, -1], dtype=np.intp)
    # 断言左侧索引和右侧索引接近期望结果
    tm.assert_almost_equal(lidx, exp_lidx)
    tm.assert_almost_equal(ridx, exp_ridx)


# 测试外连接索引器函数2
def test_outer_join_indexer2():
    # 创建两个NumPy数组idx和idx2
    idx = np.array([1, 1, 2, 5], dtype=np.int64)
    idx2 = np.array([1, 2, 5, 7, 9], dtype=np.int64)

    # 调用外连接索引器函数，返回结果、左侧索引和右侧索引
    res, lidx, ridx = libjoin.outer_join_indexer(idx2, idx)

    # 期望的结果数组
    exp_res = np.array([1, 1, 2, 5, 7, 9], dtype=np.int64)
    # 断言结果数组接近期望结果
    tm.assert_almost_equal(res, exp_res)

    # 期望的左侧索引和右侧索引
    exp_lidx = np.array([0, 0, 1, 2, 3, 4], dtype=np.intp)
    exp_ridx = np.array([0, 1, 2, 3, -1, -1], dtype=np.intp)
    # 断言左侧索引和右侧索引接近期望结果
    tm.assert_almost_equal(lidx, exp_lidx)
    tm.assert_almost_equal(ridx, exp_ridx)


# 测试内连接索引器函数2
def test_inner_join_indexer2():
    # 创建两个NumPy数组idx和idx2
    idx = np.array([1, 1, 2, 5], dtype=np.int64)
    idx2 = np.array([1, 2, 5, 7, 9], dtype=np.int64)

    # 调用内连接索引器函数，返回结果、左侧索引和右侧索引
    res, lidx, ridx = libjoin.inner_join_indexer(idx2, idx)

    # 期望的结果数组
    exp_res = np.array([1, 1, 2, 5], dtype=np.int64)
    # 断言结果数组接近期望结果
    tm.assert_almost_equal(res, exp_res)

    # 期望的左侧索引
    exp_lidx = np.array([0, 0, 1, 2], dtype=np.intp)
    # 断言左侧索引接近期望结果
    tm.assert_almost_equal(lidx, exp_lidx)
    # 使用测试工具tm来比较lidx和exp_lidx是否几乎相等
    tm.assert_almost_equal(lidx, exp_lidx)
    
    # 创建一个预期的NumPy数组exp_ridx，包含整数0到3，并指定数据类型为64位整数(np.intp)
    exp_ridx = np.array([0, 1, 2, 3], dtype=np.intp)
    # 使用测试工具tm来比较ridx和exp_ridx是否几乎相等
    tm.assert_almost_equal(ridx, exp_ridx)
```