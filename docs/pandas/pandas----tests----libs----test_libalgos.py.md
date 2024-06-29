# `D:\src\scipysrc\pandas\pandas\tests\libs\test_libalgos.py`

```
# 从 datetime 模块导入 datetime 类
from datetime import datetime
# 从 itertools 模块导入 permutations 函数
from itertools import permutations
# 导入 numpy 库，并使用 np 别名
import numpy as np
# 从 pandas._libs 中导入 algos 模块，并使用 libalgos 别名
from pandas._libs import algos as libalgos
# 从 pandas._testing 中导入 tm 模块
import pandas._testing as tm


# 定义测试函数 test_ensure_platform_int
def test_ensure_platform_int():
    # 创建一个包含100个整数的 numpy 数组 arr，数据类型为 np.intp
    arr = np.arange(100, dtype=np.intp)

    # 调用 libalgos 中的 ensure_platform_int 函数处理数组 arr
    result = libalgos.ensure_platform_int(arr)
    # 断言结果与原始数组 arr 相同
    assert result is arr


# 定义测试函数 test_is_lexsorted
def test_is_lexsorted():
    # 创建一个包含不是词典序排列的 numpy 数组列表 failure
    failure = [
        np.array(
            ([3] * 32) + ([2] * 32) + ([1] * 32) + ([0] * 32),
            dtype="int64",
        ),
        np.array(
            list(range(31))[::-1] * 4,
            dtype="int64",
        ),
    ]

    # 断言 libalgos 中的 is_lexsorted 函数对 failure 返回 False
    assert not libalgos.is_lexsorted(failure)


# 定义测试函数 test_groupsort_indexer
def test_groupsort_indexer():
    # 使用随机整数生成器创建两个 numpy 数组 a 和 b
    a = np.random.default_rng(2).integers(0, 1000, 100).astype(np.intp)
    b = np.random.default_rng(2).integers(0, 1000, 100).astype(np.intp)

    # 调用 libalgos 中的 groupsort_indexer 函数对数组 a 进行排序，并取结果的第一个返回值
    result = libalgos.groupsort_indexer(a, 1000)[0]

    # 预期结果使用稳定排序的 np.argsort 对数组 a 进行排序，并将结果转换为 np.intp 类型
    expected = np.argsort(a, kind="mergesort")
    expected = expected.astype(np.intp)

    # 使用 pandas._testing 中的 assert_numpy_array_equal 函数比较 result 和 expected
    tm.assert_numpy_array_equal(result, expected)

    # 比较使用 np.lexsort 对元组 (b, a) 进行排序后的结果，将结果转换为 np.intp 类型
    key = a * 1000 + b
    result = libalgos.groupsort_indexer(key, 1000000)[0]
    expected = np.lexsort((b, a))
    expected = expected.astype(np.intp)

    # 使用 assert_numpy_array_equal 函数再次比较 result 和 expected
    tm.assert_numpy_array_equal(result, expected)


# 定义 TestPadBackfill 类
class TestPadBackfill:
    # 定义类内的测试方法 test_backfill
    def test_backfill(self):
        # 创建两个 numpy 数组 old 和 new，数据类型为 np.int64
        old = np.array([1, 5, 10], dtype=np.int64)
        new = np.array(list(range(12)), dtype=np.int64)

        # 调用 libalgos 中的 backfill["int64_t"] 函数处理 old 和 new
        filler = libalgos.backfill["int64_t"](old, new)

        # 预期结果是一个特定的 numpy 数组 expect_filler，数据类型为 np.intp
        expect_filler = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, -1], dtype=np.intp)
        # 使用 assert_numpy_array_equal 函数比较 filler 和 expect_filler
        tm.assert_numpy_array_equal(filler, expect_filler)

        # 处理一个特殊情况，old 数组长度为 2，new 数组长度为 5
        old = np.array([1, 4], dtype=np.int64)
        new = np.array(list(range(5, 10)), dtype=np.int64)
        filler = libalgos.backfill["int64_t"](old, new)

        # 预期结果是一个特定的 numpy 数组 expect_filler，数据类型为 np.intp
        expect_filler = np.array([-1, -1, -1, -1, -1], dtype=np.intp)
        # 使用 assert_numpy_array_equal 函数比较 filler 和 expect_filler
        tm.assert_numpy_array_equal(filler, expect_filler)

    # 定义类内的测试方法 test_pad
    def test_pad(self):
        # 创建两个 numpy 数组 old 和 new，数据类型为 np.int64
        old = np.array([1, 5, 10], dtype=np.int64)
        new = np.array(list(range(12)), dtype=np.int64)

        # 调用 libalgos 中的 pad["int64_t"] 函数处理 old 和 new
        filler = libalgos.pad["int64_t"](old, new)

        # 预期结果是一个特定的 numpy 数组 expect_filler，数据类型为 np.intp
        expect_filler = np.array([-1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2], dtype=np.intp)
        # 使用 assert_numpy_array_equal 函数比较 filler 和 expect_filler
        tm.assert_numpy_array_equal(filler, expect_filler)

        # 处理一个特殊情况，old 数组长度为 2，new 数组长度为 5
        old = np.array([5, 10], dtype=np.int64)
        new = np.arange(5, dtype=np.int64)
        filler = libalgos.pad["int64_t"](old, new)
        expect_filler = np.array([-1, -1, -1, -1, -1], dtype=np.intp)
        # 使用 assert_numpy_array_equal 函数比较 filler 和 expect_filler
        tm.assert_numpy_array_equal(filler, expect_filler)
    # 定义一个测试方法，用于测试 pad 和 backfill 算法在空对象数组上的行为
    def test_pad_backfill_object_segfault(self):
        # 创建一个空的 NumPy 对象数组
        old = np.array([], dtype="O")
        # 创建一个包含一个日期时间对象的 NumPy 对象数组
        new = np.array([datetime(2010, 12, 31)], dtype="O")

        # 测试 pad 算法对空数组 old 和非空数组 new 的行为
        result = libalgos.pad["object"](old, new)
        # 期望的结果是一个包含值 -1 的整数数组
        expected = np.array([-1], dtype=np.intp)
        # 断言两个 NumPy 数组是否相等
        tm.assert_numpy_array_equal(result, expected)

        # 测试 pad 算法对非空数组 new 和空数组 old 的行为
        result = libalgos.pad["object"](new, old)
        # 期望的结果是一个空的整数数组
        expected = np.array([], dtype=np.intp)
        # 断言两个 NumPy 数组是否相等
        tm.assert_numpy_array_equal(result, expected)

        # 测试 backfill 算法对空数组 old 和非空数组 new 的行为
        result = libalgos.backfill["object"](old, new)
        # 期望的结果是一个包含值 -1 的整数数组
        expected = np.array([-1], dtype=np.intp)
        # 断言两个 NumPy 数组是否相等
        tm.assert_numpy_array_equal(result, expected)

        # 测试 backfill 算法对非空数组 new 和空数组 old 的行为
        result = libalgos.backfill["object"](new, old)
        # 期望的结果是一个空的整数数组
        expected = np.array([], dtype=np.intp)
        # 断言两个 NumPy 数组是否相等
        tm.assert_numpy_array_equal(result, expected)
class TestInfinity:
    def test_infinity_sort(self):
        # GH#13445
        # numpy's argsort can be unhappy if something is less than
        # itself.  Instead, let's give our infinities a self-consistent
        # ordering, but outside the float extended real line.

        # 创建正无穷和负无穷对象
        Inf = libalgos.Infinity()
        NegInf = libalgos.NegInfinity()

        # 参考数列，包含负无穷、浮点数负无穷、负1e100、0、1e100、浮点数正无穷、正无穷
        ref_nums = [NegInf, float("-inf"), -1e100, 0, 1e100, float("inf"), Inf]

        # 断言正无穷大于等于参考数列中的每个元素
        assert all(Inf >= x for x in ref_nums)
        # 断言正无穷大于参考数列中的每个元素，或者该元素是正无穷本身
        assert all(Inf > x or x is Inf for x in ref_nums)
        # 断言正无穷大于等于正无穷，并且正无穷等于正无穷
        assert Inf >= Inf and Inf == Inf
        # 断言正无穷不小于也不大于正无穷
        assert not Inf < Inf and not Inf > Inf
        # 断言两个正无穷对象相等
        assert libalgos.Infinity() == libalgos.Infinity()
        # 断言两个正无穷对象不不等
        assert not libalgos.Infinity() != libalgos.Infinity()

        # 断言负无穷小于等于参考数列中的每个元素
        assert all(NegInf <= x for x in ref_nums)
        # 断言负无穷小于参考数列中的每个元素，或者该元素是负无穷本身
        assert all(NegInf < x or x is NegInf for x in ref_nums)
        # 断言负无穷小于等于负无穷，并且负无穷等于负无穷
        assert NegInf <= NegInf and NegInf == NegInf
        # 断言负无穷不小于也不大于负无穷
        assert not NegInf < NegInf and not NegInf > NegInf
        # 断言两个负无穷对象相等
        assert libalgos.NegInfinity() == libalgos.NegInfinity()
        # 断言两个负无穷对象不不等
        assert not libalgos.NegInfinity() != libalgos.NegInfinity()

        # 对参考数列的所有排列进行排序，确保排序后与参考数列相同
        for perm in permutations(ref_nums):
            assert sorted(perm) == ref_nums

        # smoke tests
        # 对包含32个正无穷对象的数组进行排序
        np.array([libalgos.Infinity()] * 32).argsort()
        # 对包含32个负无穷对象的数组进行排序
        np.array([libalgos.NegInfinity()] * 32).argsort()

    def test_infinity_against_nan(self):
        # 创建正无穷和负无穷对象
        Inf = libalgos.Infinity()
        NegInf = libalgos.NegInfinity()

        # 断言正无穷不大于、不大于等于、不小于、不小于等于、不等于 NaN
        assert not Inf > np.nan
        assert not Inf >= np.nan
        assert not Inf < np.nan
        assert not Inf <= np.nan
        assert not Inf == np.nan
        assert Inf != np.nan

        # 断言负无穷不大于、不大于等于、不小于、不小于等于、不等于 NaN
        assert not NegInf > np.nan
        assert not NegInf >= np.nan
        assert not NegInf < np.nan
        assert not NegInf <= np.nan
        assert not NegInf == np.nan
        assert NegInf != np.nan
```