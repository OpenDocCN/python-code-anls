# `D:\src\scipysrc\scipy\scipy\stats\tests\test_rank.py`

```
import numpy as np  # 导入 NumPy 库，用于科学计算
from numpy.testing import assert_equal, assert_array_equal  # 导入测试模块中的断言函数
import pytest  # 导入 pytest 模块，用于单元测试

from scipy.conftest import skip_xp_invalid_arg  # 从 scipy 中导入跳过特定测试参数的函数
from scipy.stats import rankdata, tiecorrect  # 从 scipy 的统计模块中导入排名数据和关于并列排名修正的函数
from scipy._lib._util import np_long  # 从 scipy 的内部工具模块中导入 numpy 的长整型类型

class TestTieCorrect:

    def test_empty(self):
        """An empty array requires no correction, should return 1.0."""
        ranks = np.array([], dtype=np.float64)  # 创建一个空的浮点数 NumPy 数组
        c = tiecorrect(ranks)  # 计算数组 ranks 的并列排名修正系数
        assert_equal(c, 1.0)  # 使用断言函数验证修正系数是否为 1.0

    def test_one(self):
        """A single element requires no correction, should return 1.0."""
        ranks = np.array([1.0], dtype=np.float64)  # 创建一个包含单个元素的浮点数 NumPy 数组
        c = tiecorrect(ranks)  # 计算数组 ranks 的并列排名修正系数
        assert_equal(c, 1.0)  # 使用断言函数验证修正系数是否为 1.0

    def test_no_correction(self):
        """Arrays with no ties require no correction."""
        ranks = np.arange(2.0)  # 创建一个包含连续两个元素的浮点数 NumPy 数组
        c = tiecorrect(ranks)  # 计算数组 ranks 的并列排名修正系数
        assert_equal(c, 1.0)  # 使用断言函数验证修正系数是否为 1.0
        ranks = np.arange(3.0)  # 创建一个包含连续三个元素的浮点数 NumPy 数组
        c = tiecorrect(ranks)  # 计算数组 ranks 的并列排名修正系数
        assert_equal(c, 1.0)  # 使用断言函数验证修正系数是否为 1.0

    def test_basic(self):
        """Check a few basic examples of the tie correction factor."""
        # One tie of two elements
        ranks = np.array([1.0, 2.5, 2.5])  # 创建一个包含一个并列排名的浮点数 NumPy 数组
        c = tiecorrect(ranks)  # 计算数组 ranks 的并列排名修正系数
        T = 2.0
        N = ranks.size
        expected = 1.0 - (T**3 - T) / (N**3 - N)  # 根据公式计算预期的修正系数
        assert_equal(c, expected)  # 使用断言函数验证修正系数是否与预期相等

        # One tie of two elements (same as above, but tie is not at the end)
        ranks = np.array([1.5, 1.5, 3.0])  # 创建另一个包含一个并列排名的浮点数 NumPy 数组
        c = tiecorrect(ranks)  # 计算数组 ranks 的并列排名修正系数
        T = 2.0
        N = ranks.size
        expected = 1.0 - (T**3 - T) / (N**3 - N)  # 根据公式计算预期的修正系数
        assert_equal(c, expected)  # 使用断言函数验证修正系数是否与预期相等

        # One tie of three elements
        ranks = np.array([1.0, 3.0, 3.0, 3.0])  # 创建一个包含一个并列排名的浮点数 NumPy 数组
        c = tiecorrect(ranks)  # 计算数组 ranks 的并列排名修正系数
        T = 3.0
        N = ranks.size
        expected = 1.0 - (T**3 - T) / (N**3 - N)  # 根据公式计算预期的修正系数
        assert_equal(c, expected)  # 使用断言函数验证修正系数是否与预期相等

        # Two ties, lengths 2 and 3.
        ranks = np.array([1.5, 1.5, 4.0, 4.0, 4.0])  # 创建一个包含两个并列排名的浮点数 NumPy 数组
        c = tiecorrect(ranks)  # 计算数组 ranks 的并列排名修正系数
        T1 = 2.0
        T2 = 3.0
        N = ranks.size
        expected = 1.0 - ((T1**3 - T1) + (T2**3 - T2)) / (N**3 - N)  # 根据公式计算预期的修正系数
        assert_equal(c, expected)  # 使用断言函数验证修正系数是否与预期相等

    def test_overflow(self):
        ntie, k = 2000, 5
        a = np.repeat(np.arange(k), ntie)  # 创建一个包含大量重复元素的 NumPy 数组
        n = a.size  # 获取数组 a 的大小
        out = tiecorrect(rankdata(a))  # 计算数组 a 的排名并且进行并列排名修正
        assert_equal(out, 1.0 - k * (ntie**3 - ntie) / float(n**3 - n))  # 使用断言函数验证修正系数是否与预期相等


class TestRankData:

    def test_empty(self):
        """stats.rankdata([]) should return an empty array."""
        a = np.array([], dtype=int)  # 创建一个空的整型 NumPy 数组
        r = rankdata(a)  # 对数组 a 进行排名操作
        assert_array_equal(r, np.array([], dtype=np.float64))  # 使用断言函数验证排名结果是否为空数组
        r = rankdata([])  # 对空列表进行排名操作
        assert_array_equal(r, np.array([], dtype=np.float64))  # 使用断言函数验证排名结果是否为空数组

    @pytest.mark.parametrize("shape", [(0, 1, 2)])  # 参数化测试，定义不同的形状
    @pytest.mark.parametrize("axis", [None, *range(3)])  # 参数化测试，定义不同的轴
    def test_empty_multidim(self, shape, axis):
        a = np.empty(shape, dtype=int)  # 创建一个指定形状的空的整型 NumPy 数组
        r = rankdata(a, axis=axis)  # 对多维数组 a 进行排名操作，指定轴参数
        expected_shape = (0,) if axis is None else shape  # 根据轴参数计算预期的形状
        assert_equal(r.shape, expected_shape)  # 使用断言函数验证排名结果的形状是否与预期相等
        assert_equal(r.dtype, np.float64)  # 使用断言函数验证排名结果的数据类型是否为浮点数
    def test_one(self):
        """Check stats.rankdata with an array of length 1."""
        data = [100]  # 定义一个包含一个整数的列表
        a = np.array(data, dtype=int)  # 将列表转换为NumPy整数数组
        r = rankdata(a)  # 调用rankdata函数，对数组进行排名
        assert_array_equal(r, np.array([1.0], dtype=np.float64))  # 断言排名结果与预期一致
        r = rankdata(data)  # 再次调用rankdata函数，对列表进行排名
        assert_array_equal(r, np.array([1.0], dtype=np.float64))  # 断言排名结果与预期一致

    def test_basic(self):
        """Basic tests of stats.rankdata."""
        data = [100, 10, 50]  # 定义一个包含三个整数的列表
        expected = np.array([3.0, 1.0, 2.0], dtype=np.float64)  # 定义预期的排名结果数组
        a = np.array(data, dtype=int)  # 将列表转换为NumPy整数数组
        r = rankdata(a)  # 调用rankdata函数，对数组进行排名
        assert_array_equal(r, expected)  # 断言排名结果与预期一致
        r = rankdata(data)  # 再次调用rankdata函数，对列表进行排名
        assert_array_equal(r, expected)  # 断言排名结果与预期一致

        data = [40, 10, 30, 10, 50]  # 定义一个包含五个整数的列表
        expected = np.array([4.0, 1.5, 3.0, 1.5, 5.0], dtype=np.float64)  # 定义预期的排名结果数组
        a = np.array(data, dtype=int)  # 将列表转换为NumPy整数数组
        r = rankdata(a)  # 调用rankdata函数，对数组进行排名
        assert_array_equal(r, expected)  # 断言排名结果与预期一致
        r = rankdata(data)  # 再次调用rankdata函数，对列表进行排名
        assert_array_equal(r, expected)  # 断言排名结果与预期一致

        data = [20, 20, 20, 10, 10, 10]  # 定义一个包含六个整数的列表
        expected = np.array([5.0, 5.0, 5.0, 2.0, 2.0, 2.0], dtype=np.float64)  # 定义预期的排名结果数组
        a = np.array(data, dtype=int)  # 将列表转换为NumPy整数数组
        r = rankdata(a)  # 调用rankdata函数，对数组进行排名
        assert_array_equal(r, expected)  # 断言排名结果与预期一致
        r = rankdata(data)  # 再次调用rankdata函数，对列表进行排名
        assert_array_equal(r, expected)  # 断言排名结果与预期一致
        # The docstring states explicitly that the argument is flattened.
        a2d = a.reshape(2, 3)  # 将一维数组a重塑为二维数组a2d（2行3列）
        r = rankdata(a2d)  # 调用rankdata函数，对二维数组进行排名
        assert_array_equal(r, expected)  # 断言排名结果与预期一致

    @skip_xp_invalid_arg
    def test_rankdata_object_string(self):
        """Tests of stats.rankdata with object and string inputs."""
        
        def min_rank(a):
            return [1 + sum(i < j for i in a) for j in a]  # 计算最小排名

        def max_rank(a):
            return [sum(i <= j for i in a) for j in a]  # 计算最大排名

        def ordinal_rank(a):
            return min_rank([(x, i) for i, x in enumerate(a)])  # 计算序数排名

        def average_rank(a):
            return [(i + j) / 2.0 for i, j in zip(min_rank(a), max_rank(a))]  # 计算平均排名

        def dense_rank(a):
            b = np.unique(a)  # 获取唯一值数组b
            return [1 + sum(i < j for i in b) for j in a]  # 计算密集排名

        rankf = dict(min=min_rank, max=max_rank, ordinal=ordinal_rank,
                     average=average_rank, dense=dense_rank)  # 创建排名函数字典rankf

        def check_ranks(a):
            for method in 'min', 'max', 'dense', 'ordinal', 'average':
                out = rankdata(a, method=method)  # 调用rankdata函数，使用指定的排名方法
                assert_array_equal(out, rankf[method](a))  # 断言排名结果与预期一致

        val = ['foo', 'bar', 'qux', 'xyz', 'abc', 'efg', 'ace', 'qwe', 'qaz']  # 定义字符串列表val
        check_ranks(np.random.choice(val, 200))  # 调用check_ranks函数，传入随机选择的字符串数组

        check_ranks(np.random.choice(val, 200).astype('object'))  # 调用check_ranks函数，传入随机选择的字符串数组，类型为对象

        val = np.array([0, 1, 2, 2.718, 3, 3.141], dtype='object')  # 定义包含不同类型数据的对象数组val
        check_ranks(np.random.choice(val, 200).astype('object'))  # 调用check_ranks函数，传入随机选择的对象数组
    # 定义测试函数，用于测试处理大整数时的排名功能
    def test_large_int(self):
        # 创建包含两个超大整数的 NumPy 数组，数据类型为无符号 64 位整数
        data = np.array([2**60, 2**60+1], dtype=np.uint64)
        # 对数据进行排名
        r = rankdata(data)
        # 断言排名结果与预期的数组相等
        assert_array_equal(r, [1.0, 2.0])

        # 创建包含两个超大整数的 NumPy 数组，数据类型为有符号 64 位整数
        data = np.array([2**60, 2**60+1], dtype=np.int64)
        # 对数据进行排名
        r = rankdata(data)
        # 断言排名结果与预期的数组相等
        assert_array_equal(r, [1.0, 2.0])

        # 创建包含一个正超大整数和一个负超大整数的 NumPy 数组，数据类型为有符号 64 位整数
        data = np.array([2**60, -2**60+1], dtype=np.int64)
        # 对数据进行排名
        r = rankdata(data)
        # 断言排名结果与预期的数组相等
        assert_array_equal(r, [2.0, 1.0])

    # 定义测试函数，用于测试处理大规模相同值的情况下的排名功能
    def test_big_tie(self):
        # 遍历不同规模的数组
        for n in [10000, 100000, 1000000]:
            # 创建长度为 n 的全为 1 的整数类型数组
            data = np.ones(n, dtype=int)
            # 对数据进行排名
            r = rankdata(data)
            # 计算预期的排名结果
            expected_rank = 0.5 * (n + 1)
            # 断言排名结果与预期的数组相等
            assert_array_equal(r, expected_rank * data,
                               "test failed with n=%d" % n)

    # 定义测试函数，用于测试在不同轴向上的排名功能
    def test_axis(self):
        # 创建一个包含子列表的二维数组
        data = [[0, 2, 1],
                [4, 2, 2]]
        # 预期的在轴向 0 上的排名结果
        expected0 = [[1., 1.5, 1.],
                     [2., 1.5, 2.]]
        # 对数组在轴向 0 上进行排名
        r0 = rankdata(data, axis=0)
        # 断言排名结果与预期的数组相等
        assert_array_equal(r0, expected0)
        # 预期的在轴向 1 上的排名结果
        expected1 = [[1., 3., 2.],
                     [3., 1.5, 1.5]]
        # 对数组在轴向 1 上进行排名
        r1 = rankdata(data, axis=1)
        # 断言排名结果与预期的数组相等
        assert_array_equal(r1, expected1)

    # 定义一组排名方法和数据类型
    methods = ["average", "min", "max", "dense", "ordinal"]
    dtypes = [np.float64] + [np_long]*4

    # 使用 pytest 的参数化装饰器来定义测试函数，用于测试在长度为 0 的轴向上的排名功能
    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("method, dtype", zip(methods, dtypes))
    def test_size_0_axis(self, axis, method, dtype):
        # 定义一个形状为 (3, 0) 的全零数组
        shape = (3, 0)
        data = np.zeros(shape)
        # 对数组在指定轴向上进行排名，使用指定的方法和数据类型
        r = rankdata(data, method=method, axis=axis)
        # 断言排名结果的形状与预期的形状相等
        assert_equal(r.shape, shape)
        # 断言排名结果的数据类型与预期的数据类型相等
        assert_equal(r.dtype, dtype)

    # 使用 pytest 的参数化装饰器来定义测试函数，测试在三维数组中处理 NaN 时的排名功能
    @pytest.mark.parametrize('axis', range(3))
    @pytest.mark.parametrize('method', methods)
    def test_nan_policy_omit_3d(self, axis, method):
        # 定义一个形状为 (20, 21, 22) 的随机数组
        shape = (20, 21, 22)
        rng = np.random.RandomState(23983242)

        # 生成随机数组，并随机将部分元素设为 NaN 或 -inf
        a = rng.random(size=shape)
        i = rng.random(size=shape) < 0.4
        j = rng.random(size=shape) < 0.1
        k = rng.random(size=shape) < 0.1
        a[i] = np.nan
        a[j] = -np.inf
        a[k] - np.inf

        # 定义一个函数，用于处理带有 NaN 的一维数组的排名
        def rank_1d_omit(a, method):
            out = np.zeros_like(a)
            i = np.isnan(a)
            a_compressed = a[~i]
            res = rankdata(a_compressed, method)
            out[~i] = res
            out[i] = np.nan
            return out

        # 定义一个函数，用于在指定轴向上处理带有 NaN 的数组的排名
        def rank_omit(a, method, axis):
            return np.apply_along_axis(lambda a: rank_1d_omit(a, method),
                                       axis, a)

        # 对数组在指定轴向上进行排名，使用指定的方法和 NaN 策略
        res = rankdata(a, method, axis=axis, nan_policy='omit')
        res0 = rank_omit(a, method, axis=axis)

        # 断言排名结果与预期的排名结果相等
        assert_array_equal(res, res0)
    def test_nan_policy_2d_axis_none(self):
        # 定义一个测试函数，测试在二维数组中使用 axis=None 的情况
        data = [[0, np.nan, 3],
                [4, 2, np.nan],
                [1, 2, 2]]
        # 断言函数 rankdata 在 nan_policy='omit' 下的输出结果与预期数组相等
        assert_array_equal(rankdata(data, axis=None, nan_policy='omit'),
                           [1., np.nan, 6., 7., 4., np.nan, 2., 4., 4.])
        # 断言函数 rankdata 在 nan_policy='propagate' 下的输出结果与预期数组相等
        assert_array_equal(rankdata(data, axis=None, nan_policy='propagate'),
                           [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                            np.nan, np.nan, np.nan])

    def test_nan_policy_raise(self):
        # 定义一个测试函数，测试在一维数组中使用 nan_policy='raise' 的情况
        data = [0, 2, 3, -2, np.nan, np.nan]
        # 使用 pytest 检查调用 rankdata 函数时是否会引发 ValueError，并匹配指定错误信息
        with pytest.raises(ValueError, match="The input contains nan"):
            rankdata(data, nan_policy='raise')

        # 重新定义数据进行二维数组测试
        data = [[0, np.nan, 3],
                [4, 2, np.nan],
                [np.nan, 2, 2]]

        # 使用 pytest 检查调用 rankdata 函数时是否会引发 ValueError，并匹配指定错误信息（按列操作）
        with pytest.raises(ValueError, match="The input contains nan"):
            rankdata(data, axis=0, nan_policy="raise")

        # 使用 pytest 检查调用 rankdata 函数时是否会引发 ValueError，并匹配指定错误信息（按行操作）
        with pytest.raises(ValueError, match="The input contains nan"):
            rankdata(data, axis=1, nan_policy="raise")

    def test_nan_policy_propagate(self):
        # 定义一个测试函数，测试在一维数组中使用 nan_policy='propagate' 的情况
        data = [0, 2, 3, -2, np.nan, np.nan]
        # 断言函数 rankdata 在 nan_policy='propagate' 下的输出结果与预期数组相等
        assert_array_equal(rankdata(data, nan_policy='propagate'),
                           [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

        # 重新定义数据进行二维数组测试
        data = [[0, np.nan, 3],
                [4, 2, np.nan],
                [1, 2, 2]]
        # 断言函数 rankdata 在 nan_policy='propagate' 下的输出结果与预期数组相等（按列操作）
        assert_array_equal(rankdata(data, axis=0, nan_policy='propagate'),
                           [[1, np.nan, np.nan],
                            [3, np.nan, np.nan],
                            [2, np.nan, np.nan]])
        # 断言函数 rankdata 在 nan_policy='propagate' 下的输出结果与预期数组相等（按行操作）
        assert_array_equal(rankdata(data, axis=1, nan_policy='propagate'),
                           [[np.nan, np.nan, np.nan],
                            [np.nan, np.nan, np.nan],
                            [1, 2.5, 2.5]])
# 定义测试用例的元组，每个元组包含输入值列表(values)，排名方法(method)，和期望输出(expected)
_cases = (
    # 空列表的平均排名测试用例
    ([], 'average', []),
    # 空列表的最小排名测试用例
    ([], 'min', []),
    # 空列表的最大排名测试用例
    ([], 'max', []),
    # 空列表的密集排名测试用例
    ([], 'dense', []),
    # 空列表的序数排名测试用例
    ([], 'ordinal', []),
    #
    # 单元素列表 [100] 的平均排名测试用例
    ([100], 'average', [1.0]),
    # 单元素列表 [100] 的最小排名测试用例
    ([100], 'min', [1.0]),
    # 单元素列表 [100] 的最大排名测试用例
    ([100], 'max', [1.0]),
    # 单元素列表 [100] 的密集排名测试用例
    ([100], 'dense', [1.0]),
    # 单元素列表 [100] 的序数排名测试用例
    ([100], 'ordinal', [1.0]),
    #
    # 三元素列表 [100, 100, 100] 的平均排名测试用例
    ([100, 100, 100], 'average', [2.0, 2.0, 2.0]),
    # 三元素列表 [100, 100, 100] 的最小排名测试用例
    ([100, 100, 100], 'min', [1.0, 1.0, 1.0]),
    # 三元素列表 [100, 100, 100] 的最大排名测试用例
    ([100, 100, 100], 'max', [3.0, 3.0, 3.0]),
    # 三元素列表 [100, 100, 100] 的密集排名测试用例
    ([100, 100, 100], 'dense', [1.0, 1.0, 1.0]),
    # 三元素列表 [100, 100, 100] 的序数排名测试用例
    ([100, 100, 100], 'ordinal', [1.0, 2.0, 3.0]),
    #
    # 三元素列表 [100, 300, 200] 的平均排名测试用例
    ([100, 300, 200], 'average', [1.0, 3.0, 2.0]),
    # 三元素列表 [100, 300, 200] 的最小排名测试用例
    ([100, 300, 200], 'min', [1.0, 3.0, 2.0]),
    # 三元素列表 [100, 300, 200] 的最大排名测试用例
    ([100, 300, 200], 'max', [1.0, 3.0, 2.0]),
    # 三元素列表 [100, 300, 200] 的密集排名测试用例
    ([100, 300, 200], 'dense', [1.0, 3.0, 2.0]),
    # 三元素列表 [100, 300, 200] 的序数排名测试用例
    ([100, 300, 200], 'ordinal', [1.0, 3.0, 2.0]),
    #
    # 四元素列表 [100, 200, 300, 200] 的平均排名测试用例
    ([100, 200, 300, 200], 'average', [1.0, 2.5, 4.0, 2.5]),
    # 四元素列表 [100, 200, 300, 200] 的最小排名测试用例
    ([100, 200, 300, 200], 'min', [1.0, 2.0, 4.0, 2.0]),
    # 四元素列表 [100, 200, 300, 200] 的最大排名测试用例
    ([100, 200, 300, 200], 'max', [1.0, 3.0, 4.0, 3.0]),
    # 四元素列表 [100, 200, 300, 200] 的密集排名测试用例
    ([100, 200, 300, 200], 'dense', [1.0, 2.0, 3.0, 2.0]),
    # 四元素列表 [100, 200, 300, 200] 的序数排名测试用例
    ([100, 200, 300, 200], 'ordinal', [1.0, 2.0, 4.0, 3.0]),
    #
    # 五元素列表 [100, 200, 300, 200, 100] 的平均排名测试用例
    ([100, 200, 300, 200, 100], 'average', [1.5, 3.5, 5.0, 3.5, 1.5]),
    # 五元素列表 [100, 200, 300, 200, 100] 的最小排名测试用例
    ([100, 200, 300, 200, 100], 'min', [1.0, 3.0, 5.0, 3.0, 1.0]),
    # 五元素列表 [100, 200, 300, 200, 100] 的最大排名测试用例
    ([100, 200, 300, 200, 100], 'max', [2.0, 4.0, 5.0, 4.0, 2.0]),
    # 五元素列表 [100, 200, 300, 200, 100] 的密集排名测试用例
    ([100, 200, 300, 200, 100], 'dense', [1.0, 2.0, 3.0, 2.0, 1.0]),
    # 五元素列表 [100, 200, 300, 200, 100] 的序数排名测试用例
    ([100, 200, 300, 200, 100], 'ordinal', [1.0, 3.0, 5.0, 4.0, 2.0]),
    #
    # 30 个相同元素 [10] 的序数排名测试用例，期望输出为数组 [1.0, 2.0, ..., 30.0]
    ([10] * 30, 'ordinal', np.arange(1.0, 31.0)),
)


def test_cases():
    # 遍历每个测试用例，依次执行排名函数并断言结果是否符合期望
    for values, method, expected in _cases:
        r = rankdata(values, method=method)
        assert_array_equal(r, expected)
```