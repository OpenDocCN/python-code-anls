# `.\numpy\numpy\random\tests\test_randomstate_regression.py`

```
# 导入系统模块 sys
import sys

# 导入 pytest 测试框架
import pytest

# 从 numpy.testing 模块中导入断言函数和异常函数
from numpy.testing import (
    assert_, assert_array_equal, assert_raises,
    )

# 导入 numpy 模块，并使用 np 别名
import numpy as np

# 从 numpy 模块中导入 random 子模块
from numpy import random

# 定义测试类 TestRegression
class TestRegression:

    # 定义测试方法 test_VonMises_range
    def test_VonMises_range(self):
        # 确保生成的随机变量在 [-pi, pi] 范围内
        # 这是针对票号 #986 的回归测试
        for mu in np.linspace(-7., 7., 5):
            r = random.vonmises(mu, 1, 50)
            assert_(np.all(r > -np.pi) and np.all(r <= np.pi))

    # 定义测试方法 test_hypergeometric_range
    def test_hypergeometric_range(self):
        # 针对票号 #921 进行测试
        assert_(np.all(random.hypergeometric(3, 18, 11, size=10) < 4))
        assert_(np.all(random.hypergeometric(18, 3, 11, size=10) > 0))

        # 针对票号 #5623 进行测试
        args = [
            (2**20 - 2, 2**20 - 2, 2**20 - 2),  # 检查 32 位系统
        ]
        is_64bits = sys.maxsize > 2**32
        if is_64bits and sys.platform != 'win32':
            # 检查 64 位系统
            args.append((2**40 - 2, 2**40 - 2, 2**40 - 2))
        for arg in args:
            assert_(random.hypergeometric(*arg) > 0)

    # 定义测试方法 test_logseries_convergence
    def test_logseries_convergence(self):
        # 针对票号 #923 进行测试
        N = 1000
        random.seed(0)
        rvsn = random.logseries(0.8, size=N)
        # 这两个频率计数应该接近理论值，在这么大的样本中
        # 对于大 N，理论值约为 0.49706795
        freq = np.sum(rvsn == 1) / N
        msg = f'Frequency was {freq:f}, should be > 0.45'
        assert_(freq > 0.45, msg)
        # 对于大 N，理论值约为 0.19882718
        freq = np.sum(rvsn == 2) / N
        msg = f'Frequency was {freq:f}, should be < 0.23'
        assert_(freq < 0.23, msg)

    # 定义测试方法 test_shuffle_mixed_dimension
    def test_shuffle_mixed_dimension(self):
        # 针对 trac 票号 #2074 进行测试
        for t in [[1, 2, 3, None],
                  [(1, 1), (2, 2), (3, 3), None],
                  [1, (2, 2), (3, 3), None],
                  [(1, 1), 2, 3, None]]:
            random.seed(12345)
            shuffled = list(t)
            random.shuffle(shuffled)
            expected = np.array([t[0], t[3], t[1], t[2]], dtype=object)
            assert_array_equal(np.array(shuffled, dtype=object), expected)

    # 定义测试方法 test_call_within_randomstate
    def test_call_within_randomstate(self):
        # 检查自定义的 RandomState 不会调用全局状态
        m = random.RandomState()
        res = np.array([0, 8, 7, 2, 1, 9, 4, 7, 0, 3])
        for i in range(3):
            random.seed(i)
            m.seed(4321)
            # 如果 m.state 没有被尊重，结果会发生变化
            assert_array_equal(m.choice(10, size=10, p=np.ones(10)/10.), res)
    def test_multivariate_normal_size_types(self):
        # 测试多变量正态分布函数的 'size' 参数问题
        # 检查 multivariate_normal 函数的 size 参数能否接受 numpy 的整数类型
        random.multivariate_normal([0], [[0]], size=1)
        random.multivariate_normal([0], [[0]], size=np.int_(1))
        random.multivariate_normal([0], [[0]], size=np.int64(1))

    def test_beta_small_parameters(self):
        # 测试 beta 分布在参数 a 和 b 较小时不会因舍入误差导致 NaN
        # 问题来源于舍入误差导致 0 / 0，见 GitHub 问题 gh-5851
        random.seed(1234567890)
        x = random.beta(0.0001, 0.0001, size=100)
        assert_(not np.any(np.isnan(x)), 'Nans in random.beta')

    def test_choice_sum_of_probs_tolerance(self):
        # 检查概率和应该为 1.0，允许一定的容差
        # 对于低精度数据类型，容差设置过紧，见 numpy GitHub 问题 6123
        random.seed(1234)
        a = [1, 2, 3]
        counts = [4, 4, 2]
        for dt in np.float16, np.float32, np.float64:
            probs = np.array(counts, dtype=dt) / sum(counts)
            c = random.choice(a, p=probs)
            assert_(c in a)
            assert_raises(ValueError, random.choice, a, p=probs*0.9)

    def test_shuffle_of_array_of_different_length_strings(self):
        # 测试对不同长度字符串数组进行洗牌操作
        # 确保在垃圾回收时不会导致段错误
        # 测试 GitHub 问题 gh-7710
        random.seed(1234)

        a = np.array(['a', 'a' * 1000])

        for _ in range(100):
            random.shuffle(a)

        # 强制进行垃圾回收，不应该导致段错误
        import gc
        gc.collect()

    def test_shuffle_of_array_of_objects(self):
        # 测试对对象数组进行洗牌操作
        # 确保在垃圾回收时不会导致段错误
        # 见 GitHub 问题 gh-7719
        random.seed(1234)
        a = np.array([np.arange(1), np.arange(4)], dtype=object)

        for _ in range(1000):
            random.shuffle(a)

        # 强制进行垃圾回收，不应该导致段错误
        import gc
        gc.collect()

    def test_permutation_subclass(self):
        class N(np.ndarray):
            pass

        random.seed(1)
        orig = np.arange(3).view(N)
        perm = random.permutation(orig)
        assert_array_equal(perm, np.array([0, 2, 1]))
        assert_array_equal(orig, np.arange(3).view(N))

        class M:
            a = np.arange(5)

            def __array__(self, dtype=None, copy=None):
                return self.a

        random.seed(1)
        m = M()
        perm = random.permutation(m)
        assert_array_equal(perm, np.array([2, 1, 4, 0, 3]))
        assert_array_equal(m.__array__(), np.arange(5))
    def test_warns_byteorder(self):
        # GH 13159
        # 根据系统字节顺序选择相应的数据类型格式
        other_byteord_dt = '<i4' if sys.byteorder == 'big' else '>i4'
        # 使用 pytest 检查是否发出了弃用警告
        with pytest.deprecated_call(match='non-native byteorder is not'):
            random.randint(0, 200, size=10, dtype=other_byteord_dt)

    def test_named_argument_initialization(self):
        # GH 13669
        # 使用特定的种子创建两个随机状态对象
        rs1 = np.random.RandomState(123456789)
        rs2 = np.random.RandomState(seed=123456789)
        # 检查两个随机状态对象生成的随机整数是否相等
        assert rs1.randint(0, 100) == rs2.randint(0, 100)

    def test_choice_retun_dtype(self):
        # GH 9867, now long since the NumPy default changed.
        # 使用给定概率分布选择随机数，验证返回值的数据类型是否为 np.long
        c = np.random.choice(10, p=[.1]*10, size=2)
        assert c.dtype == np.dtype(np.long)
        c = np.random.choice(10, p=[.1]*10, replace=False, size=2)
        assert c.dtype == np.dtype(np.long)
        c = np.random.choice(10, size=2)
        assert c.dtype == np.dtype(np.long)
        c = np.random.choice(10, replace=False, size=2)
        assert c.dtype == np.dtype(np.long)

    @pytest.mark.skipif(np.iinfo('l').max < 2**32,
                        reason='Cannot test with 32-bit C long')
    def test_randint_117(self):
        # GH 14189
        # 使用种子初始化随机数生成器，生成指定范围内的随机整数数组，验证结果是否符合预期
        random.seed(0)
        expected = np.array([2357136044, 2546248239, 3071714933, 3626093760,
                             2588848963, 3684848379, 2340255427, 3638918503,
                             1819583497, 2678185683], dtype='int64')
        actual = random.randint(2**32, size=10)
        assert_array_equal(actual, expected)

    def test_p_zero_stream(self):
        # Regression test for gh-14522.  Ensure that future versions
        # generate the same variates as version 1.16.
        # 使用种子初始化随机数生成器，验证二项分布生成的结果是否与预期一致
        np.random.seed(12345)
        assert_array_equal(random.binomial(1, [0, 0.25, 0.5, 0.75, 1]),
                           [0, 0, 0, 1, 1])

    def test_n_zero_stream(self):
        # Regression test for gh-14522.  Ensure that future versions
        # generate the same variates as version 1.16.
        # 使用种子初始化随机数生成器，验证二项分布生成的结果是否与预期一致
        np.random.seed(8675309)
        expected = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [3, 4, 2, 3, 3, 1, 5, 3, 1, 3]])
        assert_array_equal(random.binomial([[0], [10]], 0.25, size=(2, 10)),
                           expected)
# 定义一个测试函数，用于测试 random.multinomial 函数处理空概率值的情况
def test_multinomial_empty():
    # gh-20483
    # 确保空的概率值被正确处理
    assert random.multinomial(10, []).shape == (0,)
    # 确保在指定大小的情况下，空的概率值被正确处理
    assert random.multinomial(3, [], size=(7, 5, 3)).shape == (7, 5, 3, 0)


# 定义一个测试函数，用于测试 random.multinomial 函数对一维概率值的处理
def test_multinomial_1d_pval():
    # gh-20483
    # 使用 pytest 检查是否会引发 TypeError 异常，且异常信息应匹配 "pvals must be a 1-d"
    with pytest.raises(TypeError, match="pvals must be a 1-d"):
        random.multinomial(10, 0.3)
```