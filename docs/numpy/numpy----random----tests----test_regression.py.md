# `.\numpy\numpy\random\tests\test_regression.py`

```py
# 导入必要的模块和函数
import sys
from numpy.testing import (
    assert_, assert_array_equal, assert_raises,
    )
from numpy import random
import numpy as np

# 定义测试类 TestRegression，用于测试回归功能
class TestRegression:

    # 定义测试方法 test_VonMises_range，测试 Von Mises 分布的范围
    def test_VonMises_range(self):
        # 确保生成的随机变量落在 [-pi, pi] 范围内
        # 这是对问题 #986 的回归测试
        for mu in np.linspace(-7., 7., 5):
            r = random.mtrand.vonmises(mu, 1, 50)
            assert_(np.all(r > -np.pi) and np.all(r <= np.pi))

    # 定义测试方法 test_hypergeometric_range，测试超几何分布的范围
    def test_hypergeometric_range(self):
        # 这是对问题 #921 的测试
        assert_(np.all(np.random.hypergeometric(3, 18, 11, size=10) < 4))
        assert_(np.all(np.random.hypergeometric(18, 3, 11, size=10) > 0))

        # 这是对问题 #5623 的测试
        args = [
            (2**20 - 2, 2**20 - 2, 2**20 - 2),  # 检查 32 位系统
        ]
        is_64bits = sys.maxsize > 2**32
        if is_64bits and sys.platform != 'win32':
            # 检查 64 位系统
            args.append((2**40 - 2, 2**40 - 2, 2**40 - 2))
        for arg in args:
            assert_(np.random.hypergeometric(*arg) > 0)

    # 定义测试方法 test_logseries_convergence，测试对数级数分布的收敛性
    def test_logseries_convergence(self):
        # 这是对问题 #923 的测试
        N = 1000
        np.random.seed(0)
        rvsn = np.random.logseries(0.8, size=N)
        # 下面两个频率计数应该接近理论值，对于这么大的样本
        # 理论上对于大 N 结果是 0.49706795
        freq = np.sum(rvsn == 1) / N
        msg = f'Frequency was {freq:f}, should be > 0.45'
        assert_(freq > 0.45, msg)
        # 理论上对于大 N 结果是 0.19882718
        freq = np.sum(rvsn == 2) / N
        msg = f'Frequency was {freq:f}, should be < 0.23'
        assert_(freq < 0.23, msg)

    # 定义测试方法 test_shuffle_mixed_dimension，测试混合维度的洗牌功能
    def test_shuffle_mixed_dimension(self):
        # 这是对 trac 问题 #2074 的测试
        for t in [[1, 2, 3, None],
                  [(1, 1), (2, 2), (3, 3), None],
                  [1, (2, 2), (3, 3), None],
                  [(1, 1), 2, 3, None]]:
            np.random.seed(12345)
            shuffled = list(t)
            random.shuffle(shuffled)
            expected = np.array([t[0], t[3], t[1], t[2]], dtype=object)
            assert_array_equal(np.array(shuffled, dtype=object), expected)

    # 定义测试方法 test_call_within_randomstate，测试在自定义 RandomState 中调用
    def test_call_within_randomstate(self):
        # 检查自定义的 RandomState 是否影响全局状态
        m = np.random.RandomState()
        res = np.array([0, 8, 7, 2, 1, 9, 4, 7, 0, 3])
        for i in range(3):
            np.random.seed(i)
            m.seed(4321)
            # 如果 m.state 没有被遵循，结果将会改变
            assert_array_equal(m.choice(10, size=10, p=np.ones(10)/10.), res)
    def test_multivariate_normal_size_types(self):
        # 测试多元正态分布中 'size' 参数的问题。
        # 检查 multivariate_normal 函数的 size 参数是否可以是 numpy 的整数类型。
        np.random.multivariate_normal([0], [[0]], size=1)
        np.random.multivariate_normal([0], [[0]], size=np.int_(1))
        np.random.multivariate_normal([0], [[0]], size=np.int64(1))

    def test_beta_small_parameters(self):
        # 测试 beta 分布在小的 a 和 b 参数下不会因为舍入误差导致 NaN，详情见 gh-5851
        np.random.seed(1234567890)
        x = np.random.beta(0.0001, 0.0001, size=100)
        assert_(not np.any(np.isnan(x)), 'Nans in np.random.beta')

    def test_choice_sum_of_probs_tolerance(self):
        # 确保概率的总和为 1.0，允许一定的误差范围。
        # 对于低精度的数据类型，之前的容差过于严格，参见 numpy GitHub 问题 6123。
        np.random.seed(1234)
        a = [1, 2, 3]
        counts = [4, 4, 2]
        for dt in np.float16, np.float32, np.float64:
            probs = np.array(counts, dtype=dt) / sum(counts)
            c = np.random.choice(a, p=probs)
            assert_(c in a)
            assert_raises(ValueError, np.random.choice, a, p=probs*0.9)

    def test_shuffle_of_array_of_different_length_strings(self):
        # 测试对一个包含不同长度字符串的数组进行洗牌不会在垃圾回收时导致段错误。
        # 测试 gh-7710
        np.random.seed(1234)

        a = np.array(['a', 'a' * 1000])

        for _ in range(100):
            np.random.shuffle(a)

        # 强制进行垃圾回收，不应该导致段错误。
        import gc
        gc.collect()

    def test_shuffle_of_array_of_objects(self):
        # 测试对一个包含对象的数组进行洗牌不会在垃圾回收时导致段错误。
        # 参见 gh-7719
        np.random.seed(1234)
        a = np.array([np.arange(1), np.arange(4)], dtype=object)

        for _ in range(1000):
            np.random.shuffle(a)

        # 强制进行垃圾回收，不应该导致段错误。
        import gc
        gc.collect()

    def test_permutation_subclass(self):
        class N(np.ndarray):
            pass

        np.random.seed(1)
        orig = np.arange(3).view(N)
        perm = np.random.permutation(orig)
        assert_array_equal(perm, np.array([0, 2, 1]))
        assert_array_equal(orig, np.arange(3).view(N))

        class M:
            a = np.arange(5)

            def __array__(self, dtype=None, copy=None):
                return self.a

        np.random.seed(1)
        m = M()
        perm = np.random.permutation(m)
        assert_array_equal(perm, np.array([2, 1, 4, 0, 3]))
        assert_array_equal(m.__array__(), np.arange(5))
```