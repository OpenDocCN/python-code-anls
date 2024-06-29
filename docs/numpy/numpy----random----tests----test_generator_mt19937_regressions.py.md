# `.\numpy\numpy\random\tests\test_generator_mt19937_regressions.py`

```
    # 导入必要的测试断言函数
    from numpy.testing import (assert_, assert_array_equal)
    # 导入 NumPy 库并使用别名 np
    import numpy as np
    # 导入 pytest 库，用于测试框架
    import pytest
    # 导入随机数生成器 Generator 和 MT19937 BitGenerator
    from numpy.random import Generator, MT19937

    # 定义测试类 TestRegression
    class TestRegression:

        # 每个测试方法执行前的初始化方法
        def setup_method(self):
            # 创建一个具有确定种子的随机数生成器对象
            self.mt19937 = Generator(MT19937(121263137472525314065))

        # 测试 vonmises 方法的范围
        def test_vonmises_range(self):
            # 对 mu 参数进行五个值的线性空间采样，测试随机变量的范围是否在 [-pi, pi] 内
            for mu in np.linspace(-7., 7., 5):
                r = self.mt19937.vonmises(mu, 1, 50)
                assert_(np.all(r > -np.pi) and np.all(r <= np.pi))

        # 测试 hypergeometric 方法的范围
        def test_hypergeometric_range(self):
            # 测试 hypergeometric 方法生成的随机数是否小于 4
            assert_(np.all(self.mt19937.hypergeometric(3, 18, 11, size=10) < 4))
            # 测试 hypergeometric 方法生成的随机数是否大于 0
            assert_(np.all(self.mt19937.hypergeometric(18, 3, 11, size=10) > 0))

            # 对于 32 位系统，检查 hypergeometric 方法生成的随机数是否大于 0
            args = (2**20 - 2, 2**20 - 2, 2**20 - 2)
            assert_(self.mt19937.hypergeometric(*args) > 0)

        # 测试 logseries 方法的收敛性
        def test_logseries_convergence(self):
            # 设定样本数 N
            N = 1000
            # 生成 logseries 分布的随机数序列
            rvsn = self.mt19937.logseries(0.8, size=N)
            # 计算随机数为 1 的频率
            freq = np.sum(rvsn == 1) / N
            # 断言频率大于 0.45
            msg = f'Frequency was {freq:f}, should be > 0.45'
            assert_(freq > 0.45, msg)
            # 计算随机数为 2 的频率
            freq = np.sum(rvsn == 2) / N
            # 断言频率小于 0.23
            msg = f'Frequency was {freq:f}, should be < 0.23'
            assert_(freq < 0.23, msg)

        # 测试 shuffle 方法对混合维度数组的处理
        def test_shuffle_mixed_dimension(self):
            # 对不同混合维度数组进行测试
            for t in [[1, 2, 3, None],
                      [(1, 1), (2, 2), (3, 3), None],
                      [1, (2, 2), (3, 3), None],
                      [(1, 1), 2, 3, None]]:
                # 创建一个具有确定种子的新随机数生成器对象
                mt19937 = Generator(MT19937(12345))
                # 将列表 t 转换为 numpy 数组
                shuffled = np.array(t, dtype=object)
                # 使用随机数生成器对象对数组进行乱序操作
                mt19937.shuffle(shuffled)
                # 预期的乱序结果
                expected = np.array([t[2], t[0], t[3], t[1]], dtype=object)
                # 断言乱序后的数组与预期结果相等
                assert_array_equal(np.array(shuffled, dtype=object), expected)

        # 测试自定义 BitGenerator 在调用时不影响全局状态
        def test_call_within_randomstate(self):
            # 预期的结果数组
            res = np.array([1, 8, 0, 1, 5, 3, 3, 8, 1, 4])
            # 进行三次测试
            for i in range(3):
                # 创建一个具有确定种子的新随机数生成器对象
                mt19937 = Generator(MT19937(i))
                m = Generator(MT19937(4321))
                # 断言如果 m.state 没有被尊重，则结果会改变
                assert_array_equal(m.choice(10, size=10, p=np.ones(10)/10.), res)

        # 测试 multivariate_normal 方法的 size 参数类型
        def test_multivariate_normal_size_types(self):
            # 测试 multivariate_normal 方法的 size 参数接受 numpy 整数类型
            self.mt19937.multivariate_normal([0], [[0]], size=1)
            self.mt19937.multivariate_normal([0], [[0]], size=np.int_(1))
            self.mt19937.multivariate_normal([0], [[0]], size=np.int64(1))
    def test_beta_small_parameters(self):
        # 测试当 beta 函数的参数 a 和 b 较小时，不会由于舍入误差导致产生 NaN，例如由于 0 / 0 引起的问题，参见 gh-5851
        x = self.mt19937.beta(0.0001, 0.0001, size=100)
        assert_(not np.any(np.isnan(x)), 'Nans in mt19937.beta')

    def test_beta_very_small_parameters(self):
        # gh-24203: 当参数非常小的时候，beta 函数可能会 hang。
        self.mt19937.beta(1e-49, 1e-40)

    def test_beta_ridiculously_small_parameters(self):
        # gh-24266: 当参数非常小（比如 subnormal 或最小正常数的小倍数）时，beta 函数可能会生成 NaN。
        tiny = np.finfo(1.0).tiny
        x = self.mt19937.beta(tiny/32, tiny/40, size=50)
        assert not np.any(np.isnan(x))

    def test_choice_sum_of_probs_tolerance(self):
        # 概率和应该为 1.0，允许一定的容差。
        # 对于低精度数据类型，容差设置过紧。参见 numpy github 问题 6123。
        a = [1, 2, 3]
        counts = [4, 4, 2]
        for dt in np.float16, np.float32, np.float64:
            probs = np.array(counts, dtype=dt) / sum(counts)
            c = self.mt19937.choice(a, p=probs)
            assert_(c in a)
            with pytest.raises(ValueError):
                self.mt19937.choice(a, p=probs*0.9)

    def test_shuffle_of_array_of_different_length_strings(self):
        # 测试对包含不同长度字符串的数组进行乱序排列时，不会在垃圾回收时导致段错误。
        # 测试 gh-7710

        a = np.array(['a', 'a' * 1000])

        for _ in range(100):
            self.mt19937.shuffle(a)

        # 强制进行垃圾回收 - 不应该导致段错误。
        import gc
        gc.collect()

    def test_shuffle_of_array_of_objects(self):
        # 测试对包含对象的数组进行乱序排列时，不会在垃圾回收时导致段错误。
        # 参见 gh-7719
        a = np.array([np.arange(1), np.arange(4)], dtype=object)

        for _ in range(1000):
            self.mt19937.shuffle(a)

        # 强制进行垃圾回收 - 不应该导致段错误。
        import gc
        gc.collect()

    def test_permutation_subclass(self):
        # 测试对子类化的 ndarray 进行排列时的行为。
        class N(np.ndarray):
            pass

        mt19937 = Generator(MT19937(1))
        orig = np.arange(3).view(N)
        perm = mt19937.permutation(orig)
        assert_array_equal(perm, np.array([2, 0, 1]))
        assert_array_equal(orig, np.arange(3).view(N))

        class M:
            a = np.arange(5)

            def __array__(self, dtype=None, copy=None):
                return self.a

        mt19937 = Generator(MT19937(1))
        m = M()
        perm = mt19937.permutation(m)
        assert_array_equal(perm, np.array([4, 1, 3, 0, 2]))
        assert_array_equal(m.__array__(), np.arange(5))
    # 定义测试方法，用于测试 gamma 分布中参数为 0 时的情况
    def test_gamma_0(self):
        # 断言使用 mt19937 对象生成的 gamma 分布中，参数为 0.0 时的返回值应为 0.0
        assert self.mt19937.standard_gamma(0.0) == 0.0
        # 断言使用 mt19937 对象生成的 gamma 分布中，参数为 [0.0] 时的返回值应为 0.0
        assert_array_equal(self.mt19937.standard_gamma([0.0]), 0.0)

        # 使用 mt19937 对象生成的 gamma 分布中，参数为 [0.0] 时，指定返回数据类型为 'float'
        actual = self.mt19937.standard_gamma([0.0], dtype='float')
        # 期望的返回结果是一个包含单个元素 [0.] 的 numpy 数组，数据类型为 np.float32
        expected = np.array([0.], dtype=np.float32)
        # 断言实际返回值与期望值相等
        assert_array_equal(actual, expected)

    # 定义测试方法，用于测试几何分布中极小概率情况的处理
    def test_geometric_tiny_prob(self):
        # 回归测试 gh-17007。
        # 当 p = 1e-30 时，样本超过 2**63-1 的概率是 0.9999999999907766，
        # 所以我们期望的结果是全部都是 2**63-1。
        assert_array_equal(self.mt19937.geometric(p=1e-30, size=3),
                           np.iinfo(np.int64).max)
```