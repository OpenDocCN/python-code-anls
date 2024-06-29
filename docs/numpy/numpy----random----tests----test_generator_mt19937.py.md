# `.\numpy\numpy\random\tests\test_generator_mt19937.py`

```
# 导入必要的标准库和第三方库
import os.path           # 导入处理文件路径的标准库
import sys               # 导入系统相关的标准库
import hashlib           # 导入哈希算法相关的标准库

import pytest            # 导入 pytest 测试框架

import numpy as np       # 导入 NumPy 数学库
from numpy.exceptions import AxisError       # 导入 NumPy 异常类
from numpy.linalg import LinAlgError         # 导入 NumPy 线性代数错误类
from numpy.testing import (                 # 导入 NumPy 测试工具
    assert_, assert_raises, assert_equal, assert_allclose,
    assert_warns, assert_no_warnings, assert_array_equal,
    assert_array_almost_equal, suppress_warnings, IS_WASM)

from numpy.random import Generator, MT19937, SeedSequence, RandomState   # 导入 NumPy 随机数生成相关类

# 创建一个随机数生成器对象
random = Generator(MT19937())

# 随机数生成器的跳跃测试数据集
JUMP_TEST_DATA = [
    {
        "seed": 0,
        "steps": 10,
        "initial": {"key_sha256": "bb1636883c2707b51c5b7fc26c6927af4430f2e0785a8c7bc886337f919f9edf", "pos": 9},
        "jumped": {"key_sha256": "ff682ac12bb140f2d72fba8d3506cf4e46817a0db27aae1683867629031d8d55", "pos": 598},
    },
    {
        "seed": 384908324,
        "steps": 312,
        "initial": {"key_sha256": "16b791a1e04886ccbbb4d448d6ff791267dc458ae599475d08d5cced29d11614", "pos": 311},
        "jumped": {"key_sha256": "a0110a2cf23b56be0feaed8f787a7fc84bef0cb5623003d75b26bdfa1c18002c", "pos": 276},
    },
    {
        "seed": [839438204, 980239840, 859048019, 821],
        "steps": 511,
        "initial": {"key_sha256": "d306cf01314d51bd37892d874308200951a35265ede54d200f1e065004c3e9ea", "pos": 510},
        "jumped": {"key_sha256": "0e00ab449f01a5195a83b4aee0dfbc2ce8d46466a640b92e33977d2e42f777f8", "pos": 475},
    },
]

# 定义 pytest 的 fixture 函数 endpoint，作用域为模块级别，参数为 True 或 False
@pytest.fixture(scope='module', params=[True, False])
def endpoint(request):
    return request.param

# 测试类 TestSeed，用于测试随机数生成器种子相关的功能
class TestSeed:
    # 测试标量种子情况
    def test_scalar(self):
        s = Generator(MT19937(0))
        assert_equal(s.integers(1000), 479)
        s = Generator(MT19937(4294967295))
        assert_equal(s.integers(1000), 324)

    # 测试数组种子情况
    def test_array(self):
        s = Generator(MT19937(range(10)))
        assert_equal(s.integers(1000), 465)
        s = Generator(MT19937(np.arange(10)))
        assert_equal(s.integers(1000), 465)
        s = Generator(MT19937([0]))
        assert_equal(s.integers(1000), 479)
        s = Generator(MT19937([4294967295]))
        assert_equal(s.integers(1000), 324)

    # 测试 SeedSequence 种子情况
    def test_seedsequence(self):
        s = MT19937(SeedSequence(0))
        assert_equal(s.random_raw(1), 2058676884)

    # 测试无效的标量种子情况
    def test_invalid_scalar(self):
        # 种子必须是无符号的 32 位整数
        assert_raises(TypeError, MT19937, -0.5)
        assert_raises(ValueError, MT19937, -1)

    # 测试无效的数组种子情况
    def test_invalid_array(self):
        # 种子必须是无符号整数
        assert_raises(TypeError, MT19937, [-0.5])
        assert_raises(ValueError, MT19937, [-1])
        assert_raises(ValueError, MT19937, [1, -2, 4294967296])

    # 测试未实例化比特生成器的情况
    def test_noninstantized_bitgen(self):
        assert_raises(ValueError, Generator, MT19937)

# 测试类 TestBinomial 从这里开始，下面的代码需要继续添加注释
    def test_n_zero(self):
        # 测试二项分布中 n == 0 的边界情况。
        # 对于任何 p 在 [0, 1] 区间内，binomial(0, p) 应该为零。
        # 这个测试解决了 issue #3480。
        # 创建一个包含两个整数 0 的数组
        zeros = np.zeros(2, dtype='int')
        # 对于每个 p 值进行测试：0, 0.5, 1
        for p in [0, .5, 1]:
            # 断言当 n=0 时，binomial(0, p) 的返回值为 0
            assert_(random.binomial(0, p) == 0)
            # 断言当 n 是一个包含两个 0 的数组时，binomial(zeros, p) 返回的结果也应该是 zeros
            assert_array_equal(random.binomial(zeros, p), zeros)

    def test_p_is_nan(self):
        # 处理 issue #4571。
        # 断言当 p 是 NaN 时，应该抛出 ValueError 异常
        assert_raises(ValueError, random.binomial, 1, np.nan)
class TestMultinomial:
    # 多项式分布测试类

    def test_basic(self):
        # 测试基本情况下的多项式分布
        random.multinomial(100, [0.2, 0.8])

    def test_zero_probability(self):
        # 测试概率为零的情况下的多项式分布
        random.multinomial(100, [0.2, 0.8, 0.0, 0.0, 0.0])

    def test_int_negative_interval(self):
        # 测试负整数区间的随机整数生成
        assert_(-5 <= random.integers(-5, -1) < -1)
        x = random.integers(-5, -1, 5)
        assert_(np.all(-5 <= x))
        assert_(np.all(x < -1))

    def test_size(self):
        # gh-3173
        # 测试多项式分布的不同尺寸参数
        p = [0.5, 0.5]
        assert_equal(random.multinomial(1, p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.multinomial(1, p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.multinomial(1, p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.multinomial(1, p, [2, 2]).shape, (2, 2, 2))
        assert_equal(random.multinomial(1, p, (2, 2)).shape, (2, 2, 2))
        assert_equal(random.multinomial(1, p, np.array((2, 2))).shape,
                     (2, 2, 2))

        assert_raises(TypeError, random.multinomial, 1, p,
                      float(1))

    def test_invalid_prob(self):
        # 测试无效概率参数
        assert_raises(ValueError, random.multinomial, 100, [1.1, 0.2])
        assert_raises(ValueError, random.multinomial, 100, [-.1, 0.9])

    def test_invalid_n(self):
        # 测试无效的 n 参数
        assert_raises(ValueError, random.multinomial, -1, [0.8, 0.2])
        assert_raises(ValueError, random.multinomial, [-1] * 10, [0.8, 0.2])

    def test_p_non_contiguous(self):
        # 测试非连续的 p 参数
        p = np.arange(15.)
        p /= np.sum(p[1::3])
        pvals = p[1::3]
        random = Generator(MT19937(1432985819))
        non_contig = random.multinomial(100, pvals=pvals)
        random = Generator(MT19937(1432985819))
        contig = random.multinomial(100, pvals=np.ascontiguousarray(pvals))
        assert_array_equal(non_contig, contig)

    def test_multinomial_pvals_float32(self):
        # 测试 pvals 数组为 float32 类型的情况
        x = np.array([9.9e-01, 9.9e-01, 1.0e-09, 1.0e-09, 1.0e-09, 1.0e-09,
                      1.0e-09, 1.0e-09, 1.0e-09, 1.0e-09], dtype=np.float32)
        pvals = x / x.sum()
        random = Generator(MT19937(1432985819))
        match = r"[\w\s]*pvals array is cast to 64-bit floating"
        with pytest.raises(ValueError, match=match):
            random.multinomial(1, pvals)


class TestMultivariateHypergeometric:

    def setup_method(self):
        # 设置测试方法的种子值
        self.seed = 8675309
    # 定义测试函数，用于验证参数的有效性
    def test_argument_validation(self):
        # Error cases...

        # `colors` must be a 1-d sequence
        # 断言：验证当`colors`不是一维序列时，会引发 ValueError 异常
        assert_raises(ValueError, random.multivariate_hypergeometric,
                      10, 4)

        # Negative nsample
        # 断言：验证当 nsample 为负数时，会引发 ValueError 异常
        assert_raises(ValueError, random.multivariate_hypergeometric,
                      [2, 3, 4], -1)

        # Negative color
        # 断言：验证当 colors 中存在负数时，会引发 ValueError 异常
        assert_raises(ValueError, random.multivariate_hypergeometric,
                      [-1, 2, 3], 2)

        # nsample exceeds sum(colors)
        # 断言：验证当 nsample 超过 colors 中元素总和时，会引发 ValueError 异常
        assert_raises(ValueError, random.multivariate_hypergeometric,
                      [2, 3, 4], 10)

        # nsample exceeds sum(colors) (edge case of empty colors)
        # 断言：验证当 colors 为空时，nsample 超过 0 会引发 ValueError 异常
        assert_raises(ValueError, random.multivariate_hypergeometric,
                      [], 1)

        # Validation errors associated with very large values in colors.
        # 断言：验证当 colors 中包含极大值时，会引发 ValueError 异常
        assert_raises(ValueError, random.multivariate_hypergeometric,
                      [999999999, 101], 5, 1, 'marginals')

        # Check for maximum int64 index
        # 断言：验证当 colors 中的一个元素接近 np.int64 的最大索引时，会引发 ValueError 异常
        int64_info = np.iinfo(np.int64)
        max_int64 = int64_info.max
        max_int64_index = max_int64 // int64_info.dtype.itemsize
        assert_raises(ValueError, random.multivariate_hypergeometric,
                      [max_int64_index - 100, 101], 5, 1, 'count')

    @pytest.mark.parametrize('method', ['count', 'marginals'])
    def test_edge_cases(self, method):
        # Set the seed, but in fact, all the results in this test are
        # deterministic, so we don't really need this.
        # 设置随机数生成器的种子，尽管在这个测试中所有的结果都是确定性的，实际上我们不真正需要这个设置。
        random = Generator(MT19937(self.seed))

        # Test case where all colors are zero
        # 测试当所有的 colors 都为零时的情况
        x = random.multivariate_hypergeometric([0, 0, 0], 0, method=method)
        assert_array_equal(x, [0, 0, 0])

        # Test case where colors is empty
        # 测试当 colors 为空时的情况
        x = random.multivariate_hypergeometric([], 0, method=method)
        assert_array_equal(x, [])

        # Test case where colors is empty and size=1
        # 测试当 colors 为空且 size=1 时的情况
        x = random.multivariate_hypergeometric([], 0, size=1, method=method)
        assert_array_equal(x, np.empty((1, 0), dtype=np.int64))

        # Test case where colors has non-zero values but nsample=0
        # 测试当 colors 中有非零值但 nsample=0 时的情况
        x = random.multivariate_hypergeometric([1, 2, 3], 0, method=method)
        assert_array_equal(x, [0, 0, 0])

        # Test case where colors has one non-zero value and nsample is less than it
        # 测试当 colors 中只有一个非零值且 nsample 小于该值时的情况
        x = random.multivariate_hypergeometric([9, 0, 0], 3, method=method)
        assert_array_equal(x, [3, 0, 0])

        # Test case where colors is balanced (each color has equal number of balls)
        # 测试当 colors 是平衡的情况（每种颜色的球数相等）
        colors = [1, 1, 0, 1, 1]
        x = random.multivariate_hypergeometric(colors, sum(colors),
                                               method=method)
        assert_array_equal(x, colors)

        # Test case where each color has different number of balls
        # 测试当每种颜色的球数不同的情况
        x = random.multivariate_hypergeometric([3, 4, 5], 12, size=3,
                                               method=method)
        assert_array_equal(x, [[3, 4, 5]]*3)

    # Cases for nsample:
    #     nsample < 10
    #     10 <= nsample < colors.sum()/2
    #     colors.sum()/2 < nsample < colors.sum() - 10
    #     colors.sum() - 10 < nsample < colors.sum()
    @pytest.mark.parametrize('nsample', [8, 25, 45, 55])
    @pytest.mark.parametrize('method', ['count', 'marginals'])
    @pytest.mark.parametrize('size', [5, (2, 3), 150000])
    # 测试常规情况的函数，用于验证多元超几何分布生成的样本是否符合预期
    def test_typical_cases(self, nsample, method, size):
        # 使用种子生成随机数发生器对象
        random = Generator(MT19937(self.seed))

        # 定义不同颜色球的数量数组
        colors = np.array([10, 5, 20, 25])

        # 生成多元超几何分布的样本
        sample = random.multivariate_hypergeometric(colors, nsample, size,
                                                    method=method)

        # 根据输入的 size 参数确定期望的样本形状
        if isinstance(size, int):
            expected_shape = (size,) + colors.shape
        else:
            expected_shape = size + colors.shape

        # 断言样本的形状与预期形状相同
        assert_equal(sample.shape, expected_shape)

        # 断言样本中所有元素均大于等于 0
        assert_((sample >= 0).all())

        # 断言样本中所有元素均小于等于对应颜色球的数量
        assert_((sample <= colors).all())

        # 断言样本每一组中元素之和等于 nsample
        assert_array_equal(sample.sum(axis=-1),
                           np.full(size, fill_value=nsample, dtype=int))

        # 如果 size 是整数且大于等于 100000，则比较样本均值与预期值
        if isinstance(size, int) and size >= 100000:
            # 此样本足够大，可以将其均值与预期值进行比较
            assert_allclose(sample.mean(axis=0),
                            nsample * colors / colors.sum(),
                            rtol=1e-3, atol=0.005)

    # 第一个重复性测试函数，用于验证生成的多元超几何分布样本是否可重现
    def test_repeatability1(self):
        # 使用相同的种子生成随机数发生器对象
        random = Generator(MT19937(self.seed))

        # 生成多元超几何分布的样本
        sample = random.multivariate_hypergeometric([3, 4, 5], 5, size=5,
                                                    method='count')

        # 预期的样本值
        expected = np.array([[2, 1, 2],
                             [2, 1, 2],
                             [1, 1, 3],
                             [2, 0, 3],
                             [2, 1, 2]])

        # 断言生成的样本与预期值相等
        assert_array_equal(sample, expected)

    # 第二个重复性测试函数，用于验证生成的多元超几何分布样本是否可重现
    def test_repeatability2(self):
        # 使用相同的种子生成随机数发生器对象
        random = Generator(MT19937(self.seed))

        # 生成多元超几何分布的样本
        sample = random.multivariate_hypergeometric([20, 30, 50], 50,
                                                    size=5,
                                                    method='marginals')

        # 预期的样本值
        expected = np.array([[ 9, 17, 24],
                             [ 7, 13, 30],
                             [ 9, 15, 26],
                             [ 9, 17, 24],
                             [12, 14, 24]])

        # 断言生成的样本与预期值相等
        assert_array_equal(sample, expected)

    # 第三个重复性测试函数，用于验证生成的多元超几何分布样本是否可重现
    def test_repeatability3(self):
        # 使用相同的种子生成随机数发生器对象
        random = Generator(MT19937(self.seed))

        # 生成多元超几何分布的样本
        sample = random.multivariate_hypergeometric([20, 30, 50], 12,
                                                    size=5,
                                                    method='marginals')

        # 预期的样本值
        expected = np.array([[2, 3, 7],
                             [5, 3, 4],
                             [2, 5, 5],
                             [5, 3, 4],
                             [1, 5, 6]])

        # 断言生成的样本与预期值相等
        assert_array_equal(sample, expected)
class TestSetState:
    # 设置测试方法的初始化状态
    def setup_method(self):
        # 设置种子为 1234567890
        self.seed = 1234567890
        # 使用 MT19937 算法生成器初始化随机生成器
        self.rg = Generator(MT19937(self.seed))
        # 获取位生成器对象
        self.bit_generator = self.rg.bit_generator
        # 获取当前位生成器的状态
        self.state = self.bit_generator.state
        # 保存位生成器的遗留状态
        self.legacy_state = (self.state['bit_generator'],
                             self.state['state']['key'],
                             self.state['state']['pos'])

    # 测试高斯分布重置
    def test_gaussian_reset(self):
        # 确保缓存的每个另一个高斯分布被重置。
        old = self.rg.standard_normal(size=3)
        # 恢复位生成器的状态
        self.bit_generator.state = self.state
        new = self.rg.standard_normal(size=3)
        # 断言旧的和新的数组相等
        assert_(np.all(old == new))

    # 在中途恢复高斯分布状态的测试
    def test_gaussian_reset_in_media_res(self):
        # 当状态保存了一个缓存的高斯分布时，确保恢复缓存的高斯分布。
        self.rg.standard_normal()
        state = self.bit_generator.state
        old = self.rg.standard_normal(size=3)
        self.bit_generator.state = state
        new = self.rg.standard_normal(size=3)
        assert_(np.all(old == new))

    # 测试负二项分布
    def test_negative_binomial(self):
        # 确保负二项分布结果接受浮点数参数而不截断。
        self.rg.negative_binomial(0.5, 0.5)


class TestIntegers:
    rfunc = random.integers

    # 有效的整数/布尔类型
    itype = [bool, np.int8, np.uint8, np.int16, np.uint16,
             np.int32, np.uint32, np.int64, np.uint64]

    # 测试不支持的类型
    def test_unsupported_type(self, endpoint):
        assert_raises(TypeError, self.rfunc, 1, endpoint=endpoint, dtype=float)

    # 测试边界检查
    def test_bounds_checking(self, endpoint):
        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd
            assert_raises(ValueError, self.rfunc, lbnd - 1, ubnd,
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, lbnd, ubnd + 1,
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, ubnd, lbnd,
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, 1, 0, endpoint=endpoint,
                          dtype=dt)

            assert_raises(ValueError, self.rfunc, [lbnd - 1], ubnd,
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, [lbnd], [ubnd + 1],
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, [ubnd], [lbnd],
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, 1, [0],
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, [ubnd+1], [ubnd],
                          endpoint=endpoint, dtype=dt)
    # 对于给定的端点和数据类型，执行边界检查的测试函数
    def test_bounds_checking_array(self, endpoint):
        # 遍历所有整数类型数据类型
        for dt in self.itype:
            # 如果数据类型是布尔型，则下界为0，否则使用 np.iinfo 获取数据类型的最小值
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            # 如果数据类型是布尔型，则上界为2，否则使用 np.iinfo 获取数据类型的最大值再加上端点标记
            ubnd = 2 if dt is bool else np.iinfo(dt).max + (not endpoint)

            # 测试函数应该抛出 ValueError 异常，期望的情况是下界 - 1 超出范围
            assert_raises(ValueError, self.rfunc, [lbnd - 1] * 2, [ubnd] * 2,
                          endpoint=endpoint, dtype=dt)
            # 测试函数应该抛出 ValueError 异常，期望的情况是上界 + 1 超出范围
            assert_raises(ValueError, self.rfunc, [lbnd] * 2,
                          [ubnd + 1] * 2, endpoint=endpoint, dtype=dt)
            # 测试函数应该抛出 ValueError 异常，期望的情况是上界不应该小于下界
            assert_raises(ValueError, self.rfunc, ubnd, [lbnd] * 2,
                          endpoint=endpoint, dtype=dt)
            # 测试函数应该抛出 ValueError 异常，期望的情况是下界和上界不符合规定
            assert_raises(ValueError, self.rfunc, [1] * 2, 0,
                          endpoint=endpoint, dtype=dt)

    # 测试在零值和极值情况下的随机数生成函数
    def test_rng_zero_and_extremes(self, endpoint):
        # 遍历所有整数类型数据类型
        for dt in self.itype:
            # 如果数据类型是布尔型，则下界为0，否则使用 np.iinfo 获取数据类型的最小值
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            # 如果数据类型是布尔型，则上界为2，否则使用 np.iinfo 获取数据类型的最大值再加1
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            # 如果不是端点，则将上界减1
            ubnd = ubnd - 1 if endpoint else ubnd
            # 计算是否为开区间
            is_open = not endpoint

            # 测试生成的随机数是否等于目标值
            tgt = ubnd - 1
            assert_equal(self.rfunc(tgt, tgt + is_open, size=1000,
                                    endpoint=endpoint, dtype=dt), tgt)
            assert_equal(self.rfunc([tgt], tgt + is_open, size=1000,
                                    endpoint=endpoint, dtype=dt), tgt)

            # 测试生成的随机数是否等于目标值
            tgt = lbnd
            assert_equal(self.rfunc(tgt, tgt + is_open, size=1000,
                                    endpoint=endpoint, dtype=dt), tgt)
            assert_equal(self.rfunc(tgt, [tgt + is_open], size=1000,
                                    endpoint=endpoint, dtype=dt), tgt)

            # 测试生成的随机数是否等于目标值
            tgt = (lbnd + ubnd) // 2
            assert_equal(self.rfunc(tgt, tgt + is_open, size=1000,
                                    endpoint=endpoint, dtype=dt), tgt)
            assert_equal(self.rfunc([tgt], [tgt + is_open],
                                    size=1000, endpoint=endpoint, dtype=dt),
                         tgt)
    # 测试函数，用于验证生成随机数函数在边界条件下的行为
    def test_rng_zero_and_extremes_array(self, endpoint):
        # 设定数组大小为1000
        size = 1000
        # 遍历数据类型列表
        for dt in self.itype:
            # 如果数据类型是布尔型，则下界为0，否则为该数据类型的最小值
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            # 如果数据类型是布尔型，则上界为2，否则为该数据类型的最大值加1
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            # 如果 endpoint 为 False，则上界减一
            ubnd = ubnd - 1 if endpoint else ubnd

            # 设定目标值为上界减一
            tgt = ubnd - 1
            # 断言调用生成随机数函数后的结果与目标值相等
            assert_equal(self.rfunc([tgt], [tgt + 1],
                                    size=size, dtype=dt), tgt)
            assert_equal(self.rfunc(
                [tgt] * size, [tgt + 1] * size, dtype=dt), tgt)
            assert_equal(self.rfunc(
                [tgt] * size, [tgt + 1] * size, size=size, dtype=dt), tgt)

            # 设定目标值为下界
            tgt = lbnd
            # 断言调用生成随机数函数后的结果与目标值相等
            assert_equal(self.rfunc([tgt], [tgt + 1],
                                    size=size, dtype=dt), tgt)
            assert_equal(self.rfunc(
                [tgt] * size, [tgt + 1] * size, dtype=dt), tgt)
            assert_equal(self.rfunc(
                [tgt] * size, [tgt + 1] * size, size=size, dtype=dt), tgt)

            # 设定目标值为下界与上界之间的中间值
            tgt = (lbnd + ubnd) // 2
            # 断言调用生成随机数函数后的结果与目标值相等
            assert_equal(self.rfunc([tgt], [tgt + 1],
                                    size=size, dtype=dt), tgt)
            assert_equal(self.rfunc(
                [tgt] * size, [tgt + 1] * size, dtype=dt), tgt)
            assert_equal(self.rfunc(
                [tgt] * size, [tgt + 1] * size, size=size, dtype=dt), tgt)

    # 测试函数，用于验证生成随机数函数在整个范围内的行为
    def test_full_range(self, endpoint):
        # Test for ticket #1690

        # 遍历数据类型列表
        for dt in self.itype:
            # 如果数据类型是布尔型，则下界为0，否则为该数据类型的最小值
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            # 如果数据类型是布尔型，则上界为2，否则为该数据类型的最大值加1
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            # 如果 endpoint 为 False，则上界减一
            ubnd = ubnd - 1 if endpoint else ubnd

            try:
                # 尝试调用生成随机数函数，覆盖整个范围内的数值
                self.rfunc(lbnd, ubnd, endpoint=endpoint, dtype=dt)
            except Exception as e:
                # 若有异常抛出，则断言失败
                raise AssertionError("No error should have been raised, "
                                     "but one was with the following "
                                     "message:\n\n%s" % str(e))

    # 测试函数，用于验证生成随机数函数在整个范围内的行为（数组参数）
    def test_full_range_array(self, endpoint):
        # Test for ticket #1690

        # 遍历数据类型列表
        for dt in self.itype:
            # 如果数据类型是布尔型，则下界为0，否则为该数据类型的最小值
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            # 如果数据类型是布尔型，则上界为2，否则为该数据类型的最大值加1
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            # 如果 endpoint 为 False，则上界减一
            ubnd = ubnd - 1 if endpoint else ubnd

            try:
                # 尝试调用生成随机数函数，覆盖整个范围内的数值（使用数组参数）
                self.rfunc([lbnd] * 2, [ubnd], endpoint=endpoint, dtype=dt)
            except Exception as e:
                # 若有异常抛出，则断言失败
                raise AssertionError("No error should have been raised, "
                                     "but one was with the following "
                                     "message:\n\n%s" % str(e))
    # 使用随机数生成器 Generator 初始化，不使用固定种子
    random = Generator(MT19937())

    # 遍历 self.itype 中除第一个元素外的每个数据类型
    for dt in self.itype[1:]:
        # 遍历 ubnd 取值为 [4, 8, 16]
        for ubnd in [4, 8, 16]:
            # 调用 self.rfunc 生成指定范围和大小的随机数数组 vals
            vals = self.rfunc(2, ubnd - endpoint, size=2 ** 16,
                              endpoint=endpoint, dtype=dt)
            # 断言 vals 中的最大值小于 ubnd
            assert_(vals.max() < ubnd)
            # 断言 vals 中的最小值大于等于 2

    # 生成大小为 2**16 的 bool 类型的随机数数组 vals
    vals = self.rfunc(0, 2 - endpoint, size=2 ** 16, endpoint=endpoint,
                      dtype=bool)
    # 断言 vals 中的最大值小于 2
    assert_(vals.max() < 2)
    # 断言 vals 中的最小值大于等于 0

# 测试函数，用于验证生成的随机数在指定范围内的情况
def test_in_bounds_fuzz(self, endpoint):
    # 不同数据类型的随机数生成及比较的测试
    for dt in self.itype:
        # 根据数据类型选择 lbnd 和 ubnd 的值
        lbnd = 0 if dt is bool else np.iinfo(dt).min
        ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
        # 如果 endpoint 为 True，则将 ubnd 减一
        ubnd = ubnd - 1 if endpoint else ubnd

        # 设置随机数生成的大小为 1000
        size = 1000

        # 使用种子 1234 初始化随机数生成器 Generator
        random = Generator(MT19937(1234))

        # 生成标量的随机数数组 scalar
        scalar = random.integers(lbnd, ubnd, size=size, endpoint=endpoint,
                            dtype=dt)

        # 再次使用相同种子初始化随机数生成器 Generator
        random = Generator(MT19937(1234))

        # 生成标量数组的随机数数组 scalar_array
        scalar_array = random.integers([lbnd], [ubnd], size=size,
                                  endpoint=endpoint, dtype=dt)

        # 再次使用相同种子初始化随机数生成器 Generator
        random = Generator(MT19937(1234))

        # 生成大小为 size 的数组 array
        array = random.integers([lbnd] * size, [ubnd] *
                           size, size=size, endpoint=endpoint, dtype=dt)

        # 断言 scalar 和 scalar_array 相等
        assert_array_equal(scalar, scalar_array)

        # 断言 scalar 和 array 相等
        assert_array_equal(scalar, array)
    def test_repeatability(self, endpoint):
        # 对于每种数据类型，测试生成的随机数序列的重复性
        tgt = {'bool':   '053594a9b82d656f967c54869bc6970aa0358cf94ad469c81478459c6a90eee3',
               'int16':  '54de9072b6ee9ff7f20b58329556a46a447a8a29d67db51201bf88baa6e4e5d4',
               'int32':  'd3a0d5efb04542b25ac712e50d21f39ac30f312a5052e9bbb1ad3baa791ac84b',
               'int64':  '14e224389ac4580bfbdccb5697d6190b496f91227cf67df60989de3d546389b1',
               'int8':   '0e203226ff3fbbd1580f15da4621e5f7164d0d8d6b51696dd42d004ece2cbec1',
               'uint16': '54de9072b6ee9ff7f20b58329556a46a447a8a29d67db51201bf88baa6e4e5d4',
               'uint32': 'd3a0d5efb04542b25ac712e50d21f39ac30f312a5052e9bbb1ad3baa791ac84b',
               'uint64': '14e224389ac4580bfbdccb5697d6190b496f91227cf67df60989de3d546389b1',
               'uint8':  '0e203226ff3fbbd1580f15da4621e5f7164d0d8d6b51696dd42d004ece2cbec1'}

        # 遍历所有数据类型，除了布尔型外，使用生成的随机数进行哈希计算，并断言结果与目标哈希值相等
        for dt in self.itype[1:]:
            random = Generator(MT19937(1234))

            # 如果系统字节顺序为小端序，则直接使用随机数生成
            if sys.byteorder == 'little':
                val = random.integers(0, 6 - endpoint, size=1000, endpoint=endpoint,
                                      dtype=dt)
            else:
                # 如果系统字节顺序为大端序，则对生成的随机数进行字节交换后使用
                val = random.integers(0, 6 - endpoint, size=1000, endpoint=endpoint,
                                      dtype=dt).byteswap()

            # 对生成的随机数进行 SHA-256 哈希计算，并转换为十六进制表示
            res = hashlib.sha256(val).hexdigest()
            # 断言计算得到的哈希值与预期的哈希值相等
            assert_(tgt[np.dtype(dt).name] == res)

        # 布尔型的哈希计算与字节顺序无关
        random = Generator(MT19937(1234))
        val = random.integers(0, 2 - endpoint, size=1000, endpoint=endpoint,
                              dtype=bool).view(np.int8)
        res = hashlib.sha256(val).hexdigest()
        assert_(tgt[np.dtype(bool).name] == res)

    def test_repeatability_broadcasting(self, endpoint):
        # 对于每种数据类型，测试广播方式生成的随机数序列的重复性
        for dt in self.itype:
            # 确定每种数据类型的下界和上界
            lbnd = 0 if dt in (bool, np.bool) else np.iinfo(dt).min
            ubnd = 2 if dt in (bool, np.bool) else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd

            random = Generator(MT19937(1234))
            # 生成随机数序列，确保广播方式生成的随机数与单一值方式生成的随机数相等
            val = random.integers(lbnd, ubnd, size=1000, endpoint=endpoint,
                                  dtype=dt)

            random = Generator(MT19937(1234))
            val_bc = random.integers([lbnd] * 1000, ubnd, endpoint=endpoint,
                                     dtype=dt)

            # 断言广播生成的随机数序列与单一值生成的随机数序列相等
            assert_array_equal(val, val_bc)

            random = Generator(MT19937(1234))
            val_bc = random.integers([lbnd] * 1000, [ubnd] * 1000,
                                     endpoint=endpoint, dtype=dt)

            # 断言广播生成的随机数序列与多个上下界值生成的随机数序列相等
            assert_array_equal(val, val_bc)
    @pytest.mark.parametrize(
            'bound, expected',
            [(2**32 - 1, np.array([517043486, 1364798665, 1733884389, 1353720612,
                                   3769704066, 1170797179, 4108474671])),
             (2**32, np.array([517043487, 1364798666, 1733884390, 1353720613,
                               3769704067, 1170797180, 4108474672])),
             (2**32 + 1, np.array([517043487, 1733884390, 3769704068, 4108474673,
                                   1831631863, 1215661561, 3869512430]))]
        )
        # 使用 pytest 的参数化装饰器，定义多个测试参数组合
        def test_repeatability_32bit_boundary(self, bound, expected):
            # 针对每个参数组合执行测试
            for size in [None, len(expected)]:
                # 使用种子为1234的 MT19937 随机数生成器创建 Generator 对象
                random = Generator(MT19937(1234))
                # 生成随机整数数组，范围为0到bound，尺寸为size
                x = random.integers(bound, size=size)
                # 断言生成的随机数数组与期望数组相等，若尺寸为None则与期望数组第一个元素比较
                assert_equal(x, expected if size is not None else expected[0])
    
        # 测试广播情况下的重复性，参数化测试
        def test_repeatability_32bit_boundary_broadcasting(self):
            # 定义期望的多维数组
            desired = np.array([[[1622936284, 3620788691, 1659384060],
                                 [1417365545,  760222891, 1909653332],
                                 [3788118662,  660249498, 4092002593]],
                                [[3625610153, 2979601262, 3844162757],
                                 [ 685800658,  120261497, 2694012896],
                                 [1207779440, 1586594375, 3854335050]],
                                [[3004074748, 2310761796, 3012642217],
                                 [2067714190, 2786677879, 1363865881],
                                 [ 791663441, 1867303284, 2169727960]],
                                [[1939603804, 1250951100,  298950036],
                                 [1040128489, 3791912209, 3317053765],
                                 [3155528714,   61360675, 2305155588]],
                                [[ 817688762, 1335621943, 3288952434],
                                 [1770890872, 1102951817, 1957607470],
                                 [3099996017,  798043451,   48334215]]])
            # 针对每个参数组合执行测试
            for size in [None, (5, 3, 3)]:
                # 使用种子为12345的 MT19937 随机数生成器创建 Generator 对象
                random = Generator(MT19937(12345))
                # 生成随机整数数组，范围为[[-1], [0], [1]]，分别对应[2**32 - 1, 2**32, 2**32 + 1]，尺寸为size
                x = random.integers([[-1], [0], [1]],
                                    [2**32 - 1, 2**32, 2**32 + 1],
                                    size=size)
                # 断言生成的随机数数组与期望数组相等，若尺寸为None则与期望数组第一个元素比较
                assert_array_equal(x, desired if size is not None else desired[0])
    def test_int64_uint64_broadcast_exceptions(self, endpoint):
        # 定义不同数据类型的配置字典，包括 np.uint64 和 np.int64 类型
        configs = {np.uint64: ((0, 2**65), (-1, 2**62), (10, 9), (0, 0)),
                   np.int64: ((0, 2**64), (-(2**64), 2**62), (10, 9), (0, 0),
                              (-2**63-1, -2**63-1))}
        
        # 遍历配置字典中的每个数据类型
        for dtype in configs:
            # 遍历当前数据类型的配置列表
            for config in configs[dtype]:
                # 获取配置中的下界和上界
                low, high = config
                # 根据端点值调整上界
                high = high - endpoint
                
                # 创建包含相同下界值的 numpy 数组
                low_a = np.array([[low]*10])
                # 创建包含相同上界值的 numpy 数组
                high_a = np.array([high] * 10)
                
                # 使用 random.integers 函数测试数值范围异常情况，预期引发 ValueError 异常
                assert_raises(ValueError, random.integers, low, high,
                              endpoint=endpoint, dtype=dtype)
                assert_raises(ValueError, random.integers, low_a, high,
                              endpoint=endpoint, dtype=dtype)
                assert_raises(ValueError, random.integers, low, high_a,
                              endpoint=endpoint, dtype=dtype)
                assert_raises(ValueError, random.integers, low_a, high_a,
                              endpoint=endpoint, dtype=dtype)

                # 创建包含对象类型数据的 numpy 数组
                low_o = np.array([[low]*10], dtype=object)
                high_o = np.array([high] * 10, dtype=object)
                
                # 使用 random.integers 函数测试对象类型数据范围异常情况，预期引发 ValueError 异常
                assert_raises(ValueError, random.integers, low_o, high,
                              endpoint=endpoint, dtype=dtype)
                assert_raises(ValueError, random.integers, low, high_o,
                              endpoint=endpoint, dtype=dtype)
                assert_raises(ValueError, random.integers, low_o, high_o,
                              endpoint=endpoint, dtype=dtype)

    def test_int64_uint64_corner_case(self, endpoint):
        # 在存储于 Numpy 数组中时，`lbnd` 被转换为 np.int64，`ubnd` 被转换为 np.uint64。
        # 检查 `lbnd` >= `ubnd` 以前是通过直接比较完成的，这是不正确的，
        # 因为当 Numpy 尝试比较这两个数字时，它会将它们都转换为 np.float64，
        # 由于 np.float64 无法表示 `ubnd`，导致其向下舍入到 np.iinfo(np.int64).max，
        # 这会导致 ValueError，因为现在 `lbnd` 等于新的 `ubnd`。

        # 定义数据类型为 np.int64
        dt = np.int64
        # 目标值设定为 np.int64 的最大值
        tgt = np.iinfo(np.int64).max
        # 设置下界为 np.int64 的最大值
        lbnd = np.int64(np.iinfo(np.int64).max)
        # 设置上界为 np.int64 的最大值加一再减去端点值
        ubnd = np.uint64(np.iinfo(np.int64).max + 1 - endpoint)

        # 现在这些函数调用都不应该生成 ValueError 异常
        actual = random.integers(lbnd, ubnd, endpoint=endpoint, dtype=dt)
        # 断言实际结果等于目标值
        assert_equal(actual, tgt)
    def test_respect_dtype_singleton(self, endpoint):
        # See gh-7203
        # 遍历self.itype中的数据类型
        for dt in self.itype:
            # 如果dt是bool类型，设置下限为0，否则使用np.iinfo(dt)的最小值
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            # 如果dt是bool类型，设置上限为2，否则使用np.iinfo(dt)的最大值加1
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            # 如果endpoint为True，上限减1；否则保持不变
            ubnd = ubnd - 1 if endpoint else ubnd
            # 如果dt是bool类型，设置dt为np.bool，否则保持不变
            dt = np.bool if dt is bool else dt

            # 使用self.rfunc生成一个样本，指定下限lbnd，上限ubnd，数据类型为dt
            sample = self.rfunc(lbnd, ubnd, endpoint=endpoint, dtype=dt)
            # 断言生成的样本的数据类型为dt
            assert_equal(sample.dtype, dt)

        # 再次遍历bool和int两种数据类型
        for dt in (bool, int):
            # 如果dt是bool类型，设置下限为0，否则使用np.iinfo(dt)的最小值
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            # 如果dt是bool类型，设置上限为2，否则使用np.iinfo(dt)的最大值加1
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            # 如果endpoint为True，上限减1；否则保持不变
            ubnd = ubnd - 1 if endpoint else ubnd

            # gh-7284: 确保我们得到Python数据类型
            # 使用self.rfunc生成一个样本，指定下限lbnd，上限ubnd，数据类型为dt
            sample = self.rfunc(lbnd, ubnd, endpoint=endpoint, dtype=dt)
            # 断言生成的样本没有dtype属性
            assert not hasattr(sample, 'dtype')
            # 断言生成的样本的类型为dt
            assert_equal(type(sample), dt)

    def test_respect_dtype_array(self, endpoint):
        # See gh-7203
        # 遍历self.itype中的数据类型
        for dt in self.itype:
            # 如果dt是bool类型，设置下限为0，否则使用np.iinfo(dt)的最小值
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            # 如果dt是bool类型，设置上限为2，否则使用np.iinfo(dt)的最大值加1
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            # 如果endpoint为True，上限减1；否则保持不变
            ubnd = ubnd - 1 if endpoint else ubnd
            # 如果dt是bool类型，设置dt为np.bool，否则保持不变
            dt = np.bool if dt is bool else dt

            # 使用self.rfunc生成一个样本，指定下限为[lbnd]，上限为[ubnd]，数据类型为dt
            sample = self.rfunc([lbnd], [ubnd], endpoint=endpoint, dtype=dt)
            # 断言生成的样本的数据类型为dt
            assert_equal(sample.dtype, dt)
            # 使用self.rfunc生成一个样本，指定下限为[lbnd, lbnd]，上限为[ubnd, ubnd]，数据类型为dt
            sample = self.rfunc([lbnd] * 2, [ubnd] * 2, endpoint=endpoint,
                                dtype=dt)
            # 断言生成的样本的数据类型为dt
            assert_equal(sample.dtype, dt)

    def test_zero_size(self, endpoint):
        # See gh-7203
        # 遍历self.itype中的数据类型
        for dt in self.itype:
            # 使用self.rfunc生成一个大小为(3, 0, 4)的样本，数据类型为dt
            sample = self.rfunc(0, 0, (3, 0, 4), endpoint=endpoint, dtype=dt)
            # 断言生成的样本的形状为(3, 0, 4)
            assert sample.shape == (3, 0, 4)
            # 断言生成的样本的数据类型为dt
            assert sample.dtype == dt
            # 使用self.rfunc生成一个大小为(0,)的样本，下限为0，上限为-10，数据类型为dt
            assert self.rfunc(0, -10, 0, endpoint=endpoint,
                              dtype=dt).shape == (0,)
            # 使用random.integers生成一个大小为(3, 0, 4)的样本，下限为0，上限为0，数据类型为dt
            assert_equal(random.integers(0, 0, size=(3, 0, 4)).shape,
                         (3, 0, 4))
            # 使用random.integers生成一个大小为(0,)的样本，下限为0，上限为-10，数据类型为dt
            assert_equal(random.integers(0, -10, size=0).shape, (0,))
            # 使用random.integers生成一个大小为(0,)的样本，下限为10，上限为10，数据类型为dt
            assert_equal(random.integers(10, 10, size=0).shape, (0,))

    def test_error_byteorder(self):
        # 使用'<i4'或'>i4'根据系统字节顺序设置other_byteord_dt
        other_byteord_dt = '<i4' if sys.byteorder == 'big' else '>i4'
        # 断言调用random.integers时会引发ValueError异常
        with pytest.raises(ValueError):
            random.integers(0, 200, size=10, dtype=other_byteord_dt)

    # chi2max is the maximum acceptable chi-squared value.
    @pytest.mark.slow
    @pytest.mark.parametrize('sample_size,high,dtype,chi2max',
        [(5000000, 5, np.int8, 125.0),          # p-value ~4.6e-25
         (5000000, 7, np.uint8, 150.0),         # p-value ~7.7e-30
         (10000000, 2500, np.int16, 3300.0),    # p-value ~3.0e-25
         (50000000, 5000, np.uint16, 6500.0),   # p-value ~3.5e-25
        ])
    # 定义一个测试函数，用于测试生成的随机整数数组的卡方检验
    def test_integers_small_dtype_chisquared(self, sample_size, high,
                                             dtype, chi2max):
        # 对于问题报告 gh-14774 的回归测试。
        
        # 生成指定范围内的随机整数数组
        samples = random.integers(high, size=sample_size, dtype=dtype)

        # 使用 NumPy 的 unique 函数获取数组中的唯一值和它们的计数
        values, counts = np.unique(samples, return_counts=True)
        
        # 计算期望的频率，即每个值出现的期望次数
        expected = sample_size / high
        
        # 计算卡方值，用于衡量观察值与期望值的偏差程度
        chi2 = ((counts - expected)**2 / expected).sum()
        
        # 断言卡方值小于预设的最大卡方值，用于判断是否符合期望的分布
        assert chi2 < chi2max
class TestRandomDist:
    # 确保随机分布针对给定种子返回正确的值

    def setup_method(self):
        # 设置测试方法的初始种子值
        self.seed = 1234567890

    def test_integers(self):
        # 测试生成整数的方法
        random = Generator(MT19937(self.seed))
        # 生成一个指定范围的整数数组
        actual = random.integers(-99, 99, size=(3, 2))
        # 预期的结果数组
        desired = np.array([[-80, -56], [41, 37], [-83, -16]])
        # 断言数组是否相等
        assert_array_equal(actual, desired)

    def test_integers_masked(self):
        # 测试使用掩码拒绝抽样算法生成 uint32 类型数组
        random = Generator(MT19937(self.seed))
        actual = random.integers(0, 99, size=(3, 2), dtype=np.uint32)
        desired = np.array([[9, 21], [70, 68], [8, 41]], dtype=np.uint32)
        assert_array_equal(actual, desired)

    def test_integers_closed(self):
        # 测试生成包括端点的整数数组
        random = Generator(MT19937(self.seed))
        actual = random.integers(-99, 99, size=(3, 2), endpoint=True)
        desired = np.array([[-80, -56], [ 41, 38], [-83, -15]])
        assert_array_equal(actual, desired)

    def test_integers_max_int(self):
        # 测试整数生成方法是否能够生成最大的 Python int 值
        actual = random.integers(np.iinfo('l').max, np.iinfo('l').max,
                                 endpoint=True)
        desired = np.iinfo('l').max
        assert_equal(actual, desired)

    def test_random(self):
        random = Generator(MT19937(self.seed))
        actual = random.random((3, 2))
        desired = np.array([[0.096999199829214, 0.707517457682192],
                            [0.084364834598269, 0.767731206553125],
                            [0.665069021359413, 0.715487190596693]])
        assert_array_almost_equal(actual, desired, decimal=15)

        random = Generator(MT19937(self.seed))
        actual = random.random()
        assert_array_almost_equal(actual, desired[0, 0], decimal=15)

    def test_random_float(self):
        random = Generator(MT19937(self.seed))
        actual = random.random((3, 2))
        desired = np.array([[0.0969992 , 0.70751746],
                            [0.08436483, 0.76773121],
                            [0.66506902, 0.71548719]])
        assert_array_almost_equal(actual, desired, decimal=7)

    def test_random_float_scalar(self):
        random = Generator(MT19937(self.seed))
        actual = random.random(dtype=np.float32)
        desired = 0.0969992
        assert_array_almost_equal(actual, desired, decimal=7)

    @pytest.mark.parametrize('dtype, uint_view_type',
                             [(np.float32, np.uint32),
                              (np.float64, np.uint64)])
    # 测试函数：验证随机数最低有效位的分布是否符合预期
    def test_random_distribution_of_lsb(self, dtype, uint_view_type):
        # 使用种子初始化 Mersenne Twister 伪随机数生成器
        random = Generator(MT19937(self.seed))
        # 生成指定数据类型的随机数样本
        sample = random.random(100000, dtype=dtype)
        # 将样本视图转换为指定无符号整数视图类型，并计算最低有效位为1的个数
        num_ones_in_lsb = np.count_nonzero(sample.view(uint_view_type) & 1)
        # 断言：最低有效位为1的数量应在 24100 到 25900 之间
        # 这个范围内的概率小于 5e-11
        assert 24100 < num_ones_in_lsb < 25900

    # 测试函数：验证当传入不支持的数据类型时是否引发 TypeError 异常
    def test_random_unsupported_type(self):
        assert_raises(TypeError, random.random, dtype='int32')

    # 测试函数：验证从指定元素中均匀选择并替换
    def test_choice_uniform_replace(self):
        random = Generator(MT19937(self.seed))
        # 从0到3之间均匀选择4个元素，并与期望的数组进行比较
        actual = random.choice(4, 4)
        desired = np.array([0, 0, 2, 2], dtype=np.int64)
        assert_array_equal(actual, desired)

    # 测试函数：验证从指定元素中非均匀选择并替换
    def test_choice_nonuniform_replace(self):
        random = Generator(MT19937(self.seed))
        # 从0到3之间非均匀选择4个元素，并与期望的数组进行比较
        actual = random.choice(4, 4, p=[0.4, 0.4, 0.1, 0.1])
        desired = np.array([0, 1, 0, 1], dtype=np.int64)
        assert_array_equal(actual, desired)

    # 测试函数：验证从指定元素中均匀选择且不替换
    def test_choice_uniform_noreplace(self):
        random = Generator(MT19937(self.seed))
        # 从0到3之间均匀选择3个元素且不替换，并与期望的数组进行比较
        actual = random.choice(4, 3, replace=False)
        desired = np.array([2, 0, 3], dtype=np.int64)
        assert_array_equal(actual, desired)
        # 从0到3之间均匀选择4个元素且不替换且不打乱顺序，并与期望的数组进行比较
        actual = random.choice(4, 4, replace=False, shuffle=False)
        desired = np.arange(4, dtype=np.int64)
        assert_array_equal(actual, desired)

    # 测试函数：验证从指定元素中非均匀选择且不替换
    def test_choice_nonuniform_noreplace(self):
        random = Generator(MT19937(self.seed))
        # 从0到3之间非均匀选择3个元素且不替换，并与期望的数组进行比较
        actual = random.choice(4, 3, replace=False, p=[0.1, 0.3, 0.5, 0.1])
        desired = np.array([0, 2, 3], dtype=np.int64)
        assert_array_equal(actual, desired)

    # 测试函数：验证从非整数数组中选择指定数量的元素
    def test_choice_noninteger(self):
        random = Generator(MT19937(self.seed))
        # 从 ['a', 'b', 'c', 'd'] 中选择4个元素，并与期望的数组进行比较
        actual = random.choice(['a', 'b', 'c', 'd'], 4)
        desired = np.array(['a', 'a', 'c', 'c'])
        assert_array_equal(actual, desired)

    # 测试函数：验证从多维数组中选择元素（默认轴）
    def test_choice_multidimensional_default_axis(self):
        random = Generator(MT19937(self.seed))
        # 从二维数组中选择3个元素，并与期望的数组进行比较
        actual = random.choice([[0, 1], [2, 3], [4, 5], [6, 7]], 3)
        desired = np.array([[0, 1], [0, 1], [4, 5]])
        assert_array_equal(actual, desired)

    # 测试函数：验证从多维数组中选择元素（自定义轴）
    def test_choice_multidimensional_custom_axis(self):
        random = Generator(MT19937(self.seed))
        # 从二维数组中选择每行的第1列元素，并与期望的数组进行比较
        actual = random.choice([[0, 1], [2, 3], [4, 5], [6, 7]], 1, axis=1)
        desired = np.array([[0], [2], [4], [6]])
        assert_array_equal(actual, desired)
    # 定义一个测试方法来测试随机选择函数的异常情况
    def test_choice_exceptions(self):
        # 将 random.choice 赋值给 sample 变量，便于后续调用
        sample = random.choice
        # 断言当从范围为 [-1, 3) 的整数集合中选择时会引发 ValueError 异常
        assert_raises(ValueError, sample, -1, 3)
        # 断言当从范围为 [3., 3) 的浮点数集合中选择时会引发 ValueError 异常
        assert_raises(ValueError, sample, 3., 3)
        # 断言当从空列表中选择时会引发 ValueError 异常
        assert_raises(ValueError, sample, [], 3)
        # 断言当从包含四个元素的列表中选择，同时提供了概率分布但概率分布不合法时会引发 ValueError 异常
        assert_raises(ValueError, sample, [1, 2, 3, 4], 3,
                      p=[[0.25, 0.25], [0.25, 0.25]])
        # 断言当从包含两个元素的列表中选择，同时提供了不合法的概率分布时会引发 ValueError 异常
        assert_raises(ValueError, sample, [1, 2], 3, p=[0.4, 0.4, 0.2])
        # 断言当从包含两个元素的列表中选择，同时提供了不合法的概率分布时会引发 ValueError 异常
        assert_raises(ValueError, sample, [1, 2], 3, p=[1.1, -0.1])
        # 断言当从包含两个元素的列表中选择，同时提供了不合法的概率分布时会引发 ValueError 异常
        assert_raises(ValueError, sample, [1, 2], 3, p=[0.4, 0.4])
        # 断言当从包含三个元素的列表中选择，但指定不允许替换时会引发 ValueError 异常
        assert_raises(ValueError, sample, [1, 2, 3], 4, replace=False)
        # 断言当从包含三个元素的列表中选择，但指定选择数量超出范围时会引发 ValueError 异常
        # （gh-13087 表示 GitHub 上的 issue 编号）
        assert_raises(ValueError, sample, [1, 2, 3], -2, replace=False)
        # 断言当从包含三个元素的列表中选择，但指定选择数量为元组时会引发 ValueError 异常
        assert_raises(ValueError, sample, [1, 2, 3], (-1,), replace=False)
        # 断言当从包含三个元素的列表中选择，但指定选择数量为元组且超出范围时会引发 ValueError 异常
        assert_raises(ValueError, sample, [1, 2, 3], (-1, 1), replace=False)
        # 断言当从包含三个元素的列表中选择，但提供了不合法的概率分布时会引发 ValueError 异常
        assert_raises(ValueError, sample, [1, 2, 3], 2,
                      replace=False, p=[1, 0, 0])
    def test_choice_return_shape(self):
        p = [0.1, 0.9]
        # Check scalar
        # 检查标量
        assert_(np.isscalar(random.choice(2, replace=True)))
        assert_(np.isscalar(random.choice(2, replace=False)))
        assert_(np.isscalar(random.choice(2, replace=True, p=p)))
        assert_(np.isscalar(random.choice(2, replace=False, p=p)))
        assert_(np.isscalar(random.choice([1, 2], replace=True)))
        assert_(random.choice([None], replace=True) is None)
        a = np.array([1, 2])
        arr = np.empty(1, dtype=object)
        arr[0] = a
        assert_(random.choice(arr, replace=True) is a)

        # Check 0-d array
        # 检查0维数组
        s = tuple()
        assert_(not np.isscalar(random.choice(2, s, replace=True)))
        assert_(not np.isscalar(random.choice(2, s, replace=False)))
        assert_(not np.isscalar(random.choice(2, s, replace=True, p=p)))
        assert_(not np.isscalar(random.choice(2, s, replace=False, p=p)))
        assert_(not np.isscalar(random.choice([1, 2], s, replace=True)))
        assert_(random.choice([None], s, replace=True).ndim == 0)
        a = np.array([1, 2])
        arr = np.empty(1, dtype=object)
        arr[0] = a
        assert_(random.choice(arr, s, replace=True).item() is a)

        # Check multi dimensional array
        # 检查多维数组
        s = (2, 3)
        p = [0.1, 0.1, 0.1, 0.1, 0.4, 0.2]
        assert_equal(random.choice(6, s, replace=True).shape, s)
        assert_equal(random.choice(6, s, replace=False).shape, s)
        assert_equal(random.choice(6, s, replace=True, p=p).shape, s)
        assert_equal(random.choice(6, s, replace=False, p=p).shape, s)
        assert_equal(random.choice(np.arange(6), s, replace=True).shape, s)

        # Check zero-size
        # 检查大小为0的情况
        assert_equal(random.integers(0, 0, size=(3, 0, 4)).shape, (3, 0, 4))
        assert_equal(random.integers(0, -10, size=0).shape, (0,))
        assert_equal(random.integers(10, 10, size=0).shape, (0,))
        assert_equal(random.choice(0, size=0).shape, (0,))
        assert_equal(random.choice([], size=(0,)).shape, (0,))
        assert_equal(random.choice(['a', 'b'], size=(3, 0, 4)).shape,
                     (3, 0, 4))
        assert_raises(ValueError, random.choice, [], 10)

    def test_choice_nan_probabilities(self):
        a = np.array([42, 1, 2])
        p = [None, None, None]
        assert_raises(ValueError, random.choice, a, p=p)

    def test_choice_p_non_contiguous(self):
        p = np.ones(10) / 5
        p[1::2] = 3.0
        random = Generator(MT19937(self.seed))
        non_contig = random.choice(5, 3, p=p[::2])
        random = Generator(MT19937(self.seed))
        contig = random.choice(5, 3, p=np.ascontiguousarray(p[::2]))
        assert_array_equal(non_contig, contig)
    def test_choice_return_type(self):
        # 用例编号 gh 9867
        p = np.ones(4) / 4.
        # 从范围为 0 到 3 的整数中选择 2 个数，返回结果的数据类型为 np.int64
        actual = random.choice(4, 2)
        assert actual.dtype == np.int64
        # 从范围为 0 到 3 的整数中选择 2 个数，不允许重复选择，返回结果的数据类型为 np.int64
        actual = random.choice(4, 2, replace=False)
        assert actual.dtype == np.int64
        # 从范围为 0 到 3 的整数中选择 2 个数，按给定概率分布 p 进行选择，返回结果的数据类型为 np.int64
        actual = random.choice(4, 2, p=p)
        assert actual.dtype == np.int64
        # 从范围为 0 到 3 的整数中选择 2 个数，按给定概率分布 p 进行选择，不允许重复选择，返回结果的数据类型为 np.int64
        actual = random.choice(4, 2, p=p, replace=False)
        assert actual.dtype == np.int64

    def test_choice_large_sample(self):
        # 预期结果的哈希值
        choice_hash = '4266599d12bfcfb815213303432341c06b4349f5455890446578877bb322e222'
        # 创建一个随机数生成器对象 random
        random = Generator(MT19937(self.seed))
        # 从 0 到 9999 的整数中选择 5000 个数，不允许重复选择
        actual = random.choice(10000, 5000, replace=False)
        # 如果系统的字节顺序不是小端序，则对结果进行字节交换
        if sys.byteorder != 'little':
            actual = actual.byteswap()
        # 计算结果的 SHA-256 哈希值，并以十六进制表示
        res = hashlib.sha256(actual.view(np.int8)).hexdigest()
        # 断言预期的哈希值与计算得到的哈希值相等
        assert_(choice_hash == res)

    def test_choice_array_size_empty_tuple(self):
        # 创建一个随机数生成器对象 random
        random = Generator(MT19937(self.seed))
        # 对于空的元组大小，从给定的列表中进行选择，期望结果为 np.array([1])
        assert_array_equal(random.choice([1, 2, 3], size=()), np.array(1),
                           strict=True)
        # 对于空的元组大小，从给定的列表中进行选择，期望结果为 [1, 2, 3]
        assert_array_equal(random.choice([[1, 2, 3]], size=()), [1, 2, 3])
        # 对于空的元组大小，从给定的列表中进行选择，期望结果为 [1]
        assert_array_equal(random.choice([[1]], size=()), [1], strict=True)
        # 对于空的元组大小，从给定的列表中进行选择，沿指定的轴（axis=1）选择，期望结果为 [1]
        assert_array_equal(random.choice([[1]], size=(), axis=1), [1],
                           strict=True)

    def test_bytes(self):
        # 创建一个随机数生成器对象 random
        random = Generator(MT19937(self.seed))
        # 生成长度为 10 的随机字节序列
        actual = random.bytes(10)
        # 预期的字节序列
        desired = b'\x86\xf0\xd4\x18\xe1\x81\t8%\xdd'
        # 断言生成的随机字节序列与预期的字节序列相等
        assert_equal(actual, desired)

    def test_shuffle(self):
        # 测试列表、数组（不同的数据类型）、多维版本的随机重排，包括 C 连续和非 C 连续的情况
        for conv in [lambda x: np.array([]),
                     lambda x: x,
                     lambda x: np.asarray(x).astype(np.int8),
                     lambda x: np.asarray(x).astype(np.float32),
                     lambda x: np.asarray(x).astype(np.complex64),
                     lambda x: np.asarray(x).astype(object),
                     lambda x: [(i, i) for i in x],
                     lambda x: np.asarray([[i, i] for i in x]),
                     lambda x: np.vstack([x, x]).T,
                     # gh-11442
                     lambda x: (np.asarray([(i, i) for i in x],
                                           [("a", int), ("b", int)])
                                .view(np.recarray)),
                     # gh-4270
                     lambda x: np.asarray([(i, i) for i in x],
                                          [("a", object, (1,)),
                                           ("b", np.int32, (1,))])]:
            # 创建一个随机数生成器对象 random
            random = Generator(MT19937(self.seed))
            # 转换给定的列表或数组 x，并进行随机重排
            alist = conv([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
            random.shuffle(alist)
            # 获取重排后的结果
            actual = alist
            # 预期的重排结果
            desired = conv([4, 1, 9, 8, 0, 5, 3, 6, 2, 7])
            # 断言重排后的结果与预期的结果相等
            assert_array_equal(actual, desired)
    def test_shuffle_custom_axis(self):
        # 使用确定的种子创建随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 创建一个4x4的数组，其中元素为0到15
        actual = np.arange(16).reshape((4, 4))
        # 按照指定的轴（第二个轴）对数组进行原地洗牌操作
        random.shuffle(actual, axis=1)
        # 预期的结果，按照第二个轴（列）进行了重新排列
        desired = np.array([[ 0,  3,  1,  2],
                            [ 4,  7,  5,  6],
                            [ 8, 11,  9, 10],
                            [12, 15, 13, 14]])
        # 断言洗牌后的数组与期望的结果数组相等
        assert_array_equal(actual, desired)
        # 重新使用相同的种子创建随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 重新创建一个4x4的数组，其中元素为0到15
        actual = np.arange(16).reshape((4, 4))
        # 按照指定的轴（最后一个轴）对数组进行原地洗牌操作
        random.shuffle(actual, axis=-1)
        # 断言洗牌后的数组与期望的结果数组相等
        assert_array_equal(actual, desired)

    def test_shuffle_custom_axis_empty(self):
        # 使用确定的种子创建随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 创建一个空的数组，形状为(0, 6)
        desired = np.array([]).reshape((0, 6))
        # 对于每一个轴（0和1），循环执行以下操作
        for axis in (0, 1):
            # 创建一个空的数组，形状为(0, 6)
            actual = np.array([]).reshape((0, 6))
            # 按照指定的轴对数组进行原地洗牌操作
            random.shuffle(actual, axis=axis)
            # 断言洗牌后的数组与期望的结果数组相等
            assert_array_equal(actual, desired)

    def test_shuffle_axis_nonsquare(self):
        # 创建一个2x10的数组，并初始化其元素为0到19
        y1 = np.arange(20).reshape(2, 10)
        # 创建y1的副本y2
        y2 = y1.copy()
        # 使用确定的种子创建随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 按照指定的轴（第二个轴）对数组进行原地洗牌操作
        random.shuffle(y1, axis=1)
        # 重新使用相同的种子创建随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 转置y2数组，并按照最后一个轴对其进行原地洗牌操作
        random.shuffle(y2.T)
        # 断言洗牌后的y1数组与y2数组相等
        assert_array_equal(y1, y2)

    def test_shuffle_masked(self):
        # 创建一个5x4的掩码数组，元素为0到19取模3再减1，掩码值为-1
        a = np.ma.masked_values(np.reshape(range(20), (5, 4)) % 3 - 1, -1)
        # 创建一个1维掩码数组，元素为0到19取模3再减1，掩码值为-1
        b = np.ma.masked_values(np.arange(20) % 3 - 1, -1)
        # 复制a和b的原始数组
        a_orig = a.copy()
        b_orig = b.copy()
        # 对于范围内的50次迭代
        for i in range(50):
            # 使用随机数生成器对象对数组a进行原地洗牌操作
            random.shuffle(a)
            # 断言洗牌后的a.data（非掩码部分）与原始a.data（非掩码部分）的排序后结果相等
            assert_equal(
                sorted(a.data[~a.mask]), sorted(a_orig.data[~a_orig.mask]))
            # 使用随机数生成器对象对数组b进行原地洗牌操作
            random.shuffle(b)
            # 断言洗牌后的b.data（非掩码部分）与原始b.data（非掩码部分）的排序后结果相等
            assert_equal(
                sorted(b.data[~b.mask]), sorted(b_orig.data[~b_orig.mask]))

    def test_shuffle_exceptions(self):
        # 使用确定的种子创建随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 创建一个包含0到9的数组
        arr = np.arange(10)
        # 断言调用random.shuffle(arr, 1)会抛出AxisError异常
        assert_raises(AxisError, random.shuffle, arr, 1)
        # 创建一个形状为(3, 3)的数组，元素为0到8
        arr = np.arange(9).reshape((3, 3))
        # 断言调用random.shuffle(arr, 3)会抛出AxisError异常
        assert_raises(AxisError, random.shuffle, arr, 3)
        # 断言调用random.shuffle(arr, slice(1, 2, None))会抛出TypeError异常
        assert_raises(TypeError, random.shuffle, arr, slice(1, 2, None))
        # 创建一个二维列表
        arr = [[1, 2, 3], [4, 5, 6]]
        # 断言调用random.shuffle(arr, 1)会抛出NotImplementedError异常
        assert_raises(NotImplementedError, random.shuffle, arr, 1)

        # 创建一个标量数组
        arr = np.array(3)
        # 断言调用random.shuffle(arr)会抛出TypeError异常
        assert_raises(TypeError, random.shuffle, arr)
        # 创建一个形状为(3, 2)的数组，元素为1
        arr = np.ones((3, 2))
        # 断言调用random.shuffle(arr, 2)会抛出AxisError异常
        assert_raises(AxisError, random.shuffle, arr, 2)

    def test_shuffle_not_writeable(self):
        # 使用确定的种子创建随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 创建一个包含5个零的数组
        a = np.zeros(5)
        # 将数组a设置为只读
        a.flags.writeable = False
        # 使用pytest断言调用random.shuffle(a)会抛出ValueError异常，并且异常信息包含'read-only'
        with pytest.raises(ValueError, match='read-only'):
            random.shuffle(a)
    # 定义一个测试函数，用于测试随机排列功能
    def test_permutation(self):
        # 使用给定的种子初始化随机数生成器
        random = Generator(MT19937(self.seed))
        # 创建一个整数列表
        alist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        # 进行随机排列操作
        actual = random.permutation(alist)
        # 期望的结果列表
        desired = [4, 1, 9, 8, 0, 5, 3, 6, 2, 7]
        # 断言实际结果与期望结果相等
        assert_array_equal(actual, desired)

        # 使用相同的随机数生成器再次进行测试
        random = Generator(MT19937(self.seed))
        # 创建一个二维数组，至少是二维的版本
        arr_2d = np.atleast_2d([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]).T
        # 进行二维数组的随机排列操作
        actual = random.permutation(arr_2d)
        # 断言实际结果与期望结果的二维版本相等
        assert_array_equal(actual, np.atleast_2d(desired).T)

        # 测试字符串类型的非法输入
        bad_x_str = "abcd"
        # 断言调用随机排列函数时会抛出 AxisError 异常
        assert_raises(AxisError, random.permutation, bad_x_str)

        # 测试浮点数类型的非法输入
        bad_x_float = 1.2
        # 断言调用随机排列函数时会抛出 AxisError 异常
        assert_raises(AxisError, random.permutation, bad_x_float)

        # 使用相同的随机数生成器再次进行测试
        random = Generator(MT19937(self.seed))
        # 创建一个整数值
        integer_val = 10
        # 期望的结果列表
        desired = [3, 0, 8, 7, 9, 4, 2, 5, 1, 6]
        # 对整数值进行随机排列操作
        actual = random.permutation(integer_val)
        # 断言实际结果与期望结果相等
        assert_array_equal(actual, desired)

    # 定义一个测试函数，用于测试自定义轴的随机排列功能
    def test_permutation_custom_axis(self):
        # 创建一个 4x4 的二维数组
        a = np.arange(16).reshape((4, 4))
        # 期望的结果二维数组
        desired = np.array([[ 0,  3,  1,  2],
                            [ 4,  7,  5,  6],
                            [ 8, 11,  9, 10],
                            [12, 15, 13, 14]])
        # 使用给定的种子初始化随机数生成器
        random = Generator(MT19937(self.seed))
        # 对二维数组按指定轴进行随机排列操作
        actual = random.permutation(a, axis=1)
        # 断言实际结果与期望结果相等
        assert_array_equal(actual, desired)
        # 使用相同的随机数生成器再次进行测试
        random = Generator(MT19937(self.seed))
        # 对二维数组按指定轴进行随机排列操作
        actual = random.permutation(a, axis=-1)
        # 断言实际结果与期望结果相等
        assert_array_equal(actual, desired)

    # 定义一个测试函数，用于测试随机排列函数抛出异常的情况
    def test_permutation_exceptions(self):
        # 使用给定的种子初始化随机数生成器
        random = Generator(MT19937(self.seed))
        # 创建一个一维数组
        arr = np.arange(10)
        # 断言调用随机排列函数时会抛出 AxisError 异常
        assert_raises(AxisError, random.permutation, arr, 1)
        # 创建一个 3x3 的二维数组
        arr = np.arange(9).reshape((3, 3))
        # 断言调用随机排列函数时会抛出 AxisError 异常
        assert_raises(AxisError, random.permutation, arr, 3)
        # 断言调用随机排列函数时会抛出 TypeError 异常
        assert_raises(TypeError, random.permutation, arr, slice(1, 2, None))

    # 使用 pytest 参数化装饰器定义多个测试用例，用于测试随机排列函数的不同情况
    @pytest.mark.parametrize("dtype", [int, object])
    @pytest.mark.parametrize("axis, expected",
                             [(None, np.array([[3, 7, 0, 9, 10, 11],
                                               [8, 4, 2, 5,  1,  6]])),
                              (0, np.array([[6, 1, 2, 9, 10, 11],
                                            [0, 7, 8, 3,  4,  5]])),
                              (1, np.array([[ 5, 3,  4, 0, 2, 1],
                                            [11, 9, 10, 6, 8, 7]]))])
    # 定义一个测试函数，用于测试随机排列函数的不同参数和期望结果
    def test_permuted(self, dtype, axis, expected):
        # 使用给定的种子初始化随机数生成器
        random = Generator(MT19937(self.seed))
        # 创建一个 2x6 的二维数组，并转换为指定的数据类型
        x = np.arange(12).reshape(2, 6).astype(dtype)
        # 在原地进行随机排列操作
        random.permuted(x, axis=axis, out=x)
        # 断言实际结果与期望结果相等
        assert_array_equal(x, expected)

        # 使用相同的随机数生成器再次进行测试
        random = Generator(MT19937(self.seed))
        # 创建一个 2x6 的二维数组，并转换为指定的数据类型
        x = np.arange(12).reshape(2, 6).astype(dtype)
        # 返回随机排列后的新数组
        y = random.permuted(x, axis=axis)
        # 断言返回数组的数据类型与期望相同
        assert y.dtype == dtype
        # 断言返回数组与期望结果相等
        assert_array_equal(y, expected)
    # 测试随机置换函数 permuted 的使用，使用给定的种子初始化随机数生成器
    def test_permuted_with_strides(self):
        random = Generator(MT19937(self.seed))
        # 创建一个二维数组 x0，形状为 (2, 11)，并复制给 x1
        x0 = np.arange(22).reshape(2, 11)
        x1 = x0.copy()
        # 使用步长为 3 对 x0 进行切片得到 x
        x = x0[:, ::3]
        # 调用 permuted 函数对 x 进行随机置换，指定轴为 1，并将结果存回 x 中
        y = random.permuted(x, axis=1, out=x)
        # 预期的结果数组
        expected = np.array([[0, 9, 3, 6],
                             [14, 20, 11, 17]])
        # 断言 y 与预期结果 expected 相等
        assert_array_equal(y, expected)
        # 将预期的结果 expected 赋值回 x1 的对应位置
        x1[:, ::3] = expected
        # 断言 x1 是否与原始 x0 在相同位置上被正确修改
        assert_array_equal(x1, x0)

    # 测试在空数组上调用 permuted 函数的情况
    def test_permuted_empty(self):
        # 调用 permuted 函数传入空数组，并断言其返回结果为空数组
        y = random.permuted([])
        assert_array_equal(y, [])

    # 参数化测试：测试使用不正确形状的输出数组 outshape 调用 permuted 函数时的情况
    @pytest.mark.parametrize('outshape', [(2, 3), 5])
    def test_permuted_out_with_wrong_shape(self, outshape):
        # 创建一个数组 a，并为其指定 dtype
        a = np.array([1, 2, 3])
        # 创建一个与 outshape 形状相同且 dtype 为 a 的 dtype 的全零数组 out
        out = np.zeros(outshape, dtype=a.dtype)
        # 使用 permuted 函数调用 a，并断言会引发 ValueError 异常，匹配 'same shape'
        with pytest.raises(ValueError, match='same shape'):
            random.permuted(a, out=out)

    # 测试在使用不正确类型的输出数组 out 时调用 permuted 函数的情况
    def test_permuted_out_with_wrong_type(self):
        # 创建一个形状为 (3, 5)，dtype 为 np.int32 的全零数组 out
        out = np.zeros((3, 5), dtype=np.int32)
        # 创建一个形状为 (3, 5) 的全一数组 x
        x = np.ones((3, 5))
        # 使用 permuted 函数调用 x，指定轴为 1，并传入不正确类型的 out 数组，
        # 并断言会引发 TypeError 异常，匹配 'Cannot cast'
        with pytest.raises(TypeError, match='Cannot cast'):
            random.permuted(x, axis=1, out=out)

    # 测试在不可写入数组上调用 permuted 函数的情况
    def test_permuted_not_writeable(self):
        # 创建一个形状为 (2, 5) 的全零数组 x
        x = np.zeros((2, 5))
        # 将 x 的写入标志设置为 False，使其变为只读
        x.flags.writeable = False
        # 使用 permuted 函数调用 x，指定轴为 1，并断言会引发 ValueError 异常，匹配 'read-only'
        with pytest.raises(ValueError, match='read-only'):
            random.permuted(x, axis=1, out=x)

    # 测试 beta 分布函数的使用
    def test_beta(self):
        # 使用给定的种子初始化随机数生成器
        random = Generator(MT19937(self.seed))
        # 调用 beta 函数生成一个形状为 (3, 2) 的随机数组 actual
        actual = random.beta(.1, .9, size=(3, 2))
        # 预期的结果数组
        desired = np.array(
            [[1.083029353267698e-10, 2.449965303168024e-11],
             [2.397085162969853e-02, 3.590779671820755e-08],
             [2.830254190078299e-04, 1.744709918330393e-01]])
        # 断言 actual 与预期结果 desired 在小数点后 15 位精度下相等
        assert_array_almost_equal(actual, desired, decimal=15)

    # 测试二项分布函数的使用
    def test_binomial(self):
        # 使用给定的种子初始化随机数生成器
        random = Generator(MT19937(self.seed))
        # 调用 binomial 函数生成一个形状为 (3, 2) 的随机数组 actual
        actual = random.binomial(100.123, .456, size=(3, 2))
        # 预期的结果数组
        desired = np.array([[42, 41],
                            [42, 48],
                            [44, 50]])
        # 断言 actual 与预期结果 desired 相等
        assert_array_equal(actual, desired)

        # 使用相同的种子再次初始化随机数生成器
        random = Generator(MT19937(self.seed))
        # 调用 binomial 函数生成一个标量结果 actual
        actual = random.binomial(100.123, .456)
        # 预期的结果标量
        desired = 42
        # 断言 actual 与预期结果 desired 相等
        assert_array_equal(actual, desired)

    # 测试卡方分布函数的使用
    def test_chisquare(self):
        # 使用给定的种子初始化随机数生成器
        random = Generator(MT19937(self.seed))
        # 调用 chisquare 函数生成一个形状为 (3, 2) 的随机数组 actual
        actual = random.chisquare(50, size=(3, 2))
        # 预期的结果数组
        desired = np.array([[32.9850547060149, 39.0219480493301],
                            [56.2006134779419, 57.3474165711485],
                            [55.4243733880198, 55.4209797925213]])
        # 断言 actual 与预期结果 desired 在小数点后 13 位精度下相等
        assert_array_almost_equal(actual, desired, decimal=13)
    def test_dirichlet(self):
        # 使用 MT19937 算法生成器和给定种子初始化随机数生成器
        random = Generator(MT19937(self.seed))
        # 设置 Dirichlet 分布的参数 alpha
        alpha = np.array([51.72840233779265162, 39.74494232180943953])
        # 生成一个 3x2 的 Dirichlet 分布样本
        actual = random.dirichlet(alpha, size=(3, 2))
        # 预期的结果
        desired = np.array([[[0.5439892869558927,  0.45601071304410745],
                             [0.5588917345860708,  0.4411082654139292 ]],
                            [[0.5632074165063435,  0.43679258349365657],
                             [0.54862581112627,    0.45137418887373015]],
                            [[0.49961831357047226, 0.5003816864295278 ],
                             [0.52374806183482,    0.47625193816517997]]])
        # 检查生成的样本是否与预期结果几乎相等
        assert_array_almost_equal(actual, desired, decimal=15)
        # 设置一个错误的 alpha 值，检查是否抛出 ValueError 异常
        bad_alpha = np.array([5.4e-01, -1.0e-16])
        assert_raises(ValueError, random.dirichlet, bad_alpha)

        # 重新初始化随机数生成器，设置相同的 alpha 值，生成一个单一的 Dirichlet 分布样本
        random = Generator(MT19937(self.seed))
        actual = random.dirichlet(alpha)
        # 检查生成的样本是否与预期结果的第一个元素相等
        assert_array_almost_equal(actual, desired[0, 0], decimal=15)

    def test_dirichlet_size(self):
        # 测试特定的 issue - gh-3173
        p = np.array([51.72840233779265162, 39.74494232180943953])
        # 检查不同输入大小下生成的 Dirichlet 分布样本的形状
        assert_equal(random.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.dirichlet(p, [2, 2]).shape, (2, 2, 2))
        assert_equal(random.dirichlet(p, (2, 2)).shape, (2, 2, 2))
        assert_equal(random.dirichlet(p, np.array((2, 2))).shape, (2, 2, 2))

        # 检查是否会抛出 TypeError 异常，如果 alpha 是浮点数而不是数组或列表
        assert_raises(TypeError, random.dirichlet, p, float(1))

    def test_dirichlet_bad_alpha(self):
        # 测试特定的 issue - gh-2089
        alpha = np.array([5.4e-01, -1.0e-16])
        # 检查是否会抛出 ValueError 异常，如果 alpha 包含无效的值
        assert_raises(ValueError, random.dirichlet, alpha)

        # 测试特定的 issue - gh-15876
        # 检查是否会抛出 ValueError 异常，如果 alpha 不符合 Dirichlet 分布的要求
        assert_raises(ValueError, random.dirichlet, [[5, 1]])
        assert_raises(ValueError, random.dirichlet, [[5], [1]])
        assert_raises(ValueError, random.dirichlet, [[[5], [1]], [[1], [5]]])
        assert_raises(ValueError, random.dirichlet, np.array([[5, 1], [1, 5]]))

    def test_dirichlet_alpha_non_contiguous(self):
        # 测试非连续的 alpha 数组
        a = np.array([51.72840233779265162, -1.0, 39.74494232180943953])
        # 创建一个非连续的 alpha 数组
        alpha = a[::2]
        random = Generator(MT19937(self.seed))
        # 使用非连续的 alpha 数组生成 Dirichlet 分布样本和使用连续的 alpha 数组生成样本，检查它们是否几乎相等
        non_contig = random.dirichlet(alpha, size=(3, 2))
        random = Generator(MT19937(self.seed))
        contig = random.dirichlet(np.ascontiguousarray(alpha),
                                  size=(3, 2))
        assert_array_almost_equal(non_contig, contig)
    # 测试小 alpha 值情况下的 Dirichlet 分布生成
    def test_dirichlet_small_alpha(self):
        eps = 1.0e-9  # 设置一个极小的 epsilon 值作为 alpha 的基础
        alpha = eps * np.array([1., 1.0e-3])  # 构造一个包含 epsilon 的 alpha 数组
        random = Generator(MT19937(self.seed))  # 使用指定种子创建随机数生成器对象
        actual = random.dirichlet(alpha, size=(3, 2))  # 生成指定大小的 Dirichlet 分布样本
        expected = np.array([  # 期望的 Dirichlet 分布样本
            [[1., 0.],
             [1., 0.]],
            [[1., 0.],
             [1., 0.]],
            [[1., 0.],
             [1., 0.]]
        ])
        assert_array_almost_equal(actual, expected, decimal=15)  # 断言实际结果与期望结果的接近程度

    @pytest.mark.slow  # 将当前测试标记为慢速测试
    def test_dirichlet_moderately_small_alpha(self):
        # 使用 alpha.max() < 0.1 触发分数破碎的代码路径
        alpha = np.array([0.02, 0.04, 0.03])  # 定义一个 alpha 数组
        exact_mean = alpha / alpha.sum()  # 计算精确的均值
        random = Generator(MT19937(self.seed))  # 使用指定种子创建随机数生成器对象
        sample = random.dirichlet(alpha, size=20000000)  # 生成大量大小的 Dirichlet 分布样本
        sample_mean = sample.mean(axis=0)  # 计算样本的均值
        assert_allclose(sample_mean, exact_mean, rtol=1e-3)  # 断言样本均值与精确均值的接近程度

    # 这组参数包括 alpha.max() >= 0.1 和 alpha.max() < 0.1，用于测试 Dirichlet 分布代码的两种生成方法
    @pytest.mark.parametrize(
        'alpha',
        [[5, 9, 0, 8],
         [0.5, 0, 0, 0],
         [1, 5, 0, 0, 1.5, 0, 0, 0],
         [0.01, 0.03, 0, 0.005],
         [1e-5, 0, 0, 0],
         [0.002, 0.015, 0, 0, 0.04, 0, 0, 0],
         [0.0],
         [0, 0, 0]],
    )
    def test_dirichlet_multiple_zeros_in_alpha(self, alpha):
        alpha = np.array(alpha)  # 转换成 numpy 数组
        y = random.dirichlet(alpha)  # 生成 Dirichlet 分布样本
        assert_equal(y[alpha == 0], 0.0)  # 断言对于 alpha 为零的位置，生成的值也应为零

    # 测试指数分布生成
    def test_exponential(self):
        random = Generator(MT19937(self.seed))  # 使用指定种子创建随机数生成器对象
        actual = random.exponential(1.1234, size=(3, 2))  # 生成指定大小的指数分布样本
        desired = np.array([[0.098845481066258, 1.560752510746964],
                            [0.075730916041636, 1.769098974710777],
                            [1.488602544592235, 2.49684815275751 ]])  # 期望的指数分布样本
        assert_array_almost_equal(actual, desired, decimal=15)  # 断言实际结果与期望结果的接近程度

    # 测试指数分布中 scale 参数为零的情况
    def test_exponential_0(self):
        assert_equal(random.exponential(scale=0), 0)  # 断言在 scale 参数为零时，生成的值应为零
        assert_raises(ValueError, random.exponential, scale=-0.)  # 断言对于负数 scale 参数会引发 ValueError

    # 测试 F 分布生成
    def test_f(self):
        random = Generator(MT19937(self.seed))  # 使用指定种子创建随机数生成器对象
        actual = random.f(12, 77, size=(3, 2))  # 生成指定大小的 F 分布样本
        desired = np.array([[0.461720027077085, 1.100441958872451],
                            [1.100337455217484, 0.91421736740018 ],
                            [0.500811891303113, 0.826802454552058]])  # 期望的 F 分布样本
        assert_array_almost_equal(actual, desired, decimal=15)  # 断言实际结果与期望结果的接近程度

    # 测试 Gamma 分布生成
    def test_gamma(self):
        random = Generator(MT19937(self.seed))  # 使用指定种子创建随机数生成器对象
        actual = random.gamma(5, 3, size=(3, 2))  # 生成指定大小的 Gamma 分布样本
        desired = np.array([[ 5.03850858902096,  7.9228656732049 ],
                            [18.73983605132985, 19.57961681699238],
                            [18.17897755150825, 18.17653912505234]])  # 期望的 Gamma 分布样本
        assert_array_almost_equal(actual, desired, decimal=14)  # 断言实际结果与期望结果的接近程度
    # 测试 gamma 分布中 shape=0, scale=0 的情况，预期结果为 0
    def test_gamma_0(self):
        assert_equal(random.gamma(shape=0, scale=0), 0)
        # 测试传入非法参数的情况，期望引发 ValueError
        assert_raises(ValueError, random.gamma, shape=-0., scale=-0.)

    # 测试几何分布的生成
    def test_geometric(self):
        # 使用给定的种子创建随机数生成器
        random = Generator(MT19937(self.seed))
        # 生成几何分布样本，指定概率为 0.123456789，形状为 (3, 2)
        actual = random.geometric(.123456789, size=(3, 2))
        # 期望的几何分布样本值
        desired = np.array([[1, 11],
                            [1, 12],
                            [11, 17]])
        # 断言生成的样本与期望值一致
        assert_array_equal(actual, desired)

    # 测试几何分布中的异常情况
    def test_geometric_exceptions(self):
        # 测试概率参数超出范围的情况，期望引发 ValueError
        assert_raises(ValueError, random.geometric, 1.1)
        # 测试概率参数列表中有值超出范围的情况，期望引发 ValueError
        assert_raises(ValueError, random.geometric, [1.1] * 10)
        # 测试概率参数为负数的情况，期望引发 ValueError
        assert_raises(ValueError, random.geometric, -0.1)
        # 测试概率参数列表中有负数的情况，期望引发 ValueError
        assert_raises(ValueError, random.geometric, [-0.1] * 10)
        # 在忽略无效值错误状态下，测试概率参数为 NaN 的情况，期望引发 ValueError
        with np.errstate(invalid='ignore'):
            assert_raises(ValueError, random.geometric, np.nan)
            assert_raises(ValueError, random.geometric, [np.nan] * 10)

    # 测试 Gumbel 分布的生成
    def test_gumbel(self):
        # 使用给定的种子创建随机数生成器
        random = Generator(MT19937(self.seed))
        # 生成 Gumbel 分布样本，指定位置参数为 0.123456789，尺度参数为 2.0，形状为 (3, 2)
        actual = random.gumbel(loc=.123456789, scale=2.0, size=(3, 2))
        # 期望的 Gumbel 分布样本值
        desired = np.array([[ 4.688397515056245, -0.289514845417841],
                            [ 4.981176042584683, -0.633224272589149],
                            [-0.055915275687488, -0.333962478257953]])
        # 断言生成的样本与期望值在指定精度下相近
        assert_array_almost_equal(actual, desired, decimal=15)

    # 测试 Gumbel 分布中尺度参数为 0 的情况
    def test_gumbel_0(self):
        # 测试尺度参数为 0 的 Gumbel 分布，预期结果为 0
        assert_equal(random.gumbel(scale=0), 0)
        # 测试传入非法参数的情况，期望引发 ValueError
        assert_raises(ValueError, random.gumbel, scale=-0.)

    # 测试超几何分布的生成
    def test_hypergeometric(self):
        # 使用给定的种子创建随机数生成器
        random = Generator(MT19937(self.seed))
        # 生成超几何分布样本，指定参数 ngood=10.1, nbad=5.5, nsample=14，形状为 (3, 2)
        actual = random.hypergeometric(10.1, 5.5, 14, size=(3, 2))
        # 期望的超几何分布样本值
        desired = np.array([[ 9, 9],
                            [ 9, 9],
                            [10, 9]])
        # 断言生成的样本与期望值一致
        assert_array_equal(actual, desired)

        # 测试 ngood = 0 的情况
        actual = random.hypergeometric(5, 0, 3, size=4)
        desired = np.array([3, 3, 3, 3])
        assert_array_equal(actual, desired)

        actual = random.hypergeometric(15, 0, 12, size=4)
        desired = np.array([12, 12, 12, 12])
        assert_array_equal(actual, desired)

        # 测试 nbad = 0 的情况
        actual = random.hypergeometric(0, 5, 3, size=4)
        desired = np.array([0, 0, 0, 0])
        assert_array_equal(actual, desired)

        actual = random.hypergeometric(0, 15, 12, size=4)
        desired = np.array([0, 0, 0, 0])
        assert_array_equal(actual, desired)

    # 测试拉普拉斯分布的生成
    def test_laplace(self):
        # 使用给定的种子创建随机数生成器
        random = Generator(MT19937(self.seed))
        # 生成拉普拉斯分布样本，指定位置参数为 0.123456789，尺度参数为 2.0，形状为 (3, 2)
        actual = random.laplace(loc=.123456789, scale=2.0, size=(3, 2))
        # 期望的拉普拉斯分布样本值
        desired = np.array([[-3.156353949272393,  1.195863024830054],
                            [-3.435458081645966,  1.656882398925444],
                            [ 0.924824032467446,  1.251116432209336]])
        # 断言生成的样本与期望值在指定精度下相近
        assert_array_almost_equal(actual, desired, decimal=15)

    # 测试拉普拉斯分布中尺度参数为 0 的情况
    def test_laplace_0(self):
        # 测试尺度参数为 0 的拉普拉斯分布，预期结果为 0
        assert_equal(random.laplace(scale=0), 0)
        # 测试传入非法参数的情况，期望引发 ValueError
        assert_raises(ValueError, random.laplace, scale=-0.)
    # 测试 logistic 方法的正确性
    def test_logistic(self):
        # 使用特定种子生成随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 调用 logistic 方法生成随机数数组，指定均值、标准差和数组大小
        actual = random.logistic(loc=.123456789, scale=2.0, size=(3, 2))
        # 预期的结果数组
        desired = np.array([[-4.338584631510999,  1.890171436749954],
                            [-4.64547787337966 ,  2.514545562919217],
                            [ 1.495389489198666,  1.967827627577474]])
        # 断言实际输出与预期结果的接近程度，指定精度为15位小数
        assert_array_almost_equal(actual, desired, decimal=15)

    # 测试 lognormal 方法的正确性
    def test_lognormal(self):
        # 使用特定种子生成随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 调用 lognormal 方法生成随机数数组，指定均值、标准差和数组大小
        actual = random.lognormal(mean=.123456789, sigma=2.0, size=(3, 2))
        # 预期的结果数组
        desired = np.array([[ 0.0268252166335, 13.9534486483053],
                            [ 0.1204014788936,  2.2422077497792],
                            [ 4.2484199496128, 12.0093343977523]])
        # 断言实际输出与预期结果的接近程度，指定精度为13位小数
        assert_array_almost_equal(actual, desired, decimal=13)

    # 测试 lognormal 方法当 sigma 为 0 时的异常情况
    def test_lognormal_0(self):
        # 断言 lognormal 方法在 sigma 为 0 时返回值为 1
        assert_equal(random.lognormal(sigma=0), 1)
        # 断言 lognormal 方法在 sigma 为负数时抛出 ValueError 异常
        assert_raises(ValueError, random.lognormal, sigma=-0.)

    # 测试 logseries 方法的正确性
    def test_logseries(self):
        # 使用特定种子生成随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 调用 logseries 方法生成随机数数组，指定概率和数组大小
        actual = random.logseries(p=.923456789, size=(3, 2))
        # 预期的结果数组
        desired = np.array([[14, 17],
                            [3, 18],
                            [5, 1]])
        # 断言实际输出与预期结果的完全相等
        assert_array_equal(actual, desired)

    # 测试 logseries 方法当概率 p 为 0 时的情况
    def test_logseries_zero(self):
        # 使用特定种子生成随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 断言 logseries 方法在概率 p 为 0 时返回值为 1
        assert random.logseries(0) == 1

    # 测试 logseries 方法在异常值情况下的行为
    @pytest.mark.parametrize("value", [np.nextafter(0., -1), 1., np.nan, 5.])
    def test_logseries_exceptions(self, value):
        # 使用特定种子生成随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 捕获异常并断言 logseries 方法对指定异常值抛出 ValueError 异常
        with np.errstate(invalid="ignore"):
            with pytest.raises(ValueError):
                random.logseries(value)
            with pytest.raises(ValueError):
                # 连续路径:
                random.logseries(np.array([value] * 10))
            with pytest.raises(ValueError):
                # 非连续路径:
                random.logseries(np.array([value] * 10)[::2])

    # 测试 multinomial 方法的正确性
    def test_multinomial(self):
        # 使用特定种子生成随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 调用 multinomial 方法生成多项式分布的随机数数组
        actual = random.multinomial(20, [1 / 6.] * 6, size=(3, 2))
        # 预期的结果数组
        desired = np.array([[[1, 5, 1, 6, 4, 3],
                             [4, 2, 6, 2, 4, 2]],
                            [[5, 3, 2, 6, 3, 1],
                             [4, 4, 0, 2, 3, 7]],
                            [[6, 3, 1, 5, 3, 2],
                             [5, 5, 3, 1, 2, 4]]])
        # 断言实际输出与预期结果的完全相等
        assert_array_equal(actual, desired)

    # 在 wasm 环境下跳过 fp 错误测试，标记为 pytest.skip
    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    # 参数化测试方法和协方差矩阵，确保不包含复数
    @pytest.mark.parametrize("method", ["svd", "eigh", "cholesky"])
    @pytest.mark.parametrize('mean, cov', [([0], [[1+1j]]), ([0j], [[1]])])
    def test_multivariate_normal_disallow_complex(self, mean, cov):
        # 使用特定种子生成随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 捕获异常并断言 multivariate_normal 方法对复数协方差矩阵抛出 TypeError 异常
        with pytest.raises(TypeError, match="must not be complex"):
            random.multivariate_normal(mean, cov)

    # 参数化测试方法，验证各种协方差矩阵计算方法
    @pytest.mark.parametrize("method", ["svd", "eigh", "cholesky"])
    def test_multivariate_normal_basic_stats(self, method):
        # 使用给定的随机数种子初始化随机数生成器
        random = Generator(MT19937(self.seed))
        # 设定样本数量
        n_s = 1000
        # 设定多变量正态分布的均值向量和协方差矩阵
        mean = np.array([1, 2])
        cov = np.array([[2, 1], [1, 2]])
        # 生成多变量正态分布样本数据
        s = random.multivariate_normal(mean, cov, size=(n_s,), method=method)
        # 将样本数据居中化
        s_center = s - mean
        # 计算样本的经验协方差矩阵
        cov_emp = (s_center.T @ s_center) / (n_s - 1)
        # 断言检查样本均值是否接近期望值
        assert np.all(np.abs(s_center.mean(-2)) < 0.1)
        # 断言检查经验协方差矩阵是否接近真实协方差矩阵
        assert np.all(np.abs(cov_emp - cov) < 0.2)

    def test_negative_binomial(self):
        # 使用给定的随机数种子初始化随机数生成器
        random = Generator(MT19937(self.seed))
        # 生成负二项分布的样本数据
        actual = random.negative_binomial(n=100, p=.12345, size=(3, 2))
        # 预期的负二项分布样本数据
        desired = np.array([[543, 727],
                            [775, 760],
                            [600, 674]])
        # 断言检查生成的样本数据是否与预期一致
        assert_array_equal(actual, desired)

    def test_negative_binomial_exceptions(self):
        # 忽略 numpy 中的无效操作警告
        with np.errstate(invalid='ignore'):
            # 断言检查负二项分布函数在给定异常情况下是否正确抛出 ValueError
            assert_raises(ValueError, random.negative_binomial, 100, np.nan)
            assert_raises(ValueError, random.negative_binomial, 100,
                          [np.nan] * 10)

    def test_negative_binomial_p0_exception(self):
        # 验证当 p=0 时，是否会引发 ValueError 异常
        with assert_raises(ValueError):
            x = random.negative_binomial(1, 0)

    def test_negative_binomial_invalid_p_n_combination(self):
        # 验证在可能导致溢出或无限循环的 p 和 n 值组合下是否会引发异常
        with np.errstate(invalid='ignore'):
            assert_raises(ValueError, random.negative_binomial, 2**62, 0.1)
            assert_raises(ValueError, random.negative_binomial, [2**62], [0.1])

    def test_noncentral_chisquare(self):
        # 使用给定的随机数种子初始化随机数生成器
        random = Generator(MT19937(self.seed))
        # 生成非中心卡方分布的样本数据
        actual = random.noncentral_chisquare(df=5, nonc=5, size=(3, 2))
        # 预期的非中心卡方分布样本数据
        desired = np.array([[ 1.70561552362133, 15.97378184942111],
                            [13.71483425173724, 20.17859633310629],
                            [11.3615477156643 ,  3.67891108738029]])
        # 断言检查生成的样本数据是否与预期一致，精确到小数点后14位
        assert_array_almost_equal(actual, desired, decimal=14)

        # 生成另一组非中心卡方分布的样本数据
        actual = random.noncentral_chisquare(df=.5, nonc=.2, size=(3, 2))
        # 预期的非中心卡方分布样本数据
        desired = np.array([[9.41427665607629e-04, 1.70473157518850e-04],
                            [1.14554372041263e+00, 1.38187755933435e-03],
                            [1.90659181905387e+00, 1.21772577941822e+00]])
        # 断言检查生成的样本数据是否与预期一致，精确到小数点后14位
        assert_array_almost_equal(actual, desired, decimal=14)

        # 生成另一组非中心卡方分布的样本数据，这次设定非中心参数为 0
        actual = random.noncentral_chisquare(df=5, nonc=0, size=(3, 2))
        # 预期的非中心卡方分布样本数据
        desired = np.array([[0.82947954590419, 1.80139670767078],
                            [6.58720057417794, 7.00491463609814],
                            [6.31101879073157, 6.30982307753005]])
        # 断言检查生成的样本数据是否与预期一致，精确到小数点后14位
        assert_array_almost_equal(actual, desired, decimal=14)
    # 定义测试非中心 F 分布的函数
    def test_noncentral_f(self):
        # 使用指定种子创建随机数生成器
        random = Generator(MT19937(self.seed))
        # 生成非中心 F 分布的随机样本
        actual = random.noncentral_f(dfnum=5, dfden=2, nonc=1, size=(3, 2))
        # 预期的非中心 F 分布的值
        desired = np.array([[0.060310671139  , 0.23866058175939],
                            [0.86860246709073, 0.2668510459738 ],
                            [0.23375780078364, 1.88922102885943]])
        # 检查生成的随机样本与预期值的接近程度
        assert_array_almost_equal(actual, desired, decimal=14)

    # 定义测试带 NaN 值的非中心 F 分布的函数
    def test_noncentral_f_nan(self):
        # 使用指定种子创建随机数生成器
        random = Generator(MT19937(self.seed))
        # 生成带 NaN 非中心参数的非中心 F 分布的随机样本
        actual = random.noncentral_f(dfnum=5, dfden=2, nonc=np.nan)
        # 断言生成的值是 NaN
        assert np.isnan(actual)

    # 定义测试正态分布的函数
    def test_normal(self):
        # 使用指定种子创建随机数生成器
        random = Generator(MT19937(self.seed))
        # 生成正态分布的随机样本
        actual = random.normal(loc=.123456789, scale=2.0, size=(3, 2))
        # 预期的正态分布的值
        desired = np.array([[-3.618412914693162,  2.635726692647081],
                            [-2.116923463013243,  0.807460983059643],
                            [ 1.446547137248593,  2.485684213886024]])
        # 检查生成的随机样本与预期值的接近程度
        assert_array_almost_equal(actual, desired, decimal=15)

    # 定义测试带特定条件的正态分布的函数
    def test_normal_0(self):
        # 断言在特定条件下正态分布生成的值为 0
        assert_equal(random.normal(scale=0), 0)
        # 断言在不合理的条件下正态分布抛出 ValueError 异常
        assert_raises(ValueError, random.normal, scale=-0.)

    # 定义测试帕累托分布的函数
    def test_pareto(self):
        # 使用指定种子创建随机数生成器
        random = Generator(MT19937(self.seed))
        # 生成帕累托分布的随机样本
        actual = random.pareto(a=.123456789, size=(3, 2))
        # 预期的帕累托分布的值
        desired = np.array([[1.0394926776069018e+00, 7.7142534343505773e+04],
                            [7.2640150889064703e-01, 3.4650454783825594e+05],
                            [4.5852344481994740e+04, 6.5851383009539105e+07]])
        # 检查生成的随机样本与预期值的接近程度，允许较大的数值单位最后位置差异
        np.testing.assert_array_almost_equal_nulp(actual, desired, nulp=30)

    # 定义测试泊松分布的函数
    def test_poisson(self):
        # 使用指定种子创建随机数生成器
        random = Generator(MT19937(self.seed))
        # 生成泊松分布的随机样本
        actual = random.poisson(lam=.123456789, size=(3, 2))
        # 预期的泊松分布的值
        desired = np.array([[0, 0],
                            [0, 0],
                            [0, 0]])
        # 检查生成的随机样本与预期值是否相等
        assert_array_equal(actual, desired)

    # 定义测试泊松分布异常情况的函数
    def test_poisson_exceptions(self):
        # 定义一个大的正整数和一个负数作为测试用的参数
        lambig = np.iinfo('int64').max
        lamneg = -1
        # 断言当泊松分布的参数为负数时抛出 ValueError 异常
        assert_raises(ValueError, random.poisson, lamneg)
        # 断言当泊松分布的参数列表中包含负数时抛出 ValueError 异常
        assert_raises(ValueError, random.poisson, [lamneg] * 10)
        # 断言当泊松分布的参数为较大的正整数时抛出 ValueError 异常
        assert_raises(ValueError, random.poisson, lambig)
        # 断言当泊松分布的参数列表中包含较大的正整数时抛出 ValueError 异常
        assert_raises(ValueError, random.poisson, [lambig] * 10)
        # 在忽略无效操作的情况下，断言当泊松分布的参数为 NaN 时抛出 ValueError 异常
        with np.errstate(invalid='ignore'):
            assert_raises(ValueError, random.poisson, np.nan)
            assert_raises(ValueError, random.poisson, [np.nan] * 10)
    # 定义一个测试方法，用于测试随机数发生器生成的 power 分布
    def test_power(self):
        # 使用给定种子初始化 Mersenne Twister 伪随机数生成器
        random = Generator(MT19937(self.seed))
        # 生成指定参数的 power 分布随机数，返回一个数组
        actual = random.power(a=.123456789, size=(3, 2))
        # 预期的 power 分布随机数结果数组
        desired = np.array([[1.977857368842754e-09, 9.806792196620341e-02],
                            [2.482442984543471e-10, 1.527108843266079e-01],
                            [8.188283434244285e-02, 3.950547209346948e-01]])
        # 断言生成的随机数数组与预期数组几乎相等，精确度为 15 位小数
        assert_array_almost_equal(actual, desired, decimal=15)

    # 定义一个测试方法，用于测试随机数发生器生成的 Rayleigh 分布
    def test_rayleigh(self):
        # 使用给定种子初始化 Mersenne Twister 伪随机数生成器
        random = Generator(MT19937(self.seed))
        # 生成指定参数的 Rayleigh 分布随机数，返回一个数组
        actual = random.rayleigh(scale=10, size=(3, 2))
        # 预期的 Rayleigh 分布随机数结果数组
        desired = np.array([[4.19494429102666, 16.66920198906598],
                            [3.67184544902662, 17.74695521962917],
                            [16.27935397855501, 21.08355560691792]])
        # 断言生成的随机数数组与预期数组几乎相等，精确度为 14 位小数
        assert_array_almost_equal(actual, desired, decimal=14)

    # 定义一个测试方法，用于测试 Rayleigh 分布的特殊情况：scale 为 0
    def test_rayleigh_0(self):
        # 断言当 scale 参数为 0 时，生成的 Rayleigh 随机数应为 0
        assert_equal(random.rayleigh(scale=0), 0)
        # 断言当 scale 参数为负数时，应该引发 ValueError 异常
        assert_raises(ValueError, random.rayleigh, scale=-0.)

    # 定义一个测试方法，用于测试随机数发生器生成的标准 Cauchy 分布
    def test_standard_cauchy(self):
        # 使用给定种子初始化 Mersenne Twister 伪随机数生成器
        random = Generator(MT19937(self.seed))
        # 生成指定参数的标准 Cauchy 分布随机数，返回一个数组
        actual = random.standard_cauchy(size=(3, 2))
        # 预期的标准 Cauchy 分布随机数结果数组
        desired = np.array([[-1.489437778266206, -3.275389641569784],
                            [ 0.560102864910406, -0.680780916282552],
                            [-1.314912905226277,  0.295852965660225]])
        # 断言生成的随机数数组与预期数组几乎相等，精确度为 15 位小数
        assert_array_almost_equal(actual, desired, decimal=15)

    # 定义一个测试方法，用于测试随机数发生器生成的标准指数分布
    def test_standard_exponential(self):
        # 使用给定种子初始化 Mersenne Twister 伪随机数生成器
        random = Generator(MT19937(self.seed))
        # 生成指定参数的标准指数分布随机数，返回一个数组
        actual = random.standard_exponential(size=(3, 2), method='inv')
        # 预期的标准指数分布随机数结果数组
        desired = np.array([[0.102031839440643, 1.229350298474972],
                            [0.088137284693098, 1.459859985522667],
                            [1.093830802293668, 1.256977002164613]])
        # 断言生成的随机数数组与预期数组几乎相等，精确度为 15 位小数
        assert_array_almost_equal(actual, desired, decimal=15)

    # 定义一个测试方法，用于测试标准指数分布生成时的类型错误情况
    def test_standard_expoential_type_error(self):
        # 断言当指定 dtype 为 np.int32 时，应该引发 TypeError 异常
        assert_raises(TypeError, random.standard_exponential, dtype=np.int32)

    # 定义一个测试方法，用于测试随机数发生器生成的标准 Gamma 分布
    def test_standard_gamma(self):
        # 使用给定种子初始化 Mersenne Twister 伪随机数生成器
        random = Generator(MT19937(self.seed))
        # 生成指定参数的标准 Gamma 分布随机数，返回一个数组
        actual = random.standard_gamma(shape=3, size=(3, 2))
        # 预期的标准 Gamma 分布随机数结果数组
        desired = np.array([[0.62970724056362, 1.22379851271008],
                            [3.899412530884  , 4.12479964250139],
                            [3.74994102464584, 3.74929307690815]])
        # 断言生成的随机数数组与预期数组几乎相等，精确度为 14 位小数
        assert_array_almost_equal(actual, desired, decimal=14)

    # 定义一个测试方法，用于测试生成的标准 Gamma 分布的浮点数情况
    def test_standard_gammma_scalar_float(self):
        # 使用给定种子初始化 Mersenne Twister 伪随机数生成器
        random = Generator(MT19937(self.seed))
        # 生成指定参数的标准 Gamma 分布随机数，返回一个浮点数
        actual = random.standard_gamma(3, dtype=np.float32)
        # 预期的标准 Gamma 分布随机数结果
        desired = 2.9242148399353027
        # 断言生成的随机数与预期结果几乎相等，精确度为 6 位小数
        assert_array_almost_equal(actual, desired, decimal=6)

    # 定义一个测试方法，用于测试随机数发生器生成的标准 Gamma 分布
    def test_standard_gamma_float(self):
        # 使用给定种子初始化 Mersenne Twister 伪随机数生成器
        random = Generator(MT19937(self.seed))
        # 生成指定参数的标准 Gamma 分布随机数，返回一个数组
        actual = random.standard_gamma(shape=3, size=(3, 2))
        # 预期的标准 Gamma 分布随机数结果数组
        desired = np.array([[0.62971, 1.2238 ],
                            [3.89941, 4.1248 ],
                            [3.74994, 3.74929]])
        # 断言生成的随机数数组与预期数组几乎相等，精确度为 5 位小数
        assert_array_almost_equal(actual, desired, decimal=5)
    def test_standard_gammma_float_out(self):
        # 创建一个形状为 (3, 2)，数据类型为 np.float32 的全零数组
        actual = np.zeros((3, 2), dtype=np.float32)
        # 使用给定种子创建一个随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 生成标准伽玛分布的随机数，填充到预先提供的数组 actual 中
        random.standard_gamma(10.0, out=actual, dtype=np.float32)
        # 期望的结果数组，与实际结果比较
        desired = np.array([[10.14987,  7.87012],
                             [ 9.46284, 12.56832],
                             [13.82495,  7.81533]], dtype=np.float32)
        # 断言实际数组与期望数组几乎相等，精度为小数点后五位
        assert_array_almost_equal(actual, desired, decimal=5)

        # 重新使用相同的随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 生成标准伽玛分布的随机数，填充到已存在的数组 actual 中，并指定新的形状 (3, 2)
        random.standard_gamma(10.0, out=actual, size=(3, 2), dtype=np.float32)
        # 再次断言实际数组与期望数组几乎相等，精度为小数点后五位
        assert_array_almost_equal(actual, desired, decimal=5)

    def test_standard_gamma_unknown_type(self):
        # 断言尝试生成标准伽玛分布时使用未知的数据类型会引发 TypeError 异常
        assert_raises(TypeError, random.standard_gamma, 1.,
                      dtype='int32')

    def test_out_size_mismatch(self):
        # 创建一个长度为 10 的全零数组
        out = np.zeros(10)
        # 断言生成标准伽玛分布时，指定的输出数组长度与 size 参数不匹配会引发 ValueError 异常
        assert_raises(ValueError, random.standard_gamma, 10.0, size=20,
                      out=out)
        # 断言生成标准伽玛分布时，指定的输出数组形状与 size 参数不匹配会引发 ValueError 异常
        assert_raises(ValueError, random.standard_gamma, 10.0, size=(10, 1),
                      out=out)

    def test_standard_gamma_0(self):
        # 断言生成形状为 0 的标准伽玛分布的随机数应为 0
        assert_equal(random.standard_gamma(shape=0), 0)
        # 断言尝试生成形状为负数的标准伽玛分布的随机数会引发 ValueError 异常
        assert_raises(ValueError, random.standard_gamma, shape=-0.)

    def test_standard_normal(self):
        # 使用给定种子创建一个随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 生成标准正态分布的随机数，指定形状为 (3, 2)
        actual = random.standard_normal(size=(3, 2))
        # 期望的结果数组，与实际结果比较
        desired = np.array([[-1.870934851846581,  1.25613495182354 ],
                            [-1.120190126006621,  0.342002097029821],
                            [ 0.661545174124296,  1.181113712443012]])
        # 断言实际数组与期望数组几乎相等，精度为小数点后十五位
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_standard_normal_unsupported_type(self):
        # 断言尝试生成标准正态分布时使用不支持的数据类型会引发 TypeError 异常
        assert_raises(TypeError, random.standard_normal, dtype=np.int32)

    def test_standard_t(self):
        # 使用给定种子创建一个随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 生成自由度为 10 的 t 分布的随机数，指定形状为 (3, 2)
        actual = random.standard_t(df=10, size=(3, 2))
        # 期望的结果数组，与实际结果比较
        desired = np.array([[-1.484666193042647,  0.30597891831161 ],
                            [ 1.056684299648085, -0.407312602088507],
                            [ 0.130704414281157, -2.038053410490321]])
        # 断言实际数组与期望数组几乎相等，精度为小数点后十五位
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_triangular(self):
        # 使用给定种子创建一个随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 生成三角分布的随机数，指定左边界为 5.12，众数为 10.23，右边界为 20.34，形状为 (3, 2)
        actual = random.triangular(left=5.12, mode=10.23, right=20.34,
                                   size=(3, 2))
        # 期望的结果数组，与实际结果比较
        desired = np.array([[ 7.86664070590917, 13.6313848513185 ],
                            [ 7.68152445215983, 14.36169131136546],
                            [13.16105603911429, 13.72341621856971]])
        # 断言实际数组与期望数组几乎相等，精度为小数点后十四位
        assert_array_almost_equal(actual, desired, decimal=14)
    def test_uniform(self):
        # 使用 Mersenne Twister 算法生成器和给定的种子初始化随机数生成器
        random = Generator(MT19937(self.seed))
        # 生成均匀分布随机数数组，范围在 [1.23, 10.54]，大小为 (3, 2)
        actual = random.uniform(low=1.23, high=10.54, size=(3, 2))
        # 期望的结果数组
        desired = np.array([[2.13306255040998 , 7.816987531021207],
                            [2.015436610109887, 8.377577533009589],
                            [7.421792588856135, 7.891185744455209]])
        # 断言生成的随机数数组与期望的数组几乎相等，精度为 15 位小数
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_uniform_range_bounds(self):
        # 获取浮点数的最小值和最大值
        fmin = np.finfo('float').min
        fmax = np.finfo('float').max

        # 定义 uniform 函数的别名
        func = random.uniform
        # 断言当边界超出浮点数范围时会抛出 OverflowError
        assert_raises(OverflowError, func, -np.inf, 0)
        assert_raises(OverflowError, func, 0, np.inf)
        assert_raises(OverflowError, func, fmin, fmax)
        assert_raises(OverflowError, func, [-np.inf], [0])
        assert_raises(OverflowError, func, [0], [np.inf])

        # 检验较大范围内的 uniform 调用不会抛出异常
        # 通过增加 fmin 以防止 DBL_MAX / 1e17 + DBL_MAX > DBL_MAX
        random.uniform(low=np.nextafter(fmin, 1), high=fmax / 1e17)

    def test_uniform_zero_range(self):
        # 定义 uniform 函数的别名
        func = random.uniform
        # 测试当范围为零时的 uniform 调用
        result = func(1.5, 1.5)
        assert_allclose(result, 1.5)
        result = func([0.0, np.pi], [0.0, np.pi])
        assert_allclose(result, [0.0, np.pi])
        result = func([[2145.12], [2145.12]], [2145.12, 2145.12])
        assert_allclose(result, 2145.12 + np.zeros((2, 2)))

    def test_uniform_neg_range(self):
        # 定义 uniform 函数的别名
        func = random.uniform
        # 断言当范围为负时会抛出 ValueError
        assert_raises(ValueError, func, 2, 1)
        assert_raises(ValueError, func,  [1, 2], [1, 1])
        assert_raises(ValueError, func,  [[0, 1],[2, 3]], 2)

    def test_scalar_exception_propagation(self):
        # 测试分布中的异常传播是否正确
        # 当对象转换为标量时抛出异常的情况
        #
        # gh: 8865 的回归测试

        # 定义抛出异常的浮点数类
        class ThrowingFloat(np.ndarray):
            def __float__(self):
                raise TypeError

        # 创建抛出异常的浮点数对象
        throwing_float = np.array(1.0).view(ThrowingFloat)
        # 断言调用 uniform 函数时会传播 TypeError 异常
        assert_raises(TypeError, random.uniform, throwing_float,
                      throwing_float)

        # 定义抛出异常的整数类
        class ThrowingInteger(np.ndarray):
            def __int__(self):
                raise TypeError

        # 创建抛出异常的整数对象
        throwing_int = np.array(1).view(ThrowingInteger)
        # 断言调用 hypergeometric 函数时会传播 TypeError 异常
        assert_raises(TypeError, random.hypergeometric, throwing_int, 1, 1)

    def test_vonmises(self):
        # 使用 Mersenne Twister 算法生成器和给定的种子初始化随机数生成器
        random = Generator(MT19937(self.seed))
        # 生成 von Mises 分布随机数数组，参数为 mu=1.23, kappa=1.54，大小为 (3, 2)
        actual = random.vonmises(mu=1.23, kappa=1.54, size=(3, 2))
        # 期望的结果数组
        desired = np.array([[ 1.107972248690106,  2.841536476232361],
                            [ 1.832602376042457,  1.945511926976032],
                            [-0.260147475776542,  2.058047492231698]])
        # 断言生成的随机数数组与期望的数组几乎相等，精度为 15 位小数
        assert_array_almost_equal(actual, desired, decimal=15)
    # 测试 von Mises 分布生成器的小规模数据，检查是否存在无限循环问题，参见 issue gh-4720
    def test_vonmises_small(self):
        # 使用指定种子初始化随机数生成器
        random = Generator(MT19937(self.seed))
        # 生成 von Mises 分布样本，其中 mu=0, kappa=1.1e-8，生成大小为 10**6
        r = random.vonmises(mu=0., kappa=1.1e-8, size=10**6)
        # 断言所有生成的值都是有限的
        assert_(np.isfinite(r).all())

    # 测试 von Mises 分布生成器对于 kappa 为 NaN 的情况
    def test_vonmises_nan(self):
        random = Generator(MT19937(self.seed))
        # 生成 von Mises 分布样本，其中 mu=0, kappa=np.nan
        r = random.vonmises(mu=0., kappa=np.nan)
        # 断言生成的值包含 NaN
        assert_(np.isnan(r))

    # 参数化测试，测试 von Mises 分布生成器对于大 kappa 值的情况
    @pytest.mark.parametrize("kappa", [1e4, 1e15])
    def test_vonmises_large_kappa(self, kappa):
        random = Generator(MT19937(self.seed))
        # 使用 RandomState 从 Generator 中获取随机状态
        rs = RandomState(random.bit_generator)
        state = random.bit_generator.state

        # 使用 RandomState 生成 von Mises 分布样本，其中 mu=0, kappa=当前参数化的 kappa 值，生成大小为 10
        random_state_vals = rs.vonmises(0, kappa, size=10)
        # 恢复 Generator 的随机状态
        random.bit_generator.state = state
        # 使用 Generator 生成 von Mises 分布样本，其中 mu=0, kappa=当前参数化的 kappa 值，生成大小为 10
        gen_vals = random.vonmises(0, kappa, size=10)
        
        # 根据 kappa 值的大小进行断言
        if kappa < 1e6:
            # 断言两种方式生成的值接近
            assert_allclose(random_state_vals, gen_vals)
        else:
            # 断言两种方式生成的值不完全相等
            assert np.all(random_state_vals != gen_vals)

    # 参数化测试，测试 von Mises 分布生成器对于大范围的 mu 和 kappa 值的情况
    @pytest.mark.parametrize("mu", [-7., -np.pi, -3.1, np.pi, 3.2])
    @pytest.mark.parametrize("kappa", [1e-9, 1e-6, 1, 1e3, 1e15])
    def test_vonmises_large_kappa_range(self, mu, kappa):
        random = Generator(MT19937(self.seed))
        # 生成 von Mises 分布样本，其中 mu=当前参数化的 mu 值, kappa=当前参数化的 kappa 值，生成大小为 50
        r = random.vonmises(mu, kappa, 50)
        # 断言生成的值都在 -π 到 π 的范围内
        assert_(np.all(r > -np.pi) and np.all(r <= np.pi))

    # 测试 Wald 分布生成器
    def test_wald(self):
        random = Generator(MT19937(self.seed))
        # 生成 Wald 分布样本，其中 mean=1.23, scale=1.54，生成大小为 (3, 2)
        actual = random.wald(mean=1.23, scale=1.54, size=(3, 2))
        # 期望的结果
        desired = np.array([[0.26871721804551, 3.2233942732115 ],
                            [2.20328374987066, 2.40958405189353],
                            [2.07093587449261, 0.73073890064369]])
        # 断言生成的值与期望的结果几乎相等，精度为 14 位小数
        assert_array_almost_equal(actual, desired, decimal=14)

    # 测试 Weibull 分布生成器
    def test_weibull(self):
        random = Generator(MT19937(self.seed))
        # 生成 Weibull 分布样本，其中 a=1.23，生成大小为 (3, 2)
        actual = random.weibull(a=1.23, size=(3, 2))
        # 期望的结果
        desired = np.array([[0.138613914769468, 1.306463419753191],
                            [0.111623365934763, 1.446570494646721],
                            [1.257145775276011, 1.914247725027957]])
        # 断言生成的值与期望的结果几乎相等，精度为 15 位小数
        assert_array_almost_equal(actual, desired, decimal=15)

    # 测试 a=0 时的 Weibull 分布生成器
    def test_weibull_0(self):
        random = Generator(MT19937(self.seed))
        # 断言生成的 Weibull 分布样本全为 0
        assert_equal(random.weibull(a=0, size=12), np.zeros(12))
        # 断言当 a=-0 时会引发 ValueError
        assert_raises(ValueError, random.weibull, a=-0.)

    # 测试 Zipf 分布生成器
    def test_zipf(self):
        random = Generator(MT19937(self.seed))
        # 生成 Zipf 分布样本，其中 a=1.23，生成大小为 (3, 2)
        actual = random.zipf(a=1.23, size=(3, 2))
        # 期望的结果
        desired = np.array([[  1,   1],
                            [ 10, 867],
                            [354,   2]])
        # 断言生成的值与期望的结果完全相等
        assert_array_equal(actual, desired)
# 定义一个名为 TestBroadcast 的测试类，用于测试广播函数在非标量参数情况下的行为是否正确。
class TestBroadcast:
    # 在每个测试方法执行前执行的初始化方法
    def setup_method(self):
        self.seed = 123456789

    # 测试 uniform 函数的行为
    def test_uniform(self):
        # 创建随机数生成器对象，使用 MT19937 算法和给定的种子
        random = Generator(MT19937(self.seed))
        # 定义 low 和 high 分别为 [0] 和 [1]
        low = [0]
        high = [1]
        # 获取 uniform 函数的引用
        uniform = random.uniform
        # 定义期望的结果数组
        desired = np.array([0.16693771389729, 0.19635129550675, 0.75563050964095])

        # 重新创建随机数生成器对象，确保使用相同的种子
        random = Generator(MT19937(self.seed))
        # 调用 uniform 函数，生成随机数数组，用 assert_array_almost_equal 断言检查结果准确性
        actual = random.uniform(low * 3, high)
        assert_array_almost_equal(actual, desired, decimal=14)

        # 重新创建随机数生成器对象，再次调用 uniform 函数，用 assert_array_almost_equal 断言检查结果准确性
        random = Generator(MT19937(self.seed))
        actual = random.uniform(low, high * 3)
        assert_array_almost_equal(actual, desired, decimal=14)

    # 测试 normal 函数的行为
    def test_normal(self):
        # 定义 loc 和 scale 分别为 [0] 和 [1]，以及一个错误的 scale 值列表 [-1]
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        # 创建随机数生成器对象，使用 MT19937 算法和给定的种子
        random = Generator(MT19937(self.seed))
        # 定义期望的结果数组
        desired = np.array([-0.38736406738527,  0.79594375042255,  0.0197076236097])

        # 重新创建随机数生成器对象，调用 normal 函数，用 assert_array_almost_equal 断言检查结果准确性
        random = Generator(MT19937(self.seed))
        actual = random.normal(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        # 使用 assert_raises 检查是否引发 ValueError 异常
        assert_raises(ValueError, random.normal, loc * 3, bad_scale)

        # 重新创建随机数生成器对象，再次调用 normal 函数，用 assert_array_almost_equal 断言检查结果准确性
        random = Generator(MT19937(self.seed))
        normal = random.normal
        actual = normal(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        # 使用 assert_raises 检查是否引发 ValueError 异常
        assert_raises(ValueError, normal, loc, bad_scale * 3)

    # 测试 beta 函数的行为
    def test_beta(self):
        # 定义 a 和 b 分别为 [1] 和 [2]，以及两个错误的 a 和 b 值列表 [-1] 和 [-2]
        a = [1]
        b = [2]
        bad_a = [-1]
        bad_b = [-2]
        # 定义期望的结果数组
        desired = np.array([0.18719338682602, 0.73234824491364, 0.17928615186455])

        # 创建随机数生成器对象，使用 MT19937 算法和给定的种子
        random = Generator(MT19937(self.seed))
        # 获取 beta 函数的引用
        beta = random.beta
        # 调用 beta 函数，生成随机数数组，用 assert_array_almost_equal 断言检查结果准确性
        actual = beta(a * 3, b)
        assert_array_almost_equal(actual, desired, decimal=14)
        # 使用 assert_raises 检查是否引发 ValueError 异常
        assert_raises(ValueError, beta, bad_a * 3, b)
        assert_raises(ValueError, beta, a * 3, bad_b)

        # 重新创建随机数生成器对象，再次调用 beta 函数，用 assert_array_almost_equal 断言检查结果准确性
        random = Generator(MT19937(self.seed))
        actual = random.beta(a, b * 3)
        assert_array_almost_equal(actual, desired, decimal=14)

    # 测试 exponential 函数的行为
    def test_exponential(self):
        # 定义 scale 和一个错误的 scale 值列表 [-1]
        scale = [1]
        bad_scale = [-1]
        # 定义期望的结果数组
        desired = np.array([0.67245993212806, 0.21380495318094, 0.7177848928629])

        # 创建随机数生成器对象，使用 MT19937 算法和给定的种子
        random = Generator(MT19937(self.seed))
        # 调用 exponential 函数，生成随机数数组，用 assert_array_almost_equal 断言检查结果准确性
        actual = random.exponential(scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        # 使用 assert_raises 检查是否引发 ValueError 异常
        assert_raises(ValueError, random.exponential, bad_scale * 3)

    # 测试 standard_gamma 函数的行为
    def test_standard_gamma(self):
        # 定义 shape 和一个错误的 shape 值列表 [-1]
        shape = [1]
        bad_shape = [-1]
        # 定义期望的结果数组
        desired = np.array([0.67245993212806, 0.21380495318094, 0.7177848928629])

        # 创建随机数生成器对象，使用 MT19937 算法和给定的种子
        random = Generator(MT19937(self.seed))
        # 获取 standard_gamma 函数的引用
        std_gamma = random.standard_gamma
        # 调用 standard_gamma 函数，生成随机数数组，用 assert_array_almost_equal 断言检查结果准确性
        actual = std_gamma(shape * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        # 使用 assert_raises 检查是否引发 ValueError 异常
        assert_raises(ValueError, std_gamma, bad_shape * 3)
    # 定义一个用于测试 gamma 分布函数的方法
    def test_gamma(self):
        # 设定 gamma 分布的形状参数为 [1]
        shape = [1]
        # 设定 gamma 分布的尺度参数为 [2]
        scale = [2]
        # 设定一个无效的形状参数为 [-1]
        bad_shape = [-1]
        # 设定一个无效的尺度参数为 [-2]
        bad_scale = [-2]
        # 期望得到的结果数组
        desired = np.array([1.34491986425611, 0.42760990636187, 1.4355697857258])

        # 创建一个随机数生成器，使用 MT19937 算法，种子值由类中的 self.seed 提供
        random = Generator(MT19937(self.seed))
        # 获取 gamma 方法
        gamma = random.gamma
        # 使用给定的形状和尺度参数生成 gamma 分布的实际结果
        actual = gamma(shape * 3, scale)
        # 断言实际结果与期望结果的近似性，精度为小数点后14位
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言当形状参数为负数时会引发 ValueError 异常
        assert_raises(ValueError, gamma, bad_shape * 3, scale)
        # 断言当尺度参数为负数时会引发 ValueError 异常
        assert_raises(ValueError, gamma, shape * 3, bad_scale)

        # 重新创建一个随机数生成器，以确保测试独立性，使用相同的种子值
        random = Generator(MT19937(self.seed))
        # 获取 gamma 方法
        gamma = random.gamma
        # 使用给定的形状和尺度参数生成 gamma 分布的实际结果
        actual = gamma(shape, scale * 3)
        # 断言实际结果与期望结果的近似性，精度为小数点后14位
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言当形状参数为负数时会引发 ValueError 异常
        assert_raises(ValueError, gamma, bad_shape, scale * 3)
        # 断言当尺度参数为负数时会引发 ValueError 异常
        assert_raises(ValueError, gamma, shape, bad_scale * 3)

    # 定义一个用于测试 F 分布函数的方法
    def test_f(self):
        # 设定 F 分布的自由度参数为 [1]（分子）
        dfnum = [1]
        # 设定 F 分布的自由度参数为 [2]（分母）
        dfden = [2]
        # 设定一个无效的自由度参数为 [-1]（分子）
        bad_dfnum = [-1]
        # 设定一个无效的自由度参数为 [-2]（分母）
        bad_dfden = [-2]
        # 期望得到的结果数组
        desired = np.array([0.07765056244107, 7.72951397913186, 0.05786093891763])

        # 创建一个随机数生成器，使用 MT19937 算法，种子值由类中的 self.seed 提供
        random = Generator(MT19937(self.seed))
        # 获取 f 方法（F 分布函数）
        f = random.f
        # 使用给定的自由度参数生成 F 分布的实际结果
        actual = f(dfnum * 3, dfden)
        # 断言实际结果与期望结果的近似性，精度为小数点后14位
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言当自由度参数为负数时会引发 ValueError 异常
        assert_raises(ValueError, f, bad_dfnum * 3, dfden)
        # 断言当自由度参数为负数时会引发 ValueError 异常
        assert_raises(ValueError, f, dfnum * 3, bad_dfden)

        # 重新创建一个随机数生成器，以确保测试独立性，使用相同的种子值
        random = Generator(MT19937(self.seed))
        # 获取 f 方法（F 分布函数）
        f = random.f
        # 使用给定的自由度参数生成 F 分布的实际结果
        actual = f(dfnum, dfden * 3)
        # 断言实际结果与期望结果的近似性，精度为小数点后14位
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言当自由度参数为负数时会引发 ValueError 异常
        assert_raises(ValueError, f, bad_dfnum, dfden * 3)
        # 断言当自由度参数为负数时会引发 ValueError 异常
        assert_raises(ValueError, f, dfnum, bad_dfden * 3)
    # 定义测试函数，用于测试非中心 F 分布的生成
    def test_noncentral_f(self):
        # 设定自由度参数列表
        dfnum = [2]
        dfden = [3]
        nonc = [4]
        # 不合法的自由度参数列表
        bad_dfnum = [0]
        bad_dfden = [-1]
        bad_nonc = [-2]
        # 期望得到的结果数组
        desired = np.array([2.02434240411421, 12.91838601070124, 1.24395160354629])

        # 使用特定种子生成随机数生成器
        random = Generator(MT19937(self.seed))
        # 获取生成非中心 F 分布的函数
        nonc_f = random.noncentral_f
        # 计算实际生成的非中心 F 分布，并与期望结果进行比较
        actual = nonc_f(dfnum * 3, dfden, nonc)
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言处理包含 NaN 值的情况
        assert np.all(np.isnan(nonc_f(dfnum, dfden, [np.nan] * 3)))

        # 断言处理不合法参数的情况
        assert_raises(ValueError, nonc_f, bad_dfnum * 3, dfden, nonc)
        assert_raises(ValueError, nonc_f, dfnum * 3, bad_dfden, nonc)
        assert_raises(ValueError, nonc_f, dfnum * 3, dfden, bad_nonc)

        # 重新设置随机数生成器和非中心 F 分布函数
        random = Generator(MT19937(self.seed))
        nonc_f = random.noncentral_f
        # 再次计算非中心 F 分布，并与期望结果进行比较
        actual = nonc_f(dfnum, dfden * 3, nonc)
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言处理不合法参数的情况
        assert_raises(ValueError, nonc_f, bad_dfnum, dfden * 3, nonc)
        assert_raises(ValueError, nonc_f, dfnum, bad_dfden * 3, nonc)
        assert_raises(ValueError, nonc_f, dfnum, dfden * 3, bad_nonc)

        # 重新设置随机数生成器和非中心 F 分布函数
        random = Generator(MT19937(self.seed))
        nonc_f = random.noncentral_f
        # 再次计算非中心 F 分布，并与期望结果进行比较
        actual = nonc_f(dfnum, dfden, nonc * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言处理不合法参数的情况
        assert_raises(ValueError, nonc_f, bad_dfnum, dfden, nonc * 3)
        assert_raises(ValueError, nonc_f, dfnum, bad_dfden, nonc * 3)
        assert_raises(ValueError, nonc_f, dfnum, dfden, bad_nonc * 3)

    # 定义测试函数，用于测试小自由度下的非中心 F 分布生成
    def test_noncentral_f_small_df(self):
        # 使用特定种子生成随机数生成器
        random = Generator(MT19937(self.seed))
        # 期望得到的结果数组
        desired = np.array([0.04714867120827, 0.1239390327694])
        # 计算实际生成的小自由度下的非中心 F 分布，并与期望结果进行比较
        actual = random.noncentral_f(0.9, 0.9, 2, size=2)
        assert_array_almost_equal(actual, desired, decimal=14)

    # 定义测试函数，用于测试卡方分布生成
    def test_chisquare(self):
        # 设定自由度参数列表
        df = [1]
        # 不合法的自由度参数列表
        bad_df = [-1]
        # 期望得到的结果数组
        desired = np.array([0.05573640064251, 1.47220224353539, 2.9469379318589])

        # 使用特定种子生成随机数生成器
        random = Generator(MT19937(self.seed))
        # 计算实际生成的卡方分布，并与期望结果进行比较
        actual = random.chisquare(df * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言处理不合法参数的情况
        assert_raises(ValueError, random.chisquare, bad_df * 3)
    def test_noncentral_chisquare(self):
        # 定义自由度和非中心参数的列表
        df = [1]
        nonc = [2]
        # 定义不合法的自由度和非中心参数列表
        bad_df = [-1]
        bad_nonc = [-2]
        # 期望结果
        desired = np.array([0.07710766249436, 5.27829115110304, 0.630732147399])

        # 使用指定种子创建随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 获取非中心卡方分布函数
        nonc_chi = random.noncentral_chisquare
        # 计算实际结果
        actual = nonc_chi(df * 3, nonc)
        # 断言实际结果与期望结果接近
        assert_array_almost_equal(actual, desired, decimal=14)
        # 检查是否引发值错误异常
        assert_raises(ValueError, nonc_chi, bad_df * 3, nonc)
        assert_raises(ValueError, nonc_chi, df * 3, bad_nonc)

        # 使用相同种子重新创建随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 获取非中心卡方分布函数
        nonc_chi = random.noncentral_chisquare
        # 计算实际结果
        actual = nonc_chi(df, nonc * 3)
        # 断言实际结果与期望结果接近
        assert_array_almost_equal(actual, desired, decimal=14)
        # 检查是否引发值错误异常
        assert_raises(ValueError, nonc_chi, bad_df, nonc * 3)

    def test_standard_t(self):
        # 定义自由度列表和不合法的自由度列表
        df = [1]
        bad_df = [-1]
        # 期望结果
        desired = np.array([-1.39498829447098, -1.23058658835223, 0.17207021065983])

        # 使用指定种子创建随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 计算标准 t 分布函数
        actual = random.standard_t(df * 3)
        # 断言实际结果与期望结果接近
        assert_array_almost_equal(actual, desired, decimal=14)
        # 检查是否引发值错误异常
        assert_raises(ValueError, random.standard_t, bad_df * 3)

    def test_vonmises(self):
        # 定义均值和 kappa 参数列表
        mu = [2]
        kappa = [1]
        # 定义不合法的 kappa 参数列表
        bad_kappa = [-1]
        # 期望结果
        desired = np.array([2.25935584988528, 2.23326261461399, -2.84152146503326])

        # 使用指定种子创建随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 计算 von Mises 分布函数
        actual = random.vonmises(mu * 3, kappa)
        # 断言实际结果与期望结果接近
        assert_array_almost_equal(actual, desired, decimal=14)
        # 检查是否引发值错误异常
        assert_raises(ValueError, random.vonmises, mu * 3, bad_kappa)

        # 使用相同种子重新创建随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 计算 von Mises 分布函数
        actual = random.vonmises(mu, kappa * 3)
        # 断言实际结果与期望结果接近
        assert_array_almost_equal(actual, desired, decimal=14)
        # 检查是否引发值错误异常
        assert_raises(ValueError, random.vonmises, mu, bad_kappa * 3)

    def test_pareto(self):
        # 定义参数 a 的列表和不合法的参数 a 的列表
        a = [1]
        bad_a = [-1]
        # 期望结果
        desired = np.array([0.95905052946317, 0.2383810889437 , 1.04988745750013])

        # 使用指定种子创建随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 计算 Pareto 分布函数
        actual = random.pareto(a * 3)
        # 断言实际结果与期望结果接近
        assert_array_almost_equal(actual, desired, decimal=14)
        # 检查是否引发值错误异常
        assert_raises(ValueError, random.pareto, bad_a * 3)

    def test_weibull(self):
        # 定义参数 a 的列表和不合法的参数 a 的列表
        a = [1]
        bad_a = [-1]
        # 期望结果
        desired = np.array([0.67245993212806, 0.21380495318094, 0.7177848928629])

        # 使用指定种子创建随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 计算 Weibull 分布函数
        actual = random.weibull(a * 3)
        # 断言实际结果与期望结果接近
        assert_array_almost_equal(actual, desired, decimal=14)
        # 检查是否引发值错误异常
        assert_raises(ValueError, random.weibull, bad_a * 3)

    def test_power(self):
        # 定义参数 a 的列表和不合法的参数 a 的列表
        a = [1]
        bad_a = [-1]
        # 期望结果
        desired = np.array([0.48954864361052, 0.19249412888486, 0.51216834058807])

        # 使用指定种子创建随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 计算 power 分布函数
        actual = random.power(a * 3)
        # 断言实际结果与期望结果接近
        assert_array_almost_equal(actual, desired, decimal=14)
        # 检查是否引发值错误异常
        assert_raises(ValueError, random.power, bad_a * 3)
    def test_laplace(self):
        # 定义位置参数 loc 和尺度参数 scale
        loc = [0]
        scale = [1]
        # 定义一个错误的尺度参数 bad_scale
        bad_scale = [-1]
        # 期望的结果数组
        desired = np.array([-1.09698732625119, -0.93470271947368, 0.71592671378202])

        # 使用给定的随机数生成器种子创建随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 获取拉普拉斯分布函数
        laplace = random.laplace
        # 使用 loc * 3 和 scale 调用拉普拉斯分布函数，得到实际结果
        actual = laplace(loc * 3, scale)
        # 断言实际结果与期望结果的近似程度
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言使用错误尺度参数调用函数会抛出 ValueError 异常
        assert_raises(ValueError, laplace, loc * 3, bad_scale)

        # 重新使用相同的随机数生成器种子创建随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 获取拉普拉斯分布函数
        laplace = random.laplace
        # 使用 loc 和 scale * 3 调用拉普拉斯分布函数，得到实际结果
        actual = laplace(loc, scale * 3)
        # 断言实际结果与期望结果的近似程度
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言使用错误尺度参数调用函数会抛出 ValueError 异常
        assert_raises(ValueError, laplace, loc, bad_scale * 3)

    def test_gumbel(self):
        # 定义位置参数 loc 和尺度参数 scale
        loc = [0]
        scale = [1]
        # 定义一个错误的尺度参数 bad_scale
        bad_scale = [-1]
        # 期望的结果数组
        desired = np.array([1.70020068231762, 1.52054354273631, -0.34293267607081])

        # 使用给定的随机数生成器种子创建随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 获取古贝尔分布函数
        gumbel = random.gumbel
        # 使用 loc * 3 和 scale 调用古贝尔分布函数，得到实际结果
        actual = gumbel(loc * 3, scale)
        # 断言实际结果与期望结果的近似程度
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言使用错误尺度参数调用函数会抛出 ValueError 异常
        assert_raises(ValueError, gumbel, loc * 3, bad_scale)

        # 重新使用相同的随机数生成器种子创建随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 获取古贝尔分布函数
        gumbel = random.gumbel
        # 使用 loc 和 scale * 3 调用古贝尔分布函数，得到实际结果
        actual = gumbel(loc, scale * 3)
        # 断言实际结果与期望结果的近似程度
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言使用错误尺度参数调用函数会抛出 ValueError 异常
        assert_raises(ValueError, gumbel, loc, bad_scale * 3)

    def test_logistic(self):
        # 定义位置参数 loc 和尺度参数 scale
        loc = [0]
        scale = [1]
        # 定义一个错误的尺度参数 bad_scale
        bad_scale = [-1]
        # 期望的结果数组
        desired = np.array([-1.607487640433, -1.40925686003678, 1.12887112820397])

        # 使用给定的随机数生成器种子创建随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 使用 loc * 3 和 scale 调用 logistic 函数，得到实际结果
        actual = random.logistic(loc * 3, scale)
        # 断言实际结果与期望结果的近似程度
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言使用错误尺度参数调用函数会抛出 ValueError 异常
        assert_raises(ValueError, random.logistic, loc * 3, bad_scale)

        # 重新使用相同的随机数生成器种子创建随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 使用 loc 和 scale * 3 调用 logistic 函数，得到实际结果
        actual = random.logistic(loc, scale * 3)
        # 断言实际结果与期望结果的近似程度
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言使用错误尺度参数调用函数会抛出 ValueError 异常
        assert_raises(ValueError, random.logistic, loc, bad_scale * 3)
        # 断言 logistic(1.0, 0.0) 应该返回 1.0
        assert_equal(random.logistic(1.0, 0.0), 1.0)

    def test_lognormal(self):
        # 定义均值参数 mean 和标准差参数 sigma
        mean = [0]
        sigma = [1]
        # 定义一个错误的标准差参数 bad_sigma
        bad_sigma = [-1]
        # 期望的结果数组
        desired = np.array([0.67884390500697, 2.21653186290321, 1.01990310084276])

        # 使用给定的随机数生成器种子创建随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 获取对数正态分布函数
        lognormal = random.lognormal
        # 使用 mean * 3 和 sigma 调用对数正态分布函数，得到实际结果
        actual = lognormal(mean * 3, sigma)
        # 断言实际结果与期望结果的近似程度
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言使用错误标准差参数调用函数会抛出 ValueError 异常
        assert_raises(ValueError, lognormal, mean * 3, bad_sigma)

        # 重新使用相同的随机数生成器种子创建随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 使用 mean 和 sigma * 3 调用对数正态分布函数，得到实际结果
        actual = lognormal(mean, sigma * 3)
        # 断言使用错误标准差参数调用函数会抛出 ValueError 异常
        assert_raises(ValueError, lognormal, mean, bad_sigma * 3)
    # 定义测试 Rayleigh 分布生成的方法
    def test_rayleigh(self):
        # 设置 Rayleigh 分布的尺度参数
        scale = [1]
        # 设置一个无效的尺度参数（负数）
        bad_scale = [-1]
        # 预期的生成结果
        desired = np.array(
            [1.1597068009872629,
             0.6539188836253857,
             1.1981526554349398]
        )

        # 使用特定种子创建随机数生成器
        random = Generator(MT19937(self.seed))
        # 生成 Rayleigh 分布的随机数并与预期结果进行比较
        actual = random.rayleigh(scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        # 检查是否会抛出尺度参数为负数的异常
        assert_raises(ValueError, random.rayleigh, bad_scale * 3)

    # 定义测试 Wald 分布生成的方法
    def test_wald(self):
        # 设置 Wald 分布的均值参数
        mean = [0.5]
        # 设置 Wald 分布的尺度参数
        scale = [1]
        # 设置一个无效的均值参数（零）
        bad_mean = [0]
        # 设置一个无效的尺度参数（负数）
        bad_scale = [-2]
        # 预期的生成结果
        desired = np.array([0.38052407392905, 0.50701641508592, 0.484935249864])

        # 使用特定种子创建随机数生成器
        random = Generator(MT19937(self.seed))
        # 生成 Wald 分布的随机数并与预期结果进行比较
        actual = random.wald(mean * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        # 检查是否会抛出均值参数为零的异常
        assert_raises(ValueError, random.wald, bad_mean * 3, scale)
        # 检查是否会抛出尺度参数为负数的异常
        assert_raises(ValueError, random.wald, mean * 3, bad_scale)

        # 使用特定种子创建随机数生成器
        random = Generator(MT19937(self.seed))
        # 生成 Wald 分布的随机数并与预期结果进行比较
        actual = random.wald(mean, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        # 检查是否会抛出均值参数为零的异常
        assert_raises(ValueError, random.wald, bad_mean, scale * 3)
        # 检查是否会抛出尺度参数为负数的异常
        assert_raises(ValueError, random.wald, mean, bad_scale * 3)
    def test_triangular(self):
        left = [1]  # 左边界列表
        right = [3]  # 右边界列表
        mode = [2]  # 模式列表
        bad_left_one = [3]  # 错误的左边界列表示例
        bad_mode_one = [4]  # 错误的模式列表示例
        bad_left_two, bad_mode_two = right * 2  # 错误的左边界和模式示例
        desired = np.array([1.57781954604754, 1.62665986867957, 2.30090130831326])  # 期望结果数组

        random = Generator(MT19937(self.seed))  # 创建随机数生成器对象
        triangular = random.triangular  # 获取三角分布生成方法
        actual = triangular(left * 3, mode, right)  # 调用三角分布生成方法并计算结果
        assert_array_almost_equal(actual, desired, decimal=14)  # 断言实际结果与期望结果接近

        # 以下三个断言测试函数调用与错误输入的组合
        assert_raises(ValueError, triangular, bad_left_one * 3, mode, right)
        assert_raises(ValueError, triangular, left * 3, bad_mode_one, right)
        assert_raises(ValueError, triangular, bad_left_two * 3, bad_mode_two, right)

        # 重置随机数生成器并再次进行测试
        random = Generator(MT19937(self.seed))
        triangular = random.triangular
        actual = triangular(left, mode * 3, right)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, triangular, bad_left_one, mode * 3, right)
        assert_raises(ValueError, triangular, left, bad_mode_one * 3, right)
        assert_raises(ValueError, triangular, bad_left_two, bad_mode_two * 3, right)

        # 重置随机数生成器并再次进行测试
        random = Generator(MT19937(self.seed))
        triangular = random.triangular
        actual = triangular(left, mode, right * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, triangular, bad_left_one, mode, right * 3)
        assert_raises(ValueError, triangular, left, bad_mode_one, right * 3)
        assert_raises(ValueError, triangular, bad_left_two, bad_mode_two, right * 3)

        # 额外的断言测试特定的数值输入
        assert_raises(ValueError, triangular, 10., 0., 20.)
        assert_raises(ValueError, triangular, 10., 25., 20.)
        assert_raises(ValueError, triangular, 10., 10., 10.)

    def test_binomial(self):
        n = [1]  # n 参数列表
        p = [0.5]  # p 参数列表
        bad_n = [-1]  # 错误的 n 参数列表示例
        bad_p_one = [-1]  # 错误的 p 参数列表示例
        bad_p_two = [1.5]  # 错误的 p 参数列表示例
        desired = np.array([0, 0, 1])  # 期望结果数组

        random = Generator(MT19937(self.seed))  # 创建随机数生成器对象
        binom = random.binomial  # 获取二项分布生成方法
        actual = binom(n * 3, p)  # 调用二项分布生成方法并计算结果
        assert_array_equal(actual, desired)  # 断言实际结果与期望结果相等

        # 以下三个断言测试函数调用与错误输入的组合
        assert_raises(ValueError, binom, bad_n * 3, p)
        assert_raises(ValueError, binom, n * 3, bad_p_one)
        assert_raises(ValueError, binom, n * 3, bad_p_two)

        # 重置随机数生成器并再次进行测试
        random = Generator(MT19937(self.seed))
        actual = random.binomial(n, p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, binom, bad_n, p * 3)
        assert_raises(ValueError, binom, n, bad_p_one * 3)
        assert_raises(ValueError, binom, n, bad_p_two * 3)
    # 定义测试负二项分布的函数
    def test_negative_binomial(self):
        # 设置参数 n 和 p
        n = [1]
        p = [0.5]
        # 定义错误的 n 和 p 值
        bad_n = [-1]
        bad_p_one = [-1]
        bad_p_two = [1.5]
        # 期望的结果数组
        desired = np.array([0, 2, 1], dtype=np.int64)

        # 使用指定种子生成随机数发生器
        random = Generator(MT19937(self.seed))
        # 获取负二项分布函数
        neg_binom = random.negative_binomial
        # 计算实际结果
        actual = neg_binom(n * 3, p)
        # 检查实际结果与期望结果是否一致
        assert_array_equal(actual, desired)
        # 检查对错误输入的处理，预期抛出 ValueError 异常
        assert_raises(ValueError, neg_binom, bad_n * 3, p)
        assert_raises(ValueError, neg_binom, n * 3, bad_p_one)
        assert_raises(ValueError, neg_binom, n * 3, bad_p_two)

        # 重复上述步骤，但交换 n 和 p 的位置
        random = Generator(MT19937(self.seed))
        neg_binom = random.negative_binomial
        actual = neg_binom(n, p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, neg_binom, bad_n, p * 3)
        assert_raises(ValueError, neg_binom, n, bad_p_one * 3)
        assert_raises(ValueError, neg_binom, n, bad_p_two * 3)

    # 定义测试泊松分布的函数
    def test_poisson(self):

        # 设置参数 lambda
        lam = [1]
        # 定义错误的 lambda 值
        bad_lam_one = [-1]
        # 期望的结果数组
        desired = np.array([0, 0, 3])

        # 使用指定种子生成随机数发生器
        random = Generator(MT19937(self.seed))
        # 获取泊松分布函数和最大 lambda 值
        max_lam = random._poisson_lam_max
        bad_lam_two = [max_lam * 2]
        poisson = random.poisson
        # 计算实际结果
        actual = poisson(lam * 3)
        # 检查实际结果与期望结果是否一致
        assert_array_equal(actual, desired)
        # 检查对错误输入的处理，预期抛出 ValueError 异常
        assert_raises(ValueError, poisson, bad_lam_one * 3)
        assert_raises(ValueError, poisson, bad_lam_two * 3)

    # 定义测试 Zipf 分布的函数
    def test_zipf(self):
        # 设置参数 a
        a = [2]
        # 定义错误的 a 值
        bad_a = [0]
        # 期望的结果数组
        desired = np.array([1, 8, 1])

        # 使用指定种子生成随机数发生器
        random = Generator(MT19937(self.seed))
        # 获取 Zipf 分布函数
        zipf = random.zipf
        # 计算实际结果
        actual = zipf(a * 3)
        # 检查实际结果与期望结果是否一致
        assert_array_equal(actual, desired)
        # 检查对错误输入的处理，预期抛出 ValueError 异常
        assert_raises(ValueError, zipf, bad_a * 3)
        # 在忽略无效值错误的情况下，检查对无效输入的处理
        with np.errstate(invalid='ignore'):
            assert_raises(ValueError, zipf, np.nan)
            assert_raises(ValueError, zipf, [0, 0, np.nan])

    # 定义测试几何分布的函数
    def test_geometric(self):
        # 设置参数 p
        p = [0.5]
        # 定义错误的 p 值
        bad_p_one = [-1]
        bad_p_two = [1.5]
        # 期望的结果数组
        desired = np.array([1, 1, 3])

        # 使用指定种子生成随机数发生器
        random = Generator(MT19937(self.seed))
        # 获取几何分布函数
        geometric = random.geometric
        # 计算实际结果
        actual = geometric(p * 3)
        # 检查实际结果与期望结果是否一致
        assert_array_equal(actual, desired)
        # 检查对错误输入的处理，预期抛出 ValueError 异常
        assert_raises(ValueError, geometric, bad_p_one * 3)
        assert_raises(ValueError, geometric, bad_p_two * 3)
    # 定义一个测试函数，用于测试超几何分布函数的行为
    def test_hypergeometric(self):
        # 设定三个列表，分别表示好的物品数量、坏的物品数量、样本抽取数量
        ngood = [1]
        nbad = [2]
        nsample = [2]
        # 定义一些不合法的参数组合，用于测试函数抛出 ValueError 的情况
        bad_ngood = [-1]
        bad_nbad = [-2]
        bad_nsample_one = [-1]
        bad_nsample_two = [4]
        # 期望的输出结果，一个 NumPy 数组
        desired = np.array([0, 0, 1])

        # 使用 Mersenne Twister 19937 生成器初始化一个随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 调用随机数生成器对象的超几何分布方法，生成实际结果
        actual = random.hypergeometric(ngood * 3, nbad, nsample)
        # 断言实际输出结果与期望结果相等
        assert_array_equal(actual, desired)
        # 检查是否抛出 ValueError 异常，测试不合法参数 ngood * 3
        assert_raises(ValueError, random.hypergeometric, bad_ngood * 3, nbad, nsample)
        # 检查是否抛出 ValueError 异常，测试不合法参数 nbad
        assert_raises(ValueError, random.hypergeometric, ngood * 3, bad_nbad, nsample)
        # 检查是否抛出 ValueError 异常，测试不合法参数 nsample (-1)
        assert_raises(ValueError, random.hypergeometric, ngood * 3, nbad, bad_nsample_one)
        # 检查是否抛出 ValueError 异常，测试不合法参数 nsample (4)
        assert_raises(ValueError, random.hypergeometric, ngood * 3, nbad, bad_nsample_two)

        # 重新初始化随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 调用随机数生成器对象的超几何分布方法，生成实际结果
        actual = random.hypergeometric(ngood, nbad * 3, nsample)
        # 断言实际输出结果与期望结果相等
        assert_array_equal(actual, desired)
        # 检查是否抛出 ValueError 异常，测试不合法参数 ngood
        assert_raises(ValueError, random.hypergeometric, bad_ngood, nbad * 3, nsample)
        # 检查是否抛出 ValueError 异常，测试不合法参数 nbad * 3
        assert_raises(ValueError, random.hypergeometric, ngood, bad_nbad * 3, nsample)
        # 检查是否抛出 ValueError 异常，测试不合法参数 nsample (-1)
        assert_raises(ValueError, random.hypergeometric, ngood, nbad * 3, bad_nsample_one)
        # 检查是否抛出 ValueError 异常，测试不合法参数 nsample (4)
        assert_raises(ValueError, random.hypergeometric, ngood, nbad * 3, bad_nsample_two)

        # 重新初始化随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 将随机数生成器对象的超几何分布方法赋值给一个变量
        hypergeom = random.hypergeometric
        # 调用超几何分布方法，生成实际结果
        actual = hypergeom(ngood, nbad, nsample * 3)
        # 断言实际输出结果与期望结果相等
        assert_array_equal(actual, desired)
        # 检查是否抛出 ValueError 异常，测试不合法参数 ngood
        assert_raises(ValueError, hypergeom, bad_ngood, nbad, nsample * 3)
        # 检查是否抛出 ValueError 异常，测试不合法参数 nbad
        assert_raises(ValueError, hypergeom, ngood, bad_nbad, nsample * 3)
        # 检查是否抛出 ValueError 异常，测试不合法参数 nsample (-1)
        assert_raises(ValueError, hypergeom, ngood, nbad, bad_nsample_one * 3)
        # 检查是否抛出 ValueError 异常，测试不合法参数 nsample (4)
        assert_raises(ValueError, hypergeom, ngood, nbad, bad_nsample_two * 3)

        # 检查是否抛出 ValueError 异常，测试参数 ngood 为负数
        assert_raises(ValueError, hypergeom, -1, 10, 20)
        # 检查是否抛出 ValueError 异常，测试参数 nbad 为负数
        assert_raises(ValueError, hypergeom, 10, -1, 20)
        # 检查是否抛出 ValueError 异常，测试参数 nsample 为负数
        assert_raises(ValueError, hypergeom, 10, 10, -1)
        # 检查是否抛出 ValueError 异常，测试参数 nsample 大于 ngood + nbad
        assert_raises(ValueError, hypergeom, 10, 10, 25)

        # 检查是否抛出 ValueError 异常，测试参数 ngood 超出范围
        assert_raises(ValueError, hypergeom, 2**30, 10, 20)
        # 检查是否抛出 ValueError 异常，测试参数 nbad 超出范围
        assert_raises(ValueError, hypergeom, 999, 2**31, 50)
        # 检查是否抛出 ValueError 异常，测试参数 nsample 超出范围
        assert_raises(ValueError, hypergeom, 999, [2**29, 2**30], 1000)

    # 定义一个测试函数，用于测试对数系列分布函数的行为
    def test_logseries(self):
        # 设定一个列表，表示概率 p
        p = [0.5]
        # 定义一些不合法的参数组合，用于测试函数抛出 ValueError 的情况
        bad_p_one = [2]
        bad_p_two = [-1]
        # 期望的输出结果，一个 NumPy 数组
        desired = np.array([1, 1, 1])

        # 使用 Mersenne Twister 19937 生成器初始化一个随机数生成器对象
        random = Generator(MT19937(self.seed))
        # 将随机数生成器对象的对数系列分布方法赋值给一个变量
        logseries = random.logseries
        # 调用对数系列分布方法，生成实际结果
        actual = logseries(p * 3)
        # 断言实际输出结果与期望结果相等
        assert_array_equal(actual, desired)
        # 检查是否抛出 ValueError 异常，测试不合法参数 p (2)
        assert_raises(ValueError, logseries, bad_p_one * 3)
        # 检查是否抛出 ValueError 异常，测试不合法参数 p (-1)
        assert_raises(ValueError, logseries, bad_p_two * 3)
    # 定义测试函数 test_multinomial，用于测试随机数生成器的多项式分布采样功能
    def test_multinomial(self):
        # 使用给定种子初始化随机数生成器
        random = Generator(MT19937(self.seed))
        # 生成指定参数下的多项式分布样本
        actual = random.multinomial([5, 20], [1 / 6.] * 6, size=(3, 2))
        # 预期的多项式分布样本结果
        desired = np.array([[[0, 0, 2, 1, 2, 0],
                             [2, 3, 6, 4, 2, 3]],
                            [[1, 0, 1, 0, 2, 1],
                             [7, 2, 2, 1, 4, 4]],
                            [[0, 2, 0, 1, 2, 0],
                             [3, 2, 3, 3, 4, 5]]], dtype=np.int64)
        # 断言生成的实际结果与预期结果一致
        assert_array_equal(actual, desired)

        # 重置随机数生成器
        random = Generator(MT19937(self.seed))
        # 生成指定参数下的多项式分布样本
        actual = random.multinomial([5, 20], [1 / 6.] * 6)
        # 预期的多项式分布样本结果
        desired = np.array([[0, 0, 2, 1, 2, 0],
                            [2, 3, 6, 4, 2, 3]], dtype=np.int64)
        # 断言生成的实际结果与预期结果一致
        assert_array_equal(actual, desired)

        # 重置随机数生成器
        random = Generator(MT19937(self.seed))
        # 生成指定参数下的多项式分布样本
        actual = random.multinomial([5, 20], [[1 / 6.] * 6] * 2)
        # 预期的多项式分布样本结果
        desired = np.array([[0, 0, 2, 1, 2, 0],
                            [2, 3, 6, 4, 2, 3]], dtype=np.int64)
        # 断言生成的实际结果与预期结果一致
        assert_array_equal(actual, desired)

        # 重置随机数生成器
        random = Generator(MT19937(self.seed))
        # 生成指定参数下的多项式分布样本
        actual = random.multinomial([[5], [20]], [[1 / 6.] * 6] * 2)
        # 预期的多项式分布样本结果
        desired = np.array([[[0, 0, 2, 1, 2, 0],
                             [0, 0, 2, 1, 1, 1]],
                            [[4, 2, 3, 3, 5, 3],
                             [7, 2, 2, 1, 4, 4]]], dtype=np.int64)
        # 断言生成的实际结果与预期结果一致
        assert_array_equal(actual, desired)

    # 使用 pytest 的参数化装饰器定义测试函数 test_multinomial_pval_broadcast
    @pytest.mark.parametrize("n", [10,
                                   np.array([10, 10]),
                                   np.array([[[10]], [[10]]])
                                   ]
                             )
    def test_multinomial_pval_broadcast(self, n):
        # 使用给定种子初始化随机数生成器
        random = Generator(MT19937(self.seed))
        # 定义概率数组
        pvals = np.array([1 / 4] * 4)
        # 生成指定参数下的多项式分布样本
        actual = random.multinomial(n, pvals)
        # 检查实际输出的形状是否与预期一致
        n_shape = tuple() if isinstance(n, int) else n.shape
        expected_shape = n_shape + (4,)
        assert actual.shape == expected_shape
        # 扩展概率数组的维度
        pvals = np.vstack([pvals, pvals])
        # 生成指定参数下的多项式分布样本
        actual = random.multinomial(n, pvals)
        # 计算扩展后的预期形状
        expected_shape = np.broadcast_shapes(n_shape, pvals.shape[:-1]) + (4,)
        assert actual.shape == expected_shape

        # 再次扩展概率数组的维度
        pvals = np.vstack([[pvals], [pvals]])
        # 生成指定参数下的多项式分布样本
        actual = random.multinomial(n, pvals)
        # 计算扩展后的预期形状
        expected_shape = np.broadcast_shapes(n_shape, pvals.shape[:-1])
        assert actual.shape == expected_shape + (4,)
        # 生成指定参数下的多项式分布样本，并指定输出数组的形状
        actual = random.multinomial(n, pvals, size=(3, 2) + expected_shape)
        assert actual.shape == (3, 2) + expected_shape + (4,)

        # 使用 pytest 的异常断言，确保在指定情况下引发 ValueError 异常
        with pytest.raises(ValueError):
            # 确保不能广播 size 参数
            actual = random.multinomial(n, pvals, size=(1,) * 6)

    # 定义测试函数 test_invalid_pvals_broadcast，测试无效概率数组的广播行为
    def test_invalid_pvals_broadcast(self):
        # 使用给定种子初始化随机数生成器
        random = Generator(MT19937(self.seed))
        # 定义无效的概率数组
        pvals = [[1 / 6] * 6, [1 / 4] * 6]
        # 断言在给定无效概率数组时，调用多项式分布方法会引发 ValueError 异常
        assert_raises(ValueError, random.multinomial, 1, pvals)
        assert_raises(ValueError, random.multinomial, 6, 0.5)
    # 定义一个测试方法，用于测试生成空输出的情况
    def test_empty_outputs(self):
        # 使用给定的种子初始化 Mersenne Twister 伪随机数生成器
        random = Generator(MT19937(self.seed))
        # 生成一个多项分布样本，输入的数据是一个空的 3 维数组，每个元素为 int64 类型
        actual = random.multinomial(np.empty((10, 0, 6), "i8"), [1 / 6] * 6)
        # 断言生成的样本形状应为 (10, 0, 6, 6)
        assert actual.shape == (10, 0, 6, 6)
        # 生成一个多项分布样本，输入的参数是一个整数 12 和一个空的 3 维数组
        actual = random.multinomial(12, np.empty((10, 0, 10)))
        # 断言生成的样本形状应为 (10, 0, 10)
        assert actual.shape == (10, 0, 10)
        # 生成一个多项分布样本，输入的数据分别是两个空的 4 维数组
        actual = random.multinomial(np.empty((3, 0, 7), "i8"),
                                    np.empty((3, 0, 7, 4)))
        # 断言生成的样本形状应为 (3, 0, 7, 4)
        assert actual.shape == (3, 0, 7, 4)
# 使用 pytest.mark.skipif 标记，如果 IS_WASM 为真，则跳过测试用例，理由是无法启动线程
@pytest.mark.skipif(IS_WASM, reason="can't start thread")
class TestThread:
    # 确保即使在线程中，每个状态也能产生相同的序列
    def setup_method(self):
        # 初始化种子范围为 [0, 1, 2, 3]
        self.seeds = range(4)

    def check_function(self, function, sz):
        from threading import Thread

        # 创建两个空数组，用于存放线程生成的输出
        out1 = np.empty((len(self.seeds),) + sz)
        out2 = np.empty((len(self.seeds),) + sz)

        # 多线程生成
        t = [Thread(target=function, args=(Generator(MT19937(s)), o))
             for s, o in zip(self.seeds, out1)]
        [x.start() for x in t]  # 启动所有线程
        [x.join() for x in t]   # 等待所有线程完成

        # 串行生成
        for s, o in zip(self.seeds, out2):
            function(Generator(MT19937(s)), o)

        # 检查结果是否一致，根据平台和数据类型进行精度比较
        if np.intp().dtype.itemsize == 4 and sys.platform == "win32":
            assert_array_almost_equal(out1, out2)
        else:
            assert_array_equal(out1, out2)

    def test_normal(self):
        def gen_random(state, out):
            out[...] = state.normal(size=10000)

        self.check_function(gen_random, sz=(10000,))

    def test_exp(self):
        def gen_random(state, out):
            out[...] = state.exponential(scale=np.ones((100, 1000)))

        self.check_function(gen_random, sz=(100, 1000))

    def test_multinomial(self):
        def gen_random(state, out):
            out[...] = state.multinomial(10, [1 / 6.] * 6, size=10000)

        self.check_function(gen_random, sz=(10000, 6))


# 见问题 #4263
class TestSingleEltArrayInput:
    def setup_method(self):
        # 初始化测试所需的数组和形状
        self.argOne = np.array([2])
        self.argTwo = np.array([3])
        self.argThree = np.array([4])
        self.tgtShape = (1,)

    def test_one_arg_funcs(self):
        # 需要测试的函数列表
        funcs = (random.exponential, random.standard_gamma,
                 random.chisquare, random.standard_t,
                 random.pareto, random.weibull,
                 random.power, random.rayleigh,
                 random.poisson, random.zipf,
                 random.geometric, random.logseries)

        # 需要传递概率参数的函数列表
        probfuncs = (random.geometric, random.logseries)

        # 对每个函数进行测试
        for func in funcs:
            if func in probfuncs:  # 如果是概率函数，传递参数 0.5
                out = func(np.array([0.5]))
            else:
                out = func(self.argOne)  # 否则传递预设的输入参数

            # 检查输出的形状是否符合预期
            assert_equal(out.shape, self.tgtShape)
    # 测试包含两个参数的随机函数
    def test_two_arg_funcs(self):
        # 定义一组包含随机函数的元组
        funcs = (random.uniform, random.normal,
                 random.beta, random.gamma,
                 random.f, random.noncentral_chisquare,
                 random.vonmises, random.laplace,
                 random.gumbel, random.logistic,
                 random.lognormal, random.wald,
                 random.binomial, random.negative_binomial)
    
        # 只包含两个参数的随机函数
        probfuncs = (random.binomial, random.negative_binomial)
    
        # 遍历每个随机函数
        for func in funcs:
            # 如果当前函数是概率分布函数
            if func in probfuncs:  # p <= 1
                # 设置第二个参数为固定的数组 [0.5]
                argTwo = np.array([0.5])
            else:
                # 否则，使用预定义的第二个参数
                argTwo = self.argTwo
    
            # 调用随机函数 func，并传入两个参数
            out = func(self.argOne, argTwo)
            # 断言输出的形状与预期相同
            assert_equal(out.shape, self.tgtShape)
    
            # 调用随机函数 func，传入第一个参数的第一个元素和第二个参数
            out = func(self.argOne[0], argTwo)
            # 断言输出的形状与预期相同
            assert_equal(out.shape, self.tgtShape)
    
            # 调用随机函数 func，传入第一个参数和第二个参数的第一个元素
            out = func(self.argOne, argTwo[0])
            # 断言输出的形状与预期相同
            assert_equal(out.shape, self.tgtShape)
    
    # 测试整数生成函数
    def test_integers(self, endpoint):
        # 定义整数类型的列表
        itype = [np.bool, np.int8, np.uint8, np.int16, np.uint16,
                 np.int32, np.uint32, np.int64, np.uint64]
        # 使用 random.integers 函数
        func = random.integers
        # 设置高位和低位的数组
        high = np.array([1])
        low = np.array([0])
    
        # 遍历每种整数类型
        for dt in itype:
            # 调用整数生成函数 func，并传入低位、高位、以及其他参数
            out = func(low, high, endpoint=endpoint, dtype=dt)
            # 断言输出的形状与预期相同
            assert_equal(out.shape, self.tgtShape)
    
            # 调用整数生成函数 func，传入低位的第一个元素、高位、以及其他参数
            out = func(low[0], high, endpoint=endpoint, dtype=dt)
            # 断言输出的形状与预期相同
            assert_equal(out.shape, self.tgtShape)
    
            # 调用整数生成函数 func，传入低位、高位的第一个元素、以及其他参数
            out = func(low, high[0], endpoint=endpoint, dtype=dt)
            # 断言输出的形状与预期相同
            assert_equal(out.shape, self.tgtShape)
    
    # 测试包含三个参数的随机函数
    def test_three_arg_funcs(self):
        # 定义包含三个参数的随机函数列表
        funcs = [random.noncentral_f, random.triangular,
                 random.hypergeometric]
    
        # 遍历每个随机函数
        for func in funcs:
            # 调用随机函数 func，并传入三个参数
            out = func(self.argOne, self.argTwo, self.argThree)
            # 断言输出的形状与预期相同
            assert_equal(out.shape, self.tgtShape)
    
            # 调用随机函数 func，传入第一个参数的第一个元素、第二个和第三个参数
            out = func(self.argOne[0], self.argTwo, self.argThree)
            # 断言输出的形状与预期相同
            assert_equal(out.shape, self.tgtShape)
    
            # 调用随机函数 func，传入第一个和第二个参数的第一个元素、第三个参数
            out = func(self.argOne, self.argTwo[0], self.argThree)
            # 断言输出的形状与预期相同
            assert_equal(out.shape, self.tgtShape)
# 参数化测试函数，使用JUMP_TEST_DATA中的每个配置项作为输入参数
@pytest.mark.parametrize("config", JUMP_TEST_DATA)
def test_jumped(config):
    # 从配置中获取初始种子和步数
    seed = config["seed"]
    steps = config["steps"]

    # 使用MT19937算法初始化生成器
    mt19937 = MT19937(seed)
    
    # 执行指定步数的随机数生成操作（模拟燃烧步骤）
    mt19937.random_raw(steps)
    
    # 获取当前状态的密钥并进行字节序转换（若系统字节序为大端）
    key = mt19937.state["state"]["key"]
    if sys.byteorder == 'big':
        key = key.byteswap()
    
    # 计算密钥的SHA256哈希值
    sha256 = hashlib.sha256(key)
    
    # 断言当前状态的位置与预期初始状态中的位置相同
    assert mt19937.state["state"]["pos"] == config["initial"]["pos"]
    
    # 断言计算得到的SHA256哈希值与预期初始状态中的密钥的SHA256哈希值相同
    assert sha256.hexdigest() == config["initial"]["key_sha256"]

    # 执行MT19937算法的jumped方法，获取跳跃后的生成器状态
    jumped = mt19937.jumped()
    
    # 获取跳跃后状态的密钥并进行字节序转换（若系统字节序为大端）
    key = jumped.state["state"]["key"]
    if sys.byteorder == 'big':
        key = key.byteswap()
    
    # 计算密钥的SHA256哈希值
    sha256 = hashlib.sha256(key)
    
    # 断言跳跃后状态的位置与预期跳跃后状态中的位置相同
    assert jumped.state["state"]["pos"] == config["jumped"]["pos"]
    
    # 断言计算得到的SHA256哈希值与预期跳跃后状态中的密钥的SHA256哈希值相同
    assert sha256.hexdigest() == config["jumped"]["key_sha256"]


# 测试当随机数生成函数参数化时，是否能够正确引发异常
def test_broadcast_size_error():
    # 创建形状为(3,)的均值向量mu和形状为(4,3)的标准差矩阵sigma
    mu = np.ones(3)
    sigma = np.ones((4, 3))
    
    # 定义大小为(10,4,2)的期望大小，测试是否正确引发异常
    size = (10, 4, 2)
    assert random.normal(mu, sigma, size=(5, 4, 3)).shape == (5, 4, 3)
    
    # 使用pytest检查是否正确引发大小不匹配的异常
    with pytest.raises(ValueError):
        random.normal(mu, sigma, size=size)
    with pytest.raises(ValueError):
        random.normal(mu, sigma, size=(1, 3))
    with pytest.raises(ValueError):
        random.normal(mu, sigma, size=(4, 1, 1))
    
    # 测试只有一个参数时是否正确引发异常
    shape = np.ones((4, 3))
    with pytest.raises(ValueError):
        random.standard_gamma(shape, size=size)
    with pytest.raises(ValueError):
        random.standard_gamma(shape, size=(3,))
    with pytest.raises(ValueError):
        random.standard_gamma(shape, size=3)
    
    # 检查是否正确引发输出参数异常
    out = np.empty(size)
    with pytest.raises(ValueError):
        random.standard_gamma(shape, out=out)
    
    # 检查多项式分布中参数不匹配的异常引发情况
    with pytest.raises(ValueError):
        random.binomial(1, [0.3, 0.7], size=(2, 1))
    with pytest.raises(ValueError):
        random.binomial([1, 2], 0.3, size=(2, 1))
    with pytest.raises(ValueError):
        random.binomial([1, 2], [0.3, 0.7], size=(2, 1))
    with pytest.raises(ValueError):
        random.multinomial([2, 2], [.3, .7], size=(2, 1))
    
    # 使用chi-square分布创建a、b、c，并检查非中心F分布是否引发异常
    a = random.chisquare(5, size=3)
    b = random.chisquare(5, size=(4, 3))
    c = random.chisquare(5, size=(5, 4, 3))
    assert random.noncentral_f(a, b, c).shape == (5, 4, 3)
    with pytest.raises(ValueError, match=r"Output size \(6, 5, 1, 1\) is"):
        random.noncentral_f(a, b, c, size=(6, 5, 1, 1))


# 测试当随机数生成函数参数化时，是否能够正确引发异常（标量情况）
def test_broadcast_size_scalar():
    # 创建形状为(3,)的均值向量mu和标准差向量sigma
    mu = np.ones(3)
    sigma = np.ones(3)
    
    # 测试是否正确引发大小不匹配的异常
    random.normal(mu, sigma, size=3)
    with pytest.raises(ValueError):
        random.normal(mu, sigma, size=2)


# 测试GH 18142，即当列表中存在不同类型的元素时，是否能正确进行乱序操作
def test_ragged_shuffle():
    # 定义一个包含空列表、空列表和整数1的序列
    seq = [[], [], 1]
    
    # 使用MT19937算法生成器进行乱序操作，验证是否不引发警告
    gen = Generator(MT19937(0))
    assert_no_warnings(gen.shuffle, seq)
    
    # 检查乱序后的序列是否符合预期
    assert seq == [1, [], []]


# 参数化测试函数，测试当high为-2或[-2]时的情况
@pytest.mark.parametrize("high", [-2, [-2]])
# 使用 pytest.mark.parametrize 装饰器定义一个参数化测试函数，用于测试整数生成器的异常情况
@pytest.mark.parametrize("endpoint", [True, False])
def test_single_arg_integer_exception(high, endpoint):
    # GH 14333：引用 GitHub 上的 issue 编号
    # 创建一个 Mersenne Twister 伪随机数生成器对象，种子为 0
    gen = Generator(MT19937(0))
    # 根据 endpoint 的值选择错误消息
    msg = 'high < 0' if endpoint else 'high <= 0'
    # 测试 high 参数小于等于0时是否会抛出 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        gen.integers(high, endpoint=endpoint)
    msg = 'low > high' if endpoint else 'low >= high'
    # 测试 low 参数大于 high 参数时是否会抛出 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        gen.integers(-1, high, endpoint=endpoint)
    # 测试传入列表形式的 low 参数时是否会抛出 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        gen.integers([-1], high, endpoint=endpoint)


# 使用 pytest.mark.parametrize 装饰器定义一个参数化测试函数，用于测试要求连续存储的异常情况
@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_c_contig_req_out(dtype):
    # GH 18704：引用 GitHub 上的 issue 编号
    # 创建一个空的 NumPy 数组，指定了存储顺序为 Fortran，数据类型为参数 dtype
    out = np.empty((2, 3), order="F", dtype=dtype)
    shape = [1, 2, 3]
    # 测试 random.standard_gamma 函数在指定输出数组时是否会抛出 ValueError 异常
    with pytest.raises(ValueError, match="Supplied output array"):
        random.standard_gamma(shape, out=out, dtype=dtype)
    # 测试 random.standard_gamma 函数在指定输出数组和大小时是否会抛出 ValueError 异常
    with pytest.raises(ValueError, match="Supplied output array"):
        random.standard_gamma(shape, out=out, size=out.shape, dtype=dtype)


# 使用 pytest.mark.parametrize 装饰器定义一个多参数化测试函数，用于测试要求连续存储的异常情况
@pytest.mark.parametrize("dtype", ["f4", "f8"])
@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("dist", [random.standard_normal, random.random])
def test_contig_req_out(dist, order, dtype):
    # GH 18704：引用 GitHub 上的 issue 编号
    # 创建一个空的 NumPy 数组，指定了存储顺序和数据类型
    out = np.empty((2, 3), dtype=dtype, order=order)
    # 调用指定的随机分布函数，测试其输出结果是否与指定的输出数组相同
    variates = dist(out=out, dtype=dtype)
    assert variates is out
    # 再次调用指定的随机分布函数，测试其输出结果是否与指定的输出数组和大小相同
    variates = dist(out=out, dtype=dtype, size=out.shape)
    assert variates is out


# 定义一个测试函数，用于测试旧式 pickle 格式下的生成器构造函数
def test_generator_ctor_old_style_pickle():
    # 创建一个 PCG64DXSM 类型的随机数生成器对象，种子为 0
    rg = np.random.Generator(np.random.PCG64DXSM(0))
    # 调用 standard_normal 方法生成一个随机数
    rg.standard_normal(1)
    # 调用 reduce 方法，用于序列化操作
    ctor, (bit_gen, ), _ = rg.__reduce__()
    # 检查反序列化后的对象是否与原始的 bit_gen 类型名称相同
    assert bit_gen.__class__.__name__ == "PCG64DXSM"
    # 使用构造函数名再次创建一个 bit_gen 对象
    print(ctor)
    b = ctor(*("PCG64DXSM",))
    print(b)
    # 将 bit_generator 对象的状态设置为与原始 bit_gen 对象相同
    b.bit_generator.state = bit_gen.state
    state_b = b.bit_generator.state
    assert bit_gen.state == state_b


# 定义一个测试函数，用于测试 pickle 对保留种子序列的影响
def test_pickle_preserves_seed_sequence():
    # GH 26234：引用 GitHub 上的 issue 编号
    # 导入 pickle 模块进行序列化操作
    import pickle
    # 创建一个 PCG64DXSM 类型的随机数生成器对象，种子为 20240411
    rg = np.random.Generator(np.random.PCG64DXSM(20240411))
    # 获取生成器的种子序列对象
    ss = rg.bit_generator.seed_seq
    # 将生成器对象序列化后再反序列化
    rg_plk = pickle.loads(pickle.dumps(rg))
    # 获取反序列化后的生成器的种子序列对象
    ss_plk = rg_plk.bit_generator.seed_seq
    # 检查序列化前后种子状态和种子池是否相同
    assert_equal(ss.state, ss_plk.state)
    assert_equal(ss.pool, ss_plk.pool)
    # 在原始生成器的种子序列上再执行一次 spawn 操作
    rg.bit_generator.seed_seq.spawn(10)
    # 再次进行序列化操作
    rg_plk = pickle.loads(pickle.dumps(rg))
    ss_plk = rg_plk.bit_generator.seed_seq
    # 检查序列化前后种子状态是否相同
    assert_equal(ss.state, ss_plk.state)


# 使用 pytest.mark.parametrize 装饰器定义一个参数化测试函数，用于测试旧式 pickle 格式下的生成器
@pytest.mark.parametrize("version", [121, 126])
def test_legacy_pickle(version):
    # Pickling format was changes in 1.22.x and in 2.0.x：引用 GitHub 上的 issue 编号
    # 导入 pickle 和 gzip 模块
    import pickle
    import gzip
    # 获取当前文件的路径
    base_path = os.path.split(os.path.abspath(__file__))[0]
    # 拼接生成器 pickle 文件的路径
    pkl_file = os.path.join(
        base_path, "data", f"generator_pcg64_np{version}.pkl.gz"
    )
    # 使用 gzip 打开 pickle 文件
    with gzip.open(pkl_file) as gz:
        # 从 pickle 文件中加载生成器对象
        rg = pickle.load(gz)
    # 获取生成器对象的状态
    state = rg.bit_generator.state['state']
    # 断言生成器对象的类型是 Generator 类的实例
    assert isinstance(rg, Generator)
    # 断言：确保 rg.bit_generator 是 np.random.PCG64 类型的对象
    assert isinstance(rg.bit_generator, np.random.PCG64)
    
    # 断言：确保 state 字典中 'state' 键对应的值等于 35399562948360463058890781895381311971
    assert state['state'] == 35399562948360463058890781895381311971
    
    # 断言：确保 state 字典中 'inc' 键对应的值等于 87136372517582989555478159403783844777
    assert state['inc'] == 87136372517582989555478159403783844777
```