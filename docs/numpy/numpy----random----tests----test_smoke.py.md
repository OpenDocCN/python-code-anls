# `.\numpy\numpy\random\tests\test_smoke.py`

```py
# 导入pickle模块，用于对象序列化和反序列化
import pickle
# 导入functools模块的partial函数，用于创建偏函数
from functools import partial

# 导入numpy库并重命名为np，用于科学计算
import numpy as np
# 导入pytest测试框架相关的断言函数和装饰器
import pytest
# 导入numpy.testing模块中的断言函数，用于数组断言
from numpy.testing import assert_equal, assert_, assert_array_equal
# 导入numpy.random模块中的随机数生成器相关类
from numpy.random import (Generator, MT19937, PCG64, PCG64DXSM, Philox, SFC64)

# 定义pytest的fixture，用于参数化测试，返回各种数据类型
@pytest.fixture(scope='module',
                params=(np.bool, np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64))
def dtype(request):
    return request.param


# 定义函数params_0，测试随机数生成器的输出是否为标量及其形状
def params_0(f):
    val = f()
    assert_(np.isscalar(val))
    val = f(10)
    assert_(val.shape == (10,))
    val = f((10, 10))
    assert_(val.shape == (10, 10))
    val = f((10, 10, 10))
    assert_(val.shape == (10, 10, 10))
    val = f(size=(5, 5))
    assert_(val.shape == (5, 5))


# 定义函数params_1，测试随机数生成器在不同输入条件下的输出
def params_1(f, bounded=False):
    a = 5.0
    b = np.arange(2.0, 12.0)
    c = np.arange(2.0, 102.0).reshape((10, 10))
    d = np.arange(2.0, 1002.0).reshape((10, 10, 10))
    e = np.array([2.0, 3.0])
    g = np.arange(2.0, 12.0).reshape((1, 10, 1))
    if bounded:
        a = 0.5
        b = b / (1.5 * b.max())
        c = c / (1.5 * c.max())
        d = d / (1.5 * d.max())
        e = e / (1.5 * e.max())
        g = g / (1.5 * g.max())

    # 标量
    f(a)
    # 标量 - 指定大小
    f(a, size=(10, 10))
    # 1维数组
    f(b)
    # 2维数组
    f(c)
    # 3维数组
    f(d)
    # 1维数组 - 指定大小
    f(b, size=10)
    # 2维数组 - 指定大小 - 广播
    f(e, size=(10, 2))
    # 3维数组 - 指定大小
    f(g, size=(10, 10, 10))


# 定义函数comp_state，比较两个状态是否相同
def comp_state(state1, state2):
    identical = True
    if isinstance(state1, dict):
        for key in state1:
            identical &= comp_state(state1[key], state2[key])
    elif type(state1) != type(state2):
        identical &= type(state1) == type(state2)
    else:
        if (isinstance(state1, (list, tuple, np.ndarray)) and isinstance(
                state2, (list, tuple, np.ndarray))):
            for s1, s2 in zip(state1, state2):
                identical &= comp_state(s1, s2)
        else:
            identical &= state1 == state2
    return identical


# 定义函数warmup，对随机数生成器进行预热
def warmup(rg, n=None):
    if n is None:
        n = 11 + np.random.randint(0, 20)
    rg.standard_normal(n)
    rg.standard_normal(n)
    rg.standard_normal(n, dtype=np.float32)
    rg.standard_normal(n, dtype=np.float32)
    rg.integers(0, 2 ** 24, n, dtype=np.uint64)
    rg.integers(0, 2 ** 48, n, dtype=np.uint64)
    rg.standard_gamma(11.0, n)
    rg.standard_gamma(11.0, n, dtype=np.float32)
    rg.random(n, dtype=np.float64)
    rg.random(n, dtype=np.float32)


# 定义RNG类，用于随机数生成器的设置和状态管理
class RNG:
    @classmethod
    def setup_class(cls):
        # 在测试类中被覆盖。用于静音IDE中的噪声
        cls.bit_generator = PCG64
        cls.advance = None
        cls.seed = [12345]
        cls.rg = Generator(cls.bit_generator(*cls.seed))
        cls.initial_state = cls.rg.bit_generator.state
        cls.seed_vector_bits = 64
        cls._extra_setup()

    @classmethod
    # 设置额外的类变量和初始状态，包括一个一维向量、二维向量、矩阵和错误类型
    def _extra_setup(cls):
        cls.vec_1d = np.arange(2.0, 102.0)
        cls.vec_2d = np.arange(2.0, 102.0)[None, :]
        cls.mat = np.arange(2.0, 102.0, 0.01).reshape((100, 100))
        cls.seed_error = TypeError

    # 重置对象的状态为初始状态
    def _reset_state(self):
        self.rg.bit_generator.state = self.initial_state

    # 测试随机数生成器初始化是否正确
    def test_init(self):
        rg = Generator(self.bit_generator())
        state = rg.bit_generator.state
        rg.standard_normal(1)  # 生成一个标准正态分布的随机数
        rg.standard_normal(1)  # 再生成一个标准正态分布的随机数
        rg.bit_generator.state = state  # 恢复到初始状态
        new_state = rg.bit_generator.state
        assert_(comp_state(state, new_state))  # 断言新状态与初始状态相同

    # 测试随机数生成器的状态推进功能
    def test_advance(self):
        state = self.rg.bit_generator.state
        if hasattr(self.rg.bit_generator, 'advance'):
            self.rg.bit_generator.advance(self.advance)  # 推进随机数生成器状态
            assert_(not comp_state(state, self.rg.bit_generator.state))  # 断言状态不同
        else:
            bitgen_name = self.rg.bit_generator.__class__.__name__
            pytest.skip(f'Advance is not supported by {bitgen_name}')  # 如果不支持推进功能，则跳过测试

    # 测试随机数生成器的跳跃功能
    def test_jump(self):
        state = self.rg.bit_generator.state
        if hasattr(self.rg.bit_generator, 'jumped'):
            bit_gen2 = self.rg.bit_generator.jumped()  # 执行跳跃操作
            jumped_state = bit_gen2.state
            assert_(not comp_state(state, jumped_state))  # 断言状态不同
            self.rg.random(2 * 3 * 5 * 7 * 11 * 13 * 17)  # 生成一组随机数
            self.rg.bit_generator.state = state  # 恢复初始状态
            bit_gen3 = self.rg.bit_generator.jumped()  # 再次执行跳跃操作
            rejumped_state = bit_gen3.state
            assert_(comp_state(jumped_state, rejumped_state))  # 断言再次跳跃后状态与第一次相同
        else:
            bitgen_name = self.rg.bit_generator.__class__.__name__
            if bitgen_name not in ('SFC64',):
                raise AttributeError(f'no "jumped" in {bitgen_name}')
            pytest.skip(f'Jump is not supported by {bitgen_name}')  # 如果不支持跳跃功能，则跳过测试

    # 测试生成均匀分布随机数
    def test_uniform(self):
        r = self.rg.uniform(-1.0, 0.0, size=10)  # 生成均匀分布的随机数
        assert_(len(r) == 10)  # 断言生成的随机数数量为10
        assert_((r > -1).all())  # 断言所有生成的随机数大于-1
        assert_((r <= 0).all())  # 断言所有生成的随机数小于等于0

    # 测试生成均匀分布随机数组
    def test_uniform_array(self):
        r = self.rg.uniform(np.array([-1.0] * 10), 0.0, size=10)  # 生成均匀分布的随机数数组
        assert_(len(r) == 10)  # 断言生成的随机数数量为10
        assert_((r > -1).all())  # 断言所有生成的随机数大于-1
        assert_((r <= 0).all())  # 断言所有生成的随机数小于等于0
        r = self.rg.uniform(np.array([-1.0] * 10),
                            np.array([0.0] * 10), size=10)  # 生成均匀分布的随机数数组
        assert_(len(r) == 10)  # 断言生成的随机数数量为10
        assert_((r > -1).all())  # 断言所有生成的随机数大于-1
        assert_((r <= 0).all())  # 断言所有生成的随机数小于等于0
        r = self.rg.uniform(-1.0, np.array([0.0] * 10), size=10)  # 生成均匀分布的随机数数组
        assert_(len(r) == 10)  # 断言生成的随机数数量为10
        assert_((r > -1).all())  # 断言所有生成的随机数大于-1
        assert_((r <= 0).all())  # 断言所有生成的随机数小于等于0

    # 测试生成随机数
    def test_random(self):
        assert_(len(self.rg.random(10)) == 10)  # 断言生成的随机数数量为10
        params_0(self.rg.random)  # 调用函数 params_0 进行进一步测试

    # 测试生成标准正态分布随机数（使用 Zig 方法）
    def test_standard_normal_zig(self):
        assert_(len(self.rg.standard_normal(10)) == 10)  # 断言生成的随机数数量为10

    # 测试生成标准正态分布随机数
    def test_standard_normal(self):
        assert_(len(self.rg.standard_normal(10)) == 10)  # 断言生成的随机数数量为10
        params_0(self.rg.standard_normal)  # 调用函数 params_0 进行进一步测试
    # 测试标准 Gamma 分布生成器函数
    def test_standard_gamma(self):
        # 断言生成的 Gamma 分布样本长度为 10
        assert_(len(self.rg.standard_gamma(10, 10)) == 10)
        # 断言生成的 Gamma 分布样本数组长度为 10
        assert_(len(self.rg.standard_gamma(np.array([10] * 10), 10)) == 10)
        # 使用 params_1 函数验证标准 Gamma 分布生成器函数
        params_1(self.rg.standard_gamma)

    # 测试标准指数分布生成器函数
    def test_standard_exponential(self):
        # 断言生成的指数分布样本长度为 10
        assert_(len(self.rg.standard_exponential(10)) == 10)
        # 使用 params_0 函数验证标准指数分布生成器函数
        params_0(self.rg.standard_exponential)

    # 测试生成浮点数标准指数分布的生成器函数
    def test_standard_exponential_float(self):
        # 生成浮点数标准指数分布样本，验证长度为 10
        randoms = self.rg.standard_exponential(10, dtype='float32')
        assert_(len(randoms) == 10)
        # 断言生成的浮点数标准指数分布样本数据类型为 np.float32
        assert randoms.dtype == np.float32
        # 使用 params_0 函数验证生成浮点数标准指数分布的生成器函数
        params_0(partial(self.rg.standard_exponential, dtype='float32'))

    # 测试生成浮点数标准指数分布并应用对数变换的生成器函数
    def test_standard_exponential_float_log(self):
        # 生成浮点数标准指数分布并应用对数变换，验证长度为 10
        randoms = self.rg.standard_exponential(10, dtype='float32', method='inv')
        assert_(len(randoms) == 10)
        # 断言生成的浮点数标准指数分布样本数据类型为 np.float32
        assert randoms.dtype == np.float32
        # 使用 params_0 函数验证生成浮点数标准指数分布并应用对数变换的生成器函数
        params_0(partial(self.rg.standard_exponential, dtype='float32', method='inv'))

    # 测试标准柯西分布生成器函数
    def test_standard_cauchy(self):
        # 断言生成的柯西分布样本长度为 10
        assert_(len(self.rg.standard_cauchy(10)) == 10)
        # 使用 params_0 函数验证标准柯西分布生成器函数
        params_0(self.rg.standard_cauchy)

    # 测试标准 t 分布生成器函数
    def test_standard_t(self):
        # 断言生成的 t 分布样本长度为 10
        assert_(len(self.rg.standard_t(10, 10)) == 10)
        # 使用 params_1 函数验证标准 t 分布生成器函数
        params_1(self.rg.standard_t)

    # 测试二项分布生成器函数
    def test_binomial(self):
        # 断言生成的二项分布样本大于等于 0
        assert_(self.rg.binomial(10, .5) >= 0)
        # 断言生成的二项分布样本大于等于 0
        assert_(self.rg.binomial(1000, .5) >= 0)

    # 测试随机数生成器状态重置函数
    def test_reset_state(self):
        # 保存当前随机数生成器状态
        state = self.rg.bit_generator.state
        # 生成一个整数
        int_1 = self.rg.integers(2**31)
        # 恢复之前保存的随机数生成器状态
        self.rg.bit_generator.state = state
        # 再次生成一个整数
        int_2 = self.rg.integers(2**31)
        # 断言两次生成的整数相等
        assert_(int_1 == int_2)

    # 测试随机数生成器初始化熵的函数
    def test_entropy_init(self):
        # 使用自定义比较函数验证随机数生成器的状态不同
        rg = Generator(self.bit_generator())
        rg2 = Generator(self.bit_generator())
        assert_(not comp_state(rg.bit_generator.state,
                               rg2.bit_generator.state))

    # 测试随机数生成器种子设定函数
    def test_seed(self):
        # 使用指定种子初始化两个随机数生成器对象
        rg = Generator(self.bit_generator(*self.seed))
        rg2 = Generator(self.bit_generator(*self.seed))
        # 生成随机数
        rg.random()
        rg2.random()
        # 断言两个随机数生成器的状态相同
        assert_(comp_state(rg.bit_generator.state, rg2.bit_generator.state))

    # 测试重置随机数生成器状态后生成正态分布的函数
    def test_reset_state_gauss(self):
        # 使用指定种子初始化随机数生成器对象
        rg = Generator(self.bit_generator(*self.seed))
        # 生成标准正态分布样本
        rg.standard_normal()
        # 保存当前随机数生成器状态
        state = rg.bit_generator.state
        # 生成标准正态分布的样本数组
        n1 = rg.standard_normal(size=10)
        # 使用默认种子初始化新的随机数生成器对象
        rg2 = Generator(self.bit_generator())
        # 恢复之前保存的随机数生成器状态
        rg2.bit_generator.state = state
        # 生成标准正态分布的样本数组
        n2 = rg2.standard_normal(size=10)
        # 断言两次生成的正态分布样本数组相等
        assert_array_equal(n1, n2)

    # 测试重置随机数生成器状态后生成无符号 32 位整数的函数
    def test_reset_state_uint32(self):
        # 使用指定种子初始化随机数生成器对象
        rg = Generator(self.bit_generator(*self.seed))
        # 生成指定范围内的无符号 32 位整数样本数组
        rg.integers(0, 2 ** 24, 120, dtype=np.uint32)
        # 保存当前随机数生成器状态
        state = rg.bit_generator.state
        # 生成无符号 32 位整数样本数组
        n1 = rg.integers(0, 2 ** 24, 10, dtype=np.uint32)
        # 使用默认种子初始化新的随机数生成器对象
        rg2 = Generator(self.bit_generator())
        # 恢复之前保存的随机数生成器状态
        rg2.bit_generator.state = state
        # 生成无符号 32 位整数样本数组
        n2 = rg2.integers(0, 2 ** 24, 10, dtype=np.uint32)
        # 断言两次生成的无符号 32 位整数样本数组相等
        assert_array_equal(n1, n2)
    def test_reset_state_float(self):
        # 创建一个生成器对象，使用给定的位生成器和种子初始化
        rg = Generator(self.bit_generator(*self.seed))
        # 生成浮点数随机数，改变生成器状态
        rg.random(dtype='float32')
        # 获取当前生成器的状态
        state = rg.bit_generator.state
        # 再次生成浮点数随机数，获得不同的随机数序列
        n1 = rg.random(size=10, dtype='float32')
        # 创建另一个生成器对象，使用默认的位生成器初始化
        rg2 = Generator(self.bit_generator())
        # 将第一个生成器的状态应用于第二个生成器
        rg2.bit_generator.state = state
        # 使用第二个生成器生成与第一个相同的随机数序列
        n2 = rg2.random(size=10, dtype='float32')
        # 断言两个随机数序列完全相同
        assert_((n1 == n2).all())

    def test_shuffle(self):
        # 创建一个包含200到1的整数数组
        original = np.arange(200, 0, -1)
        # 对数组进行随机排列
        permuted = self.rg.permutation(original)
        # 断言原始数组与排列后数组有不同的元素
        assert_((original != permuted).any())

    def test_permutation(self):
        # 创建一个包含200到1的整数数组
        original = np.arange(200, 0, -1)
        # 对数组进行随机排列
        permuted = self.rg.permutation(original)
        # 断言原始数组与排列后数组有不同的元素
        assert_((original != permuted).any())

    def test_beta(self):
        # 生成 beta 分布的随机数，形状为(10,)
        vals = self.rg.beta(2.0, 2.0, 10)
        # 断言生成的随机数数组长度为10
        assert_(len(vals) == 10)
        # 使用数组作为参数生成 beta 分布的随机数，形状为(10,)
        vals = self.rg.beta(np.array([2.0] * 10), 2.0)
        # 断言生成的随机数数组长度为10
        assert_(len(vals) == 10)
        # 使用数组作为参数生成 beta 分布的随机数，形状为(10,)
        vals = self.rg.beta(2.0, np.array([2.0] * 10))
        # 断言生成的随机数数组长度为10
        assert_(len(vals) == 10)
        # 使用数组作为参数生成 beta 分布的随机数，形状为(10,)
        vals = self.rg.beta(np.array([2.0] * 10), np.array([2.0] * 10))
        # 断言生成的随机数数组长度为10
        assert_(len(vals) == 10)
        # 使用二维数组作为参数生成 beta 分布的随机数，形状为(10, 10)
        vals = self.rg.beta(np.array([2.0] * 10), np.array([[2.0]] * 10))
        # 断言生成的随机数数组形状为(10, 10)
        assert_(vals.shape == (10, 10))

    def test_bytes(self):
        # 生成包含10个字节的随机字节数组
        vals = self.rg.bytes(10)
        # 断言生成的字节数组长度为10
        assert_(len(vals) == 10)

    def test_chisquare(self):
        # 生成自由度为2的卡方分布的随机数，形状为(10,)
        vals = self.rg.chisquare(2.0, 10)
        # 断言生成的随机数数组长度为10
        assert_(len(vals) == 10)
        # 调用特定的参数检查函数
        params_1(self.rg.chisquare)

    def test_exponential(self):
        # 生成参数为2的指数分布的随机数，形状为(10,)
        vals = self.rg.exponential(2.0, 10)
        # 断言生成的随机数数组长度为10
        assert_(len(vals) == 10)
        # 调用特定的参数检查函数
        params_1(self.rg.exponential)

    def test_f(self):
        # 生成自由度为(3, 1000)的 F 分布的随机数，形状为(10,)
        vals = self.rg.f(3, 1000, 10)
        # 断言生成的随机数数组长度为10
        assert_(len(vals) == 10)

    def test_gamma(self):
        # 生成形状参数为3，尺度参数为2的 Gamma 分布的随机数，形状为(10,)
        vals = self.rg.gamma(3, 2, 10)
        # 断言生成的随机数数组长度为10
        assert_(len(vals) == 10)

    def test_geometric(self):
        # 生成几何分布的随机数，成功概率为0.5，形状为(10,)
        vals = self.rg.geometric(0.5, 10)
        # 断言生成的随机数数组长度为10
        assert_(len(vals) == 10)
        # 调用特定的参数检查函数
        params_1(self.rg.exponential, bounded=True)

    def test_gumbel(self):
        # 生成 Gumbel 分布的随机数，位置参数为2.0，尺度参数为2.0，形状为(10,)
        vals = self.rg.gumbel(2.0, 2.0, 10)
        # 断言生成的随机数数组长度为10
        assert_(len(vals) == 10)

    def test_laplace(self):
        # 生成 Laplace 分布的随机数，位置参数为2.0，尺度参数为2.0，形状为(10,)
        vals = self.rg.laplace(2.0, 2.0, 10)
        # 断言生成的随机数数组长度为10
        assert_(len(vals) == 10)

    def test_logitic(self):
        # 生成 Logistic 分布的随机数，位置参数为2.0，尺度参数为2.0，形状为(10,)
        vals = self.rg.logistic(2.0, 2.0, 10)
        # 断言生成的随机数数组长度为10
        assert_(len(vals) == 10)

    def test_logseries(self):
        # 生成 Logseries 分布的随机数，概率参数为0.5，形状为(10,)
        vals = self.rg.logseries(0.5, 10)
        # 断言生成的随机数数组长度为10
        assert_(len(vals) == 10)

    def test_negative_binomial(self):
        # 生成负二项分布的随机数，参数为(10, 0.2)，形状为(10,)
        vals = self.rg.negative_binomial(10, 0.2, 10)
        # 断言生成的随机数数组长度为10
        assert_(len(vals) == 10)

    def test_noncentral_chisquare(self):
        # 生成非中心卡方分布的随机数，参数为(10, 2)，形状为(10,)
        vals = self.rg.noncentral_chisquare(10, 2, 10)
        # 断言生成的随机数数组长度为10
        assert_(len(vals) == 10)
    # 测试非中心 F 分布生成器
    def test_noncentral_f(self):
        # 生成指定参数的非中心 F 分布样本
        vals = self.rg.noncentral_f(3, 1000, 2, 10)
        # 断言样本长度为 10
        assert_(len(vals) == 10)
        # 使用数组参数生成非中心 F 分布样本
        vals = self.rg.noncentral_f(np.array([3] * 10), 1000, 2)
        # 断言样本长度为 10
        assert_(len(vals) == 10)
        # 使用数组参数生成非中心 F 分布样本
        vals = self.rg.noncentral_f(3, np.array([1000] * 10), 2)
        # 断言样本长度为 10
        assert_(len(vals) == 10)
        # 使用数组参数生成非中心 F 分布样本
        vals = self.rg.noncentral_f(3, 1000, np.array([2] * 10))
        # 断言样本长度为 10
        assert_(len(vals) == 10)

    # 测试正态分布生成器
    def test_normal(self):
        # 生成指定参数的正态分布样本
        vals = self.rg.normal(10, 0.2, 10)
        # 断言样本长度为 10
        assert_(len(vals) == 10)

    # 测试帕累托分布生成器
    def test_pareto(self):
        # 生成指定参数的帕累托分布样本
        vals = self.rg.pareto(3.0, 10)
        # 断言样本长度为 10
        assert_(len(vals) == 10)

    # 测试泊松分布生成器
    def test_poisson(self):
        # 生成指定参数的泊松分布样本
        vals = self.rg.poisson(10, 10)
        # 断言样本长度为 10
        assert_(len(vals) == 10)
        # 使用数组参数生成泊松分布样本
        vals = self.rg.poisson(np.array([10] * 10))
        # 断言样本长度为 10
        assert_(len(vals) == 10)
        # 对泊松分布生成器应用参数检查函数
        params_1(self.rg.poisson)

    # 测试幂分布生成器
    def test_power(self):
        # 生成指定参数的幂分布样本
        vals = self.rg.power(0.2, 10)
        # 断言样本长度为 10
        assert_(len(vals) == 10)

    # 测试整数分布生成器
    def test_integers(self):
        # 生成指定参数的整数分布样本
        vals = self.rg.integers(10, 20, 10)
        # 断言样本长度为 10
        assert_(len(vals) == 10)

    # 测试瑞利分布生成器
    def test_rayleigh(self):
        # 生成指定参数的瑞利分布样本
        vals = self.rg.rayleigh(0.2, 10)
        # 断言样本长度为 10
        assert_(len(vals) == 10)
        # 对瑞利分布生成器应用参数检查函数，设定边界为真
        params_1(self.rg.rayleigh, bounded=True)

    # 测试冯·米塞斯分布生成器
    def test_vonmises(self):
        # 生成指定参数的冯·米塞斯分布样本
        vals = self.rg.vonmises(10, 0.2, 10)
        # 断言样本长度为 10
        assert_(len(vals) == 10)

    # 测试瓦尔德分布生成器
    def test_wald(self):
        # 生成指定参数的瓦尔德分布样本
        vals = self.rg.wald(1.0, 1.0, 10)
        # 断言样本长度为 10
        assert_(len(vals) == 10)

    # 测试威布尔分布生成器
    def test_weibull(self):
        # 生成指定参数的威布尔分布样本
        vals = self.rg.weibull(1.0, 10)
        # 断言样本长度为 10
        assert_(len(vals) == 10)

    # 测试齐普夫分布生成器
    def test_zipf(self):
        # 生成指定参数的齐普夫分布样本
        vals = self.rg.zipf(10, 10)
        # 断言样本长度为 10
        assert_(len(vals) == 10)
        # 使用一维向量参数生成齐普夫分布样本
        vals = self.rg.zipf(self.vec_1d)
        # 断言样本长度为 100
        assert_(len(vals) == 100)
        # 使用二维向量参数生成齐普夫分布样本
        vals = self.rg.zipf(self.vec_2d)
        # 断言样本形状为 (1, 100)
        assert_(vals.shape == (1, 100))
        # 使用矩阵参数生成齐普夫分布样本
        vals = self.rg.zipf(self.mat)
        # 断言样本形状为 (100, 100)
        assert_(vals.shape == (100, 100))

    # 测试超几何分布生成器
    def test_hypergeometric(self):
        # 生成指定参数的超几何分布样本
        vals = self.rg.hypergeometric(25, 25, 20)
        # 断言返回值为标量
        assert_(np.isscalar(vals))
        # 使用数组参数生成超几何分布样本
        vals = self.rg.hypergeometric(np.array([25] * 10), 25, 20)
        # 断言返回值形状为 (10,)
        assert_(vals.shape == (10,))

    # 测试三角分布生成器
    def test_triangular(self):
        # 生成指定参数的三角分布样本
        vals = self.rg.triangular(-5, 0, 5)
        # 断言返回值为标量
        assert_(np.isscalar(vals))
        # 使用数组参数生成三角分布样本
        vals = self.rg.triangular(-5, np.array([0] * 10), 5)
        # 断言返回值形状为 (10, )
        assert_(vals.shape == (10,))

    # 测试多变量正态分布生成器
    def test_multivariate_normal(self):
        # 设定均值和协方差矩阵
        mean = [0, 0]
        cov = [[1, 0], [0, 100]]  # 对角协方差
        # 生成指定参数的多变量正态分布样本
        x = self.rg.multivariate_normal(mean, cov, 5000)
        # 断言返回值形状为 (5000, 2)
        assert_(x.shape == (5000, 2))
        # 生成另一组多变量正态分布样本
        x_zig = self.rg.multivariate_normal(mean, cov, 5000)
        # 断言返回值形状为 (5000, 2)
        assert_(x.shape == (5000, 2))
        # 生成另一组多变量正态分布样本
        x_inv = self.rg.multivariate_normal(mean, cov, 5000)
        # 断言返回值形状为 (5000, 2)
        assert_(x.shape == (5000, 2))
        # 断言生成的两组样本不完全相同
        assert_((x_zig != x_inv).any())

    # 测试多项分布生成器
    def test_multinomial(self):
        # 生成指定参数的多项分布样本
        vals = self.rg.multinomial(100, [1.0 / 3, 2.0 / 3])
        # 断言返回值形状为 (2,)
        assert_(vals.shape == (2,))
        # 生成多组指定参数的多项分布样本
        vals = self.rg.multin
    # 测试 Dirichlet 分布生成器的功能
    def test_dirichlet(self):
        # 生成一个 Dirichlet 分布样本
        s = self.rg.dirichlet((10, 5, 3), 20)
        # 断言返回的样本形状是否符合预期
        assert_(s.shape == (20, 3))

    # 测试对象的序列化和反序列化功能
    def test_pickle(self):
        # 序列化对象 self.rg 并获取 pickle 字节流
        pick = pickle.dumps(self.rg)
        # 反序列化 pickle 字节流并获取反序列化后的对象 unpick
        unpick = pickle.loads(pick)
        # 断言序列化前后的对象类型相同
        assert_((type(self.rg) == type(unpick)))
        # 断言序列化前后对象的状态是否一致
        assert_(comp_state(self.rg.bit_generator.state,
                           unpick.bit_generator.state))

        # 再次进行相同的序列化和反序列化操作，进行二次验证
        pick = pickle.dumps(self.rg)
        unpick = pickle.loads(pick)
        assert_((type(self.rg) == type(unpick)))
        assert_(comp_state(self.rg.bit_generator.state,
                           unpick.bit_generator.state))

    # 测试随机数生成器的种子数组初始化功能
    def test_seed_array(self):
        # 如果不支持向量种子，则跳过此测试
        if self.seed_vector_bits is None:
            bitgen_name = self.bit_generator.__name__
            pytest.skip(f'Vector seeding is not supported by {bitgen_name}')

        # 根据种子位数选择正确的数据类型
        if self.seed_vector_bits == 32:
            dtype = np.uint32
        else:
            dtype = np.uint64

        # 使用单个元素的数组作为种子初始化生成器，并比较状态是否一致
        seed = np.array([1], dtype=dtype)
        bg = self.bit_generator(seed)
        state1 = bg.state
        bg = self.bit_generator(1)
        state2 = bg.state
        assert_(comp_state(state1, state2))

        # 使用长度为 4 的数组作为种子初始化生成器，并比较状态是否一致
        seed = np.arange(4, dtype=dtype)
        bg = self.bit_generator(seed)
        state1 = bg.state
        bg = self.bit_generator(seed[0])
        state2 = bg.state
        assert_(not comp_state(state1, state2))

        # 使用长度为 1500 的数组作为种子初始化生成器，并比较状态是否一致
        seed = np.arange(1500, dtype=dtype)
        bg = self.bit_generator(seed)
        state1 = bg.state
        bg = self.bit_generator(seed[0])
        state2 = bg.state
        assert_(not comp_state(state1, state2))

        # 使用特定算法生成长度为 1500 的种子数组，并比较状态是否一致
        seed = 2 ** np.mod(np.arange(1500, dtype=dtype),
                           self.seed_vector_bits - 1) + 1
        bg = self.bit_generator(seed)
        state1 = bg.state
        bg = self.bit_generator(seed[0])
        state2 = bg.state
        assert_(not comp_state(state1, state2))

    # 测试生成均匀分布的随机浮点数功能
    def test_uniform_float(self):
        # 创建随机数生成器 rg，并进行初始化和预热
        rg = Generator(self.bit_generator(12345))
        warmup(rg)
        state = rg.bit_generator.state
        # 生成一组随机浮点数 r1
        r1 = rg.random(11, dtype=np.float32)
        # 创建另一个随机数生成器 rg2，并初始化和预热，并恢复状态
        rg2 = Generator(self.bit_generator())
        warmup(rg2)
        rg2.bit_generator.state = state
        # 使用相同状态生成另一组随机浮点数 r2
        r2 = rg2.random(11, dtype=np.float32)
        # 断言两组随机数数组是否相等
        assert_array_equal(r1, r2)
        # 断言生成的浮点数类型是否为 np.float32
        assert_equal(r1.dtype, np.float32)
        # 断言两个随机数生成器的状态是否一致
        assert_(comp_state(rg.bit_generator.state, rg2.bit_generator.state))

    # 测试生成 Gamma 分布的随机浮点数功能
    def test_gamma_floats(self):
        # 创建随机数生成器 rg，并进行初始化和预热
        rg = Generator(self.bit_generator())
        warmup(rg)
        state = rg.bit_generator.state
        # 生成一组 Gamma 分布的随机浮点数 r1
        r1 = rg.standard_gamma(4.0, 11, dtype=np.float32)
        # 创建另一个随机数生成器 rg2，并初始化和预热，并恢复状态
        rg2 = Generator(self.bit_generator())
        warmup(rg2)
        rg2.bit_generator.state = state
        # 使用相同状态生成另一组 Gamma 分布的随机浮点数 r2
        r2 = rg2.standard_gamma(4.0, 11, dtype=np.float32)
        # 断言两组随机数数组是否相等
        assert_array_equal(r1, r2)
        # 断言生成的浮点数类型是否为 np.float32
        assert_equal(r1.dtype, np.float32)
        # 断言两个随机数生成器的状态是否一致
        assert_(comp_state(rg.bit_generator.state, rg2.bit_generator.state))
    # 测试标准正态分布生成器的功能，验证两个独立生成的随机数数组是否相等
    def test_normal_floats(self):
        # 使用指定的比特生成器创建随机数生成器对象
        rg = Generator(self.bit_generator())
        # 对生成器进行预热，确保生成的随机数稳定
        warmup(rg)
        # 获取当前比特生成器的状态
        state = rg.bit_generator.state
        # 生成长度为 11 的 np.float32 类型的标准正态分布随机数数组
        r1 = rg.standard_normal(11, dtype=np.float32)
        # 使用同一个比特生成器对象创建另一个随机数生成器对象
        rg2 = Generator(self.bit_generator())
        # 对第二个生成器进行预热
        warmup(rg2)
        # 将第二个生成器的状态设置为之前保存的状态，确保两次生成的随机数相同
        rg2.bit_generator.state = state
        # 生成另一个长度为 11 的 np.float32 类型的标准正态分布随机数数组
        r2 = rg2.standard_normal(11, dtype=np.float32)
        # 断言两个随机数数组相等
        assert_array_equal(r1, r2)
        # 断言随机数数组的数据类型为 np.float32
        assert_equal(r1.dtype, np.float32)
        # 比较两个比特生成器的状态是否相同
        assert_(comp_state(rg.bit_generator.state, rg2.bit_generator.state))

    # 测试标准正态分布生成器的功能，验证两个独立生成的随机数数组是否相等
    def test_normal_zig_floats(self):
        # 使用指定的比特生成器创建随机数生成器对象
        rg = Generator(self.bit_generator())
        # 对生成器进行预热，确保生成的随机数稳定
        warmup(rg)
        # 获取当前比特生成器的状态
        state = rg.bit_generator.state
        # 生成长度为 11 的 np.float32 类型的标准正态分布随机数数组
        r1 = rg.standard_normal(11, dtype=np.float32)
        # 使用同一个比特生成器对象创建另一个随机数生成器对象
        rg2 = Generator(self.bit_generator())
        # 对第二个生成器进行预热
        warmup(rg2)
        # 将第二个生成器的状态设置为之前保存的状态，确保两次生成的随机数相同
        rg2.bit_generator.state = state
        # 生成另一个长度为 11 的 np.float32 类型的标准正态分布随机数数组
        r2 = rg2.standard_normal(11, dtype=np.float32)
        # 断言两个随机数数组相等
        assert_array_equal(r1, r2)
        # 断言随机数数组的数据类型为 np.float32
        assert_equal(r1.dtype, np.float32)
        # 比较两个比特生成器的状态是否相同
        assert_(comp_state(rg.bit_generator.state, rg2.bit_generator.state))

    # 测试随机数填充功能，验证使用现有数组作为输出时是否正确生成随机数
    def test_output_fill(self):
        # 使用预设的随机数生成器对象
        rg = self.rg
        # 获取当前比特生成器的状态
        state = rg.bit_generator.state
        # 创建一个形状为 (31, 7, 97) 的空 np.ndarray 数组
        size = (31, 7, 97)
        existing = np.empty(size)
        # 将比特生成器的状态恢复到之前保存的状态
        rg.bit_generator.state = state
        # 使用现有数组 existing 作为输出，生成标准正态分布的随机数填充该数组
        rg.standard_normal(out=existing)
        # 再次将比特生成器的状态恢复到之前保存的状态
        rg.bit_generator.state = state
        # 直接生成一个与 existing 相同形状的标准正态分布的随机数数组
        direct = rg.standard_normal(size=size)
        # 断言直接生成的数组与使用现有数组生成的数组相等
        assert_equal(direct, existing)

        # 创建一个形状为 (31, 7, 97) 的空 np.ndarray 数组
        sized = np.empty(size)
        # 将比特生成器的状态恢复到之前保存的状态
        rg.bit_generator.state = state
        # 使用现有数组 sized 作为输出，生成标准正态分布的随机数填充该数组
        rg.standard_normal(out=sized, size=sized.shape)

        # 创建一个形状为 (31, 7, 97) 的空 np.ndarray 数组，指定数据类型为 np.float32
        existing = np.empty(size, dtype=np.float32)
        # 将比特生成器的状态恢复到之前保存的状态
        rg.bit_generator.state = state
        # 使用现有数组 existing 作为输出，生成标准正态分布的随机数填充该数组，指定数据类型为 np.float32
        rg.standard_normal(out=existing, dtype=np.float32)
        # 再次将比特生成器的状态恢复到之前保存的状态
        rg.bit_generator.state = state
        # 直接生成一个与 existing 相同形状的标准正态分布的随机数数组，指定数据类型为 np.float32
        direct = rg.standard_normal(size=size, dtype=np.float32)
        # 断言直接生成的数组与使用现有数组生成的数组相等
        assert_equal(direct, existing)

    # 测试随机数填充功能，验证使用现有数组作为输出时是否正确生成均匀分布的随机数
    def test_output_filling_uniform(self):
        # 使用预设的随机数生成器对象
        rg = self.rg
        # 获取当前比特生成器的状态
        state = rg.bit_generator.state
        # 创建一个形状为 (31, 7, 97) 的空 np.ndarray 数组
        size = (31, 7, 97)
        existing = np.empty(size)
        # 将比特生成器的状态恢复到之前保存的状态
        rg.bit_generator.state = state
        # 使用现有数组 existing 作为输出，生成均匀分布的随机数填充该数组
        rg.random(out=existing)
        # 再次将比特生成器的状态恢复到之前保存的状态
        rg.bit_generator.state = state
        # 直接生成一个与 existing 相同形状的均匀分布的随机数数组
        direct = rg.random(size=size)
        # 断言直接生成的数组与使用现有数组生成的数组相等
        assert_equal(direct, existing)

        # 创建一个形状为 (31, 7, 97) 的空 np.ndarray 数组，指定数据类型为 np.float32
        existing = np.empty(size, dtype=np.float32)
        # 将比特生成器的状态恢复到之前保存的状态
        rg.bit_generator.state = state
        # 使用现有数组 existing 作为输出，生成均匀分布的随机数填充该数组，指定数据类型为 np.float32
        rg.random(out=existing, dtype=np.float32)
        # 再次将比特生成器的状态恢复到之前保存的状态
        rg.bit_generator.state = state
        # 直接生成一个与 existing 相同形状的均匀分布的随机数数组，指定数据类型为 np.float32
        direct = rg.random(size=size, dtype=np.float32)
        # 断言直接生成的数组与使用现有数组生成的数组相等
        assert_equal(direct, existing)
    # 测试填充指数分布的输出数组
    def test_output_filling_exponential(self):
        # 获取随机数生成器实例
        rg = self.rg
        # 保存随机数生成器的状态
        state = rg.bit_generator.state
        # 指定数组的大小
        size = (31, 7, 97)
        # 创建一个空数组用于存储生成的随机数
        existing = np.empty(size)
        # 恢复随机数生成器的状态
        rg.bit_generator.state = state
        # 使用已存在的数组填充指数分布的随机数
        rg.standard_exponential(out=existing)
        # 恢复随机数生成器的状态
        rg.bit_generator.state = state
        # 直接生成指定大小的指数分布随机数并存储在数组中
        direct = rg.standard_exponential(size=size)
        # 断言两种方式生成的结果数组应当相等
        assert_equal(direct, existing)

        # 创建一个空数组用于存储生成的随机数，指定数据类型为 float32
        existing = np.empty(size, dtype=np.float32)
        # 恢复随机数生成器的状态
        rg.bit_generator.state = state
        # 使用已存在的数组填充指数分布的随机数，指定数据类型为 float32
        rg.standard_exponential(out=existing, dtype=np.float32)
        # 恢复随机数生成器的状态
        rg.bit_generator.state = state
        # 直接生成指定大小的指数分布随机数并存储在数组中，指定数据类型为 float32
        direct = rg.standard_exponential(size=size, dtype=np.float32)
        # 断言两种方式生成的结果数组应当相等
        assert_equal(direct, existing)

    # 测试填充 Gamma 分布的输出数组
    def test_output_filling_gamma(self):
        # 获取随机数生成器实例
        rg = self.rg
        # 保存随机数生成器的状态
        state = rg.bit_generator.state
        # 指定数组的大小
        size = (31, 7, 97)
        # 创建一个全零数组用于存储生成的随机数
        existing = np.zeros(size)
        # 恢复随机数生成器的状态
        rg.bit_generator.state = state
        # 使用已存在的数组填充 Gamma 分布的随机数，参数为 1.0
        rg.standard_gamma(1.0, out=existing)
        # 恢复随机数生成器的状态
        rg.bit_generator.state = state
        # 直接生成指定大小的 Gamma 分布随机数并存储在数组中，参数为 1.0
        direct = rg.standard_gamma(1.0, size=size)
        # 断言两种方式生成的结果数组应当相等
        assert_equal(direct, existing)

        # 创建一个全零数组用于存储生成的随机数，指定数据类型为 float32
        existing = np.zeros(size, dtype=np.float32)
        # 恢复随机数生成器的状态
        rg.bit_generator.state = state
        # 使用已存在的数组填充 Gamma 分布的随机数，参数为 1.0，指定数据类型为 float32
        rg.standard_gamma(1.0, out=existing, dtype=np.float32)
        # 恢复随机数生成器的状态
        rg.bit_generator.state = state
        # 直接生成指定大小的 Gamma 分布随机数并存储在数组中，参数为 1.0，指定数据类型为 float32
        direct = rg.standard_gamma(1.0, size=size, dtype=np.float32)
        # 断言两种方式生成的结果数组应当相等
        assert_equal(direct, existing)

    # 测试广播填充 Gamma 分布的输出数组
    def test_output_filling_gamma_broadcast(self):
        # 获取随机数生成器实例
        rg = self.rg
        # 保存随机数生成器的状态
        state = rg.bit_generator.state
        # 指定数组的大小
        size = (31, 7, 97)
        # 创建一个全零数组用于存储生成的随机数
        existing = np.zeros(size)
        # 使用广播方式填充 Gamma 分布的随机数，参数为从 1.0 到 97.0 的数组
        rg.standard_gamma(np.arange(97.0) + 1.0, out=existing)
        # 恢复随机数生成器的状态
        rg.bit_generator.state = state
        # 直接生成指定大小的 Gamma 分布随机数并存储在数组中，参数为从 1.0 到 97.0 的数组
        direct = rg.standard_gamma(np.arange(97.0) + 1.0, size=size)
        # 断言两种方式生成的结果数组应当相等
        assert_equal(direct, existing)

        # 创建一个全零数组用于存储生成的随机数，指定数据类型为 float32
        existing = np.zeros(size, dtype=np.float32)
        # 恢复随机数生成器的状态
        rg.bit_generator.state = state
        # 使用广播方式填充 Gamma 分布的随机数，参数为从 1.0 到 97.0 的数组，指定数据类型为 float32
        rg.standard_gamma(np.arange(97.0) + 1.0, out=existing, dtype=np.float32)
        # 恢复随机数生成器的状态
        rg.bit_generator.state = state
        # 直接生成指定大小的 Gamma 分布随机数并存储在数组中，参数为从 1.0 到 97.0 的数组，指定数据类型为 float32
        direct = rg.standard_gamma(np.arange(97.0) + 1.0, size=size, dtype=np.float32)
        # 断言两种方式生成的结果数组应当相等
        assert_equal(direct, existing)
    # 测试函数，用于验证生成器方法在特定条件下的输出是否会触发异常

    def test_output_fill_error(self):
        # 从测试类中获取生成器对象
        rg = self.rg
        # 定义一个元组表示数组大小
        size = (31, 7, 97)
        # 创建一个空数组，用于存储生成的随机数
        existing = np.empty(size)
        # 测试：确保在指定类型错误时会引发异常
        with pytest.raises(TypeError):
            rg.standard_normal(out=existing, dtype=np.float32)
        # 测试：确保在指定值错误时会引发异常
        with pytest.raises(ValueError):
            rg.standard_normal(out=existing[::3])
        
        # 重新定义数组类型和测试异常
        existing = np.empty(size, dtype=np.float32)
        with pytest.raises(TypeError):
            rg.standard_normal(out=existing, dtype=np.float64)

        existing = np.zeros(size, dtype=np.float32)
        with pytest.raises(TypeError):
            rg.standard_gamma(1.0, out=existing, dtype=np.float64)
        with pytest.raises(ValueError):
            rg.standard_gamma(1.0, out=existing[::3])
        
        # 重新定义数组类型和测试异常
        existing = np.zeros(size, dtype=np.float64)
        with pytest.raises(TypeError):
            rg.standard_gamma(1.0, out=existing, dtype=np.float32)
        with pytest.raises(ValueError):
            rg.standard_gamma(1.0, out=existing[::3])

    # 测试函数，用于验证整数生成器在不同情况下的输出是否正确

    def test_integers_broadcast(self, dtype):
        # 根据数据类型设置上下界
        if dtype == np.bool:
            upper = 2
            lower = 0
        else:
            info = np.iinfo(dtype)
            upper = int(info.max) + 1
            lower = info.min
        # 重置生成器状态
        self._reset_state()
        # 测试生成器方法：lower为标量，upper为列表
        a = self.rg.integers(lower, [upper] * 10, dtype=dtype)
        self._reset_state()
        # 测试生成器方法：lower为列表，upper为标量
        b = self.rg.integers([lower] * 10, upper, dtype=dtype)
        # 确保两次生成的结果相等
        assert_equal(a, b)
        self._reset_state()
        # 测试生成器方法：lower和upper均为标量
        c = self.rg.integers(lower, upper, size=10, dtype=dtype)
        # 确保生成的结果与a相等
        assert_equal(a, c)
        self._reset_state()
        # 测试生成器方法：lower和upper均为数组对象
        d = self.rg.integers(np.array(
            [lower] * 10), np.array([upper], dtype=object), size=10,
            dtype=dtype)
        # 确保生成的结果与a相等
        assert_equal(a, d)
        self._reset_state()
        # 测试生成器方法：lower和upper均为数组对象
        e = self.rg.integers(
            np.array([lower] * 10), np.array([upper] * 10), size=10,
            dtype=dtype)
        # 确保生成的结果与a相等
        assert_equal(a, e)

        # 重置生成器状态
        self._reset_state()
        # 测试生成器方法：lower为标量，upper为标量
        a = self.rg.integers(0, upper, size=10, dtype=dtype)
        self._reset_state()
        # 测试生成器方法：lower为列表，upper为标量
        b = self.rg.integers([upper] * 10, dtype=dtype)
        # 确保两次生成的结果相等
        assert_equal(a, b)

    # 测试函数，用于验证整数生成器对NumPy数组的处理是否正确

    def test_integers_numpy(self, dtype):
        # 定义高和低的NumPy数组
        high = np.array([1])
        low = np.array([0])

        # 测试生成器方法：使用NumPy数组作为输入
        out = self.rg.integers(low, high, dtype=dtype)
        # 确保输出形状符合预期
        assert out.shape == (1,)

        # 测试生成器方法：使用NumPy数组作为输入，低位为标量
        out = self.rg.integers(low[0], high, dtype=dtype)
        # 确保输出形状符合预期
        assert out.shape == (1,)

        # 测试生成器方法：使用NumPy数组作为输入，高位为标量
        out = self.rg.integers(low, high[0], dtype=dtype)
        # 确保输出形状符合预期
        assert out.shape == (1,)
    # 测试整数生成函数在广播错误处理方面的行为
    def test_integers_broadcast_errors(self, dtype):
        # 如果数据类型是布尔型
        if dtype == np.bool:
            # 设置上界为2，下界为0
            upper = 2
            lower = 0
        else:
            # 获取数据类型的信息对象
            info = np.iinfo(dtype)
            # 计算上界为最大值加1后的整数值
            upper = int(info.max) + 1
            # 下界为数据类型的最小值
            lower = info.min
        # 断言调用生成整数函数时，超出上界抛出值错误
        with pytest.raises(ValueError):
            self.rg.integers(lower, [upper + 1] * 10, dtype=dtype)
        # 断言调用生成整数函数时，超出下界抛出值错误
        with pytest.raises(ValueError):
            self.rg.integers(lower - 1, [upper] * 10, dtype=dtype)
        # 断言调用生成整数函数时，下界参数不是标量抛出值错误
        with pytest.raises(ValueError):
            self.rg.integers([lower - 1], [upper] * 10, dtype=dtype)
        # 断言调用生成整数函数时，上下界参数都不是标量抛出值错误
        with pytest.raises(ValueError):
            self.rg.integers([0], [0], dtype=dtype)
class TestMT19937(RNG):
    @classmethod
    def setup_class(cls):
        # 设置比特生成器为MT19937
        cls.bit_generator = MT19937
        # 设置 advance 为None，表示没有预定义的状态前进量
        cls.advance = None
        # 设置种子为一个列表，包含一个整数值
        cls.seed = [2 ** 21 + 2 ** 16 + 2 ** 5 + 1]
        # 使用给定的比特生成器和种子创建一个生成器对象
        cls.rg = Generator(cls.bit_generator(*cls.seed))
        # 记录生成器的初始状态
        cls.initial_state = cls.rg.bit_generator.state
        # 设置种子向量的比特数为32
        cls.seed_vector_bits = 32
        # 调用额外的设置方法
        cls._extra_setup()
        # 设置种子错误为 ValueError
        cls.seed_error = ValueError

    def test_numpy_state(self):
        # 创建一个numpy的随机状态对象
        nprg = np.random.RandomState()
        # 生成99个标准正态分布的随机数
        nprg.standard_normal(99)
        # 获取当前的状态
        state = nprg.get_state()
        # 将获取到的状态设置为当前生成器的比特生成器状态
        self.rg.bit_generator.state = state
        # 再次获取当前的比特生成器状态
        state2 = self.rg.bit_generator.state
        # 断言两个状态的第一个元素相等
        assert_((state[1] == state2['state']['key']).all())
        # 断言两个状态的第二个元素相等
        assert_((state[2] == state2['state']['pos']))


```    
class TestPhilox(RNG):
    @classmethod
    def setup_class(cls):
        # 设置比特生成器为Philox
        cls.bit_generator = Philox
        # 设置 advance 为一个大整数，表示预定义的状态前进量
        cls.advance = 2**63 + 2**31 + 2**15 + 1
        # 设置种子为一个列表，包含一个整数值
        cls.seed = [12345]
        # 使用给定的比特生成器和种子创建一个生成器对象
        cls.rg = Generator(cls.bit_generator(*cls.seed))
        # 记录生成器的初始状态
        cls.initial_state = cls.rg.bit_generator.state
        # 设置种子向量的比特数为64
        cls.seed_vector_bits = 64
        # 调用额外的设置方法
        cls._extra_setup()

```py    

```    
class TestSFC64(RNG):
    @classmethod
    def setup_class(cls):
        # 设置比特生成器为SFC64
        cls.bit_generator = SFC64
        # 设置 advance 为None，表示没有预定义的状态前进量
        cls.advance = None
        # 设置种子为一个列表，包含一个整数值
        cls.seed = [12345]
        # 使用给定的比特生成器和种子创建一个生成器对象
        cls.rg = Generator(cls.bit_generator(*cls.seed))
        # 记录生成器的初始状态
        cls.initial_state = cls.rg.bit_generator.state
        # 设置种子向量的比特数为192
        cls.seed_vector_bits = 192
        # 调用额外的设置方法
        cls._extra_setup()

```py    

```    
class TestPCG64(RNG):
    @classmethod
    def setup_class(cls):
        # 设置比特生成器为PCG64
        cls.bit_generator = PCG64
        # 设置 advance 为一个大整数，表示预定义的状态前进量
        cls.advance = 2**63 + 2**31 + 2**15 + 1
        # 设置种子为一个列表，包含一个整数值
        cls.seed = [12345]
        # 使用给定的比特生成器和种子创建一个生成器对象
        cls.rg = Generator(cls.bit_generator(*cls.seed))
        # 记录生成器的初始状态
        cls.initial_state = cls.rg.bit_generator.state
        # 设置种子向量的比特数为64
        cls.seed_vector_bits = 64
        # 调用额外的设置方法
        cls._extra_setup()

```py    

```    
class TestPCG64DXSM(RNG):
    @classmethod
    def setup_class(cls):
        # 设置比特生成器为PCG64DXSM
        cls.bit_generator = PCG64DXSM
        # 设置 advance 为一个大整数，表示预定义的状态前进量
        cls.advance = 2**63 + 2**31 + 2**15 + 1
        # 设置种子为一个列表，包含一个整数值
        cls.seed = [12345]
        # 使用给定的比特生成器和种子创建一个生成器对象
        cls.rg = Generator(cls.bit_generator(*cls.seed))
        # 记录生成器的初始状态
        cls.initial_state = cls.rg.bit_generator.state
        # 设置种子向量的比特数为64
        cls.seed_vector_bits = 64
        # 调用额外的设置方法
        cls._extra_setup()

```py    

```    
class TestDefaultRNG(RNG):
    @classmethod
    def setup_class(cls):
        # 这将重复一些直接实例化新生成器对象的测试，但没关系。
        # 设置比特生成器为PCG64
        cls.bit_generator = PCG64
        # 设置 advance 为一个大整数，表示预定义的状态前进量
        cls.advance = 2**63 + 2**31 + 2**15 + 1
        # 设置种子为一个列表，包含一个整数值
        cls.seed = [12345]
        # 使用给定的种子创建一个numpy的随机生成器对象
        cls.rg = np.random.default_rng(*cls.seed)
        # 记录生成器的初始状态
        cls.initial_state = cls.rg.bit_generator.state
        # 设置种子向量的比特数为64
        cls.seed_vector_bits = 64
        # 调用额外的设置方法
        cls._extra_setup()

    def test_default_is_pcg64(self):
        # 为了改变默认的比特生成器，我们会通过一个弃用周期来迁移到一个不同的函数。
        # 断言当前生成器的比特生成器是PCG64的实例
        assert_(isinstance(self.rg.bit_generator, PCG64))
    # 定义单元测试方法 test_seed，用于测试 np.random.default_rng 函数的不同参数情况
    def test_seed(self):
        # 使用默认种子创建一个随机数生成器对象
        np.random.default_rng()
        # 使用 None 作为种子创建一个随机数生成器对象
        np.random.default_rng(None)
        # 使用整数 12345 作为种子创建一个随机数生成器对象
        np.random.default_rng(12345)
        # 使用整数 0 作为种子创建一个随机数生成器对象
        np.random.default_rng(0)
        # 使用一个非常大的整数作为种子创建一个随机数生成器对象
        np.random.default_rng(43660444402423911716352051725018508569)
        # 使用包含两个整数的列表作为种子创建一个随机数生成器对象
        np.random.default_rng([43660444402423911716352051725018508569,
                               279705150948142787361475340226491943209])
        # 测试使用负数作为种子是否会引发 ValueError 异常
        with pytest.raises(ValueError):
            np.random.default_rng(-1)
        # 测试使用包含负数的列表作为种子是否会引发 ValueError 异常
        with pytest.raises(ValueError):
            np.random.default_rng([12345, -1])
```