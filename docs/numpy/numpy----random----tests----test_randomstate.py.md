# `.\numpy\numpy\random\tests\test_randomstate.py`

```
import hashlib  # 导入用于哈希计算的 hashlib 库
import pickle  # 导入用于序列化和反序列化的 pickle 库
import sys  # 导入系统相关的模块
import warnings  # 导入警告处理模块

import numpy as np  # 导入 NumPy 库，并使用 np 别名
import pytest  # 导入 Pytest 测试框架
from numpy.testing import (  # 从 NumPy 测试模块中导入多个断言函数
        assert_, assert_raises, assert_equal, assert_warns,
        assert_no_warnings, assert_array_equal, assert_array_almost_equal,
        suppress_warnings, IS_WASM
        )
from numpy.random import MT19937, PCG64  # 从 NumPy 随机模块中导入 MT19937 和 PCG64 生成器类
from numpy import random  # 导入 NumPy 随机模块

INT_FUNCS = {'binomial': (100.0, 0.6),  # 初始化整数函数字典，包含函数名和参数元组
             'geometric': (.5,),
             'hypergeometric': (20, 20, 10),
             'logseries': (.5,),
             'multinomial': (20, np.ones(6) / 6.0),
             'negative_binomial': (100, .5),
             'poisson': (10.0,),
             'zipf': (2,),
             }

if np.iinfo(np.long).max < 2**32:
    # 如果 long 类型的最大值小于 2 的 32 次方，说明是 Windows 或某些 32 位平台，例如 ARM
    INT_FUNC_HASHES = {'binomial': '2fbead005fc63942decb5326d36a1f32fe2c9d32c904ee61e46866b88447c263',  # 初始化整数函数的哈希值字典
                       'logseries': '23ead5dcde35d4cfd4ef2c105e4c3d43304b45dc1b1444b7823b9ee4fa144ebb',
                       'geometric': '0d764db64f5c3bad48c8c33551c13b4d07a1e7b470f77629bef6c985cac76fcf',
                       'hypergeometric': '7b59bf2f1691626c5815cdcd9a49e1dd68697251d4521575219e4d2a1b8b2c67',
                       'multinomial': 'd754fa5b92943a38ec07630de92362dd2e02c43577fc147417dc5b9db94ccdd3',
                       'negative_binomial': '8eb216f7cb2a63cf55605422845caaff002fddc64a7dc8b2d45acd477a49e824',
                       'poisson': '70c891d76104013ebd6f6bcf30d403a9074b886ff62e4e6b8eb605bf1a4673b7',
                       'zipf': '01f074f97517cd5d21747148ac6ca4074dde7fcb7acbaec0a936606fecacd93f',
                       }
else:
    # 否则，长整型的最大值大于等于 2 的 32 次方，适用于大多数平台
    INT_FUNC_HASHES = {'binomial': '8626dd9d052cb608e93d8868de0a7b347258b199493871a1dc56e2a26cacb112',  # 初始化整数函数的哈希值字典
                       'geometric': '8edd53d272e49c4fc8fbbe6c7d08d563d62e482921f3131d0a0e068af30f0db9',
                       'hypergeometric': '83496cc4281c77b786c9b7ad88b74d42e01603a55c60577ebab81c3ba8d45657',
                       'logseries': '65878a38747c176bc00e930ebafebb69d4e1e16cd3a704e264ea8f5e24f548db',
                       'multinomial': '7a984ae6dca26fd25374479e118b22f55db0aedccd5a0f2584ceada33db98605',
                       'negative_binomial': 'd636d968e6a24ae92ab52fe11c46ac45b0897e98714426764e820a7d77602a61',
                       'poisson': '956552176f77e7c9cb20d0118fc9cf690be488d790ed4b4c4747b965e61b0bb4',
                       'zipf': 'f84ba7feffda41e606e20b28dfc0f1ea9964a74574513d4a4cbc98433a8bfa45',
                       }


@pytest.fixture(scope='module', params=INT_FUNCS)
def int_func(request):
    # Pytest 的夹具（fixture），提供整数函数名、参数元组和对应的哈希值
    return (request.param, INT_FUNCS[request.param],
            INT_FUNC_HASHES[request.param])


@pytest.fixture
def restore_singleton_bitgen():
    """确保每个测试后单例比特生成器被恢复的夹具"""
    orig_bitgen = np.random.get_bit_generator()  # 获取原始的比特生成器状态
    yield  # 返回当前状态，并继续执行测试
    np.random.set_bit_generator(orig_bitgen)  # 恢复原始的比特生成器状态


def assert_mt19937_state_equal(a, b):
    assert_equal(a['bit_generator'], b['bit_generator'])  # 断言两个状态字典的比特生成器字段相等
    # 断言两个数组的内容是否完全相同，用于比较字典 a 中 'state' 键下的 'key' 对应的数组与字典 b 中的相同内容
    assert_array_equal(a['state']['key'], b['state']['key'])
    
    # 断言两个数组的内容是否完全相同，用于比较字典 a 中 'state' 键下的 'pos' 对应的数组与字典 b 中的相同内容
    assert_array_equal(a['state']['pos'], b['state']['pos'])
    
    # 断言两个变量的值是否相等，用于比较字典 a 中 'has_gauss' 键的值与字典 b 中的相同内容
    assert_equal(a['has_gauss'], b['has_gauss'])
    
    # 断言两个变量的值是否相等，用于比较字典 a 中 'gauss' 键的值与字典 b 中的相同内容
    assert_equal(a['gauss'], b['gauss'])
class TestSeed:
    # 测试随机数生成器的种子设置和随机数生成的正确性

    def test_scalar(self):
        # 测试使用标量作为种子的情况
        s = random.RandomState(0)
        # 断言生成的随机整数为 684
        assert_equal(s.randint(1000), 684)
        s = random.RandomState(4294967295)
        # 断言生成的随机整数为 419
        assert_equal(s.randint(1000), 419)

    def test_array(self):
        # 测试使用数组作为种子的情况
        s = random.RandomState(range(10))
        # 断言生成的随机整数为 468
        assert_equal(s.randint(1000), 468)
        s = random.RandomState(np.arange(10))
        # 断言生成的随机整数为 468
        assert_equal(s.randint(1000), 468)
        s = random.RandomState([0])
        # 断言生成的随机整数为 973
        assert_equal(s.randint(1000), 973)
        s = random.RandomState([4294967295])
        # 断言生成的随机整数为 265
        assert_equal(s.randint(1000), 265)

    def test_invalid_scalar(self):
        # 测试传入无效的标量作为种子时的异常情况
        # 种子必须是一个无符号的32位整数
        assert_raises(TypeError, random.RandomState, -0.5)
        assert_raises(ValueError, random.RandomState, -1)

    def test_invalid_array(self):
        # 测试传入无效的数组作为种子时的异常情况
        # 种子必须是一个无符号的32位整数
        assert_raises(TypeError, random.RandomState, [-0.5])
        assert_raises(ValueError, random.RandomState, [-1])
        assert_raises(ValueError, random.RandomState, [4294967296])
        assert_raises(ValueError, random.RandomState, [1, 2, 4294967296])
        assert_raises(ValueError, random.RandomState, [1, -2, 4294967296])

    def test_invalid_array_shape(self):
        # 测试传入无效的数组形状作为种子时的异常情况
        # GitHub问题编号 #9832
        assert_raises(ValueError, random.RandomState, np.array([], dtype=np.int64))
        assert_raises(ValueError, random.RandomState, [[1, 2, 3]])
        assert_raises(ValueError, random.RandomState, [[1, 2, 3], [4, 5, 6]])

    def test_cannot_seed(self):
        # 测试无法对生成器进行种子设置的情况
        rs = random.RandomState(PCG64(0))
        with assert_raises(TypeError):
            rs.seed(1234)

    def test_invalid_initialization(self):
        # 测试传入无效的生成器初始化参数时的异常情况
        assert_raises(ValueError, random.RandomState, MT19937)


class TestBinomial:
    # 测试二项分布相关函数的正确性

    def test_n_zero(self):
        # 测试 n 等于 0 的情况
        # 二项分布中，当 n 等于 0 时，对任何 p 在 [0, 1] 的情况应该返回 0
        # 这个测试解决问题编号 #3480
        zeros = np.zeros(2, dtype='int')
        for p in [0, .5, 1]:
            assert_(random.binomial(0, p) == 0)
            assert_array_equal(random.binomial(zeros, p), zeros)

    def test_p_is_nan(self):
        # 测试 p 参数为 NaN 时的异常情况
        # GitHub问题编号 #4571
        assert_raises(ValueError, random.binomial, 1, np.nan)


class TestMultinomial:
    # 测试多项分布相关函数的正确性

    def test_basic(self):
        # 测试基本的多项分布生成情况
        random.multinomial(100, [0.2, 0.8])

    def test_zero_probability(self):
        # 测试概率为 0 的情况
        random.multinomial(100, [0.2, 0.8, 0.0, 0.0, 0.0])

    def test_int_negative_interval(self):
        # 测试生成负整数的情况
        assert_(-5 <= random.randint(-5, -1) < -1)
        x = random.randint(-5, -1, 5)
        assert_(np.all(-5 <= x))
        assert_(np.all(x < -1))
    def test_size(self):
        # 测试用例名称：test_size
        # 检测 GitHub issue 3173
        # 定义概率分布数组 p
        p = [0.5, 0.5]
        # 验证随机抽样结果的形状是否为 (1, 2)
        assert_equal(random.multinomial(1, p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.multinomial(1, p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.multinomial(1, p, np.uint32(1)).shape, (1, 2))
        # 验证随机抽样结果的形状是否为 (2, 2, 2)，使用列表作为参数
        assert_equal(random.multinomial(1, p, [2, 2]).shape, (2, 2, 2))
        # 验证随机抽样结果的形状是否为 (2, 2, 2)，使用元组作为参数
        assert_equal(random.multinomial(1, p, (2, 2)).shape, (2, 2, 2))
        # 验证随机抽样结果的形状是否为 (2, 2, 2)，使用 NumPy 数组作为参数
        assert_equal(random.multinomial(1, p, np.array((2, 2))).shape,
                     (2, 2, 2))

        # 预期抛出 TypeError 异常，因为 n 参数应该是整数类型
        assert_raises(TypeError, random.multinomial, 1, p,
                      float(1))

    def test_invalid_prob(self):
        # 测试用例名称：test_invalid_prob
        # 预期抛出 ValueError 异常，因为概率总和超过 1.0
        assert_raises(ValueError, random.multinomial, 100, [1.1, 0.2])
        # 预期抛出 ValueError 异常，因为概率值小于 0
        assert_raises(ValueError, random.multinomial, 100, [-.1, 0.9])

    def test_invalid_n(self):
        # 测试用例名称：test_invalid_n
        # 预期抛出 ValueError 异常，因为 n 参数为负数
        assert_raises(ValueError, random.multinomial, -1, [0.8, 0.2])

    def test_p_non_contiguous(self):
        # 测试用例名称：test_p_non_contiguous
        # 定义 p 数组为非连续的 NumPy 数组
        p = np.arange(15.)
        p /= np.sum(p[1::3])
        pvals = p[1::3]
        # 设置随机数种子
        random.seed(1432985819)
        # 使用非连续的 pvals 数组进行多项式抽样
        non_contig = random.multinomial(100, pvals=pvals)
        # 重新设置相同的随机数种子
        random.seed(1432985819)
        # 使用连续的 pvals 数组进行多项式抽样
        contig = random.multinomial(100, pvals=np.ascontiguousarray(pvals))
        # 验证非连续和连续两种抽样结果是否一致
        assert_array_equal(non_contig, contig)

    def test_multinomial_pvals_float32(self):
        # 测试用例名称：test_multinomial_pvals_float32
        # 定义浮点型数组 x
        x = np.array([9.9e-01, 9.9e-01, 1.0e-09, 1.0e-09, 1.0e-09, 1.0e-09,
                      1.0e-09, 1.0e-09, 1.0e-09, 1.0e-09], dtype=np.float32)
        # 计算 pvals，并保证其和为 1
        pvals = x / x.sum()
        # 预期抛出 ValueError 异常，并匹配指定的错误信息
        match = r"[\w\s]*pvals array is cast to 64-bit floating"
        with pytest.raises(ValueError, match=match):
            # 使用浮点型 pvals 进行多项式抽样
            random.multinomial(1, pvals)

    def test_multinomial_n_float(self):
        # 测试用例名称：test_multinomial_n_float
        # 非整数类型的 n 参数应该会被 gracefully truncate（优雅地截断）成整数
        random.multinomial(100.5, [0.2, 0.8])
class TestSetState:
    # 设置测试方法的初始化方法
    def setup_method(self):
        # 设定随机种子
        self.seed = 1234567890
        # 创建具有指定种子的随机状态对象
        self.random_state = random.RandomState(self.seed)
        # 获取当前随机状态的状态信息
        self.state = self.random_state.get_state()

    # 测试基本功能
    def test_basic(self):
        # 获取当前随机状态的一组随机整数
        old = self.random_state.tomaxint(16)
        # 恢复之前保存的随机状态
        self.random_state.set_state(self.state)
        # 再次获取相同种子下的一组随机整数
        new = self.random_state.tomaxint(16)
        # 断言两次获取的随机整数数组相等
        assert_(np.all(old == new))

    # 测试高斯分布重置
    def test_gaussian_reset(self):
        # 确保缓存的每隔一次高斯分布被重置
        old = self.random_state.standard_normal(size=3)
        # 恢复之前保存的随机状态
        self.random_state.set_state(self.state)
        # 再次获取相同种子下的高斯分布随机数
        new = self.random_state.standard_normal(size=3)
        # 断言两次获取的随机数数组相等
        assert_(np.all(old == new))

    # 在中途保存带有缓存高斯分布的状态时进行测试
    def test_gaussian_reset_in_media_res(self):
        # 当状态保存了缓存高斯分布时，确保缓存高斯分布被恢复
        self.random_state.standard_normal()
        state = self.random_state.get_state()
        old = self.random_state.standard_normal(size=3)
        # 恢复之前保存的随机状态
        self.random_state.set_state(state)
        new = self.random_state.standard_normal(size=3)
        # 断言两次获取的随机数数组相等
        assert_(np.all(old == new))

    # 测试向后兼容性
    def test_backwards_compatibility(self):
        # 确保可以接受不含缓存高斯值的旧状态元组
        old_state = self.state[:-2]
        x1 = self.random_state.standard_normal(size=16)
        # 设置为旧的随机状态
        self.random_state.set_state(old_state)
        x2 = self.random_state.standard_normal(size=16)
        # 恢复当前保存的随机状态
        self.random_state.set_state(self.state)
        x3 = self.random_state.standard_normal(size=16)
        # 断言三次获取的随机数数组相等
        assert_(np.all(x1 == x2))
        assert_(np.all(x1 == x3))

    # 测试负二项分布
    def test_negative_binomial(self):
        # 确保负二项分布结果接受浮点参数而不截断
        self.random_state.negative_binomial(0.5, 0.5)

    # 测试获取状态时的警告
    def test_get_state_warning(self):
        rs = random.RandomState(PCG64())
        with suppress_warnings() as sup:
            w = sup.record(RuntimeWarning)
            state = rs.get_state()
            # 断言捕获到一条运行时警告
            assert_(len(w) == 1)
            # 断言状态是一个字典对象
            assert isinstance(state, dict)
            # 断言状态中的比特生成器为'PCG64'
            assert state['bit_generator'] == 'PCG64'

    # 测试设置无效的旧状态
    def test_invalid_legacy_state_setting(self):
        state = self.random_state.get_state()
        new_state = ('Unknown', ) + state[1:]
        # 断言设置无效旧状态会引发值错误异常
        assert_raises(ValueError, self.random_state.set_state, new_state)
        # 断言设置无效旧状态会引发类型错误异常
        assert_raises(TypeError, self.random_state.set_state,
                      np.array(new_state, dtype=object))
        state = self.random_state.get_state(legacy=False)
        del state['bit_generator']
        # 断言设置无效旧状态会引发值错误异常
        assert_raises(ValueError, self.random_state.set_state, state)
    # 定义测试函数，测试序列化和反序列化的功能

    def test_pickle(self):
        # 设定随机种子为0
        self.random_state.seed(0)
        # 生成100个随机样本
        self.random_state.random_sample(100)
        # 生成标准正态分布随机数
        self.random_state.standard_normal()
        # 获取当前状态的序列化表示，legacy=False 表示使用新的序列化格式
        pickled = self.random_state.get_state(legacy=False)
        # 断言是否具有高斯状态，应为1
        assert_equal(pickled['has_gauss'], 1)
        # 反序列化恢复状态
        rs_unpick = pickle.loads(pickle.dumps(self.random_state))
        # 获取恢复状态的序列化表示
        unpickled = rs_unpick.get_state(legacy=False)
        # 断言两个 MT19937 状态是否相等
        assert_mt19937_state_equal(pickled, unpickled)

    # 定义测试函数，测试状态设置功能
    def test_state_setting(self):
        # 获取当前对象的状态
        attr_state = self.random_state.__getstate__()
        # 生成标准正态分布随机数
        self.random_state.standard_normal()
        # 恢复对象状态
        self.random_state.__setstate__(attr_state)
        # 获取恢复后的状态
        state = self.random_state.get_state(legacy=False)
        # 断言两个 MT19937 状态是否相等
        assert_mt19937_state_equal(attr_state, state)

    # 定义测试函数，测试对象的字符串表示
    def test_repr(self):
        # 断言对象的字符串表示是否以指定字符串开头
        assert repr(self.random_state).startswith('RandomState(MT19937)')
# 定义一个测试类 TestRandint，用于测试 random.randint 函数的行为
class TestRandint:

    # 将 random.randint 赋给类属性 rfunc，方便后续测试函数调用
    rfunc = random.randint

    # 定义支持的整数和布尔类型列表
    itype = [np.bool, np.int8, np.uint8, np.int16, np.uint16,
             np.int32, np.uint32, np.int64, np.uint64]

    # 测试不支持的数据类型是否会触发 TypeError 异常
    def test_unsupported_type(self):
        assert_raises(TypeError, self.rfunc, 1, dtype=float)

    # 测试边界检查功能
    def test_bounds_checking(self):
        # 遍历支持的整数和布尔类型
        for dt in self.itype:
            # 确定下界 lbnd 和上界 ubnd
            lbnd = 0 if dt is np.bool else np.iinfo(dt).min
            ubnd = 2 if dt is np.bool else np.iinfo(dt).max + 1
            # 测试是否能正确抛出 ValueError 异常
            assert_raises(ValueError, self.rfunc, lbnd - 1, ubnd, dtype=dt)
            assert_raises(ValueError, self.rfunc, lbnd, ubnd + 1, dtype=dt)
            assert_raises(ValueError, self.rfunc, ubnd, lbnd, dtype=dt)
            assert_raises(ValueError, self.rfunc, 1, 0, dtype=dt)

    # 测试随机数生成器在边界情况下的行为
    def test_rng_zero_and_extremes(self):
        # 遍历支持的整数和布尔类型
        for dt in self.itype:
            # 确定下界 lbnd 和上界 ubnd
            lbnd = 0 if dt is np.bool else np.iinfo(dt).min
            ubnd = 2 if dt is np.bool else np.iinfo(dt).max + 1

            # 测试生成恰好等于 ubnd - 1 的随机数
            tgt = ubnd - 1
            assert_equal(self.rfunc(tgt, tgt + 1, size=1000, dtype=dt), tgt)

            # 测试生成恰好等于 lbnd 的随机数
            tgt = lbnd
            assert_equal(self.rfunc(tgt, tgt + 1, size=1000, dtype=dt), tgt)

            # 测试生成介于 lbnd 和 ubnd 中间的随机数
            tgt = (lbnd + ubnd) // 2
            assert_equal(self.rfunc(tgt, tgt + 1, size=1000, dtype=dt), tgt)

    # 测试随机数生成器在整个范围内的行为
    def test_full_range(self):
        # 测试 ticket #1690 的问题
        for dt in self.itype:
            # 确定下界 lbnd 和上界 ubnd
            lbnd = 0 if dt is np.bool else np.iinfo(dt).min
            ubnd = 2 if dt is np.bool else np.iinfo(dt).max + 1

            # 测试在指定范围内生成随机数是否会抛出异常
            try:
                self.rfunc(lbnd, ubnd, dtype=dt)
            except Exception as e:
                raise AssertionError("No error should have been raised, "
                                     "but one was with the following "
                                     "message:\n\n%s" % str(e))

    # 测试在边界内进行随机生成的行为
    def test_in_bounds_fuzz(self):
        # 不使用固定的随机种子
        random.seed()

        # 遍历除 np.bool 外的整数和布尔类型
        for dt in self.itype[1:]:
            # 遍历不同的上界值
            for ubnd in [4, 8, 16]:
                # 生成大量随机数并检查是否在指定范围内
                vals = self.rfunc(2, ubnd, size=2**16, dtype=dt)
                assert_(vals.max() < ubnd)
                assert_(vals.min() >= 2)

        # 对布尔类型进行特殊测试
        vals = self.rfunc(0, 2, size=2**16, dtype=np.bool)
        assert_(vals.max() < 2)
        assert_(vals.min() >= 0)
    def test_repeatability(self):
        # 测试函数的重复性

        # 定义期望的哈希值，用于不同数据类型生成的序列
        tgt = {'bool': '509aea74d792fb931784c4b0135392c65aec64beee12b0cc167548a2c3d31e71',
               'int16': '7b07f1a920e46f6d0fe02314155a2330bcfd7635e708da50e536c5ebb631a7d4',
               'int32': 'e577bfed6c935de944424667e3da285012e741892dcb7051a8f1ce68ab05c92f',
               'int64': '0fbead0b06759df2cfb55e43148822d4a1ff953c7eb19a5b08445a63bb64fa9e',
               'int8': '001aac3a5acb935a9b186cbe14a1ca064b8bb2dd0b045d48abeacf74d0203404',
               'uint16': '7b07f1a920e46f6d0fe02314155a2330bcfd7635e708da50e536c5ebb631a7d4',
               'uint32': 'e577bfed6c935de944424667e3da285012e741892dcb7051a8f1ce68ab05c92f',
               'uint64': '0fbead0b06759df2cfb55e43148822d4a1ff953c7eb19a5b08445a63bb64fa9e',
               'uint8': '001aac3a5acb935a9b186cbe14a1ca064b8bb2dd0b045d48abeacf74d0203404'}

        # 对除了布尔类型以外的所有数据类型，生成特定范围内的随机序列，并计算其哈希值
        for dt in self.itype[1:]:
            random.seed(1234)

            # 如果系统的字节顺序是小端序列，视图将转换为小端序列以进行哈希
            if sys.byteorder == 'little':
                val = self.rfunc(0, 6, size=1000, dtype=dt)
            else:
                val = self.rfunc(0, 6, size=1000, dtype=dt).byteswap()

            # 计算序列的哈希值并转换为十六进制表示
            res = hashlib.sha256(val.view(np.int8)).hexdigest()

            # 断言计算得到的哈希值与预期的哈希值相等
            assert_(tgt[np.dtype(dt).name] == res)

        # 对于布尔类型，不依赖于字节顺序
        random.seed(1234)
        val = self.rfunc(0, 2, size=1000, dtype=bool).view(np.int8)

        # 计算布尔序列的哈希值并转换为十六进制表示
        res = hashlib.sha256(val).hexdigest()

        # 断言计算得到的哈希值与预期的哈希值相等
        assert_(tgt[np.dtype(bool).name] == res)

    @pytest.mark.skipif(np.iinfo('l').max < 2**32,
                        reason='Cannot test with 32-bit C long')
    # 定义一个测试方法，用于测试在 32 位边界广播时的重复性
    def test_repeatability_32bit_boundary_broadcasting(self):
        # 定义预期的三维 NumPy 数组，包含多个子数组，每个子数组中有三个整数
        desired = np.array([[[3992670689, 2438360420, 2557845020],
                             [4107320065, 4142558326, 3216529513],
                             [1605979228, 2807061240,  665605495]],
                            [[3211410639, 4128781000,  457175120],
                             [1712592594, 1282922662, 3081439808],
                             [3997822960, 2008322436, 1563495165]],
                            [[1398375547, 4269260146,  115316740],
                             [3414372578, 3437564012, 2112038651],
                             [3572980305, 2260248732, 3908238631]],
                            [[2561372503,  223155946, 3127879445],
                             [ 441282060, 3514786552, 2148440361],
                             [1629275283, 3479737011, 3003195987]],
                            [[ 412181688,  940383289, 3047321305],
                             [2978368172,  764731833, 2282559898],
                             [ 105711276,  720447391, 3596512484]]])
        # 对于不同的大小设置，进行测试
        for size in [None, (5, 3, 3)]:
            # 设置随机数种子以保证结果的可重复性
            random.seed(12345)
            # 调用 self.rfunc 方法生成随机数数组 x，其范围由 [-1, 0, 1] 和 [2^32 - 1, 2^32, 2^32 + 1] 决定
            # size 参数指定数组的形状
            x = self.rfunc([[-1], [0], [1]], [2**32 - 1, 2**32, 2**32 + 1],
                           size=size)
            # 断言生成的随机数数组 x 是否与预期的数组 desired 相等，如果 size 不为 None，则只比较第一个子数组
            assert_array_equal(x, desired if size is not None else desired[0])

    # 定义一个测试方法，用于测试 int64 和 uint64 边界情况
    def test_int64_uint64_corner_case(self):
        # 当存储在 NumPy 数组中时，lbnd 被转换为 np.int64，ubnd 被转换为 np.uint64
        # 检查 lbnd 是否大于等于 ubnd 曾经通过直接比较完成，这是不正确的，因为当 NumPy 尝试比较两个数时，
        # 会将它们都转换为 np.float64，但 ubnd 太大无法表示为 np.float64，导致它被舍入为 np.iinfo(np.int64).max，
        # 这引发了 ValueError，因为 lbnd 现在等于新的 ubnd。

        # 设置数据类型为 np.int64
        dt = np.int64
        # 设置目标值为 np.int64 的最大值
        tgt = np.iinfo(np.int64).max
        # 将 lbnd 设置为 np.int64 的最大值
        lbnd = np.int64(np.iinfo(np.int64).max)
        # 将 ubnd 设置为 np.int64 的最大值加一，并转换为 np.uint64
        ubnd = np.uint64(np.iinfo(np.int64).max + 1)

        # 现在这些函数调用不应该生成 ValueError
        # 使用指定的数据类型 dt 生成在 [lbnd, ubnd] 范围内的随机整数 actual
        actual = random.randint(lbnd, ubnd, dtype=dt)
        # 断言生成的随机数 actual 是否等于预期的目标值 tgt
        assert_equal(actual, tgt)
    def test_respect_dtype_singleton(self):
        # 用例 gh-7203
        # 遍历 self.itype 中的数据类型
        for dt in self.itype:
            # 如果 dt 是 np.bool，则 lbnd 为 0，否则为 np.iinfo(dt).min
            lbnd = 0 if dt is np.bool else np.iinfo(dt).min
            # 如果 dt 是 np.bool，则 ubnd 为 2，否则为 np.iinfo(dt).max + 1
            ubnd = 2 if dt is np.bool else np.iinfo(dt).max + 1

            # 使用 self.rfunc 函数生成指定范围内的随机数 sample，并指定 dtype 为 dt
            sample = self.rfunc(lbnd, ubnd, dtype=dt)
            # 断言 sample 的数据类型为 np.dtype(dt)
            assert_equal(sample.dtype, np.dtype(dt))

        # 遍历 bool 和 int 这两种数据类型
        for dt in (bool, int):
            # 对于 legacy 随机生成，当输入为 `int` 且默认 dtype 为 int 时，强制使用 "long"
            # 当 dt 是 int 时，op_dtype 为 "long"，否则为 "bool"
            op_dtype = "long" if dt is int else "bool"
            # 如果 dt 是 bool，则 lbnd 为 0，否则为 np.iinfo(op_dtype).min
            lbnd = 0 if dt is bool else np.iinfo(op_dtype).min
            # 如果 dt 是 bool，则 ubnd 为 2，否则为 np.iinfo(op_dtype).max + 1
            ubnd = 2 if dt is bool else np.iinfo(op_dtype).max + 1

            # 使用 self.rfunc 函数生成指定范围内的随机数 sample，并指定 dtype 为 dt
            sample = self.rfunc(lbnd, ubnd, dtype=dt)
            # 断言 sample 没有 'dtype' 属性
            assert_(not hasattr(sample, 'dtype'))
            # 断言 sample 的类型为 dt
            assert_equal(type(sample), dt)
class TestRandomDist:
    # 确保随机分布对于给定的种子返回正确的值

    def setup_method(self):
        # 设置测试方法的初始化操作，设置种子为固定值
        self.seed = 1234567890

    def test_rand(self):
        # 测试 random.rand 方法
        random.seed(self.seed)
        actual = random.rand(3, 2)
        desired = np.array([[0.61879477158567997, 0.59162362775974664],
                            [0.88868358904449662, 0.89165480011560816],
                            [0.4575674820298663, 0.7781880808593471]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_rand_singleton(self):
        # 测试单个值的 random.rand 方法
        random.seed(self.seed)
        actual = random.rand()
        desired = 0.61879477158567997
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_randn(self):
        # 测试 random.randn 方法
        random.seed(self.seed)
        actual = random.randn(3, 2)
        desired = np.array([[1.34016345771863121, 1.73759122771936081],
                            [1.498988344300628, -0.2286433324536169],
                            [2.031033998682787, 2.17032494605655257]])
        assert_array_almost_equal(actual, desired, decimal=15)

        # 测试单个值的 random.randn 方法
        random.seed(self.seed)
        actual = random.randn()
        assert_array_almost_equal(actual, desired[0, 0], decimal=15)

    def test_randint(self):
        # 测试 random.randint 方法
        random.seed(self.seed)
        actual = random.randint(-99, 99, size=(3, 2))
        desired = np.array([[31, 3],
                            [-52, 41],
                            [-48, -66]])
        assert_array_equal(actual, desired)

    def test_random_integers(self):
        # 测试 random.random_integers 方法
        random.seed(self.seed)
        with suppress_warnings() as sup:
            w = sup.record(DeprecationWarning)
            actual = random.random_integers(-99, 99, size=(3, 2))
            assert_(len(w) == 1)
        desired = np.array([[31, 3],
                            [-52, 41],
                            [-48, -66]])
        assert_array_equal(actual, desired)

        # 测试不建议使用的 random.random_integers 方法
        random.seed(self.seed)
        with suppress_warnings() as sup:
            w = sup.record(DeprecationWarning)
            actual = random.random_integers(198, size=(3, 2))
            assert_(len(w) == 1)
        assert_array_equal(actual, desired + 100)
    def test_tomaxint(self):
        # 设置随机数种子
        random.seed(self.seed)
        # 使用指定种子创建随机状态对象
        rs = random.RandomState(self.seed)
        # 调用随机状态对象的tomaxint方法生成随机整数数组
        actual = rs.tomaxint(size=(3, 2))
        
        # 检查当前环境下Python long类型的最大值是否为2147483647
        if np.iinfo(np.long).max == 2147483647:
            # 如果是，则设置期望的结果为指定的整数数组
            desired = np.array([[1328851649,  731237375],
                                [1270502067,  320041495],
                                [1908433478,  499156889]], dtype=np.int64)
        else:
            # 如果不是，则设置期望的结果为另一个指定的整数数组
            desired = np.array([[5707374374421908479, 5456764827585442327],
                                [8196659375100692377, 8224063923314595285],
                                [4220315081820346526, 7177518203184491332]],
                               dtype=np.int64)

        # 断言实际结果与期望结果相等
        assert_equal(actual, desired)

        # 使用相同的种子重新设定随机状态对象
        rs.seed(self.seed)
        # 再次调用tomaxint方法生成随机整数，只检查返回的第一个整数
        actual = rs.tomaxint()
        assert_equal(actual, desired[0, 0])

    def test_random_integers_max_int(self):
        # 测试random_integers方法是否能生成Python int类型的最大值
        # 这个最大值可以转换成C long类型。先前的实现在生成这个整数时会抛出OverflowError。
        with suppress_warnings() as sup:
            # 记录并忽略DeprecationWarning警告
            w = sup.record(DeprecationWarning)
            # 调用random_integers方法生成指定范围内的随机整数
            actual = random.random_integers(np.iinfo('l').max,
                                            np.iinfo('l').max)
            # 断言警告记录的数量为1
            assert_(len(w) == 1)

        # 设置期望的结果为Python int类型的最大值
        desired = np.iinfo('l').max
        # 断言实际结果与期望结果相等
        assert_equal(actual, desired)
        
        with suppress_warnings() as sup:
            # 记录并忽略DeprecationWarning警告
            w = sup.record(DeprecationWarning)
            # 获取C long类型的dtype对象
            typer = np.dtype('l').type
            # 调用random_integers方法生成指定范围内的随机整数
            actual = random.random_integers(typer(np.iinfo('l').max),
                                            typer(np.iinfo('l').max))
            # 断言警告记录的数量为1
            assert_(len(w) == 1)
        # 断言实际结果与期望结果相等
        assert_equal(actual, desired)

    def test_random_integers_deprecated(self):
        # 测试random_integers方法在特定条件下是否会抛出DeprecationWarning警告
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)

            # 断言调用random_integers方法会抛出DeprecationWarning警告，
            # 当high参数为Python int类型的最大值时
            assert_raises(DeprecationWarning,
                          random.random_integers,
                          np.iinfo('l').max)

            # 断言调用random_integers方法会抛出DeprecationWarning警告，
            # 当high参数不为None且为Python int类型的最大值时
            assert_raises(DeprecationWarning,
                          random.random_integers,
                          np.iinfo('l').max, np.iinfo('l').max)

    def test_random_sample(self):
        # 设置随机数种子
        random.seed(self.seed)
        # 调用random_sample方法生成指定形状的随机数数组
        actual = random.random_sample((3, 2))
        # 设置期望的结果为指定的随机数数组
        desired = np.array([[0.61879477158567997, 0.59162362775974664],
                            [0.88868358904449662, 0.89165480011560816],
                            [0.4575674820298663, 0.7781880808593471]])
        # 断言实际结果与期望结果在给定精度下相等
        assert_array_almost_equal(actual, desired, decimal=15)

        # 使用相同的种子重新设定随机数种子
        random.seed(self.seed)
        # 再次调用random_sample方法生成随机数，只检查返回的第一个数
        actual = random.random_sample()
        # 断言实际结果与期望结果在给定精度下相等
        assert_array_almost_equal(actual, desired[0, 0], decimal=15)
    # 定义一个测试方法，测试从一个长度为4的集合中有放回地选择4个元素
    def test_choice_uniform_replace(self):
        # 使用预设的随机种子初始化随机数生成器
        random.seed(self.seed)
        # 使用随机数生成器从集合中有放回地选择4个元素
        actual = random.choice(4, 4)
        # 预期结果是一个包含[2, 3, 2, 3]的NumPy数组
        desired = np.array([2, 3, 2, 3])
        # 断言选择的结果与预期结果相等
        assert_array_equal(actual, desired)

    # 定义一个测试方法，测试从一个长度为4的集合中有放回地选择4个元素，但选择概率不均匀
    def test_choice_nonuniform_replace(self):
        # 使用预设的随机种子初始化随机数生成器
        random.seed(self.seed)
        # 使用随机数生成器从集合中有放回地选择4个元素，并指定每个元素的选择概率
        actual = random.choice(4, 4, p=[0.4, 0.4, 0.1, 0.1])
        # 预期结果是一个包含[1, 1, 2, 2]的NumPy数组
        desired = np.array([1, 1, 2, 2])
        # 断言选择的结果与预期结果相等
        assert_array_equal(actual, desired)

    # 定义一个测试方法，测试从一个长度为4的集合中无放回地选择3个元素
    def test_choice_uniform_noreplace(self):
        # 使用预设的随机种子初始化随机数生成器
        random.seed(self.seed)
        # 使用随机数生成器从集合中无放回地选择3个元素
        actual = random.choice(4, 3, replace=False)
        # 预期结果是一个包含[0, 1, 3]的NumPy数组
        desired = np.array([0, 1, 3])
        # 断言选择的结果与预期结果相等
        assert_array_equal(actual, desired)

    # 定义一个测试方法，测试从一个长度为4的集合中无放回地选择3个元素，但选择概率不均匀
    def test_choice_nonuniform_noreplace(self):
        # 使用预设的随机种子初始化随机数生成器
        random.seed(self.seed)
        # 使用随机数生成器从集合中无放回地选择3个元素，并指定每个元素的选择概率
        actual = random.choice(4, 3, replace=False, p=[0.1, 0.3, 0.5, 0.1])
        # 预期结果是一个包含[2, 3, 1]的NumPy数组
        desired = np.array([2, 3, 1])
        # 断言选择的结果与预期结果相等
        assert_array_equal(actual, desired)

    # 定义一个测试方法，测试从一个包含字符的集合中有放回地选择4个元素
    def test_choice_noninteger(self):
        # 使用预设的随机种子初始化随机数生成器
        random.seed(self.seed)
        # 使用随机数生成器从字符集合中有放回地选择4个元素
        actual = random.choice(['a', 'b', 'c', 'd'], 4)
        # 预期结果是一个包含['c', 'd', 'c', 'd']的NumPy数组
        desired = np.array(['c', 'd', 'c', 'd'])
        # 断言选择的结果与预期结果相等
        assert_array_equal(actual, desired)

    # 定义一个测试方法，测试对异常情况的处理
    def test_choice_exceptions(self):
        # 将 random.choice 赋值给 sample，用于简化代码
        sample = random.choice
        # 断言在选择范围为负数时引发 ValueError 异常
        assert_raises(ValueError, sample, -1, 3)
        # 断言在选择范围为浮点数时引发 ValueError 异常
        assert_raises(ValueError, sample, 3., 3)
        # 断言在选择的集合为嵌套列表时引发 ValueError 异常
        assert_raises(ValueError, sample, [[1, 2], [3, 4]], 3)
        # 断言在选择的集合为空时引发 ValueError 异常
        assert_raises(ValueError, sample, [], 3)
        # 断言在选择概率矩阵维度不匹配时引发 ValueError 异常
        assert_raises(ValueError, sample, [1, 2, 3, 4], 3,
                      p=[[0.25, 0.25], [0.25, 0.25]])
        # 断言在选择概率总和不为1时引发 ValueError 异常
        assert_raises(ValueError, sample, [1, 2], 3, p=[0.4, 0.4, 0.2])
        # 断言在选择概率包含负数或大于1的值时引发 ValueError 异常
        assert_raises(ValueError, sample, [1, 2], 3, p=[1.1, -0.1])
        # 断言在无放回选择时，选择的数量大于集合大小时引发 ValueError 异常
        assert_raises(ValueError, sample, [1, 2, 3], 4, replace=False)
        # gh-13087: 断言在选择数量为负数时引发 ValueError 异常
        assert_raises(ValueError, sample, [1, 2, 3], -2, replace=False)
        # 断言在选择数量为元组且包含负数时引发 ValueError 异常
        assert_raises(ValueError, sample, [1, 2, 3], (-1,), replace=False)
        # 断言在选择数量为元组且包含负数时引发 ValueError 异常
        assert_raises(ValueError, sample, [1, 2, 3], (-1, 1), replace=False)
        # 断言在无放回选择时，选择概率矩阵与集合大小不匹配时引发 ValueError 异常
        assert_raises(ValueError, sample, [1, 2, 3], 2,
                      replace=False, p=[1, 0, 0])
    # 测试函数：验证 random.choice 函数在不同情况下的返回结果的形状和类型

    def test_choice_return_shape(self):
        # 定义概率分布
        p = [0.1, 0.9]
        
        # 检查标量情况下的返回结果
        assert_(np.isscalar(random.choice(2, replace=True)))  # 使用随机选择函数选择标量值，检查是否为标量
        assert_(np.isscalar(random.choice(2, replace=False)))  # 使用随机选择函数选择标量值，检查是否为标量
        assert_(np.isscalar(random.choice(2, replace=True, p=p)))  # 使用带有概率分布的随机选择函数选择标量值，检查是否为标量
        assert_(np.isscalar(random.choice(2, replace=False, p=p)))  # 使用带有概率分布的随机选择函数选择标量值，检查是否为标量
        assert_(np.isscalar(random.choice([1, 2], replace=True)))  # 使用随机选择函数选择标量值，检查是否为标量
        assert_(random.choice([None], replace=True) is None)  # 使用随机选择函数选择 None 类型，验证是否返回 None
        a = np.array([1, 2])
        arr = np.empty(1, dtype=object)
        arr[0] = a
        assert_(random.choice(arr, replace=True) is a)  # 使用随机选择函数选择数组中的对象，验证是否返回正确的对象

        # 检查 0 维数组情况
        s = tuple()
        assert_(not np.isscalar(random.choice(2, s, replace=True)))  # 使用随机选择函数选择数组，检查是否不是标量
        assert_(not np.isscalar(random.choice(2, s, replace=False)))  # 使用随机选择函数选择数组，检查是否不是标量
        assert_(not np.isscalar(random.choice(2, s, replace=True, p=p)))  # 使用带有概率分布的随机选择函数选择数组，检查是否不是标量
        assert_(not np.isscalar(random.choice(2, s, replace=False, p=p)))  # 使用带有概率分布的随机选择函数选择数组，检查是否不是标量
        assert_(not np.isscalar(random.choice([1, 2], s, replace=True)))  # 使用随机选择函数选择数组，检查是否不是标量
        assert_(random.choice([None], s, replace=True).ndim == 0)  # 使用随机选择函数选择 None 类型的数组，验证维度是否为 0
        a = np.array([1, 2])
        arr = np.empty(1, dtype=object)
        arr[0] = a
        assert_(random.choice(arr, s, replace=True).item() is a)  # 使用随机选择函数选择数组中的对象，验证是否返回正确的对象

        # 检查多维数组情况
        s = (2, 3)
        p = [0.1, 0.1, 0.1, 0.1, 0.4, 0.2]
        assert_equal(random.choice(6, s, replace=True).shape, s)  # 使用随机选择函数选择多维数组，验证形状是否正确
        assert_equal(random.choice(6, s, replace=False).shape, s)  # 使用随机选择函数选择多维数组，验证形状是否正确
        assert_equal(random.choice(6, s, replace=True, p=p).shape, s)  # 使用带有概率分布的随机选择函数选择多维数组，验证形状是否正确
        assert_equal(random.choice(6, s, replace=False, p=p).shape, s)  # 使用带有概率分布的随机选择函数选择多维数组，验证形状是否正确
        assert_equal(random.choice(np.arange(6), s, replace=True).shape, s)  # 使用随机选择函数选择多维数组，验证形状是否正确

        # 检查零大小情况
        assert_equal(random.randint(0, 0, size=(3, 0, 4)).shape, (3, 0, 4))  # 生成零大小的数组，验证形状是否正确
        assert_equal(random.randint(0, -10, size=0).shape, (0,))  # 生成零大小的数组，验证形状是否正确
        assert_equal(random.randint(10, 10, size=0).shape, (0,))  # 生成零大小的数组，验证形状是否正确
        assert_equal(random.choice(0, size=0).shape, (0,))  # 生成零大小的数组，验证形状是否正确
        assert_equal(random.choice([], size=(0,)).shape, (0,))  # 生成零大小的数组，验证形状是否正确
        assert_equal(random.choice(['a', 'b'], size=(3, 0, 4)).shape, (3, 0, 4))  # 使用随机选择函数选择多维数组，验证形状是否正确
        assert_raises(ValueError, random.choice, [], 10)  # 使用随机选择函数选择空列表，验证是否引发 ValueError

    # 测试函数：验证 random.choice 函数在概率分布含有 NaN 值时是否会引发 ValueError

    def test_choice_nan_probabilities(self):
        a = np.array([42, 1, 2])
        p = [None, None, None]
        assert_raises(ValueError, random.choice, a, p=p)  # 使用带有 NaN 值的概率分布调用随机选择函数，验证是否引发 ValueError

    # 测试函数：验证 random.choice 函数在使用非连续的概率分布时的结果是否与连续的概率分布一致

    def test_choice_p_non_contiguous(self):
        p = np.ones(10) / 5
        p[1::2] = 3.0
        random.seed(self.seed)
        non_contig = random.choice(5, 3, p=p[::2])  # 使用非连续的概率分布调用随机选择函数，保存结果
        random.seed(self.seed)
        contig = random.choice(5, 3, p=np.ascontiguousarray(p[::2]))  # 使用连续的概率分布调用随机选择函数，保存结果
        assert_array_equal(non_contig, contig)  # 检查两种调用方式的结果是否一致

    # 测试函数：验证 random.bytes 函数生成的随机字节流是否符合预期

    def test_bytes(self):
        random.seed(self.seed)
        actual = random.bytes(10)  # 生成指定长度的随机字节流
        desired = b'\x82Ui\x9e\xff\x97+Wf\xa5'
        assert_equal(actual, desired)  # 检查生成的随机字节流是否与预期相符
    def test_shuffle(self):
        # 测试列表、数组（不同数据类型）、多维数组以及是否连续存储的版本
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
            # 使用给定的转换函数生成输入数据，并设置随机种子
            random.seed(self.seed)
            alist = conv([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
            # 对生成的数据进行随机排列
            random.shuffle(alist)
            actual = alist
            # 预期的排序结果
            desired = conv([0, 1, 9, 6, 2, 4, 5, 8, 7, 3])
            # 断言实际结果与预期结果一致
            assert_array_equal(actual, desired)

    def test_shuffle_masked(self):
        # gh-3263
        # 创建带有掩码值的数组a和b
        a = np.ma.masked_values(np.reshape(range(20), (5, 4)) % 3 - 1, -1)
        b = np.ma.masked_values(np.arange(20) % 3 - 1, -1)
        a_orig = a.copy()
        b_orig = b.copy()
        for i in range(50):
            # 对数组a和b进行随机排列
            random.shuffle(a)
            # 断言排列后非掩码部分数据的排序与原始数组a的一致性
            assert_equal(
                sorted(a.data[~a.mask]), sorted(a_orig.data[~a_orig.mask]))
            random.shuffle(b)
            # 断言排列后非掩码部分数据的排序与原始数组b的一致性
            assert_equal(
                sorted(b.data[~b.mask]), sorted(b_orig.data[~b_orig.mask]))

    def test_shuffle_invalid_objects(self):
        # 测试对非法对象进行随机排列的情况
        x = np.array(3)
        assert_raises(TypeError, random.shuffle, x)

    def test_permutation(self):
        # 使用设定的随机种子初始化随机数生成器
        random.seed(self.seed)
        alist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        # 对列表进行随机排列
        actual = random.permutation(alist)
        # 预期的排序结果
        desired = [0, 1, 9, 6, 2, 4, 5, 8, 7, 3]
        # 断言实际结果与预期结果一致
        assert_array_equal(actual, desired)

        random.seed(self.seed)
        # 创建二维数组，并对其进行随机排列
        arr_2d = np.atleast_2d([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]).T
        actual = random.permutation(arr_2d)
        assert_array_equal(actual, np.atleast_2d(desired).T)

        random.seed(self.seed)
        # 测试对不合法的字符串进行随机排列
        bad_x_str = "abcd"
        assert_raises(IndexError, random.permutation, bad_x_str)

        random.seed(self.seed)
        # 测试对不合法的浮点数进行随机排列
        bad_x_float = 1.2
        assert_raises(IndexError, random.permutation, bad_x_float)

        integer_val = 10
        desired = [9, 0, 8, 5, 1, 3, 4, 7, 6, 2]

        random.seed(self.seed)
        # 对整数进行随机排列
        actual = random.permutation(integer_val)
        assert_array_equal(actual, desired)
    # 定义一个测试函数，用于测试 beta 分布生成的随机数是否符合预期
    def test_beta(self):
        # 设置随机数种子，保证结果可复现
        random.seed(self.seed)
        # 生成指定参数下的 beta 分布随机数数组
        actual = random.beta(.1, .9, size=(3, 2))
        # 预期的 beta 分布随机数数组
        desired = np.array(
                [[1.45341850513746058e-02, 5.31297615662868145e-04],
                 [1.85366619058432324e-06, 4.19214516800110563e-03],
                 [1.58405155108498093e-04, 1.26252891949397652e-04]])
        # 使用 assert_array_almost_equal 检查生成的随机数数组与预期数组是否几乎相等
        assert_array_almost_equal(actual, desired, decimal=15)

    # 定义一个测试函数，用于测试二项分布生成的随机数是否符合预期
    def test_binomial(self):
        # 设置随机数种子，保证结果可复现
        random.seed(self.seed)
        # 生成指定参数下的二项分布随机数数组
        actual = random.binomial(100.123, .456, size=(3, 2))
        # 预期的二项分布随机数数组
        desired = np.array([[37, 43],
                            [42, 48],
                            [46, 45]])
        # 使用 assert_array_equal 检查生成的随机数数组与预期数组是否完全相等
        assert_array_equal(actual, desired)

        # 重新设置随机数种子，生成单个二项分布随机数
        random.seed(self.seed)
        actual = random.binomial(100.123, .456)
        desired = 37
        # 使用 assert_array_equal 检查生成的单个随机数与预期值是否相等
        assert_array_equal(actual, desired)

    # 定义一个测试函数，用于测试卡方分布生成的随机数是否符合预期
    def test_chisquare(self):
        # 设置随机数种子，保证结果可复现
        random.seed(self.seed)
        # 生成指定参数下的卡方分布随机数数组
        actual = random.chisquare(50, size=(3, 2))
        # 预期的卡方分布随机数数组
        desired = np.array([[63.87858175501090585, 68.68407748911370447],
                            [65.77116116901505904, 47.09686762438974483],
                            [72.3828403199695174, 74.18408615260374006]])
        # 使用 assert_array_almost_equal 检查生成的随机数数组与预期数组是否几乎相等
        assert_array_almost_equal(actual, desired, decimal=13)

    # 定义一个测试函数，用于测试狄利克雷分布生成的随机数是否符合预期
    def test_dirichlet(self):
        # 设置随机数种子，保证结果可复现
        random.seed(self.seed)
        # 定义狄利克雷分布的参数向量
        alpha = np.array([51.72840233779265162, 39.74494232180943953])
        # 生成指定参数下的狄利克雷分布随机数数组
        actual = random.dirichlet(alpha, size=(3, 2))
        # 预期的狄利克雷分布随机数数组
        desired = np.array([[[0.54539444573611562, 0.45460555426388438],
                             [0.62345816822039413, 0.37654183177960598]],
                            [[0.55206000085785778, 0.44793999914214233],
                             [0.58964023305154301, 0.41035976694845688]],
                            [[0.59266909280647828, 0.40733090719352177],
                             [0.56974431743975207, 0.43025568256024799]]])
        # 使用 assert_array_almost_equal 检查生成的随机数数组与预期数组是否几乎相等
        assert_array_almost_equal(actual, desired, decimal=15)
        # 测试异常情况：不合法的 alpha 值应引发 ValueError 异常
        bad_alpha = np.array([5.4e-01, -1.0e-16])
        assert_raises(ValueError, random.dirichlet, bad_alpha)

        # 重新设置随机数种子，生成单个狄利克雷分布随机数
        random.seed(self.seed)
        actual = random.dirichlet(alpha)
        # 使用 assert_array_almost_equal 检查生成的单个随机数与预期值是否几乎相等
        assert_array_almost_equal(actual, desired[0, 0], decimal=15)

    # 定义一个测试函数，用于测试狄利克雷分布生成的随机数在不同 size 参数下的形状是否符合预期
    def test_dirichlet_size(self):
        # 测试 GitHub issue 3173：验证不同 size 参数下狄利克雷分布随机数的形状是否正确
        p = np.array([51.72840233779265162, 39.74494232180943953])
        assert_equal(random.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.dirichlet(p, [2, 2]).shape, (2, 2, 2))
        assert_equal(random.dirichlet(p, (2, 2)).shape, (2, 2, 2))
        assert_equal(random.dirichlet(p, np.array((2, 2))).shape, (2, 2, 2))

        # 测试异常情况：size 参数类型为 float 应引发 TypeError 异常
        assert_raises(TypeError, random.dirichlet, p, float(1))
    # 定义一个测试函数，用于测试 dirichlet 函数在 alpha 参数异常情况下的行为
    def test_dirichlet_bad_alpha(self):
        # 设定特定的 alpha 数组，包含一个正数和一个负数
        alpha = np.array([5.4e-01, -1.0e-16])
        # 断言调用 random.dirichlet(alpha) 会引发 ValueError 异常
        assert_raises(ValueError, random.dirichlet, alpha)

    # 定义一个测试函数，测试 dirichlet 函数在非连续 alpha 数组下的结果
    def test_dirichlet_alpha_non_contiguous(self):
        # 创建一个包含三个元素的数组 a
        a = np.array([51.72840233779265162, -1.0, 39.74494232180943953])
        # 从数组 a 中选取非连续的元素作为 alpha 数组
        alpha = a[::2]
        # 设定随机数种子
        random.seed(self.seed)
        # 调用 random.dirichlet(alpha, size=(3, 2)) 生成非连续 alpha 数组的 Dirichlet 分布
        non_contig = random.dirichlet(alpha, size=(3, 2))
        # 重新设定相同的随机数种子
        random.seed(self.seed)
        # 将 alpha 数组转换为连续存储方式，并调用 random.dirichlet 生成连续 alpha 数组的 Dirichlet 分布
        contig = random.dirichlet(np.ascontiguousarray(alpha),
                                  size=(3, 2))
        # 断言非连续和连续数组生成的结果几乎相等
        assert_array_almost_equal(non_contig, contig)

    # 定义一个测试函数，测试 exponential 函数的随机数生成
    def test_exponential(self):
        # 设定随机数种子
        random.seed(self.seed)
        # 调用 random.exponential(1.1234, size=(3, 2)) 生成指定参数下的指数分布随机数
        actual = random.exponential(1.1234, size=(3, 2))
        # 预期的结果数组
        desired = np.array([[1.08342649775011624, 1.00607889924557314],
                            [2.46628830085216721, 2.49668106809923884],
                            [0.68717433461363442, 1.69175666993575979]])
        # 断言生成的随机数数组与预期结果几乎相等
        assert_array_almost_equal(actual, desired, decimal=15)

    # 定义一个测试函数，测试 exponential 函数在特定边界条件下的行为
    def test_exponential_0(self):
        # 断言当 scale 参数为 0 时，random.exponential(scale=0) 返回值为 0
        assert_equal(random.exponential(scale=0), 0)
        # 断言调用 random.exponential(scale=-0.) 会引发 ValueError 异常
        assert_raises(ValueError, random.exponential, scale=-0.)

    # 定义一个测试函数，测试 f 函数的随机数生成
    def test_f(self):
        # 设定随机数种子
        random.seed(self.seed)
        # 调用 random.f(12, 77, size=(3, 2)) 生成指定参数下的 F 分布随机数
        actual = random.f(12, 77, size=(3, 2))
        # 预期的结果数组
        desired = np.array([[1.21975394418575878, 1.75135759791559775],
                            [1.44803115017146489, 1.22108959480396262],
                            [1.02176975757740629, 1.34431827623300415]])
        # 断言生成的随机数数组与预期结果几乎相等
        assert_array_almost_equal(actual, desired, decimal=15)

    # 定义一个测试函数，测试 gamma 函数的随机数生成
    def test_gamma(self):
        # 设定随机数种子
        random.seed(self.seed)
        # 调用 random.gamma(5, 3, size=(3, 2)) 生成指定参数下的 Gamma 分布随机数
        actual = random.gamma(5, 3, size=(3, 2))
        # 预期的结果数组
        desired = np.array([[24.60509188649287182, 28.54993563207210627],
                            [26.13476110204064184, 12.56988482927716078],
                            [31.71863275789960568, 33.30143302795922011]])
        # 断言生成的随机数数组与预期结果几乎相等
        assert_array_almost_equal(actual, desired, decimal=14)

    # 定义一个测试函数，测试 gamma 函数在特定边界条件下的行为
    def test_gamma_0(self):
        # 断言当 shape 和 scale 参数均为 0 时，random.gamma(shape=0, scale=0) 返回值为 0
        assert_equal(random.gamma(shape=0, scale=0), 0)
        # 断言调用 random.gamma(shape=-0., scale=-0.) 会引发 ValueError 异常
        assert_raises(ValueError, random.gamma, shape=-0., scale=-0.)

    # 定义一个测试函数，测试 geometric 函数的随机数生成
    def test_geometric(self):
        # 设定随机数种子
        random.seed(self.seed)
        # 调用 random.geometric(.123456789, size=(3, 2)) 生成指定参数下的几何分布随机数
        actual = random.geometric(.123456789, size=(3, 2))
        # 预期的结果数组
        desired = np.array([[8, 7],
                            [17, 17],
                            [5, 12]])
        # 断言生成的随机数数组与预期结果完全相等
        assert_array_equal(actual, desired)

    # 定义一个测试函数，测试 geometric 函数在异常参数情况下的行为
    def test_geometric_exceptions(self):
        # 断言调用 random.geometric(1.1) 会引发 ValueError 异常
        assert_raises(ValueError, random.geometric, 1.1)
        # 断言调用 random.geometric([1.1] * 10) 会引发 ValueError 异常
        assert_raises(ValueError, random.geometric, [1.1] * 10)
        # 断言调用 random.geometric(-0.1) 会引发 ValueError 异常
        assert_raises(ValueError, random.geometric, -0.1)
        # 断言调用 random.geometric([-0.1] * 10) 会引发 ValueError 异常
        assert_raises(ValueError, random.geometric, [-0.1] * 10)
        # 在警告抑制的上下文中，断言调用 random.geometric(np.nan) 会引发 ValueError 异常
        with suppress_warnings() as sup:
            sup.record(RuntimeWarning)
            assert_raises(ValueError, random.geometric, np.nan)
            # 断言调用 random.geometric([np.nan] * 10) 会引发 ValueError 异常
            assert_raises(ValueError, random.geometric, [np.nan] * 10)
    # 测试 Gumbel 分布生成随机数的函数
    def test_gumbel(self):
        # 使用指定的种子初始化随机数生成器
        random.seed(self.seed)
        # 生成符合 Gumbel 分布的随机数数组
        actual = random.gumbel(loc=.123456789, scale=2.0, size=(3, 2))
        # 预期的结果数组
        desired = np.array([[0.19591898743416816, 0.34405539668096674],
                            [-1.4492522252274278, -1.47374816298446865],
                            [1.10651090478803416, -0.69535848626236174]])
        # 断言生成的随机数数组与预期结果数组的近似性
        assert_array_almost_equal(actual, desired, decimal=15)

    # 测试 Gumbel 分布在 scale=0 时的特殊情况
    def test_gumbel_0(self):
        # 断言当 scale=0 时，生成的随机数应该为0
        assert_equal(random.gumbel(scale=0), 0)
        # 断言当尝试使用负数作为 scale 时，会触发 ValueError 异常
        assert_raises(ValueError, random.gumbel, scale=-0.)

    # 测试超几何分布生成随机数的函数
    def test_hypergeometric(self):
        # 使用指定的种子初始化随机数生成器
        random.seed(self.seed)
        # 生成符合超几何分布的随机数数组
        actual = random.hypergeometric(10.1, 5.5, 14, size=(3, 2))
        # 预期的结果数组
        desired = np.array([[10, 10],
                            [10, 10],
                            [9, 9]])
        # 断言生成的随机数数组与预期结果数组完全相等
        assert_array_equal(actual, desired)

        # 测试 nbad = 0 的情况
        actual = random.hypergeometric(5, 0, 3, size=4)
        desired = np.array([3, 3, 3, 3])
        assert_array_equal(actual, desired)

        actual = random.hypergeometric(15, 0, 12, size=4)
        desired = np.array([12, 12, 12, 12])
        assert_array_equal(actual, desired)

        # 测试 ngood = 0 的情况
        actual = random.hypergeometric(0, 5, 3, size=4)
        desired = np.array([0, 0, 0, 0])
        assert_array_equal(actual, desired)

        actual = random.hypergeometric(0, 15, 12, size=4)
        desired = np.array([0, 0, 0, 0])
        assert_array_equal(actual, desired)

    # 测试 Laplace 分布生成随机数的函数
    def test_laplace(self):
        # 使用指定的种子初始化随机数生成器
        random.seed(self.seed)
        # 生成符合 Laplace 分布的随机数数组
        actual = random.laplace(loc=.123456789, scale=2.0, size=(3, 2))
        # 预期的结果数组
        desired = np.array([[0.66599721112760157, 0.52829452552221945],
                            [3.12791959514407125, 3.18202813572992005],
                            [-0.05391065675859356, 1.74901336242837324]])
        # 断言生成的随机数数组与预期结果数组的近似性
        assert_array_almost_equal(actual, desired, decimal=15)

    # 测试 Laplace 分布在 scale=0 时的特殊情况
    def test_laplace_0(self):
        # 断言当 scale=0 时，生成的随机数应该为0
        assert_equal(random.laplace(scale=0), 0)
        # 断言当尝试使用负数作为 scale 时，会触发 ValueError 异常
        assert_raises(ValueError, random.laplace, scale=-0.)

    # 测试 logistic 分布生成随机数的函数
    def test_logistic(self):
        # 使用指定的种子初始化随机数生成器
        random.seed(self.seed)
        # 生成符合 logistic 分布的随机数数组
        actual = random.logistic(loc=.123456789, scale=2.0, size=(3, 2))
        # 预期的结果数组
        desired = np.array([[1.09232835305011444, 0.8648196662399954],
                            [4.27818590694950185, 4.33897006346929714],
                            [-0.21682183359214885, 2.63373365386060332]])
        # 断言生成的随机数数组与预期结果数组的近似性
        assert_array_almost_equal(actual, desired, decimal=15)

    # 测试 lognormal 分布生成随机数的函数
    def test_lognormal(self):
        # 使用指定的种子初始化随机数生成器
        random.seed(self.seed)
        # 生成符合 lognormal 分布的随机数数组
        actual = random.lognormal(mean=.123456789, sigma=2.0, size=(3, 2))
        # 预期的结果数组
        desired = np.array([[16.50698631688883822, 36.54846706092654784],
                            [22.67886599981281748, 0.71617561058995771],
                            [65.72798501792723869, 86.84341601437161273]])
        # 断言生成的随机数数组与预期结果数组的近似性
        assert_array_almost_equal(actual, desired, decimal=13)
    # 定义测试函数 test_lognormal_0，用于测试 random.lognormal 函数
    def test_lognormal_0(self):
        # 断言 random.lognormal(sigma=0) 的返回值为 1
        assert_equal(random.lognormal(sigma=0), 1)
        # 断言调用 random.lognormal(sigma=-0.) 会抛出 ValueError 异常
        assert_raises(ValueError, random.lognormal, sigma=-0.)

    # 定义测试函数 test_logseries，用于测试 random.logseries 函数
    def test_logseries(self):
        # 使用指定的种子值初始化随机数生成器
        random.seed(self.seed)
        # 调用 random.logseries(p=.923456789, size=(3, 2))，获取实际结果
        actual = random.logseries(p=.923456789, size=(3, 2))
        # 期望的结果数据
        desired = np.array([[2, 2],
                            [6, 17],
                            [3, 6]])
        # 断言实际结果与期望结果数组相等
        assert_array_equal(actual, desired)

    # 定义测试函数 test_logseries_zero，用于测试 random.logseries 函数中参数为 0 的情况
    def test_logseries_zero(self):
        # 断言调用 random.logseries(0) 的返回值为 1
        assert random.logseries(0) == 1

    # 使用 pytest.mark.parametrize 装饰器定义参数化测试函数 test_logseries_exceptions
    @pytest.mark.parametrize("value", [np.nextafter(0., -1), 1., np.nan, 5.])
    def test_logseries_exceptions(self, value):
        # 在忽略无效错误的上下文中执行以下代码块
        with np.errstate(invalid="ignore"):
            # 使用 pytest.raises 检查调用 random.logseries(value) 是否会引发 ValueError 异常
            with pytest.raises(ValueError):
                random.logseries(value)
            # 使用 pytest.raises 检查调用 random.logseries(np.array([value] * 10)) 是否会引发 ValueError 异常
            with pytest.raises(ValueError):
                # 连续路径：
                random.logseries(np.array([value] * 10))
            # 使用 pytest.raises 检查调用 random.logseries(np.array([value] * 10)[::2]) 是否会引发 ValueError 异常
            with pytest.raises(ValueError):
                # 非连续路径：
                random.logseries(np.array([value] * 10)[::2])

    # 定义测试函数 test_multinomial，用于测试 random.multinomial 函数
    def test_multinomial(self):
        # 使用指定的种子值初始化随机数生成器
        random.seed(self.seed)
        # 调用 random.multinomial(20, [1 / 6.] * 6, size=(3, 2))，获取实际结果
        actual = random.multinomial(20, [1 / 6.] * 6, size=(3, 2))
        # 期望的结果数据
        desired = np.array([[[4, 3, 5, 4, 2, 2],
                             [5, 2, 8, 2, 2, 1]],
                            [[3, 4, 3, 6, 0, 4],
                             [2, 1, 4, 3, 6, 4]],
                            [[4, 4, 2, 5, 2, 3],
                             [4, 3, 4, 2, 3, 4]]])
        # 断言实际结果与期望结果数组相等
        assert_array_equal(actual, desired)
    # 测试多变量正态分布生成函数
    def test_multivariate_normal(self):
        # 设定随机数种子，确保可复现性
        random.seed(self.seed)
        # 设定正态分布的均值和协方差矩阵
        mean = (.123456789, 10)
        cov = [[1, 0], [0, 1]]
        # 设定生成随机样本的尺寸
        size = (3, 2)
        # 调用随机生成多变量正态分布样本的函数
        actual = random.multivariate_normal(mean, cov, size)
        # 设定期望的结果样本
        desired = np.array([[[1.463620246718631, 11.73759122771936],
                             [1.622445133300628, 9.771356667546383]],
                            [[2.154490787682787, 12.170324946056553],
                             [1.719909438201865, 9.230548443648306]],
                            [[0.689515026297799, 9.880729819607714],
                             [-0.023054015651998, 9.201096623542879]]])
        # 断言实际生成的样本与期望的结果样本相等，精度为15位小数
        assert_array_almost_equal(actual, desired, decimal=15)

        # 检查默认尺寸的情况，避免引发弃用警告
        actual = random.multivariate_normal(mean, cov)
        desired = np.array([0.895289569463708, 9.17180864067987])
        assert_array_almost_equal(actual, desired, decimal=15)

        # 检查非半正定协方差矩阵是否引发 RuntimeWarning 警告
        mean = [0, 0]
        cov = [[1, 2], [2, 1]]
        assert_warns(RuntimeWarning, random.multivariate_normal, mean, cov)

        # 检查设置 check_valid='ignore' 时是否不会引发 RuntimeWarning 警告
        assert_no_warnings(random.multivariate_normal, mean, cov,
                           check_valid='ignore')

        # 检查设置 check_valid='raises' 时是否引发 ValueError 异常
        assert_raises(ValueError, random.multivariate_normal, mean, cov,
                      check_valid='raise')

        # 检查设置 numpy 数组类型半正定矩阵的情况，应不会引发警告
        cov = np.array([[1, 0.1], [0.1, 1]], dtype=np.float32)
        with suppress_warnings() as sup:
            random.multivariate_normal(mean, cov)
            w = sup.record(RuntimeWarning)
            # 断言未记录任何警告信息
            assert len(w) == 0

        # 检查各种错误情况下是否能够正确引发 ValueError 异常
        mu = np.zeros(2)
        cov = np.eye(2)
        assert_raises(ValueError, random.multivariate_normal, mean, cov,
                      check_valid='other')
        assert_raises(ValueError, random.multivariate_normal,
                      np.zeros((2, 1, 1)), cov)
        assert_raises(ValueError, random.multivariate_normal,
                      mu, np.empty((3, 2)))
        assert_raises(ValueError, random.multivariate_normal,
                      mu, np.eye(3))
    # 测试非中心卡方分布生成函数
    def test_noncentral_chisquare(self):
        # 使用指定种子初始化随机数生成器
        random.seed(self.seed)
        # 生成非中心卡方分布样本，自由度为5，非中心参数为5，生成3行2列的样本
        actual = random.noncentral_chisquare(df=5, nonc=5, size=(3, 2))
        # 期望的结果数组
        desired = np.array([[23.91905354498517511, 13.35324692733826346],
                            [31.22452661329736401, 16.60047399466177254],
                            [5.03461598262724586, 17.94973089023519464]])
        # 断言实际生成的样本与期望的结果数组近似相等，精确到小数点后14位
        assert_array_almost_equal(actual, desired, decimal=14)

        # 生成非中心卡方分布样本，自由度为0.5，非中心参数为0.2，生成3行2列的样本
        actual = random.noncentral_chisquare(df=.5, nonc=.2, size=(3, 2))
        # 期望的结果数组
        desired = np.array([[1.47145377828516666,  0.15052899268012659],
                            [0.00943803056963588,  1.02647251615666169],
                            [0.332334982684171,  0.15451287602753125]])
        # 断言实际生成的样本与期望的结果数组近似相等，精确到小数点后14位
        assert_array_almost_equal(actual, desired, decimal=14)

        # 使用指定种子重新初始化随机数生成器
        random.seed(self.seed)
        # 生成非中心卡方分布样本，自由度为5，非中心参数为0，生成3行2列的样本
        actual = random.noncentral_chisquare(df=5, nonc=0, size=(3, 2))
        # 期望的结果数组
        desired = np.array([[9.597154162763948, 11.725484450296079],
                            [10.413711048138335, 3.694475922923986],
                            [13.484222138963087, 14.377255424602957]])
        # 断言实际生成的样本与期望的结果数组近似相等，精确到小数点后14位
        assert_array_almost_equal(actual, desired, decimal=14)

    # 测试非中心 F 分布生成函数
    def test_noncentral_f(self):
        # 使用指定种子初始化随机数生成器
        random.seed(self.seed)
        # 生成非中心 F 分布样本，分子自由度为5，分母自由度为2，非中心参数为1，生成3行2列的样本
        actual = random.noncentral_f(dfnum=5, dfden=2, nonc=1, size=(3, 2))
        # 期望的结果数组
        desired = np.array([[1.40598099674926669, 0.34207973179285761],
                            [3.57715069265772545, 7.92632662577829805],
                            [0.43741599463544162, 1.1774208752428319]])
        # 断言实际生成的样本与期望的结果数组近似相等，精确到小数点后14位
        assert_array_almost_equal(actual, desired, decimal=14)

    # 测试带 NaN 非中心 F 分布生成函数
    def test_noncentral_f_nan(self):
        # 使用指定种子初始化随机数生成器
        random.seed(self.seed)
        # 生成带 NaN 的非中心 F 分布样本，分子自由度为5，分母自由度为2，非中心参数为 NaN
        actual = random.noncentral_f(dfnum=5, dfden=2, nonc=np.nan)
        # 断言实际生成的结果是 NaN
        assert np.isnan(actual)

    # 测试正态分布生成函数
    def test_normal(self):
        # 使用指定种子初始化随机数生成器
        random.seed(self.seed)
        # 生成正态分布样本，均值为0.123456789，标准差为2.0，生成3行2列的样本
        actual = random.normal(loc=.123456789, scale=2.0, size=(3, 2))
        # 期望的结果数组
        desired = np.array([[2.80378370443726244, 3.59863924443872163],
                            [3.121433477601256, -0.33382987590723379],
                            [4.18552478636557357, 4.46410668111310471]])
        # 断言实际生成的样本与期望的结果数组近似相等，精确到小数点后15位
        assert_array_almost_equal(actual, desired, decimal=15)

    # 测试标准差为0的正态分布生成函数
    def test_normal_0(self):
        # 断言生成的正态分布样本的标准差为0时，结果应为0
        assert_equal(random.normal(scale=0), 0)
        # 断言当标准差为负数时，会引发 ValueError 异常
        assert_raises(ValueError, random.normal, scale=-0.)
    # 测试 Pareto 分布的随机数生成器
    def test_pareto(self):
        # 使用指定种子初始化随机数生成器
        random.seed(self.seed)
        # 生成 Pareto 分布的随机数数组
        actual = random.pareto(a=.123456789, size=(3, 2))
        # 期望得到的 Pareto 分布的随机数数组
        desired = np.array(
                [[2.46852460439034849e+03, 1.41286880810518346e+03],
                 [5.28287797029485181e+07, 6.57720981047328785e+07],
                 [1.40840323350391515e+02, 1.98390255135251704e+05]])
        # 由于在某些环境下（32位 x86 Ubuntu 12.10），数组中的某些值可能略有不同，
        # 这里设置一个比较宽松的容差范围（以 nulp 为单位）
        np.testing.assert_array_almost_equal_nulp(actual, desired, nulp=30)

    # 测试 Poisson 分布的随机数生成器
    def test_poisson(self):
        # 使用指定种子初始化随机数生成器
        random.seed(self.seed)
        # 生成 Poisson 分布的随机数数组
        actual = random.poisson(lam=.123456789, size=(3, 2))
        # 期望得到的 Poisson 分布的随机数数组
        desired = np.array([[0, 0],
                            [1, 0],
                            [0, 0]])
        # 检查生成的随机数数组是否与期望数组相等
        assert_array_equal(actual, desired)

    # 测试 Poisson 分布的异常情况
    def test_poisson_exceptions(self):
        # 设置大到足以触发异常的 lambda 值
        lambig = np.iinfo('l').max
        lamneg = -1
        # 检查是否正确抛出 ValueError 异常
        assert_raises(ValueError, random.poisson, lamneg)
        assert_raises(ValueError, random.poisson, [lamneg] * 10)
        assert_raises(ValueError, random.poisson, lambig)
        assert_raises(ValueError, random.poisson, [lambig] * 10)
        # 检查是否正确抑制包含 NaN 的警告并抛出 ValueError 异常
        with suppress_warnings() as sup:
            sup.record(RuntimeWarning)
            assert_raises(ValueError, random.poisson, np.nan)
            assert_raises(ValueError, random.poisson, [np.nan] * 10)

    # 测试 Power 分布的随机数生成器
    def test_power(self):
        # 使用指定种子初始化随机数生成器
        random.seed(self.seed)
        # 生成 Power 分布的随机数数组
        actual = random.power(a=.123456789, size=(3, 2))
        # 期望得到的 Power 分布的随机数数组
        desired = np.array([[0.02048932883240791, 0.01424192241128213],
                            [0.38446073748535298, 0.39499689943484395],
                            [0.00177699707563439, 0.13115505880863756]])
        # 检查生成的随机数数组是否在指定精度范围内与期望数组相等
        assert_array_almost_equal(actual, desired, decimal=15)

    # 测试 Rayleigh 分布的随机数生成器
    def test_rayleigh(self):
        # 使用指定种子初始化随机数生成器
        random.seed(self.seed)
        # 生成 Rayleigh 分布的随机数数组
        actual = random.rayleigh(scale=10, size=(3, 2))
        # 期望得到的 Rayleigh 分布的随机数数组
        desired = np.array([[13.8882496494248393, 13.383318339044731],
                            [20.95413364294492098, 21.08285015800712614],
                            [11.06066537006854311, 17.35468505778271009]])
        # 检查生成的随机数数组是否在指定精度范围内与期望数组相等
        assert_array_almost_equal(actual, desired, decimal=14)

    # 测试 Rayleigh 分布中 scale=0 的情况
    def test_rayleigh_0(self):
        # 检查 scale=0 时是否生成正确的值
        assert_equal(random.rayleigh(scale=0), 0)
        # 检查对 scale=-0. 抛出 ValueError 异常
        assert_raises(ValueError, random.rayleigh, scale=-0.)

    # 测试标准 Cauchy 分布的随机数生成器
    def test_standard_cauchy(self):
        # 使用指定种子初始化随机数生成器
        random.seed(self.seed)
        # 生成标准 Cauchy 分布的随机数数组
        actual = random.standard_cauchy(size=(3, 2))
        # 期望得到的标准 Cauchy 分布的随机数数组
        desired = np.array([[0.77127660196445336, -6.55601161955910605],
                            [0.93582023391158309, -2.07479293013759447],
                            [-4.74601644297011926, 0.18338989290760804]])
        # 检查生成的随机数数组是否在指定精度范围内与期望数组相等
        assert_array_almost_equal(actual, desired, decimal=15)
    # 测试标准指数分布生成函数
    def test_standard_exponential(self):
        # 使用预设的随机种子初始化随机数生成器
        random.seed(self.seed)
        # 生成指定大小的标准指数分布随机数数组
        actual = random.standard_exponential(size=(3, 2))
        # 预期的标准指数分布随机数数组
        desired = np.array([[0.96441739162374596, 0.89556604882105506],
                            [2.1953785836319808, 2.22243285392490542],
                            [0.6116915921431676, 1.50592546727413201]])
        # 使用高精度比较函数检查生成的随机数数组与预期数组的接近程度
        assert_array_almost_equal(actual, desired, decimal=15)

    # 测试标准伽马分布生成函数
    def test_standard_gamma(self):
        # 使用预设的随机种子初始化随机数生成器
        random.seed(self.seed)
        # 生成指定形状和大小的标准伽马分布随机数数组
        actual = random.standard_gamma(shape=3, size=(3, 2))
        # 预期的标准伽马分布随机数数组
        desired = np.array([[5.50841531318455058, 6.62953470301903103],
                            [5.93988484943779227, 2.31044849402133989],
                            [7.54838614231317084, 8.012756093271868]])
        # 使用高精度比较函数检查生成的随机数数组与预期数组的接近程度
        assert_array_almost_equal(actual, desired, decimal=14)

    # 测试形状参数为0的标准伽马分布生成函数
    def test_standard_gamma_0(self):
        # 检查形状参数为0时函数返回值是否为0
        assert_equal(random.standard_gamma(shape=0), 0)
        # 检查当形状参数为负数时是否会引发 ValueError 异常
        assert_raises(ValueError, random.standard_gamma, shape=-0.)

    # 测试标准正态分布生成函数
    def test_standard_normal(self):
        # 使用预设的随机种子初始化随机数生成器
        random.seed(self.seed)
        # 生成指定大小的标准正态分布随机数数组
        actual = random.standard_normal(size=(3, 2))
        # 预期的标准正态分布随机数数组
        desired = np.array([[1.34016345771863121, 1.73759122771936081],
                            [1.498988344300628, -0.2286433324536169],
                            [2.031033998682787, 2.17032494605655257]])
        # 使用高精度比较函数检查生成的随机数数组与预期数组的接近程度
        assert_array_almost_equal(actual, desired, decimal=15)

    # 测试生成单个标准正态分布随机数的函数
    def test_randn_singleton(self):
        # 使用预设的随机种子初始化随机数生成器
        random.seed(self.seed)
        # 生成一个单个标准正态分布随机数
        actual = random.randn()
        # 预期的单个标准正态分布随机数
        desired = np.array(1.34016345771863121)
        # 使用高精度比较函数检查生成的随机数与预期数值的接近程度
        assert_array_almost_equal(actual, desired, decimal=15)

    # 测试标准 t 分布生成函数
    def test_standard_t(self):
        # 使用预设的随机种子初始化随机数生成器
        random.seed(self.seed)
        # 生成指定自由度和大小的标准 t 分布随机数数组
        actual = random.standard_t(df=10, size=(3, 2))
        # 预期的标准 t 分布随机数数组
        desired = np.array([[0.97140611862659965, -0.08830486548450577],
                            [1.36311143689505321, -0.55317463909867071],
                            [-0.18473749069684214, 0.61181537341755321]])
        # 使用高精度比较函数检查生成的随机数数组与预期数组的接近程度
        assert_array_almost_equal(actual, desired, decimal=15)

    # 测试三角分布生成函数
    def test_triangular(self):
        # 使用预设的随机种子初始化随机数生成器
        random.seed(self.seed)
        # 生成指定参数的三角分布随机数数组
        actual = random.triangular(left=5.12, mode=10.23, right=20.34,
                                   size=(3, 2))
        # 预期的三角分布随机数数组
        desired = np.array([[12.68117178949215784, 12.4129206149193152],
                            [16.20131377335158263, 16.25692138747600524],
                            [11.20400690911820263, 14.4978144835829923]])
        # 使用高精度比较函数检查生成的随机数数组与预期数组的接近程度
        assert_array_almost_equal(actual, desired, decimal=14)

    # 测试均匀分布生成函数
    def test_uniform(self):
        # 使用预设的随机种子初始化随机数生成器
        random.seed(self.seed)
        # 生成指定上下界和大小的均匀分布随机数数组
        actual = random.uniform(low=1.23, high=10.54, size=(3, 2))
        # 预期的均匀分布随机数数组
        desired = np.array([[6.99097932346268003, 6.73801597444323974],
                            [9.50364421400426274, 9.53130618907631089],
                            [5.48995325769805476, 8.47493103280052118]])
        # 使用高精度比较函数检查生成的随机数数组与预期数组的接近程度
        assert_array_almost_equal(actual, desired, decimal=15)
    def test_uniform_range_bounds(self):
        # 获取浮点数的最小值和最大值
        fmin = np.finfo('float').min
        fmax = np.finfo('float').max

        # 使用 random.uniform 函数进行测试，期望抛出 OverflowError 异常
        assert_raises(OverflowError, random.uniform, -np.inf, 0)
        assert_raises(OverflowError, random.uniform, 0, np.inf)
        assert_raises(OverflowError, random.uniform, fmin, fmax)
        assert_raises(OverflowError, random.uniform, [-np.inf], [0])
        assert_raises(OverflowError, random.uniform, [0], [np.inf])

        # (fmax / 1e17) - fmin 在范围内，因此不应该抛出异常
        # 增加 fmin 一点以考虑 i386 扩展精度 DBL_MAX / 1e17 + DBL_MAX > DBL_MAX
        random.uniform(low=np.nextafter(fmin, 1), high=fmax / 1e17)

    def test_scalar_exception_propagation(self):
        # 测试确保分布中的异常正确传播
        # 当使用会抛出异常的对象作为标量参数时。
        #
        # 修复 gh: 8865 的回归测试

        class ThrowingFloat(np.ndarray):
            def __float__(self):
                raise TypeError

        # 创建抛出异常的浮点数对象
        throwing_float = np.array(1.0).view(ThrowingFloat)
        # 确保 random.uniform 函数对抛出异常的浮点数抛出 TypeError
        assert_raises(TypeError, random.uniform, throwing_float,
                      throwing_float)

        class ThrowingInteger(np.ndarray):
            def __int__(self):
                raise TypeError

        # 创建抛出异常的整数对象
        throwing_int = np.array(1).view(ThrowingInteger)
        # 确保 random.hypergeometric 函数对抛出异常的整数抛出 TypeError
        assert_raises(TypeError, random.hypergeometric, throwing_int, 1, 1)

    def test_vonmises(self):
        # 使用指定的种子生成随机数
        random.seed(self.seed)
        # 使用 vonmises 分布生成随机数，指定 mu 和 kappa，以及生成的大小
        actual = random.vonmises(mu=1.23, kappa=1.54, size=(3, 2))
        # 期望的生成结果
        desired = np.array([[2.28567572673902042, 2.89163838442285037],
                            [0.38198375564286025, 2.57638023113890746],
                            [1.19153771588353052, 1.83509849681825354]])
        # 确保生成的随机数数组与期望数组几乎相等，精度为 15 位小数
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_vonmises_small(self):
        # 检查是否出现无限循环，gh-4720
        random.seed(self.seed)
        # 使用 vonmises 分布生成随机数，指定 mu 和非常小的 kappa 值，以及生成的大小
        r = random.vonmises(mu=0., kappa=1.1e-8, size=10**6)
        # 确保生成的随机数数组中所有值都是有限的
        assert_(np.isfinite(r).all())

    def test_vonmises_large(self):
        # 防止在修复 Generator 时对 RandomState 进行更改
        random.seed(self.seed)
        # 使用 vonmises 分布生成随机数，指定 mu 和较大的 kappa 值，以及生成的大小
        actual = random.vonmises(mu=0., kappa=1e7, size=3)
        # 期望的生成结果
        desired = np.array([4.634253748521111e-04,
                            3.558873596114509e-04,
                            -2.337119622577433e-04])
        # 确保生成的随机数数组与期望数组几乎相等，精度为 8 位小数
        assert_array_almost_equal(actual, desired, decimal=8)

    def test_vonmises_nan(self):
        # 使用指定的种子生成随机数
        random.seed(self.seed)
        # 使用 vonmises 分布生成随机数，指定 mu 和 NaN 的 kappa
        r = random.vonmises(mu=0., kappa=np.nan)
        # 确保生成的随机数是 NaN
        assert_(np.isnan(r))
    # 定义一个名为 test_wald 的测试方法，用于测试 Wald 分布的随机数生成器
    def test_wald(self):
        # 设置随机数种子，以确保结果可复现
        random.seed(self.seed)
        # 生成符合指定参数的随机数数组，形状为 (3, 2)
        actual = random.wald(mean=1.23, scale=1.54, size=(3, 2))
        # 预期的结果数组，用于与生成的随机数数组比较
        desired = np.array([[3.82935265715889983, 5.13125249184285526],
                            [0.35045403618358717, 1.50832396872003538],
                            [0.24124319895843183, 0.22031101461955038]])
        # 断言生成的随机数数组与预期数组几乎相等，精确到小数点后 14 位
        assert_array_almost_equal(actual, desired, decimal=14)

    # 定义一个名为 test_weibull 的测试方法，用于测试 Weibull 分布的随机数生成器
    def test_weibull(self):
        # 设置随机数种子，以确保结果可复现
        random.seed(self.seed)
        # 生成符合指定参数的随机数数组，形状为 (3, 2)
        actual = random.weibull(a=1.23, size=(3, 2))
        # 预期的结果数组，用于与生成的随机数数组比较
        desired = np.array([[0.97097342648766727, 0.91422896443565516],
                            [1.89517770034962929, 1.91414357960479564],
                            [0.67057783752390987, 1.39494046635066793]])
        # 断言生成的随机数数组与预期数组几乎相等，精确到小数点后 15 位
        assert_array_almost_equal(actual, desired, decimal=15)

    # 定义一个名为 test_weibull_0 的测试方法，用于测试 Weibull 分布中参数 a=0 的情况
    def test_weibull_0(self):
        # 设置随机数种子，以确保结果可复现
        random.seed(self.seed)
        # 断言当参数 a=0 时，生成的随机数数组应该全为零
        assert_equal(random.weibull(a=0, size=12), np.zeros(12))
        # 断言当参数 a=-0. 时，应该抛出 ValueError 异常
        assert_raises(ValueError, random.weibull, a=-0.)

    # 定义一个名为 test_zipf 的测试方法，用于测试 Zipf 分布的随机数生成器
    def test_zipf(self):
        # 设置随机数种子，以确保结果可复现
        random.seed(self.seed)
        # 生成符合指定参数的随机数数组，形状为 (3, 2)
        actual = random.zipf(a=1.23, size=(3, 2))
        # 预期的结果数组，用于与生成的随机数数组比较
        desired = np.array([[66, 29],
                            [1, 1],
                            [3, 13]])
        # 断言生成的随机数数组与预期数组完全相等
        assert_array_equal(actual, desired)
class TestBroadcast:
    # tests that functions that broadcast behave
    # correctly when presented with non-scalar arguments

    def setup_method(self):
        # 设置随机种子
        self.seed = 123456789

    def set_seed(self):
        # 使用设定的种子初始化随机数生成器
        random.seed(self.seed)

    def test_uniform(self):
        # 测试均匀分布函数在非标量参数情况下的行为
        low = [0]
        high = [1]
        uniform = random.uniform
        desired = np.array([0.53283302478975902,
                            0.53413660089041659,
                            0.50955303552646702])

        self.set_seed()
        # 测试第一种情况：low 数组复制3次，high 为标量
        actual = uniform(low * 3, high)
        assert_array_almost_equal(actual, desired, decimal=14)

        self.set_seed()
        # 测试第二种情况：low 为标量，high 数组复制3次
        actual = uniform(low, high * 3)
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_normal(self):
        # 测试正态分布函数在非标量参数情况下的行为
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        normal = random.normal
        desired = np.array([2.2129019979039612,
                            2.1283977976520019,
                            1.8417114045748335])

        self.set_seed()
        # 测试第一种情况：loc 数组复制3次，scale 为标量
        actual = normal(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        # 测试异常情况：scale 为负数
        assert_raises(ValueError, normal, loc * 3, bad_scale)

        self.set_seed()
        # 测试第二种情况：loc 为标量，scale 数组复制3次
        actual = normal(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        # 测试异常情况：scale 为负数
        assert_raises(ValueError, normal, loc, bad_scale * 3)

    def test_beta(self):
        # 测试 Beta 分布函数在非标量参数情况下的行为
        a = [1]
        b = [2]
        bad_a = [-1]
        bad_b = [-2]
        beta = random.beta
        desired = np.array([0.19843558305989056,
                            0.075230336409423643,
                            0.24976865978980844])

        self.set_seed()
        # 测试第一种情况：a 数组复制3次，b 为标量
        actual = beta(a * 3, b)
        assert_array_almost_equal(actual, desired, decimal=14)
        # 测试异常情况：bad_a 数组复制3次，b 为标量
        assert_raises(ValueError, beta, bad_a * 3, b)
        # 测试异常情况：a 数组复制3次，bad_b 为标量
        assert_raises(ValueError, beta, a * 3, bad_b)

        self.set_seed()
        # 测试第二种情况：a 为标量，b 数组复制3次
        actual = beta(a, b * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        # 测试异常情况：a 为标量，bad_b 数组复制3次
        assert_raises(ValueError, beta, bad_a, b * 3)
        # 测试异常情况：a 为标量，b 数组复制3次
        assert_raises(ValueError, beta, a, bad_b * 3)

    def test_exponential(self):
        # 测试指数分布函数在非标量参数情况下的行为
        scale = [1]
        bad_scale = [-1]
        exponential = random.exponential
        desired = np.array([0.76106853658845242,
                            0.76386282278691653,
                            0.71243813125891797])

        self.set_seed()
        # 测试 scale 数组复制3次的情况
        actual = exponential(scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        # 测试异常情况：bad_scale 数组复制3次
        assert_raises(ValueError, exponential, bad_scale * 3)
    # 定义测试标准 Gamma 分布的方法
    def test_standard_gamma(self):
        shape = [1]  # 设置 Gamma 分布的形状参数为 [1]
        bad_shape = [-1]  # 定义一个错误的形状参数为 [-1]
        std_gamma = random.standard_gamma  # 将 random 模块中的 standard_gamma 函数赋给 std_gamma
        desired = np.array([0.76106853658845242,
                            0.76386282278691653,
                            0.71243813125891797])  # 预期的 Gamma 分布随机数数组

        self.set_seed()  # 调用当前类的 set_seed 方法，设置随机数种子
        actual = std_gamma(shape * 3)  # 生成一个形状为 [1, 1, 1] 的标准 Gamma 分布随机数数组
        assert_array_almost_equal(actual, desired, decimal=14)  # 断言生成的随机数数组与预期数组接近

        assert_raises(ValueError, std_gamma, bad_shape * 3)  # 断言当使用错误的形状参数 [-1, -1, -1] 时，会引发 ValueError 异常

    # 定义测试 Gamma 分布的方法
    def test_gamma(self):
        shape = [1]  # 设置 Gamma 分布的形状参数为 [1]
        scale = [2]  # 设置 Gamma 分布的尺度参数为 [2]
        bad_shape = [-1]  # 定义一个错误的形状参数为 [-1]
        bad_scale = [-2]  # 定义一个错误的尺度参数为 [-2]
        gamma = random.gamma  # 将 random 模块中的 gamma 函数赋给 gamma
        desired = np.array([1.5221370731769048,
                            1.5277256455738331,
                            1.4248762625178359])  # 预期的 Gamma 分布随机数数组

        self.set_seed()  # 调用当前类的 set_seed 方法，设置随机数种子

        # 测试 shape 参数为 [1, 1, 1]，scale 参数为 [2] 时的 Gamma 分布随机数数组
        actual = gamma(shape * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)  # 断言生成的随机数数组与预期数组接近
        assert_raises(ValueError, gamma, bad_shape * 3, scale)  # 断言当使用错误的形状参数 [-1, -1, -1] 时，会引发 ValueError 异常
        assert_raises(ValueError, gamma, shape * 3, bad_scale)  # 断言当使用错误的尺度参数 [-2] 时，会引发 ValueError 异常

        # 测试 shape 参数为 [1]，scale 参数为 [2, 2, 2] 时的 Gamma 分布随机数数组
        actual = gamma(shape, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)  # 断言生成的随机数数组与预期数组接近
        assert_raises(ValueError, gamma, bad_shape, scale * 3)  # 断言当使用错误的形状参数 [-1] 时，会引发 ValueError 异常
        assert_raises(ValueError, gamma, shape, bad_scale * 3)  # 断言当使用错误的尺度参数 [-2, -2, -2] 时，会引发 ValueError 异常

    # 定义测试 F 分布的方法
    def test_f(self):
        dfnum = [1]  # 设置 F 分布的分子自由度参数为 [1]
        dfden = [2]  # 设置 F 分布的分母自由度参数为 [2]
        bad_dfnum = [-1]  # 定义一个错误的分子自由度参数为 [-1]
        bad_dfden = [-2]  # 定义一个错误的分母自由度参数为 [-2]
        f = random.f  # 将 random 模块中的 f 函数赋给 f
        desired = np.array([0.80038951638264799,
                            0.86768719635363512,
                            2.7251095168386801])  # 预期的 F 分布随机数数组

        self.set_seed()  # 调用当前类的 set_seed 方法，设置随机数种子

        # 测试 dfnum 参数为 [1, 1, 1]，dfden 参数为 [2] 时的 F 分布随机数数组
        actual = f(dfnum * 3, dfden)
        assert_array_almost_equal(actual, desired, decimal=14)  # 断言生成的随机数数组与预期数组接近
        assert_raises(ValueError, f, bad_dfnum * 3, dfden)  # 断言当使用错误的分子自由度参数 [-1, -1, -1] 时，会引发 ValueError 异常
        assert_raises(ValueError, f, dfnum * 3, bad_dfden)  # 断言当使用错误的分母自由度参数 [-2] 时，会引发 ValueError 异常

        # 测试 dfnum 参数为 [1]，dfden 参数为 [2, 2, 2] 时的 F 分布随机数数组
        actual = f(dfnum, dfden * 3)
        assert_array_almost_equal(actual, desired, decimal=14)  # 断言生成的随机数数组与预期数组接近
        assert_raises(ValueError, f, bad_dfnum, dfden * 3)  # 断言当使用错误的分子自由度参数 [-1] 时，会引发 ValueError 异常
        assert_raises(ValueError, f, dfnum, bad_dfden * 3)  # 断言当使用错误的分母自由度参数 [-2, -2, -2] 时，会引发 ValueError 异常
    # 定义测试函数 `test_noncentral_f`，用于测试非中心 F 分布生成函数的各种情况
    def test_noncentral_f(self):
        # 设置自由度参数，分别为 [2]，表示自由度参数列表
        dfnum = [2]
        # 设置分母自由度参数，为 [3]，表示分母自由度参数列表
        dfden = [3]
        # 设置非中心参数，为 [4]，表示非中心参数列表
        nonc = [4]
        # 设置不良自由度参数，为 [0]，表示不良自由度参数列表
        bad_dfnum = [0]
        # 设置不良分母自由度参数，为 [-1]，表示不良分母自由度参数列表
        bad_dfden = [-1]
        # 设置不良非中心参数，为 [-2]，表示不良非中心参数列表
        bad_nonc = [-2]
        # 将随机生成的非中心 F 分布函数赋值给变量 nonc_f
        nonc_f = random.noncentral_f
        # 设置预期结果为指定的数组，精确到小数点后14位
        desired = np.array([9.1393943263705211,
                            13.025456344595602,
                            8.8018098359100545])

        # 调用函数设置随机种子
        self.set_seed()
        # 调用非中心 F 分布函数计算实际结果
        actual = nonc_f(dfnum * 3, dfden, nonc)
        # 断言实际结果与预期结果的准确度
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言处理包含 NaN 的情况，预期会抛出 ValueError 异常
        assert np.all(np.isnan(nonc_f(dfnum, dfden, [np.nan] * 3)))

        # 断言处理不良自由度参数情况，预期会抛出 ValueError 异常
        assert_raises(ValueError, nonc_f, bad_dfnum * 3, dfden, nonc)
        assert_raises(ValueError, nonc_f, dfnum * 3, bad_dfden, nonc)
        assert_raises(ValueError, nonc_f, dfnum * 3, dfden, bad_nonc)

        # 重设随机种子
        self.set_seed()
        # 重新计算实际结果，验证处理多个自由度参数的情况
        actual = nonc_f(dfnum, dfden * 3, nonc)
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言处理不良自由度参数情况，预期会抛出 ValueError 异常
        assert_raises(ValueError, nonc_f, bad_dfnum, dfden * 3, nonc)
        assert_raises(ValueError, nonc_f, dfnum, bad_dfden * 3, nonc)
        assert_raises(ValueError, nonc_f, dfnum, dfden * 3, bad_nonc)

        # 重设随机种子
        self.set_seed()
        # 重新计算实际结果，验证处理多个非中心参数的情况
        actual = nonc_f(dfnum, dfden, nonc * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言处理不良非中心参数情况，预期会抛出 ValueError 异常
        assert_raises(ValueError, nonc_f, bad_dfnum, dfden, nonc * 3)
        assert_raises(ValueError, nonc_f, dfnum, bad_dfden, nonc * 3)
        assert_raises(ValueError, nonc_f, dfnum, dfden, bad_nonc * 3)

    # 定义测试函数 `test_noncentral_f_small_df`，用于测试小自由度情况下的非中心 F 分布生成函数
    def test_noncentral_f_small_df(self):
        # 设置随机种子
        self.set_seed()
        # 设置预期结果为指定的数组，精确到小数点后14位
        desired = np.array([6.869638627492048, 0.785880199263955])
        # 调用非中心 F 分布函数计算实际结果，指定生成数量为2
        actual = random.noncentral_f(0.9, 0.9, 2, size=2)
        # 断言实际结果与预期结果的准确度
        assert_array_almost_equal(actual, desired, decimal=14)

    # 定义测试函数 `test_chisquare`，用于测试卡方分布生成函数的各种情况
    def test_chisquare(self):
        # 设置自由度参数为 [1]，表示自由度参数列表
        df = [1]
        # 设置不良自由度参数为 [-1]，表示不良自由度参数列表
        bad_df = [-1]
        # 将随机生成的卡方分布函数赋值给变量 chisquare
        chisquare = random.chisquare
        # 设置预期结果为指定的数组，精确到小数点后14位
        desired = np.array([0.57022801133088286,
                            0.51947702108840776,
                            0.1320969254923558])

        # 调用函数设置随机种子
        self.set_seed()
        # 调用卡方分布函数计算实际结果，验证处理多个自由度参数的情况
        actual = chisquare(df * 3)
        # 断言实际结果与预期结果的准确度
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言处理不良自由度参数情况，预期会抛出 ValueError 异常
        assert_raises(ValueError, chisquare, bad_df * 3)

    # 定义测试函数 `test_noncentral_chisquare`，用于测试非中心卡方分布生成函数的各种情况
    def test_noncentral_chisquare(self):
        # 设置自由度参数为 [1]，表示自由度参数列表
        df = [1]
        # 设置非中心参数为 [2]，表示非中心参数列表
        nonc = [2]
        # 设置不良自由度参数为 [-1]，表示不良自由度参数列表
        bad_df = [-1]
        # 设置不良非中心参数为 [-2]，表示不良非中心参数列表
        bad_nonc = [-2]
        # 将随机生成的非中心卡方分布函数赋值给变量 nonc_chi
        nonc_chi = random.noncentral_chisquare
        # 设置预期结果为指定的数组，精确到小数点后14位
        desired = np.array([9.0015599467913763,
                            4.5804135049718742,
                            6.0872302432834564])

        # 调用函数设置随机种子
        self.set_seed()
        # 调用非中心卡方分布函数计算实际结果，验证处理多个自由度参数的情况
        actual = nonc_chi(df * 3, nonc)
        # 断言实际结果与预期结果的准确度
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言处理不良自由度参数情况，预期会抛出 ValueError 异常
        assert_raises(ValueError, nonc_chi, bad_df * 3, nonc)
        assert_raises(ValueError, nonc_chi, df * 3, bad_nonc)

        # 重设随机种子
        self.set_seed()
        # 重新计算实际结果，验证处理多个非中心参数的情况
        actual = nonc_chi(df, nonc * 3)
        # 断言实际结果与预期结果的准确度
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言处理不良非中心参数情况，预期会抛出 ValueError 异常
        assert_raises(ValueError, nonc_chi, bad_df, nonc * 3)
        assert_raises(ValueError, nonc_chi, df, bad_nonc * 3)
    # 定义一个测试方法，用于测试标准 t 分布的随机数生成
    def test_standard_t(self):
        # 创建一个包含单个元素的列表 df
        df = [1]
        # 创建一个包含单个负数元素的列表 bad_df
        bad_df = [-1]
        # 获取 random.standard_t 函数的引用并赋值给 t
        t = random.standard_t
        # 期望的 t 分布随机数结果，用 NumPy 数组表示
        desired = np.array([3.0702872575217643,
                            5.8560725167361607,
                            1.0274791436474273])

        # 设置随机数种子
        self.set_seed()
        # 使用 t 函数生成 df * 3 对应的 t 分布随机数，并赋值给 actual
        actual = t(df * 3)
        # 检查 actual 是否与 desired 几乎相等，精度为 14 位小数
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言 ValueError 是否会被触发，传入 bad_df * 3 作为参数调用 t 函数
        assert_raises(ValueError, t, bad_df * 3)
        # 断言 ValueError 是否会被触发，传入 bad_df * 3 作为参数调用 random.standard_t 函数

    # 定义一个测试方法，用于测试 von Mises 分布的随机数生成
    def test_vonmises(self):
        # 创建包含单个元素的列表 mu
        mu = [2]
        # 创建包含单个元素的列表 kappa
        kappa = [1]
        # 创建包含单个负数元素的列表 bad_kappa
        bad_kappa = [-1]
        # 获取 random.vonmises 函数的引用并赋值给 vonmises
        vonmises = random.vonmises
        # 期望的 von Mises 分布随机数结果，用 NumPy 数组表示
        desired = np.array([2.9883443664201312,
                            -2.7064099483995943,
                            -1.8672476700665914])

        # 设置随机数种子
        self.set_seed()
        # 使用 vonmises 函数生成 mu * 3, kappa 对应的 von Mises 分布随机数，并赋值给 actual
        actual = vonmises(mu * 3, kappa)
        # 检查 actual 是否与 desired 几乎相等，精度为 14 位小数
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言 ValueError 是否会被触发，传入 mu * 3, bad_kappa 作为参数调用 vonmises 函数

        # 设置随机数种子
        self.set_seed()
        # 使用 vonmises 函数生成 mu, kappa * 3 对应的 von Mises 分布随机数，并赋值给 actual
        actual = vonmises(mu, kappa * 3)
        # 检查 actual 是否与 desired 几乎相等，精度为 14 位小数
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言 ValueError 是否会被触发，传入 mu, bad_kappa * 3 作为参数调用 vonmises 函数

    # 定义一个测试方法，用于测试 Pareto 分布的随机数生成
    def test_pareto(self):
        # 创建包含单个元素的列表 a
        a = [1]
        # 创建包含单个负数元素的列表 bad_a
        bad_a = [-1]
        # 获取 random.pareto 函数的引用并赋值给 pareto
        pareto = random.pareto
        # 期望的 Pareto 分布随机数结果，用 NumPy 数组表示
        desired = np.array([1.1405622680198362,
                            1.1465519762044529,
                            1.0389564467453547])

        # 设置随机数种子
        self.set_seed()
        # 使用 pareto 函数生成 a * 3 对应的 Pareto 分布随机数，并赋值给 actual
        actual = pareto(a * 3)
        # 检查 actual 是否与 desired 几乎相等，精度为 14 位小数
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言 ValueError 是否会被触发，传入 bad_a * 3 作为参数调用 pareto 函数
        assert_raises(ValueError, pareto, bad_a * 3)
        # 断言 ValueError 是否会被触发，传入 bad_a * 3 作为参数调用 random.pareto 函数

    # 定义一个测试方法，用于测试 Weibull 分布的随机数生成
    def test_weibull(self):
        # 创建包含单个元素的列表 a
        a = [1]
        # 创建包含单个负数元素的列表 bad_a
        bad_a = [-1]
        # 获取 random.weibull 函数的引用并赋值给 weibull
        weibull = random.weibull
        # 期望的 Weibull 分布随机数结果，用 NumPy 数组表示
        desired = np.array([0.76106853658845242,
                            0.76386282278691653,
                            0.71243813125891797])

        # 设置随机数种子
        self.set_seed()
        # 使用 weibull 函数生成 a * 3 对应的 Weibull 分布随机数，并赋值给 actual
        actual = weibull(a * 3)
        # 检查 actual 是否与 desired 几乎相等，精度为 14 位小数
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言 ValueError 是否会被触发，传入 bad_a * 3 作为参数调用 weibull 函数
        assert_raises(ValueError, weibull, bad_a * 3)
        # 断言 ValueError 是否会被触发，传入 bad_a * 3 作为参数调用 random.weibull 函数

    # 定义一个测试方法，用于测试 Power 分布的随机数生成
    def test_power(self):
        # 创建包含单个元素的列表 a
        a = [1]
        # 创建包含单个负数元素的列表 bad_a
        bad_a = [-1]
        # 获取 random.power 函数的引用并赋值给 power
        power = random.power
        # 期望的 Power 分布随机数结果，用 NumPy 数组表示
        desired = np.array([0.53283302478975902,
                            0.53413660089041659,
                            0.50955303552646702])

        # 设置随机数种子
        self.set_seed()
        # 使用 power 函数生成 a * 3 对应的 Power 分布随机数，并赋值给 actual
        actual = power(a * 3)
        # 检查 actual 是否与 desired 几乎相等，精度为 14 位小数
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言 ValueError 是否会被触发，传入 bad_a * 3 作为参数调用 power 函数
        assert_raises(ValueError, power, bad_a * 3)
        # 断言 ValueError 是否会被触发，传入 bad_a * 3 作为参数调用 random.power 函数
    # 定义测试 Laplace 分布的方法
    def test_laplace(self):
        # 设置分布的位置参数为列表 [0]
        loc = [0]
        # 设置分布的尺度参数为列表 [1]
        scale = [1]
        # 设置错误的尺度参数为列表 [-1]
        bad_scale = [-1]
        # 使用 random 模块中的 Laplace 函数
        laplace = random.laplace
        # 预期的结果，作为 NumPy 数组
        desired = np.array([0.067921356028507157,
                            0.070715642226971326,
                            0.019290950698972624])

        # 设定随机数种子
        self.set_seed()
        # 调用 Laplace 函数生成实际结果
        actual = laplace(loc * 3, scale)
        # 断言实际结果与预期结果近似相等，精度为 14 位小数
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言调用带有错误尺度参数会引发 ValueError 异常
        assert_raises(ValueError, laplace, loc * 3, bad_scale)

        # 再次设定随机数种子
        self.set_seed()
        # 使用不同的尺度参数调用 Laplace 函数生成另一组实际结果
        actual = laplace(loc, scale * 3)
        # 断言实际结果与预期结果近似相等，精度为 14 位小数
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言调用带有错误尺度参数会引发 ValueError 异常
        assert_raises(ValueError, laplace, loc, bad_scale * 3)

    # 定义测试 Gumbel 分布的方法
    def test_gumbel(self):
        # 设置分布的位置参数为列表 [0]
        loc = [0]
        # 设置分布的尺度参数为列表 [1]
        scale = [1]
        # 设置错误的尺度参数为列表 [-1]
        bad_scale = [-1]
        # 使用 random 模块中的 Gumbel 函数
        gumbel = random.gumbel
        # 预期的结果，作为 NumPy 数组
        desired = np.array([0.2730318639556768,
                            0.26936705726291116,
                            0.33906220393037939])

        # 设定随机数种子
        self.set_seed()
        # 调用 Gumbel 函数生成实际结果
        actual = gumbel(loc * 3, scale)
        # 断言实际结果与预期结果近似相等，精度为 14 位小数
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言调用带有错误尺度参数会引发 ValueError 异常
        assert_raises(ValueError, gumbel, loc * 3, bad_scale)

        # 再次设定随机数种子
        self.set_seed()
        # 使用不同的尺度参数调用 Gumbel 函数生成另一组实际结果
        actual = gumbel(loc, scale * 3)
        # 断言实际结果与预期结果近似相等，精度为 14 位小数
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言调用带有错误尺度参数会引发 ValueError 异常
        assert_raises(ValueError, gumbel, loc, bad_scale * 3)

    # 定义测试 Logistic 分布的方法
    def test_logistic(self):
        # 设置分布的位置参数为列表 [0]
        loc = [0]
        # 设置分布的尺度参数为列表 [1]
        scale = [1]
        # 设置错误的尺度参数为列表 [-1]
        bad_scale = [-1]
        # 使用 random 模块中的 Logistic 函数
        logistic = random.logistic
        # 预期的结果，作为 NumPy 数组
        desired = np.array([0.13152135837586171,
                            0.13675915696285773,
                            0.038216792802833396])

        # 设定随机数种子
        self.set_seed()
        # 调用 Logistic 函数生成实际结果
        actual = logistic(loc * 3, scale)
        # 断言实际结果与预期结果近似相等，精度为 14 位小数
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言调用带有错误尺度参数会引发 ValueError 异常
        assert_raises(ValueError, logistic, loc * 3, bad_scale)

        # 再次设定随机数种子
        self.set_seed()
        # 使用不同的尺度参数调用 Logistic 函数生成另一组实际结果
        actual = logistic(loc, scale * 3)
        # 断言实际结果与预期结果近似相等，精度为 14 位小数
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言调用带有错误尺度参数会引发 ValueError 异常
        assert_raises(ValueError, logistic, loc, bad_scale * 3)
        # 断言调用 Logistic 函数，当尺度参数为 0 时返回 1.0
        assert_equal(random.logistic(1.0, 0.0), 1.0)

    # 定义测试 Lognormal 分布的方法
    def test_lognormal(self):
        # 设置分布的均值参数为列表 [0]
        mean = [0]
        # 设置分布的标准差参数为列表 [1]
        sigma = [1]
        # 设置错误的标准差参数为列表 [-1]
        bad_sigma = [-1]
        # 使用 random 模块中的 Lognormal 函数
        lognormal = random.lognormal
        # 预期的结果，作为 NumPy 数组
        desired = np.array([9.1422086044848427,
                            8.4013952870126261,
                            6.3073234116578671])

        # 设定随机数种子
        self.set_seed()
        # 调用 Lognormal 函数生成实际结果
        actual = lognormal(mean * 3, sigma)
        # 断言实际结果与预期结果近似相等，精度为 14 位小数
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言调用带有错误标准差参数会引发 ValueError 异常
        assert_raises(ValueError, lognormal, mean * 3, bad_sigma)
        # 断言调用 random 模块中的 Lognormal 函数，带有错误标准差参数会引发 ValueError 异常
        assert_raises(ValueError, random.lognormal, mean * 3, bad_sigma)

        # 再次设定随机数种子
        self.set_seed()
        # 使用不同的标准差参数调用 Lognormal 函数生成另一组实际结果
        actual = lognormal(mean, sigma * 3)
        # 断言实际结果与预期结果近似相等，精度为 14 位小数
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言调用带有错误标准差参数会引发 ValueError 异常
        assert_raises(ValueError, lognormal, mean, bad_sigma * 3)
        # 断言调用 random 模块中的 Lognormal 函数，带有错误标准差参数会引发 ValueError 异常
        assert_raises(ValueError, random.lognormal, mean, bad_sigma * 3)
    # 定义一个测试函数，用于测试 Rayleigh 分布生成随机数的准确性和异常处理
    def test_rayleigh(self):
        # 设定 Rayleigh 分布的参数，这里 scale 是一个包含正数的列表
        scale = [1]
        # 设定一个错误的 scale 参数，包含负数
        bad_scale = [-1]
        # 获取随机数生成函数 random.rayleigh 的引用
        rayleigh = random.rayleigh
        # 期望得到的随机数结果，使用 numpy 数组存储
        desired = np.array([1.2337491937897689,
                            1.2360119924878694,
                            1.1936818095781789])

        # 设定随机数种子
        self.set_seed()
        # 生成 Rayleigh 分布的随机数，参数为 scale * 3
        actual = rayleigh(scale * 3)
        # 断言生成的随机数结果与期望值的接近程度
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言生成具有错误 scale 参数的 Rayleigh 分布会引发 ValueError 异常
        assert_raises(ValueError, rayleigh, bad_scale * 3)

    # 定义一个测试函数，用于测试 Wald 分布生成随机数的准确性和异常处理
    def test_wald(self):
        # 设定 Wald 分布的均值参数 mean 和尺度参数 scale，分别为包含正数的列表
        mean = [0.5]
        scale = [1]
        # 设定错误的均值和尺度参数，分别包含 0 和负数
        bad_mean = [0]
        bad_scale = [-2]
        # 获取随机数生成函数 random.wald 的引用
        wald = random.wald
        # 期望得到的随机数结果，使用 numpy 数组存储
        desired = np.array([0.11873681120271318,
                            0.12450084820795027,
                            0.9096122728408238])

        # 设定随机数种子
        self.set_seed()
        # 生成 Wald 分布的随机数，mean 参数重复 3 次，scale 参数不变
        actual = wald(mean * 3, scale)
        # 断言生成的随机数结果与期望值的接近程度
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言生成具有错误均值参数的 Wald 分布会引发 ValueError 异常
        assert_raises(ValueError, wald, bad_mean * 3, scale)
        # 断言生成具有错误尺度参数的 Wald 分布会引发 ValueError 异常
        assert_raises(ValueError, wald, mean * 3, bad_scale)
        # 断言直接调用 random.wald 生成具有错误均值参数的 Wald 分布会引发 ValueError 异常
        assert_raises(ValueError, random.wald, bad_mean * 3, scale)
        # 断言直接调用 random.wald 生成具有错误尺度参数的 Wald 分布会引发 ValueError 异常
        assert_raises(ValueError, random.wald, mean * 3, bad_scale)

        # 设定随机数种子
        self.set_seed()
        # 生成 Wald 分布的随机数，mean 参数不变，scale 参数重复 3 次
        actual = wald(mean, scale * 3)
        # 断言生成的随机数结果与期望值的接近程度
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言生成具有错误均值参数的 Wald 分布会引发 ValueError 异常
        assert_raises(ValueError, wald, bad_mean, scale * 3)
        # 断言生成具有错误尺度参数的 Wald 分布会引发 ValueError 异常
        assert_raises(ValueError, wald, mean, bad_scale * 3)
        # 断言生成具有错误均值参数为 0 的 Wald 分布会引发 ValueError 异常
        assert_raises(ValueError, wald, 0.0, 1)
        # 断言生成具有错误尺度参数为 0 的 Wald 分布会引发 ValueError 异常
        assert_raises(ValueError, wald, 0.5, 0.0)
    # 定义一个测试方法，用于测试 random.triangular 函数的行为
    def test_triangular(self):
        # 准备测试数据：left、right、mode 是参数列表，bad_left_one、bad_mode_one、bad_left_two、bad_mode_two 是异常情况的参数
        left = [1]
        right = [3]
        mode = [2]
        bad_left_one = [3]
        bad_mode_one = [4]
        bad_left_two, bad_mode_two = right * 2
        # 获取 random.triangular 函数的引用
        triangular = random.triangular
        # 期望的结果数组
        desired = np.array([2.03339048710429,
                            2.0347400359389356,
                            2.0095991069536208])

        # 设置随机种子
        self.set_seed()
        # 测试 triangular 函数正常情况下的行为
        actual = triangular(left * 3, mode, right)
        assert_array_almost_equal(actual, desired, decimal=14)
        # 测试 triangular 函数在不合法参数下是否抛出 ValueError 异常
        assert_raises(ValueError, triangular, bad_left_one * 3, mode, right)
        assert_raises(ValueError, triangular, left * 3, bad_mode_one, right)
        assert_raises(ValueError, triangular, bad_left_two * 3, bad_mode_two,
                      right)

        # 设置随机种子
        self.set_seed()
        # 测试 triangular 函数在参数 mode 扩展的情况下的行为
        actual = triangular(left, mode * 3, right)
        assert_array_almost_equal(actual, desired, decimal=14)
        # 测试 triangular 函数在不合法参数下是否抛出 ValueError 异常
        assert_raises(ValueError, triangular, bad_left_one, mode * 3, right)
        assert_raises(ValueError, triangular, left, bad_mode_one * 3, right)
        assert_raises(ValueError, triangular, bad_left_two, bad_mode_two * 3,
                      right)

        # 设置随机种子
        self.set_seed()
        # 测试 triangular 函数在参数 right 扩展的情况下的行为
        actual = triangular(left, mode, right * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        # 测试 triangular 函数在不合法参数下是否抛出 ValueError 异常
        assert_raises(ValueError, triangular, bad_left_one, mode, right * 3)
        assert_raises(ValueError, triangular, left, bad_mode_one, right * 3)
        assert_raises(ValueError, triangular, bad_left_two, bad_mode_two,
                      right * 3)

        # 测试 triangular 函数在边界参数下是否抛出 ValueError 异常
        assert_raises(ValueError, triangular, 10., 0., 20.)
        assert_raises(ValueError, triangular, 10., 25., 20.)
        assert_raises(ValueError, triangular, 10., 10., 10.)

    # 定义一个测试方法，用于测试 random.binomial 函数的行为
    def test_binomial(self):
        # 准备测试数据：n、p 是参数列表，bad_n、bad_p_one、bad_p_two 是异常情况的参数
        n = [1]
        p = [0.5]
        bad_n = [-1]
        bad_p_one = [-1]
        bad_p_two = [1.5]
        # 获取 random.binomial 函数的引用
        binom = random.binomial
        # 期望的结果数组
        desired = np.array([1, 1, 1])

        # 设置随机种子
        self.set_seed()
        # 测试 binomial 函数正常情况下的行为
        actual = binom(n * 3, p)
        assert_array_equal(actual, desired)
        # 测试 binomial 函数在不合法参数下是否抛出 ValueError 异常
        assert_raises(ValueError, binom, bad_n * 3, p)
        assert_raises(ValueError, binom, n * 3, bad_p_one)
        assert_raises(ValueError, binom, n * 3, bad_p_two)

        # 设置随机种子
        self.set_seed()
        # 测试 binomial 函数在参数 p 扩展的情况下的行为
        actual = binom(n, p * 3)
        assert_array_equal(actual, desired)
        # 测试 binomial 函数在不合法参数下是否抛出 ValueError 异常
        assert_raises(ValueError, binom, bad_n, p * 3)
        assert_raises(ValueError, binom, n, bad_p_one * 3)
        assert_raises(ValueError, binom, n, bad_p_two * 3)
    # 定义测试负二项分布的方法
    def test_negative_binomial(self):
        # 初始化 n, p, bad_n, bad_p_one, bad_p_two 数组
        n = [1]
        p = [0.5]
        bad_n = [-1]
        bad_p_one = [-1]
        bad_p_two = [1.5]
        # 获取 random.negative_binomial 函数引用
        neg_binom = random.negative_binomial
        # 期望的结果数组
        desired = np.array([1, 0, 1])

        # 设置随机种子
        self.set_seed()
        # 测试三倍的 n 和单个的 p 的负二项分布
        actual = neg_binom(n * 3, p)
        assert_array_equal(actual, desired)
        # 测试负二项分布函数对 bad_n * 3 的输入是否抛出 ValueError 异常
        assert_raises(ValueError, neg_binom, bad_n * 3, p)
        # 测试负二项分布函数对 n * 3 和 bad_p_one 的输入是否抛出 ValueError 异常
        assert_raises(ValueError, neg_binom, n * 3, bad_p_one)
        # 测试负二项分布函数对 n * 3 和 bad_p_two 的输入是否抛出 ValueError 异常

        assert_raises(ValueError, neg_binom, n * 3, bad_p_two)

        # 重新设置随机种子
        self.set_seed()
        # 测试单个的 n 和三倍的 p 的负二项分布
        actual = neg_binom(n, p * 3)
        assert_array_equal(actual, desired)
        # 测试负二项分布函数对 bad_n 和 p * 3 的输入是否抛出 ValueError 异常
        assert_raises(ValueError, neg_binom, bad_n, p * 3)
        # 测试负二项分布函数对 n 和 bad_p_one * 3 的输入是否抛出 ValueError 异常
        assert_raises(ValueError, neg_binom, n, bad_p_one * 3)
        # 测试负二项分布函数对 n 和 bad_p_two * 3 的输入是否抛出 ValueError 异常
        assert_raises(ValueError, neg_binom, n, bad_p_two * 3)

    # 定义测试泊松分布的方法
    def test_poisson(self):
        # 获取最大的泊松分布参数
        max_lam = random.RandomState()._poisson_lam_max

        # 初始化 lam, bad_lam_one, bad_lam_two 数组
        lam = [1]
        bad_lam_one = [-1]
        bad_lam_two = [max_lam * 2]
        # 获取 random.poisson 函数引用
        poisson = random.poisson
        # 期望的结果数组
        desired = np.array([1, 1, 0])

        # 设置随机种子
        self.set_seed()
        # 测试三倍的 lam 的泊松分布
        actual = poisson(lam * 3)
        assert_array_equal(actual, desired)
        # 测试泊松分布函数对 bad_lam_one * 3 的输入是否抛出 ValueError 异常
        assert_raises(ValueError, poisson, bad_lam_one * 3)
        # 测试泊松分布函数对 bad_lam_two * 3 的输入是否抛出 ValueError 异常
        assert_raises(ValueError, poisson, bad_lam_two * 3)

    # 定义测试 Zipf 分布的方法
    def test_zipf(self):
        # 初始化 a, bad_a 数组
        a = [2]
        bad_a = [0]
        # 获取 random.zipf 函数引用
        zipf = random.zipf
        # 期望的结果数组
        desired = np.array([2, 2, 1])

        # 设置随机种子
        self.set_seed()
        # 测试三倍的 a 的 Zipf 分布
        actual = zipf(a * 3)
        assert_array_equal(actual, desired)
        # 测试 Zipf 分布函数对 bad_a * 3 的输入是否抛出 ValueError 异常
        assert_raises(ValueError, zipf, bad_a * 3)
        # 在忽略无效值错误的上下文中，测试 Zipf 分布函数对 np.nan 的输入是否抛出 ValueError 异常
        with np.errstate(invalid='ignore'):
            assert_raises(ValueError, zipf, np.nan)
            # 测试 Zipf 分布函数对 [0, 0, np.nan] 的输入是否抛出 ValueError 异常
            assert_raises(ValueError, zipf, [0, 0, np.nan])

    # 定义测试几何分布的方法
    def test_geometric(self):
        # 初始化 p, bad_p_one, bad_p_two 数组
        p = [0.5]
        bad_p_one = [-1]
        bad_p_two = [1.5]
        # 获取 random.geometric 函数引用
        geom = random.geometric
        # 期望的结果数组
        desired = np.array([2, 2, 2])

        # 设置随机种子
        self.set_seed()
        # 测试三倍的 p 的几何分布
        actual = geom(p * 3)
        assert_array_equal(actual, desired)
        # 测试几何分布函数对 bad_p_one * 3 的输入是否抛出 ValueError 异常
        assert_raises(ValueError, geom, bad_p_one * 3)
        # 测试几何分布函数对 bad_p_two * 3 的输入是否抛出 ValueError 异常
        assert_raises(ValueError, geom, bad_p_two * 3)
    # 定义一个测试函数，用于测试超几何分布的随机数生成函数
    def test_hypergeometric(self):
        # 设置好几个测试用例的参数
        ngood = [1]  # 成功事件数的列表
        nbad = [2]   # 失败事件数的列表
        nsample = [2]  # 抽样数量的列表
        bad_ngood = [-1]  # 错误的成功事件数列表
        bad_nbad = [-2]   # 错误的失败事件数列表
        bad_nsample_one = [0]  # 错误的抽样数量列表（情况一）
        bad_nsample_two = [4]  # 错误的抽样数量列表（情况二）
        hypergeom = random.hypergeometric  # 超几何分布的随机数生成函数
        desired = np.array([1, 1, 1])  # 期望的结果数组

        # 设置随机数种子
        self.set_seed()
        # 进行超几何分布的随机数生成，并检查结果是否符合预期
        actual = hypergeom(ngood * 3, nbad, nsample)
        assert_array_equal(actual, desired)
        # 检查是否会抛出错误，对于错误的参数情况
        assert_raises(ValueError, hypergeom, bad_ngood * 3, nbad, nsample)
        assert_raises(ValueError, hypergeom, ngood * 3, bad_nbad, nsample)
        assert_raises(ValueError, hypergeom, ngood * 3, nbad, bad_nsample_one)
        assert_raises(ValueError, hypergeom, ngood * 3, nbad, bad_nsample_two)

        # 再次设置随机数种子
        self.set_seed()
        # 变换参数顺序，重新测试超几何分布的随机数生成
        actual = hypergeom(ngood, nbad * 3, nsample)
        assert_array_equal(actual, desired)
        # 检查是否会抛出错误，对于错误的参数情况
        assert_raises(ValueError, hypergeom, bad_ngood, nbad * 3, nsample)
        assert_raises(ValueError, hypergeom, ngood, bad_nbad * 3, nsample)
        assert_raises(ValueError, hypergeom, ngood, nbad * 3, bad_nsample_one)
        assert_raises(ValueError, hypergeom, ngood, nbad * 3, bad_nsample_two)

        # 再次设置随机数种子
        self.set_seed()
        # 变换抽样数量参数，重新测试超几何分布的随机数生成
        actual = hypergeom(ngood, nbad, nsample * 3)
        assert_array_equal(actual, desired)
        # 检查是否会抛出错误，对于错误的参数情况
        assert_raises(ValueError, hypergeom, bad_ngood, nbad, nsample * 3)
        assert_raises(ValueError, hypergeom, ngood, bad_nbad, nsample * 3)
        assert_raises(ValueError, hypergeom, ngood, nbad, bad_nsample_one * 3)
        assert_raises(ValueError, hypergeom, ngood, nbad, bad_nsample_two * 3)

        # 检查一些特殊的错误情况，确保函数能正确处理异常情况
        assert_raises(ValueError, hypergeom, -1, 10, 20)
        assert_raises(ValueError, hypergeom, 10, -1, 20)
        assert_raises(ValueError, hypergeom, 10, 10, 0)
        assert_raises(ValueError, hypergeom, 10, 10, 25)

    # 定义一个测试函数，用于测试对数级数分布的随机数生成函数
    def test_logseries(self):
        p = [0.5]  # 概率参数的列表
        bad_p_one = [2]  # 错误的概率参数列表（情况一）
        bad_p_two = [-1]  # 错误的概率参数列表（情况二）
        logseries = random.logseries  # 对数级数分布的随机数生成函数
        desired = np.array([1, 1, 1])  # 期望的结果数组

        # 设置随机数种子
        self.set_seed()
        # 进行对数级数分布的随机数生成，并检查结果是否符合预期
        actual = logseries(p * 3)
        assert_array_equal(actual, desired)
        # 检查是否会抛出错误，对于错误的参数情况
        assert_raises(ValueError, logseries, bad_p_one * 3)
        assert_raises(ValueError, logseries, bad_p_two * 3)
# 如果运行环境是 WASM，跳过当前测试类，因为无法启动线程
@pytest.mark.skipif(IS_WASM, reason="can't start thread")
class TestThread:
    # 确保即使在多线程情况下，每个状态都能产生相同的随机序列
    def setup_method(self):
        # 初始化种子范围
        self.seeds = range(4)

    def check_function(self, function, sz):
        from threading import Thread

        # 创建空的数组，用于存放多个随机序列的结果
        out1 = np.empty((len(self.seeds),) + sz)
        out2 = np.empty((len(self.seeds),) + sz)

        # 多线程生成随机序列
        t = [Thread(target=function, args=(random.RandomState(s), o))
             for s, o in zip(self.seeds, out1)]
        [x.start() for x in t]  # 启动所有线程
        [x.join() for x in t]   # 等待所有线程执行完成

        # 单线程生成随机序列作为对照
        for s, o in zip(self.seeds, out2):
            function(random.RandomState(s), o)

        # 对比多线程和单线程生成的结果
        # 在某些平台（如 32 位 Windows）中，线程可能会改变 x87 FPU 的精度模式
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
        # 定义需要测试的函数列表
        funcs = (random.exponential, random.standard_gamma,
                 random.chisquare, random.standard_t,
                 random.pareto, random.weibull,
                 random.power, random.rayleigh,
                 random.poisson, random.zipf,
                 random.geometric, random.logseries)

        # 需要概率参数的函数列表
        probfuncs = (random.geometric, random.logseries)

        # 遍历每个函数并测试
        for func in funcs:
            if func in probfuncs:  # 对于需要概率小于 1.0 的函数
                out = func(np.array([0.5]))
            else:
                out = func(self.argOne)  # 使用预定义的输入参数进行函数调用

            # 断言函数返回的数组形状符合预期的形状
            assert_equal(out.shape, self.tgtShape)
    # 定义一个测试函数，用于测试接受两个参数的随机数生成函数
    def test_two_arg_funcs(self):
        # 定义包含多个随机数生成函数的元组
        funcs = (random.uniform, random.normal,
                 random.beta, random.gamma,
                 random.f, random.noncentral_chisquare,
                 random.vonmises, random.laplace,
                 random.gumbel, random.logistic,
                 random.lognormal, random.wald,
                 random.binomial, random.negative_binomial)

        # 定义接受两个参数的概率分布函数的元组
        probfuncs = (random.binomial, random.negative_binomial)

        # 遍历所有随机数生成函数
        for func in funcs:
            # 根据函数是否在概率分布函数中确定第二个参数的值
            if func in probfuncs:  # 如果是概率分布函数，第二个参数应小于等于1
                argTwo = np.array([0.5])
            else:
                argTwo = self.argTwo  # 否则使用测试对象的第二个参数

            # 测试函数的调用，传入第一个参数和确定的第二个参数
            out = func(self.argOne, argTwo)
            # 断言输出的形状与预期形状相同
            assert_equal(out.shape, self.tgtShape)

            # 测试函数的调用，传入第一个参数的第一个元素和确定的第二个参数
            out = func(self.argOne[0], argTwo)
            # 断言输出的形状与预期形状相同
            assert_equal(out.shape, self.tgtShape)

            # 测试函数的调用，传入第一个参数和确定的第二个参数的第一个元素
            out = func(self.argOne, argTwo[0])
            # 断言输出的形状与预期形状相同
            assert_equal(out.shape, self.tgtShape)

    # 定义一个测试函数，用于测试接受三个参数的随机数生成函数
    def test_three_arg_funcs(self):
        # 定义包含多个接受三个参数的随机数生成函数的列表
        funcs = [random.noncentral_f, random.triangular,
                 random.hypergeometric]

        # 遍历所有接受三个参数的随机数生成函数
        for func in funcs:
            # 测试函数的调用，传入三个测试对象作为参数
            out = func(self.argOne, self.argTwo, self.argThree)
            # 断言输出的形状与预期形状相同
            assert_equal(out.shape, self.tgtShape)

            # 测试函数的调用，传入第一个参数的第一个元素和两个测试对象作为参数
            out = func(self.argOne[0], self.argTwo, self.argThree)
            # 断言输出的形状与预期形状相同
            assert_equal(out.shape, self.tgtShape)

            # 测试函数的调用，传入两个测试对象的第一个元素和第一个测试对象作为参数
            out = func(self.argOne, self.argTwo[0], self.argThree)
            # 断言输出的形状与预期形状相同
            assert_equal(out.shape, self.tgtShape)
# 确保返回的数组在平台上的数据类型是正确的
def test_integer_dtype(int_func):
    # 设置随机数种子
    random.seed(123456789)
    # 解包传入的参数元组
    fname, args, sha256 = int_func
    # 根据函数名获取对应的随机数生成函数
    f = getattr(random, fname)
    # 调用随机数生成函数生成随机数数组
    actual = f(*args, size=2)
    # 断言生成的随机数数组的数据类型是否为 long 类型（对应于 'l' 数据类型）
    assert_(actual.dtype == np.dtype('l'))


def test_integer_repeat(int_func):
    # 设置随机数种子
    random.seed(123456789)
    # 解包传入的参数元组
    fname, args, sha256 = int_func
    # 根据函数名获取对应的随机数生成函数
    f = getattr(random, fname)
    # 调用随机数生成函数生成大规模随机数数组
    val = f(*args, size=1000000)
    # 如果系统的字节顺序不是小端（little endian），则进行字节交换
    if sys.byteorder != 'little':
        val = val.byteswap()
    # 计算生成的大规模随机数数组的哈希值
    res = hashlib.sha256(val.view(np.int8)).hexdigest()
    # 断言计算得到的哈希值与预期的哈希值是否相等
    assert_(res == sha256)


def test_broadcast_size_error():
    # 测试用例GH-16833，检查广播操作的尺寸错误
    with pytest.raises(ValueError):
        # 测试 binomial 函数，传入参数格式不符合预期将抛出 ValueError 异常
        random.binomial(1, [0.3, 0.7], size=(2, 1))
    with pytest.raises(ValueError):
        # 测试 binomial 函数，传入参数格式不符合预期将抛出 ValueError 异常
        random.binomial([1, 2], 0.3, size=(2, 1))
    with pytest.raises(ValueError):
        # 测试 binomial 函数，传入参数格式不符合预期将抛出 ValueError 异常
        random.binomial([1, 2], [0.3, 0.7], size=(2, 1))


def test_randomstate_ctor_old_style_pickle():
    # 测试用例，检查旧风格的 pickle 反序列化构造函数
    rs = np.random.RandomState(MT19937(0))
    rs.standard_normal(1)
    # 直接调用 __reduce__ 方法，用于序列化和反序列化
    ctor, args, state_a = rs.__reduce__()
    # 模拟反序列化一个只有名称的旧 pickle 对象
    assert args[0].__class__.__name__ == "MT19937"
    b = ctor(*("MT19937",))
    b.set_state(state_a)
    state_b = b.get_state(legacy=False)

    assert_equal(state_a['bit_generator'], state_b['bit_generator'])
    assert_array_equal(state_a['state']['key'], state_b['state']['key'])
    assert_array_equal(state_a['state']['pos'], state_b['state']['pos'])
    assert_equal(state_a['has_gauss'], state_b['has_gauss'])


def test_hot_swap(restore_singleton_bitgen):
    # 测试用例 GH 21808，检查热交换功能
    def_bg = np.random.default_rng(0)
    bg = def_bg.bit_generator
    np.random.set_bit_generator(bg)
    assert isinstance(np.random.mtrand._rand._bit_generator, type(bg))

    second_bg = np.random.get_bit_generator()
    assert bg is second_bg


def test_seed_alt_bit_gen(restore_singleton_bitgen):
    # 测试用例 GH 21808，检查使用备用位生成器的种子设置
    bg = PCG64(0)
    np.random.set_bit_generator(bg)
    state = np.random.get_state(legacy=False)
    np.random.seed(1)
    new_state = np.random.get_state(legacy=False)
    print(state)
    print(new_state)
    assert state["bit_generator"] == "PCG64"
    assert state["state"]["state"] != new_state["state"]["state"]
    assert state["state"]["inc"] != new_state["state"]["inc"]


def test_state_error_alt_bit_gen(restore_singleton_bitgen):
    # 测试用例 GH 21808，检查备用位生成器的状态错误处理
    state = np.random.get_state()
    bg = PCG64(0)
    np.random.set_bit_generator(bg)
    with pytest.raises(ValueError, match="state must be for a PCG64"):
        np.random.set_state(state)


def test_swap_worked(restore_singleton_bitgen):
    # 测试用例 GH 21808，检查位生成器切换功能
    np.random.seed(98765)
    vals = np.random.randint(0, 2 ** 30, 10)
    bg = PCG64(0)
    state = bg.state
    np.random.set_bit_generator(bg)
    state_direct = np.random.get_state(legacy=False)
    # 遍历 state 字典中的每个字段，确保其值与 state_direct 字典中对应字段的值相等
    for field in state:
        assert state[field] == state_direct[field]

    # 使用固定种子（98765）初始化 NumPy 的随机数生成器
    np.random.seed(98765)
    
    # 生成包含 10 个介于 0 到 2^30 之间的随机整数的数组
    pcg_vals = np.random.randint(0, 2 ** 30, 10)
    
    # 确保 vals 数组不全等于 pcg_vals 数组（即验证随机数生成是否有效）
    assert not np.all(vals == pcg_vals)

    # 将 bg 对象的 state 属性赋给 new_state 变量
    new_state = bg.state
    
    # 确保 new_state 字典中 "state" 键对应的值不等于 state 字典中同样键对应的值
    assert new_state["state"]["state"] != state["state"]["state"]
    
    # 确保 new_state 字典中 "state" 键对应的 "inc" 值等于 new_state 字典中同样键对应的 "inc" 值
    assert new_state["state"]["inc"] == new_state["state"]["inc"]
# 定义一个测试函数，用于测试“restore_singleton_bitgen”装饰器功能
def test_swapped_singleton_against_direct(restore_singleton_bitgen):
    # 设置 NumPy 的随机数生成器使用 PCG64 算法和种子 98765
    np.random.set_bit_generator(PCG64(98765))
    # 生成一个长度为 10 的随机整数数组，范围在 0 到 2^30 之间
    singleton_vals = np.random.randint(0, 2 ** 30, 10)
    # 使用指定种子创建一个 RandomState 对象
    rg = np.random.RandomState(PCG64(98765))
    # 生成一个长度为 10 的随机整数数组，范围在 0 到 2^30 之间
    non_singleton_vals = rg.randint(0, 2 ** 30, 10)
    # 断言非单例数组与单例数组相等
    assert_equal(non_singleton_vals, singleton_vals)
```