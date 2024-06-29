# `.\numpy\numpy\random\tests\test_random.py`

```py
# 导入警告模块，用于管理警告消息
import warnings

# 导入 pytest 测试框架，用于编写和运行测试用例
import pytest

# 导入 numpy 库，并从中导入测试相关的函数和类
import numpy as np
from numpy.testing import (
        assert_, assert_raises, assert_equal, assert_warns,
        assert_no_warnings, assert_array_equal, assert_array_almost_equal,
        suppress_warnings, IS_WASM
        )

# 导入 sys 模块，用于访问系统相关的参数和函数
import sys


# 定义一个测试类 TestSeed，用于测试随机种子的功能
class TestSeed:
    
    # 定义测试方法 test_scalar，测试标量种子情况
    def test_scalar(self):
        # 创建一个随机状态对象 s，种子为 0
        s = np.random.RandomState(0)
        # 断言从 s 生成的随机整数在指定范围内
        assert_equal(s.randint(1000), 684)
        
        # 创建一个随机状态对象 s，种子为 4294967295
        s = np.random.RandomState(4294967295)
        # 断言从 s 生成的随机整数在指定范围内
        assert_equal(s.randint(1000), 419)

    # 定义测试方法 test_array，测试数组种子情况
    def test_array(self):
        # 创建一个随机状态对象 s，种子为 range(10)
        s = np.random.RandomState(range(10))
        # 断言从 s 生成的随机整数在指定范围内
        assert_equal(s.randint(1000), 468)
        
        # 创建一个随机状态对象 s，种子为 np.arange(10)
        s = np.random.RandomState(np.arange(10))
        # 断言从 s 生成的随机整数在指定范围内
        assert_equal(s.randint(1000), 468)
        
        # 创建一个随机状态对象 s，种子为 [0]
        s = np.random.RandomState([0])
        # 断言从 s 生成的随机整数在指定范围内
        assert_equal(s.randint(1000), 973)
        
        # 创建一个随机状态对象 s，种子为 [4294967295]
        s = np.random.RandomState([4294967295])
        # 断言从 s 生成的随机整数在指定范围内
        assert_equal(s.randint(1000), 265)

    # 定义测试方法 test_invalid_scalar，测试无效的标量种子情况
    def test_invalid_scalar(self):
        # 断言使用无效种子 -0.5 会引发 TypeError 异常
        assert_raises(TypeError, np.random.RandomState, -0.5)
        # 断言使用无效种子 -1 会引发 ValueError 异常
        assert_raises(ValueError, np.random.RandomState, -1)

    # 定义测试方法 test_invalid_array，测试无效的数组种子情况
    def test_invalid_array(self):
        # 断言使用无效种子 [-0.5] 会引发 TypeError 异常
        assert_raises(TypeError, np.random.RandomState, [-0.5])
        # 断言使用无效种子 [-1] 会引发 ValueError 异常
        assert_raises(ValueError, np.random.RandomState, [-1])
        # 断言使用无效种子 [4294967296] 会引发 ValueError 异常
        assert_raises(ValueError, np.random.RandomState, [4294967296])
        # 断言使用无效种子 [1, 2, 4294967296] 会引发 ValueError 异常
        assert_raises(ValueError, np.random.RandomState, [1, 2, 4294967296])
        # 断言使用无效种子 [1, -2, 4294967296] 会引发 ValueError 异常
        assert_raises(ValueError, np.random.RandomState, [1, -2, 4294967296])

    # 定义测试方法 test_invalid_array_shape，测试无效的数组形状种子情况
    def test_invalid_array_shape(self):
        # 断言使用空数组作为种子会引发 ValueError 异常
        assert_raises(ValueError, np.random.RandomState, np.array([], dtype=np.int64))
        # 断言使用二维数组作为种子会引发 ValueError 异常
        assert_raises(ValueError, np.random.RandomState, [[1, 2, 3]])
        # 断言使用二维数组作为种子会引发 ValueError 异常
        assert_raises(ValueError, np.random.RandomState, [[1, 2, 3], [4, 5, 6]])


# 定义一个测试类 TestBinomial，用于测试二项分布相关功能
class TestBinomial:
    
    # 定义测试方法 test_n_zero，测试 n 为零的情况
    def test_n_zero(self):
        # 测试二项分布中 n == 0 的情况
        zeros = np.zeros(2, dtype='int')
        for p in [0, .5, 1]:
            # 断言在二项分布中 n 为 0 时，任何 p 值下的结果都应为 0
            assert_(random.binomial(0, p) == 0)
            # 断言在二项分布中 n 为数组时，任何 p 值下的结果都应为 0
            assert_array_equal(random.binomial(zeros, p), zeros)

    # 定义测试方法 test_p_is_nan，测试 p 值为 NaN 的情况
    def test_p_is_nan(self):
        # 断言当 p 值为 NaN 时，二项分布会引发 ValueError 异常
        assert_raises(ValueError, random.binomial, 1, np.nan)


# 定义一个测试类 TestMultinomial，用于测试多项分布相关功能
class TestMultinomial:
    
    # 定义基本的多项分布测试方法
    def test_basic(self):
        # 测试基本的多项分布生成
        random.multinomial(100, [0.2, 0.8])

    # 定义测试零概率的多项分布情况
    def test_zero_probability(self):
        # 测试多项分布中存在零概率的情况
        random.multinomial(100, [0.2, 0.8, 0.0, 0.0, 0.0])

    # 定义测试负整数范围的随机数生成情况
    def test_int_negative_interval(self):
        # 断言从负整数范围 [-5, -1) 生成的随机整数在指定范围内
        assert_(-5 <= random.randint(-5, -1) < -1)
        # 断言生成的随机数组 x 的所有值均在指定负整数范围内
        x = random.randint(-5, -1, 5)
        assert_(np.all(-5 <= x))
        assert_(np.all(x < -1))
    # 定义一个测试函数，用于测试 np.random.multinomial 方法的返回结果是否符合预期
    def test_size(self):
        # 测试用例1: 测试当定义概率分布 p=[0.5, 0.5] 时，返回结果的形状是否为 (1, 2)
        p = [0.5, 0.5]
        assert_equal(np.random.multinomial(1, p, np.uint32(1)).shape, (1, 2))
        # 测试用例2: 测试当定义概率分布 p=[0.5, 0.5] 时，返回结果的形状是否为 (1, 2)
        assert_equal(np.random.multinomial(1, p, np.uint32(1)).shape, (1, 2))
        # 测试用例3: 测试当定义概率分布 p=[0.5, 0.5] 时，返回结果的形状是否为 (1, 2)
        assert_equal(np.random.multinomial(1, p, np.uint32(1)).shape, (1, 2))
        # 测试用例4: 测试当定义概率分布 p=[0.5, 0.5] 时，返回结果的形状是否为 (2, 2, 2)
        assert_equal(np.random.multinomial(1, p, [2, 2]).shape, (2, 2, 2))
        # 测试用例5: 测试当定义概率分布 p=[0.5, 0.5] 时，返回结果的形状是否为 (2, 2, 2)
        assert_equal(np.random.multinomial(1, p, (2, 2)).shape, (2, 2, 2))
        # 测试用例6: 测试当定义概率分布 p=[0.5, 0.5] 时，返回结果的形状是否为 (2, 2, 2)
        assert_equal(np.random.multinomial(1, p, np.array((2, 2))).shape,
                     (2, 2, 2))
        # 测试用例7: 测试当输入参数类型错误时，是否会抛出类型错误异常
        assert_raises(TypeError, np.random.multinomial, 1, p,
                      float(1))

    # 定义一个测试函数，用于测试 np.random.multinomial 方法对多维概率分布的处理情况
    def test_multidimensional_pvals(self):
        # 测试用例1: 测试当概率分布为二维列表时，是否会抛出值错误异常
        assert_raises(ValueError, np.random.multinomial, 10, [[0, 1]])
        # 测试用例2: 测试当概率分布为二维列表时，是否会抛出值错误异常
        assert_raises(ValueError, np.random.multinomial, 10, [[0], [1]])
        # 测试用例3: 测试当概率分布为三维列表时，是否会抛出值错误异常
        assert_raises(ValueError, np.random.multinomial, 10, [[[0], [1]], [[1], [0]]])
        # 测试用例4: 测试当概率分布为 2x2 数组时，是否会抛出值错误异常
        assert_raises(ValueError, np.random.multinomial, 10, np.array([[0, 1], [1, 0]])
class TestSetState:
    # 设置测试方法的初始化状态
    def setup_method(self):
        # 设定种子值
        self.seed = 1234567890
        # 使用种子创建伪随机数生成器对象
        self.prng = random.RandomState(self.seed)
        # 获取当前生成器对象的状态
        self.state = self.prng.get_state()

    # 测试基本功能
    def test_basic(self):
        # 生成旧状态下的随机整数数组
        old = self.prng.tomaxint(16)
        # 恢复到初始状态
        self.prng.set_state(self.state)
        # 生成新状态下的随机整数数组
        new = self.prng.tomaxint(16)
        # 断言旧状态和新状态的数组元素是否全部相等
        assert_(np.all(old == new))

    # 测试高斯分布重置
    def test_gaussian_reset(self):
        # 确保缓存的每隔一个高斯值被重置
        old = self.prng.standard_normal(size=3)
        # 恢复到初始状态
        self.prng.set_state(self.state)
        # 生成新状态下的高斯分布数组
        new = self.prng.standard_normal(size=3)
        # 断言旧状态和新状态的数组元素是否全部相等
        assert_(np.all(old == new))

    # 在使用中保存了带有缓存高斯值的状态时，确保高斯值被正确恢复
    def test_gaussian_reset_in_media_res(self):
        # 进行一次标准正态分布采样，缓存高斯值
        self.prng.standard_normal()
        # 获取当前生成器对象的状态
        state = self.prng.get_state()
        # 生成旧状态下的高斯分布数组
        old = self.prng.standard_normal(size=3)
        # 恢复到指定状态
        self.prng.set_state(state)
        # 生成新状态下的高斯分布数组
        new = self.prng.standard_normal(size=3)
        # 断言旧状态和新状态的数组元素是否全部相等
        assert_(np.all(old == new))

    # 测试向后兼容性
    def test_backwards_compatibility(self):
        # 确保可以接受不带有缓存高斯值的旧状态元组
        old_state = self.state[:-2]
        # 生成旧状态下的随机整数数组
        x1 = self.prng.standard_normal(size=16)
        # 恢复到指定状态
        self.prng.set_state(old_state)
        # 生成新状态下的随机整数数组
        x2 = self.prng.standard_normal(size=16)
        # 恢复到初始状态
        self.prng.set_state(self.state)
        # 生成另一组新状态下的随机整数数组
        x3 = self.prng.standard_normal(size=16)
        # 断言x1和x2的数组元素是否全部相等
        assert_(np.all(x1 == x2))
        # 断言x1和x3的数组元素是否全部相等
        assert_(np.all(x1 == x3))

    # 测试负二项分布
    def test_negative_binomial(self):
        # 确保负二项分布可以接受浮点数参数而不会截断
        self.prng.negative_binomial(0.5, 0.5)

    # 测试设置无效状态
    def test_set_invalid_state(self):
        # 期望抛出IndexError异常
        with pytest.raises(IndexError):
            self.prng.set_state(())


class TestRandint:

    rfunc = np.random.randint

    # 合法的整数/布尔类型
    itype = [np.bool, np.int8, np.uint8, np.int16, np.uint16,
             np.int32, np.uint32, np.int64, np.uint64]

    # 测试不支持的数据类型
    def test_unsupported_type(self):
        assert_raises(TypeError, self.rfunc, 1, dtype=float)

    # 测试边界检查
    def test_bounds_checking(self):
        for dt in self.itype:
            # 确定下界
            lbnd = 0 if dt is np.bool else np.iinfo(dt).min
            # 确定上界
            ubnd = 2 if dt is np.bool else np.iinfo(dt).max + 1
            # 期望抛出ValueError异常
            assert_raises(ValueError, self.rfunc, lbnd - 1, ubnd, dtype=dt)
            # 期望抛出ValueError异常
            assert_raises(ValueError, self.rfunc, lbnd, ubnd + 1, dtype=dt)
            # 期望抛出ValueError异常
            assert_raises(ValueError, self.rfunc, ubnd, lbnd, dtype=dt)
            # 期望抛出ValueError异常
            assert_raises(ValueError, self.rfunc, 1, 0, dtype=dt)
    def test_rng_zero_and_extremes(self):
        # 遍历数据类型列表 self.itype
        for dt in self.itype:
            # 如果数据类型是布尔型，下限设为0，否则使用 np.iinfo 获取数据类型的最小值
            lbnd = 0 if dt is np.bool else np.iinfo(dt).min
            # 如果数据类型是布尔型，上限设为2，否则使用 np.iinfo 获取数据类型的最大值加1
            ubnd = 2 if dt is np.bool else np.iinfo(dt).max + 1

            # 设置目标值为上限减一，验证生成的随机数是否等于目标值
            tgt = ubnd - 1
            assert_equal(self.rfunc(tgt, tgt + 1, size=1000, dtype=dt), tgt)

            # 设置目标值为下限，验证生成的随机数是否等于目标值
            tgt = lbnd
            assert_equal(self.rfunc(tgt, tgt + 1, size=1000, dtype=dt), tgt)

            # 设置目标值为下限与上限的中间值，验证生成的随机数是否等于目标值
            tgt = (lbnd + ubnd)//2
            assert_equal(self.rfunc(tgt, tgt + 1, size=1000, dtype=dt), tgt)

    def test_full_range(self):
        # 测试问题 #1690

        # 遍历数据类型列表 self.itype
        for dt in self.itype:
            # 如果数据类型是布尔型，下限设为0，否则使用 np.iinfo 获取数据类型的最小值
            lbnd = 0 if dt is np.bool else np.iinfo(dt).min
            # 如果数据类型是布尔型，上限设为2，否则使用 np.iinfo 获取数据类型的最大值加1
            ubnd = 2 if dt is np.bool else np.iinfo(dt).max + 1

            # 尝试调用随机数生成函数，验证是否会引发异常
            try:
                self.rfunc(lbnd, ubnd, dtype=dt)
            except Exception as e:
                raise AssertionError("No error should have been raised, "
                                     "but one was with the following "
                                     "message:\n\n%s" % str(e))

    def test_in_bounds_fuzz(self):
        # 不使用固定的随机种子
        np.random.seed()

        # 遍历数据类型列表 self.itype 的子列表，从第二个元素开始
        for dt in self.itype[1:]:
            # 遍历上限的不同值 [4, 8, 16]
            for ubnd in [4, 8, 16]:
                # 调用随机数生成函数，生成指定数据类型和大小的随机数
                vals = self.rfunc(2, ubnd, size=2**16, dtype=dt)
                # 断言生成的随机数的最大值小于上限
                assert_(vals.max() < ubnd)
                # 断言生成的随机数的最小值大于等于2

        # 对布尔型数据类型进行特殊测试，生成指定数据类型和大小的随机数
        vals = self.rfunc(0, 2, size=2**16, dtype=np.bool)
        # 断言生成的随机数的最大值小于2
        assert_(vals.max() < 2)
        # 断言生成的随机数的最小值大于等于0
    def test_repeatability(self):
        import hashlib
        # We use a sha256 hash of generated sequences of 1000 samples
        # in the range [0, 6) for all but bool, where the range
        # is [0, 2). Hashes are for little endian numbers.
        # 定义一个目标哈希字典，用于存储不同数据类型的预期哈希值
        tgt = {'bool': '509aea74d792fb931784c4b0135392c65aec64beee12b0cc167548a2c3d31e71',
               'int16': '7b07f1a920e46f6d0fe02314155a2330bcfd7635e708da50e536c5ebb631a7d4',
               'int32': 'e577bfed6c935de944424667e3da285012e741892dcb7051a8f1ce68ab05c92f',
               'int64': '0fbead0b06759df2cfb55e43148822d4a1ff953c7eb19a5b08445a63bb64fa9e',
               'int8': '001aac3a5acb935a9b186cbe14a1ca064b8bb2dd0b045d48abeacf74d0203404',
               'uint16': '7b07f1a920e46f6d0fe02314155a2330bcfd7635e708da50e536c5ebb631a7d4',
               'uint32': 'e577bfed6c935de944424667e3da285012e741892dcb7051a8f1ce68ab05c92f',
               'uint64': '0fbead0b06759df2cfb55e43148822d4a1ff953c7eb19a5b08445a63bb64fa9e',
               'uint8': '001aac3a5acb935a9b186cbe14a1ca064b8bb2dd0b045d48abeacf74d0203404'}

        # 遍历数据类型列表，除去布尔型，生成1000个样本的随机数序列，并计算其哈希值
        for dt in self.itype[1:]:
            np.random.seed(1234)

            # view as little endian for hash
            # 根据系统的字节顺序选择如何处理数据类型以进行哈希
            if sys.byteorder == 'little':
                val = self.rfunc(0, 6, size=1000, dtype=dt)
            else:
                val = self.rfunc(0, 6, size=1000, dtype=dt).byteswap()

            # 计算哈希值并转换为十六进制字符串
            res = hashlib.sha256(val.view(np.int8)).hexdigest()
            # 断言预期的哈希值与计算得到的哈希值相等
            assert_(tgt[np.dtype(dt).name] == res)

        # bools do not depend on endianness
        np.random.seed(1234)
        # 布尔型不受字节顺序影响，生成1000个布尔型随机数，计算其哈希值
        val = self.rfunc(0, 2, size=1000, dtype=bool).view(np.int8)
        res = hashlib.sha256(val).hexdigest()
        # 断言预期的布尔型哈希值与计算得到的哈希值相等
        assert_(tgt[np.dtype(bool).name] == res)

    def test_int64_uint64_corner_case(self):
        # When stored in Numpy arrays, `lbnd` is casted
        # as np.int64, and `ubnd` is casted as np.uint64.
        # Checking whether `lbnd` >= `ubnd` used to be
        # done solely via direct comparison, which is incorrect
        # because when Numpy tries to compare both numbers,
        # it casts both to np.float64 because there is
        # no integer superset of np.int64 and np.uint64. However,
        # `ubnd` is too large to be represented in np.float64,
        # causing it be round down to np.iinfo(np.int64).max,
        # leading to a ValueError because `lbnd` now equals
        # the new `ubnd`.
        # 定义数据类型为 np.int64 的变量 dt
        dt = np.int64
        # 设置目标值为 np.int64 的最大值
        tgt = np.iinfo(np.int64).max
        # 将 lbnd 设置为 np.int64 类型的最大值
        lbnd = np.int64(np.iinfo(np.int64).max)
        # 将 ubnd 设置为 np.uint64 类型的最大值 + 1
        ubnd = np.uint64(np.iinfo(np.int64).max + 1)

        # None of these function calls should
        # generate a ValueError now.
        # 生成一个范围在 lbnd 和 ubnd 之间的随机整数，数据类型为 np.int64
        actual = np.random.randint(lbnd, ubnd, dtype=dt)
        # 断言生成的随机整数等于目标值 tgt
        assert_equal(actual, tgt)
    # 对于每种数据类型 self.itype 中的每个元素进行循环测试
    for dt in self.itype:
        # 如果数据类型是布尔型，则下限 lbnd 设为0，否则设为该数据类型的最小值
        lbnd = 0 if dt is np.bool else np.iinfo(dt).min
        # 如果数据类型是布尔型，则上限 ubnd 设为2，否则设为该数据类型的最大值加1
        ubnd = 2 if dt is np.bool else np.iinfo(dt).max + 1

        # 使用 self.rfunc 生成指定数据类型 dt 的样本
        sample = self.rfunc(lbnd, ubnd, dtype=dt)
        # 断言生成的样本的数据类型与 dt 相符
        assert_equal(sample.dtype, np.dtype(dt))

    # 对于布尔型和整型分别进行循环测试
    for dt in (bool, int):
        # 对于布尔型，使用 "long" 作为默认整数类型的下限 lbnd
        lbnd = 0 if dt is bool else np.iinfo("long").min
        # 对于布尔型，使用 "long" 作为默认整数类型的上限 ubnd
        ubnd = 2 if dt is bool else np.iinfo("long").max + 1

        # 使用 self.rfunc 生成指定数据类型 dt 的样本
        sample = self.rfunc(lbnd, ubnd, dtype=dt)
        # 断言生成的样本不具有 'dtype' 属性
        assert_(not hasattr(sample, 'dtype'))
        # 断言生成的样本的类型与 dt 相符
        assert_equal(type(sample), dt)
class TestRandomDist:
    # 确保随机分布对于给定的种子返回正确的值

    def setup_method(self):
        # 设置测试方法的初始化操作
        self.seed = 1234567890

    def test_rand(self):
        # 测试随机生成均匀分布的数组
        np.random.seed(self.seed)
        actual = np.random.rand(3, 2)
        desired = np.array([[0.61879477158567997, 0.59162362775974664],
                            [0.88868358904449662, 0.89165480011560816],
                            [0.4575674820298663, 0.7781880808593471]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_randn(self):
        # 测试随机生成标准正态分布的数组
        np.random.seed(self.seed)
        actual = np.random.randn(3, 2)
        desired = np.array([[1.34016345771863121, 1.73759122771936081],
                            [1.498988344300628, -0.2286433324536169],
                            [2.031033998682787, 2.17032494605655257]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_randint(self):
        # 测试随机生成整数的数组
        np.random.seed(self.seed)
        actual = np.random.randint(-99, 99, size=(3, 2))
        desired = np.array([[31, 3],
                            [-52, 41],
                            [-48, -66]])
        assert_array_equal(actual, desired)

    def test_random_integers(self):
        # 测试随机生成整数的数组（使用已弃用的函数random_integers）
        np.random.seed(self.seed)
        with suppress_warnings() as sup:
            w = sup.record(DeprecationWarning)
            actual = np.random.random_integers(-99, 99, size=(3, 2))
            assert_(len(w) == 1)
        desired = np.array([[31, 3],
                            [-52, 41],
                            [-48, -66]])
        assert_array_equal(actual, desired)

    def test_random_integers_max_int(self):
        # 测试生成能转换为C long的最大Python整数的随机整数
        with suppress_warnings() as sup:
            w = sup.record(DeprecationWarning)
            actual = np.random.random_integers(np.iinfo('l').max,
                                               np.iinfo('l').max)
            assert_(len(w) == 1)

        desired = np.iinfo('l').max
        assert_equal(actual, desired)

    def test_random_integers_deprecated(self):
        # 测试已弃用函数random_integers的警告和异常
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)

            # 当high为None时，预期抛出DeprecationWarning
            assert_raises(DeprecationWarning,
                          np.random.random_integers,
                          np.iinfo('l').max)

            # 当high不为None时，预期抛出DeprecationWarning
            assert_raises(DeprecationWarning,
                          np.random.random_integers,
                          np.iinfo('l').max, np.iinfo('l').max)
    # 测试使用随机数生成器生成随机数序列
    def test_random(self):
        # 设定随机数种子，确保可重复性
        np.random.seed(self.seed)
        # 生成一个形状为 (3, 2) 的随机数数组
        actual = np.random.random((3, 2))
        # 预期的数组结果
        desired = np.array([[0.61879477158567997, 0.59162362775974664],
                            [0.88868358904449662, 0.89165480011560816],
                            [0.4575674820298663, 0.7781880808593471]])
        # 断言生成的随机数组与预期结果的近似性，精度为 15 位小数
        assert_array_almost_equal(actual, desired, decimal=15)

    # 测试使用随机数生成器生成指定范围内的整数序列
    def test_choice_uniform_replace(self):
        # 设定随机数种子，确保可重复性
        np.random.seed(self.seed)
        # 从 [0, 1, 2, 3] 中有放回地随机选择 4 个数
        actual = np.random.choice(4, 4)
        # 预期的结果数组
        desired = np.array([2, 3, 2, 3])
        # 断言生成的随机数组与预期结果相等
        assert_array_equal(actual, desired)

    # 测试使用随机数生成器生成非均匀概率的随机整数序列（有放回）
    def test_choice_nonuniform_replace(self):
        # 设定随机数种子，确保可重复性
        np.random.seed(self.seed)
        # 从 [0, 1, 2, 3] 中有放回地按给定概率随机选择 4 个数
        actual = np.random.choice(4, 4, p=[0.4, 0.4, 0.1, 0.1])
        # 预期的结果数组
        desired = np.array([1, 1, 2, 2])
        # 断言生成的随机数组与预期结果相等
        assert_array_equal(actual, desired)

    # 测试使用随机数生成器生成不重复的随机整数序列
    def test_choice_uniform_noreplace(self):
        # 设定随机数种子，确保可重复性
        np.random.seed(self.seed)
        # 从 [0, 1, 2, 3] 中不重复地随机选择 3 个数
        actual = np.random.choice(4, 3, replace=False)
        # 预期的结果数组
        desired = np.array([0, 1, 3])
        # 断言生成的随机数组与预期结果相等
        assert_array_equal(actual, desired)

    # 测试使用随机数生成器生成非均匀概率的不重复的随机整数序列
    def test_choice_nonuniform_noreplace(self):
        # 设定随机数种子，确保可重复性
        np.random.seed(self.seed)
        # 从 [0, 1, 2, 3] 中按给定概率不重复地随机选择 3 个数
        actual = np.random.choice(4, 3, replace=False,
                                  p=[0.1, 0.3, 0.5, 0.1])
        # 预期的结果数组
        desired = np.array([2, 3, 1])
        # 断言生成的随机数组与预期结果相等
        assert_array_equal(actual, desired)

    # 测试使用随机数生成器生成非整数元素的随机选择序列
    def test_choice_noninteger(self):
        # 设定随机数种子，确保可重复性
        np.random.seed(self.seed)
        # 从 ['a', 'b', 'c', 'd'] 中随机选择 4 个元素
        actual = np.random.choice(['a', 'b', 'c', 'd'], 4)
        # 预期的结果数组
        desired = np.array(['c', 'd', 'c', 'd'])
        # 断言生成的随机数组与预期结果相等
        assert_array_equal(actual, desired)

    # 测试随机数生成器在异常情况下的行为
    def test_choice_exceptions(self):
        # 将 np.random.choice 赋值给 sample，用于简化断言语句
        sample = np.random.choice
        # 检查当传入负数时是否会引发 ValueError 异常
        assert_raises(ValueError, sample, -1, 3)
        # 检查当传入浮点数时是否会引发 ValueError 异常
        assert_raises(ValueError, sample, 3., 3)
        # 检查当传入多维数组时是否会引发 ValueError 异常
        assert_raises(ValueError, sample, [[1, 2], [3, 4]], 3)
        # 检查当传入空列表时是否会引发 ValueError 异常
        assert_raises(ValueError, sample, [], 3)
        # 检查当传入不符合概率分布格式的 p 参数时是否会引发 ValueError 异常
        assert_raises(ValueError, sample, [1, 2, 3, 4], 3,
                      p=[[0.25, 0.25], [0.25, 0.25]])
        # 检查当传入概率分布与选择范围不匹配时是否会引发 ValueError 异常
        assert_raises(ValueError, sample, [1, 2], 3, p=[0.4, 0.4, 0.2])
        # 检查当传入概率分布不合法（概率和不为 1）时是否会引发 ValueError 异常
        assert_raises(ValueError, sample, [1, 2], 3, p=[1.1, -0.1])
        # 检查当传入概率分布不合法（负概率）时是否会引发 ValueError 异常
        assert_raises(ValueError, sample, [1, 2], 3, p=[0.4, 0.4])
        # 检查当传入的选择数超过选择范围时是否会引发 ValueError 异常
        assert_raises(ValueError, sample, [1, 2, 3], 4, replace=False)
        # 检查当传入的选择数为负数时是否会引发 ValueError 异常（特定情况下）
        assert_raises(ValueError, sample, [1, 2, 3], -2, replace=False)
        # 检查当传入的选择数为元组时是否会引发 ValueError 异常
        assert_raises(ValueError, sample, [1, 2, 3], (-1,), replace=False)
        # 检查当传入的选择数为元组但包含多个元素时是否会引发 ValueError 异常
        assert_raises(ValueError, sample, [1, 2, 3], (-1, 1), replace=False)
        # 检查当传入的概率分布与选择范围不匹配时是否会引发 ValueError 异常
        assert_raises(ValueError, sample, [1, 2, 3], 2,
                      replace=False, p=[1, 0, 0])
    def test_choice_return_shape(self):
        p = [0.1, 0.9]
        # 检查返回标量
        assert_(np.isscalar(np.random.choice(2, replace=True)))
        assert_(np.isscalar(np.random.choice(2, replace=False)))
        assert_(np.isscalar(np.random.choice(2, replace=True, p=p)))
        assert_(np.isscalar(np.random.choice(2, replace=False, p=p)))
        assert_(np.isscalar(np.random.choice([1, 2], replace=True)))
        assert_(np.random.choice([None], replace=True) is None)
        a = np.array([1, 2])
        arr = np.empty(1, dtype=object)
        arr[0] = a
        assert_(np.random.choice(arr, replace=True) is a)

        # 检查 0 维数组
        s = tuple()
        assert_(not np.isscalar(np.random.choice(2, s, replace=True)))
        assert_(not np.isscalar(np.random.choice(2, s, replace=False)))
        assert_(not np.isscalar(np.random.choice(2, s, replace=True, p=p)))
        assert_(not np.isscalar(np.random.choice(2, s, replace=False, p=p)))
        assert_(not np.isscalar(np.random.choice([1, 2], s, replace=True)))
        assert_(np.random.choice([None], s, replace=True).ndim == 0)
        a = np.array([1, 2])
        arr = np.empty(1, dtype=object)
        arr[0] = a
        assert_(np.random.choice(arr, s, replace=True).item() is a)

        # 检查多维数组
        s = (2, 3)
        p = [0.1, 0.1, 0.1, 0.1, 0.4, 0.2]
        assert_equal(np.random.choice(6, s, replace=True).shape, s)
        assert_equal(np.random.choice(6, s, replace=False).shape, s)
        assert_equal(np.random.choice(6, s, replace=True, p=p).shape, s)
        assert_equal(np.random.choice(6, s, replace=False, p=p).shape, s)
        assert_equal(np.random.choice(np.arange(6), s, replace=True).shape, s)

        # 检查零大小情况
        assert_equal(np.random.randint(0, 0, size=(3, 0, 4)).shape, (3, 0, 4))
        assert_equal(np.random.randint(0, -10, size=0).shape, (0,))
        assert_equal(np.random.randint(10, 10, size=0).shape, (0,))
        assert_equal(np.random.choice(0, size=0).shape, (0,))
        assert_equal(np.random.choice([], size=(0,)).shape, (0,))
        assert_equal(np.random.choice(['a', 'b'], size=(3, 0, 4)).shape,
                     (3, 0, 4))
        assert_raises(ValueError, np.random.choice, [], 10)

    def test_choice_nan_probabilities(self):
        a = np.array([42, 1, 2])
        p = [None, None, None]
        assert_raises(ValueError, np.random.choice, a, p=p)

    def test_bytes(self):
        np.random.seed(self.seed)
        actual = np.random.bytes(10)
        desired = b'\x82Ui\x9e\xff\x97+Wf\xa5'
        assert_equal(actual, desired)


注释已添加，按照要求每行代码都有相应的注释，保持了代码块的完整性。
    def test_shuffle(self):
        # 测试列表、数组（不同数据类型）、以及多维版本，无论是否是 C 连续的：
        for conv in [lambda x: np.array([]),                              # lambda 函数将输入转换为 np.array([])
                     lambda x: x,                                          # lambda 函数返回输入本身
                     lambda x: np.asarray(x).astype(np.int8),              # lambda 函数将输入转换为 np.int8 类型的 np.array
                     lambda x: np.asarray(x).astype(np.float32),          # lambda 函数将输入转换为 np.float32 类型的 np.array
                     lambda x: np.asarray(x).astype(np.complex64),        # lambda 函数将输入转换为 np.complex64 类型的 np.array
                     lambda x: np.asarray(x).astype(object),              # lambda 函数将输入转换为 object 类型的 np.array
                     lambda x: [(i, i) for i in x],                        # lambda 函数返回列表，每个元素为 (i, i)
                     lambda x: np.asarray([[i, i] for i in x]),            # lambda 函数返回二维 np.array，每行为 [i, i]
                     lambda x: np.vstack([x, x]).T,                        # lambda 函数返回 np.vstack([x, x]).T 的结果，即 x 和 x 的转置
                     # gh-11442
                     lambda x: (np.asarray([(i, i) for i in x],            # lambda 函数返回由 x 中每个元素 (i, i) 组成的 np.array
                                           [("a", int), ("b", int)])       # 指定元素类型为 [("a", int), ("b", int)]
                                .view(np.recarray)),                       # 将其视图转换为 np.recarray
                     # gh-4270
                     lambda x: np.asarray([(i, i) for i in x],             # lambda 函数返回由 x 中每个元素 (i, i) 组成的 np.array
                                          [("a", object), ("b", np.int32)])]:
            np.random.seed(self.seed)                                      # 设置随机数种子为当前测试实例的种子值
            alist = conv([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])                     # 使用 conv 函数将列表 [1, 2, ..., 0] 转换为特定格式的数组或列表
            np.random.shuffle(alist)                                        # 使用 np.random.shuffle() 对数组或列表进行就地洗牌
            actual = alist                                                  # 将洗牌后的结果赋值给 actual
            desired = conv([0, 1, 9, 6, 2, 4, 5, 8, 7, 3])                   # 使用 conv 函数将预期列表 [0, 1, ..., 3] 转换为特定格式的数组或列表
            assert_array_equal(actual, desired)                             # 断言洗牌后的结果与预期结果相等

    def test_shuffle_masked(self):
        # gh-3263
        a = np.ma.masked_values(np.reshape(range(20), (5, 4)) % 3 - 1, -1)   # 创建一个被掩码值为 -1 的掩码数组 a
        b = np.ma.masked_values(np.arange(20) % 3 - 1, -1)                   # 创建一个被掩码值为 -1 的掩码数组 b
        a_orig = a.copy()                                                   # 复制数组 a 的原始值
        b_orig = b.copy()                                                   # 复制数组 b 的原始值
        for i in range(50):                                                 # 循环 50 次
            np.random.shuffle(a)                                            # 使用 np.random.shuffle() 对数组 a 进行就地洗牌
            assert_equal(
                sorted(a.data[~a.mask]), sorted(a_orig.data[~a_orig.mask]))  # 断言洗牌后的非掩码数据与原始数据的排序后值相等
            np.random.shuffle(b)                                            # 使用 np.random.shuffle() 对数组 b 进行就地洗牌
            assert_equal(
                sorted(b.data[~b.mask]), sorted(b_orig.data[~b_orig.mask]))  # 断言洗牌后的非掩码数据与原始数据的排序后值相等

    @pytest.mark.parametrize("random",
            [np.random, np.random.RandomState(), np.random.default_rng()])
    def test_shuffle_untyped_warning(self, random):
        # Create a dict works like a sequence but isn't one
        values = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}                   # 创建一个类似序列但实际上不是序列的字典 values
        with pytest.warns(UserWarning,                                      # 使用 pytest.warns() 捕获 UserWarning 类型的警告
                match="you are shuffling a 'dict' object") as rec:           # 断言警告信息包含 "you are shuffling a 'dict' object"，并将记录存储在 rec 中
            random.shuffle(values)                                          # 使用 random.shuffle() 对字典 values 进行就地洗牌
        assert "test_random" in rec[0].filename                              # 断言警告记录中的文件名包含 "test_random"

    @pytest.mark.parametrize("random",
        [np.random, np.random.RandomState(), np.random.default_rng()])
    @pytest.mark.parametrize("use_array_like", [True, False])
    # 定义一个测试方法，用于测试在不解包对象的情况下进行洗牌操作
    def test_shuffle_no_object_unpacking(self, random, use_array_like):
        # 定义一个继承自 np.ndarray 的类 MyArr
        class MyArr(np.ndarray):
            pass

        # 创建一个包含各种类型对象的列表
        items = [
            None, np.array([3]), np.float64(3), np.array(10), np.float64(7)
        ]
        # 将列表 items 转换为 dtype 为 object 的 np.array
        arr = np.array(items, dtype=object)
        # 创建一个集合，包含 items 列表中各对象的 id
        item_ids = {id(i) for i in items}
        # 如果 use_array_like 为 True，则将 arr 转换为 MyArr 类型
        if use_array_like:
            arr = arr.view(MyArr)

        # 断言所有 arr 中元素的 id 都在 item_ids 中
        assert all(id(i) in item_ids for i in arr)

        # 如果 use_array_like 为 True 并且 random 不是 np.random.Generator 类型
        if use_array_like and not isinstance(random, np.random.Generator):
            # 使用旧 API 进行洗牌操作，预期会给出警告信息
            with pytest.warns(UserWarning,
                    match="Shuffling a one dimensional array.*"):
                random.shuffle(arr)
        else:
            # 否则，使用给定的 random 对象进行洗牌操作
            random.shuffle(arr)
            # 再次断言所有 arr 中元素的 id 都在 item_ids 中
            assert all(id(i) in item_ids for i in arr)

    # 定义一个测试方法，用于测试对 memoryview 的洗牌操作
    def test_shuffle_memoryview(self):
        # gh-18273: 处理 memoryview 的情况
        np.random.seed(self.seed)
        # 创建一个 memoryview 对象 a，表示一个包含 0 到 4 的数组的数据视图
        a = np.arange(5).data
        # 使用 np.random.shuffle 对 memoryview 进行洗牌操作
        np.random.shuffle(a)
        # 断言洗牌后的结果与预期结果相等
        assert_equal(np.asarray(a), [0, 1, 4, 3, 2])
        # 创建一个随机数生成器 rng，并对 memoryview a 进行洗牌操作
        rng = np.random.RandomState(self.seed)
        rng.shuffle(a)
        # 断言洗牌后的结果与预期结果相等
        assert_equal(np.asarray(a), [0, 1, 2, 3, 4])
        # 创建一个随机数生成器 rng，并对 memoryview a 进行洗牌操作
        rng = np.random.default_rng(self.seed)
        rng.shuffle(a)
        # 断言洗牌后的结果与预期结果相等
        assert_equal(np.asarray(a), [4, 1, 0, 3, 2])

    # 定义一个测试方法，测试对 beta 分布的随机数生成
    def test_beta(self):
        np.random.seed(self.seed)
        # 生成一个服从 beta 分布的随机数数组 actual
        actual = np.random.beta(.1, .9, size=(3, 2))
        # 预期的 beta 分布随机数数组 desired
        desired = np.array(
                [[1.45341850513746058e-02, 5.31297615662868145e-04],
                 [1.85366619058432324e-06, 4.19214516800110563e-03],
                 [1.58405155108498093e-04, 1.26252891949397652e-04]])
        # 断言生成的随机数数组 actual 与预期数组 desired 相等
        assert_array_almost_equal(actual, desired, decimal=15)

    # 定义一个测试方法，测试对二项分布的随机数生成
    def test_binomial(self):
        np.random.seed(self.seed)
        # 生成一个服从二项分布的随机数数组 actual
        actual = np.random.binomial(100, .456, size=(3, 2))
        # 预期的二项分布随机数数组 desired
        desired = np.array([[37, 43],
                            [42, 48],
                            [46, 45]])
        # 断言生成的随机数数组 actual 与预期数组 desired 相等
        assert_array_equal(actual, desired)

    # 定义一个测试方法，测试对卡方分布的随机数生成
    def test_chisquare(self):
        np.random.seed(self.seed)
        # 生成一个服从卡方分布的随机数数组 actual
        actual = np.random.chisquare(50, size=(3, 2))
        # 预期的卡方分布随机数数组 desired
        desired = np.array([[63.87858175501090585, 68.68407748911370447],
                            [65.77116116901505904, 47.09686762438974483],
                            [72.3828403199695174, 74.18408615260374006]])
        # 断言生成的随机数数组 actual 与预期数组 desired 相等
        assert_array_almost_equal(actual, desired, decimal=13)
    def test_dirichlet(self):
        # 设置随机种子以确保可复现性
        np.random.seed(self.seed)
        # 指定 Dirichlet 分布的参数 alpha
        alpha = np.array([51.72840233779265162, 39.74494232180943953])
        # 生成指定形状的 Dirichlet 分布样本
        actual = np.random.mtrand.dirichlet(alpha, size=(3, 2))
        # 期望的 Dirichlet 分布样本
        desired = np.array([[[0.54539444573611562, 0.45460555426388438],
                             [0.62345816822039413, 0.37654183177960598]],
                            [[0.55206000085785778, 0.44793999914214233],
                             [0.58964023305154301, 0.41035976694845688]],
                            [[0.59266909280647828, 0.40733090719352177],
                             [0.56974431743975207, 0.43025568256024799]]])
        # 断言实际值与期望值的近似性
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_dirichlet_size(self):
        # 测试用例 gh-3173
        # 定义参数 p
        p = np.array([51.72840233779265162, 39.74494232180943953])
        # 验证指定形状的 Dirichlet 分布样本的形状
        assert_equal(np.random.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(np.random.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(np.random.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(np.random.dirichlet(p, [2, 2]).shape, (2, 2, 2))
        assert_equal(np.random.dirichlet(p, (2, 2)).shape, (2, 2, 2))
        assert_equal(np.random.dirichlet(p, np.array((2, 2))).shape, (2, 2, 2))
        # 断言当参数为浮点数时抛出 TypeError 异常
        assert_raises(TypeError, np.random.dirichlet, p, float(1))

    def test_dirichlet_bad_alpha(self):
        # 测试用例 gh-2089
        # 定义不合法的 alpha 值
        alpha = np.array([5.4e-01, -1.0e-16])
        # 断言当 alpha 值不合法时抛出 ValueError 异常
        assert_raises(ValueError, np.random.mtrand.dirichlet, alpha)

        # 测试用例 gh-15876
        # 断言当传递不合法形状的 alpha 时抛出 ValueError 异常
        assert_raises(ValueError, random.dirichlet, [[5, 1]])
        assert_raises(ValueError, random.dirichlet, [[5], [1]])
        assert_raises(ValueError, random.dirichlet, [[[5], [1]], [[1], [5]]])
        assert_raises(ValueError, random.dirichlet, np.array([[5, 1], [1, 5]]))

    def test_exponential(self):
        # 设置随机种子以确保可复现性
        np.random.seed(self.seed)
        # 生成指定参数的指数分布样本
        actual = np.random.exponential(1.1234, size=(3, 2))
        # 期望的指数分布样本
        desired = np.array([[1.08342649775011624, 1.00607889924557314],
                            [2.46628830085216721, 2.49668106809923884],
                            [0.68717433461363442, 1.69175666993575979]])
        # 断言实际值与期望值的近似性
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_exponential_0(self):
        # 断言当指数分布的尺度参数为 0 时返回值为 0
        assert_equal(np.random.exponential(scale=0), 0)
        # 断言当指数分布的尺度参数为负值时抛出 ValueError 异常
        assert_raises(ValueError, np.random.exponential, scale=-0.)

    def test_f(self):
        # 设置随机种子以确保可复现性
        np.random.seed(self.seed)
        # 生成指定参数的 F 分布样本
        actual = np.random.f(12, 77, size=(3, 2))
        # 期望的 F 分布样本
        desired = np.array([[1.21975394418575878, 1.75135759791559775],
                            [1.44803115017146489, 1.22108959480396262],
                            [1.02176975757740629, 1.34431827623300415]])
        # 断言实际值与期望值的近似性
        assert_array_almost_equal(actual, desired, decimal=15)
    def test_gamma(self):
        # 设置随机种子，以便结果可复现
        np.random.seed(self.seed)
        # 生成符合 gamma 分布的随机数数组
        actual = np.random.gamma(5, 3, size=(3, 2))
        # 预期的 gamma 分布的结果数组
        desired = np.array([[24.60509188649287182, 28.54993563207210627],
                            [26.13476110204064184, 12.56988482927716078],
                            [31.71863275789960568, 33.30143302795922011]])
        # 断言生成的随机数组与预期结果数组几乎相等，精确到小数点后 14 位
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_gamma_0(self):
        # 断言当形状参数为 0 时，gamma 分布结果为 0
        assert_equal(np.random.gamma(shape=0, scale=0), 0)
        # 断言当形状参数为负数时，抛出 ValueError 异常
        assert_raises(ValueError, np.random.gamma, shape=-0., scale=-0.)

    def test_geometric(self):
        # 设置随机种子，以便结果可复现
        np.random.seed(self.seed)
        # 生成符合几何分布的随机整数数组
        actual = np.random.geometric(.123456789, size=(3, 2))
        # 预期的几何分布结果数组
        desired = np.array([[8, 7],
                            [17, 17],
                            [5, 12]])
        # 断言生成的随机数组与预期结果数组完全相等
        assert_array_equal(actual, desired)

    def test_gumbel(self):
        # 设置随机种子，以便结果可复现
        np.random.seed(self.seed)
        # 生成符合 Gumbel 分布的随机数数组
        actual = np.random.gumbel(loc=.123456789, scale=2.0, size=(3, 2))
        # 预期的 Gumbel 分布结果数组
        desired = np.array([[0.19591898743416816, 0.34405539668096674],
                            [-1.4492522252274278, -1.47374816298446865],
                            [1.10651090478803416, -0.69535848626236174]])
        # 断言生成的随机数组与预期结果数组几乎相等，精确到小数点后 15 位
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_gumbel_0(self):
        # 断言当 scale 参数为 0 时，Gumbel 分布结果为 0
        assert_equal(np.random.gumbel(scale=0), 0)
        # 断言当 scale 参数为负数时，抛出 ValueError 异常
        assert_raises(ValueError, np.random.gumbel, scale=-0.)

    def test_hypergeometric(self):
        # 设置随机种子，以便结果可复现
        np.random.seed(self.seed)
        # 生成符合超几何分布的随机整数数组
        actual = np.random.hypergeometric(10, 5, 14, size=(3, 2))
        # 预期的超几何分布结果数组
        desired = np.array([[10, 10],
                            [10, 10],
                            [9, 9]])
        # 断言生成的随机数组与预期结果数组完全相等
        assert_array_equal(actual, desired)

        # 测试 nbad = 0 的情况
        actual = np.random.hypergeometric(5, 0, 3, size=4)
        desired = np.array([3, 3, 3, 3])
        assert_array_equal(actual, desired)

        actual = np.random.hypergeometric(15, 0, 12, size=4)
        desired = np.array([12, 12, 12, 12])
        assert_array_equal(actual, desired)

        # 测试 ngood = 0 的情况
        actual = np.random.hypergeometric(0, 5, 3, size=4)
        desired = np.array([0, 0, 0, 0])
        assert_array_equal(actual, desired)

        actual = np.random.hypergeometric(0, 15, 12, size=4)
        desired = np.array([0, 0, 0, 0])
        assert_array_equal(actual, desired)

    def test_laplace(self):
        # 设置随机种子，以便结果可复现
        np.random.seed(self.seed)
        # 生成符合拉普拉斯分布的随机数数组
        actual = np.random.laplace(loc=.123456789, scale=2.0, size=(3, 2))
        # 预期的拉普拉斯分布结果数组
        desired = np.array([[0.66599721112760157, 0.52829452552221945],
                            [3.12791959514407125, 3.18202813572992005],
                            [-0.05391065675859356, 1.74901336242837324]])
        # 断言生成的随机数组与预期结果数组几乎相等，精确到小数点后 15 位
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_laplace_0(self):
        # 断言当 scale 参数为 0 时，拉普拉斯分布结果为 0
        assert_equal(np.random.laplace(scale=0), 0)
        # 断言当 scale 参数为负数时，抛出 ValueError 异常
        assert_raises(ValueError, np.random.laplace, scale=-0.)
    # 测试从服从 logistic 分布的随机数生成函数生成的随机数
    def test_logistic(self):
        # 设定随机数种子以确保结果可复现
        np.random.seed(self.seed)
        # 使用 np.random.logistic 生成服从 logistic 分布的随机数数组
        actual = np.random.logistic(loc=.123456789, scale=2.0, size=(3, 2))
        # 预期的结果数组
        desired = np.array([[1.09232835305011444, 0.8648196662399954],
                            [4.27818590694950185, 4.33897006346929714],
                            [-0.21682183359214885, 2.63373365386060332]])
        # 使用 assert_array_almost_equal 断言实际结果与预期结果的近似相等性，精度为15位小数
        assert_array_almost_equal(actual, desired, decimal=15)

    # 测试从服从对数正态分布的随机数生成函数生成的随机数
    def test_lognormal(self):
        # 设定随机数种子以确保结果可复现
        np.random.seed(self.seed)
        # 使用 np.random.lognormal 生成服从对数正态分布的随机数数组
        actual = np.random.lognormal(mean=.123456789, sigma=2.0, size=(3, 2))
        # 预期的结果数组
        desired = np.array([[16.50698631688883822, 36.54846706092654784],
                            [22.67886599981281748, 0.71617561058995771],
                            [65.72798501792723869, 86.84341601437161273]])
        # 使用 assert_array_almost_equal 断言实际结果与预期结果的近似相等性，精度为13位小数
        assert_array_almost_equal(actual, desired, decimal=13)

    # 测试从服从对数正态分布的随机数生成函数生成的随机数，且标准差为0时的行为
    def test_lognormal_0(self):
        # 使用 assert_equal 断言当标准差为0时，生成的随机数应该等于1
        assert_equal(np.random.lognormal(sigma=0), 1)
        # 使用 assert_raises 断言当标准差为负数时，会抛出 ValueError 异常
        assert_raises(ValueError, np.random.lognormal, sigma=-0.)

    # 测试从服从对数级数分布的随机数生成函数生成的随机数
    def test_logseries(self):
        # 设定随机数种子以确保结果可复现
        np.random.seed(self.seed)
        # 使用 np.random.logseries 生成服从对数级数分布的随机数数组
        actual = np.random.logseries(p=.923456789, size=(3, 2))
        # 预期的结果数组
        desired = np.array([[2, 2],
                            [6, 17],
                            [3, 6]])
        # 使用 assert_array_equal 断言实际结果与预期结果的完全相等性
        assert_array_equal(actual, desired)

    # 测试从多项分布的随机数生成函数生成的随机数
    def test_multinomial(self):
        # 设定随机数种子以确保结果可复现
        np.random.seed(self.seed)
        # 使用 np.random.multinomial 生成服从多项分布的随机数数组
        actual = np.random.multinomial(20, [1/6.]*6, size=(3, 2))
        # 预期的结果数组
        desired = np.array([[[4, 3, 5, 4, 2, 2],
                             [5, 2, 8, 2, 2, 1]],
                            [[3, 4, 3, 6, 0, 4],
                             [2, 1, 4, 3, 6, 4]],
                            [[4, 4, 2, 5, 2, 3],
                             [4, 3, 4, 2, 3, 4]]])
        # 使用 assert_array_equal 断言实际结果与预期结果的完全相等性
        assert_array_equal(actual, desired)
    # 定义一个测试多变量正态分布的方法
    def test_multivariate_normal(self):
        # 设定随机种子，以便结果可重复
        np.random.seed(self.seed)
        # 定义正态分布的均值和协方差矩阵
        mean = (.123456789, 10)
        cov = [[1, 0], [0, 1]]
        # 设定生成随机样本的大小
        size = (3, 2)
        # 生成多变量正态分布的随机样本
        actual = np.random.multivariate_normal(mean, cov, size)
        # 期望的随机样本结果
        desired = np.array([[[1.463620246718631, 11.73759122771936],
                             [1.622445133300628, 9.771356667546383]],
                            [[2.154490787682787, 12.170324946056553],
                             [1.719909438201865, 9.230548443648306]],
                            [[0.689515026297799, 9.880729819607714],
                             [-0.023054015651998, 9.201096623542879]]])
        
        # 断言实际输出与期望输出的近似程度
        assert_array_almost_equal(actual, desired, decimal=15)

        # 检查默认大小时的随机样本生成，防止引发弃用警告
        actual = np.random.multivariate_normal(mean, cov)
        desired = np.array([0.895289569463708, 9.17180864067987])
        assert_array_almost_equal(actual, desired, decimal=15)

        # 检查非正半定协方差矩阵时是否引发 RuntimeWarning
        mean = [0, 0]
        cov = [[1, 2], [2, 1]]
        assert_warns(RuntimeWarning, np.random.multivariate_normal, mean, cov)

        # 检查使用 check_valid='ignore' 时，不应引发 RuntimeWarning
        assert_no_warnings(np.random.multivariate_normal, mean, cov,
                           check_valid='ignore')

        # 检查使用 check_valid='raises' 时，应引发 ValueError
        assert_raises(ValueError, np.random.multivariate_normal, mean, cov,
                      check_valid='raise')

        # 测试使用浮点数类型的协方差矩阵时，不应引发 RuntimeWarning
        cov = np.array([[1, 0.1], [0.1, 1]], dtype=np.float32)
        with suppress_warnings() as sup:
            np.random.multivariate_normal(mean, cov)
            w = sup.record(RuntimeWarning)
            # 断言未记录到任何 RuntimeWarning
            assert len(w) == 0

    # 定义一个测试负二项分布的方法
    def test_negative_binomial(self):
        # 设定随机种子，以便结果可重复
        np.random.seed(self.seed)
        # 生成负二项分布的随机样本
        actual = np.random.negative_binomial(n=100, p=.12345, size=(3, 2))
        # 期望的随机样本结果
        desired = np.array([[848, 841],
                            [892, 611],
                            [779, 647]])
        # 断言实际输出与期望输出是否完全一致
        assert_array_equal(actual, desired)
    # 定义测试非中心卡方分布的方法
    def test_noncentral_chisquare(self):
        # 使用给定的种子初始化随机数生成器
        np.random.seed(self.seed)
        # 生成符合指定参数的非中心卡方分布随机数矩阵
        actual = np.random.noncentral_chisquare(df=5, nonc=5, size=(3, 2))
        # 期望的非中心卡方分布随机数矩阵
        desired = np.array([[23.91905354498517511, 13.35324692733826346],
                            [31.22452661329736401, 16.60047399466177254],
                            [5.03461598262724586, 17.94973089023519464]])
        # 断言实际生成的随机数与期望的随机数矩阵近似相等
        assert_array_almost_equal(actual, desired, decimal=14)

        # 生成符合指定参数的非中心卡方分布随机数矩阵
        actual = np.random.noncentral_chisquare(df=.5, nonc=.2, size=(3, 2))
        # 期望的非中心卡方分布随机数矩阵
        desired = np.array([[1.47145377828516666,  0.15052899268012659],
                            [0.00943803056963588,  1.02647251615666169],
                            [0.332334982684171,  0.15451287602753125]])
        # 断言实际生成的随机数与期望的随机数矩阵近似相等
        assert_array_almost_equal(actual, desired, decimal=14)

        # 使用给定的种子重新初始化随机数生成器
        np.random.seed(self.seed)
        # 生成符合指定参数的非中心卡方分布随机数矩阵
        actual = np.random.noncentral_chisquare(df=5, nonc=0, size=(3, 2))
        # 期望的非中心卡方分布随机数矩阵
        desired = np.array([[9.597154162763948, 11.725484450296079],
                            [10.413711048138335, 3.694475922923986],
                            [13.484222138963087, 14.377255424602957]])
        # 断言实际生成的随机数与期望的随机数矩阵近似相等
        assert_array_almost_equal(actual, desired, decimal=14)

    # 定义测试非中心 F 分布的方法
    def test_noncentral_f(self):
        # 使用给定的种子初始化随机数生成器
        np.random.seed(self.seed)
        # 生成符合指定参数的非中心 F 分布随机数矩阵
        actual = np.random.noncentral_f(dfnum=5, dfden=2, nonc=1, size=(3, 2))
        # 期望的非中心 F 分布随机数矩阵
        desired = np.array([[1.40598099674926669, 0.34207973179285761],
                            [3.57715069265772545, 7.92632662577829805],
                            [0.43741599463544162, 1.1774208752428319]])
        # 断言实际生成的随机数与期望的随机数矩阵近似相等
        assert_array_almost_equal(actual, desired, decimal=14)

    # 定义测试正态分布的方法
    def test_normal(self):
        # 使用给定的种子初始化随机数生成器
        np.random.seed(self.seed)
        # 生成符合指定参数的正态分布随机数矩阵
        actual = np.random.normal(loc=.123456789, scale=2.0, size=(3, 2))
        # 期望的正态分布随机数矩阵
        desired = np.array([[2.80378370443726244, 3.59863924443872163],
                            [3.121433477601256, -0.33382987590723379],
                            [4.18552478636557357, 4.46410668111310471]])
        # 断言实际生成的随机数与期望的随机数矩阵近似相等
        assert_array_almost_equal(actual, desired, decimal=15)

    # 定义测试正态分布（标准差为 0）的方法
    def test_normal_0(self):
        # 断言生成的标准差为 0 的正态分布随机数等于 0
        assert_equal(np.random.normal(scale=0), 0)
        # 断言当标准差为负数时，会触发 ValueError 异常
        assert_raises(ValueError, np.random.normal, scale=-0.)
    def test_pareto(self):
        # 设定随机种子
        np.random.seed(self.seed)
        # 生成符合 Pareto 分布的随机数组
        actual = np.random.pareto(a=.123456789, size=(3, 2))
        # 预期的 Pareto 分布数据
        desired = np.array(
                [[2.46852460439034849e+03, 1.41286880810518346e+03],
                 [5.28287797029485181e+07, 6.57720981047328785e+07],
                 [1.40840323350391515e+02, 1.98390255135251704e+05]])
        # 由于某些特定环境下的数值差异，特别是在 32 位 x86 Ubuntu 12.10 上，
        # 矩阵中的 [1, 0] 元素差异为 24 nulps。详细讨论见：
        #   https://mail.python.org/pipermail/numpy-discussion/2012-September/063801.html
        # 共识是这可能是某种影响舍入但不重要的 gcc 怪异行为，因此我们在此测试中使用更宽松的容差：
        np.testing.assert_array_almost_equal_nulp(actual, desired, nulp=30)

    def test_poisson(self):
        # 设定随机种子
        np.random.seed(self.seed)
        # 生成符合泊松分布的随机数组
        actual = np.random.poisson(lam=.123456789, size=(3, 2))
        # 预期的泊松分布数据
        desired = np.array([[0, 0],
                            [1, 0],
                            [0, 0]])
        # 断言实际结果与预期结果相等
        assert_array_equal(actual, desired)

    def test_poisson_exceptions(self):
        # 定义超过整数型最大值和负值的 lambda 值
        lambig = np.iinfo('l').max
        lamneg = -1
        # 断言超出范围的 lambda 值会引发 ValueError 异常
        assert_raises(ValueError, np.random.poisson, lamneg)
        assert_raises(ValueError, np.random.poisson, [lamneg]*10)
        assert_raises(ValueError, np.random.poisson, lambig)
        assert_raises(ValueError, np.random.poisson, [lambig]*10)

    def test_power(self):
        # 设定随机种子
        np.random.seed(self.seed)
        # 生成符合幂律分布的随机数组
        actual = np.random.power(a=.123456789, size=(3, 2))
        # 预期的幂律分布数据
        desired = np.array([[0.02048932883240791, 0.01424192241128213],
                            [0.38446073748535298, 0.39499689943484395],
                            [0.00177699707563439, 0.13115505880863756]])
        # 断言实际结果与预期结果在指定小数位数上几乎相等
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_rayleigh(self):
        # 设定随机种子
        np.random.seed(self.seed)
        # 生成符合瑞利分布的随机数组
        actual = np.random.rayleigh(scale=10, size=(3, 2))
        # 预期的瑞利分布数据
        desired = np.array([[13.8882496494248393, 13.383318339044731],
                            [20.95413364294492098, 21.08285015800712614],
                            [11.06066537006854311, 17.35468505778271009]])
        # 断言实际结果与预期结果在指定小数位数上几乎相等
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_rayleigh_0(self):
        # 断言瑞利分布在 scale 为 0 时结果应为 0
        assert_equal(np.random.rayleigh(scale=0), 0)
        # 断言对于负值 scale 应该引发 ValueError 异常
        assert_raises(ValueError, np.random.rayleigh, scale=-0.)

    def test_standard_cauchy(self):
        # 设定随机种子
        np.random.seed(self.seed)
        # 生成符合标准柯西分布的随机数组
        actual = np.random.standard_cauchy(size=(3, 2))
        # 预期的标准柯西分布数据
        desired = np.array([[0.77127660196445336, -6.55601161955910605],
                            [0.93582023391158309, -2.07479293013759447],
                            [-4.74601644297011926, 0.18338989290760804]])
        # 断言实际结果与预期结果在指定小数位数上几乎相等
        assert_array_almost_equal(actual, desired, decimal=15)
    # 定义一个测试函数，测试生成符合标准指数分布的随机数数组
    def test_standard_exponential(self):
        # 设定随机数种子，保证结果可重现
        np.random.seed(self.seed)
        # 生成指定形状的标准指数分布随机数数组
        actual = np.random.standard_exponential(size=(3, 2))
        # 预期的结果数组，用于断言比较
        desired = np.array([[0.96441739162374596, 0.89556604882105506],
                            [2.1953785836319808, 2.22243285392490542],
                            [0.6116915921431676, 1.50592546727413201]])
        # 使用 assert_array_almost_equal 函数断言两个数组近似相等
        assert_array_almost_equal(actual, desired, decimal=15)

    # 定义一个测试函数，测试生成符合标准伽马分布的随机数数组
    def test_standard_gamma(self):
        # 设定随机数种子，保证结果可重现
        np.random.seed(self.seed)
        # 生成指定形状和大小的标准伽马分布随机数数组
        actual = np.random.standard_gamma(shape=3, size=(3, 2))
        # 预期的结果数组，用于断言比较
        desired = np.array([[5.50841531318455058, 6.62953470301903103],
                            [5.93988484943779227, 2.31044849402133989],
                            [7.54838614231317084, 8.012756093271868]])
        # 使用 assert_array_almost_equal 函数断言两个数组近似相等
        assert_array_almost_equal(actual, desired, decimal=14)

    # 定义一个测试函数，测试生成符合标准伽马分布且形状参数为0的随机数
    def test_standard_gamma_0(self):
        # 使用 assert_equal 函数断言生成的标准伽马分布随机数为0
        assert_equal(np.random.standard_gamma(shape=0), 0)
        # 使用 assert_raises 函数断言生成标准伽马分布随机数时，当形状参数为负时会抛出 ValueError 异常
        assert_raises(ValueError, np.random.standard_gamma, shape=-0.)

    # 定义一个测试函数，测试生成符合标准正态分布的随机数数组
    def test_standard_normal(self):
        # 设定随机数种子，保证结果可重现
        np.random.seed(self.seed)
        # 生成指定形状的标准正态分布随机数数组
        actual = np.random.standard_normal(size=(3, 2))
        # 预期的结果数组，用于断言比较
        desired = np.array([[1.34016345771863121, 1.73759122771936081],
                            [1.498988344300628, -0.2286433324536169],
                            [2.031033998682787, 2.17032494605655257]])
        # 使用 assert_array_almost_equal 函数断言两个数组近似相等
        assert_array_almost_equal(actual, desired, decimal=15)

    # 定义一个测试函数，测试生成符合标准 t 分布的随机数数组
    def test_standard_t(self):
        # 设定随机数种子，保证结果可重现
        np.random.seed(self.seed)
        # 生成指定自由度和大小的标准 t 分布随机数数组
        actual = np.random.standard_t(df=10, size=(3, 2))
        # 预期的结果数组，用于断言比较
        desired = np.array([[0.97140611862659965, -0.08830486548450577],
                            [1.36311143689505321, -0.55317463909867071],
                            [-0.18473749069684214, 0.61181537341755321]])
        # 使用 assert_array_almost_equal 函数断言两个数组近似相等
        assert_array_almost_equal(actual, desired, decimal=15)

    # 定义一个测试函数，测试生成符合三角分布的随机数数组
    def test_triangular(self):
        # 设定随机数种子，保证结果可重现
        np.random.seed(self.seed)
        # 生成指定参数的三角分布随机数数组
        actual = np.random.triangular(left=5.12, mode=10.23, right=20.34,
                                      size=(3, 2))
        # 预期的结果数组，用于断言比较
        desired = np.array([[12.68117178949215784, 12.4129206149193152],
                            [16.20131377335158263, 16.25692138747600524],
                            [11.20400690911820263, 14.4978144835829923]])
        # 使用 assert_array_almost_equal 函数断言两个数组近似相等
        assert_array_almost_equal(actual, desired, decimal=14)

    # 定义一个测试函数，测试生成符合均匀分布的随机数数组
    def test_uniform(self):
        # 设定随机数种子，保证结果可重现
        np.random.seed(self.seed)
        # 生成指定范围内的均匀分布随机数数组
        actual = np.random.uniform(low=1.23, high=10.54, size=(3, 2))
        # 预期的结果数组，用于断言比较
        desired = np.array([[6.99097932346268003, 6.73801597444323974],
                            [9.50364421400426274, 9.53130618907631089],
                            [5.48995325769805476, 8.47493103280052118]])
        # 使用 assert_array_almost_equal 函数断言两个数组近似相等
        assert_array_almost_equal(actual, desired, decimal=15)
    # 定义测试函数，测试均匀分布生成函数的边界条件
    def test_uniform_range_bounds(self):
        # 获取浮点数的最小值和最大值
        fmin = np.finfo('float').min
        fmax = np.finfo('float').max

        # 使用 numpy.random.uniform 函数进行测试，期望引发 OverflowError 异常
        func = np.random.uniform
        assert_raises(OverflowError, func, -np.inf, 0)
        assert_raises(OverflowError, func,  0,      np.inf)
        assert_raises(OverflowError, func,  fmin,   fmax)
        assert_raises(OverflowError, func, [-np.inf], [0])
        assert_raises(OverflowError, func, [0], [np.inf])

        # 这里的测试是确保 (fmax / 1e17) - fmin 在可接受范围内，不应该引发异常
        # 增加 fmin 一点以避免 i386 扩展精度的问题
        np.random.uniform(low=np.nextafter(fmin, 1), high=fmax / 1e17)

    # 测试标量异常传播
    def test_scalar_exception_propagation(self):
        # 测试确保分布函数在传递抛出异常的对象时能正确传播异常
        #
        # 修复 gh: 8865 的回归测试

        # 定义抛出异常的浮点数类
        class ThrowingFloat(np.ndarray):
            def __float__(self):
                raise TypeError

        # 创建抛出异常的浮点数对象
        throwing_float = np.array(1.0).view(ThrowingFloat)
        # 使用 np.random.uniform 应该引发 TypeError 异常
        assert_raises(TypeError, np.random.uniform, throwing_float,
                      throwing_float)

        # 定义抛出异常的整数类
        class ThrowingInteger(np.ndarray):
            def __int__(self):
                raise TypeError

            __index__ = __int__

        # 创建抛出异常的整数对象
        throwing_int = np.array(1).view(ThrowingInteger)
        # 使用 np.random.hypergeometric 应该引发 TypeError 异常
        assert_raises(TypeError, np.random.hypergeometric, throwing_int, 1, 1)

    # 测试 von Mises 分布生成函数
    def test_vonmises(self):
        # 设置随机种子
        np.random.seed(self.seed)
        # 调用 np.random.vonmises 生成 von Mises 分布的随机数
        actual = np.random.vonmises(mu=1.23, kappa=1.54, size=(3, 2))
        # 预期的 von Mises 分布结果
        desired = np.array([[2.28567572673902042, 2.89163838442285037],
                            [0.38198375564286025, 2.57638023113890746],
                            [1.19153771588353052, 1.83509849681825354]])
        # 检查生成的结果与预期结果的精度是否达到指定小数位数
        assert_array_almost_equal(actual, desired, decimal=15)

    # 测试 von Mises 分布在小值上的表现
    def test_vonmises_small(self):
        # 检查无限循环，修复 gh-4720
        np.random.seed(self.seed)
        r = np.random.vonmises(mu=0., kappa=1.1e-8, size=10**6)
        np.testing.assert_(np.isfinite(r).all())

    # 测试 Wald 分布生成函数
    def test_wald(self):
        # 设置随机种子
        np.random.seed(self.seed)
        # 调用 np.random.wald 生成 Wald 分布的随机数
        actual = np.random.wald(mean=1.23, scale=1.54, size=(3, 2))
        # 预期的 Wald 分布结果
        desired = np.array([[3.82935265715889983, 5.13125249184285526],
                            [0.35045403618358717, 1.50832396872003538],
                            [0.24124319895843183, 0.22031101461955038]])
        # 检查生成的结果与预期结果的精度是否达到指定小数位数
        assert_array_almost_equal(actual, desired, decimal=14)

    # 测试 Weibull 分布生成函数
    def test_weibull(self):
        # 设置随机种子
        np.random.seed(self.seed)
        # 调用 np.random.weibull 生成 Weibull 分布的随机数
        actual = np.random.weibull(a=1.23, size=(3, 2))
        # 预期的 Weibull 分布结果
        desired = np.array([[0.97097342648766727, 0.91422896443565516],
                            [1.89517770034962929, 1.91414357960479564],
                            [0.67057783752390987, 1.39494046635066793]])
        # 检查生成的结果与预期结果的精度是否达到指定小数位数
        assert_array_almost_equal(actual, desired, decimal=15)
    # 定义测试函数 test_weibull_0，测试 Weibull 分布参数 a=0 的情况
    def test_weibull_0(self):
        # 设定随机数种子，保证结果可复现性
        np.random.seed(self.seed)
        # 断言：调用 np.random.weibull 方法生成 Weibull 分布的随机数，期望结果是一组大小为 12 的全零数组
        assert_equal(np.random.weibull(a=0, size=12), np.zeros(12))
        # 断言：调用 np.random.weibull 方法传入无效参数 -0.，期望抛出 ValueError 异常
        assert_raises(ValueError, np.random.weibull, a=-0.)

    # 定义测试函数 test_zipf，测试 Zipf 分布生成随机数的情况
    def test_zipf(self):
        # 设定随机数种子，保证结果可复现性
        np.random.seed(self.seed)
        # 调用 np.random.zipf 方法生成 Zipf 分布的随机数，期望结果是一个 3x2 的数组
        actual = np.random.zipf(a=1.23, size=(3, 2))
        # 期望的结果数组
        desired = np.array([[66, 29],
                            [1, 1],
                            [3, 13]])
        # 断言：比较生成的随机数数组和期望的数组是否相等
        assert_array_equal(actual, desired)
# 定义一个名为 TestBroadcast 的测试类，用于测试广播函数在非标量参数下的行为是否正确。
class TestBroadcast:
    
    # 在每个测试方法运行前执行的设置方法
    def setup_method(self):
        self.seed = 123456789

    # 设置随机种子的方法
    def setSeed(self):
        np.random.seed(self.seed)

    # TODO: Include test for randint once it can broadcast
    # Can steal the test written in PR #6938

    # 测试 np.random.uniform 函数
    def test_uniform(self):
        # 定义均匀分布的下界
        low = [0]
        # 定义均匀分布的上界
        high = [1]
        # 获取 np.random.uniform 函数的引用
        uniform = np.random.uniform
        # 期望的均匀分布结果
        desired = np.array([0.53283302478975902,
                            0.53413660089041659,
                            0.50955303552646702])

        # 设置随机种子
        self.setSeed()
        # 测试 uniform 函数在 low * 3, high 参数下的结果
        actual = uniform(low * 3, high)
        # 断言实际结果与期望结果的近似性
        assert_array_almost_equal(actual, desired, decimal=14)

        # 再次设置随机种子
        self.setSeed()
        # 测试 uniform 函数在 low, high * 3 参数下的结果
        actual = uniform(low, high * 3)
        # 断言实际结果与期望结果的近似性
        assert_array_almost_equal(actual, desired, decimal=14)

    # 测试 np.random.normal 函数
    def test_normal(self):
        # 定义正态分布的均值
        loc = [0]
        # 定义正态分布的标准差
        scale = [1]
        # 不良标准差的定义
        bad_scale = [-1]
        # 获取 np.random.normal 函数的引用
        normal = np.random.normal
        # 期望的正态分布结果
        desired = np.array([2.2129019979039612,
                            2.1283977976520019,
                            1.8417114045748335])

        # 设置随机种子
        self.setSeed()
        # 测试 normal 函数在 loc * 3, scale 参数下的结果
        actual = normal(loc * 3, scale)
        # 断言实际结果与期望结果的近似性
        assert_array_almost_equal(actual, desired, decimal=14)
        # 测试 normal 函数在 loc * 3, bad_scale 参数下是否引发 ValueError 异常
        assert_raises(ValueError, normal, loc * 3, bad_scale)

        # 再次设置随机种子
        self.setSeed()
        # 测试 normal 函数在 loc, scale * 3 参数下的结果
        actual = normal(loc, scale * 3)
        # 断言实际结果与期望结果的近似性
        assert_array_almost_equal(actual, desired, decimal=14)
        # 测试 normal 函数在 loc, bad_scale * 3 参数下是否引发 ValueError 异常
        assert_raises(ValueError, normal, loc, bad_scale * 3)

    # 测试 np.random.beta 函数
    def test_beta(self):
        # 定义 beta 分布的参数 a
        a = [1]
        # 定义 beta 分布的参数 b
        b = [2]
        # 不良参数 a 的定义
        bad_a = [-1]
        # 不良参数 b 的定义
        bad_b = [-2]
        # 获取 np.random.beta 函数的引用
        beta = np.random.beta
        # 期望的 beta 分布结果
        desired = np.array([0.19843558305989056,
                            0.075230336409423643,
                            0.24976865978980844])

        # 设置随机种子
        self.setSeed()
        # 测试 beta 函数在 a * 3, b 参数下的结果
        actual = beta(a * 3, b)
        # 断言实际结果与期望结果的近似性
        assert_array_almost_equal(actual, desired, decimal=14)
        # 测试 beta 函数在 bad_a * 3, b 参数下是否引发 ValueError 异常
        assert_raises(ValueError, beta, bad_a * 3, b)
        # 测试 beta 函数在 a * 3, bad_b 参数下是否引发 ValueError 异常
        assert_raises(ValueError, beta, a * 3, bad_b)

        # 再次设置随机种子
        self.setSeed()
        # 测试 beta 函数在 a, b * 3 参数下的结果
        actual = beta(a, b * 3)
        # 断言实际结果与期望结果的近似性
        assert_array_almost_equal(actual, desired, decimal=14)
        # 测试 beta 函数在 a, bad_b * 3 参数下是否引发 ValueError 异常
        assert_raises(ValueError, beta, a, bad_b * 3)

    # 测试 np.random.exponential 函数
    def test_exponential(self):
        # 定义指数分布的参数 scale
        scale = [1]
        # 不良参数 scale 的定义
        bad_scale = [-1]
        # 获取 np.random.exponential 函数的引用
        exponential = np.random.exponential
        # 期望的指数分布结果
        desired = np.array([0.76106853658845242,
                            0.76386282278691653,
                            0.71243813125891797])

        # 设置随机种子
        self.setSeed()
        # 测试 exponential 函数在 scale * 3 参数下的结果
        actual = exponential(scale * 3)
        # 断言实际结果与期望结果的近似性
        assert_array_almost_equal(actual, desired, decimal=14)
        # 测试 exponential 函数在 bad_scale * 3 参数下是否引发 ValueError 异常
        assert_raises(ValueError, exponential, bad_scale * 3)
    # 定义测试标准 Gamma 分布的方法
    def test_standard_gamma(self):
        shape = [1]  # Gamma 分布的形状参数
        bad_shape = [-1]  # 错误的形状参数，预期会引发 ValueError
        std_gamma = np.random.standard_gamma  # 引用 NumPy 中的标准 Gamma 分布函数
        desired = np.array([0.76106853658845242,
                            0.76386282278691653,
                            0.71243813125891797])  # 期望得到的随机数数组

        self.setSeed()  # 设置随机数种子
        actual = std_gamma(shape * 3)  # 生成符合给定形状的标准 Gamma 分布随机数
        assert_array_almost_equal(actual, desired, decimal=14)  # 断言实际生成的随机数数组与期望的数组相近
        assert_raises(ValueError, std_gamma, bad_shape * 3)  # 断言使用错误形状参数会引发 ValueError

    # 定义测试 Gamma 分布的方法
    def test_gamma(self):
        shape = [1]  # Gamma 分布的形状参数
        scale = [2]  # Gamma 分布的尺度参数
        bad_shape = [-1]  # 错误的形状参数，预期会引发 ValueError
        bad_scale = [-2]  # 错误的尺度参数，预期会引发 ValueError
        gamma = np.random.gamma  # 引用 NumPy 中的 Gamma 分布函数
        desired = np.array([1.5221370731769048,
                            1.5277256455738331,
                            1.4248762625178359])  # 期望得到的随机数数组

        self.setSeed()  # 设置随机数种子
        actual = gamma(shape * 3, scale)  # 生成符合给定形状和尺度的 Gamma 分布随机数
        assert_array_almost_equal(actual, desired, decimal=14)  # 断言实际生成的随机数数组与期望的数组相近
        assert_raises(ValueError, gamma, bad_shape * 3, scale)  # 断言使用错误形状参数会引发 ValueError
        assert_raises(ValueError, gamma, shape * 3, bad_scale)  # 断言使用错误尺度参数会引发 ValueError

        self.setSeed()  # 重新设置随机数种子
        actual = gamma(shape, scale * 3)  # 生成符合给定形状和多倍尺度的 Gamma 分布随机数
        assert_array_almost_equal(actual, desired, decimal=14)  # 断言实际生成的随机数数组与期望的数组相近
        assert_raises(ValueError, gamma, bad_shape, scale * 3)  # 断言使用错误形状参数会引发 ValueError
        assert_raises(ValueError, gamma, shape, bad_scale * 3)  # 断言使用错误尺度参数会引发 ValueError

    # 定义测试 F 分布的方法
    def test_f(self):
        dfnum = [1]  # F 分布的分子自由度参数
        dfden = [2]  # F 分布的分母自由度参数
        bad_dfnum = [-1]  # 错误的分子自由度参数，预期会引发 ValueError
        bad_dfden = [-2]  # 错误的分母自由度参数，预期会引发 ValueError
        f = np.random.f  # 引用 NumPy 中的 F 分布函数
        desired = np.array([0.80038951638264799,
                            0.86768719635363512,
                            2.7251095168386801])  # 期望得到的随机数数组

        self.setSeed()  # 设置随机数种子
        actual = f(dfnum * 3, dfden)  # 生成符合给定自由度的 F 分布随机数
        assert_array_almost_equal(actual, desired, decimal=14)  # 断言实际生成的随机数数组与期望的数组相近
        assert_raises(ValueError, f, bad_dfnum * 3, dfden)  # 断言使用错误的分子自由度参数会引发 ValueError
        assert_raises(ValueError, f, dfnum * 3, bad_dfden)  # 断言使用错误的分母自由度参数会引发 ValueError

        self.setSeed()  # 重新设置随机数种子
        actual = f(dfnum, dfden * 3)  # 生成符合给定自由度和多倍分母自由度的 F 分布随机数
        assert_array_almost_equal(actual, desired, decimal=14)  # 断言实际生成的随机数数组与期望的数组相近
        assert_raises(ValueError, f, bad_dfnum, dfden * 3)  # 断言使用错误的分子自由度参数会引发 ValueError
        assert_raises(ValueError, f, dfnum, bad_dfden * 3)  # 断言使用错误的分母自由度参数会引发 ValueError
    def test_noncentral_f(self):
        # 定义自由度参数
        dfnum = [2]
        dfden = [3]
        nonc = [4]
        # 定义无效的自由度参数
        bad_dfnum = [0]
        bad_dfden = [-1]
        bad_nonc = [-2]
        # 获取 numpy 的非中心 F 分布函数
        nonc_f = np.random.noncentral_f
        # 期望的结果数组
        desired = np.array([9.1393943263705211,
                            13.025456344595602,
                            8.8018098359100545])

        # 设置随机种子
        self.setSeed()
        # 计算非中心 F 分布
        actual = nonc_f(dfnum * 3, dfden, nonc)
        # 断言实际结果与期望结果近似相等
        assert_array_almost_equal(actual, desired, decimal=14)
        # 检查是否引发 ValueError 异常
        assert_raises(ValueError, nonc_f, bad_dfnum * 3, dfden, nonc)
        assert_raises(ValueError, nonc_f, dfnum * 3, bad_dfden, nonc)
        assert_raises(ValueError, nonc_f, dfnum * 3, dfden, bad_nonc)

        # 重新设置随机种子
        self.setSeed()
        # 再次计算非中心 F 分布
        actual = nonc_f(dfnum, dfden * 3, nonc)
        # 断言实际结果与期望结果近似相等
        assert_array_almost_equal(actual, desired, decimal=14)
        # 检查是否引发 ValueError 异常
        assert_raises(ValueError, nonc_f, bad_dfnum, dfden * 3, nonc)
        assert_raises(ValueError, nonc_f, dfnum, bad_dfden * 3, nonc)
        assert_raises(ValueError, nonc_f, dfnum, dfden * 3, bad_nonc)

        # 再次设置随机种子
        self.setSeed()
        # 第三次计算非中心 F 分布
        actual = nonc_f(dfnum, dfden, nonc * 3)
        # 断言实际结果与期望结果近似相等
        assert_array_almost_equal(actual, desired, decimal=14)
        # 检查是否引发 ValueError 异常
        assert_raises(ValueError, nonc_f, bad_dfnum, dfden, nonc * 3)
        assert_raises(ValueError, nonc_f, dfnum, bad_dfden, nonc * 3)
        assert_raises(ValueError, nonc_f, dfnum, dfden, bad_nonc * 3)
    # 定义一个用于测试标准 t 分布的方法
    def test_standard_t(self):
        # 创建包含一个元素的列表 df
        df = [1]
        # 创建包含一个负数的不良数据列表 bad_df
        bad_df = [-1]
        # 获取 NumPy 中的标准 t 分布生成器函数
        t = np.random.standard_t
        # 期望的 t 分布样本数组
        desired = np.array([3.0702872575217643,
                            5.8560725167361607,
                            1.0274791436474273])

        # 设置随机数种子
        self.setSeed()
        # 生成 t 分布样本
        actual = t(df * 3)
        # 断言生成的样本与期望值的接近程度
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言生成 t 分布时输入不合法值会引发 ValueError 异常
        assert_raises(ValueError, t, bad_df * 3)

    # 定义一个用于测试 von Mises 分布的方法
    def test_vonmises(self):
        # 给定 von Mises 分布的均值 mu
        mu = [2]
        # 给定 von Mises 分布的集中参数 kappa
        kappa = [1]
        # 不良的 von Mises 分布集中参数列表 bad_kappa
        bad_kappa = [-1]
        # 获取 NumPy 中的 von Mises 分布生成器函数
        vonmises = np.random.vonmises
        # 期望的 von Mises 分布样本数组
        desired = np.array([2.9883443664201312,
                            -2.7064099483995943,
                            -1.8672476700665914])

        # 设置随机数种子
        self.setSeed()
        # 生成 von Mises 分布样本
        actual = vonmises(mu * 3, kappa)
        # 断言生成的样本与期望值的接近程度
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言生成 von Mises 分布时输入不合法值会引发 ValueError 异常
        assert_raises(ValueError, vonmises, mu * 3, bad_kappa)

        # 设置随机数种子
        self.setSeed()
        # 生成 von Mises 分布样本
        actual = vonmises(mu, kappa * 3)
        # 断言生成的样本与期望值的接近程度
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言生成 von Mises 分布时输入不合法值会引发 ValueError 异常
        assert_raises(ValueError, vonmises, mu, bad_kappa * 3)

    # 定义一个用于测试 Pareto 分布的方法
    def test_pareto(self):
        # 给定 Pareto 分布的形状参数 a
        a = [1]
        # 不良的 Pareto 分布形状参数列表 bad_a
        bad_a = [-1]
        # 获取 NumPy 中的 Pareto 分布生成器函数
        pareto = np.random.pareto
        # 期望的 Pareto 分布样本数组
        desired = np.array([1.1405622680198362,
                            1.1465519762044529,
                            1.0389564467453547])

        # 设置随机数种子
        self.setSeed()
        # 生成 Pareto 分布样本
        actual = pareto(a * 3)
        # 断言生成的样本与期望值的接近程度
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言生成 Pareto 分布时输入不合法值会引发 ValueError 异常
        assert_raises(ValueError, pareto, bad_a * 3)

    # 定义一个用于测试 Weibull 分布的方法
    def test_weibull(self):
        # 给定 Weibull 分布的形状参数 a
        a = [1]
        # 不良的 Weibull 分布形状参数列表 bad_a
        bad_a = [-1]
        # 获取 NumPy 中的 Weibull 分布生成器函数
        weibull = np.random.weibull
        # 期望的 Weibull 分布样本数组
        desired = np.array([0.76106853658845242,
                            0.76386282278691653,
                            0.71243813125891797])

        # 设置随机数种子
        self.setSeed()
        # 生成 Weibull 分布样本
        actual = weibull(a * 3)
        # 断言生成的样本与期望值的接近程度
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言生成 Weibull 分布时输入不合法值会引发 ValueError 异常
        assert_raises(ValueError, weibull, bad_a * 3)

    # 定义一个用于测试 Power 分布的方法
    def test_power(self):
        # 给定 Power 分布的形状参数 a
        a = [1]
        # 不良的 Power 分布形状参数列表 bad_a
        bad_a = [-1]
        # 获取 NumPy 中的 Power 分布生成器函数
        power = np.random.power
        # 期望的 Power 分布样本数组
        desired = np.array([0.53283302478975902,
                            0.53413660089041659,
                            0.50955303552646702])

        # 设置随机数种子
        self.setSeed()
        # 生成 Power 分布样本
        actual = power(a * 3)
        # 断言生成的样本与期望值的接近程度
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言生成 Power 分布时输入不合法值会引发 ValueError 异常
        assert_raises(ValueError, power, bad_a * 3)

    # 定义一个用于测试 Laplace 分布的方法
    def test_laplace(self):
        # 给定 Laplace 分布的位置参数 loc
        loc = [0]
        # 给定 Laplace 分布的尺度参数 scale
        scale = [1]
        # 不良的 Laplace 分布尺度参数列表 bad_scale
        bad_scale = [-1]
        # 获取 NumPy 中的 Laplace 分布生成器函数
        laplace = np.random.laplace
        # 期望的 Laplace 分布样本数组
        desired = np.array([0.067921356028507157,
                            0.070715642226971326,
                            0.019290950698972624])

        # 设置随机数种子
        self.setSeed()
        # 生成 Laplace 分布样本
        actual = laplace(loc * 3, scale)
        # 断言生成的样本与期望值的接近程度
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言生成 Laplace 分布时输入不合法值会引发 ValueError 异常
        assert_raises(ValueError, laplace, loc * 3, bad_scale)

        # 设置随机数种子
        self.setSeed()
        # 生成 Laplace 分布样本
        actual = laplace(loc, scale * 3)
        # 断言生成的样本与期望值的接近程度
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言生成 Laplace 分布时输入不合法值会引发 ValueError 异常
        assert_raises(ValueError, laplace, loc, bad_scale * 3)
    # 定义一个测试函数，用于测试 Gumbel 分布生成随机数的正确性
    def test_gumbel(self):
        # 设置 Gumbel 分布的位置参数
        loc = [0]
        # 设置 Gumbel 分布的尺度参数
        scale = [1]
        # 设置错误的 Gumbel 分布尺度参数
        bad_scale = [-1]
        # 获取 numpy 库中的 Gumbel 随机数生成函数
        gumbel = np.random.gumbel
        # 期望的 Gumbel 分布生成的随机数
        desired = np.array([0.2730318639556768,
                            0.26936705726291116,
                            0.33906220393037939])

        # 设置随机数生成器的种子
        self.setSeed()
        # 生成 Gumbel 分布的随机数，并与期望值进行比较
        actual = gumbel(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        # 检查当使用错误的尺度参数时是否会抛出 ValueError 异常
        assert_raises(ValueError, gumbel, loc * 3, bad_scale)

        # 重新设置随机数生成器的种子
        self.setSeed()
        # 生成 Gumbel 分布的随机数，并与期望值进行比较
        actual = gumbel(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        # 检查当使用错误的尺度参数时是否会抛出 ValueError 异常
        assert_raises(ValueError, gumbel, loc, bad_scale * 3)

    # 定义一个测试函数，用于测试 Logistic 分布生成随机数的正确性
    def test_logistic(self):
        # 设置 Logistic 分布的位置参数
        loc = [0]
        # 设置 Logistic 分布的尺度参数
        scale = [1]
        # 设置错误的 Logistic 分布尺度参数
        bad_scale = [-1]
        # 获取 numpy 库中的 Logistic 随机数生成函数
        logistic = np.random.logistic
        # 期望的 Logistic 分布生成的随机数
        desired = np.array([0.13152135837586171,
                            0.13675915696285773,
                            0.038216792802833396])

        # 设置随机数生成器的种子
        self.setSeed()
        # 生成 Logistic 分布的随机数，并与期望值进行比较
        actual = logistic(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        # 检查当使用错误的尺度参数时是否会抛出 ValueError 异常
        assert_raises(ValueError, logistic, loc * 3, bad_scale)

        # 重新设置随机数生成器的种子
        self.setSeed()
        # 生成 Logistic 分布的随机数，并与期望值进行比较
        actual = logistic(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        # 检查当使用错误的尺度参数时是否会抛出 ValueError 异常
        assert_raises(ValueError, logistic, loc, bad_scale * 3)

    # 定义一个测试函数，用于测试 Lognormal 分布生成随机数的正确性
    def test_lognormal(self):
        # 设置 Lognormal 分布的均值参数
        mean = [0]
        # 设置 Lognormal 分布的标准差参数
        sigma = [1]
        # 设置错误的 Lognormal 分布标准差参数
        bad_sigma = [-1]
        # 获取 numpy 库中的 Lognormal 随机数生成函数
        lognormal = np.random.lognormal
        # 期望的 Lognormal 分布生成的随机数
        desired = np.array([9.1422086044848427,
                            8.4013952870126261,
                            6.3073234116578671])

        # 设置随机数生成器的种子
        self.setSeed()
        # 生成 Lognormal 分布的随机数，并与期望值进行比较
        actual = lognormal(mean * 3, sigma)
        assert_array_almost_equal(actual, desired, decimal=14)
        # 检查当使用错误的标准差参数时是否会抛出 ValueError 异常
        assert_raises(ValueError, lognormal, mean * 3, bad_sigma)

        # 重新设置随机数生成器的种子
        self.setSeed()
        # 生成 Lognormal 分布的随机数，并与期望值进行比较
        actual = lognormal(mean, sigma * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        # 检查当使用错误的标准差参数时是否会抛出 ValueError 异常
        assert_raises(ValueError, lognormal, mean, bad_sigma * 3)

    # 定义一个测试函数，用于测试 Rayleigh 分布生成随机数的正确性
    def test_rayleigh(self):
        # 设置 Rayleigh 分布的尺度参数
        scale = [1]
        # 设置错误的 Rayleigh 分布尺度参数
        bad_scale = [-1]
        # 获取 numpy 库中的 Rayleigh 随机数生成函数
        rayleigh = np.random.rayleigh
        # 期望的 Rayleigh 分布生成的随机数
        desired = np.array([1.2337491937897689,
                            1.2360119924878694,
                            1.1936818095781789])

        # 设置随机数生成器的种子
        self.setSeed()
        # 生成 Rayleigh 分布的随机数，并与期望值进行比较
        actual = rayleigh(scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        # 检查当使用错误的尺度参数时是否会抛出 ValueError 异常
        assert_raises(ValueError, rayleigh, bad_scale * 3)
    # 定义名为 test_wald 的测试方法
    def test_wald(self):
        # 定义正常的均值和标度参数列表
        mean = [0.5]
        scale = [1]
        # 定义异常情况下的均值和标度参数列表
        bad_mean = [0]
        bad_scale = [-2]
        # 获取 numpy.random 模块中的 wald 函数
        wald = np.random.wald
        # 期望的结果数组
        desired = np.array([0.11873681120271318,
                            0.12450084820795027,
                            0.9096122728408238])

        # 设置随机数种子
        self.setSeed()
        # 使用 wald 函数生成实际结果
        actual = wald(mean * 3, scale)
        # 断言实际结果与期望结果的近似程度
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言对于坏的均值参数列表，会抛出 ValueError 异常
        assert_raises(ValueError, wald, bad_mean * 3, scale)
        # 断言对于坏的标度参数列表，会抛出 ValueError 异常
        assert_raises(ValueError, wald, mean * 3, bad_scale)

        # 再次设置随机数种子
        self.setSeed()
        # 使用 wald 函数生成实际结果
        actual = wald(mean, scale * 3)
        # 断言实际结果与期望结果的近似程度
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言对于坏的均值参数列表，会抛出 ValueError 异常
        assert_raises(ValueError, wald, bad_mean, scale * 3)
        # 断言对于坏的标度参数列表，会抛出 ValueError 异常
        assert_raises(ValueError, wald, mean, bad_scale * 3)
        # 断言当均值参数为 0.0 时，会抛出 ValueError 异常
        assert_raises(ValueError, wald, 0.0, 1)
        # 断言当标度参数为 0.0 时，会抛出 ValueError 异常
        assert_raises(ValueError, wald, 0.5, 0.0)

    # 定义名为 test_triangular 的测试方法
    def test_triangular(self):
        # 定义三角分布的左端参数列表
        left = [1]
        # 定义三角分布的右端参数列表
        right = [3]
        # 定义三角分布的众数参数列表
        mode = [2]
        # 定义异常情况下的左端和众数参数列表
        bad_left_one = [3]
        bad_mode_one = [4]
        # 使用右端参数列表的元素复制构成坏的左端和众数参数列表
        bad_left_two, bad_mode_two = right * 2
        # 获取 numpy.random 模块中的 triangular 函数
        triangular = np.random.triangular
        # 期望的结果数组
        desired = np.array([2.03339048710429,
                            2.0347400359389356,
                            2.0095991069536208])

        # 设置随机数种子
        self.setSeed()
        # 使用 triangular 函数生成实际结果
        actual = triangular(left * 3, mode, right)
        # 断言实际结果与期望结果的近似程度
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言对于坏的左端参数列表，会抛出 ValueError 异常
        assert_raises(ValueError, triangular, bad_left_one * 3, mode, right)
        # 断言对于坏的众数参数列表，会抛出 ValueError 异常
        assert_raises(ValueError, triangular, left * 3, bad_mode_one, right)
        # 断言对于坏的左端和众数参数列表，会抛出 ValueError 异常
        assert_raises(ValueError, triangular, bad_left_two * 3, bad_mode_two,
                      right)

        # 再次设置随机数种子
        self.setSeed()
        # 使用 triangular 函数生成实际结果
        actual = triangular(left, mode * 3, right)
        # 断言实际结果与期望结果的近似程度
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言对于坏的左端参数列表，会抛出 ValueError 异常
        assert_raises(ValueError, triangular, bad_left_one, mode * 3, right)
        # 断言对于坏的众数参数列表，会抛出 ValueError 异常
        assert_raises(ValueError, triangular, left, bad_mode_one * 3, right)
        # 断言对于坏的左端和众数参数列表，会抛出 ValueError 异常
        assert_raises(ValueError, triangular, bad_left_two, bad_mode_two * 3,
                      right)

        # 再次设置随机数种子
        self.setSeed()
        # 使用 triangular 函数生成实际结果
        actual = triangular(left, mode, right * 3)
        # 断言实际结果与期望结果的近似程度
        assert_array_almost_equal(actual, desired, decimal=14)
        # 断言对于坏的左端参数列表，会抛出 ValueError 异常
        assert_raises(ValueError, triangular, bad_left_one, mode, right * 3)
        # 断言对于坏的众数参数列表，会抛出 ValueError 异常
        assert_raises(ValueError, triangular, left, bad_mode_one, right * 3)
        # 断言对于坏的左端和众数参数列表，会抛出 ValueError 异常
        assert_raises(ValueError, triangular, bad_left_two, bad_mode_two,
                      right * 3)
    # 测试二项分布生成器函数
    def test_binomial(self):
        # 初始化参数
        n = [1]  # 成功次数的期望值
        p = [0.5]  # 每次试验成功的概率
        bad_n = [-1]  # 错误的成功次数期望值（负数）
        bad_p_one = [-1]  # 错误的成功概率（负数）
        bad_p_two = [1.5]  # 错误的成功概率（大于1）
        binom = np.random.binomial  # 二项分布生成函数
        desired = np.array([1, 1, 1])  # 期望的结果数组

        # 设置随机数种子
        self.setSeed()

        # 测试成功
        actual = binom(n * 3, p)
        assert_array_equal(actual, desired)

        # 测试错误情况：负数的成功次数期望值
        assert_raises(ValueError, binom, bad_n * 3, p)

        # 测试错误情况：负数的成功概率
        assert_raises(ValueError, binom, n * 3, bad_p_one)

        # 测试错误情况：超出范围的成功概率
        assert_raises(ValueError, binom, n * 3, bad_p_two)

        # 重新设置随机数种子，进行第二组测试
        self.setSeed()

        # 测试成功
        actual = binom(n, p * 3)
        assert_array_equal(actual, desired)

        # 测试错误情况：负数的成功次数期望值
        assert_raises(ValueError, binom, bad_n, p * 3)

        # 测试错误情况：负数的成功概率
        assert_raises(ValueError, binom, n, bad_p_one * 3)

        # 测试错误情况：超出范围的成功概率
        assert_raises(ValueError, binom, n, bad_p_two * 3)


```  
    # 测试负二项分布生成器函数
    def test_negative_binomial(self):
        # 初始化参数
        n = [1]  # 成功次数的期望值
        p = [0.5]  # 每次试验成功的概率
        bad_n = [-1]  # 错误的成功次数期望值（负数）
        bad_p_one = [-1]  # 错误的成功概率（负数）
        bad_p_two = [1.5]  # 错误的成功概率（大于1）
        neg_binom = np.random.negative_binomial  # 负二项分布生成函数
        desired = np.array([1, 0, 1])  # 期望的结果数组

        # 设置随机数种子
        self.setSeed()

        # 测试成功
        actual = neg_binom(n * 3, p)
        assert_array_equal(actual, desired)

        # 测试错误情况：负数的成功次数期望值
        assert_raises(ValueError, neg_binom, bad_n * 3, p)

        # 测试错误情况：负数的成功概率
        assert_raises(ValueError, neg_binom, n * 3, bad_p_one)

        # 测试错误情况：超出范围的成功概率
        assert_raises(ValueError, neg_binom, n * 3, bad_p_two)

        # 重新设置随机数种子，进行第二组测试
        self.setSeed()

        # 测试成功
        actual = neg_binom(n, p * 3)
        assert_array_equal(actual, desired)

        # 测试错误情况：负数的成功次数期望值
        assert_raises(ValueError, neg_binom, bad_n, p * 3)

        # 测试错误情况：负数的成功概率
        assert_raises(ValueError, neg_binom, n, bad_p_one * 3)

        # 测试错误情况：超出范围的成功概率
        assert_raises(ValueError, neg_binom, n, bad_p_two * 3)


```py  
    # 测试泊松分布生成器函数
    def test_poisson(self):
        # 获取最大允许的泊松分布参数值
        max_lam = np.random.RandomState()._poisson_lam_max

        # 初始化参数
        lam = [1]  # 泊松分布的参数 lambda
        bad_lam_one = [-1]  # 错误的 lambda 值（负数）
        bad_lam_two = [max_lam * 2]  # 错误的 lambda 值（超出允许范围）
        poisson = np.random.poisson  # 泊松分布生成函数
        desired = np.array([1, 1, 0])  # 期望的结果数组

        # 设置随机数种子
        self.setSeed()

        # 测试成功
        actual = poisson(lam * 3)
        assert_array_equal(actual, desired)

        # 测试错误情况：负数的 lambda 值
        assert_raises(ValueError, poisson, bad_lam_one * 3)

        # 测试错误情况：超出范围的 lambda 值
        assert_raises(ValueError, poisson, bad_lam_two * 3)


```  
    # 测试 Zipf 分布生成器函数
    def test_zipf(self):
        # 初始化参数
        a = [2]  # Zipf 分布的参数 a
        bad_a = [0]  # 错误的 a 值（0）
        zipf = np.random.zipf  # Zipf 分布生成函数
        desired = np.array([2, 2, 1])  # 期望的结果数组

        # 设置随机数种子
        self.setSeed()

        # 测试成功
        actual = zipf(a * 3)
        assert_array_equal(actual, desired)

        # 测试错误情况：0 的 a 值
        assert_raises(ValueError, zipf, bad_a * 3)

        # 测试 NaN 的情况，使用 np.errstate 忽略无效值错误
        with np.errstate(invalid='ignore'):
            assert_raises(ValueError, zipf, np.nan)
            assert_raises(ValueError, zipf, [0, 0, np.nan])


```py  
    # 测试几何分布生成器函数
    def test_geometric(self):
        # 初始化参数
        p = [0.5]  # 几何分布的成功概率
        bad_p_one = [-1]  # 错误的成功概率（负数）
        bad_p_two = [1.5]  # 错误的成功概率（大于1）
        geom = np.random.geometric  # 几何分布生成函数
        desired = np.array([2, 2, 2])  # 期望的结果数组

        # 设置随机数种子
        self.setSeed()

        # 测试成功
        actual = geom(p * 3)
        assert_array_equal(actual, desired)

        # 测试错误情况：负数的成功概率
        assert_raises(ValueError, geom, bad_p_one * 3)

        # 测试错误情况：超出范围的成功概率
        assert_raises(ValueError, geom, bad_p_two * 3)
    # 定义一个测试函数，用于测试超几何分布的相关功能
    def test_hypergeometric(self):
        # 定义正常情况下的参数
        ngood = [1]  # 成功事件总数列表
        nbad = [2]   # 失败事件总数列表
        nsample = [2]  # 抽样总数列表
        bad_ngood = [-1]  # 错误的成功事件总数列表
        bad_nbad = [-2]   # 错误的失败事件总数列表
        bad_nsample_one = [0]   # 错误的抽样总数列表一
        bad_nsample_two = [4]   # 错误的抽样总数列表二
        hypergeom = np.random.hypergeometric  # 获取 NumPy 中超几何分布的随机数生成器函数
        desired = np.array([1, 1, 1])  # 期望的结果数组，用于比较实际结果

        # 设置随机数种子
        self.setSeed()

        # 测试不同参数组合下的超几何分布生成情况
        actual = hypergeom(ngood * 3, nbad, nsample)
        assert_array_equal(actual, desired)  # 断言实际生成的数组与期望的数组相等
        assert_raises(ValueError, hypergeom, bad_ngood * 3, nbad, nsample)  # 断言对错误参数抛出 ValueError 异常
        assert_raises(ValueError, hypergeom, ngood * 3, bad_nbad, nsample)  # 断言对错误参数抛出 ValueError 异常
        assert_raises(ValueError, hypergeom, ngood * 3, nbad, bad_nsample_one)  # 断言对错误参数抛出 ValueError 异常
        assert_raises(ValueError, hypergeom, ngood * 3, nbad, bad_nsample_two)  # 断言对错误参数抛出 ValueError 异常

        # 设置随机数种子
        self.setSeed()

        # 测试不同参数组合下的超几何分布生成情况
        actual = hypergeom(ngood, nbad * 3, nsample)
        assert_array_equal(actual, desired)  # 断言实际生成的数组与期望的数组相等
        assert_raises(ValueError, hypergeom, bad_ngood, nbad * 3, nsample)  # 断言对错误参数抛出 ValueError 异常
        assert_raises(ValueError, hypergeom, ngood, bad_nbad * 3, nsample)  # 断言对错误参数抛出 ValueError 异常
        assert_raises(ValueError, hypergeom, ngood, nbad * 3, bad_nsample_one)  # 断言对错误参数抛出 ValueError 异常
        assert_raises(ValueError, hypergeom, ngood, nbad * 3, bad_nsample_two)  # 断言对错误参数抛出 ValueError 异常

        # 设置随机数种子
        self.setSeed()

        # 测试不同参数组合下的超几何分布生成情况
        actual = hypergeom(ngood, nbad, nsample * 3)
        assert_array_equal(actual, desired)  # 断言实际生成的数组与期望的数组相等
        assert_raises(ValueError, hypergeom, bad_ngood, nbad, nsample * 3)  # 断言对错误参数抛出 ValueError 异常
        assert_raises(ValueError, hypergeom, ngood, bad_nbad, nsample * 3)  # 断言对错误参数抛出 ValueError 异常
        assert_raises(ValueError, hypergeom, ngood, nbad, bad_nsample_one * 3)  # 断言对错误参数抛出 ValueError 异常
        assert_raises(ValueError, hypergeom, ngood, nbad, bad_nsample_two * 3)  # 断言对错误参数抛出 ValueError 异常

    # 定义一个测试函数，用于测试对数级数分布的相关功能
    def test_logseries(self):
        p = [0.5]  # 概率参数列表
        bad_p_one = [2]  # 错误的概率参数列表一
        bad_p_two = [-1]  # 错误的概率参数列表二
        logseries = np.random.logseries  # 获取 NumPy 中对数级数分布的随机数生成器函数
        desired = np.array([1, 1, 1])  # 期望的结果数组，用于比较实际结果

        # 设置随机数种子
        self.setSeed()

        # 测试不同参数组合下的对数级数分布生成情况
        actual = logseries(p * 3)
        assert_array_equal(actual, desired)  # 断言实际生成的数组与期望的数组相等
        assert_raises(ValueError, logseries, bad_p_one * 3)  # 断言对错误参数抛出 ValueError 异常
        assert_raises(ValueError, logseries, bad_p_two * 3)  # 断言对错误参数抛出 ValueError 异常
# 如果在 WASM 平台上运行测试，则跳过这个测试类，因为无法启动线程
@pytest.mark.skipif(IS_WASM, reason="can't start thread")
class TestThread:
    # 确保即使在多线程情况下，每个状态产生的随机数序列也是相同的
    def setup_method(self):
        # 初始化随机数生成器的种子
        self.seeds = range(4)

    def check_function(self, function, sz):
        from threading import Thread

        # 创建两个空数组，用于存储多个线程和串行生成的结果
        out1 = np.empty((len(self.seeds),) + sz)
        out2 = np.empty((len(self.seeds),) + sz)

        # 多线程生成随机数
        t = [Thread(target=function, args=(np.random.RandomState(s), o))
             for s, o in zip(self.seeds, out1)]
        [x.start() for x in t]
        [x.join() for x in t]

        # 串行生成随机数
        for s, o in zip(self.seeds, out2):
            function(np.random.RandomState(s), o)

        # 对于某些平台（例如 32 位 Windows），线程可能会改变 x87 FPU 的精度模式
        if np.intp().dtype.itemsize == 4 and sys.platform == "win32":
            assert_array_almost_equal(out1, out2)
        else:
            assert_array_equal(out1, out2)

    def test_normal(self):
        # 定义生成正态分布随机数的函数
        def gen_random(state, out):
            out[...] = state.normal(size=10000)
        self.check_function(gen_random, sz=(10000,))

    def test_exp(self):
        # 定义生成指数分布随机数的函数
        def gen_random(state, out):
            out[...] = state.exponential(scale=np.ones((100, 1000)))
        self.check_function(gen_random, sz=(100, 1000))

    def test_multinomial(self):
        # 定义生成多项分布随机数的函数
        def gen_random(state, out):
            out[...] = state.multinomial(10, [1/6.]*6, size=10000)
        self.check_function(gen_random, sz=(10000, 6))


# 查看问题编号 #4263
class TestSingleEltArrayInput:
    def setup_method(self):
        # 初始化测试用例的输入参数和目标形状
        self.argOne = np.array([2])
        self.argTwo = np.array([3])
        self.argThree = np.array([4])
        self.tgtShape = (1,)

    def test_one_arg_funcs(self):
        # 定义需要测试的单参数随机数生成函数
        funcs = (np.random.exponential, np.random.standard_gamma,
                 np.random.chisquare, np.random.standard_t,
                 np.random.pareto, np.random.weibull,
                 np.random.power, np.random.rayleigh,
                 np.random.poisson, np.random.zipf,
                 np.random.geometric, np.random.logseries)

        probfuncs = (np.random.geometric, np.random.logseries)

        # 遍历每个随机数生成函数
        for func in funcs:
            if func in probfuncs:  # 对于参数 p 小于 1.0 的函数
                out = func(np.array([0.5]))
            else:
                out = func(self.argOne)

            # 断言生成的随机数的形状与目标形状相匹配
            assert_equal(out.shape, self.tgtShape)
    # 测试包含两个参数的随机函数
    def test_two_arg_funcs(self):
        # 定义包含随机函数的元组
        funcs = (np.random.uniform, np.random.normal,
                 np.random.beta, np.random.gamma,
                 np.random.f, np.random.noncentral_chisquare,
                 np.random.vonmises, np.random.laplace,
                 np.random.gumbel, np.random.logistic,
                 np.random.lognormal, np.random.wald,
                 np.random.binomial, np.random.negative_binomial)

        # 只包含两个参数的概率分布函数
        probfuncs = (np.random.binomial, np.random.negative_binomial)

        # 对于每个随机函数进行迭代
        for func in funcs:
            # 如果函数属于概率函数，则设置参数 argTwo 为数组 [0.5]
            if func in probfuncs:  # p <= 1
                argTwo = np.array([0.5])
            else:
                # 否则使用类中的 self.argTwo 参数
                argTwo = self.argTwo

            # 调用函数 func，并传入 self.argOne 和 argTwo 参数，获取输出
            out = func(self.argOne, argTwo)
            # 断言输出的形状与预期相同
            assert_equal(out.shape, self.tgtShape)

            # 调用函数 func，传入 self.argOne[0] 和 argTwo 参数，获取输出
            out = func(self.argOne[0], argTwo)
            # 再次断言输出的形状与预期相同
            assert_equal(out.shape, self.tgtShape)

            # 调用函数 func，传入 self.argOne 和 argTwo[0] 参数，获取输出
            out = func(self.argOne, argTwo[0])
            # 再次断言输出的形状与预期相同
            assert_equal(out.shape, self.tgtShape)

    # 测试 np.random.randint 函数
    def test_randint(self):
        # 定义整数类型数组
        itype = [bool, np.int8, np.uint8, np.int16, np.uint16,
                 np.int32, np.uint32, np.int64, np.uint64]
        # 设置函数为 np.random.randint
        func = np.random.randint
        # 设置 high 参数为数组 [1]
        high = np.array([1])
        # 设置 low 参数为数组 [0]
        low = np.array([0])

        # 对于每种整数类型进行迭代
        for dt in itype:
            # 调用 func 函数，传入 low, high 和 dtype=dt 参数，获取输出
            out = func(low, high, dtype=dt)
            # 断言输出的形状与预期相同
            assert_equal(out.shape, self.tgtShape)

            # 调用 func 函数，传入 low[0], high 和 dtype=dt 参数，获取输出
            out = func(low[0], high, dtype=dt)
            # 再次断言输出的形状与预期相同
            assert_equal(out.shape, self.tgtShape)

            # 调用 func 函数，传入 low, high[0] 和 dtype=dt 参数，获取输出
            out = func(low, high[0], dtype=dt)
            # 再次断言输出的形状与预期相同
            assert_equal(out.shape, self.tgtShape)

    # 测试包含三个参数的随机函数
    def test_three_arg_funcs(self):
        # 定义包含三个参数的随机函数列表
        funcs = [np.random.noncentral_f, np.random.triangular,
                 np.random.hypergeometric]

        # 对于每个随机函数进行迭代
        for func in funcs:
            # 调用 func 函数，传入 self.argOne, self.argTwo 和 self.argThree 参数，获取输出
            out = func(self.argOne, self.argTwo, self.argThree)
            # 断言输出的形状与预期相同
            assert_equal(out.shape, self.tgtShape)

            # 调用 func 函数，传入 self.argOne[0], self.argTwo 和 self.argThree 参数，获取输出
            out = func(self.argOne[0], self.argTwo, self.argThree)
            # 再次断言输出的形状与预期相同
            assert_equal(out.shape, self.tgtShape)

            # 调用 func 函数，传入 self.argOne, self.argTwo[0] 和 self.argThree 参数，获取输出
            out = func(self.argOne, self.argTwo[0], self.argThree)
            # 再次断言输出的形状与预期相同
            assert_equal(out.shape, self.tgtShape)
```