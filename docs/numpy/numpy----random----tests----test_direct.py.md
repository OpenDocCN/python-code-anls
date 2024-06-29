# `.\numpy\numpy\random\tests\test_direct.py`

```py
import os  # 导入操作系统相关功能
from os.path import join  # 导入路径拼接函数
import sys  # 导入系统相关功能

import numpy as np  # 导入 NumPy 库
from numpy.testing import (assert_equal, assert_allclose, assert_array_equal,
                           assert_raises)  # 导入 NumPy 测试相关函数
import pytest  # 导入 pytest 测试框架

from numpy.random import (  # 导入 NumPy 随机数生成器及相关类
    Generator, MT19937, PCG64, PCG64DXSM, Philox, RandomState, SeedSequence,
    SFC64, default_rng
)
from numpy.random._common import interface  # 导入 NumPy 随机数生成器通用接口

try:
    import cffi  # 尝试导入 cffi 库
    MISSING_CFFI = False  # 若导入成功则设为 False
except ImportError:
    MISSING_CFFI = True  # 若导入失败则设为 True

try:
    import ctypes  # 尝试导入 ctypes 库
    MISSING_CTYPES = False  # 若导入成功则设为 False
except ImportError:
    MISSING_CTYPES = True  # 若导入失败则设为 True

if sys.flags.optimize > 1:
    # 当 Python 优化标志大于 1 时，缺少文档字符串以检查，因此无法使用 cffi
    MISSING_CFFI = True

pwd = os.path.dirname(os.path.abspath(__file__))
# 获取当前文件所在目录的绝对路径
    # 当高斯随机数列表长度小于所需的数量 n 时，继续生成随机数
    while len(gauss) < n:
        r2 = 2
        # 循环直到生成有效的随机数 r2，保证 r2 不大于 1.0 且不为 0.0
        while r2 >= 1.0 or r2 == 0.0:
            # 生成服从均匀分布的随机数，并转换为范围在 [-1, 1) 的数值
            x1 = 2.0 * doubles[loc] - 1.0
            x2 = 2.0 * doubles[loc + 1] - 1.0
            # 计算平方和 r2
            r2 = x1 * x1 + x2 * x2
            loc += 2
    
        # 计算 Box-Muller 变换中的转换因子 f
        f = np.sqrt(-2.0 * np.log(r2) / r2)
        # 根据 Box-Muller 变换生成两个服从标准正态分布的随机数，并加入高斯随机数列表
        gauss.append(f * x2)
        gauss.append(f * x1)
    
    # 返回生成的前 n 个高斯分布随机数
    return gauss[:n]
# 定义一个测试函数，用于测试 SeedSequence 相关功能
def test_seedsequence():
    # 导入必要的模块和类
    from numpy.random.bit_generator import (ISeedSequence,
                                            ISpawnableSeedSequence,
                                            SeedlessSeedSequence)

    # 创建一个 SeedSequence 实例 s1，指定种子范围、生成键和池大小
    s1 = SeedSequence(range(10), spawn_key=(1, 2), pool_size=6)
    # 生成 10 个子序列
    s1.spawn(10)
    # 使用 s1 的状态创建另一个 SeedSequence 实例 s2
    s2 = SeedSequence(**s1.state)
    # 断言 s1 和 s2 的状态相同
    assert_equal(s1.state, s2.state)
    # 断言 s1 和 s2 的生成子序列数量相同
    assert_equal(s1.n_children_spawned, s2.n_children_spawned)

    # 检查接口无法直接实例化的情况
    assert_raises(TypeError, ISeedSequence)
    assert_raises(TypeError, ISpawnableSeedSequence)
    # 创建一个 SeedlessSeedSequence 实例 dummy
    dummy = SeedlessSeedSequence()
    # 断言调用生成状态方法会抛出 NotImplementedError 异常
    assert_raises(NotImplementedError, dummy.generate_state, 10)
    # 断言 dummy 的 spawn 方法生成了长度为 10 的列表
    assert len(dummy.spawn(10)) == 10


# 定义一个测试函数，测试生成新的生成器和位生成器
def test_generator_spawning():
    """ Test spawning new generators and bit_generators directly.
    """
    # 使用 numpy 生成默认的随机数生成器 rng
    rng = np.random.default_rng()
    # 获取 rng 的位生成器的种子序列
    seq = rng.bit_generator.seed_seq
    # 生成 5 个新的种子序列
    new_ss = seq.spawn(5)
    # 期望的生成键列表
    expected_keys = [seq.spawn_key + (i,) for i in range(5)]
    # 断言新生成的种子序列的生成键与期望的一致
    assert [c.spawn_key for c in new_ss] == expected_keys

    # 生成 5 个新的位生成器
    new_bgs = rng.bit_generator.spawn(5)
    # 期望的生成键列表
    expected_keys = [seq.spawn_key + (i,) for i in range(5, 10)]
    # 断言新生成的位生成器的种子序列的生成键与期望的一致
    assert [bg.seed_seq.spawn_key for bg in new_bgs] == expected_keys

    # 生成 5 个新的随机数生成器
    new_rngs = rng.spawn(5)
    # 期望的生成键列表
    expected_keys = [seq.spawn_key + (i,) for i in range(10, 15)]
    # 获取新生成的随机数生成器的种子序列的生成键列表
    found_keys = [rng.bit_generator.seed_seq.spawn_key for rng in new_rngs]
    # 断言生成键列表与期望的一致
    assert found_keys == expected_keys

    # 检查新生成的随机数生成器的第一个和第二个是否产生了不同的均匀分布值
    assert new_rngs[0].uniform() != new_rngs[1].uniform()


# 定义一个测试函数，测试无法生成的情况
def test_non_spawnable():
    from numpy.random.bit_generator import ISeedSequence

    # 定义一个虚拟的 SeedSequence 类
    class FakeSeedSequence:
        # 生成状态的虚拟方法
        def generate_state(self, n_words, dtype=np.uint32):
            return np.zeros(n_words, dtype=dtype)

    # 将虚拟的 SeedSequence 类注册为 ISeedSequence 的子类
    ISeedSequence.register(FakeSeedSequence)

    # 使用虚拟的 SeedSequence 创建随机数生成器 rng
    rng = np.random.default_rng(FakeSeedSequence())

    # 使用 pytest 断言异常信息匹配的方式，断言生成子序列方法会抛出 TypeError 异常
    with pytest.raises(TypeError, match="The underlying SeedSequence"):
        rng.spawn(5)

    # 使用 pytest 断言异常信息匹配的方式，断言位生成器的生成子序列方法会抛出 TypeError 异常
    with pytest.raises(TypeError, match="The underlying SeedSequence"):
        rng.bit_generator.spawn(5)


class Base:
    # 类属性：指定数据类型为 np.uint64
    dtype = np.uint64
    # 类属性：data1 和 data2 均为空字典
    data2 = data1 = {}

    # 类方法：用于设置类的初始状态
    @classmethod
    def setup_class(cls):
        # 设置类的位生成器为 PCG64
        cls.bit_generator = PCG64
        # 设置类的比特数为 64
        cls.bits = 64
        # 设置类的数据类型为 np.uint64
        cls.dtype = np.uint64
        # 设置类的错误类型为 TypeError
        cls.seed_error_type = TypeError
        # 设置类的无效初始化类型为空列表
        cls.invalid_init_types = []
        # 设置类的无效初始化值为空列表

    # 类方法：用于从 CSV 文件读取数据
    @classmethod
    def _read_csv(cls, filename):
        # 打开文件
        with open(filename) as csv:
            # 读取种子并解析为整数列表
            seed = csv.readline()
            seed = seed.split(',')
            seed = [int(s.strip(), 0) for s in seed[1:]]
            # 读取数据并解析为 numpy 数组
            data = []
            for line in csv:
                data.append(int(line.split(',')[-1].strip(), 0))
            return {'seed': seed, 'data': np.array(data, dtype=cls.dtype)}
    # 测试原始随机数生成器的功能
    def test_raw(self):
        # 使用给定种子生成比特生成器
        bit_generator = self.bit_generator(*self.data1['seed'])
        # 生成指定长度的原始随机整数序列
        uints = bit_generator.random_raw(1000)
        # 断言生成的整数序列与预期数据相等
        assert_equal(uints, self.data1['data'])

        # 使用相同的种子重新生成比特生成器
        bit_generator = self.bit_generator(*self.data1['seed'])
        # 生成单个原始随机整数
        uints = bit_generator.random_raw()
        # 断言生成的单个整数与预期数据序列的第一个元素相等
        assert_equal(uints, self.data1['data'][0])

        # 使用另一组种子生成比特生成器
        bit_generator = self.bit_generator(*self.data2['seed'])
        # 生成指定长度的原始随机整数序列
        uints = bit_generator.random_raw(1000)
        # 断言生成的整数序列与预期数据相等
        assert_equal(uints, self.data2['data'])

    # 测试带有输出控制的随机原始数据生成
    def test_random_raw(self):
        # 使用给定种子生成比特生成器
        bit_generator = self.bit_generator(*self.data1['seed'])
        # 使用output=False参数生成原始随机整数序列，预期返回None
        uints = bit_generator.random_raw(output=False)
        # 断言返回值为None
        assert uints is None
        # 再次使用output=False参数生成单个原始随机整数，预期返回None
        uints = bit_generator.random_raw(1000, output=False)
        # 断言返回值为None
        assert uints is None

    # 测试生成标准正态分布随机数
    def test_gauss_inv(self):
        # 定义生成随机数的数量
        n = 25
        # 使用给定种子生成比特生成器，并初始化随机状态对象
        rs = RandomState(self.bit_generator(*self.data1['seed']))
        # 生成指定数量的标准正态分布随机数
        gauss = rs.standard_normal(n)
        # 断言生成的随机数序列与从预期数据中生成的数据经转换后的结果相等
        assert_allclose(gauss,
                        gauss_from_uint(self.data1['data'], n, self.bits))

        # 使用另一组种子生成比特生成器，并初始化随机状态对象
        rs = RandomState(self.bit_generator(*self.data2['seed']))
        # 再次生成指定数量的标准正态分布随机数
        gauss = rs.standard_normal(25)
        # 断言生成的随机数序列与从预期数据中生成的数据经转换后的结果相等
        assert_allclose(gauss,
                        gauss_from_uint(self.data2['data'], n, self.bits))

    # 测试生成均匀分布的双精度随机数
    def test_uniform_double(self):
        # 使用给定种子生成比特生成器，并初始化生成器对象
        rs = Generator(self.bit_generator(*self.data1['seed']))
        # 从预期数据生成双精度均匀分布的值
        vals = uniform_from_uint(self.data1['data'], self.bits)
        # 使用生成器对象生成与预期数据长度相等的双精度随机数序列
        uniforms = rs.random(len(vals))
        # 断言生成的随机数序列与预期的均匀分布值相等
        assert_allclose(uniforms, vals)
        # 断言生成的随机数序列的数据类型为双精度浮点数
        assert_equal(uniforms.dtype, np.float64)

        # 使用另一组种子生成比特生成器，并初始化生成器对象
        rs = Generator(self.bit_generator(*self.data2['seed']))
        # 从预期数据生成双精度均匀分布的值
        vals = uniform_from_uint(self.data2['data'], self.bits)
        # 使用生成器对象生成与预期数据长度相等的双精度随机数序列
        uniforms = rs.random(len(vals))
        # 断言生成的随机数序列与预期的均匀分布值相等
        assert_allclose(uniforms, vals)
        # 断言生成的随机数序列的数据类型为双精度浮点数
        assert_equal(uniforms.dtype, np.float64)

    # 测试生成均匀分布的单精度随机数
    def test_uniform_float(self):
        # 使用给定种子生成比特生成器，并初始化生成器对象
        rs = Generator(self.bit_generator(*self.data1['seed']))
        # 从预期数据生成单精度均匀分布的值
        vals = uniform32_from_uint(self.data1['data'], self.bits)
        # 使用生成器对象生成与预期数据长度相等的单精度随机数序列，数据类型为单精度浮点数
        uniforms = rs.random(len(vals), dtype=np.float32)
        # 断言生成的随机数序列与预期的均匀分布值相等
        assert_allclose(uniforms, vals)
        # 断言生成的随机数序列的数据类型为单精度浮点数
        assert_equal(uniforms.dtype, np.float32)

        # 使用另一组种子生成比特生成器，并初始化生成器对象
        rs = Generator(self.bit_generator(*self.data2['seed']))
        # 从预期数据生成单精度均匀分布的值
        vals = uniform32_from_uint(self.data2['data'], self.bits)
        # 使用生成器对象生成与预期数据长度相等的单精度随机数序列，数据类型为单精度浮点数
        uniforms = rs.random(len(vals), dtype=np.float32)
        # 断言生成的随机数序列与预期的均匀分布值相等
        assert_allclose(uniforms, vals)
        # 断言生成的随机数序列的数据类型为单精度浮点数
        assert_equal(uniforms.dtype, np.float32)

    # 测试生成生成器对象的字符串表示形式
    def test_repr(self):
        # 使用给定种子生成比特生成器，并初始化生成器对象
        rs = Generator(self.bit_generator(*self.data1['seed']))
        # 断言生成器对象的字符串表示中包含关键字'Generator'
        assert 'Generator' in repr(rs)
        # 断言生成器对象的字符串表示中包含生成器对象的内存地址的十六进制表示，并转换为小写字母
        assert f'{id(rs):#x}'.upper().replace('X', 'x') in repr(rs)

    # 测试生成生成器对象的字符串表示形式
    def test_str(self):
        # 使用给定种子生成比特生成器，并初始化生成器对象
        rs = Generator(self.bit_generator(*self.data1['seed']))
        # 断言生成器对象的字符串表示中包含关键字'Generator'
        assert 'Generator' in str(rs)
        # 断言生成器对象的字符串表示中包含比特生成器的名称的字符串表示形式
        assert str(self.bit_generator.__name__) in str(rs)
        # 断言生成器对象的字符串表示中不包含生成器对象的内存地址的十六进制表示，并转换为小写字母
        assert f'{id(rs):#x}'.upper().replace('X', 'x') not in str(rs)
    def test_pickle(self):
        # 导入 pickle 模块，用于对象序列化和反序列化
        import pickle

        # 使用测试数据生成比特生成器
        bit_generator = self.bit_generator(*self.data1['seed'])
        # 获取当前比特生成器的状态
        state = bit_generator.state
        # 将比特生成器对象序列化为字节流
        bitgen_pkl = pickle.dumps(bit_generator)
        # 从序列化的字节流中反序列化出对象
        reloaded = pickle.loads(bitgen_pkl)
        # 获取重新加载后的比特生成器的状态
        reloaded_state = reloaded.state
        # 断言重新加载的比特生成器产生的随机数序列与原始比特生成器一致
        assert_array_equal(Generator(bit_generator).standard_normal(1000),
                           Generator(reloaded).standard_normal(1000))
        # 断言重新加载的比特生成器对象与原始对象不是同一个对象
        assert bit_generator is not reloaded
        # 断言重新加载的比特生成器的状态与原始状态一致
        assert_state_equal(reloaded_state, state)

        # 创建一个种子序列对象
        ss = SeedSequence(100)
        # 将种子序列对象序列化为字节流并反序列化
        aa = pickle.loads(pickle.dumps(ss))
        # 断言序列化和反序列化后的种子序列状态一致
        assert_equal(ss.state, aa.state)

    def test_pickle_preserves_seed_sequence(self):
        # GH 26234
        # 添加显式测试以验证比特生成器保留种子序列
        import pickle

        # 使用测试数据生成比特生成器
        bit_generator = self.bit_generator(*self.data1['seed'])
        # 获取比特生成器的种子序列
        ss = bit_generator.seed_seq
        # 将比特生成器对象序列化为字节流并反序列化
        bg_plk = pickle.loads(pickle.dumps(bit_generator))
        # 获取反序列化后比特生成器的种子序列
        ss_plk = bg_plk.seed_seq
        # 断言序列化和反序列化后的种子序列状态一致
        assert_equal(ss.state, ss_plk.state)
        # 断言序列化和反序列化后的种子序列池（pool）一致
        assert_equal(ss.pool, ss_plk.pool)

        # 修改原始比特生成器的种子序列并生成新的比特生成器对象
        bit_generator.seed_seq.spawn(10)
        # 将新生成的比特生成器对象序列化为字节流并反序列化
        bg_plk = pickle.loads(pickle.dumps(bit_generator))
        # 获取反序列化后新生成比特生成器的种子序列
        ss_plk = bg_plk.seed_seq
        # 断言序列化和反序列化后的种子序列状态一致
        assert_equal(ss.state, ss_plk.state)
        # 断言序列化和反序列化后的种子序列已生成的子序列数目一致
        assert_equal(ss.n_children_spawned, ss_plk.n_children_spawned)

    def test_invalid_state_type(self):
        # 使用测试数据生成比特生成器
        bit_generator = self.bit_generator(*self.data1['seed'])
        # 尝试将比特生成器的状态设置为非法类型，断言会引发 TypeError 异常
        with pytest.raises(TypeError):
            bit_generator.state = {'1'}

    def test_invalid_state_value(self):
        # 使用测试数据生成比特生成器
        bit_generator = self.bit_generator(*self.data1['seed'])
        # 获取比特生成器的当前状态
        state = bit_generator.state
        # 修改状态字典中的值为非法值，断言会引发 ValueError 异常
        state['bit_generator'] = 'otherBitGenerator'
        with pytest.raises(ValueError):
            bit_generator.state = state

    def test_invalid_init_type(self):
        # 使用默认比特生成器生成对象
        bit_generator = self.bit_generator
        # 使用各种非法初始化类型，断言会引发 TypeError 异常
        for st in self.invalid_init_types:
            with pytest.raises(TypeError):
                bit_generator(*st)

    def test_invalid_init_values(self):
        # 使用默认比特生成器生成对象
        bit_generator = self.bit_generator
        # 使用各种非法初始化值，断言会引发 ValueError 或 OverflowError 异常
        for st in self.invalid_init_values:
            with pytest.raises((ValueError, OverflowError)):
                bit_generator(*st)

    def test_benchmark(self):
        # 使用测试数据生成比特生成器
        bit_generator = self.bit_generator(*self.data1['seed'])
        # 运行比特生成器的基准测试，断言会引发 ValueError 异常
        bit_generator._benchmark(1)
        bit_generator._benchmark(1, 'double')
        with pytest.raises(ValueError):
            bit_generator._benchmark(1, 'int32')

    @pytest.mark.skipif(MISSING_CFFI, reason='cffi not available')
    def test_cffi(self):
        # 使用测试数据生成比特生成器
        bit_generator = self.bit_generator(*self.data1['seed'])
        # 获取比特生成器的 CFFI 接口对象
        cffi_interface = bit_generator.cffi
        # 断言 CFFI 接口对象是 interface 类型的实例
        assert isinstance(cffi_interface, interface)
        # 再次获取比特生成器的 CFFI 接口对象，断言对象是同一个对象
        other_cffi_interface = bit_generator.cffi
        assert other_cffi_interface is cffi_interface

    @pytest.mark.skipif(MISSING_CTYPES, reason='ctypes not available')
    # 定义一个测试方法，测试 ctypes 接口的功能
    def test_ctypes(self):
        # 使用预定义的种子数据初始化比特生成器对象
        bit_generator = self.bit_generator(*self.data1['seed'])
        # 获取比特生成器对象的 ctypes 接口
        ctypes_interface = bit_generator.ctypes
        # 断言 ctypes 接口是 interface 类型的实例
        assert isinstance(ctypes_interface, interface)
        # 再次获取比特生成器对象的 ctypes 接口
        other_ctypes_interface = bit_generator.ctypes
        # 断言两次获取的 ctypes 接口是同一个对象
        assert other_ctypes_interface is ctypes_interface

    # 定义另一个测试方法，测试 __getstate__ 方法的功能
    def test_getstate(self):
        # 使用预定义的种子数据初始化比特生成器对象
        bit_generator = self.bit_generator(*self.data1['seed'])
        # 获取比特生成器对象的状态
        state = bit_generator.state
        # 调用 __getstate__ 方法获取备用状态
        alt_state = bit_generator.__getstate__()
        # 断言备用状态是一个元组
        assert isinstance(alt_state, tuple)
        # 断言比特生成器对象的状态与备用状态的第一个元素相等
        assert_state_equal(state, alt_state[0])
        # 断言备用状态的第二个元素是 SeedSequence 类型的实例
        assert isinstance(alt_state[1], SeedSequence)
# 定义一个测试类 TestPhilox，继承自 Base 类
class TestPhilox(Base):
    
    # 类方法，用于设置测试类的初始状态
    @classmethod
    def setup_class(cls):
        # 指定使用 Philox 作为随机数生成器
        cls.bit_generator = Philox
        # 指定生成的随机数位数为 64
        cls.bits = 64
        # 指定数据类型为 64 位无符号整数
        cls.dtype = np.uint64
        # 读取并设置第一个 CSV 文件的数据
        cls.data1 = cls._read_csv(join(pwd, './data/philox-testset-1.csv'))
        # 读取并设置第二个 CSV 文件的数据
        cls.data2 = cls._read_csv(join(pwd, './data/philox-testset-2.csv'))
        # 设置 seed_error_type 为 TypeError，表示 seed 错误类型为 TypeError
        cls.seed_error_type = TypeError
        # 初始化无效的初始化类型为空列表
        cls.invalid_init_types = []
        # 初始化无效的初始化数值列表，包括超出范围的值
        cls.invalid_init_values = [(1, None, 1), (-1,), (None, None, 2 ** 257 + 1)]

    # 测试方法，测试设置密钥
    def test_set_key(self):
        # 使用 data1 中的种子数据初始化 bit_generator
        bit_generator = self.bit_generator(*self.data1['seed'])
        # 获取当前 bit_generator 的状态
        state = bit_generator.state
        # 使用 bit_generator 的状态中的 counter 和 key 创建新的 bit_generator
        keyed = self.bit_generator(counter=state['state']['counter'],
                                   key=state['state']['key'])
        # 断言 bit_generator 的状态和 keyed 的状态相等
        assert_state_equal(bit_generator.state, keyed.state)


# 定义一个测试类 TestPCG64，继承自 Base 类
class TestPCG64(Base):
    
    # 类方法，用于设置测试类的初始状态
    @classmethod
    def setup_class(cls):
        # 指定使用 PCG64 作为随机数生成器
        cls.bit_generator = PCG64
        # 指定生成的随机数位数为 64
        cls.bits = 64
        # 指定数据类型为 64 位无符号整数
        cls.dtype = np.uint64
        # 读取并设置第一个 CSV 文件的数据
        cls.data1 = cls._read_csv(join(pwd, './data/pcg64-testset-1.csv'))
        # 读取并设置第二个 CSV 文件的数据
        cls.data2 = cls._read_csv(join(pwd, './data/pcg64-testset-2.csv'))
        # 设置 seed_error_type 为 ValueError 或 TypeError，表示 seed 错误类型为这两者之一
        cls.seed_error_type = (ValueError, TypeError)
        # 初始化无效的初始化类型列表
        cls.invalid_init_types = [(3.2,), ([None],), (1, None)]
        # 初始化无效的初始化数值列表，包括负数值
        cls.invalid_init_values = [(-1,)]

    # 测试方法，测试 advance 方法的对称性
    def test_advance_symmetry(self):
        # 使用 data1 中的种子数据初始化 Generator 对象
        rs = Generator(self.bit_generator(*self.data1['seed']))
        # 获取当前随机数生成器的状态
        state = rs.bit_generator.state
        # 设置步长为 -0x9e3779b97f4a7c150000000000000000，执行 advance 方法
        rs.bit_generator.advance(-0x9e3779b97f4a7c150000000000000000)
        # 生成一个小整数
        val_neg = rs.integers(10)
        # 恢复初始状态
        rs.bit_generator.state = state
        # 设置步长为 2**128 + (-0x9e3779b97f4a7c150000000000000000)，再次执行 advance 方法
        rs.bit_generator.advance(2**128 + -0x9e3779b97f4a7c150000000000000000)
        # 生成一个小整数
        val_pos = rs.integers(10)
        # 恢复初始状态
        rs.bit_generator.state = state
        # 设置步长为 10 * 2**128 + (-0x9e3779b97f4a7c150000000000000000)，再次执行 advance 方法
        rs.bit_generator.advance(10 * 2**128 + -0x9e3779b97f4a7c150000000000000000)
        # 生成一个大整数
        val_big = rs.integers(10)
        # 断言三次生成的随机数相等
        assert val_neg == val_pos
        assert val_big == val_pos

    # 测试方法，测试 advance 方法对大数值的处理
    def test_advange_large(self):
        # 使用固定的种子初始化 Generator 对象
        rs = Generator(self.bit_generator(38219308213743))
        # 获取随机数生成器对象
        pcg = rs.bit_generator
        # 获取当前状态
        state = pcg.state["state"]
        # 设置初始状态为特定值
        initial_state = 287608843259529770491897792873167516365
        # 断言当前状态与初始状态相等
        assert state["state"] == initial_state
        # 执行 advance 方法，步长为 2**i 的和，i 为 (96, 64, 32, 16, 8, 4, 2, 1)
        pcg.advance(sum(2**i for i in (96, 64, 32, 16, 8, 4, 2, 1)))
        # 获取当前状态
        state = pcg.state["state"]
        # 设置预期的高级状态值
        advanced_state = 135275564607035429730177404003164635391
        # 断言当前状态与预期的高级状态值相等
        assert state["state"] == advanced_state


# 定义一个测试类 TestPCG64DXSM，继承自 Base 类
class TestPCG64DXSM(Base):
    
    # 类方法，用于设置测试类的初始状态
    @classmethod
    def setup_class(cls):
        # 指定使用 PCG64DXSM 作为随机数生成器
        cls.bit_generator = PCG64DXSM
        # 指定生成的随机数位数为 64
        cls.bits = 64
        # 指定数据类型为 64 位无符号整数
        cls.dtype = np.uint64
        # 读取并设置第一个 CSV 文件的数据
        cls.data1 = cls._read_csv(join(pwd, './data/pcg64dxsm-testset-1.csv'))
        # 读取并设置第二个 CSV 文件的数据
        cls.data2 = cls._read_csv(join(pwd, './data/pcg64dxsm-testset-2.csv'))
        # 设置 seed_error_type 为 ValueError 或 TypeError，表示 seed 错误类型为这两者之一
        cls.seed_error_type = (ValueError, TypeError)
        # 初始化无效的初始化类型列表
        cls.invalid_init_types = [(3.2,), ([None],), (1, None)]
        # 初始化无效的初始化数值列表，包括负数值
        cls.invalid_init_values = [(-1,)]
    # 测试函数，验证随机数生成器在使用 advance 方法后的行为
    def test_advance_symmetry(self):
        # 根据给定的种子生成随机数生成器对象
        rs = Generator(self.bit_generator(*self.data1['seed']))
        # 获取当前随机数生成器的状态
        state = rs.bit_generator.state
        # 设置步长为负的常数，使用 advance 方法使随机数生成器状态向前推进
        step = -0x9e3779b97f4a7c150000000000000000
        rs.bit_generator.advance(step)
        # 生成一个范围在 0 到 9 之间的整数
        val_neg = rs.integers(10)
        # 恢复随机数生成器到之前的状态
        rs.bit_generator.state = state
        # 使用一个非常大的步长，再次调用 advance 方法推进随机数生成器状态
        rs.bit_generator.advance(2**128 + step)
        # 再次生成一个范围在 0 到 9 之间的整数
        val_pos = rs.integers(10)
        # 恢复随机数生成器到之前的状态
        rs.bit_generator.state = state
        # 使用一个更大的步长推进随机数生成器状态
        rs.bit_generator.advance(10 * 2**128 + step)
        # 再次生成一个范围在 0 到 9 之间的整数
        val_big = rs.integers(10)
        # 断言，验证使用不同步长后生成的随机数相等
        assert val_neg == val_pos
        assert val_big == val_pos

    # 测试函数，验证在给定大步长的情况下随机数生成器的 advance 方法行为
    def test_advange_large(self):
        # 根据指定的种子生成随机数生成器对象
        rs = Generator(self.bit_generator(38219308213743))
        # 获取随机数生成器对象
        pcg = rs.bit_generator
        # 获取当前随机数生成器的状态
        state = pcg.state
        # 初始状态的预期值
        initial_state = 287608843259529770491897792873167516365
        # 断言，验证当前状态与预期的初始状态相符
        assert state["state"]["state"] == initial_state
        # 使用一个大的步长，推进随机数生成器状态
        pcg.advance(sum(2**i for i in (96, 64, 32, 16, 8, 4, 2, 1)))
        # 获取推进后的状态
        state = pcg.state["state"]
        # 预期的推进后的状态值
        advanced_state = 277778083536782149546677086420637664879
        # 断言，验证推进后的状态与预期的状态相符
        assert state["state"] == advanced_state
# 定义一个测试类 TestMT19937，继承自 Base 类
class TestMT19937(Base):
    # 类方法：设置测试类的初始状态
    @classmethod
    def setup_class(cls):
        # 设定比特生成器为 MT19937
        cls.bit_generator = MT19937
        # 比特数设定为 32
        cls.bits = 32
        # 数据类型设定为 np.uint32
        cls.dtype = np.uint32
        # 从 CSV 文件中读取数据作为 cls.data1 和 cls.data2
        cls.data1 = cls._read_csv(join(pwd, './data/mt19937-testset-1.csv'))
        cls.data2 = cls._read_csv(join(pwd, './data/mt19937-testset-2.csv'))
        # 设定 seed 错误类型为 ValueError
        cls.seed_error_type = ValueError
        # 设定无效初始化类型为空列表
        cls.invalid_init_types = []
        # 设定无效初始化数值为 [(-1,)]
        cls.invalid_init_values = [(-1,)]

    # 测试方法：测试使用浮点数数组作为种子时是否会引发 TypeError 异常
    def test_seed_float_array(self):
        assert_raises(TypeError, self.bit_generator, np.array([np.pi]))
        assert_raises(TypeError, self.bit_generator, np.array([-np.pi]))
        assert_raises(TypeError, self.bit_generator, np.array([np.pi, -np.pi]))
        assert_raises(TypeError, self.bit_generator, np.array([0, np.pi]))
        assert_raises(TypeError, self.bit_generator, [np.pi])
        assert_raises(TypeError, self.bit_generator, [0, np.pi])

    # 测试方法：测试使用元组作为状态时的功能
    def test_state_tuple(self):
        # 使用 seed 数据初始化 Generator 对象 rs
        rs = Generator(self.bit_generator(*self.data1['seed']))
        # 获取 bit_generator 属性
        bit_generator = rs.bit_generator
        # 获取状态信息
        state = bit_generator.state
        # 生成一个随机数
        desired = rs.integers(2 ** 16)
        # 构建元组 tup 包含 bit_generator、state['state']['key'] 和 state['state']['pos']
        tup = (state['bit_generator'], state['state']['key'],
               state['state']['pos'])
        # 设置 bit_generator 的状态为 tup
        bit_generator.state = tup
        # 再次生成随机数
        actual = rs.integers(2 ** 16)
        # 断言生成的随机数与期望值相等
        assert_equal(actual, desired)
        # 将 tup 扩展为 tup + (0, 0.0)
        tup = tup + (0, 0.0)
        # 设置 bit_generator 的状态为扩展后的 tup
        bit_generator.state = tup
        # 再次生成随机数
        actual = rs.integers(2 ** 16)
        # 断言生成的随机数与期望值相等
        assert_equal(actual, desired)


# 定义一个测试类 TestSFC64，继承自 Base 类
class TestSFC64(Base):
    # 类方法：设置测试类的初始状态
    @classmethod
    def setup_class(cls):
        # 设定比特生成器为 SFC64
        cls.bit_generator = SFC64
        # 比特数设定为 64
        cls.bits = 64
        # 数据类型设定为 np.uint64
        cls.dtype = np.uint64
        # 从 CSV 文件中读取数据作为 cls.data1 和 cls.data2
        cls.data1 = cls._read_csv(
            join(pwd, './data/sfc64-testset-1.csv'))
        cls.data2 = cls._read_csv(
            join(pwd, './data/sfc64-testset-2.csv'))
        # 设定 seed 错误类型为 (ValueError, TypeError)
        cls.seed_error_type = (ValueError, TypeError)
        # 设定无效初始化类型为 [(3.2,), ([None],), (1, None)]
        cls.invalid_init_types = [(3.2,), ([None],), (1, None)]
        # 设定无效初始化数值为 [(-1,)]
        cls.invalid_init_values = [(-1,)]

    # 测试方法：测试旧版本 pickle 格式的兼容性
    def test_legacy_pickle(self):
        # 导入 gzip 和 pickle 模块
        import gzip
        import pickle

        # 期望的状态数据
        expected_state = np.array(
            [
                9957867060933711493,
                532597980065565856,
                14769588338631205282,
                13
            ],
            dtype=np.uint64
        )

        # 获取当前文件路径的基础路径
        base_path = os.path.split(os.path.abspath(__file__))[0]
        # 构建 pickle 文件的完整路径
        pkl_file = os.path.join(base_path, "data", f"sfc64_np126.pkl.gz")
        # 使用 gzip 打开 pickle 文件
        with gzip.open(pkl_file) as gz:
            # 从 pickle 文件中加载 SFC64 对象
            sfc = pickle.load(gz)

        # 断言 sfc 是 SFC64 类的实例
        assert isinstance(sfc, SFC64)
        # 断言 sfc 的状态中的 state['state']['state'] 与 expected_state 相等
        assert_equal(sfc.state["state"]["state"], expected_state)


# 定义一个测试类 TestDefaultRNG
class TestDefaultRNG:
    # 测试方法：测试 default_rng 函数的种子设置
    def test_seed(self):
        # 遍历不同的参数组合进行测试
        for args in [(), (None,), (1234,), ([1234, 5678],)]:
            # 调用 default_rng 函数
            rg = default_rng(*args)
            # 断言 rg 的比特生成器是 PCG64 类的实例
            assert isinstance(rg.bit_generator, PCG64)
    # 定义一个测试方法 `test_passthrough`
    def test_passthrough(self):
        # 创建一个 Philox 随机数生成器实例并赋值给 `bg`
        bg = Philox()
        # 使用默认随机数生成器 `default_rng`，传入 `bg` 作为参数创建 `rg`
        rg = default_rng(bg)
        # 断言 `rg` 的 `bit_generator` 属性确实是 `bg`
        assert rg.bit_generator is bg
        # 使用 `rg` 作为参数再次调用 `default_rng` 创建 `rg2`
        rg2 = default_rng(rg)
        # 断言 `rg2` 和 `rg` 是同一个对象
        assert rg2 is rg
        # 断言 `rg2` 的 `bit_generator` 属性也是 `bg`
        assert rg2.bit_generator is bg
```