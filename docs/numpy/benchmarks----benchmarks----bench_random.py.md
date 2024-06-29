# `.\numpy\benchmarks\benchmarks\bench_random.py`

```
from .common import Benchmark  # 导入 Benchmark 类

import numpy as np  # 导入 NumPy 库

try:
    from numpy.random import Generator  # 尝试导入 Generator 类
except ImportError:
    pass  # 如果导入失败则跳过

class Random(Benchmark):  # 定义 Random 类，继承自 Benchmark 类
    params = ['normal', 'uniform', 'weibull 1', 'binomial 10 0.5',  # 参数列表
              'poisson 10']

    def setup(self, name):  # 设置方法，初始化函数
        items = name.split()  # 将参数名按空格分割成列表
        name = items.pop(0)  # 取出第一个参数作为名称
        params = [float(x) for x in items]  # 将剩余参数转换为浮点数列表

        self.func = getattr(np.random, name)  # 获取 NumPy.random 中的函数对象
        self.params = tuple(params) + ((100, 100),)  # 设置函数调用的参数列表

    def time_rng(self, name):  # 定义时间测量方法
        self.func(*self.params)  # 调用 NumPy.random 中的函数

class Shuffle(Benchmark):  # 定义 Shuffle 类，继承自 Benchmark 类
    def setup(self):  # 设置方法，初始化函数
        self.a = np.arange(100000)  # 创建包含 100000 个元素的数组

    def time_100000(self):  # 定义时间测量方法
        np.random.shuffle(self.a)  # 对数组进行乱序操作

class Randint(Benchmark):  # 定义 Randint 类，继承自 Benchmark 类

    def time_randint_fast(self):  # 快速随机整数生成方法
        """Compare to uint32 below"""  # 进行 uint32 的比较
        np.random.randint(0, 2**30, size=10**5)  # 生成 10^5 个范围在 [0, 2^30) 的随机整数

    def time_randint_slow(self):  # 慢速随机整数生成方法
        """Compare to uint32 below"""  # 进行 uint32 的比较
        np.random.randint(0, 2**30 + 1, size=10**5)  # 生成 10^5 个范围在 [0, 2^30 + 1) 的随机整数

class Randint_dtype(Benchmark):  # 定义 Randint_dtype 类，继承自 Benchmark 类
    high = {  # 设置不同数据类型的最大值字典
        'bool': 1,
        'uint8': 2**7,
        'uint16': 2**15,
        'uint32': 2**31,
        'uint64': 2**63
    }

    param_names = ['dtype']  # 参数名称列表
    params = ['bool', 'uint8', 'uint16', 'uint32', 'uint64']  # 参数值列表

    def setup(self, name):  # 设置方法，初始化函数
        from numpy.lib import NumpyVersion  # 导入 NumpyVersion 类
        if NumpyVersion(np.__version__) < '1.11.0.dev0':  # 检查 NumPy 版本是否符合条件
            raise NotImplementedError  # 如果版本不符合则抛出错误

    def time_randint_fast(self, name):  # 快速随机整数生成方法
        high = self.high[name]  # 获取指定数据类型的最大值
        np.random.randint(0, high, size=10**5, dtype=name)  # 生成 10^5 个指定数据类型的随机整数

    def time_randint_slow(self, name):  # 慢速随机整数生成方法
        high = self.high[name]  # 获取指定数据类型的最大值
        np.random.randint(0, high + 1, size=10**5, dtype=name)  # 生成 10^5 个指定数据类型的随机整数

class Permutation(Benchmark):  # 定义 Permutation 类，继承自 Benchmark 类
    def setup(self):  # 设置方法，初始化函数
        self.n = 10000  # 设置整数 n 的值
        self.a_1d = np.random.random(self.n)  # 生成包含 n 个随机数的一维数组
        self.a_2d = np.random.random((self.n, 2))  # 生成包含 n 行 2 列的随机数二维数组

    def time_permutation_1d(self):  # 测量一维数组乱序操作的时间
        np.random.permutation(self.a_1d)  # 对一维数组进行乱序操作

    def time_permutation_2d(self):  # 测量二维数组乱序操作的时间
        np.random.permutation(self.a_2d)  # 对二维数组进行乱序操作

    def time_permutation_int(self):  # 测量整数乱序操作的时间
        np.random.permutation(self.n)  # 对整数 n 进行乱序操作

nom_size = 100000  # 设置 nom_size 变量的值

class RNG(Benchmark):  # 定义 RNG 类，继承自 Benchmark 类
    param_names = ['rng']  # 参数名称列表
    params = ['PCG64', 'MT19937', 'Philox', 'SFC64', 'numpy']  # 参数值列表

    def setup(self, bitgen):  # 设置方法，初始化函数
        if bitgen == 'numpy':  # 如果参数为 'numpy'
            self.rg = np.random.RandomState()  # 创建 NumPy 随机状态对象
        else:
            self.rg = Generator(getattr(np.random, bitgen)())  # 创建指定类型的 Generator 对象
        self.rg.random()  # 生成随机数
        self.int32info = np.iinfo(np.int32)  # 获取 np.int32 的信息
        self.uint32info = np.iinfo(np.uint32)  # 获取 np.uint32 的信息
        self.uint64info = np.iinfo(np.uint64)  # 获取 np.uint64 的信息

    def time_raw(self, bitgen):  # 测量原始数据生成时间的方法
        if bitgen == 'numpy':  # 如果参数为 'numpy'
            self.rg.random_integers(self.int32info.max, size=nom_size)  # 使用 NumPy.random 中的函数生成随机整数
        else:
            self.rg.integers(self.int32info.max, size=nom_size, endpoint=True)  # 使用 Generator 对象生成随机整数
    # 定义一个方法用于生成32位整数随机数
    def time_32bit(self, bitgen):
        # 从self.uint32info中获取最小值和最大值
        min, max = self.uint32info.min, self.uint32info.max
        # 如果bitgen为'numpy'，使用numpy的randint函数生成指定范围内的随机整数数组，数据类型为np.uint32
        if bitgen == 'numpy':
            self.rg.randint(min, max + 1, nom_size, dtype=np.uint32)
        # 否则使用self.rg的integers方法生成指定范围内的随机整数数组，数据类型为np.uint32
        else:
            self.rg.integers(min, max + 1, nom_size, dtype=np.uint32)

    # 定义一个方法用于生成64位整数随机数
    def time_64bit(self, bitgen):
        # 从self.uint64info中获取最小值和最大值
        min, max = self.uint64info.min, self.uint64info.max
        # 如果bitgen为'numpy'，使用numpy的randint函数生成指定范围内的随机整数数组，数据类型为np.uint64
        if bitgen == 'numpy':
            self.rg.randint(min, max + 1, nom_size, dtype=np.uint64)
        # 否则使用self.rg的integers方法生成指定范围内的随机整数数组，数据类型为np.uint64
        else:
            self.rg.integers(min, max + 1, nom_size, dtype=np.uint64)

    # 定义一个方法用于生成标准正态分布的随机数
    def time_normal_zig(self, bitgen):
        # 使用self.rg的standard_normal方法生成符合标准正态分布的随机数数组，数组大小为nom_size
        self.rg.standard_normal(nom_size)
# 定义一个继承自Benchmark的Bounded类
class Bounded(Benchmark):
    # 定义不同精度的无符号整数类型
    u8 = np.uint8
    u16 = np.uint16
    u32 = np.uint32
    u64 = np.uint64
    # 定义参数名列表
    param_names = ['rng', 'dt_max']
    # 定义参数组合
    params = [['PCG64', 'MT19937', 'Philox', 'SFC64', 'numpy'],
              [[u8,    95],  # 8位最差情况
               [u8,    64],  # 8位遗留最坏情况
               [u8,   127],  # 8位遗留最佳情况
               [u16,   95],  # 16位最差情况
               [u16, 1024],  # 16位遗留最坏情况
               [u16, 1535],  # 16位遗留典型平均情况
               [u16, 2047],  # 16位遗留最佳情况
               [u32, 1024],  # 32位遗留最坏情况
               [u32, 1535],  # 32位遗留典型平均情况
               [u32, 2047],  # 32位遗留最佳情况
               [u64,   95],  # 64位最差情况
               [u64, 1024],  # 64位遗留最坏情况
               [u64, 1535],  # 64位遗留典型平均情况
               [u64, 2047],  # 64位遗留最佳情况
             ]]

    # 初始化设置方法
    def setup(self, bitgen, args):
        # 设置种子值
        seed = 707250673
        # 根据不同的bitgen参数选择不同的随机数生成器
        if bitgen == 'numpy':
            self.rg = np.random.RandomState(seed)
        else:
            self.rg = Generator(getattr(np.random, bitgen)(seed))
        # 生成随机数
        self.rg.random()

    # 定义时间测量方法，用于测量有界值的计时器
    def time_bounded(self, bitgen, args):
            """
            Timer for 8-bit bounded values.

            Parameters (packed as args)
            ----------
            dt : {uint8, uint16, uint32, unit64}
                output dtype
            max : int
                Upper bound for range. Lower is always 0.  Must be <= 2**bits.
            """
            # 解包参数
            dt, max = args
            # 根据不同的bitgen参数选择不同的随机整数生成方法
            if bitgen == 'numpy':
                self.rg.randint(0, max + 1, nom_size, dtype=dt)
            else:
                self.rg.integers(0, max + 1, nom_size, dtype=dt)

# 定义一个继承自Benchmark的Choice类
class Choice(Benchmark):
    # 参数列表
    params = [1e3, 1e6, 1e8]

    # 初始化设置方法
    def setup(self, v):
        # 生成一个长度为v的数组
        self.a = np.arange(v)
        # 使用numpy默认的随机数生成器
        self.rng = np.random.default_rng()

    # 测量遗留选择方法的时间
    def time_legacy_choice(self, v):
        np.random.choice(self.a, 1000, replace=False)

    # 测量选择方法的时间
    def time_choice(self, v):
        self.rng.choice(self.a, 1000, replace=False)
```