# `D:\src\scipysrc\scipy\benchmarks\benchmarks\signal.py`

```
# 导入必要的模块和函数
from itertools import product

import numpy as np
from .common import Benchmark, safe_import

# 使用安全导入上下文，引入 scipy 的信号处理模块
with safe_import():
    import scipy.signal as signal


class Resample(Benchmark):
    
    # 参数化基准测试的参数名称和值
    param_names = ['N', 'num']
    params = [[977, 9973, 2 ** 14, 2 ** 16]] * 2

    def setup(self, N, num):
        # 创建一个从 0 到 10 的 N 个点的线性空间
        x = np.linspace(0, 10, N, endpoint=False)
        # 计算一个余弦函数的值，用于后续的重采样
        self.y = np.cos(-x**2/6.0)

    def time_complex(self, N, num):
        # 测试复数数据的信号重采样
        signal.resample(self.y + 0j, num)

    def time_real(self, N, num):
        # 测试实数数据的信号重采样
        signal.resample(self.y, num)


class CalculateWindowedFFT(Benchmark):

    def setup(self):
        # 创建用于计算的长数组
        rng = np.random.default_rng(5678)
        x = rng.standard_normal(2**20)
        y = rng.standard_normal(2**20)
        self.x = x
        self.y = y

    def time_welch(self):
        # 测试 Welch 方法计算功率谱密度
        signal.welch(self.x)

    def time_csd(self):
        # 测试交叉功率谱密度的计算
        signal.csd(self.x, self.y)

    def time_periodogram(self):
        # 测试周期图的计算
        signal.periodogram(self.x)

    def time_spectrogram(self):
        # 测试频谱图的计算
        signal.spectrogram(self.x)

    def time_coherence(self):
        # 测试相干函数的计算
        signal.coherence(self.x, self.y)


class Convolve2D(Benchmark):
    # 参数化二维卷积和相关的模式和边界条件
    param_names = ['mode', 'boundary']
    params = [
        ['full', 'valid', 'same'],
        ['fill', 'wrap', 'symm']
    ]

    def setup(self, mode, boundary):
        # 创建用于计算的多组 2D 数组对
        rng = np.random.default_rng(1234)
        pairs = []
        for ma, na, mb, nb in product((8, 13, 30, 36), repeat=4):
            a = rng.standard_normal((ma, na))
            b = rng.standard_normal((mb, nb))
            pairs.append((a, b))
        self.pairs = pairs

    def time_convolve2d(self, mode, boundary):
        # 测试二维卷积的运行时间
        for a, b in self.pairs:
            if mode == 'valid':
                if b.shape[0] > a.shape[0] or b.shape[1] > a.shape[1]:
                    continue
            signal.convolve2d(a, b, mode=mode, boundary=boundary)

    def time_correlate2d(self, mode, boundary):
        # 测试二维相关的运行时间
        for a, b in self.pairs:
            if mode == 'valid':
                if b.shape[0] > a.shape[0] or b.shape[1] > a.shape[1]:
                    continue
            signal.correlate2d(a, b, mode=mode, boundary=boundary)


class FFTConvolve(Benchmark):
    # 参数化 FFT 卷积的模式和数组大小
    param_names = ['mode', 'size']
    params = [
        ['full', 'valid', 'same'],
        [(a,b) for a,b in product((1, 2, 8, 36, 60, 150, 200, 500), repeat=2)
         if b <= a]
    ]

    def setup(self, mode, size):
        # 创建用于计算的随机数组
        rng = np.random.default_rng(1234)
        self.a = rng.standard_normal(size[0])
        self.b = rng.standard_normal(size[1])

    def time_convolve2d(self, mode, size):
        # 测试 FFT 卷积的运行时间
        signal.fftconvolve(self.a, self.b, mode=mode)


class OAConvolve(Benchmark):
    # 参数化 OA 卷积的模式和数组大小
    param_names = ['mode', 'size']
    params = [
        ['full', 'valid', 'same'],
        [(a, b) for a, b in product((40, 200, 3000), repeat=2)
         if b < a]
    ]
    # 定义一个方法 `setup`，用于初始化对象的数据
    def setup(self, mode, size):
        # 使用随机数生成器创建一个新的 RNG 对象，种子为 1234
        rng = np.random.default_rng(1234)
        # 使用 RNG 对象生成一个大小为 size[0] 的标准正态分布的随机数组，并赋值给对象的属性 self.a
        self.a = rng.standard_normal(size[0])
        # 使用 RNG 对象生成一个大小为 size[1] 的标准正态分布的随机数组，并赋值给对象的属性 self.b
        self.b = rng.standard_normal(size[1])
    
    # 定义一个方法 `time_convolve2d`，用于执行二维卷积计算
    def time_convolve2d(self, mode, size):
        # 调用 signal 模块的 oaconvolve 函数，对 self.a 和 self.b 进行二维卷积计算
        # mode 参数指定卷积模式
        signal.oaconvolve(self.a, self.b, mode=mode)
class Convolve(Benchmark):
    param_names = ['mode']
    params = [
        ['full', 'valid', 'same']
    ]

    def setup(self, mode):
        rng = np.random.default_rng(1234)
        # sample a bunch of pairs of 2d arrays
        pairs = {'1d': [], '2d': []}
        # 循环生成不同长度的一维数组对
        for ma, nb in product((1, 2, 8, 13, 30, 36, 50, 75), repeat=2):
            a = rng.standard_normal(ma)
            b = rng.standard_normal(nb)
            pairs['1d'].append((a, b))

        # 循环生成不同尺寸的二维图像和卷积核对
        for n_image in [256, 512, 1024]:
            for n_kernel in [3, 5, 7]:
                x = rng.standard_normal((n_image, n_image))
                h = rng.standard_normal((n_kernel, n_kernel))
                pairs['2d'].append((x, h))
        self.pairs = pairs

    def time_convolve(self, mode):
        # 针对一维数组对进行卷积操作
        for a, b in self.pairs['1d']:
            if b.shape[0] > a.shape[0]:
                continue
            signal.convolve(a, b, mode=mode)

    def time_convolve2d(self, mode):
        # 针对二维图像和卷积核对进行二维卷积操作
        for a, b in self.pairs['2d']:
            if mode == 'valid':
                if b.shape[0] > a.shape[0] or b.shape[1] > a.shape[1]:
                    continue
            signal.convolve(a, b, mode=mode)

    def time_correlate(self, mode):
        # 针对一维数组对进行相关操作
        for a, b in self.pairs['1d']:
            if b.shape[0] > a.shape[0]:
                continue
            signal.correlate(a, b, mode=mode)

    def time_correlate2d(self, mode):
        # 针对二维图像和卷积核对进行二维相关操作
        for a, b in self.pairs['2d']:
            if mode == 'valid':
                if b.shape[0] > a.shape[0] or b.shape[1] > a.shape[1]:
                    continue
            signal.correlate(a, b, mode=mode)


class LTI(Benchmark):

    def setup(self):
        # 初始化一个线性时不变系统对象
        self.system = signal.lti(1.0, [1, 0, 1])
        # 创建一个时间向量
        self.t = np.arange(0, 100, 0.5)
        # 创建一个正弦波作为输入信号
        self.u = np.sin(2 * self.t)

    def time_lsim(self):
        # 对线性时不变系统进行仿真
        signal.lsim(self.system, self.u, self.t)

    def time_step(self):
        # 计算线性时不变系统的阶跃响应
        signal.step(self.system, T=self.t)

    def time_impulse(self):
        # 计算线性时不变系统的冲激响应
        signal.impulse(self.system, T=self.t)

    def time_bode(self):
        # 计算线性时不变系统的频率响应
        signal.bode(self.system)


class Upfirdn1D(Benchmark):
    param_names = ['up', 'down']
    params = [
        [1, 4],
        [1, 4]
    ]

    def setup(self, up, down):
        rng = np.random.default_rng(1234)
        # sample a bunch of pairs of 2d arrays
        pairs = []
        # 循环生成一维滤波器和信号对
        for nfilt in [8, ]:
            for n in [32, 128, 512, 2048]:
                h = rng.standard_normal(nfilt)
                x = rng.standard_normal(n)
                pairs.append((h, x))

        self.pairs = pairs

    def time_upfirdn1d(self, up, down):
        # 对一维信号进行上/下采样滤波
        for h, x in self.pairs:
            signal.upfirdn(h, x, up=up, down=down)


class Upfirdn2D(Benchmark):
    param_names = ['up', 'down', 'axis']
    params = [
        [1, 4],
        [1, 4],
        [0, -1],
    ]
    # 设置函数，用于初始化对象的属性
    def setup(self, up, down, axis):
        # 创建一个随机数生成器对象，默认种子为1234
        rng = np.random.default_rng(1234)
        # 初始化空列表，用于存储多组二维数组对
        pairs = []
        # 循环生成二维数组对
        for nfilt in [8, ]:
            for n in [32, 128, 512]:
                # 生成长度为nfilt的标准正态分布随机数组h
                h = rng.standard_normal(nfilt)
                # 生成形状为(n, n)的标准正态分布随机数组x
                x = rng.standard_normal((n, n))
                # 将生成的数组对(h, x)添加到pairs列表中
                pairs.append((h, x))

        # 将生成的数组对列表赋值给对象属性self.pairs
        self.pairs = pairs

    # 测试函数，用于对self.pairs中的数组对执行upfirdn操作
    def time_upfirdn2d(self, up, down, axis):
        # 遍历self.pairs中的每对(h, x)
        for h, x in self.pairs:
            # 调用信号处理库中的upfirdn函数，对输入数组x进行上下采样和卷积操作
            signal.upfirdn(h, x, up=up, down=down, axis=axis)
# 定义一个名为 FIRLS 的类，继承自 Benchmark 类
class FIRLS(Benchmark):
    # 参数名列表，包括 'n' 和 'edges'
    param_names = ['n', 'edges']
    # 参数组合列表，分别是 n 和 edges 的不同组合
    params = [
        [21, 101, 1001, 2001],     # 不同的 n 值
        [(0.1, 0.9), (0.01, 0.99)], # 不同的 edges 值
        ]

    # 定义一个方法 time_firls，用于执行 firls 函数的性能测试
    def time_firls(self, n, edges):
        # 调用 signal 模块的 firls 函数，传入参数 n、(0,) + edges + (1,) 和 [1, 1, 0, 0]
        signal.firls(n, (0,) + edges + (1,), [1, 1, 0, 0])
```