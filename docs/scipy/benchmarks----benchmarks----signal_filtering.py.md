# `D:\src\scipysrc\scipy\benchmarks\benchmarks\signal_filtering.py`

```
import numpy as np  # 导入NumPy库，用于数值计算
import timeit  # 导入timeit库，用于性能测试
from concurrent.futures import ThreadPoolExecutor, wait  # 导入线程池和等待方法

from .common import Benchmark, safe_import  # 导入Benchmark类和safe_import函数

with safe_import():  # 使用安全导入上下文管理器，确保导入成功
    from scipy.signal import (lfilter, firwin, decimate, butter, sosfilt,
                              medfilt2d, freqz)  # 导入SciPy信号处理模块的相关函数

class Decimate(Benchmark):
    param_names = ['q', 'ftype', 'zero_phase']  # 参数名列表
    params = [
        [2, 10, 30],  # q参数的值列表
        ['iir', 'fir'],  # ftype参数的值列表
        [True, False]  # zero_phase参数的值列表
    ]

    def setup(self, q, ftype, zero_phase):
        np.random.seed(123456)  # 设定随机数种子
        sample_rate = 10000.  # 采样率
        t = np.arange(int(1e6), dtype=np.float64) / sample_rate  # 生成时间向量
        self.sig = np.sin(2*np.pi*500*t) + 0.3 * np.sin(2*np.pi*4e3*t)  # 生成示例信号

    def time_decimate(self, q, ftype, zero_phase):
        decimate(self.sig, q, ftype=ftype, zero_phase=zero_phase)  # 执行信号下采样操作


class Lfilter(Benchmark):
    param_names = ['n_samples', 'numtaps']  # 参数名列表
    params = [
        [1e3, 50e3, 1e6],  # n_samples参数的值列表
        [9, 23, 51]  # numtaps参数的值列表
    ]

    def setup(self, n_samples, numtaps):
        np.random.seed(125678)  # 设定随机数种子
        sample_rate = 25000.  # 采样率
        t = np.arange(n_samples, dtype=np.float64) / sample_rate  # 生成时间向量
        nyq_rate = sample_rate / 2.  # Nyquist频率
        cutoff_hz = 3000.0  # 截止频率
        self.sig = np.sin(2*np.pi*500*t) + 0.3 * np.sin(2*np.pi*11e3*t)  # 生成示例信号
        self.coeff = firwin(numtaps, cutoff_hz/nyq_rate)  # 设计FIR滤波器系数

    def time_lfilter(self, n_samples, numtaps):
        lfilter(self.coeff, 1.0, self.sig)  # 执行FIR滤波操作


class ParallelSosfilt(Benchmark):
    timeout = 100  # 超时时间
    timer = timeit.default_timer  # 计时器

    param_names = ['n_samples', 'threads']  # 参数名列表
    params = [
        [1e3, 10e3],  # n_samples参数的值列表
        [1, 2, 4]  # threads参数的值列表
    ]

    def setup(self, n_samples, threads):
        self.filt = butter(8, 8e-6, "lowpass", output="sos")  # 设计低通滤波器
        self.data = np.arange(int(n_samples) * 3000).reshape(int(n_samples), 3000)  # 生成示例数据
        self.chunks = np.array_split(self.data, threads)  # 将数据分割成多个部分

    def time_sosfilt(self, n_samples, threads):
        with ThreadPoolExecutor(max_workers=threads) as pool:
            futures = []
            for i in range(threads):
                futures.append(pool.submit(sosfilt, self.filt, self.chunks[i]))  # 提交sosfilt任务

            wait(futures)  # 等待所有任务完成


class Sosfilt(Benchmark):
    param_names = ['n_samples', 'order']  # 参数名列表
    params = [
        [1000, 1000000],  # n_samples参数的值列表
        [6, 20]  # order参数的值列表
    ]

    def setup(self, n_samples, order):
        self.sos = butter(order, [0.1575, 0.1625], 'band', output='sos')  # 设计带通滤波器
        self.y = np.random.RandomState(0).randn(n_samples)  # 生成随机信号

    def time_sosfilt_basic(self, n_samples, order):
        sosfilt(self.sos, self.y)  # 执行sosfilt基本操作


class MedFilt2D(Benchmark):
    param_names = ['threads']  # 参数名列表
    params = [[1, 2, 4]]  # threads参数的值列表

    def setup(self, threads):
        rng = np.random.default_rng(8176)  # 创建随机数生成器
        self.chunks = np.array_split(rng.standard_normal((250, 349)), threads)  # 将二维数据分割成多个部分

    def _medfilt2d(self, threads):
        with ThreadPoolExecutor(max_workers=threads) as pool:
            wait({pool.submit(medfilt2d, chunk, 5) for chunk in self.chunks})  # 提交medfilt2d任务并等待完成

    def time_medfilt2d(self, threads):
        self._medfilt2d(threads)  # 执行二维中值滤波操作
    # 定义一个方法 `peakmem_medfilt2d`，用于执行二维中值滤波操作，接受一个参数 `threads`
    def peakmem_medfilt2d(self, threads):
        # 调用对象的 `_medfilt2d` 方法，传递 `threads` 参数
        self._medfilt2d(threads)
# 定义一个类 FreqzRfft，继承自 Benchmark 类，用于频率响应计算的基准测试
class FreqzRfft(Benchmark):
    # 参数名称列表，包括 'whole'（是否计算整个频谱）、'nyquist'（是否包括 Nyquist 频率）、'worN'（数据点数）
    param_names = ['whole', 'nyquist', 'worN']
    # 参数取值列表：
    # - 'whole': 包括 False（不计算整个频谱）、True（计算整个频谱）
    # - 'nyquist': 包括 False（不包括 Nyquist 频率）、True（包括 Nyquist 频率）
    # - 'worN': 一系列数据点数，用于频率响应计算
    params = [
        [False, True],
        [False, True],
        [64, 65, 128, 129, 256, 257, 258, 512, 513, 65536, 65537, 65538],
    ]

    # 设置方法，在每个测试之前调用，初始化 self.y 数组
    def setup(self, whole, nyquist, worN):
        # 初始化一个长度为 worN 的零数组
        self.y = np.zeros(worN)
        # 将 self.y 数组的中间位置赋值为 1.0
        self.y[worN//2] = 1.0

    # time_freqz 方法，用于执行频率响应计算的基准测试
    def time_freqz(self, whole, nyquist, worN):
        # 调用 freqz 函数计算频率响应
        freqz(self.y, whole=whole, include_nyquist=nyquist, worN=worN)
```