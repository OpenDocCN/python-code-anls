# `D:\src\scipysrc\scipy\benchmarks\benchmarks\fft_basic.py`

```
""" Test functions for fftpack.basic module
"""
# 导入必要的库和模块
from numpy import arange, asarray, zeros, dot, exp, pi, double, cdouble
from numpy.random import rand
import numpy as np
from concurrent import futures
import os

import scipy.fftpack
import numpy.fft
from .common import Benchmark, safe_import

# 使用 safe_import() 上下文管理器导入 scipy.fft
with safe_import() as exc:
    import scipy.fft as scipy_fft
    has_scipy_fft = True
if exc.error:
    has_scipy_fft = False

# 使用 safe_import() 上下文管理器导入 pyfftw.interfaces.numpy_fft 和 pyfftw
with safe_import() as exc:
    import pyfftw.interfaces.numpy_fft as pyfftw_fft
    import pyfftw
    pyfftw.interfaces.cache.enable()
    has_pyfftw = True
if exc.error:
    pyfftw_fft = {}  # noqa: F811
    has_pyfftw = False

# 定义 PyfftwBackend 类作为 pyfftw 的后端
class PyfftwBackend:
    """Backend for pyfftw"""
    __ua_domain__ = 'numpy.scipy.fft'

    @staticmethod
    def __ua_function__(method, args, kwargs):
        kwargs.pop('overwrite_x', None)

        fn = getattr(pyfftw_fft, method.__name__, None)
        return (NotImplemented if fn is None
                else fn(*args, **kwargs))

# 定义生成随机数组的函数 random
def random(size):
    return rand(*size)

# 定义直接离散傅立叶变换函数 direct_dft
def direct_dft(x):
    x = asarray(x)
    n = len(x)
    y = zeros(n, dtype=cdouble)
    w = -arange(n)*(2j*pi/n)
    for i in range(n):
        y[i] = dot(exp(i*w), x)
    return y

# 定义直接逆离散傅立叶变换函数 direct_idft
def direct_idft(x):
    x = asarray(x)
    n = len(x)
    y = zeros(n, dtype=cdouble)
    w = arange(n)*(2j*pi/n)
    for i in range(n):
        y[i] = dot(exp(i*w), x)/n
    return y

# 定义根据模块名获取对应模块的函数 get_module
def get_module(mod_name):
    module_map = {
        'scipy.fftpack': scipy.fftpack,
        'scipy.fft': scipy_fft,
        'numpy.fft': numpy.fft
    }

    if not has_scipy_fft and mod_name == 'scipy.fft':
        raise NotImplementedError

    return module_map[mod_name]

# 定义 Fft 类作为基准测试类
class Fft(Benchmark):
    params = [
        [100, 256, 313, 512, 1000, 1024, 2048, 2048*2, 2048*4],
        ['real', 'cmplx'],
        ['scipy.fftpack', 'scipy.fft', 'numpy.fft']
    ]
    param_names = ['size', 'type', 'module']

    def setup(self, size, cmplx, module):
        if cmplx == 'cmplx':
            self.x = random([size]).astype(cdouble)+random([size]).astype(cdouble)*1j
        else:
            self.x = random([size]).astype(double)

        module = get_module(module)
        self.fft = getattr(module, 'fft')
        self.ifft = getattr(module, 'ifft')

    def time_fft(self, size, cmplx, module):
        self.fft(self.x)

    def time_ifft(self, size, cmplx, module):
        self.ifft(self.x)

# 定义 NextFastLen 类作为下一个快速长度的基准测试类
class NextFastLen(Benchmark):
    params = [
        [12, 13,  # small ones
         1021, 1024,  # 2 ** 10 and a prime
         16381, 16384,  # 2 ** 14 and a prime
         262139, 262144,  # 2 ** 17 and a prime
         999983, 1048576,  # 2 ** 20 and a prime
         ],
    ]
    param_names = ['size']

    def setup(self, size):
        if not has_scipy_fft:
            raise NotImplementedError

    def time_next_fast_len(self, size):
        scipy_fft.next_fast_len.__wrapped__(size)

    def time_next_fast_len_cached(self, size):
        scipy_fft.next_fast_len(size)
class RFft(Benchmark):
    # 参数定义：包含不同大小和模块的组合
    params = [
        [100, 256, 313, 512, 1000, 1024, 2048, 2048*2, 2048*4],  # 不同的大小
        ['scipy.fftpack', 'scipy.fft', 'numpy.fft']  # 不同的模块
    ]
    param_names = ['size', 'module']  # 参数名称：大小和模块

    def setup(self, size, module):
        # 初始化随机数组并转换为双精度浮点数
        self.x = random([size]).astype(double)

        # 获取指定模块中的rfft和irfft函数
        module = get_module(module)
        self.rfft = getattr(module, 'rfft')
        self.irfft = getattr(module, 'irfft')

        # 对self.x进行傅里叶变换，并将结果存储在self.y中
        self.y = self.rfft(self.x)

    def time_rfft(self, size, module):
        # 测试rfft函数的运行时间
        self.rfft(self.x)

    def time_irfft(self, size, module):
        # 测试irfft函数的运行时间
        self.irfft(self.y)


class RealTransforms1D(Benchmark):
    # 参数定义：包含不同大小、类型和模块的组合
    params = [
        [75, 100, 135, 256, 313, 512, 675, 1024, 2025, 2048],  # 不同的大小
        ['I', 'II', 'III', 'IV'],  # 不同的类型
        ['scipy.fftpack', 'scipy.fft']  # 不同的模块
    ]
    param_names = ['size', 'type', 'module']  # 参数名称：大小、类型和模块

    def setup(self, size, type, module):
        # 获取指定模块中的dct和dst函数
        module = get_module(module)
        self.dct = getattr(module, 'dct')
        self.dst = getattr(module, 'dst')

        # 根据类型（I、II、III、IV）选择对应的转换类型
        self.type = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}[type]

        # 对于dct/dst类型1，逻辑变换大小需要调整
        if self.type == 1:
            size += 1

        # 初始化随机数组并转换为双精度浮点数
        self.x = random([size]).astype(double)

        # 对于dst类型1，初始化self.x_dst
        if self.type == 1:
            self.x_dst = self.x[:-2].copy()

    def time_dct(self, size, type, module):
        # 测试dct函数的运行时间
        self.dct(self.x, self.type)

    def time_dst(self, size, type, module):
        # 根据类型选择self.x或self.x_dst，并测试dst函数的运行时间
        x = self.x if self.type != 1 else self.x_dst
        self.dst(x, self.type)


class Fftn(Benchmark):
    # 参数定义：包含不同大小、类型和模块的组合
    params = [
        ["100x100", "313x100", "1000x100", "256x256", "512x512"],  # 不同的大小
        ['real', 'cmplx'],  # 不同的类型
        ['scipy.fftpack', 'scipy.fft', 'numpy.fft']  # 不同的模块
    ]
    param_names = ['size', 'type', 'module']  # 参数名称：大小、类型和模块

    def setup(self, size, cmplx, module):
        # 将size转换为整数列表
        size = list(map(int, size.split("x")))

        # 根据类型（real或cmplx）初始化随机数组
        if cmplx != 'cmplx':
            self.x = random(size).astype(double)
        else:
            self.x = random(size).astype(cdouble) + random(size).astype(cdouble) * 1j

        # 获取指定模块中的fftn函数
        self.fftn = getattr(get_module(module), 'fftn')

    def time_fftn(self, size, cmplx, module):
        # 测试fftn函数的运行时间
        self.fftn(self.x)


class RealTransformsND(Benchmark):
    # 参数定义：包含不同大小、类型和模块的组合
    params = [
        ['75x75', '100x100', '135x135', '313x363', '1000x100', '256x256'],  # 不同的大小
        ['I', 'II', 'III', 'IV'],  # 不同的类型
        ['scipy.fftpack', 'scipy.fft']  # 不同的模块
    ]
    param_names = ['size', 'type', 'module']  # 参数名称：大小、类型和模块

    def setup(self, size, type, module):
        # 获取指定模块中的dctn和dstn函数
        self.dctn = getattr(get_module(module), 'dctn')
        self.dstn = getattr(get_module(module), 'dstn')

        # 根据类型（I、II、III、IV）选择对应的转换类型
        self.type = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}[type]

        # 对于dct/dst类型1，逻辑变换大小需要调整
        size = list(map(int, size.split('x')))
        if self.type == 1:
            size = (s + 1 for s in size)

        # 初始化随机数组并转换为双精度浮点数
        self.x = random(size).astype(double)

        # 对于dst类型1，初始化self.x_dst
        if self.type == 1:
            self.x_dst = self.x[:-2, :-2].copy()
    # 定义一个方法 `time_dctn`，接受三个参数 `size`, `type`, `module`
    def time_dctn(self, size, type, module):
        # 调用实例的 `dctn` 方法，传入实例属性 `self.x` 和参数 `self.type`
        self.dctn(self.x, self.type)

    # 定义一个方法 `time_dstn`，接受三个参数 `size`, `type`, `module`
    def time_dstn(self, size, type, module):
        # 如果 `self.type` 不等于 1，则将实例属性 `self.x` 赋给变量 `x`，否则将 `self.x_dst` 赋给 `x`
        x = self.x if self.type != 1 else self.x_dst
        # 调用实例的 `dstn` 方法，传入变量 `x` 和参数 `self.type`
        self.dstn(x, self.type)
# 定义一个用于基准测试的 FFT 后端类，继承自 Benchmark 类
class FftBackends(Benchmark):
    # 参数列表，包含不同的大小、类型和后端选项
    params = [
        [100, 256, 313, 512, 1000, 1024, 2048, 2048*2, 2048*4],  # 不同的数据大小
        ['real', 'cmplx'],  # 数据类型：实数或复数
        ['pocketfft', 'pyfftw', 'numpy', 'direct']  # FFT 的实现后端
    ]
    # 参数名称列表，对应于 params 列表的各个维度
    param_names = ['size', 'type', 'backend']

    # 设置函数，在每次测试之前被调用，用于初始化数据和选择后端
    def setup(self, size, cmplx, backend):
        import scipy.fft

        # 根据数据类型和大小生成相应类型的随机数数组
        if cmplx == 'cmplx':
            self.x = random([size]).astype(cdouble)+random([size]).astype(cdouble)*1j
        else:
            self.x = random([size]).astype(double)

        # 设置 FFT 和 IFFT 函数，初始使用 scipy 的实现
        self.fft = scipy.fft.fft
        self.ifft = scipy.fft.ifft

        # 根据后端选项设置全局 FFT 后端
        if backend == 'pocketfft':
            scipy.fft.set_global_backend('scipy')
        elif backend == 'pyfftw':
            # 检查是否支持 pyfftw，如果不支持则抛出异常
            if not has_pyfftw:
                raise NotImplementedError
            scipy.fft.set_global_backend(PyfftwBackend)
        elif backend == 'numpy':
            # 使用 NumPy 后端替换默认的 scipy 后端
            from scipy.fft._debug_backends import NumPyBackend
            scipy.fft.set_global_backend(NumPyBackend)
        elif backend == 'direct':
            # 使用直接的 pocketfft 后端替换默认的 scipy 后端
            import scipy.fft._pocketfft
            self.fft = scipy.fft._pocketfft.fft
            self.ifft = scipy.fft._pocketfft.ifft

    # FFT 性能测试函数，测试 FFT 操作的性能
    def time_fft(self, size, cmplx, module):
        self.fft(self.x)

    # IFFT 性能测试函数，测试 IFFT 操作的性能
    def time_ifft(self, size, cmplx, module):
        self.ifft(self.x)


# 定义另一个用于基准测试的 FFTn 后端类，继承自 Benchmark 类
class FftnBackends(Benchmark):
    # 参数列表，包含不同的大小、类型和后端选项
    params = [
        ["100x100", "313x100", "1000x100", "256x256", "512x512"],  # 不同的数据大小
        ['real', 'cmplx'],  # 数据类型：实数或复数
        ['pocketfft', 'pyfftw', 'numpy', 'direct']  # FFTn 的实现后端
    ]
    # 参数名称列表，对应于 params 列表的各个维度
    param_names = ['size', 'type', 'backend']

    # 设置函数，在每次测试之前被调用，用于初始化数据和选择后端
    def setup(self, size, cmplx, backend):
        import scipy.fft
        size = list(map(int, size.split("x")))  # 将字符串形式的大小转换为整数列表

        # 根据数据类型和大小生成相应类型的随机数数组
        if cmplx == 'cmplx':
            self.x = random(size).astype(double)+random(size).astype(double)*1j
        else:
            self.x = random(size).astype(double)

        # 设置 FFTn 和 IFFTn 函数，初始使用 scipy 的实现
        self.fftn = scipy.fft.fftn
        self.ifftn = scipy.fft.ifftn

        # 根据后端选项设置全局 FFT 后端
        if backend == 'pocketfft':
            scipy.fft.set_global_backend('scipy')
        elif backend == 'pyfftw':
            # 检查是否支持 pyfftw，如果不支持则抛出异常
            if not has_pyfftw:
                raise NotImplementedError
            scipy.fft.set_global_backend(PyfftwBackend)
        elif backend == 'numpy':
            # 使用 NumPy 后端替换默认的 scipy 后端
            from scipy.fft._debug_backends import NumPyBackend
            scipy.fft.set_global_backend(NumPyBackend)
        elif backend == 'direct':
            # 使用直接的 pocketfft 后端替换默认的 scipy 后端
            import scipy.fft._pocketfft
            self.fftn = scipy.fft._pocketfft.fftn
            self.ifftn = scipy.fft._pocketfft.ifftn

    # FFTn 性能测试函数，测试 FFTn 操作的性能
    def time_fft(self, size, cmplx, module):
        self.fftn(self.x)

    # IFFTn 性能测试函数，测试 IFFTn 操作的性能
    def time_ifft(self, size, cmplx, module):
        self.ifftn(self.x)


# 定义一个用于基准测试的 FFT 多线程类，继承自 Benchmark 类
class FftThreading(Benchmark):
    # 参数列表，包含不同的大小、并行数和方法选项
    params = [
        ['100x100', '1000x100', '256x256', '512x512'],  # 不同的数据大小
        [1, 8, 32, 100],  # 并行变换的数量
        ['workers', 'threading']  # 多线程方法选项
    ]
    # 参数名称列表，对应于 params 列表的各个维度
    param_names = ['size', 'num_transforms', 'method']
    # 设置函数，用于初始化参数和方法
    def setup(self, size, num_transforms, method):
        # 如果没有导入 scipy 的 FFT 功能，则抛出未实现错误
        if not has_scipy_fft:
            raise NotImplementedError

        # 将 size 参数按照 "宽x高" 格式分割为列表，并转换为整数
        size = list(map(int, size.split("x")))
        # 生成 num_transforms 个复数数组，每个数组的元素为随机实部加虚部的复数，类型为 np.complex128
        self.xs = [(random(size)+1j*random(size)).astype(np.complex128)
                   for _ in range(num_transforms)]

        # 如果方法为 'threading'，则创建线程池，线程数量为 CPU 核心数
        if method == 'threading':
            self.pool = futures.ThreadPoolExecutor(os.cpu_count())

    # map_thread 函数，用于并行执行给定函数 func 在每个 self.xs 数组上
    def map_thread(self, func):
        f = []
        # 对每个 self.xs 数组，提交 func 函数到线程池中执行，并将 Future 对象保存到列表 f 中
        for x in self.xs:
            f.append(self.pool.submit(func, x))
        # 等待所有 Future 对象完成
        futures.wait(f)

    # time_fft 函数，根据方法执行 scipy 的 FFT 函数在每个 self.xs 数组上
    def time_fft(self, size, num_transforms, method):
        # 如果方法为 'threading'，则并行执行 scipy 的 fft 函数在每个 self.xs 数组上
        if method == 'threading':
            self.map_thread(scipy_fft.fft)
        else:
            # 否则，顺序执行 scipy 的 fft 函数在每个 self.xs 数组上
            for x in self.xs:
                scipy_fft.fft(x, workers=-1)

    # time_fftn 函数，根据方法执行 scipy 的 FFTN 函数在每个 self.xs 数组上
    def time_fftn(self, size, num_transforms, method):
        # 如果方法为 'threading'，则并行执行 scipy 的 fftn 函数在每个 self.xs 数组上
        if method == 'threading':
            self.map_thread(scipy_fft.fftn)
        else:
            # 否则，顺序执行 scipy 的 fftn 函数在每个 self.xs 数组上
            for x in self.xs:
                scipy_fft.fftn(x, workers=-1)
```