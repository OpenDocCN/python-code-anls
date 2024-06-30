# `D:\src\scipysrc\scipy\benchmarks\benchmarks\fftpack_pseudo_diffs.py`

```
""" Benchmark functions for fftpack.pseudo_diffs module
"""
# 导入必要的库函数和模块
from numpy import arange, sin, cos, pi, exp, tanh, sign
# 从本地模块中导入Benchmark类和safe_import函数
from .common import Benchmark, safe_import

# 使用安全导入上下文管理器确保安全导入
with safe_import():
    # 在安全导入环境中，从scipy.fftpack模块导入所需的函数
    from scipy.fftpack import diff, fft, ifft, tilbert, hilbert, shift, fftfreq


# 定义直接差分函数
def direct_diff(x, k=1, period=None):
    # 对输入信号x进行傅里叶变换
    fx = fft(x)
    # 获取信号长度
    n = len(fx)
    # 如果未指定周期，设置默认周期为2*pi
    if period is None:
        period = 2*pi
    # 计算频率域上的角频率
    w = fftfreq(n)*2j*pi/period*n
    # 根据k的符号选择角频率的幂次方
    if k < 0:
        w = 1 / w**k
        w[0] = 0.0
    else:
        w = w**k
    # 如果信号长度超过2000，中间频率段置零
    if n > 2000:
        w[250:n-250] = 0.0
    # 对变换后的频谱进行逆傅里叶变换并返回实部
    return ifft(w*fx).real


# 定义直接Tilbert变换函数
def direct_tilbert(x, h=1, period=None):
    # 对输入信号x进行傅里叶变换
    fx = fft(x)
    # 获取信号长度
    n = len(fx)
    # 如果未指定周期，设置默认周期为2*pi
    if period is None:
        period = 2*pi
    # 计算频率域上的角频率
    w = fftfreq(n)*h*2*pi/period*n
    # 设置第一个频率分量为1，并计算Tilbert变换
    w[0] = 1
    w = 1j/tanh(w)
    w[0] = 0j
    # 对变换后的频谱进行逆傅里叶变换并返回结果
    return ifft(w*fx)


# 定义直接Hilbert变换函数
def direct_hilbert(x):
    # 对输入信号x进行傅里叶变换
    fx = fft(x)
    # 获取信号长度
    n = len(fx)
    # 计算频率域上的角频率
    w = fftfreq(n)*n
    # 设置Hilbert变换系数为虚数单位乘以频率符号函数
    w = 1j*sign(w)
    # 对变换后的频谱进行逆傅里叶变换并返回结果
    return ifft(w*fx)


# 定义直接频移函数
def direct_shift(x, a, period=None):
    # 获取信号长度
    n = len(x)
    # 如果未指定周期，设置默认的频率域偏移量
    if period is None:
        k = fftfreq(n)*1j*n
    else:
        k = fftfreq(n)*2j*pi/period*n
    # 对信号进行频移变换并返回结果的实部
    return ifft(fft(x)*exp(k*a)).real


# 定义Bench类，继承Benchmark类，用于性能评估
class Bench(Benchmark):
    # 设置测试参数
    params = [
        [100, 256, 512, 1000, 1024, 2048, 2048*2, 2048*4],
        ['fft', 'direct'],
    ]
    # 设置参数名称
    param_names = ['size', 'type']

    # 设置测试准备函数
    def setup(self, size, type):
        # 将size参数转换为整数
        size = int(size)

        # 生成长度为size的等间隔序列作为输入信号x
        x = arange(size)*2*pi/size
        a = 1
        self.a = a
        # 根据size的大小选择不同的测试信号
        if size < 2000:
            self.f = sin(x)*cos(4*x)+exp(sin(3*x))
            self.sf = sin(x+a)*cos(4*(x+a))+exp(sin(3*(x+a)))
        else:
            self.f = sin(x)*cos(4*x)
            self.sf = sin(x+a)*cos(4*(x+a))

    # 定义差分性能测试函数
    def time_diff(self, size, soltype):
        # 根据soltype选择使用fft或直接差分函数
        if soltype == 'fft':
            diff(self.f, 3)
        else:
            direct_diff(self.f, 3)

    # 定义Tilbert变换性能测试函数
    def time_tilbert(self, size, soltype):
        # 根据soltype选择使用fft或直接Tilbert变换函数
        if soltype == 'fft':
            tilbert(self.f, 1)
        else:
            direct_tilbert(self.f, 1)

    # 定义Hilbert变换性能测试函数
    def time_hilbert(self, size, soltype):
        # 根据soltype选择使用fft或直接Hilbert变换函数
        if soltype == 'fft':
            hilbert(self.f)
        else:
            direct_hilbert(self.f)

    # 定义频移性能测试函数
    def time_shift(self, size, soltype):
        # 根据soltype选择使用fft或直接频移函数
        if soltype == 'fft':
            shift(self.f, self.a)
        else:
            direct_shift(self.f, self.a)
```