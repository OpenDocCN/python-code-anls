# `D:\src\scipysrc\scipy\scipy\signal\windows\_windows.py`

```
# 导入所需模块和库
"""The suite of window functions."""
import operator  # 导入操作符模块
import warnings  # 导入警告模块

import numpy as np  # 导入 NumPy 库
from scipy import linalg, special, fft as sp_fft  # 导入 SciPy 库的线性代数、特殊函数和 FFT 子模块

# 定义公开的窗口函数名称列表
__all__ = ['boxcar', 'triang', 'parzen', 'bohman', 'blackman', 'nuttall',
           'blackmanharris', 'flattop', 'bartlett', 'barthann',
           'hamming', 'kaiser', 'kaiser_bessel_derived', 'gaussian',
           'general_cosine', 'general_gaussian', 'general_hamming',
           'chebwin', 'cosine', 'hann', 'exponential', 'tukey', 'taylor',
           'dpss', 'get_window', 'lanczos']


def _len_guards(M):
    """Handle small or incorrect window lengths"""
    # 检查窗口长度 M 是否是非负整数
    if int(M) != M or M < 0:
        raise ValueError('Window length M must be a non-negative integer')
    return M <= 1


def _extend(M, sym):
    """Extend window by 1 sample if needed for DFT-even symmetry"""
    # 如果不需要对称性，则扩展窗口长度 M+1，并返回 True
    if not sym:
        return M + 1, True
    else:
        return M, False


def _truncate(w, needed):
    """Truncate window by 1 sample if needed for DFT-even symmetry"""
    # 如果需要 DFT-even 对称性，则截断窗口 w 的最后一个样本
    if needed:
        return w[:-1]
    else:
        return w


def general_cosine(M, a, sym=True):
    r"""
    Generic weighted sum of cosine terms window

    Parameters
    ----------
    M : int
        Number of points in the output window
    a : array_like
        Sequence of weighting coefficients. This uses the convention of being
        centered on the origin, so these will typically all be positive
        numbers, not alternating sign.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The array of window values.

    References
    ----------
    .. [1] A. Nuttall, "Some windows with very good sidelobe behavior," IEEE
           Transactions on Acoustics, Speech, and Signal Processing, vol. 29,
           no. 1, pp. 84-91, Feb 1981. :doi:`10.1109/TASSP.1981.1163506`.
    .. [2] Heinzel G. et al., "Spectrum and spectral density estimation by the
           Discrete Fourier transform (DFT), including a comprehensive list of
           window functions and some new flat-top windows", February 15, 2002
           https://holometer.fnal.gov/GH_FFT.pdf

    Examples
    --------
    Heinzel describes a flat-top window named "HFT90D" with formula: [2]_

    .. math::  w_j = 1 - 1.942604 \cos(z) + 1.340318 \cos(2z)
               - 0.440811 \cos(3z) + 0.043097 \cos(4z)

    where

    .. math::  z = \frac{2 \pi j}{N}, j = 0...N - 1

    Since this uses the convention of starting at the origin, to reproduce the
    window, we need to convert every other coefficient to a positive number:

    >>> HFT90D = [1, 1.942604, 1.340318, 0.440811, 0.043097]

    The paper states that the highest sidelobe is at -90.2 dB.  Reproduce
    Figure 42 by plotting the window and its frequency response, and confirm
    the sidelobe level in red:

    >>> import numpy as np
    >>> from scipy.signal.windows import general_cosine
    >>> from scipy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt
    
    >>> window = general_cosine(1000, HFT90D, sym=False)
    # 使用 Scipy 中的 general_cosine 函数生成长度为 1000 的 HFT90D 窗口，sym=False 表示非对称窗口
    
    >>> plt.plot(window)
    # 绘制生成的窗口函数图像
    
    >>> plt.title("HFT90D window")
    # 设置图像标题为 "HFT90D window"
    
    >>> plt.ylabel("Amplitude")
    # 设置 y 轴标签为 "Amplitude"
    
    >>> plt.xlabel("Sample")
    # 设置 x 轴标签为 "Sample"
    
    >>> plt.figure()
    # 创建一个新的图像窗口
    
    >>> A = fft(window, 10000) / (len(window)/2.0)
    # 对生成的窗口函数进行 FFT 变换，并进行归一化处理
    
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    # 生成频率轴，范围从 -0.5 到 0.5
    
    >>> response = np.abs(fftshift(A / abs(A).max()))
    # 对 FFT 变换后的结果进行频移和幅度归一化处理
    
    >>> response = 20 * np.log10(np.maximum(response, 1e-10))
    # 将归一化后的结果转换为 dB 单位
    
    >>> plt.plot(freq, response)
    # 绘制频率响应曲线
    
    >>> plt.axis([-50/1000, 50/1000, -140, 0])
    # 设置绘图的坐标轴范围
    
    >>> plt.title("Frequency response of the HFT90D window")
    # 设置图像标题为 "Frequency response of the HFT90D window"
    
    >>> plt.ylabel("Normalized magnitude [dB]")
    # 设置 y 轴标签为 "Normalized magnitude [dB]"
    
    >>> plt.xlabel("Normalized frequency [cycles per sample]")
    # 设置 x 轴标签为 "Normalized frequency [cycles per sample]"
    
    >>> plt.axhline(-90.2, color='red')
    # 绘制一条水平线，表示 -90.2 dB 的参考线，颜色为红色
    
    >>> plt.show()
    # 显示绘制的图像
    
    """
    if _len_guards(M):
        return np.ones(M)
    # 如果输入 M 满足长度保护条件，返回长度为 M 的全为 1 的数组
    
    M, needs_trunc = _extend(M, sym)
    # 根据 sym 参数对输入 M 进行扩展处理，并检查是否需要截断
    
    fac = np.linspace(-np.pi, np.pi, M)
    # 生成一个从 -π 到 π 的等间隔数组，长度为 M
    
    w = np.zeros(M)
    # 初始化一个长度为 M 的零数组
    
    for k in range(len(a)):
        w += a[k] * np.cos(k * fac)
    # 使用给定的系数 a，生成一个余弦加权和数组 w
    
    return _truncate(w, needs_trunc)
    # 返回根据截断需求处理后的数组 w
# 返回一个矩形窗口（也称为方窗或矩形窗），相当于没有窗口效果
def boxcar(M, sym=True):
    # 检查 M 的长度保护条件，并返回全为 1 的数组
    if _len_guards(M):
        return np.ones(M)
    # 根据 sym 参数扩展 M 的长度，用于对称性处理
    M, needs_trunc = _extend(M, sym)

    # 创建一个长度为 M 的全为 1 的浮点数数组
    w = np.ones(M, float)

    # 根据 needs_trunc 参数截断窗口 w 并返回
    return _truncate(w, needs_trunc)


# 返回一个三角形窗口
def triang(M, sym=True):
    # 检查 M 的长度保护条件，并根据 sym 参数扩展 M 的长度，用于对称性或周期性处理
    M, needs_trunc = _extend(M, sym)

    # 创建一个长度为 M 的三角形窗口数组
    w = np.bartlett(M)

    # 根据 needs_trunc 参数截断窗口 w 并返回
    return _truncate(w, needs_trunc)
    # 设置 y 轴标签为 "Normalized magnitude [dB]"
    >>> plt.ylabel("Normalized magnitude [dB]")
    # 设置 x 轴标签为 "Normalized frequency [cycles per sample]"
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    如果 M 长度保护条件满足，则返回长度为 M 的全为 1 的数组
    if _len_guards(M):
        return np.ones(M)
    # 对 M 进行扩展，并检查是否需要截断
    M, needs_trunc = _extend(M, sym)

    # 创建包含整数序列 1 到 (M + 1) // 2 的 numpy 数组
    n = np.arange(1, (M + 1) // 2 + 1)
    # 判断 M 是否为偶数
    if M % 2 == 0:
        # 如果 M 是偶数，计算对称的窗口系数
        w = (2 * n - 1.0) / M
        # 将前半部分和反转后的前半部分连接起来，形成完整的对称窗口系数数组
        w = np.r_[w, w[::-1]]
    else:
        # 如果 M 是奇数，计算对称的窗口系数
        w = 2 * n / (M + 1.0)
        # 将前半部分和反转后的前半部分（不包括最后一个元素）连接起来，形成完整的对称窗口系数数组
        w = np.r_[w, w[-2::-1]]

    # 返回经过截断处理后的窗口系数数组
    return _truncate(w, needs_trunc)
def bohman(M, sym=True):
    """Return a Bohman window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero, an empty array
        is returned. An exception is thrown when it is negative.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Examples
    --------
    Plot the window and its frequency response:

    >>> import numpy as np
    >>> from scipy import signal
    >>> from scipy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    # Generate a Bohman window of length M
    window = signal.windows.bohman(M)
    # Plot the Bohman window
    plt.plot(window)
    # Labeling the plot
    plt.title("Bohman window")
    plt.ylabel("Amplitude")
    plt.xlabel("Sample")

    """
    # If M is 0, return an empty array
    if _len_guards(M):
        return np.ones(M)
    
    # Extend M if needed and determine if truncation is necessary
    M, needs_trunc = _extend(M, sym)

    # Generate the Bohman window
    n = np.arange(-(M - 1) / 2.0, (M - 1) / 2.0 + 0.5, 1.0)
    na = np.extract(n < -(M - 1) / 4.0, n)
    nb = np.extract(abs(n) <= (M - 1) / 4.0, n)
    wa = 2 * (1 - np.abs(na) / (M / 2.0)) ** 3.0
    wb = (1 - 6 * (np.abs(nb) / (M / 2.0)) ** 2.0 +
          6 * (np.abs(nb) / (M / 2.0)) ** 3.0)
    w = np.r_[wa, wb, wa[::-1]]

    # Return the window after potential truncation
    return _truncate(w, needs_trunc)
    # 设置图表的标题为 "Bohman window"
    plt.title("Bohman window")
    # 设置 y 轴标签为 "Amplitude"
    plt.ylabel("Amplitude")
    # 设置 x 轴标签为 "Sample"
    plt.xlabel("Sample")

    # 创建一个新的图表
    plt.figure()
    # 计算窗口的 Fourier 变换，并进行归一化处理
    A = fft(window, 2047) / (len(window)/2.0)
    # 生成频率坐标
    freq = np.linspace(-0.5, 0.5, len(A))
    # 计算频率响应的幅度，转换为分贝单位
    response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    # 绘制频率响应图
    plt.plot(freq, response)
    # 设置坐标轴范围
    plt.axis([-0.5, 0.5, -120, 0])
    # 设置图表的标题为 "Frequency response of the Bohman window"
    plt.title("Frequency response of the Bohman window")
    # 设置 y 轴标签为 "Normalized magnitude [dB]"
    plt.ylabel("Normalized magnitude [dB]")
    # 设置 x 轴标签为 "Normalized frequency [cycles per sample]"
    plt.xlabel("Normalized frequency [cycles per sample]")

"""
如果输入的长度满足长度保护条件，则返回一个全为 1 的数组
"""
if _len_guards(M):
    return np.ones(M)
# 将输入长度 M 和对称性参数 sym 进行扩展，获取实际长度和是否需要截断的信息
M, needs_trunc = _extend(M, sym)

# 计算 Bohman 窗口的系数
fac = np.abs(np.linspace(-1, 1, M)[1:-1])
w = (1 - fac) * np.cos(np.pi * fac) + 1.0 / np.pi * np.sin(np.pi * fac)
w = np.r_[0, w, 0]

# 返回根据截断条件处理后的窗口系数
return _truncate(w, needs_trunc)
# 定义一个函数，生成 Blackman 窗口函数
def blackman(M, sym=True):
    r"""
    Return a Blackman window.

    The Blackman window is a taper formed by using the first three terms of
    a summation of cosines. It was designed to have close to the minimal
    leakage possible.  It is close to optimal, only slightly worse than a
    Kaiser window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero, an empty array
        is returned. An exception is thrown when it is negative.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    The Blackman window is defined as

    .. math::  w(n) = 0.42 - 0.5 \cos(2\pi n/M) + 0.08 \cos(4\pi n/M)

    The "exact Blackman" window was designed to null out the third and fourth
    sidelobes, but has discontinuities at the boundaries, resulting in a
    6 dB/oct fall-off.  This window is an approximation of the "exact" window,
    which does not null the sidelobes as well, but is smooth at the edges,
    improving the fall-off rate to 18 dB/oct. [3]_

    Most references to the Blackman window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function. It is known as a
    "near optimal" tapering function, almost as good (by some measures)
    as the Kaiser window.

    References
    ----------
    .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
           spectra, Dover Publications, New York.
    .. [2] Oppenheim, A.V., and R.W. Schafer. Discrete-Time Signal Processing.
           Upper Saddle River, NJ: Prentice-Hall, 1999, pp. 468-471.
    .. [3] Harris, Fredric J. (Jan 1978). "On the use of Windows for Harmonic
           Analysis with the Discrete Fourier Transform". Proceedings of the
           IEEE 66 (1): 51-83. :doi:`10.1109/PROC.1978.10837`.

    Examples
    --------
    Plot the window and its frequency response:

    >>> import numpy as np
    >>> from scipy import signal
    >>> from scipy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    创建一个 Blackman 窗口，包含 M 个点
    window = signal.windows.blackman(M)

    绘制窗口的时域响应图
    plt.plot(window)
    plt.title("Blackman window")
    plt.ylabel("Amplitude")
    plt.xlabel("Sample")

    创建一个新的图形对象
    plt.figure()

    对窗口进行 Fourier 变换，并归一化
    A = fft(window, 2048) / (len(window)/2.0)

    计算频率轴
    freq = np.linspace(-0.5, 0.5, len(A))

    计算窗口的频率响应
    response = np.abs(fftshift(A / abs(A).max()))

    将频率响应转换为分贝单位
    response = 20 * np.log10(np.maximum(response, 1e-10))

    绘制频率响应图
    plt.plot(freq, response)
    # 设置绘图的 X 轴和 Y 轴范围
    plt.axis([-0.5, 0.5, -120, 0])
    # 设置图表的标题
    plt.title("Frequency response of the Blackman window")
    # 设置 Y 轴标签
    plt.ylabel("Normalized magnitude [dB]")
    # 设置 X 轴标签
    plt.xlabel("Normalized frequency [cycles per sample]")
    
    """
    # 返回一个使用一般余弦函数生成的黑曼窗口的数组
    # M: 窗口长度
    # [0.42, 0.50, 0.08]: 窗口的系数
    # sym: 窗口的对称性
    return general_cosine(M, [0.42, 0.50, 0.08], sym)
    """
# 定义一个名为 `nuttall` 的函数，用于生成 Nuttall 窗口函数
def nuttall(M, sym=True):
    # 返回基于 Nuttall 窗口函数变体 "Nuttall4c" 的最小 4 项 Blackman-Harris 窗口
    return general_cosine(M, [0.3635819, 0.4891775, 0.1365995, 0.0106411], sym)

# 定义一个名为 `blackmanharris` 的函数，用于生成 Blackman-Harris 窗口函数
def blackmanharris(M, sym=True):
    # 返回最小 4 项 Blackman-Harris 窗口
    # 参数 M：输出窗口中的点数。如果为零，则返回一个空数组。当 M 为负数时抛出异常。
    # 参数 sym：布尔值，默认为 True。当为 True 时生成对称窗口，用于滤波器设计；当为 False 时生成周期窗口，用于频谱分析。
    # 返回窗口 w，最大值被归一化为 1（如果 M 是偶数且 sym 为 True，则值 1 不会出现）。
    return general_cosine(M, [0.35875, 0.48829, 0.14128, 0.01168], sym)
    >>> import matplotlib.pyplot as plt  # 导入matplotlib库，并命名为plt，用于绘图
    
    >>> window = signal.windows.blackmanharris(51)  # 创建长度为51的Blackman-Harris窗口
    
    >>> plt.plot(window)  # 绘制窗口函数的图像
    >>> plt.title("Blackman-Harris window")  # 设置图像标题为"Blackman-Harris window"
    >>> plt.ylabel("Amplitude")  # 设置y轴标签为"Amplitude"
    >>> plt.xlabel("Sample")  # 设置x轴标签为"Sample"
    
    >>> plt.figure()  # 创建新的图形窗口
    
    >>> A = fft(window, 2048) / (len(window)/2.0)  # 对窗口函数进行2048点FFT，进行幅度归一化处理
    >>> freq = np.linspace(-0.5, 0.5, len(A))  # 生成频率坐标，范围为[-0.5, 0.5]
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))  # 计算幅频响应，并转换为对数刻度（dB）
    
    >>> plt.plot(freq, response)  # 绘制频率响应曲线
    >>> plt.axis([-0.5, 0.5, -120, 0])  # 设置坐标轴范围，x轴为[-0.5, 0.5]，y轴为[-120, 0]
    >>> plt.title("Frequency response of the Blackman-Harris window")  # 设置图像标题为"Frequency response of the Blackman-Harris window"
    >>> plt.ylabel("Normalized magnitude [dB]")  # 设置y轴标签为"Normalized magnitude [dB]"
    >>> plt.xlabel("Normalized frequency [cycles per sample]")  # 设置x轴标签为"Normalized frequency [cycles per sample]"
    
    """
    使用general_cosine函数生成M点的一般余弦窗口，并返回结果。
    M为窗口长度，[0.35875, 0.48829, 0.14128, 0.01168]为窗口系数，sym为对称性设置。
    """
    return general_cosine(M, [0.35875, 0.48829, 0.14128, 0.01168], sym)
# 返回一个 Bartlett 窗口函数

# 参数 M：输出窗口中的点数。如果为零，返回一个空数组。如果为负数，则引发异常。
# 参数 sym：布尔值，可选。当为 True 时（默认），生成对称窗口，用于滤波器设计。
#         当为 False 时，生成周期窗口，用于频谱分析。
def bartlett(M, sym=True):
    """
    Return a Bartlett window.

    The Bartlett window is very similar to a triangular window, except
    that the end points are at zero.  It is often used in signal
    processing for tapering a signal, without generating too much
    ripple in the frequency domain.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero, an empty array
        is returned. An exception is thrown when it is negative.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window array containing the Bartlett window.

    Notes
    -----
    The Bartlett window is also known as the triangular window. It tapers
    smoothly from the center to zero at the edges, helping to reduce ripple
    effects in the frequency domain.

    Examples
    --------
    To plot the Bartlett window and its frequency response:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy import signal

    >>> window = signal.windows.bartlett(51)
    >>> plt.plot(window)
    >>> plt.title("Bartlett window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = np.fft.fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(np.fft.fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Bartlett window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    References
    ----------
    .. [1] "Bartlett window", Wikipedia,
           https://en.wikipedia.org/wiki/Window_function#Bartlett_window
    """
    # 定义 Bartlett 窗口的系数
    a = [0.5, 0.5]
    # 调用通用余弦窗口函数，返回 Bartlett 窗口
    return general_cosine(M, a, sym)
    # Docstring adapted from NumPy's bartlett function
    
    # 检查长度是否符合要求，如果符合，则返回全为1的长度为M的数组
    if _len_guards(M):
        return np.ones(M)
    
    # 扩展长度M，并根据sym参数决定是否需要截断
    M, needs_trunc = _extend(M, sym)
    
    # 生成从0到M-1的整数序列作为窗口的索引
    n = np.arange(0, M)
    
    # 根据Bartlett窗口的定义，计算窗口函数w(n)
    # 其中，当n <= (M - 1) / 2.0时，w(n) = 2.0 * n / (M - 1)
    # 否则，w(n) = 2.0 - 2.0 * n / (M - 1)
    w = np.where(np.less_equal(n, (M - 1) / 2.0),
                 2.0 * n / (M - 1), 2.0 - 2.0 * n / (M - 1))
    
    # 返回根据sym参数截断后的窗口函数w
    return _truncate(w, needs_trunc)
# 定义一个函数 hann，用于生成 Hann 窗口函数

# Docstring 文档字符串，描述了 Hann 窗口的特性、参数和返回值
# 返回一个 Hann 窗口函数，用于平滑信号或频谱分析

def hann(M, sym=True):
    r"""
    Return a Hann window.

    The Hann window is a taper formed by using a raised cosine or sine-squared
    with ends that touch zero.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero, an empty array
        is returned. An exception is thrown when it is negative.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    The Hann window is defined as

    .. math::  w(n) = 0.5 - 0.5 \cos\left(\frac{2\pi{n}}{M-1}\right)
               \qquad 0 \leq n \leq M-1

    The window was named for Julius von Hann, an Austrian meteorologist. It is
    also known as the Cosine Bell. It is sometimes erroneously referred to as
    the "Hanning" window, from the use of "hann" as a verb in the original
    paper and confusion with the very similar Hamming window.

    Most references to the Hann window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function.

    References
    ----------
    .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
           spectra, Dover Publications, New York.
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics",
           The University of Alberta Press, 1975, pp. 106-108.
    .. [3] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function
    .. [4] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
           "Numerical Recipes", Cambridge University Press, 1986, page 425.

    Examples
    --------
    Plot the window and its frequency response:

    >>> import numpy as np
    >>> from scipy import signal
    >>> from scipy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.windows.hann(51)
    >>> plt.plot(window)
    >>> plt.title("Hann window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = np.abs(fftshift(A / abs(A).max()))
    >>> response = 20 * np.log10(np.maximum(response, 1e-10))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Hann window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    # Docstring adapted from NumPy's hanning function
    # 调用 general_hamming 函数，传入参数 M, 0.5, sym，并返回其结果
    return general_hamming(M, 0.5, sym)
def tukey(M, alpha=0.5, sym=True):
    r"""Return a Tukey window, also known as a tapered cosine window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero, an empty array
        is returned. An exception is thrown when it is negative.
    alpha : float, optional
        Shape parameter of the Tukey window, representing the fraction of the
        window inside the cosine tapered region.
        If zero, the Tukey window is equivalent to a rectangular window.
        If one, the Tukey window is equivalent to a Hann window.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    References
    ----------
    .. [1] Harris, Fredric J. (Jan 1978). "On the use of Windows for Harmonic
           Analysis with the Discrete Fourier Transform". Proceedings of the
           IEEE 66 (1): 51-83. :doi:`10.1109/PROC.1978.10837`
    .. [2] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function#Tukey_window

    Examples
    --------
    Plot the window and its frequency response:

    >>> import numpy as np
    >>> from scipy import signal
    >>> from scipy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.windows.tukey(51)
    >>> plt.plot(window)
    >>> plt.title("Tukey window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")
    >>> plt.ylim([0, 1.1])

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Tukey window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    # Guard clause: return an array of ones if M is invalid
    if _len_guards(M):
        return np.ones(M)

    # Return a rectangular window if alpha is less than or equal to zero
    if alpha <= 0:
        return np.ones(M, 'd')
    # Return a Hann window if alpha is greater than or equal to one
    elif alpha >= 1.0:
        return hann(M, sym=sym)

    # Extend the window length and check if truncation is needed
    M, needs_trunc = _extend(M, sym)

    # Generate the Tukey window
    n = np.arange(0, M)
    width = int(np.floor(alpha*(M-1)/2.0))
    n1 = n[0:width+1]
    n2 = n[width+1:M-width-1]
    n3 = n[M-width-1:]

    w1 = 0.5 * (1 + np.cos(np.pi * (-1 + 2.0*n1/alpha/(M-1))))
    w2 = np.ones(n2.shape)
    w3 = 0.5 * (1 + np.cos(np.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))

    w = np.concatenate((w1, w2, w3))

    # Truncate the window if necessary and return
    return _truncate(w, needs_trunc)
    """
    根据输入参数生成Bartlett-Hann窗口。

    Parameters
    ----------
    M : int
        窗口长度。
    sym : bool, optional
        当为True（默认）时，生成对称窗口，用于滤波设计。
        当为False时，生成周期性窗口，用于频谱分析。

    Returns
    -------
    w : ndarray
        生成的窗口，最大值归一化为1（如果M为偶数且sym为True时，最大值1并不会出现）。

    Examples
    --------
    绘制窗口及其频率响应示例：

    >>> import numpy as np
    >>> from scipy import signal
    >>> from scipy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.windows.barthann(51)
    >>> plt.plot(window)
    >>> plt.title("Bartlett-Hann窗口")
    >>> plt.ylabel("振幅")
    >>> plt.xlabel("样本")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Bartlett-Hann窗口的频率响应")
    >>> plt.ylabel("归一化幅度 [dB]")
    >>> plt.xlabel("归一化频率 [每个样本的周期数]")

    """
    # 检查长度参数M的有效性，并根据需要进行截断处理
    if _len_guards(M):
        return np.ones(M)
    # 根据sym参数调整窗口长度M和是否需要截断的标志
    M, needs_trunc = _extend(M, sym)

    # 生成Bartlett-Hann窗口的主要计算步骤
    n = np.arange(0, M)
    fac = np.abs(n / (M - 1.0) - 0.5)
    w = 0.62 - 0.48 * fac + 0.38 * np.cos(2 * np.pi * fac)

    # 根据截断标志对窗口进行截断处理，并返回最终生成的窗口
    return _truncate(w, needs_trunc)
# 定义一个函数，返回一个广义汉明窗口
def general_hamming(M, alpha, sym=True):
    r"""Return a generalized Hamming window.

    The generalized Hamming window is constructed by multiplying a rectangular
    window by one period of a cosine function [1]_.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero, an empty array
        is returned. An exception is thrown when it is negative.
    alpha : float
        The window coefficient, :math:`\alpha`
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    See Also
    --------
    hamming, hann

    Notes
    -----
    The generalized Hamming window is defined as

    .. math:: w(n) = \alpha - \left(1 - \alpha\right)
              \cos\left(\frac{2\pi{n}}{M-1}\right) \qquad 0 \leq n \leq M-1

    Both the common Hamming window and Hann window are special cases of the
    generalized Hamming window with :math:`\alpha` = 0.54 and :math:`\alpha` =
    0.5, respectively [2]_.

    References
    ----------
    .. [1] DSPRelated, "Generalized Hamming Window Family",
           https://www.dsprelated.com/freebooks/sasp/Generalized_Hamming_Window_Family.html
    .. [2] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function
    .. [3] Riccardo Piantanida ESA, "Sentinel-1 Level 1 Detailed Algorithm
           Definition",
           https://sentinel.esa.int/documents/247904/1877131/Sentinel-1-Level-1-Detailed-Algorithm-Definition
    .. [4] Matthieu Bourbigot ESA, "Sentinel-1 Product Definition",
           https://sentinel.esa.int/documents/247904/1877131/Sentinel-1-Product-Definition

    Examples
    --------
    The Sentinel-1A/B Instrument Processing Facility uses generalized Hamming
    windows in the processing of spaceborne Synthetic Aperture Radar (SAR)
    data [3]_. The facility uses various values for the :math:`\alpha`
    parameter based on operating mode of the SAR instrument. Some common
    :math:`\alpha` values include 0.75, 0.7 and 0.52 [4]_. As an example, we
    plot these different windows.

    >>> import numpy as np
    >>> from scipy.signal.windows import general_hamming
    >>> from scipy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> fig1, spatial_plot = plt.subplots()
    >>> spatial_plot.set_title("Generalized Hamming Windows")
    >>> spatial_plot.set_ylabel("Amplitude")
    >>> spatial_plot.set_xlabel("Sample")

    >>> fig2, freq_plot = plt.subplots()
    >>> freq_plot.set_title("Frequency Responses")
    >>> freq_plot.set_ylabel("Normalized magnitude [dB]")
    >>> freq_plot.set_xlabel("Normalized frequency [cycles per sample]")

    >>> for alpha in [0.75, 0.7, 0.52]:
    # 使用 general_hamming 函数生成长度为 41 的汉明窗口，alpha 是窗口参数
    window = general_hamming(41, alpha)
    # 在空间图中绘制窗口的图像，并使用 alpha 的值作为标签
    spatial_plot.plot(window, label="{:.2f}".format(alpha))
    # 对窗口进行 FFT 变换，使用 2048 作为 FFT 的点数，得到幅度归一化后的频谱 A
    A = fft(window, 2048) / (len(window)/2.0)
    # 生成频率轴，范围从 -0.5 到 0.5，与 FFT 结果 A 的长度相匹配
    freq = np.linspace(-0.5, 0.5, len(A))
    # 对 FFT 结果 A 进行零移频移，并计算其的分贝值，用作频率响应
    response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    # 在频率图中绘制频率响应曲线，并使用 alpha 的值作为标签
    freq_plot.plot(freq, response, label="{:.2f}".format(alpha))
    # 在频率图和空间图中显示图例，位置为右上角
    >>> freq_plot.legend(loc="upper right")
    >>> spatial_plot.legend(loc="upper right")
# 定义一个函数 hamming，生成 Hamming 窗口函数
def hamming(M, sym=True):
    # 文档字符串，描述 Hamming 窗口函数的特性和用途
    r"""Return a Hamming window.

    The Hamming window is a taper formed by using a raised cosine with
    non-zero endpoints, optimized to minimize the nearest side lobe.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero, an empty array
        is returned. An exception is thrown when it is negative.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    The Hamming window is defined as

    .. math::  w(n) = 0.54 - 0.46 \cos\left(\frac{2\pi{n}}{M-1}\right)
               \qquad 0 \leq n \leq M-1

    The Hamming was named for R. W. Hamming, an associate of J. W. Tukey and
    is described in Blackman and Tukey. It was recommended for smoothing the
    truncated autocovariance function in the time domain.
    Most references to the Hamming window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function.

    References
    ----------
    .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
           spectra, Dover Publications, New York.
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics", The
           University of Alberta Press, 1975, pp. 109-110.
    .. [3] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function
    .. [4] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
           "Numerical Recipes", Cambridge University Press, 1986, page 425.

    Examples
    --------
    Plot the window and its frequency response:

    >>> import numpy as np
    >>> from scipy import signal
    >>> from scipy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.windows.hamming(51)
    >>> plt.plot(window)
    >>> plt.title("Hamming window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Hamming window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    # 调用 general_hamming 函数生成 Hamming 窗口，0.54 是 Hamming 窗口函数的系数
    return general_hamming(M, 0.54, sym)
    r"""Return a Kaiser window.

    The Kaiser window is a taper formed by using a Bessel function.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero, an empty array
        is returned. An exception is thrown when it is negative.
    beta : float
        Shape parameter, determines trade-off between main-lobe width and
        side lobe level. As beta gets large, the window narrows.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    The Kaiser window is defined as

    .. math::  w(n) = I_0\left( \beta \sqrt{1-\frac{4n^2}{(M-1)^2}}
               \right)/I_0(\beta)

    with

    .. math:: \quad -\frac{M-1}{2} \leq n \leq \frac{M-1}{2},

    where :math:`I_0` is the modified zeroth-order Bessel function.

    The Kaiser was named for Jim Kaiser, who discovered a simple approximation
    to the DPSS window based on Bessel functions.
    The Kaiser window is a very good approximation to the Digital Prolate
    Spheroidal Sequence, or Slepian window, which is the transform which
    maximizes the energy in the main lobe of the window relative to total
    energy.

    The Kaiser can approximate other windows by varying the beta parameter.
    (Some literature uses alpha = beta/pi.) [4]_

    ====  =======================
    beta  Window shape
    ====  =======================
    0     Rectangular
    5     Similar to a Hamming
    6     Similar to a Hann
    8.6   Similar to a Blackman
    ====  =======================

    A beta value of 14 is probably a good starting point. Note that as beta
    gets large, the window narrows, and so the number of samples needs to be
    large enough to sample the increasingly narrow spike, otherwise NaNs will
    be returned.

    Most references to the Kaiser window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function.

    References
    ----------
    .. [1] J. F. Kaiser, "Digital Filters" - Ch 7 in "Systems analysis by
           digital computer", Editors: F.F. Kuo and J.F. Kaiser, p 218-285.
           John Wiley and Sons, New York, (1966).
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics", The
           University of Alberta Press, 1975, pp. 177-178.
    .. [3] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function
    """
    # 根据 F. J. Harris 的文献 [4]，使用 Kaiser 窗口进行信号处理中的窗口函数设计
    Examples
    --------
    绘制 Kaiser 窗口及其频率响应：
    
    >>> import numpy as np
    >>> from scipy import signal
    >>> from scipy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt
    
    >>> window = signal.windows.kaiser(51, beta=14)
    >>> plt.plot(window)
    >>> plt.title(r"Kaiser window ($\beta$=14)")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")
    
    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title(r"Frequency response of the Kaiser window ($\beta$=14)")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")
    
    """
    # Docstring adapted from NumPy's kaiser function
    
    # 如果输入参数 M 符合保护长度条件，则返回长度为 M 的单位数组
    if _len_guards(M):
        return np.ones(M)
    # 对于长度 M，根据对称性需要进行扩展处理
    M, needs_trunc = _extend(M, sym)
    
    # 创建长度为 M 的序列 n，从 0 到 M-1
    n = np.arange(0, M)
    # 计算 Kaiser 窗口的参数 alpha
    alpha = (M - 1) / 2.0
    # 计算 Kaiser 窗口的每个点的权重 w
    w = (special.i0(beta * np.sqrt(1 - ((n - alpha) / alpha) ** 2.0)) /
         special.i0(beta))
    
    # 返回处理后的 Kaiser 窗口 w，并根据需要截断
    return _truncate(w, needs_trunc)
# 返回一个高斯窗口函数，用于信号处理和滤波器设计
def gaussian(M, std, sym=True):
    # 参数 M 表示输出窗口的点数，如果为零则返回空数组，当为负数时抛出异常
    M : int
        Number of points in the output window. If zero, an empty array
        is returned. An exception is thrown when it is negative.
    # 参数 std 表示标准差 sigma
    std : float
        The standard deviation, sigma.
    # sym 是一个布尔值，仅为了兼容其他窗口函数的接口并可通过 `get_window` 调用。当 sym 为 True（默认）时，生成对称窗口，用于滤波器设计。
    sym : bool, optional
        This parameter only exists to comply with the interface offered by
        the other window functions and to be callable by `get_window`.
        When True (default), generates a symmetric window, for use in filter
        design.

    Returns
    -------
    w : ndarray
        The window, normalized to fulfil the Princen-Bradley condition.
    """
    sym : bool, optional
        当为True（默认），生成对称窗口，用于滤波设计。
        当为False，生成周期性窗口，用于频谱分析。

    Returns
    -------
    w : ndarray
        窗口数组，最大值被归一化为1（当M为偶数且sym为True时，值1不出现）。

    Notes
    -----
    高斯窗口定义如下：

    .. math::  w(n) = e^{ -\frac{1}{2}\left(\frac{n}{\sigma}\right)^2 }

    Examples
    --------
    绘制窗口及其频率响应图：

    >>> import numpy as np
    >>> from scipy import signal
    >>> from scipy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.windows.gaussian(51, std=7)
    >>> plt.plot(window)
    >>> plt.title(r"Gaussian window ($\sigma$=7)")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title(r"Frequency response of the Gaussian window ($\sigma$=7)")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    # 如果长度保护条件满足，则返回长度为M的全1数组
    if _len_guards(M):
        return np.ones(M)
    # 调整窗口长度，并确定是否需要截断
    M, needs_trunc = _extend(M, sym)

    # 生成高斯窗口的序列
    n = np.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = np.exp(-n ** 2 / sig2)

    # 根据需要截断窗口，并返回结果
    return _truncate(w, needs_trunc)
# 返回具有广义高斯形状的窗口函数
def general_gaussian(M, p, sig, sym=True):
    # 如果 M 小于等于零，返回一个空数组；若为负数，抛出异常
    if _len_guards(M):
        return np.ones(M)
    # 扩展 M 并检查是否需要截断
    M, needs_trunc = _extend(M, sym)

    # 创建序列 n，以使窗口对称
    n = np.arange(0, M) - (M - 1.0) / 2.0
    # 计算广义高斯窗口函数的值
    w = np.exp(-0.5 * np.abs(n / sig) ** (2 * p))

    # 返回处理后的窗口，根据需要截断
    return _truncate(w, needs_trunc)


# `chebwin` 由 Kumar Appaiah 贡献
def chebwin(M, at, sym=True):
    # 返回一种 Dolph-Chebyshev 窗口
    # 如果 M 小于等于零，返回一个空数组；若为负数，抛出异常
    if _len_guards(M):
        return np.ones(M)
    # 扩展 M 并检查是否需要截断
    M, needs_trunc = _extend(M, sym)

    # 返回处理后的窗口，最大值始终标准化为 1
    return _truncate(w, needs_trunc)
    """
    `M` and sidelobe equiripple attenuation `at`, using Chebyshev
    polynomials.  It was originally developed by Dolph to optimize the
    directionality of radio antenna arrays.

    Unlike most windows, the Dolph-Chebyshev is defined in terms of its
    frequency response:

    .. math:: W(k) = \frac
              {\cos\{M \cos^{-1}[\beta \cos(\frac{\pi k}{M})]\}}
              {\cosh[M \cosh^{-1}(\beta)]}

    where

    .. math:: \beta = \cosh \left [\frac{1}{M}
              \cosh^{-1}(10^\frac{A}{20}) \right ]

    and 0 <= abs(k) <= M-1. A is the attenuation in decibels (`at`).

    The time domain window is then generated using the IFFT, so
    power-of-two `M` are the fastest to generate, and prime number `M` are
    the slowest.

    The equiripple condition in the frequency domain creates impulses in the
    time domain, which appear at the ends of the window.

    References
    ----------
    .. [1] C. Dolph, "A current distribution for broadside arrays which
           optimizes the relationship between beam width and side-lobe level",
           Proceedings of the IEEE, Vol. 34, Issue 6
    .. [2] Peter Lynch, "The Dolph-Chebyshev Window: A Simple Optimal Filter",
           American Meteorological Society (April 1997)
           http://mathsci.ucd.ie/~plynch/Publications/Dolph.pdf
    .. [3] F. J. Harris, "On the use of windows for harmonic analysis with the
           discrete Fourier transforms", Proceedings of the IEEE, Vol. 66,
           No. 1, January 1978

    Examples
    --------
    Plot the window and its frequency response:

    >>> import numpy as np
    >>> from scipy import signal
    >>> from scipy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.windows.chebwin(51, at=100)
    >>> plt.plot(window)
    >>> plt.title("Dolph-Chebyshev window (100 dB)")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Dolph-Chebyshev window (100 dB)")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """

    # 如果 `at` 的绝对值小于 45dB，发出警告，因为在这种衰减值下，切比雪夫窗口不适合用于频谱分析
    if np.abs(at) < 45:
        warnings.warn("This window is not suitable for spectral analysis "
                      "for attenuation values lower than about 45dB because "
                      "the equivalent noise bandwidth of a Chebyshev window "
                      "does not grow monotonically with increasing sidelobe "
                      "attenuation when the attenuation is smaller than "
                      "about 45 dB.",
                      stacklevel=2)

    # 如果 M 的长度保护条件被满足，返回长度为 M 的全一数组
    if _len_guards(M):
        return np.ones(M)

    # 对 M 和 sym 进行扩展，返回扩展后的 M 和是否需要截断的标志
    M, needs_trunc = _extend(M, sym)

    # 计算参数 beta，其中 order = M - 1.0
    order = M - 1.0
    beta = np.cosh(1.0 / order * np.arccosh(10 ** (np.abs(at) / 20.)))
    k = np.r_[0:M] * 1.0
    x = beta * np.cos(np.pi * k / M)
    # 找到窗口的DFT系数
    # 使用解析定义的切比雪夫多项式，而不是使用scipy.special中的展开式
    # 在scipy.special中使用展开式会导致错误。

    p = np.zeros(x.shape)
    # 对于 x > 1 的情况，计算切比雪夫多项式
    p[x > 1] = np.cosh(order * np.arccosh(x[x > 1]))
    # 对于 x < -1 的情况，计算切比雪夫多项式
    p[x < -1] = (2 * (M % 2) - 1) * np.cosh(order * np.arccosh(-x[x < -1]))
    # 对于 -1 <= x <= 1 的情况，计算切比雪夫多项式
    p[np.abs(x) <= 1] = np.cos(order * np.arccos(x[np.abs(x) <= 1]))

    # 适当的IDFT和填充
    # 取决于M是偶数还是奇数
    if M % 2:
        # 对于奇数M，计算p的实部的FFT
        w = np.real(sp_fft.fft(p))
        # 计算截取长度
        n = (M + 1) // 2
        # 取FFT结果的前n项
        w = w[:n]
        # 对w进行对称填充
        w = np.concatenate((w[n - 1:0:-1], w))
    else:
        # 对于偶数M，对p进行相位调整
        p = p * np.exp(1.j * np.pi / M * np.r_[0:M])
        # 计算p的实部的FFT
        w = np.real(sp_fft.fft(p))
        # 计算截取长度
        n = M // 2 + 1
        # 对w进行对称填充
        w = np.concatenate((w[n - 1:0:-1], w[1:n]))
    # 对w进行归一化处理
    w = w / max(w)

    # 返回截断后的结果
    return _truncate(w, needs_trunc)
def cosine(M, sym=True):
    """Return a window with a simple cosine shape.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero, an empty array
        is returned. An exception is thrown when it is negative.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----

    .. versionadded:: 0.13.0

    Examples
    --------
    Plot the window and its frequency response:

    >>> import numpy as np
    >>> from scipy import signal
    >>> from scipy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.windows.cosine(51)  # Generate a cosine window of length 51
    >>> plt.plot(window)  # Plot the cosine window
    >>> plt.title("Cosine window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2047) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)  # Plot the frequency response of the window
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the cosine window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")
    >>> plt.show()

    """
    # Check if M is 0 or negative, return an empty array or raise an exception
    if _len_guards(M):
        return np.ones(M)
    # Extend M if needed based on sym parameter
    M, needs_trunc = _extend(M, sym)

    # Calculate the cosine window
    w = np.sin(np.pi / M * (np.arange(0, M) + .5))

    # Truncate the window if necessary and return
    return _truncate(w, needs_trunc)
    # 如果 sym 为 True 并且 center 不为 None，则抛出 ValueError 异常
    if sym and center is not None:
        raise ValueError("If sym==True, center must be None.")
    
    # 检查 M 的长度保护条件，如果满足，返回一个包含 M 个元素的全为 1 的数组
    if _len_guards(M):
        return np.ones(M)
    
    # 调用 _extend 函数，处理 M 和 sym，获取处理后的 M 和是否需要截断的标志 needs_trunc
    M, needs_trunc = _extend(M, sym)
    
    # 如果 center 为 None，则将 center 设置为 M-1 的一半
    if center is None:
        center = (M-1) / 2
    
    # 生成包含 M 个元素的序列 n，即从 0 到 M-1
    n = np.arange(0, M)
    
    # 根据指数衰减公式计算窗口函数 w，其中 tau 控制衰减速度
    w = np.exp(-np.abs(n-center) / tau)
    
    # 调用 _truncate 函数，根据 needs_trunc 截断窗口函数 w 的长度，然后返回
    return _truncate(w, needs_trunc)
# 定义 Taylor 窗口生成函数
def taylor(M, nbar=4, sll=30, norm=True, sym=True):
    """
    Return a Taylor window.

    The Taylor window taper function approximates the Dolph-Chebyshev window's
    constant sidelobe level for a parameterized number of near-in sidelobes,
    but then allows a taper beyond [2]_.

    The SAR (synthetic aperture radar) community commonly uses Taylor
    weighting for image formation processing because it provides strong,
    selectable sidelobe suppression with minimum broadening of the
    mainlobe [1]_.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero, an empty array
        is returned. An exception is thrown when it is negative.
    nbar : int, optional
        Number of nearly constant level sidelobes adjacent to the mainlobe.
    sll : float, optional
        Desired suppression of sidelobe level in decibels (dB) relative to the
        DC gain of the mainlobe. This should be a positive number.
    norm : bool, optional
        When True (default), divides the window by the largest (middle) value
        for odd-length windows or the value that would occur between the two
        repeated middle values for even-length windows such that all values
        are less than or equal to 1. When False the DC gain will remain at 1
        (0 dB) and the sidelobes will be `sll` dB down.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    out : array
        The window. When `norm` is True (default), the maximum value is
        normalized to 1 (though the value 1 does not appear if `M` is
        even and `sym` is True).

    See Also
    --------
    chebwin, kaiser, bartlett, blackman, hamming, hann

    References
    ----------
    .. [1] W. Carrara, R. Goodman, and R. Majewski, "Spotlight Synthetic
           Aperture Radar: Signal Processing Algorithms" Pages 512-513,
           July 1995.
    .. [2] Armin Doerry, "Catalog of Window Taper Functions for
           Sidelobe Control", 2017.
           https://www.researchgate.net/profile/Armin_Doerry/publication/316281181_Catalog_of_Window_Taper_Functions_for_Sidelobe_Control/links/58f92cb2a6fdccb121c9d54d/Catalog-of-Window-Taper-Functions-for-Sidelobe-Control.pdf

    Examples
    --------
    Plot the window and its frequency response:

    >>> import numpy as np
    >>> from scipy import signal
    >>> from scipy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.windows.taylor(51, nbar=20, sll=100, norm=False)
    # 生成 Taylor 窗口，长度为 51，20 dB 处的近端旁瓣，抑制级别为 100 dB，不进行归一化
    >>> plt.plot(window)
    # 绘制窗口的时域响应图
    >>> plt.title("Taylor window (100 dB)")
    # 设置图标题为 "Taylor 窗口 (100 dB)"
    >>> plt.ylabel("Amplitude")
    # 设置纵轴标签为 "振幅"
    >>> plt.xlabel("Sample")
    # 设置横轴标签为 "样本"

    >>> plt.figure()
    # 创建新的图形窗口
    >>> A = fft(window, 2048) / (len(window)/2.0)
    # 对窗口进行 2048 点 FFT，除以窗口长度的一半，得到幅度归一化的频域表示
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    # 生成频率轴
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    # 计算频域响应的对数幅度谱

    """
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Taylor window (100 dB)")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """  # noqa: E501
    # 如果 M 小于等于零，返回一个包含 M 个元素，每个元素为 1 的数组
    if _len_guards(M):
        return np.ones(M)
    # 扩展 M 并检查是否需要截断，返回扩展后的 M 值和是否需要截断的标志
    M, needs_trunc = _extend(M, sym)

    # 根据设定的窗函数的主瓣电平（sll），计算参数 B
    # 原始文本中使用负的副瓣电平参数，然后在计算 B 时取反。
    # 为了与其他方法保持一致，假设副瓣电平参数为正数。
    B = 10**(sll / 20)
    # 计算参数 A
    A = np.arccosh(B) / np.pi
    # 计算参数 s2
    s2 = nbar**2 / (A**2 + (nbar - 0.5)**2)
    # 创建一个从 1 到 nbar-1 的数组
    ma = np.arange(1, nbar)

    # 创建一个空数组 Fm，用于存储计算得到的系数
    Fm = np.empty(nbar-1)
    # 创建一个与 ma 大小相同的数组 signs
    signs = np.empty_like(ma)
    # 将 signs 的偶数索引位置设置为 1，奇数索引位置设置为 -1
    signs[::2] = 1
    signs[1::2] = -1
    # 计算 ma 元素的平方
    m2 = ma*ma
    # 遍历 ma 数组，计算 Fm 数组中的每个元素
    for mi, m in enumerate(ma):
        # 计算 numer 分子部分
        numer = signs[mi] * np.prod(1 - m2[mi]/s2/(A**2 + (ma - 0.5)**2))
        # 计算 denom 分母部分
        denom = 2 * np.prod(1 - m2[mi]/m2[:mi]) * np.prod(1 - m2[mi]/m2[mi+1:])
        # 计算得到 Fm 数组的每个元素
        Fm[mi] = numer / denom

    # 定义窗函数 W(n)，接受一个整数 n 作为参数
    def W(n):
        # 计算窗函数的值，使用 Fm 和余弦函数
        return 1 + 2*np.dot(Fm, np.cos(
            2*np.pi*ma[:, np.newaxis]*(n-M/2.+0.5)/M))

    # 计算窗函数 w 的值，传入一个从 0 到 M-1 的整数数组作为参数
    w = W(np.arange(M))

    # 如果需要进行归一化
    if norm:
        # 计算归一化的比例尺
        scale = 1.0 / W((M - 1) / 2)
        # 将 w 数组乘以归一化比例尺
        w *= scale

    # 返回截断后的窗函数 w
    return _truncate(w, needs_trunc)
# 定义函数 dpss，用于计算离散普洛莱特球面序列（DPSS）
def dpss(M, NW, Kmax=None, sym=True, norm=None, return_ratios=False):
    """
    Compute the Discrete Prolate Spheroidal Sequences (DPSS).

    DPSS (or Slepian sequences) are often used in multitaper power spectral
    density estimation (see [1]_). The first window in the sequence can be
    used to maximize the energy concentration in the main lobe, and is also
    called the Slepian window.

    Parameters
    ----------
    M : int
        窗口长度。
    NW : float
        标准化的半带宽，对应于 ``2*NW = BW/f0 = BW*M*dt``，其中 ``dt`` 默认为 1。
    Kmax : int | None, optional
        要返回的 DPSS 窗口数量（从阶数 ``0`` 到 ``Kmax-1``）。如果为 None（默认），则仅返回一个形状为 ``(M,)`` 的窗口，而不是形状为 ``(Kmax, M)`` 的窗口数组。
    sym : bool, optional
        当为 True（默认）时，生成对称窗口，用于滤波器设计。
        当为 False 时，生成周期性窗口，用于频谱分析。
    norm : {2, 'approximate', 'subsample'} | None, optional
        如果为 'approximate' 或 'subsample'，则通过最大值对窗口进行标准化，并应用一个修正因子来调整偶数长度窗口的大小，修正因子分别为 ``M**2/(M**2+NW)``（"approximate"）或基于 FFT 的子采样移位（"subsample"），详见注释部分。
        如果为 None，则在 ``Kmax=None`` 时使用 "approximate"，否则使用 2（即使用 l2 范数）。
    return_ratios : bool, optional
        如果为 True，则除了窗口，还返回能量集中比例。

    Returns
    -------
    v : ndarray, shape (Kmax, M) or (M,)
        DPSS 窗口。如果 `Kmax` 为 None，则返回 1 维数组。
    r : ndarray, shape (Kmax,) or float, optional
        窗口的能量集中比例。仅在 `return_ratios` 为 True 时返回。如果 `Kmax` 为 None，则返回标量。

    Notes
    -----
    这个计算使用 [2]_ 中给出的三对角特征向量形式。

    对于 ``Kmax=None`` 的默认标准化模式，即窗口生成模式，简单地使用 l-无穷范数会导致具有两个单位值的窗口，这会在偶数和奇数阶数之间造成轻微的标准化差异。为了抵消这种效应，对于偶数样本数，使用 ``M**2/float(M**2+NW)`` 的近似修正（见下面的示例）。

    对于非常长的信号（例如 1e6 元素），计算数量级较短的窗口并使用插值（例如 `scipy.interpolate.interp1d`）来获取长度为 `M` 的锥度可能会很有用，但这通常无法保持锥度之间的正交性。

    .. versionadded:: 1.1

    References
    ----------
    .. [1] Percival DB, Walden WT. Spectral Analysis for Physical Applications:
       Multitaper and Conventional Univariate Techniques.
       Cambridge University Press; 1993.
    """
    .. [2] Slepian, D. Prolate spheroidal wave functions, Fourier analysis, and
       uncertainty V: The discrete case. Bell System Technical Journal,
       Volume 57 (1978), 1371430.
    .. [3] Kaiser, JF, Schafer RW. On the Use of the I0-Sinh Window for
       Spectrum Analysis. IEEE Transactions on Acoustics, Speech and
       Signal Processing. ASSP-28 (1): 105-107; 1980.

    Examples
    --------
    We can compare the window to `kaiser`, which was invented as an alternative
    that was easier to calculate [3]_ (example adapted from
    `here <https://ccrma.stanford.edu/~jos/sasp/Kaiser_DPSS_Windows_Compared.html>`_):

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.signal import windows, freqz
    >>> M = 51
    >>> fig, axes = plt.subplots(3, 2, figsize=(5, 7))
    >>> for ai, alpha in enumerate((1, 3, 5)):
    ...     win_dpss = windows.dpss(M, alpha)
    ...     beta = alpha*np.pi
    ...     win_kaiser = windows.kaiser(M, beta)
    ...     for win, c in ((win_dpss, 'k'), (win_kaiser, 'r')):
    ...         win /= win.sum()
    ...         axes[ai, 0].plot(win, color=c, lw=1.)
    ...         axes[ai, 0].set(xlim=[0, M-1], title=r'$\\alpha$ = %s' % alpha,
    ...                         ylabel='Amplitude')
    ...         w, h = freqz(win)
    ...         axes[ai, 1].plot(w, 20 * np.log10(np.abs(h)), color=c, lw=1.)
    ...         axes[ai, 1].set(xlim=[0, np.pi],
    ...                         title=r'$\\beta$ = %0.2f' % beta,
    ...                         ylabel='Magnitude (dB)')
    >>> for ax in axes.ravel():
    ...     ax.grid(True)
    >>> axes[2, 1].legend(['DPSS', 'Kaiser'])
    >>> fig.tight_layout()
    >>> plt.show()

    And here are examples of the first four windows, along with their
    concentration ratios:

    >>> M = 512
    >>> NW = 2.5
    >>> win, eigvals = windows.dpss(M, NW, 4, return_ratios=True)
    >>> fig, ax = plt.subplots(1)
    >>> ax.plot(win.T, linewidth=1.)
    >>> ax.set(xlim=[0, M-1], ylim=[-0.1, 0.1], xlabel='Samples',
    ...        title='DPSS, M=%d, NW=%0.1f' % (M, NW))
    >>> ax.legend(['win[%d] (%0.4f)' % (ii, ratio)
    ...            for ii, ratio in enumerate(eigvals)])
    >>> fig.tight_layout()
    >>> plt.show()

    Using a standard :math:`l_{\\infty}` norm would produce two unity values
    for even `M`, but only one unity value for odd `M`. This produces uneven
    window power that can be counteracted by the approximate correction
    ``M**2/float(M**2+NW)``, which can be selected by using
    ``norm='approximate'`` (which is the same as ``norm=None`` when
    ``Kmax=None``, as is the case here). Alternatively, the slower
    ``norm='subsample'`` can be used, which uses subsample shifting in the
    frequency domain (FFT) to compute the correction:

    >>> Ms = np.arange(1, 41)
    >>> factors = (50, 20, 10, 5, 2.0001)
    >>> energy = np.empty((3, len(Ms), len(factors)))
    >>> for mi, M in enumerate(Ms):
    # 迭代每个 M 值，并在每个 M 值下迭代 factors 数组中的因子
    ...     for fi, NW in enumerate(factors):
    # 将 DPSS 窗口函数应用于给定的 M 和 NW 值，并返回窗口和特征值
    ...         win, eigvals = windows.dpss(M, NW, 4, return_ratios=True)
    # 将每个窗口函数的幅度平方和存储在能量数组中的相应位置
    ...         energy[:, mi, fi] = np.abs(win)**2
    ...     for fi, factor in enumerate(factors):
    ...         NW = M / float(factor)
    ...         # 根据经验近似修正（默认）
    ...         win = windows.dpss(M, NW)
    ...         # 计算能量，使用 l2 范数进行归一化
    ...         energy[0, mi, fi] = np.sum(win ** 2) / np.sqrt(M)
    ...         # 使用子采样偏移进行修正
    ...         win = windows.dpss(M, NW, norm='subsample')
    ...         # 计算能量，使用 l2 范数进行归一化
    ...         energy[1, mi, fi] = np.sum(win ** 2) / np.sqrt(M)
    ...         # 未修正的情况（使用 l-infinity 范数）
    ...         win /= win.max()
    ...         # 计算能量，使用 l2 范数进行归一化
    ...         energy[2, mi, fi] = np.sum(win ** 2) / np.sqrt(M)
    >>> fig, ax = plt.subplots(1)
    >>> hs = ax.plot(Ms, energy[2], '-o', markersize=4,
    ...              markeredgecolor='none')
    >>> leg = [hs[-1]]
    >>> for hi, hh in enumerate(hs):
    ...     h1 = ax.plot(Ms, energy[0, :, hi], '-o', markersize=4,
    ...                  color=hh.get_color(), markeredgecolor='none',
    ...                  alpha=0.66)
    ...     h2 = ax.plot(Ms, energy[1, :, hi], '-o', markersize=4,
    ...                  color=hh.get_color(), markeredgecolor='none',
    ...                  alpha=0.33)
    ...     if hi == len(hs) - 1:
    ...         leg.insert(0, h1[0])
    ...         leg.insert(0, h2[0])
    >>> ax.set(xlabel='M (samples)', ylabel=r'Power / $\\sqrt{M}$')
    >>> ax.legend(leg, ['Uncorrected', r'Corrected: $\\frac{M^2}{M^2+NW}$',
    ...                 'Corrected (subsample)'])
    >>> fig.tight_layout()

    """
    # 如果输入 M 不符合长度要求，则返回一个全为 1 的数组
    if _len_guards(M):
        return np.ones(M)
    # 如果未指定 norm，则根据情况设定默认值
    if norm is None:
        norm = 'approximate' if Kmax is None else 2
    # 已知的规范选项
    known_norms = (2, 'approximate', 'subsample')
    # 如果指定的 norm 不在已知规范选项中，则引发 ValueError 异常
    if norm not in known_norms:
        raise ValueError(f'norm must be one of {known_norms}, got {norm}')
    # 如果未指定 Kmax，则设定为单值情况
    if Kmax is None:
        singleton = True
        Kmax = 1
    else:
        singleton = False
    # 将 Kmax 转换为整数
    Kmax = operator.index(Kmax)
    # 如果 Kmax 不符合范围要求，则引发 ValueError 异常
    if not 0 < Kmax <= M:
        raise ValueError('Kmax must be greater than 0 and less than M')
    # 如果 NW 大于等于 M/2，则引发 ValueError 异常
    if NW >= M/2.:
        raise ValueError('NW must be less than M/2.')
    # 如果 NW 小于等于 0，则引发 ValueError 异常
    if NW <= 0:
        raise ValueError('NW must be positive')
    # 调用 _extend 函数，根据 sym 参数对 M 进行扩展，并返回扩展后的 M 和是否需要截断的标志
    M, needs_trunc = _extend(M, sym)
    # 计算 W 的值
    W = float(NW) / M
    # 生成一个包含 M 个元素的索引数组
    nidx = np.arange(M)

    # 这里我们希望设置一个优化问题，以找到一个能量在带宽 [-W, W] 内最集中的序列。
    # 因此，度量 lambda(T,W) 是该带内能量和总能量的比率。这导致了特征系统
    # (A - (l1)I)v = 0，其中对应于最大特征值的特征向量是能量最集中的序列。
    # 这个系统的特征向量集合称为 Slepian 序列，或离散椭球序列（DPSS）。仅
    # 前 K 个，K = 2NW/dt 个 DPSS 序列会展示良好的频谱集中性。
    # [详见 https://en.wikipedia.org/wiki/Spectral_concentration_problem]
    # 设置一个替代的对称三对角特征值问题，满足条件 (B - (l2)I)v = 0，其中 v 是 DPSS
    # 主对角线 = ([M-1-2*t]/2)**2 cos(2PIW)，t=[0,1,2,...,M-1]
    # 第一个非对角线 = t(M-t)/2，t=[1,2,...,M-1]
    # 参考文献 Percival 和 Walden, 1993
    d = ((M - 1 - 2 * nidx) / 2.) ** 2 * np.cos(2 * np.pi * W)
    e = nidx[1:] * (M - nidx[1:]) / 2.

    # 只计算最高的 Kmax 个特征值
    w, windows = linalg.eigh_tridiagonal(
        d, e, select='i', select_range=(M - Kmax, M - 1))
    w = w[::-1]  # 将特征值逆序排列
    windows = windows[:, ::-1].T  # 将特征向量逆序排列，并转置

    # 根据约定 (Percival 和 Walden, 1993 第 379 页)
    # * 对称锥形窗（k=0,2,4,...）应该具有正平均值。
    fix_even = (windows[::2].sum(axis=1) < 0)
    for i, f in enumerate(fix_even):
        if f:
            windows[2 * i] *= -1  # 将负的对称锥形窗转为正

    # * 反对称锥形窗应该以正的瓣开始
    #   （这取决于“瓣”的定义，在这里我们将采用第一个高于数值噪声的点，
    #   对于足够平滑的函数来说，这比依赖于使用 max(abs(w)) 的算法更健壮，
    #   后者容易受到数值噪声问题的影响）
    thresh = max(1e-7, 1. / M)
    for i, w in enumerate(windows[1::2]):
        if w[w * w > thresh][0] < 0:
            windows[2 * i + 1] *= -1  # 将负的反对称锥形窗转为正

    # 现在找到原始频谱集中问题的特征值
    # 使用 Percival 和 Walden, 1993 第 390 页的自相关序列技术
    if return_ratios:
        dpss_rxx = _fftautocorr(windows)
        r = 4 * W * np.sinc(2 * W * nidx)
        r[0] = 2 * W
        ratios = np.dot(dpss_rxx, r)
        if singleton:
            ratios = ratios[0]

    # 处理 sym 和 Kmax=None 的情况
    if norm != 2:
        windows /= windows.max()
        if M % 2 == 0:
            if norm == 'approximate':
                correction = M**2 / float(M**2 + NW)
            else:
                s = sp_fft.rfft(windows[0])
                shift = -(1 - 1./M) * np.arange(1, M//2 + 1)
                s[1:] *= 2 * np.exp(-1j * np.pi * shift)
                correction = M / s.real.sum()
            windows *= correction
    # 否则已经是 l2 归一化的，不需操作
    if needs_trunc:
        windows = windows[:, :-1]
    if singleton:
        windows = windows[0]
    return (windows, ratios) if return_ratios else windows
# 定义 Lanczos 窗口函数，也称为 sinc 窗口
def lanczos(M, *, sym=True):
    r"""Return a Lanczos window also known as a sinc window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero, an empty array
        is returned. An exception is thrown when it is negative.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    The Lanczos window is defined as

    .. math::  w(n) = sinc \left( \frac{2n}{M - 1} - 1 \right)

    where

    .. math::  sinc(x) = \frac{\sin(\pi x)}{\pi x}

    The Lanczos window has reduced Gibbs oscillations and is widely used for
    filtering climate timeseries with good properties in the physical and
    spectral domains.

    .. versionadded:: 1.10

    References
    ----------
    .. [1] Lanczos, C., and Teichmann, T. (1957). Applied analysis.
           Physics Today, 10, 44.
    .. [2] Duchon C. E. (1979) Lanczos Filtering in One and Two Dimensions.
           Journal of Applied Meteorology, Vol 18, pp 1016-1022.
    .. [3] Thomson, R. E. and Emery, W. J. (2014) Data Analysis Methods in
           Physical Oceanography (Third Edition), Elsevier, pp 593-637.
    .. [4] Wikipedia, "Window function",
           http://en.wikipedia.org/wiki/Window_function

    Examples
    --------
    Plot the window

    >>> import numpy as np
    >>> from scipy.signal.windows import lanczos
    >>> from scipy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(1)
    >>> window = lanczos(51)
    >>> ax.plot(window)
    >>> ax.set_title("Lanczos window")
    >>> ax.set_ylabel("Amplitude")
    >>> ax.set_xlabel("Sample")
    >>> fig.tight_layout()
    >>> plt.show()

    and its frequency response:

    >>> fig, ax = plt.subplots(1)
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> ax.plot(freq, response)
    >>> ax.set_xlim(-0.5, 0.5)
    >>> ax.set_ylim(-120, 0)
    >>> ax.set_title("Frequency response of the lanczos window")
    >>> ax.set_ylabel("Normalized magnitude [dB]")
    >>> ax.set_xlabel("Normalized frequency [cycles per sample]")
    >>> fig.tight_layout()
    >>> plt.show()
    """
    # 检查 M 的长度保护条件
    if _len_guards(M):
        return np.ones(M)
    # 调用 _extend 函数，根据 sym 参数扩展 M
    M, needs_trunc = _extend(M, sym)

    # 为了确保窗口是对称的，我们连接窗口的右半部分和其镜像的左半部分
    def _calc_right_side_lanczos(n, m):
        return np.sinc(2. * np.arange(n, m) / (m - 1) - 1.0)
    # 如果 M 是偶数，则使用 _calc_right_side_lanczos 函数计算 Lanczos 窗口的右侧部分
    if M % 2 == 0:
        # 计算窗口的一半长度为 M/2 的 Lanczos 窗口右侧部分
        wh = _calc_right_side_lanczos(M/2, M)
        # 将 wh 数组翻转并与原数组连接，形成完整的 Lanczos 窗口数组 w
        w = np.r_[np.flip(wh), wh]
    else:
        # 如果 M 是奇数，则计算窗口的一半长度为 (M+1)/2 的 Lanczos 窗口右侧部分
        wh = _calc_right_side_lanczos((M+1)/2, M)
        # 将 wh 数组翻转，加入中间的单个元素 1.0，再与原数组连接，形成完整的 Lanczos 窗口数组 w
        w = np.r_[np.flip(wh), 1.0, wh]

    # 调用 _truncate 函数截断数组 w，根据需要截断的长度 needs_trunc
    return _truncate(w, needs_trunc)
# 定义一个函数，用于计算实数组的自相关，并裁剪结果
def _fftautocorr(x):
    # 获取数组 x 的最后一个维度的长度
    N = x.shape[-1]
    # 计算大于等于 2*N-1 的最小的快速傅里叶变换长度
    use_N = sp_fft.next_fast_len(2*N-1)
    # 对数组 x 进行快速傅里叶变换，沿着最后一个维度
    x_fft = sp_fft.rfft(x, use_N, axis=-1)
    # 计算傅里叶变换结果的共轭乘积的逆傅里叶变换，并截取前 N 列
    cxy = sp_fft.irfft(x_fft * x_fft.conj(), n=use_N)[:, :N]
    # 或者等效地（但在大多数情况下更慢）：
    # cxy = np.array([np.convolve(xx, yy[::-1], mode='full')
    #                 for xx, yy in zip(x, x)])[:, N-1:2*N-1]
    return cxy

# 创建窗口名称和对应的函数的字典
_win_equiv_raw = {
    ('barthann', 'brthan', 'bth'): (barthann, False),
    ('bartlett', 'bart', 'brt'): (bartlett, False),
    ('blackman', 'black', 'blk'): (blackman, False),
    ('blackmanharris', 'blackharr', 'bkh'): (blackmanharris, False),
    ('bohman', 'bman', 'bmn'): (bohman, False),
    ('boxcar', 'box', 'ones',
        'rect', 'rectangular'): (boxcar, False),
    ('chebwin', 'cheb'): (chebwin, True),
    ('cosine', 'halfcosine'): (cosine, False),
    ('dpss',): (dpss, True),
    ('exponential', 'poisson'): (exponential, False),
    ('flattop', 'flat', 'flt'): (flattop, False),
    ('gaussian', 'gauss', 'gss'): (gaussian, True),
    ('general cosine', 'general_cosine'): (general_cosine, True),
    ('general gaussian', 'general_gaussian',
        'general gauss', 'general_gauss', 'ggs'): (general_gaussian, True),
    ('general hamming', 'general_hamming'): (general_hamming, True),
    ('hamming', 'hamm', 'ham'): (hamming, False),
    ('hann', 'han'): (hann, False),
    ('kaiser', 'ksr'): (kaiser, True),
    ('kaiser bessel derived', 'kbd'): (kaiser_bessel_derived, True),
    ('lanczos', 'sinc'): (lanczos, False),
    ('nuttall', 'nutl', 'nut'): (nuttall, False),
    ('parzen', 'parz', 'par'): (parzen, False),
    ('taylor', 'taylorwin'): (taylor, False),
    ('triangle', 'triang', 'tri'): (triang, False),
    ('tukey', 'tuk'): (tukey, False),
}

# 将所有有效窗口名称字符串填充到字典 _win_equiv 中
_win_equiv = {}
for k, v in _win_equiv_raw.items():
    for key in k:
        _win_equiv[key] = v[0]

# 记录哪些窗口需要额外的参数
_needs_param = set()
for k, v in _win_equiv_raw.items():
    if v[1]:
        _needs_param.update(k)

# 定义函数，根据给定长度和类型返回一个窗口
def get_window(window, Nx, fftbins=True):
    """
    返回指定长度和类型的窗口。

    Parameters
    ----------
    window : string, float, or tuple
        要创建的窗口类型。详见下方详细说明。
    Nx : int
        窗口中的样本数。
    fftbins : bool, optional
        如果为 True（默认），创建一个“周期性”窗口，准备与 `ifftshift` 结合使用，并乘以 FFT 结果
        （参见 :func:`~scipy.fft.fftfreq`）。
        如果为 False，创建一个“对称”窗口，用于滤波器设计。

    Returns
    -------
    get_window : ndarray
        返回长度为 `Nx` 和类型为 `window` 的窗口

    Notes
    -----
    窗口类型:

    - `~scipy.signal.windows.boxcar`
    - `~scipy.signal.windows.triang`
    - `~scipy.signal.windows.blackman`
    """
    """
    If the window requires no parameters, then `window` can be a string.

    If the window requires parameters, then `window` must be a tuple
    with the first argument the string name of the window, and the next
    arguments the needed parameters.

    If `window` is a floating point number, it is interpreted as the beta
    parameter of the `~scipy.signal.windows.kaiser` window.

    Each of the window types listed above is also the name of
    a function that can be called directly to create a window of
    that type.

    Examples
    --------
    >>> from scipy import signal
    >>> signal.get_window('triang', 7)
    array([ 0.125,  0.375,  0.625,  0.875,  0.875,  0.625,  0.375])
    >>> signal.get_window(('kaiser', 4.0), 9)
    array([ 0.08848053,  0.29425961,  0.56437221,  0.82160913,  0.97885093,
            0.97885093,  0.82160913,  0.56437221,  0.29425961])
    >>> signal.get_window(('exponential', None, 1.), 9)
    array([ 0.011109  ,  0.03019738,  0.082085  ,  0.22313016,  0.60653066,
            0.60653066,  0.22313016,  0.082085  ,  0.03019738])
    >>> signal.get_window(4.0, 9)
    array([ 0.08848053,  0.29425961,  0.56437221,  0.82160913,  0.97885093,
            0.97885093,  0.82160913,  0.56437221,  0.29425961])

    """
    # 根据是否为 FFT 点的对称性，确定窗口的对称性
    sym = not fftbins
    # 尝试将窗口参数转换为浮点数
    try:
        beta = float(window)
    # 处理类型错误和值错误的异常
    except (TypeError, ValueError) as e:
        # 初始化空参数元组
        args = ()
        # 如果窗口参数是元组类型
        if isinstance(window, tuple):
            # 取出第一个元素作为窗口字符串
            winstr = window[0]
            # 如果元组长度大于1，剩余部分作为参数
            if len(window) > 1:
                args = window[1:]
        # 如果窗口参数是字符串类型
        elif isinstance(window, str):
            # 如果窗口字符串在需要参数的窗口列表中
            if window in _needs_param:
                # 抛出值错误异常，提示需要传递元组作为参数
                raise ValueError("The '" + window + "' window needs one or "
                                 "more parameters -- pass a tuple.") from e
            else:
                # 否则直接使用窗口字符串
                winstr = window
        else:
            # 如果窗口参数类型不支持，抛出值错误异常
            raise ValueError("%s as window type is not supported." %
                             str(type(window))) from e

        try:
            # 根据窗口字符串从等价窗口映射中获取窗口函数
            winfunc = _win_equiv[winstr]
        except KeyError as e:
            # 如果窗口字符串不在映射中，抛出值错误异常
            raise ValueError("Unknown window type.") from e

        # 如果窗口函数是 dpss
        if winfunc is dpss:
            # 组合参数元组，包括 Nx 和额外的参数，最后一个参数置为 None
            params = (Nx,) + args + (None,)
        else:
            # 否则，组合参数元组，包括 Nx 和其他参数
            params = (Nx,) + args
    else:
        # 如果没有捕获到类型错误或值错误异常
        # 默认使用 kaiser 窗口函数
        winfunc = kaiser
        # 组合参数元组，包括 Nx 和 beta 参数
        params = (Nx, beta)

    # 返回根据窗口函数和参数计算得到的窗口函数值
    return winfunc(*params, sym=sym)
```