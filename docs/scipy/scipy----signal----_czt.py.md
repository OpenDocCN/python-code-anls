# `D:\src\scipysrc\scipy\scipy\signal\_czt.py`

```
# This program is public domain
# Authors: Paul Kienzle, Nadav Horesh
"""
Chirp z-transform.

We provide two interfaces to the chirp z-transform: an object interface
which precalculates part of the transform and can be applied efficiently
to many different data sets, and a functional interface which is applied
only to the given data set.

Transforms
----------

CZT : callable (x, axis=-1) -> array
   Define a chirp z-transform that can be applied to different signals.
ZoomFFT : callable (x, axis=-1) -> array
   Define a Fourier transform on a range of frequencies.

Functions
---------

czt : array
   Compute the chirp z-transform for a signal.
zoom_fft : array
   Compute the Fourier transform on a range of frequencies.
"""

import cmath               # 导入复数数学模块
import numbers             # 导入数字类型模块
import numpy as np         # 导入数值计算模块numpy
from numpy import pi, arange  # 导入pi和arange函数
from scipy.fft import fft, ifft, next_fast_len  # 导入傅里叶变换相关函数

__all__ = ['czt', 'zoom_fft', 'CZT', 'ZoomFFT', 'czt_points']  # 定义模块的公开接口列表

def _validate_sizes(n, m):
    """
    Validate the sizes of input parameters for CZT.

    Parameters
    ----------
    n : int
        Number of CZT data points.
    m : int or None
        Number of CZT output points.

    Returns
    -------
    m : int
        Validated number of CZT output points.

    Raises
    ------
    ValueError
        If n or m is invalid (non-positive or non-integer).
    """
    if n < 1 or not isinstance(n, numbers.Integral):
        raise ValueError('Invalid number of CZT data '
                         f'points ({n}) specified. '
                         'n must be positive and integer type.')

    if m is None:
        m = n
    elif m < 1 or not isinstance(m, numbers.Integral):
        raise ValueError('Invalid number of CZT output '
                         f'points ({m}) specified. '
                         'm must be positive and integer type.')

    return m


def czt_points(m, w=None, a=1+0j):
    """
    Return the points at which the chirp z-transform is computed.

    Parameters
    ----------
    m : int
        The number of points desired.
    w : complex, optional
        The ratio between points in each step.
        Defaults to equally spaced points around the entire unit circle.
    a : complex, optional
        The starting point in the complex plane.  Default is 1+0j.

    Returns
    -------
    out : ndarray
        The points in the Z plane at which `CZT` samples the z-transform,
        when called with arguments `m`, `w`, and `a`, as complex numbers.

    See Also
    --------
    CZT : Class that creates a callable chirp z-transform function.
    czt : Convenience function for quickly calculating CZT.

    Examples
    --------
    Plot the points of a 16-point FFT:

    >>> import numpy as np
    >>> from scipy.signal import czt_points
    >>> points = czt_points(16)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(points.real, points.imag, 'o')
    >>> plt.gca().add_patch(plt.Circle((0,0), radius=1, fill=False, alpha=.3))
    >>> plt.axis('equal')
    >>> plt.show()

    and a 91-point logarithmic spiral that crosses the unit circle:

    >>> m, w, a = 91, 0.995*np.exp(-1j*np.pi*.05), 0.8*np.exp(1j*np.pi/6)
    >>> points = czt_points(m, w, a)
    >>> plt.plot(points.real, points.imag, 'o')
    >>> plt.gca().add_patch(plt.Circle((0,0), radius=1, fill=False, alpha=.3))
    >>> plt.axis('equal')
    >>> plt.show()
    """
    # 调用 _validate_sizes 函数验证参数 m 的有效性，并返回修正后的值
    m = _validate_sizes(1, m)

    # 使用 arange 函数生成一个从 0 到 m-1 的整数数组 k
    k = arange(m)

    # 将数组 a 转换为浮点数，确保其至少为浮点数类型
    a = 1.0 * a  # at least float

    # 如果参数 w 未指定
    if w is None:
        # 未指定具体内容，默认使用 FFT，返回 a 乘以指数函数值数组
        return a * np.exp(2j * pi * k / m)
    else:
        # 如果参数 w 指定了
        # 将数组 w 转换为浮点数，确保其至少为浮点数类型
        w = 1.0 * w  # at least float
        # 返回 a 乘以 w 的负 k 次方的数组
        return a * w**-k
class CZT:
    """
    Create a callable chirp z-transform function.

    Transform to compute the frequency response around a spiral.
    Objects of this class are callables which can compute the
    chirp z-transform on their inputs.  This object precalculates the constant
    chirps used in the given transform.

    Parameters
    ----------
    n : int
        The size of the signal.
    m : int, optional
        The number of output points desired.  Default is `n`.
    w : complex, optional
        The ratio between points in each step.  This must be precise or the
        accumulated error will degrade the tail of the output sequence.
        Defaults to equally spaced points around the entire unit circle.
    a : complex, optional
        The starting point in the complex plane.  Default is 1+0j.

    Returns
    -------
    f : CZT
        Callable object ``f(x, axis=-1)`` for computing the chirp z-transform
        on `x`.

    See Also
    --------
    czt : Convenience function for quickly calculating CZT.
    ZoomFFT : Class that creates a callable partial FFT function.

    Notes
    -----
    The defaults are chosen such that ``f(x)`` is equivalent to
    ``fft.fft(x)`` and, if ``m > len(x)``, that ``f(x, m)`` is equivalent to
    ``fft.fft(x, m)``.

    If `w` does not lie on the unit circle, then the transform will be
    around a spiral with exponentially-increasing radius.  Regardless,
    angle will increase linearly.

    For transforms that do lie on the unit circle, accuracy is better when
    using `ZoomFFT`, since any numerical error in `w` is
    accumulated for long data lengths, drifting away from the unit circle.

    The chirp z-transform can be faster than an equivalent FFT with
    zero padding.  Try it with your own array sizes to see.

    However, the chirp z-transform is considerably less precise than the
    equivalent zero-padded FFT.

    As this CZT is implemented using the Bluestein algorithm, it can compute
    large prime-length Fourier transforms in O(N log N) time, rather than the
    O(N**2) time required by the direct DFT calculation.  (`scipy.fft` also
    uses Bluestein's algorithm'.)

    (The name "chirp z-transform" comes from the use of a chirp in the
    Bluestein algorithm.  It does not decompose signals into chirps, like
    other transforms with "chirp" in the name.)

    References
    ----------
    .. [1] Leo I. Bluestein, "A linear filtering approach to the computation
           of the discrete Fourier transform," Northeast Electronics Research
           and Engineering Meeting Record 10, 218-219 (1968).
    .. [2] Rabiner, Schafer, and Rader, "The chirp z-transform algorithm and
           its application," Bell Syst. Tech. J. 48, 1249-1292 (1969).

    Examples
    --------
    Compute multiple prime-length FFTs:

    >>> from scipy.signal import CZT  # Import the CZT class from scipy.signal
    >>> import numpy as np  # Import numpy for array operations
    >>> a = np.random.rand(7)  # Generate a random array `a` of length 7
    >>> b = np.random.rand(7)  # Generate a random array `b` of length 7
    >>> c = np.random.rand(7)  # Generate a random array `c` of length 7

    """
    >>> czt_7 = CZT(n=7)


# 创建一个 CZT 对象 czt_7，指定 n=7，这里初始化了一个长度为 7 的 CZT 变换器
>>> czt_7 = CZT(n=7)



    >>> A = czt_7(a)


# 对信号 a 进行 CZT 变换，返回结果存储在变量 A 中
>>> A = czt_7(a)



    >>> B = czt_7(b)


# 对信号 b 进行 CZT 变换，返回结果存储在变量 B 中
>>> B = czt_7(b)



    >>> C = czt_7(c)


# 对信号 c 进行 CZT 变换，返回结果存储在变量 C 中
>>> C = czt_7(c)



    Display the points at which the FFT is calculated:


# 显示 FFT 计算的点的位置
    Display the points at which the FFT is calculated:



    >>> czt_7.points()


# 调用 CZT 对象的 points 方法，返回 CZT 变换计算的点
>>> czt_7.points()



    array([ 1.00000000+0.j        ,  0.62348980+0.78183148j,
           -0.22252093+0.97492791j, -0.90096887+0.43388374j,
           -0.90096887-0.43388374j, -0.22252093-0.97492791j,
            0.62348980-0.78183148j])


# 返回 CZT 变换计算的点，这些点是复数数组
array([ 1.00000000+0.j        ,  0.62348980+0.78183148j,
       -0.22252093+0.97492791j, -0.90096887+0.43388374j,
       -0.90096887-0.43388374j, -0.22252093-0.97492791j,
        0.62348980-0.78183148j])



    >>> import matplotlib.pyplot as plt


# 导入 matplotlib 库的 pyplot 模块，用于绘图
>>> import matplotlib.pyplot as plt



    >>> plt.plot(czt_7.points().real, czt_7.points().imag, 'o')


# 绘制 CZT 计算的点的实部和虚部，使用圆点表示
>>> plt.plot(czt_7.points().real, czt_7.points().imag, 'o')



    >>> plt.gca().add_patch(plt.Circle((0,0), radius=1, fill=False, alpha=.3))


# 在当前图形中添加一个半径为1、不填充的透明圆形
>>> plt.gca().add_patch(plt.Circle((0,0), radius=1, fill=False, alpha=.3))



    >>> plt.axis('equal')


# 设置坐标轴的纵横比相等
>>> plt.axis('equal')



    >>> plt.show()
    """


# 显示绘制的图形
>>> plt.show()
# 定义 ZoomFFT 类，继承自 CZT 类
class ZoomFFT(CZT):
    """
    创建一个可调用的 Zoom FFT 变换函数。

    这是针对单位圆周围一组等间距频率的 chirp z-transform（`CZT`）的特化版本，用于比计算整个 FFT 并截断更高效地计算 FFT 的一部分。

    Parameters
    ----------
    n : int
        信号的大小。
    fn : array_like
        长度为2的序列 [`f1`, `f2`] 表示频率范围，或者是一个标量，假定范围为 [0, `fn`]。
    m : int, optional
        要评估的点数。默认为 `n`。
    fs : float, optional
        采样频率。例如，如果 ``fs=10`` 表示10 kHz，则 `f1` 和 `f2` 也应以 kHz 给出。
        默认采样频率为2，因此 `f1` 和 `f2` 应在 [0, 1] 范围内，以保持变换低于 Nyquist 频率。
    endpoint : bool, optional
        如果为 True，则 `f2` 是最后一个样本。否则，它不包括在内。默认为 False。

    Returns
    -------
    f : ZoomFFT
        可调用对象 ``f(x, axis=-1)`` 用于计算 `x` 上的 zoom FFT。

    See Also
    --------
    zoom_fft : 计算 zoom FFT 的便捷函数。

    Notes
    -----
    默认值选得这样，使得 ``f(x, 2)`` 等同于 ``fft.fft(x)``，并且如果 ``m > len(x)``
    那么 ``f(x, 2, m)`` 等同于 ``fft.fft(x, m)``。

    采样频率是 1/dt，`x` 信号中样本之间的时间步长。单位圆对应于从0到采样频率的频率。默认采样频率为2，
    意味着 `f1`、`f2` 值可以达到 Nyquist 频率的范围为 [0, 1)。对于以弧度表示的 `f1`、`f2` 值，
    应使用采样频率 2*pi。

    请记住，zoom FFT 只能插值现有 FFT 的点。它无法帮助解析两个分开的接近频率。
    频率分辨率只能通过增加采集时间来增加。

    这些函数使用 Bluestein 算法实现（与 `scipy.fft` 一样）。[2]_

    References
    ----------
    .. [1] Steve Alan Shilling, "A study of the chirp z-transform and its
           applications", pg 29 (1970)
           https://krex.k-state.edu/dspace/bitstream/handle/2097/7844/LD2668R41972S43.pdf
    .. [2] Leo I. Bluestein, "A linear filtering approach to the computation
           of the discrete Fourier transform," Northeast Electronics Research
           and Engineering Meeting Record 10, 218-219 (1968).

    Examples
    --------
    要绘制变换结果，请使用以下示例：

    >>> import numpy as np
    >>> from scipy.signal import ZoomFFT
    >>> t = np.linspace(0, 1, 1021)
    >>> x = np.cos(2*np.pi*15*t) + np.sin(2*np.pi*17*t)
    >>> f1, f2 = 5, 27
    >>> transform = ZoomFFT(len(x), [f1, f2], len(x), fs=1021)
    >>> X = transform(x)
    >>> f = np.linspace(f1, f2, len(x))
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(f, 20*np.log10(np.abs(X)))
    >>> plt.show()
    """
这部分代码是一个文档字符串，通常用于说明函数或类的作用和用法，这里可能是一个类的初始化方法。

    def __init__(self, n, fn, m=None, *, fs=2, endpoint=False):
初始化方法，用于初始化类的实例。

        m = _validate_sizes(n, m)
调用 `_validate_sizes` 函数验证并确定 `m` 的值。

        k = arange(max(m, n), dtype=np.min_scalar_type(-max(m, n)**2))
生成一个数组 `k`，用作后续计算中的索引。

        if np.size(fn) == 2:
            f1, f2 = fn
        elif np.size(fn) == 1:
            f1, f2 = 0.0, fn
        else:
            raise ValueError('fn must be a scalar or 2-length sequence')
根据 `fn` 的大小确定 `f1` 和 `f2` 的值，这些值是频率的范围。

        self.f1, self.f2, self.fs = f1, f2, fs
将计算得到的 `f1`、`f2` 和传入的 `fs` 赋值给实例变量。

        if endpoint:
            scale = ((f2 - f1) * m) / (fs * (m - 1))
        else:
            scale = (f2 - f1) / fs
根据 `endpoint` 的值计算 `scale`，它是一个缩放因子。

        a = cmath.exp(2j * pi * f1/fs)
计算 `a`，这是一个复数，用于后续的复数运算。

        wk2 = np.exp(-(1j * pi * scale * k**2) / m)
计算 `wk2`，这是一个复数数组，用于频谱变换。

        self.w = cmath.exp(-2j*pi/m * scale)
        self.a = a
        self.m, self.n = m, n
设置实例变量 `w`、`a`，以及 `m` 和 `n` 的值。

        ak = np.exp(-2j * pi * f1/fs * k[:n])
计算 `ak`，这是一个复数数组，用于频谱变换中的系数。

        self._Awk2 = ak * wk2[:n]
设置实例变量 `_Awk2`，它是 `ak` 与 `wk2` 的前 `n` 个元素的乘积。

        nfft = next_fast_len(n + m - 1)
计算 `nfft`，这是一个快速傅里叶变换的长度。

        self._nfft = nfft
        self._Fwk2 = fft(1/np.hstack((wk2[n-1:0:-1], wk2[:m])), nfft)
进行傅里叶变换，计算 `_Fwk2`。

        self._wk2 = wk2[:m]
设置实例变量 `_wk2`，它是 `wk2` 的前 `m` 个元素。

        self._yidx = slice(n-1, n+m-1)
设置实例变量 `_yidx`，它是一个切片对象。
# 导入必要的库函数
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import czt, czt_points

# 定义一个函数，计算在 Z 平面上螺旋形周围的频率响应
def czt(x, m=None, w=None, a=1+0j, *, axis=-1):
    """
    Compute the frequency response around a spiral in the Z plane.

    Parameters
    ----------
    x : array
        The signal to transform.
    m : int, optional
        The number of output points desired.  Default is the length of the
        input data.
    w : complex, optional
        The ratio between points in each step.  This must be precise or the
        accumulated error will degrade the tail of the output sequence.
        Defaults to equally spaced points around the entire unit circle.
    a : complex, optional
        The starting point in the complex plane.  Default is 1+0j.
    axis : int, optional
        Axis over which to compute the FFT. If not given, the last axis is
        used.

    Returns
    -------
    out : ndarray
        An array of the same dimensions as `x`, but with the length of the
        transformed axis set to `m`.

    See Also
    --------
    CZT : Class that creates a callable chirp z-transform function.
    zoom_fft : Convenience function for partial FFT calculations.

    Notes
    -----
    The defaults are chosen such that ``signal.czt(x)`` is equivalent to
    ``fft.fft(x)`` and, if ``m > len(x)``, that ``signal.czt(x, m)`` is
    equivalent to ``fft.fft(x, m)``.

    If the transform needs to be repeated, use `CZT` to construct a
    specialized transform function which can be reused without
    recomputing constants.

    An example application is in system identification, repeatedly evaluating
    small slices of the z-transform of a system, around where a pole is
    expected to exist, to refine the estimate of the pole's true location. [1]_

    References
    ----------
    .. [1] Steve Alan Shilling, "A study of the chirp z-transform and its
           applications", pg 20 (1970)
           https://krex.k-state.edu/dspace/bitstream/handle/2097/7844/LD2668R41972S43.pdf

    Examples
    --------
    Generate a sinusoid:

    >>> f1, f2, fs = 8, 10, 200  # Hz
    >>> t = np.linspace(0, 1, fs, endpoint=False)
    >>> x = np.sin(2*np.pi*t*f2)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(t, x)
    >>> plt.axis([0, 1, -1.1, 1.1])
    >>> plt.show()

    Its discrete Fourier transform has all of its energy in a single frequency
    bin:

    >>> plt.plot(rfftfreq(fs, 1/fs), abs(rfft(x)))
    >>> plt.margins(0, 0.1)
    >>> plt.show()

    However, if the sinusoid is logarithmically-decaying:

    >>> x = np.exp(-t*f1) * np.sin(2*np.pi*t*f2)
    >>> plt.plot(t, x)
    >>> plt.axis([0, 1, -1.1, 1.1])
    >>> plt.show()

    the DFT will have spectral leakage:

    >>> plt.plot(rfftfreq(fs, 1/fs), abs(rfft(x)))
    >>> plt.margins(0, 0.1)
    >>> plt.show()

    While the DFT always samples the z-transform around the unit circle, the
    chirp z-transform allows us to sample the Z-transform along any
    """
    # 执行 chirp z-transform 计算
    return czt(x, m=m, w=w, a=a, axis=axis)
    x = np.asarray(x)  # 将输入的数据 x 转换为 NumPy 数组，确保 x 是可操作的数组对象
    transform = CZT(x.shape[axis], m=m, w=w, a=a)  # 创建 CZT 变换对象，使用指定的参数进行初始化
    return transform(x, axis=axis)  # 对输入数据 x 进行 CZT 变换，返回变换后的结果
# 定义一个函数 zoom_fft，用于计算信号 `x` 的离散傅里叶变换 (DFT)，仅计算频率范围在 `fn` 内的部分。

def zoom_fft(x, fn, m=None, *, fs=2, endpoint=False, axis=-1):
    """
    Compute the DFT of `x` only for frequencies in range `fn`.

    Parameters
    ----------
    x : array
        The signal to transform.
    fn : array_like
        A length-2 sequence [`f1`, `f2`] giving the frequency range, or a
        scalar, for which the range [0, `fn`] is assumed.
    m : int, optional
        The number of points to evaluate.  The default is the length of `x`.
    fs : float, optional
        The sampling frequency.  If ``fs=10`` represented 10 kHz, for example,
        then `f1` and `f2` would also be given in kHz.
        The default sampling frequency is 2, so `f1` and `f2` should be
        in the range [0, 1] to keep the transform below the Nyquist
        frequency.
    endpoint : bool, optional
        If True, `f2` is the last sample. Otherwise, it is not included.
        Default is False.
    axis : int, optional
        Axis over which to compute the FFT. If not given, the last axis is
        used.

    Returns
    -------
    out : ndarray
        The transformed signal.  The Fourier transform will be calculated
        at the points f1, f1+df, f1+2df, ..., f2, where df=(f2-f1)/m.

    See Also
    --------
    ZoomFFT : Class that creates a callable partial FFT function.

    Notes
    -----
    The defaults are chosen such that ``signal.zoom_fft(x, 2)`` is equivalent
    to ``fft.fft(x)`` and, if ``m > len(x)``, that ``signal.zoom_fft(x, 2, m)``
    is equivalent to ``fft.fft(x, m)``.

    To graph the magnitude of the resulting transform, use::

        plot(linspace(f1, f2, m, endpoint=False), abs(zoom_fft(x, [f1, f2], m)))

    If the transform needs to be repeated, use `ZoomFFT` to construct
    a specialized transform function which can be reused without
    recomputing constants.

    Examples
    --------
    To plot the transform results use something like the following:

    >>> import numpy as np
    >>> from scipy.signal import zoom_fft
    >>> t = np.linspace(0, 1, 1021)
    >>> x = np.cos(2*np.pi*15*t) + np.sin(2*np.pi*17*t)
    >>> f1, f2 = 5, 27
    >>> X = zoom_fft(x, [f1, f2], len(x), fs=1021)
    >>> f = np.linspace(f1, f2, len(x))
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(f, 20*np.log10(np.abs(X)))
    >>> plt.show()
    """
    # 将输入信号 `x` 转换为 NumPy 数组
    x = np.asarray(x)
    # 创建一个 ZoomFFT 对象，用于计算指定频率范围内的傅里叶变换
    transform = ZoomFFT(x.shape[axis], fn, m=m, fs=fs, endpoint=endpoint)
    # 调用 transform 对象，计算指定轴上的傅里叶变换，并返回结果
    return transform(x, axis=axis)
```