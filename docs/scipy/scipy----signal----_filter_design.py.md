# `D:\src\scipysrc\scipy\scipy\signal\_filter_design.py`

```
"""Filter design."""
# 导入必要的模块和库
import math
import operator
import warnings

import numpy as np
# 导入多个 numpy 函数和类，用于处理多项式、数组操作等
from numpy import (atleast_1d, poly, polyval, roots, real, asarray,
                   resize, pi, absolute, sqrt, tan, log10,
                   arcsinh, sin, exp, cosh, arccosh, ceil, conjugate,
                   zeros, sinh, append, concatenate, prod, ones, full, array,
                   mintypecode)
# 导入 numpy.polynomial 中的特定函数
from numpy.polynomial.polynomial import polyval as npp_polyval
from numpy.polynomial.polynomial import polyvalfromroots

# 导入 scipy 中的特定模块和函数
from scipy import special, optimize, fft as sp_fft
from scipy.special import comb
from scipy._lib._util import float_factorial
from scipy.signal._arraytools import _validate_fs


__all__ = ['findfreqs', 'freqs', 'freqz', 'tf2zpk', 'zpk2tf', 'normalize',
           'lp2lp', 'lp2hp', 'lp2bp', 'lp2bs', 'bilinear', 'iirdesign',
           'iirfilter', 'butter', 'cheby1', 'cheby2', 'ellip', 'bessel',
           'band_stop_obj', 'buttord', 'cheb1ord', 'cheb2ord', 'ellipord',
           'buttap', 'cheb1ap', 'cheb2ap', 'ellipap', 'besselap',
           'BadCoefficients', 'freqs_zpk', 'freqz_zpk',
           'tf2sos', 'sos2tf', 'zpk2sos', 'sos2zpk', 'group_delay',
           'sosfreqz', 'iirnotch', 'iirpeak', 'bilinear_zpk',
           'lp2lp_zpk', 'lp2hp_zpk', 'lp2bp_zpk', 'lp2bs_zpk',
           'gammatone', 'iircomb']


class BadCoefficients(UserWarning):
    """Warning about badly conditioned filter coefficients"""
    pass


abs = absolute


def _is_int_type(x):
    """
    Check if input is of a scalar integer type (so ``5`` and ``array(5)`` will
    pass, while ``5.0`` and ``array([5])`` will fail.
    """
    if np.ndim(x) != 0:
        # Older versions of NumPy did not raise for np.array([1]).__index__()
        # This is safe to remove when support for those versions is dropped
        return False
    try:
        operator.index(x)
    except TypeError:
        return False
    else:
        return True


def findfreqs(num, den, N, kind='ba'):
    """
    Find array of frequencies for computing the response of an analog filter.

    Parameters
    ----------
    num, den : array_like, 1-D
        The polynomial coefficients of the numerator and denominator of the
        transfer function of the filter or LTI system, where the coefficients
        are ordered from highest to lowest degree. Or, the roots  of the
        transfer function numerator and denominator (i.e., zeroes and poles).
    N : int
        The length of the array to be computed.
    kind : str {'ba', 'zp'}, optional
        Specifies whether the numerator and denominator are specified by their
        polynomial coefficients ('ba'), or their roots ('zp').

    Returns
    -------
    w : (N,) ndarray
        A 1-D array of frequencies, logarithmically spaced.

    Examples
    --------
    Find a set of nine frequencies that span the "interesting part" of the
    frequency response for the filter with the transfer function

        H(s) = s / (s^2 + 8s + 25)
    """
    # 实现省略，根据输入参数计算频率数组并返回
    pass
    # 导入 scipy.signal 库中的 findfreqs 函数
    >>> from scipy import signal
    # 调用 findfreqs 函数，计算传递函数的频率响应
    >>> signal.findfreqs([1, 0], [1, 8, 25], N=9)
    # 返回频率数组，表示频率响应的离散频率点
    array([  1.00000000e-02,   3.16227766e-02,   1.00000000e-01,
             3.16227766e-01,   1.00000000e+00,   3.16227766e+00,
             1.00000000e+01,   3.16227766e+01,   1.00000000e+02])
    """
    # 如果 kind 参数为 'ba'，执行以下操作
    if kind == 'ba':
        # 计算分母多项式的根，并转换为至少为 1 维的复数数组
        ep = atleast_1d(roots(den)) + 0j
        # 计算分子多项式的根，并转换为至少为 1 维的复数数组
        tz = atleast_1d(roots(num)) + 0j
    # 如果 kind 参数为 'zp'，执行以下操作
    elif kind == 'zp':
        # 将分母多项式视为根，并转换为至少为 1 维的复数数组
        ep = atleast_1d(den) + 0j
        # 将分子多项式视为根，并转换为至少为 1 维的复数数组
        tz = atleast_1d(num) + 0j
    # 如果 kind 参数既不是 'ba' 也不是 'zp'，抛出 ValueError 异常
    else:
        raise ValueError("input must be one of {'ba', 'zp'}")
    
    # 如果根数组 ep 的长度为 0，则将其至少视为 1 维的复数数组，值设为 -1000
    if len(ep) == 0:
        ep = atleast_1d(-1000) + 0j
    
    # 将满足条件的根数组 ep 和 tz 合并为一个数组 ez
    ez = np.r_[ep[ep.imag >= 0], tz[(np.abs(tz) < 1e5) & (tz.imag >= 0)]]
    
    # 计算整数部分为零的根的数量，并将结果保存在 integ 中
    integ = np.abs(ez) < 1e-10
    # 计算高频部分的近似值，并四舍五入到最接近的整数
    hfreq = np.round(np.log10(np.max(3 * np.abs(ez.real + integ) +
                                     1.5 * ez.imag)) + 0.5)
    # 计算低频部分的近似值，并四舍五入到最接近的整数
    lfreq = np.round(np.log10(0.1 * np.min(np.abs((ez + integ).real) +
                                           2 * ez.imag)) - 0.5)
    
    # 生成以 10 为底的对数刻度的频率数组，从 lfreq 到 hfreq，共 N 个点
    w = np.logspace(lfreq, hfreq, N)
    # 返回计算得到的频率数组 w
    return w
# 计算模拟滤波器的频率响应

def freqs(b, a, worN=200, plot=None):
    """
    Compute frequency response of analog filter.

    Given the M-order numerator `b` and N-order denominator `a` of an analog
    filter, compute its frequency response::

             b[0]*(jw)**M + b[1]*(jw)**(M-1) + ... + b[M]
     H(w) = ----------------------------------------------
             a[0]*(jw)**N + a[1]*(jw)**(N-1) + ... + a[N]

    Parameters
    ----------
    b : array_like
        Numerator of a linear filter.
    a : array_like
        Denominator of a linear filter.
    worN : {None, int, array_like}, optional
        If None, then compute at 200 frequencies around the interesting parts
        of the response curve (determined by pole-zero locations). If a single
        integer, then compute at that many frequencies. Otherwise, compute the
        response at the angular frequencies (e.g., rad/s) given in `worN`.
    plot : callable, optional
        A callable that takes two arguments. If given, the return parameters
        `w` and `h` are passed to plot. Useful for plotting the frequency
        response inside `freqs`.

    Returns
    -------
    w : ndarray
        The angular frequencies at which `h` was computed.
    h : ndarray
        The frequency response.

    See Also
    --------
    freqz : Compute the frequency response of a digital filter.

    Notes
    -----
    Using Matplotlib's "plot" function as the callable for `plot` produces
    unexpected results, this plots the real part of the complex transfer
    function, not the magnitude. Try ``lambda w, h: plot(w, abs(h))``.

    Examples
    --------
    >>> from scipy.signal import freqs, iirfilter
    >>> import numpy as np

    >>> b, a = iirfilter(4, [1, 10], 1, 60, analog=True, ftype='cheby1')

    >>> w, h = freqs(b, a, worN=np.logspace(-1, 2, 1000))

    >>> import matplotlib.pyplot as plt
    >>> plt.semilogx(w, 20 * np.log10(abs(h)))
    >>> plt.xlabel('Frequency')
    >>> plt.ylabel('Amplitude response [dB]')
    >>> plt.grid(True)
    >>> plt.show()

    """
    if worN is None:
        # 对于向后兼容性
        w = findfreqs(b, a, 200)
    elif _is_int_type(worN):
        w = findfreqs(b, a, worN)
    else:
        w = atleast_1d(worN)

    # 将角频率转换为复数频率
    s = 1j * w
    # 计算滤波器的复频率响应
    h = polyval(b, s) / polyval(a, s)
    if plot is not None:
        # 如果提供了绘图函数，调用绘图函数绘制频率响应
        plot(w, h)

    # 返回计算的角频率和频率响应
    return w, h


def freqs_zpk(z, p, k, worN=200):
    """
    Compute frequency response of analog filter.

    Given the zeros `z`, poles `p`, and gain `k` of a filter, compute its
    frequency response::

                (jw-z[0]) * (jw-z[1]) * ... * (jw-z[-1])
     H(w) = k * ----------------------------------------
                (jw-p[0]) * (jw-p[1]) * ... * (jw-p[-1])

    Parameters
    ----------
    z : array_like
        Zeroes of a linear filter
    p : array_like
        Poles of a linear filter
    k : scalar
        Gain of a linear filter

    """
    k = np.asarray(k)
    # 将 k 转换为 NumPy 数组，以确保统一处理方式

    if k.size > 1:
        # 如果 k 的大小大于 1，抛出数值错误，因为 k 必须是一个标量增益
        raise ValueError('k must be a single scalar gain')

    if worN is None:
        # 如果 worN 为 None，则根据极点零点的位置，计算在感兴趣的响应曲线部分周围的 200 个频率点
        w = findfreqs(z, p, 200, kind='zp')
    elif _is_int_type(worN):
        # 如果 worN 是整数类型，则计算 worN 个频率点
        w = findfreqs(z, p, worN, kind='zp')
    else:
        # 否则，直接使用 worN 作为频率数组
        w = worN

    w = atleast_1d(w)
    # 确保 w 至少是一维数组
    s = 1j * w
    # 计算复数频率 s = jω

    num = polyvalfromroots(s, z)
    # 计算以 z 为根的多项式在 s 处的值
    den = polyvalfromroots(s, p)
    # 计算以 p 为根的多项式在 s 处的值

    h = k * num/den
    # 计算频率响应 h = k * num(s) / den(s)

    return w, h
    # 返回频率数组 w 和频率响应 h
# 定义频率响应函数 `freqz`，计算数字滤波器的频率响应

def freqz(b, a=1, worN=512, whole=False, plot=None, fs=2*pi,
          include_nyquist=False):
    """
    Compute the frequency response of a digital filter.

    Given the M-order numerator `b` and N-order denominator `a` of a digital
    filter, compute its frequency response::

                 jw                 -jw              -jwM
        jw    B(e  )    b[0] + b[1]e    + ... + b[M]e
     H(e  ) = ------ = -----------------------------------
                 jw                 -jw              -jwN
              A(e  )    a[0] + a[1]e    + ... + a[N]e

    Parameters
    ----------
    b : array_like
        Numerator of a linear filter. If `b` has dimension greater than 1,
        it is assumed that the coefficients are stored in the first dimension,
        and ``b.shape[1:]``, ``a.shape[1:]``, and the shape of the frequencies
        array must be compatible for broadcasting.
    a : array_like, optional
        Denominator of a linear filter. If `b` has dimension greater than 1,
        it is assumed that the coefficients are stored in the first dimension,
        and ``b.shape[1:]``, ``a.shape[1:]``, and the shape of the frequencies
        array must be compatible for broadcasting.
    worN : {None, int, array_like}, optional
        If a single integer, then compute at that many frequencies (default is
        N=512). This is a convenient alternative to::

            np.linspace(0, fs if whole else fs/2, N, endpoint=include_nyquist)

        Using a number that is fast for FFT computations can result in
        faster computations (see Notes).

        If an array_like, compute the response at the frequencies given.
        These are in the same units as `fs`.
    whole : bool, optional
        Normally, frequencies are computed from 0 to the Nyquist frequency,
        fs/2 (upper-half of unit-circle). If `whole` is True, compute
        frequencies from 0 to fs. Ignored if worN is array_like.
    plot : callable, optional
        A callable that takes two arguments. If given, the return parameters
        `w` and `h` are passed to plot. Useful for plotting the frequency
        response inside `freqz`.
    fs : float, optional
        The sampling frequency of the digital system. Defaults to 2*pi
        radians/sample (so w is from 0 to pi).

        .. versionadded:: 1.2.0
    include_nyquist : bool, optional
        If `whole` is False and `worN` is an integer, setting `include_nyquist`
        to True will include the last frequency (Nyquist frequency) and is
        otherwise ignored.

        .. versionadded:: 1.5.0

    Returns
    -------
    w : ndarray
        The frequencies at which `h` was computed, in the same units as `fs`.
        By default, `w` is normalized to the range [0, pi) (radians/sample).
    h : ndarray
        The frequency response, as complex numbers.

    See Also
    --------
    freqz_zpk
    sosfreqz

    Notes
    -----
    Using Matplotlib's :func:`matplotlib.pyplot.plot` function as the callable
    """
    # 函数体内部尚未提供，根据参数和选项计算数字滤波器的频率响应
    # 将输入的系数 `b` 和 `a` 转换为至少是一维数组
    b = atleast_1d(b)
    a = atleast_1d(a)
    # 确保文件系统参数 fs 有效，不允许为空
    fs = _validate_fs(fs, allow_none=False)

    # 如果 worN 为 None，则为了向后兼容性设置其为 512
    if worN is None:
        worN = 512

    h = None  # 初始化变量 h 为 None

    # 如果 worN 是整数类型
    if _is_int_type(worN):
        N = operator.index(worN)  # 将 worN 转换为整数 N
        del worN  # 删除 worN 变量
        # 如果 N 小于 0，则抛出异常
        if N < 0:
            raise ValueError(f'worN must be nonnegative, got {N}')

        # 根据参数 whole 设置最后一个点的位置
        lastpoint = 2 * pi if whole else pi

        # 根据参数 include_nyquist 和 whole 生成频率数组 w
        # 如果 include_nyquist 为 True 并且 whole 为 False，则包含终点
        w = np.linspace(0, lastpoint, N,
                        endpoint=include_nyquist and not whole)

        # 根据参数 whole 和 include_nyquist 计算 FFT 的点数 n_fft
        n_fft = N if whole else 2 * (N - 1) if include_nyquist else 2 * N

        # 如果条件满足，选择合适的 FFT 函数 fft_func 进行计算
        if (a.size == 1 and (b.ndim == 1 or (b.shape[-1] == 1))
                and n_fft >= b.shape[0]
                and n_fft > 0):
            # 根据数组类型选择 FFT 函数
            if np.isrealobj(b) and np.isrealobj(a):
                fft_func = sp_fft.rfft
            else:
                fft_func = sp_fft.fft

            # 计算 FFT 并存储结果到 h
            h = fft_func(b, n=n_fft, axis=0)[:N]
            h /= a  # 对 h 进行除法操作

            # 如果使用了实部 FFT 函数并且 whole 为 True
            if fft_func is sp_fft.rfft and whole:
                # 排除直流分量和可能的 Nyquist 频率
                stop = -1 if n_fft % 2 == 1 else -2
                h_flip = slice(stop, 0, -1)
                h = np.concatenate((h, h[h_flip].conj()))  # 连接数组 h 和其反转的共轭部分

            # 如果 b 的维度大于 1，去除 h 的最后一个维度
            if b.ndim > 1:
                h = h[..., 0]  # 去除 h 的最后一个维度
                h = np.moveaxis(h, 0, -1)  # 将 h 的第一个维度移动到最后

    else:
        w = atleast_1d(worN)  # 将 worN 至少转换为 1 维数组
        del worN
        w = 2*pi*w/fs  # 根据频率数组 w 调整值

    # 如果 h 仍然为 None，则使用频率数组 w 计算响应 h
    if h is None:
        zm1 = exp(-1j * w)
        h = (npp_polyval(zm1, b, tensor=False) /
             npp_polyval(zm1, a, tensor=False))

    w = w*(fs/(2*pi))  # 调整频率数组 w 的值

    # 如果 plot 不为 None，则绘制图形
    if plot is not None:
        plot(w, h)

    # 返回频率数组 w 和响应 h
    return w, h
def freqz_zpk(z, p, k, worN=512, whole=False, fs=2*pi):
    r"""
    Compute the frequency response of a digital filter in ZPK form.

    Given the Zeros, Poles and Gain of a digital filter, compute its frequency
    response:

    :math:`H(z)=k \prod_i (z - Z[i]) / \prod_j (z - P[j])`

    where :math:`k` is the `gain`, :math:`Z` are the `zeros` and :math:`P` are
    the `poles`.

    Parameters
    ----------
    z : array_like
        Zeroes of a linear filter
    p : array_like
        Poles of a linear filter
    k : scalar
        Gain of a linear filter
    worN : {None, int, array_like}, optional
        If a single integer, then compute at that many frequencies (default is
        N=512).

        If an array_like, compute the response at the frequencies given.
        These are in the same units as `fs`.
    whole : bool, optional
        Normally, frequencies are computed from 0 to the Nyquist frequency,
        fs/2 (upper-half of unit-circle). If `whole` is True, compute
        frequencies from 0 to fs. Ignored if w is array_like.
    fs : float, optional
        The sampling frequency of the digital system. Defaults to 2*pi
        radians/sample (so w is from 0 to pi).

        .. versionadded:: 1.2.0

    Returns
    -------
    w : ndarray
        The frequencies at which `h` was computed, in the same units as `fs`.
        By default, `w` is normalized to the range [0, pi) (radians/sample).
    h : ndarray
        The frequency response, as complex numbers.

    See Also
    --------
    freqs : Compute the frequency response of an analog filter in TF form
    freqs_zpk : Compute the frequency response of an analog filter in ZPK form
    freqz : Compute the frequency response of a digital filter in TF form

    Notes
    -----
    .. versionadded:: 0.19.0

    Examples
    --------
    Design a 4th-order digital Butterworth filter with cut-off of 100 Hz in a
    system with sample rate of 1000 Hz, and plot the frequency response:

    >>> import numpy as np
    >>> from scipy import signal
    >>> z, p, k = signal.butter(4, 100, output='zpk', fs=1000)
    >>> w, h = signal.freqz_zpk(z, p, k, fs=1000)

    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(1, 1, 1)
    >>> ax1.set_title('Digital filter frequency response')

    >>> ax1.plot(w, 20 * np.log10(abs(h)), 'b')
    >>> ax1.set_ylabel('Amplitude [dB]', color='b')
    >>> ax1.set_xlabel('Frequency [Hz]')
    >>> ax1.grid(True)

    >>> ax2 = ax1.twinx()
    >>> angles = np.unwrap(np.angle(h))
    >>> ax2.plot(w, angles, 'g')
    >>> ax2.set_ylabel('Angle [radians]', color='g')

    >>> plt.axis('tight')
    >>> plt.show()

    """
    # Ensure z and p are at least 1-dimensional arrays
    z, p = map(atleast_1d, (z, p))

    # Validate the sampling frequency fs, ensuring it is not None
    fs = _validate_fs(fs, allow_none=False)

    # Determine the endpoint of the frequency range based on 'whole' flag
    if whole:
        lastpoint = 2 * pi
    else:
        lastpoint = pi

    # If worN is None, use default number of frequencies (512)
    if worN is None:
        # For backwards compatibility, use 512 frequencies
        w = np.linspace(0, lastpoint, 512, endpoint=False)
    # 如果输入的参数 worN 是整数类型，则生成一个从 0 到 lastpoint 的等间距数组 w
    elif _is_int_type(worN):
        w = np.linspace(0, lastpoint, worN, endpoint=False)
    # 如果输入的参数 worN 不是整数类型，则将其至少视为一维数组，并转换为角频率
    else:
        w = atleast_1d(worN)
        w = 2*pi*w/fs

    # 计算单位圆上复数的指数，即 exp(j * w)
    zm1 = exp(1j * w)
    # 计算频率响应 h，使用了传递函数的根据单位圆上的复数 zm1 计算多项式值
    h = k * polyvalfromroots(zm1, z) / polyvalfromroots(zm1, p)

    # 将角频率 w 转换为以 Hz 为单位的频率
    w = w*(fs/(2*pi))

    # 返回频率 w 和频率响应 h
    return w, h
    # 如果未指定频率参数 `w`，则默认为 512，用于向后兼容
    if w is None:
        w = 512

    # 验证并确保采样频率 `fs` 是有效的，不允许为 None
    fs = _validate_fs(fs, allow_none=False)
    # 检查输入的频率参数是否为整数类型
    if _is_int_type(w):
        # 如果是整数类型，根据是否为整体频率设置不同的角频率范围
        if whole:
            w = np.linspace(0, 2 * pi, w, endpoint=False)
        else:
            w = np.linspace(0, pi, w, endpoint=False)
    else:
        # 如果不是整数类型，确保将频率参数转换为至少包含一个维度的数组
        w = np.atleast_1d(w)
        # 将非整数频率参数转换为对应的角频率
        w = 2*pi*w/fs

    # 提取系统的分子和分母系数，并确保它们至少包含一个维度
    b, a = map(np.atleast_1d, system)
    # 计算系统的传递函数的共轭形式的卷积
    c = np.convolve(b, conjugate(a[::-1]))
    # 计算卷积结果与其索引的乘积
    cr = c * np.arange(c.size)
    # 计算指数函数，用于频率响应的复数角频率
    z = np.exp(-1j * w)
    # 计算传递函数的分子多项式在复数角频率处的值
    num = np.polyval(cr[::-1], z)
    # 计算传递函数的分母多项式在复数角频率处的值
    den = np.polyval(c[::-1], z)
    # 计算群延迟（group delay），并减去系统的阶数加一
    gd = np.real(num / den) - a.size + 1
    # 检查是否存在群延迟无穷大的情况
    singular = ~np.isfinite(gd)
    # 检查是否存在分母接近零的情况
    near_singular = np.absolute(den) < 10 * EPSILON

    # 处理群延迟无穷大的情况，将其设为零，并发出警告
    if np.any(singular):
        gd[singular] = 0
        warnings.warn(
            "The group delay is singular at frequencies [{}], setting to 0".
            format(", ".join(f"{ws:.3f}" for ws in w[singular])),
            stacklevel=2
        )

    # 处理分母接近零的情况，发出警告
    elif np.any(near_singular):
        warnings.warn(
            "The filter's denominator is extremely small at frequencies [{}], \
            around which a singularity may be present".
            format(", ".join(f"{ws:.3f}" for ws in w[near_singular])),
            stacklevel=2
        )

    # 将频率参数转换为以 Hz 为单位
    w = w*(fs/(2*pi))

    # 返回转换后的频率和计算得到的群延迟
    return w, gd
def _validate_sos(sos):
    """Helper to validate a SOS input"""
    # 将输入的 sos 转换为至少是二维的 numpy 数组
    sos = np.atleast_2d(sos)
    # 如果 sos 不是二维数组，则抛出数值错误异常
    if sos.ndim != 2:
        raise ValueError('sos array must be 2D')
    # 获取 sos 数组的形状信息
    n_sections, m = sos.shape
    # 如果 sos 的每行不是长度为 6，则抛出数值错误异常
    if m != 6:
        raise ValueError('sos array must be shape (n_sections, 6)')
    # 如果 sos 每个二阶段中第四列不全为 1，则抛出数值错误异常
    if not (sos[:, 3] == 1).all():
        raise ValueError('sos[:, 3] should be all ones')
    # 返回验证后的 sos 数组以及二阶段的数量
    return sos, n_sections


def sosfreqz(sos, worN=512, whole=False, fs=2*pi):
    r"""
    Compute the frequency response of a digital filter in SOS format.

    Given `sos`, an array with shape (n, 6) of second order sections of
    a digital filter, compute the frequency response of the system function::

               B0(z)   B1(z)         B{n-1}(z)
        H(z) = ----- * ----- * ... * ---------
               A0(z)   A1(z)         A{n-1}(z)

    for z = exp(omega*1j), where B{k}(z) and A{k}(z) are numerator and
    denominator of the transfer function of the k-th second order section.

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. Each row corresponds to a second-order
        section, with the first three columns providing the numerator
        coefficients and the last three providing the denominator
        coefficients.
    worN : {None, int, array_like}, optional
        If a single integer, then compute at that many frequencies (default is
        N=512).  Using a number that is fast for FFT computations can result
        in faster computations (see Notes of `freqz`).

        If an array_like, compute the response at the frequencies given (must
        be 1-D). These are in the same units as `fs`.
    whole : bool, optional
        Normally, frequencies are computed from 0 to the Nyquist frequency,
        fs/2 (upper-half of unit-circle). If `whole` is True, compute
        frequencies from 0 to fs.
    fs : float, optional
        The sampling frequency of the digital system. Defaults to 2*pi
        radians/sample (so w is from 0 to pi).

        .. versionadded:: 1.2.0

    Returns
    -------
    w : ndarray
        The frequencies at which `h` was computed, in the same units as `fs`.
        By default, `w` is normalized to the range [0, pi) (radians/sample).
    h : ndarray
        The frequency response, as complex numbers.

    See Also
    --------
    freqz, sosfilt

    Notes
    -----
    .. versionadded:: 0.19.0

    Examples
    --------
    Design a 15th-order bandpass filter in SOS format.

    >>> from scipy import signal
    >>> import numpy as np
    >>> sos = signal.ellip(15, 0.5, 60, (0.2, 0.4), btype='bandpass',
    ...                    output='sos')

    Compute the frequency response at 1500 points from DC to Nyquist.

    >>> w, h = signal.sosfreqz(sos, worN=1500)

    Plot the response.

    >>> import matplotlib.pyplot as plt
    >>> plt.subplot(2, 1, 1)
    >>> db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
    >>> plt.plot(w/np.pi, db)
    # 设置 y 轴的坐标范围为 -75 到 5
    plt.ylim(-75, 5)
    # 显示网格线
    plt.grid(True)
    # 设置 y 轴刻度为 [0, -20, -40, -60]
    plt.yticks([0, -20, -40, -60])
    # 设置 y 轴标签为 'Gain [dB]'
    plt.ylabel('Gain [dB]')
    # 设置图表标题为 'Frequency Response'
    plt.title('Frequency Response')
    # 创建第二个子图，2 行 1 列中的第 2 个
    plt.subplot(2, 1, 2)
    # 绘制相位响应图，w/π 是 x 轴数据，np.angle(h) 是 y 轴数据
    plt.plot(w/np.pi, np.angle(h))
    # 显示网格线
    plt.grid(True)
    # 设置 y 轴刻度为 [-π, -π/2, 0, π/2, π]
    plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],
               [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    # 设置 y 轴标签为 'Phase [rad]'
    plt.ylabel('Phase [rad]')
    # 设置 x 轴标签为 'Normalized frequency (1.0 = Nyquist)'
    plt.xlabel('Normalized frequency (1.0 = Nyquist)')
    # 显示图形
    plt.show()
    
    If the same filter is implemented as a single transfer function,
    numerical error corrupts the frequency response:
    
    # 使用信号处理中的 ellip 函数生成带通滤波器的传递函数系数 b 和 a
    b, a = signal.ellip(15, 0.5, 60, (0.2, 0.4), btype='bandpass',
                       output='ba')
    # 计算频率响应 w 和 h
    w, h = signal.freqz(b, a, worN=1500)
    # 创建第一个子图，2 行 1 列中的第 1 个
    plt.subplot(2, 1, 1)
    # 计算增益的分贝值
    db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
    # 绘制频率响应图，w/π 是 x 轴数据，db 是 y 轴数据
    plt.plot(w/np.pi, db)
    # 设置 y 轴的坐标范围为 -75 到 5
    plt.ylim(-75, 5)
    # 显示网格线
    plt.grid(True)
    # 设置 y 轴刻度为 [0, -20, -40, -60]
    plt.yticks([0, -20, -40, -60])
    # 设置 y 轴标签为 'Gain [dB]'
    plt.ylabel('Gain [dB]')
    # 设置图表标题为 'Frequency Response'
    plt.title('Frequency Response')
    # 创建第二个子图，2 行 1 列中的第 2 个
    plt.subplot(2, 1, 2)
    # 绘制相位响应图，w/π 是 x 轴数据，np.angle(h) 是 y 轴数据
    plt.plot(w/np.pi, np.angle(h))
    # 显示网格线
    plt.grid(True)
    # 设置 y 轴刻度为 [-π, -π/2, 0, π/2, π]
    plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],
               [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    # 设置 y 轴标签为 'Phase [rad]'
    plt.ylabel('Phase [rad]')
    # 设置 x 轴标签为 'Normalized frequency (1.0 = Nyquist)'
    plt.xlabel('Normalized frequency (1.0 = Nyquist)')
    # 显示图形
    plt.show()
    
    """
    # 验证并确保采样率 fs 是有效的，如果为 None，则引发异常
    fs = _validate_fs(fs, allow_none=False)
    
    # 验证并确保二阶系统描述矩阵 sos 是有效的，返回 sos 和其包含的段数 n_sections
    sos, n_sections = _validate_sos(sos)
    # 如果没有段（n_sections == 0），则无法计算频率
    if n_sections == 0:
        raise ValueError('Cannot compute frequencies with no sections')
    # 初始化频率响应 h 为 1.0
    h = 1.
    # 遍历二阶系统描述矩阵 sos 的每一行
    for row in sos:
        # 计算当前行的频率响应 w 和 rowh
        w, rowh = freqz(row[:3], row[3:], worN=worN, whole=whole, fs=fs)
        # 将当前行的频率响应与总的频率响应 h 相乘
        h *= rowh
    # 返回频率响应的频率数组 w 和总的频率响应 h
    return w, h
def _cplxreal(z, tol=None):
    """
    Split into complex and real parts, combining conjugate pairs.

    The 1-D input vector `z` is split up into its complex (`zc`) and real (`zr`)
    elements. Every complex element must be part of a complex-conjugate pair,
    which are combined into a single number (with positive imaginary part) in
    the output. Two complex numbers are considered a conjugate pair if their
    real and imaginary parts differ in magnitude by less than ``tol * abs(z)``.

    Parameters
    ----------
    z : array_like
        Vector of complex numbers to be sorted and split
    tol : float, optional
        Relative tolerance for testing realness and conjugate equality.
        Default is ``100 * spacing(1)`` of `z`'s data type (i.e., 2e-14 for
        float64)

    Returns
    -------
    zc : ndarray
        Complex elements of `z`, with each pair represented by a single value
        having positive imaginary part, sorted first by real part, and then
        by magnitude of imaginary part. The pairs are averaged when combined
        to reduce error.
    zr : ndarray
        Real elements of `z` (those having imaginary part less than
        `tol` times their magnitude), sorted by value.

    Raises
    ------
    ValueError
        If there are any complex numbers in `z` for which a conjugate
        cannot be found.

    See Also
    --------
    _cplxpair

    Examples
    --------
    >>> from scipy.signal._filter_design import _cplxreal
    >>> a = [4, 3, 1, 2-2j, 2+2j, 2-1j, 2+1j, 2-1j, 2+1j, 1+1j, 1-1j]
    >>> zc, zr = _cplxreal(a)
    >>> print(zc)
    [ 1.+1.j  2.+1.j  2.+1.j  2.+2.j]
    >>> print(zr)
    [ 1.  3.  4.]
    """

    # Ensure z is at least 1-dimensional
    z = atleast_1d(z)
    if z.size == 0:
        return z, z
    elif z.ndim != 1:
        raise ValueError('_cplxreal only accepts 1-D input')

    if tol is None:
        # Set default tolerance based on the precision of z's data type
        tol = 100 * np.finfo((1.0 * z).dtype).eps

    # Sort z by increasing magnitude of imaginary part, then by real part
    z = z[np.lexsort((abs(z.imag), z.real))]

    # Identify real elements based on tolerance condition
    real_indices = abs(z.imag) <= tol * abs(z)
    zr = z[real_indices].real

    if len(zr) == len(z):
        # If all elements are real, return empty complex array and zr
        return array([]), zr

    # Separate positive and negative halves of complex conjugates
    z = z[~real_indices]
    zp = z[z.imag > 0]
    zn = z[z.imag < 0]

    if len(zp) != len(zn):
        # Raise an error if there are unmatched complex conjugates
        raise ValueError('Array contains complex value with no matching '
                         'conjugate.')

    # Find contiguous runs of nearly identical real parts
    same_real = np.diff(zp.real) <= tol * abs(zp[:-1])
    diffs = np.diff(concatenate(([0], same_real, [0])))
    run_starts = np.nonzero(diffs > 0)[0]
    run_stops = np.nonzero(diffs < 0)[0]

    # Sort each run by their imaginary parts
    # 遍历所有的运行起始索引
    for i in range(len(run_starts)):
        # 获取当前运行的起始位置和停止位置
        start = run_starts[i]
        stop = run_stops[i] + 1
        # 对于每个区块(chunk)，包括zp和zn的子数组
        for chunk in (zp[start:stop], zn[start:stop]):
            # 对chunk按照其虚部的绝对值进行排序
            chunk[...] = chunk[np.lexsort([abs(chunk.imag)])]

    # 检查负数和其共轭是否匹配
    if any(abs(zp - zn.conj()) > tol * abs(zn)):
        # 如果存在复值，并且其共轭值没有匹配，则抛出异常
        raise ValueError('Array contains complex value with no matching '
                         'conjugate.')

    # 平均化实部和虚部的数值不精确性
    zc = (zp + zn.conj()) / 2

    # 返回平均化后的复数数组和实数数组
    return zc, zr
def _cplxpair(z, tol=None):
    """
    Sort into pairs of complex conjugates.

    Complex conjugates in `z` are sorted by increasing real part. In each
    pair, the number with negative imaginary part appears first.

    If pairs have identical real parts, they are sorted by increasing
    imaginary magnitude.

    Two complex numbers are considered a conjugate pair if their real and
    imaginary parts differ in magnitude by less than ``tol * abs(z)``.  The
    pairs are forced to be exact complex conjugates by averaging the positive
    and negative values.

    Purely real numbers are also sorted, but placed after the complex
    conjugate pairs. A number is considered real if its imaginary part is
    smaller than `tol` times the magnitude of the number.

    Parameters
    ----------
    z : array_like
        1-D input array to be sorted.
    tol : float, optional
        Relative tolerance for testing realness and conjugate equality.
        Default is ``100 * spacing(1)`` of `z`'s data type (i.e., 2e-14 for
        float64)

    Returns
    -------
    y : ndarray
        Complex conjugate pairs followed by real numbers.

    Raises
    ------
    ValueError
        If there are any complex numbers in `z` for which a conjugate
        cannot be found.

    See Also
    --------
    _cplxreal

    Examples
    --------
    >>> from scipy.signal._filter_design import _cplxpair
    >>> a = [4, 3, 1, 2-2j, 2+2j, 2-1j, 2+1j, 2-1j, 2+1j, 1+1j, 1-1j]
    >>> z = _cplxpair(a)
    >>> print(z)
    [ 1.-1.j  1.+1.j  2.-1.j  2.+1.j  2.-1.j  2.+1.j  2.-2.j  2.+2.j  1.+0.j
      3.+0.j  4.+0.j]
    """

    # Convert input `z` to at least 1-D array
    z = np.atleast_1d(z)
    
    # If `z` is empty or consists only of real numbers, sort and return
    if z.size == 0 or np.isrealobj(z):
        return np.sort(z)

    # Raise an error if `z` is not 1-D
    if z.ndim != 1:
        raise ValueError('z must be 1-D')

    # Separate complex and real parts of `z`
    zc, zr = _cplxreal(z, tol)

    # Interleave complex values and their conjugates, with negative imaginary
    # parts first in each pair
    zc = np.dstack((zc.conj(), zc)).flatten()
    z = np.append(zc, zr)
    return z
    """
    将传递函数的系数标准化为b和a，并确保a的第一个系数不为零
    """
    b, a = normalize(b, a)
    """
    将b中的系数除以a中的第一个系数，确保第一个系数为1.0
    """
    b = (b + 0.0) / a[0]
    """
    将a中的系数除以a中的第一个系数，确保第一个系数为1.0
    """
    a = (a + 0.0) / a[0]
    """
    提取k值，即b中的第一个系数
    """
    k = b[0]
    """
    将b中的系数除以第一个系数，确保第一个系数为1.0
    """
    b /= b[0]
    """
    计算传递函数的零点
    """
    z = roots(b)
    """
    计算传递函数的极点
    """
    p = roots(a)
    """
    返回传递函数的零点(z)、极点(p)和增益(k)
    """
    return z, p, k
    # 定义函数 tf2sos，将传递函数表示转换为二阶节（Second-Order Sections）

    # 参数说明：
    # b : array_like
    #     分子多项式的系数。
    # a : array_like
    #     分母多项式的系数。
    # pairing : {None, 'nearest', 'keep_odd', 'minimal'}, optional
    #     用于将极点和零点成对组合成节的方法。
    #     详细信息和对 'pairing' 和 'analog' 参数的限制，请参见 'zpk2sos'。
    # analog : bool, optional
    #     如果为 True，则系统是模拟系统，否则为离散系统。

    # 返回：
    # 返回二阶节（Second-Order Sections）

    r"""
    Return second-order sections from transfer function representation

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.
    pairing : {None, 'nearest', 'keep_odd', 'minimal'}, optional
        The method to use to combine pairs of poles and zeros into sections.
        See `zpk2sos` for information and restrictions on `pairing` and
        `analog` arguments.
    analog : bool, optional
        If True, system is analog, otherwise discrete.

        .. versionadded:: 1.8.0

    Returns
    -------
    ```
    # sos: ndarray
    # 二阶滤波器系数数组，形状为 ``(n_sections, 6)``。参见 `sosfilt` 获取 SOS 滤波器格式规范。
    # 
    # See Also
    # --------
    # zpk2sos, sosfilt

    # Notes
    # -----
    # 通常不建议从传递函数 (TF) 转换到 SOS 格式，因为这样做通常不会改善数值精度误差。
    # 相反，考虑在 ZPK 格式中设计滤波器，然后直接转换到 SOS。TF 首先转换为 ZPK 格式，然后再将
    # ZPK 转换为 SOS。
    # 
    # .. versionadded:: 0.16.0

    # Examples
    # --------
    # 使用传递函数 H(s) 的多项式表示找到其二阶段 (sos) 的转移函数 'sos'。
    # 
    # .. math::
    # 
    #    H(s) = \frac{s^2 - 3.5s - 2}{s^4 + 3s^3 - 15s^2 - 19s + 30}
    #    
    # >>> from scipy.signal import tf2sos
    # >>> tf2sos([1, -3.5, -2], [1, 3, -15, -19, 30], analog=True)
    # array([[  0. ,   0. ,   1. ,   1. ,   2. , -15. ],
    #        [  1. ,  -3.5,  -2. ,   1. ,   1. ,  -2. ]])
    """
    # 返回 TF 转换为 SOS 后的结果，通过先转换为 ZPK 格式，再将 ZPK 转换为 SOS 格式来完成。
    return zpk2sos(*tf2zpk(b, a), pairing=pairing, analog=analog)
# 返回一组二阶节的单个传递函数

def sos2tf(sos):
    # 将输入的二阶滤波器系数转换为 NumPy 数组
    sos = np.asarray(sos)
    # 确定结果类型，如果类型为布尔、整数或无符号整数，则改为 np.float64
    result_type = sos.dtype
    if result_type.kind in 'bui':
        result_type = np.float64

    # 初始化分子多项式系数 b 和分母多项式系数 a
    b = np.array([1], dtype=result_type)
    a = np.array([1], dtype=result_type)
    
    # 获取二阶节的数量
    n_sections = sos.shape[0]
    # 遍历每个二阶节
    for section in range(n_sections):
        # 计算当前二阶节的分子多项式并与 b 相乘
        b = np.polymul(b, sos[section, :3])
        # 计算当前二阶节的分母多项式并与 a 相乘
        a = np.polymul(a, sos[section, 3:])
    
    # 返回计算得到的分子和分母多项式系数
    return b, a


# 返回一组二阶节的零点、极点和增益

def sos2zpk(sos):
    # 将输入的二阶滤波器系数转换为 NumPy 数组
    sos = np.asarray(sos)
    # 获取二阶节的数量
    n_sections = sos.shape[0]
    # 初始化存储零点和极点的数组，以及系统增益
    z = np.zeros(n_sections*2, np.complex128)
    p = np.zeros(n_sections*2, np.complex128)
    k = 1.

    # 遍历每个二阶节
    for section in range(n_sections):
        # 调用 tf2zpk 函数计算当前二阶节的零点、极点和增益
        zpk = tf2zpk(sos[section, :3], sos[section, 3:])
        # 将计算得到的零点放入 z 数组
        z[2*section:2*section+len(zpk[0])] = zpk[0]
        # 将计算得到的极点放入 p 数组
        p[2*section:2*section+len(zpk[1])] = zpk[1]
        # 累积计算得到的增益
        k *= zpk[2]

    # 返回零点、极点和增益
    return z, p, k


# 根据距离获取最接近的实数或复数元素的索引

def _nearest_real_complex_idx(fro, to, which):
    assert which in ('real', 'complex', 'any')
    # 根据距离对 fro 与 to 之间的元素进行排序
    order = np.argsort(np.abs(fro - to))
    if which == 'any':
        # 返回距离最近的任意元素的索引
        return order[0]
    else:
        # 根据指定的类型选择元素
        mask = np.isreal(fro[order])
        if which == 'complex':
            mask = ~mask
        # 返回最接近的实数或复数元素的索引
        return order[np.nonzero(mask)[0][0]]


# 从最多两个零点和极点创建一个二阶节

def _single_zpksos(z, p, k):
    # 初始化二阶节的系数数组
    sos = np.zeros(6)
    # 调用 zpk2tf 函数计算当前零点、极点和增益的传递函数系数
    b, a = zpk2tf(z, p, k)
    # 将计算得到的分子多项式系数存入 sos 数组
    sos[3-len(b):3] = b
    # 将计算得到的分母多项式系数存入 sos 数组
    sos[6-len(a):6] = a
    # 返回创建的二阶节
    return sos
# 根据输入的零点、极点和增益，生成二阶段系统的系数
def zpk2sos(z, p, k, pairing=None, *, analog=False):
    """Return second-order sections from zeros, poles, and gain of a system

    Parameters
    ----------
    z : array_like
        Zeros of the transfer function.
    p : array_like
        Poles of the transfer function.
    k : float
        System gain.
    pairing : {None, 'nearest', 'keep_odd', 'minimal'}, optional
        The method to use to combine pairs of poles and zeros into sections.
        If analog is False and pairing is None, pairing is set to 'nearest';
        if analog is True, pairing must be 'minimal', and is set to that if
        it is None.
    analog : bool, optional
        If True, system is analog, otherwise discrete.

        .. versionadded:: 1.8.0

    Returns
    -------
    sos : ndarray
        Array of second-order filter coefficients, with shape
        ``(n_sections, 6)``. See `sosfilt` for the SOS filter format
        specification.

    See Also
    --------
    sosfilt

    Notes
    -----
    The algorithm used to convert ZPK to SOS format is designed to
    minimize errors due to numerical precision issues. The pairing
    algorithm attempts to minimize the peak gain of each biquadratic
    section. This is done by pairing poles with the nearest zeros, starting
    with the poles closest to the unit circle for discrete-time systems, and
    poles closest to the imaginary axis for continuous-time systems.

    ``pairing='minimal'`` outputs may not be suitable for `sosfilt`,
    and ``analog=True`` outputs will never be suitable for `sosfilt`.

    *Algorithms*

    The steps in the ``pairing='nearest'``, ``pairing='keep_odd'``,
    and ``pairing='minimal'`` algorithms are mostly shared. The
    ``'nearest'`` algorithm attempts to minimize the peak gain, while
    ``'keep_odd'`` minimizes peak gain under the constraint that
    odd-order systems should retain one section as first order.
    ``'minimal'`` is similar to ``'keep_odd'``, but no additional
    poles or zeros are introduced

    The algorithm steps are as follows:

    As a pre-processing step for ``pairing='nearest'``,
    ``pairing='keep_odd'``, add poles or zeros to the origin as
    necessary to obtain the same number of poles and zeros for
    pairing.  If ``pairing == 'nearest'`` and there are an odd number
    of poles, add an additional pole and a zero at the origin.

    The following steps are then iterated over until no more poles or
    zeros remain:

    1. Take the (next remaining) pole (complex or real) closest to the
       unit circle (or imaginary axis, for ``analog=True``) to
       begin a new filter section.

    2. If the pole is real and there are no other remaining real poles [#]_,
       add the closest real zero to the section and leave it as a first
       order section. Note that after this step we are guaranteed to be
       left with an even number of real poles, complex poles, real zeros,
       and complex zeros for subsequent pairing iterations.
    3. Else:
        # 如果极点是复数且零点是唯一剩余的实数零点，则将极点与下一个最接近的复数零点配对。
        # 这是必要的，以确保最终会保留一个实零点，以便最终创建一个一阶节（从而保持奇序）。
        1. If the pole is complex and the zero is the only remaining real
           zero*, then pair the pole with the *next* closest zero
           (guaranteed to be complex). This is necessary to ensure that
           there will be a real zero remaining to eventually create a
           first-order section (thus keeping the odd order).

        # 否则将极点与最接近的剩余零点（无论是复数还是实数）配对。
        2. Else pair the pole with the closest remaining zero (complex or
           real).

        # 继续完成第二阶段节，通过向当前极点和零点添加另一个极点和零点：
        3. Proceed to complete the second-order section by adding another
           pole and zero to the current pole and zero in the section:

            # 如果当前极点和零点都是复数，则添加它们的共轭。
            1. If the current pole and zero are both complex, add their
               conjugates.

            # 否则，如果极点是复数而零点是实数，则添加共轭极点和最接近的实数零点。
            2. Else if the pole is complex and the zero is real, add the
               conjugate pole and the next closest real zero.

            # 否则，如果极点是实数而零点是复数，则添加共轭零点和最接近这些零点的实数极点。
            3. Else if the pole is real and the zero is complex, add the
               conjugate zero and the real pole closest to those zeros.

            # 否则（必然有实数极点和实数零点），添加最接近单位圆的下一个实数极点，然后添加最接近该极点的实数零点。
            4. Else (we must have a real pole and real zero) add the next
               real pole closest to the unit circle, and then add the real
               zero closest to that pole.
    # 在默认情况下使用最近匹配策略进行转换，将零点极点转换为二阶段的二阶节
    if pairing is None:
        pairing = 'minimal' if analog else 'nearest'
    
    # 定义有效的匹配策略列表，用于检查用户输入的匹配策略是否有效
    valid_pairings = ['nearest', 'keep_odd', 'minimal']
    if pairing not in valid_pairings:
        raise ValueError(f'pairing must be one of {valid_pairings}, not {pairing}')
    
    # 如果是模拟系统且匹配策略不是'minimal'，则抛出异常，因为模拟系统下只支持'minimal'策略
    if analog and pairing != 'minimal':
        raise ValueError('for analog zpk2sos conversion, '
                         'pairing must be "minimal"')
    
    # 如果输入的零点和极点均为空，则根据是否为模拟系统返回相应的默认单一一阶段的SOS
    if len(z) == len(p) == 0:
        if not analog:
            return np.array([[k, 0., 0., 1., 0., 0.]])
        else:
            return np.array([[0., 0., k, 0., 0., 1.]])
    
    # 如果匹配策略不是'minimal'，则确保零点和极点数量相同，并进行必要的复制以处理不同的匹配策略
    if pairing != 'minimal':
        # 将极点和零点数组补齐到相同长度，并计算出所需的节（section）数量
        p = np.concatenate((p, np.zeros(max(len(z) - len(p), 0))))
        z = np.concatenate((z, np.zeros(max(len(p) - len(z), 0))))
        n_sections = (max(len(p), len(z)) + 1) // 2
    
        # 如果极点数量为奇数且匹配策略为'nearest'，则添加一个额外的零点和极点以保证配对成对性
        if len(p) % 2 == 1 and pairing == 'nearest':
            p = np.concatenate((p, [0.]))
            z = np.concatenate((z, [0.]))
        assert len(p) == len(z)
    else:
        # 如果匹配策略是'minimal'，则要求极点数量不少于零点数量，用于模拟系统的转换
        if len(p) < len(z):
            raise ValueError('for analog zpk2sos conversion, '
                             'must have len(p)>=len(z)')
    
        # 计算出所需的节（section）数量
        n_sections = (len(p) + 1) // 2
    
    # 确保零点和极点成对存在，这里使用 _cplxreal 函数对零点和极点进行处理
    # 以便确保每个复数共轭对的成员都存在于数组中
    z = np.concatenate(_cplxreal(z))
    p = np.concatenate(_cplxreal(p))
    # 如果 k 不是实数，则引发值错误异常
    if not np.isreal(k):
        raise ValueError('k must be real')
    # 将 k 转换为其实部，确保 k 是实数
    k = k.real

    # 如果不是模拟滤波器（analog=False）
    if not analog:
        # digital: "worst" 是距离单位圆最近的点的索引
        def idx_worst(p):
            return np.argmin(np.abs(1 - np.abs(p)))
    else:
        # analog: "worst" 是距离虚轴最近的点的索引
        def idx_worst(p):
            return np.argmin(np.abs(np.real(p)))

    # 创建一个形状为 (n_sections, 6) 的全零数组，用于存储系统参数
    sos = np.zeros((n_sections, 6))

    # 构建系统，以便将 "worst" 系数放在最后
    for si in range(n_sections-1, -1, -1):
        # 从最后一个部分开始向前遍历
        p1_idx = idx_worst(p)
        # 找到最差的极点的索引
        p1 = p[p1_idx]
        # 获取最差的极点
        p = np.delete(p, p1_idx)
        # 从数组 p 中删除这个极点

        # 将该极点与一个零点配对

        if np.isreal(p1) and np.isreal(p).sum() == 0:
            # 特殊情况（1）：只剩下一个实数极点
            if pairing != 'minimal':
                z1_idx = _nearest_real_complex_idx(z, p1, 'real')
                # 找到与 p1 最接近的实数零点的索引
                z1 = z[z1_idx]
                # 获取找到的实数零点
                z = np.delete(z, z1_idx)
                # 从数组 z 中删除这个零点
                sos[si] = _single_zpksos([z1, 0], [p1, 0], 1)
                # 生成一个二阶节，使用 z1 和 0 作为零点，p1 和 0 作为极点
            elif len(z) > 0:
                z1_idx = _nearest_real_complex_idx(z, p1, 'real')
                z1 = z[z1_idx]
                z = np.delete(z, z1_idx)
                sos[si] = _single_zpksos([z1], [p1], 1)
                # 生成一个一阶节，使用 z1 作为零点，p1 作为极点
            else:
                sos[si] = _single_zpksos([], [p1], 1)
                # 生成一个零阶节，只有 p1 作为极点

        elif (len(p) + 1 == len(z)
              and not np.isreal(p1)
              and np.isreal(p).sum() == 1
              and np.isreal(z).sum() == 1):

            # 特殊情况（2）：只剩下一个实数极点和一个实数零点，
            # 且极点和零点数目相等，必须与一个复数零点配对

            z1_idx = _nearest_real_complex_idx(z, p1, 'complex')
            # 找到与 p1 最接近的复数零点的索引
            z1 = z[z1_idx]
            # 获取找到的复数零点
            z = np.delete(z, z1_idx)
            # 从数组 z 中删除这个零点
            sos[si] = _single_zpksos([z1, z1.conj()], [p1, p1.conj()], 1)
            # 生成一个二阶节，使用 z1 和其共轭复数作为零点，p1 和其共轭复数作为极点

        else:
            if np.isreal(p1):
                prealidx = np.flatnonzero(np.isreal(p))
                # 获取所有实数极点的索引
                p2_idx = prealidx[idx_worst(p[prealidx])]
                # 找到最差的实数极点的索引
                p2 = p[p2_idx]
                # 获取最差的实数极点
                p = np.delete(p, p2_idx)
                # 从数组 p 中删除这个极点
            else:
                p2 = p1.conj()
                # 如果 p1 是复数极点，则 p2 是其共轭

            # 找到最接近的零点
            if len(z) > 0:
                z1_idx = _nearest_real_complex_idx(z, p1, 'any')
                # 找到与 p1 最接近的任意类型的零点的索引
                z1 = z[z1_idx]
                # 获取找到的零点
                z = np.delete(z, z1_idx)
                # 从数组 z 中删除这个零点

                if not np.isreal(z1):
                    sos[si] = _single_zpksos([z1, z1.conj()], [p1, p2], 1)
                    # 生成一个二阶节，使用 z1 和其共轭复数作为零点，p1 和 p2 作为极点
                else:
                    if len(z) > 0:
                        z2_idx = _nearest_real_complex_idx(z, p1, 'real')
                        # 找到与 p1 最接近的实数零点的索引
                        z2 = z[z2_idx]
                        assert np.isreal(z2)
                        # 断言 z2 是实数
                        z = np.delete(z, z2_idx)
                        # 从数组 z 中删除这个零点
                        sos[si] = _single_zpksos([z1, z2], [p1, p2], 1)
                        # 生成一个二阶节，使用 z1 和 z2 作为零点，p1 和 p2 作为极点
                    else:
                        sos[si] = _single_zpksos([z1], [p1, p2], 1)
                        # 生成一个一阶节，使用 z1 作为零点，p1 和 p2 作为极点
            else:
                # 没有更多的零点
                sos[si] = _single_zpksos([], [p1, p2], 1)
                # 生成一个零阶节，只有 p1 和 p2 作为极点

    assert len(p) == len(z) == 0  # 我们已经用完了所有的极点和零点
    del p, z

    # 将增益放在第一个节中
    sos[0][:3] *= k
    # 返回 sOS
    return sos
def _align_nums(nums):
    """Aligns the shapes of multiple numerators.

    Given an array of numerator coefficient arrays [[a_1, a_2,...,
    a_n],..., [b_1, b_2,..., b_m]], this function pads shorter numerator
    arrays with zero's so that all numerators have the same length. Such
    alignment is necessary for functions like 'tf2ss', which needs the
    alignment when dealing with SIMO transfer functions.

    Parameters
    ----------
    nums: array_like
        Numerator or list of numerators. Not necessarily with same length.

    Returns
    -------
    nums: array
        The numerator. If `nums` input was a list of numerators then a 2-D
        array with padded zeros for shorter numerators is returned. Otherwise
        returns ``np.asarray(nums)``.
    """
    try:
        # 尝试将输入转换为 numpy 数组
        nums = asarray(nums)

        # 如果转换后的数组不是数字类型，则抛出 ValueError
        if not np.issubdtype(nums.dtype, np.number):
            raise ValueError("dtype of numerator is non-numeric")

        # 返回转换后的数组
        return nums

    except ValueError:
        # 如果转换失败，说明输入是一个包含不同类型的数值或数组的列表
        # 将每个元素至少转换为 1 维数组
        nums = [np.atleast_1d(num) for num in nums]
        
        # 计算列表中最长数组的长度
        max_width = max(num.size for num in nums)

        # 预先分配一个全零数组，用于存放填充了零的数值
        aligned_nums = np.zeros((len(nums), max_width))

        # 遍历并填充 aligned_nums 数组
        for index, num in enumerate(nums):
            aligned_nums[index, -num.size:] = num

        # 返回填充了零的数组
        return aligned_nums


def normalize(b, a):
    """Normalize numerator/denominator of a continuous-time transfer function.

    If values of `b` are too close to 0, they are removed. In that case, a
    BadCoefficients warning is emitted.

    Parameters
    ----------
    b: array_like
        Numerator of the transfer function. Can be a 2-D array to normalize
        multiple transfer functions.
    a: array_like
        Denominator of the transfer function. At most 1-D.

    Returns
    -------
    num: array
        The numerator of the normalized transfer function. At least a 1-D
        array. A 2-D array if the input `num` is a 2-D array.
    den: 1-D array
        The denominator of the normalized transfer function.

    Notes
    -----
    Coefficients for both the numerator and denominator should be specified in
    descending exponent order (e.g., ``s^2 + 3s + 5`` would be represented as
    ``[1, 3, 5]``).

    Examples
    --------
    >>> from scipy.signal import normalize

    Normalize the coefficients of the transfer function
    ``(3*s^2 - 2*s + 5) / (2*s^2 + 3*s + 1)``:

    >>> b = [3, -2, 5]
    >>> a = [2, 3, 1]
    >>> normalize(b, a)
    (array([ 1.5, -1. ,  2.5]), array([1. , 1.5, 0.5]))

    A warning is generated if, for example, the first coefficient of
    `b` is 0.  In the following example, the result is as expected:

    >>> import warnings
    >>> with warnings.catch_warnings(record=True) as w:
    ...     num, den = normalize([0, 3, 6], [2, -5, 4])

    >>> num
    """
    array([1.5, 3. ])
    >>> den
    array([ 1. , -2.5,  2. ])

    >>> print(w[0].message)
    Badly conditioned filter coefficients (numerator): the results may be meaningless

    """
    num, den = b, a  # 将参数 b 赋给 num，参数 a 赋给 den

    den = np.atleast_1d(den)  # 将 den 转换为至少一维数组
    num = np.atleast_2d(_align_nums(num))  # 将 num 对齐并转换为至少二维数组

    if den.ndim != 1:  # 如果 den 不是一维数组
        raise ValueError("Denominator polynomial must be rank-1 array.")  # 抛出数值错误异常
    if num.ndim > 2:  # 如果 num 的维度大于二
        raise ValueError("Numerator polynomial must be rank-1 or"
                         " rank-2 array.")  # 抛出数值错误异常
    if np.all(den == 0):  # 如果 den 中所有元素都为零
        raise ValueError("Denominator must have at least one nonzero element.")  # 抛出数值错误异常

    # Trim leading zeros in denominator, leave at least one.
    den = np.trim_zeros(den, 'f')  # 去除分母中前导零，至少保留一个非零元素

    # Normalize transfer function
    num, den = num / den[0], den / den[0]  # 标准化传递函数的分子和分母

    # Count numerator columns that are all zero
    leading_zeros = 0  # 计算分子中全为零的列数
    for col in num.T:  # 遍历 num 的转置列
        if np.allclose(col, 0, atol=1e-14):  # 如果列中所有元素接近于零（精度为1e-14）
            leading_zeros += 1  # 计数增加
        else:
            break

    # Trim leading zeros of numerator
    if leading_zeros > 0:  # 如果分子中有前导零
        warnings.warn("Badly conditioned filter coefficients (numerator): the "
                      "results may be meaningless",
                      BadCoefficients, stacklevel=2)  # 发出警告，表明分子的系数可能条件不佳
        # Make sure at least one column remains
        if leading_zeros == num.shape[1]:  # 如果所有列都是零
            leading_zeros -= 1  # 至少保留一列
        num = num[:, leading_zeros:]  # 去除分子中的前导零

    # Squeeze first dimension if singular
    if num.shape[0] == 1:  # 如果分子的第一维是单一的
        num = num[0, :]  # 去除这一维

    return num, den  # 返回处理后的分子和分母
# 将低通滤波器原型转换为不同频率的低通滤波器
# 返回一个模拟低通滤波器，其截止频率为 `wo`，从具有单位截止频率的模拟低通滤波器原型中，
# 返回传输函数 ('ba') 表示形式的滤波器系数。

# 参数：
# b : array_like
#     分子多项式系数。
# a : array_like
#     分母多项式系数。
# wo : float
#     所需的截止频率，作为角频率（例如，rad/s）。
#     默认为无变化。

# 返回：
# b : array_like
#     转换后的低通滤波器的分子多项式系数。
# a : array_like
#     转换后的低通滤波器的分母多项式系数。

# 参见：
# lp2hp, lp2bp, lp2bs, bilinear
# lp2lp_zpk

# 注意：
# 这是从 s-平面替换导出的

# 数学公式：
# s -> s / ω0

# 示例：
# >>> from scipy import signal
# >>> import matplotlib.pyplot as plt

def lp2lp(b, a, wo=1.0):
    a, b = map(atleast_1d, (a, b))
    try:
        wo = float(wo)
    except TypeError:
        wo = float(wo[0])
    d = len(a)
    n = len(b)
    M = max((d, n))
    pwo = pow(wo, np.arange(M - 1, -1, -1))
    start1 = max((n - d, 0))
    start2 = max((d - n, 0))
    b = b * pwo[start1] / pwo[start2:]
    a = a * pwo[start1] / pwo[start1:]
    return normalize(b, a)


# 将低通滤波器原型转换为高通滤波器
# 返回一个模拟高通滤波器，其截止频率为 `wo`，从具有单位截止频率的模拟低通滤波器原型中，
# 返回传输函数 ('ba') 表示形式的滤波器系数。

# 参数：
# b : array_like
#     分子多项式系数。
# a : array_like
#     分母多项式系数。
# wo : float
#     所需的截止频率，作为角频率（例如，rad/s）。
#     默认为无变化。

# 返回：
# b : array_like
#     转换后的高通滤波器的分子多项式系数。
# a : array_like
#     转换后的高通滤波器的分母多项式系数。

# 参见：
# lp2lp, lp2bp, lp2bs, bilinear
# lp2hp_zpk

# 注意：
# 这是从 s-平面替换导出的

# 数学公式：
# s -> ω0 / s

# 这保持了低通和高通响应在对数尺度上的对称性。

# 示例：
# >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
导入 matplotlib 库中的 pyplot 模块，用于绘制图形

    >>> lp = signal.lti([1.0], [1.0, 1.0])
创建一个低通系统，传入参数为分子 [1.0] 和分母 [1.0, 1.0]

    >>> hp = signal.lti(*signal.lp2hp(lp.num, lp.den))
将低通系统转换为高通系统，使用 lp2hp 函数，传入参数为低通系统的分子和分母

    >>> w, mag_lp, p_lp = lp.bode()
计算低通系统的频率响应，返回频率 w、幅度响应 mag_lp 和相位响应 p_lp

    >>> w, mag_hp, p_hp = hp.bode(w)
使用高通系统计算相同频率下的频率响应，返回频率 w、幅度响应 mag_hp 和相位响应 p_hp

    >>> plt.plot(w, mag_lp, label='Lowpass')
在图中绘制低通系统的幅度响应曲线，设置标签为 'Lowpass'

    >>> plt.plot(w, mag_hp, label='Highpass')
在图中绘制高通系统的幅度响应曲线，设置标签为 'Highpass'

    >>> plt.semilogx()
将 x 轴设置为对数刻度

    >>> plt.grid(True)
显示图中的网格线

    >>> plt.xlabel('Frequency [rad/s]')
设置 x 轴标签为 'Frequency [rad/s]'

    >>> plt.ylabel('Magnitude [dB]')
设置 y 轴标签为 'Magnitude [dB]'

    >>> plt.legend()
显示图例

"""
a, b = map(atleast_1d, (a, b))
将 a 和 b 转换为至少是一维数组的形式

try:
    wo = float(wo)
except TypeError:
    wo = float(wo[0])
尝试将 wo 转换为浮点数，如果出现 TypeError，则取 wo 的第一个元素并转换为浮点数

d = len(a)
获取数组 a 的长度赋值给 d

n = len(b)
获取数组 b 的长度赋值给 n

if wo != 1:
    pwo = pow(wo, np.arange(max((d, n))))
else:
    pwo = np.ones(max((d, n)), b.dtype.char)
如果 wo 不等于 1，则计算 wo 的幂，范围是 0 到 max(d, n)-1 的整数，结果赋值给 pwo；否则，生成一个元素全为 1 的数组，长度为 max(d, n)，数据类型与 b 的字符类型相同

if d >= n:
    outa = a[::-1] * pwo
    outb = resize(b, (d,))
    outb[n:] = 0.0
    outb[:n] = b[::-1] * pwo[:n]
如果 d 大于等于 n，则计算 outa 和 outb：
    - outa 是 a 数组的倒序乘以 pwo 得到的结果
    - outb 是将 b 数组重塑为长度为 d 的数组，然后将超出 n 长度的部分置为 0，前 n 部分是 b 数组的倒序乘以 pwo[:n] 得到的结果

else:
    outb = b[::-1] * pwo
    outa = resize(a, (n,))
    outa[d:] = 0.0
    outa[:d] = a[::-1] * pwo[:d]
如果 d 小于 n，则计算 outb 和 outa：
    - outb 是 b 数组的倒序乘以 pwo 得到的结果
    - outa 是将 a 数组重塑为长度为 n 的数组，然后将超出 d 长度的部分置为 0，前 d 部分是 a 数组的倒序乘以 pwo[:d] 得到的结果

return normalize(outb, outa)
返回经过标准化处理后的 outb 和 outa
# 将低通滤波器原型转换为带通滤波器的函数
def lp2bp(b, a, wo=1.0, bw=1.0):
    r"""
    Transform a lowpass filter prototype to a bandpass filter.

    Return an analog band-pass filter with center frequency `wo` and
    bandwidth `bw` from an analog low-pass filter prototype with unity
    cutoff frequency, in transfer function ('ba') representation.

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.
    wo : float
        Desired passband center, as angular frequency (e.g., rad/s).
        Defaults to no change.
    bw : float
        Desired passband width, as angular frequency (e.g., rad/s).
        Defaults to 1.

    Returns
    -------
    b : array_like
        Numerator polynomial coefficients of the transformed band-pass filter.
    a : array_like
        Denominator polynomial coefficients of the transformed band-pass filter.

    See Also
    --------
    lp2lp, lp2hp, lp2bs, bilinear
    lp2bp_zpk

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \rightarrow \frac{s^2 + {\omega_0}^2}{s \cdot \mathrm{BW}}

    This is the "wideband" transformation, producing a passband with
    geometric (log frequency) symmetry about `wo`.

    Examples
    --------
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> lp = signal.lti([1.0], [1.0, 1.0])
    >>> bp = signal.lti(*signal.lp2bp(lp.num, lp.den))
    >>> w, mag_lp, p_lp = lp.bode()
    >>> w, mag_bp, p_bp = bp.bode(w)

    >>> plt.plot(w, mag_lp, label='Lowpass')
    >>> plt.plot(w, mag_bp, label='Bandpass')
    >>> plt.semilogx()
    >>> plt.grid(True)
    >>> plt.xlabel('Frequency [rad/s]')
    >>> plt.ylabel('Magnitude [dB]')
    >>> plt.legend()
    """

    # 确保参数 b 和 a 至少是一维数组
    a, b = map(atleast_1d, (a, b))
    # 获取分子多项式的阶数
    D = len(a) - 1
    # 获取分母多项式的阶数
    N = len(b) - 1
    # 确定数据类型
    artype = mintypecode((a, b))
    # 计算最大的阶数
    ma = max([N, D])
    # 计算扩展后的分子和分母多项式的阶数
    Np = N + ma
    Dp = D + ma
    # 创建空的扩展后的分子和分母多项式数组
    bprime = np.empty(Np + 1, artype)
    aprime = np.empty(Dp + 1, artype)
    # 计算中心频率的平方
    wosq = wo * wo
    # 计算扩展后的分子多项式
    for j in range(Np + 1):
        val = 0.0
        for i in range(0, N + 1):
            for k in range(0, i + 1):
                if ma - i + 2 * k == j:
                    val += comb(i, k) * b[N - i] * (wosq) ** (i - k) / bw ** i
        bprime[Np - j] = val
    # 计算扩展后的分母多项式
    for j in range(Dp + 1):
        val = 0.0
        for i in range(0, D + 1):
            for k in range(0, i + 1):
                if ma - i + 2 * k == j:
                    val += comb(i, k) * a[D - i] * (wosq) ** (i - k) / bw ** i
        aprime[Dp - j] = val

    # 返回归一化后的分子和分母多项式系数
    return normalize(bprime, aprime)


def lp2bs(b, a, wo=1.0, bw=1.0):
    r"""
    Transform a lowpass filter prototype to a bandstop filter.

    Return an analog band-stop filter with center frequency `wo` and
    bandwidth `bw` from an analog low-pass filter prototype with unity
    cutoff frequency, in transfer function ('ba') representation.

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.
    wo : float
        Desired stopband center, as angular frequency (e.g., rad/s).
        Defaults to no change.
    bw : float
        Desired stopband width, as angular frequency (e.g., rad/s).
        Defaults to 1.

    Returns
    -------
    b : array_like
        Numerator polynomial coefficients of the transformed band-stop filter.
    a : array_like
        Denominator polynomial coefficients of the transformed band-stop filter.

    See Also
    --------
    lp2lp, lp2hp, lp2bp, bilinear
    lp2bs_zpk

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \rightarrow \frac{s \cdot \mathrm{BW}}{s^2 + {\omega_0}^2}

    This is the "wideband" transformation, producing a stopband with
    geometric (log frequency) symmetry about `wo`.

    Examples
    --------
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> lp = signal.lti([1.0], [1.0, 1.5])
    >>> bs = signal.lti(*signal.lp2bs(lp.num, lp.den))
    >>> w, mag_lp, p_lp = lp.bode()
    >>> w, mag_bs, p_bs = bs.bode(w)
    >>> plt.plot(w, mag_lp, label='Lowpass')
    >>> plt.plot(w, mag_bs, label='Bandstop')
    >>> plt.semilogx()
    >>> plt.grid(True)
    >>> plt.xlabel('Frequency [rad/s]')
    >>> plt.ylabel('Magnitude [dB]')
    >>> plt.legend()
    """
    # Ensure coefficients `a` and `b` are at least 1-dimensional arrays
    a, b = map(atleast_1d, (a, b))
    # Degree of the denominator polynomial
    D = len(a) - 1
    # Degree of the numerator polynomial
    N = len(b) - 1
    # Determine the minimum typecode for arrays `a` and `b`
    artype = mintypecode((a, b))
    # Determine the maximum degree between numerator and denominator
    M = max([N, D])
    # Length of the resulting numerator polynomial after transformation
    Np = M + M
    # Length of the resulting denominator polynomial after transformation
    Dp = M + M
    # Initialize an empty array for the transformed numerator polynomial
    bprime = np.empty(Np + 1, artype)
    # Initialize an empty array for the transformed denominator polynomial
    aprime = np.empty(Dp + 1, artype)
    # Square of the stopband center angular frequency
    wosq = wo * wo

    # Calculate coefficients for the transformed numerator polynomial
    for j in range(Np + 1):
        val = 0.0
        for i in range(0, N + 1):
            for k in range(0, M - i + 1):
                if i + 2 * k == j:
                    val += (comb(M - i, k) * b[N - i] *
                            (wosq) ** (M - i - k) * bw ** i)
        bprime[Np - j] = val

    # Calculate coefficients for the transformed denominator polynomial
    for j in range(Dp + 1):
        val = 0.0
        for i in range(0, D + 1):
            for k in range(0, M - i + 1):
                if i + 2 * k == j:
                    val += (comb(M - i, k) * a[D - i] *
                            (wosq) ** (M - i - k) * bw ** i)
        aprime[Dp - j] = val

    # Normalize and return the transformed coefficients
    return normalize(bprime, aprime)
# 使用双线性变换从模拟系统转换为数字系统的IIR滤波器
def bilinear(b, a, fs=1.0):
    r"""
    Return a digital IIR filter from an analog one using a bilinear transform.

    Transform a set of poles and zeros from the analog s-plane to the digital
    z-plane using Tustin's method, which substitutes ``2*fs*(z-1) / (z+1)`` for
    ``s``, maintaining the shape of the frequency response.

    Parameters
    ----------
    b : array_like
        Numerator of the analog filter transfer function.
    a : array_like
        Denominator of the analog filter transfer function.
    fs : float
        Sample rate, as ordinary frequency (e.g., hertz). No prewarping is
        done in this function.

    Returns
    -------
    b : ndarray
        Numerator of the transformed digital filter transfer function.
    a : ndarray
        Denominator of the transformed digital filter transfer function.

    See Also
    --------
    lp2lp, lp2hp, lp2bp, lp2bs
    bilinear_zpk

    Examples
    --------
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> fs = 100
    >>> bf = 2 * np.pi * np.array([7, 13])
    >>> filts = signal.lti(*signal.butter(4, bf, btype='bandpass',
    ...                                   analog=True))
    >>> filtz = signal.lti(*signal.bilinear(filts.num, filts.den, fs))
    >>> wz, hz = signal.freqz(filtz.num, filtz.den)
    >>> ws, hs = signal.freqs(filts.num, filts.den, worN=fs*wz)

    >>> plt.semilogx(wz*fs/(2*np.pi), 20*np.log10(np.abs(hz).clip(1e-15)),
    ...              label=r'$|H_z(e^{j \omega})|$')
    >>> plt.semilogx(wz*fs/(2*np.pi), 20*np.log10(np.abs(hs).clip(1e-15)),
    ...              label=r'$|H(j \omega)|$')
    >>> plt.legend()
    >>> plt.xlabel('Frequency [Hz]')
    >>> plt.ylabel('Magnitude [dB]')
    >>> plt.grid(True)
    """
    # 确保采样率为有效值，不允许为None
    fs = _validate_fs(fs, allow_none=False)
    # 确保a和b至少为一维数组
    a, b = map(atleast_1d, (a, b))
    # 计算系统阶数
    D = len(a) - 1
    N = len(b) - 1
    artype = float
    M = max([N, D])
    Np = M
    Dp = M
    # 初始化变换后的分子和分母系数数组
    bprime = np.empty(Np + 1, artype)
    aprime = np.empty(Dp + 1, artype)
    # 计算变换后的分子系数
    for j in range(Np + 1):
        val = 0.0
        for i in range(N + 1):
            for k in range(i + 1):
                for l in range(M - i + 1):
                    if k + l == j:
                        val += (comb(i, k) * comb(M - i, l) * b[N - i] *
                                pow(2 * fs, i) * (-1) ** k)
        bprime[j] = real(val)
    # 计算变换后的分母系数
    for j in range(Dp + 1):
        val = 0.0
        for i in range(D + 1):
            for k in range(i + 1):
                for l in range(M - i + 1):
                    if k + l == j:
                        val += (comb(i, k) * comb(M - i, l) * a[D - i] *
                                pow(2 * fs, i) * (-1) ** k)
        aprime[j] = real(val)

    # 返回标准化后的分子和分母系数数组
    return normalize(bprime, aprime)


# 验证通带增益和阻带增益是否合法
def _validate_gpass_gstop(gpass, gstop):
    # 如果通带增益不大于0，则引发异常
    if gpass <= 0.0:
        raise ValueError("gpass should be larger than 0.0")
    # 如果阻带增益不大于0，则引发异常
    elif gstop <= 0.0:
        raise ValueError("gstop should be larger than 0.0")
    # 如果 gpass 大于 gstop，则抛出数值错误异常
    elif gpass > gstop:
        raise ValueError("gpass should be smaller than gstop")
# 定义函数 iirdesign，用于设计 IIR 数字和模拟滤波器
def iirdesign(wp, ws, gpass, gstop, analog=False, ftype='ellip', output='ba',
              fs=None):
    """Complete IIR digital and analog filter design.

    Given passband and stopband frequencies and gains, construct an analog or
    digital IIR filter of minimum order for a given basic type. Return the
    output in numerator, denominator ('ba'), pole-zero ('zpk') or second order
    sections ('sos') form.

    Parameters
    ----------
    wp, ws : float or array like, shape (2,)
        Passband and stopband edge frequencies. Possible values are scalars
        (for lowpass and highpass filters) or ranges (for bandpass and bandstop
        filters).
        For digital filters, these are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. For example:

            - Lowpass:   wp = 0.2,          ws = 0.3
            - Highpass:  wp = 0.3,          ws = 0.2
            - Bandpass:  wp = [0.2, 0.5],   ws = [0.1, 0.6]
            - Bandstop:  wp = [0.1, 0.6],   ws = [0.2, 0.5]

        For analog filters, `wp` and `ws` are angular frequencies (e.g., rad/s).
        Note, that for bandpass and bandstop filters passband must lie strictly
        inside stopband or vice versa.
    gpass : float
        The maximum loss in the passband (dB).
    gstop : float
        The minimum attenuation in the stopband (dB).
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    ftype : str, optional
        The type of IIR filter to design:

            - Butterworth   : 'butter'
            - Chebyshev I   : 'cheby1'
            - Chebyshev II  : 'cheby2'
            - Cauer/elliptic: 'ellip'

    output : {'ba', 'zpk', 'sos'}, optional
        Filter form of the output:

            - second-order sections (recommended): 'sos'
            - numerator/denominator (default)    : 'ba'
            - pole-zero                          : 'zpk'

        In general the second-order sections form ('sos') is
        recommended because inferring the coefficients for the
        numerator/denominator form ('ba') suffers from numerical
        instabilities. For reasons of backward compatibility the default
        form is the numerator/denominator form ('ba'), where the 'b'
        and the 'a' in 'ba' refer to the commonly used names of the
        coefficients used.

        Note: Using the second-order sections form ('sos') is sometimes
        associated with additional computational costs: for
        data-intense use cases it is therefore recommended to also
        investigate the numerator/denominator form ('ba').

    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    ```
    # 函数实现略
    ```
    try:
        ordfunc = filter_dict[ftype][1]
    except KeyError as e:
        raise ValueError("Invalid IIR filter type: %s" % ftype) from e
    except IndexError as e:
        raise ValueError(("%s does not have order selection. Use "
                          "iirfilter function.") % ftype) from e



    # 尝试获取指定滤波器类型对应的函数（在 filter_dict 中查找）
    # 如果找不到对应的键（滤波器类型），抛出值错误异常
    # 如果索引超出范围（无法获取到所需的次序选择），抛出值错误异常
    ordfunc = filter_dict[ftype][1]



    _validate_gpass_gstop(gpass, gstop)



    # 调用函数 _validate_gpass_gstop 验证通带增益 gpass 和阻带增益 gstop 的有效性
    _validate_gpass_gstop(gpass, gstop)



    wp = atleast_1d(wp)
    ws = atleast_1d(ws)



    # 将 wp 和 ws 转换为至少是 1 维的数组
    wp = atleast_1d(wp)
    ws = atleast_1d(ws)



    fs = _validate_fs(fs, allow_none=True)



    # 调用函数 _validate_fs 验证采样频率 fs 的有效性，允许 fs 为 None
    fs = _validate_fs(fs, allow_none=True)



    if wp.shape[0] != ws.shape[0] or wp.shape not in [(1,), (2,)]:
        raise ValueError("wp and ws must have one or two elements each, and "
                         f"the same shape, got {wp.shape} and {ws.shape}")



    # 检查 wp 和 ws 的形状是否相同，且长度为 1 或 2
    # 如果不符合条件，抛出值错误异常，显示实际形状信息
    if wp.shape[0] != ws.shape[0] or wp.shape not in [(1,), (2,)]:
        raise ValueError("wp and ws must have one or two elements each, and "
                         f"the same shape, got {wp.shape} and {ws.shape}")



    if any(wp <= 0) or any(ws <= 0):
        raise ValueError("Values for wp, ws must be greater than 0")



    # 检查 wp 和 ws 中是否有小于等于 0 的值
    # 如果有，抛出值错误异常，要求 wp 和 ws 的值必须大于 0
    if any(wp <= 0) or any(ws <= 0):
        raise ValueError("Values for wp, ws must be greater than 0")



    if not analog:
        if fs is None:
            if any(wp >= 1) or any(ws >= 1):
                raise ValueError("Values for wp, ws must be less than 1")
        elif any(wp >= fs/2) or any(ws >= fs/2):
            raise ValueError("Values for wp, ws must be less than fs/2 "
                             f"(fs={fs} -> fs/2={fs/2})")



    # 如果不是模拟滤波器（digital=False）
    #   如果 fs 为 None，则检查 wp 和 ws 是否大于等于 1
    #   如果 fs 不为 None，则检查 wp 和 ws 是否大于等于 fs/2
    # 如果超过上述范围，抛出值错误异常，显示相应的限制条件
    if not analog:
        if fs is None:
            if any(wp >= 1) or any(ws >= 1):
                raise ValueError("Values for wp, ws must be less than 1")
        elif any(wp >= fs/2) or any(ws >= fs/2):
            raise ValueError("Values for wp, ws must be less than fs/2 "
                             f"(fs={fs} -> fs/2={fs/2})")
    # 检查滤波器参数 wp 的行数是否为 2
    if wp.shape[0] == 2:
        # 检查滤波器的通带必须严格位于阻带内或阻带严格位于通带内，否则引发数值错误
        if not ((ws[0] < wp[0] and wp[1] < ws[1]) or
               (wp[0] < ws[0] and ws[1] < wp[1])):
            raise ValueError("Passband must lie strictly inside stopband "
                             "or vice versa")

    # 计算带类型，基于滤波器参数 wp 的长度
    band_type = 2 * (len(wp) - 1)
    band_type += 1
    # 根据 wp 的第一个元素与 ws 的第一个元素的比较，确定带类型
    if wp[0] >= ws[0]:
        band_type += 1

    # 根据带类型选择滤波器类型 btype
    btype = {1: 'lowpass', 2: 'highpass',
             3: 'bandstop', 4: 'bandpass'}[band_type]

    # 调用 ordfunc 函数计算滤波器的阶数 N 和归一化频率 Wn
    N, Wn = ordfunc(wp, ws, gpass, gstop, analog=analog, fs=fs)
    
    # 返回使用 iirfilter 函数生成的滤波器设计
    return iirfilter(N, Wn, rp=gpass, rs=gstop, analog=analog, btype=btype,
                     ftype=ftype, output=output, fs=fs)
# 定义一个函数用于设计 IIR 数字和模拟滤波器，给定阶数和关键点
def iirfilter(N, Wn, rp=None, rs=None, btype='band', analog=False,
              ftype='butter', output='ba', fs=None):
    """
    IIR digital and analog filter design given order and critical points.

    Design an Nth-order digital or analog filter and return the filter
    coefficients.

    Parameters
    ----------
    N : int
        滤波器的阶数。
    Wn : array_like
        临界频率的标量或长度为2的序列。

        对于数字滤波器，`Wn` 的单位与 `fs` 相同。默认情况下，
        `fs` 是每样本2个半周期，因此这些频率被归一化到0到1之间，
        其中1是奈奎斯特频率。 (`Wn` 因此是每样本的半周期数。)

        对于模拟滤波器，`Wn` 是角频率 (例如，rad/s)。

        当 Wn 是长度为2的序列时，必须满足 ``Wn[0]`` 小于 ``Wn[1]``。
    rp : float, optional
        对于 Chebyshev 和 elliptic 滤波器，指定通带中的最大波动。 (dB)
    rs : float, optional
        对于 Chebyshev 和 elliptic 滤波器，指定阻带中的最小衰减。 (dB)
    btype : {'bandpass', 'lowpass', 'highpass', 'bandstop'}, optional
        滤波器的类型。默认为 'bandpass'。
    analog : bool, optional
        当为 True 时，返回模拟滤波器；否则返回数字滤波器。
    ftype : str, optional
        要设计的 IIR 滤波器类型:

            - Butterworth   : 'butter'
            - Chebyshev I   : 'cheby1'
            - Chebyshev II  : 'cheby2'
            - Cauer/elliptic: 'ellip'
            - Bessel/Thomson: 'bessel'

    output : {'ba', 'zpk', 'sos'}, optional
        输出的滤波器形式:

            - second-order sections (推荐): 'sos'
            - numerator/denominator (默认)  : 'ba'
            - pole-zero                      : 'zpk'

        通常推荐使用二阶段形式 ('sos')，因为从分子/分母形式 ('ba') 推断系数存在数值不稳定性。
        由于向后兼容性的原因，默认形式是分子/分母形式 ('ba')，其中 'b' 和 'a' 是常用的系数名称。

        注意: 使用二阶段形式 ('sos') 有时会伴随额外的计算成本：因此建议对于数据密集的用例，
        也要调查分子/分母形式 ('ba')。

    fs : float, optional
        数字系统的采样频率。

        .. versionadded:: 1.2.0

    Returns
    -------
    b, a : ndarray, ndarray
        IIR 滤波器的分子 (`b`) 和分母 (`a`) 多项式系数。仅当 ``output='ba'`` 时返回。
    z, p, k : ndarray, ndarray, float
        IIR 滤波器传递函数的零点、极点和系统增益。仅当 ``output='zpk'`` 时返回。
    """
    # 验证并适当处理采样频率参数，允许其为None
    fs = _validate_fs(fs, allow_none=True)
    # 将滤波器类型、带通类型和输出类型转换为小写
    ftype, btype, output = (x.lower() for x in (ftype, btype, output))
    # 将频率数组转换为NumPy数组
    Wn = asarray(Wn)
    # 如果指定了采样频率且为数字滤波器，则将频率范围归一化
    if fs is not None:
        if analog:
            # 如果为模拟滤波器，则不能指定采样频率
            raise ValueError("fs cannot be specified for an analog filter")
        Wn = Wn / (fs/2)

    # 如果关键频率小于等于0，则抛出值错误
    if np.any(Wn <= 0):
        raise ValueError("filter critical frequencies must be greater than 0")

    # 如果频率数组大小大于1且第一个元素不小于第二个元素，则抛出值错误
    if Wn.size > 1 and not Wn[0] < Wn[1]:
        raise ValueError("Wn[0] must be less than Wn[1]")

    # 根据带通类型从字典中获取相应的滤波器类型
    try:
        btype = band_dict[btype]
    except KeyError as e:
        raise ValueError("'%s' is an invalid bandtype for filter." % btype) from e

    # 根据滤波器类型从字典中获取对应的滤波器函数
    try:
        typefunc = filter_dict[ftype][0]
    except KeyError as e:
        raise ValueError("'%s' is not a valid basic IIR filter." % ftype) from e
    # 检查输出类型是否为合法值，如果不是，则引发值错误异常
    if output not in ['ba', 'zpk', 'sos']:
        raise ValueError("'%s' is not a valid output form." % output)

    # 如果 rp 参数不为 None 且小于 0，则引发值错误异常
    if rp is not None and rp < 0:
        raise ValueError("passband ripple (rp) must be positive")

    # 如果 rs 参数不为 None 且小于 0，则引发值错误异常
    if rs is not None and rs < 0:
        raise ValueError("stopband attenuation (rs) must be positive")

    # 根据所选的滤波器类型获取模拟低通原型滤波器的零点、极点和增益
    if typefunc == buttap:
        z, p, k = typefunc(N)
    elif typefunc == besselap:
        z, p, k = typefunc(N, norm=bessel_norms[ftype])
    elif typefunc == cheb1ap:
        # 如果 rp 参数为 None，则引发值错误异常，要求提供通带波纹参数 rp
        if rp is None:
            raise ValueError("passband ripple (rp) must be provided to "
                             "design a Chebyshev I filter.")
        z, p, k = typefunc(N, rp)
    elif typefunc == cheb2ap:
        # 如果 rs 参数为 None，则引发值错误异常，要求提供阻带衰减参数 rs
        if rs is None:
            raise ValueError("stopband attenuation (rs) must be provided to "
                             "design an Chebyshev II filter.")
        z, p, k = typefunc(N, rs)
    elif typefunc == ellipap:
        # 如果 rs 或 rp 参数为 None，则引发值错误异常，要求同时提供两者以设计椭圆滤波器
        if rs is None or rp is None:
            raise ValueError("Both rp and rs must be provided to design an "
                             "elliptic filter.")
        z, p, k = typefunc(N, rp, rs)
    else:
        # 如果选定的滤波器类型不在支持列表中，则引发未实现错误异常
        raise NotImplementedError("'%s' not implemented in iirfilter." % ftype)

    # 对于数字滤波器设计，预处理频率以进行数字滤波器设计
    if not analog:
        # 如果任何关键频率值 Wn 不在 (0, 1) 范围内，则根据情况引发值错误异常
        if np.any(Wn <= 0) or np.any(Wn >= 1):
            if fs is not None:
                raise ValueError("Digital filter critical frequencies must "
                                 f"be 0 < Wn < fs/2 (fs={fs} -> fs/2={fs/2})")
            raise ValueError("Digital filter critical frequencies "
                             "must be 0 < Wn < 1")
        fs = 2.0
        # 进行预弯曲，转换频率为数字滤波器设计所需的形式
        warped = 2 * fs * tan(pi * Wn / fs)
    else:
        warped = Wn

    # 根据滤波器类型进行转换为低通、带通、高通或带阻形式
    if btype in ('lowpass', 'highpass'):
        # 如果关键频率 Wn 不是单一值，则引发值错误异常
        if np.size(Wn) != 1:
            raise ValueError('Must specify a single critical frequency Wn '
                             'for lowpass or highpass filter')

        if btype == 'lowpass':
            # 转换为数字低通滤波器零点、极点和增益
            z, p, k = lp2lp_zpk(z, p, k, wo=warped)
        elif btype == 'highpass':
            # 转换为数字高通滤波器零点、极点和增益
            z, p, k = lp2hp_zpk(z, p, k, wo=warped)
    elif btype in ('bandpass', 'bandstop'):
        try:
            # 计算带通或带阻滤波器的带宽和中心频率
            bw = warped[1] - warped[0]
            wo = sqrt(warped[0] * warped[1])
        except IndexError as e:
            # 如果 Wn 没有正确指定起始和结束频率，则引发值错误异常
            raise ValueError('Wn must specify start and stop frequencies for '
                             'bandpass or bandstop filter') from e

        if btype == 'bandpass':
            # 转换为数字带通滤波器零点、极点和增益
            z, p, k = lp2bp_zpk(z, p, k, wo=wo, bw=bw)
        elif btype == 'bandstop':
            # 转换为数字带阻滤波器零点、极点和增益
            z, p, k = lp2bs_zpk(z, p, k, wo=wo, bw=bw)
    else:
        # 如果选定的滤波器类型不在支持列表中，则引发未实现错误异常
        raise NotImplementedError("'%s' not implemented in iirfilter." % btype)

    # 如果需要，将结果转换为适当的输出类型（极点-零点形式、状态空间形式、数值-分母形式）
    if not analog:
        # 如果是数字滤波器设计，则将结果转换为离散等效形式
        z, p, k = bilinear_zpk(z, p, k, fs=fs)
    # 如果输出类型为 'zpk'，则直接返回零极点增益元组 (z, p, k)
    if output == 'zpk':
        return z, p, k
    # 如果输出类型为 'ba'，则将零极点增益转换为传递函数分子系数和分母系数并返回
    elif output == 'ba':
        return zpk2tf(z, p, k)
    # 如果输出类型为 'sos'，则将零极点增益转换为二阶段独立结构并返回
    elif output == 'sos':
        return zpk2sos(z, p, k, analog=analog)
# 计算转移函数的相对阶数，从零点和极点
def _relative_degree(z, p):
    degree = len(p) - len(z)
    if degree < 0:
        raise ValueError("Improper transfer function. "
                         "Must have at least as many poles as zeros.")
    else:
        return degree


# 使用双线性变换从模拟滤波器转换为数字滤波器
def bilinear_zpk(z, p, k, fs):
    """
    使用双线性变换将模拟系统的零点和极点转换为数字系统的零点和极点。
    双线性变换使用Tustin方法，用``2*fs*(z-1) / (z+1)``代替``s``，以保持频率响应的形状。

    Parameters
    ----------
    z : array_like
        模拟滤波器传递函数的零点。
    p : array_like
        模拟滤波器传递函数的极点。
    k : float
        模拟滤波器传递函数的系统增益。
    fs : float
        采样率，普通频率（例如赫兹）。此函数中不进行预扭曲。

    Returns
    -------
    z : ndarray
        转换后的数字滤波器传递函数的零点。
    p : ndarray
        转换后的数字滤波器传递函数的极点。
    k : float
        转换后的数字滤波器的系统增益。

    See Also
    --------
    lp2lp_zpk, lp2hp_zpk, lp2bp_zpk, lp2bs_zpk
    bilinear

    Notes
    -----
    .. versionadded:: 1.1.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> fs = 100
    >>> bf = 2 * np.pi * np.array([7, 13])
    >>> filts = signal.lti(*signal.butter(4, bf, btype='bandpass', analog=True,
    ...                                   output='zpk'))
    >>> filtz = signal.lti(*signal.bilinear_zpk(filts.zeros, filts.poles,
    ...                                         filts.gain, fs))
    >>> wz, hz = signal.freqz_zpk(filtz.zeros, filtz.poles, filtz.gain)
    >>> ws, hs = signal.freqs_zpk(filts.zeros, filts.poles, filts.gain,
    ...                           worN=fs*wz)
    >>> plt.semilogx(wz*fs/(2*np.pi), 20*np.log10(np.abs(hz).clip(1e-15)),
    ...              label=r'$|H_z(e^{j \omega})|$')
    >>> plt.semilogx(wz*fs/(2*np.pi), 20*np.log10(np.abs(hs).clip(1e-15)),
    ...              label=r'$|H(j \omega)|$')
    >>> plt.legend()
    >>> plt.xlabel('Frequency [Hz]')
    >>> plt.ylabel('Magnitude [dB]')
    >>> plt.grid(True)
    """
    z = atleast_1d(z)
    p = atleast_1d(p)

    # 验证采样率的有效性
    fs = _validate_fs(fs, allow_none=False)

    # 计算相对阶数
    degree = _relative_degree(z, p)

    # 计算两倍采样率
    fs2 = 2.0 * fs

    # 对极点和零点进行双线性变换
    z_z = (fs2 + z) / (fs2 - z)
    p_z = (fs2 + p) / (fs2 - p)

    # 将原本位于无穷远处的零点移动到奈奎斯特频率
    z_z = append(z_z, -ones(degree))

    # 补偿增益变化
    k_z = k * real(prod(fs2 - z) / prod(fs2 - p))

    return z_z, p_z, k_z


def lp2lp_zpk(z, p, k, wo=1.0):
    r"""
    """
    Transform a lowpass filter prototype to a different frequency.

    Return an analog low-pass filter with cutoff frequency `wo`
    from an analog low-pass filter prototype with unity cutoff frequency,
    using zeros, poles, and gain ('zpk') representation.

    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    wo : float
        Desired cutoff, as angular frequency (e.g., rad/s).
        Defaults to no change.

    Returns
    -------
    z : ndarray
        Zeros of the transformed low-pass filter transfer function.
    p : ndarray
        Poles of the transformed low-pass filter transfer function.
    k : float
        System gain of the transformed low-pass filter.

    See Also
    --------
    lp2hp_zpk, lp2bp_zpk, lp2bs_zpk, bilinear
    lp2lp

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \rightarrow \frac{s}{\omega_0}

    .. versionadded:: 1.1.0

    Examples
    --------
    Use the 'zpk' (Zero-Pole-Gain) representation of a lowpass filter to 
    transform it to a new 'zpk' representation associated with a cutoff frequency wo.

    >>> from scipy.signal import lp2lp_zpk
    >>> z   = [7,   2]
    >>> p   = [5,   13]
    >>> k   = 0.8
    >>> wo  = 0.4
    >>> lp2lp_zpk(z, p, k, wo)
    (   array([2.8, 0.8]), array([2. , 5.2]), 0.8)
    """

    # Ensure z and p are arrays
    z = atleast_1d(z)
    p = atleast_1d(p)
    # Convert wo to float to avoid integer wraparound issues
    wo = float(wo)  # Avoid int wraparound

    # Calculate the relative degree of the filter
    degree = _relative_degree(z, p)

    # Scale zeros and poles to shift cutoff frequency
    z_lp = wo * z
    p_lp = wo * p

    # Adjust the gain to maintain overall system gain
    k_lp = k * wo**degree

    # Return transformed zeros, poles, and gain
    return z_lp, p_lp, k_lp
# 将低通滤波器原型转换为高通滤波器

def lp2hp_zpk(z, p, k, wo=1.0):
    """
    Transform a lowpass filter prototype to a highpass filter.

    Return an analog high-pass filter with cutoff frequency `wo`
    from an analog low-pass filter prototype with unity cutoff frequency,
    using zeros, poles, and gain ('zpk') representation.

    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    wo : float
        Desired cutoff, as angular frequency (e.g., rad/s).
        Defaults to no change.

    Returns
    -------
    z : ndarray
        Zeros of the transformed high-pass filter transfer function.
    p : ndarray
        Poles of the transformed high-pass filter transfer function.
    k : float
        System gain of the transformed high-pass filter.

    See Also
    --------
    lp2lp_zpk, lp2bp_zpk, lp2bs_zpk, bilinear
    lp2hp

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \rightarrow \frac{\omega_0}{s}

    This maintains symmetry of the lowpass and highpass responses on a
    logarithmic scale.

    .. versionadded:: 1.1.0
    
    Examples
    --------
    Use the 'zpk' (Zero-Pole-Gain) representation of a lowpass filter to 
    transform it to a highpass filter with a cutoff frequency wo.
    
    >>> from scipy.signal import lp2hp_zpk
    >>> z   = [ -2 + 3j ,  -0.5 - 0.8j ]
    >>> p   = [ -1      ,  -4          ]
    >>> k   = 10
    >>> wo  = 0.6
    >>> lp2hp_zpk(z, p, k, wo)
    (   array([-0.09230769-0.13846154j, -0.33707865+0.53932584j]),
        array([-0.6 , -0.15]),
        8.5)
    """
    # Ensure z, p are arrays
    z = atleast_1d(z)
    p = atleast_1d(p)
    wo = float(wo)

    # Calculate the relative degree of the system
    degree = _relative_degree(z, p)

    # Invert positions radially about unit circle to convert LPF to HPF
    z_hp = wo / z
    p_hp = wo / p

    # If lowpass had zeros at infinity, inverting moves them to origin.
    z_hp = append(z_hp, zeros(degree))

    # Cancel out gain change caused by inversion
    k_hp = k * real(prod(-z) / prod(-p))

    return z_hp, p_hp, k_hp


# 将低通滤波器原型转换为带通滤波器

def lp2bp_zpk(z, p, k, wo=1.0, bw=1.0):
    """
    Transform a lowpass filter prototype to a bandpass filter.

    Return an analog band-pass filter with center frequency `wo` and
    bandwidth `bw` from an analog low-pass filter prototype with unity
    cutoff frequency, using zeros, poles, and gain ('zpk') representation.

    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    wo : float
        Desired passband center, as angular frequency (e.g., rad/s).
        Defaults to no change.
    bw : float
        Bandwidth of the bandpass filter.
        Defaults to unity bandwidth.
    # 将输入的零点 z 和极点 p 至少转换为一维数组
    z = atleast_1d(z)
    p = atleast_1d(p)
    # 将中心频率 wo 和带宽 bw 转换为浮点数
    wo = float(wo)
    bw = float(bw)

    # 计算零点和极点的相对阶数
    degree = _relative_degree(z, p)

    # 将零点和极点按照指定的带宽进行缩放
    z_lp = z * bw/2
    p_lp = p * bw/2

    # 确保平方根计算产生复数结果，而不是 NaN
    z_lp = z_lp.astype(complex)
    p_lp = p_lp.astype(complex)

    # 复制零点和极点，并将它们从基带移至 +wo 和 -wo
    z_bp = concatenate((z_lp + sqrt(z_lp**2 - wo**2),
                        z_lp - sqrt(z_lp**2 - wo**2)))
    p_bp = concatenate((p_lp + sqrt(p_lp**2 - wo**2),
                        p_lp - sqrt(p_lp**2 - wo**2)))

    # 将相对阶数的零点移到原点，剩余的零点移到无穷远处形成带通滤波器
    z_bp = append(z_bp, zeros(degree))

    # 取消因频率缩放而产生的增益变化
    k_bp = k * bw**degree

    # 返回变换后的带通滤波器的零点、极点和增益
    return z_bp, p_bp, k_bp
def lp2bs_zpk(z, p, k, wo=1.0, bw=1.0):
    """
    Transform a lowpass filter prototype to a bandstop filter.

    Return an analog band-stop filter with center frequency `wo` and
    stopband width `bw` from an analog low-pass filter prototype with unity
    cutoff frequency, using zeros, poles, and gain ('zpk') representation.

    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    wo : float
        Desired stopband center, as angular frequency (e.g., rad/s).
        Defaults to no change.
    bw : float
        Desired stopband width, as angular frequency (e.g., rad/s).
        Defaults to 1.

    Returns
    -------
    z_bs : ndarray
        Zeros of the transformed band-stop filter transfer function.
    p_bs : ndarray
        Poles of the transformed band-stop filter transfer function.
    k_bs : float
        System gain of the transformed band-stop filter.

    See Also
    --------
    lp2lp_zpk, lp2hp_zpk, lp2bp_zpk, bilinear
    lp2bs

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \rightarrow \frac{s \cdot \mathrm{BW}}{s^2 + {\omega_0}^2}

    This is the "wideband" transformation, producing a stopband with
    geometric (log frequency) symmetry about `wo`.

    .. versionadded:: 1.1.0

    Examples
    --------
    Transform a low-pass filter represented in 'zpk' (Zero-Pole-Gain) form 
    into a bandstop filter represented in 'zpk' form, with a center frequency wo and
    bandwidth bw.
    
    >>> from scipy.signal import lp2bs_zpk
    >>> z   = [             ]  # Zeros of the original low-pass filter transfer function
    >>> p   = [ 0.7 ,    -1 ]  # Poles of the original low-pass filter transfer function
    >>> k   = 9                # System gain of the original low-pass filter
    >>> wo  = 0.5              # Desired stopband center frequency (angular frequency)
    >>> bw  = 10               # Desired stopband width (angular frequency)
    >>> lp2bs_zpk(z, p, k, wo, bw)
    (   array([0.+0.5j, 0.+0.5j, 0.-0.5j, 0.-0.5j]), 
        array([14.2681928 +0.j, -0.02506281+0.j,  0.01752149+0.j, -9.97493719+0.j]), 
        -12.857142857142858)
    """
    z = atleast_1d(z)   # Ensure zeros are at least one-dimensional
    p = atleast_1d(p)   # Ensure poles are at least one-dimensional
    wo = float(wo)      # Convert wo to float
    bw = float(bw)      # Convert bw to float

    degree = _relative_degree(z, p)  # Compute relative degree of the system

    # Invert to a highpass filter with desired bandwidth
    z_hp = (bw/2) / z
    p_hp = (bw/2) / p

    # Square root needs to produce complex result, not NaN
    z_hp = z_hp.astype(complex)
    p_hp = p_hp.astype(complex)

    # Duplicate poles and zeros and shift from baseband to +wo and -wo
    z_bs = concatenate((z_hp + sqrt(z_hp**2 - wo**2),
                        z_hp - sqrt(z_hp**2 - wo**2)))
    p_bs = concatenate((p_hp + sqrt(p_hp**2 - wo**2),
                        p_hp - sqrt(p_hp**2 - wo**2)))

    # Move any zeros that were at infinity to the center of the stopband
    z_bs = append(z_bs, full(degree, +1j*wo))
    z_bs = append(z_bs, full(degree, -1j*wo))

    # Cancel out gain change caused by inversion
    k_bs = k * real(prod(-z) / prod(-p))

    return z_bs, p_bs, k_bs
# 定义 Butterworth 数字和模拟滤波器设计函数
def butter(N, Wn, btype='low', analog=False, output='ba', fs=None):
    """
    Butterworth digital and analog filter design.

    Design an Nth-order digital or analog Butterworth filter and return
    the filter coefficients.

    Parameters
    ----------
    N : int
        The order of the filter. For 'bandpass' and 'bandstop' filters,
        the resulting order of the final second-order sections ('sos')
        matrix is ``2*N``, with `N` the number of biquad sections
        of the desired system.
    Wn : array_like
        The critical frequency or frequencies. For lowpass and highpass
        filters, Wn is a scalar; for bandpass and bandstop filters,
        Wn is a length-2 sequence.

        For a Butterworth filter, this is the point at which the gain
        drops to 1/sqrt(2) that of the passband (the "-3 dB point").

        For digital filters, if `fs` is not specified, `Wn` units are
        normalized from 0 to 1, where 1 is the Nyquist frequency (`Wn` is
        thus in half cycles / sample and defined as 2*critical frequencies
        / `fs`). If `fs` is specified, `Wn` is in the same units as `fs`.

        For analog filters, `Wn` is an angular frequency (e.g. rad/s).
    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        The type of filter.  Default is 'lowpass'.
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    output : {'ba', 'zpk', 'sos'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        second-order sections ('sos'). Default is 'ba' for backwards
        compatibility, but 'sos' should be used for general-purpose filtering.
    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.
    sos : ndarray
        Second-order sections representation of the IIR filter.
        Only returned if ``output='sos'``.

    See Also
    --------
    buttord, buttap

    Notes
    -----
    The Butterworth filter has maximally flat frequency response in the
    passband.

    The ``'sos'`` output parameter was added in 0.16.0.

    If the transfer function form ``[b, a]`` is requested, numerical
    problems can occur since the conversion between roots and
    the polynomial coefficients is a numerically sensitive operation,
    even for N >= 4. It is recommended to work with the SOS
    representation.
    """
    # 返回一个 Butterworth IIR 滤波器的系数
    return iirfilter(N, Wn, btype=btype, analog=analog,
                     output=output, ftype='butter', fs=fs)
# 定义 Chebyshev Type I 数字和模拟滤波器设计函数
def cheby1(N, rp, Wn, btype='low', analog=False, output='ba', fs=None):
    """
    Chebyshev type I digital and analog filter design.

    Design an Nth-order digital or analog Chebyshev type I filter and
    return the filter coefficients.

    Parameters
    ----------
    N : int
        The order of the filter.
    rp : float
        The maximum ripple allowed below unity gain in the passband.
        Specified in decibels, as a positive number.
    Wn : array_like
        A scalar or length-2 sequence giving the critical frequencies.
        For Type I filters, this is the point in the transition band at which
        the gain first drops below -`rp`.

        For digital filters, `Wn` are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`Wn` is thus in
        half-cycles / sample.)

        For analog filters, `Wn` is an angular frequency (e.g., rad/s).
    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        The type of filter.  Default is 'lowpass'.
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    output : {'ba', 'zpk', 'sos'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        second-order sections ('sos'). Default is 'ba' for backwards
        compatibility, but 'sos' should be used for general-purpose filtering.
    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.
    sos : ndarray
        Second-order sections representation of the IIR filter.
        Only returned if ``output='sos'``.

    See Also
    --------
    cheb1ord, cheb1ap

    Notes
    -----
    The Chebyshev type I filter maximizes the rate of cutoff between the
    frequency response's passband and stopband, at the expense of ripple in
    the passband and increased ringing in the step response.

    Type I filters roll off faster than Type II (`cheby2`), but Type II
    filters do not have any ripple in the passband.

    The equiripple passband has N maxima or minima (for example, a
    5th-order filter has 3 maxima and 2 minima). Consequently, the DC gain is
    unity for odd-order filters, or -rp dB for even-order filters.

    The ``'sos'`` output parameter was added in 0.16.0.

    Examples
    --------
    Design an analog filter and plot its frequency response, showing the
    critical points:

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    """
    pass  # 这里只是函数定义，实际实现在函数体内进行
    >>> b, a = signal.cheby1(4, 5, 100, 'low', analog=True)
    # 使用 scipy.signal 库中的 cheby1 函数设计一个四阶 Chebyshev Type I 低通滤波器
    >>> w, h = signal.freqs(b, a)
    # 计算滤波器的频率响应
    >>> plt.semilogx(w, 20 * np.log10(abs(h)))
    # 绘制频率响应曲线，横轴为频率（对数刻度），纵轴为增益（单位：dB）
    >>> plt.title('Chebyshev Type I frequency response (rp=5)')
    # 设置图标题为 Chebyshev Type I 滤波器的频率响应（rp=5）
    >>> plt.xlabel('Frequency [radians / second]')
    # 设置横轴标签为频率（单位：弧度/秒）
    >>> plt.ylabel('Amplitude [dB]')
    # 设置纵轴标签为振幅（单位：dB）
    >>> plt.margins(0, 0.1)
    # 设置图的边距
    >>> plt.grid(which='both', axis='both')
    # 显示网格线，包括主要网格和次要网格，横纵均包括
    >>> plt.axvline(100, color='green') # cutoff frequency
    # 在频率为 100 的位置画一条垂直绿色线，表示截止频率
    >>> plt.axhline(-5, color='green') # rp
    # 在增益为 -5 dB 的位置画一条水平绿色线，表示通带波动限制 rp
    >>> plt.show()
    # 显示绘制的图形

    Generate a signal made up of 10 Hz and 20 Hz, sampled at 1 kHz

    >>> t = np.linspace(0, 1, 1000, False)  # 1 second
    # 生成一个 1 秒长的时间向量 t，包含 1000 个点，采样频率为 1 kHz
    >>> sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)
    # 生成一个由 10 Hz 和 20 Hz 正弦信号叠加而成的信号 sig
    >>> fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    # 创建一个包含两个子图的图像窗口，共享横轴
    >>> ax1.plot(t, sig)
    # 在第一个子图 ax1 上绘制信号 sig 随时间的波形图
    >>> ax1.set_title('10 Hz and 20 Hz sinusoids')
    # 设置第一个子图的标题为 '10 Hz and 20 Hz sinusoids'
    >>> ax1.axis([0, 1, -2, 2])
    # 设置第一个子图的坐标轴范围为 x 轴 [0, 1]，y 轴 [-2, 2]

    Design a digital high-pass filter at 15 Hz to remove the 10 Hz tone, and
    apply it to the signal. (It's recommended to use second-order sections
    format when filtering, to avoid numerical error with transfer function
    (``ba``) format):

    >>> sos = signal.cheby1(10, 1, 15, 'hp', fs=1000, output='sos')
    # 使用 scipy.signal 库中的 cheby1 函数设计一个阶数为 10 的 Chebyshev Type I 高通数字滤波器，
    # 截止频率为 15 Hz，采样频率为 1 kHz，输出为 second-order sections 形式
    >>> filtered = signal.sosfilt(sos, sig)
    # 对信号 sig 应用设计好的滤波器 sos 进行滤波
    >>> ax2.plot(t, filtered)
    # 在第二个子图 ax2 上绘制滤波后的信号随时间的波形图
    >>> ax2.set_title('After 15 Hz high-pass filter')
    # 设置第二个子图的标题为 'After 15 Hz high-pass filter'
    >>> ax2.axis([0, 1, -2, 2])
    # 设置第二个子图的坐标轴范围为 x 轴 [0, 1]，y 轴 [-2, 2]
    >>> ax2.set_xlabel('Time [seconds]')
    # 设置第二个子图的横轴标签为 'Time [seconds]'
    >>> plt.tight_layout()
    # 调整子图的布局使其更加紧凑
    >>> plt.show()
    # 显示绘制的图形
    """
    return iirfilter(N, Wn, rp=rp, btype=btype, analog=analog,
                     output=output, ftype='cheby1', fs=fs)
    # 返回一个 Chebyshev Type I 滤波器的数字实现，使用 iirfilter 函数进行设计，参数包括阶数 N，
    # 截止频率 Wn，通带波动限制 rp，滤波器类型 btype，模拟/数字滤波器选择 analog，输出格式 output，
    # 滤波器类型 ftype 设为 'cheby1'，采样频率 fs
# 定义 Chebyshev Type II 数字和模拟滤波器设计函数
def cheby2(N, rs, Wn, btype='low', analog=False, output='ba', fs=None):
    """
    Chebyshev type II digital and analog filter design.

    Design an Nth-order digital or analog Chebyshev type II filter and
    return the filter coefficients.

    Parameters
    ----------
    N : int
        滤波器的阶数。
    rs : float
        在阻带中所需的最小衰减。
        以分贝为单位，为正数。
    Wn : array_like
        一个标量或长度为2的序列，给出关键频率。
        对于 Type II 滤波器，这是增益首次达到 -`rs` 的过渡带中的点。

        对于数字滤波器，`Wn` 的单位与 `fs` 相同。默认情况下，
        `fs` 为每个样本的2半周期，因此这些值被归一化为从0到1，
        其中1是奈奎斯特频率。(`Wn` 因此以半周期/样本表示。)

        对于模拟滤波器，`Wn` 是角频率（例如，rad/s）。
    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        滤波器的类型。默认为 'lowpass'。
    analog : bool, optional
        当为 True 时，返回模拟滤波器；否则返回数字滤波器。
    output : {'ba', 'zpk', 'sos'}, optional
        输出类型：分子/分母 ('ba')，极点-零点 ('zpk') 或
        二阶段 ('sos')。默认为 'ba' 以保持向后兼容性，
        但通常应使用 'sos' 进行通用滤波。
    fs : float, optional
        数字系统的采样频率。

        .. versionadded:: 1.2.0

    Returns
    -------
    b, a : ndarray, ndarray
        IIR 滤波器的分子 (`b`) 和分母 (`a`) 多项式。
        仅在 ``output='ba'`` 时返回。
    z, p, k : ndarray, ndarray, float
        IIR 滤波器传递函数的零点、极点和系统增益。
        仅在 ``output='zpk'`` 时返回。
    sos : ndarray
        IIR 滤波器的二阶段表示。
        仅在 ``output='sos'`` 时返回。

    See Also
    --------
    cheb2ord, cheb2ap

    Notes
    -----
    Chebyshev Type II 滤波器在频率响应的通带和阻带之间最大化截止率，
    但阻带中会有波动，并且在阶跃响应中增加振荡。

    Type II 滤波器的衰减不如 Type I (`cheby1`) 快。

    ``'sos'`` 输出参数在版本 0.16.0 中添加。

    Examples
    --------
    设计一个模拟滤波器并绘制其频率响应，显示关键点：

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> b, a = signal.cheby2(4, 40, 100, 'low', analog=True)
    >>> w, h = signal.freqs(b, a)
    >>> plt.semilogx(w, 20 * np.log10(abs(h)))
    >>> plt.title('Chebyshev Type II frequency response (rs=40)')
    >>> plt.xlabel('Frequency [radians / second]')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.margins(0, 0.1)
    >>> plt.grid(which='both', axis='both')
    # 在图表中显示网格，包括主要和次要网格线
    >>> plt.axvline(100, color='green') # cutoff frequency
    # 在 x = 100 处画一条垂直线，颜色为绿色，表示截止频率
    >>> plt.axhline(-40, color='green') # rs
    # 在 y = -40 处画一条水平线，颜色为绿色，表示衰减比 rs
    >>> plt.show()
    # 显示图形

    Generate a signal made up of 10 Hz and 20 Hz, sampled at 1 kHz

    >>> t = np.linspace(0, 1, 1000, False)  # 1 second
    # 生成一个时间数组 t，包含 1000 个点，用于表示 0 到 1 秒之间的时间
    >>> sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)
    # 生成一个信号 sig，由两个正弦波叠加而成，分别为 10 Hz 和 20 Hz
    >>> fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    # 创建一个包含两个子图的图形窗口，两个子图共享 x 轴
    >>> ax1.plot(t, sig)
    # 在第一个子图 ax1 上绘制时间 t 对应的信号 sig
    >>> ax1.set_title('10 Hz and 20 Hz sinusoids')
    # 设置第一个子图的标题为 '10 Hz and 20 Hz sinusoids'
    >>> ax1.axis([0, 1, -2, 2])
    # 设置第一个子图的坐标轴范围为 x 轴从 0 到 1，y 轴从 -2 到 2

    Design a digital high-pass filter at 17 Hz to remove the 10 Hz tone, and
    apply it to the signal. (It's recommended to use second-order sections
    format when filtering, to avoid numerical error with transfer function
    (``ba``) format):

    >>> sos = signal.cheby2(12, 20, 17, 'hp', fs=1000, output='sos')
    # 设计一个 12 阶的切比雪夫二型数字高通滤波器，在 17 Hz 处截止，采样频率为 1000 Hz
    >>> filtered = signal.sosfilt(sos, sig)
    # 使用设计好的滤波器 sos 对信号 sig 进行滤波
    >>> ax2.plot(t, filtered)
    # 在第二个子图 ax2 上绘制滤波后的信号 filtered
    >>> ax2.set_title('After 17 Hz high-pass filter')
    # 设置第二个子图的标题为 'After 17 Hz high-pass filter'
    >>> ax2.axis([0, 1, -2, 2])
    # 设置第二个子图的坐标轴范围为 x 轴从 0 到 1，y 轴从 -2 到 2
    >>> ax2.set_xlabel('Time [seconds]')
    # 设置第二个子图的 x 轴标签为 'Time [seconds]'
    >>> plt.show()
    # 显示图形
    """
    return iirfilter(N, Wn, rs=rs, btype=btype, analog=analog,
                     output=output, ftype='cheby2', fs=fs)
    # 返回一个 IIR 滤波器设计，使用切比雪夫二型滤波器 (cheby2)，参数由 N、Wn、rs、btype 等指定，采样频率为 fs
# 定义一个函数用于设计椭圆（Cauer）数字和模拟滤波器
def ellip(N, rp, rs, Wn, btype='low', analog=False, output='ba', fs=None):
    """
    Elliptic (Cauer) digital and analog filter design.

    Design an Nth-order digital or analog elliptic filter and return
    the filter coefficients.

    Parameters
    ----------
    N : int
        滤波器的阶数。
    rp : float
        在通带中允许的最大纹波。以分贝表示，为正数。
    rs : float
        在阻带中所需的最小衰减。以分贝表示，为正数。
    Wn : array_like
        一个标量或长度为2的序列，给出关键频率。
        对于椭圆滤波器，这是在转变带中增益首次下降到低于-rp的点。

        对于数字滤波器，Wn以与fs相同的单位表示。默认情况下，fs为每样本2个半周期，因此这些值被标准化为0到1，
        其中1是奈奎斯特频率。（因此，Wn以每半周期/样本的单位表示。）

        对于模拟滤波器，Wn是一个角频率（例如，rad/s）。
    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        滤波器的类型。默认为'lowpass'。
    analog : bool, optional
        当为True时，返回模拟滤波器；否则返回数字滤波器。
    output : {'ba', 'zpk', 'sos'}, optional
        输出类型：分子/分母（'ba'）、极点-零点（'zpk'）或二阶段（'sos'）。默认为'ba'以保持向后兼容性，
        但一般情况下应使用'sos'进行通用滤波。
    fs : float, optional
        数字系统的采样频率。

        .. versionadded:: 1.2.0

    Returns
    -------
    b, a : ndarray, ndarray
        IIR滤波器的分子（b）和分母（a）多项式系数。仅在output='ba'时返回。
    z, p, k : ndarray, ndarray, float
        IIR滤波器传递函数的零点、极点和系统增益。仅在output='zpk'时返回。
    sos : ndarray
        IIR滤波器的二阶段表示。仅在output='sos'时返回。

    See Also
    --------
    ellipord, ellipap

    Notes
    -----
    椭圆（Cauer）滤波器也称为Cauer或Zolotarev滤波器，它在通带和阻带的波动都很小，
    并在频率响应的通带和阻带之间的过渡区域内最大化过渡速率，
    但会导致阶跃响应中的增加振荡。

    当rp接近0时，椭圆滤波器变成切比雪夫II型滤波器（`cheby2`）。
    当rs接近0时，变成切比雪夫I型滤波器（`cheby1`）。
    当rp和rs都接近0时，变成巴特沃斯滤波器（`butter`）。

    等波纹通带具有N个最大值或最小值（例如，5阶滤波器有3个最大值和2个最小值）。
    因此，奇数阶滤波器的直流增益为1，偶数阶滤波器为-rp dB。

    """
    """
    设计一个 IIR 滤波器（无限脉冲响应滤波器），根据指定的参数生成一个低通或高通滤波器。

    Parameters
    ----------
    N : int
        滤波器的阶数（次数）
    Wn : float or tuple
        滤波器的归一化截止频率，可以是单个数字或者一对数字（对于带通或带阻滤波器）
    rs : float, optional
        带阻滤波器的停止带最小衰减（dB）
    rp : float, optional
        带通滤波器的最大通带允许波纹（dB）
    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        滤波器的类型：低通、高通、带通或带阻
    analog : bool, optional
        指定是否设计模拟滤波器，默认为 True
    output : {'ba', 'zpk', 'sos'}, optional
        指定输出的滤波器表示形式：系数（'ba'）、零极点（'zpk'）、二阶段（'sos'）
    fs : float, optional
        数字滤波器的采样频率（Hz），仅在设计数字滤波器时使用
    ftype : str, optional
        滤波器的类型，这里是 'elliptic' 表示椭圆滤波器

    Returns
    -------
    iirfilter : ndarray or tuple
        根据指定参数生成的 IIR 滤波器的系数或二阶段描述（根据输出参数的不同而定）

    Notes
    -----
    - `'sos'` 输出参数是在 0.16.0 版本中新增的。
    - 这段代码的示例演示了如何使用 scipy 中的信号处理模块设计和应用椭圆滤波器，并绘制其频率响应。

    Examples
    --------
    设计一个模拟低通椭圆滤波器并绘制其频率响应，显示关键点：

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> b, a = signal.ellip(4, 5, 40, 100, 'low', analog=True)
    >>> w, h = signal.freqs(b, a)
    >>> plt.semilogx(w, 20 * np.log10(abs(h)))
    >>> plt.title('Elliptic filter frequency response (rp=5, rs=40)')
    >>> plt.xlabel('Frequency [radians / second]')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.margins(0, 0.1)
    >>> plt.grid(which='both', axis='both')
    >>> plt.axvline(100, color='green') # 截止频率
    >>> plt.axhline(-40, color='green') # rs
    >>> plt.axhline(-5, color='green') # rp
    >>> plt.show()

    生成一个由 10 Hz 和 20 Hz 组成的信号，采样频率为 1 kHz：

    >>> t = np.linspace(0, 1, 1000, False)  # 1 秒
    >>> sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)
    >>> fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    >>> ax1.plot(t, sig)
    >>> ax1.set_title('10 Hz and 20 Hz sinusoids')
    >>> ax1.axis([0, 1, -2, 2])

    设计一个数字高通滤波器，在 17 Hz 处移除 10 Hz 信号，并将其应用于信号。（推荐使用二阶段格式以避免传递函数格式的数值误差）：

    >>> sos = signal.ellip(8, 1, 100, 17, 'hp', fs=1000, output='sos')
    >>> filtered = signal.sosfilt(sos, sig)
    >>> ax2.plot(t, filtered)
    >>> ax2.set_title('After 17 Hz high-pass filter')
    >>> ax2.axis([0, 1, -2, 2])
    >>> ax2.set_xlabel('Time [seconds]')
    >>> plt.tight_layout()
    >>> plt.show()
    """
    return iirfilter(N, Wn, rs=rs, rp=rp, btype=btype, analog=analog,
                     output=output, ftype='elliptic', fs=fs)
# 定义函数bessel，用于设计Bessel/Thomson数字和模拟滤波器
def bessel(N, Wn, btype='low', analog=False, output='ba', norm='phase',
           fs=None):
    """
    Bessel/Thomson digital and analog filter design.

    Design an Nth-order digital or analog Bessel filter and return the
    filter coefficients.

    Parameters
    ----------
    N : int
        滤波器的阶数。
    Wn : array_like
        临界频率，可以是标量或长度为2的序列，根据 `norm` 参数定义。
        对于模拟滤波器，`Wn` 是一个角频率（例如，rad/s）。
        对于数字滤波器，`Wn` 的单位与 `fs` 相同。默认情况下，`fs` 是每样本2个半周期，因此这些频率被归一化为0到1的范围，
        其中1是奈奎斯特频率。 (`Wn` 因此是半周期/样本。)
    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        滤波器类型。默认为 'lowpass'。
    analog : bool, optional
        当为True时返回模拟滤波器，否则返回数字滤波器。（参见注释。）
    output : {'ba', 'zpk', 'sos'}, optional
        输出类型：分子/分母（'ba'），极点/零点（'zpk'），或二阶段（'sos'）。默认为 'ba'。
    norm : {'phase', 'delay', 'mag'}, optional
        临界频率归一化：

        ``phase``
            滤波器归一化，使得在角频率 `Wn` 处相位响应达到其中点。这适用于低通和高通滤波器，因此这是“相位匹配”情况。

            幅度响应渐近线和具有相同的布特沃斯滤波器阶数，在截止频率为 `Wn` 处。

            这是默认值，与MATLAB的实现相匹配。

        ``delay``
            滤波器归一化，使得通带中的群延迟为1/ `Wn`（例如，秒）。这是通过解Bessel多项式得到的“自然”类型。

        ``mag``
            滤波器归一化，使得在角频率 `Wn` 处增益幅度为-3 dB。

        .. versionadded:: 0.18.0
    fs : float, optional
        数字系统的采样频率。

        .. versionadded:: 1.2.0

    Returns
    -------
    b, a : ndarray, ndarray
        IIR滤波器的分子（`b`）和分母（`a`）多项式。
        仅在 ``output='ba'`` 时返回。
    z, p, k : ndarray, ndarray, float
        IIR滤波器传递函数的零点、极点和系统增益。
        仅在 ``output='zpk'`` 时返回。
    sos : ndarray
        IIR滤波器的二阶段表示。
        仅在 ``output='sos'`` 时返回。

    Notes
    -----
    Bessel滤波器，也称为汤姆逊滤波器，具有最大平坦的群延迟和最大线性的相位响应，步跃响应中几乎没有振铃现象。 [1]_

    """
    The Bessel is inherently an analog filter. This function generates digital
    Bessel filters using the bilinear transform, which does not preserve the
    phase response of the analog filter. As such, it is only approximately
    correct at frequencies below about fs/4. To get maximally-flat group
    delay at higher frequencies, the analog Bessel filter must be transformed
    using phase-preserving techniques.

    See `besselap` for implementation details and references.

    The ``'sos'`` output parameter was added in 0.16.0.

    References
    ----------
    .. [1] Thomson, W.E., "Delay Networks having Maximally Flat Frequency
           Characteristics", Proceedings of the Institution of Electrical
           Engineers, Part III, November 1949, Vol. 96, No. 44, pp. 487-490.

    Examples
    --------
    Plot the phase-normalized frequency response, showing the relationship
    to the Butterworth's cutoff frequency (green):

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> b, a = signal.butter(4, 100, 'low', analog=True)  # Generate a 4th-order low-pass Butterworth analog filter
    >>> w, h = signal.freqs(b, a)  # Compute the frequency response of the filter
    >>> plt.semilogx(w, 20 * np.log10(np.abs(h)), color='silver', ls='dashed')  # Plot the magnitude response in dB
    >>> b, a = signal.bessel(4, 100, 'low', analog=True, norm='phase')  # Generate a 4th-order low-pass analog Bessel filter with phase normalization
    >>> w, h = signal.freqs(b, a)  # Compute the frequency response of the Bessel filter
    >>> plt.semilogx(w, 20 * np.log10(np.abs(h)))  # Plot the magnitude response in dB
    >>> plt.title('Bessel filter magnitude response (with Butterworth)')  # Set plot title
    >>> plt.xlabel('Frequency [radians / second]')  # Set x-axis label
    >>> plt.ylabel('Amplitude [dB]')  # Set y-axis label
    >>> plt.margins(0, 0.1)  # Set plot margins
    >>> plt.grid(which='both', axis='both')  # Enable grid
    >>> plt.axvline(100, color='green')  # Add vertical line at cutoff frequency
    >>> plt.show()  # Display the plot

    and the phase midpoint:

    >>> plt.figure()
    >>> plt.semilogx(w, np.unwrap(np.angle(h)))  # Plot the phase response, unwrapped
    >>> plt.axvline(100, color='green')  # Add vertical line at cutoff frequency
    >>> plt.axhline(-np.pi, color='red')  # Add horizontal line at phase midpoint
    >>> plt.title('Bessel filter phase response')  # Set plot title
    >>> plt.xlabel('Frequency [radians / second]')  # Set x-axis label
    >>> plt.ylabel('Phase [radians]')  # Set y-axis label
    >>> plt.margins(0, 0.1)  # Set plot margins
    >>> plt.grid(which='both', axis='both')  # Enable grid
    >>> plt.show()  # Display the plot

    Plot the magnitude-normalized frequency response, showing the -3 dB cutoff:

    >>> b, a = signal.bessel(3, 10, 'low', analog=True, norm='mag')  # Generate a 3rd-order low-pass analog Bessel filter with magnitude normalization
    >>> w, h = signal.freqs(b, a)  # Compute the frequency response of the filter
    >>> plt.semilogx(w, 20 * np.log10(np.abs(h)))  # Plot the magnitude response in dB
    >>> plt.axhline(-3, color='red')  # Add horizontal line at -3 dB magnitude
    >>> plt.axvline(10, color='green')  # Add vertical line at cutoff frequency
    >>> plt.title('Magnitude-normalized Bessel filter frequency response')  # Set plot title
    >>> plt.xlabel('Frequency [radians / second]')  # Set x-axis label
    >>> plt.ylabel('Amplitude [dB]')  # Set y-axis label
    >>> plt.margins(0, 0.1)  # Set plot margins
    >>> plt.grid(which='both', axis='both')  # Enable grid
    >>> plt.show()  # Display the plot

    Plot the delay-normalized filter, showing the maximally-flat group delay
    at 0.1 seconds:

    >>> b, a = signal.bessel(5, 1/0.1, 'low', analog=True, norm='delay')  # Generate a 5th-order low-pass analog Bessel filter with delay normalization
    >>> w, h = signal.freqs(b, a)  # Compute the frequency response of the filter
    >>> plt.figure()
    # 绘制半对数坐标系中的信号相位非连续性角度差除以频率差，用于显示系统的群延迟
    >>> plt.semilogx(w[1:], -np.diff(np.unwrap(np.angle(h)))/np.diff(w)
    
    # 在图上添加一条红色水平线，表示0.1秒的群延迟
    >>> plt.axhline(0.1, color='red')  # 0.1 seconds group delay
    
    # 设置图的标题为“Bessel filter group delay”
    >>> plt.title('Bessel filter group delay')
    
    # 设置图的横轴标签为“Frequency [radians / second]”
    >>> plt.xlabel('Frequency [radians / second]')
    
    # 设置图的纵轴标签为“Group delay [seconds]”
    >>> plt.ylabel('Group delay [seconds]')
    
    # 设置图的边距，使得图的内容不紧贴坐标轴
    >>> plt.margins(0, 0.1)
    
    # 显示图，包括网格线，网格线显示在图的所有轴上
    >>> plt.grid(which='both', axis='both')
    
    # 显示绘制好的图形
    >>> plt.show()
    
    
    
    # 使用指定的参数创建一个Bessel滤波器
    return iirfilter(N, Wn, btype=btype, analog=analog,
                     output=output, ftype='bessel_'+norm, fs=fs)
# 定义一个空函数 `maxflat()`，暂时没有实现任何功能
def maxflat():
    pass

# 定义一个空函数 `yulewalk()`，暂时没有实现任何功能
def yulewalk():
    pass

# 带阻滤波器的优化目标函数，用于最小化阶数

"""
Returns the non-integer order for an analog band stop filter.

Parameters
----------
wp : scalar
    通带边缘 `passb` 的边缘。
ind : int, {0, 1}
    指定要变化的 `passb` 边缘的索引 (0 或 1)。
passb : ndarray
    两个固定通带边缘的序列。
stopb : ndarray
    两个固定阻带边缘的序列。
gstop : float
    阻带中的衰减量（以分贝为单位）。
gpass : float
    通带中的波动量（以分贝为单位）。
type : {'butter', 'cheby', 'ellip'}
    滤波器类型。

Returns
-------
n : scalar
    滤波器阶数（可能是非整数）。
"""
def band_stop_obj(wp, ind, passb, stopb, gpass, gstop, type):

    # 验证 gpass 和 gstop 的有效性
    _validate_gpass_gstop(gpass, gstop)

    # 复制 passb 数组，以便修改其中的一个边缘值
    passbC = passb.copy()
    passbC[ind] = wp

    # 计算自然频率 nat
    nat = (stopb * (passbC[0] - passbC[1]) /
           (stopb ** 2 - passbC[0] * passbC[1]))
    nat = min(abs(nat))

    if type == 'butter':
        GSTOP = 10 ** (0.1 * abs(gstop))
        GPASS = 10 ** (0.1 * abs(gpass))
        n = (log10((GSTOP - 1.0) / (GPASS - 1.0)) / (2 * log10(nat)))
    elif type == 'cheby':
        GSTOP = 10 ** (0.1 * abs(gstop))
        GPASS = 10 ** (0.1 * abs(gpass))
        n = arccosh(sqrt((GSTOP - 1.0) / (GPASS - 1.0))) / arccosh(nat)
    elif type == 'ellip':
        GSTOP = 10 ** (0.1 * gstop)
        GPASS = 10 ** (0.1 * gpass)
        arg1 = sqrt((GPASS - 1.0) / (GSTOP - 1.0))
        arg0 = 1.0 / nat
        d0 = special.ellipk([arg0 ** 2, 1 - arg0 ** 2])
        d1 = special.ellipk([arg1 ** 2, 1 - arg1 ** 2])
        n = (d0[0] * d1[1] / (d0[1] * d1[0]))
    else:
        raise ValueError("Incorrect type: %s" % type)

    return n


# 数字滤波器设计前预处理频率
def _pre_warp(wp, ws, analog):
    # 数字滤波器设计前预处理频率
    if not analog:
        passb = np.tan(pi * wp / 2.0)
        stopb = np.tan(pi * ws / 2.0)
    else:
        passb = wp * 1.0
        stopb = ws * 1.0
    return passb, stopb


# 验证数字滤波器设计中的 wp 和 ws 参数
def _validate_wp_ws(wp, ws, fs, analog):
    wp = atleast_1d(wp)
    ws = atleast_1d(ws)
    if fs is not None:
        if analog:
            raise ValueError("fs cannot be specified for an analog filter")
        wp = 2 * wp / fs
        ws = 2 * ws / fs

    filter_type = 2 * (len(wp) - 1) + 1
    if wp[0] >= ws[0]:
        filter_type += 1

    return wp, ws, filter_type


# 查找自然频率函数
def _find_nat_freq(stopb, passb, gpass, gstop, filter_type, filter_kind):
    if filter_type == 1:            # 低通
        nat = stopb / passb
    elif filter_type == 2:          # 高通
        nat = passb / stopb
    elif filter_type == 3:          # 如果滤波器类型为3（停止带）

        ### breakpoint()

        # 使用 optimize.fminbound 函数找到最小化 band_stop_obj 函数的参数 wp0
        wp0 = optimize.fminbound(band_stop_obj, passb[0], stopb[0] - 1e-12,
                                 args=(0, passb, stopb, gpass, gstop,
                                       filter_kind),
                                 disp=0)
        # 更新 passb[0] 的值为 wp0，以便进一步优化
        passb[0] = wp0

        # 使用 optimize.fminbound 函数找到最小化 band_stop_obj 函数的参数 wp1
        wp1 = optimize.fminbound(band_stop_obj, stopb[1] + 1e-12, passb[1],
                                 args=(1, passb, stopb, gpass, gstop,
                                       filter_kind),
                                 disp=0)
        # 更新 passb[1] 的值为 wp1，以便进一步优化
        passb[1] = wp1

        # 计算自然频率 nat
        nat = ((stopb * (passb[0] - passb[1])) /
               (stopb ** 2 - passb[0] * passb[1]))

    elif filter_type == 4:          # 如果滤波器类型为4（通过带）

        # 计算自然频率 nat
        nat = ((stopb ** 2 - passb[0] * passb[1]) /
               (stopb * (passb[0] - passb[1])))

    else:
        # 抛出 ValueError 异常，因为不应该出现未定义的 filter_type 值
        raise ValueError(f"should not happen: {filter_type =}.")

    # 取自然频率 nat 的绝对值的最小值
    nat = min(abs(nat))
    # 返回自然频率 nat 和更新后的 passb 数组
    return nat, passb
def _postprocess_wn(WN, analog, fs):
    # 根据 analong 参数决定是否对 WN 进行处理，如果是数字滤波器则直接使用 WN，如果是模拟滤波器则计算其正切函数
    wn = WN if analog else np.arctan(WN) * 2.0 / pi
    # 如果 wn 是长度为 1 的数组，则将其转换为标量
    if len(wn) == 1:
        wn = wn[0]
    # 如果提供了采样频率 fs，则对 wn 进行归一化处理
    if fs is not None:
        wn = wn * fs / 2
    return wn

# Butterworth 滤波器阶数选择函数
def buttord(wp, ws, gpass, gstop, analog=False, fs=None):
    """Butterworth filter order selection.

    Return the order of the lowest order digital or analog Butterworth filter
    that loses no more than `gpass` dB in the passband and has at least
    `gstop` dB attenuation in the stopband.

    Parameters
    ----------
    wp, ws : float
        Passband and stopband edge frequencies.

        For digital filters, these are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`wp` and `ws` are thus in
        half-cycles / sample.) For example:

            - Lowpass:   wp = 0.2,          ws = 0.3
            - Highpass:  wp = 0.3,          ws = 0.2
            - Bandpass:  wp = [0.2, 0.5],   ws = [0.1, 0.6]
            - Bandstop:  wp = [0.1, 0.6],   ws = [0.2, 0.5]

        For analog filters, `wp` and `ws` are angular frequencies (e.g., rad/s).
    gpass : float
        The maximum loss in the passband (dB).
    gstop : float
        The minimum attenuation in the stopband (dB).
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    ord : int
        The lowest order for a Butterworth filter which meets specs.
    wn : ndarray or float
        The Butterworth natural frequency (i.e. the "3dB frequency"). Should
        be used with `butter` to give filter results. If `fs` is specified,
        this is in the same units, and `fs` must also be passed to `butter`.

    See Also
    --------
    butter : Filter design using order and critical points
    cheb1ord : Find order and critical points from passband and stopband spec
    cheb2ord, ellipord
    iirfilter : General filter design using order and critical frequencies
    iirdesign : General filter design using passband and stopband spec

    Examples
    --------
    Design an analog bandpass filter with passband within 3 dB from 20 to
    50 rad/s, while rejecting at least -40 dB below 14 and above 60 rad/s.
    Plot its frequency response, showing the passband and stopband
    constraints in gray.

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> N, Wn = signal.buttord([20, 50], [14, 60], 3, 40, True)
    >>> b, a = signal.butter(N, Wn, 'band', True)
    >>> w, h = signal.freqs(b, a, np.logspace(1, 2, 500))
    >>> plt.semilogx(w, 20 * np.log10(abs(h)))
    >>> plt.title('Butterworth bandpass filter fit to constraints')
    >>> plt.xlabel('Frequency [radians / second]')
    >>> plt.ylabel('Amplitude [dB]')
    # 设置网格显示，包括主刻度和次刻度
    >>> plt.grid(which='both', axis='both')
    
    # 绘制填充区域，表示停止带
    >>> plt.fill([1,  14,  14,   1], [-40, -40, 99, 99], '0.9', lw=0) # stop
    
    # 绘制填充区域，表示通过带
    >>> plt.fill([20, 20,  50,  50], [-99, -3, -3, -99], '0.9', lw=0) # pass
    
    # 绘制填充区域，表示停止带
    >>> plt.fill([60, 60, 1e9, 1e9], [99, -40, -40, 99], '0.9', lw=0) # stop
    
    # 设置图形的坐标轴范围
    >>> plt.axis([10, 100, -60, 3])
    
    # 显示绘制的图形
    >>> plt.show()
    
    """
    # 验证 gpass 和 gstop 的有效性
    _validate_gpass_gstop(gpass, gstop)
    
    # 验证 fs 的有效性，并允许其为 None
    fs = _validate_fs(fs, allow_none=True)
    
    # 验证并处理 wp, ws，返回滤波器的截止频率和阻止带频率以及滤波器类型
    wp, ws, filter_type = _validate_wp_ws(wp, ws, fs, analog)
    
    # 预处理，对 wp 和 ws 进行预变换
    passb, stopb = _pre_warp(wp, ws, analog)
    
    # 查找自然频率，返回自然频率和处理后的阻止带频率
    nat, passb = _find_nat_freq(stopb, passb, gpass, gstop, filter_type, 'butter')
    
    # 计算 gstop 对应的值
    GSTOP = 10 ** (0.1 * abs(gstop))
    
    # 计算 gpass 对应的值
    GPASS = 10 ** (0.1 * abs(gpass))
    
    # 计算滤波器的阶数
    ord = int(ceil(log10((GSTOP - 1.0) / (GPASS - 1.0)) / (2 * log10(nat))))
    
    # 计算 Butterworth 滤波器的自然频率 WN（或者称为 "3dB" 频率）
    try:
        W0 = (GPASS - 1.0) ** (-1.0 / (2.0 * ord))
    except ZeroDivisionError:
        W0 = 1.0
        # 若阶数为零，则发出警告
        warnings.warn("Order is zero...check input parameters.",
                      RuntimeWarning, stacklevel=2)
    
    # 将此频率从低通原型转换回到原始的模拟滤波器
    if filter_type == 1:  # 低通滤波器
        WN = W0 * passb
    elif filter_type == 2:  # 高通滤波器
        WN = passb / W0
    elif filter_type == 3:  # 阻止带滤波器
        WN = np.empty(2, float)
        discr = sqrt((passb[1] - passb[0]) ** 2 +
                     4 * W0 ** 2 * passb[0] * passb[1])
        WN[0] = ((passb[1] - passb[0]) + discr) / (2 * W0)
        WN[1] = ((passb[1] - passb[0]) - discr) / (2 * W0)
        WN = np.sort(abs(WN))
    elif filter_type == 4:  # 通过带滤波器
        W0 = np.array([-W0, W0], float)
        WN = (-W0 * (passb[1] - passb[0]) / 2.0 +
              sqrt(W0 ** 2 / 4.0 * (passb[1] - passb[0]) ** 2 +
                   passb[0] * passb[1]))
        WN = np.sort(abs(WN))
    else:
        raise ValueError("Bad type: %s" % filter_type)
    
    # 后处理 WN，根据模拟或数字滤波器以及采样率处理频率
    wn = _postprocess_wn(WN, analog, fs)
    
    # 返回计算得到的滤波器阶数和频率
    return ord, wn
# Chebyshev Type I 滤波器阶数选择函数。

Return the order of the lowest order digital or analog Chebyshev Type I
filter that loses no more than `gpass` dB in the passband and has at
least `gstop` dB attenuation in the stopband.

Parameters
----------
wp, ws : float
    Passband and stopband edge frequencies.

    For digital filters, these are in the same units as `fs`. By default,
    `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
    where 1 is the Nyquist frequency. (`wp` and `ws` are thus in
    half-cycles / sample.)  For example:

        - Lowpass:   wp = 0.2,          ws = 0.3
        - Highpass:  wp = 0.3,          ws = 0.2
        - Bandpass:  wp = [0.2, 0.5],   ws = [0.1, 0.6]
        - Bandstop:  wp = [0.1, 0.6],   ws = [0.2, 0.5]

    For analog filters, `wp` and `ws` are angular frequencies (e.g., rad/s).
gpass : float
    The maximum loss in the passband (dB).
gstop : float
    The minimum attenuation in the stopband (dB).
analog : bool, optional
    When True, return an analog filter, otherwise a digital filter is
    returned.
fs : float, optional
    The sampling frequency of the digital system.

    .. versionadded:: 1.2.0

Returns
-------
ord : int
    The lowest order for a Chebyshev type I filter that meets specs.
wn : ndarray or float
    The Chebyshev natural frequency (the "3dB frequency") for use with
    `cheby1` to give filter results. If `fs` is specified,
    this is in the same units, and `fs` must also be passed to `cheby1`.

See Also
--------
cheby1 : Filter design using order and critical points
buttord : Find order and critical points from passband and stopband spec
cheb2ord, ellipord
iirfilter : General filter design using order and critical frequencies
iirdesign : General filter design using passband and stopband spec

Examples
--------
Design a digital lowpass filter such that the passband is within 3 dB up
to 0.2*(fs/2), while rejecting at least -40 dB above 0.3*(fs/2). Plot its
frequency response, showing the passband and stopband constraints in gray.

>>> from scipy import signal
>>> import matplotlib.pyplot as plt
>>> import numpy as np

>>> N, Wn = signal.cheb1ord(0.2, 0.3, 3, 40)
>>> b, a = signal.cheby1(N, 3, Wn, 'low')
>>> w, h = signal.freqz(b, a)
>>> plt.semilogx(w / np.pi, 20 * np.log10(abs(h)))
>>> plt.title('Chebyshev I lowpass filter fit to constraints')
>>> plt.xlabel('Normalized frequency')
>>> plt.ylabel('Amplitude [dB]')
>>> plt.grid(which='both', axis='both')
>>> plt.fill([.01, 0.2, 0.2, .01], [-3, -3, -99, -99], '0.9', lw=0) # stop
>>> plt.fill([0.3, 0.3,   2,   2], [ 9, -40, -40,  9], '0.9', lw=0) # pass
>>> plt.axis([0.08, 1, -60, 3])
>>> plt.show()
    # 确保文件系统有效性，并允许其为 None
    fs = _validate_fs(fs, allow_none=True)
    # 验证通带增益和阻带增益的有效性
    _validate_gpass_gstop(gpass, gstop)
    # 验证截止频率和通带宽度，并返回处理后的参数
    wp, ws, filter_type = _validate_wp_ws(wp, ws, fs, analog)
    # 根据截止频率和通带宽度进行预变形
    passb, stopb = _pre_warp(wp, ws, analog)
    # 查找自然频率
    nat, passb = _find_nat_freq(stopb, passb, gpass, gstop, filter_type, 'cheby')

    # 计算阻带增益和通带增益的幅度值
    GSTOP = 10 ** (0.1 * abs(gstop))
    GPASS = 10 ** (0.1 * abs(gpass))
    # 计算过渡区的超伽马值
    v_pass_stop = np.arccosh(np.sqrt((GSTOP - 1.0) / (GPASS - 1.0)))
    # 计算滤波器的阶数
    ord = int(ceil(v_pass_stop / np.arccosh(nat)))

    # 后处理自然频率，如果是模拟滤波器则考虑采样频率
    wn = _postprocess_wn(passb, analog, fs)

    # 返回滤波器的阶数和归一化频率
    return ord, wn
# Chebyshev Type II 滤波器阶数选择函数
def cheb2ord(wp, ws, gpass, gstop, analog=False, fs=None):
    """Chebyshev type II filter order selection.

    返回一个数字或模拟 Chebyshev Type II 滤波器的最低阶数，
    在通带最多损失 `gpass` dB 并且在阻带至少有 `gstop` dB 的衰减。

    Parameters
    ----------
    wp, ws : float
        通带和阻带的边缘频率。

        对于数字滤波器，这些频率与 `fs` 的单位相同。默认情况下，
        `fs` 是每样本 2 个半周期，因此这些频率标准化为 0 到 1，
        其中 1 是奈奎斯特频率。(`wp` 和 `ws` 因此是半周期 / 样本。) 例如：

            - 低通滤波器：   wp = 0.2,          ws = 0.3
            - 高通滤波器：  wp = 0.3,          ws = 0.2
            - 带通滤波器：  wp = [0.2, 0.5],   ws = [0.1, 0.6]
            - 带阻滤波器：  wp = [0.1, 0.6],   ws = [0.2, 0.5]

        对于模拟滤波器，`wp` 和 `ws` 是角频率 (例如，弧度/秒)。
    gpass : float
        通带最大损失 (dB)。
    gstop : float
        阻带最小衰减 (dB)。
    analog : bool, optional
        当为 True 时，返回模拟滤波器，否则返回数字滤波器。
    fs : float, optional
        数字系统的采样频率。

        .. versionadded:: 1.2.0

    Returns
    -------
    ord : int
        满足规格要求的 Chebyshev Type II 滤波器的最低阶数。
    wn : ndarray or float
        Chebyshev 自然频率（“3dB 频率”），用于 `cheby2` 函数以产生滤波结果。
        如果指定了 `fs`，则以相同单位表示，同时必须将 `fs` 传递给 `cheby2` 函数。

    See Also
    --------
    cheby2 : 使用阶数和临界点进行滤波器设计
    buttord : 从通带和阻带规格中查找阶数和临界点
    cheb1ord, ellipord
    iirfilter : 使用阶数和临界频率进行一般滤波器设计
    iirdesign : 使用通带和阻带规格进行一般滤波器设计

    Examples
    --------
    设计一个数字带阻滤波器，其在从 0.2*(fs/2) 到 0.5*(fs/2) 之间拒绝 -60 dB 的信号，
    同时在 0.1*(fs/2) 以下或 0.6*(fs/2) 以上保持在 3 dB 以下。绘制其频率响应，
    并在灰色中显示通带和阻带约束。

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> N, Wn = signal.cheb2ord([0.1, 0.6], [0.2, 0.5], 3, 60)
    >>> b, a = signal.cheby2(N, 60, Wn, 'stop')
    >>> w, h = signal.freqz(b, a)
    >>> plt.semilogx(w / np.pi, 20 * np.log10(abs(h)))
    >>> plt.title('Chebyshev II bandstop filter fit to constraints')
    >>> plt.xlabel('Normalized frequency')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.grid(which='both', axis='both')
    >>> plt.fill([.01, .1, .1, .01], [-3,  -3, -99, -99], '0.9', lw=0) # 阻带
    >>> plt.fill([.2,  .2, .5,  .5], [ 9, -60, -60,   9], '0.9', lw=0) # 通带
    fs = _validate_fs(fs, allow_none=True)  # 调用函数 _validate_fs 验证采样频率 fs 的有效性，允许 None 值
    _validate_gpass_gstop(gpass, gstop)     # 调用函数 _validate_gpass_gstop 验证通带和阻带的幅度 gpass, gstop 的有效性
    wp, ws, filter_type = _validate_wp_ws(wp, ws, fs, analog)  # 调用函数 _validate_wp_ws 验证角频率 wp, ws 和滤波器类型 filter_type 的有效性
    passb, stopb = _pre_warp(wp, ws, analog)  # 调用函数 _pre_warp 对角频率进行预变换，返回预变换后的通带和阻带
    nat, passb = _find_nat_freq(stopb, passb, gpass, gstop, filter_type, 'cheby')  # 调用函数 _find_nat_freq 寻找自然频率 nat 和经过预变换后的通带 passb

    GSTOP = 10 ** (0.1 * abs(gstop))  # 计算阻带的增益
    GPASS = 10 ** (0.1 * abs(gpass))  # 计算通带的增益
    v_pass_stop = np.arccosh(np.sqrt((GSTOP - 1.0) / (GPASS - 1.0)))  # 计算传递带宽和阻带宽之间的参数
    ord = int(ceil(v_pass_stop / arccosh(nat)))  # 计算滤波器的阶数

    # 找到模拟响应为 -gpass dB 的频率。
    # 然后从低通原型转换回原始滤波器。

    new_freq = cosh(1.0 / ord * v_pass_stop)  # 计算新的频率
    new_freq = 1.0 / new_freq  # 更新新的频率

    if filter_type == 1:
        nat = passb / new_freq  # 更新自然频率，根据滤波器类型
    elif filter_type == 2:
        nat = passb * new_freq  # 更新自然频率，根据滤波器类型
    elif filter_type == 3:
        nat = np.empty(2, float)  # 创建一个空的数组，存储两个浮点数
        nat[0] = (new_freq / 2.0 * (passb[0] - passb[1]) +
                  sqrt(new_freq ** 2 * (passb[1] - passb[0]) ** 2 / 4.0 +
                       passb[1] * passb[0]))  # 更新自然频率，根据滤波器类型
        nat[1] = passb[1] * passb[0] / nat[0]  # 更新自然频率，根据滤波器类型
    elif filter_type == 4:
        nat = np.empty(2, float)  # 创建一个空的数组，存储两个浮点数
        nat[0] = (1.0 / (2.0 * new_freq) * (passb[0] - passb[1]) +
                  sqrt((passb[1] - passb[0]) ** 2 / (4.0 * new_freq ** 2) +
                       passb[1] * passb[0]))  # 更新自然频率，根据滤波器类型
        nat[1] = passb[0] * passb[1] / nat[0]  # 更新自然频率，根据滤波器类型

    wn = _postprocess_wn(nat, analog, fs)  # 调用函数 _postprocess_wn 对自然频率进行后处理，返回数字滤波器的角频率

    return ord, wn  # 返回滤波器的阶数和数字滤波器的角频率
# 计算常数 _POW10_LOG10，其值为以 10 为底的自然对数
_POW10_LOG10 = np.log(10)

# 定义函数 _pow10m1，计算近似于 10 ** x - 1 的值，当 x 接近 0 时
def _pow10m1(x):
    return np.expm1(_POW10_LOG10 * x)

# 定义函数 ellipord，用于选择椭圆（Cauer）滤波器的阶数

"""Elliptic (Cauer) filter order selection.

Return the order of the lowest order digital or analog elliptic filter
that loses no more than `gpass` dB in the passband and has at least
`gstop` dB attenuation in the stopband.

Parameters
----------
wp, ws : float
    Passband and stopband edge frequencies.

    For digital filters, these are in the same units as `fs`. By default,
    `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
    where 1 is the Nyquist frequency. (`wp` and `ws` are thus in
    half-cycles / sample.) For example:

        - Lowpass:   wp = 0.2,          ws = 0.3
        - Highpass:  wp = 0.3,          ws = 0.2
        - Bandpass:  wp = [0.2, 0.5],   ws = [0.1, 0.6]
        - Bandstop:  wp = [0.1, 0.6],   ws = [0.2, 0.5]

    For analog filters, `wp` and `ws` are angular frequencies (e.g., rad/s).
gpass : float
    The maximum loss in the passband (dB).
gstop : float
    The minimum attenuation in the stopband (dB).
analog : bool, optional
    When True, return an analog filter, otherwise a digital filter is
    returned.
fs : float, optional
    The sampling frequency of the digital system.

    .. versionadded:: 1.2.0

Returns
-------
ord : int
    The lowest order for an Elliptic (Cauer) filter that meets specs.
wn : ndarray or float
    The Chebyshev natural frequency (the "3dB frequency") for use with
    `ellip` to give filter results. If `fs` is specified,
    this is in the same units, and `fs` must also be passed to `ellip`.

See Also
--------
ellip : Filter design using order and critical points
buttord : Find order and critical points from passband and stopband spec
cheb1ord, cheb2ord
iirfilter : General filter design using order and critical frequencies
iirdesign : General filter design using passband and stopband spec

Examples
--------
Design an analog highpass filter such that the passband is within 3 dB
above 30 rad/s, while rejecting -60 dB at 10 rad/s. Plot its
frequency response, showing the passband and stopband constraints in gray.

>>> from scipy import signal
>>> import matplotlib.pyplot as plt
>>> import numpy as np

>>> N, Wn = signal.ellipord(30, 10, 3, 60, True)
>>> b, a = signal.ellip(N, 3, 60, Wn, 'high', True)
>>> w, h = signal.freqs(b, a, np.logspace(0, 3, 500))
>>> plt.semilogx(w, 20 * np.log10(abs(h)))
>>> plt.title('Elliptical highpass filter fit to constraints')
>>> plt.xlabel('Frequency [radians / second]')
>>> plt.ylabel('Amplitude [dB]')
>>> plt.grid(which='both', axis='both')
>>> plt.fill([.1, 10,  10,  .1], [1e4, 1e4, -60, -60], '0.9', lw=0) # stop
"""
def ellipord(wp, ws, gpass, gstop, analog=False, fs=None):
    # 实现省略，根据参数计算滤波器的阶数
    pass
    # 调用 matplotlib 的 fill 函数绘制填充多边形，定义了四个顶点坐标
    >>> plt.fill([30, 30, 1e9, 1e9], [-99, -3, -3, -99], '0.9', lw=0) # pass
    
    # 设置当前图的坐标轴范围
    >>> plt.axis([1, 300, -80, 3])
    
    # 显示当前图形
    >>> plt.show()
    
    """
    # 对输入的采样频率进行验证，并允许其为空
    fs = _validate_fs(fs, allow_none=True)
    
    # 验证通带和阻带的参数 gpass 和 gstop
    _validate_gpass_gstop(gpass, gstop)
    
    # 验证和计算截止频率 wp、ws，以及滤波器类型 filter_type
    wp, ws, filter_type = _validate_wp_ws(wp, ws, fs, analog)
    
    # 对截止频率进行预处理和变换，返回预处理后的通带和阻带
    passb, stopb = _pre_warp(wp, ws, analog)
    
    # 查找自然频率 nat，更新通带 passb
    nat, passb = _find_nat_freq(stopb, passb, gpass, gstop, filter_type, 'ellip')
    
    # 计算参数 arg1_sq，用于特定函数调用
    arg1_sq = _pow10m1(0.1 * gpass) / _pow10m1(0.1 * gstop)
    
    # 计算参数 arg0
    arg0 = 1.0 / nat
    
    # 计算椭圆函数值 d0 和 d1
    d0 = special.ellipk(arg0 ** 2), special.ellipkm1(arg0 ** 2)
    d1 = special.ellipk(arg1_sq), special.ellipkm1(arg1_sq)
    
    # 计算滤波器的阶数 ord
    ord = int(ceil(d0[0] * d1[1] / (d0[1] * d1[0])))
    
    # 后处理计算得到的自然频率 wn
    wn = _postprocess_wn(passb, analog, fs)
    
    # 返回滤波器的阶数 ord 和后处理的自然频率 wn
    return ord, wn
# Solve the degree equation using nomes to determine the maximum number of terms
def _ellipdeg(n, m1):
    """
    Solve degree equation using nomes to determine the maximum number of terms

    Parameters
    ----------
    n : int
        Degree of the elliptic function
    m1 : float
        Parameter related to the elliptic function

    Returns
    -------
    ndarray
        Array containing the solutions to the degree equation
    """
    Given n, m1, solve
       n * K(m) / K'(m) = K1(m1) / K1'(m1)
    for m

    See [1], Eq. (49)

    References
    ----------
    .. [1] Orfanidis, "Lecture Notes on Elliptic Filter Design",
           https://www.ece.rutgers.edu/~orfanidi/ece521/notes.pdf
    """
    # 计算参数 m1 对应的第一类完全椭圆积分 K1(m1)
    K1 = special.ellipk(m1)
    # 计算参数 m1 对应的第一类完全椭圆积分的导数 K1'(m1)
    K1p = special.ellipkm1(m1)

    # 计算 q1 = exp(-π * K1'(m1) / K1(m1))
    q1 = np.exp(-np.pi * K1p / K1)
    # 计算 q = q1^(1/n)
    q = q1 ** (1/n)

    # 创建一个数组，包含整数序列从 0 到 _ELLIPDEG_MMAX
    mnum = np.arange(_ELLIPDEG_MMAX + 1)
    # 创建一个数组，包含整数序列从 1 到 _ELLIPDEG_MMAX + 1
    mden = np.arange(1, _ELLIPDEG_MMAX + 2)

    # 计算 num = Σ(q^(mnum * (mnum + 1)))
    num = np.sum(q ** (mnum * (mnum+1)))
    # 计算 den = 1 + 2 * Σ(q^(mden^2))
    den = 1 + 2 * np.sum(q ** (mden**2))

    # 返回 Elliptic Filter Design 中的结果
    return 16 * q * (num / den) ** 4
# Landen变换递归序列中的最大迭代次数。10是保守的选择；单元测试通过4，Orfanidis建议使用5。
_ARC_JAC_SN_MAXITER = 10

# 计算反Jacobean椭圆函数sn的逆函数
def _arc_jac_sn(w, m):
    """Inverse Jacobian elliptic sn
    
    解方程 w = sn(z, m)，求解 z
    
    Parameters
    ----------
    w : 复数标量
        自变量
    m : 标量
        模数；取值范围为 [0, 1]

    See [1], Eq. (56)

    References
    ----------
    .. [1] Orfanidis, "Lecture Notes on Elliptic Filter Design",
           https://www.ece.rutgers.edu/~orfanidi/ece521/notes.pdf
    """

    # 计算补充参数kx
    def _complement(kx):
        # (1-k**2) ** 0.5; 下面的表达式适用于小的kx
        return ((1 - kx) * (1 + kx)) ** 0.5

    # 计算k，即m的平方根
    k = m ** 0.5

    # 处理k超过1的情况
    if k > 1:
        return np.nan
    elif k == 1:
        return np.arctanh(w)

    # 使用Landen变换求解过程中的参数列表
    ks = [k]
    niter = 0
    while ks[-1] != 0:
        k_ = ks[-1]
        k_p = _complement(k_)
        ks.append((1 - k_p) / (1 + k_p))
        niter += 1
        if niter > _ARC_JAC_SN_MAXITER:
            raise ValueError('Landen transformation not converging')

    # 计算K值，是ks列表中元素的累积乘积乘以π/2
    K = np.prod(1 + np.array(ks[1:])) * np.pi/2

    # 计算wns列表，用于最终计算z值
    wns = [w]
    for kn, knext in zip(ks[:-1], ks[1:]):
        wn = wns[-1]
        wnext = (2 * wn /
                 ((1 + knext) * (1 + _complement(kn * wn))))
        wns.append(wnext)

    # 计算最终z值
    u = 2 / np.pi * np.arcsin(wns[-1])
    z = K * u
    return z


# 计算实部为w，模数为1-m的实Jacobean椭圆函数sc的逆函数
def _arc_jac_sc1(w, m):
    """Real inverse Jacobian sc, with complementary modulus
    
    解方程 w = sc(z, 1-m)，求解 z
    
    w - 实数标量
    m - 模数
    
    From [1], sc(z, m) = -i * sn(i * z, 1 - m)

    References
    ----------
    # noqa: E501
    .. [1] https://functions.wolfram.com/EllipticFunctions/JacobiSC/introductions/JacobiPQs/ShowAll.html,
       "Representations through other Jacobi functions"
    """

    # 调用_arc_jac_sn函数求解复数域中的sn函数的逆函数
    zcomplex = _arc_jac_sn(1j * w, m)
    
    # 确保zcomplex的实部接近零，否则抛出错误
    if abs(zcomplex.real) > 1e-14:
        raise ValueError

    return zcomplex.imag


# 返回N阶椭圆模拟低通滤波器的(z,p,k)参数
def ellipap(N, rp, rs):
    """Return (z,p,k) of Nth-order elliptic analog lowpass filter.
    
    返回N阶椭圆模拟低通滤波器的(z,p,k)参数。
    该滤波器是一个归一化的原型，通带中有rp分贝的波动，并且在停带中下降rs分贝。

    The filter's angular (e.g., rad/s) cutoff frequency is normalized to 1,
    defined as the point at which the gain first drops below ``-rp``.

    See Also
    --------
    ellip : 使用该原型进行滤波器设计

    References
    ----------
    .. [1] Lutovac, Tosic, and Evans, "Filter Design for Signal Processing",
           Chapters 5 and 12.

    .. [2] Orfanidis, "Lecture Notes on Elliptic Filter Design",
           https://www.ece.rutgers.edu/~orfanidi/ece521/notes.pdf
    """
    
    # 确保N是非负整数
    if abs(int(N)) != N:
        raise ValueError("Filter order must be a nonnegative integer")
    elif N == 0:
        # 避免除以零警告，偶数阶滤波器的直流增益为-rp dB
        return np.array([]), np.array([]), 10**(-rp/20)
    # 如果满足条件 N == 1，执行以下操作
    elif N == 1:
        # 计算 -sqrt(1.0 / _pow10m1(0.1 * rp)) 并赋给 p
        p = -sqrt(1.0 / _pow10m1(0.1 * rp))
        # 将 p 赋给 k
        k = -p
        # 创建一个空列表 z
        z = []
        # 返回 z 转换为数组的形式，以及 p 和 k 转换为数组的形式
        return asarray(z), asarray(p), k

    # 计算 eps_sq = _pow10m1(0.1 * rp)
    eps_sq = _pow10m1(0.1 * rp)

    # 计算 eps = np.sqrt(eps_sq)
    eps = np.sqrt(eps_sq)
    
    # 计算 ck1_sq = eps_sq / _pow10m1(0.1 * rs)
    ck1_sq = eps_sq / _pow10m1(0.1 * rs)
    
    # 如果 ck1_sq 等于 0，则抛出 ValueError 异常
    if ck1_sq == 0:
        raise ValueError("Cannot design a filter with given rp and rs"
                         " specifications.")

    # 计算 val = special.ellipk(ck1_sq), special.ellipkm1(ck1_sq)
    val = special.ellipk(ck1_sq), special.ellipkm1(ck1_sq)

    # 计算 m = _ellipdeg(N, ck1_sq)
    m = _ellipdeg(N, ck1_sq)

    # 计算 capk = special.ellipk(m)
    capk = special.ellipk(m)

    # 计算 j = np.arange(1 - N % 2, N, 2)，jj = len(j)
    j = np.arange(1 - N % 2, N, 2)
    jj = len(j)

    # 计算 [s, c, d, phi] = special.ellipj(j * capk / N, m * np.ones(jj))
    [s, c, d, phi] = special.ellipj(j * capk / N, m * np.ones(jj))
    
    # 从 s 中压缩出绝对值大于 EPSILON 的元素赋给 snew
    snew = np.compress(abs(s) > EPSILON, s, axis=-1)
    
    # 计算 z = 1.0 / (sqrt(m) * snew)
    z = 1.0 / (sqrt(m) * snew)
    
    # 将 z 转换为虚部为 1 的复数
    z = 1j * z
    
    # 将 z 与其共轭连接起来形成新的 z
    z = np.concatenate((z, conjugate(z)))

    # 计算 r = _arc_jac_sc1(1. / eps, ck1_sq)
    r = _arc_jac_sc1(1. / eps, ck1_sq)
    
    # 计算 v0 = capk * r / (N * val[0])
    v0 = capk * r / (N * val[0])

    # 计算 [sv, cv, dv, phi] = special.ellipj(v0, 1 - m)
    [sv, cv, dv, phi] = special.ellipj(v0, 1 - m)
    
    # 计算 p = -(c * d * sv * cv + 1j * s * dv) / (1 - (d * sv) ** 2.0)
    p = -(c * d * sv * cv + 1j * s * dv) / (1 - (d * sv) ** 2.0)

    # 如果 N 为偶数，执行以下操作
    if N % 2:
        # 压缩出满足条件 abs(p.imag) > EPSILON * np.sqrt(np.sum(p * np.conjugate(p), axis=0).real) 的 p 元素，并赋给 newp
        newp = np.compress(
            abs(p.imag) > EPSILON * np.sqrt(np.sum(p * np.conjugate(p), axis=0).real),
            p, axis=-1
        )
        # 将 p 与其共轭连接起来形成新的 p
        p = np.concatenate((p, conjugate(newp)))
    else:
        # 将 p 与其共轭连接起来形成新的 p
        p = np.concatenate((p, conjugate(p)))

    # 计算 k = (np.prod(-p, axis=0) / np.prod(-z, axis=0)).real
    k = (np.prod(-p, axis=0) / np.prod(-z, axis=0)).real
    
    # 如果 N 为偶数，计算 k = k / np.sqrt(1 + eps_sq)
    if N % 2 == 0:
        k = k / np.sqrt(1 + eps_sq)

    # 返回 z, p, k 作为结果
    return z, p, k
# TODO: Make this a real public function scipy.misc.ff
# 将此函数改造为真正的公共函数 scipy.misc.ff

def _falling_factorial(x, n):
    r"""
    Return the factorial of `x` to the `n` falling.

    This is defined as:

    .. math::   x^\underline n = (x)_n = x (x-1) \cdots (x-n+1)

    This can more efficiently calculate ratios of factorials, since:

    n!/m! == falling_factorial(n, n-m)

    where n >= m

    skipping the factors that cancel out

    the usual factorial n! == ff(n, n)
    """
    val = 1
    for k in range(x - n + 1, x + 1):
        val *= k
    return val


def _bessel_poly(n, reverse=False):
    """
    Return the coefficients of Bessel polynomial of degree `n`

    If `reverse` is true, a reverse Bessel polynomial is output.

    Output is a list of coefficients:
    [1]                   = 1
    [1,  1]               = 1*s   +  1
    [1,  3,  3]           = 1*s^2 +  3*s   +  3
    [1,  6, 15, 15]       = 1*s^3 +  6*s^2 + 15*s   +  15
    [1, 10, 45, 105, 105] = 1*s^4 + 10*s^3 + 45*s^2 + 105*s + 105
    etc.

    Output is a Python list of arbitrary precision long ints, so n is only
    limited by your hardware's memory.

    Sequence is http://oeis.org/A001498, and output can be confirmed to
    match http://oeis.org/A001498/b001498.txt :

    >>> from scipy.signal._filter_design import _bessel_poly
    >>> i = 0
    >>> for n in range(51):
    ...     for x in _bessel_poly(n, reverse=True):
    ...         print(i, x)
    ...         i += 1

    """
    if abs(int(n)) != n:
        raise ValueError("Polynomial order must be a nonnegative integer")
    else:
        n = int(n)  # np.int32 doesn't work, for instance

    out = []
    for k in range(n + 1):
        num = _falling_factorial(2*n - k, n)
        den = 2**(n - k) * math.factorial(k)
        out.append(num // den)

    if reverse:
        return out[::-1]
    else:
        return out


def _campos_zeros(n):
    """
    Return approximate zero locations of Bessel polynomials y_n(x) for order
    `n` using polynomial fit (Campos-Calderon 2011)
    """
    if n == 1:
        return asarray([-1+0j])

    s = npp_polyval(n, [0, 0, 2, 0, -3, 1])
    b3 = npp_polyval(n, [16, -8]) / s
    b2 = npp_polyval(n, [-24, -12, 12]) / s
    b1 = npp_polyval(n, [8, 24, -12, -2]) / s
    b0 = npp_polyval(n, [0, -6, 0, 5, -1]) / s

    r = npp_polyval(n, [0, 0, 2, 1])
    a1 = npp_polyval(n, [-6, -6]) / r
    a2 = 6 / r

    k = np.arange(1, n+1)
    x = npp_polyval(k, [0, a1, a2])
    y = npp_polyval(k, [b0, b1, b2, b3])

    return x + 1j*y


def _aberth(f, fp, x0, tol=1e-15, maxiter=50):
    """
    Given a function `f`, its first derivative `fp`, and a set of initial
    guesses `x0`, simultaneously find the roots of the polynomial using the
    Aberth-Ehrlich method.

    ``len(x0)`` should equal the number of roots of `f`.

    (This is not a complete implementation of Bini's algorithm.)
    """

    N = len(x0)

    x = array(x0, complex)
    beta = np.empty_like(x0)
    # 对于给定的最大迭代次数，执行迭代求解过程
    for iteration in range(maxiter):
        # 计算当前位置处的函数值与导数值，应用牛顿法计算步长 alpha
        alpha = -f(x) / fp(x)  # Newton's method

        # 计算每个零点之间的“排斥力”，即模型中的相互排斥效应
        for k in range(N):
            # 计算当前零点与其后各零点的距离倒数之和
            beta[k] = np.sum(1/(x[k] - x[k+1:]))
            # 加上当前零点与其前各零点的距离倒数之和
            beta[k] += np.sum(1/(x[k] - x[:k]))

        # 根据计算得到的步长 alpha 更新零点位置 x
        x += alpha / (1 + alpha * beta)

        # 检查更新后的零点是否全部为有限值
        if not all(np.isfinite(x)):
            # 若有非有限值出现，则抛出运行时错误
            raise RuntimeError('Root-finding calculation failed')

        # 判断是否所有零点的更新步长都已经小于设定的容差 tol
        # 若是，则停止迭代过程
        if all(abs(alpha) <= tol):
            break
    else:
        # 若迭代未能收敛，则抛出异常
        raise Exception('Zeros failed to converge')

    # 返回最终计算得到的零点位置 x
    return x
def _bessel_zeros(N):
    """
    Find zeros of ordinary Bessel polynomial of order `N`, by root-finding of
    modified Bessel function of the second kind
    """
    if N == 0:
        # 如果阶数 N 为 0，返回空数组
        return asarray([])

    # 生成起始点
    x0 = _campos_zeros(N)

    # 零点相同于 exp(1/x)*K_{N+0.5}(1/x) 和 N 阶普通贝塞尔多项式 y_N(x)
    def f(x):
        # 定义函数 f(x) 为修正贝塞尔函数的第二类 K_{N+0.5}(1/x)
        return special.kve(N+0.5, 1/x)

    # f(x) 的一阶导数
    def fp(x):
        # 返回修正贝塞尔函数的第二类 K_{N-0.5}(1/x)/(2*x**2) -
        # special.kve(N+0.5, 1/x)/(x**2) +
        # special.kve(N+1.5, 1/x)/(2*x**2) 的结果
        return (special.kve(N-0.5, 1/x)/(2*x**2) -
                special.kve(N+0.5, 1/x)/(x**2) +
                special.kve(N+1.5, 1/x)/(2*x**2))

    # 使用阿伯特法找到起始点的根
    x = _aberth(f, fp, x0)

    # 对每个根使用牛顿法提高精度
    for i in range(len(x)):
        x[i] = optimize.newton(f, x[i], fp, tol=1e-15)

    # 平均复共轭根以确保对称性
    x = np.mean((x, x[::-1].conj()), 0)

    # 根的总和应为 -1
    if abs(np.sum(x) + 1) > 1e-15:
        # 如果根的总和不接近 -1，抛出运行时错误
        raise RuntimeError('Generated zeros are inaccurate')

    return x


def _norm_factor(p, k):
    """
    Numerically find frequency shift to apply to delay-normalized filter such
    that -3 dB point is at 1 rad/sec.

    `p` is an array_like of polynomial poles
    `k` is a float gain

    First 10 values are listed in "Bessel Scale Factors" table,
    "Bessel Filters Polynomials, Poles and Circuit Elements 2003, C. Bond."
    """
    p = asarray(p, dtype=complex)

    def G(w):
        """
        Gain of filter
        """
        # 返回滤波器的增益
        return abs(k / prod(1j*w - p))

    def cutoff(w):
        """
        When gain = -3 dB, return 0
        """
        # 当增益为 -3 dB 时，返回值为 0
        return G(w) - 1/np.sqrt(2)

    # 使用牛顿法找到使得增益降低到 -3 dB 的角频率
    return optimize.newton(cutoff, 1.5)


def besselap(N, norm='phase'):
    """
    Return (z,p,k) for analog prototype of an Nth-order Bessel filter.

    Parameters
    ----------
    N : int
        The order of the filter.
    norm : {'phase', 'delay', 'mag'}, optional
        Frequency normalization:

        ``phase``
            The filter is normalized such that the phase response reaches its
            midpoint at an angular (e.g., rad/s) cutoff frequency of 1. This
            happens for both low-pass and high-pass filters, so this is the
            "phase-matched" case. [6]_

            The magnitude response asymptotes are the same as a Butterworth
            filter of the same order with a cutoff of `Wn`.

            This is the default, and matches MATLAB's implementation.

        ``delay``
            The filter is normalized such that the group delay in the passband
            is 1 (e.g., 1 second). This is the "natural" type obtained by
            solving Bessel polynomials

        ``mag``
            The filter is normalized such that the gain magnitude is -3 dB at
            angular frequency 1. This is called "frequency normalization" by
            Bond. [1]_

        .. versionadded:: 0.18.0

    Returns
    -------
    ```
    """
    z : ndarray
        Zeros of the transfer function. Is always an empty array.
    p : ndarray
        Poles of the transfer function.
    k : scalar
        Gain of the transfer function. For phase-normalized, this is always 1.

    See Also
    --------
    bessel : Filter design function using this prototype

    Notes
    -----
    To find the pole locations, approximate starting points are generated [2]_
    for the zeros of the ordinary Bessel polynomial [3]_, then the
    Aberth-Ehrlich method [4]_ [5]_ is used on the Kv(x) Bessel function to
    calculate more accurate zeros, and these locations are then inverted about
    the unit circle.

    References
    ----------
    .. [1] C.R. Bond, "Bessel Filter Constants",
           http://www.crbond.com/papers/bsf.pdf
    .. [2] Campos and Calderon, "Approximate closed-form formulas for the
           zeros of the Bessel Polynomials", :arXiv:`1105.0957`.
    .. [3] Thomson, W.E., "Delay Networks having Maximally Flat Frequency
           Characteristics", Proceedings of the Institution of Electrical
           Engineers, Part III, November 1949, Vol. 96, No. 44, pp. 487-490.
    .. [4] Aberth, "Iteration Methods for Finding all Zeros of a Polynomial
           Simultaneously", Mathematics of Computation, Vol. 27, No. 122,
           April 1973
    .. [5] Ehrlich, "A modified Newton method for polynomials", Communications
           of the ACM, Vol. 10, Issue 2, pp. 107-108, Feb. 1967,
           :DOI:`10.1145/363067.363115`
    .. [6] Miller and Bohn, "A Bessel Filter Crossover, and Its Relation to
           Others", RaneNote 147, 1998,
           https://www.ranecommercial.com/legacy/note147.html

    """
    # 如果 N 不是整数或者是负数，则抛出异常
    if abs(int(N)) != N:
        raise ValueError("Filter order must be a nonnegative integer")

    # 将 N 转换为整数类型，因为下面的计算有时超出 np.int64 的范围
    N = int(N)
    # 如果 N 等于 0，则传输函数没有极点（poles为空数组），增益（gain）为1
    if N == 0:
        p = []  # 没有极点
        k = 1   # 增益为1
    else:
        # 找到反向 Bessel 多项式的根（极点）
        p = 1 / _bessel_zeros(N)

        # 计算最后一个系数 a_last
        a_last = _falling_factorial(2 * N, N) // 2**N

        # 根据需要将极点转换为不同的归一化方式
        if norm in ('delay', 'mag'):
            # 对于单位群延迟归一化
            k = a_last
            if norm == 'mag':
                # -3 dB 频率点在 1 弧度/秒处
                norm_factor = _norm_factor(p, k)
                p /= norm_factor
                k = norm_factor**-N * a_last
        elif norm == 'phase':
            # 相位匹配归一化（1 弧度/秒处的最大相位移的一半）
            # 渐近线与 Butterworth 滤波器相同
            p *= 10**(-math.log10(a_last) / N)
            k = 1
        else:
            raise ValueError('normalization not understood')  # 不理解的归一化方式

    # 返回空数组作为零点、包含极点的复数数组以及增益的浮点数
    return asarray([]), asarray(p, dtype=complex), float(k)
# 设计二阶 IIR 陷波数字滤波器

def iirnotch(w0, Q, fs=2.0):
    """
    Design second-order IIR notch digital filter.

    A notch filter is a band-stop filter with a narrow bandwidth
    (high quality factor). It rejects a narrow frequency band and
    leaves the rest of the spectrum little changed.

    Parameters
    ----------
    w0 : float
        Frequency to remove from a signal. If `fs` is specified, this is in
        the same units as `fs`. By default, it is a normalized scalar that must
        satisfy  ``0 < w0 < 1``, with ``w0 = 1`` corresponding to half of the
        sampling frequency.
    Q : float
        Quality factor. Dimensionless parameter that characterizes
        notch filter -3 dB bandwidth ``bw`` relative to its center
        frequency, ``Q = w0/bw``.
    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (``b``) and denominator (``a``) polynomials
        of the IIR filter.

    See Also
    --------
    iirpeak

    Notes
    -----
    .. versionadded:: 0.19.0

    References
    ----------
    .. [1] Sophocles J. Orfanidis, "Introduction To Signal Processing",
           Prentice-Hall, 1996

    Examples
    --------
    Design and plot filter to remove the 60 Hz component from a
    signal sampled at 200 Hz, using a quality factor Q = 30

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> fs = 200.0  # Sample frequency (Hz)
    >>> f0 = 60.0  # Frequency to be removed from signal (Hz)
    >>> Q = 30.0  # Quality factor
    >>> # Design notch filter
    >>> b, a = signal.iirnotch(f0, Q, fs)

    >>> # Frequency response
    >>> freq, h = signal.freqz(b, a, fs=fs)
    >>> # Plot
    >>> fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    >>> ax[0].plot(freq, 20*np.log10(abs(h)), color='blue')
    >>> ax[0].set_title("Frequency Response")
    >>> ax[0].set_ylabel("Amplitude (dB)", color='blue')
    >>> ax[0].set_xlim([0, 100])
    >>> ax[0].set_ylim([-25, 10])
    >>> ax[0].grid(True)
    >>> ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='green')
    >>> ax[1].set_ylabel("Angle (degrees)", color='green')
    >>> ax[1].set_xlabel("Frequency (Hz)")
    >>> ax[1].set_xlim([0, 100])
    >>> ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
    >>> ax[1].set_ylim([-90, 90])
    >>> ax[1].grid(True)
    >>> plt.show()
    """

    # 调用内部函数 _design_notch_peak_filter 来实现陷波滤波器的设计和返回滤波器参数
    return _design_notch_peak_filter(w0, Q, "notch", fs)


def iirpeak(w0, Q, fs=2.0):
    """
    Design second-order IIR peak (resonant) digital filter.

    A peak filter is a band-pass filter with a narrow bandwidth
    (high quality factor). It rejects components outside a narrow
    frequency band.

    Parameters
    ----------
    w0 : float
        Frequency to be retained in a signal. If `fs` is specified, this is in
        the same units as `fs`. By default, it is a normalized scalar that must
        satisfy  ``0 < w0 < 1``, with ``w0 = 1`` corresponding to half of the
        sampling frequency.
    Q : float
        Quality factor. Dimensionless parameter that characterizes
        peak filter -3 dB bandwidth ``bw`` relative to its center
        frequency, ``Q = w0/bw``.
    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (``b``) and denominator (``a``) polynomials
        of the IIR filter.

    See Also
    --------
    iirnotch

    Notes
    -----
    .. versionadded:: 0.19.0

    References
    ----------
    .. [1] Sophocles J. Orfanidis, "Introduction To Signal Processing",
           Prentice-Hall, 1996

    Examples
    --------
    Design and plot filter to remove the frequencies other than the 300 Hz
    component from a signal sampled at 1000 Hz, using a quality factor Q = 30

    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> fs = 1000.0  # Sample frequency (Hz)
    >>> f0 = 300.0  # Frequency to be retained (Hz)
    >>> Q = 30.0  # Quality factor
    >>> # Design peak filter
    >>> b, a = signal.iirpeak(f0, Q, fs)

    >>> # Frequency response
    >>> freq, h = signal.freqz(b, a, fs=fs)
    >>> # Plot
    >>> fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    >>> ax[0].plot(freq, 20*np.log10(np.maximum(abs(h), 1e-5)), color='blue')
    >>> ax[0].set_title("Frequency Response")
    >>> ax[0].set_ylabel("Amplitude (dB)", color='blue')
    >>> ax[0].set_xlim([0, 500])
    >>> ax[0].set_ylim([-50, 10])
    >>> ax[0].grid(True)
    >>> ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='green')
    >>> ax[1].set_ylabel("Angle (degrees)", color='green')
    >>> ax[1].set_xlabel("Frequency (Hz)")
    >>> ax[1].set_xlim([0, 500])
    >>> ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
    >>> ax[1].set_ylim([-90, 90])
    >>> ax[1].grid(True)
    >>> plt.show()
    """

    return _design_notch_peak_filter(w0, Q, "peak", fs)


注释：

    # 返回设计的峰值滤波器的分子（b）和分母（a）多项式系数
    return _design_notch_peak_filter(w0, Q, "peak", fs)
def iircomb(w0, Q, ftype='notch', fs=2.0, *, pass_zero=False):
    """
    Design IIR notching or peaking digital comb filter.

    A notching comb filter consists of regularly-spaced band-stop filters with
    a narrow bandwidth (high quality factor). Each rejects a narrow frequency
    band and leaves the rest of the spectrum little changed.

    A peaking comb filter consists of regularly-spaced band-pass filters with
    a narrow bandwidth (high quality factor). Each rejects components outside
    a narrow frequency band.

    Parameters
    ----------
    w0 : float
        Normalized frequency to remove or emphasize in the signal. If `fs` is
        specified, this is in the same units as `fs`. By default, it is a
        normalized scalar that must satisfy ``0 < w0 < 1``, with ``w0 = 1``
        corresponding to half of the sampling frequency.
    Q : float
        Quality factor. Dimensionless parameter that characterizes the notch
        filter -3 dB bandwidth ``bw`` relative to its center frequency,
        ``Q = w0/bw``.
    ftype : str, optional
        The type of IIR filter to design:
            - 'notch' for notching comb filter.
            - 'peak' for peaking comb filter.
    fs : float, optional
        The sampling frequency of the digital system.
    pass_zero : bool, optional
        Whether to design a filter that passes through zero frequency.

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (``b``) and denominator (``a``) polynomials
        of the IIR filter.
    """
    fs = _validate_fs(fs, allow_none=False)  # Validate and ensure fs is a float

    w0 = float(w0)  # Ensure w0 is a float
    Q = float(Q)    # Ensure Q is a float

    w0 = 2 * w0 / fs  # Normalize w0 with respect to fs

    if w0 > 1.0 or w0 < 0.0:  # Check if normalized w0 is within valid range
        raise ValueError("w0 should be such that 0 < w0 < 1")

    bw = w0 / Q  # Compute bandwidth bw

    bw = bw * np.pi  # Normalize bandwidth in radians
    w0 = w0 * np.pi   # Normalize w0 in radians

    if ftype not in ("notch", "peak"):  # Validate ftype
        raise ValueError("Unknown ftype.")

    beta = np.tan(bw / 2.0)  # Compute beta parameter using tangent function

    gain = 1.0 / (1.0 + beta)  # Compute gain

    if ftype == "notch":
        # Design notch filter numerator b and denominator a
        b = gain * np.array([1.0, -2.0 * np.cos(w0), 1.0])
    else:
        # Design peak filter numerator b and denominator a
        b = (1.0 - gain) * np.array([1.0, 0.0, -1.0])

    a = np.array([1.0, -2.0 * gain * np.cos(w0), (2.0 * gain - 1.0)])  # Design denominator a

    return b, a  # Return numerator and denominator polynomials of the IIR filter
    w0 : float
        # 梳状滤波器的基频（其峰值之间的间距）。此值必须能整除采样频率。如果指定了 `fs`，则单位与 `fs` 相同。默认情况下，这是一个归一化的标量，必须满足 ``0 < w0 < 1``，其中 ``w0 = 1`` 对应于采样频率的一半。
    Q : float
        # 质量因子。无量纲参数，用于描述陷波滤波器的 -3 dB 带宽 ``bw`` 相对于其中心频率的特性， ``Q = w0/bw``。
    ftype : {'notch', 'peak'}
        # 函数生成的梳状滤波器的类型。如果是 'notch'，则 Q 因子应用于陷波；如果是 'peak'，则 Q 因子应用于峰值。默认为 'notch'。
    fs : float, optional
        # 信号的采样频率。默认值为 2.0。
    pass_zero : bool, optional
        # 如果为 False（默认），滤波器的零点（空洞）位于频率 [0, w0, 2*w0, ...] 处，峰值位于中点 [w0/2, 3*w0/2, 5*w0/2, ...] 处。如果为 True，则峰值位于 [0, w0, 2*w0, ...] 处（通过零频率），反之亦然。
        # 
        # .. versionadded:: 1.9.0

    Returns
    -------
    b, a : ndarray, ndarray
        # IIR 滤波器的分子（``b``）和分母（``a``）多项式。

    Raises
    ------
    ValueError
        # 如果 `w0` 小于或等于 0，或大于或等于 ``fs/2``，如果 `fs` 不能被 `w0` 整除，或者 `ftype` 不是 'notch' 或 'peak'。

    See Also
    --------
    iirnotch
    iirpeak

    Notes
    -----
    # 有关实现细节，请参见 [1]_。由于使用了一个重复的单极点，梳状滤波器的 TF 实现即使在较高阶次下也是数值稳定的，不会因精度损失而受影响。

    References
    ----------
    .. [1] Sophocles J. Orfanidis, "Introduction To Signal Processing",
           Prentice-Hall, 1996, ch. 11, "Digital Filter Design"

    Examples
    --------
    # 设计并绘制采样频率为 200 Hz 的信号中 20 Hz 处的陷波梳状滤波器，使用质量因子 Q = 30

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> fs = 200.0  # 采样频率（Hz）
    >>> f0 = 20.0   # 从信号中移除的频率（Hz）
    >>> Q = 30.0    # 质量因子
    >>> # 设计陷波梳状滤波器
    >>> b, a = signal.iircomb(f0, Q, ftype='notch', fs=fs)

    >>> # 频率响应
    >>> freq, h = signal.freqz(b, a, fs=fs)
    >>> response = abs(h)
    >>> # 为了在绘图时避免除以零
    >>> response[response == 0] = 1e-20
    >>> # 绘图
    >>> fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    >>> ax[0].plot(freq, 20*np.log10(abs(response)), color='blue')
    >>> ax[0].set_title("频率响应")
    >>> ax[0].set_ylabel("幅度（dB）", color='blue')
    >>> ax[0].set_xlim([0, 100])
    # Convert w0, Q, and fs to float
    w0 = float(w0)  # 将 w0 转换为浮点数
    Q = float(Q)  # 将 Q 转换为浮点数
    fs = _validate_fs(fs, allow_none=False)  # 调用 _validate_fs 函数验证并获取有效的采样频率 fs

    # Check for invalid cutoff frequency or filter type
    ftype = ftype.lower()  # 将滤波器类型转换为小写
    if not 0 < w0 < fs / 2:
        raise ValueError(f"w0 must be between 0 and {fs / 2}"
                         f" (nyquist), but given {w0}.")  # 如果 w0 不在有效范围内，则引发值错误异常
    if ftype not in ('notch', 'peak'):
        raise ValueError('ftype must be either notch or peak.')  # 如果滤波器类型不是 'notch' 或 'peak'，则引发值错误异常

    # Compute the order of the filter
    N = round(fs / w0)  # 计算滤波器的阶数

    # Check for cutoff frequency divisibility
    if abs(w0 - fs/N)/fs > 1e-14:
        raise ValueError('fs must be divisible by w0.')  # 如果 w0 和 fs/N 之间的差异大于指定的精度限制，则引发值错误异常

    # Compute frequency in radians and filter bandwidth
    # Eq. 11.3.1 (p. 574) from reference [1]
    w0 = (2 * np.pi * w0) / fs  # 计算频率 w0 的弧度表示
    w_delta = w0 / Q  # 计算滤波器的带宽

    # Define base gain values depending on notch or peak filter
    # Compute -3dB attenuation
    # Eqs. 11.4.1 and 11.4.2 (p. 582) from reference [1]
    if ftype == 'notch':
        G0, G = 1, 0  # 如果是陷波滤波器，则设定 G0=1, G=0
    elif ftype == 'peak':
        G0, G = 0, 1  # 如果是峰值滤波器，则设定 G0=0, G=1

    # Compute beta according to Eq. 11.5.3 (p. 591) from reference [1]. Due to
    # assuming a -3 dB attenuation value, i.e, assuming GB = 1 / np.sqrt(2),
    # the following term simplifies to:
    #   np.sqrt((GB**2 - G0**2) / (G**2 - GB**2)) = 1
    beta = np.tan(N * w_delta / 4)  # 根据公式计算 beta 值
    # 计算滤波器系数

    # 根据参考文献[1]中的方程式 11.5.1（第590页），计算变量 a, b, c
    ax = (1 - beta) / (1 + beta)  # 计算 a_x 系数
    bx = (G0 + G * beta) / (1 + beta)  # 计算 b_x 系数
    cx = (G0 - G * beta) / (1 + beta)  # 计算 c_x 系数

    # 最后一个系数是负数，以获得通过零的峰值组合或不通过零的凹槽组合
    negative_coef = ((ftype == 'peak' and pass_zero) or
                     (ftype == 'notch' and not pass_zero))

    # 计算分子系数

    # 根据参考文献[1]中的方程式 11.5.1（第590页）或方程式 11.5.4（第591页），计算 b 系数
    # b - cz^-N 或 b + cz^-N
    b = np.zeros(N + 1)  # 初始化长度为 N+1 的零数组
    b[0] = bx  # 设置 b[0] 系数
    if negative_coef:
        b[-1] = -cx  # 如果是负系数，设置 b[-1] 为 -c_x
    else:
        b[-1] = +cx  # 如果是正系数，设置 b[-1] 为 +c_x

    # 计算分母系数

    # 根据参考文献[1]中的方程式 11.5.1（第590页）或方程式 11.5.4（第591页），计算 a 系数
    # 1 - az^-N 或 1 + az^-N
    a = np.zeros(N + 1)  # 初始化长度为 N+1 的零数组
    a[0] = 1  # 设置 a[0] 系数为 1
    if negative_coef:
        a[-1] = -ax  # 如果是负系数，设置 a[-1] 为 -a_x
    else:
        a[-1] = +ax  # 如果是正系数，设置 a[-1] 为 +a_x

    return b, a
# 将频率（单位：Hz）转换为等效矩形带宽（ERB）的实用工具函数
def _hz_to_erb(hz):
    """
    Utility for converting from frequency (Hz) to the
    Equivalent Rectangular Bandwidth (ERB) scale
    ERB = frequency / EarQ + minBW
    """
    # EarQ 是 ERB 公式中的常数
    EarQ = 9.26449
    # minBW 是 ERB 公式中的最小带宽常数
    minBW = 24.7
    # 根据 ERB 公式计算并返回给定频率的等效矩形带宽（ERB）
    return hz / EarQ + minBW


# 设计 Gammatone 滤波器
def gammatone(freq, ftype, order=None, numtaps=None, fs=None):
    """
    Gammatone filter design.

    This function computes the coefficients of an FIR or IIR gammatone
    digital filter [1]_.

    Parameters
    ----------
    freq : float
        Center frequency of the filter (expressed in the same units
        as `fs`).
    ftype : {'fir', 'iir'}
        The type of filter the function generates. If 'fir', the function
        will generate an Nth order FIR gammatone filter. If 'iir', the
        function will generate an 8th order digital IIR filter, modeled as
        as 4th order gammatone filter.
    order : int, optional
        The order of the filter. Only used when ``ftype='fir'``.
        Default is 4 to model the human auditory system. Must be between
        0 and 24.
    numtaps : int, optional
        Length of the filter. Only used when ``ftype='fir'``.
        Default is ``fs*0.015`` if `fs` is greater than 1000,
        15 if `fs` is less than or equal to 1000.
    fs : float, optional
        The sampling frequency of the signal. `freq` must be between
        0 and ``fs/2``. Default is 2.

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (``b``) and denominator (``a``) polynomials of the filter.

    Raises
    ------
    ValueError
        If `freq` is less than or equal to 0 or greater than or equal to
        ``fs/2``, if `ftype` is not 'fir' or 'iir', if `order` is less than
        or equal to 0 or greater than 24 when ``ftype='fir'``

    See Also
    --------
    firwin
    iirfilter

    References
    ----------
    .. [1] Slaney, Malcolm, "An Efficient Implementation of the
        Patterson-Holdsworth Auditory Filter Bank", Apple Computer
        Technical Report 35, 1993, pp.3-8, 34-39.

    Examples
    --------
    16-sample 4th order FIR Gammatone filter centered at 440 Hz

    >>> from scipy import signal
    >>> signal.gammatone(440, 'fir', numtaps=16, fs=16000)
    (array([ 0.00000000e+00,  2.22196719e-07,  1.64942101e-06,  4.99298227e-06,
        1.01993969e-05,  1.63125770e-05,  2.14648940e-05,  2.29947263e-05,
        1.76776931e-05,  2.04980537e-06, -2.72062858e-05, -7.28455299e-05,
       -1.36651076e-04, -2.19066855e-04, -3.18905076e-04, -4.33156712e-04]),
       [1.0])

    IIR Gammatone filter centered at 440 Hz

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> b, a = signal.gammatone(440, 'iir', fs=16000)
    >>> w, h = signal.freqz(b, a)
    >>> plt.plot(w / ((2 * np.pi) / 16000), 20 * np.log10(abs(h)))
    >>> plt.xscale('log')
    >>> plt.title('Gammatone filter frequency response')
    >>> plt.xlabel('Frequency')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.margins(0, 0.1)
    """
    # 根据提供的频率和滤波器类型，计算并返回 Gammatone 滤波器的系数
    # 首先计算 ERB 频率
    ERB = _hz_to_erb(freq)
    
    if ftype == 'fir':
        # FIR 滤波器类型，根据 order 和 numtaps 参数生成 N 阶 FIR Gammatone 滤波器
        if order is None:
            order = 4
        if numtaps is None:
            numtaps = int(fs * 0.015) if fs > 1000 else 15
        # 使用 scipy 的 firwin 函数生成 FIR 滤波器系数
        b = signal.firwin(numtaps, cutoff=ERB, fs=fs)
        a = 1.0
    elif ftype == 'iir':
        # IIR 滤波器类型，生成 8 阶数字 IIR 滤波器，模拟 4 阶 Gammatone 滤波器
        order = 8
        # 使用 scipy 的 iirfilter 函数生成 IIR 滤波器系数
        b, a = signal.iirfilter(order, [ERB], fs=fs)
    else:
        # 如果滤波器类型既不是 'fir' 也不是 'iir'，抛出 ValueError 异常
        raise ValueError("Unknown filter type. Please use 'fir' or 'iir'.")
    
    # 返回生成的滤波器系数
    return b, a
    # 将频率 freq 转换为浮点数
    freq = float(freq)

    # 如果未传入采样率 fs，则设置为默认值 2
    if fs is None:
        fs = 2
    fs = _validate_fs(fs, allow_none=False)  # 确保采样率 fs 合法

    # 检查无效的截止频率或滤波器类型
    ftype = ftype.lower()  # 将滤波器类型转换为小写
    filter_types = ['fir', 'iir']
    if not 0 < freq < fs / 2:
        raise ValueError(f"The frequency must be between 0 and {fs / 2}"
                         f" (nyquist), but given {freq}.")  # 抛出值错误，如果频率超出有效范围
    if ftype not in filter_types:
        raise ValueError('ftype must be either fir or iir.')  # 抛出值错误，如果滤波器类型不是 fir 或 iir

    # 计算 FIR gammatone 滤波器
    if ftype == 'fir':
        # 如果未传入阶数 order，则设置为默认值 4
        if order is None:
            order = 4
        order = operator.index(order)  # 转换阶数为整数

        # 如果未传入 numtaps，则根据采样率计算默认值
        if numtaps is None:
            numtaps = max(int(fs * 0.015), 15)
        numtaps = operator.index(numtaps)  # 转换 numtaps 为整数

        # 检查阶数是否有效
        if not 0 < order <= 24:
            raise ValueError("Invalid order: order must be > 0 and <= 24.")  # 抛出值错误，如果阶数不在有效范围内

        # 设置 Gammatone 脉冲响应参数
        t = np.arange(numtaps) / fs
        bw = 1.019 * _hz_to_erb(freq)

        # 计算 FIR Gammatone 滤波器
        b = (t ** (order - 1)) * np.exp(-2 * np.pi * bw * t)
        b *= np.cos(2 * np.pi * freq * t)

        # 缩放 FIR 滤波器，使截止频率处的频率响应为 1
        scale_factor = 2 * (2 * np.pi * bw) ** (order)
        scale_factor /= float_factorial(order - 1)
        scale_factor /= fs
        b *= scale_factor
        a = [1.0]  # 设置 FIR 滤波器的分母系数为 [1.0]

    # 计算 IIR gammatone 滤波器
    elif ftype == 'iir':
        # 如果ftype为'iir'，则执行以下操作（IIR滤波器）

        # 如果传入了order或者numtaps，则发出警告信息
        if order is not None:
            warnings.warn('order is not used for IIR gammatone filter.', stacklevel=2)
        if numtaps is not None:
            warnings.warn('numtaps is not used for IIR gammatone filter.', stacklevel=2)

        # 设置Gammatone冲激响应的参数
        T = 1./fs  # 计算采样周期
        bw = 2 * np.pi * 1.019 * _hz_to_erb(freq)  # 计算带宽
        fr = 2 * freq * np.pi * T  # 计算频率响应
        bwT = bw * T  # 计算带宽乘以采样周期

        # 计算用于在中心频率上归一化音量的增益
        g1 = -2 * np.exp(2j * fr) * T
        g2 = 2 * np.exp(-(bwT) + 1j * fr) * T
        g3 = np.sqrt(3 + 2 ** (3 / 2)) * np.sin(fr)
        g4 = np.sqrt(3 - 2 ** (3 / 2)) * np.sin(fr)
        g5 = np.exp(2j * fr)

        g = g1 + g2 * (np.cos(fr) - g4)
        g *= (g1 + g2 * (np.cos(fr) + g4))
        g *= (g1 + g2 * (np.cos(fr) - g3))
        g *= (g1 + g2 * (np.cos(fr) + g3))
        g /= ((-2 / np.exp(2 * bwT) - 2 * g5 + 2 * (1 + g5) / np.exp(bwT)) ** 4)
        g = np.abs(g)

        # 创建空的滤波器系数列表
        b = np.empty(5)
        a = np.empty(9)

        # 计算分子系数
        b[0] = (T ** 4) / g
        b[1] = -4 * T ** 4 * np.cos(fr) / np.exp(bw * T) / g
        b[2] = 6 * T ** 4 * np.cos(2 * fr) / np.exp(2 * bw * T) / g
        b[3] = -4 * T ** 4 * np.cos(3 * fr) / np.exp(3 * bw * T) / g
        b[4] = T ** 4 * np.cos(4 * fr) / np.exp(4 * bw * T) / g

        # 计算分母系数
        a[0] = 1
        a[1] = -8 * np.cos(fr) / np.exp(bw * T)
        a[2] = 4 * (4 + 3 * np.cos(2 * fr)) / np.exp(2 * bw * T)
        a[3] = -8 * (6 * np.cos(fr) + np.cos(3 * fr))
        a[3] /= np.exp(3 * bw * T)
        a[4] = 2 * (18 + 16 * np.cos(2 * fr) + np.cos(4 * fr))
        a[4] /= np.exp(4 * bw * T)
        a[5] = -8 * (6 * np.cos(fr) + np.cos(3 * fr))
        a[5] /= np.exp(5 * bw * T)
        a[6] = 4 * (4 + 3 * np.cos(2 * fr)) / np.exp(6 * bw * T)
        a[7] = -8 * np.cos(fr) / np.exp(7 * bw * T)
        a[8] = np.exp(-8 * bw * T)

    return b, a
# 定义滤波器类型到对应的滤波器设计函数和计算函数的映射字典
filter_dict = {'butter': [buttap, buttord],            # 'butter' 对应 Butterworth 滤波器的设计和计算函数列表
               'butterworth': [buttap, buttord],      # 'butterworth' 同样对应 Butterworth 滤波器
               'cauer': [ellipap, ellipord],           # 'cauer' 对应 Cauer (Elliptic) 滤波器的设计和计算函数列表
               'elliptic': [ellipap, ellipord],        # 'elliptic' 同样对应 Cauer (Elliptic) 滤波器
               'ellip': [ellipap, ellipord],           # 'ellip' 同样对应 Cauer (Elliptic) 滤波器
               'bessel': [besselap],                   # 'bessel' 对应 Bessel 滤波器的设计函数列表
               'bessel_phase': [besselap],             # 'bessel_phase' 同样对应 Bessel 滤波器
               'bessel_delay': [besselap],             # 'bessel_delay' 同样对应 Bessel 滤波器
               'bessel_mag': [besselap],               # 'bessel_mag' 同样对应 Bessel 滤波器
               'cheby1': [cheb1ap, cheb1ord],          # 'cheby1' 对应 Chebyshev Type I 滤波器的设计和计算函数列表
               'chebyshev1': [cheb1ap, cheb1ord],      # 'chebyshev1' 同样对应 Chebyshev Type I 滤波器
               'chebyshevi': [cheb1ap, cheb1ord],      # 'chebyshevi' 同样对应 Chebyshev Type I 滤波器
               'cheby2': [cheb2ap, cheb2ord],          # 'cheby2' 对应 Chebyshev Type II 滤波器的设计和计算函数列表
               'chebyshev2': [cheb2ap, cheb2ord],      # 'chebyshev2' 同样对应 Chebyshev Type II 滤波器
               'chebyshevii': [cheb2ap, cheb2ord],     # 'chebyshevii' 同样对应 Chebyshev Type II 滤波器
               }

# 定义频带类型到对应的滤波器类型的映射字典
band_dict = {'band': 'bandpass',                       # 'band' 对应带通滤波器类型
             'bandpass': 'bandpass',                  # 'bandpass' 同样对应带通滤波器类型
             'pass': 'bandpass',                       # 'pass' 同样对应带通滤波器类型
             'bp': 'bandpass',                         # 'bp' 同样对应带通滤波器类型
             
             'bs': 'bandstop',                         # 'bs' 对应带阻滤波器类型
             'bandstop': 'bandstop',                   # 'bandstop' 同样对应带阻滤波器类型
             'bands': 'bandstop',                      # 'bands' 同样对应带阻滤波器类型
             'stop': 'bandstop',                       # 'stop' 同样对应带阻滤波器类型
             
             'l': 'lowpass',                           # 'l' 对应低通滤波器类型
             'low': 'lowpass',                         # 'low' 同样对应低通滤波器类型
             'lowpass': 'lowpass',                     # 'lowpass' 同样对应低通滤波器类型
             'lp': 'lowpass',                           # 'lp' 同样对应低通滤波器类型
             
             'high': 'highpass',                       # 'high' 对应高通滤波器类型
             'highpass': 'highpass',                   # 'highpass' 同样对应高通滤波器类型
             'h': 'highpass',                          # 'h' 同样对应高通滤波器类型
             'hp': 'highpass',                         # 'hp' 同样对应高通滤波器类型
             }

# 定义 Bessel 滤波器类型到归一化方式的映射字典
bessel_norms = {'bessel': 'phase',                     # 'bessel' 对应 Bessel 滤波器的相位归一化
                'bessel_phase': 'phase',               # 'bessel_phase' 同样对应 Bessel 滤波器的相位归一化
                'bessel_delay': 'delay',               # 'bessel_delay' 对应 Bessel 滤波器的时延归一化
                'bessel_mag': 'mag'}                   # 'bessel_mag' 对应 Bessel 滤波器的幅度归一化
```