# `D:\src\scipysrc\scipy\scipy\signal\_fir_filter_design.py`

```
# 引入必要的数学函数和操作符
from math import ceil, log
import operator
# 警告库，用于处理潜在的异常或警告情况
import warnings
# 用于类型提示的工具
from typing import Literal, Optional

# 引入科学计算库 NumPy
import numpy as np
# 引入 NumPy 中的 FFT 相关函数
from numpy.fft import irfft, fft, ifft
# 引入 SciPy 中的 sinc 函数
from scipy.special import sinc
# 引入 SciPy 中的线性代数函数和异常处理工具
from scipy.linalg import (toeplitz, hankel, solve, LinAlgError, LinAlgWarning,
                          lstsq)
# 引入 SciPy 中信号处理模块的工具函数
from scipy.signal._arraytools import _validate_fs

# 从当前包中引入 _sigtools 模块
from . import _sigtools

# 设置模块的公开接口，这些函数可以被外部使用
__all__ = ['kaiser_beta', 'kaiser_atten', 'kaiserord',
           'firwin', 'firwin2', 'remez', 'firls', 'minimum_phase']


# 以下是函数定义和文档字符串的注释
# Some notes on function parameters:
#
# `cutoff` and `width` are given as numbers between 0 and 1.  These are
# relative frequencies, expressed as a fraction of the Nyquist frequency.
# For example, if the Nyquist frequency is 2 KHz, then width=0.15 is a width
# of 300 Hz.
#
# The `order` of a FIR filter is one less than the number of taps.
# This is a potential source of confusion, so in the following code,
# we will always use the number of taps as the parameterization of
# the 'size' of the filter. The "number of taps" means the number
# of coefficients, which is the same as the length of the impulse
# response of the filter.


def kaiser_beta(a):
    """Compute the Kaiser parameter `beta`, given the attenuation `a`.

    Parameters
    ----------
    a : float
        The desired attenuation in the stopband and maximum ripple in
        the passband, in dB.  This should be a *positive* number.

    Returns
    -------
    beta : float
        The `beta` parameter to be used in the formula for a Kaiser window.

    References
    ----------
    Oppenheim, Schafer, "Discrete-Time Signal Processing", p.475-476.

    Examples
    --------
    Suppose we want to design a lowpass filter, with 65 dB attenuation
    in the stop band.  The Kaiser window parameter to be used in the
    window method is computed by ``kaiser_beta(65)``:

    >>> from scipy.signal import kaiser_beta
    >>> kaiser_beta(65)
    6.20426

    """
    if a > 50:
        beta = 0.1102 * (a - 8.7)
    elif a > 21:
        beta = 0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21)
    else:
        beta = 0.0
    return beta


def kaiser_atten(numtaps, width):
    """Compute the attenuation of a Kaiser FIR filter.

    Given the number of taps `N` and the transition width `width`, compute the
    attenuation `a` in dB, given by Kaiser's formula:

        a = 2.285 * (N - 1) * pi * width + 7.95

    Parameters
    ----------
    numtaps : int
        The number of taps in the FIR filter.
    width : float
        The desired width of the transition region between passband and
        stopband (or, in general, at any discontinuity) for the filter,
        expressed as a fraction of the Nyquist frequency.

    Returns
    -------
    a : float
        The attenuation of the ripple, in dB.

    See Also
    --------
    kaiserord, kaiser_beta

    Examples
    --------
    Suppose we want to design a FIR filter using the Kaiser window method
    # 计算 Kaiser 窗口设计的近似衰减（以分贝为单位）
    def kaiser_atten(numtaps, width):
        # 使用 Kaiser 窗口的公式计算衰减值
        a = 2.285 * (numtaps - 1) * np.pi * width + 7.95
        # 返回计算得到的衰减值
        return a
def kaiserord(ripple, width):
    """
    Determine the filter window parameters for the Kaiser window method.

    The parameters returned by this function are generally used to create
    a finite impulse response filter using the window method, with either
    `firwin` or `firwin2`.

    Parameters
    ----------
    ripple : float
        Upper bound for the deviation (in dB) of the magnitude of the
        filter's frequency response from that of the desired filter (not
        including frequencies in any transition intervals). That is, if w
        is the frequency expressed as a fraction of the Nyquist frequency,
        A(w) is the actual frequency response of the filter and D(w) is the
        desired frequency response, the design requirement is that::

            abs(A(w) - D(w))) < 10**(-ripple/20)

        for 0 <= w <= 1 and w not in a transition interval.
    width : float
        Width of transition region, normalized so that 1 corresponds to pi
        radians / sample. That is, the frequency is expressed as a fraction
        of the Nyquist frequency.

    Returns
    -------
    numtaps : int
        The length of the Kaiser window.
    beta : float
        The beta parameter for the Kaiser window.

    See Also
    --------
    kaiser_beta, kaiser_atten

    Notes
    -----
    There are several ways to obtain the Kaiser window:

    - ``signal.windows.kaiser(numtaps, beta, sym=True)``
    - ``signal.get_window(beta, numtaps)``
    - ``signal.get_window(('kaiser', beta), numtaps)``

    The empirical equations discovered by Kaiser are used.

    References
    ----------
    Oppenheim, Schafer, "Discrete-Time Signal Processing", pp.475-476.

    Examples
    --------
    We will use the Kaiser window method to design a lowpass FIR filter
    for a signal that is sampled at 1000 Hz.

    We want at least 65 dB rejection in the stop band, and in the pass
    band the gain should vary no more than 0.5%.

    We want a cutoff frequency of 175 Hz, with a transition between the
    pass band and the stop band of 24 Hz. That is, in the band [0, 163],
    the gain varies no more than 0.5%, and in the band [187, 500], the
    signal is attenuated by at least 65 dB.

    >>> import numpy as np
    >>> from scipy.signal import kaiserord, firwin, freqz
    >>> import matplotlib.pyplot as plt
    >>> fs = 1000.0
    >>> cutoff = 175
    >>> width = 24

    The Kaiser method accepts just a single parameter to control the pass
    band ripple and the stop band rejection, so we use the more restrictive
    of the two. In this case, the pass band ripple is 0.005, or 46.02 dB,
    so we will use 65 dB as the design parameter.

    Use `kaiserord` to determine the length of the filter and the
    parameter for the Kaiser window.

    >>> numtaps, beta = kaiserord(65, width/(0.5*fs))
    >>> numtaps
    167
    >>> beta
    6.20426

    Use `firwin` to create the FIR filter.

    >>> taps = firwin(numtaps, cutoff, window=('kaiser', beta),
    """
    # 使用 Kaiser 窗口方法确定滤波器的窗口参数
    # 确定 Kaiser 窗口的长度和 beta 参数
    numtaps = int(np.ceil((ripple - 7.95) / (2.285 * np.pi * width / np.pi)))
    beta = kaiser_beta(ripple)
    return numtaps, beta
    A = abs(ripple)  # 计算波动的绝对值，用于确保理解
    if A < 8:
        # 如果波动太小，Kaiser 公式不再有效。
        raise ValueError("请求的最大波动衰减 %f 对于Kaiser公式来说太小。" % A)
    beta = kaiser_beta(A)

    # 根据Kaiser窗口方法，计算滤波器阶数。
    # Kaiser公式（来自Oppenheim和Schafer的描述）是针对滤波器阶数的，因此需要加1来获得taps的数量。
    numtaps = (A - 7.95) / 2.285 / (np.pi * width) + 1

    return int(ceil(numtaps)), beta
# 使用窗口法设计 FIR 滤波器
def firwin(numtaps, cutoff, *, width=None, window='hamming', pass_zero=True,
           scale=True, fs=None):
    """
    FIR filter design using the window method.

    This function computes the coefficients of a finite impulse response
    filter. The filter will have linear phase; it will be Type I if
    `numtaps` is odd and Type II if `numtaps` is even.

    Type II filters always have zero response at the Nyquist frequency, so a
    ValueError exception is raised if firwin is called with `numtaps` even and
    having a passband whose right end is at the Nyquist frequency.

    Parameters
    ----------
    numtaps : int
        Length of the filter (number of coefficients, i.e. the filter
        order + 1).  `numtaps` must be odd if a passband includes the
        Nyquist frequency.
    cutoff : float or 1-D array_like
        Cutoff frequency of filter (expressed in the same units as `fs`)
        OR an array of cutoff frequencies (that is, band edges). In the
        latter case, the frequencies in `cutoff` should be positive and
        monotonically increasing between 0 and `fs/2`. The values 0 and
        `fs/2` must not be included in `cutoff`.
    width : float or None, optional
        If `width` is not None, then assume it is the approximate width
        of the transition region (expressed in the same units as `fs`)
        for use in Kaiser FIR filter design. In this case, the `window`
        argument is ignored.
    window : string or tuple of string and parameter values, optional
        Desired window to use. See `scipy.signal.get_window` for a list
        of windows and required parameters.
    pass_zero : {True, False, 'bandpass', 'lowpass', 'highpass', 'bandstop'}, optional
        If True, the gain at the frequency 0 (i.e., the "DC gain") is 1.
        If False, the DC gain is 0. Can also be a string argument for the
        desired filter type (equivalent to ``btype`` in IIR design functions).

        .. versionadded:: 1.3.0
           Support for string arguments.
    scale : bool, optional
        Set to True to scale the coefficients so that the frequency
        response is exactly unity at a certain frequency.
        That frequency is either:

        - 0 (DC) if the first passband starts at 0 (i.e. pass_zero
          is True)
        - `fs/2` (the Nyquist frequency) if the first passband ends at
          `fs/2` (i.e the filter is a single band highpass filter);
          center of first passband otherwise

    fs : float, optional
        The sampling frequency of the signal. Each frequency in `cutoff`
        must be between 0 and ``fs/2``.  Default is 2.

    Returns
    -------
    h : (numtaps,) ndarray
        Coefficients of length `numtaps` FIR filter.

    Raises
    ------
    ValueError
        If `numtaps` is even and `pass_zero` is True with a passband ending at the Nyquist frequency.
    """
    """
        ValueError
            If any value in `cutoff` is less than or equal to 0 or greater
            than or equal to ``fs/2``, if the values in `cutoff` are not strictly
            monotonically increasing, or if `numtaps` is even but a passband
            includes the Nyquist frequency.
    
        See Also
        --------
        firwin2
        firls
        minimum_phase
        remez
    
        Examples
        --------
        Low-pass from 0 to f:
    
        >>> from scipy import signal
        >>> numtaps = 3
        >>> f = 0.1
        >>> signal.firwin(numtaps, f)
        array([ 0.06799017,  0.86401967,  0.06799017])
    
        Use a specific window function:
    
        >>> signal.firwin(numtaps, f, window='nuttall')
        array([  3.56607041e-04,   9.99286786e-01,   3.56607041e-04])
    
        High-pass ('stop' from 0 to f):
    
        >>> signal.firwin(numtaps, f, pass_zero=False)
        array([-0.00859313,  0.98281375, -0.00859313])
    
        Band-pass:
    
        >>> f1, f2 = 0.1, 0.2
        >>> signal.firwin(numtaps, [f1, f2], pass_zero=False)
        array([ 0.06301614,  0.88770441,  0.06301614])
    
        Band-stop:
    
        >>> signal.firwin(numtaps, [f1, f2])
        array([-0.00801395,  1.0160279 , -0.00801395])
    
        Multi-band (passbands are [0, f1], [f2, f3] and [f4, 1]):
    
        >>> f3, f4 = 0.3, 0.4
        >>> signal.firwin(numtaps, [f1, f2, f3, f4])
        array([-0.01376344,  1.02752689, -0.01376344])
    
        Multi-band (passbands are [f1, f2] and [f3,f4]):
    
        >>> signal.firwin(numtaps, [f1, f2, f3, f4], pass_zero=False)
        array([ 0.04890915,  0.91284326,  0.04890915])
    
        """
        # The major enhancements to this function added in November 2010 were
        # developed by Tom Krauss (see ticket #902).
    
        # Validate and set the sampling frequency `fs`, defaulting to 2 if `fs` is None.
        fs = _validate_fs(fs, allow_none=True)
        fs = 2 if fs is None else fs
    
        # Calculate the Nyquist frequency.
        nyq = 0.5 * fs
    
        # Normalize the cutoff frequencies relative to the Nyquist frequency.
        cutoff = np.atleast_1d(cutoff) / float(nyq)
    
        # Check for invalid input.
        if cutoff.ndim > 1:
            raise ValueError("The cutoff argument must be at most "
                             "one-dimensional.")
        if cutoff.size == 0:
            raise ValueError("At least one cutoff frequency must be given.")
        if cutoff.min() <= 0 or cutoff.max() >= 1:
            raise ValueError("Invalid cutoff frequency: frequencies must be "
                             "greater than 0 and less than fs/2.")
        if np.any(np.diff(cutoff) <= 0):
            raise ValueError("Invalid cutoff frequencies: the frequencies "
                             "must be strictly increasing.")
    
        if width is not None:
            # Calculate the Kaiser window parameter `beta` if `width` is provided,
            # which overrides any `window` value passed in.
            atten = kaiser_atten(numtaps, float(width) / nyq)
            beta = kaiser_beta(atten)
            window = ('kaiser', beta)
    # 如果 pass_zero 是字符串类型，则根据其取值进行条件判断
    if isinstance(pass_zero, str):
        # 如果 pass_zero 是 'bandstop' 或者 'lowpass'，则设置 pass_zero 为 True
        if pass_zero in ('bandstop', 'lowpass'):
            # 如果 pass_zero 是 'lowpass'，则检查 cutoff 的长度是否为1，否则引发异常
            if pass_zero == 'lowpass':
                if cutoff.size != 1:
                    raise ValueError('cutoff must have one element if '
                                     f'pass_zero=="lowpass", got {cutoff.shape}')
            # 如果 pass_zero 不是 'lowpass'，则检查 cutoff 的长度是否小于等于1，否则引发异常
            elif cutoff.size <= 1:
                raise ValueError('cutoff must have at least two elements if '
                                 f'pass_zero=="bandstop", got {cutoff.shape}')
            # 将 pass_zero 设置为 True
            pass_zero = True
        # 如果 pass_zero 是 'bandpass' 或者 'highpass'，则设置 pass_zero 为 False
        elif pass_zero in ('bandpass', 'highpass'):
            # 如果 pass_zero 是 'highpass'，则检查 cutoff 的长度是否为1，否则引发异常
            if pass_zero == 'highpass':
                if cutoff.size != 1:
                    raise ValueError('cutoff must have one element if '
                                     f'pass_zero=="highpass", got {cutoff.shape}')
            # 如果 pass_zero 不是 'highpass'，则检查 cutoff 的长度是否小于等于1，否则引发异常
            elif cutoff.size <= 1:
                raise ValueError('cutoff must have at least two elements if '
                                 f'pass_zero=="bandpass", got {cutoff.shape}')
            # 将 pass_zero 设置为 False
            pass_zero = False
        else:
            # 如果 pass_zero 的取值不在预定义的范围内，则引发异常
            raise ValueError('pass_zero must be True, False, "bandpass", '
                             '"lowpass", "highpass", or "bandstop", got '
                             f'{pass_zero}')
    
    # 确保 pass_zero 是布尔类型
    pass_zero = bool(operator.index(pass_zero))  # ensure bool-like

    # 计算 pass_nyquist，用于确定是否需要在 cutoff 的末尾添加 0 或 1
    pass_nyquist = bool(cutoff.size & 1) ^ pass_zero
    # 如果 pass_nyquist 为 True，并且 numtaps 是偶数，则引发异常
    if pass_nyquist and numtaps % 2 == 0:
        raise ValueError("A filter with an even number of coefficients must "
                         "have zero response at the Nyquist frequency.")

    # 在 cutoff 的两端插入 0 和/或 1，使得 cutoff 的长度为偶数，并且每对值对应一个通带
    cutoff = np.hstack(([0.0] * pass_zero, cutoff, [1.0] * pass_nyquist))

    # 将 cutoff 转换为 2-D 数组 bands，每行表示一个通带的左右边界
    bands = cutoff.reshape(-1, 2)

    # 构建滤波器系数
    alpha = 0.5 * (numtaps - 1)
    m = np.arange(0, numtaps) - alpha
    h = 0
    for left, right in bands:
        h += right * sinc(right * m)
        h -= left * sinc(left * m)

    # 获取并应用窗口函数
    from .windows import get_window
    win = get_window(window, numtaps, fftbins=False)
    h *= win

    # 如果需要，进行缩放处理
    if scale:
        # 获取第一个通带的左右边界
        left, right = bands[0]
        if left == 0:
            scale_frequency = 0.0
        elif right == 1:
            scale_frequency = 1.0
        else:
            scale_frequency = 0.5 * (left + right)
        c = np.cos(np.pi * m * scale_frequency)
        s = np.sum(h * c)
        h /= s

    # 返回滤波器的频率响应
    return h
# Original version of firwin2 from scipy ticket #457, submitted by "tash".
#
# Rewritten by Warren Weckesser, 2010.
def firwin2(numtaps, freq, gain, *, nfreqs=None, window='hamming',
            antisymmetric=False, fs=None):
    """
    FIR filter design using the window method.

    From the given frequencies `freq` and corresponding gains `gain`,
    this function constructs an FIR filter with linear phase and
    (approximately) the given frequency response.

    Parameters
    ----------
    numtaps : int
        The number of taps in the FIR filter.  `numtaps` must be less than
        `nfreqs`.
    freq : array_like, 1-D
        The frequency sampling points. Typically 0.0 to 1.0 with 1.0 being
        Nyquist.  The Nyquist frequency is half `fs`.
        The values in `freq` must be nondecreasing. A value can be repeated
        once to implement a discontinuity. The first value in `freq` must
        be 0, and the last value must be ``fs/2``. Values 0 and ``fs/2`` must
        not be repeated.
    gain : array_like
        The filter gains at the frequency sampling points. Certain
        constraints to gain values, depending on the filter type, are applied,
        see Notes for details.
    nfreqs : int, optional
        The size of the interpolation mesh used to construct the filter.
        For most efficient behavior, this should be a power of 2 plus 1
        (e.g, 129, 257, etc). The default is one more than the smallest
        power of 2 that is not less than `numtaps`. `nfreqs` must be greater
        than `numtaps`.
    window : string or (string, float) or float, or None, optional
        Window function to use. Default is "hamming". See
        `scipy.signal.get_window` for the complete list of possible values.
        If None, no window function is applied.
    antisymmetric : bool, optional
        Whether resulting impulse response is symmetric/antisymmetric.
        See Notes for more details.
    fs : float, optional
        The sampling frequency of the signal. Each frequency in `cutoff`
        must be between 0 and ``fs/2``. Default is 2.

    Returns
    -------
    taps : ndarray
        The filter coefficients of the FIR filter, as a 1-D array of length
        `numtaps`.

    See Also
    --------
    firls
    firwin
    minimum_phase
    remez

    Notes
    -----
    From the given set of frequencies and gains, the desired response is
    constructed in the frequency domain. The inverse FFT is applied to the
    desired response to create the associated convolution kernel, and the
    first `numtaps` coefficients of this kernel, scaled by `window`, are
    returned.

    The FIR filter will have linear phase. The type of filter is determined by
    the value of 'numtaps` and `antisymmetric` flag.
    """

    # 根据给定的频率和增益，使用窗口法设计 FIR 滤波器
    # 构造频率响应，并进行频率域的反向 FFT 处理，生成相关的卷积核
    # 返回这个卷积核的前 `numtaps` 个系数，经过 `window` 函数的缩放
    pass
    """
    There are four possible combinations:
    
       - odd  `numtaps`, `antisymmetric` is False, type I filter is produced
       - even `numtaps`, `antisymmetric` is False, type II filter is produced
       - odd  `numtaps`, `antisymmetric` is True, type III filter is produced
       - even `numtaps`, `antisymmetric` is True, type IV filter is produced
    
    Magnitude response of all but type I filters are subjects to following
    constraints:
    
       - type II  -- zero at the Nyquist frequency
       - type III -- zero at zero and Nyquist frequencies
       - type IV  -- zero at zero frequency
    
    .. versionadded:: 0.9.0
    
    References
    ----------
    .. [1] Oppenheim, A. V. and Schafer, R. W., "Discrete-Time Signal
       Processing", Prentice-Hall, Englewood Cliffs, New Jersey (1989).
       (See, for example, Section 7.4.)
    
    .. [2] Smith, Steven W., "The Scientist and Engineer's Guide to Digital
       Signal Processing", Ch. 17. http://www.dspguide.com/ch17/1.htm
    
    Examples
    --------
    A lowpass FIR filter with a response that is 1 on [0.0, 0.5], and
    that decreases linearly on [0.5, 1.0] from 1 to 0:
    
    >>> from scipy import signal
    >>> taps = signal.firwin2(150, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0])
    >>> print(taps[72:78])
    [-0.02286961 -0.06362756  0.57310236  0.57310236 -0.06362756 -0.02286961]
    
    """
    
    # Validate and set the sampling frequency `fs`, defaulting to 2 if `fs` is None
    fs = _validate_fs(fs, allow_none=True)
    fs = 2 if fs is None else fs
    
    # Calculate the Nyquist frequency
    nyq = 0.5 * fs
    
    # Check if `freq` and `gain` arrays have the same length
    if len(freq) != len(gain):
        raise ValueError('freq and gain must be of same length.')
    
    # Check if `numtaps` is less than `nfreqs` if `nfreqs` is specified
    if nfreqs is not None and numtaps >= nfreqs:
        raise ValueError(('ntaps must be less than nfreqs, but firwin2 was '
                          'called with ntaps=%d and nfreqs=%s') %
                         (numtaps, nfreqs))
    
    # Ensure `freq` starts with 0 and ends with `nyq`
    if freq[0] != 0 or freq[-1] != nyq:
        raise ValueError('freq must start with 0 and end with fs/2.')
    
    # Check if `freq` values are non-decreasing
    d = np.diff(freq)
    if (d < 0).any():
        raise ValueError('The values in freq must be nondecreasing.')
    
    # Check if any value in `freq` occurs more than twice consecutively
    d2 = d[:-1] + d[1:]
    if (d2 == 0).any():
        raise ValueError('A value in freq must not occur more than twice.')
    
    # Ensure value 0 is not repeated in `freq`
    if freq[1] == 0:
        raise ValueError('Value 0 must not be repeated in freq')
    
    # Ensure value `nyq` is not repeated in `freq`
    if freq[-2] == nyq:
        raise ValueError('Value fs/2 must not be repeated in freq')
    
    # Determine filter type based on `numtaps` and `antisymmetric`
    if antisymmetric:
        if numtaps % 2 == 0:
            ftype = 4  # Type IV filter
        else:
            ftype = 3  # Type III filter
    else:
        if numtaps % 2 == 0:
            ftype = 2  # Type II filter
        else:
            ftype = 1  # Type I filter
    
    # Validate filter type constraints based on `ftype` and `gain`
    if ftype == 2 and gain[-1] != 0.0:
        raise ValueError("A Type II filter must have zero gain at the "
                         "Nyquist frequency.")
    elif ftype == 3 and (gain[0] != 0.0 or gain[-1] != 0.0):
        raise ValueError("A Type III filter must have zero gain at zero "
                         "and Nyquist frequencies.")
    # 如果滤波器类型为4且增益不为0，则引发值错误异常
    elif ftype == 4 and gain[0] != 0.0:
        raise ValueError("A Type IV filter must have zero gain at zero "
                         "frequency.")

    # 如果未指定频率数量，则计算默认的频率数量
    if nfreqs is None:
        nfreqs = 1 + 2 ** int(ceil(log(numtaps, 2)))

    # 如果频率数组中存在任何零元素，执行以下操作
    if (d == 0).any():
        # 复制频率数组以便进行调整
        freq = np.array(freq, copy=True)
        # 计算机器精度乘以奈奎斯特频率，作为微调的阈值
        eps = np.finfo(float).eps * nyq
        # 循环遍历频率数组，调整重复值，以确保插值操作正常进行
        for k in range(len(freq) - 1):
            if freq[k] == freq[k + 1]:
                freq[k] = freq[k] - eps
                freq[k + 1] = freq[k + 1] + eps
        # 检查调整后的频率数组是否严格递增，若不是则引发值错误异常
        d = np.diff(freq)
        if (d <= 0).any():
            raise ValueError("freq cannot contain numbers that are too close "
                             "(within eps * (fs/2): "
                             f"{eps}) to a repeated value")

    # 在0到奈奎斯特频率之间均匀生成nfreqs个点的网格x
    x = np.linspace(0.0, nyq, nfreqs)
    # 对频率响应进行线性插值，得到在x点处的插值结果fx
    fx = np.interp(x, freq, gain)

    # 调整系数的相位，使得反FFT的前ntaps个点为期望的滤波器系数
    shift = np.exp(-(numtaps - 1) / 2. * 1.j * np.pi * x / nyq)
    if ftype > 2:
        shift *= 1j

    # 对插值后的频率响应乘以相位调整系数，得到fx2
    fx2 = fx * shift

    # 使用逆FFT计算得到滤波器的时域响应
    out_full = irfft(fx2)

    # 如果指定了窗口函数，则根据窗口函数对滤波器系数进行加权处理
    if window is not None:
        # 导入窗口函数并获取相应窗口
        from .windows import get_window
        wind = get_window(window, numtaps, fftbins=False)
    else:
        wind = 1

    # 保留out_full中的前numtaps个系数，并乘以窗口函数
    out = out_full[:numtaps] * wind

    # 如果滤波器类型为3，将out中间的一个点设为0
    if ftype == 3:
        out[out.size // 2] = 0.0

    # 返回滤波器的时域响应系数
    return out
# 使用 Remez 交换算法计算最小最大误差的最优滤波器系数

def remez(numtaps, bands, desired, *, weight=None, type='bandpass',
          maxiter=25, grid_density=16, fs=None):
    """
    Calculate the minimax optimal filter using the Remez exchange algorithm.

    Parameters
    ----------
    numtaps : int
        滤波器中期望的阶数（或者说滤波器的阶数加一）
    bands : array_like
        包含频带边缘的单调序列，所有元素必须非负且小于采样频率的一半（由 `fs` 给出）
    desired : array_like
        在每个指定频带中的期望增益的一半大小序列
    weight : array_like, optional
        给定每个频带区域的相对加权值。`weight` 的长度必须是 `bands` 长度的一半
    type : {'bandpass', 'differentiator', 'hilbert'}, optional
        滤波器的类型:

          * 'bandpass' : 在频带内具有平坦的响应。默认值。
          * 'differentiator' : 在频带内具有频率比例响应。
          * 'hilbert' : 具有奇对称性的滤波器，即类型 III（偶阶）或类型 IV（奇阶）线性相位滤波器。

    maxiter : int, optional
        算法的最大迭代次数。默认值为 25。
    grid_density : int, optional
        网格密度。`remez` 中使用的密集网格大小为 ``(numtaps + 1) * grid_density``。默认为 16。
    fs : float, optional
        信号的采样频率。默认为 1。

    Returns
    -------
    out : ndarray
        包含最优（从最小最大误差的角度）滤波器系数的一维数组。

    See Also
    --------
    firls
    firwin
    firwin2
    minimum_phase

    References
    ----------
    .. [1] J. H. McClellan and T. W. Parks, "A unified approach to the
           design of optimum FIR linear phase digital filters",
           IEEE Trans. Circuit Theory, vol. CT-20, pp. 697-701, 1973.
    .. [2] J. H. McClellan, T. W. Parks and L. R. Rabiner, "A Computer
           Program for Designing Optimum FIR Linear Phase Digital
           Filters", IEEE Trans. Audio Electroacoust., vol. AU-21,
           pp. 506-525, 1973.

    Examples
    --------
    下面的示例展示了如何使用 `remez` 设计低通、高通、带通和带阻滤波器。每个滤波器的定义参数包括滤波器阶数、频带边界、边界过渡宽度、每个频带中的期望增益和采样频率。

    在所有示例中，我们使用了 22050 Hz 的采样频率。
    """
    example, the desired gain in each band is either 0 (for a stop band)
    or 1 (for a pass band).

    `freqz` is used to compute the frequency response of each filter, and
    the utility function ``plot_response`` defined below is used to plot
    the response.

    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> fs = 22050   # Sample rate, Hz

    >>> def plot_response(w, h, title):
    ...     "Utility function to plot response functions"
    ...     fig = plt.figure()  # 创建新的图形对象
    ...     ax = fig.add_subplot(111)  # 在图形对象中添加子图，1行1列第1个位置
    ...     ax.plot(w, 20*np.log10(np.abs(h)))  # 绘制频率响应曲线
    ...     ax.set_ylim(-40, 5)  # 设置纵轴范围
    ...     ax.grid(True)  # 显示网格
    ...     ax.set_xlabel('Frequency (Hz)')  # 设置横轴标签
    ...     ax.set_ylabel('Gain (dB)')  # 设置纵轴标签
    ...     ax.set_title(title)  # 设置图标题

    The first example is a low-pass filter, with cutoff frequency 8 kHz.
    The filter length is 325, and the transition width from pass to stop
    is 100 Hz.

    >>> cutoff = 8000.0    # Desired cutoff frequency, Hz  # 截止频率
    >>> trans_width = 100  # Width of transition from pass to stop, Hz  # 过渡带宽
    >>> numtaps = 325      # Size of the FIR filter.  # 滤波器长度
    >>> taps = signal.remez(numtaps, [0, cutoff, cutoff + trans_width, 0.5*fs],
    ...                     [1, 0], fs=fs)  # 设计低通滤波器系数
    >>> w, h = signal.freqz(taps, [1], worN=2000, fs=fs)  # 计算频率响应
    >>> plot_response(w, h, "Low-pass Filter")  # 绘制低通滤波器的响应曲线
    >>> plt.show()  # 显示图形

    This example shows a high-pass filter:

    >>> cutoff = 2000.0    # Desired cutoff frequency, Hz  # 截止频率
    >>> trans_width = 250  # Width of transition from pass to stop, Hz  # 过渡带宽
    >>> numtaps = 125      # Size of the FIR filter.  # 滤波器长度
    >>> taps = signal.remez(numtaps, [0, cutoff - trans_width, cutoff, 0.5*fs],
    ...                     [0, 1], fs=fs)  # 设计高通滤波器系数
    >>> w, h = signal.freqz(taps, [1], worN=2000, fs=fs)  # 计算频率响应
    >>> plot_response(w, h, "High-pass Filter")  # 绘制高通滤波器的响应曲线
    >>> plt.show()  # 显示图形

    This example shows a band-pass filter with a pass-band from 2 kHz to
    5 kHz.  The transition width is 260 Hz and the length of the filter
    is 63, which is smaller than in the other examples:

    >>> band = [2000, 5000]  # Desired pass band, Hz  # 通带频率范围
    >>> trans_width = 260    # Width of transition from pass to stop, Hz  # 过渡带宽
    >>> numtaps = 63         # Size of the FIR filter.  # 滤波器长度
    >>> edges = [0, band[0] - trans_width, band[0], band[1],
    ...          band[1] + trans_width, 0.5*fs]
    >>> taps = signal.remez(numtaps, edges, [0, 1, 0], fs=fs)  # 设计带通滤波器系数
    >>> w, h = signal.freqz(taps, [1], worN=2000, fs=fs)  # 计算频率响应
    >>> plot_response(w, h, "Band-pass Filter")  # 绘制带通滤波器的响应曲线
    >>> plt.show()  # 显示图形

    The low order leads to higher ripple and less steep transitions.

    The next example shows a band-stop filter.

    >>> band = [6000, 8000]  # Desired stop band, Hz  # 阻带频率范围
    >>> trans_width = 200    # Width of transition from pass to stop, Hz  # 过渡带宽
    >>> numtaps = 175        # Size of the FIR filter.  # 滤波器长度
    >>> edges = [0, band[0] - trans_width, band[0], band[1],
    ...          band[1] + trans_width, 0.5*fs]
    >>> taps = signal.remez(numtaps, edges, [1, 0, 1], fs=fs)
    >>> w, h = signal.freqz(taps, [1], worN=2000, fs=fs)
    >>> plot_response(w, h, "Band-stop Filter")
    >>> plt.show()

    """
    fs = _validate_fs(fs, allow_none=True)  # 调用 _validate_fs 函数验证并处理采样率参数 fs，允许其为 None
    fs = 1.0 if fs is None else fs  # 如果 fs 为 None，则设置为默认值 1.0

    # Convert type
    try:
        tnum = {'bandpass': 1, 'differentiator': 2, 'hilbert': 3}[type]  # 根据 type 参数选择滤波器类型编号
    except KeyError as e:
        raise ValueError("Type must be 'bandpass', 'differentiator', "
                         "or 'hilbert'") from e  # 如果 type 不是预定义的值，则抛出错误

    # Convert weight
    if weight is None:
        weight = [1] * len(desired)  # 如果权重参数 weight 为 None，则设置为与 desired 相同长度的全 1 列表

    bands = np.asarray(bands).copy()  # 将频带边界数组 bands 转换为 NumPy 数组的副本
    return _sigtools._remez(numtaps, bands, desired, weight, tnum, fs,
                            maxiter, grid_density)  # 调用底层函数 _remez 进行最小最大法设计滤波器的计算
# 定义一个函数用于设计最小二乘误差的有限脉冲响应（FIR）滤波器

def firls(numtaps, bands, desired, *, weight=None, fs=None):
    """
    FIR filter design using least-squares error minimization.

    Calculate the filter coefficients for the linear-phase finite
    impulse response (FIR) filter which has the best approximation
    to the desired frequency response described by `bands` and
    `desired` in the least squares sense (i.e., the integral of the
    weighted mean-squared error within the specified bands is
    minimized).

    Parameters
    ----------
    numtaps : int
        The number of taps in the FIR filter. `numtaps` must be odd.
    bands : array_like
        A monotonic nondecreasing sequence containing the band edges in
        Hz. All elements must be non-negative and less than or equal to
        the Nyquist frequency given by `nyq`. The bands are specified as
        frequency pairs, thus, if using a 1D array, its length must be
        even, e.g., `np.array([0, 1, 2, 3, 4, 5])`. Alternatively, the
        bands can be specified as an nx2 sized 2D array, where n is the
        number of bands, e.g, `np.array([[0, 1], [2, 3], [4, 5]])`.
    desired : array_like
        A sequence the same size as `bands` containing the desired gain
        at the start and end point of each band.
    weight : array_like, optional
        A relative weighting to give to each band region when solving
        the least squares problem. `weight` has to be half the size of
        `bands`.
    fs : float, optional
        The sampling frequency of the signal. Each frequency in `bands`
        must be between 0 and ``fs/2`` (inclusive). Default is 2.

    Returns
    -------
    coeffs : ndarray
        Coefficients of the optimal (in a least squares sense) FIR filter.

    See Also
    --------
    firwin
    firwin2
    minimum_phase
    remez

    Notes
    -----
    This implementation follows the algorithm given in [1]_.
    As noted there, least squares design has multiple advantages:

        1. Optimal in a least-squares sense.
        2. Simple, non-iterative method.
        3. The general solution can obtained by solving a linear
           system of equations.
        4. Allows the use of a frequency dependent weighting function.

    This function constructs a Type I linear phase FIR filter, which
    contains an odd number of `coeffs` satisfying for :math:`n < numtaps`:

    .. math:: coeffs(n) = coeffs(numtaps - 1 - n)

    The odd number of coefficients and filter symmetry avoid boundary
    conditions that could otherwise occur at the Nyquist and 0 frequencies
    (e.g., for Type II, III, or IV variants).

    .. versionadded:: 0.18

    References
    ----------
    .. [1] Ivan Selesnick, Linear-Phase Fir Filter Design By Least Squares.
           OpenStax CNX. Aug 9, 2005.
           http://cnx.org/contents/eb1ecb35-03a9-4610-ba87-41cd771c95f2@7

    Examples
    --------
    We want to construct a band-pass filter. Note that the behavior in the
    """
    """
    frequency ranges between our stop bands and pass bands is unspecified,
    and thus may overshoot depending on the parameters of our filter:

    导入所需的库和模块
    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    创建一个包含两个子图的图形对象
    >>> fig, axs = plt.subplots(2)

    定义采样率为 10 Hz
    >>> fs = 10.0  # Hz

    定义期望的频率响应：0-2 Hz 和 4-5 Hz 的通带，1-4 Hz 的阻带
    >>> desired = (0, 0, 1, 1, 0, 0)

    遍历两组滤波器设计参数
    >>> for bi, bands in enumerate(((0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 4.5, 5))):
    ...     # 使用 firls 方法设计 FIR 滤波器
    ...     fir_firls = signal.firls(73, bands, desired, fs=fs)
    ...     # 使用 remez 方法设计 FIR 滤波器
    ...     fir_remez = signal.remez(73, bands, desired[::2], fs=fs)
    ...     # 使用 firwin2 方法设计 FIR 滤波器
    ...     fir_firwin2 = signal.firwin2(73, bands, desired, fs=fs)
    ...     # 存储当前子图和其响应的句柄
    ...     hs = list()
    ...     ax = axs[bi]
    ...     # 对每个滤波器响应进行频率分析
    ...     for fir in (fir_firls, fir_remez, fir_firwin2):
    ...         freq, response = signal.freqz(fir)
    ...         hs.append(ax.semilogy(0.5*fs*freq/np.pi, np.abs(response))[0])
    ...     # 绘制阻带和通带的理想响应曲线
    ...     for band, gains in zip(zip(bands[::2], bands[1::2]),
    ...                            zip(desired[::2], desired[1::2])):
    ...         ax.semilogy(band, np.maximum(gains, 1e-7), 'k--', linewidth=2)
    ...     # 添加图例到第一个子图
    ...     if bi == 0:
    ...         ax.legend(hs, ('firls', 'remez', 'firwin2'),
    ...                   loc='lower center', frameon=False)
    ...     else:
    ...         ax.set_xlabel('Frequency (Hz)')
    ...     # 开启网格线显示
    ...     ax.grid(True)
    ...     # 设置子图标题和纵坐标标签
    ...     ax.set(title='Band-pass %d-%d Hz' % bands[2:4], ylabel='Magnitude')
    ...

    调整子图布局
    >>> fig.tight_layout()

    显示图形
    >>> plt.show()

    """

    # 验证并获取有效的采样率
    fs = _validate_fs(fs, allow_none=True)
    fs = 2 if fs is None else fs

    # 计算奈奎斯特频率
    nyq = 0.5 * fs

    # 确保滤波器的长度为奇数且大于等于 1
    numtaps = int(numtaps)
    if numtaps % 2 == 0 or numtaps < 1:
        raise ValueError("numtaps must be odd and >= 1")
    M = (numtaps-1) // 2

    # 将频带参数归一化到 0 到 1 之间，并将其整理为频带对的形式
    nyq = float(nyq)
    if nyq <= 0:
        raise ValueError('nyq must be positive, got %s <= 0.' % nyq)
    bands = np.asarray(bands).flatten() / nyq
    if len(bands) % 2 != 0:
        raise ValueError("bands must contain frequency pairs.")
    if (bands < 0).any() or (bands > 1).any():
        raise ValueError("bands must be between 0 and 1 relative to Nyquist")
    bands.shape = (-1, 2)

    # 检查期望的频率响应参数
    desired = np.asarray(desired).flatten()
    if bands.size != desired.size:
        raise ValueError(
            f"desired must have one entry per frequency, got {desired.size} "
            f"gains for {bands.size} frequencies."
        )
    desired.shape = (-1, 2)
    if (np.diff(bands) <= 0).any() or (np.diff(bands[:, 0]) < 0).any():
        raise ValueError("bands must be monotonically nondecreasing and have "
                         "width > 0.")
    if (bands[:-1, 1] > bands[1:, 0]).any():
        raise ValueError("bands must not overlap.")
    if (desired < 0).any():
        raise ValueError("desired must be non-negative.")

    # 检查权重参数，如果未指定则默认为全为 1 的权重
    if weight is None:
        weight = np.ones(len(desired))
    weight = np.asarray(weight).flatten()
    # 检查权重和期望数组的长度是否相等，若不相等则抛出数值错误异常
    if len(weight) != len(desired):
        raise ValueError("weight must be the same size as the number of "
                         f"band pairs ({len(bands)}).")

    # 检查权重数组中是否有负值，若有则抛出数值错误异常
    if (weight < 0).any():
        raise ValueError("weight must be non-negative.")

    # 设置线性矩阵方程 Qa = b

    # 我们可以表达 Q(k,n) = 0.5 Q1(k,n) + 0.5 Q2(k,n)
    # 其中 Q1(k,n)=q(k-n) 且 Q2(k,n)=q(k+n)，即一个Toeplitz加上一个Hankel。

    # 在系数计算时，我们省略了上述公式中的0.5倍数因子。

    # 我们也省略了Q和b方程中的1/π，因为它们在求解过程中会相互抵消。

    # 我们有以下关系式：
    #     q(n) = 1/π ∫W(ω)cos(nω)dω (在0到π上的积分)
    # 使用我们的归一化ω=πf，并且在每个区间f1->f2上有一个恒定的权重W，我们得到：
    #     q(n) = W∫cos(πnf)df (0到1) = Wf sin(πnf)/πnf
    # 在每个f1->f2对中积分（即f2处的值减去f1处的值）。

    # 创建一个包含numtaps个元素的一维数组，其中每个元素是一个包含两个空维度的二维数组
    n = np.arange(numtaps)[:, np.newaxis, np.newaxis]
    # 计算q(n)，这里使用了差分sinc函数乘以频带和权重的结果
    q = np.dot(np.diff(np.sinc(bands * n) * bands, axis=2)[:, :, 0], weight)

    # 现在我们组装我们的Toeplitz和Hankel矩阵
    Q1 = toeplitz(q[:M+1])  # 创建Toeplitz矩阵Q1
    Q2 = hankel(q[:M+1], q[M:])  # 创建Hankel矩阵Q2
    Q = Q1 + Q2  # 组合得到Q矩阵

    # 现在对于b(n)我们有：
    #     b(n) = 1/π ∫ W(ω)D(ω)cos(nω)dω (在0到π上的积分)
    # 使用我们的归一化ω=πf，并且在每个区间上有一个恒定的权重和一个线性项D(ω)，我们得到（在每个f1->f2区间）：
    #     b(n) = W ∫ (mf+c)cos(πnf)df
    #          = f(mf+c)sin(πnf)/πnf + mf**2 cos(nπf)/(πnf)**2
    # 在每个f1->f2对中积分（即f2处的值减去f1处的值）。

    n = n[:M + 1]  # 仅需要这么多个系数
    # 选择m和c以便我们在起始和结束权重处
    m = (np.diff(desired, axis=1) / np.diff(bands, axis=1))
    c = desired[:, [0]] - bands[:, [0]] * m
    # 计算b，这里使用了带有sinc函数的频带和权重的乘积
    b = bands * (m*bands + c) * np.sinc(bands * n)
    # 在n=0处使用L'Hospital法则进行cos(nπf)/(πnf)**2的计算
    b[0] -= m * bands * bands / 2.
    b[1:] += m * np.cos(n[1:] * np.pi * bands) / (np.pi * n[1:]) ** 2
    b = np.dot(np.diff(b, axis=2)[:, :, 0], weight)

    # 现在我们可以求解方程
    try:  # 尝试使用快速方法求解
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            a = solve(Q, b, assume_a="pos", check_finite=False)
        for ww in w:
            if (ww.category == LinAlgWarning and
                    str(ww.message).startswith('Ill-conditioned matrix')):
                raise LinAlgError(str(ww.message))
    except LinAlgError:  # 如果Q的秩不足
        # 这比pinvh更快，即使我们没有显式使用对称性。在一些非穷尽测试中，gelsy比gelsd和gelss更快。
        a = lstsq(Q, b, lapack_driver='gelsy')[0]

    # 使系数对称（线性相位）
    coeffs = np.hstack((a[:0:-1], 2 * a[0], a[1:]))
    return coeffs
# 计算修改后的一维离散希尔伯特变换

def _dhtm(mag):
    """Compute the modified 1-D discrete Hilbert transform

    Parameters
    ----------
    mag : ndarray
        The magnitude spectrum. Should be 1-D with an even length, and
        preferably a fast length for FFT/IFFT.
    """
    # 基于 Niranjan Damera-Venkata, Brian L. Evans 和 Shawn R. McCaslin 的代码进行调整
    sig = np.zeros(len(mag))
    # 将 Nyquist 和 DC 分量设置为 0，因为 np.abs(fftfreq(N)[midpt]) == 0.5
    midpt = len(mag) // 2
    sig[1:midpt] = 1
    sig[midpt+1:] = -1
    # 如果未来需要支持复杂滤波器，应在对数函数内部对 mag 进行 np.abs() 处理，并移除 .real
    recon = ifft(mag * np.exp(fft(sig * ifft(np.log(mag))))).real
    return recon


def minimum_phase(h: np.ndarray,
                  method: Literal['homomorphic', 'hilbert'] = 'homomorphic',
                  n_fft: Optional[int] = None, *, half: bool = True) -> np.ndarray:
    """Convert a linear-phase FIR filter to minimum phase

    Parameters
    ----------
    h : array
        Linear-phase FIR filter coefficients.
    method : {'hilbert', 'homomorphic'}
        The provided methods are:

            'homomorphic' (default)
                This method [4]_ [5]_ works best with filters with an
                odd number of taps, and the resulting minimum phase filter
                will have a magnitude response that approximates the square
                root of the original filter's magnitude response using half
                the number of taps when ``half=True`` (default), or the
                original magnitude spectrum using the same number of taps
                when ``half=False``.

            'hilbert'
                This method [1]_ is designed to be used with equiripple
                filters (e.g., from `remez`) with unity or zero gain
                regions.

    n_fft : int
        The number of points to use for the FFT. Should be at least a
        few times larger than the signal length (see Notes).
    half : bool
        If ``True``, create a filter that is half the length of the original, with a
        magnitude spectrum that is the square root of the original. If ``False``,
        create a filter that is the same length as the original, with a magnitude
        spectrum that is designed to match the original (only supported when
        ``method='homomorphic'``).

        .. versionadded:: 1.14.0

    Returns
    -------
    h_minimum : array
        The minimum-phase version of the filter, with length
        ``(len(h) + 1) // 2`` when ``half is True`` or ``len(h)`` otherwise.

    See Also
    --------
    firwin
    firwin2
    remez

    Notes
    -----
    Both the Hilbert [1]_ or homomorphic [4]_ [5]_ methods require selection
    of an FFT length to estimate the complex cepstrum of the filter.

    In the case of the Hilbert method, the deviation from the ideal
    spectrum ``epsilon`` is related to the number of stopband zeros
    ``n_stop`` and FFT length ``n_fft`` as::

        epsilon = 2. * n_stop / n_fft


    For example, with 100 stopband zeros and a FFT length of 2048,
    ``epsilon = 0.0976``. If we conservatively assume that the number of
    stopband zeros is one less than the filter length, we can take the FFT
    length to be the next power of 2 that satisfies ``epsilon=0.01`` as::

        n_fft = 2 ** int(np.ceil(np.log2(2 * (len(h) - 1) / 0.01)))


    This gives reasonable results for both the Hilbert and homomorphic
    methods, and gives the value used when ``n_fft=None``.


    Alternative implementations exist for creating minimum-phase filters,
    including zero inversion [2]_ and spectral factorization [3]_ [4]_.
    For more information, see `this DSPGuru page
    <http://dspguru.com/dsp/howtos/how-to-design-minimum-phase-fir-filters>`__.


    References
    ----------
    .. [1] N. Damera-Venkata and B. L. Evans, "Optimal design of real and
           complex minimum phase digital FIR filters," Acoustics, Speech,
           and Signal Processing, 1999. Proceedings., 1999 IEEE International
           Conference on, Phoenix, AZ, 1999, pp. 1145-1148 vol.3.
           :doi:`10.1109/ICASSP.1999.756179`
    .. [2] X. Chen and T. W. Parks, "Design of optimal minimum phase FIR
           filters by direct factorization," Signal Processing,
           vol. 10, no. 4, pp. 369-383, Jun. 1986.
    .. [3] T. Saramaki, "Finite Impulse Response Filter Design," in
           Handbook for Digital Signal Processing, chapter 4,
           New York: Wiley-Interscience, 1993.
    .. [4] J. S. Lim, Advanced Topics in Signal Processing.
           Englewood Cliffs, N.J.: Prentice Hall, 1988.
    .. [5] A. V. Oppenheim, R. W. Schafer, and J. R. Buck,
           "Discrete-Time Signal Processing," 3rd edition.
           Upper Saddle River, N.J.: Pearson, 2009.


    Examples
    --------
    Create an optimal linear-phase low-pass filter `h` with a transition band of
    [0.2, 0.3] (assuming a Nyquist frequency of 1):

    >>> import numpy as np
    >>> from scipy.signal import remez, minimum_phase, freqz, group_delay
    >>> import matplotlib.pyplot as plt
    >>> freq = [0, 0.2, 0.3, 1.0]
    >>> desired = [1, 0]
    >>> h_linear = remez(151, freq, desired, fs=2)


    Convert it to minimum phase:

    >>> h_hil = minimum_phase(h_linear, method='hilbert')
    >>> h_hom = minimum_phase(h_linear, method='homomorphic')
    >>> h_hom_full = minimum_phase(h_linear, method='homomorphic', half=False)


    Compare the impulse and frequency response of the four filters:

    >>> fig0, ax0 = plt.subplots(figsize=(6, 3), tight_layout=True)
    >>> fig1, axs = plt.subplots(3, sharex='all', figsize=(6, 6), tight_layout=True)
    >>> ax0.set_title("Impulse response")
    >>> ax0.set(xlabel='Samples', ylabel='Amplitude', xlim=(0, len(h_linear) - 1))
    >>> axs[0].set_title("Frequency Response")
    # 将滤波器系数 `h` 转换为 NumPy 数组
    h = np.asarray(h)
    # 如果滤波器系数是复数类型，则抛出异常，不支持复数滤波器
    if np.iscomplexobj(h):
        raise ValueError('Complex filters not supported')
    # 如果滤波器系数不是一维数组，或者长度小于等于2，则抛出异常
    if h.ndim != 1 or h.size <= 2:
        raise ValueError('h must be 1-D and at least 2 samples long')
    # 计算滤波器系数的一半长度
    n_half = len(h) // 2
    # 检查滤波器系数是否对称，如果不对称则发出警告
    if not np.allclose(h[-n_half:][::-1], h[:n_half]):
        warnings.warn('h does not appear to by symmetric, conversion may fail',
                      RuntimeWarning, stacklevel=2)
    # 检查方法参数是否为字符串类型且取值为 'homomorphic' 或 'hilbert'，否则抛出异常
    if not isinstance(method, str) or method not in ('homomorphic', 'hilbert',):
        raise ValueError(f'method must be "homomorphic" or "hilbert", got {method!r}')
    # 如果方法为 'hilbert' 且 half=False，则抛出异常，因为此时仅支持 `method='homomorphic'`
    if method == "hilbert" and not half:
        raise ValueError("`half=False` is only supported when `method='homomorphic'`")
    # 如果未指定 n_fft 的值，则根据公式计算其值，确保足够大以包含滤波器系数的有效信息
    if n_fft is None:
        n_fft = 2 ** int(np.ceil(np.log2(2 * (len(h) - 1) / 0.01)))
    # 将 n_fft 转换为整数类型
    n_fft = int(n_fft)
    # 如果 n_fft 小于滤波器系数的长度，则抛出异常
    if n_fft < len(h):
        raise ValueError('n_fft must be at least len(h)==%s' % len(h))
    if method == 'hilbert':
        # 根据希尔伯特方法计算滤波器的频率响应
        w = np.arange(n_fft) * (2 * np.pi / n_fft * n_half)
        # 计算希尔伯特变换的频率响应
        H = np.real(fft(h, n_fft) * np.exp(1j * w))
        # 计算通带和阻带的幅度差
        dp = max(H) - 1
        ds = 0 - min(H)
        # 计算滤波器的频率响应S
        S = 4. / (np.sqrt(1+dp+ds) + np.sqrt(1-dp+ds)) ** 2
        # 对频率响应加上阻带幅度
        H += ds
        # 将频率响应乘以S
        H *= S
        # 对频率响应进行开方，结果写回H
        H = np.sqrt(H, out=H)
        # 确保对数计算时不会发散，加上一个微小值
        H += 1e-10  # ensure that the log does not explode
        # 计算希尔伯特最小相位滤波器的最小相位
        h_minimum = _dhtm(H)
    else:  # method == 'homomorphic'
        # 零填充；计算DFT
        h_temp = np.abs(fft(h, n_fft))
        # 避免对数计算发散，加上一个微小值
        h_temp += 1e-7 * h_temp[h_temp > 0].min()  # don't let log blow up
        # 对频谱取对数
        np.log(h_temp, out=h_temp)
        if half:  # 可选的频谱减半
            h_temp *= 0.5
        # IDFT
        h_temp = ifft(h_temp).real
        # 逐点乘以同态滤波器
        # lmin[n] = 2u[n] - d[n]
        # 即，双倍正频率并将负频率置零；
        # Oppenheim+Shafer第3版p991 eq13.42b和p1004 fig13.7
        win = np.zeros(n_fft)
        win[0] = 1
        stop = n_fft // 2
        win[1:stop] = 2
        if n_fft % 2:
            win[stop] = 1
        h_temp *= win
        h_temp = ifft(np.exp(fft(h_temp)))
        # 计算同态滤波后的最小相位响应
        h_minimum = h_temp.real
    # 计算输出信号的长度
    n_out = (n_half + len(h) % 2) if half else len(h)
    # 返回最小相位响应的前n_out个值
    return h_minimum[:n_out]
```