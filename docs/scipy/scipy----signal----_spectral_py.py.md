# `D:\src\scipysrc\scipy\scipy\signal\_spectral_py.py`

```
"""Tools for spectral analysis.
"""
# 导入所需的库和模块
import numpy as np  # 导入 NumPy 库并重命名为 np
from scipy import fft as sp_fft  # 从 SciPy 库中导入 fft 模块，并重命名为 sp_fft
from . import _signaltools  # 从当前包中导入 _signaltools 模块
from .windows import get_window  # 从当前包中的 windows 模块导入 get_window 函数
from ._spectral import _lombscargle  # 从当前包中的 _spectral 模块导入 _lombscargle 函数
from ._arraytools import const_ext, even_ext, odd_ext, zero_ext  # 从当前包中的 _arraytools 模块导入指定函数
import warnings  # 导入警告模块


__all__ = ['periodogram', 'welch', 'lombscargle', 'csd', 'coherence',
           'spectrogram', 'stft', 'istft', 'check_COLA', 'check_NOLA']

# 定义函数 lombscargle，用于计算 Lomb-Scargle 周期图
def lombscargle(x,
                y,
                freqs,
                precenter=False,
                normalize=False):
    """
    lombscargle(x, y, freqs)

    Computes the Lomb-Scargle periodogram.

    The Lomb-Scargle periodogram was developed by Lomb [1]_ and further
    extended by Scargle [2]_ to find, and test the significance of weak
    periodic signals with uneven temporal sampling.

    When *normalize* is False (default) the computed periodogram
    is unnormalized, it takes the value ``(A**2) * N/4`` for a harmonic
    signal with amplitude A for sufficiently large N.

    When *normalize* is True the computed periodogram is normalized by
    the residuals of the data around a constant reference model (at zero).

    Input arrays should be 1-D and will be cast to float64.

    Parameters
    ----------
    x : array_like
        Sample times.
    y : array_like
        Measurement values.
    freqs : array_like
        Angular frequencies for output periodogram.
    precenter : bool, optional
        Pre-center measurement values by subtracting the mean.
    normalize : bool, optional
        Compute normalized periodogram.

    Returns
    -------
    pgram : array_like
        Lomb-Scargle periodogram.

    Raises
    ------
    ValueError
        If the input arrays `x` and `y` do not have the same shape.

    See Also
    --------
    istft: Inverse Short Time Fourier Transform
    check_COLA: Check whether the Constant OverLap Add (COLA) constraint is met
    welch: Power spectral density by Welch's method
    spectrogram: Spectrogram by Welch's method
    csd: Cross spectral density by Welch's method

    Notes
    -----
    This subroutine calculates the periodogram using a slightly
    modified algorithm due to Townsend [3]_ which allows the
    periodogram to be calculated using only a single pass through
    the input arrays for each frequency.

    The algorithm running time scales roughly as O(x * freqs) or O(N^2)
    for a large number of samples and frequencies.

    References
    ----------
    .. [1] N.R. Lomb "Least-squares frequency analysis of unequally spaced
           data", Astrophysics and Space Science, vol 39, pp. 447-462, 1976

    .. [2] J.D. Scargle "Studies in astronomical time series analysis. II -
           Statistical aspects of spectral analysis of unevenly spaced data",
           The Astrophysical Journal, vol 263, pp. 835-853, 1982
    """
    # 函数内部实现略，将在实际代码中完成
    pass
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    # 将输入的时间数组 x 转换为连续的内存布局，使用 np.float64 类型
    y = np.ascontiguousarray(y, dtype=np.float64)
    # 将输入的数据数组 y 转换为连续的内存布局，使用 np.float64 类型
    freqs = np.ascontiguousarray(freqs, dtype=np.float64)
    # 将输入的频率数组 freqs 转换为连续的内存布局，使用 np.float64 类型

    assert x.ndim == 1
    # 确保 x 是一维数组
    assert y.ndim == 1
    # 确保 y 是一维数组
    assert freqs.ndim == 1
    # 确保 freqs 是一维数组

    if precenter:
        # 如果 precenter 为 True，则对 y 减去其均值后调用 _lombscargle 函数
        pgram = _lombscargle(x, y - y.mean(), freqs)
    else:
        # 如果 precenter 为 False，则直接调用 _lombscargle 函数
        pgram = _lombscargle(x, y, freqs)

    if normalize:
        # 如果 normalize 为 True，则对 pgram 进行归一化处理
        pgram *= 2 / np.dot(y, y)

    return pgram
    """
# 使用 periodogram 方法估算信号的功率谱密度
def periodogram(x, fs=1.0, window='boxcar', nfft=None, detrend='constant',
                return_onesided=True, scaling='density', axis=-1):
    """
    Estimate power spectral density using a periodogram.

    Parameters
    ----------
    x : array_like
        Time series of measurement values
    fs : float, optional
        Sampling frequency of the `x` time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be equal to the length
        of the axis over which the periodogram is computed. Defaults
        to 'boxcar'.
    nfft : int, optional
        Length of the FFT used. If `None` the length of `x` will be
        used.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `True`, but for
        complex data, a two-sided spectrum is always returned.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the power spectral density ('density')
        where `Pxx` has units of V**2/Hz and computing the squared magnitude
        spectrum ('spectrum') where `Pxx` has units of V**2, if `x`
        is measured in V and `fs` is measured in Hz. Defaults to
        'density'
    axis : int, optional
        Axis along which the periodogram is computed; the default is
        over the last axis (i.e. ``axis=-1``).

    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    Pxx : ndarray
        Power spectral density or power spectrum of `x`.

    See Also
    --------
    welch: Estimate power spectral density using Welch's method
    lombscargle: Lomb-Scargle periodogram for unevenly sampled data

    Notes
    -----
    Consult the :ref:`tutorial_SpectralAnalysis` section of the :ref:`user_guide`
    for a discussion of the scalings of the power spectral density and
    the magnitude (squared) spectrum.

    .. versionadded:: 0.12.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()

    Generate a test signal, a 2 Vrms sine wave at 1234 Hz, corrupted by
    0.001 V**2/Hz of white noise sampled at 10 kHz.

    >>> fs = 10e3
    >>> N = 1e5
    >>> amp = 2*np.sqrt(2)
    # 设置信号的频率
    freq = 1234.0
    # 计算噪声功率
    noise_power = 0.001 * fs / 2
    # 生成时间序列
    time = np.arange(N) / fs
    # 生成正弦信号
    x = amp*np.sin(2*np.pi*freq*time)
    # 添加高斯白噪声到信号中
    x += rng.normal(scale=np.sqrt(noise_power), size=time.shape)

    # 计算并绘制功率谱密度
    f, Pxx_den = signal.periodogram(x, fs)
    plt.semilogy(f, Pxx_den)
    plt.ylim([1e-7, 1e2])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()

    # 如果我们对频谱密度的后半部分进行平均，排除峰值，我们可以恢复信号上的噪声功率
    np.mean(Pxx_den[25000:])

    # 计算并绘制功率谱
    f, Pxx_spec = signal.periodogram(x, fs, 'flattop', scaling='spectrum')
    plt.figure()
    plt.semilogy(f, np.sqrt(Pxx_spec))
    plt.ylim([1e-4, 1e1])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Linear spectrum [V RMS]')
    plt.show()

    # 功率谱中的峰值高度是RMS振幅的估算
    np.sqrt(Pxx_spec.max())
    
    """
    将输入数据转换为NumPy数组
    """
    x = np.asarray(x)

    if x.size == 0:
        return np.empty(x.shape), np.empty(x.shape)

    if window is None:
        window = 'boxcar'

    if nfft is None:
        nperseg = x.shape[axis]
    elif nfft == x.shape[axis]:
        nperseg = nfft
    elif nfft > x.shape[axis]:
        s = [np.s_[:]]*len(x.shape)
        s[axis] = np.s_[:nfft]
        x = x[tuple(s)]
        nperseg = nfft
        nfft = None

    if hasattr(window, 'size'):
        if window.size != nperseg:
            raise ValueError('窗口的大小必须与指定轴上的输入大小相同')

    # 调用welch函数进行功率谱密度估计
    return welch(x, fs=fs, window=window, nperseg=nperseg, noverlap=0,
                 nfft=nfft, detrend=detrend, return_onesided=return_onesided,
                 scaling=scaling, axis=axis)
# 使用 Welch 方法估算功率谱密度。

def welch(x, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None,
          detrend='constant', return_onesided=True, scaling='density',
          axis=-1, average='mean'):
    r"""
    使用 Welch 方法估算功率谱密度。

    Welch 方法 [1]_ 将数据分成重叠的片段，计算每个片段的修改后的周期图，并对周期图进行平均，
    从而计算功率谱密度估计。

    Parameters
    ----------
    x : array_like
        测量值的时间序列
    fs : float, optional
        `x` 时间序列的采样频率。默认为 1.0。
    window : str or tuple or array_like, optional
        所需的窗口类型。如果 `window` 是字符串或元组，将传递给 `get_window` 以生成窗口值，
        默认情况下为 DFT-even。查看 `get_window` 获取窗口列表和所需参数。如果 `window` 是
        array_like，则直接使用它作为窗口，其长度必须为 `nperseg`。默认为汉宁窗。
    nperseg : int, optional
        每个段的长度。默认为 None，但如果 `window` 是字符串或元组，则设置为 256，
        如果 `window` 是 array_like，则设置为窗口的长度。
    noverlap : int, optional
        段之间重叠的点数。如果为 `None`，则 ``noverlap = nperseg // 2``。默认为 `None`。
    nfft : int, optional
        所使用的 FFT 的长度，如果需要零填充 FFT。如果为 `None`，FFT 长度为 `nperseg`。默认为 `None`。
    detrend : str or function or `False`, optional
        指定如何去趋势化每个段。如果 `detrend` 是字符串，则作为 `type` 参数传递给 `detrend` 函数。
        如果是函数，则接受一个段并返回去趋势化的段。如果 `detrend` 是 `False`，则不执行去趋势化。
        默认为 'constant'。
    return_onesided : bool, optional
        如果为 `True`，对于实数数据返回单边谱。如果为 `False`，返回双边谱。默认为 `True`，
        但对于复杂数据，始终返回双边谱。
    scaling : { 'density', 'spectrum' }, optional
        选择计算功率谱密度 ('density') 或计算幅度谱的平方 ('spectrum')，如果 `x` 是以 V 为单位，
        `fs` 是以 Hz 为单位的话，`Pxx` 的单位为 V**2/Hz。默认为 'density'。
    axis : int, optional
        计算周期图的轴；默认为最后一个轴（即 ``axis=-1``）。
    average : { 'mean', 'median' }, optional
        平均周期图时使用的方法。默认为 'mean'。

        .. versionadded:: 1.2.0

    Returns
    -------
    f : ndarray
        样本频率的数组。
    Pxx : ndarray
        x 的功率谱密度或功率谱。
    See Also
    --------
    periodogram: Simple, optionally modified periodogram
    lombscargle: Lomb-Scargle periodogram for unevenly sampled data

    Notes
    -----
    An appropriate amount of overlap will depend on the choice of window
    and on your requirements. For the default Hann window an overlap of
    50% is a reasonable trade off between accurately estimating the
    signal power, while not over counting any of the data. Narrower
    windows may require a larger overlap.

    If `noverlap` is 0, this method is equivalent to Bartlett's method
    [2]_.

    Consult the :ref:`tutorial_SpectralAnalysis` section of the :ref:`user_guide`
    for a discussion of the scalings of the power spectral density and
    the (squared) magnitude spectrum.

    .. versionadded:: 0.12.0

    References
    ----------
    .. [1] P. Welch, "The use of the fast Fourier transform for the
           estimation of power spectra: A method based on time averaging
           over short, modified periodograms", IEEE Trans. Audio
           Electroacoust. vol. 15, pp. 70-73, 1967.
    .. [2] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",
           Biometrika, vol. 37, pp. 1-16, 1950.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()

    Generate a test signal, a 2 Vrms sine wave at 1234 Hz, corrupted by
    0.001 V**2/Hz of white noise sampled at 10 kHz.

    >>> fs = 10e3
    >>> N = 1e5
    >>> amp = 2*np.sqrt(2)
    >>> freq = 1234.0
    >>> noise_power = 0.001 * fs / 2
    >>> time = np.arange(N) / fs
    >>> x = amp*np.sin(2*np.pi*freq*time)
    >>> x += rng.normal(scale=np.sqrt(noise_power), size=time.shape)

    Compute and plot the power spectral density.

    >>> f, Pxx_den = signal.welch(x, fs, nperseg=1024)
    >>> plt.semilogy(f, Pxx_den)
    >>> plt.ylim([0.5e-3, 1])
    >>> plt.xlabel('frequency [Hz]')
    >>> plt.ylabel('PSD [V**2/Hz]')
    >>> plt.show()

    If we average the last half of the spectral density, to exclude the
    peak, we can recover the noise power on the signal.

    >>> np.mean(Pxx_den[256:])
    0.0009924865443739191

    Now compute and plot the power spectrum.

    >>> f, Pxx_spec = signal.welch(x, fs, 'flattop', 1024, scaling='spectrum')
    >>> plt.figure()
    >>> plt.semilogy(f, np.sqrt(Pxx_spec))
    >>> plt.xlabel('frequency [Hz]')
    >>> plt.ylabel('Linear spectrum [V RMS]')
    >>> plt.show()

    The peak height in the power spectrum is an estimate of the RMS
    amplitude.

    >>> np.sqrt(Pxx_spec.max())
    2.0077340678640727

    If we now introduce a discontinuity in the signal, by increasing the
    amplitude of a small portion of the signal by 50, we can see the
    corruption of the mean average power spectral density, but using a
    median average better estimates the normal behaviour.

    >>> x[int(N//2):int(N//2)+10] *= 50.
    # 使用信号处理库中的 welch 函数计算信号 x 的功率谱密度估计
    >>> f, Pxx_den = signal.welch(x, fs, nperseg=1024)

    # 使用 welch 函数计算信号 x 的功率谱密度估计，使用中位数进行平均
    >>> f_med, Pxx_den_med = signal.welch(x, fs, nperseg=1024, average='median')

    # 绘制功率谱密度估计结果的半对数图（y轴对数），标记为平均值
    >>> plt.semilogy(f, Pxx_den, label='mean')

    # 绘制功率谱密度估计结果的半对数图（y轴对数），标记为中位数
    >>> plt.semilogy(f_med, Pxx_den_med, label='median')

    # 设置 y 轴的显示范围
    >>> plt.ylim([0.5e-3, 1])

    # 设置 x 轴的标签
    >>> plt.xlabel('frequency [Hz]')

    # 设置 y 轴的标签
    >>> plt.ylabel('PSD [V**2/Hz]')

    # 显示图例
    >>> plt.legend()

    # 显示绘制的图形
    >>> plt.show()

    """
    # 使用交叉功率谱密度估计函数 csd 计算信号 x 的频率和功率谱密度估计，返回实部
    freqs, Pxx = csd(x, x, fs=fs, window=window, nperseg=nperseg,
                     noverlap=noverlap, nfft=nfft, detrend=detrend,
                     return_onesided=return_onesided, scaling=scaling,
                     axis=axis, average=average)

    # 返回计算得到的频率和功率谱密度估计的实部
    return freqs, Pxx.real
# 使用Welch方法估算交叉功率谱密度Pxy

def csd(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None,
        detrend='constant', return_onesided=True, scaling='density',
        axis=-1, average='mean'):
    r"""
    Estimate the cross power spectral density, Pxy, using Welch's method.

    Parameters
    ----------
    x : array_like
        Time series of measurement values
    y : array_like
        Time series of measurement values
    fs : float, optional
        Sampling frequency of the `x` and `y` time series. Defaults
        to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap: int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `True`, but for
        complex data, a two-sided spectrum is always returned.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the cross spectral density ('density')
        where `Pxy` has units of V**2/Hz and computing the cross spectrum
        ('spectrum') where `Pxy` has units of V**2, if `x` and `y` are
        measured in V and `fs` is measured in Hz. Defaults to 'density'
    axis : int, optional
        Axis along which the CSD is computed for both inputs; the
        default is over the last axis (i.e. ``axis=-1``).
    average : { 'mean', 'median' }, optional
        Method to use when averaging periodograms. If the spectrum is
        complex, the average is computed separately for the real and
        imaginary parts. Defaults to 'mean'.

        .. versionadded:: 1.2.0

    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    Pxy : ndarray
        Cross spectral density or cross power spectrum of x,y.

    See Also
    --------
    """
    # periodogram: 简单的周期图，可选择修改
    # lombscargle: 用于不均匀采样数据的Lomb-Scargle周期图
    # welch: 使用Welch方法计算功率谱密度。[相当于csd(x, x)]
    # coherence: 使用Welch方法计算幅度平方相干度

    # 注意事项
    # -----
    # 按照惯例，Pxy是通过X的共轭FFT乘以Y的FFT计算得到的。
    # 如果输入序列长度不同，较短的序列将会被零填充以匹配长度。
    # 合适的重叠量取决于窗口的选择和您的要求。对于默认的Hann窗口，50%的重叠是在准确估计信号功率和不过度计数任何数据之间的合理折衷。
    # 较窄的窗口可能需要更大的重叠。
    
    # 请参阅 :ref:`tutorial_SpectralAnalysis` 部分的 :ref:`user_guide`，讨论谱密度和（振幅）频谱的标度问题。

    # .. versionadded:: 0.16.0

    # 参考文献
    # ----------
    # .. [1] P. Welch, "The use of the fast Fourier transform for the
    #        estimation of power spectra: A method based on time averaging
    #        over short, modified periodograms", IEEE Trans. Audio
    #        Electroacoust. vol. 15, pp. 70-73, 1967.
    # .. [2] Rabiner, Lawrence R., and B. Gold. "Theory and Application of
    #        Digital Signal Processing" Prentice-Hall, pp. 414-419, 1975

    # 示例
    # --------
    # >>> import numpy as np
    # >>> from scipy import signal
    # >>> import matplotlib.pyplot as plt
    # >>> rng = np.random.default_rng()

    # 生成两个具有某些共同特征的测试信号。

    # >>> fs = 10e3
    # >>> N = 1e5
    # >>> amp = 20
    # >>> freq = 1234.0
    # >>> noise_power = 0.001 * fs / 2
    # >>> time = np.arange(N) / fs
    # >>> b, a = signal.butter(2, 0.25, 'low')
    # >>> x = rng.normal(scale=np.sqrt(noise_power), size=time.shape)
    # >>> y = signal.lfilter(b, a, x)
    # >>> x += amp*np.sin(2*np.pi*freq*time)
    # >>> y += rng.normal(scale=0.1*np.sqrt(noise_power), size=time.shape)

    # 计算并绘制交叉谱密度的幅度。

    # >>> f, Pxy = signal.csd(x, y, fs, nperseg=1024)
    # >>> plt.semilogy(f, np.abs(Pxy))
    # >>> plt.xlabel('frequency [Hz]')
    # >>> plt.ylabel('CSD [V**2/Hz]')
    # >>> plt.show()
    """
    # 使用_spectral_helper函数计算频谱密度
    freqs, _, Pxy = _spectral_helper(x, y, fs, window, nperseg, noverlap,
                                     nfft, detrend, return_onesided, scaling,
                                     axis, mode='psd')

    # 对窗口平均
    # 检查 Pxy 的维度是否大于等于2且非空
    if len(Pxy.shape) >= 2 and Pxy.size > 0:
        # 检查 Pxy 的最后一个维度是否大于1
        if Pxy.shape[-1] > 1:
            # 如果 average 参数为 'median'
            if average == 'median':
                # 计算中位数偏差修正因子
                bias = _median_bias(Pxy.shape[-1])
                # 如果 Pxy 是复数对象，则分别对实部和虚部取中位数
                if np.iscomplexobj(Pxy):
                    Pxy = (np.median(np.real(Pxy), axis=-1)
                           + 1j * np.median(np.imag(Pxy), axis=-1))
                else:
                    # 否则直接对 Pxy 取中位数
                    Pxy = np.median(Pxy, axis=-1)
                # 对 Pxy 应用偏差修正
                Pxy /= bias
            # 如果 average 参数为 'mean'，则对 Pxy 沿最后一个维度取均值
            elif average == 'mean':
                Pxy = Pxy.mean(axis=-1)
            else:
                # 如果 average 参数既不是 'median' 也不是 'mean'，抛出数值错误
                raise ValueError(f'average must be "median" or "mean", got {average}')
        else:
            # 如果 Pxy 的最后一个维度不大于1，则重新调整 Pxy 的形状
            Pxy = np.reshape(Pxy, Pxy.shape[:-1])

    # 返回计算结果中的频率 freqs 和处理后的 Pxy
    return freqs, Pxy
# 计算一个频谱图的函数（遗留函数）

Spectrograms可以用作可视化非稳态信号随时间变化的频率内容的一种方式。

.. legacy:: function

    :class:`ShortTimeFFT` 是一个新的STFT / ISTFT实现，具有更多功能，还包括一个 :meth:`~ShortTimeFFT.spectrogram` 方法。
    在 :ref:`tutorial_stft` 部分的 :ref:`user_guide` 中可以找到这些实现之间的 :ref:`comparison <tutorial_stft_legacy_stft>`。

Parameters
----------
x : array_like
    测量值时间序列
fs : float, optional
    `x` 时间序列的采样频率。默认为 1.0。
window : str or tuple or array_like, optional
    所需的窗口类型。如果 `window` 是一个字符串或元组，则传递给 `get_window` 以生成窗口值，默认情况下是DFT-even。
    有关窗口和必需参数的列表，请参阅 `get_window`。
    如果 `window` 是 array_like，则直接使用它作为窗口，并且其长度必须为 nperseg。
    默认为一个带有形状参数为0.25的Tukey窗口。
nperseg : int, optional
    每个段的长度。默认为 None，但如果 window 是 str 或 tuple，则设置为 256；如果 window 是 array_like，则设置为窗口的长度。
noverlap : int, optional
    段之间重叠的点数。如果 `None`，则 ``noverlap = nperseg // 8``。默认为 `None`。
nfft : int, optional
    所使用的FFT的长度，如果需要进行零填充的FFT。如果 `None`，FFT长度为 `nperseg`。默认为 `None`。
detrend : str or function or `False`, optional
    指定如何去趋势化每个段。如果 `detrend` 是一个字符串，则作为 `detrend` 函数的 `type` 参数传递。
    如果它是一个函数，则它接受一个段并返回去趋势化后的段。
    如果 `detrend` 是 `False`，则不进行去趋势化。默认为 'constant'。
return_onesided : bool, optional
    如果为 `True`，对于实数数据返回单侧频谱。如果为 `False`，返回双侧频谱。默认为 `True`，
    但对于复数数据，始终返回双侧频谱。
scaling : { 'density', 'spectrum' }, optional
    选择计算功率谱密度 ('density') 还是计算功率谱 ('spectrum')。
    如果 `x` 以 V 为单位，`fs` 以 Hz 为单位，则 'density' 的单位为 V**2/Hz，'spectrum' 的单位为 V**2。
    默认为 'density'。
    axis : int, optional
        要计算声谱图的轴；默认是最后一个轴（即 ``axis=-1``）。
    mode : str, optional
        定义预期返回的数值类型。选项有 ['psd', 'complex', 'magnitude', 'angle', 'phase']。
        'complex' 等同于 `stft` 的输出，没有填充或边界扩展。
        'magnitude' 返回 STFT 的绝对幅度。
        'angle' 和 'phase' 返回 STFT 的复角度，有和无解包装，分别对应。
        
    Returns
    -------
    f : ndarray
        样本频率的数组。
    t : ndarray
        分段时间的数组。
    Sxx : ndarray
        x 的声谱图。默认情况下，Sxx 的最后一个轴对应于分段时间。
    
    See Also
    --------
    periodogram: 简单的，可选修改后的周期图
    lombscargle: 不均匀采样数据的Lomb-Scargle周期图
    welch: Welch方法计算功率谱密度
    csd: Welch方法计算的交叉谱密度
    ShortTimeFFT: 提供更多功能的新版STFT/ISTFT实现，包括 :meth:`~ShortTimeFFT.spectrogram` 方法。

    Notes
    -----
    合适的重叠量取决于窗口的选择和您的需求。与welch方法相反，该方法平均整个数据流，当计算声谱图时，
    可能希望使用较小的重叠（或根本不重叠），以保持各个段之间的某种统计独立性。
    出于这个原因，默认窗口是图基窗口，每端重叠1/8个窗口长度。

    .. versionadded:: 0.16.0

    References
    ----------
    .. [1] Oppenheim, Alan V., Ronald W. Schafer, John R. Buck
           "Discrete-Time Signal Processing", Prentice Hall, 1999.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> from scipy.fft import fftshift
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()

    生成一个测试信号，一个2 Vrms的正弦波，其频率围绕3kHz缓慢调制，受到以10 kHz采样的指数衰减白噪声的污染。

    >>> fs = 10e3
    >>> N = 1e5
    >>> amp = 2 * np.sqrt(2)
    >>> noise_power = 0.01 * fs / 2
    >>> time = np.arange(N) / float(fs)
    >>> mod = 500*np.cos(2*np.pi*0.25*time)
    >>> carrier = amp * np.sin(2*np.pi*3e3*time + mod)
    >>> noise = rng.normal(scale=np.sqrt(noise_power), size=time.shape)
    >>> noise *= np.exp(-time/5)
    >>> x = carrier + noise

    计算并绘制声谱图。

    >>> f, t, Sxx = signal.spectrogram(x, fs)
    >>> plt.pcolormesh(t, f, Sxx, shading='gouraud')
    >>> plt.ylabel('Frequency [Hz]')
    >>> plt.xlabel('Time [sec]')
    >>> plt.show()
    # 定义可接受的模式列表，用于检查输入的频谱分析模式是否有效
    modelist = ['psd', 'complex', 'magnitude', 'angle', 'phase']

    # 如果输入的模式不在允许的模式列表中，则引发值错误异常
    if mode not in modelist:
        raise ValueError(f'unknown value for mode {mode}, must be one of {modelist}')

    # 确定 nperseg 的默认值，这是窗口长度的一个重要参数
    window, nperseg = _triage_segments(window, nperseg,
                                       input_length=x.shape[axis])

    # 如果未指定 noverlap 参数，则根据 nperseg 计算默认的重叠长度
    if noverlap is None:
        noverlap = nperseg // 8

    # 根据不同的频谱分析模式选择不同的处理方式
    if mode == 'psd':
        # 计算功率谱密度（PSD）
        freqs, time, Sxx = _spectral_helper(x, x, fs, window, nperseg,
                                            noverlap, nfft, detrend,
                                            return_onesided, scaling, axis,
                                            mode='psd')
    else:
        # 计算短时傅里叶变换（STFT）或复数形式的频谱数据
        freqs, time, Sxx = _spectral_helper(x, x, fs, window, nperseg,
                                            noverlap, nfft, detrend,
                                            return_onesided, scaling, axis,
                                            mode='stft')

        # 根据模式对频谱数据进行进一步处理
        if mode == 'magnitude':
            # 转换为幅度谱
            Sxx = np.abs(Sxx)
        elif mode in ['angle', 'phase']:
            # 转换为相位谱或角度谱
            Sxx = np.angle(Sxx)
            if mode == 'phase':
                # 如果是相位谱，对相位进行解包裹处理
                # Sxx 在时间步长上有一个额外的维度
                if axis < 0:
                    axis -= 1
                Sxx = np.unwrap(Sxx, axis=axis)

    # 返回频率、时间和处理后的频谱数据
    return freqs, time, Sxx
# 检查常量重叠加法（COLA）约束是否得到满足
def check_COLA(window, nperseg, noverlap, tol=1e-10):
    r"""Check whether the Constant OverLap Add (COLA) constraint is met.

    Parameters
    ----------
    window : str or tuple or array_like
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg.
    nperseg : int
        Length of each segment.
    noverlap : int
        Number of points to overlap between segments.
    tol : float, optional
        The allowed variance of a bin's weighted sum from the median bin
        sum.

    Returns
    -------
    verdict : bool
        `True` if chosen combination satisfies COLA within `tol`,
        `False` otherwise

    See Also
    --------
    check_NOLA: Check whether the Nonzero Overlap Add (NOLA) constraint is met
    stft: Short Time Fourier Transform
    istft: Inverse Short Time Fourier Transform

    Notes
    -----
    In order to enable inversion of an STFT via the inverse STFT in
    `istft`, it is sufficient that the signal windowing obeys the constraint of
    "Constant OverLap Add" (COLA). This ensures that every point in the input
    data is equally weighted, thereby avoiding aliasing and allowing full
    reconstruction.

    Some examples of windows that satisfy COLA:
        - Rectangular window at overlap of 0, 1/2, 2/3, 3/4, ...
        - Bartlett window at overlap of 1/2, 3/4, 5/6, ...
        - Hann window at 1/2, 2/3, 3/4, ...
        - Any Blackman family window at 2/3 overlap
        - Any window with ``noverlap = nperseg-1``

    A very comprehensive list of other windows may be found in [2]_,
    wherein the COLA condition is satisfied when the "Amplitude
    Flatness" is unity.

    .. versionadded:: 0.19.0

    References
    ----------
    .. [1] Julius O. Smith III, "Spectral Audio Signal Processing", W3K
           Publishing, 2011,ISBN 978-0-9745607-3-1.
    .. [2] G. Heinzel, A. Ruediger and R. Schilling, "Spectrum and
           spectral density estimation by the Discrete Fourier transform
           (DFT), including a comprehensive list of window functions and
           some new at-top windows", 2002,
           http://hdl.handle.net/11858/00-001M-0000-0013-557A-5

    Examples
    --------
    >>> from scipy import signal

    Confirm COLA condition for rectangular window of 75% (3/4) overlap:

    >>> signal.check_COLA(signal.windows.boxcar(100), 100, 75)
    True

    COLA is not true for 25% (1/4) overlap, though:

    >>> signal.check_COLA(signal.windows.boxcar(100), 100, 25)
    False

    "Symmetrical" Hann window (for filter design) is not COLA:

    >>> signal.check_COLA(signal.windows.hann(120, sym=True), 120, 60)
    False

    "Periodic" or "DFT-even" Hann window (for FFT analysis) is COLA for
    """
    检查给定信号是否满足COLA（重叠加法性）条件。

    >>> signal.check_COLA(signal.windows.hann(120, sym=False), 120, 60)
    返回 True，表明窗口长度为120的Hann窗口，60的重叠满足COLA条件。

    >>> signal.check_COLA(signal.windows.hann(120, sym=False), 120, 80)
    返回 True，表明窗口长度为120的Hann窗口，80的重叠满足COLA条件。

    >>> signal.check_COLA(signal.windows.hann(120, sym=False), 120, 90)
    返回 True，表明窗口长度为120的Hann窗口，90的重叠满足COLA条件。

    """
    # 将nperseg转换为整数
    nperseg = int(nperseg)

    # 如果nperseg小于1，抛出数值错误
    if nperseg < 1:
        raise ValueError('nperseg must be a positive integer')

    # 将noverlap转换为整数
    noverlap = int(noverlap)

    # 如果noverlap大于等于nperseg，抛出数值错误
    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')

    # 如果window是字符串或元组，则获取对应的窗口函数
    if isinstance(window, str) or type(window) is tuple:
        win = get_window(window, nperseg)
    else:
        # 否则，将window转换为NumPy数组
        win = np.asarray(window)
        # 检查window是否为1维数组
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        # 检查window的长度是否与nperseg相等
        if win.shape[0] != nperseg:
            raise ValueError('window must have length of nperseg')

    # 计算步长（窗口长度减去重叠长度）
    step = nperseg - noverlap

    # 计算窗口在不重叠部分的和
    binsums = sum(win[ii*step:(ii+1)*step] for ii in range(nperseg//step))

    # 如果窗口长度不能整除步长，则将剩余部分加到对应位置
    if nperseg % step != 0:
        binsums[:nperseg % step] += win[-(nperseg % step):]

    # 计算偏差，即窗口和的中位数与窗口和的差值
    deviation = binsums - np.median(binsums)

    # 返回偏差的绝对值的最大值是否小于tol
    return np.max(np.abs(deviation)) < tol
def check_NOLA(window, nperseg, noverlap, tol=1e-10):
    r"""Check whether the Nonzero Overlap Add (NOLA) constraint is met.

    Parameters
    ----------
    window : str or tuple or array_like
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg.
    nperseg : int
        Length of each segment.
    noverlap : int
        Number of points to overlap between segments.
    tol : float, optional
        The allowed variance of a bin's weighted sum from the median bin
        sum.

    Returns
    -------
    verdict : bool
        `True` if chosen combination satisfies the NOLA constraint within
        `tol`, `False` otherwise

    See Also
    --------
    check_COLA: Check whether the Constant OverLap Add (COLA) constraint is met
    stft: Short Time Fourier Transform
    istft: Inverse Short Time Fourier Transform

    Notes
    -----
    In order to enable inversion of an STFT via the inverse STFT in
    `istft`, the signal windowing must obey the constraint of "nonzero
    overlap add" (NOLA):

    .. math:: \sum_{t}w^{2}[n-tH] \ne 0

    for all :math:`n`, where :math:`w` is the window function, :math:`t` is the
    frame index, and :math:`H` is the hop size (:math:`H` = `nperseg` -
    `noverlap`).

    This ensures that the normalization factors in the denominator of the
    overlap-add inversion equation are not zero. Only very pathological windows
    will fail the NOLA constraint.

    .. versionadded:: 1.2.0

    References
    ----------
    .. [1] Julius O. Smith III, "Spectral Audio Signal Processing", W3K
           Publishing, 2011,ISBN 978-0-9745607-3-1.
    .. [2] G. Heinzel, A. Ruediger and R. Schilling, "Spectrum and
           spectral density estimation by the Discrete Fourier transform
           (DFT), including a comprehensive list of window functions and
           some new at-top windows", 2002,
           http://hdl.handle.net/11858/00-001M-0000-0013-557A-5

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal

    Confirm NOLA condition for rectangular window of 75% (3/4) overlap:

    >>> signal.check_NOLA(signal.windows.boxcar(100), 100, 75)
    True

    NOLA is also true for 25% (1/4) overlap:

    >>> signal.check_NOLA(signal.windows.boxcar(100), 100, 25)
    True

    "Symmetrical" Hann window (for filter design) is also NOLA:

    >>> signal.check_NOLA(signal.windows.hann(120, sym=True), 120, 60)
    True

    As long as there is overlap, it takes quite a pathological window to fail
    NOLA:

    >>> w = np.ones(64, dtype="float")
    >>> w[::2] = 0
    >>> signal.check_NOLA(w, 64, 32)
    False

    If there is not enough overlap, a window with zeros at the ends will not
    work:
    ```
    # 检查信号窗口是否满足 NOLA（Non-overlapping Additivity）条件
    >>> signal.check_NOLA(signal.windows.hann(64), 64, 0)
    False
    >>> signal.check_NOLA(signal.windows.hann(64), 64, 1)
    False
    >>> signal.check_NOLA(signal.windows.hann(64), 64, 2)
    True

    """
    # 将 nperseg 转换为整数类型
    nperseg = int(nperseg)

    # 如果 nperseg 小于 1，则抛出数值错误
    if nperseg < 1:
        raise ValueError('nperseg must be a positive integer')

    # 如果 noverlap 大于等于 nperseg，则抛出数值错误
    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg')
    # 如果 noverlap 小于 0，则抛出数值错误
    if noverlap < 0:
        raise ValueError('noverlap must be a nonnegative integer')
    # 将 noverlap 转换为整数类型
    noverlap = int(noverlap)

    # 如果 window 是字符串或者元组，则通过 get_window 函数获取窗口向量
    if isinstance(window, str) or type(window) is tuple:
        win = get_window(window, nperseg)
    else:
        # 否则，将 window 转换为 NumPy 数组
        win = np.asarray(window)
        # 如果 win 的维度不为 1，则抛出数值错误
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        # 如果 win 的长度不等于 nperseg，则抛出数值错误
        if win.shape[0] != nperseg:
            raise ValueError('window must have length of nperseg')

    # 计算步长为 nperseg - noverlap
    step = nperseg - noverlap
    # 计算每个分段的平方和并求和
    binsums = sum(win[ii*step:(ii+1)*step]**2 for ii in range(nperseg//step))

    # 如果 nperseg 不能被 step 整除，则对剩余部分进行平方和计算
    if nperseg % step != 0:
        binsums[:nperseg % step] += win[-(nperseg % step):]**2

    # 返回是否满足 NOLA 条件的布尔值
    return np.min(binsums) > tol
# 计算短时傅里叶变换（STFT）的函数，用于分析非平稳信号的频率和相位随时间的变化
def stft(x, fs=1.0, window='hann', nperseg=256, noverlap=None, nfft=None,
         detrend=False, return_onesided=True, boundary='zeros', padded=True,
         axis=-1, scaling='spectrum'):
    r"""Compute the Short Time Fourier Transform (legacy function).

    STFTs can be used as a way of quantifying the change of a
    nonstationary signal's frequency and phase content over time.

    .. legacy:: function

        `ShortTimeFFT` is a newer STFT / ISTFT implementation with more
        features. A :ref:`comparison <tutorial_stft_legacy_stft>` between the
        implementations can be found in the :ref:`tutorial_stft` section of the
        :ref:`user_guide`.

    Parameters
    ----------
    x : array_like
        Time series of measurement values
    fs : float, optional
        Sampling frequency of the `x` time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.
    nperseg : int, optional
        Length of each segment. Defaults to 256.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`. When
        specified, the COLA constraint must be met (see Notes below).
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to `False`.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `True`, but for
        complex data, a two-sided spectrum is always returned.
    boundary : str or None, optional
        Specifies whether the input signal is extended at both ends, and
        how to generate the new values, in order to center the first
        windowed segment on the first input point. This has the benefit
        of enabling reconstruction of the first input point when the
        employed window function starts at zero. Valid options are
        ``['even', 'odd', 'constant', 'zeros', None]``. Defaults to
        'zeros', for zero padding extension. I.e. ``[1, 2, 3, 4]`` is
        extended to ``[0, 1, 2, 3, 4, 0]`` for ``nperseg=3``.
    padded : bool, optional
        Not used in the legacy implementation. Defaults to `True`.
    axis : int, optional
        Not used in the legacy implementation. Defaults to `-1`.
    scaling : {'spectrum', 'density'}, optional
        Not used in the legacy implementation. Defaults to 'spectrum'.
    padded : bool, optional
        # 指定输入信号是否在末尾进行零填充，以确保信号正好能够被整数个窗口段包含在内，从而所有信号都包含在输出中。
        Defaults to `True`. Padding occurs after boundary extension, if
        `boundary` is not `None`, and `padded` is `True`, as is the
        default.
    axis : int, optional
        # 计算STFT的轴向， 默认为最后一个轴（即 ``axis=-1``）。
    scaling: {'spectrum', 'psd'}
        # 默认的 'spectrum' 缩放选项允许将 `Zxx` 的每个频率线解释为幅度谱。
        'psd' 选项将每行缩放为功率谱密度 - 允许通过数值积分 ``abs(Zxx)**2`` 计算信号的能量。

        .. versionadded:: 1.9.0

    Returns
    -------
    f : ndarray
        # 样本频率数组。
    t : ndarray
        # 段时间数组。
    Zxx : ndarray
        # `x` 的STFT。默认情况下，`Zxx` 的最后一个轴对应于段时间。

    See Also
    --------
    istft: 逆短时傅里叶变换
    ShortTimeFFT: 提供更多功能的新版STFT/ISTFT实现。
    check_COLA: 检查是否满足常数重叠添加（COLA）约束
    check_NOLA: 检查是否满足非零重叠添加（NOLA）约束
    welch: Welch方法计算功率谱密度。
    spectrogram: Welch方法计算的谱图。
    csd: Welch方法计算的交叉谱密度。
    lombscargle: 不均匀采样数据的Lomb-Scargle周期图。

    Notes
    -----
    # 为了通过 `istft` 实现STFT的反演，信号的窗口必须遵循“非零重叠添加”（NOLA）约束，且输入信号必须有完整的窗口覆盖。
    （即 ``(x.shape[axis] - nperseg) % (nperseg-noverlap) == 0``）。`padded` 参数可用于实现这一点。

    给定时域信号 :math:`x[n]`，窗口 :math:`w[n]`，和跳跃大小 :math:`H` = `nperseg - noverlap`，
    在时间索引 :math:`t` 处的窗口帧为

    .. math:: x_{t}[n]=x[n]w[n-tH]

    重叠添加（OLA）重构方程为

    .. math:: x[n]=\frac{\sum_{t}x_{t}[n]w[n-tH]}{\sum_{t}w^{2}[n-tH]}

    NOLA约束确保OLA重构方程中出现的每个归一化项的分母都不为零。
    可以使用 `check_NOLA` 来测试 `window`、`nperseg` 和 `noverlap` 的选择是否满足这一约束。

    .. versionadded:: 0.19.0

    References
    ----------
    .. [1] Oppenheim, Alan V., Ronald W. Schafer, John R. Buck
           "Discrete-Time Signal Processing", Prentice Hall, 1999.
    """
    根据给定的参数计算信号的短时傅里叶变换（STFT）并返回频率、时间和频谱信息。
    
    Parameters
    ----------
    x : array_like
        输入信号数组。
    fs : float
        信号的采样频率。
    nperseg : int
        每个段的长度（窗口的长度）。
    return_onesided : bool, optional
        如果为 True，则返回单侧频谱，否则返回双侧频谱。默认为 True。
    scaling : {'density', 'spectrum'}, optional
        频谱密度的标度方式。'density' 对应 PSD（功率谱密度），'spectrum' 对应普通频谱。默认为 'density'。
    axis : int, optional
        沿着哪个轴计算 STFT。默认为 -1，即最后一个轴。
    window : str or tuple or array_like, optional
        窗口函数。字符串或者窗口的数据。默认为 'hann'。
    noverlap : int, optional
        重叠段的长度。默认为 None。
    nfft : int, optional
        FFT 的点数。默认为 None，使用 nperseg。
    detrend : str or function or False, optional
        消除线性趋势的方法。默认为 False。
    mode : {'stft', 'psd', 'complex'}, optional
        返回的数据模式：'stft' 返回幅度和相位；'psd' 返回功率谱密度；'complex' 返回幅度和相位。默认为 'stft'。
    boundary : str or None, optional
        插值的边界处理方式。默认为 None。
    padded : bool, optional
        是否对输入信号进行填充。默认为 True。
    
    Returns
    -------
    freqs : ndarray
        频率轴上的频率值。
    time : ndarray
        时间轴上的时间值。
    Zxx : ndarray
        STFT 的结果，可以是幅度、相位或者复数形式，取决于 mode 参数。
    
    Raises
    ------
    ValueError
        如果 scaling 参数既不是 'spectrum' 也不是 'psd'，则抛出该异常。
    
    Notes
    -----
    根据所选的 scaling 参数，可以得到普通频谱或者功率谱密度（PSD）。
    """
    if scaling == 'psd':
        scaling = 'density'  # 将 'psd' 标度转换为 'density'
    elif scaling != 'spectrum':
        raise ValueError(f"Parameter {scaling=} not in ['spectrum', 'psd']!")  # 如果 scaling 不是 'spectrum' 或 'psd'，则抛出异常
    
    # 调用内部函数 _spectral_helper 计算信号的 STFT，并获取返回的频率、时间和频谱信息
    freqs, time, Zxx = _spectral_helper(x, x, fs, window, nperseg, noverlap,
                                        nfft, detrend, return_onesided,
                                        scaling=scaling, axis=axis,
                                        mode='stft', boundary=boundary,
                                        padded=padded)
    
    return freqs, time, Zxx  # 返回计算得到的频率、时间和频谱信息
def istft(Zxx, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None,
          input_onesided=True, boundary=True, time_axis=-1, freq_axis=-2,
          scaling='spectrum'):
    r"""Perform the inverse Short Time Fourier transform (legacy function).

    .. legacy:: function

        `ShortTimeFFT` is a newer STFT / ISTFT implementation with more
        features. A :ref:`comparison <tutorial_stft_legacy_stft>` between the
        implementations can be found in the :ref:`tutorial_stft` section of the
        :ref:`user_guide`.

    Parameters
    ----------
    Zxx : array_like
        STFT of the signal to be reconstructed. If a purely real array
        is passed, it will be cast to a complex data type.
    fs : float, optional
        Sampling frequency of the time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window. Must match the window used to generate the
        STFT for faithful inversion.
    nperseg : int, optional
        Number of data points corresponding to each STFT segment. This
        parameter must be specified if the number of data points per
        segment is odd, or if the STFT was padded via ``nfft >
        nperseg``. If `None`, the value depends on the shape of
        `Zxx` and `input_onesided`. If `input_onesided` is `True`,
        ``nperseg=2*(Zxx.shape[freq_axis] - 1)``. Otherwise,
        ``nperseg=Zxx.shape[freq_axis]``. Defaults to `None`.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`, half
        of the segment length. Defaults to `None`. When specified, the
        COLA constraint must be met (see Notes below), and should match
        the parameter used to generate the STFT. Defaults to `None`.
    nfft : int, optional
        Number of FFT points corresponding to each STFT segment. This
        parameter must be specified if the STFT was padded via ``nfft >
        nperseg``. If `None`, the default values are the same as for
        `nperseg`, detailed above, with one exception: if
        `input_onesided` is True and
        ``nperseg==2*Zxx.shape[freq_axis] - 1``, `nfft` also takes on
        that value. This case allows the proper inversion of an
        odd-length unpadded STFT using ``nfft=None``. Defaults to
        `None`.
    input_onesided : bool, optional
        If `True`, interpret the input array as one-sided FFTs, such
        as is returned by `stft` with ``return_onesided=True`` and
        `numpy.fft.rfft`. If `False`, interpret the input as a a
        two-sided FFT. Defaults to `True`.
    """

    # 实现逆短时傅里叶变换（STFT的反变换）
    # 此函数是遗留函数，有新功能的STFT / ISTFT实现请使用ShortTimeFFT
    # 更多信息和实现对比请参考用户指南中的教程

    # 在这里实现逆STFT的具体算法，还可以根据参数进行调整
    pass
    boundary : bool, optional
        指定输入信号是否在其边界处通过向 `stft` 提供非 `None` 的 ``boundary`` 参数而得到扩展。默认为 `True`。

    time_axis : int, optional
        STFT 的时间片段所在的轴；默认为最后一个轴（即 ``axis=-1``）。

    freq_axis : int, optional
        STFT 的频率轴所在的轴；默认为倒数第二个轴（即 ``axis=-2``）。

    scaling: {'spectrum', 'psd'}
        'spectrum' 缩放选项允许将 `Zxx` 的每个频率线解释为幅度谱。'psd' 选项将每条线缩放为功率谱密度，它允许通过对 ``abs(Zxx)**2`` 进行数值积分来计算信号的能量。

    Returns
    -------
    t : ndarray
        输出数据的时间数组。
    x : ndarray
        `Zxx` 的逆STFT。

    See Also
    --------
    stft: 短时傅里叶变换
    ShortTimeFFT: 提供更多功能的新版STFT/逆STFT实现。
    check_COLA: 检查是否满足常数重叠添加（COLA）约束。
    check_NOLA: 检查是否满足非零重叠添加（NOLA）约束。

    Notes
    -----
    为了通过逆STFT进行STFT的反演，信号窗口必须遵守“非零重叠添加”（NOLA）约束：

    .. math:: \sum_{t}w^{2}[n-tH] \ne 0

    这确保了重叠添加重构方程的分母中出现的归一化因子不为零：

    .. math:: x[n]=\frac{\sum_{t}x_{t}[n]w[n-tH]}{\sum_{t}w^{2}[n-tH]}

    可以使用 `check_NOLA` 函数检查NOLA约束。

    经过修改（例如掩码或其他方式），的STFT不能保证对应于完全可实现的信号。此函数实现了基于最小二乘估计算法的逆STFT，该算法详细说明在 [2]_ 中，其生成的信号使得返回信号的STFT与修改后的STFT之间的均方误差最小化。

    .. versionadded:: 0.19.0

    References
    ----------
    .. [1] Oppenheim, Alan V., Ronald W. Schafer, John R. Buck
           "Discrete-Time Signal Processing", Prentice Hall, 1999.
    .. [2] Daniel W. Griffin, Jae S. Lim "Signal Estimation from
           Modified Short-Time Fourier Transform", IEEE 1984,
           10.1109/TASSP.1984.1164317

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()

    生成一个测试信号，一个受到50Hz正弦波干扰的2 Vrms的信号，其白噪声功率为0.001 V**2/Hz，采样频率为1024 Hz。

    >>> fs = 1024
    >>> N = 10*fs
    >>> nperseg = 512
    >>> amp = 2 * np.sqrt(2)
    >>> noise_power = 0.001 * fs / 2
    >>> time = np.arange(N) / float(fs)
    >>> carrier = amp * np.sin(2*np.pi*50*time)
    # 用指定的均值和形状创建正态分布噪声
    >>> noise = rng.normal(scale=np.sqrt(noise_power),
    ...                    size=time.shape)
    
    # 将噪声叠加到载波信号上，形成含噪声的信号
    >>> x = carrier + noise

    # 计算短时傅里叶变换（STFT），并绘制其幅度图
    >>> f, t, Zxx = signal.stft(x, fs=fs, nperseg=nperseg)
    >>> plt.figure()
    >>> plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp, shading='gouraud')
    >>> plt.ylim([f[1], f[-1]])
    >>> plt.title('STFT Magnitude')
    >>> plt.ylabel('Frequency [Hz]')
    >>> plt.xlabel('Time [sec]')
    >>> plt.yscale('log')
    >>> plt.show()

    # 将幅度低于载波幅度的10%的分量置零，然后通过逆STFT转换回时间序列
    >>> Zxx = np.where(np.abs(Zxx) >= amp/10, Zxx, 0)
    >>> _, xrec = signal.istft(Zxx, fs)

    # 将清洁后的信号与原始信号和真实载波信号进行比较
    >>> plt.figure()
    >>> plt.plot(time, x, time, xrec, time, carrier)
    >>> plt.xlim([2, 2.1])
    >>> plt.xlabel('Time [sec]')
    >>> plt.ylabel('Signal')
    >>> plt.legend(['Carrier + Noise', 'Filtered via STFT', 'True Carrier'])
    >>> plt.show()

    # 注意，清洁后的信号不像原始信号那样突然开始，因为一些瞬态的系数也被移除了
    >>> plt.figure()
    >>> plt.plot(time, x, time, xrec, time, carrier)
    >>> plt.xlim([0, 0.1])
    >>> plt.xlabel('Time [sec]')
    >>> plt.ylabel('Signal')
    >>> plt.legend(['Carrier + Noise', 'Filtered via STFT', 'True Carrier'])
    >>> plt.show()

    """
    # 确保输入是适当复杂数据类型的ndarray
    Zxx = np.asarray(Zxx) + 0j
    freq_axis = int(freq_axis)
    time_axis = int(time_axis)

    # 如果STFT的维度小于2，则引发值错误
    if Zxx.ndim < 2:
        raise ValueError('Input stft must be at least 2d!')

    # 如果时间轴和频率轴相同，则引发值错误
    if freq_axis == time_axis:
        raise ValueError('Must specify differing time and frequency axes!')

    # 获取分段数
    nseg = Zxx.shape[time_axis]

    # 如果输入是单边的，假设分段长度是偶数
    if input_onesided:
        n_default = 2*(Zxx.shape[freq_axis] - 1)
    else:
        n_default = Zxx.shape[freq_axis]

    # 检查窗口参数
    if nperseg is None:
        nperseg = n_default
    else:
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError('nperseg must be a positive integer')

    # 检查FFT长度参数
    if nfft is None:
        if (input_onesided) and (nperseg == n_default + 1):
            nfft = nperseg
        else:
            nfft = n_default
    elif nfft < nperseg:
        raise ValueError('nfft must be greater than or equal to nperseg.')
    else:
        nfft = int(nfft)

    # 检查重叠参数
    if noverlap is None:
        noverlap = nperseg//2
    else:
        noverlap = int(noverlap)
    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')
    nstep = nperseg - noverlap

    # 如果需要，重新排列轴
    # 如果时间轴或频率轴与 Zxx 的维度不匹配，则进行轴交换操作
    if time_axis != Zxx.ndim-1 or freq_axis != Zxx.ndim-2:
        # 将负索引转换为正索引，以便于 transpose 的调用
        if freq_axis < 0:
            freq_axis = Zxx.ndim + freq_axis
        if time_axis < 0:
            time_axis = Zxx.ndim + time_axis
        
        # 构造新的轴顺序列表，排除掉 time_axis 和 freq_axis
        zouter = list(range(Zxx.ndim))
        for ax in sorted([time_axis, freq_axis], reverse=True):
            zouter.pop(ax)
        
        # 使用 np.transpose 进行轴交换
        Zxx = np.transpose(Zxx, zouter + [freq_axis, time_axis])

    # 将 window 参数转换为数组
    if isinstance(window, str) or type(window) is tuple:
        win = get_window(window, nperseg)
    else:
        win = np.asarray(window)
        # 检查 window 是否为一维数组
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        # 检查 window 的长度是否与 nperseg 相符
        if win.shape[0] != nperseg:
            raise ValueError(f'window must have length of {nperseg}')

    # 根据 input_onesided 参数选择 ifft 或 irfft 函数
    ifunc = sp_fft.irfft if input_onesided else sp_fft.ifft
    # 进行 ifft 或 irfft 计算，生成 xsubs
    xsubs = ifunc(Zxx, axis=-2, n=nfft)[..., :nperseg, :]

    # 初始化输出数组 x 和 normalization 数组 norm
    outputlength = nperseg + (nseg-1) * nstep
    x = np.zeros(list(Zxx.shape[:-2]) + [outputlength], dtype=xsubs.dtype)
    norm = np.zeros(outputlength, dtype=xsubs.dtype)

    # 检查 win 和 xsubs 的数据类型是否一致，如果不一致则转换 win 的数据类型
    if np.result_type(win, xsubs) != xsubs.dtype:
        win = win.astype(xsubs.dtype)

    # 根据 scaling 参数对 xsubs 进行缩放处理
    if scaling == 'spectrum':
        xsubs *= win.sum()
    elif scaling == 'psd':
        xsubs *= np.sqrt(fs * sum(win**2))
    else:
        raise ValueError(f"Parameter {scaling=} not in ['spectrum', 'psd']!")

    # 根据 ifft 分段结果构造输出数组 x 和 norm
    # 这个循环可能可以通过向量化或者其他方式优化...
    for ii in range(nseg):
        # 对 ifft 结果应用窗口函数 win
        x[..., ii*nstep:ii*nstep+nperseg] += xsubs[..., ii] * win
        norm[..., ii*nstep:ii*nstep+nperseg] += win**2

    # 如果 boundary=True，则移除扩展点
    if boundary:
        x = x[..., nperseg//2:-(nperseg//2)]
        norm = norm[..., nperseg//2:-(nperseg//2)]

    # 根据非零 normalization 进行除法处理
    if np.sum(norm > 1e-10) != len(norm):
        # 发出警告，指示 STFT 可能不可逆
        warnings.warn(
            "NOLA condition failed, STFT may not be invertible."
            + (" Possibly due to missing boundary" if not boundary else ""),
            stacklevel=2
        )
    x /= np.where(norm > 1e-10, norm, 1.0)

    # 如果 input_onesided=True，则取 x 的实部
    if input_onesided:
        x = x.real

    # 如果 x 的维度大于 1，则尝试将轴重新放回原来的位置
    if x.ndim > 1:
        if time_axis != Zxx.ndim-1:
            if freq_axis < time_axis:
                time_axis -= 1
            x = np.moveaxis(x, -1, time_axis)

    # 根据 x 的形状生成时间数组 time
    time = np.arange(x.shape[0]) / float(fs)
    # 返回时间数组和重构后的信号数组 x
    return time, x
# 估算离散时间信号 X 和 Y 的幅度平方相干性估计 Cxy，使用 Welch 方法
def coherence(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None,
              nfft=None, detrend='constant', axis=-1):
    r"""
    Estimate the magnitude squared coherence estimate, Cxy, of
    discrete-time signals X and Y using Welch's method.

    ``Cxy = abs(Pxy)**2/(Pxx*Pyy)``, where `Pxx` and `Pyy` are power
    spectral density estimates of X and Y, and `Pxy` is the cross
    spectral density estimate of X and Y.

    Parameters
    ----------
    x : array_like
        Time series of measurement values
    y : array_like
        Time series of measurement values
    fs : float, optional
        Sampling frequency of the `x` and `y` time series. Defaults
        to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap: int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    axis : int, optional
        Axis along which the coherence is computed for both inputs; the
        default is over the last axis (i.e. ``axis=-1``).

    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    Cxy : ndarray
        Magnitude squared coherence of x and y.

    See Also
    --------
    periodogram: Simple, optionally modified periodogram
    lombscargle: Lomb-Scargle periodogram for unevenly sampled data
    welch: Power spectral density by Welch's method.
    csd: Cross spectral density by Welch's method.

    Notes
    -----
    An appropriate amount of overlap will depend on the choice of window
    and on your requirements. For the default Hann window an overlap of
    50% is a reasonable trade off between accurately estimating the
    signal power, while not over counting any of the data. Narrower
    windows may require a larger overlap.

    .. versionadded:: 0.16.0

    References
    ----------

    """
    """
    Compute the coherence between two signals x and y using Welch's method.
    
    Parameters:
    - x : array_like
        First input signal.
    - y : array_like
        Second input signal.
    - fs : float
        The sampling frequency (in Hz).
    - window : str or tuple or array_like, optional
        Desired window to use. Default is 'hann'.
    - nperseg : int, optional
        Length of each segment. Defaults to 256.
    - noverlap : int, optional
        Number of points to overlap between segments. Defaults to None.
    - nfft : int, optional
        Length of the FFT used. If None, the FFT length is nperseg. Defaults to None.
    - detrend : str or function or False, optional
        Specifies how to detrend each segment. Defaults to 'constant'.
    - axis : int, optional
        Axis along which the coherence is computed. Defaults to -1.
    
    Returns:
    - freqs : ndarray
        Array of sample frequencies.
    - Cxy : ndarray
        Coherence between x and y.
    
    References:
    - [1] P. Welch, "The use of the fast Fourier transform for the
           estimation of power spectra: A method based on time averaging
           over short, modified periodograms", IEEE Trans. Audio
           Electroacoust. vol. 15, pp. 70-73, 1967.
    - [2] Stoica, Petre, and Randolph Moses, "Spectral Analysis of
           Signals" Prentice Hall, 2005
    
    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    
    Generate two test signals with some common features.
    
    >>> fs = 10e3
    >>> N = 1e5
    >>> amp = 20
    >>> freq = 1234.0
    >>> noise_power = 0.001 * fs / 2
    >>> time = np.arange(N) / fs
    >>> b, a = signal.butter(2, 0.25, 'low')
    >>> x = rng.normal(scale=np.sqrt(noise_power), size=time.shape)
    >>> y = signal.lfilter(b, a, x)
    >>> x += amp*np.sin(2*np.pi*freq*time)
    >>> y += rng.normal(scale=0.1*np.sqrt(noise_power), size=time.shape)
    
    Compute and plot the coherence.
    
    >>> f, Cxy = signal.coherence(x, y, fs, nperseg=1024)
    >>> plt.semilogy(f, Cxy)
    >>> plt.xlabel('frequency [Hz]')
    >>> plt.ylabel('Coherence')
    >>> plt.show()
    """
    freqs, Pxx = welch(x, fs=fs, window=window, nperseg=nperseg,
                       noverlap=noverlap, nfft=nfft, detrend=detrend,
                       axis=axis)
    _, Pyy = welch(y, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
                   nfft=nfft, detrend=detrend, axis=axis)
    _, Pxy = csd(x, y, fs=fs, window=window, nperseg=nperseg,
                 noverlap=noverlap, nfft=nfft, detrend=detrend, axis=axis)
    
    Cxy = np.abs(Pxy)**2 / Pxx / Pyy
    
    return freqs, Cxy
# 定义一个辅助函数，用于计算不同形式的窗口FFT，如PSD、CSD等
# 这个函数不设计为外部调用，而是供stft、psd、csd和spectrogram等函数内部使用

def _spectral_helper(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None,
                     nfft=None, detrend='constant', return_onesided=True,
                     scaling='density', axis=-1, mode='psd', boundary=None,
                     padded=False):
    """Calculate various forms of windowed FFTs for PSD, CSD, etc.

    Parameters
    ----------
    x : array_like
        Array or sequence containing the data to be analyzed.
    y : array_like
        Array or sequence containing the data to be analyzed. If this is
        the same object in memory as `x` (i.e. ``_spectral_helper(x,
        x, ...)``), the extra computations are spared.
    fs : float, optional
        Sampling frequency of the time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `True`, but for
        complex data, a two-sided spectrum is always returned.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the cross spectral density ('density')
        where `Pxy` has units of V**2/Hz and computing the cross
        spectrum ('spectrum') where `Pxy` has units of V**2, if `x`
        and `y` are measured in V and `fs` is measured in Hz.
        Defaults to 'density'
    # axis : int, optional
    #     FFT计算的轴向，默认在最后一个轴向计算（即 ``axis=-1``）。
    # mode: str {'psd', 'stft'}, optional
    #     定义期望的返回值类型。默认为 'psd'。
    # boundary : str or None, optional
    #     指定输入信号是否在两端扩展，并如何生成新值，以便将第一个窗口化段置于第一个输入点的中心。
    #     这有助于在使用窗口函数从零开始时重建第一个输入点。有效选项为 ``['even', 'odd', 'constant', 'zeros', None]``。
    #     默认为 `None`。
    # padded : bool, optional
    #     指定输入信号是否在末尾进行零填充，以确保信号完全适合整数个窗口段，从而将所有信号包含在输出中。
    #     默认为 `False`。如果 `boundary` 不是 `None`，并且 `padded` 为 `True`，则会在边界扩展之后进行填充。
    
    Returns
    -------
    freqs : ndarray
        样本频率的数组。
    t : ndarray
        对应于每个数据段的时间数组。
    result : ndarray
        输出数据的数组，其内容依赖于 *mode* 关键字。
    
    Notes
    -----
    从 matplotlib.mlab 改编
    
    .. versionadded:: 0.16.0
    """
    # 如果 mode 不在 ['psd', 'stft'] 中，则引发值错误
    if mode not in ['psd', 'stft']:
        raise ValueError("Unknown value for mode %s, must be one of: "
                         "{'psd', 'stft'}" % mode)
    
    # 定义边界函数映射
    boundary_funcs = {'even': even_ext,
                      'odd': odd_ext,
                      'constant': const_ext,
                      'zeros': zero_ext,
                      None: None}
    
    # 如果 boundary 不在 boundary_funcs 中，则引发值错误
    if boundary not in boundary_funcs:
        raise ValueError("Unknown boundary option '{}', must be one of: {}"
                         .format(boundary, list(boundary_funcs.keys())))
    
    # 如果 x 和 y 是同一个对象，可以节省计算量
    same_data = y is x
    
    # 如果不是同一个数据且 mode 不是 'psd'，则引发值错误
    if not same_data and mode != 'psd':
        raise ValueError("x and y must be equal if mode is 'stft'")
    
    # 将 axis 转换为整数类型
    axis = int(axis)
    
    # 确保 x 和 y 是 np.arrays，并获取输出数据类型
    x = np.asarray(x)
    if not same_data:
        y = np.asarray(y)
        outdtype = np.result_type(x, y, np.complex64)
    else:
        outdtype = np.result_type(x, np.complex64)
    
    # 如果不是同一个数据，检查是否可以广播外部轴
    if not same_data:
        xouter = list(x.shape)
        youter = list(y.shape)
        xouter.pop(axis)
        youter.pop(axis)
        try:
            outershape = np.broadcast(np.empty(xouter), np.empty(youter)).shape
        except ValueError as e:
            raise ValueError('x and y cannot be broadcast together.') from e
    
    # 如果是同一个数据且 x 的大小为 0，则返回空数组
    if same_data:
        if x.size == 0:
            return np.empty(x.shape), np.empty(x.shape), np.empty(x.shape)
    else:
        # 如果不是相同的数据类型，且其中一个数组为空，则返回空的输出数组
        if x.size == 0 or y.size == 0:
            # 计算输出数组的形状，将最小的轴长度加入到outershape中
            outshape = outershape + (min([x.shape[axis], y.shape[axis]]),)
            # 创建一个空的输出数组，按照指定的轴重新排列
            emptyout = np.moveaxis(np.empty(outshape), -1, axis)
            return emptyout, emptyout, emptyout

    if x.ndim > 1:
        # 如果数组x的维度大于1，并且指定的轴不是最后一个轴，则将指定轴移动到最后一个位置
        if axis != -1:
            x = np.moveaxis(x, axis, -1)
            # 如果x和y的数据类型不同，并且数组y的维度也大于1，则将y中指定的轴移动到最后一个位置
            if not same_data and y.ndim > 1:
                y = np.moveaxis(y, axis, -1)

    # 检查x和y的长度是否相同，如果不相同则进行零填充
    if not same_data:
        if x.shape[-1] != y.shape[-1]:
            # 如果x的长度小于y，则对x进行零填充使其长度与y相同
            if x.shape[-1] < y.shape[-1]:
                pad_shape = list(x.shape)
                pad_shape[-1] = y.shape[-1] - x.shape[-1]
                x = np.concatenate((x, np.zeros(pad_shape)), -1)
            else:
                # 如果y的长度小于x，则对y进行零填充使其长度与x相同
                pad_shape = list(y.shape)
                pad_shape[-1] = x.shape[-1] - y.shape[-1]
                y = np.concatenate((y, np.zeros(pad_shape)), -1)

    if nperseg is not None:  # 如果用户指定了nperseg
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError('nperseg must be a positive integer')

    # 解析窗口；如果窗口是数组形式，则设置nperseg为窗口的形状
    win, nperseg = _triage_segments(window, nperseg, input_length=x.shape[-1])

    if nfft is None:
        # 如果未指定nfft，则将其设为nperseg的值
        nfft = nperseg
    elif nfft < nperseg:
        # 如果指定了nfft，但是小于nperseg，则引发异常
        raise ValueError('nfft must be greater than or equal to nperseg.')
    else:
        nfft = int(nfft)

    if noverlap is None:
        # 如果未指定noverlap，则设置为nperseg的一半
        noverlap = nperseg//2
    else:
        noverlap = int(noverlap)
    if noverlap >= nperseg:
        # 如果noverlap大于等于nperseg，则引发异常
        raise ValueError('noverlap must be less than nperseg.')
    nstep = nperseg - noverlap

    # 边界处理：如果指定了boundary，则使用相应的边界函数进行扩展
    if boundary is not None:
        ext_func = boundary_funcs[boundary]
        x = ext_func(x, nperseg//2, axis=-1)
        # 如果x和y的数据类型不同，则对y也进行相同的边界扩展
        if not same_data:
            y = ext_func(y, nperseg//2, axis=-1)

    if padded:
        # 如果指定了padded，则进行填充，使得数组长度为整数个窗口段
        nadd = (-(x.shape[-1]-nperseg) % nstep) % nperseg
        zeros_shape = list(x.shape[:-1]) + [nadd]
        x = np.concatenate((x, np.zeros(zeros_shape)), axis=-1)
        # 如果x和y的数据类型不同，则对y也进行相同的填充
        if not same_data:
            zeros_shape = list(y.shape[:-1]) + [nadd]
            y = np.concatenate((y, np.zeros(zeros_shape)), axis=-1)

    # 处理去趋势化和窗口函数
    if not detrend:
        # 如果不需要去趋势化，则定义一个恒等函数
        def detrend_func(d):
            return d
    elif not hasattr(detrend, '__call__'):
        # 如果指定的去趋势化不是函数，则使用内置的去趋势化函数
        def detrend_func(d):
            return _signaltools.detrend(d, type=detrend, axis=-1)
    # 如果 axis 不等于 -1，定义一个新的 detrend_func 函数，对数据进行去趋势处理
    elif axis != -1:
        # 将函数包装，使其接收一个预期的形状
        def detrend_func(d):
            # 将最后一个轴移动到指定的 axis 位置
            d = np.moveaxis(d, -1, axis)
            # 对数据进行去趋势处理
            d = detrend(d)
            # 将 axis 位置的轴移回到最后一个位置
            return np.moveaxis(d, axis, -1)
    else:
        # 如果 axis 等于 -1，则直接使用现有的 detrend 函数
        detrend_func = detrend

    # 如果 win 的结果类型与 outdtype 不匹配，则将 win 转换为指定的 outdtype 类型
    if np.result_type(win, np.complex64) != outdtype:
        win = win.astype(outdtype)

    # 根据 scaling 参数计算 scale
    if scaling == 'density':
        scale = 1.0 / (fs * (win*win).sum())
    elif scaling == 'spectrum':
        scale = 1.0 / win.sum()**2
    else:
        # 如果 scaling 参数不是 'density' 或 'spectrum'，则抛出 ValueError 异常
        raise ValueError('Unknown scaling: %r' % scaling)

    # 如果 mode 是 'stft'，则对 scale 进行平方根处理
    if mode == 'stft':
        scale = np.sqrt(scale)

    # 如果 return_onesided 为 True，确定 sides 的类型
    if return_onesided:
        # 如果输入数据 x 是复数类型，则 sides 设置为 'twosided'，并发出警告信息
        if np.iscomplexobj(x):
            sides = 'twosided'
            warnings.warn('Input data is complex, switching to return_onesided=False',
                          stacklevel=3)
        else:
            # 如果输入数据 x 不是复数类型，则 sides 设置为 'onesided'
            sides = 'onesided'
            # 如果不同的数据集 (same_data 为 False)，且输入数据 y 是复数类型，则 sides 设置为 'twosided'，并发出警告信息
            if not same_data:
                if np.iscomplexobj(y):
                    sides = 'twosided'
                    warnings.warn('Input data is complex, switching to '
                                  'return_onesided=False',
                                  stacklevel=3)
    else:
        # 如果 return_onesided 为 False，则 sides 设置为 'twosided'
        sides = 'twosided'

    # 根据 sides 的类型选择频率数组的计算方法
    if sides == 'twosided':
        freqs = sp_fft.fftfreq(nfft, 1/fs)
    elif sides == 'onesided':
        freqs = sp_fft.rfftfreq(nfft, 1/fs)

    # 执行窗口化 FFT 运算，得到结果
    result = _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides)

    # 如果不是同一个数据集 (same_data 为 False)，对 y 数据也进行相同的 FFT 运算，并将结果与 x 数据的共轭相乘
    if not same_data:
        result_y = _fft_helper(y, win, detrend_func, nperseg, noverlap, nfft,
                               sides)
        result = np.conjugate(result) * result_y
    elif mode == 'psd':
        # 如果是同一个数据集且 mode 是 'psd'，则将结果与自身的共轭相乘
        result = np.conjugate(result) * result

    # 结果乘以 scale
    result *= scale

    # 如果 sides 是 'onesided' 并且 mode 是 'psd'
    if sides == 'onesided' and mode == 'psd':
        # 如果 nfft 是奇数，则将结果的第二个以后的部分乘以 2
        if nfft % 2:
            result[..., 1:] *= 2
        else:
            # 如果 nfft 是偶数，则不要对最后一个点（奈奎斯特频率点）乘以 2
            result[..., 1:-1] *= 2

    # 计算时间轴
    time = np.arange(nperseg/2, x.shape[-1] - nperseg/2 + 1,
                     nperseg - noverlap)/float(fs)
    # 如果设置了 boundary 参数，则调整时间轴
    if boundary is not None:
        time -= (nperseg/2) / fs

    # 将结果转换为指定的 outdtype 类型
    result = result.astype(outdtype)

    # 如果 same_data 为 True 且 mode 不是 'stft'，则只保留结果的实部
    if same_data and mode != 'stft':
        result = result.real

    # 将结果的最后一个轴移动回数据来自的 axis 轴
    result = np.moveaxis(result, -1, axis)

    # 返回频率数组 freqs、时间数组 time 和结果 result
    return freqs, time, result
# 解释 _fft_helper 函数，用于计算窗口化的 FFT，供 scipy.signal._spectral_helper 内部使用

def _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides):
    """
    Calculate windowed FFT, for internal use by
    `scipy.signal._spectral_helper`.

    This is a helper function that does the main FFT calculation for
    `_spectral_helper`. All input validation is performed there, and the
    data axis is assumed to be the last axis of x. It is not designed to
    be called externally. The windows are not averaged over; the result
    from each window is returned.

    Returns
    -------
    result : ndarray
        Array of FFT data

    Notes
    -----
    Adapted from matplotlib.mlab

    .. versionadded:: 0.16.0
    """
    # 创建数组的滑动窗口视图
    if nperseg == 1 and noverlap == 0:
        result = x[..., np.newaxis]
    else:
        step = nperseg - noverlap
        result = np.lib.stride_tricks.sliding_window_view(
            x, window_shape=nperseg, axis=-1, writeable=True
        )
        result = result[..., 0::step, :]

    # 对每个数据段进行去趋势处理
    result = detrend_func(result)

    # 通过乘法应用窗口
    result = win * result

    # 执行 FFT。默认在最后一个轴上操作，自动进行零填充
    if sides == 'twosided':
        func = sp_fft.fft
    else:
        result = result.real
        func = sp_fft.rfft
    result = func(result, n=nfft)

    return result


# 解释 _triage_segments 函数，用于解析 spectrogram 和 _spectral_helper 的窗口和 nperseg 参数

def _triage_segments(window, nperseg, input_length):
    """
    Parses window and nperseg arguments for spectrogram and _spectral_helper.
    This is a helper function, not meant to be called externally.

    Parameters
    ----------
    window : string, tuple, or ndarray
        If window is specified by a string or tuple and nperseg is not
        specified, nperseg is set to the default of 256 and returns a window of
        that length.
        If instead the window is array_like and nperseg is not specified, then
        nperseg is set to the length of the window. A ValueError is raised if
        the user supplies both an array_like window and a value for nperseg but
        nperseg does not equal the length of the window.

    nperseg : int
        Length of each segment

    input_length: int
        Length of input signal, i.e. x.shape[-1]. Used to test for errors.

    Returns
    -------
    win : ndarray
        window. If function was called with string or tuple than this will hold
        the actual array used as a window.

    nperseg : int
        Length of each segment. If window is str or tuple, nperseg is set to
        256. If window is array_like, nperseg is set to the length of the
        window.
    """
    # 解析窗口；如果是类似数组，则设置 nperseg = win.shape
    # 检查窗口参数是否为字符串或元组类型
    if isinstance(window, str) or isinstance(window, tuple):
        # 如果未指定 nperseg 参数，则设为默认值 256
        if nperseg is None:
            nperseg = 256  # then change to default
        # 如果 nperseg 大于输入信号长度
        if nperseg > input_length:
            # 发出警告，指出 nperseg 大于输入长度，并使用输入长度作为 nperseg 的值
            warnings.warn(f'nperseg = {nperseg:d} is greater than input length '
                          f' = {input_length:d}, using nperseg = {input_length:d}',
                          stacklevel=3)
            nperseg = input_length
        # 根据指定的窗口类型获取窗口函数
        win = get_window(window, nperseg)
    else:
        # 如果窗口参数不是字符串或元组，则将其转换为 NumPy 数组
        win = np.asarray(window)
        # 检查窗口数组是否为一维数组
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        # 检查窗口数组长度是否超过输入信号长度
        if input_length < win.shape[-1]:
            raise ValueError('window is longer than input signal')
        # 如果未指定 nperseg 参数，则设为窗口数组的长度
        if nperseg is None:
            nperseg = win.shape[0]
        elif nperseg is not None:
            # 如果指定了 nperseg 参数，并且其值与窗口数组长度不同，则引发异常
            if nperseg != win.shape[0]:
                raise ValueError("value specified for nperseg is different"
                                 " from length of window")
    # 返回窗口数组和最终确定的 nperseg 参数值
    return win, nperseg
# 计算一组周期图的中位数相对于平均值的偏差。

Returns the bias of the median of a set of periodograms relative to
the mean.

# 从文献 [1] 的附录 B 查看详细信息。

See Appendix B from [1]_ for details.

# 参数 n: 周期图的数量，即被平均的周期图数目。

Parameters
----------
n : int
    Numbers of periodograms being averaged.

# 返回计算得到的偏差 bias。

Returns
-------
bias : float
    Calculated bias.

# 参考文献 [1] 提到的文章详细信息。

References
----------
.. [1] B. Allen, W.G. Anderson, P.R. Brady, D.A. Brown, J.D.E. Creighton.
       "FINDCHIRP: an algorithm for detection of gravitational waves from
       inspiraling compact binaries", Physical Review D 85, 2012,
       :arxiv:`gr-qc/0509116`
```