# `D:\src\scipysrc\matplotlib\lib\matplotlib\mlab.py`

```
"""
Numerical Python functions written for compatibility with MATLAB
commands with the same names. Most numerical Python functions can be found in
the `NumPy`_ and `SciPy`_ libraries. What remains here is code for performing
spectral computations and kernel density estimations.

.. _NumPy: https://numpy.org
.. _SciPy: https://www.scipy.org

Spectral functions
------------------

`cohere`
    Coherence (normalized cross spectral density)

`csd`
    Cross spectral density using Welch's average periodogram

`detrend`
    Remove the mean or best fit line from an array

`psd`
    Power spectral density using Welch's average periodogram

`specgram`
    Spectrogram (spectrum over segments of time)

`complex_spectrum`
    Return the complex-valued frequency spectrum of a signal

`magnitude_spectrum`
    Return the magnitude of the frequency spectrum of a signal

`angle_spectrum`
    Return the angle (wrapped phase) of the frequency spectrum of a signal

`phase_spectrum`
    Return the phase (unwrapped angle) of the frequency spectrum of a signal

`detrend_mean`
    Remove the mean from a line.

`detrend_linear`
    Remove the best fit line from a line.

`detrend_none`
    Return the original line.
"""

import functools
from numbers import Number

import numpy as np

from matplotlib import _api, _docstring, cbook


def window_hanning(x):
    """
    Return *x* times the Hanning (or Hann) window of len(*x*).

    See Also
    --------
    window_none : Another window algorithm.
    """
    # 返回输入数组 x 乘以长度为 len(x) 的汉宁窗口
    return np.hanning(len(x)) * x


def window_none(x):
    """
    No window function; simply return *x*.

    See Also
    --------
    window_hanning : Another window algorithm.
    """
    # 直接返回输入数组 x，不应用任何窗口函数
    return x


def detrend(x, key=None, axis=None):
    """
    Return *x* with its trend removed.

    Parameters
    ----------
    x : array or sequence
        Array or sequence containing the data.

    key : {'default', 'constant', 'mean', 'linear', 'none'} or function
        The detrending algorithm to use. 'default', 'mean', and 'constant' are
        the same as `detrend_mean`. 'linear' is the same as `detrend_linear`.
        'none' is the same as `detrend_none`. The default is 'mean'. See the
        corresponding functions for more details regarding the algorithms. Can
        also be a function that carries out the detrend operation.

    axis : int
        The axis along which to do the detrending.

    See Also
    --------
    detrend_mean : Implementation of the 'mean' algorithm.
    detrend_linear : Implementation of the 'linear' algorithm.
    detrend_none : Implementation of the 'none' algorithm.
    """
    # 如果 key 为 None 或者 'constant', 'mean', 'default' 中的一种，使用 detrend_mean 进行去趋势化处理
    if key is None or key in ['constant', 'mean', 'default']:
        return detrend(x, key=detrend_mean, axis=axis)
    # 如果 key 为 'linear'，使用 detrend_linear 进行去趋势化处理
    elif key == 'linear':
        return detrend(x, key=detrend_linear, axis=axis)
    # 如果 key 为 'none'，使用 detrend_none 返回原始数据（不进行处理）
    elif key == 'none':
        return detrend(x, key=detrend_none, axis=axis)
    # 如果key是一个可调用对象（函数），则进行如下操作
    elif callable(key):
        # 将x转换为NumPy数组
        x = np.asarray(x)
        # 如果指定了axis，并且axis + 1超出了数组的维度范围，抛出值错误异常
        if axis is not None and axis + 1 > x.ndim:
            raise ValueError(f'axis(={axis}) out of bounds')
        # 如果axis为None且数组维度为0，或者axis为0且数组维度为1，直接对x应用key函数并返回结果
        if (axis is None and x.ndim == 0) or (not axis and x.ndim == 1):
            return key(x)
        # 尝试在函数支持axis参数的情况下使用axis参数
        # 否则使用apply_along_axis函数在指定axis上应用key函数
        try:
            return key(x, axis=axis)
        except TypeError:
            return np.apply_along_axis(key, axis=axis, arr=x)
    # 如果key既不是字符串也不是可调用对象，则抛出值错误异常
    else:
        raise ValueError(
            f"Unknown value for key: {key!r}, must be one of: 'default', "
            f"'constant', 'mean', 'linear', or a function")
# 如果 y 参数为 None，则使用 x 参数作为 y 参数，表示处理相同的数据
same_data = True



# 如果 y 参数不为 None，则检查是否 y 和 x 是同一个对象，以便在不进行额外计算的情况下实现 psd()、csd() 和 spectrogram() 的核心功能
same_data = y is x



# 如果未指定采样频率 Fs，则默认设置为 2
if Fs is None:
    Fs = 2



# 如果未指定重叠部分 noverlap，则默认设置为 0
if noverlap is None:
    noverlap = 0



# 如果未指定去趋势函数 detrend_func，则默认使用 detrend_none 函数
if detrend_func is None:
    detrend_func = detrend_none



# 如果未指定窗口函数 window，则默认使用 window_hanning 窗口函数
if window is None:
    window = window_hanning



# 如果未指定 FFT 点数 NFFT，则默认设置为 256
# NFFT 表示 FFT 的窗口大小
if NFFT is None:
    NFFT = 256
    # 如果重叠长度大于或等于FFT长度，抛出数值错误
    if noverlap >= NFFT:
        raise ValueError('noverlap must be less than NFFT')

    # 如果模式为None或者'default'，将模式设置为'psd'
    if mode is None or mode == 'default':
        mode = 'psd'
    # 检查模式是否在允许的列表中
    _api.check_in_list(
        ['default', 'psd', 'complex', 'magnitude', 'angle', 'phase'],
        mode=mode)

    # 如果数据不相同且模式不是'psd'，抛出数值错误
    if not same_data and mode != 'psd':
        raise ValueError("x and y must be equal if mode is not 'psd'")

    # 确保x和y是numpy数组，如果x和y是同一个对象，则保持不变
    x = np.asarray(x)
    if not same_data:
        y = np.asarray(y)

    # 如果侧边未指定或者为'default'，根据x是否为复数确定侧边
    if sides is None or sides == 'default':
        if np.iscomplexobj(x):
            sides = 'twosided'
        else:
            sides = 'onesided'
    # 检查侧边是否在允许的列表中
    _api.check_in_list(['default', 'onesided', 'twosided'], sides=sides)

    # 如果x的长度小于NFFT，则对x进行零填充
    if len(x) < NFFT:
        n = len(x)
        x = np.resize(x, NFFT)
        x[n:] = 0

    # 如果数据不相同且y的长度小于NFFT，则对y进行零填充
    if not same_data and len(y) < NFFT:
        n = len(y)
        y = np.resize(y, NFFT)
        y[n:] = 0

    # 如果未指定pad_to，则设置为NFFT
    if pad_to is None:
        pad_to = NFFT

    # 如果模式不是'psd'，则scale_by_freq为False；如果scale_by_freq未指定，则为True
    if mode != 'psd':
        scale_by_freq = False
    elif scale_by_freq is None:
        scale_by_freq = True

    # 对于实数x，除非另有说明，忽略负频率
    if sides == 'twosided':
        numFreqs = pad_to
        if pad_to % 2:
            freqcenter = (pad_to - 1)//2 + 1
        else:
            freqcenter = pad_to//2
        scaling_factor = 1.
    elif sides == 'onesided':
        if pad_to % 2:
            numFreqs = (pad_to + 1)//2
        else:
            numFreqs = pad_to//2 + 1
        scaling_factor = 2.

    # 如果窗口不是可迭代对象，则使用窗口函数创建一个窗口
    if not np.iterable(window):
        window = window(np.ones(NFFT, x.dtype))
    # 窗口长度必须与数据的第一个维度匹配，否则抛出数值错误
    if len(window) != NFFT:
        raise ValueError(
            "The window length must match the data's first dimension")

    # 对x进行滑动窗口处理，并对结果进行去趋势处理
    result = np.lib.stride_tricks.sliding_window_view(
        x, NFFT, axis=0)[::NFFT - noverlap].T
    result = detrend(result, detrend_func, axis=0)
    result = result * window.reshape((-1, 1))
    result = np.fft.fft(result, n=pad_to, axis=0)[:numFreqs, :]
    freqs = np.fft.fftfreq(pad_to, 1/Fs)[:numFreqs]

    # 如果数据不同，根据模式处理resultY并与result求乘积
    if not same_data:
        # 如果same_data为False，模式必须是'psd'
        resultY = np.lib.stride_tricks.sliding_window_view(
            y, NFFT, axis=0)[::NFFT - noverlap].T
        resultY = detrend(resultY, detrend_func, axis=0)
        resultY = resultY * window.reshape((-1, 1))
        resultY = np.fft.fft(resultY, n=pad_to, axis=0)[:numFreqs, :]
        result = np.conj(result) * resultY
    elif mode == 'psd':
        # 如果same_data为True且模式是'psd'，则处理result
        result = np.conj(result) * result
    elif mode == 'magnitude':
        # 如果模式是'magnitude'，计算结果的幅值并除以窗口总和
        result = np.abs(result) / window.sum()
    elif mode == 'angle' or mode == 'phase':
        # 如果模式是'angle'或'phase'，计算结果的角度（相位）
        # 这里将在后面处理单边和双边的情况
        result = np.angle(result)
    elif mode == 'complex':
        # 如果模式是'complex'，将结果除以窗口总和
        result /= window.sum()
    # 如果模式为 'psd'，则执行以下操作：
    # 这里包括对单边密度的缩放因子以及按需除以采样频率的操作。对除了直流分量和NFFT/2分量以外的所有内容进行缩放：

    # 如果频率数量是偶数，不对NFFT/2进行缩放
    if not NFFT % 2:
        slc = slice(1, -1, None)
    # 如果频率数量是奇数，只不对直流分量进行缩放
    else:
        slc = slice(1, None, None)

    # 对结果的指定切片进行缩放操作
    result[slc] *= scaling_factor

    # 如果需要按照频率进行缩放：
    if scale_by_freq:
        # 对结果除以采样频率，以使密度函数单位为dB/Hz，并且可以通过绘制的频率值进行积分
        result /= Fs
        # 根据窗口的范数缩放谱以补偿窗口损失；参见Bendat & Piersol Sec 11.5.2
        result /= (window**2).sum()
    else:
        # 在这种情况下，保持段中的功率，而不是振幅
        result /= window.sum()**2

    # 根据NFFT、len(x)和noverlap计算时间向量t
    t = np.arange(NFFT/2, len(x) - NFFT/2 + 1, NFFT - noverlap)/Fs

    # 如果选择 'twosided' 频谱：
    if sides == 'twosided':
        # 将频率范围中心对齐到零点
        freqs = np.roll(freqs, -freqcenter, axis=0)
        result = np.roll(result, -freqcenter, axis=0)
    # 如果pad_to不是偶数，需要正确处理最后一个频率值
    elif not pad_to % 2:
        freqs[-1] *= -1

    # 如果模式为 'phase'，解包相位以处理单边与双边情况
    if mode == 'phase':
        result = np.unwrap(result, axis=0)

    # 返回处理后的结果、频率向量freqs和时间向量t
    return result, freqs, t
# 定义一个私有辅助函数，用于实现复杂、幅度、角度和相位频谱之间的共性操作
def _single_spectrum_helper(
        mode, x, Fs=None, window=None, pad_to=None, sides=None):
    """
    Private helper implementing the commonality between the complex, magnitude,
    angle, and phase spectrums.
    """
    # 检查 mode 参数是否在指定的列表中
    _api.check_in_list(['complex', 'magnitude', 'angle', 'phase'], mode=mode)

    # 如果 pad_to 参数未指定，将其设为输入信号 x 的长度
    if pad_to is None:
        pad_to = len(x)

    # 调用 _spectral_helper 函数，获取频谱 spec、频率 freqs 和额外信息 _
    spec, freqs, _ = _spectral_helper(x=x, y=None, NFFT=len(x), Fs=Fs,
                                      detrend_func=detrend_none, window=window,
                                      noverlap=0, pad_to=pad_to,
                                      sides=sides,
                                      scale_by_freq=False,
                                      mode=mode)
    
    # 如果 mode 不是 'complex'，取 spec 的实部
    if mode != 'complex':
        spec = spec.real

    # 如果 spec 是二维数组且第二维长度为 1，将其转为一维数组
    if spec.ndim == 2 and spec.shape[1] == 1:
        spec = spec[:, 0]

    # 返回处理后的频谱 spec 和频率 freqs
    return spec, freqs


# 将下列关键字文档拆分出来，以便在其他地方使用
_docstring.interpd.update(
    Spectral="""\
Fs : float, default: 2
    The sampling frequency (samples per time unit).  It is used to calculate
    the Fourier frequencies, *freqs*, in cycles per time unit.

window : callable or ndarray, default: `.window_hanning`
    A function or a vector of length *NFFT*.  To create window vectors see
    `.window_hanning`, `.window_none`, `numpy.blackman`, `numpy.hamming`,
    `numpy.bartlett`, `scipy.signal`, `scipy.signal.get_window`, etc.  If a
    function is passed as the argument, it must take a data segment as an
    argument and return the windowed version of the segment.

sides : {'default', 'onesided', 'twosided'}, optional
    Which sides of the spectrum to return. 'default' is one-sided for real
    data and two-sided for complex data. 'onesided' forces the return of a
    one-sided spectrum, while 'twosided' forces two-sided.""",

    Single_Spectrum="""\
pad_to : int, optional
    The number of points to which the data segment is padded when performing
    the FFT.  While not increasing the actual resolution of the spectrum (the
    minimum distance between resolvable peaks), this can give more points in
    the plot, allowing for more detail. This corresponds to the *n* parameter
    in the call to `~numpy.fft.fft`.  The default is None, which sets *pad_to*
    equal to the length of the input signal (i.e. no padding).""",

    PSD="""\
pad_to : int, optional
    The number of points to which the data segment is padded when performing
    the FFT.  This can be different from *NFFT*, which specifies the number
    of data points used.  While not increasing the actual resolution of the
    spectrum (the minimum distance between resolvable peaks), this can give
    more points in the plot, allowing for more detail. This corresponds to
    the *n* parameter in the call to `~numpy.fft.fft`. The default is None,
    which sets *pad_to* equal to *NFFT*

NFFT : int, default: 256
    The number of data points used in each block for the FFT.  A power 2 is
    most efficient.  This should *NOT* be used to get zero padding, or the
    scaling of the result will be incorrect; use *pad_to* for this instead.


    这段注释是对某段代码的解释说明，建议不要使用该段代码来获取零填充，因为这样会导致结果的缩放不正确；而应该使用 *pad_to* 来进行零填充。
detrend : {'none', 'mean', 'linear'} or callable, default: 'none'
    # detrend 参数用于指定在进行 FFT 前对每个数据段应用的去趋势方法，可选项包括 'none'、'mean'、'linear' 或自定义函数。
    # 这些选项分别对应于不去趋势、去除平均值趋势和去除线性趋势。在 Matplotlib 中，与 MATLAB 不同，detrend 参数是一个函数。
    # 模块 :mod:`~matplotlib.mlab` 定义了 `.detrend_none`、`.detrend_mean` 和 `.detrend_linear`，也可以使用自定义函数。
    # 字符串形式的 'none' 对应 `.detrend_none`，'mean' 对应 `.detrend_mean`，'linear' 对应 `.detrend_linear`。

scale_by_freq : bool, default: True
    # scale_by_freq 参数控制是否将结果密度值按比例缩放以给出单位为 1/Hz 的密度。这允许对返回的频率值进行积分。
    # 默认值为 True，以保持与 MATLAB 的兼容性。
    """
    Calculate the cross spectrum using Welch's average periodogram method. The vectors *x* and *y* are divided into
    *NFFT* length segments. Each segment is detrended by function *detrend* and windowed by function *window*. *noverlap* 
    gives the length of the overlap between segments. The product of the direct FFTs of *x* and *y* are averaged over each 
    segment to compute :math:`P_{xy}`, with a scaling to correct for power loss due to windowing.

    If len(*x*) < *NFFT* or len(*y*) < *NFFT*, they will be zero padded to *NFFT*.

    Parameters
    ----------
    x, y : 1-D arrays or sequences
        Arrays or sequences containing the data

    %(Spectral)s
        Parameters related to spectral density estimation.

    %(PSD)s
        Parameters related to power spectral density estimation.

    noverlap : int, default: 0 (no overlap)
        The number of points of overlap between segments.

    Returns
    -------
    Pxy : 1-D array
        The values for the cross spectrum :math:`P_{xy}` before scaling (real valued)

    freqs : 1-D array
        The frequencies corresponding to the elements in *Pxy*

    References
    ----------
    Bendat & Piersol -- Random Data: Analysis and Measurement Procedures, John
    Wiley & Sons (1986)

    See Also
    --------
    psd : equivalent to setting ``y = x``.
    """
    # If NFFT is not specified, set it to 256 by default
    if NFFT is None:
        NFFT = 256
    # Compute the cross spectrum using helper function _spectral_helper
    Pxy, freqs, _ = _spectral_helper(x=x, y=y, NFFT=NFFT, Fs=Fs,
                                     detrend_func=detrend, window=window,
                                     noverlap=noverlap, pad_to=pad_to,
                                     sides=sides, scale_by_freq=scale_by_freq,
                                     mode='psd')

    # If Pxy has two dimensions, average across the second dimension
    if Pxy.ndim == 2:
        if Pxy.shape[1] > 1:
            Pxy = Pxy.mean(axis=1)
        else:
            Pxy = Pxy[:, 0]
    
    # Return the computed cross spectrum and frequencies
    return Pxy, freqs
# 定义一个字符串，包含单个频谱文档的模板，用于描述各种频谱计算方法的参数和返回值
_single_spectrum_docs = """\
Compute the {quantity} of *x*.
Data is padded to a length of *pad_to* and the windowing function *window* is
applied to the signal.

Parameters
----------
x : 1-D array or sequence
    Array or sequence containing the data

{Spectral}

{Single_Spectrum}

Returns
-------
spectrum : 1-D array
    The {quantity}.
freqs : 1-D array
    The frequencies corresponding to the elements in *spectrum*.

See Also
--------
psd
    Returns the power spectral density.
complex_spectrum
    Returns the complex-valued frequency spectrum.
magnitude_spectrum
    Returns the absolute value of the `complex_spectrum`.
angle_spectrum
    Returns the angle of the `complex_spectrum`.
phase_spectrum
    Returns the phase (unwrapped angle) of the `complex_spectrum`.
specgram
    Can return the complex spectrum of segments within the signal.
"""

# 定义偏函数，将_single_spectrum_helper函数应用到"complex"类型频谱计算上，并设置其文档字符串
complex_spectrum = functools.partial(_single_spectrum_helper, "complex")
complex_spectrum.__doc__ = _single_spectrum_docs.format(
    quantity="complex-valued frequency spectrum",
    **_docstring.interpd.params)

# 定义偏函数，将_single_spectrum_helper函数应用到"magnitude"类型频谱计算上，并设置其文档字符串
magnitude_spectrum = functools.partial(_single_spectrum_helper, "magnitude")
magnitude_spectrum.__doc__ = _single_spectrum_docs.format(
    quantity="magnitude (absolute value) of the frequency spectrum",
    **_docstring.interpd.params)

# 定义偏函数，将_single_spectrum_helper函数应用到"angle"类型频谱计算上，并设置其文档字符串
angle_spectrum = functools.partial(_single_spectrum_helper, "angle")
angle_spectrum.__doc__ = _single_spectrum_docs.format(
    quantity="angle of the frequency spectrum (wrapped phase spectrum)",
    **_docstring.interpd.params)

# 定义偏函数，将_single_spectrum_helper函数应用到"phase"类型频谱计算上，并设置其文档字符串
phase_spectrum = functools.partial(_single_spectrum_helper, "phase")
phase_spectrum.__doc__ = _single_spectrum_docs.format(
    quantity="phase of the frequency spectrum (unwrapped phase spectrum)",
    **_docstring.interpd.params)

# 定义specgram函数，用于计算并绘制信号的谱图
@_docstring.dedent_interpd
def specgram(x, NFFT=None, Fs=None, detrend=None, window=None,
             noverlap=None, pad_to=None, sides=None, scale_by_freq=None,
             mode=None):
    """
    Compute a spectrogram.

    Compute and plot a spectrogram of data in *x*.  Data are split into
    *NFFT* length segments and the spectrum of each section is
    computed.  The windowing function *window* is applied to each
    segment, and the amount of overlap of each segment is
    specified with *noverlap*.

    Parameters
    ----------
    x : array-like
        1-D array or sequence.

    %(Spectral)s

    %(PSD)s

    noverlap : int, default: 128
        The number of points of overlap between blocks.
    mode : str, default: 'psd'
        What sort of spectrum to use:
            'psd'
                Returns the power spectral density.
            'complex'
                Returns the complex-valued frequency spectrum.
            'magnitude'
                Returns the magnitude spectrum.
            'angle'
                Returns the phase spectrum without unwrapping.
            'phase'
                Returns the phase spectrum with unwrapping.

    Returns
    -------
    """
    """
    spectrum : array-like
        2D array, columns are the periodograms of successive segments.

    freqs : array-like
        1-D array, frequencies corresponding to the rows in *spectrum*.

    t : array-like
        1-D array, the times corresponding to midpoints of segments
        (i.e the columns in *spectrum*).

    See Also
    --------
    psd : differs in the overlap and in the return values.
    complex_spectrum : similar, but with complex valued frequencies.
    magnitude_spectrum : similar single segment when *mode* is 'magnitude'.
    angle_spectrum : similar to single segment when *mode* is 'angle'.
    phase_spectrum : similar to single segment when *mode* is 'phase'.

    Notes
    -----
    *detrend* and *scale_by_freq* only apply when *mode* is set to 'psd'.
    """

    # 如果没有指定重叠量，则使用默认值 128，_spectral_helper() 中的默认值为 noverlap = 0
    if noverlap is None:
        noverlap = 128  # default in _spectral_helper() is noverlap = 0
    
    # 如果没有指定 NFFT（FFT 点数），则使用默认值 256，_spectral_helper() 中也使用相同的默认值
    if NFFT is None:
        NFFT = 256  # same default as in _spectral_helper()
    
    # 如果信号长度小于等于 NFFT，则发出警告，并只计算一个片段
    if len(x) <= NFFT:
        _api.warn_external("Only one segment is calculated since parameter "
                           f"NFFT (={NFFT}) >= signal length (={len(x)}).")
    
    # 调用 _spectral_helper 函数计算频谱
    spec, freqs, t = _spectral_helper(x=x, y=None, NFFT=NFFT, Fs=Fs,
                                      detrend_func=detrend, window=window,
                                      noverlap=noverlap, pad_to=pad_to,
                                      sides=sides,
                                      scale_by_freq=scale_by_freq,
                                      mode=mode)

    # 如果 mode 不是 'complex'，则只保留 spec 的实部，因为 helper 函数是通用实现
    if mode != 'complex':
        spec = spec.real  # Needed since helper implements generically

    # 返回计算得到的频谱 spec，频率数组 freqs，和时间数组 t
    return spec, freqs, t
@_docstring.dedent_interpd
# 使用装饰器 dedent_interpd 对函数文档字符串进行格式化和插值处理
def cohere(x, y, NFFT=256, Fs=2, detrend=detrend_none, window=window_hanning,
           noverlap=0, pad_to=None, sides='default', scale_by_freq=None):
    r"""
    The coherence between *x* and *y*.  Coherence is the normalized
    cross spectral density:

    .. math::

        C_{xy} = \frac{|P_{xy}|^2}{P_{xx}P_{yy}}

    Parameters
    ----------
    x, y
        Array or sequence containing the data

    %(Spectral)s
        插入 "Spectral" 部分的描述内容

    %(PSD)s
        插入 "PSD" 部分的描述内容

    noverlap : int, default: 0 (no overlap)
        The number of points of overlap between segments.

    Returns
    -------
    Cxy : 1-D array
        The coherence vector.
    freqs : 1-D array
        The frequencies for the elements in *Cxy*.

    See Also
    --------
    :func:`psd`, :func:`csd` :
        For information about the methods used to compute :math:`P_{xy}`,
        :math:`P_{xx}` and :math:`P_{yy}`.
    """
    if len(x) < 2 * NFFT:
        raise ValueError(
            "Coherence is calculated by averaging over *NFFT* length "
            "segments.  Your signal is too short for your choice of *NFFT*.")
    Pxx, f = psd(x, NFFT, Fs, detrend, window, noverlap, pad_to, sides,
                 scale_by_freq)
    # 计算信号 x 的功率谱密度 Pxx 和对应的频率数组 f
    Pyy, f = psd(y, NFFT, Fs, detrend, window, noverlap, pad_to, sides,
                 scale_by_freq)
    # 计算信号 y 的功率谱密度 Pyy 和对应的频率数组 f
    Pxy, f = csd(x, y, NFFT, Fs, detrend, window, noverlap, pad_to, sides,
                 scale_by_freq)
    # 计算信号 x 和 y 的交叉功率谱密度 Pxy 和对应的频率数组 f
    Cxy = np.abs(Pxy) ** 2 / (Pxx * Pyy)
    # 计算并返回归一化的相干度 Cxy 和频率数组 f
    return Cxy, f


class GaussianKDE:
    """
    Representation of a kernel-density estimate using Gaussian kernels.

    Parameters
    ----------
    dataset : array-like
        Datapoints to estimate from. In case of univariate data this is a 1-D
        array, otherwise a 2D array with shape (# of dims, # of data).
    bw_method : {'scott', 'silverman'} or float or callable, optional
        The method used to calculate the estimator bandwidth.  If a
        float, this will be used directly as `kde.factor`.  If a
        callable, it should take a `GaussianKDE` instance as only
        parameter and return a float. If None (default), 'scott' is used.

    Attributes
    ----------
    dataset : ndarray
        The dataset passed to the constructor.
    dim : int
        Number of dimensions.
    num_dp : int
        Number of datapoints.
    factor : float
        The bandwidth factor, obtained from `kde.covariance_factor`, with which
        the covariance matrix is multiplied.
    covariance : ndarray
        The covariance matrix of *dataset*, scaled by the calculated bandwidth
        (`kde.factor`).
    inv_cov : ndarray
        The inverse of *covariance*.

    Methods
    -------
    kde.evaluate(points) : ndarray
        Evaluate the estimated pdf on a provided set of points.
    kde(points) : ndarray
        Same as kde.evaluate(points)
    """

    # This implementation with minor modification was too good to pass up.
    # 这个实现经过轻微修改，实在太好了，不能放过。
    # 从 scipy 的 GitHub 仓库中的 kde.py 文件获取的代码段
    def __init__(self, dataset, bw_method=None):
        # 将输入数据集至少转换为二维数组
        self.dataset = np.atleast_2d(dataset)
        # 检查数据集是否包含多个元素，否则抛出数值错误异常
        if not np.array(self.dataset).size > 1:
            raise ValueError("`dataset` 输入应包含多个元素。")

        # 计算数据集的维度和样本数
        self.dim, self.num_dp = np.array(self.dataset).shape

        # 根据不同的带宽方法设置协方差因子函数
        if bw_method is None:
            pass
        elif cbook._str_equal(bw_method, 'scott'):
            # 使用 Scott 方法设置协方差因子函数
            self.covariance_factor = self.scotts_factor
        elif cbook._str_equal(bw_method, 'silverman'):
            # 使用 Silverman 方法设置协方差因子函数
            self.covariance_factor = self.silverman_factor
        elif isinstance(bw_method, Number):
            # 使用常数设置协方差因子函数
            self._bw_method = 'use constant'
            self.covariance_factor = lambda: bw_method
        elif callable(bw_method):
            # 使用可调用对象设置协方差因子函数
            self._bw_method = bw_method
            self.covariance_factor = lambda: self._bw_method(self)
        else:
            # 抛出数值错误异常，提示 `bw_method` 应为 'scott', 'silverman', 标量或可调用对象
            raise ValueError("`bw_method` 应为 'scott', 'silverman', 标量或可调用对象")

        # 计算协方差矩阵，并使用 covariance_factor() 进行缩放
        self.factor = self.covariance_factor()
        
        # 如果不存在 '_data_inv_cov' 属性，则缓存数据的协方差和逆协方差矩阵
        if not hasattr(self, '_data_inv_cov'):
            self.data_covariance = np.atleast_2d(
                np.cov(
                    self.dataset,
                    rowvar=1,
                    bias=False))
            self.data_inv_cov = np.linalg.inv(self.data_covariance)

        # 计算缩放后的协方差和逆协方差矩阵
        self.covariance = self.data_covariance * self.factor ** 2
        self.inv_cov = self.data_inv_cov / self.factor ** 2
        
        # 计算标准化因子，用于高斯核密度估计
        self.norm_factor = (np.sqrt(np.linalg.det(2 * np.pi * self.covariance))
                            * self.num_dp)

    # 返回 Scott 方法计算的协方差因子
    def scotts_factor(self):
        return np.power(self.num_dp, -1. / (self.dim + 4))

    # 返回 Silverman 方法计算的协方差因子
    def silverman_factor(self):
        return np.power(
            self.num_dp * (self.dim + 2.0) / 4.0, -1. / (self.dim + 4))

    # 默认的带宽计算方法，可以被子类覆盖重写
    covariance_factor = scotts_factor
    # 定义一个方法用于评估估计的概率密度函数在一组点上的取值

    def evaluate(self, points):
        """
        Evaluate the estimated pdf on a set of points.

        Parameters
        ----------
        points : (# of dimensions, # of points)-array
            Alternatively, a (# of dimensions,) vector can be passed in and
            treated as a single point.

        Returns
        -------
        (# of points,)-array
            The values at each point.

        Raises
        ------
        ValueError : if the dimensionality of the input points is different
                     than the dimensionality of the KDE.

        """
        # 将输入的点至少转换为二维数组
        points = np.atleast_2d(points)

        # 获取点的维度和数量
        dim, num_m = np.array(points).shape

        # 如果输入点的维度与 KDE 的维度不同，则抛出 ValueError 异常
        if dim != self.dim:
            raise ValueError(f"points have dimension {dim}, dataset has "
                             f"dimension {self.dim}")

        # 初始化结果数组
        result = np.zeros(num_m)

        # 如果点的数量大于等于数据集的数据点数量
        if num_m >= self.num_dp:
            # 遍历数据集中的每个数据点
            for i in range(self.num_dp):
                # 计算数据点与给定点之间的差值
                diff = self.dataset[:, i, np.newaxis] - points
                # 计算差值与协方差矩阵的乘积
                tdiff = np.dot(self.inv_cov, diff)
                # 计算能量项
                energy = np.sum(diff * tdiff, axis=0) / 2.0
                # 计算结果中对应点的贡献并累加
                result = result + np.exp(-energy)
        else:
            # 遍历给定的每个点
            for i in range(num_m):
                # 计算数据集中每个数据点与给定点之间的差值
                diff = self.dataset - points[:, i, np.newaxis]
                # 计算差值与协方差矩阵的乘积
                tdiff = np.dot(self.inv_cov, diff)
                # 计算能量项
                energy = np.sum(diff * tdiff, axis=0) / 2.0
                # 计算结果中对应点的贡献并存储
                result[i] = np.sum(np.exp(-energy), axis=0)

        # 将结果数组归一化
        result = result / self.norm_factor

        # 返回最终的评估结果
        return result

    # 将 evaluate 方法作为 __call__ 方法的别名
    __call__ = evaluate
```