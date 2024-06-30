# `D:\src\scipysrc\scipy\scipy\signal\_waveforms.py`

```
# 导入 NumPy 库，并从中导入所需的函数和常量
import numpy as np
# 从 NumPy 中导入 asarray, zeros, place, nan, mod, pi, extract, log, sqrt, exp, cos, sin, polyval, polyint 函数
from numpy import asarray, zeros, place, nan, mod, pi, extract, log, sqrt, exp, cos, sin, polyval, polyint

# 定义公开的函数列表，这些函数可以被外部访问到
__all__ = ['sawtooth', 'square', 'gausspulse', 'chirp', 'sweep_poly',
           'unit_impulse']

# 定义 sawtooth 函数，返回周期锯齿波形或三角波形
def sawtooth(t, width=1):
    """
    Return a periodic sawtooth or triangle waveform.

    The sawtooth waveform has a period ``2*pi``, rises from -1 to 1 on the
    interval 0 to ``width*2*pi``, then drops from 1 to -1 on the interval
    ``width*2*pi`` to ``2*pi``. `width` must be in the interval [0, 1].

    Note that this is not band-limited.  It produces an infinite number
    of harmonics, which are aliased back and forth across the frequency
    spectrum.

    Parameters
    ----------
    t : array_like
        Time.
    width : array_like, optional
        Width of the rising ramp as a proportion of the total cycle.
        Default is 1, producing a rising ramp, while 0 produces a falling
        ramp.  `width` = 0.5 produces a triangle wave.
        If an array, causes wave shape to change over time, and must be the
        same length as t.

    Returns
    -------
    y : ndarray
        Output array containing the sawtooth waveform.

    Examples
    --------
    A 5 Hz waveform sampled at 500 Hz for 1 second:

    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> t = np.linspace(0, 1, 500)
    >>> plt.plot(t, signal.sawtooth(2 * np.pi * 5 * t))

    """
    # 将 t 和 width 转换为 NumPy 数组
    t, w = asarray(t), asarray(width)
    # 强制将 w 和 t 的不合适值设置为零
    w = asarray(w + (t - t))
    t = asarray(t + (w - w))
    # 根据 t 的数据类型确定 y 的数据类型
    if t.dtype.char in ['fFdD']:
        ytype = t.dtype.char
    else:
        ytype = 'd'
    # 初始化 y 数组，用于存储输出波形
    y = zeros(t.shape, ytype)

    # 确保 width 在 [0, 1] 的范围内
    mask1 = (w > 1) | (w < 0)
    place(y, mask1, nan)

    # 将 t 取模 2*pi
    tmod = mod(t, 2 * pi)

    # 在区间 0 到 width*2*pi 的函数为 tmod / (pi*w) - 1
    mask2 = (1 - mask1) & (tmod < w * 2 * pi)
    tsub = extract(mask2, tmod)
    wsub = extract(mask2, w)
    place(y, mask2, tsub / (pi * wsub) - 1)

    # 在区间 width*2*pi 到 2*pi 的函数为 (pi*(w+1)-tmod) / (pi*(1-w))
    mask3 = (1 - mask1) & (1 - mask2)
    tsub = extract(mask3, tmod)
    wsub = extract(mask3, w)
    place(y, mask3, (pi * (wsub + 1) - tsub) / (pi * (1 - wsub)))
    
    # 返回计算结果的数组 y
    return y

# 定义 square 函数，返回周期方波波形
def square(t, duty=0.5):
    """
    Return a periodic square-wave waveform.

    The square wave has a period ``2*pi``, has value +1 from 0 to
    ``2*pi*duty`` and -1 from ``2*pi*duty`` to ``2*pi``. `duty` must be in
    the interval [0,1].

    Note that this is not band-limited.  It produces an infinite number
    of harmonics, which are aliased back and forth across the frequency
    spectrum.

    Parameters
    ----------
    t : array_like
        The input time array.
    t, w = asarray(t), asarray(duty)
    # 将输入的时间序列 t 和占空比序列 duty 转换为 ndarray 类型
    w = asarray(w + (t - t))
    # 如果占空比 duty 是数组，则按元素相加，否则保持不变
    t = asarray(t + (w - w))
    # 如果时间序列 t 是数组，则按元素相加，否则保持不变
    if t.dtype.char in ['fFdD']:
        # 如果 t 的数据类型是单精度或双精度浮点数，则 y 的数据类型设为 t 的数据类型
        ytype = t.dtype.char
    else:
        # 否则设为双精度浮点数
        ytype = 'd'

    y = zeros(t.shape, ytype)
    # 创建一个与 t 形状相同的零数组，数据类型为 ytype

    # 宽度必须在 0 到 1 之间
    mask1 = (w > 1) | (w < 0)
    # 创建一个布尔掩码，标记超出 0 到 1 范围的值
    place(y, mask1, nan)
    # 将 y 中 mask1 对应位置的值设为 NaN

    # 在区间 0 到 duty*2*pi 内函数值为 1
    tmod = mod(t, 2 * pi)
    # 计算 t 对 2*pi 取模后的结果
    mask2 = (1 - mask1) & (tmod < w * 2 * pi)
    # 创建一个布尔掩码，标记在条件 1 - mask1 和 tmod < w * 2 * pi 下为真的位置
    place(y, mask2, 1)
    # 将 y 中 mask2 对应位置的值设为 1

    # 在区间 duty*2*pi 到 2*pi 内函数值为 (pi*(w+1)-tmod) / (pi*(1-w))
    mask3 = (1 - mask1) & (1 - mask2)
    # 创建一个布尔掩码，标记在条件 1 - mask1 和 1 - mask2 下为真的位置
    place(y, mask3, -1)
    # 将 y 中 mask3 对应位置的值设为 -1
    return y
    # 返回生成的波形数组 y
# 定义一个函数，生成高斯调制正弦波信号
def gausspulse(t, fc=1000, bw=0.5, bwr=-6, tpr=-60, retquad=False,
               retenv=False):
    """
    Return a Gaussian modulated sinusoid:

        ``exp(-a t^2) exp(1j*2*pi*fc*t).``

    If `retquad` is True, then return the real and imaginary parts
    (in-phase and quadrature).
    If `retenv` is True, then return the envelope (unmodulated signal).
    Otherwise, return the real part of the modulated sinusoid.

    Parameters
    ----------
    t : ndarray or the string 'cutoff'
        Input array.
    fc : float, optional
        Center frequency (e.g. Hz).  Default is 1000.
    bw : float, optional
        Fractional bandwidth in frequency domain of pulse (e.g. Hz).
        Default is 0.5.
    bwr : float, optional
        Reference level at which fractional bandwidth is calculated (dB).
        Default is -6.
    tpr : float, optional
        If `t` is 'cutoff', then the function returns the cutoff
        time for when the pulse amplitude falls below `tpr` (in dB).
        Default is -60.
    retquad : bool, optional
        If True, return the quadrature (imaginary) as well as the real part
        of the signal.  Default is False.
    retenv : bool, optional
        If True, return the envelope of the signal.  Default is False.

    Returns
    -------
    yI : ndarray
        Real part of signal.  Always returned.
    yQ : ndarray
        Imaginary part of signal.  Only returned if `retquad` is True.
    yenv : ndarray
        Envelope of signal.  Only returned if `retenv` is True.

    Examples
    --------
    Plot real component, imaginary component, and envelope for a 5 Hz pulse,
    sampled at 100 Hz for 2 seconds:

    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> t = np.linspace(-1, 1, 2 * 100, endpoint=False)
    >>> i, q, e = signal.gausspulse(t, fc=5, retquad=True, retenv=True)
    >>> plt.plot(t, i, t, q, t, e, '--')

    """

    # 检查中心频率是否非负，如果为负则引发 ValueError 异常
    if fc < 0:
        raise ValueError("Center frequency (fc=%.2f) must be >=0." % fc)
    
    # 检查带宽是否大于零，如果不是则引发 ValueError 异常
    if bw <= 0:
        raise ValueError("Fractional bandwidth (bw=%.2f) must be > 0." % bw)
    
    # 检查参考级别是否小于零分贝，如果不是则引发 ValueError 异常
    if bwr >= 0:
        raise ValueError("Reference level for bandwidth (bwr=%.2f) must "
                         "be < 0 dB" % bwr)

    # 计算用于高斯调制的系数 a
    # a 的计算基于参考级别和频带宽度
    ref = pow(10.0, bwr / 20.0)
    # 计算 a 的值，用于高斯信号的生成
    a = -(pi * fc * bw) ** 2 / (4.0 * log(ref))
    # 如果输入参数 t 是一个字符串
    if isinstance(t, str):
        # 如果 t 等于 'cutoff'，计算截止点
        if t == 'cutoff':  
            # 求解 exp(-a tc**2) = tref 中的 tc
            # 其中 tc = sqrt(-log(tref) / a)，其中 tref = 10^(tpr/20)
            if tpr >= 0:
                # 如果参考时间截止级别 tpr 大于等于 0 dB，则引发数值错误
                raise ValueError("Reference level for time cutoff must "
                                 "be < 0 dB")
            # 计算 tref = 10^(tpr/20)
            tref = pow(10.0, tpr / 20.0)
            # 返回计算得到的 tc = sqrt(-log(tref) / a)
            return sqrt(-log(tref) / a)
        else:
            # 如果 t 是一个字符串但不是 'cutoff'，则引发数值错误
            raise ValueError("If `t` is a string, it must be 'cutoff'")

    # 计算信号的包络 yenv = exp(-a * t^2)
    yenv = exp(-a * t * t)
    # 计算 I 分量 yI = yenv * cos(2 * pi * fc * t)
    yI = yenv * cos(2 * pi * fc * t)
    # 计算 Q 分量 yQ = yenv * sin(2 * pi * fc * t)
    yQ = yenv * sin(2 * pi * fc * t)

    # 根据参数 retquad 和 retenv 决定返回值
    if not retquad and not retenv:
        # 如果不返回 Q 分量和包络，只返回 I 分量
        return yI
    elif not retquad and retenv:
        # 如果不返回 Q 分量但返回包络，返回 I 分量和包络
        return yI, yenv
    elif retquad and not retenv:
        # 如果返回 Q 分量但不返回包络，返回 I 分量和 Q 分量
        return yI, yQ
    else:
        # 如果同时返回 Q 分量和包络，返回 I 分量、Q 分量和包络
        return yI, yQ, yenv
# 定义频率扫描余弦波生成器函数
def chirp(t, f0, t1, f1, method='linear', phi=0, vertex_zero=True):
    """Frequency-swept cosine generator.

    In the following, 'Hz' should be interpreted as 'cycles per unit';
    there is no requirement here that the unit is one second.  The
    important distinction is that the units of rotation are cycles, not
    radians. Likewise, `t` could be a measurement of space instead of time.

    Parameters
    ----------
    t : array_like
        Times at which to evaluate the waveform.
    f0 : float
        Frequency (e.g. Hz) at time t=0.
    t1 : float
        Time at which `f1` is specified.
    f1 : float
        Frequency (e.g. Hz) of the waveform at time `t1`.
    method : {'linear', 'quadratic', 'logarithmic', 'hyperbolic'}, optional
        Kind of frequency sweep.  If not given, `linear` is assumed.  See
        Notes below for more details.
    phi : float, optional
        Phase offset, in degrees. Default is 0.
    vertex_zero : bool, optional
        This parameter is only used when `method` is 'quadratic'.
        It determines whether the vertex of the parabola that is the graph
        of the frequency is at t=0 or t=t1.

    Returns
    -------
    y : ndarray
        A numpy array containing the signal evaluated at `t` with the
        requested time-varying frequency.  More precisely, the function
        returns ``cos(phase + (pi/180)*phi)`` where `phase` is the integral
        (from 0 to `t`) of ``2*pi*f(t)``. ``f(t)`` is defined below.

    See Also
    --------
    sweep_poly

    Notes
    -----
    There are four options for the `method`.  The following formulas give
    the instantaneous frequency (in Hz) of the signal generated by
    `chirp()`.  For convenience, the shorter names shown below may also be
    used.

    linear, lin, li:

        ``f(t) = f0 + (f1 - f0) * t / t1``

    quadratic, quad, q:

        The graph of the frequency f(t) is a parabola through (0, f0) and
        (t1, f1).  By default, the vertex of the parabola is at (0, f0).
        If `vertex_zero` is False, then the vertex is at (t1, f1).  The
        formula is:

        if vertex_zero is True:

            ``f(t) = f0 + (f1 - f0) * t**2 / t1**2``

        else:

            ``f(t) = f1 - (f1 - f0) * (t1 - t)**2 / t1**2``

        To use a more general quadratic function, or an arbitrary
        polynomial, use the function `scipy.signal.sweep_poly`.

    logarithmic, log, lo:

        ``f(t) = f0 * (f1/f0)**(t/t1)``

        f0 and f1 must be nonzero and have the same sign.

        This signal is also known as a geometric or exponential chirp.

    hyperbolic, hyp:

        ``f(t) = f0*f1*t1 / ((f0 - f1)*t + f1*t1)``

        f0 and f1 must be nonzero.

    Examples
    --------
    The following will be used in the examples:

    >>> import numpy as np
    >>> from scipy.signal import chirp, spectrogram
    >>> import matplotlib.pyplot as plt

    For the first example, we'll plot the waveform for a linear chirp
    """
    # 计算相位 'phase'，在 _chirp_phase 中计算，方便测试。
    phase = _chirp_phase(t, f0, t1, f1, method, vertex_zero)
    # 将 phi 转换为弧度制。
    phi *= pi / 180
    # 返回余弦相位加上 phi 的结果。
    return cos(phase + phi)
# 计算由 `chirp` 函数生成输出时使用的相位。

# 查看 `chirp` 函数以获取参数说明。

def _chirp_phase(t, f0, t1, f1, method='linear', vertex_zero=True):
    # 将输入的时间数组转换为NumPy数组
    t = asarray(t)
    # 将起始频率转换为浮点数
    f0 = float(f0)
    # 将结束时间转换为浮点数
    t1 = float(t1)
    # 将结束频率转换为浮点数
    f1 = float(f1)

    # 根据不同的方法计算相位
    if method in ['linear', 'lin', 'li']:
        # 计算线性变化的斜率
        beta = (f1 - f0) / t1
        # 计算线性变化的相位
        phase = 2 * pi * (f0 * t + 0.5 * beta * t * t)

    elif method in ['quadratic', 'quad', 'q']:
        # 计算二次变化的斜率
        beta = (f1 - f0) / (t1 ** 2)
        if vertex_zero:
            # 如果顶点为零，则计算二次变化的相位
            phase = 2 * pi * (f0 * t + beta * t ** 3 / 3)
        else:
            # 如果顶点不为零，则计算二次变化的相位
            phase = 2 * pi * (f1 * t + beta * ((t1 - t) ** 3 - t1 ** 3) / 3)

    elif method in ['logarithmic', 'log', 'lo']:
        # 对数变化的情况
        if f0 * f1 <= 0.0:
            # 如果起始频率和结束频率乘积小于等于零，抛出异常
            raise ValueError("For a logarithmic chirp, f0 and f1 must be "
                             "nonzero and have the same sign.")
        if f0 == f1:
            # 如果起始频率等于结束频率，则计算对数变化的相位
            phase = 2 * pi * f0 * t
        else:
            # 计算对数变化的斜率
            beta = t1 / log(f1 / f0)
            # 计算对数变化的相位
            phase = 2 * pi * beta * f0 * (pow(f1 / f0, t / t1) - 1.0)

    elif method in ['hyperbolic', 'hyp']:
        # 双曲线变化的情况
        if f0 == 0 or f1 == 0:
            # 如果起始频率或结束频率为零，抛出异常
            raise ValueError("For a hyperbolic chirp, f0 and f1 must be "
                             "nonzero.")
        if f0 == f1:
            # 如果起始频率等于结束频率，则计算常频率的相位
            phase = 2 * pi * f0 * t
        else:
            # 计算双曲线变化的奇异点
            sing = -f1 * t1 / (f0 - f1)
            # 计算双曲线变化的相位
            phase = 2 * pi * (-sing * f0) * log(np.abs(1 - t/sing))

    else:
        # 如果方法不在预定义的列表中，抛出异常
        raise ValueError("method must be 'linear', 'quadratic', 'logarithmic',"
                         " or 'hyperbolic', but a value of %r was given."
                         % method)

    # 返回计算得到的相位
    return phase
    # sweep_poly : ndarray
    #     A numpy array containing the signal evaluated at `t` with the
    #     requested time-varying frequency.  More precisely, the function
    #     returns ``cos(phase + (pi/180)*phi)``, where `phase` is the integral
    #     (from 0 to t) of ``2 * pi * f(t)``; ``f(t)`` is defined above.
    #
    # See Also
    # --------
    # chirp
    #
    # Notes
    # -----
    # .. versionadded:: 0.8.0
    #
    # If `poly` is a list or ndarray of length `n`, then the elements of
    # `poly` are the coefficients of the polynomial, and the instantaneous
    # frequency is:
    #
    #     ``f(t) = poly[0]*t**(n-1) + poly[1]*t**(n-2) + ... + poly[n-1]``
    #
    # If `poly` is an instance of `numpy.poly1d`, then the instantaneous
    # frequency is:
    #
    #       ``f(t) = poly(t)``
    #
    # Finally, the output `s` is:
    #
    #     ``cos(phase + (pi/180)*phi)``
    #
    # where `phase` is the integral from 0 to `t` of ``2 * pi * f(t)``,
    # ``f(t)`` as defined above.
    #
    # Examples
    # --------
    # Compute the waveform with instantaneous frequency::
    #
    #     f(t) = 0.025*t**3 - 0.36*t**2 + 1.25*t + 2
    #
    # over the interval 0 <= t <= 10.
    #
    # >>> import numpy as np
    # >>> from scipy.signal import sweep_poly
    # >>> p = np.poly1d([0.025, -0.36, 1.25, 2.0])
    # >>> t = np.linspace(0, 10, 5001)
    # >>> w = sweep_poly(t, p)
    #
    # Plot it:
    #
    # >>> import matplotlib.pyplot as plt
    # >>> plt.subplot(2, 1, 1)
    # >>> plt.plot(t, w)
    # >>> plt.title("Sweep Poly\\nwith frequency " +
    # ...           "$f(t) = 0.025t^3 - 0.36t^2 + 1.25t + 2$")
    # >>> plt.subplot(2, 1, 2)
    # >>> plt.plot(t, p(t), 'r', label='f(t)')
    # >>> plt.legend()
    # >>> plt.xlabel('t')
    # >>> plt.tight_layout()
    # >>> plt.show()
    """
    # 'phase' is computed in _sweep_poly_phase, to make testing easier.
    # 计算相位 'phase'，在 _sweep_poly_phase 函数中计算，以便于测试。
    phase = _sweep_poly_phase(t, poly)
    # Convert to radians.
    # 将角度转换为弧度。
    phi *= pi / 180
    # 返回计算得到的信号值，使用余弦函数计算相位加上角度转换后的角度的值。
    return cos(phase + phi)
def _sweep_poly_phase(t, poly):
    """
    Calculate the phase used by sweep_poly to generate its output.

    See `sweep_poly` for a description of the arguments.

    """
    # 使用 polyint 函数处理 poly，它自动处理列表、ndarray 和 poly1d 实例。
    intpoly = polyint(poly)
    # 计算相位，使用 polyval 函数计算多项式在 t 处的值，乘以 2π。
    phase = 2 * pi * polyval(intpoly, t)
    return phase


def unit_impulse(shape, idx=None, dtype=float):
    """
    Unit impulse signal (discrete delta function) or unit basis vector.

    Parameters
    ----------
    shape : int or tuple of int
        Number of samples in the output (1-D), or a tuple that represents the
        shape of the output (N-D).
    idx : None or int or tuple of int or 'mid', optional
        Index at which the value is 1.  If None, defaults to the 0th element.
        If ``idx='mid'``, the impulse will be centered at ``shape // 2`` in
        all dimensions.  If an int, the impulse will be at `idx` in all
        dimensions.
    dtype : data-type, optional
        The desired data-type for the array, e.g., ``numpy.int8``.  Default is
        ``numpy.float64``.

    Returns
    -------
    y : ndarray
        Output array containing an impulse signal.

    Notes
    -----
    The 1D case is also known as the Kronecker delta.

    .. versionadded:: 0.19.0

    Examples
    --------
    An impulse at the 0th element (:math:`\\delta[n]`):

    >>> from scipy import signal
    >>> signal.unit_impulse(8)
    array([ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])

    Impulse offset by 2 samples (:math:`\\delta[n-2]`):

    >>> signal.unit_impulse(7, 2)
    array([ 0.,  0.,  1.,  0.,  0.,  0.,  0.])

    2-dimensional impulse, centered:

    >>> signal.unit_impulse((3, 3), 'mid')
    array([[ 0.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  0.]])

    Impulse at (2, 2), using broadcasting:

    >>> signal.unit_impulse((4, 4), 2)
    array([[ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  0.]])

    Plot the impulse response of a 4th-order Butterworth lowpass filter:

    >>> imp = signal.unit_impulse(100, 'mid')
    >>> b, a = signal.butter(4, 0.2)
    >>> response = signal.lfilter(b, a, imp)

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(np.arange(-50, 50), imp)
    >>> plt.plot(np.arange(-50, 50), response)
    >>> plt.margins(0.1, 0.1)
    >>> plt.xlabel('Time [samples]')
    >>> plt.ylabel('Amplitude')
    >>> plt.grid(True)
    >>> plt.show()

    """
    # 创建一个全零数组，其形状由参数 shape 决定，数据类型为 dtype
    out = zeros(shape, dtype)

    # 将 shape 转换为至少是一维的数组
    shape = np.atleast_1d(shape)

    # 根据 idx 的不同取值，确定单位冲激的位置
    if idx is None:
        idx = (0,) * len(shape)
    elif idx == 'mid':
        idx = tuple(shape // 2)
    elif not hasattr(idx, "__iter__"):
        idx = (idx,) * len(shape)

    # 在数组 out 中指定的位置 idx 设置为 1
    out[idx] = 1
    return out
```