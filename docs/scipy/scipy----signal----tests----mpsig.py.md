# `D:\src\scipysrc\scipy\scipy\signal\tests\mpsig.py`

```
"""
Some signal functions implemented using mpmath.
"""

# 尝试导入 mpmath 库
try:
    import mpmath
except ImportError:
    mpmath = None


def _prod(seq):
    """Returns the product of the elements in the sequence `seq`."""
    p = 1
    for elem in seq:
        p *= elem
    return p


def _relative_degree(z, p):
    """
    Return relative degree of transfer function from zeros and poles.

    This is simply len(p) - len(z), which must be nonnegative.
    A ValueError is raised if len(p) < len(z).
    """
    # 计算传递函数的相对阶数
    degree = len(p) - len(z)
    if degree < 0:
        raise ValueError("Improper transfer function. "
                         "Must have at least as many poles as zeros.")
    return degree


def _zpkbilinear(z, p, k, fs):
    """Bilinear transformation to convert a filter from analog to digital."""

    degree = _relative_degree(z, p)

    fs2 = 2*fs

    # 对极点和零点进行双线性变换
    z_z = [(fs2 + z1) / (fs2 - z1) for z1 in z]
    p_z = [(fs2 + p1) / (fs2 - p1) for p1 in p]

    # 将原本位于无穷远的零点移动到 Nyquist 频率
    z_z.extend([-1] * degree)

    # 补偿增益变化
    numer = _prod(fs2 - z1 for z1 in z)
    denom = _prod(fs2 - p1 for p1 in p)
    k_z = k * numer / denom

    return z_z, p_z, k_z.real


def _zpklp2lp(z, p, k, wo=1):
    """Transform a lowpass filter to a different cutoff frequency."""

    degree = _relative_degree(z, p)

    # 将所有点从原点向外径向缩放，以改变截止频率
    z_lp = [wo * z1 for z1 in z]
    p_lp = [wo * p1 for p1 in p]

    # 每个移动的极点按 wo 减少增益，每个移动的零点增加增益
    # 取消净变化以保持总增益不变
    k_lp = k * wo**degree

    return z_lp, p_lp, k_lp


def _butter_analog_poles(n):
    """
    Poles of an analog Butterworth lowpass filter.

    This is the same calculation as scipy.signal.buttap(n) or
    scipy.signal.butter(n, 1, analog=True, output='zpk'), but mpmath is used,
    and only the poles are returned.
    """
    # 计算模拟 Butterworth 低通滤波器的极点
    poles = [-mpmath.exp(1j*mpmath.pi*k/(2*n)) for k in range(-n+1, n, 2)]
    return poles


def butter_lp(n, Wn):
    """
    Lowpass Butterworth digital filter design.

    This computes the same result as scipy.signal.butter(n, Wn, output='zpk'),
    but it uses mpmath, and the results are returned in lists instead of NumPy
    arrays.
    """
    zeros = []
    poles = _butter_analog_poles(n)
    k = 1
    fs = 2
    warped = 2 * fs * mpmath.tan(mpmath.pi * Wn / fs)
    z, p, k = _zpklp2lp(zeros, poles, k, wo=warped)
    z, p, k = _zpkbilinear(z, p, k, fs=fs)
    return z, p, k


def zpkfreqz(z, p, k, worN=None):
    """
    Frequency response of a filter in zpk format, using mpmath.

    This is the same calculation as scipy.signal.freqz, but the input is in
    zpk format, the calculation is performed using mpath, and the results are
    returned in lists instead of NumPy arrays.
    """
    # 此函数计算 zpk 格式滤波器的频率响应
    # 如果 worN 为 None 或者是整数类型，则将 N 设置为 worN 的值，如果 worN 为 None，则 N 默认为 512
    # 否则，如果 worN 是其他类型（如列表），直接使用 worN 作为 ws
    if worN is None or isinstance(worN, int):
        N = worN or 512
        # 生成角频率列表 ws，其中每个元素是 mpmath.pi * mpmath.mpf(j) / N
        ws = [mpmath.pi * mpmath.mpf(j) / N for j in range(N)]
    else:
        # 如果 worN 不是 None 且不是整数类型，直接使用 worN 作为 ws
        ws = worN
    
    h = []
    # 对于 ws 中的每个角频率 wk，执行以下操作：
    for wk in ws:
        # 计算 exp(1j * wk)，其中 1j 是虚数单位，zm1 是结果
        zm1 = mpmath.exp(1j * wk)
        # 计算数值部分，_prod 函数将每个 zm1 - t 中的 t 取值从 z 中取出进行连乘
        numer = _prod([zm1 - t for t in z])
        # 计算分母部分，同样是使用 _prod 函数，将每个 zm1 - t 中的 t 取值从 p 中取出进行连乘
        denom = _prod([zm1 - t for t in p])
        # 计算 h(k)，其中 k 是给定的常数系数，hk 是最终的结果
        hk = k * numer / denom
        # 将 hk 加入到列表 h 中
        h.append(hk)
    # 返回计算得到的角频率列表 ws 和对应的 h 响应函数列表 h
    return ws, h
```