# `D:\src\scipysrc\scipy\scipy\fft\_fftlog_backend.py`

```
import numpy as np
from warnings import warn
from ._basic import rfft, irfft
from ..special import loggamma, poch

from scipy._lib._array_api import array_namespace, copy

__all__ = ['fht', 'ifht', 'fhtoffset']

# 常量
LN_2 = np.log(2)


def fht(a, dln, mu, offset=0.0, bias=0.0):
    xp = array_namespace(a)

    # 计算变换的大小
    n = a.shape[-1]

    # 对输入数组进行偏置处理
    if bias != 0:
        # a_q(r) = a(r) (r/r_c)^{-q}
        j_c = (n-1)/2
        j = xp.arange(n, dtype=xp.float64)
        a = a * xp.exp(-bias*(j - j_c)*dln)

    # 计算快速 Hankel 变换系数
    u = xp.asarray(fhtcoeff(n, dln, mu, offset=offset, bias=bias))

    # 进行变换
    A = _fhtq(a, u, xp=xp)

    # 对输出数组进行偏置处理
    if bias != 0:
        # A(k) = A_q(k) (k/k_c)^{-q} (k_c r_c)^{-q}
        A *= xp.exp(-bias*((j - j_c)*dln + offset))

    return A


def ifht(A, dln, mu, offset=0.0, bias=0.0):
    xp = array_namespace(A)

    # 计算变换的大小
    n = A.shape[-1]

    # 对输入数组进行偏置处理
    if bias != 0:
        # A_q(k) = A(k) (k/k_c)^{q} (k_c r_c)^{q}
        j_c = (n-1)/2
        j = xp.arange(n, dtype=xp.float64)
        A = A * xp.exp(bias*((j - j_c)*dln + offset))

    # 计算快速 Hankel 变换系数
    u = xp.asarray(fhtcoeff(n, dln, mu, offset=offset, bias=bias, inverse=True))

    # 进行变换
    a = _fhtq(A, u, inverse=True, xp=xp)

    # 对输出数组进行偏置处理
    if bias != 0:
        # a(r) = a_q(r) (r/r_c)^{q}
        a /= xp.exp(-bias*(j - j_c)*dln)

    return a


def fhtcoeff(n, dln, mu, offset=0.0, bias=0.0, inverse=False):
    """计算快速 Hankel 变换的系数数组。"""
    lnkr, q = offset, bias

    # Hankel 变换系数
    # u_m = (kr)^{-i 2m pi/(n dlnr)} U_mu(q + i 2m pi/(n dlnr))
    # 其中 U_mu(x) = 2^x Gamma((mu+1+x)/2)/Gamma((mu+1-x)/2)
    xp = (mu+1+q)/2
    xm = (mu+1-q)/2
    y = np.linspace(0, np.pi*(n//2)/(n*dln), n//2+1)
    u = np.empty(n//2+1, dtype=complex)
    v = np.empty(n//2+1, dtype=complex)
    u.imag[:] = y
    u.real[:] = xm
    loggamma(u, out=v)
    u.real[:] = xp
    loggamma(u, out=u)
    y *= 2*(LN_2 - lnkr)
    u.real -= v.real
    u.real += LN_2*q
    u.imag += v.imag
    u.imag += y
    np.exp(u, out=u)

    # 修正最后一个系数为实数
    u.imag[-1] = 0

    # 处理特殊情况
    if not np.isfinite(u[0]):
        # 设定 u_0 = 2^q Gamma(xp)/Gamma(xm) = 2^q poch(xm, xp-xm)
        # poch() 正确处理负整数的特殊情况
        u[0] = 2**q * poch(xm, xp-xm)
        # 如果系数为 inf 或 0，意味着变换或逆变换可能奇异

    # 检查变换是否奇异或逆变换是否奇异
    if np.isinf(u[0]) and not inverse:
        warn('singular transform; consider changing the bias', stacklevel=3)
        # 修正系数以获得（可能正确的）变换
        u = copy(u)
        u[0] = 0
    elif u[0] == 0 and inverse:
        # 如果首个元素为0并且进行逆变换，则发出警告，建议考虑改变偏置
        warn('singular inverse transform; consider changing the bias', stacklevel=3)
        # 复制数组u，以便修改其中的元素而不影响原始数组
        u = copy(u)
        # 将首个元素修改为无穷大，以强制获得可能正确的逆变换结果
        u[0] = np.inf

    # 返回修改后的数组u
    return u
def fhtoffset(dln, mu, initial=0.0, bias=0.0):
    """Return optimal offset for a fast Hankel transform.

    Returns an offset close to `initial` that fulfils the low-ringing
    condition of [1]_ for the fast Hankel transform `fht` with logarithmic
    spacing `dln`, order `mu` and bias `bias`.

    Parameters
    ----------
    dln : float
        Uniform logarithmic spacing of the transform.
    mu : float
        Order of the Hankel transform, any positive or negative real number.
    initial : float, optional
        Initial value for the offset. Returns the closest value that fulfils
        the low-ringing condition.
    bias : float, optional
        Exponent of power law bias, any positive or negative real number.

    Returns
    -------
    offset : float
        Optimal offset of the uniform logarithmic spacing of the transform that
        fulfils a low-ringing condition.

    Examples
    --------
    >>> from scipy.fft import fhtoffset
    >>> dln = 0.1
    >>> mu = 2.0
    >>> initial = 0.5
    >>> bias = 0.0
    >>> offset = fhtoffset(dln, mu, initial, bias)
    >>> offset
    0.5454581477676637

    See Also
    --------
    fht : Definition of the fast Hankel transform.

    References
    ----------
    .. [1] Hamilton A. J. S., 2000, MNRAS, 312, 257 (astro-ph/9905191)

    """

    lnkr, q = initial, bias  # 设置 lnkr 和 q 初始值为 initial 和 bias

    xp = (mu+1+q)/2  # 计算 xp
    xm = (mu+1-q)/2  # 计算 xm
    y = np.pi/(2*dln)  # 计算 y
    zp = loggamma(xp + 1j*y)  # 计算 zp
    zm = loggamma(xm + 1j*y)  # 计算 zm
    arg = (LN_2 - lnkr)/dln + (zp.imag + zm.imag)/np.pi  # 计算 arg
    return lnkr + (arg - np.round(arg))*dln  # 返回计算出的 lnkr 值加上微调后的结果


def _fhtq(a, u, inverse=False, *, xp=None):
    """Compute the biased fast Hankel transform.

    This is the basic FFTLog routine.
    """
    if xp is None:
        xp = np  # 如果 xp 为 None，则设置 xp 为 numpy

    # size of transform
    n = a.shape[-1]  # 获取变换的大小

    # biased fast Hankel transform via real FFT
    A = rfft(a, axis=-1)  # 对 a 进行快速傅里叶变换，得到 A
    if not inverse:
        # forward transform
        A *= u  # 前向变换时，乘以 u
    else:
        # backward transform
        A /= xp.conj(u)  # 反向变换时，除以 u 的共轭
    A = irfft(A, n, axis=-1)  # 对 A 进行反快速傅里叶变换，得到反变换结果 A
    A = xp.flip(A, axis=-1)  # 对 A 进行翻转操作，沿着最后一个轴

    return A  # 返回变换后的结果
```