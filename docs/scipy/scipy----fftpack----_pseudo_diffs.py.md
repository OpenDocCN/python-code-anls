# `D:\src\scipysrc\scipy\scipy\fftpack\_pseudo_diffs.py`

```
"""
Differential and pseudo-differential operators.
"""
# 由Pearu Peterson于2002年9月创建

__all__ = ['diff',
           'tilbert','itilbert','hilbert','ihilbert',
           'cs_diff','cc_diff','sc_diff','ss_diff',
           'shift']

from numpy import pi, asarray, sin, cos, sinh, cosh, tanh, iscomplexobj
from . import convolve  # 导入本地模块convolve
from scipy.fft._pocketfft.helper import _datacopied  # 导入Scipy的FFT模块中的_datacopied函数

_cache = {}  # 定义空的缓存字典


def diff(x,order=1,period=None, _cache=_cache):
    """
    Return kth derivative (or integral) of a periodic sequence x.

    If x_j and y_j are Fourier coefficients of periodic functions x
    and y, respectively, then::

      y_j = pow(sqrt(-1)*j*2*pi/period, order) * x_j
      y_0 = 0 if order is not 0.

    Parameters
    ----------
    x : array_like
        Input array.
    order : int, optional
        The order of differentiation. Default order is 1. If order is
        negative, then integration is carried out under the assumption
        that ``x_0 == 0``.
    period : float, optional
        The assumed period of the sequence. Default is ``2*pi``.

    Notes
    -----
    If ``sum(x, axis=0) = 0`` then ``diff(diff(x, k), -k) == x`` (within
    numerical accuracy).

    For odd order and even ``len(x)``, the Nyquist mode is taken zero.

    """
    tmp = asarray(x)  # 将输入x转换为NumPy数组
    if order == 0:
        return tmp  # 如果order为0，直接返回原数组
    if iscomplexobj(tmp):
        return diff(tmp.real,order,period)+1j*diff(tmp.imag,order,period)  # 如果输入数组为复数，则分别对实部和虚部进行求导
    if period is not None:
        c = 2*pi/period  # 计算c为2*pi/period
    else:
        c = 1.0  # 默认情况下，c为1.0
    n = len(x)  # 获取数组x的长度
    omega = _cache.get((n,order,c))  # 从缓存中获取对应的omega值
    if omega is None:
        if len(_cache) > 20:
            while _cache:
                _cache.popitem()  # 如果缓存中的项数超过20，清空缓存
        # 定义卷积核函数
        def kernel(k,order=order,c=c):
            if k:
                return pow(c*k,order)
            return 0
        # 初始化卷积核omega
        omega = convolve.init_convolution_kernel(n,kernel,d=order,
                                                 zero_nyquist=1)
        _cache[(n,order,c)] = omega  # 将计算得到的omega存入缓存
    overwrite_x = _datacopied(tmp, x)  # 检查是否复制了输入数组tmp
    return convolve.convolve(tmp,omega,swap_real_imag=order % 2,
                             overwrite_x=overwrite_x)  # 返回卷积结果


del _cache  # 删除缓存变量_cache


_cache = {}  # 重新定义空的缓存字典


def tilbert(x, h, period=None, _cache=_cache):
    """
    Return h-Tilbert transform of a periodic sequence x.

    If x_j and y_j are Fourier coefficients of periodic functions x
    and y, respectively, then::

        y_j = sqrt(-1)*coth(j*h*2*pi/period) * x_j
        y_0 = 0

    Parameters
    ----------
    x : array_like
        The input array to transform.
    h : float
        Defines the parameter of the Tilbert transform.
    period : float, optional
        The assumed period of the sequence. Default period is ``2*pi``.

    Returns
    -------
    tilbert : ndarray
        The result of the transform.

    Notes
    -----
    If ``sum(x, axis=0) == 0`` and ``n = len(x)`` is odd, then
    ``tilbert(itilbert(x)) == x``.

    If ``2 * pi * h / period`` is approximately 10 or larger, then
    numerically ``tilbert == hilbert``

    """
    # 将输入数组x转换为NumPy数组
    tmp = asarray(x)
    # 定义参数c为2*pi/period或者默认为1.0
    if period is not None:
        c = 2*pi/period
    else:
        c = 1.0
    # 获取输入数组的长度n
    n = len(x)
    # 从缓存中获取omega值
    omega = _cache.get((n, h, c))
    if omega is None:
        if len(_cache) > 20:
            while _cache:
                _cache.popitem()  # 清空缓存
        # 定义Tilbert变换的核函数
        def kernel(k, h=h, c=c):
            if k:
                return 1j / tanh(k * h * c)
            return 0
        # 初始化卷积核omega
        omega = convolve.init_convolution_kernel(n, kernel)
        _cache[(n, h, c)] = omega  # 将计算得到的omega存入缓存
    # 检查是否复制了输入数组tmp
    overwrite_x = _datacopied(tmp, x)
    # 返回卷积结果
    return convolve.convolve(tmp, omega, overwrite_x=overwrite_x)
    """
    计算实部和虚部的 Tilbert 变换，并返回复数结果。
    """
    # 将输入 x 转换为 ndarray
    tmp = asarray(x)
    # 如果 tmp 是复数对象，则分别对其实部和虚部进行 Tilbert 变换，并返回复数结果
    if iscomplexobj(tmp):
        return tilbert(tmp.real, h, period) + \
               1j * tilbert(tmp.imag, h, period)

    # 如果指定了周期 period，则重新计算 h
    if period is not None:
        h = h * 2 * pi / period

    # 获取输入 x 的长度 n
    n = len(x)
    # 尝试从缓存 _cache 中获取对应的 omega
    omega = _cache.get((n, h))
    if omega is None:
        # 如果 _cache 的长度超过 20，则清空 _cache
        if len(_cache) > 20:
            while _cache:
                _cache.popitem()

        # 定义核函数 kernel
        def kernel(k, h=h):
            # 如果 k 不为零，则返回 1/tanh(h*k)
            if k:
                return 1.0/tanh(h*k)
            # 如果 k 为零，则返回 0
            return 0

        # 初始化卷积核 omega
        omega = convolve.init_convolution_kernel(n, kernel, d=1)
        # 将计算结果存入缓存 _cache
        _cache[(n,h)] = omega

    # 检查并复制 tmp 到 x，获取是否需要覆盖原始输入数据
    overwrite_x = _datacopied(tmp, x)
    # 执行卷积操作，返回结果
    return convolve.convolve(tmp, omega, swap_real_imag=1, overwrite_x=overwrite_x)
# 删除全局变量 _cache
del _cache

# 创建空字典 _cache 用于存储函数的计算结果以提高性能
_cache = {}

# 定义函数 itilbert，计算周期序列 x 的逆 h-Tilbert 变换
def itilbert(x, h, period=None, _cache=_cache):
    """
    Return inverse h-Tilbert transform of a periodic sequence x.

    If ``x_j`` and ``y_j`` are Fourier coefficients of periodic functions x
    and y, respectively, then::

      y_j = -sqrt(-1)*tanh(j*h*2*pi/period) * x_j
      y_0 = 0

    For more details, see `tilbert`.

    """
    tmp = asarray(x)

    # 如果输入是复数类型，则分别对实部和虚部进行逆 h-Tilbert 变换
    if iscomplexobj(tmp):
        return itilbert(tmp.real, h, period) + 1j * itilbert(tmp.imag, h, period)
    
    # 如果给定了周期 period，则重新计算 h
    if period is not None:
        h = h * 2 * pi / period
    
    n = len(x)
    # 从缓存中获取计算结果
    omega = _cache.get((n, h))

    # 如果缓存中未找到结果，则进行计算
    if omega is None:
        # 如果缓存中的项数超过 20，则清空缓存
        if len(_cache) > 20:
            while _cache:
                _cache.popitem()

        # 定义核函数 kernel，并计算 omega
        def kernel(k, h=h):
            if k:
                return -tanh(h * k)
            return 0

        omega = convolve.init_convolution_kernel(n, kernel, d=1)
        _cache[(n, h)] = omega
    
    # 检查是否需要复制数据，并返回卷积结果
    overwrite_x = _datacopied(tmp, x)
    return convolve.convolve(tmp, omega, swap_real_imag=1, overwrite_x=overwrite_x)


# 删除全局变量 _cache
del _cache

# 创建空字典 _cache 用于存储函数的计算结果以提高性能
_cache = {}

# 定义函数 hilbert，计算周期序列 x 的 Hilbert 变换
def hilbert(x, _cache=_cache):
    """
    Return Hilbert transform of a periodic sequence x.

    If x_j and y_j are Fourier coefficients of periodic functions x
    and y, respectively, then::

      y_j = sqrt(-1)*sign(j) * x_j
      y_0 = 0

    Parameters
    ----------
    x : array_like
        The input array, should be periodic.
    _cache : dict, optional
        Dictionary that contains the kernel used to do a convolution with.

    Returns
    -------
    y : ndarray
        The transformed input.

    See Also
    --------
    scipy.signal.hilbert : Compute the analytic signal, using the Hilbert
                           transform.

    Notes
    -----
    If ``sum(x, axis=0) == 0`` then ``hilbert(ihilbert(x)) == x``.

    For even len(x), the Nyquist mode of x is taken zero.

    The sign of the returned transform does not have a factor -1 that is more
    often than not found in the definition of the Hilbert transform. Note also
    that `scipy.signal.hilbert` does have an extra -1 factor compared to this
    function.

    """
    tmp = asarray(x)

    # 如果输入是复数类型，则分别对实部和虚部进行 Hilbert 变换
    if iscomplexobj(tmp):
        return hilbert(tmp.real) + 1j * hilbert(tmp.imag)

    n = len(x)
    # 从缓存中获取计算结果
    omega = _cache.get(n)

    # 如果缓存中未找到结果，则进行计算
    if omega is None:
        # 如果缓存中的项数超过 20，则清空缓存
        if len(_cache) > 20:
            while _cache:
                _cache.popitem()

        # 定义核函数 kernel，并计算 omega
        def kernel(k):
            if k > 0:
                return 1.0
            elif k < 0:
                return -1.0
            return 0.0

        omega = convolve.init_convolution_kernel(n, kernel, d=1)
        _cache[n] = omega
    
    # 检查是否需要复制数据，并返回卷积结果
    overwrite_x = _datacopied(tmp, x)
    return convolve.convolve(tmp, omega, swap_real_imag=1, overwrite_x=overwrite_x)


# 删除全局变量 _cache
del _cache

# 定义函数 ihilbert，计算周期序列 x 的逆 Hilbert 变换
def ihilbert(x):
    """
    Return inverse Hilbert transform of a periodic sequence x.

    If ``x_j`` and ``y_j`` are Fourier coefficients of periodic functions x
    and y, respectively, then::

      y_j = -sqrt(-1)*sign(j) * x_j
      y_0 = 0

    """
    # 实现为空，未提供具体实现
    # 返回负的希尔伯特变换结果，即计算负希尔伯特变换并返回其结果
    return -hilbert(x)
# 初始化一个空的缓存字典，用于存储中间结果，避免重复计算
_cache = {}

# 定义函数 cs_diff，计算周期序列 x 的 (a,b)-cosh/sinh 伪导数
def cs_diff(x, a, b, period=None, _cache=_cache):
    """
    Return (a,b)-cosh/sinh pseudo-derivative of a periodic sequence.

    If ``x_j`` and ``y_j`` are Fourier coefficients of periodic functions x
    and y, respectively, then::

      y_j = -sqrt(-1)*cosh(j*a*2*pi/period)/sinh(j*b*2*pi/period) * x_j
      y_0 = 0

    Parameters
    ----------
    x : array_like
        The array to take the pseudo-derivative from.
    a, b : float
        Defines the parameters of the cosh/sinh pseudo-differential
        operator.
    period : float, optional
        The period of the sequence. Default period is ``2*pi``.

    Returns
    -------
    cs_diff : ndarray
        Pseudo-derivative of periodic sequence `x`.

    Notes
    -----
    For even len(`x`), the Nyquist mode of `x` is taken as zero.
    """
    # 将输入 x 转换为数组
    tmp = asarray(x)
    
    # 如果 x 是复数对象，则分别计算其实部和虚部的伪导数
    if iscomplexobj(tmp):
        return cs_diff(tmp.real,a,b,period) + \
               1j*cs_diff(tmp.imag,a,b,period)
    
    # 如果指定了周期 period，则重新计算参数 a 和 b
    if period is not None:
        a = a*2*pi/period
        b = b*2*pi/period
    
    # 获取序列 x 的长度 n
    n = len(x)
    
    # 尝试从缓存中获取已计算的结果 omega
    omega = _cache.get((n,a,b))
    
    # 如果 omega 不存在，则计算新的伪导数核
    if omega is None:
        # 如果缓存长度超过 20，则清空缓存
        if len(_cache) > 20:
            while _cache:
                _cache.popitem()

        # 定义伪导数核函数 kernel(k,a,b)，当 k 不为零时返回 -cosh(a*k)/sinh(b*k)，否则返回 0
        def kernel(k,a=a,b=b):
            if k:
                return -cosh(a*k)/sinh(b*k)
            return 0
        
        # 初始化卷积核 omega
        omega = convolve.init_convolution_kernel(n,kernel,d=1)
        
        # 将计算结果存入缓存
        _cache[(n,a,b)] = omega
    
    # 检查是否需要覆写输入的数据
    overwrite_x = _datacopied(tmp, x)
    
    # 返回卷积运算的结果，swap_real_imag=1 表示交换实部和虚部，overwrite_x 表示是否覆写输入数据
    return convolve.convolve(tmp,omega,swap_real_imag=1,overwrite_x=overwrite_x)

# 删除 _cache 变量，释放内存
del _cache


# 重新初始化一个空的缓存字典
_cache = {}

# 定义函数 sc_diff，计算周期序列 x 的 (a,b)-sinh/cosh 伪导数
def sc_diff(x, a, b, period=None, _cache=_cache):
    """
    Return (a,b)-sinh/cosh pseudo-derivative of a periodic sequence x.

    If x_j and y_j are Fourier coefficients of periodic functions x
    and y, respectively, then::

      y_j = sqrt(-1)*sinh(j*a*2*pi/period)/cosh(j*b*2*pi/period) * x_j
      y_0 = 0

    Parameters
    ----------
    x : array_like
        Input array.
    a,b : float
        Defines the parameters of the sinh/cosh pseudo-differential
        operator.
    period : float, optional
        The period of the sequence x. Default is 2*pi.

    Returns
    -------
    sc_diff : ndarray
        Pseudo-derivative of periodic sequence x.

    Notes
    -----
    ``sc_diff(cs_diff(x,a,b),b,a) == x``
    For even ``len(x)``, the Nyquist mode of x is taken as zero.
    """
    # 将输入 x 转换为数组
    tmp = asarray(x)
    
    # 如果 x 是复数对象，则分别计算其实部和虚部的伪导数
    if iscomplexobj(tmp):
        return sc_diff(tmp.real,a,b,period) + \
               1j*sc_diff(tmp.imag,a,b,period)
    
    # 如果指定了周期 period，则重新计算参数 a 和 b
    if period is not None:
        a = a*2*pi/period
        b = b*2*pi/period
    
    # 获取序列 x 的长度 n
    n = len(x)
    
    # 尝试从缓存中获取已计算的结果 omega
    omega = _cache.get((n,a,b))
    
    # 如果 omega 不存在，则计算新的伪导数核
    if omega is None:
        # 如果缓存长度超过 20，则清空缓存
        if len(_cache) > 20:
            while _cache:
                _cache.popitem()

        # 定义伪导数核函数 kernel(k,a,b)，当 k 不为零时返回 sinh(a*k)/cosh(b*k)，否则返回 0
        def kernel(k,a=a,b=b):
            if k:
                return sinh(a*k)/cosh(b*k)
            return 0
        
        # 初始化卷积核 omega
        omega = convolve.init_convolution_kernel(n,kernel,d=1)
        
        # 将计算结果存入缓存
        _cache[(n,a,b)] = omega
    
    # 检查是否需要覆写输入的数据
    overwrite_x = _datacopied(tmp, x)
    
    # 返回卷积运算的结果，swap_real_imag=1 表示交换实部和虚部，overwrite_x 表示是否覆写输入数据
    return convolve.convolve(tmp,omega,swap_real_imag=1,overwrite_x=overwrite_x)

# 删除 _cache 变量，释放内存
del _cache
# 全局变量，用于缓存计算结果
_cache = {}

# 定义了一个函数 ss_diff，计算周期序列 x 的 (a,b)-sinh/sinh 伪导数
def ss_diff(x, a, b, period=None, _cache=_cache):
    """
    Return (a,b)-sinh/sinh pseudo-derivative of a periodic sequence x.

    If x_j and y_j are Fourier coefficients of periodic functions x
    and y, respectively, then::

      y_j = sinh(j*a*2*pi/period)/sinh(j*b*2*pi/period) * x_j
      y_0 = a/b * x_0

    Parameters
    ----------
    x : array_like
        The array to take the pseudo-derivative from.
    a,b
        Defines the parameters of the sinh/sinh pseudo-differential
        operator.
    period : float, optional
        The period of the sequence x. Default is ``2*pi``.

    Notes
    -----
    ``ss_diff(ss_diff(x,a,b),b,a) == x``

    """
    # 将输入转换为数组 tmp
    tmp = asarray(x)
    # 如果输入是复数对象，则分别对实部和虚部计算伪导数
    if iscomplexobj(tmp):
        return ss_diff(tmp.real,a,b,period) + \
               1j*ss_diff(tmp.imag,a,b,period)
    # 如果指定了 period，则根据 period 调整 a 和 b
    if period is not None:
        a = a*2*pi/period
        b = b*2*pi/period
    # 获取序列 x 的长度 n
    n = len(x)
    # 从缓存中获取已计算的 omega
    omega = _cache.get((n,a,b))
    # 如果 omega 不存在，则计算它
    if omega is None:
        # 如果缓存的条目超过 20 条，则清空缓存
        if len(_cache) > 20:
            while _cache:
                _cache.popitem()

        # 定义核函数 kernel，用于计算伪导数的权重
        def kernel(k,a=a,b=b):
            if k:
                return sinh(a*k)/sinh(b*k)
            return float(a)/b
        # 初始化卷积核 omega
        omega = convolve.init_convolution_kernel(n,kernel)
        # 将计算结果存入缓存
        _cache[(n,a,b)] = omega
    # 复制输入数据，以防止修改原始数据
    overwrite_x = _datacopied(tmp, x)
    # 返回卷积操作的结果
    return convolve.convolve(tmp,omega,overwrite_x=overwrite_x)


# 删除全局变量 _cache，避免全局变量之间的干扰
del _cache

# 重新定义全局变量 _cache，用于缓存计算结果
_cache = {}

# 定义了一个函数 cc_diff，计算周期序列 x 的 (a,b)-cosh/cosh 伪导数
def cc_diff(x, a, b, period=None, _cache=_cache):
    """
    Return (a,b)-cosh/cosh pseudo-derivative of a periodic sequence.

    If x_j and y_j are Fourier coefficients of periodic functions x
    and y, respectively, then::

      y_j = cosh(j*a*2*pi/period)/cosh(j*b*2*pi/period) * x_j

    Parameters
    ----------
    x : array_like
        The array to take the pseudo-derivative from.
    a,b : float
        Defines the parameters of the sinh/sinh pseudo-differential
        operator.
    period : float, optional
        The period of the sequence x. Default is ``2*pi``.

    Returns
    -------
    cc_diff : ndarray
        Pseudo-derivative of periodic sequence `x`.

    Notes
    -----
    ``cc_diff(cc_diff(x,a,b),b,a) == x``

    """
    # 将输入转换为数组 tmp
    tmp = asarray(x)
    # 如果输入是复数对象，则分别对实部和虚部计算伪导数
    if iscomplexobj(tmp):
        return cc_diff(tmp.real,a,b,period) + \
               1j*cc_diff(tmp.imag,a,b,period)
    # 如果指定了 period，则根据 period 调整 a 和 b
    if period is not None:
        a = a*2*pi/period
        b = b*2*pi/period
    # 获取序列 x 的长度 n
    n = len(x)
    # 从缓存中获取已计算的 omega
    omega = _cache.get((n,a,b))
    # 如果 omega 不存在，则计算它
    if omega is None:
        # 如果缓存的条目超过 20 条，则清空缓存
        if len(_cache) > 20:
            while _cache:
                _cache.popitem()

        # 定义核函数 kernel，用于计算伪导数的权重
        def kernel(k,a=a,b=b):
            return cosh(a*k)/cosh(b*k)
        # 初始化卷积核 omega
        omega = convolve.init_convolution_kernel(n,kernel)
        # 将计算结果存入缓存
        _cache[(n,a,b)] = omega
    # 复制输入数据，以防止修改原始数据
    overwrite_x = _datacopied(tmp, x)
    # 返回卷积操作的结果
    return convolve.convolve(tmp,omega,overwrite_x=overwrite_x)


# 删除全局变量 _cache，避免全局变量之间的干扰
del _cache

# 定义了一个函数 shift，用于对周期序列 x 进行平移操作
def shift(x, a, period=None, _cache=_cache):
    """
    Shift periodic sequence x by a: y(u) = x(u+a).

    """
    """
    If x_j and y_j are Fourier coefficients of periodic functions x
    and y, respectively, then::

          y_j = exp(j*a*2*pi/period*sqrt(-1)) * x_f

    Parameters
    ----------
    x : array_like
        The array to take the pseudo-derivative from.
    a : float
        Defines the parameters of the sinh/sinh pseudo-differential
    period : float, optional
        The period of the sequences x and y. Default period is ``2*pi``.
    """
    # 将输入数组 x 转换为 ndarray 类型
    tmp = asarray(x)
    # 如果 tmp 是复数对象，则对实部和虚部进行位移
    if iscomplexobj(tmp):
        return shift(tmp.real,a,period)+1j*shift(tmp.imag,a,period)
    # 如果指定了 period，则重新计算 a 的值
    if period is not None:
        a = a*2*pi/period
    # 获取输入数组 x 的长度 n
    n = len(x)
    # 从缓存中获取或初始化 omega
    omega = _cache.get((n,a))
    if omega is None:
        # 如果缓存中的项目数超过 20，则清空缓存
        if len(_cache) > 20:
            while _cache:
                _cache.popitem()

        # 定义实部和虚部的卷积核函数
        def kernel_real(k,a=a):
            return cos(a*k)

        def kernel_imag(k,a=a):
            return sin(a*k)
        
        # 初始化卷积核 omega_real 和 omega_imag
        omega_real = convolve.init_convolution_kernel(n,kernel_real,d=0,
                                                      zero_nyquist=0)
        omega_imag = convolve.init_convolution_kernel(n,kernel_imag,d=1,
                                                      zero_nyquist=0)
        # 将计算结果存入缓存
        _cache[(n,a)] = omega_real,omega_imag
    else:
        omega_real,omega_imag = omega
    
    # 检查是否需要复制输入数据 tmp 到 x
    overwrite_x = _datacopied(tmp, x)
    # 返回卷积运算的结果
    return convolve.convolve_z(tmp,omega_real,omega_imag,
                               overwrite_x=overwrite_x)
# 删除变量 _cache，从当前命名空间中移除该变量
del _cache
```