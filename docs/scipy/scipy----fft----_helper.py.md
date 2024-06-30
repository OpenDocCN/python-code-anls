# `D:\src\scipysrc\scipy\scipy\fft\_helper.py`

```
# 导入必要的模块和函数：update_wrapper 用于复制函数属性，lru_cache 用于缓存函数调用结果，inspect 用于检查函数签名
from functools import update_wrapper, lru_cache
import inspect

# 导入 C 函数 _helper.good_size 作为 _helper
from ._pocketfft import helper as _helper

# 导入 numpy 库，并从 scipy._lib._array_api 中导入 array_namespace 函数
import numpy as np
from scipy._lib._array_api import array_namespace

# 定义函数 next_fast_len，用于查找比输入数据长度更快的 FFT 大小，用于零填充等操作
def next_fast_len(target, real=False):
    """Find the next fast size of input data to ``fft``, for zero-padding, etc.

    SciPy's FFT algorithms gain their speed by a recursive divide and conquer
    strategy. This relies on efficient functions for small prime factors of the
    input length. Thus, the transforms are fastest when using composites of the
    prime factors handled by the fft implementation. If there are efficient
    functions for all radices <= `n`, then the result will be a number `x`
    >= ``target`` with only prime factors < `n`. (Also known as `n`-smooth
    numbers)

    Parameters
    ----------
    target : int
        Length to start searching from. Must be a positive integer.
    real : bool, optional
        True if the FFT involves real input or output (e.g., `rfft` or `hfft`
        but not `fft`). Defaults to False.

    Returns
    -------
    out : int
        The smallest fast length greater than or equal to ``target``.

    Notes
    -----
    The result of this function may change in future as performance
    considerations change, for example, if new prime factors are added.

    Calling `fft` or `ifft` with real input data performs an ``'R2C'``
    transform internally.

    Examples
    --------
    On a particular machine, an FFT of prime length takes 11.4 ms:

    >>> from scipy import fft
    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> min_len = 93059  # prime length is worst case for speed
    >>> a = rng.standard_normal(min_len)
    >>> b = fft.fft(a)

    Zero-padding to the next regular length reduces computation time to
    1.6 ms, a speedup of 7.3 times:

    >>> fft.next_fast_len(min_len, real=True)
    93312
    >>> b = fft.fft(a, 93312)

    Rounding up to the next power of 2 is not optimal, taking 3.0 ms to
    compute; 1.9 times longer than the size given by ``next_fast_len``:

    >>> b = fft.fft(a, 131072)

    """
    pass

# 使用 update_wrapper 复制 next_fast_len 函数的属性到 _helper.good_size 函数
_sig = inspect.signature(next_fast_len)
next_fast_len = update_wrapper(lru_cache(_helper.good_size), next_fast_len)
# 将 _helper.good_size 函数作为 lru_cache 的装饰器应用到 next_fast_len 上
next_fast_len.__wrapped__ = _helper.good_size
# 将函数签名恢复为原始的 next_fast_len 的签名
next_fast_len.__signature__ = _sig


# 定义函数 prev_fast_len，用于查找比输入数据长度更快的 FFT 大小，适用于在 FFT 之前丢弃最小数量的样本
def prev_fast_len(target, real=False):
    """Find the previous fast size of input data to ``fft``.
    Useful for discarding a minimal number of samples before FFT.

    SciPy's FFT algorithms gain their speed by a recursive divide and conquer
    strategy. This relies on efficient functions for small prime factors of the
    input length. Thus, the transforms are fastest when using composites of the
    prime factors handled by the fft implementation. If there are efficient
    functions for all radices <= `n`, then the result will be a number `x`
    # <= ``target`` with only prime factors <= `n`. (Also known as `n`-smooth
    # numbers)
    
    # Parameters
    # ----------
    # target : int
    #     Maximum length to search until. Must be a positive integer.
    # real : bool, optional
    #     True if the FFT involves real input or output (e.g., `rfft` or `hfft`
    #     but not `fft`). Defaults to False.
    
    # Returns
    # -------
    # out : int
    #     The largest fast length less than or equal to ``target``.
    
    # Notes
    # -----
    # The result of this function may change in future as performance
    # considerations change, for example, if new prime factors are added.
    
    # Calling `fft` or `ifft` with real input data performs an ``'R2C'``
    # transform internally.
    
    # In the current implementation, prev_fast_len assumes radices of
    # 2,3,5,7,11 for complex FFT and 2,3,5 for real FFT.
    
    # Examples
    # --------
    # On a particular machine, an FFT of prime length takes 16.2 ms:
    
    # >>> from scipy import fft
    # >>> import numpy as np
    # >>> rng = np.random.default_rng()
    # >>> max_len = 93059  # prime length is worst case for speed
    # >>> a = rng.standard_normal(max_len)
    # >>> b = fft.fft(a)
    
    # Performing FFT on the maximum fast length less than max_len
    # reduces the computation time to 1.5 ms, a speedup of 10.5 times:
    
    # >>> fft.prev_fast_len(max_len, real=True)
    # 92160
    # >>> c = fft.fft(a[:92160]) # discard last 899 samples
# 直接包装 c 函数 prev_good_size，但从上面的 prev_fast_len 函数获取文档字符串等信息
_sig_prev_fast_len = inspect.signature(prev_fast_len)
# 使用 lru_cache 装饰器包装 prev_good_size 函数，并用 prev_fast_len 的信息更新其文档和签名
prev_fast_len = update_wrapper(lru_cache()(_helper.prev_good_size), prev_fast_len)
# 将 __wrapped__ 属性设置为 _helper.prev_good_size，以便在需要时访问原始函数
prev_fast_len.__wrapped__ = _helper.prev_good_size
# 将函数签名设置为 _sig_prev_fast_len，以确保保留正确的签名信息
prev_fast_len.__signature__ = _sig_prev_fast_len


def _init_nd_shape_and_axes(x, shape, axes):
    """处理 N 维变换的 shape 和 axes 参数。

    返回标准形式的 shape 和 axes，考虑负值并检查各种潜在的错误。

    Parameters
    ----------
    x : array_like
        输入数组。
    shape : int 或者 int 数组或者 None
        结果的形状。如果 `shape` 和 `axes`（见下文）都为 None，则 `shape` 是 `x.shape`；
        如果 `shape` 为 None 但 `axes` 不为 None，则 `shape` 是 `numpy.take(x.shape, axes, axis=0)`。
        如果 `shape` 为 -1，则使用 `x` 对应维度的大小。
    axes : int 或者 int 数组或者 None
        计算进行的轴。
        默认为所有轴。
        负索引会自动转换为其正值。

    Returns
    -------
    shape : tuple
        结果的形状，以整数元组形式。
    axes : list
        计算进行的轴，以整数列表形式。

    """
    x = np.asarray(x)
    return _helper._init_nd_shape_and_axes(x, shape, axes)


def fftfreq(n, d=1.0, *, xp=None, device=None):
    """返回离散傅里叶变换的样本频率。

    返回的浮点数组 `f` 包含以单位采样间隔的周期数为单位的频率 bin 中心（从零开始）。
    例如，如果采样间隔单位为秒，则频率单位为每秒的周期数。

    给定窗口长度 `n` 和采样间隔 `d`::

      f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   如果 n 是偶数
      f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   如果 n 是奇数

    Parameters
    ----------
    n : int
        窗口长度。
    d : scalar, 可选
        采样间隔（采样率的倒数）。默认为 1。
    xp : array_namespace, 可选
        返回数组的命名空间。默认为 None，即使用 NumPy。
    device : device, 可选
        返回数组的设备。
        仅当 `xp.fft.fftfreq` 实现了设备参数时有效。

    Returns
    -------
    f : ndarray
        长度为 `n` 的样本频率数组。

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.fft
    >>> signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)
    >>> fourier = scipy.fft.fft(signal)
    >>> n = signal.size
    >>> timestep = 0.1
    >>> freq = scipy.fft.fftfreq(n, d=timestep)
    >>> freq
    array([ 0.  ,  1.25,  2.5 , ..., -3.75, -2.5 , -1.25])

    """
    xp = np if xp is None else xp
    # 设置 xp 变量为 numpy，如果 xp 为 None 则使用原来的值
    # numpy 目前不支持 `device` 关键字
    # 当 numpy 兼容时应移除 `xp.__name__ != 'numpy'`
    if hasattr(xp, 'fft') and xp.__name__ != 'numpy':
        # 如果 xp 变量有 fft 属性并且不是 numpy 模块，则调用 xp 的 fft.fftfreq 方法
        return xp.fft.fftfreq(n, d=d, device=device)
    if device is not None:
        # 如果 device 参数不为 None，则抛出数值错误异常，因为输入数组类型不支持设备参数
        raise ValueError('device parameter is not supported for input array type')
    # 返回 numpy 的 fft.fftfreq 方法计算得到的频率数组
    return np.fft.fftfreq(n, d=d)
# 定义一个函数 rfftfreq，用于返回离散傅里叶变换的采样频率数组
def rfftfreq(n, d=1.0, *, xp=None, device=None):
    """Return the Discrete Fourier Transform sample frequencies
    (for usage with rfft, irfft).

    The returned float array `f` contains the frequency bin centers in cycles
    per unit of the sample spacing (with zero at the start).  For instance, if
    the sample spacing is in seconds, then the frequency unit is cycles/second.

    Given a window length `n` and a sample spacing `d`::

      f = [0, 1, ...,     n/2-1,     n/2] / (d*n)   if n is even
      f = [0, 1, ..., (n-1)/2-1, (n-1)/2] / (d*n)   if n is odd

    Unlike `fftfreq` (but like `scipy.fftpack.rfftfreq`)
    the Nyquist frequency component is considered to be positive.

    Parameters
    ----------
    n : int
        窗口长度。
    d : scalar, optional
        采样间隔（采样率的倒数）。默认为1。
    xp : array_namespace, optional
        返回数组的命名空间。默认为 None，表示使用 NumPy。
    device : device, optional
        返回数组的设备。
        仅当 `xp.fft.rfftfreq` 实现了设备参数时有效。

    Returns
    -------
    f : ndarray
        长度为 ``n//2 + 1`` 的采样频率数组。

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.fft
    >>> signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5, -3, 4], dtype=float)
    >>> fourier = scipy.fft.rfft(signal)
    >>> n = signal.size
    >>> sample_rate = 100
    >>> freq = scipy.fft.fftfreq(n, d=1./sample_rate)
    >>> freq
    array([  0.,  10.,  20., ..., -30., -20., -10.])
    >>> freq = scipy.fft.rfftfreq(n, d=1./sample_rate)
    >>> freq
    array([  0.,  10.,  20.,  30.,  40.,  50.])

    """
    # 如果 xp 为 None，则使用 NumPy；否则使用传入的 xp 参数
    xp = np if xp is None else xp
    # 如果 xp 具有 'fft' 属性且 xp 的名称不是 'numpy'，调用 xp.fft.rfftfreq 函数
    if hasattr(xp, 'fft') and xp.__name__ != 'numpy':
        return xp.fft.rfftfreq(n, d=d, device=device)
    # 如果设备参数不为 None，则抛出值错误异常，因为对于输入数组类型，不支持设备参数
    if device is not None:
        raise ValueError('device parameter is not supported for input array type')
    # 否则调用 NumPy 的 np.fft.rfftfreq 函数
    return np.fft.rfftfreq(n, d=d)


# 定义一个函数 fftshift，用于将零频率分量移动到频谱中心
def fftshift(x, axes=None):
    """Shift the zero-frequency component to the center of the spectrum.

    This function swaps half-spaces for all axes listed (defaults to all).
    Note that ``y[0]`` is the Nyquist component only if ``len(x)`` is even.

    Parameters
    ----------
    x : array_like
        输入数组。
    axes : int or shape tuple, optional
        要移动的轴。默认为 None，表示移动所有轴。

    Returns
    -------
    y : ndarray
        移动后的数组。

    See Also
    --------
    ifftshift : `fftshift` 的反变换。

    Examples
    --------
    >>> import numpy as np
    >>> freqs = np.fft.fftfreq(10, 0.1)
    >>> freqs
    array([ 0.,  1.,  2., ..., -3., -2., -1.])
    >>> np.fft.fftshift(freqs)
    array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])


    """
    # 调用 NumPy 的 np.fft.fftshift 函数，将输入数组 x 的零频率分量移动到频谱中心
    return np.fft.fftshift(x, axes=axes)
    Shift the zero-frequency component only along the second axis:

    >>> freqs = np.fft.fftfreq(9, d=1./9).reshape(3, 3)
    >>> freqs
    array([[ 0.,  1.,  2.],
           [ 3.,  4., -4.],
           [-3., -2., -1.]])
    >>> np.fft.fftshift(freqs, axes=(1,))
    array([[ 2.,  0.,  1.],
           [-4.,  3.,  4.],
           [-1., -3., -2.]])

    """
    xp = array_namespace(x)  # 使用 array_namespace 函数处理输入的数组 x，返回处理后的数组命名空间 xp
    if hasattr(xp, 'fft'):   # 检查 xp 是否具有 fft 属性（假设这是一个扩展的数组库）
        return xp.fft.fftshift(x, axes=axes)  # 如果有 fft 属性，调用 xp 的 fftshift 方法对 x 进行频率移位
    x = np.asarray(x)  # 将输入 x 转换为 NumPy 数组
    y = np.fft.fftshift(x, axes=axes)  # 对 x 进行频率移位，结果保存在 y 中
    return xp.asarray(y)  # 将 y 转换为 xp 类型的数组并返回
def ifftshift(x, axes=None):
    """The inverse of `fftshift`. Although identical for even-length `x`, the
    functions differ by one sample for odd-length `x`.

    Parameters
    ----------
    x : array_like
        Input array.
    axes : int or shape tuple, optional
        Axes over which to calculate.  Defaults to None, which shifts all axes.

    Returns
    -------
    y : ndarray
        The shifted array.

    See Also
    --------
    fftshift : Shift zero-frequency component to the center of the spectrum.

    Examples
    --------
    >>> import numpy as np
    >>> freqs = np.fft.fftfreq(9, d=1./9).reshape(3, 3)
    >>> freqs
    array([[ 0.,  1.,  2.],
           [ 3.,  4., -4.],
           [-3., -2., -1.]])
    >>> np.fft.ifftshift(np.fft.fftshift(freqs))
    array([[ 0.,  1.,  2.],
           [ 3.,  4., -4.],
           [-3., -2., -1.]])
    
    """
    xp = array_namespace(x)
    # 如果输入数组 x 所属的命名空间 xp 具有 'fft' 属性，则使用其 ifftshift 方法
    if hasattr(xp, 'fft'):
        return xp.fft.ifftshift(x, axes=axes)
    # 否则，将 x 转换为 NumPy 数组
    x = np.asarray(x)
    # 对 x 进行反移位操作，返回结果
    y = np.fft.ifftshift(x, axes=axes)
    return xp.asarray(y)
```