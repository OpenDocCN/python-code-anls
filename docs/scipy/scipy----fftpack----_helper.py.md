# `D:\src\scipysrc\scipy\scipy\fftpack\_helper.py`

```
import operator  # 导入 Python 内置的 operator 模块

import numpy as np  # 导入 NumPy 库，并使用 np 别名
from numpy.fft import fftshift, ifftshift, fftfreq  # 从 NumPy 的 fft 子模块中导入 fftshift, ifftshift, fftfreq 函数

import scipy.fft._pocketfft.helper as _helper  # 导入 scipy 中 FFT 相关模块的 helper 函数

__all__ = ['fftshift', 'ifftshift', 'fftfreq', 'rfftfreq', 'next_fast_len']  # 定义模块中公开的接口列表


def rfftfreq(n, d=1.0):
    """DFT sample frequencies (for usage with rfft, irfft).

    The returned float array contains the frequency bins in
    cycles/unit (with zero at the start) given a window length `n` and a
    sample spacing `d`::

      f = [0,1,1,2,2,...,n/2-1,n/2-1,n/2]/(d*n)   if n is even
      f = [0,1,1,2,2,...,n/2-1,n/2-1,n/2,n/2]/(d*n)   if n is odd

    Parameters
    ----------
    n : int
        Window length.
    d : scalar, optional
        Sample spacing. Default is 1.

    Returns
    -------
    out : ndarray
        The array of length `n`, containing the sample frequencies.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import fftpack
    >>> sig = np.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)
    >>> sig_fft = fftpack.rfft(sig)
    >>> n = sig_fft.size
    >>> timestep = 0.1
    >>> freq = fftpack.rfftfreq(n, d=timestep)
    >>> freq
    array([ 0.  ,  1.25,  1.25,  2.5 ,  2.5 ,  3.75,  3.75,  5.  ])

    """
    n = operator.index(n)  # 将 n 转换为整数索引类型
    if n < 0:
        raise ValueError("n = %s is not valid. "
                         "n must be a nonnegative integer." % n)

    return (np.arange(1, n + 1, dtype=int) // 2) / float(n * d)  # 返回频率数组


def next_fast_len(target):
    """
    Find the next fast size of input data to `fft`, for zero-padding, etc.

    SciPy's FFTPACK has efficient functions for radix {2, 3, 4, 5}, so this
    returns the next composite of the prime factors 2, 3, and 5 which is
    greater than or equal to `target`. (These are also known as 5-smooth
    numbers, regular numbers, or Hamming numbers.)

    Parameters
    ----------
    target : int
        Length to start searching from. Must be a positive integer.

    Returns
    -------
    out : int
        The first 5-smooth number greater than or equal to `target`.

    Notes
    -----
    .. versionadded:: 0.18.0

    Examples
    --------
    On a particular machine, an FFT of prime length takes 133 ms:

    >>> from scipy import fftpack
    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> min_len = 10007  # prime length is worst case for speed
    >>> a = rng.standard_normal(min_len)
    >>> b = fftpack.fft(a)

    Zero-padding to the next 5-smooth length reduces computation time to
    211 us, a speedup of 630 times:

    >>> fftpack.next_fast_len(min_len)
    10125
    >>> b = fftpack.fft(a, 10125)

    Rounding up to the next power of 2 is not optimal, taking 367 us to
    compute, 1.7 times as long as the 5-smooth size:

    >>> b = fftpack.fft(a, 16384)

    """
    # Real transforms use regular sizes so this is backwards compatible
    return _helper.good_size(target, True)  # 调用 helper 模块中的 good_size 函数，返回最接近的 5-smooth 数字


def _good_shape(x, shape, axes):
    """Ensure that shape argument is valid for scipy.fftpack
    """
    scipy.fftpack does not support len(shape) < x.ndim when axes is not given.
    """
    如果 shape 不为 None 并且 axes 为 None：
        # 将 shape 转换为整数的可迭代对象，如果不是整数会引发异常
        shape = _helper._iterable_of_int(shape, 'shape')
        # 如果 shape 的长度不等于 x 的维度数，抛出数值错误异常
        if len(shape) != np.ndim(x):
            raise ValueError("when given, axes and shape arguments"
                             " have to be of the same length")
    # 返回 shape
    return shape
```