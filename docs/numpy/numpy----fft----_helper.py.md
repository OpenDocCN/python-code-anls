# `.\numpy\numpy\fft\_helper.py`

```py
"""
Discrete Fourier Transforms - _helper.py

"""
from numpy._core import integer, empty, arange, asarray, roll
from numpy._core.overrides import array_function_dispatch, set_module

# Created by Pearu Peterson, September 2002

__all__ = ['fftshift', 'ifftshift', 'fftfreq', 'rfftfreq']

integer_types = (int, integer)


def _fftshift_dispatcher(x, axes=None):
    return (x,)


@array_function_dispatch(_fftshift_dispatcher, module='numpy.fft')
def fftshift(x, axes=None):
    """
    Shift the zero-frequency component to the center of the spectrum.

    This function swaps half-spaces for all axes listed (defaults to all).
    Note that ``y[0]`` is the Nyquist component only if ``len(x)`` is even.

    Parameters
    ----------
    x : array_like
        Input array.
    axes : int or shape tuple, optional
        Axes over which to shift.  Default is None, which shifts all axes.

    Returns
    -------
    y : ndarray
        The shifted array.

    See Also
    --------
    ifftshift : The inverse of `fftshift`.

    Examples
    --------
    >>> freqs = np.fft.fftfreq(10, 0.1)
    >>> freqs
    array([ 0.,  1.,  2., ..., -3., -2., -1.])
    >>> np.fft.fftshift(freqs)
    array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])

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
    # 将输入数组转换为 ndarray 类型
    x = asarray(x)
    # 如果未指定轴，则默认对所有轴进行移动
    if axes is None:
        axes = tuple(range(x.ndim))
        # 计算每个轴要移动的步数
        shift = [dim // 2 for dim in x.shape]
    # 如果 axes 是整数类型，则只移动指定轴的一半长度
    elif isinstance(axes, integer_types):
        shift = x.shape[axes] // 2
    # 否则，移动每个指定轴的一半长度
    else:
        shift = [x.shape[ax] // 2 for ax in axes]

    # 对数组进行轴向移动
    return roll(x, shift, axes)


@array_function_dispatch(_fftshift_dispatcher, module='numpy.fft')
def ifftshift(x, axes=None):
    """
    The inverse of `fftshift`. Although identical for even-length `x`, the
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
    # 将输入数组转换为 ndarray 类型
    x = asarray(x)
    # 如果未指定轴，则默认对所有轴进行移动
    if axes is None:
        axes = tuple(range(x.ndim))
        # 计算每个轴要移动的步数（负数表示反向移动）
        shift = [-(dim // 2) for dim in x.shape]
    # 如果 axes 是整数类型，则计算要滚动的偏移量
    elif isinstance(axes, integer_types):
        shift = -(x.shape[axes] // 2)
    # 如果 axes 是一个迭代对象（通常是一个列表或元组），则分别计算每个轴的偏移量
    else:
        shift = [-(x.shape[ax] // 2) for ax in axes]

    # 使用 roll 函数对数组 x 进行滚动操作，滚动的偏移量由 shift 决定，作用在指定的轴 axes 上
    return roll(x, shift, axes)
# 设置模块为 'numpy.fft'
@set_module('numpy.fft')
# 定义函数 fftfreq，返回离散傅里叶变换的采样频率
def fftfreq(n, d=1.0, device=None):
    """
    Return the Discrete Fourier Transform sample frequencies.

    The returned float array `f` contains the frequency bin centers in cycles
    per unit of the sample spacing (with zero at the start).  For instance, if
    the sample spacing is in seconds, then the frequency unit is cycles/second.

    Given a window length `n` and a sample spacing `d`::

      f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
      f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd

    Parameters
    ----------
    n : int
        窗口长度。
    d : scalar, optional
        采样间隔（采样率的倒数）。默认为 1。
    device : str, optional
        创建数组时指定的设备。默认为 ``None``。
        仅供 Array-API 兼容性使用，如果传入，必须是 ``"cpu"``。

        .. versionadded:: 2.0.0

    Returns
    -------
    f : ndarray
        长度为 `n` 的数组，包含采样频率。

    Examples
    --------
    >>> signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)
    >>> fourier = np.fft.fft(signal)
    >>> n = signal.size
    >>> timestep = 0.1
    >>> freq = np.fft.fftfreq(n, d=timestep)
    >>> freq
    array([ 0.  ,  1.25,  2.5 , ..., -3.75, -2.5 , -1.25])

    """
    # 检查 n 是否为整数类型
    if not isinstance(n, integer_types):
        raise ValueError("n should be an integer")
    # 计算 val，用于后续频率计算
    val = 1.0 / (n * d)
    # 创建一个长度为 n 的空数组 results，数据类型为 int，可指定设备
    results = empty(n, int, device=device)
    # 计算 N，为正频率部分的长度
    N = (n-1)//2 + 1
    # 生成从 0 到 N-1 的整数数组 p1，数据类型为 int，可指定设备
    p1 = arange(0, N, dtype=int, device=device)
    # 将 p1 复制到 results 的前 N 个元素
    results[:N] = p1
    # 生成从 -(n//2) 到 -1 的整数数组 p2，数据类型为 int，可指定设备
    p2 = arange(-(n//2), 0, dtype=int, device=device)
    # 将 p2 复制到 results 的后半部分
    results[N:] = p2
    # 返回结果数组乘以 val
    return results * val


# 设置模块为 'numpy.fft'
@set_module('numpy.fft')
# 定义函数 rfftfreq，返回用于实部快速傅里叶变换的采样频率
def rfftfreq(n, d=1.0, device=None):
    """
    Return the Discrete Fourier Transform sample frequencies
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
        采样间隔（采样率的倒数）。默认为 1。
    device : str, optional
        创建数组时指定的设备。默认为 ``None``。
        仅供 Array-API 兼容性使用，如果传入，必须是 ``"cpu"``。

        .. versionadded:: 2.0.0

    Returns
    -------
    f : ndarray
        长度为 ``n//2 + 1`` 的数组，包含采样频率。

    Examples
    --------
    # 创建一个 numpy 数组，表示输入的信号，其中包含浮点数
    signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5, -3, 4], dtype=float)
    
    # 对信号进行快速傅立叶变换（FFT），返回变换后的复数数组
    fourier = np.fft.rfft(signal)
    
    # 获取信号的长度（即样本数）
    n = signal.size
    
    # 设置采样率为 100
    sample_rate = 100
    
    # 使用 FFT 计算频率数组，以实数形式返回频率值
    freq = np.fft.fftfreq(n, d=1./sample_rate)
    
    # 输出频率数组
    freq
    array([  0.,  10.,  20., ..., -30., -20., -10.])
    
    # 使用 rfft 方法计算频率数组，以实数形式返回频率值，仅包括非负部分
    freq = np.fft.rfftfreq(n, d=1./sample_rate)
    
    # 输出频率数组
    freq
    array([  0.,  10.,  20.,  30.,  40.,  50.])
    
    """
    如果 n 不是整数类型（integer_types），则抛出数值错误
    n 应该是一个整数
    val = 1.0/(n*d)  # 计算频率间隔
    N = n//2 + 1  # 计算频率数组的长度
    results = arange(0, N, dtype=int, device=device)  # 生成一个整数数组，表示频率索引
    return results * val  # 返回频率数组乘以频率间隔后的结果
```