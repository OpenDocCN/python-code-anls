# `D:\src\scipysrc\scipy\scipy\fftpack\_basic.py`

```
"""
Discrete Fourier Transforms - _basic.py
"""
# Created by Pearu Peterson, August,September 2002
# 导入必要的函数和模块
__all__ = ['fft','ifft','fftn','ifftn','rfft','irfft',
           'fft2','ifft2']

# 导入_scipy.fft模块中的_pocketfft函数和_helper模块中的_good_shape函数
from scipy.fft import _pocketfft
from ._helper import _good_shape


def fft(x, n=None, axis=-1, overwrite_x=False):
    """
    Return discrete Fourier transform of real or complex sequence.

    The returned complex array contains ``y(0), y(1),..., y(n-1)``, where

    ``y(j) = (x * exp(-2*pi*sqrt(-1)*j*np.arange(n)/n)).sum()``.

    Parameters
    ----------
    x : array_like
        Array to Fourier transform.
    n : int, optional
        Length of the Fourier transform. If ``n < x.shape[axis]``, `x` is
        truncated. If ``n > x.shape[axis]``, `x` is zero-padded. The
        default results in ``n = x.shape[axis]``.
    axis : int, optional
        Axis along which the fft's are computed; the default is over the
        last axis (i.e., ``axis=-1``).
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.

    Returns
    -------
    z : complex ndarray
        with the elements::

            [y(0),y(1),..,y(n/2),y(1-n/2),...,y(-1)]        if n is even
            [y(0),y(1),..,y((n-1)/2),y(-(n-1)/2),...,y(-1)]  if n is odd

        where::

            y(j) = sum[k=0..n-1] x[k] * exp(-sqrt(-1)*j*k* 2*pi/n), j = 0..n-1

    See Also
    --------
    ifft : Inverse FFT
    rfft : FFT of a real sequence

    Notes
    -----
    The packing of the result is "standard": If ``A = fft(a, n)``, then
    ``A[0]`` contains the zero-frequency term, ``A[1:n/2]`` contains the
    positive-frequency terms, and ``A[n/2:]`` contains the negative-frequency
    terms, in order of decreasingly negative frequency. So ,for an 8-point
    transform, the frequencies of the result are [0, 1, 2, 3, -4, -3, -2, -1].
    To rearrange the fft output so that the zero-frequency component is
    centered, like [-4, -3, -2, -1,  0,  1,  2,  3], use `fftshift`.

    Both single and double precision routines are implemented. Half precision
    inputs will be converted to single precision. Non-floating-point inputs
    will be converted to double precision. Long-double precision inputs are
    not supported.

    This function is most efficient when `n` is a power of two, and least
    efficient when `n` is prime.

    Note that if ``x`` is real-valued, then ``A[j] == A[n-j].conjugate()``.
    If ``x`` is real-valued and ``n`` is even, then ``A[n/2]`` is real.

    If the data type of `x` is real, a "real FFT" algorithm is automatically
    used, which roughly halves the computation time. To increase efficiency
    a little further, use `rfft`, which does the same calculation, but only
    outputs half of the symmetrical spectrum. If the data is both real and
    symmetrical, the `dct` can again double the efficiency by generating
    half of the spectrum from half of the signal.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.fftpack import fft, ifft
    导入 SciPy 库中的 FFT 和 IFFT 函数

    >>> x = np.arange(5)
    创建一个 NumPy 数组 x，包含从 0 到 4 的整数序列

    >>> np.allclose(fft(ifft(x)), x, atol=1e-15)  # within numerical accuracy.
    对 x 进行 IFFT（逆傅里叶变换），然后再对结果进行 FFT（傅里叶变换），并检查其是否与 x 在数值精度内相等

    """
    return _pocketfft.fft(x, n, axis, None, overwrite_x)
    调用 _pocketfft 库中的 fft 函数对输入 x 进行 FFT 变换，返回变换后的结果
# 返回实数或复数序列的离散逆傅里叶变换
def ifft(x, n=None, axis=-1, overwrite_x=False):
    # 返回的复数数组包含“y(0), y(1),..., y(n-1)”，其中
    # “y(j) = (x * exp(2*pi*sqrt(-1)*j*np.arange(n)/n)).mean()”.
    
    # x: 要反转的变换数据
    # n: 逆傅里叶变换的长度。如果“n < x.shape[axis]”，则截断x。如果“n > x.shape[axis]”，则对x进行零填充。默认结果为“n = x.shape[axis]”。
    # axis: 计算逆傅里叶变换的轴；默认为最后一个轴（即“axis=-1”）。
    # overwrite_x: 如果为True，则可以销毁x的内容；默认为False。
    
    # 返回逆离散傅里叶变换
    return _pocketfft.ifft(x, n, axis, None, overwrite_x)


# 返回实序列的离散傅里叶变换
def rfft(x, n=None, axis=-1, overwrite_x=False):
    # x: 要转换的数据，必须是实数
    # n: 定义傅里叶变换的长度。如果未指定n（默认情况下），则“n = x.shape[axis]”。如果“n < x.shape[axis]”，则截断x，如果“n > x.shape[axis]”，则对x进行零填充。
    # axis: 应用变换的轴。默认为最后一个轴。
    # overwrite_x: 如果设置为true，则可以覆盖x的内容。默认为False。
    
    # 返回实数数组包含：
    # [y(0),Re(y(1)),Im(y(1)),...,Re(y(n/2))]              如果n是偶数
    # [y(0),Re(y(1)),Im(y(1)),...,Re(y(n/2)),Im(y(n/2))]   如果n是奇数
    
    # 其中：
    # y(j) = sum[k=0..n-1] x[k] * exp(-sqrt(-1)*j*k*2*pi/n)
    # j = 0..n-1
    
    # 返回实数数组
    return z
    """
    使用 _pocketfft.rfft_fftpack 函数对输入信号进行快速傅里叶变换（FFT）。
    支持单精度和双精度的数据处理。半精度的输入会被转换为单精度处理，
    非浮点数输入会被转换为双精度处理。不支持长双精度输入。

    若要获得复数数据类型的输出，请考虑使用较新的函数 `scipy.fft.rfft`。

    示例
    --------
    从 scipy.fftpack 模块导入 fft, rfft 函数
    >>> from scipy.fftpack import fft, rfft
    定义输入数组 a
    >>> a = [9, -9, 1, 3]
    对数组 a 进行 FFT 变换
    >>> fft(a)
    返回数组：[  4. +0.j,   8.+12.j,  16. +0.j,   8.-12.j]
    对数组 a 进行实部FFT变换
    >>> rfft(a)
    array([  4.,   8.,  12.,  16.])

    """
    # 使用 _pocketfft.rfft_fftpack 函数进行快速傅里叶变换，返回结果
    return _pocketfft.rfft_fftpack(x, n, axis, None, overwrite_x)
# 返回实序列 x 的逆离散傅里叶变换结果
def irfft(x, n=None, axis=-1, overwrite_x=False):
    """
    Return inverse discrete Fourier transform of real sequence x.

    The contents of `x` are interpreted as the output of the `rfft`
    function.

    Parameters
    ----------
    x : array_like
        Transformed data to invert.
    n : int, optional
        Length of the inverse Fourier transform.
        If n < x.shape[axis], x is truncated.
        If n > x.shape[axis], x is zero-padded.
        The default results in n = x.shape[axis].
    axis : int, optional
        Axis along which the ifft's are computed; the default is over
        the last axis (i.e., axis=-1).
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.

    Returns
    -------
    irfft : ndarray of floats
        The inverse discrete Fourier transform.

    See Also
    --------
    rfft, ifft, scipy.fft.irfft

    Notes
    -----
    The returned real array contains::

        [y(0),y(1),...,y(n-1)]

    where for n is even::

        y(j) = 1/n (sum[k=1..n/2-1] (x[2*k-1]+sqrt(-1)*x[2*k])
                                     * exp(sqrt(-1)*j*k* 2*pi/n)
                    + c.c. + x[0] + (-1)**(j) x[n-1])

    and for n is odd::

        y(j) = 1/n (sum[k=1..(n-1)/2] (x[2*k-1]+sqrt(-1)*x[2*k])
                                     * exp(sqrt(-1)*j*k* 2*pi/n)
                    + c.c. + x[0])

    c.c. denotes complex conjugate of preceding expression.

    For details on input parameters, see `rfft`.

    To process (conjugate-symmetric) frequency-domain data with a complex
    datatype, consider using the newer function `scipy.fft.irfft`.

    Examples
    --------
    >>> from scipy.fftpack import rfft, irfft
    >>> a = [1.0, 2.0, 3.0, 4.0, 5.0]
    >>> irfft(a)
    array([ 2.6       , -3.16405192,  1.24398433, -1.14955713,  1.46962473])
    >>> irfft(rfft(a))
    array([1., 2., 3., 4., 5.])

    """
    # 调用底层函数 _pocketfft.irfft_fftpack 完成实际的逆离散傅里叶变换
    return _pocketfft.irfft_fftpack(x, n, axis, None, overwrite_x)


# 返回多维离散傅里叶变换的结果
def fftn(x, shape=None, axes=None, overwrite_x=False):
    """
    Return multidimensional discrete Fourier transform.

    The returned array contains::

      y[j_1,..,j_d] = sum[k_1=0..n_1-1, ..., k_d=0..n_d-1]
         x[k_1,..,k_d] * prod[i=1..d] exp(-sqrt(-1)*2*pi/n_i * j_i * k_i)

    where d = len(x.shape) and n = x.shape.

    Parameters
    ----------
    x : array_like
        The (N-D) array to transform.
    shape : int or array_like of ints or None, optional
        The shape of the result. If both `shape` and `axes` (see below) are
        None, `shape` is ``x.shape``; if `shape` is None but `axes` is
        not None, then `shape` is ``numpy.take(x.shape, axes, axis=0)``.
        If ``shape[i] > x.shape[i]``, the ith dimension is padded with zeros.
        If ``shape[i] < x.shape[i]``, the ith dimension is truncated to
        length ``shape[i]``.
        If any element of `shape` is -1, the size of the corresponding
        dimension of `x` is used.

    axes : int or tuple of ints, optional
        Axes over which to compute the DFT. If not given, the last `d`
        dimensions are used.

    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.

    Returns
    -------
    ftn : ndarray
        The N-D discrete Fourier transform.

    See Also
    --------
    ifftn, scipy.fft.fftn, rfftn

    Notes
    -----
    The multidimensional DFT is defined as follows::

        F[k_1,..,k_d] = sum[m_1=0..n_1-1, ..., m_d=0..n_d-1]
            x[m_1,..,m_d] * prod[i=1..d] exp(-sqrt(-1)*2*pi/n_i * k_i * m_i)

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.fftpack import fftn
    >>> a = np.zeros((3, 3))
    >>> fftn(a)
    array([[0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j]])

    """
    # 调用底层函数 _pocketfft.fftn_fftpack 完成实际的多维离散傅里叶变换
    return _pocketfft.fftn_fftpack(x, shape, axes, None, overwrite_x)
    axes : int or array_like of ints or None, optional
        # 定义参数 `axes`，可以是整数、整数数组或者 None，表示在 `x` 的哪些轴（如果 `shape` 不为 None 则是 `y` 的轴）上应用变换。
        The axes of `x` (`y` if `shape` is not None) along which the
        transform is applied.
        # 默认情况下是在所有轴上进行变换。

    overwrite_x : bool, optional
        # 是否允许改写 `x` 的内容。默认为 False。
        If True, the contents of `x` can be destroyed. Default is False.

    Returns
    -------
    y : complex-valued N-D NumPy array
        # 返回一个复数类型的 N 维 NumPy 数组，表示输入数组的（N 维）离散傅里叶变换（DFT）。

    See Also
    --------
    ifftn
        # 参见逆离散傅里叶变换 ifftn。

    Notes
    -----
    If ``x`` is real-valued, then
    ``y[..., j_i, ...] == y[..., n_i-j_i, ...].conjugate()``.
        # 如果 `x` 是实数值，则 `y[..., j_i, ...] == y[..., n_i-j_i, ...].conjugate()`。

    Both single and double precision routines are implemented. Half precision
    inputs will be converted to single precision. Non-floating-point inputs
    will be converted to double precision. Long-double precision inputs are
    not supported.
        # 实现了单精度和双精度例程。半精度输入将转换为单精度。非浮点数输入将转换为双精度。不支持长双精度输入。

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.fftpack import fftn, ifftn
    >>> y = (-np.arange(16), 8 - np.arange(16), np.arange(16))
    >>> np.allclose(y, fftn(ifftn(y)))
    True
        # 示例展示了如何使用 fftn 和 ifftn 进行离散傅里叶变换和逆变换，并验证它们的一致性。

    """
    shape = _good_shape(x, shape, axes)
    # 调用 _good_shape 函数，确定输入数组 `x` 的有效形状，根据给定的 `shape` 和 `axes` 参数。
    return _pocketfft.fftn(x, shape, axes, None, overwrite_x)
    # 调用底层 fftn 函数进行 N 维离散傅里叶变换，使用指定的形状 `shape`、轴 `axes`，并可选地允许改写 `x` 的内容。```markdown
    axes : int or array_like of ints or None, optional
        # Define the parameter `axes`, which can be an integer, an array of integers, or None, indicating the axes of `x` (or `y` if `shape` is not None) along which the transform is applied.
        The axes of `x` (`y` if `shape` is not None) along which the
        transform is applied.
        # By default, the transform is applied over all axes.

    overwrite_x : bool, optional
        # Indicates whether the contents of `x` can be overwritten. Defaults to False.
        If True, the contents of `x` can be destroyed. Default is False.

    Returns
    -------
    y : complex-valued N-D NumPy array
        # Returns a complex-valued N-D NumPy array representing the (N-D) Discrete Fourier Transform (DFT) of the input array.

    See Also
    --------
    ifftn
        # See the inverse Discrete Fourier Transform ifftn.

    Notes
    -----
    If ``x`` is real-valued, then
    ``y[..., j_i, ...] == y[..., n_i-j_i, ...].conjugate()``.
        # If `x` is real-valued, then ``y[..., j_i, ...] == y[..., n_i-j_i, ...].conjugate()``.

    Both single and double precision routines are implemented. Half precision
    inputs will be converted to single precision. Non-floating-point inputs
    will be converted to double precision. Long-double precision inputs are
    not supported.
        # Both single and double precision routines are implemented. Half-precision inputs will be converted to single precision. Non-floating-point inputs will be converted to double precision. Long-double precision inputs are not supported.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.fftpack import fftn, ifftn
    >>> y = (-np.arange(16), 8 - np.arange(16), np.arange(16))
    >>> np.allclose(y, fftn(ifftn(y)))
    True
        # Example demonstrating the use of fftn and ifftn for discrete Fourier transform and inverse transform, validating their consistency.

    """
    shape = _good_shape(x, shape, axes)
    # Call the _good_shape function to determine the valid shape of the input array `x`, based on the given `shape` and `axes` parameters.
    return _pocketfft.fftn(x, shape, axes, None, overwrite_x)
    # Call the underlying fftn function for N-dimensional Discrete Fourier Transform, using the specified shape `shape`, axes `axes`, and optionally allowing overwriting of `x`.
# 返回多维离散傅里叶逆变换结果
def ifftn(x, shape=None, axes=None, overwrite_x=False):
    # 确定合适的形状以进行傅里叶逆变换
    shape = _good_shape(x, shape, axes)
    # 调用底层库执行多维离散傅里叶逆变换
    return _pocketfft.ifftn(x, shape, axes, None, overwrite_x)


# 二维离散傅里叶变换
def fft2(x, shape=None, axes=(-2,-1), overwrite_x=False):
    # 调用多维离散傅里叶变换函数 fftn，实现二维离散傅里叶变换
    return fftn(x, shape, axes, overwrite_x)


# 二维离散傅里叶逆变换
def ifft2(x, shape=None, axes=(-2,-1), overwrite_x=False):
    # 调用多维离散傅里叶逆变换函数 ifftn，实现二维离散傅里叶逆变换
    return ifftn(x, shape, axes, overwrite_x)
```