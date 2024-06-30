# `D:\src\scipysrc\scipy\scipy\fft\_basic.py`

```
from scipy._lib.uarray import generate_multimethod, Dispatchable  # 导入 uarray 模块中的两个函数
import numpy as np  # 导入 NumPy 库


def _x_replacer(args, kwargs, dispatchables):
    """
    uarray argument replacer to replace the transform input array (``x``)
    """
    if len(args) > 0:
        return (dispatchables[0],) + args[1:], kwargs
    kw = kwargs.copy()
    kw['x'] = dispatchables[0]
    return args, kw


def _dispatch(func):
    """
    Function annotation that creates a uarray multimethod from the function
    """
    return generate_multimethod(func, _x_replacer, domain="numpy.scipy.fft")


@_dispatch
def fft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
        plan=None):
    """
    Compute the 1-D discrete Fourier Transform.

    This function computes the 1-D *n*-point discrete Fourier
    Transform (DFT) with the efficient Fast Fourier Transform (FFT)
    algorithm [1]_.

    Parameters
    ----------
    x : array_like
        Input array, can be complex.
    n : int, optional
        Length of the transformed axis of the output.
        If `n` is smaller than the length of the input, the input is cropped.
        If it is larger, the input is padded with zeros. If `n` is not given,
        the length of the input along the axis specified by `axis` is used.
    axis : int, optional
        Axis over which to compute the FFT. If not given, the last axis is
        used.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode. Default is "backward", meaning no normalization on
        the forward transforms and scaling by ``1/n`` on the `ifft`.
        "forward" instead applies the ``1/n`` factor on the forward transform.
        For ``norm="ortho"``, both directions are scaled by ``1/sqrt(n)``.

        .. versionadded:: 1.6.0
           ``norm={"forward", "backward"}`` options were added

    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See the notes below for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``. See below for more
        details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.

    Raises
    ------
    IndexError
        if `axes` is larger than the last axis of `x`.

    See Also
    --------
    ifft : The inverse of `fft`.
    fft2 : The 2-D FFT.
    fftn : The N-D FFT.
    rfftn : The N-D FFT of real input.
    fftfreq : Frequency bins for given FFT parameters.
    next_fast_len : Size to pad input to for most efficient transforms

    Notes
    -----
    This function is decorated with `_dispatch`, which enables it to be
    called with different argument types, supporting uarray's multimethod
    functionality.
    """
    # FFT（快速傅里叶变换）是一种离散傅里叶变换（DFT）的高效计算方法，利用了计算项中的对称性。当 `n` 是2的幂时，对称性最高，因此在这些大小下变换效率最高。对于难以分解的大小，`scipy.fft` 使用 Bluestein 算法 [2]_，因此时间复杂度永远不会超过 O(`n` log `n`)。通过使用 `next_fast_len` 对输入进行零填充，可能进一步提高性能。

    # 如果 ``x`` 是一维数组，则 `fft` 的计算等效于 ::
    # 
    #     y[k] = np.sum(x * np.exp(-2j * np.pi * k * np.arange(n)/n))
    # 
    # 频率项 ``f=k/n`` 位于 ``y[k]`` 处。在 ``y[n/2]`` 处达到奈奎斯特频率，并环绕到负频率项。因此，对于8点变换，结果的频率是 [0, 1, 2, 3, -4, -3, -2, -1]。要重新排列 fft 输出，使零频率分量位于中心位置，如 [-4, -3, -2, -1, 0, 1, 2, 3]，可以使用 `fftshift`。

    # 变换可以以单精度、双精度或扩展精度（长双精度）浮点数完成。如果 ``x`` 的数据类型是实数，则自动使用“实数 FFT”算法，大致可以减半计算时间。为了进一步提高效率，可以使用 `rfft`，它执行相同的计算，但仅输出对称谱的一半。如果数据既是实数又是对称的，则 `dct` 可以再次提高效率，通过从信号的一半生成一半的谱。

    # 当指定 ``overwrite_x=True`` 时，实现可能会以任何方式使用 ``x`` 引用的内存。这可能包括重用该内存作为结果，但这无法保证。在变换后，不应依赖于 ``x`` 的内容，因为这可能在未来发生变化而没有警告。

    # ``workers`` 参数指定最大的并行作业数，用于将 FFT 计算分成独立的1-D FFTs。因此， ``x`` 必须至少是2维的，并且未变换的轴必须足够大，以便分成块。如果 ``x`` 太小，则可能使用少于请求的作业数。

    # 参考文献
    # ----------
    # .. [1] Cooley, James W., 和 John W. Tukey, 1965, "An algorithm for the
    #        machine calculation of complex Fourier series," *Math. Comput.*
    #        19: 297-301.
    # .. [2] Bluestein, L., 1970, "A linear filtering approach to the
    #        computation of discrete Fourier transform". *IEEE Transactions on
    #        Audio and Electroacoustics.* 18 (4): 451-455.

    # 示例
    # --------
    # >>> import scipy.fft
    # >>> import numpy as np
    # >>> scipy.fft.fft(np.exp(2j * np.pi * np.arange(8) / 8))
    # 返回一个包含单个元组的元组，元组的第一个元素是 Dispatchable 对象，第二个元素是 np.ndarray 对象
        array([-2.33486982e-16+1.14423775e-17j,  8.00000000e+00-1.25557246e-15j,
                2.33486982e-16+2.33486982e-16j,  0.00000000e+00+1.22464680e-16j,
               -1.14423775e-17+2.33486982e-16j,  0.00000000e+00+5.20784380e-16j,
                1.14423775e-17+1.14423775e-17j,  0.00000000e+00+1.22464680e-16j])
    
        # 在这个例子中，实际输入的 FFT 是 Hermitian 的，即在实部中是对称的，在虚部中是反对称的：
        # 导入所需的库函数和模块
        >>> from scipy.fft import fft, fftfreq, fftshift
        >>> import matplotlib.pyplot as plt
        # 创建时间序列 t，包含 256 个元素
        >>> t = np.arange(256)
        # 对 sin(t) 序列进行 FFT 变换，并进行频率轴的移位
        >>> sp = fftshift(fft(np.sin(t)))
        # 获取移位后的频率轴
        >>> freq = fftshift(fftfreq(t.shape[-1]))
        # 绘制频率与实部、虚部的图像
        >>> plt.plot(freq, sp.real, freq, sp.imag)
        # 显示绘制的图像
        [<matplotlib.lines.Line2D object at 0x...>,
         <matplotlib.lines.Line2D object at 0x...>]
        >>> plt.show()
    
        """
        # 返回包含一个元组的元组，第一个元素是 Dispatchable 对象，第二个元素是 np.ndarray 对象
        return (Dispatchable(x, np.ndarray),)
# 定义 ifft 函数，用于计算一维逆离散傅立叶变换
@_dispatch
def ifft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
         plan=None):
    """
    Compute the 1-D inverse discrete Fourier Transform.

    This function computes the inverse of the 1-D *n*-point
    discrete Fourier transform computed by `fft`.  In other words,
    ``ifft(fft(x)) == x`` to within numerical accuracy.

    The input should be ordered in the same way as is returned by `fft`,
    i.e.,

    * ``x[0]`` should contain the zero frequency term,
    * ``x[1:n//2]`` should contain the positive-frequency terms,
    * ``x[n//2 + 1:]`` should contain the negative-frequency terms, in
      increasing order starting from the most negative frequency.

    For an even number of input points, ``x[n//2]`` represents the sum of
    the values at the positive and negative Nyquist frequencies, as the two
    are aliased together. See `fft` for details.

    Parameters
    ----------
    x : array_like
        Input array, can be complex.
    n : int, optional
        Length of the transformed axis of the output.
        If `n` is smaller than the length of the input, the input is cropped.
        If it is larger, the input is padded with zeros. If `n` is not given,
        the length of the input along the axis specified by `axis` is used.
        See notes about padding issues.
    axis : int, optional
        Axis over which to compute the inverse DFT. If not given, the last
        axis is used.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.

    Raises
    ------
    IndexError
        If `axes` is larger than the last axis of `x`.

    See Also
    --------
    fft : The 1-D (forward) FFT, of which `ifft` is the inverse.
    ifft2 : The 2-D inverse FFT.
    ifftn : The N-D inverse FFT.

    Notes
    -----
    If the input parameter `n` is larger than the size of the input, the input
    is padded by appending zeros at the end. Even though this is the common
    approach, it might lead to surprising results. If a different padding is
    desired, it must be performed before calling `ifft`.
    """
    # 如果 ``x`` 是一个一维数组，那么 `ifft` 等价于以下操作：
    #
    #     y[k] = np.sum(x * np.exp(2j * np.pi * k * np.arange(n)/n)) / len(x)
    #
    # 与 `fft` 类似，`ifft` 支持所有浮点数类型，并且对于实数输入进行了优化。
    #
    # 示例：
    # >>> import scipy.fft
    # >>> import numpy as np
    # >>> scipy.fft.ifft([0, 4, 0, 0])
    # array([ 1.+0.j,  0.+1.j, -1.+0.j,  0.-1.j]) # 结果可能有所不同
    #
    # 创建并绘制一个带限信号，带有随机相位：
    #
    # >>> import matplotlib.pyplot as plt
    # >>> rng = np.random.default_rng()
    # >>> t = np.arange(400)
    # >>> n = np.zeros((400,), dtype=complex)
    # >>> n[40:60] = np.exp(1j*rng.uniform(0, 2*np.pi, (20,)))
    # >>> s = scipy.fft.ifft(n)
    # >>> plt.plot(t, s.real, 'b-', t, s.imag, 'r--')
    # [<matplotlib.lines.Line2D object at ...>, <matplotlib.lines.Line2D object at ...>]
    # >>> plt.legend(('real', 'imaginary'))
    # <matplotlib.legend.Legend object at ...>
    # >>> plt.show()
    #
    # """
    # 返回一个元组，包含 x 和 np.ndarray 的 Dispatchable 对象。
    return (Dispatchable(x, np.ndarray),)
@_dispatch
def rfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
         plan=None):
    """
    Compute the 1-D discrete Fourier Transform for real input.

    This function computes the 1-D *n*-point discrete Fourier
    Transform (DFT) of a real-valued array by means of an efficient algorithm
    called the Fast Fourier Transform (FFT).

    Parameters
    ----------
    x : array_like
        Input array
    n : int, optional
        Number of points along transformation axis in the input to use.
        If `n` is smaller than the length of the input, the input is cropped.
        If it is larger, the input is padded with zeros. If `n` is not given,
        the length of the input along the axis specified by `axis` is used.
    axis : int, optional
        Axis over which to compute the FFT. If not given, the last axis is
        used.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.
        If `n` is even, the length of the transformed axis is ``(n/2)+1``.
        If `n` is odd, the length is ``(n+1)/2``.

    Raises
    ------
    IndexError
        If `axis` is larger than the last axis of `a`.

    See Also
    --------
    irfft : The inverse of `rfft`.
    fft : The 1-D FFT of general (complex) input.
    fftn : The N-D FFT.
    rfft2 : The 2-D FFT of real input.
    rfftn : The N-D FFT of real input.

    Notes
    -----
    When the DFT is computed for purely real input, the output is
    Hermitian-symmetric, i.e., the negative frequency terms are just the complex
    conjugates of the corresponding positive-frequency terms, and the
    negative-frequency terms are therefore redundant. This function does not
    compute the negative frequency terms, and the length of the transformed
    axis of the output is therefore ``n//2 + 1``.

    When ``X = rfft(x)`` and fs is the sampling frequency, ``X[0]`` contains
    the zero-frequency term 0*fs, which is real due to Hermitian symmetry.

    If `n` is even, ``A[-1]`` contains the term representing both positive
    and negative Nyquist frequency (+fs/2 and -fs/2), and must also be purely
    real.
    """
    # 实现对实数输入的一维离散傅里叶变换（DFT）
    # 使用快速傅里叶变换（FFT）算法实现高效计算
    # 获取输入数组 `x` 的长度
    Nx = len(x)
    
    # 如果未指定 `n`，则使用输入数组 `x` 沿指定轴 `axis` 的长度
    if n is None:
        n = Nx
        
    # 根据 `n` 和 `overwrite_x` 参数调整输入数组 `x`
    if overwrite_x:
        # 如果 `overwrite_x` 为 True，则直接修改 `x` 的内容
        ret = _fft.fft(x, n=n, axis=axis, overwrite_x=True, plan=plan)
    else:
        # 如果 `overwrite_x` 为 False，则复制 `x` 后再进行变换
        ret = _fft.fft(x, n=n, axis=axis, overwrite_x=False, plan=plan)
    
    # 根据变换后的结果 `ret` 和 `norm` 进行归一化处理
    if norm is None or norm == "backward":
        return _normalization(ret, n, axis, 'backward')
    elif norm == "ortho":
        return _normalization(ret, n, axis, 'ortho')
    elif norm == "forward":
        return ret
    else:
        raise ValueError(f"Invalid norm value {norm}")
    # 返回一个包含 Dispatchable 对象和 numpy 数组的元组
    return (Dispatchable(x, np.ndarray),)
# 定义一个装饰器函数，用于分派调用
@_dispatch
# 计算反向快速傅立叶变换（IFFT）的函数，与 `rfft` 相对应
def irfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
          plan=None):
    """
    Computes the inverse of `rfft`.

    This function computes the inverse of the 1-D *n*-point
    discrete Fourier Transform of real input computed by `rfft`.
    In other words, ``irfft(rfft(x), len(x)) == x`` to within numerical
    accuracy. (See Notes below for why ``len(a)`` is necessary here.)

    The input is expected to be in the form returned by `rfft`, i.e., the
    real zero-frequency term followed by the complex positive frequency terms
    in order of increasing frequency. Since the discrete Fourier Transform of
    real input is Hermitian-symmetric, the negative frequency terms are taken
    to be the complex conjugates of the corresponding positive frequency terms.

    Parameters
    ----------
    x : array_like
        The input array.
    n : int, optional
        Length of the transformed axis of the output.
        For `n` output points, ``n//2+1`` input points are necessary. If the
        input is longer than this, it is cropped. If it is shorter than this,
        it is padded with zeros. If `n` is not given, it is taken to be
        ``2*(m-1)``, where ``m`` is the length of the input along the axis
        specified by `axis`.
    axis : int, optional
        Axis over which to compute the inverse FFT. If not given, the last
        axis is used.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.
        The length of the transformed axis is `n`, or, if `n` is not given,
        ``2*(m-1)`` where ``m`` is the length of the transformed axis of the
        input. To get an odd number of output points, `n` must be specified.

    Raises
    ------
    IndexError
        If `axis` is larger than the last axis of `x`.

    See Also
    --------
    rfft : The 1-D FFT of real input, of which `irfft` is inverse.
    fft : The 1-D FFT.
    irfft2 : The inverse of the 2-D FFT of real input.
    irfftn : The inverse of the N-D FFT of real input.

    Notes
    -----
    Returns the real valued `n`-point inverse discrete Fourier transform
    """
    返回一个包含一个元组的函数结果，元组的第一个元素是 Dispatchable 对象，表示 x 的数据类型是 np.ndarray。
    ```
# 装饰器函数，用于分派不同的函数调用
@_dispatch
# 计算具有厄米特对称性（即实频谱）信号的 FFT
def hfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
         plan=None):
    """
    Compute the FFT of a signal that has Hermitian symmetry, i.e., a real
    spectrum.
    
    Parameters
    ----------
    x : array_like
        输入数组。
    n : int, optional
        输出变换轴的长度。对于 `n` 个输出点，需要 `n//2 + 1` 个输入点。
        如果输入比这更长，会被裁剪；如果比这更短，会用零填充。如果未给出 `n`，
        则取为 ``2*(m-1)``，其中 `m` 是沿着由 `axis` 指定的输入的长度。
    axis : int, optional
        进行 FFT 的轴。如果未给出，则使用最后一个轴。
    norm : {"backward", "ortho", "forward"}, optional
        归一化模式（参见 `fft`）。默认为 "backward"。
    overwrite_x : bool, optional
        如果为 True，则可以销毁 `x` 的内容；默认为 False。
        更多细节见 `fft`。
    workers : int, optional
        用于并行计算的最大工作进程数。如果为负数，则从 ``os.cpu_count()`` 循环获取。
        更多细节见 :func:`~scipy.fft.fft`。
    plan : object, optional
        此参数保留以传递由下游 FFT 供应商提供的预计算计划。目前在 SciPy 中未使用。

        .. versionadded:: 1.5.0

    Returns
    -------
    out : ndarray
        截断或零填充的输入，沿着由 `axis` 指示的轴进行变换，或者如果未指定 `axis`，
        则沿着最后一个轴。变换轴的长度为 `n`，如果未给出 `n`，则为 ``2*m - 2``，
        其中 `m` 是输入变换轴的长度。要获取奇数个输出点，必须指定 `n`，例如，
        在典型情况下，设为 ``2*m - 1``，

    Raises
    ------
    IndexError
        如果 `axis` 大于 `x` 的最后一个轴。

    See Also
    --------
    rfft : 计算实输入的一维 FFT。
    ihfft : `hfft` 的逆操作。
    hfftn : 计算具有厄米特信号的 N 维 FFT。

    Notes
    -----
    `hfft`/`ihfft` 是一对类似于 `rfft`/`irfft` 的函数，但适用于相反的情况：
    这里信号在时间域具有厄米特对称性，在频率域中是实数。因此，在这里，使用 `hfft`，
    如果要得到奇数个结果，则必须提供结果的长度。
    * 偶数情况：``ihfft(hfft(a, 2*len(a) - 2) == a``，在舍入误差内成立，
    * 奇数情况：``ihfft(hfft(a, 2*len(a) - 1) == a``，在舍入误差内成立。

    Examples
    --------
    >>> from scipy.fft import fft, hfft
    >>> import numpy as np
    >>> a = 2 * np.pi * np.arange(10) / 10
    >>> signal = np.cos(a) + 3j * np.sin(3 * a)
    >>> fft(signal).round(10)
    # 创建一个复数数组，表示实部和虚部均为零的复数
    array([ -0.+0.j,   5.+0.j,  -0.+0.j,  15.-0.j,   0.+0.j,   0.+0.j,
            -0.+0.j, -15.-0.j,   0.+0.j,   5.+0.j])
    # 对信号的前半部分进行 Hermite 傅里叶变换，并保留小数点后十位
    >>> hfft(signal[:6]).round(10) # Input first half of signal
    # 返回处理后的复数数组，表示信号前半部分的 Hermite 傅里叶变换结果
    array([  0.,   5.,   0.,  15.,  -0.,   0.,   0., -15.,  -0.,   5.])
    # 对整个信号进行 Hermite 傅里叶变换，并截断到长度 10
    >>> hfft(signal, 10)  # Input entire signal and truncate
    # 返回处理后的复数数组，表示整个信号的 Hermite 傅里叶变换结果，并截断到长度 10
    array([  0.,   5.,   0.,  15.,  -0.,   0.,   0., -15.,  -0.,   5.])
    """
    # 返回一个元组，包含 Dispatchable 对象和 np.ndarray 对象
    return (Dispatchable(x, np.ndarray),)
# 使用装饰器 @_dispatch 标记该函数为分派函数，用于支持多态分发
def ihfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
          plan=None):
    """
    Compute the inverse FFT of a signal that has Hermitian symmetry.

    Parameters
    ----------
    x : array_like
        输入数组。
    n : int, optional
        逆FFT的长度，在输入变换轴上要使用的点数。如果 `n` 小于输入的长度，
        则输入被裁剪。如果 `n` 大于输入长度，则用零填充。如果未给出 `n`，
        则使用由 `axis` 指定的轴上的输入长度。
    axis : int, optional
        计算逆FFT的轴。如果未给出，则使用最后一个轴。
    norm : {"backward", "ortho", "forward"}, optional
        归一化模式（参见 `fft`）。默认为 "backward"。
    overwrite_x : bool, optional
        如果为 True，则可以销毁 `x` 的内容；默认为 False。详见 `fft`。
    workers : int, optional
        用于并行计算的最大工作线程数。如果为负数，则从 ``os.cpu_count()`` 循环。
        更多细节请参见 :func:`~scipy.fft.fft`。
    plan : object, optional
        此参数保留用于传递下游FFT供应商提供的预先计算的计划。目前在SciPy中未使用。

        .. versionadded:: 1.5.0

    Returns
    -------
    out : complex ndarray
        被截断或零填充的输入，沿着由 `axis` 指示的轴变换，或者如果未指定 `axis`，则是最后一个轴。
        变换轴的长度为 ``n//2 + 1``。

    See Also
    --------
    hfft, irfft

    Notes
    -----
    `hfft`/`ihfft` 是 `rfft`/`irfft` 的对应对，但逆向情况：
    这里，信号在时间域具有共轭对称性，在频率域中是实数。因此，这里使用 `hfft`，
    如果结果长度是奇数，则必须提供其长度：
    * 偶数: ``ihfft(hfft(a, 2*len(a) - 2) == a``，在舍入误差内，
    * 奇数: ``ihfft(hfft(a, 2*len(a) - 1) == a``，在舍入误差内。

    Examples
    --------
    >>> from scipy.fft import ifft, ihfft
    >>> import numpy as np
    >>> spectrum = np.array([ 15, -4, 0, -1, 0, -4])
    >>> ifft(spectrum)
    array([1.+0.j,  2.+0.j,  3.+0.j,  4.+0.j,  3.+0.j,  2.+0.j]) # 可能有所不同
    >>> ihfft(spectrum)
    array([ 1.-0.j,  2.-0.j,  3.-0.j,  4.-0.j]) # 可能有所不同
    """
    # 返回一个元组，包含被 Dispatchable 封装的输入数组 `x` 和 `np.ndarray`
    return (Dispatchable(x, np.ndarray),)


# 使用装饰器 @_dispatch 标记该函数为分派函数，用于支持多态分发
def fftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *,
         plan=None):
    """
    Compute the N-D discrete Fourier Transform.

    This function computes the N-D discrete Fourier Transform over
    any number of axes in an M-D array by means of the Fast Fourier
    Transform (FFT).

    Parameters
    ----------
    x : array_like
        输入数组，可以是复数。
    s : sequence of ints, optional
        用于计算FFT的输入形状。对于长度为 `M` 的轴，FFT的长度将为 `s[k]`，
        其中 `k` 是轴的索引。如果未提供，则使用 `x` 的形状。
    axes : sequence of ints, optional
        沿其计算FFT的轴。如果未提供，则对输入的最后一个轴进行FFT。
    norm : {"backward", "ortho", "forward"}, optional
        归一化模式（参见 `fft`）。默认为 "backward"。
    overwrite_x : bool, optional
        如果为 True，则可以销毁 `x` 的内容；默认为 False。详见 `fft`。
    workers : int, optional
        用于并行计算的最大工作线程数。如果为负数，则从 ``os.cpu_count()`` 循环。
        更多细节请参见 :func:`~scipy.fft.fft`。
    plan : object, optional
        此参数保留用于传递下游FFT供应商提供的预先计算的计划。目前在SciPy中未使用。

    Returns
    -------
    out : complex ndarray
        与输入数组 `x` 具有相同形状的数组，其中包含通过FFT变换得到的结果。

    Notes
    -----
    FFT是一种快速傅里叶变换方法，用于计算信号的频率域表示。

    Examples
    --------
    >>> from scipy.fft import fftn
    >>> import numpy as np
    >>> x = np.array([[1, 2], [3, 4]])
    >>> fftn(x)
    array([[10.+0.j, -2.+0.j],
           [-2.+0.j,  0.+0.j]]) # 可能有所不同
    """
    # 返回一个元组，包含被 Dispatchable 封装的输入数组 `x` 和 `np.ndarray`
    return (Dispatchable(x, np.ndarray),)
    # s: 整数序列，可选参数
    # 输出的每个转换轴的形状（``s[0]``指示轴0，``s[1]``指示轴1等）。
    # 这对应于 ``fft(x, n)`` 中的 `n`。
    # 沿任何轴，如果给定的形状小于输入的形状，则输入被裁剪。
    # 如果形状大于输入，则用零填充输入。
    # 如果未给出 `s`，则使用由 `axes` 指定的轴的输入形状。
    axes: 整数序列，可选参数
        # 用于计算FFT的轴。如果未给出，则使用最后的 `len(s)` 轴，或者如果 `s` 也未指定，则使用所有轴。
    norm: {"backward", "ortho", "forward"}，可选参数
        # 归一化模式（参见 `fft`）。默认为 "backward"。
    overwrite_x: 布尔值，可选参数
        # 如果为True，则可以销毁 `x` 的内容；默认为False。
        # 查看 :func:`fft` 获取更多详情。
    workers: 整数，可选参数
        # 用于并行计算的最大工作线程数。如果为负数，则值从 ``os.cpu_count()`` 循环回来。
        # 查看 :func:`~scipy.fft.fft` 获取更多详情。
    plan: 对象，可选参数
        # 此参数保留用于通过下游FFT供应商提供的预先计算的计划。
        # 当前在 SciPy 中未使用。

        # .. versionadded:: 1.5.0
    # 对输入数组 x 进行二维傅里叶变换，结果为一个2x2的数组，指定在0和1轴进行变换
    >>> scipy.fft.fftn(x, (2, 2), axes=(0, 1))
    array([[[ 2.+0.j,  2.+0.j,  2.+0.j],  # 可能会有变化
            [ 0.+0.j,  0.+0.j,  0.+0.j]],
           [[-2.+0.j, -2.+0.j, -2.+0.j],
            [ 0.+0.j,  0.+0.j,  0.+0.j]]])
    
    # 导入 matplotlib.pyplot 库，起别名为 plt
    >>> import matplotlib.pyplot as plt
    
    # 使用 np.random.default_rng() 创建随机数生成器 rng
    >>> rng = np.random.default_rng()
    
    # 使用 np.meshgrid 创建 X 和 Y，分别是200个元素的数组，用于后续计算
    >>> [X, Y] = np.meshgrid(2 * np.pi * np.arange(200) / 12,
    ...                      2 * np.pi * np.arange(200) / 34)
    
    # 根据 X 和 Y 的网格创建 S，其中包括 sin(X)、cos(Y) 和从均匀分布中取值的随机数
    >>> S = np.sin(X) + np.cos(Y) + rng.uniform(0, 1, X.shape)
    
    # 对 S 进行傅里叶变换得到 FS
    >>> FS = scipy.fft.fftn(S)
    
    # 显示经过傅里叶变换后 FS 的幅度谱的对数值，通过 plt.imshow 绘制
    >>> plt.imshow(np.log(np.abs(scipy.fft.fftshift(FS))**2))
    <matplotlib.image.AxesImage object at 0x...>  # 显示的是 AxesImage 对象的地址
    
    # 显示绘制的图像
    >>> plt.show()
    
    """
    返回一个元组，包含 x 和 np.ndarray 的 Dispatchable 对象
    """
    return (Dispatchable(x, np.ndarray),)
# 定义装饰器，用于分派函数调用
@_dispatch
# 定义 ifftn 函数，用于计算 N 维逆离散傅里叶变换（IDFT）
def ifftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *,
          plan=None):
    """
    Compute the N-D inverse discrete Fourier Transform.

    This function computes the inverse of the N-D discrete
    Fourier Transform over any number of axes in an M-D array by
    means of the Fast Fourier Transform (FFT).  In other words,
    ``ifftn(fftn(x)) == x`` to within numerical accuracy.

    The input, analogously to `ifft`, should be ordered in the same way as is
    returned by `fftn`, i.e., it should have the term for zero frequency
    in all axes in the low-order corner, the positive frequency terms in the
    first half of all axes, the term for the Nyquist frequency in the middle
    of all axes and the negative frequency terms in the second half of all
    axes, in order of decreasingly negative frequency.

    Parameters
    ----------
    x : array_like
        Input array, can be complex.
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        This corresponds to ``n`` for ``ifft(x, n)``.
        Along any axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.
        if `s` is not given, the shape of the input along the axes specified
        by `axes` is used. See notes for issue on `ifft` zero padding.
    axes : sequence of ints, optional
        Axes over which to compute the IFFT.  If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or by a combination of `s` or `x`,
        as explained in the parameters section above.

    Raises
    ------
    ValueError
        If `s` and `axes` have different length.
    IndexError
        If an element of `axes` is larger than the number of axes of `x`.

    See Also
    --------
    fftn : The forward N-D FFT, of which `ifftn` is the inverse.
    ifft : The 1-D inverse FFT.
    ifft2 : The 2-D inverse FFT.
    """
    # 返回一个元组，包含一个 Dispatchable 对象和一个 np.ndarray 对象，用于表示逆傅里叶变换的结果
    return (Dispatchable(x, np.ndarray),)
# 定义装饰器函数，用于处理 FFT 相关函数的分派
@_dispatch
# 定义二维快速傅里叶变换函数 fft2
def fft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None, *,
         plan=None):
    """
    Compute the 2-D discrete Fourier Transform
    
    This function computes the N-D discrete Fourier Transform
    over any axes in an M-D array by means of the
    Fast Fourier Transform (FFT). By default, the transform is computed over
    the last two axes of the input array, i.e., a 2-dimensional FFT.
    
    Parameters
    ----------
    x : array_like
        Input array, can be complex
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        This corresponds to ``n`` for ``fft(x, n)``.
        Along each axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.
        if `s` is not given, the shape of the input along the axes specified
        by `axes` is used.
    axes : sequence of ints, optional
        Axes over which to compute the FFT. If not given, the last two axes are
        used.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.
        
        .. versionadded:: 1.5.0
        
    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or the last two axes if `axes` is not given.
    
    Raises
    ------
    ValueError
        If `s` and `axes` have different length, or `axes` not given and
        ``len(s) != 2``.
    IndexError
        If an element of `axes` is larger than the number of axes of `x`.
    
    See Also
    --------
    ifft2 : The inverse 2-D FFT.
    fft : The 1-D FFT.
    fftn : The N-D FFT.
    fftshift : Shifts zero-frequency terms to the center of the array.
        For 2-D input, swaps first and third quadrants, and second
        and fourth quadrants.
    
    Notes
    -----
    `fft2` is just `fftn` with a different default for `axes`.
    
    The output, analogously to `fft`, contains the term for zero frequency in
    the low-order corner of the transformed axes, the positive frequency terms
    in the first half of these axes, the term for the Nyquist frequency in the
    middle of the axes and the negative frequency terms in the second half of
    the axes, in order of decreasingly negative frequency.
    """
    """
        See `fftn` for details and a plotting example, and `fft` for
        definitions and conventions used.
    
        Examples
        --------
        >>> import scipy.fft
        >>> import numpy as np
        >>> x = np.mgrid[:5, :5][0]
        >>> scipy.fft.fft2(x)
        array([[ 50.  +0.j        ,   0.  +0.j        ,   0.  +0.j        , # may vary
                  0.  +0.j        ,   0.  +0.j        ],
               [-12.5+17.20477401j,   0.  +0.j        ,   0.  +0.j        ,
                  0.  +0.j        ,   0.  +0.j        ],
               [-12.5 +4.0614962j ,   0.  +0.j        ,   0.  +0.j        ,
                  0.  +0.j        ,   0.  +0.j        ],
               [-12.5 -4.0614962j ,   0.  +0.j        ,   0.  +0.j        ,
                  0.  +0.j        ,   0.  +0.j        ],
               [-12.5-17.20477401j,   0.  +0.j        ,   0.  +0.j        ,
                  0.  +0.j        ,   0.  +0.j        ]])
    
        """
        # 返回一个包含 x 和 np.ndarray 的 Dispatchable 对象的元组
        return (Dispatchable(x, np.ndarray),)
# 定义一个装饰器，用于分派不同的函数调用
@_dispatch
# 定义 ifft2 函数，用于计算二维逆离散傅立叶变换
def ifft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None, *,
          plan=None):
    """
    Compute the 2-D inverse discrete Fourier Transform.

    This function computes the inverse of the 2-D discrete Fourier
    Transform over any number of axes in an M-D array by means of
    the Fast Fourier Transform (FFT). In other words, ``ifft2(fft2(x)) == x``
    to within numerical accuracy. By default, the inverse transform is
    computed over the last two axes of the input array.

    The input, analogously to `ifft`, should be ordered in the same way as is
    returned by `fft2`, i.e., it should have the term for zero frequency
    in the low-order corner of the two axes, the positive frequency terms in
    the first half of these axes, the term for the Nyquist frequency in the
    middle of the axes and the negative frequency terms in the second half of
    both axes, in order of decreasingly negative frequency.

    Parameters
    ----------
    x : array_like
        Input array, can be complex.
    s : sequence of ints, optional
        Shape (length of each axis) of the output (``s[0]`` refers to axis 0,
        ``s[1]`` to axis 1, etc.). This corresponds to `n` for ``ifft(x, n)``.
        Along each axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.
        if `s` is not given, the shape of the input along the axes specified
        by `axes` is used.  See notes for issue on `ifft` zero padding.
    axes : sequence of ints, optional
        Axes over which to compute the FFT. If not given, the last two
        axes are used.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or the last two axes if `axes` is not given.

    Raises
    ------
    ValueError
        If `s` and `axes` have different length, or `axes` not given and
        ``len(s) != 2``.
    IndexError
        If an element of `axes` is larger than the number of axes of `x`.

    See Also
    --------
    fft2 : The forward 2-D FFT, of which `ifft2` is the inverse.
    ifftn : The inverse of the N-D FFT.
    fft : The 1-D FFT.
    ifft : The 1-D inverse FFT.

    Notes
    -----
    This function computes the inverse 2-D FFT of an input array `x` over
    specified axes. It provides options for shape manipulation (`s`), FFT
    normalization (`norm`), and multiprocessing (`workers`). The `plan`
    parameter is reserved for future use and currently not implemented.
    """
    # `ifft2` 是 `ifftn` 的一个变体，默认对轴的处理方式不同。
    #
    # 详见 `ifftn` 的说明和绘图示例，以及 `fft` 的定义和约定。
    #
    # 类似于 `ifft`，通过在指定维度上向输入追加零来执行零填充。
    # 尽管这是常见的方法，但可能导致意外的结果。
    # 如果需要使用另一种形式的零填充，则必须在调用 `ifft2` 之前执行。
    #
    # 示例
    # ----
    # >>> import scipy.fft
    # >>> import numpy as np
    # >>> x = 4 * np.eye(4)
    # >>> scipy.fft.ifft2(x)
    # array([[1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],  # 结果可能有所不同
    #        [0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j],
    #        [0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
    #        [0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j]])
    """
    返回一个元组，包含 `x` 和 `np.ndarray` 的 `Dispatchable` 对象。
    """
    return (Dispatchable(x, np.ndarray),)
# 定义一个装饰器，用于分派不同输入类型到相应函数的装饰器函数
@_dispatch
# 定义实现 N 维实数输入的快速傅里叶变换（FFT）的函数
def rfftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *,
          plan=None):
    """
    Compute the N-D discrete Fourier Transform for real input.

    This function computes the N-D discrete Fourier Transform over
    any number of axes in an M-D real array by means of the Fast
    Fourier Transform (FFT). By default, all axes are transformed, with the
    real transform performed over the last axis, while the remaining
    transforms are complex.

    Parameters
    ----------
    x : array_like
        Input array, taken to be real.
    s : sequence of ints, optional
        Shape (length along each transformed axis) to use from the input.
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        The final element of `s` corresponds to `n` for ``rfft(x, n)``, while
        for the remaining axes, it corresponds to `n` for ``fft(x, n)``.
        Along any axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.
        if `s` is not given, the shape of the input along the axes specified
        by `axes` is used.
    axes : sequence of ints, optional
        Axes over which to compute the FFT. If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or by a combination of `s` and `x`,
        as explained in the parameters section above.
        The length of the last axis transformed will be ``s[-1]//2+1``,
        while the remaining transformed axes will have lengths according to
        `s`, or unchanged from the input.

    Raises
    ------
    ValueError
        If `s` and `axes` have different length.
    IndexError
        If an element of `axes` is larger than the number of axes of `x`.

    See Also
    --------
    irfftn : The inverse of `rfftn`, i.e., the inverse of the N-D FFT
         of real input.
    fft : The 1-D FFT, with definitions and conventions used.
    rfft : The 1-D FFT of real input.
    fftn : The N-D FFT.
    rfft2 : The 2-D FFT of real input.

    Notes
    -----
    """
    The transform for real input is performed over the last transformation
    axis, as by `rfft`, then the transform over the remaining axes is
    performed as by `fftn`. The order of the output is as for `rfft` for the
    final transformation axis, and as for `fftn` for the remaining
    transformation axes.

    See `fft` for details, definitions and conventions used.

    Examples
    --------
    >>> import scipy.fft
    >>> import numpy as np
    >>> x = np.ones((2, 2, 2))
    >>> scipy.fft.rfftn(x)
    array([[[8.+0.j,  0.+0.j], # may vary
            [0.+0.j,  0.+0.j]],
           [[0.+0.j,  0.+0.j],
            [0.+0.j,  0.+0.j]]])

    >>> scipy.fft.rfftn(x, axes=(2, 0))
    array([[[4.+0.j,  0.+0.j], # may vary
            [4.+0.j,  0.+0.j]],
           [[0.+0.j,  0.+0.j],
            [0.+0.j,  0.+0.j]]])

    """
    # 返回一个包含 Dispatchable 对象和 np.ndarray 的元组
    return (Dispatchable(x, np.ndarray),)
# 定义 `_dispatch` 装饰器，用于函数的多态分发
@_dispatch
# 定义函数 `rfft2`，用于计算实数组的二维快速傅里叶变换（FFT）
def rfft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None, *,
          plan=None):
    """
    Compute the 2-D FFT of a real array.

    Parameters
    ----------
    x : array
        Input array, taken to be real.
    s : sequence of ints, optional
        Shape of the FFT.
    axes : sequence of ints, optional
        Axes over which to compute the FFT.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : ndarray
        The result of the real 2-D FFT.

    See Also
    --------
    irfft2 : The inverse of the 2-D FFT of real input.
    rfft : The 1-D FFT of real input.
    rfftn : Compute the N-D discrete Fourier Transform for real
            input.

    Notes
    -----
    This is really just `rfftn` with different default behavior.
    For more details see `rfftn`.
    """
    # 返回一个元组，其中包含调度对象 `Dispatchable` 和 `np.ndarray`
    return (Dispatchable(x, np.ndarray),)


# 定义 `_dispatch` 装饰器，用于函数的多态分发
@_dispatch
# 定义函数 `irfftn`，用于计算 `rfftn` 的逆变换
def irfftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *,
           plan=None):
    """
    Computes the inverse of `rfftn`

    This function computes the inverse of the N-D discrete
    Fourier Transform for real input over any number of axes in an
    M-D array by means of the Fast Fourier Transform (FFT). In
    other words, ``irfftn(rfftn(x), x.shape) == x`` to within numerical
    accuracy. (The ``a.shape`` is necessary like ``len(a)`` is for `irfft`,
    and for the same reason.)

    The input should be ordered in the same way as is returned by `rfftn`,
    i.e., as for `irfft` for the final transformation axis, and as for `ifftn`
    along all the other axes.

    Parameters
    ----------
    x : array_like
        Input array.
    """
    s : sequence of ints, optional
        # 输出数组的形状（每个变换轴的长度）
        Shape (length of each transformed axis) of the output
        # s[0] 表示轴 0 的长度，s[1] 表示轴 1 的长度，依此类推。
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        # 对于最后一个轴，使用 ``s[-1]//2+1`` 个输入点。
        `s` is also the number of input points used along this axis, except for the last axis,
        where ``s[-1]//2+1`` points of the input are used.
        # 如果指定的 `s` 小于输入的形状，则进行裁剪。如果大于，则用零填充。
        Along any axis, if the shape indicated by `s` is smaller than that of
        the input, the input is cropped. If it is larger, the input is padded
        with zeros.
        # 如果 `s` 未给出，则使用指定的轴的输入形状。最后一个轴的默认形状是 ``2*(m-1)``，其中 ``m`` 是该轴上输入的长度。
        If `s` is not given, the shape of the input along the axes specified by axes is used. Except for the last axis which is taken to be
        ``2*(m-1)``, where ``m`` is the length of the input along that axis.

    axes : sequence of ints, optional
        # 用于计算逆FFT的轴。如果未给出，则使用最后的 `len(s)` 个轴，或者如果 `s` 也未指定，则使用所有轴。
        Axes over which to compute the inverse FFT. If not given, the last
        `len(s)` axes are used, or all axes if `s` is also not specified.

    norm : {"backward", "ortho", "forward"}, optional
        # 规范化模式（参见 `fft`）。默认为 "backward"。
        Normalization mode (see `fft`). Default is "backward".

    overwrite_x : bool, optional
        # 如果为 True，则可以销毁 `x` 的内容；默认为 False。
        If True, the contents of `x` can be destroyed; the default is False.
        # 更多详情请参阅 :func:`fft`。
        See :func:`fft` for more details.

    workers : int, optional
        # 用于并行计算的最大工作线程数。如果为负数，则从 ``os.cpu_count()`` 循环。
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        # 更多详情请参阅 :func:`~scipy.fft.fft`。
        See :func:`~scipy.fft.fft` for more details.

    plan : object, optional
        # 保留用于传递预先计算的计划的参数，由下游FFT供应商提供。
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors.
        # 目前在 SciPy 中未使用。
        It is currently not used in SciPy.
        # 自版本 1.5.0 添加。
        .. versionadded:: 1.5.0

    Returns
    -------
    out : ndarray
        # 变换后的截断或零填充输入，沿 `axes` 指示的轴变换，或者根据 `s` 或 `x` 的组合进行变换，详见上面的参数部分。
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or by a combination of `s` or `x`,
        as explained in the parameters section above.
        # 每个变换轴的长度由相应的 `s` 元素给出，或者如果未给出 `s`，则在最后一个变换轴上的输出长度为 ``2*(m-1)``，其中 ``m`` 是输入最后一个变换轴上的长度。
        The length of each transformed axis is as given by the corresponding
        element of `s`, or the length of the input in every axis except for the
        last one if `s` is not given. In the final transformed axis the length
        of the output when `s` is not given is ``2*(m-1)``, where ``m`` is the
        length of the final transformed axis of the input.
        # 若要在最后一个轴上获取奇数个输出点，必须指定 `s`。

    Raises
    ------
    ValueError
        # 如果 `s` 和 `axes` 的长度不同。
        If `s` and `axes` have different length.
    IndexError
        # 如果 `axes` 中的元素大于 `x` 的轴数。
        If an element of `axes` is larger than the number of axes of `x`.

    See Also
    --------
    rfftn : The forward N-D FFT of real input,
            of which `ifftn` is the inverse.
    fft : The 1-D FFT, with definitions and conventions used.
    irfft : The inverse of the 1-D FFT of real input.
    irfft2 : The inverse of the 2-D FFT of real input.

    Notes
    -----
    # 查看 `fft` 获取使用的定义和约定。
    See `fft` for definitions and conventions used.

    # 查看 `rfft` 获取实数输入使用的定义和约定。
    See `rfft` for definitions and conventions used for real input.

    # `s` 的默认值假设最终轴上的输出长度为偶数。

    The default value of `s` assumes an even output length in the final
    """
    返回一个元组，其中包含一个 x 对象，它是一个 Dispatchable 对象和一个 np.ndarray 对象的实例。

    Parameters
    ----------
    x : np.ndarray
        输入的 N 维数组，用于执行逆实部域 N 维傅里叶变换的操作。

    Returns
    -------
    tuple
        包含一个 Dispatchable 对象和一个 np.ndarray 对象的元组，这两个对象分别是输入 x 的封装和原始的 N 维逆实部域傅里叶变换结果。

    Notes
    -----
    在执行最终的复数到实数变换时，由于埃尔米特对称性要求沿着该轴的最后一个虚部分量必须为 0，因此被忽略。为了避免丢失信息，必须提供正确的实数输入长度。

    Examples
    --------
    >>> import scipy.fft
    >>> import numpy as np
    >>> x = np.zeros((3, 2, 2))
    >>> x[0, 0, 0] = 3 * 2 * 2
    >>> scipy.fft.irfftn(x)
    array([[[1.,  1.],
            [1.,  1.]],
           [[1.,  1.],
            [1.,  1.]],
           [[1.,  1.],
            [1.,  1.]]])

    """
    return (Dispatchable(x, np.ndarray),)
@_dispatch
# 定义一个分派函数，用于分发函数调用
def irfft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None, *,
           plan=None):
    """
    Computes the inverse of `rfft2`

    Parameters
    ----------
    x : array_like
        The input array
    s : sequence of ints, optional
        Shape of the real output to the inverse FFT.
    axes : sequence of ints, optional
        The axes over which to compute the inverse fft.
        Default is the last two axes.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : ndarray
        The result of the inverse real 2-D FFT.

    See Also
    --------
    rfft2 : The 2-D FFT of real input.
    irfft : The inverse of the 1-D FFT of real input.
    irfftn : The inverse of the N-D FFT of real input.

    Notes
    -----
    This is really `irfftn` with different defaults.
    For more details see `irfftn`.

    """
    # 返回一个包含 Dispatchable 对象和 numpy.ndarray 类型的元组
    return (Dispatchable(x, np.ndarray),)


@_dispatch
# 定义一个分派函数，用于分发函数调用
def hfftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *,
          plan=None):
    """
    Compute the N-D FFT of Hermitian symmetric complex input, i.e., a
    signal with a real spectrum.

    This function computes the N-D discrete Fourier Transform for a
    Hermitian symmetric complex input over any number of axes in an
    M-D array by means of the Fast Fourier Transform (FFT). In other
    words, ``ihfftn(hfftn(x, s)) == x`` to within numerical accuracy. (``s``
    here is ``x.shape`` with ``s[-1] = x.shape[-1] * 2 - 1``, this is necessary
    for the same reason ``x.shape`` would be necessary for `irfft`.)

    Parameters
    ----------
    x : array_like
        Input array.

    s : sequence of ints, optional
        Shape of the output array. The last dimension is assumed to
        have length ``x.shape[-1] // 2 + 1``, corresponding to
        non-redundant FFT result. Default is ``x.shape``.

    axes : sequence of ints, optional
        Axes over which to compute the FFT. If not given, the last
        ``len(s)`` dimensions are used.

    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".

    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.

    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.

    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : complex ndarray
        The result of the N-D FFT of the Hermitian symmetric input `x`.

    See Also
    --------
    hfft : The 1-D FFT of a Hermitian symmetric input.
    ihfft : The inverse of the 1-D FFT of a Hermitian symmetric input.
    ihfftn : The inverse of the N-D FFT of a Hermitian symmetric input.

    Notes
    -----
    This function computes the N-D FFT of a Hermitian symmetric input,
    exploiting the redundancy in the FFT of real inputs to compute the
    result in the same space as a complex FFT.

    The input `x` must have exactly `x.shape[-1] // 2 + 1` along the
    last dimension, representing the non-redundant FFT result. The output
    will have `s` as its shape, with the last dimension having length
    `x.shape[-1] // 2 + 1`.

    """
    # 定义函数 ifftn，用于执行多维逆傅里叶变换
    s : sequence of ints, optional
        # 输出数组的形状，即每个转换轴的长度
        Shape (length of each transformed axis) of the output
        # （``s[0]`` 指代轴 0，``s[1]`` 指代轴 1，以此类推）。
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.). `s` is also the
        # 对于最后一个轴，使用 ``s[-1]//2+1`` 个输入点。
        number of input points used along this axis, except for the last axis,
        where ``s[-1]//2+1`` points of the input are used.
        # 如果 `s` 指定的形状小于输入的形状，则进行裁剪。如果大于，则用零填充。
        Along any axis, if the shape indicated by `s` is smaller than that of
        the input, the input is cropped. If it is larger, the input is padded
        with zeros. If `s` is not given, the shape of the input along the axes
        specified by axes is used. Except for the last axis which is taken to be
        ``2*(m-1)`` where ``m`` is the length of the input along that axis.
    axes : sequence of ints, optional
        # 用于计算逆 FFT 的轴。如果未指定，则使用最后 `len(s)` 个轴，或者如果 `s` 也未指定，则使用所有轴。
        Axes over which to compute the inverse FFT. If not given, the last
        `len(s)` axes are used, or all axes if `s` is also not specified.
    norm : {"backward", "ortho", "forward"}, optional
        # 归一化模式（参见 `fft`）。默认为 "backward"。
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        # 如果为 True，则可以破坏 `x` 的内容；默认为 False。
        If True, the contents of `x` can be destroyed; the default is False.
        # 更多细节请参见 :func:`fft`。
        See :func:`fft` for more details.
    workers : int, optional
        # 用于并行计算的最大工作线程数。如果为负数，则从 ``os.cpu_count()`` 循环回绕。
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        # 更多细节请参见 :func:`~scipy.fft.fft`。
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        # 保留参数，用于传递由下游 FFT 供应商提供的预计算计划。目前在 SciPy 中未使用。
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : ndarray
        # 被截断或者用零填充的输入，沿着由 `axes` 指示的轴进行变换，或者由 `s` 或 `x` 的组合，如上面参数部分解释的那样。
        The truncated or zero-padded input, transformed along the axes
        # 每个转换轴的长度由相应的 `s` 元素给出，如果 `s` 未给出，则除了最后一个轴外每个轴的长度均为输入的长度。
        indicated by `axes`, or by a combination of `s` or `x`,
        as explained in the parameters section above.
        # 当 `s` 未给出时，最后一个转换轴的输出长度为 ``2*(m-1)``，其中 ``m`` 是输入最后一个转换轴的长度。
        The length of each transformed axis is as given by the corresponding
        element of `s`, or the length of the input in every axis except for the
        last one if `s` is not given.  In the final transformed axis the length
        of the output when `s` is not given is ``2*(m-1)`` where ``m`` is the
        length of the final transformed axis of the input.
        # 若要在最终轴中得到奇数个输出点，必须指定 `s`。
        To get an odd number of output points in the final axis, `s` must be specified.

    Raises
    ------
    ValueError
        # 如果 `s` 和 `axes` 的长度不同。
        If `s` and `axes` have different length.
    IndexError
        # 如果 `axes` 中的元素大于 `x` 的轴数。
        If an element of `axes` is larger than the number of axes of `x`.

    See Also
    --------
    ihfftn : The inverse N-D FFT with real spectrum. Inverse of `hfftn`.
    fft : The 1-D FFT, with definitions and conventions used.
    rfft : Forward FFT of real input.

    Notes
    -----
    # 对于 1-D 信号 ``x`` 具有实谱，它必须满足 Hermite 性质：
    For a 1-D signal ``x`` to have a real spectrum, it must satisfy
        # 对于所有的 i，都有 x[i] == np.conj(x[-i])。
        x[i] == np.conj(x[-i]) for all i
    # 通过依次在每个轴上反射来推广到更高维度：
    This generalizes into higher dimensions by reflecting over each axis in
        # 对于所有的 i, j, k, ...，都有 x[i, j, k, ...] == np.conj(x[-i, -j, -k, ...])。
        turn::

        x[i, j, k, ...] == np.conj(x[-i, -j, -k, ...]) for all i, j, k, ...
    This should not be confused with a Hermitian matrix, for which the
    transpose is its own conjugate::

        x[i, j] == np.conj(x[j, i]) for all i, j


    The default value of `s` assumes an even output length in the final
    transformation axis. When performing the final complex to real
    transformation, the Hermitian symmetry requires that the last imaginary
    component along that axis must be 0 and so it is ignored. To avoid losing
    information, the correct length of the real input *must* be given.

    Examples
    --------
    >>> import scipy.fft
    >>> import numpy as np
    >>> x = np.ones((3, 2, 2))
    >>> scipy.fft.hfftn(x)
    array([[[12.,  0.],
            [ 0.,  0.]],
           [[ 0.,  0.],
            [ 0.,  0.]],
           [[ 0.,  0.],
            [ 0.,  0.]]])

    """
    # 返回一个包含 x 和 np.ndarray 的 Dispatchable 对象的元组
    return (Dispatchable(x, np.ndarray),)
# 声明一个装饰器函数，用于分发函数调用
@_dispatch
# 定义函数 hfft2，用于计算二维 Hermitian 复数数组的 FFT
def hfft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None, *,
          plan=None):
    """
    Compute the 2-D FFT of a Hermitian complex array.

    Parameters
    ----------
    x : array
        Input array, taken to be Hermitian complex.
    s : sequence of ints, optional
        Shape of the real output.
    axes : sequence of ints, optional
        Axes over which to compute the FFT.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See `fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : ndarray
        The real result of the 2-D Hermitian complex real FFT.

    See Also
    --------
    hfftn : Compute the N-D discrete Fourier Transform for Hermitian
            complex input.

    Notes
    -----
    This is really just `hfftn` with different default behavior.
    For more details see `hfftn`.

    """
    # 返回一个包含 Dispatchable 对象的元组，该对象包装了输入 x 和 np.ndarray
    return (Dispatchable(x, np.ndarray),)


# 声明一个装饰器函数，用于分发函数调用
@_dispatch
# 定义函数 ihfftn，用于计算实数频谱的 N 维逆离散傅里叶变换（IDFT）
def ihfftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *,
           plan=None):
    """
    Compute the N-D inverse discrete Fourier Transform for a real
    spectrum.

    This function computes the N-D inverse discrete Fourier Transform
    over any number of axes in an M-D real array by means of the Fast
    Fourier Transform (FFT). By default, all axes are transformed, with the
    real transform performed over the last axis, while the remaining transforms
    are complex.

    Parameters
    ----------
    x : array_like
        Input array, taken to be real.
    s : sequence of ints, optional
        Shape (length along each transformed axis) to use from the input.
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        Along any axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.
        if `s` is not given, the shape of the input along the axes specified
        by `axes` is used.
    axes : sequence of ints, optional
        Axes over which to compute the FFT. If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".

    """
    overwrite_x : bool, optional
        如果为 True，则允许销毁 `x` 的内容；默认为 False。
        详见 :func:`fft` 获取更多细节。
    workers : int, optional
        最大并行计算的工作线程数。如果为负数，则从 ``os.cpu_count()`` 循环计数。
        详见 :func:`~scipy.fft.fft` 获取更多细节。
    plan : object, optional
        此参数保留用于传递下游 FFT 供应商提供的预计算计划。目前在 SciPy 中未使用。

        .. versionadded:: 1.5.0

    Returns
    -------
    out : complex ndarray
        截断或零填充的输入，在由 `axes` 指示的轴上进行变换，或者由 `s` 和 `x` 的组合进行变换，
        如上面参数部分所述。
        最后一个轴的长度将为 ``s[-1]//2+1``，而其余变换的轴的长度将按照 `s` 或保持输入不变。

    Raises
    ------
    ValueError
        如果 `s` 和 `axes` 的长度不同。
    IndexError
        如果 `axes` 的元素大于 `x` 的轴数。

    See Also
    --------
    hfftn : 厄米矩阵输入的 N 维正向 FFT。
    hfft : 厄米矩阵输入的 1 维 FFT。
    fft : 使用的定义和约定的 1 维 FFT。
    fftn : N 维 FFT。
    hfft2 : 厄米矩阵输入的 2 维 FFT。

    Notes
    -----
    对于实数输入，变换在最后一个变换轴上执行，类似于 `ihfft`，然后在剩余轴上执行 `ifftn`。
    输出的顺序是 Hermitian 输出信号的正部分，格式与 `rfft` 相同。

    Examples
    --------
    >>> import scipy.fft
    >>> import numpy as np
    >>> x = np.ones((2, 2, 2))
    >>> scipy.fft.ihfftn(x)
    array([[[1.+0.j,  0.+0.j], # 结果可能有所不同
            [0.+0.j,  0.+0.j]],
           [[0.+0.j,  0.+0.j],
            [0.+0.j,  0.+0.j]]])
    >>> scipy.fft.ihfftn(x, axes=(2, 0))
    array([[[1.+0.j,  0.+0.j], # 结果可能有所不同
            [1.+0.j,  0.+0.j]],
           [[0.+0.j,  0.+0.j],
            [0.+0.j,  0.+0.j]]])

    """
    return (Dispatchable(x, np.ndarray),)
# 定义一个装饰器，用于分派函数调用的处理
@_dispatch
# 定义一个函数，用于计算二维实数频谱的逆FFT
def ihfft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None, *,
           plan=None):
    """
    Compute the 2-D inverse FFT of a real spectrum.

    Parameters
    ----------
    x : array_like
        输入数组
    s : sequence of ints, optional
        输入数组的实数部分的形状
    axes : sequence of ints, optional
        计算逆FFT的轴。默认为最后两个轴。
    norm : {"backward", "ortho", "forward"}, optional
        归一化模式（参见 `fft`）。默认为 "backward"。
    overwrite_x : bool, optional
        如果为True，则可以销毁 `x` 的内容；默认为False。
        更多细节请参见 :func:`fft`。
    workers : int, optional
        并行计算时使用的最大工作线程数。如果为负数，则从 `os.cpu_count()` 循环取值。
        更多细节请参见 :func:`~scipy.fft.fft`。
    plan : object, optional
        预留参数，用于传递下游FFT供应商提供的预先计算的计划。
        当前在 SciPy 中未使用。

        .. versionadded:: 1.5.0

    Returns
    -------
    out : ndarray
        逆实数二维FFT的结果数组。

    See Also
    --------
    ihfftn : 计算埃尔米特输入的N维FFT的逆变换。

    Notes
    -----
    实际上这是 `ihfftn`，只是默认参数不同。
    更多细节请参见 `ihfftn`。
    """
    # 返回一个元组，包含输入数组 `x` 的分发对象和 `np.ndarray` 类型
    return (Dispatchable(x, np.ndarray),)
```