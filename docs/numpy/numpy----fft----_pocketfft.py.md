# `.\numpy\numpy\fft\_pocketfft.py`

```
# 定义一个偏函数，用于分派数组函数调用，指定模块为 'numpy.fft'
array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy.fft')

# 函数 `_raw_fft` 实现原始的 FFT/IFFT 操作，处理参数 a 是输入数组，n 是数据点数，axis 是轴向，
# is_real 表示数据是否为实数，is_forward 表示是否进行正向变换，norm 是规范化参数，out 是输出数组
def _raw_fft(a, n, axis, is_real, is_forward, norm, out=None):
    # 如果 n 小于 1，则抛出数值错误
    if n < 1:
        raise ValueError(f"Invalid number of FFT data points ({n}) specified.")

    # 计算规范化因子，传入数组的数据类型，避免在可能的 sqrt 或 reciprocal 运算中丢失精度
    if not is_forward:
        norm = _swap_direction(norm)

    # 获取数组实部的数据类型
    real_dtype = result_type(a.real.dtype, 1.0)

    # 根据 norm 参数确定规范化因子 fct
    if norm is None or norm == "backward":
        fct = 1
    elif norm == "ortho":
        fct = reciprocal(sqrt(n, dtype=real_dtype))
    elif norm == "forward":
        fct = reciprocal(n, dtype=real_dtype)
    else:
        raise ValueError(f'Invalid norm value {norm}; should be "backward",'
                         '"ortho" or "forward".')

    # 如果数据是实数类型
    if is_real:
        # 如果是正向变换
        if is_forward:
            # 根据 n 的奇偶性选择相应的实数 FFT 函数
            ufunc = pfu.rfft_n_even if n % 2 == 0 else pfu.rfft_n_odd
            n_out = n // 2 + 1  # 更新输出数据点数
        else:
            ufunc = pfu.irfft  # 反向实数 FFT
    else:
        ufunc = pfu.fft if is_forward else pfu.ifft  # 复数 FFT 或 IFFT

    # 标准化轴的索引
    axis = normalize_axis_index(axis, a.ndim)
    # 如果输出参数 out 为 None
    if out is None:
        # 如果是实数且非正向变换（irfft），则输出类型为实数的数据类型
        if is_real and not is_forward:  # irfft, complex in, real output.
            out_dtype = real_dtype
        else:  # 其他情况，输出为复数类型
            out_dtype = result_type(a.dtype, 1j)
        
        # 创建一个空的输出数组，保持输入数组的维度（除了指定轴的维度改为 n_out）
        out = empty(a.shape[:axis] + (n_out,) + a.shape[axis+1:],
                    dtype=out_dtype)
    
    # 如果输出参数 out 不为 None，并且其 shape 属性存在
    elif ((shape := getattr(out, "shape", None)) is not None
          and (len(shape) != a.ndim or shape[axis] != n_out)):
        # 如果输出数组的维度与输入数组不匹配，抛出 ValueError 异常
        raise ValueError("output array has wrong shape.")
    
    # 调用 ufunc 函数进行计算，并指定相关参数
    return ufunc(a, fct, axes=[(axis,), (), (axis,)], out=out)


这段代码是一个函数内的逻辑判断和返回语句。根据给定的条件，它决定如何处理输出数组 `out`，并调用 `ufunc` 函数进行计算。
# 定义一个映射表，用于将给定的 norm 参数值映射为对应的方向字符串
_SWAP_DIRECTION_MAP = {"backward": "forward", None: "forward",
                       "ortho": "ortho", "forward": "backward"}

# 根据给定的 norm 参数值返回相应的方向字符串
def _swap_direction(norm):
    try:
        return _SWAP_DIRECTION_MAP[norm]
    except KeyError:
        # 如果 norm 参数值不在映射表中，则抛出 ValueError 异常
        raise ValueError(f'Invalid norm value {norm}; should be "backward", '
                         '"ortho" or "forward".') from None

# 定义一个函数调度器 _fft_dispatcher，返回输入数组 a 和输出数组 out
def _fft_dispatcher(a, n=None, axis=None, norm=None, out=None):
    return (a, out)

# 使用装饰器 array_function_dispatch 将 fft 函数与 _fft_dispatcher 函数关联起来
@array_function_dispatch(_fft_dispatcher)
def fft(a, n=None, axis=-1, norm=None, out=None):
    """
    Compute the one-dimensional discrete Fourier Transform.

    This function computes the one-dimensional *n*-point discrete Fourier
    Transform (DFT) with the efficient Fast Fourier Transform (FFT)
    algorithm [CT].

    Parameters
    ----------
    a : array_like
        Input array, can be complex.
    n : int, optional
        Length of the transformed axis of the output.
        If `n` is smaller than the length of the input, the input is cropped.
        If it is larger, the input is padded with zeros.  If `n` is not given,
        the length of the input along the axis specified by `axis` is used.
    axis : int, optional
        Axis over which to compute the FFT.  If not given, the last axis is
        used.
    norm : {"backward", "ortho", "forward"}, optional
        .. versionadded:: 1.10.0

        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

        .. versionadded:: 1.20.0

            The "backward", "forward" values were added.
    out : complex ndarray, optional
        If provided, the result will be placed in this array. It should be
        of the appropriate shape and dtype.

        .. versionadded:: 2.0.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.

    Raises
    ------
    IndexError
        If `axis` is not a valid axis of `a`.

    See Also
    --------
    numpy.fft : for definition of the DFT and conventions used.
    ifft : The inverse of `fft`.
    fft2 : The two-dimensional FFT.
    fftn : The *n*-dimensional FFT.
    rfftn : The *n*-dimensional FFT of real input.
    fftfreq : Frequency bins for given FFT parameters.

    Notes
    -----
    FFT (Fast Fourier Transform) refers to a way the discrete Fourier
    Transform (DFT) can be calculated efficiently, by using symmetries in the
    calculated terms.  The symmetry is highest when `n` is a power of 2, and
    the transform is therefore most efficient for these sizes.

    The DFT is defined, with the conventions used in this implementation, in
    the documentation for the `numpy.fft` module.

    References
    ----------

    """
    # 将输入转换为数组，如果已经是数组则不进行转换
    a = asarray(a)
    # 如果未指定变换的长度n，则使用数组a在指定轴上的长度
    if n is None:
        n = a.shape[axis]
    # 调用底层FFT函数进行计算，并返回结果
    output = _raw_fft(a, n, axis, False, True, norm, out)
    # 返回FFT的计算结果
    return output
# 将 ifft 函数的调用分派到 _fft_dispatcher
@array_function_dispatch(_fft_dispatcher)
# 定义 ifft 函数
def ifft(a, n=None, axis=-1, norm=None, out=None):
    """
    Compute the one-dimensional inverse discrete Fourier Transform.

    This function computes the inverse of the one-dimensional *n*-point
    discrete Fourier transform computed by `fft`.  In other words,
    ``ifft(fft(a)) == a`` to within numerical accuracy.
    For a general description of the algorithm and definitions,
    see `numpy.fft`.

    # The input should be ordered in the same way as is returned by `fft`,
    # i.e.,

    # ``a[0]`` should contain the zero frequency term,
    # ``a[1:n//2]`` should contain the positive-frequency terms,
    # ``a[n//2 + 1:]`` should contain the negative-frequency terms, in
    # increasing order starting from the most negative frequency.

    # For an even number of input points, ``A[n//2]`` represents the sum of
    # the values at the positive and negative Nyquist frequencies, as the two
    # are aliased together. See `numpy.fft` for details.

    # Parameters
    # a : array_like
    #     Input array, can be complex.
    # n : int, optional
    #     Length of the transformed axis of the output.
    #     If `n` is smaller than the length of the input, the input is cropped.
    #     If it is larger, the input is padded with zeros.  If `n` is not given,
    #     the length of the input along the axis specified by `axis` is used.
    #     See notes about padding issues.
    # axis : int, optional
    #     Axis over which to compute the inverse DFT.  If not given, the last
    #     axis is used.
    # norm : {"backward", "ortho", "forward"}, optional
    #     .. versionadded:: 1.10.0

    #     Normalization mode (see `numpy.fft`). Default is "backward".
    #     Indicates which direction of the forward/backward pair of transforms
    #     is scaled and with what normalization factor.

    #     .. versionadded:: 1.20.0

    #         The "backward", "forward" values were added.

    # out : complex ndarray, optional
    #     If provided, the result will be placed in this array. It should be
    #     of the appropriate shape and dtype.

    #     .. versionadded:: 2.0.0

    # Returns
    # -------
    # out : complex ndarray
    #     The truncated or zero-padded input, transformed along the axis
    #     indicated by `axis`, or the last one if `axis` is not specified.

    # Raises
    # ------
    # IndexError
    #     If `axis` is not a valid axis of `a`.

    # See Also
    # --------
    # numpy.fft : An introduction, with definitions and general explanations.
    # fft : The one-dimensional (forward) FFT, of which `ifft` is the inverse
    # ifft2 : The two-dimensional inverse FFT.
    # ifftn : The n-dimensional inverse FFT.

    # Notes
    # -----
    # If the input parameter `n` is larger than the size of the input, the input
    # is padded by appending zeros at the end.  Even though this is the common
    # approach, it might lead to surprising results.  If a different padding is
    # desired, it must be performed before calling `ifft`.
    # 将输入转换为 ndarray 对象
    a = asarray(a)
    # 如果 n 为 None，则设为数组 a 在指定轴上的形状
    if n is None:
        n = a.shape[axis]
    # 执行原始 FFT 算法，生成频谱数据
    output = _raw_fft(a, n, axis, False, False, norm, out=out)
    # 返回 FFT 的输出结果
    return output
@array_function_dispatch(_fft_dispatcher)
# 使用 array_function_dispatch 装饰器来分派实现不同的 FFT 操作
def rfft(a, n=None, axis=-1, norm=None, out=None):
    """
    Compute the one-dimensional discrete Fourier Transform for real input.

    This function computes the one-dimensional *n*-point discrete Fourier
    Transform (DFT) of a real-valued array by means of an efficient algorithm
    called the Fast Fourier Transform (FFT).

    Parameters
    ----------
    a : array_like
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
        .. versionadded:: 1.10.0

        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

        .. versionadded:: 1.20.0

            The "backward", "forward" values were added.

    out : complex ndarray, optional
        If provided, the result will be placed in this array. It should be
        of the appropriate shape and dtype.

        .. versionadded:: 2.0.0

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
        If `axis` is not a valid axis of `a`.

    See Also
    --------
    numpy.fft : For definition of the DFT and conventions used.
    irfft : The inverse of `rfft`.
    fft : The one-dimensional FFT of general (complex) input.
    fftn : The *n*-dimensional FFT.
    rfftn : The *n*-dimensional FFT of real input.

    Notes
    -----
    When the DFT is computed for purely real input, the output is
    Hermitian-symmetric, i.e. the negative frequency terms are just the complex
    conjugates of the corresponding positive-frequency terms, and the
    negative-frequency terms are therefore redundant.  This function does not
    compute the negative frequency terms, and the length of the transformed
    axis of the output is therefore ``n//2 + 1``.

    When ``A = rfft(a)`` and fs is the sampling frequency, ``A[0]`` contains
    the zero-frequency term 0*fs, which is real due to Hermitian symmetry.

    If `n` is even, ``A[-1]`` contains the term representing both positive
    and negative Nyquist frequency (+fs/2 and -fs/2), and must also be purely
    real. If `n` is odd, there is no term at fs/2; ``A[-1]`` contains
    """
    # 实现一维实数输入的快速傅里叶变换（FFT）
    # 这个函数通过FFT算法计算实值数组的一维n点离散傅里叶变换（DFT）

    # ...
    # 具体实现部分，根据输入参数进行相应的FFT计算
    # 将输入数组 `a` 转换为 `numpy` 数组，确保操作的一致性和正确性
    a = asarray(a)
    
    # 如果未指定变换长度 `n`，则默认为输入数组 `a` 在指定轴 `axis` 上的长度
    if n is None:
        n = a.shape[axis]
    
    # 调用 `_raw_fft` 函数执行快速傅里叶变换，返回变换后的结果
    # `True, True, norm, out=out` 分别代表：是否要进行正规化，是否要进行轴对称，正规化参数，输出参数
    output = _raw_fft(a, n, axis, True, True, norm, out=out)
    
    # 返回傅里叶变换的结果
    return output
# 使用装饰器将函数注册到数组函数分派机制中，这可以根据输入数组的类型调用不同的函数版本
@array_function_dispatch(_fft_dispatcher)
# 定义函数 irfft，计算 rfft 的逆操作
def irfft(a, n=None, axis=-1, norm=None, out=None):
    """
    Computes the inverse of `rfft`.

    This function computes the inverse of the one-dimensional *n*-point
    discrete Fourier Transform of real input computed by `rfft`.
    In other words, ``irfft(rfft(a), len(a)) == a`` to within numerical
    accuracy. (See Notes below for why ``len(a)`` is necessary here.)

    The input is expected to be in the form returned by `rfft`, i.e. the
    real zero-frequency term followed by the complex positive frequency terms
    in order of increasing frequency.  Since the discrete Fourier Transform of
    real input is Hermitian-symmetric, the negative frequency terms are taken
    to be the complex conjugates of the corresponding positive frequency terms.

    Parameters
    ----------
    a : array_like
        The input array.
    n : int, optional
        Length of the transformed axis of the output.
        For `n` output points, ``n//2+1`` input points are necessary.  If the
        input is longer than this, it is cropped.  If it is shorter than this,
        it is padded with zeros.  If `n` is not given, it is taken to be
        ``2*(m-1)`` where ``m`` is the length of the input along the axis
        specified by `axis`.
    axis : int, optional
        Axis over which to compute the inverse FFT. If not given, the last
        axis is used.
    norm : {"backward", "ortho", "forward"}, optional
        .. versionadded:: 1.10.0

        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

        .. versionadded:: 1.20.0

            The "backward", "forward" values were added.

    out : ndarray, optional
        If provided, the result will be placed in this array. It should be
        of the appropriate shape and dtype.

        .. versionadded:: 2.0.0

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
        If `axis` is not a valid axis of `a`.

    See Also
    --------
    numpy.fft : For definition of the DFT and conventions used.
    rfft : The one-dimensional FFT of real input, of which `irfft` is inverse.
    fft : The one-dimensional FFT.
    irfft2 : The inverse of the two-dimensional FFT of real input.
    irfftn : The inverse of the *n*-dimensional FFT of real input.

    Notes
    -----
    Returns the real valued `n`-point inverse discrete Fourier transform
    of `a`, where `a` contains the non-negative frequency terms of a
    """
    a = asarray(a)
    # 将输入的数组 `a` 转换为 NumPy 数组，确保可以进行 FFT 操作

    if n is None:
        # 如果输入的输出长度 `n` 为 None，则根据输入数组的维度来确定输出长度
        n = (a.shape[axis] - 1) * 2

    # 调用内部函数 `_raw_fft` 进行 FFT 操作，生成输出结果
    # 参数含义依次为：输入数组 `a`，输出长度 `n`，操作轴 `axis`，进行逆变换 `True`，不进行归一化 `False`，指定输出数组 `out`
    output = _raw_fft(a, n, axis, True, False, norm, out=out)

    # 返回 FFT 变换后的输出结果
    return output
# 使用装饰器实现函数分派，将该函数与适当的_fft_dispatcher分派器相关联
@array_function_dispatch(_fft_dispatcher)
# 定义一个函数hfft，用于计算具有Hermitian对称性的信号的FFT，即实部频谱
def hfft(a, n=None, axis=-1, norm=None, out=None):
    """
    Compute the FFT of a signal that has Hermitian symmetry, i.e., a real
    spectrum.

    Parameters
    ----------
    a : array_like
        The input array.
    n : int, optional
        Length of the transformed axis of the output. For `n` output
        points, ``n//2 + 1`` input points are necessary.  If the input is
        longer than this, it is cropped.  If it is shorter than this, it is
        padded with zeros.  If `n` is not given, it is taken to be ``2*(m-1)``
        where ``m`` is the length of the input along the axis specified by
        `axis`.
    axis : int, optional
        Axis over which to compute the FFT. If not given, the last
        axis is used.
    norm : {"backward", "ortho", "forward"}, optional
        .. versionadded:: 1.10.0

        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

        .. versionadded:: 1.20.0

            The "backward", "forward" values were added.

    out : ndarray, optional
        If provided, the result will be placed in this array. It should be
        of the appropriate shape and dtype.

        .. versionadded:: 2.0.0

    Returns
    -------
    out : ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.
        The length of the transformed axis is `n`, or, if `n` is not given,
        ``2*m - 2`` where ``m`` is the length of the transformed axis of
        the input. To get an odd number of output points, `n` must be
        specified, for instance as ``2*m - 1`` in the typical case,

    Raises
    ------
    IndexError
        If `axis` is not a valid axis of `a`.

    See also
    --------
    rfft : Compute the one-dimensional FFT for real input.
    ihfft : The inverse of `hfft`.

    Notes
    -----
    `hfft`/`ihfft` are a pair analogous to `rfft`/`irfft`, but for the
    opposite case: here the signal has Hermitian symmetry in the time
    domain and is real in the frequency domain. So here it's `hfft` for
    which you must supply the length of the result if it is to be odd.

    * even: ``ihfft(hfft(a, 2*len(a) - 2)) == a``, within roundoff error,
    * odd: ``ihfft(hfft(a, 2*len(a) - 1)) == a``, within roundoff error.

    The correct interpretation of the hermitian input depends on the length of
    the original data, as given by `n`. This is because each input shape could
    correspond to either an odd or even length signal. By default, `hfft`
    assumes an even output length which puts the last entry at the Nyquist
    frequency; aliasing with its symmetric counterpart. By Hermitian symmetry,
    the value is thus treated as purely real. To avoid losing information, the
    shape of the full signal **must** be given.
    """
    # 实现代码逻辑在此处
    # 将输入数组 `a` 转换为 NumPy 数组
    a = asarray(a)
    # 如果未指定频谱长度 `n`，则设定为 `(a.shape[axis] - 1) * 2`
    if n is None:
        n = (a.shape[axis] - 1) * 2
    # 交换规范化方向 `norm` 的定义，返回新的规范化方式
    new_norm = _swap_direction(norm)
    # 对输入数组 `a` 进行共轭处理后，执行逆快速傅里叶变换
    # 返回频谱，可以指定频谱长度 `n`，轴 `axis`，以及输出数组 `out` 的选项
    output = irfft(conjugate(a), n, axis, norm=new_norm, out=None)
    return output
# 使用array_function_dispatch修饰符，将函数_ihfft分派给_fft_dispatcher
def ihfft(a, n=None, axis=-1, norm=None, out=None):
    """
    Compute the inverse FFT of a signal that has Hermitian symmetry.

    Parameters
    ----------
    a : array_like
        Input array.
    n : int, optional
        Length of the inverse FFT, the number of points along
        transformation axis in the input to use.  If `n` is smaller than
        the length of the input, the input is cropped.  If it is larger,
        the input is padded with zeros. If `n` is not given, the length of
        the input along the axis specified by `axis` is used.
    axis : int, optional
        Axis over which to compute the inverse FFT. If not given, the last
        axis is used.
    norm : {"backward", "ortho", "forward"}, optional
        .. versionadded:: 1.10.0

        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

        .. versionadded:: 1.20.0

            The "backward", "forward" values were added.

    out : complex ndarray, optional
        If provided, the result will be placed in this array. It should be
        of the appropriate shape and dtype.

        .. versionadded:: 2.0.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.
        The length of the transformed axis is ``n//2 + 1``.

    See also
    --------
    hfft, irfft

    Notes
    -----
    `hfft`/`ihfft` are a pair analogous to `rfft`/`irfft`, but for the
    opposite case: here the signal has Hermitian symmetry in the time
    domain and is real in the frequency domain. So here it's `hfft` for
    which you must supply the length of the result if it is to be odd:

    * even: ``ihfft(hfft(a, 2*len(a) - 2)) == a``, within roundoff error,
    * odd: ``ihfft(hfft(a, 2*len(a) - 1)) == a``, within roundoff error.

    Examples
    --------
    >>> spectrum = np.array([ 15, -4, 0, -1, 0, -4])
    >>> np.fft.ifft(spectrum)
    array([1.+0.j,  2.+0.j,  3.+0.j,  4.+0.j,  3.+0.j,  2.+0.j]) # may vary
    >>> np.fft.ihfft(spectrum)
    array([ 1.-0.j,  2.-0.j,  3.-0.j,  4.-0.j]) # may vary

    """
    # 将a转换为数组
    a = asarray(a)
    # 如果n为None，则n等于a在axis轴上的形状
    if n is None:
        n = a.shape[axis]
    # 将norm参数用于交换方向，并赋值给new_norm
    new_norm = _swap_direction(norm)
    # 使用rfft函数计算FFT的逆变换，并将结果存入out中
    out = rfft(a, n, axis, norm=new_norm, out=out)
    # 返回结果的共轭
    return conjugate(out, out=out)


def _cook_nd_args(a, s=None, axes=None, invreal=0):
    # 如果形状s为None
    if s is None:
        shapeless = True
        # 如果轴为None，则s等于a的形状的列表形式
        if axes is None:
            s = list(a.shape)
        # 否则，s等于在轴上获取a的形状的列表形式
        else:
            s = take(a.shape, axes)
    else:
        shapeless = False
    # s转换为列表形式
    s = list(s)
    # 如果 axes 参数为 None
    if axes is None:
        # 如果 shapeless 参数为 False
        if not shapeless:
            # 提示消息，警告用户在未来版本中将不再支持 axes 参数为 None 的情况
            msg = ("`axes` should not be `None` if `s` is not `None` "
                   "(Deprecated in NumPy 2.0). In a future version of NumPy, "
                   "this will raise an error and `s[i]` will correspond to "
                   "the size along the transformed axis specified by "
                   "`axes[i]`. To retain current behaviour, pass a sequence "
                   "[0, ..., k-1] to `axes` for an array of dimension k.")
            # 发出警告消息
            warnings.warn(msg, DeprecationWarning, stacklevel=3)
        # 设置 axes 为一个从负数索引到 -1 的列表，用于表示数组的各个维度
        axes = list(range(-len(s), 0))
    
    # 如果 s 的长度与 axes 的长度不同，抛出数值错误
    if len(s) != len(axes):
        raise ValueError("Shape and axes have different lengths.")
    
    # 如果 invreal 为 True 且 shapeless 为 True
    if invreal and shapeless:
        # 计算 s 中最后一个元素的值，用于对应维度的变换
        s[-1] = (a.shape[axes[-1]] - 1) * 2
    
    # 如果 s 中包含 None 值
    if None in s:
        # 提示消息，警告用户在未来版本中不再支持 s 中包含 None 值的情况
        msg = ("Passing an array containing `None` values to `s` is "
               "deprecated in NumPy 2.0 and will raise an error in "
               "a future version of NumPy. To use the default behaviour "
               "of the corresponding 1-D transform, pass the value matching "
               "the default for its `n` parameter. To use the default "
               "behaviour for every axis, the `s` argument can be omitted.")
        # 发出警告消息
        warnings.warn(msg, DeprecationWarning, stacklevel=3)
    
    # 根据 s 和 axes 的定义，构建新的 s 数组，用于描述变换后的数组形状
    s = [a.shape[_a] if _s == -1 else _s for _s, _a in zip(s, axes)]
    
    # 返回 s 数组和 axes 数组
    return s, axes
# 定义一个函数 `_raw_fftnd`，用于执行 N 维 FFT 变换
def _raw_fftnd(a, s=None, axes=None, function=fft, norm=None, out=None):
    # 将输入参数 `a` 转换为 ndarray 类型
    a = asarray(a)
    # 根据传入的参数 `a`, `s`, `axes`，获取处理后的 `s` 和 `axes`
    s, axes = _cook_nd_args(a, s, axes)
    # 倒序遍历 `axes`，对数组 `a` 执行 FFT 变换
    itl = list(range(len(axes)))
    itl.reverse()
    for ii in itl:
        a = function(a, n=s[ii], axis=axes[ii], norm=norm, out=out)
    # 返回变换后的数组 `a`
    return a


# 定义一个函数 `_fftn_dispatcher`，用于 FFTN 的分派器，返回 `(a, out)`
def _fftn_dispatcher(a, s=None, axes=None, norm=None, out=None):
    return (a, out)


# 使用装饰器 `array_function_dispatch`，将 `_fftn_dispatcher` 注册为 `fftn` 的分派器
@array_function_dispatch(_fftn_dispatcher)
# 定义函数 `fftn`，用于计算 N 维离散傅里叶变换（DFT）
def fftn(a, s=None, axes=None, norm=None, out=None):
    """
    Compute the N-dimensional discrete Fourier Transform.

    This function computes the *N*-dimensional discrete Fourier Transform over
    any number of axes in an *M*-dimensional array by means of the Fast Fourier
    Transform (FFT).

    Parameters
    ----------
    a : array_like
        Input array, can be complex.
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        This corresponds to ``n`` for ``fft(x, n)``.
        Along any axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.

        .. versionchanged:: 2.0

            If it is ``-1``, the whole input is used (no padding/trimming).

        If `s` is not given, the shape of the input along the axes specified
        by `axes` is used.

        .. deprecated:: 2.0

            If `s` is not ``None``, `axes` must not be ``None`` either.

        .. deprecated:: 2.0

            `s` must contain only ``int`` s, not ``None`` values. ``None``
            values currently mean that the default value for ``n`` is used
            in the corresponding 1-D transform, but this behaviour is
            deprecated.

    axes : sequence of ints, optional
        Axes over which to compute the FFT.  If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
        Repeated indices in `axes` means that the transform over that axis is
        performed multiple times.

        .. deprecated:: 2.0

            If `s` is specified, the corresponding `axes` to be transformed
            must be explicitly specified too.

    norm : {"backward", "ortho", "forward"}, optional
        .. versionadded:: 1.10.0

        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

        .. versionadded:: 1.20.0

            The "backward", "forward" values were added.

    out : complex ndarray, optional
        If provided, the result will be placed in this array. It should be
        of the appropriate shape and dtype for all axes (and hence is
        incompatible with passing in all but the trivial ``s``).

        .. versionadded:: 2.0.0

    Returns
    -------
    ```
    # out : 复数类型的多维数组
    #     截断或者零填充的输入数据，沿着由 `axes` 指示的轴或 `s` 和 `a` 的组合进行变换，详见上述参数部分。

    # Raises
    # ------
    # ValueError
    #     如果 `s` 和 `axes` 的长度不同。
    # IndexError
    #     如果 `axes` 中的某个元素大于 `a` 的轴数。

    # See Also
    # --------
    # numpy.fft : 整体查看离散傅里叶变换，包括使用的定义和约定。
    # ifftn : `fftn` 的逆 *n* 维 FFT。
    # fft : 一维 FFT，包括使用的定义和约定。
    # rfftn : 实数输入的 *n* 维 FFT。
    # fft2 : 二维 FFT。
    # fftshift : 将零频率项移动到数组的中心。

    # Notes
    # -----
    # 输出与 `fft` 类似，在所有轴上包含零频率项的低阶角落，所有轴上的正频率项在第一半部分，
    # 所有轴上的奈奎斯特频率项在中间，所有轴上的负频率项按照递减负频率的顺序排列。

    # 详见 `numpy.fft` 获取更多细节，定义和使用的约定。

    # Examples
    # --------
    # >>> a = np.mgrid[:3, :3, :3][0]
    # >>> np.fft.fftn(a, axes=(1, 2))
    # array([[[ 0.+0.j,   0.+0.j,   0.+0.j], # 可能会有变化
    #         [ 0.+0.j,   0.+0.j,   0.+0.j],
    #         [ 0.+0.j,   0.+0.j,   0.+0.j]],
    #        [[ 9.+0.j,   0.+0.j,   0.+0.j],
    #         [ 0.+0.j,   0.+0.j,   0.+0.j],
    #         [ 0.+0.j,   0.+0.j,   0.+0.j]],
    #        [[18.+0.j,   0.+0.j,   0.+0.j],
    #         [ 0.+0.j,   0.+0.j,   0.+0.j],
    #         [ 0.+0.j,   0.+0.j,   0.+0.j]]])
    # >>> np.fft.fftn(a, (2, 2), axes=(0, 1))
    # array([[[ 2.+0.j,  2.+0.j,  2.+0.j], # 可能会有变化
    #         [ 0.+0.j,  0.+0.j,  0.+0.j]],
    #        [[-2.+0.j, -2.+0.j, -2.+0.j],
    #         [ 0.+0.j,  0.+0.j,  0.+0.j]]])

    # >>> import matplotlib.pyplot as plt
    # >>> [X, Y] = np.meshgrid(2 * np.pi * np.arange(200) / 12,
    # ...                      2 * np.pi * np.arange(200) / 34)
    # >>> S = np.sin(X) + np.cos(Y) + np.random.uniform(0, 1, X.shape)
    # >>> FS = np.fft.fftn(S)
    # >>> plt.imshow(np.log(np.abs(np.fft.fftshift(FS))**2))
    # <matplotlib.image.AxesImage object at 0x...>
    # >>> plt.show()
    """
    使用 `_raw_fftnd` 函数对输入数据 `a` 进行多维 FFT 变换，指定了`s`、`axes`、`fft`、`norm` 参数，
    并将结果存储到 `out` 中。
    """
    return _raw_fftnd(a, s, axes, fft, norm, out=out)
# 使用装饰器进行函数分派，指定了 _fftn_dispatcher 作为分派函数
@array_function_dispatch(_fftn_dispatcher)
# 定义了 ifftn 函数，用于计算 N 维逆离散傅里叶变换（IDFT）
def ifftn(a, s=None, axes=None, norm=None, out=None):
    """
    计算 N 维逆离散傅里叶变换。

    该函数通过快速傅里叶变换（FFT）在 M 维数组的任意数量的轴上计算逆 N 维离散傅里叶变换。
    换句话说，``ifftn(fftn(a)) == a``，在数值精度范围内成立。
    有关所使用的定义和约定的描述，请参阅 `numpy.fft`。

    输入应该按照与 `fftn` 返回的顺序相同的方式排序，即所有轴的零频率项应位于低序角落，
    所有轴的正频率项应位于前半部分，所有轴的奈奎斯特频率项应位于中间，
    所有轴的负频率项应按照递减负频率的顺序排列。

    Parameters
    ----------
    a : array_like
        输入数组，可以是复数。
    s : sequence of ints, optional
        输出的形状（每个变换轴的长度）。(`s[0]` 对应轴 0, `s[1]` 对应轴 1, 等等)。
        对应于 `ifft(x, n)` 中的 `n`。
        沿任何轴，如果给定的形状小于输入的形状，则输入被截断。
        如果大于输入，则用零填充输入。

        .. versionchanged:: 2.0

            如果是 `-1`，则使用整个输入（无填充/裁剪）。

        如果未给出 `s`，则使用由 `axes` 指定的轴的输入形状。请参阅 `ifft` 的零填充问题。

        .. deprecated:: 2.0

            如果 `s` 不是 `None`，`axes` 也不应为 `None`。

        .. deprecated:: 2.0

            `s` 必须只包含 `int`，不包含 `None` 值。`None` 值当前表示在相应的一维变换中使用默认值 `n`，
            但这种行为已经不推荐使用。

    axes : sequence of ints, optional
        进行逆傅里叶变换的轴。如果未给出，则使用最后 `len(s)` 个轴，如果 `s` 也未指定，则使用所有轴。
        `axes` 中的重复索引意味着在该轴上执行多次逆变换。

        .. deprecated:: 2.0

            如果指定了 `s`，则必须显式指定要变换的相应 `axes`。

    norm : {"backward", "ortho", "forward"}, optional
        .. versionadded:: 1.10.0

        规范化模式（参见 `numpy.fft`）。默认为 "backward"。
        指示哪个方向的正向/反向变换对被缩放，并使用什么规范化因子。

        .. versionadded:: 1.20.0

            添加了 "backward"、"forward" 值。
    out : ndarray, optional
        输出数组，结果将被放置在其中。它应该具有与预期输出相同的形状和数据类型，
        否则，将引发异常。如果未提供，则将分配新的数组。

        .. versionadded:: 1.16.0
    """
    # out: 复数的 ndarray，可选参数
    #     如果提供了此参数，则结果将存储在该数组中。它应该具有适合所有轴的形状和 dtype
    #     （因此与除了平凡的 "s" 外的所有传入参数不兼容）。
    #
    #     .. versionadded:: 2.0.0
    #
    # Returns
    # -------
    # out: 复数的 ndarray
    #     截断或零填充的输入，沿着 `axes` 指定的轴或 `s` 或 `a` 的组合转换，如上面参数部分所述。
    #
    # Raises
    # ------
    # ValueError
    #     如果 `s` 和 `axes` 的长度不同。
    # IndexError
    #     如果 `axes` 中的一个元素大于 `a` 的轴数。
    #
    # See Also
    # --------
    # numpy.fft : 离散 Fourier 变换的整体视图，包括使用的定义和约定。
    # fftn : 前向 *n*-维 FFT，其中 `ifftn` 是其逆变换。
    # ifft : 一维逆 FFT。
    # ifft2 : 二维逆 FFT。
    # ifftshift : 撤消 `fftshift`，将零频率项移动到数组开头。
    #
    # Notes
    # -----
    # 查看 `numpy.fft` 获取使用的定义和约定。
    #
    # 零填充与 `ifft` 类似，通过在指定的维度末尾附加零来执行。尽管这是常见的方法，但可能会导致意外的结果。
    # 如果需要其他形式的零填充，必须在调用 `ifftn` 之前执行。
    #
    # Examples
    # --------
    # >>> a = np.eye(4)
    # >>> np.fft.ifftn(np.fft.fftn(a, axes=(0,)), axes=(1,))
    # array([[1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j], # 可能会有所不同
    #        [0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
    #        [0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
    #        [0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j]])
    #
    # 创建并绘制带限频率内容的图像：
    #
    # >>> import matplotlib.pyplot as plt
    # >>> n = np.zeros((200,200), dtype=complex)
    # >>> n[60:80, 20:40] = np.exp(1j*np.random.uniform(0, 2*np.pi, (20, 20)))
    # >>> im = np.fft.ifftn(n).real
    # >>> plt.imshow(im)
    # <matplotlib.image.AxesImage object at 0x...>
    # >>> plt.show()
    """
    return _raw_fftnd(a, s, axes, ifft, norm, out=out)
@array_function_dispatch(_fftn_dispatcher)
# 使用装饰器指定特定的函数调度器，用于快速傅里叶变换（FFT）的调度
def fft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    """
    Compute the 2-dimensional discrete Fourier Transform.

    This function computes the *n*-dimensional discrete Fourier Transform
    over any axes in an *M*-dimensional array by means of the
    Fast Fourier Transform (FFT).  By default, the transform is computed over
    the last two axes of the input array, i.e., a 2-dimensional FFT.

    Parameters
    ----------
    a : array_like
        Input array, can be complex
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        This corresponds to ``n`` for ``fft(x, n)``.
        Along each axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.

        .. versionchanged:: 2.0

            If it is ``-1``, the whole input is used (no padding/trimming).

        If `s` is not given, the shape of the input along the axes specified
        by `axes` is used.

        .. deprecated:: 2.0

            If `s` is not ``None``, `axes` must not be ``None`` either.

        .. deprecated:: 2.0

            `s` must contain only ``int`` s, not ``None`` values. ``None``
            values currently mean that the default value for ``n`` is used
            in the corresponding 1-D transform, but this behaviour is
            deprecated.

    axes : sequence of ints, optional
        Axes over which to compute the FFT.  If not given, the last two
        axes are used.  A repeated index in `axes` means the transform over
        that axis is performed multiple times.  A one-element sequence means
        that a one-dimensional FFT is performed. Default: ``(-2, -1)``.

        .. deprecated:: 2.0

            If `s` is specified, the corresponding `axes` to be transformed
            must not be ``None``.

    norm : {"backward", "ortho", "forward"}, optional
        .. versionadded:: 1.10.0

        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

        .. versionadded:: 1.20.0

            The "backward", "forward" values were added.

    out : complex ndarray, optional
        If provided, the result will be placed in this array. It should be
        of the appropriate shape and dtype for all axes (and hence only the
        last axis can have ``s`` not equal to the shape at that axis).

        .. versionadded:: 2.0.0

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
    """
    # 返回 `a` 的多维傅里叶变换结果。
    # `_raw_fftnd` 是实际执行傅里叶变换的函数。
    # `a` 是输入的数组，其元素类型通常为复数。
    # `s` 是指定每个维度的大小的元组，决定了输出数组的形状。
    # `axes` 是指定要在哪些轴上应用 FFT 的整数列表。
    # `fft` 是指定要使用的 FFT 函数，通常为 numpy.fft.fft。
    # `norm` 是一个布尔值，指定是否对变换结果进行归一化。
    # `out` 是可选的输出数组，用于存储结果。
    # 如果 `axes` 中的元素大于 `a` 的轴数，将引发 IndexError。
    
    See Also
    --------
    numpy.fft : 提供离散傅里叶变换的整体视图，包括定义和使用的约定。
    ifft2 : 二维逆傅里叶变换。
    fft : 一维傅里叶变换。
    fftn : *n* 维傅里叶变换。
    fftshift : 将零频率项移至数组中心。
        对于二维输入，交换第一和第三象限，以及第二和第四象限。
    
    Notes
    -----
    `fft2` 实际上是 `fftn` 的一种变体，其在 `axes` 参数上有不同的默认设置。
    
    输出类似于 `fft`，在转换后的轴的低阶角落包含零频率项，第一半轴包含正频率项，
    轴中间包含 Nyquist 频率项，第二半轴按递减负频率顺序排列。
    
    详细信息和绘图示例，请参阅 `fftn`，以及 `numpy.fft` 提供的定义和使用约定。
    
    Examples
    --------
    >>> a = np.mgrid[:5, :5][0]
    >>> np.fft.fft2(a)
    array([[ 50.  +0.j        ,   0.  +0.j        ,   0.  +0.j        , # 结果可能会有所不同
              0.  +0.j        ,   0.  +0.j        ],
           [-12.5+17.20477401j,   0.  +0.j        ,   0.  +0.j        ,
              0.  +0.j        ,   0.  +0.j        ],
           [-12.5 +4.0614962j ,   0.  +0.j        ,   0.  +0.j        ,
              0.  +0.j        ,   0.  +0.j        ],
           [-12.5 -4.0614962j ,   0.  +0.j        ,   0.  +0.j        ,
              0.  +0.j        ,   0.  +0.j        ],
           [-12.5-17.20477401j,   0.  +0.j        ,   0.  +0.j        ,
              0.  +0.j        ,   0.  +0.j        ]])
# 使用 array_function_dispatch 装饰器，将函数注册为 ifftn 调度器的一部分
@array_function_dispatch(_fftn_dispatcher)
# 定义 ifft2 函数，用于计算二维反离散傅里叶变换的逆变换
def ifft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    """
    Compute the 2-dimensional inverse discrete Fourier Transform.

    This function computes the inverse of the 2-dimensional discrete Fourier
    Transform over any number of axes in an M-dimensional array by means of
    the Fast Fourier Transform (FFT).  In other words, ``ifft2(fft2(a)) == a``
    to within numerical accuracy.  By default, the inverse transform is
    computed over the last two axes of the input array.

    The input, analogously to `ifft`, should be ordered in the same way as is
    returned by `fft2`, i.e. it should have the term for zero frequency
    in the low-order corner of the two axes, the positive frequency terms in
    the first half of these axes, the term for the Nyquist frequency in the
    middle of the axes and the negative frequency terms in the second half of
    both axes, in order of decreasingly negative frequency.

    Parameters
    ----------
    a : array_like
        Input array, can be complex.
    s : sequence of ints, optional
        Shape (length of each axis) of the output (``s[0]`` refers to axis 0,
        ``s[1]`` to axis 1, etc.).  This corresponds to `n` for ``ifft(x, n)``.
        Along each axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.

        .. versionchanged:: 2.0

            If it is ``-1``, the whole input is used (no padding/trimming).

        If `s` is not given, the shape of the input along the axes specified
        by `axes` is used.  See notes for issue on `ifft` zero padding.

        .. deprecated:: 2.0

            If `s` is not ``None``, `axes` must not be ``None`` either.

        .. deprecated:: 2.0

            `s` must contain only ``int`` s, not ``None`` values. ``None``
            values currently mean that the default value for ``n`` is used
            in the corresponding 1-D transform, but this behaviour is
            deprecated.

    axes : sequence of ints, optional
        Axes over which to compute the FFT.  If not given, the last two
        axes are used.  A repeated index in `axes` means the transform over
        that axis is performed multiple times.  A one-element sequence means
        that a one-dimensional FFT is performed. Default: ``(-2, -1)``.

        .. deprecated:: 2.0

            If `s` is specified, the corresponding `axes` to be transformed
            must not be ``None``.

    norm : {"backward", "ortho", "forward"}, optional
        .. versionadded:: 1.10.0

        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

        .. versionadded:: 1.20.0

            The "backward", "forward" values were added.
    """
    return _raw_fftnd(a, s, axes, ifft, norm, out=None)



# 调用 `_raw_fftnd` 函数进行 n 维 FFT 或逆 FFT 变换
return _raw_fftnd(a, s, axes, ifft, norm, out=None)


这段代码是一个函数的返回语句，调用了名为 `_raw_fftnd` 的函数，用于执行 n 维的 FFT（快速傅里叶变换）或逆 FFT 变换。函数的参数解释如下：

- `a`: 输入数组，进行 FFT 或逆 FFT 变换的原始数据。
- `s`: 可选参数，用于指定输出数组的形状。如果提供了 `out` 参数，`s` 应与 `out` 的形状兼容。
- `axes`: 可选参数，指定沿着哪些轴进行变换。如果未给出，则默认为最后两个轴。
- `ifft`: 可选参数，布尔值，指示是否执行逆 FFT 变换。如果为 `True`，执行逆 FFT；如果为 `False`，执行正向 FFT。
- `norm`: 可选参数，指定是否进行归一化处理。
- `out`: 可选参数，指定变换结果存储的目标数组。

函数返回变换后的结果数组。
# 使用 array_function_dispatch 装饰器将该函数分派给 _fftn_dispatcher 处理
@array_function_dispatch(_fftn_dispatcher)
# 定义 rfftn 函数，用于计算 N 维实数输入的离散傅里叶变换
def rfftn(a, s=None, axes=None, norm=None, out=None):
    """
    Compute the N-dimensional discrete Fourier Transform for real input.

    This function computes the N-dimensional discrete Fourier Transform over
    any number of axes in an M-dimensional real array by means of the Fast
    Fourier Transform (FFT).  By default, all axes are transformed, with the
    real transform performed over the last axis, while the remaining
    transforms are complex.

    Parameters
    ----------
    a : array_like
        Input array, taken to be real.
    s : sequence of ints, optional
        Shape (length along each transformed axis) to use from the input.
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        The final element of `s` corresponds to `n` for ``rfft(x, n)``, while
        for the remaining axes, it corresponds to `n` for ``fft(x, n)``.
        Along any axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.

        .. versionchanged:: 2.0

            If it is ``-1``, the whole input is used (no padding/trimming).

        If `s` is not given, the shape of the input along the axes specified
        by `axes` is used.

        .. deprecated:: 2.0

            If `s` is not ``None``, `axes` must not be ``None`` either.

        .. deprecated:: 2.0

            `s` must contain only ``int`` s, not ``None`` values. ``None``
            values currently mean that the default value for ``n`` is used
            in the corresponding 1-D transform, but this behaviour is
            deprecated.

    axes : sequence of ints, optional
        Axes over which to compute the FFT.  If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.

        .. deprecated:: 2.0

            If `s` is specified, the corresponding `axes` to be transformed
            must be explicitly specified too.

    norm : {"backward", "ortho", "forward"}, optional
        .. versionadded:: 1.10.0

        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

        .. versionadded:: 1.20.0

            The "backward", "forward" values were added.

    out : complex ndarray, optional
        If provided, the result will be placed in this array. It should be
        of the appropriate shape and dtype for all axes (and hence is
        incompatible with passing in all but the trivial ``s``).

        .. versionadded:: 2.0.0

    Returns
    -------
    # 将输入数组转换为 ndarray 类型，确保输入合法性
    a = asarray(a)
    
    # 根据输入参数及数组 a，处理并返回处理后的 s 和 axes
    s, axes = _cook_nd_args(a, s, axes)
    
    # 对数组 a 进行一维实数输入的 FFT 变换，根据最后一个轴进行变换
    # 变换长度为 s[-1]//2+1，使用 axes[-1] 指定的轴
    a = rfft(a, s[-1], axes[-1], norm, out=out)
    
    # 对除了最后一个轴之外的其他轴进行 FFT 变换
    for ii in range(len(axes)-1):
        a = fft(a, s[ii], axes[ii], norm, out=out)
    
    # 返回变换后的数组 a
    return a
# 使用装饰器将函数分派到 `_fftn_dispatcher`，用于处理 FFT 相关的数组函数调度
@array_function_dispatch(_fftn_dispatcher)
# 定义函数 `rfft2`，计算实数组的二维快速傅里叶变换（FFT）
def rfft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    """
    Compute the 2-dimensional FFT of a real array.

    Parameters
    ----------
    a : array
        Input array, taken to be real.
    s : sequence of ints, optional
        Shape of the FFT.

        .. versionchanged:: 2.0

            If it is ``-1``, the whole input is used (no padding/trimming).

        .. deprecated:: 2.0

            If `s` is not ``None``, `axes` must not be ``None`` either.

        .. deprecated:: 2.0

            `s` must contain only ``int`` s, not ``None`` values. ``None``
            values currently mean that the default value for ``n`` is used
            in the corresponding 1-D transform, but this behaviour is
            deprecated.

    axes : sequence of ints, optional
        Axes over which to compute the FFT. Default: ``(-2, -1)``.

        .. deprecated:: 2.0

            If `s` is specified, the corresponding `axes` to be transformed
            must not be ``None``.

    norm : {"backward", "ortho", "forward"}, optional
        .. versionadded:: 1.10.0

        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

        .. versionadded:: 1.20.0

            The "backward", "forward" values were added.

    out : complex ndarray, optional
        If provided, the result will be placed in this array. It should be
        of the appropriate shape and dtype for the last inverse transform.
        incompatible with passing in all but the trivial ``s``).

        .. versionadded:: 2.0.0

    Returns
    -------
    out : ndarray
        The result of the real 2-D FFT.

    See Also
    --------
    rfftn : Compute the N-dimensional discrete Fourier Transform for real
            input.

    Notes
    -----
    This is really just `rfftn` with different default behavior.
    For more details see `rfftn`.

    Examples
    --------
    >>> a = np.mgrid[:5, :5][0]
    >>> np.fft.rfft2(a)
    array([[ 50.  +0.j        ,   0.  +0.j        ,   0.  +0.j        ],
           [-12.5+17.20477401j,   0.  +0.j        ,   0.  +0.j        ],
           [-12.5 +4.0614962j ,   0.  +0.j        ,   0.  +0.j        ],
           [-12.5 -4.0614962j ,   0.  +0.j        ,   0.  +0.j        ],
           [-12.5-17.20477401j,   0.  +0.j        ,   0.  +0.j        ]])
    """
    # 调用 `rfftn` 函数，传入参数并返回结果，其中 `out` 参数允许传入预分配的输出数组
    return rfftn(a, s, axes, norm, out=out)


# 使用装饰器将函数分派到 `_fftn_dispatcher`，用于处理 FFT 相关的数组函数调度
@array_function_dispatch(_fftn_dispatcher)
# 定义函数 `irfftn`，计算 `rfftn` 的反变换（逆 FFT）
def irfftn(a, s=None, axes=None, norm=None, out=None):
    """
    Computes the inverse of `rfftn`.

    This function computes the inverse of the N-dimensional discrete
    Fourier Transform for real input over any number of axes in an
    M-dimensional array by means of the Fast Fourier Transform (FFT).  In
    other words, ``irfftn(rfftn(a), a.shape) == a`` to within numerical
    """
    # 这里的函数体未完整提供，无法添加更多注释
    # 确保 `a.shape` 类似于 `irfft` 的 `len(a)`，为此是必要的。
    # 
    # 输入应按照与 `rfftn` 返回相同的顺序排序，
    # 即对于最终转换轴的 `irfft`，以及对于所有其他轴的 `ifftn`。
    # 
    # Parameters
    # ----------
    # a : array_like
    #     输入数组。
    # s : 序列 of ints, 可选
    #     输出的形状（每个转换轴的长度）（`s[0]` 指的是轴 0，`s[1]` 指的是轴 1，等等）。
    #     `s` 也是沿此轴使用的输入点数，除了最后一个轴，
    #     其中使用输入的点数为 `s[-1]//2+1`。
    #     沿任何轴，如果由 `s` 指示的形状比输入小，则会裁剪输入。
    #     如果比输入大，则用零填充输入。
    # 
    #     .. versionchanged:: 2.0
    # 
    #         如果为 `-1`，则使用整个输入（无填充/裁剪）。
    # 
    #     如果未给出 `s`，则使用沿 `axes` 指定的轴的输入形状。
    #     最后一个轴被认为是 `2*(m-1)`，其中 `m` 是沿该轴的输入长度。
    # 
    #     .. deprecated:: 2.0
    # 
    #         如果 `s` 不为 `None`，`axes` 也不能为 `None`。
    # 
    #     .. deprecated:: 2.0
    # 
    #         `s` 必须仅包含 `int` 值，而不是 `None` 值。
    #         `None` 值目前意味着在相应的 1-D 转换中使用 `n` 的默认值，
    #         但此行为已过时。
    # 
    # axes : 序列 of ints, 可选
    #     计算逆 FFT 的轴。如果未给出，则使用最后 `len(s)` 个轴，或者如果 `s` 也未指定，则使用所有轴。
    #     在 `axes` 中重复的索引意味着在该轴上执行多次逆变换。
    # 
    #     .. deprecated:: 2.0
    # 
    #         如果指定了 `s`，则必须显式指定要转换的相应 `axes`。
    # 
    # norm : {"backward", "ortho", "forward"}, 可选
    #     .. versionadded:: 1.10.0
    # 
    #     归一化模式（参见 `numpy.fft`）。默认为 "backward"。
    #     表示前向/后向转换对中哪个方向被缩放以及使用哪个归一化因子。
    # 
    #     .. versionadded:: 1.20.0
    # 
    #         添加了 "backward"、"forward" 值。
    # 
    # out : ndarray, 可选
    #     如果提供，则结果将放置在此数组中。它应该是最后转换的适当形状和 dtype。
    # 
    #     .. versionadded:: 2.0.0
    # 
    # Returns
    # -------
    # 将输入数组 `a` 转换为 NumPy 数组，确保可以进行后续操作
    a = asarray(a)
    
    # 根据输入的参数 `a`, `s`, `axes`，以及 `invreal=1` 的设置，处理并返回 `s` 和 `axes`
    # 这些参数会影响后续的 FFT 变换
    s, axes = _cook_nd_args(a, s, axes, invreal=1)
    
    # 对除最后一个轴外的所有轴进行逆 FFT 变换，使用指定的 `s`、`axes` 和 `norm`
    # 循环次数由 `len(axes)-1` 决定
    for ii in range(len(axes)-1):
        a = ifft(a, s[ii], axes[ii], norm)
    
    # 对最后一个轴进行逆 FFT 变换，使用指定的 `s[-1]`、`axes[-1]` 和 `norm`
    # 输出结果存储在 `out` 中，这里调用了 `irfft` 函数
    a = irfft(a, s[-1], axes[-1], norm, out=out)
    
    # 返回逆 FFT 变换后的结果数组 `a`
    return a
# 用于分派到正确的函数，根据 `_fftn_dispatcher` 分派处理
@array_function_dispatch(_fftn_dispatcher)
# 定义了 `irfft2` 函数，计算二维实数输入的反向傅里叶变换的逆过程
def irfft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    """
    Computes the inverse of `rfft2`.

    Parameters
    ----------
    a : array_like
        The input array
    s : sequence of ints, optional
        Shape of the real output to the inverse FFT.

        .. versionchanged:: 2.0

            If it is ``-1``, the whole input is used (no padding/trimming).

        .. deprecated:: 2.0

            If `s` is not ``None``, `axes` must not be ``None`` either.

        .. deprecated:: 2.0

            `s` must contain only ``int`` s, not ``None`` values. ``None``
            values currently mean that the default value for ``n`` is used
            in the corresponding 1-D transform, but this behaviour is
            deprecated.

    axes : sequence of ints, optional
        The axes over which to compute the inverse fft.
        Default: ``(-2, -1)``, the last two axes.

        .. deprecated:: 2.0

            If `s` is specified, the corresponding `axes` to be transformed
            must not be ``None``.

    norm : {"backward", "ortho", "forward"}, optional
        .. versionadded:: 1.10.0

        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

        .. versionadded:: 1.20.0

            The "backward", "forward" values were added.

    out : ndarray, optional
        If provided, the result will be placed in this array. It should be
        of the appropriate shape and dtype for the last transformation.

        .. versionadded:: 2.0.0

    Returns
    -------
    out : ndarray
        The result of the inverse real 2-D FFT.

    See Also
    --------
    rfft2 : The forward two-dimensional FFT of real input,
            of which `irfft2` is the inverse.
    rfft : The one-dimensional FFT for real input.
    irfft : The inverse of the one-dimensional FFT of real input.
    irfftn : Compute the inverse of the N-dimensional FFT of real input.

    Notes
    -----
    This is really `irfftn` with different defaults.
    For more details see `irfftn`.

    Examples
    --------
    >>> a = np.mgrid[:5, :5][0]
    >>> A = np.fft.rfft2(a)
    >>> np.fft.irfft2(A, s=a.shape)
    array([[0., 0., 0., 0., 0.],
           [1., 1., 1., 1., 1.],
           [2., 2., 2., 2., 2.],
           [3., 3., 3., 3., 3.],
           [4., 4., 4., 4., 4.]])
    """
    # 调用 `irfftn` 函数，计算 N 维实数输入的反向傅里叶变换的逆过程
    return irfftn(a, s, axes, norm, out=None)
```