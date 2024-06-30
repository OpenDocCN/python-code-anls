# `D:\src\scipysrc\scipy\scipy\signal\tests\_scipy_spectral_test_shim.py`

```
"""Helpers to utilize existing stft / istft tests for testing `ShortTimeFFT`.

This module provides the functions stft_compare() and istft_compare(), which,
compares the output between the existing (i)stft() and the shortTimeFFT based
_(i)stft_wrapper() implementations in this module.

For testing add the following imports to the file ``tests/test_spectral.py``::

    from ._scipy_spectral_test_shim import stft_compare as stft
    from ._scipy_spectral_test_shim import istft_compare as istft

and remove the existing imports of stft and istft.

The idea of these wrappers is not to provide a backward-compatible interface
but to demonstrate that the ShortTimeFFT implementation is at least as capable
as the existing one and delivers comparable results. Furthermore, the
wrappers highlight the different philosophies of the implementations,
especially in the border handling.
"""
import platform
from typing import cast, Literal

import numpy as np
from numpy.testing import assert_allclose

from scipy.signal import ShortTimeFFT
from scipy.signal import csd, get_window, stft, istft
from scipy.signal._arraytools import const_ext, even_ext, odd_ext, zero_ext
from scipy.signal._short_time_fft import FFT_MODE_TYPE
from scipy.signal._spectral_py import _spectral_helper, _triage_segments, \
    _median_bias


def _stft_wrapper(x, fs=1.0, window='hann', nperseg=256, noverlap=None,
                  nfft=None, detrend=False, return_onesided=True,
                  boundary='zeros', padded=True, axis=-1, scaling='spectrum'):
    """Wrapper for the SciPy `stft()` function based on `ShortTimeFFT` for
    unit testing.

    Handling the boundary and padding is where `ShortTimeFFT` and `stft()`
    differ in behavior. Parts of `_spectral_helper()` were copied to mimic
    the` stft()` behavior.

    This function is meant to be solely used by `stft_compare()`.

    Parameters:
    - x: array_like
        Input signal array.
    - fs: float, optional
        Sampling frequency of the input signal.
    - window: str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is passed
        to `get_window` to generate the window values, which are then
        normalized.
    - nperseg: int, optional
        Length of each segment.
    - noverlap: int, optional
        Number of points to overlap between segments.
    - nfft: int, optional
        Length of the FFT used, if a zero-padded FFT is desired.
    - detrend: bool, optional
        Specifies whether to detrend each segment.
    - return_onesided: bool, optional
        Whether to return one-sided or two-sided spectrum.
    - boundary: str or None, optional
        Specifies the boundary extension method.
    - padded: bool, optional
        Specifies whether the signal is padded.
    - axis: int, optional
        Specifies the axis along which the STFT is computed.
    - scaling: {'spectrum', 'psd'}, optional
        Specifies the scaling of the spectrum.

    Returns:
    - f: ndarray
        Array of sample frequencies.
    - t: ndarray
        Array of segment times.
    - Zxx: ndarray
        STFT of `x`.

    Raises:
    - ValueError
        If `scaling` is not one of {'spectrum', 'psd'}.
        If `boundary` is not recognized.
        If `nperseg` is less than 1.

    Notes:
    - This function closely mimics the behavior of `stft()` but utilizes
      `ShortTimeFFT` for comparison in unit testing scenarios.
    """
    if scaling not in ('psd', 'spectrum'):  # same errors as in original stft:
        raise ValueError(f"Parameter {scaling=} not in ['spectrum', 'psd']!")

    # The following lines are taken from the original _spectral_helper():
    boundary_funcs = {'even': even_ext,
                      'odd': odd_ext,
                      'constant': const_ext,
                      'zeros': zero_ext,
                      None: None}

    if boundary not in boundary_funcs:
        raise ValueError(f"Unknown boundary option '{boundary}', must be one" +
                         f" of: {list(boundary_funcs.keys())}")
    if x.size == 0:
        return np.empty(x.shape), np.empty(x.shape), np.empty(x.shape)

    if nperseg is not None:  # if specified by user
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError('nperseg must be a positive integer')

    # parse window; if array like, then set nperseg = win.shape
    win, nperseg = _triage_segments(window, nperseg,
                                    input_length=x.shape[axis])

    if nfft is None:
        nfft = nperseg

    # Rest of the function implementation continues here and is not included in this example.
    # 如果 nfft 小于 nperseg，则引发值错误
    elif nfft < nperseg:
        raise ValueError('nfft must be greater than or equal to nperseg.')
    else:
        nfft = int(nfft)

    # 如果没有指定 noverlap，则将其设置为 nperseg 的一半
    if noverlap is None:
        noverlap = nperseg//2
    else:
        noverlap = int(noverlap)

    # 如果 noverlap 大于等于 nperseg，则引发值错误
    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')

    # 计算每步的步长
    nstep = nperseg - noverlap
    n = x.shape[axis]

    # 如果指定了 boundary，则使用相应的边界函数对信号进行扩展
    if boundary is not None:
        ext_func = boundary_funcs[boundary]
        # 在前后各扩展 nperseg//2
        x = ext_func(x, nperseg//2, axis=axis)

    # 如果指定了 padded，则对信号进行填充，使其长度成为窗口段的整数倍
    if padded:
        x = np.moveaxis(x, axis, -1)

        # 处理一个特殊情况，确保 x 的最后一次切片被正确处理
        if n % 2 == 1 and nperseg % 2 == 1 and noverlap % 2 == 1:
            x = x[..., :axis - 1]

        nadd = (-(x.shape[-1]-nperseg) % nstep) % nperseg
        zeros_shape = list(x.shape[:-1]) + [nadd]
        x = np.concatenate((x, np.zeros(zeros_shape)), axis=-1)
        x = np.moveaxis(x, -1, axis)

    # 定义用于缩放的类型，根据 scaling 参数决定是 'magnitude' 还是 'psd'
    scale_to = {'spectrum': 'magnitude', 'psd': 'psd'}[scaling]

    # 如果输入信号是复数且要求返回的结果是单侧的，则将 return_onesided 设置为 False
    if np.iscomplexobj(x) and return_onesided:
        return_onesided = False

    # 根据 return_onesided 的值确定 FFT 的模式
    fft_mode = cast(FFT_MODE_TYPE, 'onesided' if return_onesided else 'twosided')

    # 创建 ShortTimeFFT 对象
    ST = ShortTimeFFT(win, nstep, fs, fft_mode=fft_mode, mfft=nfft,
                      scale_to=scale_to, phase_shift=None)

    # 计算 p0 和 p1 的值，用于确定边界处理的起始和结束位置
    k_off = nperseg // 2
    p0 = 0  # ST.lower_border_end[1] + 1
    nn = x.shape[axis] if padded else n+k_off+1
    p1 = ST.upper_border_begin(nn)[1]  # ST.p_max(n) + 1

    # 处理特定测试 test_roundtrip_boundary_extension() 的边界情况
    if padded is True and nperseg - noverlap == 1:
        p1 -= nperseg // 2 - 1  # the reasoning behind this is not clear to me

    # 根据 detrend 参数对信号进行去趋势处理，计算 STFT
    detr = None if detrend is False else detrend
    Sxx = ST.stft_detrend(x, detr, p0, p1, k_offset=k_off, axis=axis)

    # 计算时间轴 t
    t = ST.t(nn, 0, p1 - p0, k_offset=0 if boundary is not None else k_off)

    # 如果输入信号是 np.float32 或 np.complex64 类型，则将 Sxx 转换为 np.complex64 类型
    if x.dtype in (np.float32, np.complex64):
        Sxx = Sxx.astype(np.complex64)

    # 处理测试 test_average_all_segments() 的边界情况
    if boundary is None and padded is False:
        t, Sxx = t[1:-1], Sxx[..., :-2]
        t -= k_off / fs

    # 返回频率轴 f, 时间轴 t, 和 STFT 结果 Sxx
    return ST.f, t, Sxx
    # *** Lines are taken from _spectral_py.istft() ***:
    # 检查输入的短时傅里叶变换 Zxx 是否至少是二维的
    if Zxx.ndim < 2:
        raise ValueError('Input stft must be at least 2d!')

    # 检查时间轴和频率轴是否指定不同的轴
    if freq_axis == time_axis:
        raise ValueError('Must specify differing time and frequency axes!')

    # 获取时间段数目
    nseg = Zxx.shape[time_axis]

    # 如果输入是单边的，假设段长度为偶数
    if input_onesided:
        n_default = 2*(Zxx.shape[freq_axis] - 1)
    else:
        n_default = Zxx.shape[freq_axis]

    # 检查窗口参数
    if nperseg is None:
        nperseg = n_default
    else:
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError('nperseg must be a positive integer')

    # 如果未指定 nfft，则根据是否单边输入和段长度是否奇数确定 nfft
    if nfft is None:
        if input_onesided and (nperseg == n_default + 1):
            # 奇数长度的段，不进行 FFT 填充
            nfft = nperseg
        else:
            nfft = n_default
    elif nfft < nperseg:
        raise ValueError('nfft must be greater than or equal to nperseg.')
    else:
        nfft = int(nfft)

    # 如果未指定 noverlap，则根据 nperseg 计算其值
    if noverlap is None:
        noverlap = nperseg//2
    else:
        noverlap = int(noverlap)
    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')
    nstep = nperseg - noverlap

    # 获取窗口数组
    if isinstance(window, str) or type(window) is tuple:
        win = get_window(window, nperseg)
    else:
        win = np.asarray(window)
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        if win.shape[0] != nperseg:
            raise ValueError(f'window must have length of {nperseg}')

    # 计算输出长度
    outputlength = nperseg + (nseg-1)*nstep
    # *** End block of: Taken from _spectral_py.istft() ***

    # 使用 cast() 来确保类型兼容性，使 mypy 检查通过
    fft_mode = cast(FFT_MODE_TYPE, 'onesided' if input_onesided else 'twosided')
    # 确定要缩放到的值（幅度谱或功率谱）
    scale_to = cast(Literal['magnitude', 'psd'],
                    {'spectrum': 'magnitude', 'psd': 'psd'}[scaling])

    # 创建 ShortTimeFFT 对象 ST
    ST = ShortTimeFFT(win, nstep, fs, fft_mode=fft_mode, mfft=nfft,
                      scale_to=scale_to, phase_shift=None)

    # 如果 boundary=True
    if boundary:
        # 计算 k0 和 k1
        j = nperseg if nperseg % 2 == 0 else nperseg - 1
        k0 = ST.k_min + nperseg // 2
        k1 = outputlength - j + k0
    else:
        # 如果 boundary=False，则抛出未实现的错误
        raise NotImplementedError("boundary=False does not make sense with" +
                                  "ShortTimeFFT.istft()!")

    # 调用 ShortTimeFFT 对象的 istft 方法，获取时域信号 x
    x = ST.istft(Zxx, k0=k0, k1=k1, f_axis=freq_axis, t_axis=time_axis)
    # 生成一个从 k0 到 k1-1 的整数序列，并乘以常数 ST.T，得到时间数组 t
    t = np.arange(k1 - k0) * ST.T
    
    # 调用 ST 对象的 upper_border_begin 方法，传入 k1 - k0 作为参数，返回一个包含 k_hi 值的元组，取其第一个元素赋给 k_hi
    k_hi = ST.upper_border_begin(k1 - k0)[0]
    
    # 使用 cast() 函数以满足类型检查工具 mypy 的要求，然后将 t, x, (ST.lower_border_end[0], k_hi) 作为结果返回
    return t, x, (ST.lower_border_end[0], k_hi)
# 定义一个名为 _csd_wrapper 的函数，用于封装 csd() 函数的调用，并供单元测试使用
def _csd_wrapper(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None,
                 nfft=None, detrend='constant', return_onesided=True,
                 scaling='density', axis=-1, average='mean'):
    """Wrapper for the `csd()` function based on `ShortTimeFFT` for
        unit testing.
    """
    # 调用 _csd_test_shim() 函数，返回频率(freqs)、时间(t)、交叉功率谱密度(Pxy)结果
    freqs, _, Pxy = _csd_test_shim(x, y, fs, window, nperseg, noverlap, nfft,
                                   detrend, return_onesided, scaling, axis)

    # 以下代码段源自 csd() 函数：
    if len(Pxy.shape) >= 2 and Pxy.size > 0:
        # 如果 Pxy 的维度大于等于2且非空
        if Pxy.shape[-1] > 1:
            # 如果最后一个维度的大小大于1
            if average == 'median':
                # 若 average 参数为 'median'，计算中位数偏差，并分别对实部和虚部进行中位数计算
                bias = _median_bias(Pxy.shape[-1])
                if np.iscomplexobj(Pxy):
                    # 如果 Pxy 是复数对象，则分别对实部和虚部进行中位数计算
                    Pxy = (np.median(np.real(Pxy), axis=-1)
                           + 1j * np.median(np.imag(Pxy), axis=-1))
                else:
                    # 如果 Pxy 是实数对象，则直接计算中位数
                    Pxy = np.median(Pxy, axis=-1)
                # 对 Pxy 进行偏差调整
                Pxy /= bias
            elif average == 'mean':
                # 若 average 参数为 'mean'，则对 Pxy 沿着最后一个维度进行平均
                Pxy = Pxy.mean(axis=-1)
            else:
                # 若 average 参数不是 'median' 或 'mean'，抛出 ValueError 异常
                raise ValueError(f'average must be "median" or "mean", got {average}')
        else:
            # 如果 Pxy 的最后一个维度大小为1，则重新整形 Pxy
            Pxy = np.reshape(Pxy, Pxy.shape[:-1])

    # 返回频率和处理后的 Pxy
    return freqs, Pxy


# 定义一个名为 _csd_test_shim 的函数，用于比较 _spectral_helper() 和 ShortTimeFFT
# 中的 _spect_helper_csd() 在 csd_wrapper() 中的输出是否一致
def _csd_test_shim(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None,
                   nfft=None, detrend='constant', return_onesided=True,
                   scaling='density', axis=-1):
    """Compare output of  _spectral_helper() and ShortTimeFFT, more
    precisely _spect_helper_csd() for used in csd_wrapper().

   The motivation of this function is to test if the ShortTimeFFT-based
   wrapper `_spect_helper_csd()` returns the same values as `_spectral_helper`.
   This function should only be usd by csd() in (unit) testing.
   """
    # 调用 _spectral_helper() 函数获取频率(freqs)、时间(t)、功率谱密度(Pxy)结果
    freqs, t, Pxy = _spectral_helper(x, y, fs, window, nperseg, noverlap, nfft,
                                     detrend, return_onesided, scaling, axis,
                                     mode='psd')
    # 调用 _spect_helper_csd() 函数获取频率(freqs1)和功率谱密度(Pxy1)结果
    freqs1, Pxy1 = _spect_helper_csd(x, y, fs, window, nperseg, noverlap, nfft,
                                     detrend, return_onesided, scaling, axis)

    # 使用 np.testing.assert_allclose() 检查 freqs1 和 freqs 是否在误差范围内相等
    np.testing.assert_allclose(freqs1, freqs)
    # 计算 Pxy 的最大绝对值，并根据需要为大 Pxy 值设定的绝对误差 atol
    amax_Pxy = max(np.abs(Pxy).max(), 1) if Pxy.size else 1
    atol = np.finfo(Pxy.dtype).resolution * amax_Pxy  # needed for large Pxy
    # 使用 np.testing.assert_allclose() 检查 Pxy1 和 Pxy 是否在误差范围内相等
    np.testing.assert_allclose(Pxy1, Pxy, atol=atol)
    # 返回频率、时间和功率谱密度
    return freqs, t, Pxy


# 定义一个名为 _spect_helper_csd 的函数，用 ShortTimeFFT 替代 _spectral_helper()
# 用于 csd() 函数
def _spect_helper_csd(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None,
                      nfft=None, detrend='constant', return_onesided=True,
                      scaling='density', axis=-1):
    """Wrapper for replacing _spectral_helper() by using the ShortTimeFFT
      for use by csd().

    This function should be only used by _csd_test_shim() and is only useful
    """
    # 调用 _spectral_helper() 函数获取频率(freqs)和功率谱密度(Pxy)
    freqs, t, Pxy = _spectral_helper(x, y, fs, window, nperseg, noverlap, nfft,
                                     detrend, return_onesided, scaling, axis,
                                     mode='psd')
    # 返回频率和功率谱密度
    return freqs, Pxy
    for testing the ShortTimeFFT implementation.
    """

    # The following lines are taken from the original _spectral_helper():
    # Check if y is the same object as x
    same_data = y is x
    # Convert axis to integer type
    axis = int(axis)

    # Ensure we have np.arrays, get outdtype
    # Convert x to numpy array
    x = np.asarray(x)
    # If y is not the same object as x, convert y to numpy array as well
    if not same_data:
        y = np.asarray(y)
    #     outdtype = np.result_type(x, y, np.complex64)
    # else:
    #     outdtype = np.result_type(x, np.complex64)

    if not same_data:
        # Check if we can broadcast the outer axes together
        xouter = list(x.shape)
        youter = list(y.shape)
        # Remove the axis dimension from the shape lists
        xouter.pop(axis)
        youter.pop(axis)
        try:
            # Broadcast empty arrays to determine the shape of outer axes
            outershape = np.broadcast(np.empty(xouter), np.empty(youter)).shape
        except ValueError as e:
            # Raise ValueError if x and y cannot be broadcast together
            raise ValueError('x and y cannot be broadcast together.') from e

    if same_data:
        # If x and y are the same object
        if x.size == 0:
            # Return empty arrays with the shape of x if x is empty
            return np.empty(x.shape), np.empty(x.shape)
    else:
        # If x and y are different objects
        if x.size == 0 or y.size == 0:
            # Calculate the shape of the output arrays
            outshape = outershape + (min([x.shape[axis], y.shape[axis]]),)
            # Create empty arrays with the calculated shape
            emptyout = np.moveaxis(np.empty(outshape), -1, axis)
            return emptyout, emptyout

    if nperseg is not None:  # if specified by user
        # Convert nperseg to integer type
        nperseg = int(nperseg)
        if nperseg < 1:
            # Raise ValueError if nperseg is not a positive integer
            raise ValueError('nperseg must be a positive integer')

    # parse window; if array like, then set nperseg = win.shape
    # Determine the length of the signal (x or y)
    n = x.shape[axis] if same_data else max(x.shape[axis], y.shape[axis])
    # Triages the window and nperseg
    win, nperseg = _triage_segments(window, nperseg, input_length=n)

    if nfft is None:
        # If nfft is not specified, set it to nperseg
        nfft = nperseg
    elif nfft < nperseg:
        # Raise ValueError if nfft is less than nperseg
        raise ValueError('nfft must be greater than or equal to nperseg.')
    else:
        # Convert nfft to integer type
        nfft = int(nfft)

    if noverlap is None:
        # If noverlap is not specified, set it to nperseg // 2
        noverlap = nperseg // 2
    else:
        # Convert noverlap to integer type
        noverlap = int(noverlap)
    if noverlap >= nperseg:
        # Raise ValueError if noverlap is not less than nperseg
        raise ValueError('noverlap must be less than nperseg.')
    # Calculate the step size between segments
    nstep = nperseg - noverlap

    if np.iscomplexobj(x) and return_onesided:
        # If x is complex and return_onesided is True, set return_onesided to False
        return_onesided = False

    # using cast() to make mypy happy:
    # Determine the FFT mode based on return_onesided
    fft_mode = cast(FFT_MODE_TYPE, 'onesided' if return_onesided
                    else 'twosided')
    # Determine the scaling method based on the 'scaling' parameter
    scale = {'spectrum': 'magnitude', 'density': 'psd'}[scaling]
    # Initialize ShortTimeFFT object with specified parameters
    SFT = ShortTimeFFT(win, nstep, fs, fft_mode=fft_mode, mfft=nfft,
                       scale_to=scale, phase_shift=None)

    # _spectral_helper() calculates X.conj()*Y instead of X*Y.conj():
    # Calculate the spectrogram and take the conjugate
    Pxy = SFT.spectrogram(y, x, detr=None if detrend is False else detrend,
                          p0=0, p1=(n-noverlap)//SFT.hop, k_offset=nperseg//2,
                          axis=axis).conj()
    # Note:
    # 'onesided2X' scaling of ShortTimeFFT conflicts with the
    # scaling='spectrum' parameter, since it doubles the squared magnitude,
    # which in the view of the ShortTimeFFT implementation does not make sense.
    # Hence, the doubling of the square is implemented here:
    # 如果设置了 return_onesided 参数为 True，则执行以下操作
    f_axis = Pxy.ndim - 1 + axis if axis < 0 else axis
    # 将 Pxy 数组中的一个轴移到最后一个位置
    Pxy = np.moveaxis(Pxy, f_axis, -1)
    # 如果 SFT.mfft 是偶数，对 Pxy 数组的中心部分进行修正
    Pxy[..., 1:-1 if SFT.mfft % 2 == 0 else None] *= 2
    # 将之前移动的轴移回原来的位置
    Pxy = np.moveaxis(Pxy, -1, f_axis)

# 返回 SFT.f 和处理后的 Pxy 数组作为结果
return SFT.f, Pxy
# 对比并确保现有的 `stft()` 和 `_stft_wrapper()` 函数返回的结果非常接近。

"""Assert that the results from the existing `stft()` and `_stft_wrapper()`
are close to each other.

For comparing the STFT values an absolute tolerance of the floating point
resolution was added to circumvent problems with the following tests:
* For float32 the tolerances are much higher in
  TestSTFT.test_roundtrip_float32()).
* The TestSTFT.test_roundtrip_scaling() has a high relative deviation.
  Interestingly this did not appear in Scipy 1.9.1 but only in the current
  development version.
"""

def stft_compare(x, fs=1.0, window='hann', nperseg=256, noverlap=None,
                 nfft=None, detrend=False, return_onesided=True,
                 boundary='zeros', padded=True, axis=-1, scaling='spectrum'):
    # 构建关键字参数字典，传递给 `stft()` 函数和 `_stft_wrapper()` 函数
    kw = dict(x=x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
              nfft=nfft, detrend=detrend, return_onesided=return_onesided,
              boundary=boundary, padded=padded, axis=axis, scaling=scaling)
    # 调用 `stft()` 函数并获取返回的频率、时间和STFT结果
    f, t, Zxx = stft(**kw)
    # 调用 `_stft_wrapper()` 函数并获取返回的频率、时间和STFT结果
    f_wrapper, t_wrapper, Zxx_wrapper = _stft_wrapper(**kw)

    # 构建误差消息的一部分，用于显示在断言失败时的错误消息中
    e_msg_part = " of `stft_wrapper()` differ from `stft()`."
    # 比较频率数组 `f_wrapper` 和 `f`
    assert_allclose(f_wrapper, f, err_msg=f"Frequencies {e_msg_part}")
    # 比较时间数组 `t_wrapper` 和 `t`
    assert_allclose(t_wrapper, t, err_msg=f"Time slices {e_msg_part}")

    # 适应性容差，以解决浮点数精度问题，根据 `Zxx` 数组的数据类型动态计算
    atol = np.finfo(Zxx.dtype).resolution * 2
    # 比较 STFT 结果 `Zxx_wrapper` 和 `Zxx`，使用预设的绝对容差 `atol`
    assert_allclose(Zxx_wrapper, Zxx, atol=atol,
                    err_msg=f"STFT values {e_msg_part}")
    # 返回频率、时间和STFT结果
    return f, t, Zxx



# 对比并确保现有的 `istft()` 和 `_istft_wrapper()` 函数返回的结果非常接近。

"""Assert that the results from the existing `istft()` and
`_istft_wrapper()` are close to each other.

Quirks:
* If ``boundary=False`` the comparison is skipped, since it does not
  make sense with ShortTimeFFT.istft(). Only used in test
  TestSTFT.test_roundtrip_boundary_extension().
* If ShortTimeFFT.istft() decides the STFT is not invertible, the
  comparison is skipped, since istft() only emits a warning and does not
  return a correct result. Only used in
  ShortTimeFFT.test_roundtrip_not_nola().
* For comparing the signals an absolute tolerance of the floating point
  resolution was added to account for the low accuracy of float32 (Occurs
  only in TestSTFT.test_roundtrip_float32()).
"""

def istft_compare(Zxx, fs=1.0, window='hann', nperseg=None, noverlap=None,
                  nfft=None, input_onesided=True, boundary=True, time_axis=-1,
                  freq_axis=-2, scaling='spectrum'):
    # 构建关键字参数字典，传递给 `istft()` 函数和 `_istft_wrapper()` 函数
    kw = dict(Zxx=Zxx, fs=fs, window=window, nperseg=nperseg,
              noverlap=noverlap, nfft=nfft, input_onesided=input_onesided,
              boundary=boundary, time_axis=time_axis, freq_axis=freq_axis,
              scaling=scaling)

    # 调用 `istft()` 函数并获取返回的时间和信号 `t`、`x`
    t, x = istft(**kw)
    # 如果 `boundary` 参数为 `False`，则跳过对比，因为在这种情况下比较没有意义
    if not boundary:
        return t, x  # _istft_wrapper does() not implement this case

    try:
        # 尝试调用 `_istft_wrapper()` 函数并获取返回的时间、信号以及 `(k_lo, k_hi)` 元组
        t_wrapper, x_wrapper, (k_lo, k_hi) = _istft_wrapper(**kw)
    except:
        # 如果逆向转换失败，则跳过比较，因为 `istft()` 只会发出警告并且不会返回正确结果
        pass
    # 如果值错误是因为短时傅里叶变换不可逆，直接返回原始的时间和信号数据
    except ValueError as v:
        if v.args[0] == "Short-time Fourier Transform not invertible!":
            return t, x
        # 如果错误不是由于不可逆的短时傅里叶变换引起的，则抛出原始的值错误异常
        raise v

    # 准备错误消息的一部分，用于断言检查 `istft_wrapper()` 和 `istft()` 的差异
    e_msg_part = " of `istft_wrapper()` differ from `istft()`"
    # 断言检查 `t` 和 `t_wrapper` 是否在允许的误差范围内接近
    assert_allclose(t, t_wrapper, err_msg=f"Sample times {e_msg_part}")

    # 根据信号数据类型动态调整绝对误差容差 `atol`，避免分辨率损失
    atol = np.finfo(x.dtype).resolution * 2  # 使用信号数据类型的分辨率的两倍作为默认绝对误差容差
    rtol = 1e-7  # 默认的相对误差容差，适用于 np.allclose()

    # 在 32 位平台上放宽 `atol` 以通过持续集成测试
    if x.dtype == np.float32 and platform.machine() == 'i686':
        # 对于 float32 类型，使用来自特定测试的容差，以避免持续集成问题
        atol, rtol = 1e-4, 1e-5
    elif platform.machine() in ('aarch64', 'i386', 'i686'):
        # 对于一些特定的 32 位平台，确保 `atol` 至少为 1e-12，以避免过于严格的容差
        atol = max(atol, 1e-12)

    # 断言检查截取的信号数据 `x_wrapper[k_lo:k_hi]` 和 `x[k_lo:k_hi]` 是否在给定的误差范围内接近
    assert_allclose(x_wrapper[k_lo:k_hi], x[k_lo:k_hi], atol=atol, rtol=rtol,
                    err_msg=f"Signal values {e_msg_part}")
    # 返回最终的时间和信号数据
    return t, x
# 定义一个函数，用于比较两个频率信号的交叉功率谱密度（CSD），并断言两种计算方法的结果接近
def csd_compare(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None,
                nfft=None, detrend='constant', return_onesided=True,
                scaling='density', axis=-1, average='mean'):
    """Assert that the results from the existing `csd()` and `_csd_wrapper()`
    are close to each other. """

    # 将传入的参数整理成一个关键字参数字典
    kw = dict(x=x, y=y, fs=fs, window=window, nperseg=nperseg,
              noverlap=noverlap, nfft=nfft, detrend=detrend,
              return_onesided=return_onesided, scaling=scaling, axis=axis,
              average=average)
    
    # 调用 csd() 函数计算频率和交叉功率谱密度
    freqs0, Pxy0 = csd(**kw)
    
    # 调用 _csd_wrapper() 函数计算频率和交叉功率谱密度
    freqs1, Pxy1 = _csd_wrapper(**kw)

    # 使用 assert_allclose 函数断言 freqs1 与 freqs0 的接近程度
    assert_allclose(freqs1, freqs0)
    
    # 使用 assert_allclose 函数断言 Pxy1 与 Pxy0 的接近程度
    assert_allclose(Pxy1, Pxy0)
    
    # 再次使用 assert_allclose 函数断言 freqs1 与 freqs0 的接近程度（这行似乎多余）
    assert_allclose(freqs1, freqs0)
    
    # 返回计算得到的频率和交叉功率谱密度
    return freqs0, Pxy0
```