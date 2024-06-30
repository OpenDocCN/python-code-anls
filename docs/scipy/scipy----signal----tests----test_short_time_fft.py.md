# `D:\src\scipysrc\scipy\scipy\signal\tests\test_short_time_fft.py`

```
"""Unit tests for module `_short_time_fft`.

This file's structure loosely groups the tests into the following sequential
categories:

1. Test function `_calc_dual_canonical_window`.
2. Test for invalid parameters and exceptions in `ShortTimeFFT` (until the
    `test_from_window` function).
3. Test algorithmic properties of STFT/ISTFT. Some tests were ported from
   ``test_spectral.py``.

Notes
-----
* Mypy 0.990 does interpret the line::

        from scipy.stats import norm as normal_distribution

  incorrectly (but the code works), hence a ``type: ignore`` was appended.
"""
# 导入所需的模块和库
import math
from itertools import product
from typing import cast, get_args, Literal

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from scipy.fft import fftshift
from scipy.stats import norm as normal_distribution  # type: ignore
from scipy.signal import get_window, welch, stft, istft, spectrogram

# 导入 `_short_time_fft` 模块中的特定功能和常量
from scipy.signal._short_time_fft import FFT_MODE_TYPE, \
    _calc_dual_canonical_window, ShortTimeFFT, PAD_TYPE
from scipy.signal.windows import gaussian


# 定义测试函数，测试 `_calc_dual_canonical_window` 函数的正确性
def test__calc_dual_canonical_window_roundtrip():
    """Test dual window calculation with a round trip to verify duality.

    Note that this works only for canonical window pairs (having minimal
    energy) like a Gaussian.

    The window is the same as in the example of `from ShortTimeFFT.from_dual`.
    """
    # 创建一个高斯窗口
    win = gaussian(51, std=10, sym=True)
    # 计算对偶窗口
    d_win = _calc_dual_canonical_window(win, 10)
    # 再次计算对偶窗口，验证计算的正确性
    win2 = _calc_dual_canonical_window(d_win, 10)
    # 断言两个窗口的值非常接近，以验证窗口对偶性
    assert_allclose(win2, win)


# 定义测试函数，测试 `_calc_dual_canonical_window` 函数中的异常情况
def test__calc_dual_canonical_window_exceptions():
    """Raise all exceptions in `_calc_dual_canonical_window`."""
    # 验证计算可能失败的情况:
    with pytest.raises(ValueError, match="hop=5 is larger than window len.*"):
        _calc_dual_canonical_window(np.ones(4), 5)
    with pytest.raises(ValueError, match=".* Transform not invertible!"):
        _calc_dual_canonical_window(np.array([.1, .2, .3, 0]), 4)

    # 验证参数 `win` 不能为整数的情况:
    with pytest.raises(ValueError, match="Parameter 'win' cannot be of int.*"):
        _calc_dual_canonical_window(np.ones(4, dtype=int), 1)


# 定义测试函数，验证在实例化 ShortTimeFFT 时参数的有效性
def test_invalid_initializer_parameters():
    """Verify that exceptions get raised on invalid parameters when
    instantiating ShortTimeFFT. """
    with pytest.raises(ValueError, match=r"Parameter win must be 1d, " +
                                         r"but win.shape=\(2, 2\)!"):
        ShortTimeFFT(np.ones((2, 2)), hop=4, fs=1)
    with pytest.raises(ValueError, match="Parameter win must have " +
                                         "finite entries"):
        ShortTimeFFT(np.array([1, np.inf, 2, 3]), hop=4, fs=1)
    with pytest.raises(ValueError, match="Parameter hop=0 is not " +
                                         "an integer >= 1!"):
        ShortTimeFFT(np.ones(4), hop=0, fs=1)
    # 使用 pytest 模块来验证是否会抛出 ValueError 异常，并且匹配特定的错误信息提示
    with pytest.raises(ValueError, match="Parameter hop=2.0 is not " +
                                         "an integer >= 1!"):
        # 忽略 PyTypeChecker 的警告，因为 hop 参数应为整数而不是浮点数
        # 初始化 ShortTimeFFT 对象，传入长度为 4 的全一数组作为输入信号
        ShortTimeFFT(np.ones(4), hop=2.0, fs=1)
    # 使用 pytest 模块来验证是否会抛出 ValueError 异常，并且匹配特定的错误信息提示
    with pytest.raises(ValueError, match=r"dual_win.shape=\(5,\) must equal " +
                                         r"win.shape=\(4,\)!"):
        # 初始化 ShortTimeFFT 对象，传入长度为 4 的全一数组作为输入信号
        # 同时传入长度为 5 的全一数组作为双窗口参数，但双窗口参数应该与窗口参数的长度相同
        ShortTimeFFT(np.ones(4), hop=2, fs=1, dual_win=np.ones(5))
    # 使用 pytest 模块来验证是否会抛出 ValueError 异常，并且匹配特定的错误信息提示
    with pytest.raises(ValueError, match="Parameter dual_win must be " +
                                         "a finite array!"):
        # 初始化 ShortTimeFFT 对象，传入长度为 3 的全一数组作为输入信号
        # 同时传入包含 NaN 值的数组作为双窗口参数，双窗口参数应该是一个有限数组
        ShortTimeFFT(np.ones(3), hop=2, fs=1,
                     dual_win=np.array([np.nan, 2, 3]))
def test_exceptions_properties_methods():
    """验证在设置属性或调用 ShortTimeFFT 的方法时，当使用无效值时会引发异常。"""
    # 创建 ShortTimeFFT 实例，初始化参数为长度为8的全1数组，hop=4，fs=1
    SFT = ShortTimeFFT(np.ones(8), hop=4, fs=1)

    # 测试设置属性 T 为 -1 是否会引发 ValueError 异常
    with pytest.raises(ValueError, match="Sampling interval T=-1 must be " +
                                         "positive!"):
        SFT.T = -1

    # 测试设置属性 fs 为 -1 是否会引发 ValueError 异常
    with pytest.raises(ValueError, match="Sampling frequency fs=-1 must be " +
                                         "positive!"):
        SFT.fs = -1

    # 测试设置属性 fft_mode 为 'invalid_typ' 是否会引发 ValueError 异常
    with pytest.raises(ValueError, match="fft_mode='invalid_typ' not in " +
                                         r"\('twosided', 'centered', " +
                                         r"'onesided', 'onesided2X'\)!"):
        SFT.fft_mode = 'invalid_typ'

    # 测试设置属性 fft_mode 为 'onesided2X' 是否会引发 ValueError 异常
    with pytest.raises(ValueError, match="For scaling is None, " +
                                         "fft_mode='onesided2X' is invalid.*"):
        SFT.fft_mode = 'onesided2X'

    # 测试设置属性 mfft 为 7 是否会引发 ValueError 异常
    with pytest.raises(ValueError, match="Attribute mfft=7 needs to be " +
                                         "at least the window length.*"):
        SFT.mfft = 7

    # 测试调用方法 scale_to('invalid') 是否会引发 ValueError 异常
    with pytest.raises(ValueError, match="scaling='invalid' not in.*"):
        SFT.scale_to('invalid')

    # 测试设置属性 phase_shift 为 3.0 是否会引发 ValueError 异常
    with pytest.raises(ValueError, match="phase_shift=3.0 has the unit .*"):
        SFT.phase_shift = 3.0

    # 测试设置属性 phase_shift 为 2*SFT.mfft 是否会引发 ValueError 异常
    with pytest.raises(ValueError, match="-mfft < phase_shift < mfft " +
                                         "does not hold.*"):
        SFT.phase_shift = 2*SFT.mfft

    # 测试调用方法 _x_slices(...) 时，传入参数 padding='invalid' 是否会引发 ValueError 异常
    with pytest.raises(ValueError, match="Parameter padding='invalid' not.*"):
        g = SFT._x_slices(np.zeros(16), k_off=0, p0=0, p1=1, padding='invalid')
        next(g)  # execute generator

    # 测试调用方法 stft_detrend(...) 时，传入参数 detr='invalid' 是否会引发 ValueError 异常
    with pytest.raises(ValueError, match="Trend type must be 'linear' " +
                                         "or 'constant'"):
        SFT.stft_detrend(np.zeros(16), detr='invalid')

    # 测试调用方法 stft_detrend(...) 时，传入参数 detr=np.nan 是否会引发 ValueError 异常
    with pytest.raises(ValueError, match="Parameter detr=nan is not a str, " +
                                         "function or None!"):
        SFT.stft_detrend(np.zeros(16), detr=np.nan)

    # 测试调用方法 p_range(...) 时，传入参数 p0=0, p1=200 是否会引发 ValueError 异常
    with pytest.raises(ValueError, match="Invalid Parameter p0=0, p1=200.*"):
        SFT.p_range(100, 0, 200)

    # 测试调用方法 istft(...) 时，传入参数 t_axis=0, f_axis=0 是否会引发 ValueError 异常
    with pytest.raises(ValueError, match="f_axis=0 may not be equal to " +
                                         "t_axis=0!"):
        SFT.istft(np.zeros((SFT.f_pts, 2)), t_axis=0, f_axis=0)

    # 测试调用方法 istft(...) 时，传入参数 S.shape[t_axis]=1 是否会引发 ValueError 异常
    with pytest.raises(ValueError, match=r"S.shape\[t_axis\]=1 needs to have" +
                                         " at least 2 slices.*"):
        SFT.istft(np.zeros((SFT.f_pts, 1)))

    # 测试调用方法 istft(...) 时，传入参数 S.shape[f_axis]=2 是否会引发 ValueError 异常
    with pytest.raises(ValueError, match=r"S.shape\[f_axis\]=2 must be equal" +
                                         " to self.f_pts=5.*"):
        SFT.istft(np.zeros((2, 2)))
    # 使用 pytest 检测是否会抛出 ValueError 异常，并匹配特定的错误消息格式
    with pytest.raises(ValueError, match=r".*\(k1=100\) <= \(k_max=12\) " +
                                         "is false!$"):
        # 调用 SFT.istft 函数并传入指定参数，期望抛出 ValueError 异常
        SFT.istft(np.zeros((SFT.f_pts, 3)), k1=100)

    # 使用 pytest 检测是否会抛出 ValueError 异常，并匹配特定的错误消息格式
    with pytest.raises(ValueError, match=r"\(k1=1\) - \(k0=0\) = 1 has to " +
                                         "be at least.* length 4!"):
        # 调用 SFT.istft 函数并传入指定参数，期望抛出 ValueError 异常
        SFT.istft(np.zeros((SFT.f_pts, 3)), k0=0, k1=1)

    # 使用 pytest 检测是否会抛出 ValueError 异常，并匹配特定的错误消息格式
    with pytest.raises(ValueError, match=r"Parameter axes_seq='invalid' " +
                                         r"not in \['tf', 'ft'\]!"):
        # 调用 SFT.extent 函数并传入指定参数，期望抛出 ValueError 异常
        SFT.extent(n=100, axes_seq='invalid')
    
    # 使用 pytest 检测是否会抛出 ValueError 异常，并匹配特定的错误消息格式
    with pytest.raises(ValueError, match="Attribute fft_mode=twosided must.*"):
        # 修改 SFT 对象的 fft_mode 属性
        SFT.fft_mode = 'twosided'
        # 调用 SFT.extent 函数并传入指定参数，期望抛出 ValueError 异常
        SFT.extent(n=100)
# 使用 pytest 的参数化功能，依次对 'onesided' 和 'onesided2X' 两个参数进行测试
@pytest.mark.parametrize('m', ('onesided', 'onesided2X'))
def test_exceptions_fft_mode_complex_win(m: FFT_MODE_TYPE):
    """Verify that one-sided spectra are not allowed with complex-valued
    windows or with complex-valued signals.

    The reason being, the `rfft` function only accepts real-valued input.
    """

    # 测试 ShortTimeFFT 对象在给定复数输入时是否抛出 ValueError 异常
    with pytest.raises(ValueError,
                       match=f"One-sided spectra, i.e., fft_mode='{m}'.*"):
        ShortTimeFFT(np.ones(8)*1j, hop=4, fs=1, fft_mode=m)

    # 创建 ShortTimeFFT 对象并更改 fft_mode 属性，预期抛出 ValueError 异常
    SFT = ShortTimeFFT(np.ones(8)*1j, hop=4, fs=1, fft_mode='twosided')
    with pytest.raises(ValueError,
                       match=f"One-sided spectra, i.e., fft_mode='{m}'.*"):
        SFT.fft_mode = m

    # 创建 ShortTimeFFT 对象并指定 fft_mode='onesided'，预期抛出 ValueError 异常
    SFT = ShortTimeFFT(np.ones(8), hop=4, fs=1, scale_to='psd', fft_mode='onesided')
    with pytest.raises(ValueError, match="Complex-valued `x` not allowed for self.*"):
        SFT.stft(np.ones(8)*1j)

    # 更改 fft_mode 属性为 'onesided2X'，预期抛出 ValueError 异常
    SFT.fft_mode = 'onesided2X'
    with pytest.raises(ValueError, match="Complex-valued `x` not allowed for self.*"):
        SFT.stft(np.ones(8)*1j)


# 测试当 fft_mode 属性设置为无效值时是否引发 RuntimeError 异常
def test_invalid_fft_mode_RuntimeError():
    """Ensure exception gets raised when property `fft_mode` is invalid. """
    # 创建 ShortTimeFFT 对象并将 _fft_mode 属性设置为无效值 'invalid_typ'
    SFT = ShortTimeFFT(np.ones(8), hop=4, fs=1)
    SFT._fft_mode = 'invalid_typ'

    # 测试访问 f 属性时是否引发 RuntimeError 异常
    with pytest.raises(RuntimeError):
        _ = SFT.f

    # 测试调用 _fft_func 方法时是否引发 RuntimeError 异常
    with pytest.raises(RuntimeError):
        SFT._fft_func(np.ones(8))

    # 测试调用 _ifft_func 方法时是否引发 RuntimeError 异常
    with pytest.raises(RuntimeError):
        SFT._ifft_func(np.ones(8))


# 使用 pytest 的参数化功能，依次测试不同窗函数参数的处理
@pytest.mark.parametrize('win_params, Nx', [(('gaussian', 2.), 9),  # in docstr
                                            ('triang', 7),
                                            (('kaiser', 4.0), 9),
                                            (('exponential', None, 1.), 9),
                                            (4.0, 9)])
def test_from_window(win_params, Nx: int):
    """Verify that `from_window()` handles parameters correctly.

    The window parameterizations are documented in the `get_window` docstring.
    """

    # 根据 win_params 参数获取对称和周期的窗函数及其长度
    w_sym, fs = get_window(win_params, Nx, fftbins=False), 16.
    w_per = get_window(win_params, Nx, fftbins=True)

    # 创建 ShortTimeFFT 对象 SFT0，测试其从给定窗函数参数创建时的行为
    SFT0 = ShortTimeFFT(w_sym, hop=3, fs=fs, fft_mode='twosided',
                        scale_to='psd', phase_shift=1)

    # 计算窗口长度
    nperseg = len(w_sym)
    noverlap = nperseg - SFT0.hop

    # 使用 from_window 方法创建 ShortTimeFFT 对象 SFT1，测试对称窗口的处理
    SFT1 = ShortTimeFFT.from_window(win_params, fs, nperseg, noverlap,
                                    symmetric_win=True, fft_mode='twosided',
                                    scale_to='psd', phase_shift=1)

    # 使用 from_window 方法创建 ShortTimeFFT 对象 SFT2，测试周期窗口的处理
    SFT2 = ShortTimeFFT.from_window(win_params, fs, nperseg, noverlap,
                                    symmetric_win=False, fft_mode='twosided',
                                    scale_to='psd', phase_shift=1)

    # 在比较实例时提供详细信息，确保窗口处理的正确性
    assert_equal(SFT1.win, SFT0.win)
    assert_allclose(SFT2.win, w_per / np.sqrt(sum(w_per**2) * fs))
    # 对于给定的参数列表，依次获取 SFT0, SFT1, SFT2 中每个对象的相应属性值
    for n_ in ('hop', 'T', 'fft_mode', 'mfft', 'scaling', 'phase_shift'):
        # 从 SFT0, SFT1, SFT2 中获取属性 n_ 的值
        v0, v1, v2 = (getattr(SFT_, n_) for SFT_ in (SFT0, SFT1, SFT2))
        # 断言 SFT1 和 SFT0 的属性值相等，如果不相等则触发断言异常
        assert v1 == v0, f"SFT1.{n_}={v1} does not equal SFT0.{n_}={v0}"
        # 断言 SFT2 和 SFT0 的属性值相等，如果不相等则触发断言异常
        assert v2 == v0, f"SFT2.{n_}={v2} does not equal SFT0.{n_}={v0}"
# 验证 `win` 和 `dual_win` 的对偶性。
# 
# 注意，此测试并不适用于任意窗口，因为对偶窗口并不唯一。它仅在非重叠的情况下适用于可逆的短时傅里叶变换。
def test_dual_win_roundtrip():
    """Verify the duality of `win` and `dual_win`.

    Note that this test does not work for arbitrary windows, since dual windows
    are not unique. It always works for invertible STFTs if the windows do not
    overlap.
    """
    # 使用非标准关键字参数值（除了 `scale_to`）：
    kw = dict(hop=4, fs=1, fft_mode='twosided', mfft=8, scale_to=None,
              phase_shift=2)
    # 创建 ShortTimeFFT 对象 SFT0，使用全1数组作为参数，并传入关键字参数
    SFT0 = ShortTimeFFT(np.ones(4), **kw)
    # 使用 SFT0 的对偶窗口创建 ShortTimeFFT 对象 SFT1，传入相同的关键字参数
    SFT1 = ShortTimeFFT.from_dual(SFT0.dual_win, **kw)
    # 断言 SFT1 的对偶窗口与 SFT0 的窗口相等
    assert_allclose(SFT1.dual_win, SFT0.win)


@pytest.mark.parametrize('scale_to, fac_psd, fac_mag',
                         [(None, 0.25, 0.125),
                          ('magnitude', 2.0, 1),
                          ('psd', 1, 0.5)])
def test_scaling(scale_to: Literal['magnitude', 'psd'], fac_psd, fac_mag):
    """Verify scaling calculations.

    * Verify passing `scale_to` parameter to `__init__()`.
    * Roundtrip while changing scaling factor.
    """
    # 创建 ShortTimeFFT 对象 SFT，使用全2的数组作为参数，并传入关键字参数
    SFT = ShortTimeFFT(np.ones(4) * 2, hop=4, fs=1, scale_to=scale_to)
    # 断言 SFT 的频谱密度因子等于 fac_psd
    assert SFT.fac_psd == fac_psd
    # 断言 SFT 的幅度因子等于 fac_mag
    assert SFT.fac_magnitude == fac_mag
    # 增加覆盖范围，两次访问属性：
    assert SFT.fac_psd == fac_psd
    assert SFT.fac_magnitude == fac_mag

    x = np.fft.irfft([0, 0, 7, 0, 0, 0, 0])  # 周期信号
    Sx = SFT.stft(x)
    Sx_mag, Sx_psd = Sx * SFT.fac_magnitude, Sx * SFT.fac_psd

    # 将 `scale_to` 设置为 'magnitude'
    SFT.scale_to('magnitude')
    # 对 Sx_mag 进行反向短时傅里叶变换，应当接近原始信号 x
    x_mag = SFT.istft(Sx_mag, k1=len(x))
    assert_allclose(x_mag, x)

    # 将 `scale_to` 设置为 'psd'
    SFT.scale_to('psd')
    # 对 Sx_psd 进行反向短时傅里叶变换，应当接近原始信号 x
    x_psd = SFT.istft(Sx_psd, k1=len(x))
    assert_allclose(x_psd, x)


def test_scale_to():
    """Verify `scale_to()` method."""
    # 创建 ShortTimeFFT 对象 SFT，使用全2的数组作为参数，并传入关键字参数
    SFT = ShortTimeFFT(np.ones(4) * 2, hop=4, fs=1, scale_to=None)

    # 将 `scale_to` 设置为 'magnitude'，并断言相关属性
    SFT.scale_to('magnitude')
    assert SFT.scaling == 'magnitude'
    assert SFT.fac_psd == 2.0
    assert SFT.fac_magnitude == 1

    # 将 `scale_to` 设置为 'psd'，并断言相关属性
    SFT.scale_to('psd')
    assert SFT.scaling == 'psd'
    assert SFT.fac_psd == 1
    assert SFT.fac_magnitude == 0.5

    # 再次将 `scale_to` 设置为 'psd'，以增加覆盖范围
    SFT.scale_to('psd')

    # 对于每个 scale 和 s_fac 的组合，分别创建 ShortTimeFFT 对象 SFT，并检查对偶窗口是否按预期缩放
    for scale, s_fac in zip(('magnitude', 'psd'), (8, 4)):
        SFT = ShortTimeFFT(np.ones(4) * 2, hop=4, fs=1, scale_to=None)
        dual_win = SFT.dual_win.copy()

        SFT.scale_to(cast(Literal['magnitude', 'psd'], scale))
        assert_allclose(SFT.dual_win, dual_win * s_fac)


def test_x_slices_padding():
    """Verify padding.

    The reference arrays were taken from the docstrings of `zero_ext`,
    `const_ext`, `odd_ext()`, and `even_ext()` from the _array_tools module.
    """
    # 创建 ShortTimeFFT 对象 SFT，使用全1的数组作为参数，并传入关键字参数
    SFT = ShortTimeFFT(np.ones(5), hop=4, fs=1)
    x = np.array([[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]], dtype=float)
    d = {'zeros': [[[0, 0, 1, 2, 3], [0, 0, 0, 1, 4]],
                   [[3, 4, 5, 0, 0], [4, 9, 16, 0, 0]]],
         'edge': [[[1, 1, 1, 2, 3], [0, 0, 0, 1, 4]],
                  [[3, 4, 5, 5, 5], [4, 9, 16, 16, 16]]],
         'even': [[[3, 2, 1, 2, 3], [4, 1, 0, 1, 4]],
                  [[3, 4, 5, 4, 3], [4, 9, 16, 9, 4]]],
         'odd': [[[-1, 0, 1, 2, 3], [-4, -1, 0, 1, 4]],
                 [[3, 4, 5, 6, 7], [4, 9, 16, 23, 28]]]}
    # 遍历字典中的每个键值对，键为填充类型，值为预期结果列表
    for p_, xx in d.items():
        # 使用 SFT 类的 _x_slices 方法生成填充后的结果
        gen = SFT._x_slices(np.array(x), 0, 0, 2, padding=cast(PAD_TYPE, p_))
        # 将生成器 gen 的内容复制为 numpy 数组 yy，由于是原地复制，所以使用 y_.copy()
        yy = np.array([y_.copy() for y_ in gen])  # due to inplace copying
        # 断言生成的填充结果 yy 与预期的 xx 相等，如果不相等则抛出错误信息
        assert_equal(yy, xx, err_msg=f"Failed '{p_}' padding.")
def test_in`
def test_invertible():
    """Verify `invertible` property. """
    # 创建一个 ShortTimeFFT 对象，使用一个长度为 8 的全 1 数组，步长为 4，采样频率为 1
    SFT = ShortTimeFFT(np.ones(8), hop=4, fs=1)
    # 检查 SFT 是否是可逆的
    assert SFT.invertible
    # 创建另一个 ShortTimeFFT 对象，使用一个长度为 8 的全 1 数组，步长为 9，采样频率为 1
    SFT = ShortTimeFFT(np.ones(8), hop=9, fs=1)
    # 检查 SFT 是否不可逆
    assert not SFT.invertible


def test_border_values():
    """Ensure that minimum and maximum values of slices are correct."""
    # 创建一个 ShortTimeFFT 对象，使用一个长度为 8 的全 1 数组，步长为 4，采样频率为 1
    SFT = ShortTimeFFT(np.ones(8), hop=4, fs=1)
    # 检查 p_min 是否等于 0
    assert SFT.p_min == 0
    # 检查 k_min 是否等于 -4
    assert SFT.k_min == -4
    # 检查 lower_border_end 是否等于 (4, 1)
    assert SFT.lower_border_end == (4, 1)
    # 再次检查 lower_border_end 是否等于 (4, 1)，用于测试缓存
    assert SFT.lower_border_end == (4, 1)
    # 检查 p_max(10) 是否等于 4
    assert SFT.p_max(10) == 4
    # 检查 k_max(10) 是否等于 16
    assert SFT.k_max(10) == 16
    # 检查 upper_border_begin(10) 是否等于 (4, 2)
    assert SFT.upper_border_begin(10) == (4, 2)


def test_border_values_exotic():
    """Ensure that the border calculations are correct for windows with zeros. """
    # 创建一个包含一个非零值的数组，作为窗口
    w = np.array([0, 0, 0, 0, 0, 0, 0, 1.])
    # 创建一个 ShortTimeFFT 对象，使用上述窗口，步长为 1，采样频率为 1
    SFT = ShortTimeFFT(w, hop=1, fs=1)
    # 检查 lower_border_end 是否等于 (0, 0)
    assert SFT.lower_border_end == (0, 0)

    # 创建一个 ShortTimeFFT 对象，使用翻转的窗口，步长为 20，采样频率为 1
    SFT = ShortTimeFFT(np.flip(w), hop=20, fs=1)
    # 检查 upper_border_begin(4) 是否等于 (0, 0)
    assert SFT.upper_border_begin(4) == (0, 0)

    # 设置步长为 -1，触发无法到达的行
    SFT._hop = -1
    # 检查 k_max(4) 是否会抛出 RuntimeError
    with pytest.raises(RuntimeError):
        _ = SFT.k_max(4)
    # 检查 k_min 是否会抛出 RuntimeError
    with pytest.raises(RuntimeError):
        _ = SFT.k_min


def test_t():
    """Verify that the times of the slices are correct. """
    # 创建一个 ShortTimeFFT 对象，使用一个长度为 8 的全 1 数组，步长为 4，采样频率为 2
    SFT = ShortTimeFFT(np.ones(8), hop=4, fs=2)
    # 检查 T 是否等于 1/2
    assert SFT.T == 1/2
    # 检查采样频率 fs 是否等于 2
    assert SFT.fs == 2.
    # 检查 delta_t 是否等于 4 * 1/2
    assert SFT.delta_t == 4 * 1/2
    # 生成时间轴，范围为 0 到 p_max(10)，步长为 delta_t
    t_stft = np.arange(0, SFT.p_max(10)) * SFT.delta_t
    # 检查 t(10) 是否等于 t_stft
    assert_equal(SFT.t(10), t_stft)
    # 检查 t(10, 1, 3) 是否等于 t_stft[1:3]
    assert_equal(SFT.t(10, 1, 3), t_stft[1:3])
    # 设置 T 为 1/4
    SFT.T = 1/4
    # 检查 T 是否等于 1/4
    assert SFT.T == 1/4
    # 检查采样频率 fs 是否等于 4
    assert SFT.fs == 4
    # 设置采样频率 fs 为 1/8
    SFT.fs = 1/8
    # 检查采样频率 fs 是否等于 1/8
    assert SFT.fs == 1/8
    # 检查 T 是否等于 8
    assert SFT.T == 8


@pytest.mark.parametrize('fft_mode, f',
                         [('onesided', [0., 1., 2.]),
                          ('onesided2X', [0., 1., 2.]),
                          ('twosided', [0., 1., 2., -2., -1.]),
                          ('centered', [-2., -1., 0., 1., 2.])])
def test_f(fft_mode: FFT_MODE_TYPE, f):
    """Verify the frequency values property `f`."""
    # 创建一个 ShortTimeFFT 对象，使用一个长度为 5 的全 1 数组，步长为 4，采样频率为 5，fft_mode 为 fft_mode，scale_to 为 'psd'
    SFT = ShortTimeFFT(np.ones(5), hop=4, fs=5, fft_mode=fft_mode,
                       scale_to='psd')
    # 检查 f 是否等于 f
    assert_equal(SFT.f, f)


def test_extent():
    """Ensure that the `extent()` method is correct. """
    # 创建一个 ShortTimeFFT 对象，使用一个长度为 32 的全 1 数组，步长为 4，采样频率为 32，fft_mode 为 'onesided'
    SFT = ShortTimeFFT(np.ones(32), hop=4, fs=32, fft_mode='onesided')
    # 检查 extent(100, 'tf', False) 是否等于 (-0.375, 3.625, 0.0, 17.0)
    assert SFT.extent(100, 'tf', False) == (-0.375, 3.625, 0.0, 17.0)
    # 检查 extent(100, 'ft', False) 是否等于 (0.0, 17.0, -0.375, 3.625)
    assert SFT.extent(100, 'ft', False) == (0.0, 17.0, -0.375, 3.625)
    # 检查 extent(100, 'tf', True) 是否等于 (-0.4375, 3.5625, -0.5, 16.5)
    assert SFT.extent(100, 'tf', True) == (-0.4375, 3.5625, -0.5, 16.5)
    # 检查 extent(100, 'ft', True) 是否等于 (-0.5, 16.5, -0.4375, 3.5625)
    assert SFT.extent(100, 'ft', True) == (-0.5, 16.5, -0.4375, 3.5625)

    # 创建一个 ShortTimeFFT 对象，使用一个长度为 32 的全 1 数组，步长为 4，采样频率为 32，fft_mode 为 'centered'
    SFT = ShortTimeFFT(np.ones(32), hop=4, fs=32, fft_mode='centered')
    # 检查 extent(100, 'tf', False) 是否等于 (-0.375, 3.625, -16.0, 15.0)
    assert SFT.extent(100, 'tf', False) == (-0.375, 3.625, -16.0, 15.0)


def test_spectrogram():
    """Verify spectrogram and cross-spectrogram methods. """
    # 创建一个 ShortTimeFFT 对象，使用一个长度为 8 的全 1 数组，步长为 4，采样频率为 1
    SFT = ShortTimeFFT(np.ones(8), hop=4, fs=1)
    # 创建一个长度为 10 的全 1 数组 x 和一个范围为 0 到 9 的数组 y
    x, y```
def test_invertible():
    """Verify `invertible` property. """
    # 创建 ShortTimeFFT 对象，使用全 1 的输入信号，设置 hop=4，采样率 fs=1
    SFT = ShortTimeFFT(np.ones(8), hop=4, fs=1)
    # 断言 ShortTimeFFT 对象的 invertible 属性为 True
    assert SFT.invertible
    # 创建另一个 ShortTimeFFT 对象，使用全 1 的输入信号，设置 hop=9，采样率 fs=1
    SFT = ShortTimeFFT(np.ones(8), hop=9, fs=1)
    # 断言 ShortTimeFFT 对象的 invertible 属性为 False
    assert not SFT.invertible


def test_border_values():
    """Ensure that minimum and maximum values of slices are correct."""
    # 创建 ShortTimeFFT 对象，使用全 1 的输入信号，设置 hop=4，采样率 fs=1
    SFT = ShortTimeFFT(np.ones(8), hop=4, fs=1)
    # 断言 ShortTimeFFT 对象的 p_min 属性为 0
    assert SFT.p_min == 0
    # 断言 ShortTimeFFT 对象的 k_min 属性为 -4
    assert SFT.k_min == -4
    # 断言 ShortTimeFFT 对象的 lower_border_end 属性为 (4, 1)
    assert SFT.lower_border_end == (4, 1)
    # 用于测试缓存，再次断言 ShortTimeFFT 对象的 lower_border_end 属性为 (4, 1)
    assert SFT.lower_border_end == (4, 1)
    # 断言 ShortTimeFFT 对象调用 p_max(10) 方法返回 4
    assert SFT.p_max(10) == 4
    # 断言 ShortTimeFFT 对象调用 k_max(10) 方法返回 16
    assert SFT.k_max(10) == 16
    # 断言 ShortTimeFFT 对象调用 upper_border_begin(10) 方法返回 (4, 2)
    assert SFT.upper_border_begin(10) == (4, 2)


def test_border_values_exotic():
    """Ensure that the border calculations are correct for windows with
    zeros. """
    # 创建窗口 w，包含 [0, 0, 0, 0, 0, 0, 0, 1.] 的数组
    w = np.array([0, 0, 0, 0, 0, 0, 0, 1.])
    # 创建 ShortTimeFFT 对象，使用窗口 w，设置 hop=1，采样率 fs=1
    SFT = ShortTimeFFT(w, hop=1, fs=1)
    # 断言 ShortTimeFFT 对象的 lower_border_end 属性为 (0, 0)
    assert SFT.lower_border_end == (0, 0)

    # 创建 ShortTimeFFT 对象，使用 np.flip(w) 反转后的窗口，设置 hop=20，采样率 fs=1
    SFT = ShortTimeFFT(np```
def test_invertible():
    """Verify `invertible` property. """
    # 创建 ShortTimeFFT 对象，使用全 1 的输入信号，设置 hop=4，采样率 fs=1
    SFT = ShortTimeFFT(np.ones(8), hop=4, fs=1)
    # 断言 ShortTimeFFT 对象的 invertible 属性为 True
    assert SFT.invertible
    # 创建另一个 ShortTimeFFT 对象，使用全 1 的输入信号，设置 hop=9，采样率 fs=1
    SFT = ShortTimeFFT(np.ones(8), hop=9, fs=1)
    # 断言 ShortTimeFFT 对象的 invertible 属性为 False
    assert not SFT.invertible


def test_border_values():
    """Ensure that minimum and maximum values of slices are correct."""
    # 创建 ShortTimeFFT 对象，使用全 1 的输入信号，设置 hop=4，采样率 fs=1
    SFT = ShortTimeFFT(np.ones(8), hop=4, fs=1)
    # 断言 ShortTimeFFT 对象的 p_min 属性为 0
    assert SFT.p_min == 0
    # 断言 ShortTimeFFT 对象的 k_min 属性为 -4
    assert SFT.k_min == -4
    # 断言 ShortTimeFFT 对象的 lower_border_end 属性为 (4, 1)
    assert SFT.lower_border_end == (4, 1)
    # 用于测试缓存，再次断言 ShortTimeFFT 对象的 lower_border_end 属性为 (4, 1)
    assert SFT.lower_border_end == (4, 1)
    # 断言 ShortTimeFFT 对象调用 p_max(10) 方法返回 4
    assert SFT.p_max(10) == 4
    # 断言 ShortTimeFFT 对象调用 k_max(10) 方法返回 16
    assert SFT.k_max(10) == 16
    # 断言 ShortTimeFFT 对象调用 upper_border_begin(10) 方法返回 (4, 2)
    assert SFT.upper_border_begin(10) == (4, 2)


def test_border_values_exotic():
    """Ensure that the border calculations are correct for windows with
    zeros. """
    # 创建窗口 w，包含 [0, 0, 0, 0, 0, 0, 0, 1.] 的数组
    w = np.array([0, 0, 0, 0, 0, 0, 0, 1.])
    # 创建 ShortTimeFFT 对象，使用窗口 w，设置 hop=1，采样率 fs=1
    SFT = ShortTimeFFT(w, hop=1, fs=1)
    # 断言 ShortTimeFFT 对象的 lower_border_end 属性为 (0, 0)
    assert SFT.lower_border_end == (0, 0)

    # 创建 ShortTimeFFT 对象，使用 np.flip(w) 反转后的窗口，设置 hop=20，采样率 fs=1
    SFT = ShortTimeFFT(np.flip(w), hop=20, fs=1)
    # 断言 ShortTimeFFT 对象调用 upper_border_begin(4) 方法返回 (0, 0)
    assert SFT.upper_border_begin(4) == (0, 0)

    # 设置 ShortTimeFFT 对象的 _hop 属性为 -1，用于触发不可达行
    SFT._hop = -1  # provoke unreachable line
    # 使用 pytest 检查是否引发 RuntimeError 异常
    with pytest.raises(RuntimeError):
        _ = SFT.k_max(4)
    with pytest.raises(RuntimeError):
        _ = SFT.k_min


def test_t():
    """Verify that the times of the slices are correct. """
    # 创建 ShortTimeFFT 对象，使用全 1 的输入信号，设置 hop=4，采样率 fs=2
    SFT = ShortTimeFFT(np.ones(8), hop=4, fs=2)
    # 断言 ShortTimeFFT 对象的 T 属性为 1/2
    assert SFT.T == 1/2
    # 断言 ShortTimeFFT 对象的 fs 属性为 2.
    assert SFT.fs == 2.
    # 断言 ShortTimeFFT 对象的 delta_t 属性为 4 * 1/2
    assert SFT.delta_t == 4 * 1/2
    # 计算 t_stft，作为 np.arange(0, SFT.p_max(10)) * SFT.delta_t 的结果
    t_stft = np.arange(0, SFT.p_max(10)) * SFT.delta_t
    # 使用 assert_equal 函数断言 ShortTimeFFT 对象调用 t(10) 方法返回 t_stft
    assert_equal(SFT.t(10), t_stft)
    # 使用 assert_equal 函数断言 ShortTimeFFT 对象调用 t(10, 1, 3) 方法返回 t_stft[1:3]
    assert_equal(SFT.t(10, 1, 3), t_stft[1:3])
    # 设置 ShortTimeFFT 对象的 T 属性为 1/4
    SFT.T = 1/4
    # 断言 ShortTimeFFT 对象的 T 属性为 1/4
    assert SFT.T == 1/4
    # 断言 ShortTimeFFT 对象的 fs 属性为 4
    assert SFT.fs == 4
    # 设置 ShortTimeFFT 对象的 fs 属性为 1/8
    SFT.fs = 1/8
    # 断言 ShortTimeFFT 对象的 fs 属性为 1/8
    assert SFT.fs == 1/8
    # 断言 ShortTimeFFT 对象的 T 属性为 8
    assert SFT.T == 8


@pytest.mark.parametrize('fft_mode, f',
                         [('onesided', [0., 1., 2.]),
                          ('onesided2X', [0., 1., 2.]),
                          ('twosided', [0., 1., 2., -2., -1.]),
                          ('centered', [-2., -1., 0., 1., 2.])])
def test_f(fft_mode: FFT_MODE_TYPE, f):
    """Verify the frequency values property `f`."""
    # 创建 ShortTimeFFT 对象，使用全 1 的输入信号，设置 hop=4，采样率 fs=5，fft_mode 和 scale_to 参数根据参数化测试动态变化
    SFT = ShortTimeFFT(np.ones(5), hop=4, fs=5, fft_mode=fft_mode,
                       scale_to='psd')
    # 使用 assert_equal 函数断言 ShortTimeFFT 对象的 f 属性与预期的 f 数组相等
    assert_equal(SFT.f, f)


def test_extent():
    """Ensure that the `extent()` method is correct. """
    # 创建 ShortTimeFFT 对象，使用全 1 的输入信号，设置 hop=4，采样率 fs=32，fft_mode 参数为 'onesided'
    SFT = ShortTimeFFT(np.ones(32), hop=4, fs=32, fft_mode='onesided')
    # 断言 ShortTimeFFT 对象调用 extent(100, 'tf', False) 方法返回 (-0.375, 3.625, 0.0, 17.0)
    assert SFT.extent(100, 'tf', False) == (-0.375, 3.625, 0.0, 17.0)
    # 断言 ShortTimeFFT 对象调用 extent(100, 'ft', False) 方法返回 (0.0, 17.0, -0.375, 3.625)
    assert SFT.extent(100, 'ft', False) == (0.0, 17.0, -0.375, 3.625)
    # 断言 ShortTimeFFT 对象调用 extent(100, 'tf', True) 方法返回 (-0.4375, 3.5625, -0.5, 16.5)
    assert SFT.extent(100, 'tf', True) == (-0.4375, 3.5625, -0.5, 16.5)
    # 断言 ShortTimeFFT 对象调用 extent(100, 'ft', True) 方法返回 (-0.5, 16.5, -0.4375, 3.5625)
    assert SFT.extent(100, 'ft', True) == (-0.5, 16.5, -0.4375, 3.5625)

    # 创建 ShortTimeFFT 对象，使用全 1 的输入信号，设置 hop=4，采样率 fs=32，fft_mode 参数为 'centered'
    SFT = ShortTimeFFT(np.ones(32), hop=4, fs=32, fft_mode='centered')
    # 断言 ShortTimeFFT 对
    # 使用 assert_allclose 函数来验证 SFT.spectrogram(x, y) 的输出是否接近于 X * Y.conj()
    assert_allclose(SFT.spectrogram(x, y), X * Y.conj())
@pytest.mark.parametrize('n', [8, 9])
def test_fft_func_roundtrip(n: int):
    """Test roundtrip `ifft_func(fft_func(x)) == x` for all permutations of
    relevant parameters. """
    # 设置随机种子以确保可重复性
    np.random.seed(2394795)
    # 生成长度为 n 的随机数组 x0
    x0 = np.random.rand(n)
    # 初始化 w 为全 1 数组，h_n 为 4
    w, h_n = np.ones(n), 4

    # 定义参数组合字典 pp
    pp = dict(
        fft_mode=get_args(FFT_MODE_TYPE),
        mfft=[None, n, n+1, n+2],
        scaling=[None, 'magnitude', 'psd'],
        phase_shift=[None, -n+1, 0, n // 2, n-1])
    # 遍历 pp 的笛卡尔积
    for f_typ, mfft, scaling, phase_shift in product(*pp.values()):
        # 跳过特定组合
        if f_typ == 'onesided2X' and scaling is None:
            continue  # this combination is forbidden
        # 创建 ShortTimeFFT 对象 SFT
        SFT = ShortTimeFFT(w, h_n, fs=n, fft_mode=f_typ, mfft=mfft,
                           scale_to=scaling, phase_shift=phase_shift)
        # 对 x0 进行 FFT 变换得到 X0
        X0 = SFT._fft_func(x0)
        # 对 X0 进行逆FFT变换得到 x1
        x1 = SFT._ifft_func(X0)
        # 断言 _fft_func() 和 _ifft_func() 的结果接近
        assert_allclose(x0, x1, err_msg="_fft_func() roundtrip failed for " +
                        f"{f_typ=}, {mfft=}, {scaling=}, {phase_shift=}")

    # 创建 SFT 对象并设定一个无效的 fft_mode，预期引发 RuntimeError 异常
    SFT = ShortTimeFFT(w, h_n, fs=1)
    SFT._fft_mode = 'invalid_fft'  # type: ignore
    with pytest.raises(RuntimeError):
        SFT._fft_func(x0)
    with pytest.raises(RuntimeError):
        SFT._ifft_func(x0)


@pytest.mark.parametrize('i', range(19))
def test_impulse_roundtrip(i):
    """Roundtrip for an impulse being at different positions `i`."""
    n = 19
    w, h_n = np.ones(8), 3
    # 创建长度为 n 的零数组 x，并在位置 i 处设置脉冲
    x = np.zeros(n)
    x[i] = 1

    # 创建 ShortTimeFFT 对象 SFT
    SFT = ShortTimeFFT(w, hop=h_n, fs=1, scale_to=None, phase_shift=None)
    # 对信号 x 进行短时傅里叶变换得到 Sx
    Sx = SFT.stft(x)
    # 测试将输入信号分成两部分的切片
    n_q = SFT.nearest_k_p(n // 2)
    Sx0 = SFT.stft(x[:n_q], padding='zeros')
    Sx1 = SFT.stft(x[n_q:], padding='zeros')
    q0_ub = SFT.upper_border_begin(n_q)[1] - SFT.p_min
    q1_le = SFT.lower_border_end[1] - SFT.p_min
    # 断言切片后的结果接近原始的 Sx
    assert_allclose(Sx0[:, :q0_ub], Sx[:, :q0_ub], err_msg=f"{i=}")
    assert_allclose(Sx1[:, q1_le:], Sx[:, q1_le-Sx1.shape[1]:],
                    err_msg=f"{i=}")

    # 合并切片后的 Sx0 和 Sx1 得到 Sx01
    Sx01 = np.hstack((Sx0[:, :q0_ub],
                      Sx0[:, q0_ub:] + Sx1[:, :q1_le],
                      Sx1[:, q1_le:]))
    # 断言合并后的 Sx01 与原始 Sx 接近
    assert_allclose(Sx, Sx01, atol=1e-8, err_msg=f"{i=}")

    # 对 Sx 进行逆短时傅里叶变换得到重建信号 y
    y = SFT.istft(Sx, 0, n)
    # 断言重建信号 y 与原始信号 x 接近
    assert_allclose(y, x, atol=1e-8, err_msg=f"{i=}")
    # 对 Sx 进行逆短时傅里叶变换得到部分重建信号 y0
    y0 = SFT.istft(Sx, 0, n//2)
    # 断言部分重建信号 y0 与原始信号的前半部分接近
    assert_allclose(x[:n//2], y0, atol=1e-8, err_msg=f"{i=}")
    # 对 Sx 进行逆短时傅里叶变换得到部分重建信号 y1
    y1 = SFT.istft(Sx, n // 2, n)
    # 断言部分重建信号 y1 与原始信号的后半部分接近
    assert_allclose(x[n // 2:], y1, atol=1e-8, err_msg=f"{i=}")


@pytest.mark.parametrize('hop', [1, 7, 8])
def test_asymmetric_window_roundtrip(hop: int):
    """An asymmetric window could uncover indexing problems. """
    # 设置随机种子以确保可重复性
    np.random.seed(23371)

    # 创建一个长度为 16 的浮点型数组 w，前半部分递增，后半部分为 1
    w = np.arange(16) / 8  # must be of type float
    w[len(w)//2:] = 1
    # 创建 ShortTimeFFT 对象 SFT
    SFT = ShortTimeFFT(w, hop, fs=1)

    # 创建长度为 64 的随机信号 x
    x = 10 * np.random.randn(64)
    # 对信号 x 进行短时傅里叶变换得到 Sx
    Sx = SFT.stft(x)
    # 对 Sx 进行逆短时傅里叶变换得到重建信号 x1
    x1 = SFT.istft(Sx, k1=len(x))
    # 断言重建信号 x1 与原始信号 x 接近
    assert_allclose(x1, x1, err_msg="Roundtrip for asymmetric window with " +
                                    f" {hop=} failed!")
def test_minimal_length_signal(m_num):
    """Verify that the shortest allowed signal works. """
    # 创建 ShortTimeFFT 对象，使用全为1的数组作为信号，窗口长度为 m_num//2，采样率为1
    SFT = ShortTimeFFT(np.ones(m_num), m_num//2, fs=1)
    # 向上取整 m_num/2 的值
    n = math.ceil(m_num/2)
    # 创建长度为 n 的全为1的数组 x
    x = np.ones(n)
    # 对信号 x 进行短时傅里叶变换
    Sx = SFT.stft(x)
    # 对经过傅里叶变换的信号进行逆变换
    x1 = SFT.istft(Sx, k1=n)
    # 检查逆变换结果 x1 是否与原始信号 x 一致，若不一致则抛出错误信息
    assert_allclose(x1, x, err_msg=f"Roundtrip minimal length signal ({n=})" +
                                   f" for {m_num} sample window failed!")
    # 检查当输入信号 x 的长度比 n 小1时，是否抛出 ValueError 异常
    with pytest.raises(ValueError, match=rf"len\(x\)={n-1} must be >= ceil.*"):
        SFT.stft(x[:-1])
    # 检查当傅里叶变换结果 Sx 的第二维长度比 n-1 小1时，是否抛出 ValueError 异常
    with pytest.raises(ValueError, match=rf"S.shape\[t_axis\]={Sx.shape[1]-1}"
                       f" needs to have at least {Sx.shape[1]} slices"):
        SFT.istft(Sx[:, :-1], k1=n)


def test_tutorial_stft_sliding_win():
    """Verify example in "Sliding Windows" subsection from the "User Guide".

    In :ref:`tutorial_stft_sliding_win` (file ``signal.rst``) of the
    :ref:`user_guide` the behavior the border behavior of
    ``ShortTimeFFT(np.ones(6), 2, fs=1)`` with a 50 sample signal is discussed.
    This test verifies the presented indexes.
    """
    # 创建 ShortTimeFFT 对象，使用全为1的数组作为信号，窗口长度为2，采样率为1
    SFT = ShortTimeFFT(np.ones(6), 2, fs=1)

    # 验证下边界:
    # 检查中间切片的索引是否为3
    assert SFT.m_num_mid == 3, f"Slice middle is not 3 but {SFT.m_num_mid=}"
    # 检查最小切片的索引是否为-1
    assert SFT.p_min == -1, f"Lowest slice {SFT.p_min=} is not -1"
    # 检查最小切片的样本索引是否为-5
    assert SFT.k_min == -5, f"Lowest slice sample {SFT.p_min=} is not -5"
    # 获取下边界结束时的样本索引和切片索引
    k_lb, p_lb = SFT.lower_border_end
    # 检查第一个未受影响切片的索引是否为2
    assert p_lb == 2, f"First unaffected slice {p_lb=} is not 2"
    # 检查第一个未受影响样本的索引是否为5
    assert k_lb == 5, f"First unaffected sample {k_lb=} is not 5"

    n = 50  # 信号的上边界
    # 检查最大切片的索引是否为27
    assert (p_max := SFT.p_max(n)) == 27, f"Last slice {p_max=} must be 27"
    # 检查最大样本的索引是否为55
    assert (k_max := SFT.k_max(n)) == 55, f"Last sample {k_max=} must be 55"
    # 获取上边界开始时的样本索引和切片索引
    k_ub, p_ub = SFT.upper_border_begin(n)
    # 检查第一个上边界切片的索引是否为24
    assert p_ub == 24, f"First upper border slice {p_ub=} must be 24"
    # 检查第一个上边界样本的索引是否为45
    assert k_ub == 45, f"First upper border slice {k_ub=} must be 45"


def test_tutorial_stft_legacy_stft():
    """Verify STFT example in "Comparison with Legacy Implementation" from the
    "User Guide".

    In :ref:`tutorial_stft_legacy_stft` (file ``signal.rst``) of the
    :ref:`user_guide` the legacy and the new implementation are compared.
    """
    fs, N = 200, 1001  # 200 Hz 采样率的5秒信号
    t_z = np.arange(N) / fs  # 信号的时间索引
    z = np.exp(2j*np.pi * 70 * (t_z - 0.2 * t_z ** 2))  # 复值的频率扫描信号

    nperseg, noverlap = 50, 40
    win = ('gaussian', 1e-2 * fs)  # 标准差为0.01秒的高斯窗口

    # 传统的 STFT:
    f0_u, t0, Sz0_u = stft(z, fs, win, nperseg, noverlap,
                           return_onesided=False, scaling='spectrum')
    Sz0 = fftshift(Sz0_u, axes=0)

    # 新的 STFT:
    # 使用窗口参数创建 ShortTimeFFT 对象
    SFT = ShortTimeFFT.from_window(win, fs, nperseg, noverlap,
                                   fft_mode='centered',
                                   scale_to='magnitude', phase_shift=None)
    # 对信号 z 进行短时傅里叶变换
    Sz1 = SFT.stft(z)

    # 检查两种方法得到的结果是否近似相等
    assert_allclose(Sz0, Sz1[:, 2:-1])
    # 断言Sz1[:, 1]的绝对值的最小值和最大值分别为6.925060911593139e-07和8.00271269218721e-07
    assert_allclose((abs(Sz1[:, 1]).min(), abs(Sz1[:, 1]).max()),
                    (6.925060911593139e-07, 8.00271269218721e-07))
    
    # 使用逆短时傅里叶变换（ISTFT）函数istft对Sz0_u进行反变换，得到时域信号t0_r和z0_r
    t0_r, z0_r = istft(Sz0_u, fs, win, nperseg, noverlap, input_onesided=False,
                       scaling='spectrum')
    
    # 使用类SFT的ISTFT方法对Sz1进行反变换，得到时域信号z1_r
    z1_r = SFT.istft(Sz1, k1=N)
    
    # 断言z0_r的长度为N + 9
    assert len(z0_r) == N + 9
    
    # 断言z0_r的前N个元素与z数组的值相近
    assert_allclose(z0_r[:N], z)
    
    # 断言z1_r与z数组的值相近
    assert_allclose(z1_r, z)
    
    # 断言SFT类的spectrogram方法返回的结果与Sz1的绝对值平方相近
    # 即频谱图是STFT的绝对值的平方
    assert_allclose(SFT.spectrogram(z), abs(Sz1) ** 2)
# 测试函数，验证在“用户指南”的“比较传统实现”部分中的示例中的频谱图示例。
def test_tutorial_stft_legacy_spectrogram():
    """Verify spectrogram example in "Comparison with Legacy Implementation"
    from the "User Guide".

    In :ref:`tutorial_stft_legacy_stft` (file ``signal.rst``) of the
    :ref:`user_guide` the legacy and the new implementation are compared.
    """
    fs, N = 200, 1001  # 采样率为200 Hz，信号长度为1001个样本点（大约5秒的信号）
    t_z = np.arange(N) / fs  # 信号的时间索引
    z = np.exp(2j*np.pi*70 * (t_z - 0.2*t_z**2))  # 复值信号的扫频信号

    nperseg, noverlap = 50, 40
    win = ('gaussian', 1e-2 * fs)  # 带有0.01秒标准差的高斯窗口

    # 传统的频谱图：
    f2_u, t2, Sz2_u = spectrogram(z, fs, win, nperseg, noverlap, detrend=None,
                                  return_onesided=False, scaling='spectrum',
                                  mode='complex')

    f2, Sz2 = fftshift(f2_u), fftshift(Sz2_u, axes=0)

    # 新的短时傅里叶变换（STFT）：
    SFT = ShortTimeFFT.from_window(win, fs, nperseg, noverlap,
                                   fft_mode='centered', scale_to='magnitude',
                                   phase_shift=None)
    Sz3 = SFT.stft(z, p0=0, p1=(N-noverlap) // SFT.hop, k_offset=nperseg // 2)
    t3 = SFT.t(N, p0=0, p1=(N-noverlap) // SFT.hop, k_offset=nperseg // 2)

    assert_allclose(t2, t3)  # 检查时间轴的一致性
    assert_allclose(f2, SFT.f)  # 检查频率轴的一致性
    assert_allclose(Sz2, Sz3)  # 检查频谱数据的一致性


# 验证四维信号通过置换其形状的正确性。
def test_permute_axes():
    """Verify correctness of four-dimensional signal by permuting its
    shape. """
    n = 25
    SFT = ShortTimeFFT(np.ones(8)/8, hop=3, fs=n)
    x0 = np.arange(n)
    Sx0 = SFT.stft(x0)
    Sx0 = Sx0.reshape((Sx0.shape[0], 1, 1, 1, Sx0.shape[-1]))
    SxT = np.moveaxis(Sx0, (0, -1), (-1, 0))

    atol = 2 * np.finfo(SFT.win.dtype).resolution
    for i in range(4):
        y = np.reshape(x0, np.roll((n, 1, 1, 1), i))
        Sy = SFT.stft(y, axis=i)
        assert_allclose(Sy, np.moveaxis(Sx0, 0, i))  # 检查移动轴后的一致性

        yb0 = SFT.istft(Sy, k1=n, f_axis=i)
        assert_allclose(yb0, y, atol=atol)  # 检查逆STFT的一致性
        # 显式t轴参数（用于覆盖）：
        yb1 = SFT.istft(Sy, k1=n, f_axis=i, t_axis=Sy.ndim-1)
        assert_allclose(yb1, y, atol=atol)  # 检查逆STFT的一致性

        SyT = np.moveaxis(Sy, (i, -1), (-1, i))
        assert_allclose(SyT, np.moveaxis(SxT, 0, i))  # 检查移动轴后的一致性

        ybT = SFT.istft(SyT, k1=n, t_axis=i, f_axis=-1)
        assert_allclose(ybT, y, atol=atol)  # 检查逆STFT的一致性


@pytest.mark.parametrize("fft_mode",
                         ('twosided', 'centered', 'onesided', 'onesided2X'))
def test_roundtrip_multidimensional(fft_mode: FFT_MODE_TYPE):
    """Test roundtrip of a multidimensional input signal versus its components.

    This test can uncover potential problems with `fftshift()`.
    """
    n = 9
    x = np.arange(4*n*2).reshape(4, n, 2)
    SFT = ShortTimeFFT(get_window('hann', 4), hop=2, fs=1,
                       scale_to='magnitude', fft_mode=fft_mode)
    Sx = SFT.stft(x, axis=1)
    y = SFT.istft(Sx, k1=n, f_axis=1, t_axis=-1)
    assert_allclose(y, x, err_msg='Multidim. roundtrip failed!')  # 检查多维信号的往复一致性
    # 使用 product 函数生成两个范围迭代器，分别对应 x 数组的第一维和第三维
    # x.shape[0] 表示 x 数组的第一维长度，x.shape[2] 表示 x 数组的第三维长度
    for i, j in product(range(x.shape[0]), range(x.shape[2])):
        # 从 Sx 数组中获取对应的部分 Sx[i, :, j, :]，并对其进行逆短时傅里叶变换（ISTFT）
        y_ = SFT.istft(Sx[i, :, j, :], k1=n)
        
        # 断言 ISTFT 的结果 y_ 与 x 数组的对应部分 x[i, :, j] 非常接近
        assert_allclose(y_, x[i, :, j], err_msg="Multidim. roundtrip for component " +
                        f"x[{i}, :, {j}] and {fft_mode=} failed!")
@pytest.mark.parametrize('window, n, nperseg, noverlap',
                         [('boxcar', 100, 10, 0),     # Test no overlap
                          ('boxcar', 100, 10, 9),     # Test high overlap
                          ('bartlett', 101, 51, 26),  # Test odd nperseg
                          ('hann', 1024, 256, 128),   # Test defaults
                          (('tukey', 0.5), 1152, 256, 64),  # Test Tukey
                          ('hann', 1024, 256, 255),   # Test overlapped hann
                          ('boxcar', 100, 10, 3),     # NOLA True, COLA False
                          ('bartlett', 101, 51, 37),  # NOLA True, COLA False
                          ('hann', 1024, 256, 127),   # NOLA True, COLA False
                          # NOLA True, COLA False:
                          (('tukey', 0.5), 1152, 256, 14),
                          ('hann', 1024, 256, 5)])    # NOLA True, COLA False
def test_roundtrip_windows(window, n: int, nperseg: int, noverlap: int):
    """Roundtrip test adapted from `test_spectral.TestSTFT`.

    The parameters are taken from the methods test_roundtrip_real(),
    test_roundtrip_nola_not_cola(), test_roundtrip_float32(),
    test_roundtrip_complex().
    """
    np.random.seed(2394655)  # 设置随机数种子，确保测试可复现性

    w = get_window(window, nperseg)  # 获取指定窗口类型和长度的窗口函数
    SFT = ShortTimeFFT(w, nperseg - noverlap, fs=1, fft_mode='twosided',
                       phase_shift=None)  # 创建短时傅里叶变换对象

    z = 10 * np.random.randn(n) + 10j * np.random.randn(n)  # 生成复数序列 z
    Sz = SFT.stft(z)  # 对 z 进行短时傅里叶变换
    z1 = SFT.istft(Sz, k1=len(z))  # 对 Sz 进行逆短时傅里叶变换
    assert_allclose(z, z1, err_msg="Roundtrip for complex values failed")  # 检查复数值序列的往复变换精度

    x = 10 * np.random.randn(n)  # 生成实数序列 x
    Sx = SFT.stft(x)  # 对 x 进行短时傅里叶变换
    x1 = SFT.istft(Sx, k1=len(z))  # 对 Sx 进行逆短时傅里叶变换
    assert_allclose(x, x1, err_msg="Roundtrip for float values failed")  # 检查实数值序列的往复变换精度

    x32 = x.astype(np.float32)  # 将实数序列转换为32位浮点数
    Sx32 = SFT.stft(x32)  # 对32位浮点数序列进行短时傅里叶变换
    x32_1 = SFT.istft(Sx32, k1=len(x32))  # 对 Sx32 进行逆短时傅里叶变换
    assert_allclose(x32, x32_1,
                    err_msg="Roundtrip for 32 Bit float values failed")  # 检查32位浮点数序列的往复变换精度


@pytest.mark.parametrize('signal_type', ('real', 'complex'))
def test_roundtrip_complex_window(signal_type):
    """Test roundtrip for complex-valued window function

    The purpose of this test is to check if the dual window is calculated
    correctly for complex-valued windows.
    """
    np.random.seed(1354654)  # 设置随机数种子，确保测试可复现性
    win = np.exp(2j*np.linspace(0, np.pi, 8))  # 生成复数值的窗口函数
    SFT = ShortTimeFFT(win, 3, fs=1, fft_mode='twosided')  # 创建短时傅里叶变换对象

    z = 10 * np.random.randn(11)  # 生成复数序列 z
    if signal_type == 'complex':
        z = z + 2j * z  # 若信号类型为复数，则 z 为复数序列
    Sz = SFT.stft(z)  # 对 z 进行短时傅里叶变换
    z1 = SFT.istft(Sz, k1=len(z))  # 对 Sz 进行逆短时傅里叶变换
    assert_allclose(z, z1,
                    err_msg="Roundtrip for complex-valued window failed")  # 检查复数值窗口函数的往复变换精度


def test_average_all_segments():
    """Compare `welch` function with stft mean.

    Ported from `TestSpectrogram.test_average_all_segments` from file
    ``test__spectral.py``.
    """
    x = np.random.randn(1024)  # 生成1024个随机数的实数序列

    fs = 1.0  # 设置采样频率
    window = ('tukey', 0.25)  # 使用 Tukey 窗口
    nperseg, noverlap = 16, 2  # 设置段长度和重叠量
    fw, Pw = welch(x, fs, window, nperseg, noverlap)  # 计算功率谱密度估计
    # 使用 ShortTimeFFT 类从给定窗口参数创建对象 SFT
    SFT = ShortTimeFFT.from_window(window, fs, nperseg, noverlap,
                                   fft_mode='onesided2X', scale_to='psd',
                                   phase_shift=None)
    # 使用 SFT 对象计算信号 x 的时频谱图 P
    # `welch` 与 STFT 将窗口放置位置不同：
    P = SFT.spectrogram(x, detr='constant', p0=0,
                        p1=(len(x)-noverlap)//SFT.hop, k_offset=nperseg//2)

    # 断言 SFT 对象的频率 f 与预期的频率 fw 很接近
    assert_allclose(SFT.f, fw)
    # 断言 P 的平均值沿着最后一个轴的值与预期的 Pw 很接近
    assert_allclose(np.mean(P, axis=-1), Pw)
@pytest.mark.parametrize('window, N, nperseg, noverlap, mfft',
                         # 来自于 test_roundtrip_padded_FFT:
                         [('hann', 1024, 256, 128, 512),
                          ('hann', 1024, 256, 128, 501),
                          ('boxcar', 100, 10, 0, 33),
                          (('tukey', 0.5), 1152, 256, 64, 1024),
                          # 来自于 test_roundtrip_padded_signal:
                          ('boxcar', 101, 10, 0, None),
                          ('hann', 1000, 256, 128, None),
                          # 来自于 test_roundtrip_boundary_extension:
                          ('boxcar', 100, 10, 0, None),
                          ('boxcar', 100, 10, 9, None)])
@pytest.mark.parametrize('padding', get_args(PAD_TYPE))
def test_stft_padding_roundtrip(window, N: int, nperseg: int, noverlap: int,
                                mfft: int, padding):
    """Test the parameter 'padding' of `stft` with roundtrips.

    The STFT parametrizations were taken from the methods
    `test_roundtrip_padded_FFT`, `test_roundtrip_padded_signal` and
    `test_roundtrip_boundary_extension` from class `TestSTFT` in file
    ``test_spectral.py``. Note that the ShortTimeFFT does not need the
    concept of "boundary extension".
    """
    x = normal_distribution.rvs(size=N, random_state=2909)  # 生成实数信号
    z = x * np.exp(1j * np.pi / 4)  # 生成复数信号

    # 从给定的窗口和参数创建 ShortTimeFFT 对象
    SFT = ShortTimeFFT.from_window(window, 1, nperseg, noverlap,
                                   fft_mode='twosided', mfft=mfft)
    # 对实数信号进行 STFT 变换，使用指定的 padding 参数
    Sx = SFT.stft(x, padding=padding)
    # 对 STFT 结果进行逆变换，应得到原始信号 x
    x1 = SFT.istft(Sx, k1=N)
    # 检查逆变换后的结果是否接近原始信号 x
    assert_allclose(x1, x,
                    err_msg=f"Failed real roundtrip with '{padding}' padding")

    # 对复数信号进行 STFT 变换，使用指定的 padding 参数
    Sz = SFT.stft(z, padding=padding)
    # 对 STFT 结果进行逆变换，应得到原始信号 z
    z1 = SFT.istft(Sz, k1=N)
    # 检查逆变换后的结果是否接近原始信号 z
    assert_allclose(z1, z, err_msg="Failed complex roundtrip with " +
                    f" '{padding}' padding")


@pytest.mark.parametrize('N_x', (128, 129, 255, 256, 1337))  # 信号长度
@pytest.mark.parametrize('w_size', (128, 256))  # 窗口长度
@pytest.mark.parametrize('t_step', (4, 64))  # SFT 时间步长
@pytest.mark.parametrize('f_c', (7., 23.))  # 输入正弦波的频率
def test_energy_conservation(N_x: int, w_size: int, t_step: int, f_c: float):
    """Test if a `psd`-scaled STFT conserves the L2 norm.

    This test is adapted from MNE-Python [1]_. Besides being battle-tested,
    this test has the benefit of using non-standard window including
    non-positive values and a 2d input signal.

    Since `ShortTimeFFT` requires the signal length `N_x` to be at least the
    window length `w_size`, the parameter `N_x` was changed from
    ``(127, 128, 255, 256, 1337)`` to ``(128, 129, 255, 256, 1337)`` to be
    more useful.

    .. [1] File ``test_stft.py`` of MNE-Python
        https://github.com/mne-tools/mne-python/blob/main/mne/time_frequency/tests/test_stft.py
    """
    # 根据窗口长度创建具有非正值的窗口
    window = np.sin(np.arange(.5, w_size + .5) / w_size * np.pi)
    # 使用给定参数创建 ShortTimeFFT 对象 SFT，用于短时傅里叶变换
    SFT = ShortTimeFFT(window, t_step, fs=1000, fft_mode='onesided2X',
                       scale_to='psd')
    # 计算数值精度容差
    atol = 2*np.finfo(window.dtype).resolution
    # 确保 N_x 至少为 w_size
    N_x = max(N_x, w_size)  # minimal sing

    # 使用低频信号进行测试
    t = np.arange(N_x).astype(np.float64)
    # 构造正弦波信号 x
    x = np.sin(2 * np.pi * f_c * t * SFT.T)
    # 将 x 复制成一个包含两个信号的数组
    x = np.array([x, x + 1.])
    # 对信号 x 进行短时傅里叶变换得到频谱 X
    X = SFT.stft(x)
    # 对频谱 X 进行逆短时傅里叶变换得到 xp
    xp = SFT.istft(X, k1=N_x)

    # 计算频谱 X 的最大频率
    max_freq = SFT.f[np.argmax(np.sum(np.abs(X[0]) ** 2, axis=1))]

    # 断言：频谱 X 的列数应与 SFT 中的 f_pts 相等
    assert X.shape[1] == SFT.f_pts
    # 断言：SFT 中的频率 f 全部大于等于 0
    assert np.all(SFT.f >= 0.)
    # 断言：频率的最大值与预期频率 f_c 的差应小于 1
    assert np.abs(max_freq - f_c) < 1.
    # 断言：x 与 xp 之间的数值接近度满足指定的容差
    assert_allclose(x, xp, atol=atol)

    # 检查 L2-范数平方（即能量）是否守恒：
    # 计算原始信号 x 的能量 E_x（数值积分）
    E_x = np.sum(x**2, axis=-1) * SFT.T
    # 计算频谱 X 的能量 E_X
    aX2 = X.real**2 + X.imag.real**2
    E_X = np.sum(np.sum(aX2, axis=-1) * SFT.delta_t, axis=-1) * SFT.delta_f
    # 断言：E_X 与 E_x 之间的数值接近度满足指定的容差
    assert_allclose(E_X, E_x, atol=atol)

    # 使用随机信号进行测试
    np.random.seed(2392795)
    x = np.random.randn(2, N_x)
    # 对随机信号 x 进行短时傅里叶变换得到频谱 X
    X = SFT.stft(x)
    # 对频谱 X 进行逆短时傅里叶变换得到 xp
    xp = SFT.istft(X, k1=N_x)

    # 断言：频谱 X 的列数应与 SFT 中的 f_pts 相等
    assert X.shape[1] == SFT.f_pts
    # 断言：SFT 中的频率 f 全部大于等于 0
    assert np.all(SFT.f >= 0.)
    # 断言：频率的最大值与预期频率 f_c 的差应小于 1
    assert np.abs(max_freq - f_c) < 1.
    # 断言：x 与 xp 之间的数值接近度满足指定的容差
    assert_allclose(x, xp, atol=atol)

    # 检查 L2-范数平方（即能量）是否守恒：
    # 计算原始信号 x 的能量 E_x（数值积分）
    E_x = np.sum(x**2, axis=-1) * SFT.T
    # 计算频谱 X 的能量 E_X
    aX2 = X.real ** 2 + X.imag.real ** 2
    E_X = np.sum(np.sum(aX2, axis=-1) * SFT.delta_t, axis=-1) * SFT.delta_f
    # 断言：E_X 与 E_x 之间的数值接近度满足指定的容差
    assert_allclose(E_X, E_x, atol=atol)

    # 尝试使用空数组进行测试
    x = np.zeros((0, N_x))
    # 对空数组 x 进行短时傅里叶变换得到频谱 X
    X = SFT.stft(x)
    # 对频谱 X 进行逆短时傅里叶变换得到 xp
    xp = SFT.istft(X, k1=N_x)
    # 断言：xp 的形状应与 x 相同
    assert xp.shape == x.shape
```