# `D:\src\scipysrc\scipy\scipy\signal\_short_time_fft.py`

```
"""Implementation of an FFT-based Short-time Fourier Transform. """

# Implementation Notes for this file (as of 2023-07)
# --------------------------------------------------
# * MyPy version 1.1.1 does not seem to support decorated property methods
#   properly. Hence, applying ``@property`` to methods decorated with `@cache``
#   (as tried with the ``lower_border_end`` method) causes a mypy error when
#   accessing it as an index (e.g., ``SFT.lower_border_end[0]``).
# * Since the method `stft` and `istft` have identical names as the legacy
#   functions in the signal module, referencing them as HTML link in the
#   docstrings has to be done by an explicit `~ShortTimeFFT.stft` instead of an
#   ambiguous `stft` (The ``~`` hides the class / module name).
# * The HTML documentation currently renders each method/property on a separate
#   page without reference to the parent class. Thus, a link to `ShortTimeFFT`
#   was added to the "See Also" section of each method/property. These links
#   can be removed, when SciPy updates ``pydata-sphinx-theme`` to >= 0.13.3
#   (currently 0.9). Consult Issue 18512 and PR 16660 for further details.
#

# Provides typing union operator ``|`` in Python 3.9:
from __future__ import annotations
# Linter does not allow to import ``Generator`` from ``typing`` module:
from collections.abc import Generator
from functools import cache, lru_cache, partial
from typing import Callable, get_args, Literal

import numpy as np

import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window

__all__ = ['ShortTimeFFT']


#: Allowed values for parameter `padding` of method `ShortTimeFFT.stft()`:
PAD_TYPE = Literal['zeros', 'edge', 'even', 'odd']

#: Allowed values for property `ShortTimeFFT.fft_mode`:
FFT_MODE_TYPE = Literal['twosided', 'centered', 'onesided', 'onesided2X']


def _calc_dual_canonical_window(win: np.ndarray, hop: int) -> np.ndarray:
    """Calculate canonical dual window for 1d window `win` and a time step
    of `hop` samples.

    A ``ValueError`` is raised, if the inversion fails.

    This is a separate function not a method, since it is also used in the
    class method ``ShortTimeFFT.from_dual()``.
    """
    # Check if the hop size is greater than the length of the window
    if hop > len(win):
        raise ValueError(f"{hop=} is larger than window length of {len(win)}" +
                         " => STFT not invertible!")
    # Check if the dtype of `win` is integer, which is not allowed
    if issubclass(win.dtype.type, np.integer):
        raise ValueError("Parameter 'win' cannot be of integer type, but " +
                         f"{win.dtype=} => STFT not invertible!")
        # The calculation of `relative_resolution` does not work for ints.
        # Furthermore, `win / DD` casts the integers away, thus an implicit
        # cast is avoided, which can always cause confusion when using 32-Bit
        # floats.

    # Calculate the squared magnitude of the window
    w2 = win.real**2 + win.imag**2  # win*win.conj() does not ensure w2 is real
    # Create a copy of w2
    DD = w2.copy()
    # 对于给定的窗口 `win`，从跳数 `hop` 开始，每隔 `hop` 步长，更新 `DD` 数组
    for k_ in range(hop, len(win), hop):
        # 将 `w2` 的前 `len(win) - k_` 部分加到 `DD` 的后 `k_` 部分
        DD[k_:] += w2[:-k_]
        # 将 `w2` 的后 `k_` 部分加到 `DD` 的前 `len(win) - k_` 部分
        DD[:-k_] += w2[k_:]

    # 检查 `DD` 中的所有元素是否大于零
    # 计算相对分辨率，作为窗口 `win` 数据类型的精度与 `DD` 中最大值的乘积
    relative_resolution = np.finfo(win.dtype).resolution * max(DD)
    # 如果 `DD` 中有任何元素小于相对分辨率，则抛出错误，指出短时傅里叶变换不可逆
    if not np.all(DD >= relative_resolution):
        raise ValueError("Short-time Fourier Transform not invertible!")

    # 返回经过调整后的窗口 `win`，使其与 `DD` 元素相除
    return win / DD
# 忽略 PyShadowingNames 警告，用于避免名称遮蔽的情况发生
class ShortTimeFFT:
    r"""提供参数化的离散短时傅里叶变换（STFT）及其逆变换（ISTFT）。

    .. currentmodule:: scipy.signal.ShortTimeFFT

    `~ShortTimeFFT.stft` 方法通过滑动窗口 (`win`) 并以 `hop` 步长递进地计算输入信号的连续 FFT。
    可用于量化信号频谱随时间的变化。

    `~ShortTimeFFT.stft` 返回一个复数矩阵 S[q,p]，其中第 p 列代表以时间 t[p] = p * `delta_t` = p * `hop` * `T`（其中 `T` 是输入信号的采样间隔）为中心的 FFT。
    第 q 行代表频率 f[q] = q * `delta_f`，其中 `delta_f` = 1 / (`mfft` * `T`) 是 FFT 的频率分辨率。

    逆 STFT `~ShortTimeFFT.istft` 通过反向 STFT 步骤计算：对 S[q,p] 的第 p 切片进行逆 FFT，并乘以称为双窗口的因子（参见 `dual_win`）。
    将结果移动 p * `delta_t`，并将结果添加到先前移动的结果中以重建信号。
    如果只知道双窗口并且 STFT 是可逆的，则可以使用 `from_dual` 方法来实例化这个类。

    由于时间 t = 0 通常被定义为输入信号的第一个样本，STFT 值通常具有负时间槽。
    因此，负索引如 `p_min` 或 `k_min` 并不表示从数组末尾倒数计数，而是表示 t = 0 的左侧。

    可在 :ref:`user_guide` 的 :ref:`tutorial_stft` 部分找到更详细的信息。

    注意，除了 `scale_to`（使用 `scaling`）之外，初始化器的所有参数都具有相同的命名属性。

    Parameters
    ----------
    win : np.ndarray
        窗口必须是实值或复值的 1 维数组。
    hop : int
        窗口每步移动的采样数增量。
    fs : float
        输入信号和窗口的采样频率。与采样间隔 `T` 的关系为 ``T = 1 / fs``。
    fft_mode : 'twosided', 'centered', 'onesided', 'onesided2X'
        使用的 FFT 模式（默认为 'onesided'）。
        有关详细信息，请参阅属性 `fft_mode`。
    mfft: int | None
        所使用的 FFT 的长度，如果需要进行零填充的 FFT。
        如果为 ``None``（默认值），则使用窗口 `win` 的长度。
    dual_win : np.ndarray | None
        `win` 的双窗口。如果为 ``None``，在需要时会计算。
    scale_to : 'magnitude', 'psd' | None
        如果不为 ``None``（默认），则对窗口函数进行缩放，使得每个 STFT 列代表 'magnitude' 或功率谱密度（'psd'）谱。
        此参数将属性 `scaling` 设置为相同的值。有关详细信息，请参阅方法 `scale_to`。
    # phase_shift : int | None
    # 如果设置，对每个频率 `f` 添加线性相位 `phase_shift` / `mfft` * `f`。
    # 默认值 0 确保零时刻（t=0）处的片段没有相位移动（居中）。
    # 更多详情请参见 `phase_shift` 属性。

    # Examples
    # --------
    # 下面的示例展示了具有变化频率 :math:`f_i(t)` 的正弦波的短时傅里叶变换（STFT）的幅度（在图中用红色虚线标记）：

    # >>> import numpy as np
    # >>> import matplotlib.pyplot as plt
    # >>> from scipy.signal import ShortTimeFFT
    # >>> from scipy.signal.windows import gaussian
    # ...
    # >>> T_x, N = 1 / 20, 1000  # 20 Hz 采样率，50 秒信号
    # >>> t_x = np.arange(N) * T_x  # 信号的时间索引
    # >>> f_i = 1 * np.arctan((t_x - t_x[N // 2]) / 2) + 5  # 变化的频率
    # >>> x = np.sin(2*np.pi*np.cumsum(f_i)*T_x) # 信号

    # 使用的高斯窗口长度为 50 个样本或 2.5 秒。`ShortTimeFFT` 中的参数 `mfft=200` 导致频谱过采样了 4 倍：

    # >>> g_std = 8  # 高斯窗口的标准差（单位：样本数）
    # >>> w = gaussian(50, std=g_std, sym=True)  # 对称高斯窗口
    # >>> SFT = ShortTimeFFT(w, hop=10, fs=1/T_x, mfft=200, scale_to='magnitude')
    # >>> Sx = SFT.stft(x)  # 执行 STFT

    # 在图中，信号 `x` 的时间范围由垂直虚线标记。注意，STFT 产生的值超出了 `x` 的时间范围。
    # 左右的阴影区域指示由于窗口片段未完全位于 `x` 的时间范围内而导致的边界效应。

    # >>> fig1, ax1 = plt.subplots(figsize=(6., 4.))  # 放大图表
    # >>> t_lo, t_hi = SFT.extent(N)[:2]  # 绘图的时间范围
    # >>> ax1.set_title(rf"STFT ({SFT.m_num*SFT.T:g}$\,s$ 高斯窗口, " +
    # ...               rf"$\sigma_t={g_std*SFT.T}\,$s)")
    # >>> ax1.set(xlabel=f"Time $t$ in seconds ({SFT.p_num(N)} slices, " +
    # ...                rf"$\Delta t = {SFT.delta_t:g}\,$s)",
    # ...         ylabel=f"Freq. $f$ in Hz ({SFT.f_pts} bins, " +
    # ...                rf"$\Delta f = {SFT.delta_f:g}\,$Hz)",
    # ...         xlim=(t_lo, t_hi))
    # ...
    # >>> im1 = ax1.imshow(abs(Sx), origin='lower', aspect='auto',
    # ...                  extent=SFT.extent(N), cmap='viridis')
    # >>> ax1.plot(t_x, f_i, 'r--', alpha=.5, label='$f_i(t)$')
    # >>> fig1.colorbar(im1, label="Magnitude $|S_x(t, f)|$")
    # ...
    # >>> # 标记窗口片段超出侧边的区域：
    # >>> for t0_, t1_ in [(t_lo, SFT.lower_border_end[0] * SFT.T),
    # ...                  (SFT.upper_border_begin(N)[0] * SFT.T, t_hi)]:
    # ...     ax1.axvspan(t0_, t1_, color='w', linewidth=0, alpha=.2)
    # >>> for t_ in [0, N * SFT.T]:  # 使用垂直线标记信号边界：
    # ...     ax1.axvline(t_, color='y', linestyle='--', alpha=0.5)
    # >>> ax1.legend()
    # >>> fig1.tight_layout()
    # >>> plt.show()
    Reconstructing the signal with the `~ShortTimeFFT.istft` is
    straightforward, but note that the length of `x1` should be specified,
    since the SFT length increases in `hop` steps:

    >>> SFT.invertible  # check if invertible
    True
    >>> x1 = SFT.istft(Sx, k1=N)
    >>> np.allclose(x, x1)
    True

    It is possible to calculate the SFT of signal parts:

    >>> p_q = SFT.nearest_k_p(N // 2)
    >>> Sx0 = SFT.stft(x[:p_q])
    >>> Sx1 = SFT.stft(x[p_q:])

    When assembling sequential STFT parts together, the overlap needs to be
    considered:

    >>> p0_ub = SFT.upper_border_begin(p_q)[1] - SFT.p_min
    >>> p1_le = SFT.lower_border_end[1] - SFT.p_min
    >>> Sx01 = np.hstack((Sx0[:, :p0_ub],
    ...                   Sx0[:, p0_ub:] + Sx1[:, :p1_le],
    ...                   Sx1[:, p1_le:]))
    >>> np.allclose(Sx01, Sx)  # Compare with SFT of complete signal
    True

    It is also possible to calculate the `istft` for signal parts:

    >>> y_p = SFT.istft(Sx, N//3, N//2)
    >>> np.allclose(y_p, x[N//3:N//2])
    True

    """
    # 不可变属性（只有获取器没有设置器）：
    _win: np.ndarray  # 窗口数组
    _dual_win: np.ndarray | None = None  # 规范双窗口
    _hop: int  # STFT步长，以样本数为单位

    # 可变属性：
    _fs: float  # 输入信号和窗口的采样频率
    _fft_mode: FFT_MODE_TYPE = 'onesided'  # 使用的FFT模式
    _mfft: int  # 使用的FFT长度，默认为win的长度
    _scaling: Literal['magnitude', 'psd'] | None = None  # _win的缩放类型
    _phase_shift: int | None  # FFT相位偏移量，以样本数为单位

    # 用于缓存计算值的属性：
    _fac_mag: float | None = None  # 幅度缩放因子
    _fac_psd: float | None = None  # PSD缩放因子
    _lower_border_end: tuple[int, int] | None = None  # 下边界的结束位置元组
    def __init__(self, win: np.ndarray, hop: int, fs: float, *,
                 fft_mode: FFT_MODE_TYPE = 'onesided',
                 mfft: int | None = None,
                 dual_win: np.ndarray | None = None,
                 scale_to: Literal['magnitude', 'psd'] | None = None,
                 phase_shift: int | None = 0):
        # 检查窗口数组是否为一维且长度大于0
        if not (win.ndim == 1 and win.size > 0):
            raise ValueError(f"Parameter win must be 1d, but {win.shape=}!")
        # 检查窗口数组是否包含有限的数值
        if not all(np.isfinite(win)):
            raise ValueError("Parameter win must have finite entries!")
        # 检查跳跃步长是否为整数且大于等于1
        if not (hop >= 1 and isinstance(hop, int)):
            raise ValueError(f"Parameter {hop=} is not an integer >= 1!")
        # 将参数赋值给对象的私有属性
        self._win, self._hop, self.fs = win, hop, fs

        # 如果未指定 mfft 的长度，则默认为窗口数组的长度
        self.mfft = len(win) if mfft is None else mfft

        # 如果存在双窗口参数，需确保其形状与主窗口数组相同
        if dual_win is not None:
            if dual_win.shape != win.shape:
                raise ValueError(f"{dual_win.shape=} must equal {win.shape=}!")
            # 检查双窗口数组是否包含有限的数值
            if not all(np.isfinite(dual_win)):
                raise ValueError("Parameter dual_win must be a finite array!")
        # 设置对象的双窗口属性，在缩放之前需要先设置这个属性
        self._dual_win = dual_win  # needs to be set before scaling

        # 如果指定了缩放目标，则进行缩放操作
        if scale_to is not None:  # needs to be set before fft_mode
            self.scale_to(scale_to)

        # 设置对象的 FFT 模式和相位偏移属性
        self.fft_mode, self.phase_shift = fft_mode, phase_shift

    @classmethod
    @property
    def win(self) -> np.ndarray:
        """Window function as real- or complex-valued 1d array.

        This attribute is read only, since `dual_win` depends on it.

        See Also
        --------
        dual_win: Canonical dual window.
        m_num: Number of samples in window `win`.
        m_num_mid: Center index of window `win`.
        mfft: Length of input for the FFT used - may be larger than `m_num`.
        hop: Time increment in signal samples for sliding window.
        win: Window function as real- or complex-valued 1d array.
        ShortTimeFFT: Class this property belongs to.
        """
        # 返回对象的窗口函数属性
        return self._win

    @property
    def hop(self) -> int:
        """Time increment in signal samples for sliding window.

        This attribute is read only, since `dual_win` depends on it.

        See Also
        --------
        delta_t: Time increment of STFT (``hop*T``)
        m_num: Number of samples in window `win`.
        m_num_mid: Center index of window `win`.
        mfft: Length of input for the FFT used - may be larger than `m_num`.
        T: Sampling interval of input signal and of the window.
        win: Window function as real- or complex-valued 1d array.
        ShortTimeFFT: Class this property belongs to.
        """
        # 返回对象的跳跃步长属性
        return self._hop

    @property
    def fs(self) -> float:
        """Sampling frequency of input signal.

        This attribute is read only.

        See Also
        --------
        win: Window function as real- or complex-valued 1d array.
        hop: Time increment in signal samples for sliding window.
        """
        # 返回对象的采样频率属性
        return self._fs
    def T(self) -> float:
        """
        Sampling interval of input signal and of the window.

        A ``ValueError`` is raised if it is set to a non-positive value.

        See Also
        --------
        delta_t: Time increment of STFT (``hop*T``)
        hop: Time increment in signal samples for sliding window.
        fs: Sampling frequency (being ``1/T``)
        t: Times of STFT for an input signal with `n` samples.
        ShortTimeFFT: Class this property belongs to.
        """
        # 返回采样间隔 T，其值为采样频率的倒数
        return 1 / self._fs

    @T.setter
    def T(self, v: float):
        """
        Sampling interval of input signal and of the window.

        A ``ValueError`` is raised if it is set to a non-positive value.
        """
        # 如果设置的采样间隔 v 不是正数，抛出 ValueError 异常
        if not (v > 0):
            raise ValueError(f"Sampling interval T={v} must be positive!")
        # 设置采样频率 fs 为采样间隔的倒数
        self._fs = 1 / v

    @property
    def fs(self) -> float:
        """
        Sampling frequency of input signal and of the window.

        The sampling frequency is the inverse of the sampling interval `T`.
        A ``ValueError`` is raised if it is set to a non-positive value.

        See Also
        --------
        delta_t: Time increment of STFT (``hop*T``)
        hop: Time increment in signal samples for sliding window.
        T: Sampling interval of input signal and of the window (``1/fs``).
        ShortTimeFFT: Class this property belongs to.
        """
        # 返回当前的采样频率 fs
        return self._fs

    @fs.setter
    def fs(self, v: float):
        """
        Sampling frequency of input signal and of the window.

        The sampling frequency is the inverse of the sampling interval `T`.
        A ``ValueError`` is raised if it is set to a non-positive value.
        """
        # 如果设置的采样频率 v 不是正数，抛出 ValueError 异常
        if not (v > 0):
            raise ValueError(f"Sampling frequency fs={v} must be positive!")
        # 设置采样频率 fs 为给定值 v
        self._fs = v

    @property
    def fft_mode(self) -> FFT_MODE_TYPE:
        """Returns the current mode of FFT ('twosided', 'centered', 'onesided' or
        'onesided2X').

        It retrieves and returns the current mode of FFT used by this class instance.
        This property is read-only.

        Returns
        -------
        FFT_MODE_TYPE
            Current FFT mode used.

        Raises
        ------
        None
            No exceptions are raised.

        Notes
        -----
        This property provides access to the current FFT mode, which determines
        how the FFT is computed and what kind of frequency spectrum is produced.

        See Also
        --------
        fft_mode : Set the mode of utilized FFT.
        """
        return self._fft_mode

    @fft_mode.setter
    def fft_mode(self, t: FFT_MODE_TYPE):
        """Set the mode of FFT.

        Sets the mode of FFT used by this class instance to the specified value.
        The allowed values are 'twosided', 'centered', 'onesided', 'onesided2X'.

        Parameters
        ----------
        t : FFT_MODE_TYPE
            The new FFT mode to set.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the provided mode `t` is not one of the allowed modes.
            If `t` is 'onesided' or 'onesided2X' and the window `self.win` is complex.
            If `t` is 'onesided2X' and `self.scaling` is None.

        Notes
        -----
        This setter method allows changing the FFT mode used by the class instance.
        It performs validation checks on the input mode `t` and raises errors if
        the mode is not allowed or if the conditions for using 'onesided' or 'onesided2X'
        are not met.

        See Also
        --------
        fft_mode : Get the current mode of utilized FFT.
        """
        if t not in (fft_mode_types := get_args(FFT_MODE_TYPE)):
            raise ValueError(f"fft_mode='{t}' not in {fft_mode_types}!")

        if t in {'onesided', 'onesided2X'} and np.iscomplexobj(self.win):
            raise ValueError(f"One-sided spectra, i.e., fft_mode='{t}', " +
                             "are not allowed for complex-valued windows!")

        if t == 'onesided2X' and self.scaling is None:
            raise ValueError(f"For scaling is None, fft_mode='{t}' is invalid!"
                             "Do scale_to('psd') or scale_to('magnitude')!")
        self._fft_mode = t
    def mfft(self) -> int:
        """
        Return the length of input for the FFT used, which can be larger than
        the window length `m_num`. If not set, defaults to `m_num`.

        See Also
        --------
        f_pts: Number of points along the frequency axis.
        f: Frequencies values of the STFT.
        m_num: Number of samples in window `win`.
        ShortTimeFFT: Class this property belongs to.
        """
        return self._mfft

    @mfft.setter
    def mfft(self, n_: int):
        """
        Setter for the length of FFT utilized.

        Parameters
        ----------
        n_ : int
            Length of FFT to set.

        Raises
        ------
        ValueError
            If `n_` is less than the window length `m_num`.

        See Also
        --------
        mfft: Property for getting the length of FFT utilized.
        """
        if not (n_ >= self.m_num):
            raise ValueError(f"Attribute mfft={n_} needs to be at least the " +
                             f"window length m_num={self.m_num}!")
        self._mfft = n_

    @property
    def scaling(self) -> Literal['magnitude', 'psd'] | None:
        """
        Returns the normalization applied to the window function
        ('magnitude', 'psd' or ``None``).

        Notes
        -----
        If not ``None``, FFTs can be interpreted as either a magnitude or
        a power spectral density spectrum. Scaling can be set using the
        `scale_to` method or the initializer parameter ``scale_to``.

        See Also
        --------
        fac_magnitude: Scaling factor for a magnitude spectrum.
        fac_psd: Scaling factor for a power spectral density spectrum.
        fft_mode: Mode of utilized FFT
        scale_to: Scale window to obtain 'magnitude' or 'psd' scaling.
        ShortTimeFFT: Class this property belongs to.
        """
        return self._scaling
    def scale_to(self, scaling: Literal['magnitude', 'psd']):
        """
        Scale window to obtain 'magnitude' or 'psd' scaling for the STFT.

        The window of a 'magnitude' spectrum has an integral of one, i.e., unit
        area for non-negative windows. This ensures that absolute the values of
        spectrum does not change if the length of the window changes (given
        the input signal is stationary).

        To represent the power spectral density ('psd') for varying length
        windows the area of the absolute square of the window needs to be
        unity.

        The `scaling` property shows the current scaling. The properties
        `fac_magnitude` and `fac_psd` show the scaling factors required to
        scale the STFT values to a magnitude or a psd spectrum.

        This method is called, if the initializer parameter `scale_to` is set.

        See Also
        --------
        fac_magnitude: Scaling factor for to  a magnitude spectrum.
        fac_psd: Scaling factor for to  a power spectral density spectrum.
        fft_mode: Mode of utilized FFT
        scaling: Normalization applied to the window function.
        ShortTimeFFT: Class this method belongs to.
        """
        # 检查 scaling 是否为 'magnitude' 或 'psd'，否则抛出 ValueError
        if scaling not in (scaling_values := {'magnitude', 'psd'}):
            raise ValueError(f"{scaling=} not in {scaling_values}!")
        # 如果当前 scaling 已经是所需的 scaling，则不进行任何操作，直接返回
        if self._scaling == scaling:  # do nothing
            return

        # 根据 scaling 选择相应的缩放因子
        s_fac = self.fac_psd if scaling == 'psd' else self.fac_magnitude
        # 对窗口数组应用缩放因子
        self._win = self._win * s_fac
        # 如果存在双窗口，则对其应用相反的缩放因子
        if self._dual_win is not None:
            self._dual_win = self._dual_win / s_fac
        # 重置缩放因子属性
        self._fac_mag, self._fac_psd = None, None  # reset scaling factors
        # 更新当前 scaling 属性
        self._scaling = scaling

    @property
    def phase_shift(self) -> int | None:
        """
        If set, add linear phase `phase_shift` / `mfft` * `f` to each FFT
        slice of frequency `f`.

        Shifting (more precisely `rolling`) an `mfft`-point FFT input by
        `phase_shift` samples results in a multiplication of the output by
        ``np.exp(2j*np.pi*q*phase_shift/mfft)`` at the frequency q * `delta_f`.

        The default value 0 ensures that there is no phase shift on the
        zeroth slice (in which t=0 is centered).
        No phase shift (``phase_shift is None``) is equivalent to
        ``phase_shift = -mfft//2``. In this case slices are not shifted
        before calculating the FFT.

        The absolute value of `phase_shift` is limited to be less than `mfft`.

        See Also
        --------
        delta_f: Width of the frequency bins of the STFT.
        f: Frequencies values of the STFT.
        mfft: Length of input for the FFT used
        ShortTimeFFT: Class this property belongs to.
        """
        # 返回当前的相位偏移值
        return self._phase_shift

    @phase_shift.setter
    def phase_shift(self, v: int | None):
        """设置相位移，其绝对值应小于 mfft 的样本数。

        查看 `phase_shift` 获取器方法以获取更多详细信息。
        """
        if v is None:
            self._phase_shift = v
            return
        if not isinstance(v, int):
            raise ValueError(f"phase_shift={v} 是样本单位。因此必须是整数或者可以为 None!")
        if not (-self.mfft < v < self.mfft):
            raise ValueError(f"对于 mfft={self.mfft}, phase_shift={v}，不满足 -mfft < phase_shift < mfft 的条件！")
        self._phase_shift = v

    def _x_slices(self, x: np.ndarray, k_off: int, p0: int, p1: int,
                  padding: PAD_TYPE) -> Generator[np.ndarray, None, None]:
        """生成沿着 `x` 的最后一个轴的信号切片。

        此方法仅由 `stft_detrend` 使用。参数在 `~ShortTimeFFT.stft` 中有描述。
        """
        if padding not in (padding_types := get_args(PAD_TYPE)):
            raise ValueError(f"参数 {padding=} 不在 {padding_types} 中！")
        pad_kws: dict[str, dict] = {  # 可传递给 np.pad 的可能关键字：
            'zeros': dict(mode='constant', constant_values=(0, 0)),
            'edge': dict(mode='edge'),
            'even': dict(mode='reflect', reflect_type='even'),
            'odd': dict(mode='reflect', reflect_type='odd'),
           }  # pad_kws 的类型注解是为了使 mypy 满意

        n, n1 = x.shape[-1], (p1 - p0) * self.hop
        k0 = p0 * self.hop - self.m_num_mid + k_off  # 起始样本
        k1 = k0 + n1 + self.m_num  # 结束样本

        i0, i1 = max(k0, 0), min(k1, n)  # 缩短 x 的索引
        # 用于填充 x 的维度：
        pad_width = [(0, 0)] * (x.ndim-1) + [(-min(k0, 0), max(k1 - n, 0))]

        x1 = np.pad(x[..., i0:i1], pad_width, **pad_kws[padding])
        for k_ in range(0, n1, self.hop):
            yield x1[..., k_:k_ + self.m_num]
    def stft(self, x: np.ndarray, p0: int | None = None,
             p1: int | None = None, *, k_offset: int = 0,
             padding: PAD_TYPE = 'zeros', axis: int = -1) \
            -> np.ndarray:
        """Perform the short-time Fourier transform.

        A two-dimensional matrix with ``p1-p0`` columns is calculated.
        The `f_pts` rows represent value at the frequencies `f`. The q-th
        column of the windowed FFT with the window `win` is centered at t[q].
        The columns represent the values at the frequencies `f`.

        Parameters
        ----------
        x
            The input signal as real or complex valued array. For complex values, the
            property `fft_mode` must be set to 'twosided' or 'centered'.
        p0
            The first element of the range of slices to calculate. If ``None``
            then it is set to :attr:`p_min`, which is the smallest possible
            slice.
        p1
            The end of the array. If ``None`` then `p_max(n)` is used.
        k_offset
            Index of first sample (t = 0) in `x`.
        padding
            Kind of values which are added, when the sliding window sticks out
            on either the lower or upper end of the input `x`. Zeros are added
            if the default 'zeros' is set. For 'edge' either the first or the
            last value of `x` is used. 'even' pads by reflecting the
            signal on the first or last sample and 'odd' additionally
            multiplies it with -1.
        axis
            The axis of `x` over which to compute the STFT.
            If not given, the last axis is used.

        Returns
        -------
        S
            A complex array is returned with the dimension always being larger
            by one than of `x`. The last axis always represent the time slices
            of the STFT. `axis` defines the frequency axis (default second to
            last). E.g., for a one-dimensional `x`, a complex 2d array is
            returned, with axis 0 representing frequency and axis 1 the time
            slices.

        See Also
        --------
        delta_f: Width of the frequency bins of the STFT.
        delta_t: Time increment of STFT
        f: Frequencies values of the STFT.
        invertible: Check if STFT is invertible.
        :meth:`~ShortTimeFFT.istft`: Inverse short-time Fourier transform.
        p_range: Determine and validate slice index range.
        stft_detrend: STFT with detrended segments.
        t: Times of STFT for an input signal with `n` samples.
        :class:`scipy.signal.ShortTimeFFT`: Class this method belongs to.
        """
        # 调用 stft_detrend 方法进行短时傅里叶变换，返回结果
        return self.stft_detrend(x, None, p0, p1, k_offset=k_offset,
                                 padding=padding, axis=axis)
    def stft_detrend(self, x: np.ndarray,
                     detr: Callable[[np.ndarray], np.ndarray] | Literal['linear', 'constant'] | None,  # noqa: E501
                     p0: int | None = None, p1: int | None = None, *,
                     k_offset: int = 0, padding: PAD_TYPE = 'zeros',
                     axis: int = -1) \
            -> np.ndarray:
        """Short-time Fourier transform with a trend being subtracted from each
        segment beforehand.

        If `detr` is set to 'constant', the mean is subtracted, if set to
        "linear", the linear trend is removed. This is achieved by calling
        :func:`scipy.signal.detrend`. If `detr` is a function, `detr` is
        applied to each segment.
        All other parameters have the same meaning as in `~ShortTimeFFT.stft`.

        Note that due to the detrending, the original signal cannot be
        reconstructed by the `~ShortTimeFFT.istft`.

        See Also
        --------
        invertible: Check if STFT is invertible.
        :meth:`~ShortTimeFFT.istft`: Inverse short-time Fourier transform.
        :meth:`~ShortTimeFFT.stft`: Short-time Fourier transform
                                   (without detrending).
        :class:`scipy.signal.ShortTimeFFT`: Class this method belongs to.
        """
        # 检查是否允许使用复数类型的输入数据
        if self.onesided_fft and np.iscomplexobj(x):
            raise ValueError(f"Complex-valued `x` not allowed for {self.fft_mode=}'! "
                             "Set property `fft_mode` to 'twosided' or 'centered'.")
        
        # 如果 detr 是字符串，则部分应用 detrend 函数进行趋势去除
        if isinstance(detr, str):
            detr = partial(detrend, type=detr)
        # 如果 detr 不是 None 或者可调用对象，则引发错误
        elif not (detr is None or callable(detr)):
            raise ValueError(f"Parameter {detr=} is not a str, function or " +
                             "None!")
        
        # 获取信号 x 在指定轴上的长度
        n = x.shape[axis]
        
        # 检查信号长度是否满足要求
        if not (n >= (m2p := self.m_num-self.m_num_mid)):
            e_str = f'{len(x)=}' if x.ndim == 1 else f'of {axis=} of {x.shape}'
            raise ValueError(f"{e_str} must be >= ceil(m_num/2) = {m2p}!")
        
        # 如果信号是多维的，则根据 NumPy 的广播机制重新排列轴
        if x.ndim > 1:  # motivated by the NumPy broadcasting mechanisms:
            x = np.moveaxis(x, axis, -1)
        
        # 确定切片的索引范围
        p0, p1 = self.p_range(n, p0, p1)
        
        # 初始化输出的 STFT 结果 S 的形状
        S_shape_1d = (self.f_pts, p1 - p0)
        S_shape = x.shape[:-1] + S_shape_1d if x.ndim > 1 else S_shape_1d
        S = np.zeros(S_shape, dtype=complex)
        
        # 遍历信号 x 的每个切片，应用 detr 函数（如果 detr 不为 None）
        for p_, x_ in enumerate(self._x_slices(x, k_offset, p0, p1, padding)):
            if detr is not None:
                x_ = detr(x_)
            # 计算 STFT 并存储到 S 中
            S[..., :, p_] = self._fft_func(x_ * self.win.conj())
        
        # 如果信号是多维的，则恢复原始的轴顺序
        if x.ndim > 1:
            return np.moveaxis(S, -2, axis if axis >= 0 else axis-1)
        
        # 返回计算得到的 STFT 结果 S
        return S
    def dual_win(self) -> np.ndarray:
        """Canonical dual window.

        A STFT can be interpreted as the input signal being expressed as a
        weighted sum of modulated and time-shifted dual windows. Note that for
        a given window there exist many dual windows. The canonical window is
        the one with the minimal energy (i.e., :math:`L_2` norm).

        `dual_win` has same length as `win`, namely `m_num` samples.

        If the dual window cannot be calculated a ``ValueError`` is raised.
        This attribute is read only and calculated lazily.

        See Also
        --------
        dual_win: Canonical dual window.
        m_num: Number of samples in window `win`.
        win: Window function as real- or complex-valued 1d array.
        ShortTimeFFT: Class this property belongs to.
        """
        # 如果尚未计算 `_dual_win`，则调用 `_calc_dual_canonical_window` 函数计算并赋值给 `_dual_win`
        if self._dual_win is None:
            self._dual_win = _calc_dual_canonical_window(self.win, self.hop)
        # 返回计算得到的 canonical dual window
        return self._dual_win

    @property
    def invertible(self) -> bool:
        """Check if STFT is invertible.

        This is achieved by trying to calculate the canonical dual window.

        See Also
        --------
        :meth:`~ShortTimeFFT.istft`: Inverse short-time Fourier transform.
        m_num: Number of samples in window `win` and `dual_win`.
        dual_win: Canonical dual window.
        win: Window for STFT.
        ShortTimeFFT: Class this property belongs to.
        """
        try:
            # 检查是否能够成功调用 `self.dual_win()`，如果能则说明可以计算 canonical dual window
            return len(self.dual_win) > 0
        except ValueError:
            # 如果计算失败则返回 False，表示不可逆
            return False

    @property
    def fac_magnitude(self) -> float:
        """Factor to multiply the STFT values by to scale each frequency slice
        to a magnitude spectrum.

        It is 1 if attribute ``scaling == 'magnitude'``.
        The window can be scaled to a magnitude spectrum by using the method
        `scale_to`.

        See Also
        --------
        fac_psd: Scaling factor for to a power spectral density spectrum.
        scale_to: Scale window to obtain 'magnitude' or 'psd' scaling.
        scaling: Normalization applied to the window function.
        ShortTimeFFT: Class this property belongs to.
        """
        # 如果 `scaling` 属性为 'magnitude'，则返回因子 1
        if self.scaling == 'magnitude':
            return 1
        # 如果尚未计算 `_fac_mag`，则计算并赋值为 `1 / abs(sum(self.win))`
        if self._fac_mag is None:
            self._fac_mag = 1 / abs(sum(self.win))
        # 返回计算得到的 scaling factor
        return self._fac_mag
    def fac_psd(self) -> float:
        """Factor to multiply the STFT values by to scale each frequency slice
        to a power spectral density (PSD).

        It is 1 if attribute ``scaling == 'psd'``.
        The window can be scaled to a psd spectrum by using the method
        `scale_to`.

        See Also
        --------
        fac_magnitude: Scaling factor for to a magnitude spectrum.
        scale_to: Scale window to obtain 'magnitude' or 'psd' scaling.
        scaling: Normalization applied to the window function.
        ShortTimeFFT: Class this property belongs to.
        """
        # 如果 scaling 属性为 'psd'，则返回因子 1
        if self.scaling == 'psd':
            return 1
        # 如果 _fac_psd 属性为 None，则计算并返回 _fac_psd
        if self._fac_psd is None:
            # 计算 _fac_psd，为 win 实部和虚部的平方和除以 T 的平方根的倒数
            self._fac_psd = 1 / np.sqrt(
                sum(self.win.real**2 + self.win.imag**2) / self.T)
        # 返回计算好的 _fac_psd
        return self._fac_psd

    @property
    def m_num(self) -> int:
        """Number of samples in window `win`.

        Note that the FFT can be oversampled by zero-padding. This is achieved
        by setting the `mfft` property.

        See Also
        --------
        m_num_mid: Center index of window `win`.
        mfft: Length of input for the FFT used - may be larger than `m_num`.
        hop: Time increment in signal samples for sliding window.
        win: Window function as real- or complex-valued 1d array.
        ShortTimeFFT: Class this property belongs to.
        """
        # 返回窗口 win 中的样本数
        return len(self.win)

    @property
    def m_num_mid(self) -> int:
        """Center index of window `win`.

        For odd `m_num`, ``(m_num - 1) / 2`` is returned and
        for even `m_num` (per definition) ``m_num / 2`` is returned.

        See Also
        --------
        m_num: Number of samples in window `win`.
        mfft: Length of input for the FFT used - may be larger than `m_num`.
        hop: Time increment in signal samples for sliding window.
        win: Window function as real- or complex-valued 1d array.
        ShortTimeFFT: Class this property belongs to.
        """
        # 返回窗口 win 的中心索引
        return self.m_num // 2

    @cache
    def _pre_padding(self) -> tuple[int, int]:
        """Smallest signal index and slice index due to padding.

         Since, per convention, for time t=0, n,q is zero, the returned values
         are negative or zero.
         """
        # 计算由于填充而导致的最小信号索引和切片索引
        w2 = self.win.real**2 + self.win.imag**2
        # 将窗口向左移动，直到与 t >= 0 的重叠消失为止：
        n0 = -self.m_num_mid
        for q_, n_ in enumerate(range(n0, n0 - self.m_num - 1, -self.hop)):
            n_next = n_ - self.hop
            if n_next + self.m_num <= 0 or all(w2[n_next:] == 0):
                return n_, -q_
        # 如果执行到这一行，说明存在问题，抛出运行时错误
        raise RuntimeError("This is code line should not have been reached!")
        # 如果到达这种情况，可能意味着应返回第一个切片，即：return n0, 0

    @property
    def k_min(self) -> int:
        """The smallest possible signal index of the STFT.

        `k_min` is the index of the left-most non-zero value of the lowest
        slice `p_min`. Since the zeroth slice is centered over the zeroth
        sample of the input signal, `k_min` is never positive.
        A detailed example is provided in the :ref:`tutorial_stft_sliding_win`
        section of the :ref:`user_guide`.

        See Also
        --------
        k_max: First sample index after signal end not touched by a time slice.
        lower_border_end: Where pre-padding effects end.
        p_min: The smallest possible slice index.
        p_max: Index of first non-overlapping upper time slice.
        p_num: Number of time slices, i.e., `p_max` - `p_min`.
        p_range: Determine and validate slice index range.
        upper_border_begin: Where post-padding effects start.
        ShortTimeFFT: Class this property belongs to.
        """
        return self._pre_padding()[0]




    @property
    def p_min(self) -> int:
        """The smallest possible slice index.

        `p_min` is the index of the left-most slice, where the window still
        sticks into the signal, i.e., has non-zero part for t >= 0.
        `k_min` is the smallest index where the window function of the slice
        `p_min` is non-zero.

        Since, per convention the zeroth slice is centered at t=0,
        `p_min` <= 0 always holds.
        A detailed example is provided in the :ref:`tutorial_stft_sliding_win`
        section of the :ref:`user_guide`.

        See Also
        --------
        k_min: The smallest possible signal index.
        k_max: First sample index after signal end not touched by a time slice.
        p_max: Index of first non-overlapping upper time slice.
        p_num: Number of time slices, i.e., `p_max` - `p_min`.
        p_range: Determine and validate slice index range.
        ShortTimeFFT: Class this property belongs to.
        """
        return self._pre_padding()[1]




    @lru_cache(maxsize=256)
    def _post_padding(self, n: int) -> tuple[int, int]:
        """Largest signal index and slice index due to padding."""
        w2 = self.win.real**2 + self.win.imag**2
        # move window to the right until the overlap for t < t[n] vanishes:
        q1 = n // self.hop   # last slice index with t[p1] <= t[n]
        k1 = q1 * self.hop - self.m_num_mid
        for q_, k_ in enumerate(range(k1, n+self.m_num, self.hop), start=q1):
            n_next = k_ + self.hop
            if n_next >= n or all(w2[:n-n_next] == 0):
                return k_ + self.m_num, q_ + 1
        raise RuntimeError("This is code line should not have been reached!")
        # If this case is reached, it probably means the last slice should be
        # returned, i.e.: return k1 + self.m_num - self.m_num_mid, q1 + 1
    def k_max(self, n: int) -> int:
        """获取信号结束后第一个未被时间片段触及的样本索引。

        `k_max` - 1 是切片 `p_max` 中最大的样本索引，对于包含 `n` 个样本的输入信号。
        在 :ref:`tutorial_stft_sliding_win` 部分的 :ref:`user_guide` 中提供了详细示例。

        See Also
        --------
        k_min: 最小可能信号索引。
        p_min: 最小可能切片索引。
        p_max: 第一个不重叠的上限时间片段的索引。
        p_num: 时间片段的数量，即 `p_max` - `p_min`。
        p_range: 确定和验证切片索引范围。
        ShortTimeFFT: 包含此方法的类。
        """
        return self._post_padding(n)[0]

    def p_max(self, n: int) -> int:
        """对于包含 `n` 个样本的输入，获取第一个不重叠的上限时间片段的索引。

        注意，中心点 t[p_max] = (p_max(n)-1) * `delta_t` 通常大于最后的时间索引 t[n-1] == (`n`-1) * `T`。
        窗口切片覆盖的样本索引的上限边界由 `k_max` 给出。
        此外，`p_max` 不表示切片数量 `p_num`，因为 `p_min` 通常小于零。
        在 :ref:`tutorial_stft_sliding_win` 部分的 :ref:`user_guide` 中提供了详细示例。

        See Also
        --------
        k_min: 最小可能信号索引。
        k_max: 信号结束后第一个未被时间片段触及的样本索引。
        p_min: 最小可能切片索引。
        p_num: 时间片段的数量，即 `p_max` - `p_min`。
        p_range: 确定和验证切片索引范围。
        ShortTimeFFT: 包含此方法的类。
        """
        return self._post_padding(n)[1]

    def p_num(self, n: int) -> int:
        """对于包含 `n` 个样本的输入信号，获取时间片段的数量。

        `p_num` = `p_max` - `p_min`，其中 `p_min` 通常为负数。
        在 :ref:`tutorial_stft_sliding_win` 部分的 :ref:`user_guide` 中提供了详细示例。

        See Also
        --------
        k_min: 最小可能信号索引。
        k_max: 信号结束后第一个未被时间片段触及的样本索引。
        lower_border_end: 预填充效果结束的地方。
        p_min: 最小可能切片索引。
        p_max: 第一个不重叠的上限时间片段的索引。
        p_range: 确定和验证切片索引范围。
        upper_border_begin: 后填充效果开始的地方。
        ShortTimeFFT: 包含此方法的类。
        """
        return self.p_max(n) - self.p_min
    def lower_border_end(self) -> tuple[int, int]:
        """
        返回第一个信号索引和第一个未受预填充影响的时间片索引的元组。

        描述窗口不再超出信号域左侧的点。
        在 :ref:`tutorial_stft_sliding_win` 部分的 :ref:`user_guide` 中提供了详细示例。

        See Also
        --------
        k_min: 最小可能的信号索引。
        k_max: 信号结束后第一个未受时间片影响的样本索引。
        lower_border_end: 预填充效果结束的位置。
        p_min: 最小可能的时间片索引。
        p_max: 第一个非重叠的上限时间片的索引。
        p_num: 时间片数量，即 `p_max` - `p_min`。
        p_range: 确定和验证时间片索引范围。
        upper_border_begin: 后填充效果开始的位置。
        ShortTimeFFT: 属于该属性的类。
        """
        # 由于 MyPy 的限制，不使用 @cache 装饰器

        # 如果已经计算过，直接返回保存的结果
        if self._lower_border_end is not None:
            return self._lower_border_end

        # self.win 中第一个非零元素的索引:
        m0 = np.flatnonzero(self.win.real**2 + self.win.imag**2)[0]

        # 将窗口向右移动，直到不再超出左侧边界:
        k0 = -self.m_num_mid + m0
        for q_, k_ in enumerate(range(k0, self.hop + 1, self.hop)):
            if k_ + self.hop >= 0:  # 下一个条目不再超出左侧边界
                self._lower_border_end = (k_ + self.m_num, q_ + 1)
                return self._lower_border_end
        # 如果循环结束仍未找到满足条件的 k，返回默认值
        self._lower_border_end = (0, max(self.p_min, 0))  # 在第一个时间片结束
        return self._lower_border_end

    @lru_cache(maxsize=256)
    def upper_border_begin(self, n: int) -> tuple[int, int]:
        """
        返回第一个信号索引和首个受后填充影响的切片索引。

        描述窗口开始向右突出到信号域之外的位置。
        详细示例请参见 :ref:`tutorial_stft_sliding_win` 中的 :ref:`user_guide` 部分。

        See Also
        --------
        k_min: 最小可能的信号索引。
        k_max: 信号结束后第一个样本索引，不被时间片段触及。
        lower_border_end: 前填充效果结束的位置。
        p_min: 最小可能的切片索引。
        p_max: 第一个非重叠的上限时间片索引。
        p_num: 时间片数量，即 `p_max` - `p_min`。
        p_range: 确定和验证切片索引范围。
        ShortTimeFFT: 此方法所属的类。
        """
        w2 = self.win.real**2 + self.win.imag**2
        q2 = n // self.hop + 1  # 第一个 t[q] >= t[n]
        q1 = max((n-self.m_num) // self.hop - 1, -1)
        # 向左移动窗口，直到不再向右突出：
        for q_ in range(q2, q1, -1):
            k_ = q_ * self.hop + (self.m_num - self.m_num_mid)
            if k_ < n or all(w2[n-k_:] == 0):
                return (q_ + 1) * self.hop - self.m_num_mid, q_ + 1
        return 0, 0  # 边界从第一个切片开始

    @property
    def delta_t(self) -> float:
        """
        STFT 的时间增量。

        时间增量 `delta_t` = `T` * `hop` 表示基于采样间隔 `T` 将样本增量 `hop` 转换为时间。

        See Also
        --------
        delta_f: STFT 频率 bin 的宽度。
        hop: 滑动窗口的信号样本跳跃大小。
        t: 具有 `n` 个样本的输入信号的 STFT 时间。
        T: 输入信号和窗口 `win` 的采样间隔。
        ShortTimeFFT: 此属性所属的类。
        """
        return self.T * self.hop
    # 定义一个方法 p_range，用于确定和验证切片索引范围
    def p_range(self, n: int, p0: int | None = None,
                p1: int | None = None) -> tuple[int, int]:
        """Determine and validate slice index range.

        Parameters
        ----------
        n : int
            Number of samples of input signal, assuming t[0] = 0.
        p0 : int | None
            First slice index. If 0 then the first slice is centered at t = 0.
            If ``None`` then `p_min` is used. Note that p0 may be < 0 if
            slices are left of t = 0.
        p1 : int | None
            End of interval (last value is p1-1).
            If ``None`` then `p_max(n)` is used.


        Returns
        -------
        p0_ : int
            The fist slice index
        p1_ : int
            End of interval (last value is p1-1).

        Notes
        -----
        A ``ValueError`` is raised if ``p_min <= p0 < p1 <= p_max(n)`` does not
        hold.

        See Also
        --------
        k_min: The smallest possible signal index.
        k_max: First sample index after signal end not touched by a time slice.
        lower_border_end: Where pre-padding effects end.
        p_min: The smallest possible slice index.
        p_max: Index of first non-overlapping upper time slice.
        p_num: Number of time slices, i.e., `p_max` - `p_min`.
        upper_border_begin: Where post-padding effects start.
        ShortTimeFFT: Class this property belongs to.
        """
        # 获取 p_max(n) 的值并赋给 p_max
        p_max = self.p_max(n)  # shorthand
        # 如果 p0 为 None，则使用 self.p_min 的值
        p0_ = self.p_min if p0 is None else p0
        # 如果 p1 为 None，则使用 p_max 的值
        p1_ = p_max if p1 is None else p1
        # 检查条件是否满足：self.p_min <= p0_ < p1_ <= p_max
        if not (self.p_min <= p0_ < p1_ <= p_max):
            # 如果条件不满足，抛出 ValueError 异常，说明参数无效
            raise ValueError(f"Invalid Parameter {p0=}, {p1=}, i.e., " +
                             f"{self.p_min=} <= p0 < p1 <= {p_max=} " +
                             f"does not hold for signal length {n=}!")
        # 返回确定的切片索引范围
        return p0_, p1_

    @lru_cache(maxsize=1)
    def t(self, n: int, p0: int | None = None, p1: int | None = None,
          k_offset: int = 0) -> np.ndarray:
        """Times of STFT for an input signal with `n` samples.

        Returns a 1d array with times of the `~ShortTimeFFT.stft` values with
        the same  parametrization. Note that the slices are
        ``delta_t = hop * T`` time units apart.

         Parameters
        ----------
        n
            Number of sample of the input signal.
        p0
            The first element of the range of slices to calculate. If ``None``
            then it is set to :attr:`p_min`, which is the smallest possible
            slice.
        p1
            The end of the array. If ``None`` then `p_max(n)` is used.
        k_offset
            Index of first sample (t = 0) in `x`.


        See Also
        --------
        delta_t: Time increment of STFT (``hop*T``)
        hop: Time increment in signal samples for sliding window.
        nearest_k_p: Nearest sample index k_p for which t[k_p] == t[p] holds.
        T: Sampling interval of input signal and of the window (``1/fs``).
        fs: Sampling frequency (being ``1/T``)
        ShortTimeFFT: Class this method belongs to.
        """
        # Calculate the actual start and end indices of the time range
        p0, p1 = self.p_range(n, p0, p1)
        # Generate a 1D array of time values corresponding to STFT slices
        return np.arange(p0, p1) * self.delta_t + k_offset * self.T

    def nearest_k_p(self, k: int, left: bool = True) -> int:
        """Return nearest sample index k_p for which t[k_p] == t[p] holds.

        The nearest next smaller time sample p (where t[p] is the center
        position of the window of the p-th slice) is p_k = k // `hop`.
        If `hop` is a divisor of `k` than `k` is returned.
        If `left` is set than p_k * `hop` is returned else (p_k+1) * `hop`.

        This method can be used to slice an input signal into chunks for
        calculating the STFT and iSTFT incrementally.

        See Also
        --------
        delta_t: Time increment of STFT (``hop*T``)
        hop: Time increment in signal samples for sliding window.
        T: Sampling interval of input signal and of the window (``1/fs``).
        fs: Sampling frequency (being ``1/T``)
        t: Times of STFT for an input signal with `n` samples.
        ShortTimeFFT: Class this method belongs to.
        """
        # Compute the nearest sample index k_p for which t[k_p] == t[p] holds
        p_q, remainder = divmod(k, self.hop)
        if remainder == 0:
            return k
        return p_q * self.hop if left else (p_q + 1) * self.hop
    def delta_f(self) -> float:
        """Width of the frequency bins of the STFT.

        Return the frequency interval `delta_f` = 1 / (`mfft` * `T`).

        See Also
        --------
        delta_t: Time increment of STFT.
        f_pts: Number of points along the frequency axis.
        f: Frequencies values of the STFT.
        mfft: Length of the input for FFT used.
        T: Sampling interval.
        t: Times of STFT for an input signal with `n` samples.
        ShortTimeFFT: Class this property belongs to.
        """
        return 1 / (self.mfft * self.T)

    @property
    def f_pts(self) -> int:
        """Number of points along the frequency axis.

        See Also
        --------
        delta_f: Width of the frequency bins of the STFT.
        f: Frequencies values of the STFT.
        mfft: Length of the input for FFT used.
        ShortTimeFFT: Class this property belongs to.
        """
        return self.mfft // 2 + 1 if self.onesided_fft else self.mfft

    @property
    def onesided_fft(self) -> bool:
        """Return True if a one-sided FFT is used.

        Returns ``True`` if `fft_mode` is either 'onesided' or 'onesided2X'.

        See Also
        --------
        fft_mode: Utilized FFT ('twosided', 'centered', 'onesided' or
                 'onesided2X')
        ShortTimeFFT: Class this property belongs to.
        """
        return self.fft_mode in {'onesided', 'onesided2X'}

    @property
    def f(self) -> np.ndarray:
        """Frequencies values of the STFT.

        A 1d array of length `f_pts` with `delta_f` spaced entries is returned.

        See Also
        --------
        delta_f: Width of the frequency bins of the STFT.
        f_pts: Number of points along the frequency axis.
        mfft: Length of the input for FFT used.
        ShortTimeFFT: Class this property belongs to.
        """
        if self.fft_mode in {'onesided', 'onesided2X'}:
            # Return the real-valued FFT frequencies using rfftfreq from fft_lib
            return fft_lib.rfftfreq(self.mfft, self.T)
        elif self.fft_mode == 'twosided':
            # Return the two-sided FFT frequencies using fftfreq from fft_lib
            return fft_lib.fftfreq(self.mfft, self.T)
        elif self.fft_mode == 'centered':
            # Return the centered FFT frequencies using fftfreq and fftshift from fft_lib
            return fft_lib.fftshift(fft_lib.fftfreq(self.mfft, self.T))
        # This should never happen but makes the Linters happy:
        fft_modes = get_args(FFT_MODE_TYPE)
        # Raise a RuntimeError if the fft_mode is not recognized
        raise RuntimeError(f"{self.fft_mode=} not in {fft_modes}!")
    def _fft_func(self, x: np.ndarray) -> np.ndarray:
        """根据 `fft_mode`, `mfft`, `scaling` 和 `phase_shift` 属性进行 FFT 变换。

        对于多维数组，变换在最后一个轴上进行。
        """
        if self.phase_shift is not None:
            if x.shape[-1] < self.mfft:  # 如果需要，进行零填充
                z_shape = list(x.shape)
                z_shape[-1] = self.mfft - x.shape[-1]
                x = np.hstack((x, np.zeros(z_shape, dtype=x.dtype)))
            p_s = (self.phase_shift + self.m_num_mid) % self.m_num
            x = np.roll(x, -p_s, axis=-1)  # 对数组进行相位移动

        if self.fft_mode == 'twosided':
            return fft_lib.fft(x, n=self.mfft, axis=-1)  # 使用 FFT 进行变换
        if self.fft_mode == 'centered':
            return fft_lib.fftshift(fft_lib.fft(x, self.mfft, axis=-1), axes=-1)  # 中心化后再进行 FFT
        if self.fft_mode == 'onesided':
            return fft_lib.rfft(x, n=self.mfft, axis=-1)  # 单边 FFT 变换
        if self.fft_mode == 'onesided2X':
            X = fft_lib.rfft(x, n=self.mfft, axis=-1)
            # 要么是平方幅度（psd），要么是幅度加倍：
            fac = np.sqrt(2) if self.scaling == 'psd' else 2
            # 对于偶数长度的输入，最后一个条目是单独的：
            X[..., 1: -1 if self.mfft % 2 == 0 else None] *= fac
            return X
        # 这种情况实际上不应发生，但为了让 Linter 不报错：
        fft_modes = get_args(FFT_MODE_TYPE)
        raise RuntimeError(f"{self.fft_mode=} not in {fft_modes}!")

    def _ifft_func(self, X: np.ndarray) -> np.ndarray:
        """_fft_func 的逆操作。

        返回长度为 `m_num` 的数组。如果 FFT 是单边的，则返回浮点数组，否则返回复数数组。
        对于多维数组，变换在最后一个轴上进行。
        """
        if self.fft_mode == 'twosided':
            x = fft_lib.ifft(X, n=self.mfft, axis=-1)  # 反变换为时域信号
        elif self.fft_mode == 'centered':
            x = fft_lib.ifft(fft_lib.ifftshift(X, axes=-1), n=self.mfft, axis=-1)  # 中心化后反变换为时域信号
        elif self.fft_mode == 'onesided':
            x = fft_lib.irfft(X, n=self.mfft, axis=-1)  # 反变换为时域信号（单边）
        elif self.fft_mode == 'onesided2X':
            Xc = X.copy()  # 不修改函数参数
            fac = np.sqrt(2) if self.scaling == 'psd' else 2
            # 对于偶数长度 X，最后一个值没有对应的负值在双边 FFT 中：
            q1 = -1 if self.mfft % 2 == 0 else None
            Xc[..., 1:q1] /= fac
            x = fft_lib.irfft(Xc, n=self.mfft, axis=-1)  # 反变换为时域信号（单边），且校正放大因子
        else:  # 这种情况实际上不应发生，但为了让 Linter 不报错：
            error_str = f"{self.fft_mode=} not in {get_args(FFT_MODE_TYPE)}!"
            raise RuntimeError(error_str)

        if self.phase_shift is None:
            return x[:self.m_num]  # 返回长度为 m_num 的部分
        p_s = (self.phase_shift + self.m_num_mid) % self.m_num
        return np.roll(x, p_s, axis=-1)[:self.m_num]  # 返回经相位移动后的长度为 m_num 的部分
    def extent(self, n: int, axes_seq: Literal['tf', 'ft'] = 'tf',
               center_bins: bool = False) -> tuple[float, float, float, float]:
        """Return minimum and maximum values time-frequency values.

        A tuple with four floats  ``(t0, t1, f0, f1)`` for 'tf' and
        ``(f0, f1, t0, t1)`` for 'ft' is returned describing the corners
        of the time-frequency domain of the `~ShortTimeFFT.stft`.
        That tuple can be passed to `matplotlib.pyplot.imshow` as a parameter
        with the same name.

        Parameters
        ----------
        n : int
            Number of samples in input signal.
        axes_seq : {'tf', 'ft'}
            Return time extent first and then frequency extent or vice-versa.
        center_bins: bool
            If set (default ``False``), the values of the time slots and
            frequency bins are moved from the side the middle. This is useful,
            when plotting the `~ShortTimeFFT.stft` values as step functions,
            i.e., with no interpolation.

        See Also
        --------
        :func:`matplotlib.pyplot.imshow`: Display data as an image.
        :class:`scipy.signal.ShortTimeFFT`: Class this method belongs to.
        """
        # 检查 axes_seq 是否合法，必须是 'tf' 或 'ft'
        if axes_seq not in ('tf', 'ft'):
            raise ValueError(f"Parameter {axes_seq=} not in ['tf', 'ft']!")

        # 根据属性 onesided_fft 和 fft_mode 计算频率区间的边界
        if self.onesided_fft:
            q0, q1 = 0, self.f_pts
        elif self.fft_mode == 'centered':
            # 计算中心对齐 FFT 模式下的频率范围
            q0 = -self.mfft // 2
            q1 = self.mfft // 2 - 1 if self.mfft % 2 == 0 else self.mfft // 2
        else:
            # 抛出错误，要求 fft_mode 属性必须是 'centered', 'onesided' 或 'onesided2X'
            raise ValueError(f"Attribute fft_mode={self.fft_mode} must be " +
                             "in ['centered', 'onesided', 'onesided2X']")

        # 使用 self.p_min 和 self.p_max(n) 计算时间范围的缩写形式
        p0, p1 = self.p_min, self.p_max(n)  # shorthand

        # 根据 center_bins 决定是否将时间和频率槽移到中间位置
        if center_bins:
            t0, t1 = self.delta_t * (p0 - 0.5), self.delta_t * (p1 - 0.5)
            f0, f1 = self.delta_f * (q0 - 0.5), self.delta_f * (q1 - 0.5)
        else:
            t0, t1 = self.delta_t * p0, self.delta_t * p1
            f0, f1 = self.delta_f * q0, self.delta_f * q1

        # 根据 axes_seq 返回不同顺序的时间和频率范围的元组
        return (t0, t1, f0, f1) if axes_seq == 'tf' else (f0, f1, t0, t1)
```