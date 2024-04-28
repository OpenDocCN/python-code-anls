# `.\transformers\audio_utils.py`

```
# coding=utf-8
# 版权归 2023 年的 HuggingFace Inc. 团队和 librosa & torchaudio 作者所有。
#
# 根据 Apache 许可证 2.0 版（“许可证”）进行许可；
# 除非符合许可证的要求，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件
# 均按“原样”分发，不提供任何形式的明示或暗示的保证
# 请参阅许可证以了解特定的语言规定和限制
"""
从音频波形中提取特征的音频处理函数。此代码纯粹使用 numpy 编写，以支持所有框架并消除不必要的依赖项。
"""
import warnings
from typing import Optional, Union

import numpy as np


def hertz_to_mel(freq: Union[float, np.ndarray], mel_scale: str = "htk") -> Union[float, np.ndarray]:
    """
    将频率从赫兹转换为梅尔。

    Args:
        freq (`float` or `np.ndarray`):
            赫兹（Hz）中的频率，或多个频率。
        mel_scale (`str`, *optional*, defaults to `"htk"`):
            要使用的梅尔频率刻度， `"htk"`、`"kaldi"` 或 `"slaney"`。

    Returns:
        `float` or `np.ndarray`: 梅尔刻度上的频率。
    """

    if mel_scale not in ["slaney", "htk", "kaldi"]:
        raise ValueError('mel_scale should be one of "htk", "slaney" or "kaldi".')

    if mel_scale == "htk":
        return 2595.0 * np.log10(1.0 + (freq / 700.0))
    elif mel_scale == "kaldi":
        return 1127.0 * np.log(1.0 + (freq / 700.0))

    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    mels = 3.0 * freq / 200.0

    if isinstance(freq, np.ndarray):
        log_region = freq >= min_log_hertz
        mels[log_region] = min_log_mel + np.log(freq[log_region] / min_log_hertz) * logstep
    elif freq >= min_log_hertz:
        mels = min_log_mel + np.log(freq / min_log_hertz) * logstep

    return mels


def mel_to_hertz(mels: Union[float, np.ndarray], mel_scale: str = "htk") -> Union[float, np.ndarray]:
    """
    将频率从梅尔转换为赫兹。

    Args:
        mels (`float` or `np.ndarray`):
            梅尔中的频率，或多个频率。
        mel_scale (`str`, *optional*, `"htk"`):
            要使用的梅尔频率刻度， `"htk"`、`"kaldi"` 或 `"slaney"`。

    Returns:
        `float` or `np.ndarray`: 赫兹中的频率。
    """

    if mel_scale not in ["slaney", "htk", "kaldi"]:
        raise ValueError('mel_scale should be one of "htk", "slaney" or "kaldi".')

    if mel_scale == "htk":
        return 700.0 * (np.power(10, mels / 2595.0) - 1.0)
    elif mel_scale == "kaldi":
        return 700.0 * (np.exp(mels / 1127.0) - 1.0)

    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    freq = 200.0 * mels / 3.0
    # 检查输入的 mels 是否为 numpy 数组
    if isinstance(mels, np.ndarray):
        # 创建一个布尔数组，标记 mels 中大于等于 min_log_mel 的元素
        log_region = mels >= min_log_mel
        # 对于满足条件的元素，根据公式计算频率值
        freq[log_region] = min_log_hertz * np.exp(logstep * (mels[log_region] - min_log_mel))
    # 如果 mels 不是 numpy 数组，且大于等于 min_log_mel
    elif mels >= min_log_mel:
        # 根据公式计算频率值
        freq = min_log_hertz * np.exp(logstep * (mels - min_log_mel))

    # 返回计算得到的频率值
    return freq
# 创建一个三角滤波器组
def _create_triangular_filter_bank(fft_freqs: np.ndarray, filter_freqs: np.ndarray) -> np.ndarray:
    """
    创建一个三角滤波器组。

    从 *torchaudio* 和 *librosa* 中改编而来。

    Args:
        fft_freqs (`np.ndarray` of shape `(num_frequency_bins,)`):
            FFT 信号频率的离散频率，单位为赫兹。
        filter_freqs (`np.ndarray` of shape `(num_mel_filters,)`):
            要创建的三角滤波器的中心频率，单位为赫兹。

    Returns:
        `np.ndarray` of shape `(num_frequency_bins, num_mel_filters)`
    """
    # 计算中心频率之间的差异
    filter_diff = np.diff(filter_freqs)
    # 计算斜率
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    # 计算向下斜率
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    # 计算向上斜率
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    # 返回向上斜率和向下斜率的最大值
    return np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))


# 创建梅尔滤波器组
def mel_filter_bank(
    num_frequency_bins: int,
    num_mel_filters: int,
    min_frequency: float,
    max_frequency: float,
    sampling_rate: int,
    norm: Optional[str] = None,
    mel_scale: str = "htk",
    triangularize_in_mel_space: bool = False,
) -> np.ndarray:
    """
    创建用于获取梅尔频谱图的频率 bin 转换矩阵。这称为 *梅尔滤波器组*，存在各种实现，这些实现在滤波器的数量、滤波器的形状、
    滤波器的间距、滤波器的带宽以及频谱变形的方式上有所不同。这些特性的目标是近似人类对频率变化的非线性感知。

    文献中引入了不同的梅尔滤波器组。支持以下变体：

    - MFCC FB-20：由 Davis 和 Mermelstein 于 1980 年引入，假设采样频率为 10 kHz，语音带宽为 `[0, 4600]` Hz。
    - MFCC FB-24 HTK：来自剑桥 HMM 工具包 (HTK) (1995)，使用 24 个滤波器的滤波器组，语音带宽为 `[0, 8000]` Hz。假设采样率 ≥ 16 kHz。
    - MFCC FB-40：来自 Slaney 于 1998 年为 MATLAB 编写的 Auditory Toolbox，假设采样率为 16 kHz，语音带宽为 `[133, 6854]` Hz。此版本还包括面积归一化。
    - HFCC-E FB-29 (Human Factor Cepstral Coefficients)：由 Skowronski 和 Harris (2004) 提出，假设采样率为 12.5 kHz，语音带宽为 `[0, 6250]` Hz。

    此代码改编自 *torchaudio* 和 *librosa*。请注意，torchaudio 的 `melscale_fbanks` 的默认参数实现了 `"htk"` 滤波器，而 librosa 使用 `"slaney"` 实现。
    """
    Args:
        num_frequency_bins (`int`):
            频率数量，用于计算频谱图（应与`stft`中相同）。
        num_mel_filters (`int`):
            要生成的梅尔滤波器数量。
        min_frequency (`float`):
            感兴趣的最低频率（单位：赫兹）。
        max_frequency (`float`):
            感兴趣的最高频率（单位：赫兹）。这不应超过`sampling_rate / 2`。
        sampling_rate (`int`):
            音频波形的采样率。
        norm (`str`, *optional*):
            如果为`"slaney"`，则通过梅尔带的宽度来归一化三角形梅尔权重（面积归一化）。
        mel_scale (`str`, *optional*, 默认为`"htk"`):
            要使用的梅尔频率标度，`"htk"`、`"kaldi"`或`"slaney"`。
        triangularize_in_mel_space (`bool`, *optional*, 默认为`False`):
            如果启用此选项，则在梅尔空间而不是频率空间中应用三角形滤波器。为了获得与计算梅尔滤波器时相同的结果，应将其设置为`true`。

    Returns:
        `np.ndarray` of shape (`num_frequency_bins`, `num_mel_filters`):
            三角形滤波器组成的矩阵。这是从频谱图到梅尔频谱图的投影矩阵。
    """
    if norm is not None and norm != "slaney":
        raise ValueError('norm must be one of None or "slaney"')

    # 三角形梅尔滤波器的中心点
    mel_min = hertz_to_mel(min_frequency, mel_scale=mel_scale)
    mel_max = hertz_to_mel(max_frequency, mel_scale=mel_scale)
    mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    filter_freqs = mel_to_hertz(mel_freqs, mel_scale=mel_scale)

    if triangularize_in_mel_space:
        # 在Hz中的FFT bin频率，但在梅尔空间中三角化的滤波器
        fft_bin_width = sampling_rate / (num_frequency_bins * 2)
        fft_freqs = hertz_to_mel(fft_bin_width * np.arange(num_frequency_bins), mel_scale=mel_scale)
        filter_freqs = mel_freqs
    else:
        # 在Hz中的FFT bin频率
        fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)

    mel_filters = _create_triangular_filter_bank(fft_freqs, filter_freqs)

    if norm is not None and norm == "slaney":
        # Slaney风格的梅尔被缩放为近似每个通道的常量能量
        enorm = 2.0 / (filter_freqs[2 : num_mel_filters + 2] - filter_freqs[:num_mel_filters])
        mel_filters *= np.expand_dims(enorm, 0)

    if (mel_filters.max(axis=0) == 0.0).any():
        warnings.warn(
            "至少有一个梅尔滤波器具有所有零值。"
            f"可能设置了过高的`num_mel_filters`值（{num_mel_filters}）。"
            f"或者，可能设置了过低的`num_frequency_bins`值（{num_frequency_bins}）。"
        )

    return mel_filters
# 找到给定窗口长度的最佳 FFT 输入大小。如果窗口长度不是2的幂，则将其向上舍入到下一个2的幂。
def optimal_fft_length(window_length: int) -> int:
    return 2 ** int(np.ceil(np.log2(window_length)))

# 返回包含指定窗口的数组。此窗口旨在与 `stft` 一起使用。
def window_function(
    window_length: int,
    name: str = "hann",
    periodic: bool = True,
    frame_length: Optional[int] = None,
    center: bool = True,
) -> np.ndarray:
    # 如果窗口是周期性的，则长度加1，否则保持原始长度
    length = window_length + 1 if periodic else window_length

    # 根据窗口函数的名称选择相应的窗口类型
    if name == "boxcar":
        window = np.ones(length)
    elif name in ["hamming", "hamming_window"]:
        window = np.hamming(length)
    elif name in ["hann", "hann_window"]:
        window = np.hanning(length)
    elif name in ["povey"]:
        window = np.power(np.hanning(length), 0.85)
    else:
        raise ValueError(f"Unknown window function '{name}'")

    # 如果窗口是周期性的，则去掉最后一个元素
    if periodic:
        window = window[:-1]

    # 如果未提供帧长度，则返回窗口
    if frame_length is None:
        return window

    # 如果窗口长度大于帧长度，则引发异常
    if window_length > frame_length:
        raise ValueError(
            f"Length of the window ({window_length}) may not be larger than frame_length ({frame_length})"
        )

    # 创建一个全零数组，将窗口放置在其中，并根据中心标志确定偏移量
    padded_window = np.zeros(frame_length)
    offset = (frame_length - window_length) // 2 if center else 0
    padded_window[offset : offset + window_length] = window
    return padded_window
# TODO This method does not support batching yet as we are mainly focused on inference.
# 定义一个函数，用于计算音频波形的频谱图，主要用于推断，暂不支持批处理
def spectrogram(
    waveform: np.ndarray,  # 输入的音频波形数据
    window: np.ndarray,  # 窗口函数
    frame_length: int,  # 分帧长度
    hop_length: int,  # 帧移长度
    fft_length: Optional[int] = None,  # FFT长度，默认为None
    power: Optional[float] = 1.0,  # 功率，默认为1.0
    center: bool = True,  # 是否居中，默认为True
    pad_mode: str = "reflect",  # 填充模式，默认为"reflect"
    onesided: bool = True,  # 是否单边，默认为True
    preemphasis: Optional[float] = None,  # 预加重系数，默认为None
    mel_filters: Optional[np.ndarray] = None,  # 梅尔滤波器，默认为None
    mel_floor: float = 1e-10,  # 梅尔底部值，默认为1e-10
    log_mel: Optional[str] = None,  # 对数梅尔，默认为None
    reference: float = 1.0,  # 参考值，默认为1.0
    min_value: float = 1e-10,  # 最小值，默认为1e-10
    db_range: Optional[float] = None,  # dB范围，默认为None
    remove_dc_offset: Optional[bool] = None,  # 是否去除直流偏移，默认为None
    dtype: np.dtype = np.float32,  # 数据类型，默认为np.float32
) -> np.ndarray:
    """
    Calculates a spectrogram over one waveform using the Short-Time Fourier Transform.

    This function can create the following kinds of spectrograms:

      - amplitude spectrogram (`power = 1.0`)
      - power spectrogram (`power = 2.0`)
      - complex-valued spectrogram (`power = None`)
      - log spectrogram (use `log_mel` argument)
      - mel spectrogram (provide `mel_filters`)
      - log-mel spectrogram (provide `mel_filters` and `log_mel`)

    How this works:

      1. The input waveform is split into frames of size `frame_length` that are partially overlapping by `frame_length
         - hop_length` samples.
      2. Each frame is multiplied by the window and placed into a buffer of size `fft_length`.
      3. The DFT is taken of each windowed frame.
      4. The results are stacked into a spectrogram.

    We make a distinction between the following "blocks" of sample data, each of which may have a different lengths:

      - The analysis frame. This is the size of the time slices that the input waveform is split into.
      - The window. Each analysis frame is multiplied by the window to avoid spectral leakage.
      - The FFT input buffer. The length of this determines how many frequency bins are in the spectrogram.

    In this implementation, the window is assumed to be zero-padded to have the same size as the analysis frame. A
    padded window can be obtained from `window_function()`. The FFT input buffer may be larger than the analysis frame,
    typically the next power of two.

    Note: This function is not optimized for speed yet. It should be mostly compatible with `librosa.stft` and
    `torchaudio.functional.transforms.Spectrogram`, although it is more flexible due to the different ways spectrograms
    can be constructed.

    Returns:
        `nd.array` containing a spectrogram of shape `(num_frequency_bins, length)` for a regular spectrogram or shape
        `(num_mel_filters, length)` for a mel spectrogram.
    """
    window_length = len(window)

    if fft_length is None:
        fft_length = frame_length

    if frame_length > fft_length:
        raise ValueError(f"frame_length ({frame_length}) may not be larger than fft_length ({fft_length})")
    # 检查窗口长度是否等于帧长度，如果不相等则抛出数值错误
    if window_length != frame_length:
        raise ValueError(f"Length of the window ({window_length}) must equal frame_length ({frame_length})")

    # 检查跳跃长度是否小于等于零，如果是则抛出数值错误
    if hop_length <= 0:
        raise ValueError("hop_length must be greater than zero")

    # 检查输入波形的维度是否为1，如果不是则抛出数值错误
    if waveform.ndim != 1:
        raise ValueError(f"Input waveform must have only one dimension, shape is {waveform.shape}")

    # 检查输入波形是否为复数类型，如果是则抛出数值错误
    if np.iscomplexobj(waveform):
        raise ValueError("Complex-valued input waveforms are not currently supported")

    # 如果需要中心填充波形，则进行中心填充
    if center:
        padding = [(int(frame_length // 2), int(frame_length // 2))]
        waveform = np.pad(waveform, padding, mode=pad_mode)

    # 将波形和窗口转换为float64类型，因为np.fft内部使用float64
    waveform = waveform.astype(np.float64)
    window = window.astype(np.float64)

    # 将波形分割成帧，每帧长度为frame_length
    num_frames = int(1 + np.floor((waveform.size - frame_length) / hop_length))

    # 计算频率的数量，根据是否单边谱确定
    num_frequency_bins = (fft_length // 2) + 1 if onesided else fft_length
    spectrogram = np.empty((num_frames, num_frequency_bins), dtype=np.complex64)

    # 根据是否单边谱选择使用rfft还是fft
    fft_func = np.fft.rfft if onesided else np.fft.fft
    buffer = np.zeros(fft_length)

    timestep = 0
    for frame_idx in range(num_frames):
        buffer[:frame_length] = waveform[timestep : timestep + frame_length]

        # 如果需要去除直流偏移，则进行处理
        if remove_dc_offset:
            buffer[:frame_length] = buffer[:frame_length] - buffer[:frame_length].mean()

        # 如果有预加重参数，则进行预加重处理
        if preemphasis is not None:
            buffer[1:frame_length] -= preemphasis * buffer[: frame_length - 1]
            buffer[0] *= 1 - preemphasis

        buffer[:frame_length] *= window

        spectrogram[frame_idx] = fft_func(buffer)
        timestep += hop_length

    # 注意：** 操作比 np.power 更快
    if power is not None:
        spectrogram = np.abs(spectrogram, dtype=np.float64) ** power

    spectrogram = spectrogram.T

    # 如果有Mel滤波器，则应用
    if mel_filters is not None:
        spectrogram = np.maximum(mel_floor, np.dot(mel_filters.T, spectrogram))

    # 如果有功率和对数Mel参数，则应用
    if power is not None and log_mel is not None:
        if log_mel == "log":
            spectrogram = np.log(spectrogram)
        elif log_mel == "log10":
            spectrogram = np.log10(spectrogram)
        elif log_mel == "dB":
            if power == 1.0:
                spectrogram = amplitude_to_db(spectrogram, reference, min_value, db_range)
            elif power == 2.0:
                spectrogram = power_to_db(spectrogram, reference, min_value, db_range)
            else:
                raise ValueError(f"Cannot use log_mel option '{log_mel}' with power {power}")
        else:
            raise ValueError(f"Unknown log_mel option: {log_mel}")

        spectrogram = np.asarray(spectrogram, dtype)

    return spectrogram
# 将功率谱图转换为分贝刻度。这计算 `10 * log10(spectrogram / reference)`，使用基本的对数性质以确保数值稳定性。
def power_to_db(
    spectrogram: np.ndarray,
    reference: float = 1.0,
    min_value: float = 1e-10,
    db_range: Optional[float] = None,
) -> np.ndarray:
    """
    Converts a power spectrogram to the decibel scale. This computes `10 * log10(spectrogram / reference)`, using basic
    logarithm properties for numerical stability.

    The motivation behind applying the log function on the (mel) spectrogram is that humans do not hear loudness on a
    linear scale. Generally to double the perceived volume of a sound we need to put 8 times as much energy into it.
    This means that large variations in energy may not sound all that different if the sound is loud to begin with.
    This compression operation makes the (mel) spectrogram features match more closely what humans actually hear.

    Based on the implementation of `librosa.power_to_db`.

    Args:
        spectrogram (`np.ndarray`):
            The input power (mel) spectrogram. Note that a power spectrogram has the amplitudes squared!
        reference (`float`, *optional*, defaults to 1.0):
            Sets the input spectrogram value that corresponds to 0 dB. For example, use `np.max(spectrogram)` to set
            the loudest part to 0 dB. Must be greater than zero.
        min_value (`float`, *optional*, defaults to `1e-10`):
            The spectrogram will be clipped to this minimum value before conversion to decibels, to avoid taking
            `log(0)`. The default of `1e-10` corresponds to a minimum of -100 dB. Must be greater than zero.
        db_range (`float`, *optional*):
            Sets the maximum dynamic range in decibels. For example, if `db_range = 80`, the difference between the
            peak value and the smallest value will never be more than 80 dB. Must be greater than zero.

    Returns:
        `np.ndarray`: the spectrogram in decibels
    """
    # 如果参考值小于等于0，则引发值错误
    if reference <= 0.0:
        raise ValueError("reference must be greater than zero")
    # 如果最小值小于等于0，则引发值错误
    if min_value <= 0.0:
        raise ValueError("min_value must be greater than zero")

    # 参考值取最小值和参考值中的最大值
    reference = max(min_value, reference)

    # 将谱图剪切到最小值和无穷大之间
    spectrogram = np.clip(spectrogram, a_min=min_value, a_max=None)
    # 计算谱图的分贝值
    spectrogram = 10.0 * (np.log10(spectrogram) - np.log10(reference))

    # 如果动态范围不为None
    if db_range is not None:
        # 如果动态范围小于等于0，则引发值错误
        if db_range <= 0.0:
            raise ValueError("db_range must be greater than zero")
        # 将谱图剪切到最大值和无穷大之间
        spectrogram = np.clip(spectrogram, a_min=spectrogram.max() - db_range, a_max=None)

    # 返回转换后的谱图
    return spectrogram


# 将振幅谱图转换为分贝刻度。这计算 `20 * log10(spectrogram / reference)`，使用基本的对数性质以确保数值稳定性。
def amplitude_to_db(
    spectrogram: np.ndarray,
    reference: float = 1.0,
    min_value: float = 1e-5,
    db_range: Optional[float] = None,
) -> np.ndarray:
    """
    Converts an amplitude spectrogram to the decibel scale. This computes `20 * log10(spectrogram / reference)`, using
    basic logarithm properties for numerical stability.

    The motivation behind applying the log function on the (mel) spectrogram is that humans do not hear loudness on a
    def amplitude_to_db(spectrogram, reference=1.0, min_value=1e-5, db_range=None):
        """
        Convert an amplitude spectrogram to dB (decibels) scale.
    
        This function converts the input amplitude (mel) spectrogram to dB scale, which is a logarithmic scale used to represent sound intensity. It compresses the dynamic range of the spectrogram to match more closely what humans actually hear.
    
        Args:
            spectrogram (`np.ndarray`):
                The input amplitude (mel) spectrogram.
            reference (`float`, *optional*, defaults to 1.0):
                Sets the input spectrogram value that corresponds to 0 dB. For example, use `np.max(spectrogram)` to set
                the loudest part to 0 dB. Must be greater than zero.
            min_value (`float`, *optional*, defaults to `1e-5`):
                The spectrogram will be clipped to this minimum value before conversion to decibels, to avoid taking
                `log(0)`. The default of `1e-5` corresponds to a minimum of -100 dB. Must be greater than zero.
            db_range (`float`, *optional*):
                Sets the maximum dynamic range in decibels. For example, if `db_range = 80`, the difference between the
                peak value and the smallest value will never be more than 80 dB. Must be greater than zero.
    
        Returns:
            `np.ndarray`: the spectrogram in decibels
        """
        # Check if reference is valid
        if reference <= 0.0:
            raise ValueError("reference must be greater than zero")
        # Check if min_value is valid
        if min_value <= 0.0:
            raise ValueError("min_value must be greater than zero")
    
        # Ensure reference is not less than min_value
        reference = max(min_value, reference)
    
        # Clip the spectrogram to min_value
        spectrogram = np.clip(spectrogram, a_min=min_value, a_max=None)
        # Convert amplitude spectrogram to dB scale
        spectrogram = 20.0 * (np.log10(spectrogram) - np.log10(reference))
    
        # If db_range is specified, clip the spectrogram to the specified dynamic range
        if db_range is not None:
            if db_range <= 0.0:
                raise ValueError("db_range must be greater than zero")
            spectrogram = np.clip(spectrogram, a_min=spectrogram.max() - db_range, a_max=None)
    
        return spectrogram
# 以下代码为不推荐使用的函数

# 获取梅尔滤波器组
def get_mel_filter_banks(
    nb_frequency_bins: int,
    nb_mel_filters: int,
    frequency_min: float,
    frequency_max: float,
    sample_rate: int,
    norm: Optional[str] = None,
    mel_scale: str = "htk",
) -> np.array:
    # 发出警告，提醒函数将在 Transformers 版本 4.31.0 中被移除
    warnings.warn(
        "The function `get_mel_filter_banks` is deprecated and will be removed in version 4.31.0 of Transformers",
        FutureWarning,
    )
    # 调用 mel_filter_bank 函数获取梅尔滤波器组
    return mel_filter_bank(
        num_frequency_bins=nb_frequency_bins,
        num_mel_filters=nb_mel_filters,
        min_frequency=frequency_min,
        max_frequency=frequency_max,
        sampling_rate=sample_rate,
        norm=norm,
        mel_scale=mel_scale,
    )

# 分帧波形
def fram_wave(waveform: np.array, hop_length: int = 160, fft_window_size: int = 400, center: bool = True):
    """
    In order to compute the short time fourier transform, the waveform needs to be split in overlapping windowed
    segments called `frames`.

    The window length (window_length) defines how much of the signal is contained in each frame, while the hop length
    defines the step between the beginning of each new frame.


    Args:
        waveform (`np.array` of shape `(sample_length,)`):
            The raw waveform which will be split into smaller chunks.
        hop_length (`int`, *optional*, defaults to 160):
            Step between each window of the waveform.
        fft_window_size (`int`, *optional*, defaults to 400):
            Defines the size of the window.
        center (`bool`, defaults to `True`):
            Whether or not to center each frame around the middle of the frame. Centering is done by reflecting the
            waveform on the left and on the right.

    Return:
        framed_waveform (`np.array` of shape `(waveform.shape // hop_length , fft_window_size)`):
            The framed waveforms that can be fed to `np.fft`.
    """
    # 发出警告，提醒函数将在 Transformers 版本 4.31.0 中被移除
    warnings.warn(
        "The function `fram_wave` is deprecated and will be removed in version 4.31.0 of Transformers",
        FutureWarning,
    )
    # 初始化一个空列表用于存储帧
    frames = []
    # 遍历波形的每个采样点，以指定的跳跃长度作为步长
    for i in range(0, waveform.shape[0] + 1, hop_length):
        # 如果需要在中心帧上操作
        if center:
            # 计算窗口的一半大小，确保在边界时不会超出范围
            half_window = (fft_window_size - 1) // 2 + 1
            # 确定当前帧的起始和结束位置
            start = i - half_window if i > half_window else 0
            end = i + half_window if i < waveform.shape[0] - half_window else waveform.shape[0]
            # 提取当前帧的波形数据
            frame = waveform[start:end]
            # 如果起始位置为0，需要在左侧填充
            if start == 0:
                padd_width = (-i + half_window, 0)
                frame = np.pad(frame, pad_width=padd_width, mode="reflect")
            # 如果结束位置为波形长度，需要在右侧填充
            elif end == waveform.shape[0]:
                padd_width = (0, (i - waveform.shape[0] + half_window))
                frame = np.pad(frame, pad_width=padd_width, mode="reflect")
        # 如果不需要在中心帧上操作
        else:
            # 提取以当前采样点为起始的窗口大小的波形数据
            frame = waveform[i : i + fft_window_size]
            # 计算帧的宽度
            frame_width = frame.shape[0]
            # 如果帧的宽度小于窗口大小，需要在右侧填充0
            if frame_width < waveform.shape[0]:
                frame = np.lib.pad(
                    frame, pad_width=(0, fft_window_size - frame_width), mode="constant", constant_values=0
                )
        # 将处理后的帧添加到帧列表中
        frames.append(frame)

    # 将帧列表转换为NumPy数组
    frames = np.stack(frames, 0)
    # 返回帧数组
    return frames
def stft(frames: np.array, windowing_function: np.array, fft_window_size: int = None):
    """
    Calculates the complex Short-Time Fourier Transform (STFT) of the given framed signal. Should give the same results
    as `torch.stft`.

    Args:
        frames (`np.array` of dimension `(num_frames, fft_window_size)`):
            A framed audio signal obtained using `audio_utils.fram_wav`.
        windowing_function (`np.array` of dimension `(nb_frequency_bins, nb_mel_filters)`:
            A array representing the function that will be used to reduce the amplitude of the discontinuities at the
            boundaries of each frame when computing the STFT. Each frame will be multiplied by the windowing_function.
            For more information on the discontinuities, called *Spectral leakage*, refer to [this
            tutorial]https://download.ni.com/evaluation/pxi/Understanding%20FFTs%20and%20Windowing.pdf
        fft_window_size (`int`, *optional*):
            Size of the window on which the Fourier transform is applied. This controls the frequency resolution of the
            spectrogram. 400 means that the Fourier transform is computed on windows of 400 samples. The number of
            frequency bins (`nb_frequency_bins`) used to divide the window into equal strips is equal to
            `(1+fft_window_size)//2`. An increase of the fft_window_size slows the calculus time proportionally.

    Example:

    ```python
    >>> from transformers.audio_utils import stft, fram_wave
    >>> import numpy as np

    >>> audio = np.random.rand(50)
    >>> fft_window_size = 10
    >>> hop_length = 2
    >>> framed_audio = fram_wave(audio, hop_length, fft_window_size)
    >>> spectrogram = stft(framed_audio, np.hanning(fft_window_size + 1))
    ```

    Returns:
        spectrogram (`np.ndarray`):
            A spectrogram of shape `(num_frames, nb_frequency_bins)` obtained using the STFT algorithm
    """
    # 发出警告，提醒用户函数将在未来版本中被移除
    warnings.warn(
        "The function `stft` is deprecated and will be removed in version 4.31.0 of Transformers",
        FutureWarning,
    )
    # 获取每帧的大小
    frame_size = frames.shape[1]

    # 如果未指定 fft_window_size，则使用帧的大小
    if fft_window_size is None:
        fft_window_size = frame_size

    # 如果 fft_window_size 小于帧的大小，则引发 ValueError
    if fft_window_size < frame_size:
        raise ValueError("FFT size must greater or equal the frame size")
    # 计算存储的 FFT bin 的数量
    nb_frequency_bins = (fft_window_size >> 1) + 1

    # 创建一个空的数组来存储频谱图，使用复数数据类型
    spectrogram = np.empty((len(frames), nb_frequency_bins), dtype=np.complex64)
    # 创建一个全零数组来存储 FFT 信号
    fft_signal = np.zeros(fft_window_size)

    # 对于每个帧，执行 STFT 计算
    for f, frame in enumerate(frames):
        # 如果存在窗函数，则将窗函数应用于帧
        if windowing_function is not None:
            # 将窗函数乘以帧，结果存储到 fft_signal 中
            np.multiply(frame, windowing_function, out=fft_signal[:frame_size])
        else:
            # 如果没有窗函数，则直接将帧复制到 fft_signal 中
            fft_signal[:frame_size] = frame
        # 对 fft_signal 执行 FFT，结果存储到 spectrogram 中
        spectrogram[f] = np.fft.fft(fft_signal, axis=0)[:nb_frequency_bins]
    # 返回频谱图的转置（行列转置）
    return spectrogram.T
```