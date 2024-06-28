# `.\audio_utils.py`

```py
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team and the librosa & torchaudio authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Audio processing functions to extract features from audio waveforms. This code is pure numpy to support all frameworks
and remove unnecessary dependencies.
"""
import warnings
from typing import Optional, Tuple, Union

import numpy as np


def hertz_to_mel(freq: Union[float, np.ndarray], mel_scale: str = "htk") -> Union[float, np.ndarray]:
    """
    Convert frequency from hertz to mels.

    Args:
        freq (`float` or `np.ndarray`):
            The frequency, or multiple frequencies, in hertz (Hz).
        mel_scale (`str`, *optional*, defaults to `"htk"`):
            The mel frequency scale to use, `"htk"`, `"kaldi"` or `"slaney"`.

    Returns:
        `float` or `np.ndarray`: The frequencies on the mel scale.
    """

    if mel_scale not in ["slaney", "htk", "kaldi"]:
        raise ValueError('mel_scale should be one of "htk", "slaney" or "kaldi".')
    
    # Convert frequencies to mels based on the specified mel scale
    if mel_scale == "htk":
        return 2595.0 * np.log10(1.0 + (freq / 700.0))
    elif mel_scale == "kaldi":
        return 1127.0 * np.log(1.0 + (freq / 700.0))

    # For "slaney" scale, compute mels using specific formulas
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    mels = 3.0 * freq / 200.0

    # Handle cases where freq is a numpy array or a scalar
    if isinstance(freq, np.ndarray):
        log_region = freq >= min_log_hertz
        mels[log_region] = min_log_mel + np.log(freq[log_region] / min_log_hertz) * logstep
    elif freq >= min_log_hertz:
        mels = min_log_mel + np.log(freq / min_log_hertz) * logstep

    return mels


def mel_to_hertz(mels: Union[float, np.ndarray], mel_scale: str = "htk") -> Union[float, np.ndarray]:
    """
    Convert frequency from mels to hertz.

    Args:
        mels (`float` or `np.ndarray`):
            The frequency, or multiple frequencies, in mels.
        mel_scale (`str`, *optional*, `"htk"`):
            The mel frequency scale to use, `"htk"`, `"kaldi"` or `"slaney"`.

    Returns:
        `float` or `np.ndarray`: The frequencies in hertz.
    """

    if mel_scale not in ["slaney", "htk", "kaldi"]:
        raise ValueError('mel_scale should be one of "htk", "slaney" or "kaldi".')
    
    # Convert mels to frequencies based on the specified mel scale
    if mel_scale == "htk":
        return 700.0 * (np.power(10, mels / 2595.0) - 1.0)
    elif mel_scale == "kaldi":
        return 700.0 * (np.exp(mels / 1127.0) - 1.0)

    # For "slaney" scale, compute frequencies using specific formulas
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    freq = 200.0 * mels / 3.0

    return freq
    # 如果输入的 mels 是一个 NumPy 数组
    if isinstance(mels, np.ndarray):
        # 创建一个布尔数组 log_region，标记所有 mels 中大于等于 min_log_mel 的元素位置
        log_region = mels >= min_log_mel
        # 对于 log_region 中为 True 的位置，根据公式计算对应的频率值并存储在 freq 中
        freq[log_region] = min_log_hertz * np.exp(logstep * (mels[log_region] - min_log_mel))
    # 如果输入的 mels 不是 NumPy 数组，而是单个数值
    elif mels >= min_log_mel:
        # 根据公式计算频率值并存储在 freq 中
        freq = min_log_hertz * np.exp(logstep * (mels - min_log_mel))
    
    # 返回计算得到的频率 freq
    return freq
# 创建一个函数用于将频率从赫兹转换为分数音阶数
def hertz_to_octave(
    freq: Union[float, np.ndarray], tuning: Optional[float] = 0.0, bins_per_octave: Optional[int] = 12
):
    """
    Convert frequency from hertz to fractional octave numbers.
    Adapted from *librosa*.

    Args:
        freq (`float` or `np.ndarray`):
            The frequency, or multiple frequencies, in hertz (Hz).
        tuning (`float`, defaults to `0.`):
            Tuning deviation from the Stuttgart pitch (A440) in (fractional) bins per octave.
        bins_per_octave (`int`, defaults to `12`):
            Number of bins per octave.

    Returns:
        `float` or `np.ndarray`: The frequencies on the octave scale.
    """
    # 计算按照斯图加特音高(A440)偏移后的基准频率
    stuttgart_pitch = 440.0 * 2.0 ** (tuning / bins_per_octave)
    # 计算频率的分数音阶数
    octave = np.log2(freq / (float(stuttgart_pitch) / 16))
    return octave


# 创建一个函数用于生成三角形滤波器组成的滤波器组
def _create_triangular_filter_bank(fft_freqs: np.ndarray, filter_freqs: np.ndarray) -> np.ndarray:
    """
    Creates a triangular filter bank.

    Adapted from *torchaudio* and *librosa*.

    Args:
        fft_freqs (`np.ndarray` of shape `(num_frequency_bins,)`):
            Discrete frequencies of the FFT bins in Hz.
        filter_freqs (`np.ndarray` of shape `(num_mel_filters,)`):
            Center frequencies of the triangular filters to create, in Hz.

    Returns:
        `np.ndarray` of shape `(num_frequency_bins, num_mel_filters)`
    """
    # 计算滤波器中心频率的差异
    filter_diff = np.diff(filter_freqs)
    # 计算每个频率 bin 对每个滤波器的下坡斜率
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    # 计算每个频率 bin 对每个滤波器的上坡斜率
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    # 返回下坡和上坡斜率中的较小值
    return np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))


# 创建一个函数用于生成色度滤波器组成的滤波器组
def chroma_filter_bank(
    num_frequency_bins: int,
    num_chroma: int,
    sampling_rate: int,
    tuning: float = 0.0,
    power: Optional[float] = 2.0,
    weighting_parameters: Optional[Tuple[float]] = (5.0, 2),
    start_at_c_chroma: Optional[bool] = True,
):
    """
    Creates a chroma filter bank, i.e a linear transformation to project spectrogram bins onto chroma bins.

    Adapted from *librosa*.
    """
    # 在默认情况下，如果从C音开始，色度数设置为12
    # 否则，设置为24
    # 每个bin中的频率
    # 获取FFT的频率bins，不包括直流分量
    frequencies = np.linspace(0, sampling_rate, num_frequency_bins, endpoint=False)[1:]

    # 将频率bins转换为chroma bins，基于给定的调谐和每个八度内的bins数
    freq_bins = num_chroma * hertz_to_octave(frequencies, tuning=tuning, bins_per_octave=num_chroma)

    # 为0 Hz的频率bin赋值，假设它比bin 1低1.5个八度
    # （这样chroma就会从bin 1旋转50%，bin宽度较宽）
    freq_bins = np.concatenate(([freq_bins[0] - 1.5 * num_chroma], freq_bins))

    # 计算每个bin的宽度
    bins_width = np.concatenate((np.maximum(freq_bins[1:] - freq_bins[:-1], 1.0), [1]))

    # 创建chroma滤波器，计算每个频率bin与每个chroma bin之间的差异
    chroma_filters = np.subtract.outer(freq_bins, np.arange(0, num_chroma, dtype="d")).T

    # 将chroma滤波器投影到范围 -num_chroma/2 .. num_chroma/2
    # 添加一个固定偏移量确保所有传递给rem的值都是正数
    num_chroma2 = np.round(float(num_chroma) / 2)
    chroma_filters = np.remainder(chroma_filters + num_chroma2 + 10 * num_chroma, num_chroma) - num_chroma2

    # 创建高斯形状的chroma滤波器，使它们更窄
    chroma_filters = np.exp(-0.5 * (2 * chroma_filters / np.tile(bins_width, (num_chroma, 1))) ** 2)

    # 如果指定了power，则对每列进行归一化
    if power is not None:
        chroma_filters = chroma_filters / np.sum(chroma_filters**power, axis=0, keepdims=True) ** (1.0 / power)

    # 如果指定了weighting_parameters，则应用高斯加权
    if weighting_parameters is not None:
        center, half_width = weighting_parameters
        chroma_filters *= np.tile(
            np.exp(-0.5 * (((freq_bins / num_chroma - center) / half_width) ** 2)),
            (num_chroma, 1),
        )

    # 如果start_at_c_chroma为True，则将chroma_filters数组向左滚动，以从'C'音调类开始
    if start_at_c_chroma:
        chroma_filters = np.roll(chroma_filters, -3 * (num_chroma // 12), axis=0)

    # 去除别名列，并复制以确保行连续性，返回numpy数组
    return np.ascontiguousarray(chroma_filters[:, : int(1 + num_frequency_bins / 2)])
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
    创建用于生成梅尔频谱图的频率 bin 转换矩阵，称为梅尔滤波器组。存在多种实现方式，这些方式在滤波器数量、滤波器形状、
    滤波器间距、滤波器带宽以及频谱扭曲方式上都有所不同。这些特性旨在近似人类对频率变化的非线性感知。

    文献中引入了不同的梅尔滤波器组变体。以下几种变体是支持的：

    - MFCC FB-20: 由Davis和Mermelstein于1980年引入，假设采样频率为10 kHz，语音带宽为 `[0, 4600]` Hz。
    - MFCC FB-24 HTK: 来自于剑桥HMM工具包（HTK）（1995年），使用24个滤波器的滤波器组，语音带宽为 `[0, 8000]` Hz。
      假设采样率 ≥ 16 kHz。
    - MFCC FB-40: 来自于Slaney在1998年为MATLAB编写的听觉工具箱，假设采样率为16 kHz，语音带宽为 `[133, 6854]` Hz。
      此版本还包括区域归一化。
    - HFCC-E FB-29（人因谱系数）：由Skowronski和Harris于2004年提出，假设采样率为12.5 kHz，语音带宽为 `[0, 6250]` Hz。

    此代码改编自 *torchaudio* 和 *librosa*。请注意，torchaudio 的 `melscale_fbanks` 的默认参数实现了 `"htk"` 滤波器，
    而 librosa 使用 `"slaney"` 实现。

    Args:
        num_frequency_bins (`int`):
            用于计算频谱图的频率数量（应与 `stft` 中的相同）。
        num_mel_filters (`int`):
            要生成的梅尔滤波器数量。
        min_frequency (`float`):
            兴趣的最低频率（单位：Hz）。
        max_frequency (`float`):
            兴趣的最高频率（单位：Hz）。不应超过 `sampling_rate / 2`。
        sampling_rate (`int`):
            音频波形的采样率。
        norm (`str`, *optional*):
            如果是 `"slaney"`，将三角形梅尔权重除以梅尔带宽的宽度（区域归一化）。
        mel_scale (`str`, *optional*, defaults to `"htk"`):
            要使用的梅尔频率刻度，可选 `"htk"`、`"kaldi"` 或 `"slaney"`。
        triangularize_in_mel_space (`bool`, *optional*, defaults to `False`):
            如果启用此选项，则在梅尔空间而不是频率空间中应用三角形滤波器。在计算梅尔滤波器时应将其设置为 `True`，以便获得与 `torchaudio` 相同的结果。
    """
    # 在这里实现梅尔滤波器组的计算和返回
    pass
    if norm is not None and norm != "slaney":
        # 如果指定了 norm 参数但不是 "slaney"，则抛出数值错误异常
        raise ValueError('norm must be one of None or "slaney"')

    # 计算三角形梅尔滤波器的中心点频率
    mel_min = hertz_to_mel(min_frequency, mel_scale=mel_scale)
    mel_max = hertz_to_mel(max_frequency, mel_scale=mel_scale)
    mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    # 将梅尔频率转换为普通频率
    filter_freqs = mel_to_hertz(mel_freqs, mel_scale=mel_scale)

    if triangularize_in_mel_space:
        # 如果在梅尔空间中进行三角化，则使用FFT频率的梅尔频率宽度
        fft_bin_width = sampling_rate / (num_frequency_bins * 2)
        fft_freqs = hertz_to_mel(fft_bin_width * np.arange(num_frequency_bins), mel_scale=mel_scale)
        filter_freqs = mel_freqs
    else:
        # 否则使用普通的FFT频率范围
        fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)

    # 创建三角形滤波器组成的滤波器组
    mel_filters = _create_triangular_filter_bank(fft_freqs, filter_freqs)

    if norm is not None and norm == "slaney":
        # 如果使用 Slaney 格式的梅尔滤波器，则进行能量归一化
        enorm = 2.0 / (filter_freqs[2 : num_mel_filters + 2] - filter_freqs[:num_mel_filters])
        mel_filters *= np.expand_dims(enorm, 0)

    if (mel_filters.max(axis=0) == 0.0).any():
        # 如果有至少一个梅尔滤波器全部为零，则发出警告
        warnings.warn(
            "At least one mel filter has all zero values. "
            f"The value for `num_mel_filters` ({num_mel_filters}) may be set too high. "
            f"Or, the value for `num_frequency_bins` ({num_frequency_bins}) may be set too low."
        )

    # 返回三角形滤波器组成的矩阵，用于从频谱图到梅尔频谱图的投影
    return mel_filters
def optimal_fft_length(window_length: int) -> int:
    """
    Finds the best FFT input size for a given `window_length`. This function takes a given window length and, if not
    already a power of two, rounds it up to the next power or two.

    The FFT algorithm works fastest when the length of the input is a power of two, which may be larger than the size
    of the window or analysis frame. For example, if the window is 400 samples, using an FFT input size of 512 samples
    is more optimal than an FFT size of 400 samples. Using a larger FFT size does not affect the detected frequencies,
    it simply gives a higher frequency resolution (i.e. the frequency bins are smaller).
    """
    # 计算大于等于 `window_length` 的最小的 2 的幂次方数
    return 2 ** int(np.ceil(np.log2(window_length)))


def window_function(
    window_length: int,
    name: str = "hann",
    periodic: bool = True,
    frame_length: Optional[int] = None,
    center: bool = True,
) -> np.ndarray:
    """
    Returns an array containing the specified window. This window is intended to be used with `stft`.

    The following window types are supported:

        - `"boxcar"`: a rectangular window
        - `"hamming"`: the Hamming window
        - `"hann"`: the Hann window
        - `"povey"`: the Povey window

    Args:
        window_length (`int`):
            The length of the window in samples.
        name (`str`, *optional*, defaults to `"hann"`):
            The name of the window function.
        periodic (`bool`, *optional*, defaults to `True`):
            Whether the window is periodic or symmetric.
        frame_length (`int`, *optional*):
            The length of the analysis frames in samples. Provide a value for `frame_length` if the window is smaller
            than the frame length, so that it will be zero-padded.
        center (`bool`, *optional*, defaults to `True`):
            Whether to center the window inside the FFT buffer. Only used when `frame_length` is provided.

    Returns:
        `np.ndarray` of shape `(window_length,)` or `(frame_length,)` containing the window.
    """
    # 如果 `periodic` 为真，则增加窗口长度以适应周期性需求
    length = window_length + 1 if periodic else window_length

    if name == "boxcar":
        # 返回一个长度为 `length` 的全一数组，即矩形窗口
        window = np.ones(length)
    elif name in ["hamming", "hamming_window"]:
        # 返回一个 Hamming 窗口
        window = np.hamming(length)
    elif name in ["hann", "hann_window"]:
        # 返回一个 Hann 窗口
        window = np.hanning(length)
    elif name in ["povey"]:
        # 返回一个经过幂次变换的 Hann 窗口
        window = np.power(np.hanning(length), 0.85)
    else:
        # 如果窗口类型未知，则抛出错误
        raise ValueError(f"Unknown window function '{name}'")

    if periodic:
        # 如果窗口需要周期性，则移除最后一个元素
        window = window[:-1]

    if frame_length is None:
        # 如果没有提供 `frame_length`，直接返回窗口数组
        return window

    if window_length > frame_length:
        # 如果窗口长度大于 `frame_length`，则抛出错误
        raise ValueError(
            f"Length of the window ({window_length}) may not be larger than frame_length ({frame_length})"
        )

    # 创建一个长度为 `frame_length` 的零数组，并将窗口数组放置到合适的位置
    padded_window = np.zeros(frame_length)
    offset = (frame_length - window_length) // 2 if center else 0
    padded_window[offset : offset + window_length] = window
    return padded_window
# TODO This method does not support batching yet as we are mainly focused on inference.
def spectrogram(
    waveform: np.ndarray,
    window: np.ndarray,
    frame_length: int,
    hop_length: int,
    fft_length: Optional[int] = None,
    power: Optional[float] = 1.0,
    center: bool = True,
    pad_mode: str = "reflect",
    onesided: bool = True,
    preemphasis: Optional[float] = None,
    mel_filters: Optional[np.ndarray] = None,
    mel_floor: float = 1e-10,
    log_mel: Optional[str] = None,
    reference: float = 1.0,
    min_value: float = 1e-10,
    db_range: Optional[float] = None,
    remove_dc_offset: Optional[bool] = None,
    dtype: np.dtype = np.float32,
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

      1. The input waveform is split into frames of size `frame_length` that are partially overlapping by `hop_length` samples.
      2. Each frame is multiplied by the window and placed into a buffer of size `fft_length`.
      3. The DFT is taken of each windowed frame.
      4. The results are stacked into a spectrogram.

    We make a distinction between the following "blocks" of sample data, each of which may have different lengths:

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
    # Determine the length of the window
    window_length = len(window)

    # If fft_length is not provided, set it equal to frame_length
    if fft_length is None:
        fft_length = frame_length

    # Check if frame_length is greater than fft_length; raise ValueError if true
    if frame_length > fft_length:
        raise ValueError(f"frame_length ({frame_length}) may not be larger than fft_length ({fft_length})")
    # 检查窗口长度与帧长度是否相等，若不相等则引发数值错误异常
    if window_length != frame_length:
        raise ValueError(f"Length of the window ({window_length}) must equal frame_length ({frame_length})")

    # 检查跳跃长度是否小于等于零，若是则引发数值错误异常
    if hop_length <= 0:
        raise ValueError("hop_length must be greater than zero")

    # 检查波形的维度是否为一维，若不是则引发数值错误异常，同时给出其形状信息
    if waveform.ndim != 1:
        raise ValueError(f"Input waveform must have only one dimension, shape is {waveform.shape}")

    # 检查波形是否为复数类型对象，若是则引发数值错误异常，因为目前不支持复数类型的波形
    if np.iscomplexobj(waveform):
        raise ValueError("Complex-valued input waveforms are not currently supported")

    # 若功率参数为None且mel滤波器不为None，则引发数值错误异常，指出不支持复数谱图计算的情况
    if power is None and mel_filters is not None:
        raise ValueError(
            "You have provided `mel_filters` but `power` is `None`. Mel spectrogram computation is not yet supported for complex-valued spectrogram."
            "Specify `power` to fix this issue."
        )

    # 若center参数为True，则在波形两端进行中心填充，填充长度为帧长度的一半
    if center:
        padding = [(int(frame_length // 2), int(frame_length // 2))]
        waveform = np.pad(waveform, padding, mode=pad_mode)

    # 将波形数据类型转换为float64，因为np.fft内部使用float64类型进行计算
    waveform = waveform.astype(np.float64)
    window = window.astype(np.float64)

    # 将波形分割为帧，每帧长度为frame_length，计算帧的数量
    num_frames = int(1 + np.floor((waveform.size - frame_length) / hop_length))

    # 计算FFT后的频率bins的数量，如果是单边谱则为(fft_length // 2) + 1，否则为fft_length
    num_frequency_bins = (fft_length // 2) + 1 if onesided else fft_length
    # 创建一个空的复数64位数组作为谱图，大小为(num_frames, num_frequency_bins)
    spectrogram = np.empty((num_frames, num_frequency_bins), dtype=np.complex64)

    # 根据是否单边谱选择FFT函数，rfft比fft更快
    fft_func = np.fft.rfft if onesided else np.fft.fft
    # 创建一个长度为fft_length的零填充数组，用于存储FFT输入数据
    buffer = np.zeros(fft_length)

    timestep = 0
    # 遍历每一帧进行FFT计算
    for frame_idx in range(num_frames):
        # 将波形数据填充到buffer中的前frame_length位置
        buffer[:frame_length] = waveform[timestep : timestep + frame_length]

        # 若remove_dc_offset为True，则移除直流偏移
        if remove_dc_offset:
            buffer[:frame_length] = buffer[:frame_length] - buffer[:frame_length].mean()

        # 若preemphasis参数不为None，则对帧进行预加重处理
        if preemphasis is not None:
            buffer[1:frame_length] -= preemphasis * buffer[: frame_length - 1]
            buffer[0] *= 1 - preemphasis

        # 对帧数据应用窗口函数
        buffer[:frame_length] *= window

        # 计算FFT并将结果存入spectrogram中的当前帧索引位置
        spectrogram[frame_idx] = fft_func(buffer)
        timestep += hop_length

    # 若power参数不为None，则对谱图进行幅度平方计算，使用**操作符比np.power更快
    if power is not None:
        spectrogram = np.abs(spectrogram, dtype=np.float64) ** power

    # 将谱图转置，使得频率bins成为第一维度
    spectrogram = spectrogram.T

    # 若mel_filters参数不为None，则进行mel滤波器的应用，并确保谱图值不低于mel_floor
    if mel_filters is not None:
        spectrogram = np.maximum(mel_floor, np.dot(mel_filters.T, spectrogram))
    # 检查是否同时指定了 power 和 log_mel 参数
    if power is not None and log_mel is not None:
        # 如果 log_mel 参数为 "log"，应用自然对数变换到 spectrogram
        if log_mel == "log":
            spectrogram = np.log(spectrogram)
        # 如果 log_mel 参数为 "log10"，应用以 10 为底的对数变换到 spectrogram
        elif log_mel == "log10":
            spectrogram = np.log10(spectrogram)
        # 如果 log_mel 参数为 "dB"
        elif log_mel == "dB":
            # 根据 power 参数选择不同的转换方法
            if power == 1.0:
                spectrogram = amplitude_to_db(spectrogram, reference, min_value, db_range)
            elif power == 2.0:
                spectrogram = power_to_db(spectrogram, reference, min_value, db_range)
            else:
                # 如果 power 参数不为 1.0 或 2.0，则抛出 ValueError 异常
                raise ValueError(f"Cannot use log_mel option '{log_mel}' with power {power}")
        else:
            # 如果 log_mel 参数不是 "log"、"log10" 或 "dB"，则抛出 ValueError 异常
            raise ValueError(f"Unknown log_mel option: {log_mel}")

        # 将 spectrogram 转换为指定的 dtype 类型
        spectrogram = np.asarray(spectrogram, dtype)

    # 返回处理后的 spectrogram
    return spectrogram
# 将功率谱图转换为分贝（dB）刻度。使用基本对数属性以确保数值稳定性。
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
    # 检查参考值是否小于等于零，如果是，则引发异常
    if reference <= 0.0:
        raise ValueError("reference must be greater than zero")
    # 检查最小值是否小于等于零，如果是，则引发异常
    if min_value <= 0.0:
        raise ValueError("min_value must be greater than zero")

    # 确保参考值不小于最小值
    reference = max(min_value, reference)

    # 将谱图限制在最小值和无上限之间
    spectrogram = np.clip(spectrogram, a_min=min_value, a_max=None)
    # 计算功率谱图的分贝值
    spectrogram = 10.0 * (np.log10(spectrogram) - np.log10(reference))

    # 如果指定了动态范围（db_range），则进一步限制谱图在指定范围内
    if db_range is not None:
        # 检查动态范围是否小于等于零，如果是，则引发异常
        if db_range <= 0.0:
            raise ValueError("db_range must be greater than zero")
        # 将谱图限制在最小值和无上限之间
        spectrogram = np.clip(spectrogram, a_min=spectrogram.max() - db_range, a_max=None)

    return spectrogram


# 将幅度谱图转换为分贝（dB）刻度。使用基本对数属性以确保数值稳定性。
def amplitude_to_db(
    spectrogram: np.ndarray,
    reference: float = 1.0,
    min_value: float = 1e-5,
    db_range: Optional[float] = None,
) -> np.ndarray:
    """
    Converts an amplitude spectrogram to the decibel scale. This computes `20 * log10(spectrogram / reference)`, using
    basic logarithm properties for numerical stability.
    """
    # 将输入的振幅（mel）频谱图转换为分贝表示的频谱图。
    def amplitude_to_db(spectrogram, reference=1.0, min_value=1e-5, db_range=None):
        # 如果参考值小于等于0，抛出数值错误异常
        if reference <= 0.0:
            raise ValueError("reference must be greater than zero")
        # 如果最小值小于等于0，抛出数值错误异常
        if min_value <= 0.0:
            raise ValueError("min_value must be greater than zero")
    
        # 将参考值设置为最小值和自身的最大值之间的最大值，确保参考值不小于最小值
        reference = max(min_value, reference)
    
        # 将频谱图限制在[min_value, None]的范围内
        spectrogram = np.clip(spectrogram, a_min=min_value, a_max=None)
        # 将振幅转换为分贝值，公式为 20 * log10(spectrogram / reference)
        spectrogram = 20.0 * (np.log10(spectrogram) - np.log10(reference))
    
        # 如果提供了db_range参数，则将频谱图限制在[spectrogram.max() - db_range, None]的范围内
        if db_range is not None:
            # 如果db_range小于等于0，抛出数值错误异常
            if db_range <= 0.0:
                raise ValueError("db_range must be greater than zero")
            # 将频谱图限制在[spectrogram.max() - db_range, None]的范围内
            spectrogram = np.clip(spectrogram, a_min=spectrogram.max() - db_range, a_max=None)
    
        # 返回转换后的频谱图
        return spectrogram
### deprecated functions below this line ###

# 警告：此函数已弃用，将在 Transformers 版本 4.31.0 中移除
def get_mel_filter_banks(
    nb_frequency_bins: int,
    nb_mel_filters: int,
    frequency_min: float,
    frequency_max: float,
    sample_rate: int,
    norm: Optional[str] = None,
    mel_scale: str = "htk",
) -> np.array:
    # 发出未来警告，提醒函数即将被移除
    warnings.warn(
        "The function `get_mel_filter_banks` is deprecated and will be removed in version 4.31.0 of Transformers",
        FutureWarning,
    )
    # 调用 mel_filter_bank 函数，返回梅尔滤波器组
    return mel_filter_bank(
        num_frequency_bins=nb_frequency_bins,
        num_mel_filters=nb_mel_filters,
        min_frequency=frequency_min,
        max_frequency=frequency_max,
        sampling_rate=sample_rate,
        norm=norm,
        mel_scale=mel_scale,
    )


def fram_wave(waveform: np.array, hop_length: int = 160, fft_window_size: int = 400, center: bool = True):
    """
    为了计算短时傅里叶变换，需要将波形分割成重叠的窗口化片段，称为“帧”。

    Args:
        waveform (`np.array` of shape `(sample_length,)`):
            将被分割成较小块的原始波形。
        hop_length (`int`, *optional*, defaults to 160):
            波形的每个窗口之间的步长。
        fft_window_size (`int`, *optional*, defaults to 400):
            窗口的大小。
        center (`bool`, defaults to `True`):
            是否将每个帧居中于帧的中间。居中通过在左右两侧反射波形来实现。

    Return:
        framed_waveform (`np.array` of shape `(waveform.shape // hop_length , fft_window_size)`):
            可供 `np.fft` 使用的帧化波形。
    """
    # 发出未来警告，提醒函数即将被移除
    warnings.warn(
        "The function `fram_wave` is deprecated and will be removed in version 4.31.0 of Transformers",
        FutureWarning,
    )
    # 初始化帧列表
    frames = []
    # 对波形数据进行帧分割，每一帧作为一个数据片段进行处理
    for i in range(0, waveform.shape[0] + 1, hop_length):
        # 如果指定居中处理
        if center:
            # 计算帧的一半窗口大小
            half_window = (fft_window_size - 1) // 2 + 1
            # 计算帧的起始和结束位置
            start = i - half_window if i > half_window else 0
            end = i + half_window if i < waveform.shape[0] - half_window else waveform.shape[0]
            # 提取波形中的帧数据
            frame = waveform[start:end]
            # 如果起始位置是0，使用反射填充来扩展帧
            if start == 0:
                padd_width = (-i + half_window, 0)
                frame = np.pad(frame, pad_width=padd_width, mode="reflect")
            # 如果结束位置是波形的末尾，使用反射填充来扩展帧
            elif end == waveform.shape[0]:
                padd_width = (0, (i - waveform.shape[0] + half_window))
                frame = np.pad(frame, pad_width=padd_width, mode="reflect")
    
        # 如果不居中处理
        else:
            # 直接从波形中提取指定大小的帧数据
            frame = waveform[i : i + fft_window_size]
            # 获取帧的宽度
            frame_width = frame.shape[0]
            # 如果帧宽度小于指定的窗口大小，使用常数值填充
            if frame_width < waveform.shape[0]:
                frame = np.lib.pad(
                    frame, pad_width=(0, fft_window_size - frame_width), mode="constant", constant_values=0
                )
        # 将处理好的帧数据添加到帧列表中
        frames.append(frame)
    
    # 将帧列表转换为 numpy 数组形式
    frames = np.stack(frames, 0)
    # 返回所有帧数据的 numpy 数组
    return frames
def stft(frames: np.array, windowing_function: np.array, fft_window_size: int = None):
    """
    Calculates the complex Short-Time Fourier Transform (STFT) of the given framed signal. Should give the same results
    as `torch.stft`.

    Args:
        frames (`np.array` of dimension `(num_frames, fft_window_size)`):
            A framed audio signal obtained using `audio_utils.fram_wav`.
        windowing_function (`np.array` of dimension `(nb_frequency_bins, nb_mel_filters)`:
            An array representing the function used to reduce amplitude discontinuities at frame boundaries when computing STFT.
            Each frame is multiplied by this windowing function. For details on these discontinuities (Spectral leakage),
            refer to [this tutorial](https://download.ni.com/evaluation/pxi/Understanding%20FFTs%20and%20Windowing.pdf).
        fft_window_size (`int`, *optional*):
            Size of the window on which the Fourier transform is applied, controlling frequency resolution of the spectrogram.
            Default is `None`, where it defaults to `frame_size`. Increasing `fft_window_size` slows computation but improves resolution.

    Example:

    ```
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
            Spectrogram of shape `(num_frames, nb_frequency_bins)` obtained using STFT algorithm
    """
    warnings.warn(
        "The function `stft` is deprecated and will be removed in version 4.31.0 of Transformers",
        FutureWarning,
    )
    # Determine the frame size from input frames
    frame_size = frames.shape[1]

    # Set fft_window_size to frame_size if not provided
    if fft_window_size is None:
        fft_window_size = frame_size

    # Validate fft_window_size against frame_size
    if fft_window_size < frame_size:
        raise ValueError("FFT size must be greater or equal to the frame size")

    # Calculate the number of FFT bins to store
    nb_frequency_bins = (fft_window_size >> 1) + 1

    # Initialize an empty array for the spectrogram
    spectrogram = np.empty((len(frames), nb_frequency_bins), dtype=np.complex64)

    # Initialize an array for the FFT signal
    fft_signal = np.zeros(fft_window_size)

    # Iterate over frames and compute STFT
    for f, frame in enumerate(frames):
        # Apply windowing function to the frame if provided
        if windowing_function is not None:
            np.multiply(frame, windowing_function, out=fft_signal[:frame_size])
        else:
            fft_signal[:frame_size] = frame
        
        # Compute FFT and store in the spectrogram array
        spectrogram[f] = np.fft.fft(fft_signal, axis=0)[:nb_frequency_bins]

    # Transpose the spectrogram and return
    return spectrogram.T
```