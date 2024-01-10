# `so-vits-svc\vencoder\whisper\audio.py`

```
# 导入必要的模块和库
from functools import lru_cache
from typing import Union
import ffmpeg
import numpy as np
import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
from .utils import exact_div

# 定义音频的硬编码超参数
SAMPLE_RATE = 16000  # 采样率
N_FFT = 400  # FFT 窗口大小
N_MELS = 80  # 梅尔频谱的数量
HOP_LENGTH = 160  # 帧移
CHUNK_LENGTH = 30  # 音频块的长度
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000: 音频块中的样本数
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000: 梅尔频谱输入中的帧数

def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    打开音频文件并将其读取为单声道波形，必要时进行重采样

    Parameters
    ----------
    file: str
        要打开的音频文件

    sr: int
        如果需要，要重采样的采样率

    Returns
    -------
    包含音频波形的 NumPy 数组，数据类型为 float32。
    """
    try:
        # 这会启动一个子进程来解码音频，同时根据需要进行混音和重采样。
        # 需要安装 ffmpeg CLI 和 `ffmpeg-python` 包。
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    将音频数组填充或修剪到 N_SAMPLES，以符合编码器的预期。
    """
    # 检查输入的array是否为PyTorch张量
    if torch.is_tensor(array):
        # 如果array在指定的axis上的长度大于指定的length，则进行index_select操作，截取指定长度的数据
        if array.shape[axis] > length:
            array = array.index_select(dim=axis, index=torch.arange(length, device=array.device))
    
        # 如果array在指定的axis上的长度小于指定的length，则进行pad操作，填充指定长度的数据
        if array.shape[axis] < length:
            # 创建pad_widths列表，用于指定每个维度的填充宽度
            pad_widths = [(0, 0)] * array.ndim
            # 更新指定axis上的填充宽度
            pad_widths[axis] = (0, length - array.shape[axis])
            # 对array进行pad操作
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        # 如果array不是PyTorch张量，则进行相应的操作
        if array.shape[axis] > length:
            # 如果array在指定的axis上的长度大于指定的length，则进行take操作，截取指定长度的数据
            array = array.take(indices=range(length), axis=axis)
    
        # 如果array在指定的axis上的长度小于指定的length，则进行pad操作，填充指定长度的数据
        if array.shape[axis] < length:
            # 创建pad_widths列表，用于指定每个维度的填充宽度
            pad_widths = [(0, 0)] * array.ndim
            # 更新指定axis上的填充宽度
            pad_widths[axis] = (0, length - array.shape[axis])
            # 对array进行pad操作
            array = np.pad(array, pad_widths)
    
    # 返回处理后的array
    return array
# 使用 lru_cache 装饰器缓存函数的结果，避免重复计算
@lru_cache(maxsize=None)
# 定义函数，加载用于将 STFT 投影到 Mel 频谱图的 Mel 滤波器矩阵
def mel_filters(device, n_mels: int = N_MELS) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    # 断言 n_mels 必须为 80，否则抛出异常
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    # 从 NumPy 数组转换为 Tensor，并将其移动到指定设备上
    return torch.from_numpy(librosa_mel_fn(sr=SAMPLE_RATE,n_fft=N_FFT,n_mels=n_mels)).to(device)


# 定义函数，计算音频的对数 Mel 频谱图
def log_mel_spectrogram(audio: Union[str, np.ndarray, torch.Tensor], n_mels: int = N_MELS):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    # 如果输入不是 Tensor 类型，则根据类型加载音频数据
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    # 创建汉宁窗口
    window = torch.hann_window(N_FFT).to(audio.device)
    # 计算短时傅里叶变换
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    # 计算幅度谱
    magnitudes = stft[..., :-1].abs() ** 2

    # 获取 Mel 滤波器
    filters = mel_filters(audio.device, n_mels)
    # 计算 Mel 频谱图
    mel_spec = filters @ magnitudes

    # 对 Mel 频谱图进行对数变换，并进行限幅处理
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec
```