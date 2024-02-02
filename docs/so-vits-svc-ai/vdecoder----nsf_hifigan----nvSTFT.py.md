# `so-vits-svc\vdecoder\nsf_hifigan\nvSTFT.py`

```py
# 导入所需的库
import os
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn

# 设置环境变量LRU_CACHE_CAPACITY为3
os.environ["LRU_CACHE_CAPACITY"] = "3"

# 将音频文件加载为PyTorch张量
def load_wav_to_torch(full_path, target_sr=None, return_empty_on_exception=False):
    # 初始化采样率
    sampling_rate = None
    try:
        # 使用soundfile库读取音频文件数据，并指定always_2d=True以确保返回二维数组
        data, sampling_rate = sf.read(full_path, always_2d=True)
    except Exception as ex:
        # 如果加载失败，打印异常信息
        print(f"'{full_path}' failed to load.\nException:")
        print(ex)
        # 如果设置了return_empty_on_exception为True，则返回空数组和采样率，否则抛出异常
        if return_empty_on_exception:
            return [], sampling_rate or target_sr or 48000
        else:
            raise Exception(ex)
    
    # 如果数据维度大于1，则取第一列数据
    if len(data.shape) > 1:
        data = data[:, 0]
        # 检查音频文件的持续时间是否大于2个样本（因为否则切片操作可能在错误的维度上）
        assert len(data) > 2
    
    # 如果音频数据类型为整数
    if np.issubdtype(data.dtype, np.integer):
        # 计算最大幅度
        max_mag = -np.iinfo(data.dtype).min
    else:
        # 如果音频数据类型为fp32
        max_mag = max(np.amax(data), -np.amin(data))
        # 根据最大幅度的值确定数据类型应为16位INT、32位INT或[-1,1]的float32
        max_mag = (2**31)+1 if max_mag > (2**15) else ((2**15)+1 if max_mag > 1.01 else 1.0)
    
    # 将数据转换为PyTorch的FloatTensor，并进行幅度归一化
    data = torch.FloatTensor(data.astype(np.float32))/max_mag
    
    # 如果数据中存在无穷大或NaN，并且设置了return_empty_on_exception为True，则返回空数组和采样率
    if (torch.isinf(data) | torch.isnan(data)).any() and return_empty_on_exception:
        return [], sampling_rate or target_sr or 48000
    # 如果指定了目标采样率且与当前采样率不同，则进行重采样
    if target_sr is not None and sampling_rate != target_sr:
        data = torch.from_numpy(librosa.core.resample(data.numpy(), orig_sr=sampling_rate, target_sr=target_sr))
        sampling_rate = target_sr
    
    # 返回处理后的数据和采样率
    return data, sampling_rate

# 实现动态范围压缩
def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)
# 定义动态范围解压缩函数，将输入数据进行指数运算并除以常数C
def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C

# 定义动态范围压缩函数，使用torch库进行对数运算和限制最小值，并乘以常数C
def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

# 定义动态范围解压缩函数，使用torch库进行指数运算并除以常数C
def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

# 定义STFT类
class STFT():
    # 初始化函数，设置一些参数和变量
    def __init__(self, sr=22050, n_mels=80, n_fft=1024, win_size=1024, hop_length=256, fmin=20, fmax=11025, clip_val=1e-5):
        self.target_sr = sr
        self.n_mels     = n_mels
        self.n_fft      = n_fft
        self.win_size   = win_size
        self.hop_length = hop_length
        self.fmin     = fmin
        self.fmax     = fmax
        self.clip_val = clip_val
        self.mel_basis = {}  # 存储梅尔滤波器组
        self.hann_window = {}  # 存储汉宁窗口

    # 调用函数，用于处理音频数据
    def __call__(self, audiopath):
        # 调用load_wav_to_torch函数加载音频数据
        audio, sr = load_wav_to_torch(audiopath, target_sr=self.target_sr)
        # 调用get_mel函数获取梅尔频谱
        spect = self.get_mel(audio.unsqueeze(0)).squeeze(0)
        return spect

# 创建STFT对象
stft = STFT()
```