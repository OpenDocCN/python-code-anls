# `so-vits-svc\vdecoder\hifiganwithsnake\nvSTFT.py`

```py
# 导入所需的库
import os
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn

# 设置环境变量LRU_CACHE_CAPACITY为3
os.environ["LRU_CACHE_CAPACITY"] = "3"

# 将音频文件加载为torch张量
def load_wav_to_torch(full_path, target_sr=None, return_empty_on_exception=False):
    # 初始化采样率
    sampling_rate = None
    try:
        # 读取音频文件数据和采样率
        data, sampling_rate = sf.read(full_path, always_2d=True)
    except Exception as ex:
        # 如果加载失败，打印异常信息
        print(f"'{full_path}' failed to load.\nException:")
        print(ex)
        # 如果设置了返回空数组，返回空数组和采样率或目标采样率或32000
        if return_empty_on_exception:
            return [], sampling_rate or target_sr or 32000
        else:
            # 否则抛出异常
            raise Exception(ex)
    
    # 如果数据维度大于1，取第一列数据
    if len(data.shape) > 1:
        data = data[:, 0]
        # 检查音频文件持续时间是否大于2个样本（否则切片操作可能在错误的维度上）
        assert len(data) > 2
    
    # 如果音频数据类型为整数
    if np.issubdtype(data.dtype, np.integer):
        # 计算最大幅度
        max_mag = -np.iinfo(data.dtype).min
    else:
        # 如果音频数据类型为fp32
        max_mag = max(np.amax(data), -np.amin(data))
        # 根据最大幅度确定数据类型
        max_mag = (2**31)+1 if max_mag > (2**15) else ((2**15)+1 if max_mag > 1.01 else 1.0)
    
    # 将数据转换为torch张量并进行幅度归一化
    data = torch.FloatTensor(data.astype(np.float32))/max_mag
    
    # 如果数据包含无穷大或NaN，并且设置了返回空数组，返回空数组和采样率或目标采样率或32000
    if (torch.isinf(data) | torch.isnan(data)).any() and return_empty_on_exception:
        return [], sampling_rate or target_sr or 32000
    # 如果设置了目标采样率且采样率不等于目标采样率，进行重采样
    if target_sr is not None and sampling_rate != target_sr:
        data = torch.from_numpy(librosa.core.resample(data.numpy(), orig_sr=sampling_rate, target_sr=target_sr))
        sampling_rate = target_sr
    
    # 返回处理后的数据和采样率
    return data, sampling_rate

# 动态范围压缩函数
def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

# 动态范围解压缩函数
def dynamic_range_decompression(x, C=1):
    # 返回 x 的指数函数值除以常数 C
    return np.exp(x) / C
# 定义动态范围压缩函数，使用 torch 库
def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    # 对输入张量进行对数变换，并限制最小值，乘以常数 C
    return torch.log(torch.clamp(x, min=clip_val) * C)

# 定义动态范围解压缩函数，使用 torch 库
def dynamic_range_decompression_torch(x, C=1):
    # 对输入张量进行指数变换，除以常数 C
    return torch.exp(x) / C

# 定义 STFT 类
class STFT():
    # 初始化函数
    def __init__(self, sr=22050, n_mels=80, n_fft=1024, win_size=1024, hop_length=256, fmin=20, fmax=11025, clip_val=1e-5):
        # 设置目标采样率
        self.target_sr = sr
        
        # 设置梅尔频率数量、FFT 点数、窗口大小、跳跃长度、最小频率、最大频率、限制值
        self.n_mels     = n_mels
        self.n_fft      = n_fft
        self.win_size   = win_size
        self.hop_length = hop_length
        self.fmin     = fmin
        self.fmax     = fmax
        self.clip_val = clip_val
        # 初始化梅尔基础和汉宁窗口的空字典
        self.mel_basis = {}
        self.hann_window = {}
    # 获取梅尔频谱
    def get_mel(self, y, center=False):
        # 设置采样率
        sampling_rate = self.target_sr
        # 设置梅尔频谱的数量
        n_mels     = self.n_mels
        # 设置FFT的点数
        n_fft      = self.n_fft
        # 设置窗口大小
        win_size   = self.win_size
        # 设置跳跃长度
        hop_length = self.hop_length
        # 设置最小频率
        fmin       = self.fmin
        # 设置最大频率
        fmax       = self.fmax
        # 设置剪切值
        clip_val   = self.clip_val
        
        # 如果音频的最小值小于-1，则打印最小值
        if torch.min(y) < -1.:
            print('min value is ', torch.min(y))
        # 如果音频的最大值大于1，则打印最大值
        if torch.max(y) > 1.:
            print('max value is ', torch.max(y))
        
        # 如果最大频率不在梅尔基础中
        if fmax not in self.mel_basis:
            # 计算梅尔频谱
            mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
            # 将梅尔频谱转换为张量，并存储在设备上
            self.mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
            # 创建汉宁窗口并存储在设备上
            self.hann_window[str(y.device)] = torch.hann_window(self.win_size).to(y.device)
        
        # 在y的第一个维度上添加填充，模式为反射
        y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_length)/2), int((n_fft-hop_length)/2)), mode='reflect')
        # 去除y的第一个维度
        y = y.squeeze(1)
        
        # 计算短时傅里叶变换
        spec = torch.stft(y, n_fft, hop_length=hop_length, win_length=win_size, window=self.hann_window[str(y.device)],
                          center=center, pad_mode='reflect', normalized=False, onesided=True)
        # 计算幅度谱
        spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))
        # 计算梅尔频谱
        spec = torch.matmul(self.mel_basis[str(fmax)+'_'+str(y.device)], spec)
        # 对频谱进行动态范围压缩
        spec = dynamic_range_compression_torch(spec, clip_val=clip_val)
        # 返回梅尔频谱
        return spec
    
    # 调用函数
    def __call__(self, audiopath):
        # 加载音频并返回音频张量和采样率
        audio, sr = load_wav_to_torch(audiopath, target_sr=self.target_sr)
        # 获取梅尔频谱
        spect = self.get_mel(audio.unsqueeze(0)).squeeze(0)
        # 返回梅尔频谱
        return spect
# 创建一个STFT对象实例
stft = STFT()
```