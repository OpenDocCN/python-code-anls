# `so-vits-svc\modules\F0Predictor\rmvpe\spec.py`

```
# 导入 numpy 库，并使用别名 np
import numpy as np
# 导入 torch 库
import torch
# 从 torch.nn.functional 模块中导入别名 F
import torch.nn.functional as F
# 从 librosa.filters 模块中导入 mel 函数
from librosa.filters import mel

# 定义 MelSpectrogram 类，继承自 torch.nn.Module
class MelSpectrogram(torch.nn.Module):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        n_mel_channels,
        sampling_rate,
        win_length,
        hop_length,
        n_fft=None,
        mel_fmin=0,
        mel_fmax=None,
        clamp = 1e-5
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 如果 n_fft 为 None，则将其设为 win_length
        n_fft = win_length if n_fft is None else n_fft
        # 初始化 hann_window 为空字典
        self.hann_window = {}
        # 使用 librosa.filters.mel 函数生成梅尔滤波器组
        mel_basis = mel(
            sr=sampling_rate,
            n_fft=n_fft, 
            n_mels=n_mel_channels, 
            fmin=mel_fmin, 
            fmax=mel_fmax,
            htk=True)
        # 将 mel_basis 转换为 torch 的 float 类型，并赋值给 mel_basis
        mel_basis = torch.from_numpy(mel_basis).float()
        # 将 mel_basis 注册为模型的缓冲区
        self.register_buffer("mel_basis", mel_basis)
        # 将 n_fft 赋值给 self.n_fft，如果 n_fft 为 None，则将 win_length 赋值给 self.n_fft
        self.n_fft = win_length if n_fft is None else n_fft
        # 将 hop_length 赋值给 self.hop_length
        self.hop_length = hop_length
        # 将 win_length 赋值给 self.win_length
        self.win_length = win_length
        # 将 sampling_rate 赋值给 self.sampling_rate
        self.sampling_rate = sampling_rate
        # 将 n_mel_channels 赋值给 self.n_mel_channels
        self.n_mel_channels = n_mel_channels
        # 将 clamp 赋值给 self.clamp
        self.clamp = clamp
    # 对音频进行前向处理，包括音调转换、速度调整等
    def forward(self, audio, keyshift=0, speed=1, center=True):
        # 根据音调变化计算频率变化因子
        factor = 2 ** (keyshift / 12)       
        # 根据频率变化因子计算新的 FFT 点数
        n_fft_new = int(np.round(self.n_fft * factor))
        # 根据频率变化因子计算新的窗口长度
        win_length_new = int(np.round(self.win_length * factor))
        # 根据速度因子计算新的帧移长度
        hop_length_new = int(np.round(self.hop_length * speed))
        
        # 生成音调变化和设备的组合键
        keyshift_key = str(keyshift)+'_'+str(audio.device)
        # 如果组合键不在汉宁窗口字典中，则创建对应的汉宁窗口并存储
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_length_new).to(audio.device)
            
        # 对音频进行短时傅里叶变换
        fft = torch.stft(
            audio,
            n_fft=n_fft_new,
            hop_length=hop_length_new,
            win_length=win_length_new,
            window=self.hann_window[keyshift_key],
            center=center,
            return_complex=True)
        # 计算幅度谱
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        
        # 如果音调发生变化
        if keyshift != 0:
            # 计算幅度谱的大小
            size = self.n_fft // 2 + 1
            resize = magnitude.size(1)
            # 如果幅度谱大小小于预期大小，则进行填充
            if resize < size:
                magnitude = F.pad(magnitude, (0, 0, 0, size-resize))
            # 根据窗口长度变化调整幅度谱
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new
            
        # 计算梅尔频谱
        mel_output = torch.matmul(self.mel_basis, magnitude)
        # 计算对数梅尔谱
        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        # 返回对数梅尔谱
        return log_mel_spec
```