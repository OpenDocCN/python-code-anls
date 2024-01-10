# `so-vits-svc\diffusion\vocoder.py`

```
# 导入 torch 库
import torch
# 从 torchaudio.transforms 模块中导入 Resample 类
from torchaudio.transforms import Resample

# 从 vdecoder.nsf_hifigan.models 模块中导入 load_config 和 load_model 函数
from vdecoder.nsf_hifigan.models import load_config, load_model
# 从 vdecoder.nsf_hifigan.nvSTFT 模块中导入 STFT 类
from vdecoder.nsf_hifigan.nvSTFT import STFT

# 定义 Vocoder 类
class Vocoder:
    # 初始化方法
    def __init__(self, vocoder_type, vocoder_ckpt, device = None):
        # 如果设备为空，则根据是否有 CUDA 加速选择设备
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # 根据不同的 vocoder_type 初始化不同的 vocoder
        if vocoder_type == 'nsf-hifigan':
            self.vocoder = NsfHifiGAN(vocoder_ckpt, device = device)
        elif vocoder_type == 'nsf-hifigan-log10':
            self.vocoder = NsfHifiGANLog10(vocoder_ckpt, device = device)
        else:
            # 如果 vocoder_type 不在预期范围内，则抛出 ValueError 异常
            raise ValueError(f" [x] Unknown vocoder: {vocoder_type}")
            
        # 初始化 resample_kernel 字典
        self.resample_kernel = {}
        # 获取 vocoder 的采样率、hop 大小和维度
        self.vocoder_sample_rate = self.vocoder.sample_rate()
        self.vocoder_hop_size = self.vocoder.hop_size()
        self.dimension = self.vocoder.dimension()
        
    # 提取方法
    def extract(self, audio, sample_rate, keyshift=0):
                
        # 如果采样率与 vocoder 采样率相同，则不进行重采样
        if sample_rate == self.vocoder_sample_rate:
            audio_res = audio
        else:
            # 否则进行重采样
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, self.vocoder_sample_rate, lowpass_filter_width = 128).to(self.device)
            audio_res = self.resample_kernel[key_str](audio)    
        
        # 提取 mel 频谱
        mel = self.vocoder.extract(audio_res, keyshift=keyshift) # B, n_frames, bins
        return mel
   
    # 推断方法
    def infer(self, mel, f0):
        # 从 f0 中获取与 mel 相同帧数的数据
        f0 = f0[:,:mel.size(1),0] # B, n_frames
        # 使用 vocoder 生成音频
        audio = self.vocoder(mel, f0)
        return audio
        
        
# 定义 NsfHifiGAN 类，继承自 torch.nn.Module
class NsfHifiGAN(torch.nn.Module):
    # 初始化方法，接受模型路径和设备参数
    def __init__(self, model_path, device=None):
        # 调用父类的初始化方法
        super().__init__()
        # 如果设备参数为空，则根据是否有 CUDA 设备选择设备
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 设置设备参数和模型路径
        self.device = device
        self.model_path = model_path
        self.model = None
        # 加载模型配置信息
        self.h = load_config(model_path)
        # 初始化短时傅里叶变换对象
        self.stft = STFT(
                self.h.sampling_rate, 
                self.h.num_mels, 
                self.h.n_fft, 
                self.h.win_size, 
                self.h.hop_size, 
                self.h.fmin, 
                self.h.fmax)
    
    # 返回采样率
    def sample_rate(self):
        return self.h.sampling_rate
        
    # 返回帧移大小
    def hop_size(self):
        return self.h.hop_size
    
    # 返回梅尔频率数量
    def dimension(self):
        return self.h.num_mels
        
    # 提取音频特征
    def extract(self, audio, keyshift=0):       
        # 获取梅尔频谱
        mel = self.stft.get_mel(audio, keyshift=keyshift).transpose(1, 2) # B, n_frames, bins
        return mel
    
    # 前向传播方法
    def forward(self, mel, f0):
        # 如果模型为空，则加载模型
        if self.model is None:
            print('| Load HifiGAN: ', self.model_path)
            self.model, self.h = load_model(self.model_path, device=self.device)
        # 使用无梯度计算上下文
        with torch.no_grad():
            c = mel.transpose(1, 2)
            # 生成音频
            audio = self.model(c, f0)
            return audio
# 创建一个名为 NsfHifiGANLog10 的类，它继承自 NsfHifiGAN 类
class NsfHifiGANLog10(NsfHifiGAN):    
    # 定义 forward 方法，接受 mel 和 f0 两个参数
    def forward(self, mel, f0):
        # 如果模型为空，则加载 HifiGAN 模型，并打印加载信息
        if self.model is None:
            print('| Load HifiGAN: ', self.model_path)
            self.model, self.h = load_model(self.model_path, device=self.device)
        # 使用 torch.no_grad() 上下文管理器，确保在前向传播过程中不计算梯度
        with torch.no_grad():
            # 计算 mel 的转置乘以 0.434294，并赋值给 c
            c = 0.434294 * mel.transpose(1, 2)
            # 使用模型生成音频数据，并返回结果
            audio = self.model(c, f0)
            return audio
```