# `so-vits-svc\modules\enhancer.py`

```py
import numpy as np  # 导入 numpy 库，用于数值计算
import torch  # 导入 torch 库，用于构建神经网络
import torch.nn.functional as F  # 导入 torch 中的神经网络函数
from torchaudio.transforms import Resample  # 从 torchaudio 库中导入 Resample 变换

from vdecoder.nsf_hifigan.models import load_model  # 从指定路径导入 load_model 函数
from vdecoder.nsf_hifigan.nvSTFT import STFT  # 从指定路径导入 STFT 类


class Enhancer:  # 定义 Enhancer 类
    def __init__(self, enhancer_type, enhancer_ckpt, device=None):  # 初始化方法
        if device is None:  # 如果设备为空
            device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 如果有 GPU 则使用 cuda，否则使用 cpu
        self.device = device  # 将设备信息保存到实例变量中
        
        if enhancer_type == 'nsf-hifigan':  # 如果增强器类型是 'nsf-hifigan'
            self.enhancer = NsfHifiGAN(enhancer_ckpt, device=self.device)  # 创建 NsfHifiGAN 实例
        else:  # 如果增强器类型不是 'nsf-hifigan'
            raise ValueError(f" [x] Unknown enhancer: {enhancer_type}")  # 抛出异常，显示未知的增强器类型
        
        self.resample_kernel = {}  # 初始化 resample_kernel 字典
        self.enhancer_sample_rate = self.enhancer.sample_rate()  # 获取增强器的采样率
        self.enhancer_hop_size = self.enhancer.hop_size()  # 获取增强器的 hop size


class NsfHifiGAN(torch.nn.Module):  # 定义 NsfHifiGAN 类，继承自 torch.nn.Module
    def __init__(self, model_path, device=None):  # 初始化方法
        super().__init__()  # 调用父类的初始化方法
        if device is None:  # 如果设备为空
            device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 如果有 GPU 则使用 cuda，否则使用 cpu
        self.device = device  # 将设备信息保存到实例变量中
        print('| Load HifiGAN: ', model_path)  # 打印加载 HifiGAN 模型的信息
        self.model, self.h = load_model(model_path, device=self.device)  # 调用 load_model 函数加载模型，并保存到实例变量中
    
    def sample_rate(self):  # 定义 sample_rate 方法
        return self.h.sampling_rate  # 返回采样率信息
        
    def hop_size(self):  # 定义 hop_size 方法
        return self.h.hop_size  # 返回 hop size 信息
        
    def forward(self, audio, f0):  # 定义 forward 方法，用于模型前向传播
        stft = STFT(  # 创建 STFT 实例
                self.h.sampling_rate,  # 采样率
                self.h.num_mels,  # 梅尔频率数量
                self.h.n_fft,  # FFT 点数
                self.h.win_size,  # 窗口大小
                self.h.hop_size,  # hop size
                self.h.fmin,  # 最小频率
                self.h.fmax)  # 最大频率
        with torch.no_grad():  # 禁用梯度计算
            mel = stft.get_mel(audio)  # 获取音频的梅尔频谱
            enhanced_audio = self.model(mel, f0[:,:mel.size(-1)]).view(-1)  # 使用模型增强音频
            return enhanced_audio, self.h.sampling_rate  # 返回增强后的音频和采样率
```