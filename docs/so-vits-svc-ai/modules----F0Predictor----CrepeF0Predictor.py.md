# `so-vits-svc\modules\F0Predictor\CrepeF0Predictor.py`

```py
# 导入 torch 库
import torch

# 从 modules.F0Predictor.crepe 中导入 CrepePitchExtractor 类
from modules.F0Predictor.crepe import CrepePitchExtractor
# 从 modules.F0Predictor 中导入 F0Predictor 类
from modules.F0Predictor.F0Predictor import F0Predictor

# 定义 CrepeF0Predictor 类，继承自 F0Predictor 类
class CrepeF0Predictor(F0Predictor):
    # 初始化方法，设置默认参数
    def __init__(self,hop_length=512,f0_min=50,f0_max=1100,device=None,sampling_rate=44100,threshold=0.05,model="full"):
        # 创建 CrepePitchExtractor 对象
        self.F0Creper = CrepePitchExtractor(hop_length=hop_length,f0_min=f0_min,f0_max=f0_max,device=device,threshold=threshold,model=model)
        # 设置属性值
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.device = device
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.name = "crepe"

    # 计算音频的基频方法
    def compute_f0(self,wav,p_len=None):
        # 将音频数据转换为 torch 的 FloatTensor 类型，并移动到指定设备上
        x = torch.FloatTensor(wav).to(self.device)
        # 如果未指定音频片段长度，则根据 hop_length 计算默认长度
        if p_len is None:
            p_len = x.shape[0]//self.hop_length
        else:
            # 如果指定了音频片段长度，则检查是否与默认长度相差不超过 4
            assert abs(p_len-x.shape[0]//self.hop_length) < 4, "pad length error"
        # 使用 CrepePitchExtractor 对象计算音频的基频和声音是否有声音的标志
        f0,uv = self.F0Creper(x[None,:].float(),self.sampling_rate,pad_to=p_len)
        # 返回基频
        return f0
    
    # 计算音频的基频和声音是否有声音的标志方法
    def compute_f0_uv(self,wav,p_len=None):
        # 将音频数据转换为 torch 的 FloatTensor 类型，并移动到指定设备上
        x = torch.FloatTensor(wav).to(self.device)
        # 如果未指定音频片段长度，则根据 hop_length 计算默认长度
        if p_len is None:
            p_len = x.shape[0]//self.hop_length
        else:
            # 如果指定了音频片段长度，则检查是否与默认长度相差不超过 4
            assert abs(p_len-x.shape[0]//self.hop_length) < 4, "pad length error"
        # 使用 CrepePitchExtractor 对象计算音频的基频和声音是否有声音的标志
        f0,uv = self.F0Creper(x[None,:].float(),self.sampling_rate,pad_to=p_len)
        # 返回基频和声音是否有声音的标志
        return f0,uv
```