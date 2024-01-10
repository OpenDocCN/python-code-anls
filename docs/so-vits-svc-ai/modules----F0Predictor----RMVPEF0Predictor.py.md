# `so-vits-svc\modules\F0Predictor\RMVPEF0Predictor.py`

```
# 导入必要的模块
from typing import Union
import numpy as np
import torch
import torch.nn.functional as F
from modules.F0Predictor.F0Predictor import F0Predictor
from .rmvpe import RMVPE

# 创建 RMVPEF0Predictor 类，继承自 F0Predictor
class RMVPEF0Predictor(F0Predictor):
    # 初始化方法，设置默认参数
    def __init__(self, hop_length=512, f0_min=50, f0_max=1100, dtype=torch.float32, device=None, sampling_rate=44100, threshold=0.05):
        # 初始化 RMVPE 对象，加载预训练模型
        self.rmvpe = RMVPE(model_path="pretrain/rmvpe.pt", dtype=dtype, device=device)
        # 设置音频处理的参数
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        # 检查是否指定了设备，如果没有则根据是否有 CUDA 支持选择设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        # 设置阈值和采样率
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.dtype = dtype
        self.name = "rmvpe"

    # 定义 repeat_expand 方法，用于将内容重复扩展到指定长度
    def repeat_expand(
        self, content: Union[torch.Tensor, np.ndarray], target_len: int, mode: str = "nearest"
    ):
        # 获取内容的维度
        ndim = content.ndim

        # 如果内容是一维数组，则将其转换为二维数组
        if content.ndim == 1:
            content = content[None, None]
        # 如果内容是二维数组，则将其转换为三维数组
        elif content.ndim == 2:
            content = content[None]

        # 断言内容为三维数组
        assert content.ndim == 3

        # 检查内容是否为 numpy 数组，如果是则转换为 torch.Tensor
        is_np = isinstance(content, np.ndarray)
        if is_np:
            content = torch.from_numpy(content)

        # 使用插值方法将内容扩展到指定长度
        results = torch.nn.functional.interpolate(content, size=target_len, mode=mode)

        # 如果内容原本为 numpy 数组，则将结果转换为 numpy 数组
        if is_np:
            results = results.numpy()

        # 根据原始内容的维度返回结果
        if ndim == 1:
            return results[0, 0]
        elif ndim == 2:
            return results[0]
    # 对提取的音频信号进行后处理，包括频率、采样率、填充长度
    def post_process(self, x, sampling_rate, f0, pad_to):
        # 如果频率是 numpy 数组，则转换为 torch 张量，并移动到相同的设备上
        if isinstance(f0, np.ndarray):
            f0 = torch.from_numpy(f0).float().to(x.device)

        # 如果不需要填充，则直接返回频率
        if pad_to is None:
            return f0

        # 对频率进行重复扩展，使其长度达到填充长度
        f0 = self.repeat_expand(f0, pad_to)
        
        # 创建与频率相同大小的零张量，并根据频率值设置对应位置的值为1或0
        vuv_vector = torch.zeros_like(f0)
        vuv_vector[f0 > 0.0] = 1.0
        vuv_vector[f0 <= 0.0] = 0.0
        
        # 去掉0频率, 并线性插值
        nzindex = torch.nonzero(f0).squeeze()
        f0 = torch.index_select(f0, dim=0, index=nzindex).cpu().numpy()
        time_org = self.hop_length / sampling_rate * nzindex.cpu().numpy()
        time_frame = np.arange(pad_to) * self.hop_length / sampling_rate
        
        # 使用线性插值方法对频率进行插值，使其长度达到填充长度
        vuv_vector = F.interpolate(vuv_vector[None,None,:],size=pad_to)[0][0]

        # 如果频率长度小于等于0，则返回填充长度的零频率和对应的声门向量
        if f0.shape[0] <= 0:
            return torch.zeros(pad_to, dtype=torch.float, device=x.device).cpu().numpy(),vuv_vector.cpu().numpy()
        # 如果频率长度等于1，则返回填充长度的相同频率和对应的声门向量
        if f0.shape[0] == 1:
            return (torch.ones(pad_to, dtype=torch.float, device=x.device) * f0[0]).cpu().numpy() ,vuv_vector.cpu().numpy()
    
        # 使用 numpy 的插值方法对频率进行插值，使其长度达到填充长度
        f0 = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1])
        
        # 返回插值后的频率和对应的声门向量
        return f0,vuv_vector.cpu().numpy()

    # 计算音频信号的频率
    def compute_f0(self,wav,p_len=None):
        # 将音频信号转换为 torch 张量，并移动到相同的设备上
        x = torch.FloatTensor(wav).to(self.dtype).to(self.device)
        # 如果未指定填充长度，则默认为音频信号长度除以跳跃长度
        if p_len is None:
            p_len = x.shape[0]//self.hop_length
        else:
            # 如果指定了填充长度，则检查填充长度与音频信号长度除以跳跃长度的差值是否小于4
            assert abs(p_len-x.shape[0]//self.hop_length) < 4, "pad length error"
        # 从音频信号中推断频率
        f0 = self.rmvpe.infer_from_audio(x,self.sampling_rate,self.threshold)
        # 如果所有频率值都为0，则返回相应长度的零频率和对应的声门向量
        if torch.all(f0 == 0):
            rtn = f0.cpu().numpy() if p_len is None else np.zeros(p_len)
            return rtn,rtn
        # 对推断得到的频率进行后处理，返回处理后的频率
        return self.post_process(x,self.sampling_rate,f0,p_len)[0]
    # 计算音频的基频和声音是否有声音
    def compute_f0_uv(self, wav, p_len=None):
        # 将音频数据转换为PyTorch张量，并移动到指定的数据类型和设备上
        x = torch.FloatTensor(wav).to(self.dtype).to(self.device)
        # 如果未指定p_len，则根据音频长度和帧移计算默认的p_len
        if p_len is None:
            p_len = x.shape[0] // self.hop_length
        else:
            # 如果指定了p_len，则检查其与计算得到的长度之差是否小于4，否则抛出异常
            assert abs(p_len - x.shape[0] // self.hop_length) < 4, "pad length error"
        # 从音频中推断基频
        f0 = self.rmvpe.infer_from_audio(x, self.sampling_rate, self.threshold)
        # 如果基频全为0，则返回与p_len相同长度的全0数组
        if torch.all(f0 == 0):
            rtn = f0.cpu().numpy() if p_len is None else np.zeros(p_len)
            return rtn, rtn
        # 对基频进行后处理，返回处理后的结果
        return self.post_process(x, self.sampling_rate, f0, p_len)
```