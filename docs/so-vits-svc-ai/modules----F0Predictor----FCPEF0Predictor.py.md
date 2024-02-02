# `so-vits-svc\modules\F0Predictor\FCPEF0Predictor.py`

```py
# 导入必要的模块
from typing import Union
import numpy as np
import torch
import torch.nn.functional as F
from modules.F0Predictor.F0Predictor import F0Predictor
from .fcpe.model import FCPEInfer

# 创建 FCPEF0Predictor 类，继承自 F0Predictor 类
class FCPEF0Predictor(F0Predictor):
    # 初始化方法，设置默认参数
    def __init__(self, hop_length=512, f0_min=50, f0_max=1100, dtype=torch.float32, device=None, sampling_rate=44100, threshold=0.05):
        # 初始化 FCPEInfer 模型
        self.fcpe = FCPEInfer(model_path="pretrain/fcpe.pt", device=device, dtype=dtype)
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        # 如果设备为空，则根据是否有 CUDA 设备选择设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.dtype = dtype
        self.name = "fcpe"

    # 定义 repeat_expand 方法，用于对输入内容进行重复扩展
    def repeat_expand(
            self, content: Union[torch.Tensor, np.ndarray], target_len: int, mode: str = "nearest"
    ):
        # 获取内容的维度
        ndim = content.ndim

        # 如果内容维度为 1，则将其扩展为二维
        if content.ndim == 1:
            content = content[None, None]
        # 如果内容维度为 2，则将其扩展为三维
        elif content.ndim == 2:
            content = content[None]

        # 断言内容维度为 3
        assert content.ndim == 3

        # 判断内容是否为 numpy 数组，如果是，则转换为 torch.Tensor
        is_np = isinstance(content, np.ndarray)
        if is_np:
            content = torch.from_numpy(content)

        # 使用 interpolate 方法对内容进行插值，使其长度达到目标长度
        results = torch.nn.functional.interpolate(content, size=target_len, mode=mode)

        # 如果内容为 numpy 数组，则将结果转换为 numpy 数组
        if is_np:
            results = results.numpy()

        # 根据原始维度返回结果
        if ndim == 1:
            return results[0, 0]
        elif ndim == 2:
            return results[0]
    # 对音频信号进行后处理，包括频率、采样率、填充长度
    def post_process(self, x, sampling_rate, f0, pad_to):
        # 如果频率是 numpy 数组，则转换为 torch 张量，并移动到指定设备
        if isinstance(f0, np.ndarray):
            f0 = torch.from_numpy(f0).float().to(x.device)

        # 如果不需要填充，则直接返回频率
        if pad_to is None:
            return f0

        # 将频率进行重复扩展，使其长度等于填充长度
        f0 = self.repeat_expand(f0, pad_to)

        # 创建与频率相同长度的零向量
        vuv_vector = torch.zeros_like(f0)
        # 根据频率值设置声音有无的标志位
        vuv_vector[f0 > 0.0] = 1.0
        vuv_vector[f0 <= 0.0] = 0.0

        # 去掉0频率, 并线性插值
        nzindex = torch.nonzero(f0).squeeze()
        f0 = torch.index_select(f0, dim=0, index=nzindex).cpu().numpy()
        time_org = self.hop_length / sampling_rate * nzindex.cpu().numpy()
        time_frame = np.arange(pad_to) * self.hop_length / sampling_rate

        # 使用线性插值对声音有无的标志位进行插值
        vuv_vector = F.interpolate(vuv_vector[None, None, :], size=pad_to)[0][0]

        # 如果频率长度小于等于0，则返回填充长度的零频率和声音有无的标志位
        if f0.shape[0] <= 0:
            return torch.zeros(pad_to, dtype=torch.float, device=x.device).cpu().numpy(), vuv_vector.cpu().numpy()
        # 如果频率长度等于1，则返回填充长度的相同频率和声音有无的标志位
        if f0.shape[0] == 1:
            return (torch.ones(pad_to, dtype=torch.float, device=x.device) * f0[0]).cpu().numpy(), vuv_vector.cpu().numpy()

        # 使用 numpy 的插值函数对频率进行插值
        f0 = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1])
        # 使用 torch 重写声音有无的标志位
        # vuv_vector = np.ceil(scipy.ndimage.zoom(vuv_vector,pad_to/len(vuv_vector),order = 0))

        # 返回插值后的频率和声音有无的标志位
        return f0, vuv_vector.cpu().numpy()

    # 计算音频信号的频率
    def compute_f0(self, wav, p_len=None):
        # 将音频信号转换为 torch 张量，并移动到指定设备
        x = torch.FloatTensor(wav).to(self.dtype).to(self.device)
        # 如果填充长度未指定，则根据音频信号长度和跳跃长度计算填充长度
        if p_len is None:
            p_len = x.shape[0] // self.hop_length
        else:
            # 如果填充长度与音频信号长度和跳跃长度计算的长度差值大于4，则抛出异常
            assert abs(p_len - x.shape[0] // self.hop_length) < 4, "pad length error"
        # 使用预测模型计算音频信号的频率
        f0 = self.fcpe(x, sr=self.sampling_rate, threshold=self.threshold)[0,:,0]
        # 如果频率全为0，则返回填充长度的零频率和声音有无的标志位
        if torch.all(f0 == 0):
            rtn = f0.cpu().numpy() if p_len is None else np.zeros(p_len)
            return rtn, rtn
        # 对预测的频率进行后处理，返回处理后的频率
        return self.post_process(x, self.sampling_rate, f0, p_len)[0]
    # 计算音频的基频和声音是否有声音
    def compute_f0_uv(self, wav, p_len=None):
        # 将音频数据转换为PyTorch张量，并移动到指定的数据类型和设备上
        x = torch.FloatTensor(wav).to(self.dtype).to(self.device)
        # 如果未指定填充长度，则根据音频长度和跳跃长度计算填充长度
        if p_len is None:
            p_len = x.shape[0] // self.hop_length
        else:
            # 如果指定了填充长度，则确保填充长度与音频长度和跳跃长度计算的长度相差不超过4
            assert abs(p_len - x.shape[0] // self.hop_length) < 4, "pad length error"
        # 使用声音分割模型计算音频的基频
        f0 = self.fcpe(x, sr=self.sampling_rate, threshold=self.threshold)[0,:,0]
        # 如果基频全为0，则返回填充长度的全0数组
        if torch.all(f0 == 0):
            rtn = f0.cpu().numpy() if p_len is None else np.zeros(p_len)
            return rtn, rtn
        # 对计算得到的基频进行后处理，返回基频和声音是否有声音的结果
        return self.post_process(x, self.sampling_rate, f0, p_len)
```