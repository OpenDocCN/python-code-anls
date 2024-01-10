# `so-vits-svc\modules\F0Predictor\HarvestF0Predictor.py`

```
import numpy as np  # 导入 NumPy 库，用于处理数组和矩阵运算
import pyworld  # 导入 pyworld 库，用于声学参数提取

from modules.F0Predictor.F0Predictor import F0Predictor  # 从 F0Predictor 模块中导入 F0Predictor 类


class HarvestF0Predictor(F0Predictor):  # 定义 HarvestF0Predictor 类，继承自 F0Predictor 类
    def __init__(self,hop_length=512,f0_min=50,f0_max=1100,sampling_rate=44100):  # 定义初始化方法，设置默认参数
        self.hop_length = hop_length  # 设置帧移长度
        self.f0_min = f0_min  # 设置最小基频
        self.f0_max = f0_max  # 设置最大基频
        self.sampling_rate = sampling_rate  # 设置采样率
        self.name = "harvest"  # 设置名称为 "harvest"

    def interpolate_f0(self,f0):  # 定义插值处理方法，接收基频作为参数
        '''
        对F0进行插值处理
        '''
        vuv_vector = np.zeros_like(f0, dtype=np.float32)  # 创建与基频相同形状的零向量
        vuv_vector[f0 > 0.0] = 1.0  # 将基频大于0的位置设置为1
        vuv_vector[f0 <= 0.0] = 0.0  # 将基频小于等于0的位置设置为0
    
        nzindex = np.nonzero(f0)[0]  # 获取非零基频的索引
        data = f0[nzindex]  # 获取非零基频的值
        nzindex = nzindex.astype(np.float32)  # 将索引转换为浮点数类型
        time_org = self.hop_length / self.sampling_rate * nzindex  # 计算原始时间
        time_frame = np.arange(f0.shape[0]) * self.hop_length / self.sampling_rate  # 计算帧时间

        if data.shape[0] <= 0:  # 如果数据长度小于等于0
            return np.zeros(f0.shape[0], dtype=np.float32),vuv_vector  # 返回与基频相同长度的零向量和声音激活向量

        if data.shape[0] == 1:  # 如果数据长度等于1
            return np.ones(f0.shape[0], dtype=np.float32) * f0[0],vuv_vector  # 返回与基频相同长度的全为第一个基频值的向量和声音激活向量

        f0 = np.interp(time_frame, time_org, data, left=data[0], right=data[-1])  # 对基频进行插值处理
        
        return f0,vuv_vector  # 返回插值后的基频和声音激活向量

    def resize_f0(self,x, target_len):  # 定义基频调整大小方法，接收基频和目标长度作为参数
        source = np.array(x)  # 将输入的基频转换为数组
        source[source<0.001] = np.nan  # 将小于0.001的值设置为 NaN
        target = np.interp(np.arange(0, len(source)*target_len, len(source))/ target_len, np.arange(0, len(source)), source)  # 对基频进行插值调整
        res = np.nan_to_num(target)  # 将 NaN 值替换为 0
        return res  # 返回调整大小后的基频
        
    def compute_f0(self,wav,p_len=None):  # 定义计算基频方法，接收音频波形和帧数作为参数
        if p_len is None:  # 如果帧数为空
            p_len = wav.shape[0]//self.hop_length  # 计算帧数
        f0, t = pyworld.harvest(  # 使用 pyworld 库的 harvest 方法提取基频
                wav.astype(np.double),  # 将音频波形转换为双精度浮点数
                fs=self.hop_length,  # 设置帧移长度
                f0_ceil=self.f0_max,  # 设置最大基频
                f0_floor=self.f0_min,  # 设置最小基频
                frame_period=1000 * self.hop_length / self.sampling_rate,  # 设置帧周期
            )
        f0 = pyworld.stonemask(wav.astype(np.double), f0, t, self.fs)  # 使用 pyworld 库的 stonemask 方法对基频进行处理
        return self.interpolate_f0(self.resize_f0(f0, p_len))[0]  # 返回插值处理和调整大小后的基频
    # 计算音频信号的基频和声音激活度
    def compute_f0_uv(self, wav, p_len=None):
        # 如果未指定帧数，则默认为音频信号长度除以每帧的采样点数
        if p_len is None:
            p_len = wav.shape[0] // self.hop_length
        # 使用pyworld库的harvest函数计算音频信号的基频
        f0, t = pyworld.harvest(
            wav.astype(np.double),  # 将音频信号转换为双精度浮点数
            fs=self.sampling_rate,  # 设置采样率
            f0_floor=self.f0_min,  # 设置基频的最小值
            f0_ceil=self.f0_max,  # 设置基频的最大值
            frame_period=1000 * self.hop_length / self.sampling_rate,  # 设置帧间隔
        )
        # 使用pyworld库的stonemask函数对基频进行修正
        f0 = pyworld.stonemask(wav.astype(np.double), f0, t, self.sampling_rate)
        # 返回插值和调整后的基频
        return self.interpolate_f0(self.resize_f0(f0, p_len))
```