# `so-vits-svc\modules\F0Predictor\DioF0Predictor.py`

```py
import numpy as np  # 导入 NumPy 库，用于科学计算
import pyworld  # 导入 pyworld 库，用于声码器相关功能

from modules.F0Predictor.F0Predictor import F0Predictor  # 从 F0Predictor 模块中导入 F0Predictor 类


class DioF0Predictor(F0Predictor):  # 定义 DioF0Predictor 类，继承自 F0Predictor 类
    def __init__(self,hop_length=512,f0_min=50,f0_max=1100,sampling_rate=44100):  # 定义初始化方法，设置默认参数
        self.hop_length = hop_length  # 设置帧移长度
        self.f0_min = f0_min  # 设置最小基频
        self.f0_max = f0_max  # 设置最大基频
        self.sampling_rate = sampling_rate  # 设置采样率
        self.name = "dio"  # 设置名称为 "dio"

    def interpolate_f0(self,f0):  # 定义插值处理方法，接收基频数据作为参数
        '''
        对F0进行插值处理
        '''
        vuv_vector = np.zeros_like(f0, dtype=np.float32)  # 创建与 f0 相同形状的零数组
        vuv_vector[f0 > 0.0] = 1.0  # 将 f0 大于 0 的位置设置为 1
        vuv_vector[f0 <= 0.0] = 0.0  # 将 f0 小于等于 0 的位置设置为 0
    
        nzindex = np.nonzero(f0)[0]  # 获取 f0 中非零元素的索引
        data = f0[nzindex]  # 获取 f0 中非零元素的值
        nzindex = nzindex.astype(np.float32)  # 将索引转换为 float32 类型
        time_org = self.hop_length / self.sampling_rate * nzindex  # 计算原始时间
        time_frame = np.arange(f0.shape[0]) * self.hop_length / self.sampling_rate  # 计算帧时间

        if data.shape[0] <= 0:  # 如果 data 的长度小于等于 0
            return np.zeros(f0.shape[0], dtype=np.float32),vuv_vector  # 返回与 f0 相同长度的零数组和 vuv_vector

        if data.shape[0] == 1:  # 如果 data 的长度等于 1
            return np.ones(f0.shape[0], dtype=np.float32) * f0[0],vuv_vector  # 返回与 f0 相同长度的数组，元素值为 f0[0]，和 vuv_vector

        f0 = np.interp(time_frame, time_org, data, left=data[0], right=data[-1])  # 对 f0 进行线性插值处理
        
        return f0,vuv_vector  # 返回插值处理后的 f0 和 vuv_vector

    def resize_f0(self,x, target_len):  # 定义基频调整方法，接收基频数据和目标长度作为参数
        source = np.array(x)  # 将输入的基频数据转换为 NumPy 数组
        source[source<0.001] = np.nan  # 将小于 0.001 的值设置为 NaN
        target = np.interp(np.arange(0, len(source)*target_len, len(source))/ target_len, np.arange(0, len(source)), source)  # 对基频数据进行插值处理
        res = np.nan_to_num(target)  # 将 NaN 值替换为 0
        return res  # 返回调整后的基频数据
    # 计算音频信号的基频（F0）
    def compute_f0(self,wav,p_len=None):
        # 如果未指定分段长度，则默认为音频长度除以每个分段的采样点数
        if p_len is None:
            p_len = wav.shape[0]//self.hop_length
        # 使用 pyworld 库的 dio 函数计算基频和对应的时间
        f0, t = pyworld.dio(
            wav.astype(np.double),
            fs=self.sampling_rate,
            f0_floor=self.f0_min,
            f0_ceil=self.f0_max,
            frame_period=1000 * self.hop_length / self.sampling_rate,
        )
        # 对基频进行修正，去除非声音部分
        f0 = pyworld.stonemask(wav.astype(np.double), f0, t, self.sampling_rate)
        # 对每个基频值进行四舍五入保留一位小数
        for index, pitch in enumerate(f0):
            f0[index] = round(pitch, 1)
        # 对修正后的基频进行插值和调整长度，返回插值后的基频
        return self.interpolate_f0(self.resize_f0(f0, p_len))[0]

    # 计算音频信号的基频（F0）和声音与无声音的标记
    def compute_f0_uv(self,wav,p_len=None):
        # 如果未指定分段长度，则默认为音频长度除以每个分段的采样点数
        if p_len is None:
            p_len = wav.shape[0]//self.hop_length
        # 使用 pyworld 库的 dio 函数计算基频和对应的时间
        f0, t = pyworld.dio(
            wav.astype(np.double),
            fs=self.sampling_rate,
            f0_floor=self.f0_min,
            f0_ceil=self.f0_max,
            frame_period=1000 * self.hop_length / self.sampling_rate,
        )
        # 对基频进行修正，去除非声音部分
        f0 = pyworld.stonemask(wav.astype(np.double), f0, t, self.sampling_rate)
        # 对每个基频值进行四舍五入保留一位小数
        for index, pitch in enumerate(f0):
            f0[index] = round(pitch, 1)
        # 对修正后的基频进行插值和调整长度，返回插值后的基频和无声音标记
        return self.interpolate_f0(self.resize_f0(f0, p_len))
```