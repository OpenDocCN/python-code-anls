# `so-vits-svc\modules\F0Predictor\PMF0Predictor.py`

```py
import numpy as np  # 导入 NumPy 库，用于处理数组和矩阵
import parselmouth  # 导入 parselmouth 库，用于音频分析

from modules.F0Predictor.F0Predictor import F0Predictor  # 从 F0Predictor 模块中导入 F0Predictor 类


class PMF0Predictor(F0Predictor):  # 定义 PMF0Predictor 类，继承自 F0Predictor 类
    def __init__(self,hop_length=512,f0_min=50,f0_max=1100,sampling_rate=44100):  # 定义初始化方法，设置默认参数
        self.hop_length = hop_length  # 设置帧移长度
        self.f0_min = f0_min  # 设置最小基频
        self.f0_max = f0_max  # 设置最大基频
        self.sampling_rate = sampling_rate  # 设置采样率
        self.name = "pm"  # 设置名称为 "pm"
    
    def interpolate_f0(self,f0):  # 定义插值处理方法，接收基频数组作为参数
        '''
        对F0进行插值处理
        '''
        vuv_vector = np.zeros_like(f0, dtype=np.float32)  # 创建与基频数组相同大小的零向量
        vuv_vector[f0 > 0.0] = 1.0  # 将基频大于0的位置设置为1
        vuv_vector[f0 <= 0.0] = 0.0  # 将基频小于等于0的位置设置为0
    
        nzindex = np.nonzero(f0)[0]  # 获取基频数组中非零元素的索引
        data = f0[nzindex]  # 获取非零基频数据
        nzindex = nzindex.astype(np.float32)  # 将非零基频索引转换为浮点数类型
        time_org = self.hop_length / self.sampling_rate * nzindex  # 计算原始时间
        time_frame = np.arange(f0.shape[0]) * self.hop_length / self.sampling_rate  # 计算帧时间

        if data.shape[0] <= 0:  # 如果数据长度小于等于0
            return np.zeros(f0.shape[0], dtype=np.float32),vuv_vector  # 返回与基频数组相同大小的零数组和声音激活向量

        if data.shape[0] == 1:  # 如果数据长度等于1
            return np.ones(f0.shape[0], dtype=np.float32) * f0[0],vuv_vector  # 返回与基频数组相同大小的数组，每个元素都是第一个基频值，和声音激活向量

        f0 = np.interp(time_frame, time_org, data, left=data[0], right=data[-1])  # 对基频进行插值处理
        
        return f0,vuv_vector  # 返回插值后的基频数组和声音激活向量
    

    def compute_f0(self,wav,p_len=None):  # 定义计算基频方法，接收音频数据和填充长度作为参数
        x = wav  # 将音频数据赋值给变量 x
        if p_len is None:  # 如果填充长度为 None
            p_len = x.shape[0]//self.hop_length  # 计算填充长度
        else:  # 否则
            assert abs(p_len-x.shape[0]//self.hop_length) < 4, "pad length error"  # 断言填充长度与音频数据长度除以帧移长度的差的绝对值小于4，否则抛出错误信息
        time_step = self.hop_length / self.sampling_rate * 1000  # 计算时间步长
        f0 = parselmouth.Sound(x, self.sampling_rate).to_pitch_ac(  # 使用 parselmouth 库计算音频的基频
            time_step=time_step / 1000, voicing_threshold=0.6,
            pitch_floor=self.f0_min, pitch_ceiling=self.f0_max).selected_array['frequency']  # 设置参数并获取基频数组

        pad_size=(p_len - len(f0) + 1) // 2  # 计算填充大小
        if(pad_size>0 or p_len - len(f0) - pad_size>0):  # 如果填充大小大于0或者填充后的长度大于0
            f0 = np.pad(f0,[[pad_size,p_len - len(f0) - pad_size]], mode='constant')  # 对基频数组进行填充
        f0,uv = self.interpolate_f0(f0)  # 调用插值处理方法对基频数组进行插值处理
        return f0  # 返回插值后的基频数组
    # 计算基频和声音是否有声音的标志
    def compute_f0_uv(self,wav,p_len=None):
        # 将输入的波形赋值给变量x
        x = wav
        # 如果未指定p_len，则根据波形长度和帧移计算p_len
        if p_len is None:
            p_len = x.shape[0]//self.hop_length
        # 如果指定了p_len，则检查计算得到的p_len与指定的p_len之间的差值是否小于4，否则抛出异常
        else:
            assert abs(p_len-x.shape[0]//self.hop_length) < 4, "pad length error"
        # 计算每个帧的时间间隔
        time_step = self.hop_length / self.sampling_rate * 1000
        # 使用Praat库计算音高，并将结果存储在f0中
        f0 = parselmouth.Sound(x, self.sampling_rate).to_pitch_ac(
            time_step=time_step / 1000, voicing_threshold=0.6,
            pitch_floor=self.f0_min, pitch_ceiling=self.f0_max).selected_array['frequency']

        # 计算需要填充的大小
        pad_size=(p_len - len(f0) + 1) // 2
        # 如果需要填充的大小大于0，或者p_len与f0长度的差值大于0，则进行填充
        if(pad_size>0 or p_len - len(f0) - pad_size>0):
            f0 = np.pad(f0,[[pad_size,p_len - len(f0) - pad_size]], mode='constant')
        # 对f0进行插值处理，得到插值后的f0和声音是否有声音的标志uv
        f0,uv = self.interpolate_f0(f0)
        # 返回计算得到的f0和uv
        return f0,uv
```