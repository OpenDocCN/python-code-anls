# `.\Chat-Haruhi-Suzumiya\yuki_builder\audio_feature_ext\data_utils\speed_perturb.py`

```py
import random  # 导入随机数模块

import numpy as np  # 导入NumPy库


class SpeedPerturbAugmentor(object):
    """音频速度扰动增强器类"""

    def __init__(self, min_speed_rate=0.9, max_speed_rate=1.1, num_rates=3, prob=0.5):
        """
        初始化方法

        :param min_speed_rate: 最小的采样速率，不应小于0.9
        :type min_speed_rate: float
        :param max_speed_rate: 最大的采样速率，不应大于1.1
        :type max_speed_rate: float
        :param num_rates: 速率数量
        :type num_rates: int
        :param prob: 数据增强的概率
        :type prob: float
        """
        if min_speed_rate < 0.9:
            raise ValueError("Sampling speed below 0.9 can cause unnatural effects")  # 抛出数值错误异常
        if max_speed_rate > 1.1:
            raise ValueError("Sampling speed above 1.1 can cause unnatural effects")  # 抛出数值错误异常
        self.prob = prob  # 设置数据增强的概率
        self._min_speed_rate = min_speed_rate  # 设置最小速率
        self._max_speed_rate = max_speed_rate  # 设置最大速率
        self._num_rates = num_rates  # 设置速率数量
        if num_rates > 0:
            self._rates = np.linspace(self._min_speed_rate, self._max_speed_rate, self._num_rates, endpoint=True)  # 生成速率列表

    def __call__(self, wav):
        """改变音频语速的方法

        :param wav: librosa 读取的音频数据
        :type wav: ndarray
        """
        if random.random() > self.prob: return wav  # 根据概率决定是否进行增强
        if self._num_rates < 0:
            speed_rate = random.uniform(self._min_speed_rate, self._max_speed_rate)  # 在指定范围内随机选择速率
        else:
            speed_rate = random.choice(self._rates)  # 从预定义的速率列表中随机选择一个速率
        if speed_rate == 1.0: return wav  # 如果速率为1.0，则返回原始音频数据

        old_length = wav.shape[0]  # 获取原始音频数据长度
        new_length = int(old_length / speed_rate)  # 根据速率计算新的音频长度
        old_indices = np.arange(old_length)  # 生成原始音频数据的索引
        new_indices = np.linspace(start=0, stop=old_length, num=new_length)  # 生成新的索引序列
        wav = np.interp(new_indices, old_indices, wav)  # 插值计算新速率下的音频数据
        return wav  # 返回改变语速后的音频数据
```