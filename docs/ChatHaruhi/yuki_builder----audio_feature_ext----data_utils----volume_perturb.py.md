# `.\Chat-Haruhi-Suzumiya\yuki_builder\audio_feature_ext\data_utils\volume_perturb.py`

```py
import random  # 导入 random 库，用于生成随机数

class VolumePerturbAugmentor(object):
    """添加随机音量大小

    :param min_gain_dBFS: 最小增益
    :type min_gain_dBFS: int
    :param max_gain_dBFS: 最大增益
    :type max_gain_dBFS: int
    :param prob: 数据增强的概率
    :type prob: float
    """

    def __init__(self, min_gain_dBFS=-15, max_gain_dBFS=15, prob=0.5):
        # 初始化函数，设置增益范围和增强概率
        self.prob = prob  # 设置数据增强的概率
        self._min_gain_dBFS = min_gain_dBFS  # 设置最小增益（单位：分贝）
        self._max_gain_dBFS = max_gain_dBFS  # 设置最大增益（单位：分贝）

    def __call__(self, wav):
        """改变音量大小

        :param wav: librosa 读取的数据
        :type wav: ndarray
        """
        # 如果随机数大于增强概率，直接返回音频数据
        if random.random() > self.prob: return wav
        # 在设定的增益范围内随机选择一个增益值
        gain = random.uniform(self._min_gain_dBFS, self._max_gain_dBFS)
        # 根据增益值调整音频的音量大小
        wav *= 10.**(gain / 20.)
        return wav  # 返回调整后的音频数据
```