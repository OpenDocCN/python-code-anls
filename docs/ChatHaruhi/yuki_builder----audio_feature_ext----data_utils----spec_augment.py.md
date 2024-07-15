# `.\Chat-Haruhi-Suzumiya\yuki_builder\audio_feature_ext\data_utils\spec_augment.py`

```py
import random  # 导入random模块，用于生成随机数

import numpy as np  # 导入numpy库，用于处理数组和数值计算
from PIL import Image  # 从PIL库中导入Image模块，用于处理图像
from PIL.Image import BICUBIC  # 从PIL库中导入BICUBIC插值方法，用于图像插值处理


class SpecAugmentor(object):
    """Augmentation model for Time warping, Frequency masking, Time masking.

    SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
        https://arxiv.org/abs/1904.08779
    SpecAugment on Large Scale Datasets
        https://arxiv.org/abs/1912.05533
    """

    def __init__(self,
                 F=10,
                 T=50,
                 n_freq_masks=2,
                 n_time_masks=2,
                 p=1.0,
                 W=40,
                 adaptive_number_ratio=0,
                 adaptive_size_ratio=0,
                 max_n_time_masks=20,
                 replace_with_zero=True,
                 prob=0.5):
        """SpecAugment class.
        Args:
            :param F: 频率屏蔽参数，控制频率掩蔽的程度
            :type F: int
            :param T: 时间屏蔽参数，控制时间掩蔽的程度
            :type T: int
            :param n_freq_masks: 频率屏蔽数量，指定应用的频率掩蔽的数量
            :type n_freq_masks: int
            :param n_time_masks: 时间屏蔽数量，指定应用的时间掩蔽的数量
            :type n_time_masks: int
            :param p: 时间屏蔽上限参数，时间掩蔽的最大比例
            :type p: float
            :param W: 时间变形参数，用于时间扭曲操作的参数
            :type W: int
            :param adaptive_number_ratio: 时间屏蔽的自适应多重比，根据输入自适应调整时间掩蔽数量
            :type adaptive_number_ratio: float
            :param adaptive_size_ratio: 时间屏蔽的自适应大小比，根据输入自适应调整时间掩蔽大小
            :type adaptive_size_ratio: float
            :param max_n_time_masks: 时间屏蔽的最大数目，控制最大时间掩蔽数量
            :type max_n_time_masks: int
            :param replace_with_zero: 如果为真，在pad补0，否则使用平均值进行填充
            :type replace_with_zero: bool
            :param prob: 数据增强的概率，控制应用增强的概率
            :type prob: float
        """
        super().__init__()
        self.inplace = True  # 设定inplace参数为True，表示原地操作

        self.replace_with_zero = replace_with_zero  # 设置是否用0替换的标志位

        self.prob = prob  # 数据增强的概率
        self.W = W  # 时间变形参数
        self.F = F  # 频率屏蔽参数
        self.T = T  # 时间屏蔽参数
        self.n_freq_masks = n_freq_masks  # 频率屏蔽数量
        self.n_time_masks = n_time_masks  # 时间屏蔽数量
        self.p = p  # 时间屏蔽上限参数

        # adaptive SpecAugment
        self.adaptive_number_ratio = adaptive_number_ratio  # 时间屏蔽的自适应多重比
        self.adaptive_size_ratio = adaptive_size_ratio  # 时间屏蔽的自适应大小比
        self.max_n_time_masks = max_n_time_masks  # 时间屏蔽的最大数目

        if adaptive_number_ratio > 0:
            self.n_time_masks = 0  # 如果自适应多重比大于0，则将时间屏蔽数量设为0
        if adaptive_size_ratio > 0:
            self.T = 0  # 如果自适应大小比大于0，则将时间屏蔽参数设为0

        self._freq_mask = None  # 频率掩蔽对象
        self._time_mask = None  # 时间掩蔽对象

    @property
    def freq_mask(self):
        return self._freq_mask  # 返回频率掩蔽对象

    @property
    def time_mask(self):
        return self._time_mask  # 返回时间掩蔽对象

    def __repr__(self):
        return f"specaug: F-{self.F}, T-{self.T}, F-n-{self.n_freq_masks}, T-n-{self.n_time_masks}"

    def __call__(self, x):
        """
        数据增强函数，对输入的预处理音频数据进行增强处理。
        :param x: 经过预处理的音频数据
        :type x: ndarray
        """
        if random.random() > self.prob: return x  # 根据概率决定是否应用增强，如果不应用则直接返回原始数据
        return self.transform_feature(x)  # 应用增强方法对音频数据进行变换处理
    def time_warp(self, x):
        """对于特定增强的时间扭曲函数
        将随机中心帧移动一个随机宽度，宽度范围为 uniform(-window, window)

        Args:
            x (np.ndarray): 频谱图 (time, freq)

        Raises:
            NotImplementedError: 未实现的异常
            NotImplementedError: 未实现的异常

        Returns:
            np.ndarray: 经过时间扭曲的频谱图 (time, freq)
        """
        window = self.W
        if window == 0:
            return x
        t = x.shape[0]
        if t - window <= window:
            return x
        # NOTE: randrange(a, b) 产生 a 到 b-1 之间的随机整数
        center = random.randrange(window, t - window)
        # 随机选择一个中心点，并在 [center - window, center + window] 范围内进行偏移
        warped = random.randrange(center - window, center + window) + 1  # 1 ... t - 1
        left = Image.fromarray(x[:center]).resize((x.shape[1], warped), BICUBIC)
        right = Image.fromarray(x[center:]).resize((x.shape[1], t - warped), BICUBIC)
        if self.inplace:
            x[:warped] = left
            x[warped:] = right
            return x
        return np.concatenate((left, right), 0)

    def mask_freq(self, x, replace_with_zero=False):
        """频率掩蔽

        Args:
            x (np.ndarray): 频谱图 (time, freq)
            replace_with_zero (bool, optional): 是否用零替换，默认为 False

        Returns:
            np.ndarray: 频率掩蔽后的频谱图 (time, freq)
        """
        n_bins = x.shape[1]
        for i in range(0, self.n_freq_masks):
            f = int(random.uniform(a=0, b=self.F))
            f_0 = int(random.uniform(a=0, b=n_bins - f))
            assert f_0 <= f_0 + f
            if replace_with_zero:
                x[:, f_0:f_0 + f] = 0
            else:
                x[:, f_0:f_0 + f] = x.mean()
            self._freq_mask = (f_0, f_0 + f)
        return x

    def mask_time(self, x, replace_with_zero=False):
        """时间掩蔽

        Args:
            x (np.ndarray): 频谱图 (time, freq)
            replace_with_zero (bool, optional): 是否用零替换，默认为 False

        Returns:
            np.ndarray: 时间掩蔽后的频谱图 (time, freq)
        """
        n_frames = x.shape[0]

        if self.adaptive_number_ratio > 0:
            n_masks = int(n_frames * self.adaptive_number_ratio)
            n_masks = min(n_masks, self.max_n_time_masks)
        else:
            n_masks = self.n_time_masks

        if self.adaptive_size_ratio > 0:
            T = self.adaptive_size_ratio * n_frames
        else:
            T = self.T

        for i in range(n_masks):
            t = int(random.uniform(a=0, b=T))
            t = min(t, int(n_frames * self.p))
            t_0 = int(random.uniform(a=0, b=n_frames - t))
            assert t_0 <= t_0 + t
            if replace_with_zero:
                x[t_0:t_0 + t, :] = 0
            else:
                x[t_0:t_0 + t, :] = x.mean()
            self._time_mask = (t_0, t_0 + t)
        return x
    # 定义一个方法 `transform_feature`，用于对特征数据进行变换
    def transform_feature(self, x: np.ndarray):
        """
        Args:
            x (np.ndarray): `[T, F]`，输入的特征数据，T 表示时间步，F 表示特征维度
        Returns:
            x (np.ndarray): `[T, F]`，经过变换后的特征数据
        """
        # 断言输入的 x 是 numpy 数组
        assert isinstance(x, np.ndarray)
        # 断言输入的 x 是二维数组（即具有时间步和特征维度两个维度）
        assert x.ndim == 2
        # 调用 self 对象的 time_warp 方法，对时间维度进行变换
        x = self.time_warp(x)
        # 调用 self 对象的 mask_freq 方法，对特征维度进行频率掩码处理
        x = self.mask_freq(x, self.replace_with_zero)
        # 调用 self 对象的 mask_time 方法，对时间维度进行时间掩码处理
        x = self.mask_time(x, self.replace_with_zero)
        # 返回经过所有变换处理后的特征数据
        return x
```