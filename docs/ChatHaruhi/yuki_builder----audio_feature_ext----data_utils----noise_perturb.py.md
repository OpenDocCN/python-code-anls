# `.\Chat-Haruhi-Suzumiya\yuki_builder\audio_feature_ext\data_utils\noise_perturb.py`

```py
import os  # 导入操作系统接口模块
import random  # 导入随机数生成模块
import warnings  # 导入警告处理模块

import numpy as np  # 导入数值计算库numpy

warnings.filterwarnings("ignore")  # 忽略警告信息
import librosa  # 导入音频处理库librosa


class NoisePerturbAugmentor(object):
    """用于添加背景噪声的增强模型

    :param min_snr_dB: 最小的信噪比，以分贝为单位
    :type min_snr_dB: int
    :param max_snr_dB: 最大的信噪比，以分贝为单位
    :type max_snr_dB: int
    :param noise_path: 噪声文件夹
    :type noise_path: str
    :param sr: 音频采样率，必须跟训练数据的一样
    :type sr: int
    :param prob: 数据增强的概率
    :type prob: float
    """

    def __init__(self, min_snr_dB=10, max_snr_dB=30, noise_path="dataset/noise", sr=16000, prob=0.5):
        self.prob = prob  # 设置数据增强的概率
        self.sr = sr  # 设置音频采样率
        self._min_snr_dB = min_snr_dB  # 设置最小信噪比（分贝）
        self._max_snr_dB = max_snr_dB  # 设置最大信噪比（分贝）
        self._noise_files = self.get_noise_file(noise_path=noise_path)  # 获取噪声文件列表

    # 获取全部噪声数据
    @staticmethod
    def get_noise_file(noise_path):
        """获取指定文件夹中的所有噪声文件

        :param noise_path: 噪声文件夹路径
        :type noise_path: str
        :return: 噪声文件路径列表
        :rtype: list
        """
        noise_files = []  # 初始化噪声文件列表
        if not os.path.exists(noise_path): return noise_files  # 如果噪声文件夹不存在则返回空列表
        for file in os.listdir(noise_path):  # 遍历噪声文件夹中的文件
            noise_files.append(os.path.join(noise_path, file))  # 将文件路径加入列表
        return noise_files  # 返回噪声文件路径列表

    @staticmethod
    def rms_db(wav):
        """返回以分贝为单位的音频均方根能量

        :param wav: 音频数据
        :type wav: ndarray
        :return: 音频均方根能量（分贝）
        :rtype: float
        """
        mean_square = np.mean(wav ** 2)  # 计算音频均方根能量
        return 10 * np.log10(mean_square)  # 返回以分贝为单位的均方根能量

    def __call__(self, wav):
        """添加背景噪音音频

        :param wav: librosa 读取的数据
        :type wav: ndarray
        :return: 添加了背景噪音的音频数据
        :rtype: ndarray
        """
        if random.random() > self.prob: return wav  # 根据概率决定是否执行数据增强
        if len(self._noise_files) == 0: return wav  # 如果没有噪声文件，则直接返回原始音频数据
        noise, r = librosa.load(random.choice(self._noise_files), sr=self.sr)  # 随机选择一个噪声文件并加载
        snr_dB = random.uniform(self._min_snr_dB, self._max_snr_dB)  # 随机生成信噪比
        noise_gain_db = min(self.rms_db(wav) - self.rms_db(noise) - snr_dB, 300)  # 计算噪声增益
        noise *= 10. ** (noise_gain_db / 20.)  # 根据增益调整噪声幅度
        noise_new = np.zeros(wav.shape, dtype=np.float32)  # 创建与音频数据相同大小的空白噪声数组
        if noise.shape[0] >= wav.shape[0]:  # 如果噪声比音频数据长
            start = random.randint(0, noise.shape[0] - wav.shape[0])  # 随机选择噪声开始位置
            noise_new[:wav.shape[0]] = noise[start: start + wav.shape[0]]  # 将截取的噪声复制到新数组中
        else:  # 如果音频数据比噪声长
            start = random.randint(0, wav.shape[0] - noise.shape[0])  # 随机选择音频数据开始位置
            noise_new[start:start + noise.shape[0]] = noise[:]  # 将噪声数据复制到对应位置
        wav += noise_new  # 将噪声添加到音频数据中
        return wav  # 返回添加了噪声的音频数据
```