# `.\models\clap\feature_extraction_clap.py`

```
# 设置文件编码为 UTF-8
# 版权声明和保留声明
#
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按"原样"分发本软件
# 无论是明示的还是暗示的，都没有任何保证或条件
# 请参阅许可证以了解特定语言的详细信息
"""CLAP 的特征提取器类。"""


import copy  # 导入拷贝模块，用于对象的复制操作
from typing import Any, Dict, List, Optional, Union  # 导入类型提示模块

import numpy as np  # 导入 NumPy 库，用于数值计算
import torch  # 导入 PyTorch 库，用于深度学习

from ...audio_utils import mel_filter_bank, spectrogram, window_function  # 导入音频处理相关函数
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor  # 导入序列特征提取器
from ...feature_extraction_utils import BatchFeature  # 导入批次特征处理工具
from ...utils import TensorType, logging  # 导入工具函数和日志记录

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


class ClapFeatureExtractor(SequenceFeatureExtractor):
    r"""
    构建一个 CLAP 特征提取器。

    此特征提取器继承自 [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`]，其中包含
    大多数主要方法。用户应参考此超类以获取有关这些方法的更多信息。

    此类使用自定义的 NumPy 实现的 *短时傅里叶变换* (STFT) 从原始语音中提取梅尔滤波器组特征，该实现应与
    PyTorch 的 `torch.stft` 等效。

    """

    model_input_names = ["input_features", "is_longer"]  # 模型输入名称列表

    def __init__(
        self,
        feature_size=64,  # 特征大小，默认为 64
        sampling_rate=48_000,  # 采样率，默认为 48000 Hz
        hop_length=480,  # 跳跃长度，默认为 480
        max_length_s=10,  # 最大长度（秒），默认为 10 秒
        fft_window_size=1024,  # FFT 窗口大小，默认为 1024
        padding_value=0.0,  # 填充值，默认为 0.0
        return_attention_mask=False,  # 是否返回注意力掩码，默认为 False
        frequency_min: float = 0,  # 最小频率，默认为 0 Hz
        frequency_max: float = 14_000,  # 最大频率，默认为 14000 Hz
        top_db: int = None,  # 上分贝数，默认为 None
        truncation: str = "fusion",  # 截断方式，默认为 "fusion"
        padding: str = "repeatpad",  # 填充方式，默认为 "repeatpad"
        **kwargs,  # 其他参数
    ):
        super().__init__()  # 调用父类的构造函数
        # 在这里可以添加特定于 CLAP 特征提取器的初始化逻辑
    ):
        # 调用父类构造函数，初始化特征提取器的参数
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        # 设置顶部动态范围的阈值
        self.top_db = top_db
        # 设置是否截断音频的标志
        self.truncation = truncation
        # 设置填充值
        self.padding = padding
        # 设置FFT窗口大小
        self.fft_window_size = fft_window_size
        # 计算频率频段的数量
        self.nb_frequency_bins = (fft_window_size >> 1) + 1
        # 设置帧移长度
        self.hop_length = hop_length
        # 设置最大长度（秒）的样本数量
        self.nb_max_samples = max_length_s * sampling_rate
        # 设置采样率
        self.sampling_rate = sampling_rate
        # 设置最小频率
        self.frequency_min = frequency_min
        # 设置最大频率
        self.frequency_max = frequency_max
        # 创建Mel滤波器组，基于HTK标准
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=self.nb_frequency_bins,
            num_mel_filters=feature_size,
            min_frequency=frequency_min,
            max_frequency=frequency_max,
            sampling_rate=sampling_rate,
            norm=None,
            mel_scale="htk",
        )
        # 创建Slaney标准的Mel滤波器组
        self.mel_filters_slaney = mel_filter_bank(
            num_frequency_bins=self.nb_frequency_bins,
            num_mel_filters=feature_size,
            min_frequency=frequency_min,
            max_frequency=frequency_max,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance, except for the
            mel filter banks, which do not need to be saved or printed as they are too long.
        """
        # 深拷贝当前实例的所有属性
        output = copy.deepcopy(self.__dict__)
        # 添加特征提取器的类型信息到输出字典中
        output["feature_extractor_type"] = self.__class__.__name__
        # 如果存在Mel滤波器，从输出中删除，因为它们太长不需要被保存或打印
        if "mel_filters" in output:
            del output["mel_filters"]
        # 如果存在Slaney标准的Mel滤波器，从输出中删除
        if "mel_filters_slaney" in output:
            del output["mel_filters_slaney"]
        # 返回序列化后的字典
        return output
    def _np_extract_fbank_features(self, waveform: np.array, mel_filters: Optional[np.array] = None) -> np.ndarray:
        """
        使用汉宁窗口计算给定 `waveform` 的对数梅尔频谱。在CLAP中，根据截断模式使用两种不同的滤波器组：
            - `self.mel_filters`：这些对应于`torchaudio`的默认参数，可以通过调用`torchaudio.transforms.MelSpectrogram().mel_scale.fb`获得。
              当`truncation`设置为`"fusion"`时使用这些滤波器。
            - `self.mel_filteres_slaney`：这些对应于`librosa`的默认参数，在计算梅尔频谱时使用`librosa.filters.mel`。
              在原始实现中，仅当截断模式不是`"fusion"`时才使用这些滤波器。
        """
        # 计算对数梅尔频谱
        log_mel_spectrogram = spectrogram(
            waveform,
            window_function(self.fft_window_size, "hann"),  # 使用汉宁窗口函数
            frame_length=self.fft_window_size,
            hop_length=self.hop_length,
            power=2.0,
            mel_filters=mel_filters,
            log_mel="dB",  # 返回对数梅尔值
        )
        return log_mel_spectrogram.T  # 返回转置后的对数梅尔频谱

    def _random_mel_fusion(self, mel, total_frames, chunk_frames):
        ranges = np.array_split(list(range(0, total_frames - chunk_frames + 1)), 3)
        if len(ranges[1]) == 0:
            # 如果音频太短，只使用第一个块
            ranges[1] = [0]
        if len(ranges[2]) == 0:
            # 如果音频太短，只使用第一个块
            ranges[2] = [0]
        # 随机选择每个部分的索引
        idx_front = np.random.choice(ranges[0])
        idx_middle = np.random.choice(ranges[1])
        idx_back = np.random.choice(ranges[2])

        # 提取前、中、后各部分的梅尔频谱块
        mel_chunk_front = mel[idx_front : idx_front + chunk_frames, :]
        mel_chunk_middle = mel[idx_middle : idx_middle + chunk_frames, :]
        mel_chunk_back = mel[idx_back : idx_back + chunk_frames, :]

        # 对输入的mel进行调整大小
        mel = torch.tensor(mel[None, None, :])
        mel_shrink = torch.nn.functional.interpolate(
            mel, size=[chunk_frames, 64], mode="bilinear", align_corners=False
        )
        mel_shrink = mel_shrink[0][0].numpy()

        # 合并各部分的梅尔频谱块
        mel_fusion = np.stack([mel_shrink, mel_chunk_front, mel_chunk_middle, mel_chunk_back], axis=0)
        return mel_fusion

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        truncation: str = None,
        padding: Optional[str] = None,
        max_length: Optional[int] = None,
        sampling_rate: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
```