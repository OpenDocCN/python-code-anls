# `.\models\tvlt\feature_extraction_tvlt.py`

```
# 设置代码文件的编码格式为UTF-8
# 版权声明：2023年由HuggingFace Inc.团队保留所有权利。
#
# 根据Apache许可证2.0版（“许可证”）获得许可；除非符合许可证要求，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发的软件
# 没有任何明示或暗示的保证或条件。
# 有关特定语言的权限，请参阅许可证。
"""TVLT的特征提取器类。"""

from math import ceil  # 导入ceil函数，用于向上取整
from typing import List, Optional, Union  # 引入类型提示模块

import numpy as np  # 导入NumPy库

from ...audio_utils import mel_filter_bank, spectrogram, window_function  # 导入音频处理函数
from ...feature_extraction_sequence_utils import BatchFeature, SequenceFeatureExtractor  # 导入序列特征提取器
from ...utils import TensorType, logging  # 导入Tensor类型和日志记录工具

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

class TvltFeatureExtractor(SequenceFeatureExtractor):
    r"""
    构造一个TVLT音频特征提取器。此特征提取器用于准备模型的音频输入数据。

    此特征提取器继承自[`FeatureExtractionMixin`]，其中包含大多数主要方法。用户
    应参考此超类以获取有关这些方法的更多信息。

    Args:
        spectrogram_length (`Dict[str, int]` *可选*, 默认为2048):
            每个音频频谱图的时间长度。
        num_channels (`int` *可选*, 默认为1):
            音频通道数。
        patch_size (`List[int]` *可选*, 默认为`[16, 16]`):
            音频补丁嵌入的补丁大小。
        feature_size (`int`, *可选*, 默认为128):
            音频频谱图的频率长度。
        sampling_rate (`int`, *可选*, 默认为44100):
            应数字化音频文件的采样率，以赫兹（Hz）表示。
        hop_length_to_sampling_rate (`int`, *可选*, 默认为86):
            Hop length是用于获取Mel频率系数的STFT的重叠窗口的长度。
            例如，对于采样率44100，跳跃长度为512，即44100 / 512 = 86。
        n_fft (`int`, *可选*, 默认为2048):
            傅里叶变换的大小。
        padding_value (`float`, *可选*, 默认为0.0):
            用于填充音频的填充值。应该对应于静音部分。
    """

    model_input_names = ["audio_values", "audio_mask"]

    def __init__(
        self,
        spectrogram_length=2048,
        num_channels=1,
        patch_size=[16, 16],
        feature_size=128,
        sampling_rate=44100,
        hop_length_to_sampling_rate=86,
        n_fft=2048,
        padding_value=0.0,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        # 调用父类构造函数，初始化特征大小、采样率、填充值等参数

        self.spectrogram_length = spectrogram_length
        # 设置频谱长度属性

        self.num_channels = num_channels
        # 设置通道数属性

        self.patch_size = patch_size
        # 设置补丁大小属性

        self.freq_len = feature_size // self.patch_size[1]
        # 计算频率长度，特征大小除以补丁大小的第二个维度

        self.n_fft = n_fft
        # 设置 FFT 窗口大小属性

        self.hop_length = sampling_rate // hop_length_to_sampling_rate
        # 计算帧移大小属性

        self.sampling_rate = sampling_rate
        # 设置采样率属性

        self.padding_value = padding_value
        # 设置填充值属性

        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + n_fft // 2,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=22050.0,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        ).T
        # 使用 mel_filter_bank 函数生成梅尔滤波器系数矩阵并转置后赋给 mel_filters 属性

    def _np_extract_fbank_features(self, waveform: np.array) -> np.ndarray:
        """
        Compute the log-mel spectrogram of the provided audio, gives similar results to Whisper's original torch
        implementation with 1e-5 tolerance.
        """
        # 计算提供音频的对数梅尔频谱，使用 hann 窗口函数，帧长度为 n_fft，帧移为 hop_length
        # 使用 mel_filters.T 作为梅尔滤波器，输出对数梅尔频谱，单位为 dB，幅度范围为 80.0 dB
        log_spec = spectrogram(
            waveform,
            window_function(self.n_fft, "hann"),
            frame_length=self.n_fft,
            hop_length=self.hop_length,
            power=2.0,
            mel_filters=self.mel_filters.T,
            log_mel="dB",
            db_range=80.0,
        )
        log_spec = log_spec[:, :-1]  # 去除最后一列，用于兼容 Whisper 实现
        log_spec = log_spec - 20.0  # 对数梅尔频谱减去 20.0
        log_spec = np.clip(log_spec / 40.0, -2.0, 0.0) + 1.0  # 裁剪、缩放对数梅尔频谱
        return log_spec

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: Optional[bool] = True,
        sampling_rate: Optional[int] = None,
        resample: bool = False,
        mask_audio: bool = False,
        **kwargs,
```