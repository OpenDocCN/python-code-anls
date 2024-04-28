# `.\transformers\models\tvlt\feature_extraction_tvlt.py`

```
# 设置代码文件的编码格式为 UTF-8
# 版权声明
# 根据 Apache 许可证版本 2.0 授权使用该文件
# 未经许可不得使用该文件
# 可以在下面链接找到该许可证
# http://www.apache.org/licenses/LICENSE-2.0
# 软件按照“现状”提供，没有任何明示或暗示的担保或条件
# 可以根据许可证获取更多详细信息
# 限制在许可证下开展的活动和设置的条件
# 用于 TVLT 的特征提取器类。

from math import ceil
from typing import List, Optional, Union

# 导入第三方库 numpy，使用缩写 np
import numpy as np

# 导入音频工具模块
from ...audio_utils import mel_filter_bank, spectrogram, window_function
# 导入序列特征提取器工具模块
from ...feature_extraction_sequence_utils import BatchFeature, SequenceFeatureExtractor
# 导入工具模块中的 TensorType 类型和 logging 函数
from ...utils import TensorType, logging

# 获取 logger
logger = logging.get_logger(__name__)

# TVLT 特征提取器类继承自序列特征提取器
class TvltFeatureExtractor(SequenceFeatureExtractor):
    r"""
    构建 TVLT 音频特征提取器。此特征提取器可用于为模型准备音频。

    此特征提取器继承自 [`FeatureExtractionMixin`]，其中包含大部分主要方法。用户应参考此超类以获取有关这些方法的更多信息。

    参数:
        spectrogram_length (`Dict[str, int]` *可选*, 默认为 2048):
            每个音频频谱图的时间长度。
        num_channels (`int` *可选*, 默认为 1):
            音频通道数。
        patch_size (`List[int]` *可选*, 默认为 `[16, 16]`):
            音频补丁嵌入的补丁大小。
        feature_size (`int`, *可选*, 默认为 128):
            音频频谱图的频率长度。
        sampling_rate (`int`, *可选*, 默认为 44100):
            音频文件应数字化的采样率，以赫兹（Hz）表示。
        hop_length_to_sampling_rate (`int`, *可选*, 默认为 86):
            Hop length 是用于获得 Mel 频率系数的 STFT 的重叠窗口的长度。
            例如，对于采样率 44100，跳跃长度为 512，即 44100 / 512 = 86
        n_fft (`int`, *可选*, 默认为 2048):
            傅里叶变换的大小。
        padding_value (`float`, *可选*, 默认为 0.0):
            用于填充音频的填充值。 应对应沉默。
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
    # 定义构造函数，设置参数 feature_size, sampling_rate, padding_value 和其它关键字参数
    def __init__(
        self,
        feature_size=feature_size,
        sampling_rate=sampling_rate,
        padding_value=padding_value,
        **kwargs,
    ):
        # 调用父类的构造函数，传入参数
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )

        # 初始化音频的频谱长度
        self.spectrogram_length = spectrogram_length
        # 初始化音频的通道数
        self.num_channels = num_channels
        # 初始化音频的 patch 尺寸
        self.patch_size = patch_size
        # 初始化频谱频率长度
        self.freq_len = feature_size // self.patch_size[1]
        # 初始化快速傅立叶变换中的采样点数
        self.n_fft = n_fft
        # 初始化帧移长度
        self.hop_length = sampling_rate // hop_length_to_sampling_rate
        # 初始化采样率
        self.sampling_rate = sampling_rate
        # 初始化填充值
        self.padding_value = padding_value
        # 初始化梅尔滤波器
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + n_fft // 2,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=22050.0,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        ).T

    # 定义私有函数，用于提取音频的梅尔频谱特征
    def _np_extract_fbank_features(self, waveform: np.array) -> np.ndarray:
        """
        Compute the log-mel spectrogram of the provided audio, gives similar results to Whisper's original torch
        implementation with 1e-5 tolerance.
        """
        # 计算音频的对数梅尔频谱
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
        # 删除最后一列数据
        log_spec = log_spec[:, :-1]
        # 数据减去 20
        log_spec = log_spec - 20.0
        # 将数据限制在 -2 到 0 之间，并加上 1
        log_spec = np.clip(log_spec / 40.0, -2.0, 0.0) + 1.0
        return log_spec

    # 定义 __call__ 方法，处理原始语音输入
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