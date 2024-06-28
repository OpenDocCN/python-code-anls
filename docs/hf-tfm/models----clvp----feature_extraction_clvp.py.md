# `.\models\clvp\feature_extraction_clvp.py`

```
# coding=utf-8
# 定义了文件编码格式为 UTF-8

# 版权声明，声明代码版权归 HuggingFace Inc. 团队所有
# 根据 Apache 许可证版本 2.0 发布，除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0

# 如果适用法律要求或书面同意，以“原样”方式分发本软件，不提供任何形式的担保或条件
# 请参阅许可证了解具体的法律条文及其约束
# 此处导入需要的模块和类
"""
Feature extractor class for CLVP
"""

# 导入必要的模块
from typing import List, Optional, Union

# 导入 NumPy 库，用于处理数值计算
import numpy as np

# 导入音频相关的工具函数
from ...audio_utils import mel_filter_bank, spectrogram, window_function

# 导入特征提取的序列工具函数
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor

# 导入特征提取的批处理功能
from ...feature_extraction_utils import BatchFeature

# 导入自定义的张量类型和日志记录工具
from ...utils import TensorType, logging

# 获取当前文件的日志记录器
logger = logging.get_logger(__name__)


# 定义 CLVP 特征提取器类，继承自 SequenceFeatureExtractor 类
class ClvpFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a CLVP feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts log-mel-spectrogram features from raw speech using a custom numpy implementation of the `Short
    Time Fourier Transform` which should match pytorch's `torch.stft` equivalent.
    """
    Args:
        feature_size (`int`, *optional*, defaults to 80):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 22050):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        default_audio_length (`int`, *optional*, defaults to 6):
            The default length of raw audio in seconds. If `max_length` is not set during `__call__` then it will
            automatically be set to default_audio_length * `self.sampling_rate`.
        hop_length (`int`, *optional*, defaults to 256):
            Length of the overlapping windows for the STFT used to obtain the Mel Frequency coefficients.
        chunk_length (`int`, *optional*, defaults to 30):
            The maximum number of chunks of `sampling_rate` samples used to trim and pad longer or shorter audio
            sequences.
        n_fft (`int`, *optional*, defaults to 1024):
            Size of the Fourier transform.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio. Should correspond to silences.
        mel_norms (`list` of length `feature_size`, *optional*):
            If `mel_norms` is provided then it will be used to normalize the log-mel spectrograms along each
            mel-filter.
        return_attention_mask (`bool`, *optional*, defaults to `False`):
            Whether to return the attention mask. If left to the default, it will return the attention mask.

            [What are attention masks?](../glossary#attention-mask)
    """
    # 定义模型输入的名称，包括输入特征和注意力掩码
    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        feature_size=80,
        sampling_rate=22050,
        default_audio_length=6,
        hop_length=256,
        chunk_length=30,
        n_fft=1024,
        padding_value=0.0,
        mel_norms=None,
        return_attention_mask=False,  # pad inputs to max length with silence token (zero) and no attention mask
        **kwargs,
    ):
        # 调用父类的初始化方法，设置基本参数
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        # 设置其他参数和属性
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sampling_rate  # 计算每个片段的采样数
        self.nb_max_frames = self.n_samples // hop_length  # 计算最大帧数
        self.sampling_rate = sampling_rate
        self.default_audio_length = default_audio_length
        self.mel_norms = mel_norms
        # 计算梅尔滤波器组
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + (n_fft // 2),
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="htk",
        )
    def _np_extract_fbank_features(self, waveform: np.array) -> np.ndarray:
        """
        This method first computes the log-mel spectrogram of the provided audio then applies normalization along the
        each mel-filterbank, if `mel_norms` is provided.
        """
        # 计算音频的对数梅尔频谱图
        log_spec = spectrogram(
            waveform,
            window_function(self.n_fft, "hann"),
            frame_length=self.n_fft,
            hop_length=self.hop_length,
            power=2.0,
            mel_filters=self.mel_filters,
            log_mel=None,
        )

        # 对计算得到的对数梅尔频谱图进行对数处理，并进行上下限裁剪
        log_spec = np.log(np.clip(log_spec, a_min=1e-5, a_max=None))

        # 如果提供了 `mel_norms`，则对对数梅尔频谱图进行归一化
        if self.mel_norms is not None:
            log_spec = log_spec / np.array(self.mel_norms)[:, None]

        # 返回处理后的对数梅尔频谱图作为结果
        return log_spec

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        sampling_rate: Optional[int] = None,
        truncation: bool = True,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: Optional[bool] = True,
        padding: Optional[str] = "max_length",
        max_length: Optional[int] = None,
        **kwargs,
```