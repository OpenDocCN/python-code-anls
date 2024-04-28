# `.\transformers\models\clvp\feature_extraction_clvp.py`

```
# coding=utf-8
# 声明文件编码为 UTF-8
# Copyright 2023 The HuggingFace Inc. team.
# 版权声明，指明版权归 The HuggingFace Inc. team 所有
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 以 Apache 许可证版本 2.0 进行许可
# you may not use this file except in compliance with the License.
# 除非符合许可证的规定，否则不能使用该文件
# You may obtain a copy of the License at
# 你可以在以下网址获取许可证的一份拷贝
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 在适用法律允许的范围内，按“原样”提供软件，不提供任何担保或条件，无论是明示的还是暗示的。
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 没有任何形式的担保或条件，无论是明示的还是暗示的
# See the License for the specific language governing permissions and
# limitations under the License.
# 请参阅许可证，了解特定语言的权限和限制。
#
"""
Feature extractor class for CLVP
"""
# CLVP 特征提取器类

from typing import List, Optional, Union

import numpy as np

from ...audio_utils import mel_filter_bank, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, logging

# 导入必要的模块和类

logger = logging.get_logger(__name__)

# 获取 logger 实例

class ClvpFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a CLVP feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts log-mel-spectrogram features from raw speech using a custom numpy implementation of the `Short
    Time Fourier Transform` which should match pytorch's `torch.stft` equivalent.
    """
    # 构造一个 CLVP 特征提取器
    # 该特征提取器继承自 [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`]，其中包含大多数主要方法。用户应参考此超类，了解有关这些方法的更多信息。
    # 此类从原始语音中提取对数梅尔频谱图特征，使用自定义的 numpy 实现了 `短时傅里叶变换`，应与 pytorch 的 `torch.stft` 等效。
``` 
    Args:
        feature_size (`int`, *optional*, defaults to 80):
            提取特征的特征维度。
        sampling_rate (`int`, *optional*, defaults to 22050):
            表示音频文件应以何种频率（赫兹）进行数字化的采样率。
        default_audio_length (`int`, *optional*, defaults to 6):
            原始音频的默认长度，以秒为单位。如果在 `__call__` 中未设置 `max_length`，则默认为 `default_audio_length` * `self.sampling_rate`。
        hop_length (`int`, *optional*, defaults to 256):
            用于获取梅尔频率系数的 STFT 中的重叠窗口的长度。
        chunk_length (`int`, *optional*, defaults to 30):
            用于修剪和填充较长或较短音频序列的 `sampling_rate` 个样本的最大块数。
        n_fft (`int`, *optional*, defaults to 1024):
            傅立叶变换的大小。
        padding_value (`float`, *optional*, defaults to 0.0):
            用于填充音频的填充值。应对应于静音。
        mel_norms (`list` of length `feature_size`, *optional*):
            如果提供了 `mel_norms`，则将用于沿每个梅尔滤波器对数梅尔频谱进行归一化。
        return_attention_mask (`bool`, *optional*, defaults to `False`):
            是否返回注意力掩码。如果保持默认设置，它将返回注意力掩码。

            [注意力掩码是什么？](../glossary#attention-mask)
    """

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
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sampling_rate
        self.nb_max_frames = self.n_samples // hop_length
        self.sampling_rate = sampling_rate
        self.default_audio_length = default_audio_length
        self.mel_norms = mel_norms
        # 创建梅尔滤波器组成的数组
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
        # 计算提供音频的对数梅尔频谱图，然后根据 `mel_norms` 进行每个梅尔滤波器的归一化
        log_spec = spectrogram(
            waveform,  # 输入音频波形
            window_function(self.n_fft, "hann"),  # 汉宁窗口函数
            frame_length=self.n_fft,  # 帧长度
            hop_length=self.hop_length,  # 帧之间的跳跃长度
            power=2.0,  # 指定功率
            mel_filters=self.mel_filters,  # 梅尔滤波器的参数
            log_mel=None,  # 不进行对数变换
        )

        # 对计算出的对数梅尔频谱图进行对数变换，并进行截断和裁剪
        log_spec = np.log(np.clip(log_spec, a_min=1e-5, a_max=None))

        # 如果提供了 `mel_norms`，则对对数梅尔频谱图进行归一化处理
        if self.mel_norms is not None:
            log_spec = log_spec / np.array(self.mel_norms)[:, None]

        # 返回对数梅尔频谱图
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