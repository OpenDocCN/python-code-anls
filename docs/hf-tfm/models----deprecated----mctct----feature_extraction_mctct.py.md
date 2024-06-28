# `.\models\deprecated\mctct\feature_extraction_mctct.py`

```py
# coding=utf-8
# 声明代码文件采用UTF-8编码格式

# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
# 版权声明，保留所有权利

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据Apache许可证2.0版本授权使用该代码

# you may not use this file except in compliance with the License.
# 只有在遵守许可证的情况下才能使用本文件

# You may obtain a copy of the License at
# 您可以在以下网址获取许可证的副本

#     http://www.apache.org/licenses/LICENSE-2.0
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非法律要求或书面同意，否则依据许可证分发的软件都是基于“原样”提供的，不带任何形式的担保或条件。

# See the License for the specific language governing permissions and
# limitations under the License.
# 请查阅许可证，了解具体语言授权和限制

"""
Feature extractor class for M-CTC-T
"""

# 导入必要的库和模块
from typing import List, Optional, Union

import numpy as np

# 导入音频处理相关的工具函数和类
from ....audio_utils import mel_filter_bank, optimal_fft_length, spectrogram, window_function
# 导入特征提取的序列工具函数和类
from ....feature_extraction_sequence_utils import SequenceFeatureExtractor
# 导入特征提取相关的批量特征类
from ....feature_extraction_utils import BatchFeature
# 导入文件处理相关的工具函数和类
from ....file_utils import PaddingStrategy, TensorType
# 导入日志记录工具
from ....utils import logging

# 获取本模块的日志记录器
logger = logging.get_logger(__name__)

# 定义 M-CTC-T 特征提取器类，继承自 SequenceFeatureExtractor 类
class MCTCTFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a M-CTC-T feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods. This
    code has been adapted from Flashlight's C++ code. For more information about the implementation, one can refer to
    this [notebook](https://colab.research.google.com/drive/1GLtINkkhzms-IsdcGy_-tVCkv0qNF-Gt#scrollTo=pMCRGMmUC_an)
    that takes the user step-by-step in the implementation.
    """
    """
    Args:
        feature_size (`int`, defaults to 80):
            The feature dimension of the extracted features. This is the number of mel_frequency
            coefficients to compute per frame.
        sampling_rate (`int`, defaults to 16000):
            The sampling rate at which the audio files should be digitized, expressed in hertz (Hz).
        padding_value (`float`, defaults to 0.0):
            The value used to fill the padding frames.
        hop_length (`int`, defaults to 10):
            Number of audio samples between consecutive frames.
        win_length (`int`, defaults to 25):
            Length of the window function applied to each frame, in milliseconds.
        win_function (`str`, defaults to `"hamming_window"`):
            Name for the window function used for windowing, must be accessible via `torch.{win_function}`.
        frame_signal_scale (`float`, defaults to 32768.0):
            Scaling factor applied to the audio frames before applying Discrete Fourier Transform (DFT).
        preemphasis_coeff (`float`, defaults to 0.97):
            Coefficient applied in pre-emphasis filtering of audio signals before DFT.
        mel_floor (`float`, defaults to 1.0):
            Minimum value enforced for mel frequency bank values.
        normalize_means (`bool`, *optional*, defaults to `True`):
            Whether to zero-mean normalize the extracted features.
        normalize_vars (`bool`, *optional*, defaults to `True`):
            Whether to unit-variance normalize the extracted features.
    """

    # List of input names expected by the model
    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        padding_value=0.0,
        hop_length=10,
        win_length=25,
        win_function="hamming_window",
        frame_signal_scale=32768.0,
        preemphasis_coeff=0.97,
        mel_floor=1.0,
        normalize_means=True,
        normalize_vars=True,
        return_attention_mask=False,
        **kwargs,
    ):
        # Call parent constructor with specified arguments
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)

        # Initialize instance variables with provided or default values
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value
        self.hop_length = hop_length
        self.win_length = win_length
        self.frame_signal_scale = frame_signal_scale
        self.preemphasis_coeff = preemphasis_coeff
        self.mel_floor = mel_floor
        self.normalize_means = normalize_means
        self.normalize_vars = normalize_vars
        self.win_function = win_function
        self.return_attention_mask = return_attention_mask

        # Calculate sample size and stride in samples
        self.sample_size = win_length * sampling_rate // 1000
        self.sample_stride = hop_length * sampling_rate // 1000

        # Determine the optimal FFT length based on sample size
        self.n_fft = optimal_fft_length(self.sample_size)
        # Calculate number of frequencies in the FFT output
        self.n_freqs = (self.n_fft // 2) + 1
    # Extracts MFSC (Mel Frequency Spectral Coefficients) features from a single waveform vector.
    # Adapted from Flashlight's C++ MFSC code.
    def _extract_mfsc_features(self, one_waveform: np.array) -> np.ndarray:
        # Determine the window function based on the specified type ('hamming_window' or default).
        if self.win_function == "hamming_window":
            window = window_function(window_length=self.sample_size, name=self.win_function, periodic=False)
        else:
            window = window_function(window_length=self.sample_size, name=self.win_function)

        # Compute mel filter banks for the given audio properties.
        fbanks = mel_filter_bank(
            num_frequency_bins=self.n_freqs,
            num_mel_filters=self.feature_size,
            min_frequency=0.0,
            max_frequency=self.sampling_rate / 2.0,
            sampling_rate=self.sampling_rate,
        )

        # Compute MFSC features using the spectrogram function.
        msfc_features = spectrogram(
            one_waveform * self.frame_signal_scale,  # Scale the waveform
            window=window,
            frame_length=self.sample_size,
            hop_length=self.sample_stride,
            fft_length=self.n_fft,
            center=False,
            preemphasis=self.preemphasis_coeff,
            mel_filters=fbanks,
            mel_floor=self.mel_floor,
            log_mel="log",  # Logarithmic mel scaling
        )
        
        # Transpose the features to have time steps as rows and features as columns.
        return msfc_features.T

    # Normalize a single array 'x' based on specified normalization options and input length.
    def _normalize_one(self, x, input_length, padding_value):
        # Ensure that we normalize float32 arrays.
        if self.normalize_means:
            mean = x[:input_length].mean(axis=0)
            x = np.subtract(x, mean)
        if self.normalize_vars:
            std = x[:input_length].std(axis=0)
            x = np.divide(x, std)

        # If the input length is less than the array length, pad the array.
        if input_length < x.shape[0]:
            x[input_length:] = padding_value

        # Ensure the array is of type float32.
        x = x.astype(np.float32)

        return x

    # Normalize a list of input features, optionally using an attention mask.
    def normalize(
        self, input_features: List[np.ndarray], attention_mask: Optional[np.ndarray] = None
    ) -> List[np.ndarray]:
        # Calculate lengths of sequences based on attention mask if available, otherwise use array lengths.
        lengths = attention_mask.sum(-1) if attention_mask is not None else [x.shape[0] for x in input_features]
        # Normalize each input feature array using _normalize_one method and return the normalized list.
        return [self._normalize_one(x, n, self.padding_value) for x, n in zip(input_features, lengths)]

    # Callable method for preprocessing raw speech data with various options.
    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        padding: Union[bool, str, PaddingStrategy] = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        sampling_rate: Optional[int] = None,
        **kwargs,
    ):
```