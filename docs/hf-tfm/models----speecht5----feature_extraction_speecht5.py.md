# `.\models\speecht5\feature_extraction_speecht5.py`

```
# coding=utf-8
# 版权 2023 年 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非符合许可证的要求，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件，
# 没有任何形式的明示或暗示的担保或条件。
# 有关特定语言的权限，请参阅许可证。
"""SpeechT5 的特征提取器类。"""

import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np

# 导入音频处理工具函数
from ...audio_utils import mel_filter_bank, optimal_fft_length, spectrogram, window_function
# 导入特征提取序列工具函数
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
# 导入批次特征类
from ...feature_extraction_utils import BatchFeature
# 导入日志记录工具
from ...utils import PaddingStrategy, TensorType, logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义 SpeechT5FeatureExtractor 类，继承自 SequenceFeatureExtractor 类
class SpeechT5FeatureExtractor(SequenceFeatureExtractor):
    r"""
    构造一个 SpeechT5 特征提取器。

    此类可以通过（可选地）将原始语音信号归一化为零均值单位方差，以供 SpeechT5 语音编码器预网络使用。

    此类还可以从原始语音中提取对数梅尔滤波器组特征，以供 SpeechT5 语音解码器预网络使用。

    此特征提取器继承自 [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`]，其中包含大多数主要方法。
    用户应参考此超类以获取有关这些方法的更多信息。
    """
    """
    Args:
        feature_size (`int`, *optional*, defaults to 1):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        padding_value (`float`, *optional*, defaults to 0.0):
            The value that is used to fill the padding values.
        do_normalize (`bool`, *optional*, defaults to `False`):
            Whether or not to zero-mean unit-variance normalize the input. Normalizing can help to significantly
            improve the performance for some models.
        num_mel_bins (`int`, *optional*, defaults to 80):
            The number of mel-frequency bins in the extracted spectrogram features.
        hop_length (`int`, *optional*, defaults to 16):
            Number of ms between windows. Otherwise referred to as "shift" in many papers.
        win_length (`int`, *optional*, defaults to 64):
            Number of ms per window.
        win_function (`str`, *optional*, defaults to `"hann_window"`):
            Name for the window function used for windowing, must be accessible via `torch.{win_function}`
        frame_signal_scale (`float`, *optional*, defaults to 1.0):
            Constant multiplied in creating the frames before applying DFT. This argument is deprecated.
        fmin (`float`, *optional*, defaults to 80):
            Minimum mel frequency in Hz.
        fmax (`float`, *optional*, defaults to 7600):
            Maximum mel frequency in Hz.
        mel_floor (`float`, *optional*, defaults to 1e-10):
            Minimum value of mel frequency banks.
        reduction_factor (`int`, *optional*, defaults to 2):
            Spectrogram length reduction factor. This argument is deprecated.
        return_attention_mask (`bool`, *optional*, defaults to `True`):
            Whether or not [`~SpeechT5FeatureExtractor.__call__`] should return `attention_mask`.
    """

    # 初始定义模型的输入名称列表
    model_input_names = ["input_values", "attention_mask"]

    # 初始化函数，设置音频特征提取器的参数
    def __init__(
        self,
        feature_size: int = 1,
        sampling_rate: int = 16000,
        padding_value: float = 0.0,
        do_normalize: bool = False,
        num_mel_bins: int = 80,
        hop_length: int = 16,
        win_length: int = 64,
        win_function: str = "hann_window",
        frame_signal_scale: float = 1.0,
        fmin: float = 80,
        fmax: float = 7600,
        mel_floor: float = 1e-10,
        reduction_factor: int = 2,
        return_attention_mask: bool = True,
        **kwargs,
    ):
    ):
        # 调用父类初始化函数，设置特征大小、采样率、填充值等参数
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        # 是否进行归一化处理
        self.do_normalize = do_normalize
        # 是否返回注意力掩码
        self.return_attention_mask = return_attention_mask

        # 梅尔滤波器参数
        self.num_mel_bins = num_mel_bins
        self.hop_length = hop_length
        self.win_length = win_length
        self.win_function = win_function
        self.frame_signal_scale = frame_signal_scale
        self.fmin = fmin
        self.fmax = fmax
        self.mel_floor = mel_floor
        self.reduction_factor = reduction_factor

        # 根据窗口长度和采样率计算样本大小和样本步长
        self.sample_size = win_length * sampling_rate // 1000
        self.sample_stride = hop_length * sampling_rate // 1000
        # 计算最优的 FFT 长度
        self.n_fft = optimal_fft_length(self.sample_size)
        # 计算频率数量
        self.n_freqs = (self.n_fft // 2) + 1

        # 设置窗口函数
        self.window = window_function(window_length=self.sample_size, name=self.win_function, periodic=True)

        # 创建梅尔滤波器组
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=self.n_freqs,
            num_mel_filters=self.num_mel_bins,
            min_frequency=self.fmin,
            max_frequency=self.fmax,
            sampling_rate=self.sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )

        # 如果帧信号比例不为1.0，发出警告
        if frame_signal_scale != 1.0:
            warnings.warn(
                "The argument `frame_signal_scale` is deprecated and will be removed in version 4.30.0 of Transformers",
                FutureWarning,
            )
        # 如果减少因子不为2.0，发出警告
        if reduction_factor != 2.0:
            warnings.warn(
                "The argument `reduction_factor` is deprecated and will be removed in version 4.30.0 of Transformers",
                FutureWarning,
            )

    @staticmethod
    # 从 transformers.models.wav2vec2.feature_extraction_wav2vec2.Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm 复制而来
    def zero_mean_unit_var_norm(
        input_values: List[np.ndarray], attention_mask: List[np.ndarray], padding_value: float = 0.0
    ) -> List[np.ndarray]:
        """
        每个数组在列表中被归一化为零均值和单位方差
        """
        if attention_mask is not None:
            attention_mask = np.array(attention_mask, np.int32)
            normed_input_values = []

            for vector, length in zip(input_values, attention_mask.sum(-1)):
                # 计算归一化的切片，确保未定义的部分用填充值填充
                normed_slice = (vector - vector[:length].mean()) / np.sqrt(vector[:length].var() + 1e-7)
                if length < normed_slice.shape[0]:
                    normed_slice[length:] = padding_value

                normed_input_values.append(normed_slice)
        else:
            # 对于没有注意力掩码的情况，对所有输入值进行归一化
            normed_input_values = [(x - x.mean()) / np.sqrt(x.var() + 1e-7) for x in input_values]

        return normed_input_values

    def _extract_mel_features(
        self,
        one_waveform: np.ndarray,
    ) -> np.ndarray:
        """
        Extracts log-mel filterbank features for one waveform array (unbatched).
        """
        # Compute log-mel spectrogram features for a single waveform array
        log_mel_spec = spectrogram(
            one_waveform,
            window=self.window,
            frame_length=self.sample_size,
            hop_length=self.sample_stride,
            fft_length=self.n_fft,
            mel_filters=self.mel_filters,
            mel_floor=self.mel_floor,
            log_mel="log10",
        )
        # Transpose the spectrogram matrix
        return log_mel_spec.T

    def __call__(
        self,
        audio: Optional[Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]]] = None,
        audio_target: Optional[Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]]] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        sampling_rate: Optional[int] = None,
        **kwargs,
    ):
        """
        Process audio input according to specified parameters.
        """
        def _process_audio(
            self,
            speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
            is_target: bool = False,
            padding: Union[bool, str, PaddingStrategy] = False,
            max_length: Optional[int] = None,
            truncation: bool = False,
            pad_to_multiple_of: Optional[int] = None,
            return_attention_mask: Optional[bool] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            **kwargs,
        ):
            """
            Internal method to process audio data.
            """

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the object's properties to a dictionary representation.
        """
        # Start with the base class's dictionary representation
        output = super().to_dict()

        # Remove derived properties from serialization
        names = ["window", "mel_filters", "sample_size", "sample_stride", "n_fft", "n_freqs"]
        for name in names:
            if name in output:
                del output[name]

        return output
```