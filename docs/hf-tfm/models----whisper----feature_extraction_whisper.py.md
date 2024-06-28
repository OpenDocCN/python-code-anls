# `.\models\whisper\feature_extraction_whisper.py`

```py
# 设置文件编码为 UTF-8，确保支持非英文字符的正确解析
# 版权声明，使用 Apache 许可证 2.0 版本
#
# 根据 Apache 许可证 2.0 版本规定，除非符合许可证要求，否则禁止使用本文件中的代码
#
# 引入必要的库和模块
"""
Feature extractor class for Whisper
"""
# 引入类型提示相关模块
from typing import List, Optional, Union

# 引入 NumPy 库，并使用 np 别名
import numpy as np

# 引入 Hugging Face 提供的 is_torch_available 函数，用于检查是否安装了 Torch
from ... import is_torch_available

# 从 Hugging Face 的 audio_utils 模块中引入 mel_filter_bank、spectrogram 和 window_function 函数
from ...audio_utils import mel_filter_bank, spectrogram, window_function

# 从 feature_extraction_sequence_utils 模块中引入 SequenceFeatureExtractor 类
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor

# 从 feature_extraction_utils 模块中引入 BatchFeature 类
from ...feature_extraction_utils import BatchFeature

# 从 utils 模块中引入 TensorType 和 logging 函数
from ...utils import TensorType, logging

# 如果 Torch 可用，导入 Torch 库
if is_torch_available():
    import torch

# 从 logging 模块中获取 logger 对象，并命名为 __name__，用于日志记录
logger = logging.get_logger(__name__)

# 定义 WhisperFeatureExtractor 类，继承自 SequenceFeatureExtractor 类
class WhisperFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Whisper feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts mel-filter bank features from raw speech using a custom numpy implementation of the `Short Time
    Fourier Transform` which should match pytorch's `torch.stft` equivalent.

    Args:
        feature_size (`int`, defaults to 80):
            The feature dimension of the extracted features.
        sampling_rate (`int`, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        hop_length (`int`, defaults to 160):
            Length of the overlaping windows for the STFT used to obtain the Mel Frequency coefficients.
        chunk_length (`int`, defaults to 30):
            The maximum number of chuncks of `sampling_rate` samples used to trim and pad longer or shorter audio
            sequences.
        n_fft (`int`, defaults to 400):
            Size of the Fourier transform.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio. Should correspond to silences.
    """

    # 类变量 model_input_names，指定输入模型的名称为 "input_features"
    model_input_names = ["input_features"]

    # 初始化方法，用于创建 WhisperFeatureExtractor 实例
    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
        padding_value=0.0,
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
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + n_fft // 2,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )


    def _np_extract_fbank_features(self, waveform: np.array) -> np.ndarray:
        """
        Compute the log-mel spectrogram of the provided audio, gives similar results to Whisper's original torch
        implementation with 1e-5 tolerance.
        """
        log_spec = spectrogram(
            waveform,
            window_function(self.n_fft, "hann"),
            frame_length=self.n_fft,
            hop_length=self.hop_length,
            power=2.0,
            mel_filters=self.mel_filters,
            log_mel="log10",
        )
        log_spec = log_spec[:, :-1]  # Remove the last frame to match expected shape
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)  # Clamp values to ensure numerical stability
        log_spec = (log_spec + 4.0) / 4.0  # Scale values to a range suitable for neural network input
        return log_spec


    def _torch_extract_fbank_features(self, waveform: np.array) -> np.ndarray:
        """
        Compute the log-mel spectrogram of the provided audio using the PyTorch STFT implementation.
        """
        waveform = torch.from_numpy(waveform).type(torch.float32)  # Convert waveform to a PyTorch tensor of float32 type

        window = torch.hann_window(self.n_fft)  # Create a Hann window tensor for STFT
        stft = torch.stft(waveform, self.n_fft, self.hop_length, window=window, return_complex=True)  # Compute STFT of waveform

        magnitudes = stft[..., :-1].abs() ** 2  # Compute magnitude squared of STFT, excluding the last frame

        mel_filters = torch.from_numpy(self.mel_filters).type(torch.float32)  # Convert mel filters to PyTorch tensor
        mel_spec = mel_filters.T @ magnitudes  # Apply mel filters to the magnitude spectrogram

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()  # Apply logarithm after clamping for numerical stability
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)  # Clamp values to ensure numerical stability
        log_spec = (log_spec + 4.0) / 4.0  # Scale values to a range suitable for neural network input
        return log_spec.numpy()  # Convert back to NumPy array for compatibility


    @staticmethod
    # Copied from transformers.models.wav2vec2.feature_extraction_wav2vec2.Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm
    def zero_mean_unit_var_norm(
        input_values: List[np.ndarray], attention_mask: List[np.ndarray], padding_value: float = 0.0
    ) -> List[np.ndarray]:
        """
        Every array in the list is normalized to have zero mean and unit variance
        """
        # 如果提供了注意力掩码，则将其转换为 np.int32 类型的数组
        if attention_mask is not None:
            attention_mask = np.array(attention_mask, np.int32)
            # 初始化一个空列表来存储归一化后的输入值
            normed_input_values = []

            # 遍历输入值列表和对应的注意力掩码长度
            for vector, length in zip(input_values, attention_mask.sum(-1)):
                # 计算当前向量的均值和方差，并进行归一化处理
                normed_slice = (vector - vector[:length].mean()) / np.sqrt(vector[:length].var() + 1e-7)
                # 如果当前向量长度小于归一化后的切片长度，则填充指定的 padding_value
                if length < normed_slice.shape[0]:
                    normed_slice[length:] = padding_value

                # 将归一化后的切片添加到结果列表中
                normed_input_values.append(normed_slice)
        else:
            # 如果没有提供注意力掩码，则对输入值列表中的每个数组进行归一化处理
            normed_input_values = [(x - x.mean()) / np.sqrt(x.var() + 1e-7) for x in input_values]

        # 返回归一化后的输入值列表
        return normed_input_values

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        truncation: bool = True,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: Optional[bool] = None,
        padding: Optional[str] = "max_length",
        max_length: Optional[int] = None,
        sampling_rate: Optional[int] = None,
        do_normalize: Optional[bool] = None,
        **kwargs,
```