# `.\transformers\models\seamless_m4t\feature_extraction_seamless_m4t.py`

```py
# 导入所需的库和模块
import numpy as np
from typing import List, Optional, Union
from ...utils import is_torch_available
if is_torch_available():
    import torch
from ...audio_utils import mel_filter_bank, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 SeamlessM4TFeatureExtractor 类, 该类继承自 SequenceFeatureExtractor
class SeamlessM4TFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a SeamlessM4T feature extractor.

    This feature extractor inherits from [`SequenceFeatureExtractor`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    This class extracts mel-filter bank features from raw speech.

    Args:
        feature_size (`int`, *optional*, defaults to 80):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        num_mel_bins (`int`, *optional*, defaults to 80):
            Number of Mel-frequency bins.
        padding_value (`float`, *optional*, defaults to 0.0):
            The value that is used to fill the padding vectors.
        stride (`int`, *optional*, defaults to 2):
            Stride used to reshape audios from shape (batch_size,num_frames,num_mel_bins) to
            (batch_size,num_frames//stride,num_mel_bins*stride).
    """

    # 定义模型输入名称
    model_input_names = ["input_features", "attention_mask"]

    # 初始化方法
    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        num_mel_bins=80,
        padding_value=0.0,
        stride=2,
        **kwargs,
    ):
    # 定义类的构造函数
    def __init__(
        self,
        num_mel_bins: int = 80,
        stride: int = 320,
        sampling_rate: int = 16000,
        feature_size: int = 1,
        padding_value: float = 0.0,
        **kwargs,
    ):
        # 设置类的属性
        self.num_mel_bins = num_mel_bins
        self.return_attention_mask = True
        self.stride = stride

        # 使用给定参数生成 Mel 滤波器组
        mel_filters = mel_filter_bank(
            num_frequency_bins=256,  # 频率分辨率的数量
            num_mel_filters=self.num_mel_bins,  # Mel 滤波器的数量
            min_frequency=20,  # 最低频率
            max_frequency=sampling_rate // 2,  # 最高频率
            sampling_rate=sampling_rate,  # 采样率
            norm=None,
            mel_scale="kaldi",
            triangularize_in_mel_space=True,
        )

        # 对 Mel 滤波器组进行补0操作
        self.mel_filters = np.pad(mel_filters, ((0, 1), (0, 0)))

        # 生成 400 点的波形窗口函数
        self.window = window_function(400, "povey", periodic=False)

        # 调用父类构造函数初始化超类
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)

    @staticmethod
    # 从 transformers.models.wav2vec2.feature_extraction_wav2vec2.Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm 复制过来
    def zero_mean_unit_var_norm(
        input_values: List[np.ndarray], attention_mask: List[np.ndarray], padding_value: float = 0.0
    ) -> List[np.ndarray]:
        """
        将输入列表中的每个数组归一化为零均值和单位方差
        """
        # 如果 attention_mask 不为 None，则进行归一化处理
        if attention_mask is not None:
            attention_mask = np.array(attention_mask, np.int32)
            normed_input_values = []

            # 遍历输入值和注意力掩码的列表，对每个数组进行归一化处理
            for vector, length in zip(input_values, attention_mask.sum(-1)):
                normed_slice = (vector - vector[:length].mean()) / np.sqrt(vector[:length].var() + 1e-7)
                if length < normed_slice.shape[0]:
                    normed_slice[length:] = padding_value

                normed_input_values.append(normed_slice)
        else:
            # 如果 attention_mask 为 None，则直接对输入值进行归一化处理
            normed_input_values = [(x - x.mean()) / np.sqrt(x.var() + 1e-7) for x in input_values]

        # 返回归一化后的结果列表
        return normed_input_values

    def _extract_fbank_features(
        self,
        waveform: np.ndarray,
    ) -> np.ndarray:
        """
        使用 TorchAudio 获得 Mel 滤波器组特征。注意，TorchAudio 要��输入的波形为 16 位有符号整数，因此在特征提取之前，波形不应进行标准化处理。
        """
        # 默认情况下，如果是双声道，则提取左声道
        if len(waveform.shape) == 2:
            waveform = waveform[0]

        waveform = np.squeeze(waveform) * (2**15)  # Kaldi 的兼容性要求：16 位有符号整数
        features = spectrogram(
            waveform,
            self.window,
            frame_length=400,
            hop_length=160,
            fft_length=512,
            power=2.0,
            center=False,
            preemphasis=0.97,
            mel_filters=self.mel_filters,
            log_mel="log",
            mel_floor=1.192092955078125e-07,
            remove_dc_offset=True,
        ).T
        return features
    # 定义一个魔术方法，使对象可以像函数一样被调用
    def __call__(
        # 声明输入参数raw_speech，可以是numpy数组、浮点数列表、numpy数组列表或浮点数二维列表
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        # 声明填充参数，可以是布尔值、字符串或填充策略对象，默认为True
        padding: Union[bool, str, PaddingStrategy] = True,
        # 声明填充到的倍数，可选参数，默认为2
        pad_to_multiple_of: Optional[int] = 2,
        # 声明最大长度，可选参数，默认为None
        max_length: Optional[int] = None,
        # 声明截断策略，可选参数，默认为False
        truncation: bool = False,
        # 声明返回的张量类型，可选参数，默认为None
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 声明采样率，可选参数，默认为None
        sampling_rate: Optional[int] = None,
        # 声明是否返回注意力掩码，可选参数，默认为None
        return_attention_mask: Optional[bool] = None,
        # 声明是否对每个mel频带进行归一化，可选参数，默认为True
        do_normalize_per_mel_bins: Optional[bool] = True,
        # **kwargs接收额外的关键字参数
        **kwargs,
```