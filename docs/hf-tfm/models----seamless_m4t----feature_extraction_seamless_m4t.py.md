# `.\models\seamless_m4t\feature_extraction_seamless_m4t.py`

```
"""
Feature extractor class for SeamlessM4T
"""

# 导入所需的模块和类
from typing import List, Optional, Union  # 引入类型提示所需的类和函数

import numpy as np  # 引入 NumPy 库，用于数值计算

# 检查是否可用 Torch，如果可用则导入
from ...utils import is_torch_available
if is_torch_available():
    import torch

# 导入音频处理相关的函数和类
from ...audio_utils import mel_filter_bank, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义 SeamlessM4TFeatureExtractor 类，继承自 SequenceFeatureExtractor
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

    # 定义模型输入的名称列表
    model_input_names = ["input_features", "attention_mask"]

    # 初始化方法，定义特征提取器的参数
    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        num_mel_bins=80,
        padding_value=0.0,
        stride=2,
        **kwargs,
    ):
    ):
        self.num_mel_bins = num_mel_bins  # 设置类的属性：梅尔滤波器的数量
        self.return_attention_mask = True  # 设置类的属性：是否返回注意力掩码
        self.stride = stride  # 设置类的属性：步长

        mel_filters = mel_filter_bank(
            num_frequency_bins=256,  # 频率分箱数量
            num_mel_filters=self.num_mel_bins,  # 梅尔滤波器的数量
            min_frequency=20,  # 最小频率
            max_frequency=sampling_rate // 2,  # 最大频率
            sampling_rate=sampling_rate,  # 采样率
            norm=None,  # 归一化方式
            mel_scale="kaldi",  # 梅尔标度
            triangularize_in_mel_space=True,  # 在梅尔空间中三角化
        )

        self.mel_filters = np.pad(mel_filters, ((0, 1), (0, 0)))  # 对梅尔滤波器进行填充操作
        self.window = window_function(400, "povey", periodic=False)  # 创建窗函数对象，窗长为400，类型为'povey'，非周期性

        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)

    @staticmethod
    # 从 transformers.models.wav2vec2.feature_extraction_wav2vec2.Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm 复制而来
    def zero_mean_unit_var_norm(
        input_values: List[np.ndarray], attention_mask: List[np.ndarray], padding_value: float = 0.0
    ) -> List[np.ndarray]:
        """
        每个数组都被归一化为零均值和单位方差
        """
        if attention_mask is not None:
            attention_mask = np.array(attention_mask, np.int32)  # 将注意力掩码转换为NumPy数组类型
            normed_input_values = []

            for vector, length in zip(input_values, attention_mask.sum(-1)):
                normed_slice = (vector - vector[:length].mean()) / np.sqrt(vector[:length].var() + 1e-7)  # 计算归一化后的切片
                if length < normed_slice.shape[0]:
                    normed_slice[length:] = padding_value  # 如果长度小于切片的形状，则使用填充值填充

                normed_input_values.append(normed_slice)  # 将归一化后的切片添加到列表中
        else:
            normed_input_values = [(x - x.mean()) / np.sqrt(x.var() + 1e-7) for x in input_values]  # 对每个输入值进行归一化处理

        return normed_input_values  # 返回归一化后的输入值列表

    def _extract_fbank_features(
        self,
        waveform: np.ndarray,
    ) -> np.ndarray:
        """
        使用 TorchAudio 获取梅尔滤波器组特征。注意，TorchAudio 要求输入为16位有符号整数，因此在特征提取之前不应对波形进行归一化。
        """
        # 默认情况下，如果是立体声，则提取左声道
        if len(waveform.shape) == 2:
            waveform = waveform[0]

        waveform = np.squeeze(waveform) * (2**15)  # Kaldi兼容性要求：16位有符号整数
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
        return features  # 返回梅尔滤波器组特征
    # 定义一个 __call__ 方法，使对象可以像函数一样被调用
    def __call__(
        # raw_speech 参数可以是 numpy 数组、浮点数列表、numpy 数组列表或浮点数列表的列表
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        # padding 参数用于控制是否进行填充，可以是布尔值、字符串或填充策略对象
        padding: Union[bool, str, PaddingStrategy] = True,
        # pad_to_multiple_of 参数指定填充到的长度的倍数，可选
        pad_to_multiple_of: Optional[int] = 2,
        # max_length 参数指定最大长度，可选
        max_length: Optional[int] = None,
        # truncation 参数控制是否截断序列，布尔值
        truncation: bool = False,
        # return_tensors 参数指定返回的张量类型，可选
        return_tensors: Optional[Union[str, TensorType]] = None,
        # sampling_rate 参数指定采样率，可选
        sampling_rate: Optional[int] = None,
        # return_attention_mask 参数控制是否返回注意力掩码，可选
        return_attention_mask: Optional[bool] = None,
        # do_normalize_per_mel_bins 参数控制是否对每个 Mel 频段进行归一化，默认为 True
        do_normalize_per_mel_bins: Optional[bool] = True,
        # **kwargs 表示接受额外的关键字参数
        **kwargs,
```