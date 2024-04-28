# `.\models\deprecated\mctct\feature_extraction_mctct.py`

```
# 设置编码格式为 utf-8

# 版权声明

# 导入所需的模块和包
from typing import List, Optional, Union
import numpy as np
# 导入音频工具模块中的函数和类
from ....audio_utils import mel_filter_bank, optimal_fft_length, spectrogram, window_function
# 导入特征提取序列工具模块中的类和函数
from ....feature_extraction_sequence_utils import SequenceFeatureExtractor
# 导入特征提取工具模块中的类和函数
from ....feature_extraction_utils import BatchFeature
# 导入文件工具模块中的类和函数
from ....file_utils import PaddingStrategy, TensorType
# 导入日志工具模块中的函数
from ....utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 M-CTC-T 特征提取器类，继承自 SequenceFeatureExtractor 类
class MCTCTFeatureExtractor(SequenceFeatureExtractor):
    r"""
    构建一个 M-CTC-T 特征提取器。

    该特征提取器继承自 [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`]，其中包含大部分主要方法。
    用户应该参考这个超类以获取有关这些方法的更多信息。该代码已经改编自 Flashlight 的 C++ 代码。
    有关实现的更多信息，可以参考这个 [notebook](https://colab.research.google.com/drive/1GLtINkkhzms-IsdcGy_-tVCkv0qNF-Gt#scrollTo=pMCRGMmUC_an)
    ，该 notebook 逐步引导用户实现。
    # 定义函数参数及其默认值
    Args:
        feature_size (`int`, defaults to 80):
            提取特征的特征维度。这是梅尔频率的数量。
        sampling_rate (`int`, defaults to 16000):
            音频文件应该被数字化的采样率，以赫兹（Hz）表示。
        padding_value (`float`, defaults to 0.0):
            用于填充值的数值。
        hop_length (`int`, defaults to 10):
            窗口之间的音频样本数。在许多论文中称为“shift”。
        win_length (`int`, defaults to 25):
            每个窗口的毫秒数。
        win_function (`str`, defaults to `"hamming_window"`):
            用于窗口处理的窗口函数的名称，必须可以通过 `torch.{win_function}` 访问。
        frame_signal_scale (`float`, defaults to 32768.0):
            在应用 DFT 之前创建帧时的乘数。
        preemphasis_coeff (`float`, defaults to 0.97):
            在应用 DFT 之前应用预加重时的乘数。
        mel_floor (`float` defaults to 1.0):
            梅尔频率组的最小值。
        normalize_means (`bool`, *optional*, defaults to `True`):
            是否对提取的特征进行零均值归一化。
        normalize_vars (`bool`, *optional*, defaults to `True`):
            是否对提取的特征进行单位方差归一化。
    """

    # 定义模型输入的名称
    model_input_names = ["input_features", "attention_mask"]

    # 初始化函数
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
        # 调用父类的初始化函数
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)

        # 设置类的属性
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

        # 计算样本大小和样本步长
        self.sample_size = win_length * sampling_rate // 1000
        self.sample_stride = hop_length * sampling_rate // 1000

        # 计算最佳 FFT 长度和频率数量
        self.n_fft = optimal_fft_length(self.sample_size)
        self.n_freqs = (self.n_fft // 2) + 1
    def _extract_mfsc_features(self, one_waveform: np.array) -> np.ndarray:
        """
        Extracts MFSC Features for one waveform vector (unbatched). Adapted from Flashlight's C++ MFSC code.
        """
        # 根据所选窗函数类型生成窗函数对象
        if self.win_function == "hamming_window":
            window = window_function(window_length=self.sample_size, name=self.win_function, periodic=False)
        else:
            window = window_function(window_length=self.sample_size, name=self.win_function)

        # 生成梅尔滤波器组
        fbanks = mel_filter_bank(
            num_frequency_bins=self.n_freqs,
            num_mel_filters=self.feature_size,
            min_frequency=0.0,
            max_frequency=self.sampling_rate / 2.0,
            sampling_rate=self.sampling_rate,
        )

        # 计算 MFSC 特征
        msfc_features = spectrogram(
            one_waveform * self.frame_signal_scale,
            window=window,
            frame_length=self.sample_size,
            hop_length=self.sample_stride,
            fft_length=self.n_fft,
            center=False,
            preemphasis=self.preemphasis_coeff,
            mel_filters=fbanks,
            mel_floor=self.mel_floor,
            log_mel="log",
        )
        return msfc_features.T

    def _normalize_one(self, x, input_length, padding_value):
        # 确保输入数组是 float32 类型
        # 根据需要进行均值归一化
        if self.normalize_means:
            mean = x[:input_length].mean(axis=0)
            x = np.subtract(x, mean)
        # 根据需要进行方差归一化
        if self.normalize_vars:
            std = x[:input_length].std(axis=0)
            x = np.divide(x, std)

        # 对长度超过输入长度的部分进行填充
        if input_length < x.shape[0]:
            x[input_length:] = padding_value

        # 确保数组为 float32 类型
        x = x.astype(np.float32)

        return x

    def normalize(
        self, input_features: List[np.ndarray], attention_mask: Optional[np.ndarray] = None
    ) -> List[np.ndarray]:
        # 计算输入特征的长度
        lengths = attention_mask.sum(-1) if attention_mask is not None else [x.shape[0] for x in input_features]
        # 对每个输入特征进行归一化处理
        return [self._normalize_one(x, n, self.padding_value) for x, n in zip(input_features, lengths)]

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
```