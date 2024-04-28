# `.\transformers\models\speecht5\feature_extraction_speecht5.py`

```
# 设置代码文件的编码格式为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证，只有在遵守许可证的情况下才能使用此文件
# 您可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非法律要求或书面同意，在遵守许可证的情况下分发的软件将基于“AS IS”方式分发，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限申明和限制的详细信息
"""Feature extractor class for SpeechT5."""
# 导入所需的模块和库
import warnings
from typing import Any, Dict, List, Optional, Union
import numpy as np

# 导入自定义的模块和函数
from ...audio_utils import mel_filter_bank, optimal_fft_length, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 SpeechT5FeatureExtractor 类，继承自 SequenceFeatureExtractor 类
class SpeechT5FeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a SpeechT5 feature extractor.

    This class can pre-process a raw speech signal by (optionally) normalizing to zero-mean unit-variance, for use by
    the SpeechT5 speech encoder prenet.

    This class can also extract log-mel filter bank features from raw speech, for use by the SpeechT5 speech decoder
    prenet.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.
    Args:
        feature_size (`int`, *optional*, defaults to 1):
            提取特征的特征维度。
        sampling_rate (`int`, *optional*, defaults to 16000):
            表示音频文件数字化时的采样率，以赫兹（Hz）表示。
        padding_value (`float`, *optional*, defaults to 0.0):
            用于填充填充值的值。
        do_normalize (`bool`, *optional*, defaults to `False`):
            是否对输入进行零均值单位方差标准化。标准化可以显著提高某些模型的性能。
        num_mel_bins (`int`, *optional*, defaults to 80):
            提取的梅尔频谱特征中的梅尔频率 bin 的数量。
        hop_length (`int`, *optional*, defaults to 16):
            窗口之间的毫秒数。在许多论文中也称为“shift”。
        win_length (`int`, *optional*, defaults to 64):
            每个窗口的毫秒数。
        win_function (`str`, *optional*, defaults to `"hann_window"`):
            用于窗口化的窗口函数的名称，必须通过 `torch.{win_function}` 访问。
        frame_signal_scale (`float`, *optional*, defaults to 1.0):
            在应用 DFT 前创建帧时乘以的常数。此参数已弃用。
        fmin (`float`, *optional*, defaults to 80):
            Hz 中的最小梅尔频率。
        fmax (`float`, *optional*, defaults to 7600):
            Hz 中的最大梅尔频率。
        mel_floor (`float`, *optional*, defaults to 1e-10):
            梅尔频率 bank 的最小值。
        reduction_factor (`int`, *optional*, defaults to 2):
            频谱长度缩减因子。此参数已弃用。
        return_attention_mask (`bool`, *optional*, defaults to `True`):
            [`~SpeechT5FeatureExtractor.__call__`] 是否应返回 `attention_mask`。
    """

    model_input_names = ["input_values", "attention_mask"]

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
    # 初始化函数，设置特征提取器的参数
    def __init__(
        self,
        feature_size: int,
        sampling_rate: int,
        padding_value: float = 0.0,
        do_normalize: bool = True,
        return_attention_mask: bool = False,
        num_mel_bins: int = 80,
        hop_length: int = 10,
        win_length: int = 25,
        win_function: str = "hann",
        frame_signal_scale: float = 1.0,
        fmin: float = 0.0,
        fmax: float = 4000.0,
        mel_floor: float = 1e-6,
        reduction_factor: float = 2.0,
        **kwargs
    ):
        # 调用父类的初始化函数，设置特征大小、采样率、填充值等参数
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        # 设置是否进行标准化的标志
        self.do_normalize = do_normalize
        # 设置是否返回注意力掩码的标志
        self.return_attention_mask = return_attention_mask

        # 设置梅尔滤波器的参数
        self.num_mel_bins = num_mel_bins
        self.hop_length = hop_length
        self.win_length = win_length
        self.win_function = win_function
        self.frame_signal_scale = frame_signal_scale
        self.fmin = fmin
        self.fmax = fmax
        self.mel_floor = mel_floor
        self.reduction_factor = reduction_factor

        # 计算样本大小和样本步长
        self.sample_size = win_length * sampling_rate // 1000
        self.sample_stride = hop_length * sampling_rate // 1000
        # 计算 FFT 长度
        self.n_fft = optimal_fft_length(self.sample_size)
        # 计算频率数量
        self.n_freqs = (self.n_fft // 2) + 1

        # 初始化窗口函数
        self.window = window_function(window_length=self.sample_size, name=self.win_function, periodic=True)

        # 初始化梅尔滤波器组
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=self.n_freqs,
            num_mel_filters=self.num_mel_bins,
            min_frequency=self.fmin,
            max_frequency=self.fmax,
            sampling_rate=self.sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )

        # 发出警告，提醒用户已弃用的参数将在未来版本中移除
        if frame_signal_scale != 1.0:
            warnings.warn(
                "The argument `frame_signal_scale` is deprecated and will be removed in version 4.30.0 of Transformers",
                FutureWarning,
            )
        if reduction_factor != 2.0:
            warnings.warn(
                "The argument `reduction_factor` is deprecated and will be removed in version 4.30.0 of Transformers",
                FutureWarning,
            )

    @staticmethod
    # 从 transformers.models.wav2vec2.feature_extraction_wav2vec2.Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm 复制而来
    # 函数用于将输入的列表中的每个数组进行零均值单位方差归一化处理
    def zero_mean_unit_var_norm(
        input_values: List[np.ndarray], attention_mask: List[np.ndarray], padding_value: float = 0.0
    ) -> List[np.ndarray]:
        """
        Every array in the list is normalized to have zero mean and unit variance
        """
        # 如果存在注意力掩码，则进行处理
        if attention_mask is not None:
            # 将注意力掩码转换为 NumPy 数组
            attention_mask = np.array(attention_mask, np.int32)
            normed_input_values = []

            # 遍历输入值和对应的注意力掩码
            for vector, length in zip(input_values, attention_mask.sum(-1)):
                # 计算零均值单位方差归一化处理后的切片
                normed_slice = (vector - vector[:length].mean()) / np.sqrt(vector[:length].var() + 1e-7)
                # 如果长度小于切片的长度，则将剩余部分填充为指定填充值
                if length < normed_slice.shape[0]:
                    normed_slice[length:] = padding_value

                # 将归一化处理后的切片添加到列表中
                normed_input_values.append(normed_slice)
        else:
            # 如果不存在注意力掩码，则对每个输入值进行零均值单位方差归一化处理
            normed_input_values = [(x - x.mean()) / np.sqrt(x.var() + 1e-7) for x in input_values]

        # 返回归一化处理后的输入值列表
        return normed_input_values

    # 提取梅尔特征的私有方法
    def _extract_mel_features(
        self,
        one_waveform: np.ndarray,
    # 定义函数，接收一个波形数组，返回对应的log-mel滤波器特征
    ) -> np.ndarray:
        """
        Extracts log-mel filterbank features for one waveform array (unbatched).
        """
        # 使用给定参数计算波形数组的log-mel谱图
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
        # 返回log-mel谱图的转置
        return log_mel_spec.T

    # 定义函数，接收音频数组并进行预处理
    def __call__(
        self,
        audio: Optional[Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]] = None,
        audio_target: Optional[Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        sampling_rate: Optional[int] = None,
        **kwargs,
    # 定义函数，处理音频数组并返回处理后的结果
    def _process_audio(
        self,
        speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]],
        is_target: bool = False,
        padding: Union[bool, str, PaddingStrategy] = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    # 将当前对象转换为字典形式，并返回该字典
    def to_dict(self) -> Dict[str, Any]:
        # 调用父类的to_dict方法，获取输出字典
        output = super().to_dict()

        # 从输出字典中删除特定属性，因为它们是由其他属性派生而来
        names = ["window", "mel_filters", "sample_size", "sample_stride", "n_fft", "n_freqs"]
        for name in names:
            # 如果属性存在于输出字典中，则删除它
            if name in output:
                del output[name]

        # 返回处理后的输出字典
        return output
```