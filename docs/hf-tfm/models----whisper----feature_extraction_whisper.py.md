# `.\transformers\models\whisper\feature_extraction_whisper.py`

```
# 设置代码文件的编码格式为 utf-8
# 版权声明，许可证信息等
"""
Feature extractor class for Whisper
"""

# 导入需要的模块和库
from typing import List, Optional, Union
import numpy as np
from ... import is_torch_available  # 导入 is_torch_available 函数
from ...audio_utils import mel_filter_bank, spectrogram, window_function  # 导入音频处理相关的函数
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor  # 导入序列特征提取器
from ...feature_extraction_utils import BatchFeature  # 导入 BatchFeature 类
from ...utils import TensorType, logging  # 导入 TensorType、logging 类/函数

if is_torch_available():  # 如果 torch 可用
    import torch  # 导入 torch 模块

logger = logging.get_logger(__name__)  # 获取 logger 实例

# 定义 WhisperFeatureExtractor 类，继承自 SequenceFeatureExtractor 类
class WhisperFeatureExtractor(SequenceFeatureExtractor):
    r"""
    构建 Whisper 特征提取器。

    该特征提取器继承自 [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`]，其中包含大部分主要方法。
    用户应该参考这个超类以获得关于这些方法的更多信息。

    该类使用自定义的 numpy 实现的 `短时傅里叶变换` 从原始语音中提取 mel 滤波器组特征，应该与 pytorch 的 `torch.stft` 等效。

    参数:
        feature_size (`int`, 默认为 80):
            提取到的特征的维度。
        sampling_rate (`int`, 默认为 16000):
            音频文件应该以赫兹为单位数字化的采样率。
        hop_length (`int`, 默认为 160):
            用于获取 Mel 频率系数的重叠窗口的长度。
        chunk_length (`int`, 默认为 30):
            用于修剪和填充更长或更短的音频序列的最大数量的 `sampling_rate` 样本的块数。
        n_fft (`int`, 默认为 400):
            傅里叶变换的大小。
        padding_value (`float`, *可选*, 默认为 0.0):
            用于填充音频的填充值。应该对应于静音部分。
    """

    model_input_names = ["input_features"]  # 模型输入的名称为 "input_features"

    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
        padding_value=0.0,
        return_attention_mask=False,  # 使用静默令牌（零）填充输入并没有注意力遮罩
        **kwargs,
    # 此类是 Wav2Vec2FeatureExtractor 的子类，继承其功能并扩展了一些新的功能
    class CustomFeatureExtractor(Wav2Vec2FeatureExtractor):
        # 初始化方法，接受一些参数并赋值给类属性
        def __init__(
            self,
            feature_size=64,
            sampling_rate=16000,
            padding_value=0.0,
            return_attention_mask=True,
            n_fft=400,
            hop_length=160,
            chunk_length=1600,
            **kwargs
        ):
            # 调用父类的初始化方法，设置一些基本属性
            super().__init__(
                feature_size=feature_size,
                sampling_rate=sampling_rate,
                padding_value=padding_value,
                return_attention_mask=return_attention_mask,
                **kwargs,
            )
            # 设置一些新的属性，如 n_fft、hop_length、chunk_length 等
            self.n_fft = n_fft
            self.hop_length = hop_length
            self.chunk_length = chunk_length
            self.n_samples = chunk_length * sampling_rate
            self.nb_max_frames = self.n_samples // hop_length
            self.sampling_rate = sampling_rate
            # 计算 mel 滤波器组
            self.mel_filters = mel_filter_bank(
                num_frequency_bins=1 + n_fft // 2,
                num_mel_filters=feature_size,
                min_frequency=0.0,
                max_frequency=8000.0,
                sampling_rate=sampling_rate,
                norm="slaney",
                mel_scale="slaney",
            )
    
        # 使用 NumPy 实现的 log-mel 谱特征提取方法
        def _np_extract_fbank_features(self, waveform: np.array) -> np.ndarray:
            """
            计算提供音频的对数 mel 频谱，与 Whisper 的原始 PyTorch 实现给出的结果非常接近（容差为 1e-5）。
            """
            # 计算声谱图
            log_spec = spectrogram(
                waveform,
                window_function(self.n_fft, "hann"),
                frame_length=self.n_fft,
                hop_length=self.hop_length,
                power=2.0,
                mel_filters=self.mel_filters,
                log_mel="log10",
            )
            # 对结果进行一些后处理
            log_spec = log_spec[:, :-1]
            log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
            log_spec = (log_spec + 4.0) / 4.0
            return log_spec
    
        # 使用 PyTorch 实现的 log-mel 谱特征提取方法
        def _torch_extract_fbank_features(self, waveform: np.array) -> np.ndarray:
            """
            使用 PyTorch 的 STFT 实现计算提供音频的对数 mel 频谱。
            """
            # 将输入转换为 PyTorch 张量
            waveform = torch.from_numpy(waveform).type(torch.float32)
            # 计算短时傅里叶变换
            window = torch.hann_window(self.n_fft)
            stft = torch.stft(waveform, self.n_fft, self.hop_length, window=window, return_complex=True)
            magnitudes = stft[..., :-1].abs() ** 2
            # 计算 mel 频谱
            mel_filters = torch.from_numpy(self.mel_filters).type(torch.float32)
            mel_spec = mel_filters.T @ magnitudes
            # 对结果进行一些后处理
            log_spec = torch.clamp(mel_spec, min=1e-10).log10()
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
            log_spec = (log_spec + 4.0) / 4.0
            return log_spec.numpy()
    
        # 静态方法，用于对输入值进行零均值单位方差归一化
        @staticmethod
        def zero_mean_unit_var_norm(
            input_values: List[np.ndarray], attention_mask: List[np.ndarray], padding_value: float = 0.0
    ) -> List[np.ndarray]:
        """
        每个列表中的数组都被标准化为零均值和单位方差
        """
        # 如果存在attention_mask，则将其转换为np.int32类型的数组
        if attention_mask is not None:
            attention_mask = np.array(attention_mask, np.int32)
            normed_input_values = []

            # 遍历input_values和attention_mask的长度
            for vector, length in zip(input_values, attention_mask.sum(-1)):
                # 对每个向量进行标准化处理并添加到normed_input_values列表中
                normed_slice = (vector - vector[:length].mean()) / np.sqrt(vector[:length].var() + 1e-7)
                # 如果长度小于标准化后的长度，则用padding_value进行填充
                if length < normed_slice.shape[0]:
                    normed_slice[length:] = padding_value

                normed_input_values.append(normed_slice)
        else:
            # 如果不存在attention_mask，则对input_values中的每个数组进行标准化处理并存储在normed_input_values中
            normed_input_values = [(x - x.mean()) / np.sqrt(x.var() + 1e-7) for x in input_values]

        return normed_input_values

    # 函数调用
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