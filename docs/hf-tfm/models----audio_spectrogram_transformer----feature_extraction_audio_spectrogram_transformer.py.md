# `.\models\audio_spectrogram_transformer\feature_extraction_audio_spectrogram_transformer.py`

```py
"""
Feature extractor class for Audio Spectrogram Transformer.
"""

# 导入必要的库
from typing import List, Optional, Union

import numpy as np  # 导入 NumPy 库

# 导入音频处理相关的函数和类
from ...audio_utils import mel_filter_bank, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, is_speech_available, is_torch_available, logging

# 如果 TorchAudio 可用，则导入相应的模块
if is_speech_available():
    import torchaudio.compliance.kaldi as ta_kaldi

# 如果 Torch 可用，则导入 Torch 库
if is_torch_available():
    import torch

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 Audio Spectrogram Transformer (AST) 特征提取器类
class ASTFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Audio Spectrogram Transformer (AST) feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts mel-filter bank features from raw speech using TorchAudio if installed or using numpy
    otherwise, pads/truncates them to a fixed length and normalizes them using a mean and standard deviation.

    Args:
        feature_size (`int`, *optional*, defaults to 1):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        num_mel_bins (`int`, *optional*, defaults to 128):
            Number of Mel-frequency bins.
        max_length (`int`, *optional*, defaults to 1024):
            Maximum length to which to pad/truncate the extracted features.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the log-Mel features using `mean` and `std`.
        mean (`float`, *optional*, defaults to -4.2677393):
            The mean value used to normalize the log-Mel features. Uses the AudioSet mean by default.
        std (`float`, *optional*, defaults to 4.5689974):
            The standard deviation value used to normalize the log-Mel features. Uses the AudioSet standard deviation
            by default.
        return_attention_mask (`bool`, *optional*, defaults to `False`):
            Whether or not [`~ASTFeatureExtractor.__call__`] should return `attention_mask`.
    """
    model_input_names = ["input_values", "attention_mask"]  # 定义模型输入的名称列表，包括输入值和注意力掩码

    def __init__(  # 初始化方法，用于设置模型参数和属性
        self,
        feature_size=1,  # 特征大小，默认为1
        sampling_rate=16000,  # 采样率，默认为16000
        num_mel_bins=128,  # 梅尔频谱的梅尔频道数，默认为128
        max_length=1024,  # 最大长度，默认为1024
        padding_value=0.0,  # 填充值，默认为0.0
        do_normalize=True,  # 是否进行归一化，默认为True
        mean=-4.2677393,  # 均值，默认为-4.2677393
        std=4.5689974,  # 标准差，默认为4.5689974
        return_attention_mask=False,  # 是否返回注意力掩码，默认为False
        **kwargs,  # 其他关键字参数
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.num_mel_bins = num_mel_bins  # 设置梅尔频道数
        self.max_length = max_length  # 设置最大长度
        self.do_normalize = do_normalize  # 设置是否归一化
        self.mean = mean  # 设置均值
        self.std = std  # 设置标准差
        self.return_attention_mask = return_attention_mask  # 设置是否返回注意力掩码

        if not is_speech_available():  # 如果语音处理不可用
            mel_filters = mel_filter_bank(  # 生成梅尔滤波器组
                num_frequency_bins=256,
                num_mel_filters=self.num_mel_bins,
                min_frequency=20,
                max_frequency=sampling_rate // 2,
                sampling_rate=sampling_rate,
                norm=None,
                mel_scale="kaldi",
                triangularize_in_mel_space=True,
            )

            self.mel_filters = np.pad(mel_filters, ((0, 1), (0, 0)))  # 对梅尔滤波器进行填充以适应需求
            self.window = window_function(400, "hann", periodic=False)  # 创建窗函数对象

    def _extract_fbank_features(  # 提取梅尔滤波器组特征的方法
        self,
        waveform: np.ndarray,  # 输入波形数据，numpy数组类型
        max_length: int,  # 最大长度，整数类型
    ) -> np.ndarray:
        """
        Get mel-filter bank features using TorchAudio. Note that TorchAudio requires 16-bit signed integers as inputs
        and hence the waveform should not be normalized before feature extraction.
        """
        # waveform = waveform * (2**15)  # Kaldi compliance: 16-bit signed integers
        if is_speech_available():  # 如果语音处理可用
            waveform = torch.from_numpy(waveform).unsqueeze(0)  # 将波形数据转换为PyTorch张量
            fbank = ta_kaldi.fbank(  # 使用TorchAudio的Kaldi库提取梅尔滤波器组特征
                waveform,
                sample_frequency=self.sampling_rate,
                window_type="hanning",
                num_mel_bins=self.num_mel_bins,
            )
        else:
            waveform = np.squeeze(waveform)  # 去除波形数据中的单维度
            fbank = spectrogram(  # 使用自定义的频谱图方法提取梅尔滤波器组特征
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

            fbank = torch.from_numpy(fbank)  # 将特征数据转换为PyTorch张量

        n_frames = fbank.shape[0]  # 获取特征张量的帧数
        difference = max_length - n_frames  # 计算需要填充或截断的帧数差异

        # pad or truncate, depending on difference
        if difference > 0:  # 如果差异大于0，进行填充操作
            pad_module = torch.nn.ZeroPad2d((0, 0, 0, difference))  # 创建填充模块对象
            fbank = pad_module(fbank)  # 对特征张量进行填充
        elif difference < 0:  # 如果差异小于0，进行截断操作
            fbank = fbank[0:max_length, :]  # 截取指定长度的特征数据

        fbank = fbank.numpy()  # 将PyTorch张量转换为numpy数组

        return fbank  # 返回梅尔滤波器组特征数组
    # 根据给定的均值和标准差对输入值进行标准化处理
    def normalize(self, input_values: np.ndarray) -> np.ndarray:
        return (input_values - (self.mean)) / (self.std * 2)

    # 实现对象的可调用接口，用于处理原始语音数据
    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        sampling_rate: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
```