# `.\transformers\models\audio_spectrogram_transformer\feature_extraction_audio_spectrogram_transformer.py`

```
# 设置文件编码为 UTF-8
# 版权声明：版权归 2022 年 HuggingFace Inc. 团队所有。
#
# 根据 Apache 许可证 2.0 版（"许可证"）的规定，除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按"原样"分发软件
# 在没有任何明示或暗示的情况下分发，包括但不限于适销性、特定用途适用性和非侵权性的保证。
# 有关许可证的详细信息，请参阅许可证。
"""
用于音频频谱变换器的特征提取器类。
"""

from typing import List, Optional, Union  # 引入类型提示

import numpy as np  # 引入 NumPy 库

from ...audio_utils import mel_filter_bank, spectrogram, window_function  # 导入音频工具函数
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor  # 导入序列特征提取器
from ...feature_extraction_utils import BatchFeature  # 导入批量特征
from ...utils import TensorType, is_speech_available, is_torch_available, logging  # 导入其他实用函数

# 如果语音可用，引入 TorchAudio 的 Kaldi 兼容模块
if is_speech_available():
    import torchaudio.compliance.kaldi as ta_kaldi

# 如果 Torch 可用，引入 Torch 模块
if is_torch_available():
    import torch

logger = logging.get_logger(__name__)  # 获取日志记录器

# 定义 ASTFeatureExtractor 类，继承自 SequenceFeatureExtractor 类
class ASTFeatureExtractor(SequenceFeatureExtractor):
    r"""
    构建音频频谱变换器（AST）特征提取器。

    此特征提取器继承自 [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`]，其中包含大部分主要方法。
    用户应该参考此超类以获取有关这些方法的更多信息。

    此类使用 TorchAudio（如果已安装）或否则使用 NumPy 从原始语音中提取梅尔滤波器组特征，将其填充/截断到固定长度，并使用均值和标准差对其进行归一化。

    参数:
        feature_size (`int`, *optional*, defaults to 1):
            提取特征的特征维度。
        sampling_rate (`int`, *optional*, defaults to 16000):
            表示音频文件应数字化的采样率，以赫兹（Hz）为单位。
        num_mel_bins (`int`, *optional*, defaults to 128):
            梅尔频率箱的数量。
        max_length (`int`, *optional*, defaults to 1024):
            要填充/截断提取特征的最大长度。
        do_normalize (`bool`, *optional*, defaults to `True`):
            是否对对数梅尔特征进行归一化，使用 `mean` 和 `std`。
        mean (`float`, *optional*, defaults to -4.2677393):
            用于归一化对数梅尔特征的均值。默认使用 AudioSet 的均值。
        std (`float`, *optional*, defaults to 4.5689974):
            用于归一化对数梅尔特征的标准差。默认使用 AudioSet 的标准差。
        return_attention_mask (`bool`, *optional*, defaults to `False`):
            是否 [`~ASTFeatureExtractor.__call__`] 应该返回 `attention_mask`。
    """
    # 定义模型输入的名称列表
    model_input_names = ["input_values", "attention_mask"]

    # 初始化函数，设置特征大小、采样率、梅尔频率数量、最大长度、填充值、是否归一化、均值、标准差、是否返回注意力掩码等参数
    def __init__(
        self,
        feature_size=1,
        sampling_rate=16000,
        num_mel_bins=128,
        max_length=1024,
        padding_value=0.0,
        do_normalize=True,
        mean=-4.2677393,
        std=4.5689974,
        return_attention_mask=False,
        **kwargs,
    ):
        # 调用父类的初始化函数，设置特征大小、采样率、填充值等参数
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.num_mel_bins = num_mel_bins
        self.max_length = max_length
        self.do_normalize = do_normalize
        self.mean = mean
        self.std = std
        self.return_attention_mask = return_attention_mask

        # 如果语音可用，则生成梅尔滤波器
        if not is_speech_available():
            mel_filters = mel_filter_bank(
                num_frequency_bins=256,
                num_mel_filters=self.num_mel_bins,
                min_frequency=20,
                max_frequency=sampling_rate // 2,
                sampling_rate=sampling_rate,
                norm=None,
                mel_scale="kaldi",
                triangularize_in_mel_space=True,
            )

            self.mel_filters = np.pad(mel_filters, ((0, 1), (0, 0)))
            self.window = window_function(400, "hann", periodic=False)

    # 提取梅尔滤波器特征
    def _extract_fbank_features(
        self,
        waveform: np.ndarray,
        max_length: int,
    ) -> np.ndarray:
        """
        Get mel-filter bank features using TorchAudio. Note that TorchAudio requires 16-bit signed integers as inputs
        and hence the waveform should not be normalized before feature extraction.
        """
        # waveform = waveform * (2**15)  # Kaldi compliance: 16-bit signed integers
        # 如果语音可用，则使用 TorchAudio 提取梅尔滤波器特征
        if is_speech_available():
            waveform = torch.from_numpy(waveform).unsqueeze(0)
            fbank = ta_kaldi.fbank(
                waveform,
                sample_frequency=self.sampling_rate,
                window_type="hanning",
                num_mel_bins=self.num_mel_bins,
            )
        else:
            waveform = np.squeeze(waveform)
            # 使用自定义的频谱图函数提取梅尔滤波器特征
            fbank = spectrogram(
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

            fbank = torch.from_numpy(fbank)

        n_frames = fbank.shape[0]
        difference = max_length - n_frames

        # 根据差值进行填充或截断
        if difference > 0:
            pad_module = torch.nn.ZeroPad2d((0, 0, 0, difference))
            fbank = pad_module(fbank)
        elif difference < 0:
            fbank = fbank[0:max_length, :]

        fbank = fbank.numpy()

        return fbank
    # 对输入数据进行标准化处理，减去均值后除以两倍的标准差
    def normalize(self, input_values: np.ndarray) -> np.ndarray:
        return (input_values - (self.mean)) / (self.std * 2)

    # 调用函数，接受原始语音数据和其他参数
    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        sampling_rate: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
```