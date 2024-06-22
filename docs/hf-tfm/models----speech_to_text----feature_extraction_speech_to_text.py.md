# `.\transformers\models\speech_to_text\feature_extraction_speech_to_text.py`

```py
# 设置脚本编码为 UTF-8
# 版权声明：2021 年由 HuggingFace Inc. 团队保留所有权利
#
# 根据 Apache 许可证 2.0 版本（“许可证”）进行许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件是基于“按原样”提供的，
# 不附带任何明示或暗示的保证或条件。
# 有关许可证的详细信息，请参阅许可证。
"""
用于 Speech2Text 的特征提取器类
"""

from typing import List, Optional, Union  # 导入类型提示相关模块

import numpy as np  # 导入 NumPy 库

from ...audio_utils import mel_filter_bank, spectrogram, window_function  # 导入音频处理相关函数
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor  # 导入特征提取相关函数
from ...feature_extraction_utils import BatchFeature  # 导入特征处理相关函数
from ...utils import PaddingStrategy, TensorType, is_speech_available, logging  # 导入工具函数

# 如果语音处理可用，则导入相关库
if is_speech_available():
    import torch  # 导入 PyTorch 库
    import torchaudio.compliance.kaldi as ta_kaldi  # 导入 TorchAudio 库中的 Kaldi 模块

logger = logging.get_logger(__name__)  # 获取日志记录器


class Speech2TextFeatureExtractor(SequenceFeatureExtractor):
    r"""
    构建一个 Speech2Text 特征提取器。

    这个特征提取器继承自 [`Speech2TextFeatureExtractor`]，其中包含大多数主要方法。用户应参考此超类以获取有关这些方法的更多信息。

    这个类从原始语音中提取 mel-filter bank 特征，如果安装了 TorchAudio 则使用 TorchAudio，否则使用 numpy，并对提取的特征应用语句级别的倒谱均值和方差归一化。

    Args:
        feature_size (`int`, *optional*, 默认为 80):
            提取特征的特征维度。
        sampling_rate (`int`, *optional*, 默认为 16000):
            音频文件数字化的采样率，以赫兹（Hz）表示。
        num_mel_bins (`int`, *optional*, 默认为 80):
            Mel 频率箱的数量。
        padding_value (`float`, *optional*, 默认为 0.0):
            用于填充向量的值。
        do_ceptral_normalize (`bool`, *optional*, 默认为 `True`):
            是否对提取的特征应用语句级别的倒谱均值和方差归一化。
        normalize_means (`bool`, *optional*, 默认为 `True`):
            是否对提取的特征进行零均值归一化。
        normalize_vars (`bool`, *optional*, 默认为 `True`):
            是否对提取的特征进行单位方差归一化。
    """

    model_input_names = ["input_features", "attention_mask"]  # 定义模型输入的名称列表

    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        num_mel_bins=80,
        padding_value=0.0,
        do_ceptral_normalize=True,
        normalize_means=True,
        normalize_vars=True,
        **kwargs,
    ):
        # 调用父类初始化方法，设置特征大小、采样率、填充值等参数
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        # 设置梅尔滤波器数量
        self.num_mel_bins = num_mel_bins
        # 是否执行梅尔频谱归一化
        self.do_ceptral_normalize = do_ceptral_normalize
        # 是否归一化均值
        self.normalize_means = normalize_means
        # 是否归一化方差
        self.normalize_vars = normalize_vars
        # 返回注意力掩码
        self.return_attention_mask = True

        # 如果语音识别不可用
        if not is_speech_available():
            # 生成梅尔滤波器
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
            # 填充梅尔滤波器
            self.mel_filters = np.pad(mel_filters, ((0, 1), (0, 0)))
            # 设置窗口函数
            self.window = window_function(400, "povey", periodic=False)

    def _extract_fbank_features(
        self,
        waveform: np.ndarray,
    ) -> np.ndarray:
        """
        使用 TorchAudio 获取梅尔滤波器组特征。注意 TorchAudio 要求以 16 位有符号整数作为输入，
        因此在提取特征之前波形不应进行归一化。
        """
        # Kaldi 兼容：16 位有符号整数
        waveform = waveform * (2**15)
        # 如果语音识别可用
        if is_speech_available():
            # 转换为 Torch 张量并增加维度
            waveform = torch.from_numpy(waveform).unsqueeze(0)
            # 使用 TorchAudio 提取梅尔频率倒谱系数
            features = ta_kaldi.fbank(waveform, num_mel_bins=self.num_mel_bins, sample_frequency=self.sampling_rate)
            # 转换为 NumPy 数组
            features = features.numpy()
        else:
            waveform = np.squeeze(waveform)
            # 使用自定义函数提取梅尔频谱
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

    @staticmethod
    def utterance_cmvn(
        x: np.ndarray,
        input_length: int,
        normalize_means: Optional[bool] = True,
        normalize_vars: Optional[bool] = True,
        padding_value: float = 0.0,
    ) -> np.ndarray:
        # 确保我们归一化 float32 数组
        if normalize_means:
            mean = x[:input_length].mean(axis=0)
            x = np.subtract(x, mean)
        if normalize_vars:
            std = x[:input_length].std(axis=0)
            x = np.divide(x, std)

        # 如果输入长度小于数组长度，填充数组
        if input_length < x.shape[0]:
            x[input_length:] = padding_value

        # 确保数组为 float32 类型
        x = x.astype(np.float32)

        return x

    def normalize(
        self, input_features: List[np.ndarray], attention_mask: Optional[np.ndarray] = None
    # 定义一个方法，接受输入特征列表和注意力掩码，返回一个由归一化后的特征组成的列表
    ) -> List[np.ndarray]:
        # 如果有注意力掩码，则计算每个特征的长度，否则默认使用特征的第一个维度作为长度
        lengths = attention_mask.sum(-1) if attention_mask is not None else [x.shape[0] for x in input_features]
        # 返回归一化后的特征列表，对每个特征进行utterance_cmvn方法的处理，传入长度信息和其他参数
        return [
            self.utterance_cmvn(x, n, self.normalize_means, self.normalize_vars, self.padding_value)
            for x, n in zip(input_features, lengths)
        ]

    # 定义__call__方法，用于调用实例
    def __call__(
        # 接受原始语音数据，可以是numpy数组、列表、嵌套列表
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        # 表示是否进行填充，默认为False；可以是填充策略的字符串表示或PaddingStrategy实例
        padding: Union[bool, str, PaddingStrategy] = False,
        # 最大长度，默认为None
        max_length: Optional[int] = None,
        # 是否截断，默认为False
        truncation: bool = False,
        # 填充到的长度的倍数，默认为None
        pad_to_multiple_of: Optional[int] = None,
        # 返回的张量类型，默认为None
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 采样率，默认为None
        sampling_rate: Optional[int] = None,
        # 是否返回注意力掩码，默认为None
        return_attention_mask: Optional[bool] = None,
        # 其他关键字参数
        **kwargs,
```