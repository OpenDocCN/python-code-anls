# `.\transformers\models\clap\feature_extraction_clap.py`

```py
# 设定文件编码为UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，您可以使用此文件，但需遵守许可证中的规定
# 可在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证副本
# 除非适用法律要求或书面同意，否则本软件是基于“按原样”的基础分发，不附带任何形式的担保或条件
# 请查看许可证以了解许可证下的特定语言规定和限制

"""CLAP 的特征提取器类"""

# 导入所需库
import copy
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

# 导入自定义模块
from ...audio_utils import mel_filter_bank, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, logging

# 获取日志记录器
logger = logging.get_logger(__name__)


class ClapFeatureExtractor(SequenceFeatureExtractor):
    r"""
    构建 CLAP 特征提取器。

    此特征提取器继承自 [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`]，其中包含大部分主要方法。
    用户应参考此超类以获取有关这些方法的更多信息。

    此类从原始语音中提取 mel 滤波器组特征，使用自定义的 numpy 实现的 *短时傅里叶变换*（STFT），
    该实现应与 pytorch 的 `torch.stft` 等效。

    """

    # 模型输入的名称列表
    model_input_names = ["input_features", "is_longer"]

    def __init__(
        self,
        feature_size=64,
        sampling_rate=48_000,
        hop_length=480,
        max_length_s=10,
        fft_window_size=1024,
        padding_value=0.0,
        return_attention_mask=False,  # pad inputs to max length with silence token (zero) and no attention mask
        frequency_min: float = 0,
        frequency_max: float = 14_000,
        top_db: int = None,
        truncation: str = "fusion",
        padding: str = "repeatpad",
        **kwargs,
    ):
        # 调用父类的初始化方法，并传入参数
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        # 设置顶部的分贝数
        self.top_db = top_db
        # 设置截断值
        self.truncation = truncation
        # 设置填充值
        self.padding = padding
        # 设置 FFT 窗口大小
        self.fft_window_size = fft_window_size
        # 计算频率 bin 数量
        self.nb_frequency_bins = (fft_window_size >> 1) + 1
        # 设置跳跃长度
        self.hop_length = hop_length
        # 设置最大长度（秒）
        self.max_length_s = max_length_s
        # 计算最大采样数
        self.nb_max_samples = max_length_s * sampling_rate
        # 设置采样率
        self.sampling_rate = sampling_rate
        # 设置最小频率
        self.frequency_min = frequency_min
        # 设置最大频率
        self.frequency_max = frequency_max
        # 计算 Mel 滤波器组
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=self.nb_frequency_bins,
            num_mel_filters=feature_size,
            min_frequency=frequency_min,
            max_frequency=frequency_max,
            sampling_rate=sampling_rate,
            norm=None,
            mel_scale="htk",
        )
        # 计算 Slaney Mel 滤波器组
        self.mel_filters_slaney = mel_filter_bank(
            num_frequency_bins=self.nb_frequency_bins,
            num_mel_filters=feature_size,
            min_frequency=frequency_min,
            max_frequency=frequency_max,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )

    # 将当前实例序列化为 Python 字典
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance, excpet for the
            mel filter banks, which do not need to be saved or printed as they are too long.
        """
        # 复制实例属性的深层副本
        output = copy.deepcopy(self.__dict__)
        # 将实例的类名添加到字典中
        output["feature_extractor_type"] = self.__class__.__name__
        # 如果字典中有 "mel_filters" 键，删除它
        if "mel_filters" in output:
            del output["mel_filters"]
        # 如果字典中有 "mel_filters_slaney" 键，删除它
        if "mel_filters_slaney" in output:
            del output["mel_filters_slaney"]
        # 返回字典
        return output
    def _np_extract_fbank_features(self, waveform: np.array, mel_filters: Optional[np.array] = None) -> np.ndarray:
        """
        Compute the log-mel spectrogram of the provided `waveform` using the Hann window. In CLAP, two different filter
        banks are used depending on the truncation pattern:
            - `self.mel_filters`: they correspond to the default parameters of `torchaudio` which can be obtained from
              calling `torchaudio.transforms.MelSpectrogram().mel_scale.fb`. These filters are used when `truncation`
              is set to `"fusion"`.
            - `self.mel_filteres_slaney` : they correspond to the default parameters of `librosa` which used
              `librosa.filters.mel` when computing the mel spectrogram. These filters were only used in the original
              implementation when the truncation mode is not `"fusion"`.
        """
        # 计算提供的 `waveform` 的对数梅尔频谱图，使用汉宁窗口。在 CLAP 中，根据截断模式使用两种不同的滤波器组：
        # - `self.mel_filters`：它们对应于 `torchaudio` 的默认参数，可以通过调用 `torchaudio.transforms.MelSpectrogram().mel_scale.fb` 获得。
        #   当 `truncation` 设置为 `"fusion"` 时使用这些滤波器。
        # - `self.mel_filteres_slaney`：它们对应于 `librosa` 的默认参数，在计算梅尔频谱图时使用 `librosa.filters.mel`。
        #   这些滤波器仅在原始实现中当截断模式不是 `"fusion"` 时使用。
        log_mel_spectrogram = spectrogram(
            waveform,
            window_function(self.fft_window_size, "hann"),
            frame_length=self.fft_window_size,
            hop_length=self.hop_length,
            power=2.0,
            mel_filters=mel_filters,
            log_mel="dB",
        )
        return log_mel_spectrogram.T

    def _random_mel_fusion(self, mel, total_frames, chunk_frames):
        ranges = np.array_split(list(range(0, total_frames - chunk_frames + 1)), 3)
        if len(ranges[1]) == 0:
            # 如果音频太短，我们只使用第一个块
            ranges[1] = [0]
        if len(ranges[2]) == 0:
            # 如果音频太短，我们只使用第一个块
            ranges[2] = [0]
        # 随机选择每个部分的索引
        idx_front = np.random.choice(ranges[0])
        idx_middle = np.random.choice(ranges[1])
        idx_back = np.random.choice(ranges[2])

        mel_chunk_front = mel[idx_front : idx_front + chunk_frames, :]
        mel_chunk_middle = mel[idx_middle : idx_middle + chunk_frames, :]
        mel_chunk_back = mel[idx_back : idx_back + chunk_frames, :]

        mel = torch.tensor(mel[None, None, :])
        mel_shrink = torch.nn.functional.interpolate(
            mel, size=[chunk_frames, 64], mode="bilinear", align_corners=False
        )
        mel_shrink = mel_shrink[0][0].numpy()
        mel_fusion = np.stack([mel_shrink, mel_chunk_front, mel_chunk_middle, mel_chunk_back], axis=0)
        return mel_fusion

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]],
        truncation: str = None,
        padding: Optional[str] = None,
        max_length: Optional[int] = None,
        sampling_rate: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
```