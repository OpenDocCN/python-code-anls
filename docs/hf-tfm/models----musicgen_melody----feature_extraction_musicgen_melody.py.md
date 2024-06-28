# `.\models\musicgen_melody\feature_extraction_musicgen_melody.py`

```py
"""
Feature extractor class for Musicgen Melody
"""
# 导入必要的模块和类
import copy  # 导入深拷贝模块
from typing import Any, Dict, List, Optional, Union  # 导入类型提示模块

import numpy as np  # 导入 NumPy 库

# 导入音频处理相关的实用函数和类
from ...audio_utils import chroma_filter_bank  
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor  
from ...feature_extraction_utils import BatchFeature  
from ...utils import TensorType, is_torch_available, is_torchaudio_available, logging  

# 如果 Torch 可用，导入 Torch 库
if is_torch_available():
    import torch  

# 如果 Torchaudio 可用，导入 Torchaudio 库
if is_torchaudio_available():
    import torchaudio  

# 获取日志记录器对象
logger = logging.get_logger(__name__)  


class MusicgenMelodyFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a MusicgenMelody feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts chroma features from audio processed by [Demucs](https://github.com/adefossez/demucs/tree/main) or
    directly from raw audio waveform.
    """
    # 默认参数：特征向量的维度为12
    # The default feature size of the extracted features is 12
    feature_size=12,
    
    # 默认参数：音频文件的数字化采样率为32000赫兹（Hz）
    # The default sampling rate at which audio files are digitized is 32000 Hz
    sampling_rate=32000,
    
    # 默认参数：用于获取Mel频率系数的STFT中的重叠窗口长度为4096
    # Length of overlapping windows for STFT used to obtain Mel Frequency coefficients
    hop_length=4096,
    
    # 默认参数：用于裁剪和填充较长或较短音频序列的最大采样率样本数为30个
    # Maximum number of chunks of sampling_rate samples used to trim and pad longer or shorter audio sequences
    chunk_length=30,
    
    # 默认参数：傅里叶变换的大小为16384
    # Size of the Fourier transform
    n_fft=16384,
    
    # 默认参数：使用的色度频带数为12
    # Number of chroma bins to use
    num_chroma=12,
    
    # 默认参数：用于填充音频的填充值为0.0
    # Padding value used to pad the audio
    padding_value=0.0,
    
    # 默认参数：是否返回注意力掩码，默认为False
    # Whether to return the attention mask, default is False
    return_attention_mask=False,
    
    # 默认参数：如果Demucs输出作为输入，要提取的干索引为[3, 2]
    # Stem channels to extract if Demucs outputs are passed
    stem_indices=[3, 2],
    
    model_input_names = ["input_features"]

    def __init__(
        self,
        feature_size=12,
        sampling_rate=32000,
        hop_length=4096,
        chunk_length=30,
        n_fft=16384,
        num_chroma=12,
        padding_value=0.0,
        return_attention_mask=False,  # pad inputs to max length with silence token (zero) and no attention mask
        stem_indices=[3, 2],
        **kwargs,
    ):
        # 调用父类构造函数，初始化特征提取器的参数
        # Call the parent class constructor to initialize parameters of the feature extractor
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        
        # 设置本地变量n_fft、hop_length、chunk_length和n_samples的值
        # Set values for local variables n_fft, hop_length, chunk_length, and n_samples
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sampling_rate
        self.sampling_rate = sampling_rate
        
        # 使用给定的参数初始化色度滤波器
        # Initialize chroma filters using given parameters
        self.chroma_filters = torch.from_numpy(
            chroma_filter_bank(sampling_rate=sampling_rate, num_frequency_bins=n_fft, tuning=0, num_chroma=num_chroma)
        ).float()
        
        # 初始化频谱图变换器，用于生成音频的频谱图
        # Initialize spectrogram transformer for generating spectrograms of audio
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, win_length=n_fft, hop_length=hop_length, power=2, center=True, pad=0, normalized=True
        )
        
        # 设置提取的干索引列表
        # Set list of stem indices to extract
        self.stem_indices = stem_indices
    def _torch_extract_fbank_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute the chroma spectrogram of the provided audio using the torchaudio spectrogram implementation and the librosa chroma features.
        """

        # if wav length is not long enough, pad it
        wav_length = waveform.shape[-1]
        if wav_length < self.n_fft:
            # Calculate padding required to match `n_fft`
            pad = self.n_fft - wav_length
            rest = 0 if pad % 2 == 0 else 1
            # Pad the waveform symmetrically to ensure correct length
            waveform = torch.nn.functional.pad(waveform, (pad // 2, pad // 2 + rest), "constant", 0)

        # squeeze alongside channel dimension
        spec = self.spectrogram(waveform).squeeze(1)

        # sum along the frequency dimension
        raw_chroma = torch.einsum("cf, ...ft->...ct", self.chroma_filters, spec)

        # normalise with max value
        norm_chroma = torch.nn.functional.normalize(raw_chroma, p=float("inf"), dim=-2, eps=1e-6)

        # transpose time and chroma dimension -> (batch, time, chroma)
        norm_chroma = norm_chroma.transpose(1, 2)

        # replace max value alongside chroma dimension with 1 and replace the rest with 0
        idx = norm_chroma.argmax(-1, keepdim=True)
        norm_chroma[:] = 0
        norm_chroma.scatter_(dim=-1, index=idx, value=1)

        return norm_chroma




    def _extract_stem_indices(self, audio, sampling_rate=None):
        """
        Extracts stems from the output of the [Demucs](https://github.com/adefossez/demucs/tree/main) audio separation model,
        then converts to mono-channel and resample to the feature extractor sampling rate.

        Args:
            audio (`torch.Tensor` of shape `(batch_size, num_stems, channel_size, audio_length)`):
                The output of the Demucs model to be processed.
            sampling_rate (`int`, *optional*):
                Demucs sampling rate. If not specified, defaults to `44000`.
        """
        sampling_rate = 44000 if sampling_rate is None else sampling_rate

        # extract "vocals" and "others" sources from audio encoder (demucs) output
        # [batch_size, num_stems, channel_size, audio_length]
        wav = audio[:, torch.tensor(self.stem_indices)]

        # merge extracted stems to single waveform
        wav = wav.sum(1)

        # convert to mono-channel waveform
        wav = wav.mean(dim=1, keepdim=True)

        # resample to model sampling rate
        # not equivalent to julius.resample
        if sampling_rate != self.sampling_rate:
            # Resample the waveform to match the feature extractor's sampling rate
            wav = torchaudio.functional.resample(
                wav, sampling_rate, self.sampling_rate, rolloff=0.945, lowpass_filter_width=24
            )

        # [batch_size, 1, audio_length] -> [batch_size, audio_length]
        wav = wav.squeeze(1)

        return wav
    def __call__(
        self,
        audio: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        truncation: bool = True,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: Optional[bool] = None,
        padding: Optional[str] = True,
        max_length: Optional[int] = None,
        sampling_rate: Optional[int] = None,
        **kwargs,
    ):
        """
        调用函数，用于处理音频数据并返回处理后的结果。

        Parameters:
            audio (Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]]):
                输入的音频数据，可以是 numpy 数组，列表或嵌套列表形式。
            truncation (bool, optional):
                是否对音频进行截断，默认为 True。
            pad_to_multiple_of (Optional[int], optional):
                可选参数，对音频进行填充的倍数。
            return_tensors (Optional[Union[str, TensorType]], optional):
                可选参数，指定返回的数据类型，如字符串或张量类型。
            return_attention_mask (Optional[bool], optional):
                可选参数，是否返回注意力掩码。
            padding (Optional[str], optional):
                可选参数，是否进行填充，默认为 True。
            max_length (Optional[int], optional):
                可选参数，最大长度限制。
            sampling_rate (Optional[int], optional):
                可选参数，采样率。
            **kwargs:
                其他关键字参数。

        Returns:
            返回处理后的音频数据或特征。

        """
        # 在这里实现音频数据的处理逻辑，具体步骤依赖于输入参数和处理逻辑
        pass

    def to_dict(self) -> Dict[str, Any]:
        """
        将当前实例序列化为 Python 字典。

        Returns:
            `Dict[str, Any]`: 包含所有配置实例属性的字典。

        """
        # 深拷贝当前实例的 __dict__ 属性
        output = copy.deepcopy(self.__dict__)
        # 添加特定的类别信息到输出字典中
        output["feature_extractor_type"] = self.__class__.__name__
        # 如果存在特定的键，则从输出字典中删除相应的条目
        if "mel_filters" in output:
            del output["mel_filters"]
        if "window" in output:
            del output["window"]
        if "chroma_filters" in output:
            del output["chroma_filters"]
        if "spectrogram" in output:
            del output["spectrogram"]
        # 返回最终的输出字典
        return output
```