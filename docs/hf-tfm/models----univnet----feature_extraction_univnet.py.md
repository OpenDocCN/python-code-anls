# `.\transformers\models\univnet\feature_extraction_univnet.py`

```py
# 版权声明
# 本代码版权归HuggingFace团队所有
# 根据Apache License, Version 2.0许可，您不得使用这个文件，除非符合许可条件
# 您可以在以下位置获取许可证的拷贝
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可分发的软件是基于“AS IS”基础，没有任何形式的保证或条件
# 请查看许可证以获取关于权限和限制的具体信息

"""UnivNetModel的特征提取器类"""
# 导入所需的库和模块
from typing import Any, Dict, List, Optional, Union
import numpy as np
from ...audio_utils import mel_filter_bank, optimal_fft_length, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging

# 获取logger对象
logger = logging.get_logger(__name__)


class UnivNetFeatureExtractor(SequenceFeatureExtractor):
    r"""
    构建UnivNet特征提取器类

    该类使用短时傅里叶变换(STFT)从原始语音中提取对数梅尔滤波器组特征，STFT实现遵循TacoTron 2和Hifi-GAN的方法。

    这个特征提取器继承自[`~feature_extraction_sequence_utils.SequenceFeatureExtractor`]，其中包含大部分主要方法。
    用户可以参考这个超类获取更多关于这些方法的信息。

    """

    # 模型的输入名称列表
    model_input_names = ["input_features", "noise_sequence", "padding_mask"]

    def __init__(
        self,
        feature_size: int = 1,
        sampling_rate: int = 24000,
        padding_value: float = 0.0,
        do_normalize: bool = False,
        num_mel_bins: int = 100,
        hop_length: int = 256,
        win_length: int = 1024,
        win_function: str = "hann_window",
        filter_length: Optional[int] = 1024,
        max_length_s: int = 10,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        mel_floor: float = 1e-9,
        center: bool = False,
        compression_factor: float = 1.0,
        compression_clip_val: float = 1e-5,
        normalize_min: float = -11.512925148010254,
        normalize_max: float = 2.3143386840820312,
        model_in_channels: int = 64,
        pad_end_length: int = 10,
        return_attention_mask=True,
        **kwargs,
    ):
        # 调用父类的构造函数，初始化参数
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )

        # 初始化是否进行归一化的标志
        self.do_normalize = do_normalize

        # 初始化梅尔频谱的参数
        self.num_mel_bins = num_mel_bins
        self.hop_length = hop_length
        self.win_length = win_length
        self.win_function = win_function
        self.filter_length = filter_length
        self.fmin = fmin
        if fmax is None:
            # 如果未指定最大频率，则根据采样率计算
            # 遵循 librosa.filters.mel 的实现
            fmax = float(sampling_rate) / 2
        self.fmax = fmax
        self.mel_floor = mel_floor

        # 初始化最大长度的参数
        self.max_length_s = max_length_s
        self.num_max_samples = max_length_s * sampling_rate

        # 计算 FFT 长度
        if self.filter_length is None:
            self.n_fft = optimal_fft_length(self.win_length)
        else:
            self.n_fft = self.filter_length
        self.n_freqs = (self.n_fft // 2) + 1

        # 根据窗口函数及参数初始化窗口
        self.window = window_function(window_length=self.win_length, name=self.win_function, periodic=True)

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

        # 初始化其他参数
        self.center = center
        self.compression_factor = compression_factor
        self.compression_clip_val = compression_clip_val
        self.normalize_min = normalize_min
        self.normalize_max = normalize_max
        self.model_in_channels = model_in_channels
        self.pad_end_length = pad_end_length

    # 定义归一化函数
    def normalize(self, spectrogram):
        return 2 * ((spectrogram - self.normalize_min) / (self.normalize_max - self.normalize_min)) - 1

    # 定义反归一化函数
    def denormalize(self, spectrogram):
        return self.normalize_min + (self.normalize_max - self.normalize_min) * ((spectrogram + 1) / 2)
    # 计算给定波形的对数 MEL 频谱图
    def mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        # 根据 MelGAN 和 Hifi-GAN 的实现,对输入波形进行自定义填充,使用"reflect"模式填充
        waveform = np.pad(
            waveform,
            (int((self.n_fft - self.hop_length) / 2), int((self.n_fft - self.hop_length) / 2)),
            mode="reflect",
        )
    
        # 获取复数频谱图,注意 waveform 必须是单个波形(未批处理)
        complex_spectrogram = spectrogram(
            waveform,
            window=self.window,
            frame_length=self.n_fft,
            hop_length=self.hop_length,
            fft_length=self.n_fft,
            power=None,
            center=self.center,
            mel_filters=None,
            mel_floor=None,
        )
    
        # 手动应用 MEL 滤波器组和 MEL 下限,因为 UnivNet 使用了略有不同的实现
        amplitude_spectrogram = np.sqrt(
            np.real(complex_spectrogram) ** 2 + np.imag(complex_spectrogram) ** 2 + self.mel_floor
        )
        mel_spectrogram = np.matmul(self.mel_filters.T, amplitude_spectrogram)
    
        # 执行频谱归一化以获得对数 MEL 频谱图
        log_mel_spectrogram = np.log(
            np.clip(mel_spectrogram, a_min=self.compression_clip_val, a_max=None) * self.compression_factor
        )
    
        # 返回频谱图,num_mel_bins 在最后一维
        return log_mel_spectrogram.T
    
    # 生成噪音
    def generate_noise(
        self,
        noise_length: int,
        generator: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        生成标准高斯噪声的随机序列，用于 [`UnivNetModel.forward`] 方法中的 `noise_sequence` 参数。

        Args:
            spectrogram_length (`int`):
                生成的噪声的长度（dim 0）。
            model_in_channels (`int`, *optional*, defaults to `None`):
                生成的噪声的特征数（dim 1）。这应该对应于 [`UnivNetGan`] 模型的 `model_in_channels`。
                如果未设置，则默认为 `self.config.model_in_channels`。
            generator (`numpy.random.Generator`, *optional*, defaults to `None`):
                一个可选的 `numpy.random.Generator` 随机数生成器，用于控制噪声的生成。如果未设置，将创建一个带有新熵的新生成器。

        Returns:
            `numpy.ndarray`: 包含形状为 `(noise_length, model_in_channels)` 的随机标准高斯噪声的数组。
        """
        # 如果未提供随机数生成器，则使用默认的随机数生成器
        if generator is None:
            generator = np.random.default_rng()

        # 定义噪声的形状
        noise_shape = (noise_length, self.model_in_channels)
        # 生成随机标准高斯噪声
        noise = generator.standard_normal(noise_shape, dtype=np.float32)

        return noise

    def batch_decode(self, waveforms, waveform_lengths=None) -> List[np.ndarray]:
        r"""
        在运行 [`UnivNetModel.forward`] 后移除生成音频的填充。这返回一个不规则列表，其中包含 1D 音频波形数组，
        而不是单个张量/数组，因为一般来说，移除填充后波形的长度会不同。

        Args:
            waveforms (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                [`UnivNetModel`] 的批量输出波形。
            waveform_lengths (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
                每个波形在填充之前的批量长度。

        Returns:
            `List[np.ndarray]`: 一个不规则列表，其中包含已移除填充的 1D 波形数组。
        """
        # 将批量的波形张量折叠为 1D 音频波形列表
        waveforms = [waveform.detach().clone().cpu().numpy() for waveform in waveforms]

        # 如果提供了波形长度，则根据波形长度移除填充
        if waveform_lengths is not None:
            waveforms = [waveform[: waveform_lengths[i]] for i, waveform in enumerate(waveforms)]

        return waveforms
    # 定义一个方法，允许对象以函数的形式被调用，接受原始语音数据、采样率等参数
    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        sampling_rate: Optional[int] = None,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        pad_to_multiple_of: Optional[int] = None,
        return_noise: bool = True,
        generator: Optional[np.random.Generator] = None,
        pad_end: bool = False,
        pad_length: Optional[int] = None,
        do_normalize: Optional[str] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    # 转换当前对象为字典类型，返回字典
    def to_dict(self) -> Dict[str, Any]:
        # 调用父类方法返回字典对象
        output = super().to_dict()

        # 删除特定属性，因为这些属性可以从其他属性推导出来
        names = ["window", "mel_filters", "n_fft", "n_freqs", "num_max_samples"]
        for name in names:
            if name in output:
                del output[name]

        return output
```