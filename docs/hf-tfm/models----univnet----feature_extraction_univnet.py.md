# `.\models\univnet\feature_extraction_univnet.py`

```py
# 版权声明和许可信息，指明代码版权和使用许可条件
# 详细描述使用 Apache 许可证 2.0 版本，允许在遵守许可的前提下使用此代码
#
# 导入必要的库和模块
from typing import Any, Dict, List, Optional, Union

import numpy as np  # 导入 NumPy 库，用于数值计算

# 导入音频处理相关工具函数
from ...audio_utils import mel_filter_bank, optimal_fft_length, spectrogram, window_function
# 导入特征提取序列工具类
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
# 导入特征提取批处理类
from ...feature_extraction_utils import BatchFeature
# 导入通用工具类
from ...utils import PaddingStrategy, TensorType, logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义 UnivNetFeatureExtractor 类，继承自 SequenceFeatureExtractor 类
class UnivNetFeatureExtractor(SequenceFeatureExtractor):
    r"""
    构建 UnivNet 特征提取器。

    此类使用短时傅里叶变换 (STFT) 从原始语音中提取对数梅尔滤波器组特征。
    STFT 实现遵循 TacoTron 2 和 Hifi-GAN 的实现方式。

    此特征提取器继承自 [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`]，
    该超类包含大部分主要方法。用户应参考此超类以获取有关这些方法的更多信息。
    """

    # 模型输入名称列表
    model_input_names = ["input_features", "noise_sequence", "padding_mask"]

    # 初始化方法，设置特征提取器的各种参数
    def __init__(
        self,
        feature_size: int = 1,  # 特征大小，默认为 1
        sampling_rate: int = 24000,  # 采样率，默认为 24000 Hz
        padding_value: float = 0.0,  # 填充值，默认为 0.0
        do_normalize: bool = False,  # 是否进行归一化，默认为 False
        num_mel_bins: int = 100,  # 梅尔滤波器组数目，默认为 100
        hop_length: int = 256,  # 跳跃长度，默认为 256
        win_length: int = 1024,  # 窗口长度，默认为 1024
        win_function: str = "hann_window",  # 窗函数类型，默认为 "hann_window"
        filter_length: Optional[int] = 1024,  # 滤波器长度，默认为 1024
        max_length_s: int = 10,  # 最大长度（秒），默认为 10
        fmin: float = 0.0,  # 最低频率，默认为 0.0 Hz
        fmax: Optional[float] = None,  # 最高频率，可选参数
        mel_floor: float = 1e-9,  # 梅尔值下限，默认为 1e-9
        center: bool = False,  # 是否居中，默认为 False
        compression_factor: float = 1.0,  # 压缩因子，默认为 1.0
        compression_clip_val: float = 1e-5,  # 压缩剪切值，默认为 1e-5
        normalize_min: float = -11.512925148010254,  # 归一化最小值，默认为 -11.512925148010254
        normalize_max: float = 2.3143386840820312,  # 归一化最大值，默认为 2.3143386840820312
        model_in_channels: int = 64,  # 模型输入通道数，默认为 64
        pad_end_length: int = 10,  # 结尾填充长度，默认为 10
        return_attention_mask=True,  # 是否返回注意力掩码，默认为 True
        **kwargs,  # 其他可选关键字参数
    ):
        # 调用父类的初始化方法
        super().__init__()
        ):
            # 调用父类的构造函数，初始化对象
            super().__init__(
                feature_size=feature_size,
                sampling_rate=sampling_rate,
                padding_value=padding_value,
                return_attention_mask=return_attention_mask,
                **kwargs,
            )

            # 设置是否进行归一化的标志
            self.do_normalize = do_normalize

            # 设置 Mel 频率滤波器的参数
            self.num_mel_bins = num_mel_bins
            self.hop_length = hop_length
            self.win_length = win_length
            self.win_function = win_function
            self.filter_length = filter_length
            self.fmin = fmin
            if fmax is None:
                # 如果未指定 fmax，则根据采样率计算最大频率
                # 遵循 librosa.filters.mel 的实现
                fmax = float(sampling_rate) / 2
            self.fmax = fmax
            self.mel_floor = mel_floor

            # 设置最大长度（秒）及其对应的最大样本数
            self.max_length_s = max_length_s
            self.num_max_samples = max_length_s * sampling_rate

            # 根据是否指定了 filter_length 来决定使用的 FFT 长度
            if self.filter_length is None:
                self.n_fft = optimal_fft_length(self.win_length)
            else:
                self.n_fft = self.filter_length
            self.n_freqs = (self.n_fft // 2) + 1

            # 初始化窗口函数
            self.window = window_function(window_length=self.win_length, name=self.win_function, periodic=True)

            # 初始化 Mel 频率滤波器组
            self.mel_filters = mel_filter_bank(
                num_frequency_bins=self.n_freqs,
                num_mel_filters=self.num_mel_bins,
                min_frequency=self.fmin,
                max_frequency=self.fmax,
                sampling_rate=self.sampling_rate,
                norm="slaney",
                mel_scale="slaney",
            )

            # 设置中心化标志及其它相关参数
            self.center = center
            self.compression_factor = compression_factor
            self.compression_clip_val = compression_clip_val
            self.normalize_min = normalize_min
            self.normalize_max = normalize_max
            self.model_in_channels = model_in_channels
            self.pad_end_length = pad_end_length

        def normalize(self, spectrogram):
            # 对频谱进行归一化处理
            return 2 * ((spectrogram - self.normalize_min) / (self.normalize_max - self.normalize_min)) - 1

        def denormalize(self, spectrogram):
            # 对归一化后的频谱进行反归一化处理
            return self.normalize_min + (self.normalize_max - self.normalize_min) * ((spectrogram + 1) / 2)
    def mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """
        Calculates log MEL spectrograms from a batch of waveforms. Note that the input waveform(s) will be padded by
        `int(self.n_fft - self.hop_length) / 2` on both sides using the `reflect` padding mode.

        Args:
            waveform (`np.ndarray` of shape `(length,)`):
                The input waveform. This must be a single real-valued, mono waveform.

        Returns:
            `numpy.ndarray`: Array containing a log-mel spectrogram of shape `(num_frames, num_mel_bins)`.
        """
        # 根据 MelGAN 和 Hifi-GAN 实现的方式，自定义填充波形
        waveform = np.pad(
            waveform,
            (int((self.n_fft - self.hop_length) / 2), int((self.n_fft - self.hop_length) / 2)),
            mode="reflect",
        )

        # 获取复杂谱图
        # 注意：由于 spectrogram(...) 的实现方式，目前必须对波形进行解批处理
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

        # 手动应用 MEL 滤波器组和 MEL floor，因为 UnivNet 使用了稍微不同的实现方式
        amplitude_spectrogram = np.sqrt(
            np.real(complex_spectrogram) ** 2 + np.imag(complex_spectrogram) ** 2 + self.mel_floor
        )
        mel_spectrogram = np.matmul(self.mel_filters.T, amplitude_spectrogram)

        # 执行谱归一化以获得对数 MEL 谱图
        log_mel_spectrogram = np.log(
            np.clip(mel_spectrogram, a_min=self.compression_clip_val, a_max=None) * self.compression_factor
        )

        # 返回最后一个维度是 num_mel_bins 的谱图
        return log_mel_spectrogram.T

    def generate_noise(
        self,
        noise_length: int,
        generator: Optional[np.random.Generator] = None,
    def noise_sequence(self, noise_length: int, generator: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Generates a random noise sequence of standard Gaussian noise for use in the `noise_sequence` argument of
        [`UnivNetModel.forward`].

        Args:
            noise_length (`int`):
                The length of the generated noise sequence.
            generator (`numpy.random.Generator`, *optional*, defaults to `None`):
                An optional random number generator to control noise generation. If not provided, a new generator
                instance will be created using `np.random.default_rng()`.

        Returns:
            `numpy.ndarray`: Array containing random standard Gaussian noise of shape `(noise_length,
            self.model_in_channels)`.
        """
        # If no generator is provided, create a new default generator
        if generator is None:
            generator = np.random.default_rng()

        # Define the shape of the noise array based on noise_length and self.model_in_channels
        noise_shape = (noise_length, self.model_in_channels)
        
        # Generate standard normal noise using the generator
        noise = generator.standard_normal(noise_shape, dtype=np.float32)

        return noise

    def batch_decode(self, waveforms, waveform_lengths=None) -> List[np.ndarray]:
        r"""
        Removes padding from generated audio after running [`UnivNetModel.forward`]. This returns a ragged list of 1D
        audio waveform arrays and not a single tensor/array because in general the waveforms will have different
        lengths after removing padding.

        Args:
            waveforms (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                The batched output waveforms from the [`UnivNetModel`].
            waveform_lengths (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
                The batched lengths of each waveform before padding.

        Returns:
            `List[np.ndarray]`: A ragged list of 1D waveform arrays with padding removed.
        """
        # Convert each batched waveform tensor to a 1D numpy array
        waveforms = [waveform.detach().clone().cpu().numpy() for waveform in waveforms]

        # If waveform_lengths is provided, truncate each waveform according to its length
        if waveform_lengths is not None:
            waveforms = [waveform[: waveform_lengths[i]] for i, waveform in enumerate(waveforms)]

        return waveforms
    # 定义一个方法 __call__，用于处理语音数据的预处理和转换
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
    ):
        # 调用父类的 to_dict 方法，获取基类的属性字典
        output = super().to_dict()

        # 从属性字典中删除不需要序列化的属性
        names = ["window", "mel_filters", "n_fft", "n_freqs", "num_max_samples"]
        for name in names:
            if name in output:
                del output[name]

        # 返回处理后的属性字典
        return output
```