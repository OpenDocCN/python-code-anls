# `.\transformers\models\pop2piano\feature_extraction_pop2piano.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证要求，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的具体语言
""" Feature extractor class for Pop2Piano"""

# 引入警告模块
import warnings
# 引入类型提示相关模块
from typing import List, Optional, Union

# 引入 numpy 库
import numpy
# 别名为 np 的引入 numpy 库
import numpy as np

# 引入音频工具相关模块
from ...audio_utils import mel_filter_bank, spectrogram
# 引入特征提取序列工具相关模块
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
# 引入特征提取工具相关模块
from ...feature_extraction_utils import BatchFeature
# 引入工具函数相关模块
from ...utils import (
    TensorType,
    is_essentia_available,
    is_librosa_available,
    is_scipy_available,
    logging,
    requires_backends,
)

# 如果 essentia 可用，则引入 essentia 相关模块
if is_essentia_available():
    import essentia
    import essentia.standard

# 如果 librosa 可用，则引入 librosa 相关模块
if is_librosa_available():
    import librosa

# 如果 scipy 可用，则引入 scipy 相关模块
if is_scipy_available():
    import scipy

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 Pop2PianoFeatureExtractor 类，继承自 SequenceFeatureExtractor 类
class Pop2PianoFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Pop2Piano feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts rhythm and preprocesses the audio before it is passed to the model. First the audio is passed
    to `RhythmExtractor2013` algorithm which extracts the beat_times, beat positions and estimates their confidence as
    well as tempo in bpm, then beat_times is interpolated and to get beatsteps. Later we calculate
    extrapolated_beatsteps from it to be used in tokenizer. On the other hand audio is resampled to self.sampling_rate
    and preprocessed and then log mel spectogram is computed from that to be used in our transformer model.
    # 该类用于从音频信号中提取特征
    class AudioFeatureExtractor:
        """
        # 该类用于从音频信号中提取特征
    
        Args:
            # 目标采样率
            sampling_rate (`int`, *optional*, defaults to 22050):
                Target Sampling rate of audio signal. It's the sampling rate that we forward to the model.
            # 填充值
            padding_value (`int`, *optional*, defaults to 0):
                Padding value used to pad the audio. Should correspond to silences.
            # 窗口大小
            window_size (`int`, *optional*, defaults to 4096):
                Length of the window in samples to which the Fourier transform is applied.
            # 帧移
            hop_length (`int`, *optional*, defaults to 1024):
                Step size between each window of the waveform, in samples.
            # 最低频率
            min_frequency (`float`, *optional*, defaults to 10.0):
                Lowest frequency that will be used in the log-mel spectrogram.
            # 特征维度
            feature_size (`int`, *optional*, defaults to 512):
                The feature dimension of the extracted features.
            # 条形数
            num_bars (`int`, *optional*, defaults to 2):
                Determines interval between each sequence.
        """
    
        # 定义模型输入名称
        model_input_names = ["input_features", "beatsteps", "extrapolated_beatstep"]
    
        # 初始化函数
        def __init__(
            self,
            sampling_rate: int = 22050,
            padding_value: int = 0,
            window_size: int = 4096,
            hop_length: int = 1024,
            min_frequency: float = 10.0,
            feature_size: int = 512,
            num_bars: int = 2,
            **kwargs,
        ):
            # 调用父类初始化函数
            super().__init__(
                feature_size=feature_size,
                sampling_rate=sampling_rate,
                padding_value=padding_value,
                **kwargs,
            )
            # 设置成员变量
            self.sampling_rate = sampling_rate
            self.padding_value = padding_value
            self.window_size = window_size
            self.hop_length = hop_length
            self.min_frequency = min_frequency
            self.feature_size = feature_size
            self.num_bars = num_bars
            # 计算 mel 滤波器组
            self.mel_filters = mel_filter_bank(
                num_frequency_bins=(self.window_size // 2) + 1,
                num_mel_filters=self.feature_size,
                min_frequency=self.min_frequency,
                max_frequency=float(self.sampling_rate // 2),
                sampling_rate=self.sampling_rate,
                norm=None,
                mel_scale="htk",
            )
    
        # 计算梅尔频谱
        def mel_spectrogram(self, sequence: np.ndarray):
            """
            Generates MelSpectrogram.
    
            Args:
                sequence (`numpy.ndarray`):
                    The sequence of which the mel-spectrogram will be computed.
            """
            mel_specs = []
            for seq in sequence:
                # 应用汉宁窗
                window = np.hanning(self.window_size + 1)[:-1]
                # 计算梅尔频谱
                mel_specs.append(
                    spectrogram(
                        waveform=seq,
                        window=window,
                        frame_length=self.window_size,
                        hop_length=self.hop_length,
                        power=2.0,
                        mel_filters=self.mel_filters,
                    )
                )
            mel_specs = np.array(mel_specs)
    
            return mel_specs
    def extract_rhythm(self, audio: np.ndarray):
        """
        This algorithm(`RhythmExtractor2013`) extracts the beat positions and estimates their confidence as well as
        tempo in bpm for an audio signal. For more information please visit
        https://essentia.upf.edu/reference/std_RhythmExtractor2013.html .

        Args:
            audio(`numpy.ndarray`):
                raw audio waveform which is passed to the Rhythm Extractor.
        """
        # 检查是否需要 essentia 后端
        requires_backends(self, ["essentia"])
        # 使用 RhythmExtractor2013 算法提取节奏信息
        essentia_tracker = essentia.standard.RhythmExtractor2013(method="multifeature")
        # 获取节拍数、节拍时间、置信度、估计以及节拍间隔
        bpm, beat_times, confidence, estimates, essentia_beat_intervals = essentia_tracker(audio)

        return bpm, beat_times, confidence, estimates, essentia_beat_intervals

    def interpolate_beat_times(
        self, beat_times: numpy.ndarray, steps_per_beat: numpy.ndarray, n_extend: numpy.ndarray
    ):
        """
        This method takes beat_times and then interpolates that using `scipy.interpolate.interp1d` and the output is
        then used to convert raw audio to log-mel-spectrogram.

        Args:
            beat_times (`numpy.ndarray`):
                beat_times is passed into `scipy.interpolate.interp1d` for processing.
            steps_per_beat (`int`):
                used as an parameter to control the interpolation.
            n_extend (`int`):
                used as an parameter to control the interpolation.
        """

        # 检查是否需要 scipy 后端
        requires_backends(self, ["scipy"])
        # 使用 interp1d 函数对节拍时间进行插值
        beat_times_function = scipy.interpolate.interp1d(
            np.arange(beat_times.size),
            beat_times,
            bounds_error=False,
            fill_value="extrapolate",
        )

        # 扩展节拍时间序列
        ext_beats = beat_times_function(
            np.linspace(0, beat_times.size + n_extend - 1, beat_times.size * steps_per_beat + n_extend)
        )

        return ext_beats
    def preprocess_mel(self, audio: np.ndarray, beatstep: np.ndarray):
        """
        Preprocessing for log-mel-spectrogram

        Args:
            audio (`numpy.ndarray` of shape `(audio_length, )` ):
                Raw audio waveform to be processed.
            beatstep (`numpy.ndarray`):
                Interpolated values of the raw audio. If beatstep[0] is greater than 0.0, then it will be shifted by
                the value at beatstep[0].
        """

        # 检查输入的音频是否为单声道
        if audio is not None and len(audio.shape) != 1:
            raise ValueError(
                f"Expected `audio` to be a single channel audio input of shape `(n, )` but found shape {audio.shape}."
            )
        
        # 如果第一个节拍时间大于0，将所有节拍时间减去第一个节拍时间
        if beatstep[0] > 0.0:
            beatstep = beatstep - beatstep[0]

        # 计算所需的时间步数
        num_steps = self.num_bars * 4
        num_target_steps = len(beatstep)
        
        # 根据节拍时间和每拍的时间步骤数进行插值
        extrapolated_beatstep = self.interpolate_beat_times(
            beat_times=beatstep, steps_per_beat=1, n_extend=(self.num_bars + 1) * 4 + 1
        )

        sample_indices = []
        max_feature_length = 0
        
        # 遍历节拍时间，组成采样索引对和找到最大特征长度
        for i in range(0, num_target_steps, num_steps):
            start_idx = i
            end_idx = min(i + num_steps, num_target_steps)
            start_sample = int(extrapolated_beatstep[start_idx] * self.sampling_rate)
            end_sample = int(extrapolated_beatstep[end_idx] * self.sampling_rate)
            sample_indices.append((start_sample, end_sample))
            max_feature_length = max(max_feature_length, end_sample - start_sample)
        
        padded_batch = []
        
        # 对每个采样索引对进行填充和截取
        for start_sample, end_sample in sample_indices:
            feature = audio[start_sample:end_sample]
            padded_feature = np.pad(
                feature,
                ((0, max_feature_length - feature.shape[0]),),
                "constant",
                constant_values=0,
            )
            padded_batch.append(padded_feature)

        # 将填充后的批量特征转换为数组并返回
        padded_batch = np.asarray(padded_batch)
        return padded_batch, extrapolated_beatstep
```  
    # 定义一个函数用于对输入特征进行填充
    def _pad(self, features: np.ndarray, add_zero_line=True):
        # 获取输入特征的形状
        features_shapes = [each_feature.shape for each_feature in features]
        # 初始化填充后的特征和注意力掩码列表
        attention_masks, padded_features = [], []
        
        # 遍历每个输入特征
        for i, each_feature in enumerate(features):
            # 如果特征形状是3维的（"input_features"）
            if len(each_feature.shape) == 3:
                # 计算需要填充的大小
                features_pad_value = max([*zip(*features_shapes)][1]) - features_shapes[i][1]
                # 创建注意力掩码
                attention_mask = np.ones(features_shapes[i][:2], dtype=np.int64)
                # 创建特征填充和注意力掩码填充
                feature_padding = ((0, 0), (0, features_pad_value), (0, 0))
                attention_mask_padding = (feature_padding[0], feature_padding[1])
            # 如果特征形状是2维的（"beatsteps"和"extrapolated_beatstep"）
            else:
                # 将特征reshape为1维
                each_feature = each_feature.reshape(1, -1)
                # 计算需要填充的大小
                features_pad_value = max([*zip(*features_shapes)][0]) - features_shapes[i][0]
                # 创建注意力掩码
                attention_mask = np.ones(features_shapes[i], dtype=np.int64).reshape(1, -1)
                # 创建特征填充和注意力掩码填充
                feature_padding = attention_mask_padding = ((0, 0), (0, features_pad_value))
            
            # 对特征和注意力掩码进行填充
            each_padded_feature = np.pad(each_feature, feature_padding, "constant", constant_values=self.padding_value)
            attention_mask = np.pad(
                attention_mask, attention_mask_padding, "constant", constant_values=self.padding_value
            )
            
            # 如果需要添加零行
            if add_zero_line:
                # 获取最大的特征长度
                zero_array_len = max([*zip(*features_shapes)][1])
                # 在特征和注意力掩码后添加零行
                each_padded_feature = np.concatenate(
                    [each_padded_feature, np.zeros([1, zero_array_len, self.feature_size])], axis=0
                )
                attention_mask = np.concatenate(
                    [attention_mask, np.zeros([1, zero_array_len], dtype=attention_mask.dtype)], axis=0
                )
            
            # 将填充后的特征和注意力掩码添加到列表中
            padded_features.append(each_padded_feature)
            attention_masks.append(attention_mask)
        
        # 将所有填充后的特征和注意力掩码合并为单个数组
        padded_features = np.concatenate(padded_features, axis=0).astype(np.float32)
        attention_masks = np.concatenate(attention_masks, axis=0).astype(np.int64)
        
        # 返回填充后的特征和注意力掩码
        return padded_features, attention_masks
    
    # 定义一个函数用于对输入进行填充
    def pad(
        self,
        inputs: BatchFeature,
        is_batched: bool,
        return_attention_mask: bool,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ):
    ):
        """
        Pads the inputs to the same length and returns attention_mask.

        Args:
            inputs (`BatchFeature`):
                Processed audio features.
            is_batched (`bool`):
                Whether inputs are batched or not.
            return_attention_mask (`bool`):
                Whether to return attention mask or not.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of a list of Python integers. Acceptable values are:
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
                If nothing is specified, it will return a list of `np.ndarray` arrays.
        Return:
            `BatchFeature` with attention_mask, attention_mask_beatsteps, and attention_mask_extrapolated_beatstep added
            to it:
            - **attention_mask** numpy.ndarray of shape `(batch_size, max_input_features_seq_length)` --
                Example:
                    1, 1, 1, 0, 0 (audio 1, also padded to a maximum length of 5 with 2 zeros at
                    the end indicating padding)

                    0, 0, 0, 0, 0 (zero pad to separate audio 1 and 2)

                    1, 1, 1, 1, 1 (audio 2)

                    0, 0, 0, 0, 0 (zero pad to separate audio 2 and 3)

                    1, 1, 1, 1, 1 (audio 3)
            - **attention_mask_beatsteps** numpy.ndarray of shape `(batch_size, max_beatsteps_seq_length)`
            - **attention_mask_extrapolated_beatstep** numpy.ndarray of shape `(batch_size,
              max_extrapolated_beatstep_seq_length)`
        """

        processed_features_dict = {}
        # Iterate over the items in the inputs dictionary
        for feature_name, feature_value in inputs.items():
            # Check if the feature is "input_features" to determine padding conditions
            if feature_name == "input_features":
                # Call the _pad method to pad the feature values and get the attention mask
                padded_feature_values, attention_mask = self._pad(feature_value, add_zero_line=True)
                processed_features_dict[feature_name] = padded_feature_values
                # Add the attention mask to the processed features dictionary if specified
                if return_attention_mask:
                    processed_features_dict["attention_mask"] = attention_mask
            else:
                # Call the _pad method to pad the feature values and get the attention mask
                padded_feature_values, attention_mask = self._pad(feature_value, add_zero_line=False)
                processed_features_dict[feature_name] = padded_feature_values
                # Add the specific attention mask to the processed features dictionary if specified
                if return_attention_mask:
                    processed_features_dict[f"attention_mask_{feature_name}"] = attention_mask

        # If processing a single example and not returning attention mask, remove the zero array line
        if not is_batched and not return_attention_mask:
            processed_features_dict["input_features"] = processed_features_dict["input_features"][:-1, ...]

        # Create a BatchFeature object based on processed features and return it
        outputs = BatchFeature(processed_features_dict, tensor_type=return_tensors)

        return outputs
    # 定义一个特殊方法 __call__，用于对象的调用
    def __call__(
        # 输入参数 audio 可以是 numpy 数组，单个或列表形式的浮点数，单个或列表形式的 numpy 数组，或列表形式的列表浮点数
        self,
        audio: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        # 输入参数 sampling_rate 可以是整数，或整数的列表
        sampling_rate: Union[int, List[int]],
        # 默认参数 steps_per_beat 设置为 2
        steps_per_beat: int = 2,
        # 默认参数 resample 设置为 True，表示可以重新采样
        resample: Optional[bool] = True,
        # 默认参数 return_attention_mask 设置为 False，表示不返回注意力掩码
        return_attention_mask: Optional[bool] = False,
        # 默认参数 return_tensors 设置为 None，表示不返回张量
        return_tensors: Optional[Union[str, TensorType]] = None,
        # **kwargs 表示接受任意其他关键字参数
```