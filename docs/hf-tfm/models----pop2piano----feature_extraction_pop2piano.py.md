# `.\models\pop2piano\feature_extraction_pop2piano.py`

```py
    r"""
    Constructs a Pop2Piano feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts rhythm and preprocesses the audio before it is passed to the model. First the audio is passed
    to `RhythmExtractor2013` algorithm which extracts the beat_times, beat positions and estimates their confidence as
    well as tempo in bpm, then beat_times is interpolated and to get beatsteps. Later we calculate
    extrapolated_beatsteps from it to be used in tokenizer. On the other hand audio is resampled to self.sampling_rate
    and preprocessed and then log mel spectogram is computed from that to be used in our transformer model.
    """
    
    # 引入警告模块，用于可能的警告信息输出
    import warnings
    # 引入类型提示模块，用于类型检查和提示
    from typing import List, Optional, Union

    # 引入 numpy 库，并给其起一个别名 np
    import numpy
    import numpy as np

    # 引入音频处理相关的函数和工具
    from ...audio_utils import mel_filter_bank, spectrogram
    # 引入特征提取序列工具函数
    from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
    # 引入批处理特征工具
    from ...feature_extraction_utils import BatchFeature
    # 引入常用工具函数
    from ...utils import (
        TensorType,
        is_essentia_available,
        is_librosa_available,
        is_scipy_available,
        logging,
        requires_backends,
    )

    # 如果 Essentia 库可用，则导入相关模块
    if is_essentia_available():
        import essentia
        import essentia.standard

    # 如果 Librosa 库可用，则导入 Librosa 模块
    if is_librosa_available():
        import librosa

    # 如果 Scipy 库可用，则导入 Scipy 模块
    if is_scipy_available():
        import scipy

    # 获取日志记录器对象
    logger = logging.get_logger(__name__)


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
    """
    model_input_names = ["input_features", "beatsteps", "extrapolated_beatstep"]



    # 定义模型输入的名称列表，包括输入特征、节拍步长和外推的节拍步长
    model_input_names = ["input_features", "beatsteps", "extrapolated_beatstep"]



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
        # 调用父类初始化方法，设置特征大小、采样率和填充值等参数
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        # 设置对象的属性，包括采样率、填充值、窗口大小、跳跃长度、最小频率、特征大小和节拍条数
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value
        self.window_size = window_size
        self.hop_length = hop_length
        self.min_frequency = min_frequency
        self.feature_size = feature_size
        self.num_bars = num_bars
        # 计算梅尔滤波器组，用于后续的梅尔频谱计算
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=(self.window_size // 2) + 1,
            num_mel_filters=self.feature_size,
            min_frequency=self.min_frequency,
            max_frequency=float(self.sampling_rate // 2),
            sampling_rate=self.sampling_rate,
            norm=None,
            mel_scale="htk",
        )




    def mel_spectrogram(self, sequence: np.ndarray):
        """
        Generates MelSpectrogram.

        Args:
            sequence (`numpy.ndarray`):
                The sequence of which the mel-spectrogram will be computed.
        """
        # 初始化空的梅尔频谱列表
        mel_specs = []
        # 对输入的每个序列进行处理
        for seq in sequence:
            # 应用汉宁窗口函数，用于信号的加窗处理
            window = np.hanning(self.window_size + 1)[:-1]
            # 计算当前序列的梅尔频谱，并加入到梅尔频谱列表中
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
        # 将梅尔频谱列表转换为 numpy 数组并返回
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
        # 检查必需的后端库是否存在
        requires_backends(self, ["essentia"])
        # 创建 RhythmExtractor2013 对象，使用多特征方法
        essentia_tracker = essentia.standard.RhythmExtractor2013(method="multifeature")
        # 调用 RhythmExtractor2013 对象处理音频，返回节奏信息
        bpm, beat_times, confidence, estimates, essentia_beat_intervals = essentia_tracker(audio)

        # 返回节拍频率、节拍时间、置信度、估计值和节拍间隔
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

        # 检查必需的后端库是否存在
        requires_backends(self, ["scipy"])
        # 创建用于插值的 interp1d 函数对象
        beat_times_function = scipy.interpolate.interp1d(
            np.arange(beat_times.size),
            beat_times,
            bounds_error=False,
            fill_value="extrapolate",
        )

        # 使用插值函数对节拍时间进行插值扩展
        ext_beats = beat_times_function(
            np.linspace(0, beat_times.size + n_extend - 1, beat_times.size * steps_per_beat + n_extend)
        )

        # 返回插值后的节拍时间
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

        # 检查输入的音频是否为单声道，并且不为 None
        if audio is not None and len(audio.shape) != 1:
            raise ValueError(
                f"Expected `audio` to be a single channel audio input of shape `(n, )` but found shape {audio.shape}."
            )
        
        # 如果 beatstep 的第一个值大于 0.0，则将整个 beatstep 数组向左平移至第一个元素为 0.0
        if beatstep[0] > 0.0:
            beatstep = beatstep - beatstep[0]

        # 计算预处理后的数据点数
        num_steps = self.num_bars * 4
        num_target_steps = len(beatstep)
        
        # 对节拍时间进行插值，以扩展为与处理后的步数匹配的时间点
        extrapolated_beatstep = self.interpolate_beat_times(
            beat_times=beatstep, steps_per_beat=1, n_extend=(self.num_bars + 1) * 4 + 1
        )

        # 初始化样本索引列表和最大特征长度
        sample_indices = []
        max_feature_length = 0
        
        # 划分样本段并计算每个段的特征长度
        for i in range(0, num_target_steps, num_steps):
            start_idx = i
            end_idx = min(i + num_steps, num_target_steps)
            start_sample = int(extrapolated_beatstep[start_idx] * self.sampling_rate)
            end_sample = int(extrapolated_beatstep[end_idx] * self.sampling_rate)
            sample_indices.append((start_sample, end_sample))
            max_feature_length = max(max_feature_length, end_sample - start_sample)
        
        # 初始化填充后的特征批处理列表
        padded_batch = []
        
        # 对每个样本段进行填充处理
        for start_sample, end_sample in sample_indices:
            feature = audio[start_sample:end_sample]
            padded_feature = np.pad(
                feature,
                ((0, max_feature_length - feature.shape[0]),),  # 在最后一维进行填充
                "constant",
                constant_values=0,  # 使用常数值 0 进行填充
            )
            padded_batch.append(padded_feature)

        # 将填充后的特征批处理列表转换为 numpy 数组
        padded_batch = np.asarray(padded_batch)
        
        # 返回填充后的特征批处理和插值后的节拍时间
        return padded_batch, extrapolated_beatstep
    # 定义一个内部方法用于填充特征数据
    def _pad(self, features: np.ndarray, add_zero_line=True):
        # 计算每个特征数据的形状并存储在列表中
        features_shapes = [each_feature.shape for each_feature in features]
        # 初始化存放注意力掩码和填充后特征数据的列表
        attention_masks, padded_features = [], []

        # 遍历每个特征数据及其索引
        for i, each_feature in enumerate(features):
            # 如果特征数据是三维的，则进行"input_features"的填充
            if len(each_feature.shape) == 3:
                # 计算需要填充的值，即特征数据第二维的差值
                features_pad_value = max([*zip(*features_shapes)][1]) - features_shapes[i][1]
                # 创建全为1的注意力掩码
                attention_mask = np.ones(features_shapes[i][:2], dtype=np.int64)
                # 设置特征数据的填充方式和注意力掩码的填充方式
                feature_padding = ((0, 0), (0, features_pad_value), (0, 0))
                attention_mask_padding = (feature_padding[0], feature_padding[1])

            # 如果特征数据是其他维度的，则进行"beatsteps"和"extrapolated_beatstep"的填充
            else:
                # 将特征数据reshape为二维
                each_feature = each_feature.reshape(1, -1)
                # 计算需要填充的值，即特征数据第一维的差值
                features_pad_value = max([*zip(*features_shapes)][0]) - features_shapes[i][0]
                # 创建全为1的注意力掩码并reshape为二维
                attention_mask = np.ones(features_shapes[i], dtype=np.int64).reshape(1, -1)
                # 设置特征数据的填充方式和注意力掩码的填充方式
                feature_padding = attention_mask_padding = ((0, 0), (0, features_pad_value))

            # 对每个特征数据进行填充，使用常数值self.padding_value
            each_padded_feature = np.pad(each_feature, feature_padding, "constant", constant_values=self.padding_value)
            # 对注意力掩码进行填充，使用常数值self.padding_value
            attention_mask = np.pad(
                attention_mask, attention_mask_padding, "constant", constant_values=self.padding_value
            )

            # 如果需要添加零行（add_zero_line为True）
            if add_zero_line:
                # 计算零数组的长度，即特征数据第二维的最大值
                zero_array_len = max([*zip(*features_shapes)][1])

                # 将零数组行连接到每个填充后的特征数据末尾
                each_padded_feature = np.concatenate(
                    [each_padded_feature, np.zeros([1, zero_array_len, self.feature_size])], axis=0
                )
                # 将零数组行连接到每个填充后的注意力掩码末尾
                attention_mask = np.concatenate(
                    [attention_mask, np.zeros([1, zero_array_len], dtype=attention_mask.dtype)], axis=0
                )

            # 将填充后的特征数据和注意力掩码添加到对应的列表中
            padded_features.append(each_padded_feature)
            attention_masks.append(attention_mask)

        # 将所有填充后的特征数据连接成一个numpy数组，并转换为float32类型
        padded_features = np.concatenate(padded_features, axis=0).astype(np.float32)
        # 将所有填充后的注意力掩码连接成一个numpy数组，并转换为int64类型
        attention_masks = np.concatenate(attention_masks, axis=0).astype(np.int64)

        # 返回填充后的特征数据和注意力掩码
        return padded_features, attention_masks
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
                    1, 1, 1, 0, 0 (audio 1, padded to a max length of 5 with 2 zeros indicating padding)

                    0, 0, 0, 0, 0 (zero padding to separate audio 1 and 2)

                    1, 1, 1, 1, 1 (audio 2)

                    0, 0, 0, 0, 0 (zero padding to separate audio 2 and 3)

                    1, 1, 1, 1, 1 (audio 3)
            - **attention_mask_beatsteps** numpy.ndarray of shape `(batch_size, max_beatsteps_seq_length)`
            - **attention_mask_extrapolated_beatstep** numpy.ndarray of shape `(batch_size,
              max_extrapolated_beatstep_seq_length)`
        """

        processed_features_dict = {}
        # Iterate through each feature and pad its values
        for feature_name, feature_value in inputs.items():
            # If the feature is 'input_features', pad it with an additional zero line
            if feature_name == "input_features":
                padded_feature_values, attention_mask = self._pad(feature_value, add_zero_line=True)
                processed_features_dict[feature_name] = padded_feature_values
                # Optionally add attention_mask to processed_features_dict
                if return_attention_mask:
                    processed_features_dict["attention_mask"] = attention_mask
            else:
                # For other features, pad without adding an extra zero line
                padded_feature_values, attention_mask = self._pad(feature_value, add_zero_line=False)
                processed_features_dict[feature_name] = padded_feature_values
                # Optionally add feature-specific attention_mask to processed_features_dict
                if return_attention_mask:
                    processed_features_dict[f"attention_mask_{feature_name}"] = attention_mask

        # If processing a single example and not returning attention_mask, remove the last zero array line
        if not is_batched and not return_attention_mask:
            processed_features_dict["input_features"] = processed_features_dict["input_features"][:-1, ...]

        # Create BatchFeature object with processed features and optionally convert to specified tensor type
        outputs = BatchFeature(processed_features_dict, tensor_type=return_tensors)

        return outputs
    # 定义一个特殊方法 `__call__`，使得对象可以像函数一样被调用
    def __call__(
        # 输入参数 audio 可以是 numpy 数组、浮点数列表、numpy 数组列表或浮点数列表的列表
        self,
        # 输入参数 sampling_rate 可以是整数或整数列表，表示采样率
        audio: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        # 每个拍子的步数，默认为 2
        sampling_rate: Union[int, List[int]],
        # 每个拍子的步数，默认为 2
        steps_per_beat: int = 2,
        # 是否重新采样音频，默认为 True
        resample: Optional[bool] = True,
        # 是否返回注意力掩码，默认为 False
        return_attention_mask: Optional[bool] = False,
        # 是否返回张量，默认为 None（即不返回张量）
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 其他关键字参数，以字典形式接收
        **kwargs,
```