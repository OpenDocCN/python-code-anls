# `.\models\speech_to_text\feature_extraction_speech_to_text.py`

```py
        feature_size=80,
        sampling_rate=16000,
        num_mel_bins=80,
        padding_value=0.0,
        do_ceptral_normalize=True,
        normalize_means=True,
        normalize_vars=True,
        **kwargs,

# 初始化函数，构造一个Speech2Text特征提取器对象，继承自SequenceFeatureExtractor类。接受多个参数来配置特征提取器的行为。


        super().__init__(**kwargs)

        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.num_mel_bins = num_mel_bins
        self.padding_value = padding_value
        self.do_ceptral_normalize = do_ceptral_normalize
        self.normalize_means = normalize_means
        self.normalize_vars = normalize_vars

# 调用父类SequenceFeatureExtractor的初始化函数，并设置特征提取器的各项参数。


    def _extract_mel_features(self, signal: np.ndarray) -> np.ndarray:
        """
        Extracts Mel-filter bank features from raw speech signal.

        Args:
            signal (`np.ndarray`): Raw speech signal.

        Returns:
            `np.ndarray`: Extracted Mel-filter bank features.
        """
        # 使用mel_filter_bank函数提取信号的Mel频率滤波器组特征
        return mel_filter_bank(
            signal,
            self.sampling_rate,
            self.num_mel_bins,
        )

# 提取原始语音信号的Mel频率滤波器组特征。


    def _apply_cmvn(self, features: np.ndarray) -> np.ndarray:
        """
        Applies utterance-level cepstral mean and variance normalization (CMVN) to the extracted features.

        Args:
            features (`np.ndarray`): Extracted features.

        Returns:
            `np.ndarray`: Features after CMVN.
        """
        # 计算特征的均值和方差，用于CMVN
        means = np.mean(features, axis=1, keepdims=True)
        variances = np.var(features, axis=1, keepdims=True)
        
        # 如果开启了归一化均值，则进行均值归一化
        if self.normalize_means:
            features -= means
        
        # 如果开启了归一化方差，则进行方差归一化
        if self.normalize_vars:
            features /= np.sqrt(variances + 1e-5)
        
        return features

# 对提取的特征应用语句级别的倒谱均值和方差归一化（CMVN）。


    def _extract_features(self, signal: np.ndarray) -> BatchFeature:
        """
        Extracts features from the raw speech signal.

        Args:
            signal (`np.ndarray`): Raw speech signal.

        Returns:
            `BatchFeature`: Batch of extracted features.
        """
        # 提取信号的频谱图
        spectrogram_feats = spectrogram(
            signal,
            self.sampling_rate,
            window_function,
        )

        # 提取频谱图的Mel频率滤波器组特征
        mel_feats = self._extract_mel_features(spectrogram_feats)

        # 如果开启了CMVN，则应用CMVN
        if self.do_ceptral_normalize:
            mel_feats = self._apply_cmvn(mel_feats)

        # 创建BatchFeature对象，封装提取的特征
        return BatchFeature(input_features=mel_feats, attention_mask=np.ones_like(mel_feats, dtype=np.float32))

# 从原始语音信号中提取特征。
    ):
        # 调用父类构造函数，设置特征大小、采样率、填充值等参数
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        # 设置梅尔频谱特征的参数
        self.num_mel_bins = num_mel_bins
        self.do_ceptral_normalize = do_ceptral_normalize
        self.normalize_means = normalize_means
        self.normalize_vars = normalize_vars
        self.return_attention_mask = True

        # 如果没有语音可用，则生成梅尔滤波器和窗口函数
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
            # 填充梅尔滤波器并设置窗口函数
            self.mel_filters = np.pad(mel_filters, ((0, 1), (0, 0)))
            self.window = window_function(400, "povey", periodic=False)

    def _extract_fbank_features(
        self,
        waveform: np.ndarray,
    ) -> np.ndarray:
        """
        Get mel-filter bank features using TorchAudio. Note that TorchAudio requires 16-bit signed integers as inputs
        and hence the waveform should not be normalized before feature extraction.
        """
        # 转换波形数据以符合Kaldi的要求：16位有符号整数
        waveform = waveform * (2**15)
        # 如果有语音可用，则使用TorchAudio提取梅尔滤波器特征
        if is_speech_available():
            waveform = torch.from_numpy(waveform).unsqueeze(0)
            features = ta_kaldi.fbank(waveform, num_mel_bins=self.num_mel_bins, sample_frequency=self.sampling_rate)
            features = features.numpy()
        else:
            # 如果没有语音可用，则使用自定义的声谱图函数进行特征提取
            waveform = np.squeeze(waveform)
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
        # 确保对float32数组进行归一化处理
        if normalize_means:
            mean = x[:input_length].mean(axis=0)
            x = np.subtract(x, mean)
        if normalize_vars:
            std = x[:input_length].std(axis=0)
            x = np.divide(x, std)

        # 如果输入长度小于数组长度，则用填充值填充数组
        if input_length < x.shape[0]:
            x[input_length:] = padding_value

        # 确保数组是float32类型
        x = x.astype(np.float32)

        return x

    def normalize(
        self, input_features: List[np.ndarray], attention_mask: Optional[np.ndarray] = None
    ) -> List[np.ndarray]:  
        # 函数定义，指定返回类型为包含 np.ndarray 的列表
        lengths = attention_mask.sum(-1) if attention_mask is not None else [x.shape[0] for x in input_features]  
        # 如果存在 attention_mask，则计算每个输入特征的有效长度；否则使用每个输入特征的长度
        return [  
            # 返回列表推导式，对每个输入特征应用 utterance_cmvn 函数，生成处理后的特征列表
            self.utterance_cmvn(x, n, self.normalize_means, self.normalize_vars, self.padding_value)
            for x, n in zip(input_features, lengths)  
            # 遍历 input_features 和 lengths 的并行列表，用于函数参数
        ]

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        padding: Union[bool, str, PaddingStrategy] = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        sampling_rate: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        **kwargs,
        # 定义 __call__ 方法，接收多个参数，包括 raw_speech、padding 等等，还有可变数量的关键字参数 kwargs
```