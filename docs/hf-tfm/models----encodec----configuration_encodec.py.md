# `.\models\encodec\configuration_encodec.py`

```py
# 定义了一个配置类 EncodecConfig，用于存储 EncodecModel 的配置信息
class EncodecConfig(PretrainedConfig):
    # 模型类型为 encodec
    model_type = "encodec"

    # 初始化函数，用于设定模型的各项配置参数
    def __init__(
        # 目标带宽列表，默认为 [1.5, 3.0, 6.0, 12.0, 24.0]
        self,
        target_bandwidths=[1.5, 3.0, 6.0, 12.0, 24.0],
        # 采样率，默认为 24,000 Hz
        sampling_rate=24_000,
        # 音频通道数，默认为 1
        audio_channels=1,
        # 是否进行归一化，默认为 False
        normalize=False,
        # 分块长度（秒），默认为 None
        chunk_length_s=None,
        # 重叠长度，默认为 None
        overlap=None,
        # 隐藏层大小，默认为 128
        hidden_size=128,
        # 滤波器数量，默认为 32
        num_filters=32,
        # 残差层数量，默认为 1
        num_residual_layers=1,
        # 上采样比例列表，默认为 [8, 5, 4, 2]
        upsampling_ratios=[8, 5, 4, 2],
        # 归一化类型，默认为 "weight_norm"
        norm_type="weight_norm",
        # 卷积核大小，默认为 7
        kernel_size=7,
        # 最后一层卷积核大小，默认为 7
        last_kernel_size=7,
        # 残差卷积核大小，默认为 3
        residual_kernel_size=3,
        # 膨胀增长率，默认为 2
        dilation_growth_rate=2,
        # 是否使用因果卷积，默认为 True
        use_causal_conv=True,
        # 填充模式，默认为 "reflect"
        pad_mode="reflect",
        # 压缩比例，默认为 2
        compress=2,
        # LSTM 层数，默认为 2
        num_lstm_layers=2,
        # 右侧修剪比例，默认为 1.0
        trim_right_ratio=1.0,
        # 代码本大小，默认为 1024
        codebook_size=1024,
        # 代码本维度，默认为 None
        codebook_dim=None,
        # 是否使用卷积作为快捷连接，默认为 True
        use_conv_shortcut=True,
        # 其他参数，用于接收未指定的配置参数
        **kwargs,
        # 设置目标带宽
        self.target_bandwidths = target_bandwidths
        # 设置采样率
        self.sampling_rate = sampling_rate
        # 设置音频通道数
        self.audio_channels = audio_channels
        # 设置是否进行归一化
        self.normalize = normalize
        # 设置分块长度（秒）
        self.chunk_length_s = chunk_length_s
        # 设置重叠率
        self.overlap = overlap
        # 设置隐藏层大小
        self.hidden_size = hidden_size
        # 设置卷积核数量
        self.num_filters = num_filters
        # 设置残差层数量
        self.num_residual_layers = num_residual_layers
        # 设置上采样比例
        self.upsampling_ratios = upsampling_ratios
        # 设置规范化类型
        self.norm_type = norm_type
        # 设置卷积核大小
        self.kernel_size = kernel_size
        # 设置最后一个卷积核大小
        self.last_kernel_size = last_kernel_size
        # 设置残差卷积核大小
        self.residual_kernel_size = residual_kernel_size
        # 设置扩张增长率
        self.dilation_growth_rate = dilation_growth_rate
        # 设置是否使用因果卷积
        self.use_causal_conv = use_causal_conv
        # 设置填充模式
        self.pad_mode = pad_mode
        # 设置是否压缩
        self.compress = compress
        # 设置LSTM层数
        self.num_lstm_layers = num_lstm_layers
        # 设置右边裁剪比例
        self.trim_right_ratio = trim_right_ratio
        # 设置码本大小
        self.codebook_size = codebook_size
        # 设置码本维度
        self.codebook_dim = codebook_dim if codebook_dim is not None else hidden_size
        # 设置是否使用卷积快捷方式
        self.use_conv_shortcut = use_conv_shortcut

        # 检查规范化类型是否有效，如果不是则抛出异常
        if self.norm_type not in ["weight_norm", "time_group_norm"]:
            raise ValueError(
                f'self.norm_type must be one of `"weight_norm"`, `"time_group_norm"`), got {self.norm_type}'
            )

        # 继承父类初始化方法
        super().__init__(**kwargs)

    # 这是一个属性，因为你可能想要随时更改分块长度（秒）
    @property
    def chunk_length(self) -> Optional[int]:
        # 如果分块长度（秒）为None，则返回None
        if self.chunk_length_s is None:
            return None
        else:
            # 否则返回采样率乘以分块长度的结果（采样点数）
            return int(self.chunk_length_s * self.sampling_rate)

    # 这是一个属性，因为你可能想要随时更改分块长度（秒）
    @property
    def chunk_stride(self) -> Optional[int]:
        # 如果分块长度（秒）或者重叠率为None，则返回None
        if self.chunk_length_s is None or self.overlap is None:
            return None
        else:
            # 否则返回分块长度乘以（1.0 - 重叠率）的结果的最大值
            return max(1, int((1.0 - self.overlap) * self.chunk_length))

    # 返回帧率
    @property
    def frame_rate(self) -> int:
        # 计算跳跃长度
        hop_length = np.prod(self.upsampling_ratios)
        # 返回采样率除以跳跃长度的结果向上取整
        return math.ceil(self.sampling_rate / hop_length)

    # 返回量化器数量
    @property
    def num_quantizers(self) -> int:
        # 返回目标带宽最后一个值乘以1000除以（帧率乘以10）的结果向下取整
        return int(1000 * self.target_bandwidths[-1] // (self.frame_rate * 10))
```