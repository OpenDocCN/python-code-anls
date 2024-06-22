# `.\transformers\models\vits\configuration_vits.py`

```py
# 导入必要的模块和函数
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练模型配置文件的映射，将预训练模型名称映射到其配置文件的 URL
VITS_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/mms-tts-eng": "https://huggingface.co/facebook/mms-tts-eng/resolve/main/config.json",
}

# VitsConfig 类，用于存储 VITS 模型的配置信息
class VitsConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VitsModel`]. It is used to instantiate a VITS
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the VITS
    [facebook/mms-tts-eng](https://huggingface.co/facebook/mms-tts-eng) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```python
    >>> from transformers import VitsModel, VitsConfig

    >>> # Initializing a "facebook/mms-tts-eng" style configuration
    >>> configuration = VitsConfig()

    >>> # Initializing a model (with random weights) from the "facebook/mms-tts-eng" style configuration
    >>> model = VitsModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```py"""

    # 模型类型标识为 "vits"
    model_type = "vits"
    # 初始化方法定义，接受多个参数设定模型的属性
    def __init__(
        self,
        vocab_size=38, # 词汇大小
        hidden_size=192, # 隐藏层的大小
        num_hidden_layers=6, # 隐藏层数量
        num_attention_heads=2, # 注意力机制的头数
        window_size=4, # 窗口大小
        use_bias=True, # 是否使用偏置
        ffn_dim=768, # 前馈网络的维度
        layerdrop=0.1, # 层丢失率
        ffn_kernel_size=3, # 前馈网络内核大小
        flow_size=192, # 流的大小
        spectrogram_bins=513, # 频谱图的分桶数量
        hidden_act="relu", # 隐藏层激活函数
        hidden_dropout=0.1, # 隐藏层的丢失率
        attention_dropout=0.1, # 注意力机制的丢失率
        activation_dropout=0.1, # 激活函数的丢失率
        initializer_range=0.02, # 初始化范围
        layer_norm_eps=1e-5, # 层归一化的epsilon值
        use_stochastic_duration_prediction=True, # 是否使用随机持续时间预测
        num_speakers=1, # 说话人数量
        speaker_embedding_size=0, # 说话人嵌入的大小
        upsample_initial_channel=512, # 上采样的初始通道数
        upsample_rates=[8, 8, 2, 2], # 上采样的比率
        upsample_kernel_sizes=[16, 16, 4, 4], # 上采样的核大小
        resblock_kernel_sizes=[3, 7, 11], # 残差块的核大小
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]], # 残差块的膨胀大小
        leaky_relu_slope=0.1, # LeakyReLU斜率
        depth_separable_channels=2, # 深度可分离卷积的通道数
        depth_separable_num_layers=3, # 深度可分离卷积的层数
        duration_predictor_flow_bins=10, # 持续时间预测器的流区间数
        duration_predictor_tail_bound=5.0, # 持续时间预测器的尾部界限
        duration_predictor_kernel_size=3, # 持续时间预测器的核大小
        duration_predictor_dropout=0.5, # 持续时间预测器的丢失率
        duration_predictor_num_flows=4, # 持续时间预测器的流数量
        duration_predictor_filter_channels=256, # 持续时间预测器的过滤通道数
        prior_encoder_num_flows=4, # 先前编码器的流数量
        prior_encoder_num_wavenet_layers=4, # 先前编码器的WaveNet层数
        posterior_encoder_num_wavenet_layers=16, # 后续编码器的WaveNet层数
        wavenet_kernel_size=5, # WaveNet的核大小
        wavenet_dilation_rate=1, # WaveNet的膨胀率
        wavenet_dropout=0.0, # WaveNet的丢失率
        speaking_rate=1.0, # 说话速率
        noise_scale=0.667, # 噪声比例
        noise_scale_duration=0.8, # 持续时间的噪声比例
        sampling_rate=16_000, # 采样率
        **kwargs, # 其他任意关键字参数
    # 设置语音合成模型的相关参数
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        window_size: int,
        use_bias: bool = False,
        ffn_dim: int = 2048,
        layerdrop: float = 0.0,
        ffn_kernel_size: int = 3,
        flow_size: int = 256,
        spectrogram_bins: int = 256,
        hidden_act: str = "gelu",
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-05,
        use_stochastic_duration_prediction: bool = False,
        num_speakers: int = 0,
        speaker_embedding_size: int = 0,
        upsample_initial_channel: int = 192,
        upsample_rates: List[int] = [2, 2, 2, 2],
        upsample_kernel_sizes: List[int] = [7, 7, 7, 7],
        resblock_kernel_sizes: List[List[int]] = [[3, 3], [1, 1]],
        resblock_dilation_sizes: List[List[int]] = [[1, 1], [1, 1]],
        leaky_relu_slope: float = 0.1,
        depth_separable_channels: int = 192,
        depth_separable_num_layers: int = 1,
        duration_predictor_flow_bins: int = 256,
        duration_predictor_tail_bound: float = 15.0,
        duration_predictor_kernel_size: int = 3,
        duration_predictor_dropout: float = 0.5,
        duration_predictor_num_flows: int = 2,
        duration_predictor_filter_channels: int = 256,
        prior_encoder_num_flows: int = 2,
        prior_encoder_num_wavenet_layers: int = 3,
        posterior_encoder_num_wavenet_layers: int = 3,
        wavenet_kernel_size: int = 3,
        wavenet_dilation_rate: List[int] = [1, 3, 9, 27],
        wavenet_dropout: float = 0.5,
        speaking_rate: float = 1.0,
        noise_scale: float = 0.3333333,
        noise_scale_duration: float = 0.2,
        sampling_rate: int = 22050,
        **kwargs
    ):
        # 设定语音合成模型的参数
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.window_size = window_size
        self.use_bias = use_bias
        self.ffn_dim = ffn_dim
        self.layerdrop = layerdrop
        self.ffn_kernel_size = ffn_kernel_size
        self.flow_size = flow_size
        self.spectrogram_bins = spectrogram_bins
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_stochastic_duration_prediction = use_stochastic_duration_prediction
        self.num_speakers = num_speakers
        self.speaker_embedding_size = speaker_embedding_size
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.leaky_relu_slope = leaky_relu_slope
        self.depth_separable_channels = depth_separable_channels
        self.depth_separable_num_layers = depth_separable_num_layers
        self.duration_predictor_flow_bins = duration_predictor_flow_bins
        self.duration_predictor_tail_bound = duration_predictor_tail_bound
        self.duration_predictor_kernel_size = duration_predictor_kernel_size
        self.duration_predictor_dropout = duration_predictor_dropout
        self.duration_predictor_num_flows = duration_predictor_num_flows
        self.duration_predictor_filter_channels = duration_predictor_filter_channels
        self.prior_encoder_num_flows = prior_encoder_num_flows
        self.prior_encoder_num_wavenet_layers = prior_encoder_num_wavenet_layers
        self.posterior_encoder_num_wavenet_layers = posterior_encoder_num_wavenet_layers
        self.wavenet_kernel_size = wavenet_kernel_size
        self.wavenet_dilation_rate = wavenet_dilation_rate
        self.wavenet_dropout = wavenet_dropout
        self.speaking_rate = speaking_rate
        self.noise_scale = noise_scale
        self.noise_scale_duration = noise_scale_duration
        self.sampling_rate = sampling_rate

        # 检查参数 `upsample_kernel_sizes` 和 `upsample_rates` 是否长度相同
        if len(upsample_kernel_sizes) != len(upsample_rates):
            raise ValueError(
                f"The length of `upsample_kernel_sizes` ({len(upsample_kernel_sizes)}) must match the length of "
                f"`upsample_rates` ({len(upsample_rates)})"
            )

        super().__init__(**kwargs)
```