# `.\transformers\models\speecht5\configuration_speecht5.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 遵循 Apache License, Version 2.0 开源协议
# SpeechT5 模型配置

# 导入必要的模块
import functools
import operator
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义预训练模型配置文件的下载链接
SPEECHT5_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/speecht5_asr": "https://huggingface.co/microsoft/speecht5_asr/resolve/main/config.json",
    "microsoft/speecht5_tts": "https://huggingface.co/microsoft/speecht5_tts/resolve/main/config.json",
    "microsoft/speecht5_vc": "https://huggingface.co/microsoft/speecht5_vc/resolve/main/config.json",
}

SPEECHT5_PRETRAINED_HIFIGAN_CONFIG_ARCHIVE_MAP = {
    "microsoft/speecht5_hifigan": "https://huggingface.co/microsoft/speecht5_hifigan/resolve/main/config.json",
}

# 定义 SpeechT5Config 类，用于存储 SpeechT5 模型的配置信息
class SpeechT5Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SpeechT5Model`]. It is used to instantiate a
    SpeechT5 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the SpeechT5
    [microsoft/speecht5_asr](https://huggingface.co/microsoft/speecht5_asr) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    >>> from transformers import SpeechT5Model, SpeechT5Config

    >>> # Initializing a "microsoft/speecht5_asr" style configuration
    >>> configuration = SpeechT5Config()

    >>> # Initializing a model (with random weights) from the "microsoft/speecht5_asr" style configuration
    >>> model = SpeechT5Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    """
    # 定义模型类型
    model_type = "speecht5"
    # 定义属性映射
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "num_hidden_layers": "encoder_layers"}
    # Transformer-TTS 模型的初始化方法，用于初始化模型的各项参数
    def __init__(
        self,
        # 词汇表大小，默认为 81
        vocab_size=81,
        # 隐藏层大小，默认为 768
        hidden_size=768,
        # 编码器层数，默认为 12
        encoder_layers=12,
        # 编码器注意力头数，默认为 12
        encoder_attention_heads=12,
        # 编码器中 FFN 层的维度，默认为 3072
        encoder_ffn_dim=3072,
        # 编码器层级丢弃率，默认为 0.1
        encoder_layerdrop=0.1,
        # 解码器层数，默认为 6
        decoder_layers=6,
        # 解码器中 FFN 层的维度，默认为 3072
        decoder_ffn_dim=3072,
        # 解码器注意力头数，默认为 12
        decoder_attention_heads=12,
        # 解码器层级丢弃率，默认为 0.1
        decoder_layerdrop=0.1,
        # 隐藏层激活函数，默认为 "gelu"
        hidden_act="gelu",
        # 位置编码丢弃率，默认为 0.1
        positional_dropout=0.1,
        # 隐藏层丢弃率，默认为 0.1
        hidden_dropout=0.1,
        # 注意力机制丢弃率，默认为 0.1
        attention_dropout=0.1,
        # 激活函数丢弃率，默认为 0.1
        activation_dropout=0.1,
        # 初始化范围，默认为 0.02
        initializer_range=0.02,
        # Layer Normalization 的 epsilon 值，默认为 1e-5
        layer_norm_eps=1e-5,
        # 是否对嵌入进行缩放，默认为 False
        scale_embedding=False,
        # 特征提取的归一化方式，默认为 "group"
        feat_extract_norm="group",
        # 特征投影的丢弃率，默认为 0.0
        feat_proj_dropout=0.0,
        # 特征提取的激活函数，默认为 "gelu"
        feat_extract_activation="gelu",
        # 卷积层维度，默认为 (512, 512, 512, 512, 512, 512, 512)
        conv_dim=(512, 512, 512, 512, 512, 512, 512),
        # 卷积层步长，默认为 (5, 2, 2, 2, 2, 2, 2)
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        # 卷积层卷积核大小，默认为 (10, 3, 3, 3, 3, 2, 2)
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        # 是否包含卷积层偏置，默认为 False
        conv_bias=False,
        # 卷积位置编码的数量，默认为 128
        num_conv_pos_embeddings=128,
        # 卷积位置编码的分组数量，默认为 16
        num_conv_pos_embedding_groups=16,
        # 是否应用特定的数据增强，默认为 True
        apply_spec_augment=True,
        # 时间掩码的概率，默认为 0.05
        mask_time_prob=0.05,
        # 时间掩码的长度，默认为 10
        mask_time_length=10,
        # 时间掩码的最小数量，默认为 2
        mask_time_min_masks=2,
        # 特征掩码的概率，默认为 0.0
        mask_feature_prob=0.0,
        # 特征掩码的长度，默认为 10
        mask_feature_length=10,
        # 特征掩码的最小数量，默认为 0
        mask_feature_min_masks=0,
        # 填充标记的 ID，默认为 1
        pad_token_id=1,
        # 起始标记的 ID，默认为 0
        bos_token_id=0,
        # 结束标记的 ID，默认为 2
        eos_token_id=2,
        # 解码器起始标记的 ID，默认为 2
        decoder_start_token_id=2,
        # 梅尔频谱的数量，默认为 80
        num_mel_bins=80,
        # 语音解码器的预处理层级数，默认为 2
        speech_decoder_prenet_layers=2,
        # 语音解码器的预处理单元数，默认为 256
        speech_decoder_prenet_units=256,
        # 语音解码器的预处理层级丢弃率，默认为 0.5
        speech_decoder_prenet_dropout=0.5,
        # 说话者嵌入维度，默认为 512
        speaker_embedding_dim=512,
        # 语音解码器的后处理层级数，默认为 5
        speech_decoder_postnet_layers=5,
        # 语音解码器的后处理单元数，默认为 256
        speech_decoder_postnet_units=256,
        # 语音解码器的后处理卷积核大小，默认为 5
        speech_decoder_postnet_kernel=5,
        # 语音解码器的后处理层级丢弃率，默认为 0.5
        speech_decoder_postnet_dropout=0.5,
        # 缩小因子，默认为 2
        reduction_factor=2,
        # 最大语音位置，默认为 4000
        max_speech_positions=4000,
        # 最大文本位置，默认为 450
        max_text_positions=450,
        # 编码器最大相对位置，默认为 160
        encoder_max_relative_position=160,
        # 是否使用引导注意力损失，默认为 True
        use_guided_attention_loss=True,
        # 引导注意力损失的头数，默认为 2
        guided_attention_loss_num_heads=2,
        # 引导注意力损失的标准差，默认为 0.4
        guided_attention_loss_sigma=0.4,
        # 引导注意力损失的缩放因子，默认为 10.0
        guided_attention_loss_scale=10.0,
        #
class SpeechT5HifiGanConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SpeechT5HifiGanModel`]. It is used to instantiate
    a SpeechT5 HiFi-GAN vocoder model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the SpeechT5
    [microsoft/speecht5_hifigan](https://huggingface.co/microsoft/speecht5_hifigan) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        model_in_dim (`int`, *optional*, defaults to 80):
            The number of frequency bins in the input log-mel spectrogram. 模型输入 log-mel 频谱图中的频率分 bin 数。
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the output audio will be generated, expressed in hertz (Hz). 输出音频的采样率，单位为赫兹（Hz）。
        upsample_initial_channel (`int`, *optional*, defaults to 512):
            The number of input channels into the upsampling network. 上采样网络的输入通道数。
        upsample_rates (`Tuple[int]` or `List[int]`, *optional*, defaults to `[4, 4, 4, 4]`):
            A tuple of integers defining the stride of each 1D convolutional layer in the upsampling network. The
            length of *upsample_rates* defines the number of convolutional layers and has to match the length of
            *upsample_kernel_sizes*. 上采样网络中每个 1D 卷积层的步长的整数元组。*upsample_rates* 的长度定义了卷积层的数量，必须与 *upsample_kernel_sizes* 的长度匹配。
        upsample_kernel_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[8, 8, 8, 8]`):
            A tuple of integers defining the kernel size of each 1D convolutional layer in the upsampling network. The
            length of *upsample_kernel_sizes* defines the number of convolutional layers and has to match the length of
            *upsample_rates*. 上采样网络中每个 1D 卷积层的内核大小的整数元组。*upsample_kernel_sizes* 的长度定义了卷积层的数量，必须与 *upsample_rates* 的长度匹配。
        resblock_kernel_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[3, 7, 11]`):
            A tuple of integers defining the kernel sizes of the 1D convolutional layers in the multi-receptive field
            fusion (MRF) module. 多感受野融合（MRF）模块中 1D 卷积层的内核大小的整数元组。
        resblock_dilation_sizes (`Tuple[Tuple[int]]` or `List[List[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`):
            A nested tuple of integers defining the dilation rates of the dilated 1D convolutional layers in the
            multi-receptive field fusion (MRF) module. 多感受野融合（MRF）模块中扩张的 1D 卷积层的膨胀率的嵌套元组。
        initializer_range (`float`, *optional*, defaults to 0.01):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices. 用于初始化所有权重矩阵的截断正态初始化器的标准偏差。
        leaky_relu_slope (`float`, *optional*, defaults to 0.1):
            The angle of the negative slope used by the leaky ReLU activation. leaky ReLU 激活函数使用的负斜率的角度。
        normalize_before (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the spectrogram before vocoding using the vocoder's learned mean and variance.
            是否在进行语音编码之前对频谱图进行归一化，使用语音编码器学习的均值和方差。

    Example:

    ```python
    >>> from transformers import SpeechT5HifiGan, SpeechT5HifiGanConfig

    >>> # 初始化一个 "microsoft/speecht5_hifigan" 风格的配置
    >>> configuration = SpeechT5HifiGanConfig()

    >>> # 使用 "microsoft/speecht5_hifigan" 风格的配置初始化一个模型（带有随机权重）
    >>> model = SpeechT5HifiGan(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py

    # 定义模型类型为 "hifigan"
    model_type = "hifigan"

    # 初始化函数
    def __init__(
        self,
        model_in_dim=80,
        sampling_rate=16000,
        upsample_initial_channel=512,
        upsample_rates=[4, 4, 4, 4],
        upsample_kernel_sizes=[8, 8, 8, 8],
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        initializer_range=0.01,
        leaky_relu_slope=0.1,
        normalize_before=True,
        **kwargs,
    ):
        # 输入维度
        self.model_in_dim = model_in_dim
        # 采样率
        self.sampling_rate = sampling_rate
        # 上采样的初始通道数
        self.upsample_initial_channel = upsample_initial_channel
        # 上采样率列表
        self.upsample_rates = upsample_rates
        # 上采样的卷积核大小列表
        self.upsample_kernel_sizes = upsample_kernel_sizes
        # ResBlock 的卷积核大小列表
        self.resblock_kernel_sizes = resblock_kernel_sizes
        # ResBlock 的膨胀大小列表
        self.resblock_dilation_sizes = resblock_dilation_sizes
        # 初始化范围
        self.initializer_range = initializer_range
        # LeakyReLU 斜率
        self.leaky_relu_slope = leaky_relu_slope
        # 是否在归一化之前进行操作
        self.normalize_before = normalize_before
        # 调用父类初始化函数
        super().__init__(**kwargs)
```