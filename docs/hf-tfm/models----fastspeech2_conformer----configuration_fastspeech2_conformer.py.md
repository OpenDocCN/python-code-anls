# `.\models\fastspeech2_conformer\configuration_fastspeech2_conformer.py`

```
# 设置文件编码格式为utf-8
# 版权声明
from typing import Dict
# 导入相关的模块和类

# 导入日志记录模块
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 定义预训练配置和对应的URL映射字典
FASTSPEECH2_CONFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "espnet/fastspeech2_conformer": "https://huggingface.co/espnet/fastspeech2_conformer/raw/main/config.json",
}

FASTSPEECH2_CONFORMER_HIFIGAN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "espnet/fastspeech2_conformer_hifigan": "https://huggingface.co/espnet/fastspeech2_conformer_hifigan/raw/main/config.json",
}

FASTSPEECH2_CONFORMER_WITH_HIFIGAN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "espnet/fastspeech2_conformer_with_hifigan": "https://huggingface.co/espnet/fastspeech2_conformer_with_hifigan/raw/main/config.json",
}

# FastSpeech2ConformerConfig类继承自PretrainedConfig类
class FastSpeech2ConformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FastSpeech2ConformerModel`]. It is used to
    instantiate a FastSpeech2Conformer model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the
    FastSpeech2Conformer [espnet/fastspeech2_conformer](https://huggingface.co/espnet/fastspeech2_conformer)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```python
    >>> from transformers import FastSpeech2ConformerModel, FastSpeech2ConformerConfig

    >>> # Initializing a FastSpeech2Conformer style configuration
    >>> configuration = FastSpeech2ConformerConfig()

    >>> # Initializing a model from the FastSpeech2Conformer style configuration
    >>> model = FastSpeech2ConformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "fastspeech2_conformer"
    attribute_map = {"num_hidden_layers": "encoder_layers", "num_attention_heads": "encoder_num_attention_heads"}

# End of code
    # 初始化函数，用于初始化模型的各项参数
    def __init__(
        self,
        # 隐藏层的大小，默认为384
        hidden_size=384,
        # 词汇表的大小，默认为78
        vocab_size=78,
        # 梅尔频谱的频道数，默认为80
        num_mel_bins=80,
        # 编码器的注意力头数，默认为2
        encoder_num_attention_heads=2,
        # 编码器的层数，默认为4
        encoder_layers=4,
        # 编码器线性单元的数量，默认为1536
        encoder_linear_units=1536,
        # 解码器的层数，默认为4
        decoder_layers=4,
        # 解码器的注意力头数，默认为2
        decoder_num_attention_heads=2,
        # 解码器线性单元的数量，默认为1536
        decoder_linear_units=1536,
        # 语音解码器后处理网络的层数，默认为5
        speech_decoder_postnet_layers=5,
        # 语音解码器后处理网络的单元数量，默认为256
        speech_decoder_postnet_units=256,
        # 语音解码器后处理网络的卷积核大小，默认为5
        speech_decoder_postnet_kernel=5,
        # 位置卷积的核大小，默认为3
        positionwise_conv_kernel_size=3,
        # 编码器是否在归一化前进行拼接，默认为False
        encoder_normalize_before=False,
        # 解码器是否在归一化前进行拼接，默认为False
        decoder_normalize_before=False,
        # 编码器是否在拼接后进行拼接，默认为False
        encoder_concat_after=False,
        # 解码器是否在拼接后进行拼接，默认为False
        decoder_concat_after=False,
        # 缩减因子，默认为1
        reduction_factor=1,
        # 说话速度，默认为1.0
        speaking_speed=1.0,
        # 在Conformer中使用Macaron风格，默认为True
        use_macaron_style_in_conformer=True,
        # 在Conformer中使用CNN，默认为True
        use_cnn_in_conformer=True,
        # 编码器的卷积核大小，默认为7
        encoder_kernel_size=7,
        # 解码器的卷积核大小，默认为31
        decoder_kernel_size=31,
        # 时长预测器的层数，默认为2
        duration_predictor_layers=2,
        # 时长预测器的通道数，默认为256
        duration_predictor_channels=256,
        # 时长预测器的卷积核大小，默认为3
        duration_predictor_kernel_size=3,
        # 能量预测器的层数，默认为2
        energy_predictor_layers=2,
        # 能量预测器的通道数，默认为256
        energy_predictor_channels=256,
        # 能量预测器的卷积核大小，默认为3
        energy_predictor_kernel_size=3,
        # 能量预测器的丢弃率，默认为0.5
        energy_predictor_dropout=0.5,
        # 能量嵌入的卷积核大小，默认为1
        energy_embed_kernel_size=1,
        # 能量嵌入的丢弃率，默认为0.0
        energy_embed_dropout=0.0,
        # 是否从能量预测器中停止梯度，默认为False
        stop_gradient_from_energy_predictor=False,
        # 音高预测器的层数，默认为5
        pitch_predictor_layers=5,
        # 音高预测器的通道数，默认为256
        pitch_predictor_channels=256,
        # 音高预测器的卷积核大小，默认为5
        pitch_predictor_kernel_size=5,
        # 音高预测器的丢弃率，默认为0.5
        pitch_predictor_dropout=0.5,
        # 音高嵌入的卷积核大小，默认为1
        pitch_embed_kernel_size=1,
        # 音高嵌入的丢弃率，默认为0.0
        pitch_embed_dropout=0.0,
        # 是否从音高预测器中停止梯度，默认为True
        stop_gradient_from_pitch_predictor=True,
        # 编码器的丢弃率，默认为0.2
        encoder_dropout_rate=0.2,
        # 编码器位置丢弃率，默认为0.2
        encoder_positional_dropout_rate=0.2,
        # 编码器注意力丢弃率，默认为0.2
        encoder_attention_dropout_rate=0.2,
        # 解码器的丢弃率，默认为0.2
        decoder_dropout_rate=0.2,
        # 解码器位置丢弃率，默认为0.2
        decoder_positional_dropout_rate=0.2,
        # 解码器注意力丢弃率，默认为0.2
        decoder_attention_dropout_rate=0.2,
        # 时长预测器的丢弃率，默认为0.2
        duration_predictor_dropout_rate=0.2,
        # 语音解码器后处理的丢弃率，默认为0.5
        speech_decoder_postnet_dropout=0.5,
        # 最大源位置，默认为5000
        max_source_positions=5000,
        # 是否使用遮蔽，默认为True
        use_masking=True,
        # 是否使用加权遮蔽，默认为False
        use_weighted_masking=False,
        # 说话者数量，默认为None
        num_speakers=None,
        # 语言数量，默认为None
        num_languages=None,
        # 说话者嵌入维度，默认为None
        speaker_embed_dim=None,
        # 是否为编码器-解码器模型，默认为True
        is_encoder_decoder=True,
        **kwargs,
# 定义 FastSpeech2ConformerHifiGanConfig 类，用于存储 FastSpeech2ConformerHifiGanModel 的配置
class FastSpeech2ConformerHifiGanConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FastSpeech2ConformerHifiGanModel`]. It is used to
    instantiate a FastSpeech2Conformer HiFi-GAN vocoder model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the
    FastSpeech2Conformer
    [espnet/fastspeech2_conformer_hifigan](https://huggingface.co/espnet/fastspeech2_conformer_hifigan) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        model_in_dim (`int`, *optional*, defaults to 80):
            The number of frequency bins in the input log-mel spectrogram.
        upsample_initial_channel (`int`, *optional*, defaults to 512):
            The number of input channels into the upsampling network.
        upsample_rates (`Tuple[int]` or `List[int]`, *optional*, defaults to `[8, 8, 2, 2]`):
            A tuple of integers defining the stride of each 1D convolutional layer in the upsampling network. The
            length of *upsample_rates* defines the number of convolutional layers and has to match the length of
            *upsample_kernel_sizes*.
        upsample_kernel_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[16, 16, 4, 4]`):
            A tuple of integers defining the kernel size of each 1D convolutional layer in the upsampling network. The
            length of *upsample_kernel_sizes* defines the number of convolutional layers and has to match the length of
            *upsample_rates*.
        resblock_kernel_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[3, 7, 11]`):
            A tuple of integers defining the kernel sizes of the 1D convolutional layers in the multi-receptive field
            fusion (MRF) module.
        resblock_dilation_sizes (`Tuple[Tuple[int]]` or `List[List[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`):
            A nested tuple of integers defining the dilation rates of the dilated 1D convolutional layers in the
            multi-receptive field fusion (MRF) module.
        initializer_range (`float`, *optional*, defaults to 0.01):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        leaky_relu_slope (`float`, *optional*, defaults to 0.1):
            The angle of the negative slope used by the leaky ReLU activation.
        normalize_before (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the spectrogram before vocoding using the vocoder's learned mean and variance.

    Example:

    ```python
    >>> from transformers import FastSpeech2ConformerHifiGan, FastSpeech2ConformerHifiGanConfig
    # 初始化一个 FastSpeech2ConformerHifiGan 配置
    configuration = FastSpeech2ConformerHifiGanConfig()

    # 根据配置初始化一个模型（具有随机权重）
    model = FastSpeech2ConformerHifiGan(configuration)

    # 访问模型配置
    configuration = model.config
    ```"""

    # 模型类型为"hifigan"
    model_type = "hifigan"

    def __init__(
        self,
        model_in_dim=80,
        upsample_initial_channel=512,
        upsample_rates=[8, 8, 2, 2],
        upsample_kernel_sizes=[16, 16, 4, 4],
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        initializer_range=0.01,
        leaky_relu_slope=0.1,
        normalize_before=True,
        **kwargs,
    ):
        # 定义各种模型超参数
        self.model_in_dim = model_in_dim
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.initializer_range = initializer_range
        self.leaky_relu_slope = leaky_relu_slope
        self.normalize_before = normalize_before
        # 调用父类的初始化方法
        super().__init__(**kwargs)
class FastSpeech2ConformerWithHifiGanConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`FastSpeech2ConformerWithHifiGan`]. It is used to
    instantiate a `FastSpeech2ConformerWithHifiGanModel` model according to the specified sub-models configurations,
    defining the model architecture.

    Instantiating a configuration with the defaults will yield a similar configuration to that of the
    FastSpeech2ConformerModel [espnet/fastspeech2_conformer](https://huggingface.co/espnet/fastspeech2_conformer) and
    FastSpeech2ConformerHifiGan
    [espnet/fastspeech2_conformer_hifigan](https://huggingface.co/espnet/fastspeech2_conformer_hifigan) architectures.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        model_config (`typing.Dict`, *optional*):
            Configuration of the text-to-speech model.
        vocoder_config (`typing.Dict`, *optional*):
            Configuration of the vocoder model.
    model_config ([`FastSpeech2ConformerConfig`], *optional*):
        Configuration of the text-to-speech model.
    vocoder_config ([`FastSpeech2ConformerHiFiGanConfig`], *optional*):
        Configuration of the vocoder model.

    Example:

    ```python
    >>> from transformers import (
    ...     FastSpeech2ConformerConfig,
    ...     FastSpeech2ConformerHifiGanConfig,
    ...     FastSpeech2ConformerWithHifiGanConfig,
    ...     FastSpeech2ConformerWithHifiGan,
    ... )

    >>> # Initializing FastSpeech2ConformerWithHifiGan sub-modules configurations.
    >>> model_config = FastSpeech2ConformerConfig()
    >>> vocoder_config = FastSpeech2ConformerHifiGanConfig()

    >>> # Initializing a FastSpeech2ConformerWithHifiGan module style configuration
    >>> configuration = FastSpeech2ConformerWithHifiGanConfig(model_config.to_dict(), vocoder_config.to_dict())

    >>> # Initializing a model (with random weights)
    >>> model = FastSpeech2ConformerWithHifiGan(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # 定义模型类型为 fastspeech2_conformer_with_hifigan
    model_type = "fastspeech2_conformer_with_hifigan"
    # 指示该类为组合配置类
    is_composition = True

    def __init__(
        self,
        model_config: Dict = None,
        vocoder_config: Dict = None,
        **kwargs,
    ):
        # 如果 model_config 为 None，则初始化为空字典，并记录日志
        if model_config is None:
            model_config = {}
            logger.info("model_config is None. initializing the model with default values.")

        # 如果 vocoder_config 为 None，则初始化为空字典，并记录日志
        if vocoder_config is None:
            vocoder_config = {}
            logger.info("vocoder_config is None. initializing the coarse model with default values.")

        # 使用 model_config 和 vocoder_config 初始化 FastSpeech2ConformerConfig 和 FastSpeech2ConformerHifiGanConfig
        self.model_config = FastSpeech2ConformerConfig(**model_config)
        self.vocoder_config = FastSpeech2ConformerHifiGanConfig(**vocoder_config)

        # 调用父类的初始化方法
        super().__init__(**kwargs)

```  
```