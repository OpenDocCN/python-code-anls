# `.\models\speecht5\configuration_speecht5.py`

```
# 设置文件编码为 UTF-8
# 版权声明和许可信息，指明了代码的版权和使用许可
#
# 此处导入 functools 和 operator 模块
# configuration_utils 模块中的 PretrainedConfig 类导入
# logging 模块中的 get_logger 函数导入
import functools
import operator

from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取 logger 对象，用于记录日志
logger = logging.get_logger(__name__)

# 定义了不同预训练模型名称到其配置文件 URL 的映射
SPEECHT5_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/speecht5_asr": "https://huggingface.co/microsoft/speecht5_asr/resolve/main/config.json",
    "microsoft/speecht5_tts": "https://huggingface.co/microsoft/speecht5_tts/resolve/main/config.json",
    "microsoft/speecht5_vc": "https://huggingface.co/microsoft/speecht5_vc/resolve/main/config.json",
}

# 定义了 HiFi-GAN 预训练模型名称到其配置文件 URL 的映射
SPEECHT5_PRETRAINED_HIFIGAN_CONFIG_ARCHIVE_MAP = {
    "microsoft/speecht5_hifigan": "https://huggingface.co/microsoft/speecht5_hifigan/resolve/main/config.json",
}

# SpeechT5Config 类，继承自 PretrainedConfig，用于存储 SpeechT5 模型的配置信息
# 提供了 SpeechT5 模型的配置，可用于实例化 SpeechT5 模型，定义模型架构
# 默认配置与 microsoft/speecht5_asr 架构类似
# 配置对象继承自 PretrainedConfig，可用于控制模型的输出
# 详细信息请参阅 PretrainedConfig 的文档
class SpeechT5Config(PretrainedConfig):
    # 模型类型为 speecht5
    model_type = "speecht5"
    # 属性映射表，将 num_attention_heads 映射为 encoder_attention_heads，将 num_hidden_layers 映射为 encoder_layers
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "num_hidden_layers": "encoder_layers"}
    # 定义一个类的初始化方法，用于初始化模型的各种参数和配置
    def __init__(
        self,
        vocab_size=81,  # 词汇表大小，默认为81
        hidden_size=768,  # 隐藏层大小，默认为768
        encoder_layers=12,  # 编码器层数，默认为12
        encoder_attention_heads=12,  # 编码器注意力头数，默认为12
        encoder_ffn_dim=3072,  # 编码器前馈网络维度，默认为3072
        encoder_layerdrop=0.1,  # 编码器层丢弃率，默认为0.1
        decoder_layers=6,  # 解码器层数，默认为6
        decoder_ffn_dim=3072,  # 解码器前馈网络维度，默认为3072
        decoder_attention_heads=12,  # 解码器注意力头数，默认为12
        decoder_layerdrop=0.1,  # 解码器层丢弃率，默认为0.1
        hidden_act="gelu",  # 隐藏层激活函数，默认为GELU
        positional_dropout=0.1,  # 位置编码的dropout率，默认为0.1
        hidden_dropout=0.1,  # 隐藏层dropout率，默认为0.1
        attention_dropout=0.1,  # 注意力dropout率，默认为0.1
        activation_dropout=0.1,  # 激活函数dropout率，默认为0.1
        initializer_range=0.02,  # 初始化权重范围，默认为0.02
        layer_norm_eps=1e-5,  # LayerNorm层的epsilon，默认为1e-5
        scale_embedding=False,  # 是否对embedding进行缩放，默认为False
        feat_extract_norm="group",  # 特征提取的归一化方式，默认为"group"
        feat_proj_dropout=0.0,  # 特征投影的dropout率，默认为0.0
        feat_extract_activation="gelu",  # 特征提取的激活函数，默认为GELU
        conv_dim=(512, 512, 512, 512, 512, 512, 512),  # 卷积维度序列，默认为(512, 512, 512, 512, 512, 512, 512)
        conv_stride=(5, 2, 2, 2, 2, 2, 2),  # 卷积步幅序列，默认为(5, 2, 2, 2, 2, 2, 2)
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),  # 卷积核大小序列，默认为(10, 3, 3, 3, 3, 2, 2)
        conv_bias=False,  # 是否使用卷积层的偏置，默认为False
        num_conv_pos_embeddings=128,  # 卷积位置编码的数量，默认为128
        num_conv_pos_embedding_groups=16,  # 卷积位置编码的分组数量，默认为16
        apply_spec_augment=True,  # 是否应用频谱增强，默认为True
        mask_time_prob=0.05,  # 时间掩码的概率，默认为0.05
        mask_time_length=10,  # 时间掩码的长度，默认为10
        mask_time_min_masks=2,  # 时间掩码的最小掩码数，默认为2
        mask_feature_prob=0.0,  # 特征掩码的概率，默认为0.0
        mask_feature_length=10,  # 特征掩码的长度，默认为10
        mask_feature_min_masks=0,  # 特征掩码的最小掩码数，默认为0
        pad_token_id=1,  # 填充token的ID，默认为1
        bos_token_id=0,  # 起始token的ID，默认为0
        eos_token_id=2,  # 结束token的ID，默认为2
        decoder_start_token_id=2,  # 解码器起始token的ID，默认为2
        num_mel_bins=80,  # 梅尔频率的数量，默认为80
        speech_decoder_prenet_layers=2,  # 语音解码器预网络层数，默认为2
        speech_decoder_prenet_units=256,  # 语音解码器预网络单元数，默认为256
        speech_decoder_prenet_dropout=0.5,  # 语音解码器预网络的dropout率，默认为0.5
        speaker_embedding_dim=512,  # 发声者嵌入的维度，默认为512
        speech_decoder_postnet_layers=5,  # 语音解码器后网络层数，默认为5
        speech_decoder_postnet_units=256,  # 语音解码器后网络单元数，默认为256
        speech_decoder_postnet_kernel=5,  # 语音解码器后网络卷积核大小，默认为5
        speech_decoder_postnet_dropout=0.5,  # 语音解码器后网络的dropout率，默认为0.5
        reduction_factor=2,  # 缩减因子，默认为2
        max_speech_positions=4000,  # 最大语音位置，默认为4000
        max_text_positions=450,  # 最大文本位置，默认为450
        encoder_max_relative_position=160,  # 编码器最大相对位置，默认为160
        use_guided_attention_loss=True,  # 是否使用引导注意力损失，默认为True
        guided_attention_loss_num_heads=2,  # 引导注意力损失的头数，默认为2
        guided_attention_loss_sigma=0.4,  # 引导注意力损失的sigma，默认为0.4
        guided_attention_loss_scale=10.0,  # 引导注意力损失的缩放因子，默认为10.0
        use_cache=True,  # 是否使用缓存，默认为True
        is_encoder_decoder=True,  # 是否是编码器-解码器结构，默认为True
        **kwargs,  # 其他未命名参数
    ):
        # 定义一个方法，用于计算输入到logits的比例
        def inputs_to_logits_ratio(self):
            # 使用functools.reduce函数对卷积步幅序列中的所有值进行累乘，初始值为1
            return functools.reduce(operator.mul, self.conv_stride, 1)
# 定义一个配置类，用于存储 [`SpeechT5HifiGanModel`] 的配置信息。这个类被用来实例化一个 SpeechT5 HiFi-GAN 语音合成模型，
# 根据指定的参数来定义模型架构。
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
            The number of frequency bins in the input log-mel spectrogram.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the output audio will be generated, expressed in hertz (Hz).
        upsample_initial_channel (`int`, *optional*, defaults to 512):
            The number of input channels into the upsampling network.
        upsample_rates (`Tuple[int]` or `List[int]`, *optional*, defaults to `[4, 4, 4, 4]`):
            A tuple of integers defining the stride of each 1D convolutional layer in the upsampling network. The
            length of *upsample_rates* defines the number of convolutional layers and has to match the length of
            *upsample_kernel_sizes*.
        upsample_kernel_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[8, 8, 8, 8]`):
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
    # 定义模型类型为 "hifigan"
    model_type = "hifigan"

    # 初始化模型类，设置模型参数
    def __init__(
        self,
        model_in_dim=80,  # 输入维度，默认为80
        sampling_rate=16000,  # 采样率，默认为16000
        upsample_initial_channel=512,  # 上采样初始通道数，默认为512
        upsample_rates=[4, 4, 4, 4],  # 上采样倍率列表，默认为[4, 4, 4, 4]
        upsample_kernel_sizes=[8, 8, 8, 8],  # 上采样卷积核大小列表，默认为[8, 8, 8, 8]
        resblock_kernel_sizes=[3, 7, 11],  # ResBlock 的卷积核大小列表，默认为[3, 7, 11]
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],  # ResBlock 的膨胀率列表，默认为[[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        initializer_range=0.01,  # 初始化范围，默认为0.01
        leaky_relu_slope=0.1,  # LeakyReLU 的斜率，默认为0.1
        normalize_before=True,  # 是否在前归一化，默认为True
        **kwargs,  # 其他参数
    ):
        self.model_in_dim = model_in_dim  # 设置模型输入维度
        self.sampling_rate = sampling_rate  # 设置采样率
        self.upsample_initial_channel = upsample_initial_channel  # 设置上采样初始通道数
        self.upsample_rates = upsample_rates  # 设置上采样倍率列表
        self.upsample_kernel_sizes = upsample_kernel_sizes  # 设置上采样卷积核大小列表
        self.resblock_kernel_sizes = resblock_kernel_sizes  # 设置ResBlock的卷积核大小列表
        self.resblock_dilation_sizes = resblock_dilation_sizes  # 设置ResBlock的膨胀率列表
        self.initializer_range = initializer_range  # 设置初始化范围
        self.leaky_relu_slope = leaky_relu_slope  # 设置LeakyReLU的斜率
        self.normalize_before = normalize_before  # 设置是否在前归一化
        super().__init__(**kwargs)  # 调用父类的初始化方法，传入其他参数
```