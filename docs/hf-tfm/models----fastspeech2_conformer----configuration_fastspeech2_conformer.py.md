# `.\models\fastspeech2_conformer\configuration_fastspeech2_conformer.py`

```
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" FastSpeech2Conformer model configuration"""

from typing import Dict

from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取全局日志记录器对象
logger = logging.get_logger(__name__)

# 定义 FastSpeech2Conformer 预训练模型配置文件的存档映射字典
FASTSPEECH2_CONFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "espnet/fastspeech2_conformer": "https://huggingface.co/espnet/fastspeech2_conformer/raw/main/config.json",
}

# 定义包含 HiFi-GAN 的 FastSpeech2Conformer 预训练模型配置文件的存档映射字典
FASTSPEECH2_CONFORMER_HIFIGAN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "espnet/fastspeech2_conformer_hifigan": "https://huggingface.co/espnet/fastspeech2_conformer_hifigan/raw/main/config.json",
}

# 定义包含带 HiFi-GAN 的 FastSpeech2Conformer 预训练模型配置文件的存档映射字典
FASTSPEECH2_CONFORMER_WITH_HIFIGAN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "espnet/fastspeech2_conformer_with_hifigan": "https://huggingface.co/espnet/fastspeech2_conformer_with_hifigan/raw/main/config.json",
}

# 定义 FastSpeech2ConformerConfig 类，用于存储 FastSpeech2Conformer 模型的配置信息
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
    ```
    """

    # 定义模型类型为 "fastspeech2_conformer"
    model_type = "fastspeech2_conformer"
    # 定义属性映射，将配置参数映射到 FastSpeech2ConformerModel 的参数上
    attribute_map = {"num_hidden_layers": "encoder_layers", "num_attention_heads": "encoder_num_attention_heads"}
    # 定义一个初始化函数，用于初始化一个神经网络模型
    def __init__(
        self,
        hidden_size=384,  # 隐藏层大小，默认为384
        vocab_size=78,  # 词汇表大小，默认为78
        num_mel_bins=80,  # 梅尔频谱特征的频道数，默认为80
        encoder_num_attention_heads=2,  # 编码器注意力头数，默认为2
        encoder_layers=4,  # 编码器层数，默认为4
        encoder_linear_units=1536,  # 编码器线性单元数，默认为1536
        decoder_layers=4,  # 解码器层数，默认为4
        decoder_num_attention_heads=2,  # 解码器注意力头数，默认为2
        decoder_linear_units=1536,  # 解码器线性单元数，默认为1536
        speech_decoder_postnet_layers=5,  # 语音解码器后处理网络层数，默认为5
        speech_decoder_postnet_units=256,  # 语音解码器后处理网络单元数，默认为256
        speech_decoder_postnet_kernel=5,  # 语音解码器后处理网络核大小，默认为5
        positionwise_conv_kernel_size=3,  # 位置卷积核大小，默认为3
        encoder_normalize_before=False,  # 编码器归一化层在前，默认为False
        decoder_normalize_before=False,  # 解码器归一化层在前，默认为False
        encoder_concat_after=False,  # 编码器拼接后，默认为False
        decoder_concat_after=False,  # 解码器拼接后，默认为False
        reduction_factor=1,  # 缩减因子，默认为1
        speaking_speed=1.0,  # 说话速度，默认为1.0
        use_macaron_style_in_conformer=True,  # 在Conformer中使用Macaron风格，默认为True
        use_cnn_in_conformer=True,  # 在Conformer中使用CNN，默认为True
        encoder_kernel_size=7,  # 编码器卷积核大小，默认为7
        decoder_kernel_size=31,  # 解码器卷积核大小，默认为31
        duration_predictor_layers=2,  # 持续时间预测器层数，默认为2
        duration_predictor_channels=256,  # 持续时间预测器通道数，默认为256
        duration_predictor_kernel_size=3,  # 持续时间预测器卷积核大小，默认为3
        energy_predictor_layers=2,  # 能量预测器层数，默认为2
        energy_predictor_channels=256,  # 能量预测器通道数，默认为256
        energy_predictor_kernel_size=3,  # 能量预测器卷积核大小，默认为3
        energy_predictor_dropout=0.5,  # 能量预测器dropout率，默认为0.5
        energy_embed_kernel_size=1,  # 能量嵌入卷积核大小，默认为1
        energy_embed_dropout=0.0,  # 能量嵌入dropout率，默认为0.0
        stop_gradient_from_energy_predictor=False,  # 是否从能量预测器停止梯度，默认为False
        pitch_predictor_layers=5,  # 音高预测器层数，默认为5
        pitch_predictor_channels=256,  # 音高预测器通道数，默认为256
        pitch_predictor_kernel_size=5,  # 音高预测器卷积核大小，默认为5
        pitch_predictor_dropout=0.5,  # 音高预测器dropout率，默认为0.5
        pitch_embed_kernel_size=1,  # 音高嵌入卷积核大小，默认为1
        pitch_embed_dropout=0.0,  # 音高嵌入dropout率，默认为0.0
        stop_gradient_from_pitch_predictor=True,  # 是否从音高预测器停止梯度，默认为True
        encoder_dropout_rate=0.2,  # 编码器dropout率，默认为0.2
        encoder_positional_dropout_rate=0.2,  # 编码器位置dropout率，默认为0.2
        encoder_attention_dropout_rate=0.2,  # 编码器注意力dropout率，默认为0.2
        decoder_dropout_rate=0.2,  # 解码器dropout率，默认为0.2
        decoder_positional_dropout_rate=0.2,  # 解码器位置dropout率，默认为0.2
        decoder_attention_dropout_rate=0.2,  # 解码器注意力dropout率，默认为0.2
        duration_predictor_dropout_rate=0.2,  # 持续时间预测器dropout率，默认为0.2
        speech_decoder_postnet_dropout=0.5,  # 语音解码器后处理dropout率，默认为0.5
        max_source_positions=5000,  # 最大源位置数，默认为5000
        use_masking=True,  # 是否使用掩码，默认为True
        use_weighted_masking=False,  # 是否使用加权掩码，默认为False
        num_speakers=None,  # 说话者数量，默认为None
        num_languages=None,  # 语言数量，默认为None
        speaker_embed_dim=None,  # 说话者嵌入维度，默认为None
        is_encoder_decoder=True,  # 是否为编码器-解码器结构，默认为True
        **kwargs,  # 其他参数，以字典形式接收
`
class FastSpeech2ConformerHifiGanConfig(PretrainedConfig):
    # 定义 FastSpeech2ConformerHifiGanConfig 类，继承自 PretrainedConfig 类
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
    """
    # 初始化配置类，设置默认参数
    def __init__(self, model_in_dim=80, upsample_initial_channel=512, upsample_rates=[8, 8, 2, 2],
                 upsample_kernel_sizes=[16, 16, 4, 4], resblock_kernel_sizes=[3, 7, 11],
                 resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]], initializer_range=0.01,
                 leaky_relu_slope=0.1, normalize_before=True, **kwargs):
        # 调用父类构造函数初始化
        super().__init__(**kwargs)
        # 初始化参数
        self.model_in_dim = model_in_dim
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.initializer_range = initializer_range
        self.leaky_relu_slope = leaky_relu_slope
        self.normalize_before = normalize_before
    # 设置模型类型为 "hifigan"
    model_type = "hifigan"

    # 定义 FastSpeech2ConformerHifiGan 类，继承自父类
    def __init__(
        self,
        model_in_dim=80,  # 设置模型输入维度为 80
        upsample_initial_channel=512,  # 设置初始上采样通道数为 512
        upsample_rates=[8, 8, 2, 2],  # 设置上采样率数组
        upsample_kernel_sizes=[16, 16, 4, 4],  # 设置上采样卷积核大小数组
        resblock_kernel_sizes=[3, 7, 11],  # 设置残差块卷积核大小数组
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],  # 设置残差块膨胀卷积尺寸数组
        initializer_range=0.01,  # 设置参数初始化范围
        leaky_relu_slope=0.1,  # 设置 LeakyReLU 斜率
        normalize_before=True,  # 设置是否在归一化前执行操作
        **kwargs,  # 其他可变关键字参数
    ):
        self.model_in_dim = model_in_dim  # 初始化模型输入维度
        self.upsample_initial_channel = upsample_initial_channel  # 初始化初始上采样通道数
        self.upsample_rates = upsample_rates  # 初始化上采样率数组
        self.upsample_kernel_sizes = upsample_kernel_sizes  # 初始化上采样卷积核大小数组
        self.resblock_kernel_sizes = resblock_kernel_sizes  # 初始化残差块卷积核大小数组
        self.resblock_dilation_sizes = resblock_dilation_sizes  # 初始化残差块膨胀卷积尺寸数组
        self.initializer_range = initializer_range  # 初始化参数初始化范围
        self.leaky_relu_slope = leaky_relu_slope  # 初始化 LeakyReLU 斜率
        self.normalize_before = normalize_before  # 初始化归一化前操作标志
        super().__init__(**kwargs)  # 调用父类的初始化方法，传入其他关键字参数
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

    model_type = "fastspeech2_conformer_with_hifigan"
    is_composition = True

    def __init__(
        self,
        model_config: Dict = None,
        vocoder_config: Dict = None,
        **kwargs,
    ):
        # 如果 `model_config` 为 None，则使用默认值初始化文本转语音模型配置
        if model_config is None:
            model_config = {}
            logger.info("model_config is None. initializing the model with default values.")

        # 如果 `vocoder_config` 为 None，则使用默认值初始化声码器模型配置
        if vocoder_config is None:
            vocoder_config = {}
            logger.info("vocoder_config is None. initializing the coarse model with default values.")

        # 使用给定的 `model_config` 字典初始化 FastSpeech2ConformerConfig 对象
        self.model_config = FastSpeech2ConformerConfig(**model_config)
        # 使用给定的 `vocoder_config` 字典初始化 FastSpeech2ConformerHifiGanConfig 对象
        self.vocoder_config = FastSpeech2ConformerHifiGanConfig(**vocoder_config)

        # 调用父类的构造函数，传递额外的关键字参数
        super().__init__(**kwargs)
```