# `.\models\musicgen_melody\configuration_musicgen_melody.py`

```py
# coding=utf-8
# Copyright 2024 Meta AI and The HuggingFace Inc. team. All rights reserved.
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
""" Musicgen Melody model configuration"""

# 导入所需的模块和函数
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig

# 获取logger对象用于记录日志
logger = logging.get_logger(__name__)

# 预训练模型配置文件映射，指定预训练模型名称及其配置文件URL
MUSICGEN_MELODY_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/musicgen-melody": "https://huggingface.co/facebook/musicgen-melody/resolve/main/config.json",
}

# MusicgenMelodyDecoderConfig类，继承自PretrainedConfig，用于存储Musicgen Melody解码器的配置信息
class MusicgenMelodyDecoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`MusicgenMelodyDecoder`]. It is used to instantiate a
    Musicgen Melody decoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Musicgen Melody
    [facebook/musicgen-melody](https://huggingface.co/facebook/musicgen-melody) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """

    # 模型类型，用于标识模型配置类别
    model_type = "musicgen_melody_decoder"
    # 推断过程中忽略的关键字列表
    keys_to_ignore_at_inference = ["past_key_values"]

    # 初始化方法，定义了多个模型配置参数
    def __init__(
        self,
        vocab_size=2048,
        max_position_embeddings=2048,
        num_hidden_layers=24,
        ffn_dim=4096,
        num_attention_heads=16,
        layerdrop=0.0,
        use_cache=True,
        activation_function="gelu",
        hidden_size=1024,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        initializer_factor=0.02,
        scale_embedding=False,
        num_codebooks=4,
        audio_channels=1,
        pad_token_id=2048,
        bos_token_id=2048,
        eos_token_id=None,
        tie_word_embeddings=False,
        **kwargs,
    ):
        # 调用父类的初始化方法，传递参数给父类PretrainedConfig
        super().__init__(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            num_hidden_layers=num_hidden_layers,
            ffn_dim=ffn_dim,
            num_attention_heads=num_attention_heads,
            layerdrop=layerdrop,
            use_cache=use_cache,
            activation_function=activation_function,
            hidden_size=hidden_size,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            initializer_factor=initializer_factor,
            scale_embedding=scale_embedding,
            num_codebooks=num_codebooks,
            audio_channels=audio_channels,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        # 初始化BERT模型的配置参数
        self.vocab_size = vocab_size  # 词汇表大小
        self.max_position_embeddings = max_position_embeddings  # 最大位置嵌入长度
        self.hidden_size = hidden_size  # 隐藏层大小
        self.ffn_dim = ffn_dim  # 前馈神经网络维度
        self.num_hidden_layers = num_hidden_layers  # 隐藏层数量
        self.num_attention_heads = num_attention_heads  # 注意力头的数量
        self.dropout = dropout  # 普通的dropout概率
        self.attention_dropout = attention_dropout  # 注意力机制中的dropout概率
        self.activation_dropout = activation_dropout  # 激活函数中的dropout概率
        self.activation_function = activation_function  # 激活函数类型
        self.initializer_factor = initializer_factor  # 初始化因子
        self.layerdrop = layerdrop  # 层级dropout比例
        self.use_cache = use_cache  # 是否使用缓存
        self.scale_embedding = scale_embedding  # 如果为True，则嵌入的缩放因子将是sqrt(d_model)
        self.num_codebooks = num_codebooks  # 编码书的数量

        # 检查音频通道数是否为1（单声道）或2（立体声）
        if audio_channels not in [1, 2]:
            raise ValueError(f"Expected 1 (mono) or 2 (stereo) audio channels, got {audio_channels} channels.")
        self.audio_channels = audio_channels  # 音频通道数

        # 调用父类构造函数，传递额外参数和BERT模型的特定参数
        super().__init__(
            pad_token_id=pad_token_id,  # 填充标记ID
            bos_token_id=bos_token_id,  # 起始标记ID
            eos_token_id=eos_token_id,  # 结束标记ID
            tie_word_embeddings=tie_word_embeddings,  # 是否绑定词嵌入
            **kwargs,  # 其他未命名参数
        )
# 定义 MusicgenMelodyConfig 类，继承自 PretrainedConfig 类，用于存储 Musicgen Melody 模型的配置信息
class MusicgenMelodyConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MusicgenMelodyModel`]. It is used to instantiate a
    Musicgen Melody model according to the specified arguments, defining the text encoder, audio encoder and Musicgen Melody decoder
    configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the Musicgen Melody
    [facebook/musicgen-melody](https://huggingface.co/facebook/musicgen-melody) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_chroma (`int`, *optional*, defaults to 12): Number of chroma bins to use.
        chroma_length (`int`, *optional*, defaults to 235):
            Maximum chroma duration if audio is used to condition the model. Corresponds to the maximum duration used during training.
        kwargs (*optional*):
            Dictionary of keyword arguments. Notably:

                - **text_encoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that
                  defines the text encoder config.
                - **audio_encoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that
                  defines the audio encoder config.
                - **decoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that defines
                  the decoder config.

    Example:

    ```
    >>> from transformers import (
    ...     MusicgenMelodyConfig,
    ...     MusicgenMelodyDecoderConfig,
    ...     T5Config,
    ...     EncodecConfig,
    ...     MusicgenMelodyForConditionalGeneration,
    ... )

    >>> # Initializing text encoder, audio encoder, and decoder model configurations
    >>> text_encoder_config = T5Config()
    >>> audio_encoder_config = EncodecConfig()
    >>> decoder_config = MusicgenMelodyDecoderConfig()

    >>> configuration = MusicgenMelodyConfig.from_sub_models_config(
    ...     text_encoder_config, audio_encoder_config, decoder_config
    ... )

    >>> # Initializing a MusicgenMelodyForConditionalGeneration (with random weights) from the facebook/musicgen-melody style configuration
    >>> model = MusicgenMelodyForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    >>> config_text_encoder = model.config.text_encoder
    >>> config_audio_encoder = model.config.audio_encoder
    >>> config_decoder = model.config.decoder

    >>> # Saving the model, including its configuration
    >>> model.save_pretrained("musicgen_melody-model")

    >>> # loading model and config from pretrained folder
    >>> musicgen_melody_config = MusicgenMelodyConfig.from_pretrained("musicgen_melody-model")

    ```
    # 使用预训练模型名称和配置创建音乐生成模型对象
    model = MusicgenMelodyForConditionalGeneration.from_pretrained("musicgen_melody-model", config=musicgen_melody_config)



    # 设置模型类型为音乐生成旋律
    model_type = "musicgen_melody"
    # 标记此模型为一个生成作品
    is_composition = True



    def __init__(
        self,
        num_chroma=12,
        chroma_length=235,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # 检查是否初始化了 text_encoder、audio_encoder 和 decoder 配置，否则抛出值错误
        if "text_encoder" not in kwargs or "audio_encoder" not in kwargs or "decoder" not in kwargs:
            raise ValueError("Config has to be initialized with text_encoder, audio_encoder and decoder config")
        
        # 弹出并初始化文本编码器配置
        text_encoder_config = kwargs.pop("text_encoder")
        text_encoder_model_type = text_encoder_config.pop("model_type")

        # 弹出并初始化音频编码器配置
        audio_encoder_config = kwargs.pop("audio_encoder")
        audio_encoder_model_type = audio_encoder_config.pop("model_type")

        # 弹出并初始化解码器配置
        decoder_config = kwargs.pop("decoder")

        # 使用 AutoConfig 根据模型类型和配置初始化文本编码器、音频编码器和解码器
        self.text_encoder = AutoConfig.for_model(text_encoder_model_type, **text_encoder_config)
        self.audio_encoder = AutoConfig.for_model(audio_encoder_model_type, **audio_encoder_config)
        self.decoder = MusicgenMelodyDecoderConfig(**decoder_config)
        self.is_encoder_decoder = False

        # 设置音调数量和音调长度
        self.num_chroma = num_chroma
        self.chroma_length = chroma_length



    @classmethod
    def from_sub_models_config(
        cls,
        text_encoder_config: PretrainedConfig,
        audio_encoder_config: PretrainedConfig,
        decoder_config: MusicgenMelodyDecoderConfig,
        **kwargs,
    ):
        r"""
        从文本编码器、音频编码器和解码器配置实例化一个 MusicgenMelodyConfig（或其派生类）。

        Returns:
            [`MusicgenMelodyConfig`]: 配置对象的一个实例
        """

        # 使用给定的配置实例化当前类的对象
        return cls(
            text_encoder=text_encoder_config.to_dict(),
            audio_encoder=audio_encoder_config.to_dict(),
            decoder=decoder_config.to_dict(),
            **kwargs,
        )



    @property
    # 这是一个属性，因为您可能想要动态更改编解码器模型
    def sampling_rate(self):
        # 返回音频编码器的采样率
        return self.audio_encoder.sampling_rate
```