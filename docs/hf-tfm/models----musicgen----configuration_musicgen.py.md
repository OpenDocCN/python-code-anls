# `.\models\musicgen\configuration_musicgen.py`

```py
# coding=utf-8
# Copyright 2023 Meta AI and The HuggingFace Inc. team. All rights reserved.
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

""" MusicGen model configuration"""

# 导入所需模块和函数
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 预训练模型配置文件映射字典，将模型名称映射到其配置文件的URL
MUSICGEN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/musicgen-small": "https://huggingface.co/facebook/musicgen-small/resolve/main/config.json",
    # 查看所有Musicgen模型：https://huggingface.co/models?filter=musicgen
}

# MusicgenDecoderConfig类，继承自PretrainedConfig类
class MusicgenDecoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`MusicgenDecoder`]. It is used to instantiate a
    MusicGen decoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MusicGen
    [facebook/musicgen-small](https://huggingface.co/facebook/musicgen-small) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # 定义 MusicgenDecoder 模型的参数及其默认值
    Args:
        vocab_size (`int`, *optional*, defaults to 2048):
            MusicgenDecoder 模型的词汇表大小，定义了在调用 `MusicgenDecoder` 时输入 `inputs_ids` 可表示的不同标记数量。
        hidden_size (`int`, *optional*, defaults to 1024):
            层和池化层的维度。
        num_hidden_layers (`int`, *optional*, defaults to 24):
            解码器层的数量。
        num_attention_heads (`int`, *optional*, defaults to 16):
            Transformer 块中每个注意力层的注意力头数量。
        ffn_dim (`int`, *optional*, defaults to 4096):
            Transformer 块中“中间”（通常称为前馈）层的维度。
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            解码器和池化器中的非线性激活函数（函数或字符串）。支持的字符串包括 `"gelu"`, `"relu"`, `"silu"` 和 `"gelu_new"`。
        dropout (`float`, *optional*, defaults to 0.1):
            嵌入层、文本编码器和池化器中所有全连接层的 dropout 概率。
        attention_dropout (`float`, *optional*, defaults to 0.0):
            注意力概率的 dropout 比率。
        activation_dropout (`float`, *optional*, defaults to 0.0):
            全连接层内部激活的 dropout 比率。
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            模型可能使用的最大序列长度。通常设置为一个很大的值（例如 512、1024 或 2048）。
        initializer_factor (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态初始化器的标准差。
        layerdrop (`float`, *optional*, defaults to 0.0):
            解码器的 LayerDrop 概率。详细信息请参阅 LayerDrop 论文（见 https://arxiv.org/abs/1909.11556）。
        scale_embedding (`bool`, *optional*, defaults to `False`):
            是否通过 sqrt(hidden_size) 缩放嵌入。
        use_cache (`bool`, *optional*, defaults to `True`):
            模型是否应返回最后的 key/values 注意力（并非所有模型都使用）。
        num_codebooks (`int`, *optional*, defaults to 4):
            转发到模型的并行码书数量。
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            是否应绑定输入和输出词嵌入。
        audio_channels (`int`, *optional*, defaults to 1):
            音频数据中的通道数。单声道为 1，立体声为 2。立体声模型生成左/右输出通道的单独音频流，单声道模型生成单一音频流输出。
    # 定义模型类型为 "musicgen_decoder"
    model_type = "musicgen_decoder"
    
    # 在推断阶段忽略的键列表，这些键不会用于推断过程中
    keys_to_ignore_at_inference = ["past_key_values"]
    
    # 定义模型类，包含各种初始化参数
    def __init__(
        self,
        vocab_size=2048,  # 词汇表大小，默认为 2048
        max_position_embeddings=2048,  # 最大位置编码长度，默认为 2048
        num_hidden_layers=24,  # 隐藏层的数量，默认为 24
        ffn_dim=4096,  # FeedForward 层的维度，默认为 4096
        num_attention_heads=16,  # 注意力头的数量，默认为 16
        layerdrop=0.0,  # LayerDrop 参数，默认为 0.0
        use_cache=True,  # 是否使用缓存，默认为 True
        activation_function="gelu",  # 激活函数类型，默认为 "gelu"
        hidden_size=1024,  # 隐藏层大小，默认为 1024
        dropout=0.1,  # 全连接层和注意力层的 Dropout 概率，默认为 0.1
        attention_dropout=0.0,  # 注意力模型的 Dropout 概率，默认为 0.0
        activation_dropout=0.0,  # 激活函数的 Dropout 概率，默认为 0.0
        initializer_factor=0.02,  # 初始化因子，默认为 0.02
        scale_embedding=False,  # 是否缩放嵌入层，默认为 False；若为 True，则缩放因子为 sqrt(d_model)
        num_codebooks=4,  # 编码书的数量，默认为 4
        audio_channels=1,  # 音频通道数，默认为 1
        pad_token_id=2048,  # 填充标记的 ID，默认为 2048
        bos_token_id=2048,  # 起始标记的 ID，默认为 2048
        eos_token_id=None,  # 终止标记的 ID，默认为 None
        tie_word_embeddings=False,  # 是否绑定词嵌入，默认为 False
        **kwargs,  # 其他关键字参数
    ):
        # 初始化模型参数
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.ffn_dim = ffn_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.initializer_factor = initializer_factor
        self.layerdrop = layerdrop
        self.use_cache = use_cache
        self.scale_embedding = scale_embedding  # 若为 True，则嵌入层缩放因子为 sqrt(d_model)
        self.num_codebooks = num_codebooks
    
        # 检查音频通道数是否为合法值（1 或 2）
        if audio_channels not in [1, 2]:
            raise ValueError(f"Expected 1 (mono) or 2 (stereo) audio channels, got {audio_channels} channels.")
        self.audio_channels = audio_channels
    
        # 调用父类的初始化方法，设置特殊的标记 ID 和其他参数
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
class MusicgenConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MusicgenModel`]. It is used to instantiate a
    MusicGen model according to the specified arguments, defining the text encoder, audio encoder and MusicGen decoder
    configs.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
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
    ...     MusicgenConfig,
    ...     MusicgenDecoderConfig,
    ...     T5Config,
    ...     EncodecConfig,
    ...     MusicgenForConditionalGeneration,
    ... )

    >>> # Initializing text encoder, audio encoder, and decoder model configurations
    >>> text_encoder_config = T5Config()
    >>> audio_encoder_config = EncodecConfig()
    >>> decoder_config = MusicgenDecoderConfig()

    >>> configuration = MusicgenConfig.from_sub_models_config(
    ...     text_encoder_config, audio_encoder_config, decoder_config
    ... )

    >>> # Initializing a MusicgenForConditionalGeneration (with random weights) from the facebook/musicgen-small style configuration
    >>> model = MusicgenForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    >>> config_text_encoder = model.config.text_encoder
    >>> config_audio_encoder = model.config.audio_encoder
    >>> config_decoder = model.config.decoder

    >>> # Saving the model, including its configuration
    >>> model.save_pretrained("musicgen-model")

    >>> # loading model and config from pretrained folder
    >>> musicgen_config = MusicgenConfig.from_pretrained("musicgen-model")
    >>> model = MusicgenForConditionalGeneration.from_pretrained("musicgen-model", config=musicgen_config)
    ```

    Assigning the model_type class attribute for identification as a 'musicgen' model type.
    This attribute helps in distinguishing different model types in a system.
    """
    model_type = "musicgen"
    is_composition = True
    # 初始化方法，接受任意关键字参数
    def __init__(self, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 检查是否缺少必要的参数：text_encoder、audio_encoder、decoder
        if "text_encoder" not in kwargs or "audio_encoder" not in kwargs or "decoder" not in kwargs:
            # 如果缺少任一参数，则抛出数值错误异常
            raise ValueError("Config has to be initialized with text_encoder, audio_encoder and decoder config")

        # 从关键字参数中取出并移除 text_encoder 的配置
        text_encoder_config = kwargs.pop("text_encoder")
        # 从 text_encoder_config 中取出并移除 model_type 字段
        text_encoder_model_type = text_encoder_config.pop("model_type")

        # 从关键字参数中取出并移除 audio_encoder 的配置
        audio_encoder_config = kwargs.pop("audio_encoder")
        # 从 audio_encoder_config 中取出并移除 model_type 字段
        audio_encoder_model_type = audio_encoder_config.pop("model_type")

        # 从关键字参数中取出 decoder 的配置
        decoder_config = kwargs.pop("decoder")

        # 使用 text_encoder_model_type 和 text_encoder_config 创建文本编码器的配置
        self.text_encoder = AutoConfig.for_model(text_encoder_model_type, **text_encoder_config)
        # 使用 audio_encoder_model_type 和 audio_encoder_config 创建音频编码器的配置
        self.audio_encoder = AutoConfig.for_model(audio_encoder_model_type, **audio_encoder_config)
        # 使用 decoder_config 创建解码器的配置
        self.decoder = MusicgenDecoderConfig(**decoder_config)
        # 标记对象为编码器-解码器类型
        self.is_encoder_decoder = True

    @classmethod
    # 类方法：从子模型的配置创建 MusicgenConfig 对象
    def from_sub_models_config(
        cls,
        # 文本编码器的预训练配置对象
        text_encoder_config: PretrainedConfig,
        # 音频编码器的预训练配置对象
        audio_encoder_config: PretrainedConfig,
        # 解码器的音乐生成配置对象
        decoder_config: MusicgenDecoderConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`MusicgenConfig`] (or a derived class) from text encoder, audio encoder and decoder
        configurations.

        Returns:
            [`MusicgenConfig`]: An instance of a configuration object
        """
        # 使用传入的配置创建一个 MusicgenConfig 对象，并返回
        return cls(
            text_encoder=text_encoder_config.to_dict(),
            audio_encoder=audio_encoder_config.to_dict(),
            decoder=decoder_config.to_dict(),
            **kwargs,
        )

    @property
    # 属性方法：返回音频编码器的采样率
    # 这是一个属性方法，因为可能需要动态改变编解码器模型
    def sampling_rate(self):
        return self.audio_encoder.sampling_rate
```