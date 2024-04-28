# `.\transformers\models\musicgen\configuration_musicgen.py`

```py
# 设置文件编码为UTF-8
# 版权所有 2023 年 Meta AI 和 HuggingFace 公司团队。保留所有权利。
#
# 根据 Apache 许可证，第 2 版（“许可证”），您只有在遵守该许可证的情况下才能使用此文件。
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律要求或书面同意，否则根据许可证分发的软件是“按原样”
# 的基础上分发，不附带任何担保或条件，无论是明示的还是隐含的。
# 请参阅许可证以查看特定语言规定的权限和限制。
""" MusicGen 模型配置"""

# 从模块中导入相关内容
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# MusicGen 预训练配置和相应归档映射
MUSICGEN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/musicgen-small": "https://huggingface.co/facebook/musicgen-small/resolve/main/config.json",
    # 查看所有 MusicGen 模型，请访问 https://huggingface.co/models?filter=musicgen
}

# MusicGen 解码器配置类
class MusicgenDecoderConfig(PretrainedConfig):
    r"""
    这是用于存储[`MusicgenDecoder`]配置的配置类。用于根据指定参数实例化 MusicGen 解码器，
    定义模型架构。使用默认值实例化配置将产生类似于 MusicGen [facebook/musicgen-small](https://huggingface.co/facebook/musicgen-small)
    架构的配置。

    配置对象继承自[`PretrainedConfig`]，用于控制模型输出。有关更多信息，请阅读[`PretrainedConfig`]的文档。
    Args:
        vocab_size (`int`, *optional*, defaults to 2048):
            MusicgenDecoder 模型的词汇表大小。定义在调用 [`MusicgenDecoder`] 时可以表示的不同标记数量。
        hidden_size (`int`, *optional*, defaults to 1024):
            层和汇聚层的维度。
        num_hidden_layers (`int`, *optional*, defaults to 24):
            解码器层数。
        num_attention_heads (`int`, *optional*, defaults to 16):
            每个Transformer块中的注意力层的注意头数。
        ffn_dim (`int`, *optional*, defaults to 4096):
            Transformer 块中 "中间"（通常命名为feed-forward）层的维度。
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            解码器和汇聚层中非线性激活函数（函数或字符串）。支持字符串 `"gelu"`, `"relu"`, `"silu"` 和 `"gelu_new"`。
        dropout (`float`, *optional*, defaults to 0.1):
            用于所有嵌入层、文本编码器和汇聚层中所有全连接层的丢失概率。
        attention_dropout (`float`, *optional*, defaults to 0.0):
            注意力概率的丢失比率。
        activation_dropout (`float`, *optional*, defaults to 0.0):
            全连接层内激活的丢失比率。
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            模型可能使用的最大序列长度。通常设置为较大的值（例如 512、1024 或 2048）以防万一。
        initializer_factor (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态初始化器的标准差。
        layerdrop (`float`, *optional*, defaults to 0.0):
            解码器的 LayerDrop 概率。有关更多详细信息，请参阅 [LayerDrop paper](see https://arxiv.org/abs/1909.11556)。
        scale_embedding (`bool`, *optional*, defaults to `False`):
            通过除以 sqrt(hidden_size) 来缩放嵌入。
        use_cache (`bool`, *optional*, defaults to `True`):
            模型是否应返回最后一个键/值注意力（并非所有模型都使用）。
        num_codebooks (`int`, *optional*, defaults to 4):
            并行转发到模型的码书数量。
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            输入和输出单词嵌入是否应绑定。
        audio_channels (`int`, *optional*, defaults to 1):
            音频数据中的通道数。单声道为1，立体声为2。立体声模型为左/右输出通道生成单独的音频流。单声道模型生成单个音频流输出。
    ```
    # 设置模型类型为音乐生成解码器
    model_type = "musicgen_decoder"
    # 在推断阶段忽略的关键字列表
    keys_to_ignore_at_inference = ["past_key_values"]
    
    # 初始化函数，设置模型参数
    def __init__(
        self,
        vocab_size=2048,  # 词汇量大小
        max_position_embeddings=2048,  # 最大位置编码数
        num_hidden_layers=24,  # 隐藏层的数量
        ffn_dim=4096,  # 前馈神经网络的维度
        num_attention_heads=16,  # 注意力头的数量
        layerdrop=0.0,  # 层丢弃率
        use_cache=True,  # 是否使用缓存
        activation_function="gelu",  # 激活函数
        hidden_size=1024,  # 隐藏层的尺寸
        dropout=0.1,  # 丢弃率
        attention_dropout=0.0,  # 注意力模块的丢弃率
        activation_dropout=0.0,  # 激活函数的丢弃率
        initializer_factor=0.02,  # 初始化因子
        scale_embedding=False,  # 是否缩放嵌入，如果为True，则缩放因子为sqrt(d_model)
        num_codebooks=4,  # 量化编码书的数量
        audio_channels=1,  # 音频通道数
        pad_token_id=2048,  # 填充标记 ID
        bos_token_id=2048,  # 序列开始标记 ID
        eos_token_id=None,  # 序列结束标记 ID
        tie_word_embeddings=False,  # 是否绑定词嵌入
        **kwargs,  # 其他参数
    ):
        # 设置模型参数
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
        self.scale_embedding = scale_embedding  # 如果为True，则缩放因子为sqrt(d_model)
        self.num_codebooks = num_codebooks
    
        # 如果音频通道数不是1或2，则引发值错误
        if audio_channels not in [1, 2]:
            raise ValueError(f"Expected 1 (mono) or 2 (stereo) audio channels, got {audio_channels} channels.")
        self.audio_channels = audio_channels
    
        # 调用父类的初始化函数，并传递参数
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
# 定义 Musicgen 模型的配置类，用于存储 MusicgenModel 的配置信息，包括文本编码器、音频编码器和 MusicGen 解码器的配置。
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

    ```py
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
    ```"""

    # 模型类型为 "musicgen"
    model_type = "musicgen"
    # 标记该模型是进行组合的
    is_composition = True
    # 初始化方法，接受任意关键字参数
    def __init__(self, **kwargs):
        # 调用父类初始化方法
        super().__init__(**kwargs)
        # 检查是否传入了文本编码器、音频编码器和解码器配置信息，若没有则抛出数值错误
        if "text_encoder" not in kwargs or "audio_encoder" not in kwargs or "decoder" not in kwargs:
            raise ValueError("Config has to be initialized with text_encoder, audio_encoder and decoder config")

        # 弹出文本编码器配置信息
        text_encoder_config = kwargs.pop("text_encoder")
        # 弹出文本编码器模型类型
        text_encoder_model_type = text_encoder_config.pop("model_type")

        # 弹出音频编码器配置信息
        audio_encoder_config = kwargs.pop("audio_encoder")
        # 弹出音频编码器模型类型
        audio_encoder_model_type = audio_encoder_config.pop("model_type")

        # 弹出解码器配置信息
        decoder_config = kwargs.pop("decoder")

        # 使用文本编码器模型类型和配置信息创建文本编码器对象
        self.text_encoder = AutoConfig.for_model(text_encoder_model_type, **text_encoder_config)
        # 使用音频编码器模型类型和配置信息创建音频编码器对象
        self.audio_encoder = AutoConfig.for_model(audio_encoder_model_type, **audio_encoder_config)
        # 使用解码器配置信息创建解码器对象
        self.decoder = MusicgenDecoderConfig(**decoder_config)
        # 标记为编码器-解码器结构
        self.is_encoder_decoder = True

    @classmethod
    # 从子模型配置信息实例化一个 MusicgenConfig 对象
    def from_sub_models_config(
        cls,
        text_encoder_config: PretrainedConfig,
        audio_encoder_config: PretrainedConfig,
        decoder_config: MusicgenDecoderConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`MusicgenConfig`] (or a derived class) from text encoder, audio encoder and decoder
        configurations.

        Returns:
            [`MusicgenConfig`]: An instance of a configuration object
        """
        # 返回一个基于文本编码器、音频编码器和解码器配置信息的 MusicgenConfig 对象
        return cls(
            text_encoder=text_encoder_config.to_dict(),
            audio_encoder=audio_encoder_config.to_dict(),
            decoder=decoder_config.to_dict(),
            **kwargs,
        )

    @property
    # 这是一个属性，因为您可能想要动态更改编解码器模型
    def sampling_rate(self):
        # 返回音频编码器的采样率属性
        return self.audio_encoder.sampling_rate
```