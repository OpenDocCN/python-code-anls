# `.\models\speech_encoder_decoder\configuration_speech_encoder_decoder.py`

```
# coding=utf-8
# 版权所有 2021 年 HuggingFace Inc. 团队.
# 版权所有 2018 年 NVIDIA 公司. 保留所有权利.
#
# 根据 Apache 许可证 2.0 版本 ("许可证") 进行许可;
# 您不得使用此文件，除非符合许可证的规定。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件按"原样"分发，
# 没有任何形式的明示或暗示担保或条件。
# 有关更多信息，请参阅许可证。

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig

# 获取日志记录器，以记录与当前模块相关的日志消息
logger = logging.get_logger(__name__)

# 继承自 PretrainedConfig 的配置类，用于存储 SpeechEncoderDecoderModel 的配置信息
class SpeechEncoderDecoderConfig(PretrainedConfig):
    r"""
    [`SpeechEncoderDecoderConfig`] 是用于存储 [`SpeechEncoderDecoderModel`] 配置的类。
    根据指定的参数实例化一个 Encoder-Decoder 模型，定义编码器和解码器的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    Args:
        kwargs (*可选*):
            关键字参数的字典。特别是:

                - **encoder** ([`PretrainedConfig`], *可选*) -- 定义编码器配置的配置对象实例。
                - **decoder** ([`PretrainedConfig`], *可选*) -- 定义解码器配置的配置对象实例。

    Examples:

    ```python
    >>> from transformers import BertConfig, Wav2Vec2Config, SpeechEncoderDecoderConfig, SpeechEncoderDecoderModel

    >>> # 初始化一个 Wav2Vec2 和 BERT 风格的配置
    >>> config_encoder = Wav2Vec2Config()
    >>> config_decoder = BertConfig()

    >>> config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

    >>> # 从 Wav2Vec2 和 google-bert/bert-base-uncased 风格的配置初始化一个 Wav2Vec2Bert 模型
    >>> model = SpeechEncoderDecoderModel(config=config)

    >>> # 访问模型配置
    >>> config_encoder = model.config.encoder
    >>> config_decoder = model.config.decoder
    >>> # 将解码器配置设置为 causal lm
    >>> config_decoder.is_decoder = True
    >>> config_decoder.add_cross_attention = True

    >>> # 保存模型，包括其配置
    >>> model.save_pretrained("my-model")

    >>> # 从预训练文件夹加载模型和配置
    >>> encoder_decoder_config = SpeechEncoderDecoderConfig.from_pretrained("my-model")
    >>> model = SpeechEncoderDecoderModel.from_pretrained("my-model", config=encoder_decoder_config)
    ```"""
    
    # 模型类型标识为 "speech-encoder-decoder"
    model_type = "speech-encoder-decoder"
    # 定义一个类变量 is_composition 并初始化为 True，表示此类是一个组合类
    is_composition = True

    # 构造函数，初始化对象时被调用
    def __init__(self, **kwargs):
        # 调用父类的构造函数，传入所有的关键字参数
        super().__init__(**kwargs)
        # 检查是否传入了 "encoder" 和 "decoder" 参数，如果没有则抛出 ValueError 异常
        if "encoder" not in kwargs or "decoder" not in kwargs:
            raise ValueError(
                f"A configuraton of type {self.model_type} cannot be instantiated because not both `encoder` and"
                f" `decoder` sub-configurations are passed, but only {kwargs}"
            )

        # 从 kwargs 中弹出 "encoder" 和 "decoder" 参数的配置，并获取它们的 model_type
        encoder_config = kwargs.pop("encoder")
        encoder_model_type = encoder_config.pop("model_type")
        decoder_config = kwargs.pop("decoder")
        decoder_model_type = decoder_config.pop("model_type")

        # 使用 AutoConfig 类为 encoder 和 decoder 创建配置对象
        self.encoder = AutoConfig.for_model(encoder_model_type, **encoder_config)
        self.decoder = AutoConfig.for_model(decoder_model_type, **decoder_config)
        # 设置对象的 is_encoder_decoder 属性为 True，表示这是一个 encoder-decoder 架构
        self.is_encoder_decoder = True

    @classmethod
    # 类方法，用于从预训练的 encoder 和 decoder 配置中实例化一个 SpeechEncoderDecoderConfig 对象
    def from_encoder_decoder_configs(
        cls, encoder_config: PretrainedConfig, decoder_config: PretrainedConfig, **kwargs
    ) -> PretrainedConfig:
        """
        从预训练的 encoder 和 decoder 配置实例化一个 [`SpeechEncoderDecoderConfig`] (或其派生类) 对象。

        Returns:
            [`SpeechEncoderDecoderConfig`]: 配置对象的一个实例
        """
        # 记录日志信息，设置 decoder_config 的 is_decoder=True 和 add_cross_attention=True
        logger.info("Setting `config.is_decoder=True` and `config.add_cross_attention=True` for decoder_config")
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True

        # 使用传入的 encoder_config 和 decoder_config 的字典形式，以及其他关键字参数，实例化一个类对象
        return cls(encoder=encoder_config.to_dict(), decoder=decoder_config.to_dict(), **kwargs)
```