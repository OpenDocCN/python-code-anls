# `.\transformers\models\speech_encoder_decoder\configuration_speech_encoder_decoder.py`

```
# 引入必要的库和模块
from ...configuration_utils import PretrainedConfig  # 从配置工具中导入预训练配置类
from ...utils import logging  # 从工具集中导入日志模块
from ..auto.configuration_auto import AutoConfig  # 从自动配置模块中导入自动配置类

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义语音编码器解码器配置类，继承自预训练配置类
class SpeechEncoderDecoderConfig(PretrainedConfig):
    r"""
    [`SpeechEncoderDecoderConfig`] 是用于存储 [`SpeechEncoderDecoderModel`] 的配置类。
    该类用于根据指定参数实例化一个编码器-解码器模型，定义了编码器和解码器的配置。

    配置对象继承自 [`PretrainedConfig`] 并可用于控制模型的输出。
    阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    Args:
        kwargs (*optional*):
            字典类型的关键字参数。特别是:

                - **encoder** ([`PretrainedConfig`], *optional*) -- 定义编码器配置的配置对象实例。
                - **decoder** ([`PretrainedConfig`], *optional*) -- 定义解码器配置的配置对象实例。

    Examples:

    ```python
    >>> from transformers import BertConfig, Wav2Vec2Config, SpeechEncoderDecoderConfig, SpeechEncoderDecoderModel

    >>> # 初始化一个 Wav2Vec2 & BERT 风格的配置
    >>> config_encoder = Wav2Vec2Config()
    >>> config_decoder = BertConfig()

    >>> config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

    >>> # 从 Wav2Vec2 & bert-base-uncased 风格的配置初始化一个 Wav2Vec2Bert 模型
    >>> model = SpeechEncoderDecoderModel(config=config)

    >>> # 访问模型配置
    >>> config_encoder = model.config.encoder
    >>> config_decoder = model.config.decoder
    >>> # 将解码器配置设置为因果语言模型
    >>> config_decoder.is_decoder = True
    >>> config_decoder.add_cross_attention = True

    >>> # 保存模型，包括其配置
    >>> model.save_pretrained("my-model")

    >>> # 从预训练文件夹加载模型和配置
    >>> encoder_decoder_config = SpeechEncoderDecoderConfig.from_pretrained("my-model")
    >>> model = SpeechEncoderDecoderModel.from_pretrained("my-model", config=encoder_decoder_config)
    ```"""

    # 模型类型
    model_type = "speech-encoder-decoder"
```  
    # 设置标志表示这是一个组合模型
    is_composition = True

    # 初始化方法，接受关键字参数
    def __init__(self, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 检查是否传入了编码器和解码器配置
        if "encoder" not in kwargs or "decoder" not in kwargs:
            # 如果没有，则抛出值错误异常
            raise ValueError(
                f"A configuraton of type {self.model_type} cannot be instantiated because not both `encoder` and"
                f" `decoder` sub-configurations are passed, but only {kwargs}"
            )

        # 从关键字参数中取出编码器和解码器的配置
        encoder_config = kwargs.pop("encoder")
        encoder_model_type = encoder_config.pop("model_type")
        decoder_config = kwargs.pop("decoder")
        decoder_model_type = decoder_config.pop("model_type")

        # 使用编码器和解码器的配置初始化编码器和解码器对象
        self.encoder = AutoConfig.for_model(encoder_model_type, **encoder_config)
        self.decoder = AutoConfig.for_model(decoder_model_type, **decoder_config)
        # 设置标志表示这是一个编码器-解码器模型
        self.is_encoder_decoder = True

    # 从编码器和解码器的配置实例化模型配置对象的类方法
    @classmethod
    def from_encoder_decoder_configs(
        cls, encoder_config: PretrainedConfig, decoder_config: PretrainedConfig, **kwargs
    ) -> PretrainedConfig:
        r"""
        从预训练编码器模型配置和解码器模型配置实例化一个 [`SpeechEncoderDecoderConfig`]（或派生类）。

        返回：
            [`SpeechEncoderDecoderConfig`]：配置对象的实例
        """
        # 记录日志，设置解码器配置的一些属性
        logger.info("Setting `config.is_decoder=True` and `config.add_cross_attention=True` for decoder_config")
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True

        # 通过编码器和解码器配置实例化模型配置对象
        return cls(encoder=encoder_config.to_dict(), decoder=decoder_config.to_dict(), **kwargs)
```