# `.\models\encoder_decoder\configuration_encoder_decoder.py`

```py
# 设定编码为UTF-8
# 版权声明
# Apache License 2.0
# 导入相关模块
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义一个`EncoderDecoderConfig`类，继承自`PretrainedConfig`类
class EncoderDecoderConfig(PretrainedConfig):
    r"""
    [`EncoderDecoderConfig`] is the configuration class to store the configuration of a [`EncoderDecoderModel`]. It is
    used to instantiate an Encoder Decoder model according to the specified arguments, defining the encoder and decoder
    configs.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        kwargs (*optional*):
            Dictionary of keyword arguments. Notably:

                - **encoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that defines
                  the encoder config.
                - **decoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that defines
                  the decoder config.

    Examples:

    ```python
    >>> from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel

    >>> # Initializing a BERT bert-base-uncased style configuration
    >>> config_encoder = BertConfig()
    >>> config_decoder = BertConfig()

    >>> config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

    >>> # Initializing a Bert2Bert model (with random weights) from the bert-base-uncased style configurations
    >>> model = EncoderDecoderModel(config=config)

    >>> # Accessing the model configuration
    >>> config_encoder = model.config.encoder
    >>> config_decoder = model.config.decoder
    >>> # set decoder config to causal lm
    >>> config_decoder.is_decoder = True
    >>> config_decoder.add_cross_attention = True

    >>> # Saving the model, including its configuration
    >>> model.save_pretrained("my-model")

    >>> # loading model and config from pretrained folder
    >>> encoder_decoder_config = EncoderDecoderConfig.from_pretrained("my-model")
    >>> model = EncoderDecoderModel.from_pretrained("my-model", config=encoder_decoder_config)
    ```py"""

    # 类型描述信息
    model_type = "encoder-decoder"
    # 是否是合成的
    is_composition = True
    # 初始化方法，接收字典类型的可变关键字参数
    def __init__(self, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 确保关键字参数中包含 "encoder" 和 "decoder" 键
        assert (
            "encoder" in kwargs and "decoder" in kwargs
        ), "Config has to be initialized with encoder and decoder config"
        # 从关键字参数中提取编码器配置并移除该参数
        encoder_config = kwargs.pop("encoder")
        # 从编码器配置中提取模型类型并移除该参数
        encoder_model_type = encoder_config.pop("model_type")
        # 从关键字参数中提取解码器配置并移除该参数
        decoder_config = kwargs.pop("decoder")
        # 从解码器配置中提取模型类型并移除该参数
        decoder_model_type = decoder_config.pop("model_type")

        # 导入自动配置模块
        from ..auto.configuration_auto import AutoConfig

        # 根据编码器模型类型和配置创建编码器配置对象
        self.encoder = AutoConfig.for_model(encoder_model_type, **encoder_config)
        # 根据解码器模型类型和配置创建解码器配置对象
        self.decoder = AutoConfig.for_model(decoder_model_type, **decoder_config)
        # 标记为是编码器-解码器模式
        self.is_encoder_decoder = True

    # 类方法，接收编码器配置和解码器配置，返回一个新的配置实例
    @classmethod
    def from_encoder_decoder_configs(
        cls, encoder_config: PretrainedConfig, decoder_config: PretrainedConfig, **kwargs
    ) -> PretrainedConfig:
        # 说明该方法会实例化一个配置对象
        r"""
        Instantiate a [`EncoderDecoderConfig`] (or a derived class) from a pre-trained encoder model configuration and
        decoder model configuration.

        Returns:
            [`EncoderDecoderConfig`]: An instance of a configuration object
        """
        # 记录日志，指出对解码器配置的修改
        logger.info("Set `config.is_decoder=True` and `config.add_cross_attention=True` for decoder_config")
        # 将解码器配置的 `is_decoder` 标记设为 True
        decoder_config.is_decoder = True
        # 将解码器配置的 `add_cross_attention` 标记设为 True
        decoder_config.add_cross_attention = True

        # 返回一个新的配置实例，传入编码器和解码器配置的字典形式，以及其他参数
        return cls(encoder=encoder_config.to_dict(), decoder=decoder_config.to_dict(), **kwargs)
```