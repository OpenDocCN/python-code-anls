# `.\models\encoder_decoder\configuration_encoder_decoder.py`

```py
# 设置文件编码为UTF-8
# 版权声明：2020年由HuggingFace Inc.团队版权所有。
# 版权声明：2018年，NVIDIA CORPORATION版权所有。
#
# 根据Apache许可证2.0版（“许可证”）授权，除非符合许可证规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按“原样”分发，不提供任何明示或暗示的担保或条件。
# 有关详细信息，请参阅许可证。

# 导入必要的模块
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取名为__name__的当前日志记录器
logger = logging.get_logger(__name__)

# 定义EncoderDecoderConfig类，继承自PretrainedConfig
class EncoderDecoderConfig(PretrainedConfig):
    r"""
    [`EncoderDecoderConfig`]是用于存储[`EncoderDecoderModel`]配置的配置类。它用于根据指定的参数实例化编码器和解码器模型。

    配置对象继承自[`PretrainedConfig`]，可用于控制模型输出。有关更多信息，请阅读[`PretrainedConfig`]的文档。

    Args:
        kwargs (*可选参数*):
            关键字参数的字典。特别是:

                - **encoder** ([`PretrainedConfig`]，*可选*) -- 定义编码器配置的配置对象实例。
                - **decoder** ([`PretrainedConfig`]，*可选*) -- 定义解码器配置的配置对象实例。

    Examples:

    ```
    >>> from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel

    >>> # 初始化一个Bert google-bert/bert-base-uncased风格的配置
    >>> config_encoder = BertConfig()
    >>> config_decoder = BertConfig()

    >>> config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

    >>> # 初始化一个Bert2Bert模型（带有随机权重），从google-bert/bert-base-uncased风格的配置开始
    >>> model = EncoderDecoderModel(config=config)

    >>> # 访问模型配置
    >>> config_encoder = model.config.encoder
    >>> config_decoder = model.config.decoder
    >>> # 将解码器配置设置为因果语言模型
    >>> config_decoder.is_decoder = True
    >>> config_decoder.add_cross_attention = True

    >>> # 保存模型，包括其配置
    >>> model.save_pretrained("my-model")

    >>> # 从预训练文件夹加载模型和配置
    >>> encoder_decoder_config = EncoderDecoderConfig.from_pretrained("my-model")
    >>> model = EncoderDecoderModel.from_pretrained("my-model", config=encoder_decoder_config)
    ```"""
    
    # 模型类型为“encoder-decoder”
    model_type = "encoder-decoder"
    # 是复合对象
    is_composition = True
    # 初始化方法，继承自父类并接收关键字参数
    def __init__(self, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 断言确保参数中包含 "encoder" 和 "decoder"，否则抛出异常
        assert (
            "encoder" in kwargs and "decoder" in kwargs
        ), "Config has to be initialized with encoder and decoder config"
        # 从参数中弹出 "encoder" 和 "decoder" 的配置信息
        encoder_config = kwargs.pop("encoder")
        # 获取编码器模型类型并弹出其配置信息
        encoder_model_type = encoder_config.pop("model_type")
        # 获取解码器配置信息并弹出其模型类型
        decoder_config = kwargs.pop("decoder")
        decoder_model_type = decoder_config.pop("model_type")

        # 导入自动配置模块
        from ..auto.configuration_auto import AutoConfig

        # 使用自动配置模块为编码器创建配置对象
        self.encoder = AutoConfig.for_model(encoder_model_type, **encoder_config)
        # 使用自动配置模块为解码器创建配置对象
        self.decoder = AutoConfig.for_model(decoder_model_type, **decoder_config)
        # 设置标志，表明这是一个编码器-解码器结构
        self.is_encoder_decoder = True

    # 类方法：根据预训练的编码器和解码器配置实例化一个编码器-解码器配置对象
    @classmethod
    def from_encoder_decoder_configs(
        cls, encoder_config: PretrainedConfig, decoder_config: PretrainedConfig, **kwargs
    ) -> PretrainedConfig:
        r"""
        Instantiate a [`EncoderDecoderConfig`] (or a derived class) from a pre-trained encoder model configuration and
        decoder model configuration.

        Returns:
            [`EncoderDecoderConfig`]: An instance of a configuration object
        """
        # 记录信息：为解码器配置设置 `is_decoder=True` 和 `add_cross_attention=True`
        logger.info("Set `config.is_decoder=True` and `config.add_cross_attention=True` for decoder_config")
        # 设置解码器配置为解码器类型，并启用交叉注意力机制
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True

        # 使用当前类构造函数创建一个编码器-解码器配置对象，并返回
        return cls(encoder=encoder_config.to_dict(), decoder=decoder_config.to_dict(), **kwargs)
```