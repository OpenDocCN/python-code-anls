# `.\models\led\configuration_led.py`

```py
# 设置文件编码为 UTF-8
# 版权声明和许可协议
# 2021 年版权归 Iz Beltagy、Matthew E. Peters、Arman Cohan 和 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版（"许可证"）授权
# 除非适用法律要求或书面同意，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 本软件按原样提供，不提供任何形式的保证或条件，明示或暗示
# 有关详细信息，请参阅许可证
"""LED 模型配置"""

from typing import List, Union  # 导入类型提示所需的模块

from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...utils import logging  # 导入日志模块

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练模型配置映射字典，将模型名称映射到对应的配置文件 URL
LED_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "allenai/led-base-16384": "https://huggingface.co/allenai/led-base-16384/resolve/main/config.json",
    # 查看所有 LED 模型 https://huggingface.co/models?filter=led
}

# LED 模型配置类
class LEDConfig(PretrainedConfig):
    r"""
    这是用于存储 [`LEDModel`] 配置的配置类。根据指定的参数实例化一个 LED 模型，定义模型架构。
    使用默认值实例化配置将产生与 LED [allenai/led-base-16384](https://huggingface.co/allenai/led-base-16384) 架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档获取更多信息。

    示例:

    ```python
    >>> from transformers import LEDModel, LEDConfig

    >>> # 初始化一个 LED allenai/led-base-16384 风格的配置
    >>> configuration = LEDConfig()

    >>> # 根据 allenai/led-base-16384 风格的配置初始化一个模型
    >>> model = LEDModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py"""

    model_type = "led"  # 模型类型为 LED
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",  # 属性映射：num_attention_heads -> encoder_attention_heads
        "hidden_size": "d_model",  # 属性映射：hidden_size -> d_model
        "attention_probs_dropout_prob": "attention_dropout",  # 属性映射：attention_probs_dropout_prob -> attention_dropout
        "initializer_range": "init_std",  # 属性映射：initializer_range -> init_std
    }
    # 初始化函数，设置各种参数的默认数值
    def __init__(
        self,
        vocab_size=50265,  # 词汇表大小，默认为50265
        max_encoder_position_embeddings=16384,  # 编码器位置嵌入的最大长度，默认为16384
        max_decoder_position_embeddings=1024,  # 解码器位置嵌入的最大长度，默认为1024
        encoder_layers=12,  # 编码器层数，默认为12
        encoder_ffn_dim=4096,  # 编码器中FFN层的维度，默认为4096
        encoder_attention_heads=16,  # 编码器中注意力头的数量，默认为16
        decoder_layers=12,  # 解码器层数，默认为12
        decoder_ffn_dim=4096,  # 解码器中FFN层的维度，默认为4096
        decoder_attention_heads=16,  # 解码器中注意力头的数量，默认为16
        encoder_layerdrop=0.0,  # 编码器中层的丢弃率，默认为0.0
        decoder_layerdrop=0.0,  # 解码器中层的丢弃率，默认为0.0
        use_cache=True,  # 是否使用缓存，默认为True
        is_encoder_decoder=True,  # 是否是编码解码模型，默认为True
        activation_function="gelu",  # 激活函数类型，默认为GELU
        d_model=1024,  # 模型维度，默认为1024
        dropout=0.1,  # 全连接层的丢弃率，默认为0.1
        attention_dropout=0.0,  # 注意力层的丢弃率，默认为0.0
        activation_dropout=0.0,  # 激活函数的丢弃率，默认为0.0
        init_std=0.02,  # 初始化的标准差，默认为0.02
        decoder_start_token_id=2,  # 解码器的起始标记ID，默认为2
        classifier_dropout=0.0,  # 分类器的丢弃率，默认为0.0
        pad_token_id=1,  # 填充标记ID，默认为1
        bos_token_id=0,  # 起始标记ID，默认为0
        eos_token_id=2,  # 结束标记ID，默认为2
        attention_window: Union[List[int], int] = 512,  # 注意力窗口的大小，默认为512
        **kwargs,  # 其他参数
    ):
        self.vocab_size = vocab_size
        self.max_encoder_position_embeddings = max_encoder_position_embeddings
        self.max_decoder_position_embeddings = max_decoder_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.attention_window = attention_window

        # 调用父类的初始化函数
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )
```