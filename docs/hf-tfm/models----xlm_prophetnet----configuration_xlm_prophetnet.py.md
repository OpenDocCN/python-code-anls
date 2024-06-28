# `.\models\xlm_prophetnet\configuration_xlm_prophetnet.py`

```
# 设置文件编码为UTF-8，确保代码中的中文和特殊字符能正确处理
# 版权声明和许可条款，指明代码的使用和分发规则
# 引入所需的模块和类，包括预训练配置和日志记录工具
from typing import Callable, Optional, Union

# 从配置工具中导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 从工具模块中导入日志记录功能
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 预训练模型及其配置文件的映射字典，指定了模型名称和对应的配置文件URL
XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/xprophetnet-large-wiki100-cased": (
        "https://huggingface.co/microsoft/xprophetnet-large-wiki100-cased/resolve/main/config.json"
    ),
}

# XLM-ProphetNet模型的配置类，继承自PretrainedConfig
class XLMProphetNetConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`XLMProphetNetModel`]. It is used to instantiate a
    XLMProphetNet model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the XLMProphetNet
    [microsoft/xprophetnet-large-wiki100-cased](https://huggingface.co/microsoft/xprophetnet-large-wiki100-cased)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    """

    # 指定模型类型为"xlm-prophetnet"
    model_type = "xlm-prophetnet"
    # 在推理过程中需要忽略的键名列表
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射字典，将配置属性映射到模型中的具体参数
    attribute_map = {
        "num_attention_heads": "num_encoder_attention_heads",
    }
    # 初始化方法，用于设置模型的各种参数和选项
    def __init__(
        self,
        activation_dropout: Optional[float] = 0.1,  # 激活函数的dropout比例，默认为0.1
        activation_function: Optional[Union[str, Callable]] = "gelu",  # 激活函数的类型，默认为gelu
        vocab_size: Optional[int] = 30522,  # 词汇表大小，默认为30522
        hidden_size: Optional[int] = 1024,  # 隐藏层的尺寸，默认为1024
        encoder_ffn_dim: Optional[int] = 4096,  # 编码器中FFN层的维度，默认为4096
        num_encoder_layers: Optional[int] = 12,  # 编码器的层数，默认为12
        num_encoder_attention_heads: Optional[int] = 16,  # 编码器注意力头的数量，默认为16
        decoder_ffn_dim: Optional[int] = 4096,  # 解码器中FFN层的维度，默认为4096
        num_decoder_layers: Optional[int] = 12,  # 解码器的层数，默认为12
        num_decoder_attention_heads: Optional[int] = 16,  # 解码器注意力头的数量，默认为16
        attention_dropout: Optional[float] = 0.1,  # 注意力机制中的dropout比例，默认为0.1
        dropout: Optional[float] = 0.1,  # 全连接层的dropout比例，默认为0.1
        max_position_embeddings: Optional[int] = 512,  # 最大位置编码数，默认为512
        init_std: Optional[float] = 0.02,  # 初始化的标准差，默认为0.02
        is_encoder_decoder: Optional[bool] = True,  # 是否为编码-解码模型，默认为True
        add_cross_attention: Optional[bool] = True,  # 是否添加交叉注意力机制，默认为True
        decoder_start_token_id: Optional[int] = 0,  # 解码器起始token的ID，默认为0
        ngram: Optional[int] = 2,  # n-gram大小，默认为2
        num_buckets: Optional[int] = 32,  # 桶的数量，默认为32
        relative_max_distance: Optional[int] = 128,  # 相对最大距离，默认为128
        disable_ngram_loss: Optional[bool] = False,  # 是否禁用n-gram损失，默认为False
        eps: Optional[float] = 0.0,  # 用于数值稳定性的小常数，默认为0.0
        use_cache: Optional[bool] = True,  # 是否使用缓存，默认为True
        pad_token_id: Optional[int] = 0,  # 填充token的ID，默认为0
        bos_token_id: Optional[int] = 1,  # 开始token的ID，默认为1
        eos_token_id: Optional[int] = 2,  # 结束token的ID，默认为2
        **kwargs,
    ):
        self.vocab_size = vocab_size  # 设置词汇表大小
        self.hidden_size = hidden_size  # 设置隐藏层大小
        self.encoder_ffn_dim = encoder_ffn_dim  # 设置编码器中FFN层的维度
        self.num_encoder_layers = num_encoder_layers  # 设置编码器层数
        self.num_encoder_attention_heads = num_encoder_attention_heads  # 设置编码器注意力头数
        self.decoder_ffn_dim = decoder_ffn_dim  # 设置解码器中FFN层的维度
        self.num_decoder_layers = num_decoder_layers  # 设置解码器层数
        self.num_decoder_attention_heads = num_decoder_attention_heads  # 设置解码器注意力头数
        self.max_position_embeddings = max_position_embeddings  # 设置最大位置编码数
        self.init_std = init_std  # 设置初始化标准差
        self.activation_function = activation_function  # 设置激活函数类型

        # 用于XLMProphetNet的特定参数
        self.ngram = ngram  # 设置n-gram大小
        self.num_buckets = num_buckets  # 设置桶的数量
        self.relative_max_distance = relative_max_distance  # 设置相对最大距离
        self.disable_ngram_loss = disable_ngram_loss  # 设置是否禁用n-gram损失
        self.eps = eps  # 设置数值稳定性的小常数

        # 三种类型的dropout
        self.attention_dropout = attention_dropout  # 设置注意力机制的dropout比例
        self.activation_dropout = activation_dropout  # 设置激活函数的dropout比例
        self.dropout = dropout  # 设置全连接层的dropout比例

        self.use_cache = use_cache  # 设置是否使用缓存

        # 调用父类的初始化方法，设置其他参数
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            add_cross_attention=add_cross_attention,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )

    @property
    def num_hidden_layers(self) -> int:
        return self.num_encoder_layers + self.num_decoder_layers

    @num_hidden_layers.setter
    # 定义一个方法 `num_hidden_layers`，用于设置隐藏层数量，这里抛出未实现错误
    def num_hidden_layers(self, value):
        # 抛出未实现错误，指示该模型不支持设置隐藏层数量
        raise NotImplementedError(
            "This model does not support the setting of `num_hidden_layers`. Please set `num_encoder_layers` and"
            " `num_decoder_layers`."
        )
```