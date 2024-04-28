# `.\transformers\models\xlm_prophetnet\configuration_xlm_prophetnet.py`

```
# 指定 Python 文件的字符编码为 UTF-8
# 声明代码的版权归属于 Microsoft 和 HuggingFace 团队
# 许可证信息：根据 Apache 许可证第 2 版进行授权，许可证条款规定了代码的使用条件
# 链接到 Apache 许可证，提供查看许可证全文的链接
# 警告：软件按 "AS IS" 的基础分发，不保证其功能和可靠性
# 版权限制：规定了许可证的限制条件
""" XLM-ProphetNet 模型配置"""

# 导入 Callable、Optional、Union 类型用于类型提示
from typing import Callable, Optional, Union

# 导入预训练配置类和日志工具
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 创建日志对象，用于记录日志信息
logger = logging.get_logger(__name__)

# 定义 XLM-ProphetNet 预训练配置的存档映射
XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    # 预定义的模型配置，提供配置文件的下载链接
    "microsoft/xprophetnet-large-wiki100-cased": (
        "https://huggingface.co/microsoft/xprophetnet-large-wiki100-cased/resolve/main/config.json"
    ),
}

# 定义 XLMProphetNetConfig 类，继承自 PretrainedConfig
class XLMProphetNetConfig(PretrainedConfig):
    r"""
    这是用于存储 [`XLMProphetNetModel`] 的配置的配置类。
    它用于根据指定参数实例化 XLMProphetNet 模型，定义模型架构。
    默认配置将生成类似于 [microsoft/xprophetnet-large-wiki100-cased]
    (https://huggingface.co/microsoft/xprophetnet-large-wiki100-cased) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。
    阅读 [`PretrainedConfig`] 文档以获取更多信息。
    """

    # 模型类型为 "xlm-prophetnet"
    model_type = "xlm-prophetnet"
    # 在推理时忽略的键，指定为 ["past_key_values"]
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射，用于映射不同属性的名称
    attribute_map = {
        "num_attention_heads": "num_encoder_attention_heads",
    }
    # 初始化函数，设置模型的各项参数
    def __init__(
        self,
        activation_dropout: Optional[float] = 0.1,  # 激活函数的dropout概率，默认为0.1
        activation_function: Optional[Union[str, Callable]] = "gelu",  # 激活函数的类型，默认为gelu
        vocab_size: Optional[int] = 30522,  # 词汇表大小，默认为30522
        hidden_size: Optional[int] = 1024,  # 隐藏层维度，默认为1024
        encoder_ffn_dim: Optional[int] = 4096,  # 编码器前馈网络维度，默认为4096
        num_encoder_layers: Optional[int] = 12,  # 编码器层数，默认为12
        num_encoder_attention_heads: Optional[int] = 16,  # 编码器自注意力头数，默认为16
        decoder_ffn_dim: Optional[int] = 4096,  # 解码器前馈网络维度，默认为4096
        num_decoder_layers: Optional[int] = 12,  # 解码器层数，默认为12
        num_decoder_attention_heads: Optional[int] = 16,  # 解码器自注意力头数，默认为16
        attention_dropout: Optional[float] = 0.1,  # 注意力机制的dropout概率，默认为0.1
        dropout: Optional[float] = 0.1,  # 普通dropout概率，默认为0.1
        max_position_embeddings: Optional[int] = 512,  # 最大位置编码数，默认为512
        init_std: Optional[float] = 0.02,  # 初始化标准差，默认为0.02
        is_encoder_decoder: Optional[bool] = True,  # 是否是编码解码结构，默认为True
        add_cross_attention: Optional[bool] = True,  # 是否添加交叉注意力，默认为True
        decoder_start_token_id: Optional[int] = 0,  # 解码器起始标记ID，默认为0
        ngram: Optional[int] = 2,  # n-gram大小，默认为2
        num_buckets: Optional[int] = 32,  # 桶的数量，默认为32
        relative_max_distance: Optional[int] = 128,  # 相对最大距离，默认为128
        disable_ngram_loss: Optional[bool] = False,  # 是否禁用n-gram损失，默认为False
        eps: Optional[float] = 0.0,  # epsilon参数，默认为0.0
        use_cache: Optional[bool] = True,  # 是否使用缓存，默认为True
        pad_token_id: Optional[int] = 0,  # 填充标记ID，默认为0
        bos_token_id: Optional[int] = 1,  # 起始标记ID，默认为1
        eos_token_id: Optional[int] = 2,  # 结束标记ID，默认为2
        **kwargs,
    ):
        self.vocab_size = vocab_size  # 设置模型的词汇表大小
        self.hidden_size = hidden_size  # 设置模型的隐藏层维度
        self.encoder_ffn_dim = encoder_ffn_dim  # 设置编码器前馈网络维度
        self.num_encoder_layers = num_encoder_layers  # 设置编码器层数
        self.num_encoder_attention_heads = num_encoder_attention_heads  # 设置编码器自注意力头数
        self.decoder_ffn_dim = decoder_ffn_dim  # 设置解码器前馈网络维度
        self.num_decoder_layers = num_decoder_layers  # 设置解码器层数
        self.num_decoder_attention_heads = num_decoder_attention_heads  # 设置解码器自注意力头数
        self.max_position_embeddings = max_position_embeddings  # 设置最大位置编码数
        self.init_std = init_std  # 设置初始化标准差，服从正态分布，均值为0，标准差为此参数
        self.activation_function = activation_function  # 设置激活函数类型

        # 参数用于xlmprophetnet
        self.ngram = ngram  # 设置n-gram大小
        self.num_buckets = num_buckets  # 设置桶的数量
        self.relative_max_distance = relative_max_distance  # 设置相对最大距离
        self.disable_ngram_loss = disable_ngram_loss  # 设置是否禁用n-gram损失
        self.eps = eps  # 设置epsilon参数

        # 3种类型的dropout
        self.attention_dropout = attention_dropout  # 设置注意力机制的dropout概率
        self.activation_dropout = activation_dropout  # 设置激活函数的dropout概率
        self.dropout = dropout  # 设置普通dropout概率

        self.use_cache = use_cache  # 设置是否使用缓存

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
        return self.num_encoder_layers + self.num_decoder_layers  # 返回编码器和解码器的层数之和

    @num_hidden_layers.setter
    # 定义一个方法，用于设置隐藏层的数量，这里抛出一个未实现的错误，提示不支持设置隐藏层数量，并建议设置编码器和解码器层数
    def num_hidden_layers(self, value):
        raise NotImplementedError(
            "This model does not support the setting of `num_hidden_layers`. Please set `num_encoder_layers` and"
            " `num_decoder_layers`."
        )
```