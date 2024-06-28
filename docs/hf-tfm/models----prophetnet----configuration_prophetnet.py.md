# `.\models\prophetnet\configuration_prophetnet.py`

```py
# coding=utf-8
# 定义了文件的编码格式为 UTF-8

# 版权声明，指出此代码的版权归属于 Microsoft Authors 和 HuggingFace Inc. 团队
# 代码使用 Apache License, Version 2.0 进行许可，表示在遵守许可的情况下可以使用此代码
# 可以通过指定的链接获取许可的详细信息
# http://www.apache.org/licenses/LICENSE-2.0

# 导入所需模块和库
""" ProphetNet model configuration"""

# 从 typing 模块导入 Callable, Optional, Union 类型提示
from typing import Callable, Optional, Union

# 从配置工具模块中导入预训练配置类 PretrainedConfig
from ...configuration_utils import PretrainedConfig
# 从工具模块中导入日志记录工具
from ...utils import logging

# 获取 logger 对象，用于日志记录
logger = logging.get_logger(__name__)

# 预训练配置映射字典，将预训练模型名称映射到其配置文件的 URL
PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/prophetnet-large-uncased": (
        "https://huggingface.co/microsoft/prophetnet-large-uncased/resolve/main/config.json"
    ),
}


# ProphetNetConfig 类，继承自 PretrainedConfig 类
class ProphetNetConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ProphetNetModel`]. It is used to instantiate a
    ProphetNet model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the ProphetNet
    [microsoft/prophetnet-large-uncased](https://huggingface.co/microsoft/prophetnet-large-uncased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    """

    # 模型类型声明为 prophetnet
    model_type = "prophetnet"
    # 推断过程中需要忽略的键名列表
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射字典，将配置中的 num_attention_heads 映射到 num_encoder_attention_heads
    attribute_map = {
        "num_attention_heads": "num_encoder_attention_heads",
    }
    # 定义初始化方法，设置模型的各项参数
    def __init__(
        self,
        activation_dropout: Optional[float] = 0.1,  # 激活函数的dropout率，默认为0.1
        activation_function: Optional[Union[str, Callable]] = "gelu",  # 激活函数的类型，默认为GELU
        vocab_size: Optional[int] = 30522,  # 词汇表大小，默认为30522
        hidden_size: Optional[int] = 1024,  # 隐藏层大小，默认为1024
        encoder_ffn_dim: Optional[int] = 4096,  # 编码器中FFN层的维度，默认为4096
        num_encoder_layers: Optional[int] = 12,  # 编码器层数，默认为12
        num_encoder_attention_heads: Optional[int] = 16,  # 编码器中注意力头的数量，默认为16
        decoder_ffn_dim: Optional[int] = 4096,  # 解码器中FFN层的维度，默认为4096
        num_decoder_layers: Optional[int] = 12,  # 解码器层数，默认为12
        num_decoder_attention_heads: Optional[int] = 16,  # 解码器中注意力头的数量，默认为16
        attention_dropout: Optional[float] = 0.1,  # 注意力机制的dropout率，默认为0.1
        dropout: Optional[float] = 0.1,  # 通用的dropout率，默认为0.1
        max_position_embeddings: Optional[int] = 512,  # 最大位置编码数，默认为512
        init_std: Optional[float] = 0.02,  # 权重初始化的标准差，默认为0.02
        is_encoder_decoder: Optional[bool] = True,  # 是否为编码解码模型，默认为True
        add_cross_attention: Optional[bool] = True,  # 是否添加交叉注意力，默认为True
        decoder_start_token_id: Optional[int] = 0,  # 解码器起始标记的ID，默认为0
        ngram: Optional[int] = 2,  # ProphetNet模型中的ngram大小，默认为2
        num_buckets: Optional[int] = 32,  # ProphetNet模型中的桶数量，默认为32
        relative_max_distance: Optional[int] = 128,  # ProphetNet模型中的最大相对距离，默认为128
        disable_ngram_loss: Optional[bool] = False,  # 是否禁用ngram损失，默认为False
        eps: Optional[float] = 0.0,  # 极小值，用于数值稳定性，默认为0.0
        use_cache: Optional[bool] = True,  # 是否使用缓存，默认为True
        pad_token_id: Optional[int] = 0,  # 填充标记的ID，默认为0
        bos_token_id: Optional[int] = 1,  # 起始标记的ID，默认为1
        eos_token_id: Optional[int] = 2,  # 结束标记的ID，默认为2
        **kwargs,
    ):
        self.vocab_size = vocab_size  # 设置词汇表大小
        self.hidden_size = hidden_size  # 设置隐藏层大小
        self.encoder_ffn_dim = encoder_ffn_dim  # 设置编码器FFN层的维度
        self.num_encoder_layers = num_encoder_layers  # 设置编码器层数
        self.num_encoder_attention_heads = num_encoder_attention_heads  # 设置编码器注意力头的数量
        self.decoder_ffn_dim = decoder_ffn_dim  # 设置解码器FFN层的维度
        self.num_decoder_layers = num_decoder_layers  # 设置解码器层数
        self.num_decoder_attention_heads = num_decoder_attention_heads  # 设置解码器注意力头的数量
        self.max_position_embeddings = max_position_embeddings  # 设置最大位置编码数
        self.init_std = init_std  # 设置权重初始化的标准差
        self.activation_function = activation_function  # 设置激活函数类型

        # ProphetNet模型的参数
        self.ngram = ngram  # 设置ngram大小
        self.num_buckets = num_buckets  # 设置桶数量
        self.relative_max_distance = relative_max_distance  # 设置最大相对距离
        self.disable_ngram_loss = disable_ngram_loss  # 设置是否禁用ngram损失
        self.eps = eps  # 设置极小值

        # 三种类型的dropout
        self.attention_dropout = attention_dropout  # 设置注意力机制的dropout率
        self.activation_dropout = activation_dropout  # 设置激活函数的dropout率
        self.dropout = dropout  # 设置通用的dropout率

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
    # 定义一个方法 `num_hidden_layers`，该方法接受一个参数 `value`
    def num_hidden_layers(self, value):
        # 抛出 `NotImplementedError` 异常，表示这个模型不支持设置 `num_hidden_layers` 参数
        raise NotImplementedError(
            "This model does not support the setting of `num_hidden_layers`. Please set `num_encoder_layers` and"
            " `num_decoder_layers`."
        )
```