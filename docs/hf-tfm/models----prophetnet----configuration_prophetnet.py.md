# `.\transformers\models\prophetnet\configuration_prophetnet.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 2020 年版权归微软作者和 HuggingFace 公司团队所有
# 根据 Apache 许可证 2.0 版进行许可
# 除非符合许可证的规定，否则您不得使用此文件
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按"原样"提供，
# 不提供任何形式的担保或条件，无论是明示的还是暗示的。
# 有关权限，请参阅许可证。
"""ProphetNet 模型配置"""

# 导入必要的模块
from typing import Callable, Optional, Union
# 导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 导入日志工具
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练配置的存档映射
PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/prophetnet-large-uncased": (
        "https://huggingface.co/microsoft/prophetnet-large-uncased/resolve/main/config.json"
    ),
}

# ProphetNet 配置类，继承自 PretrainedConfig
class ProphetNetConfig(PretrainedConfig):
    r"""
    这是用于存储 [`ProphetNetModel`] 配置的配置类。根据指定的参数，它用于实例化
    ProphetNet 模型，定义模型架构。使用默认值实例化配置将产生类似于 ProphetNet
    [microsoft/prophetnet-large-uncased](https://huggingface.co/microsoft/prophetnet-large-uncased) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。有关更多信息，请阅读
    [`PretrainedConfig`] 的文档。

    """

    # 模型类型
    model_type = "prophetnet"
    # 推理时忽略的键
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射
    attribute_map = {
        "num_attention_heads": "num_encoder_attention_heads",
    }
    def __init__(
        self,
        activation_dropout: Optional[float] = 0.1,  # 激活函数的 dropout 概率，默认为 0.1
        activation_function: Optional[Union[str, Callable]] = "gelu",  # 激活函数类型，默认为 GELU
        vocab_size: Optional[int] = 30522,  # 词汇表大小，默认为 30522
        hidden_size: Optional[int] = 1024,  # 隐藏层大小，默认为 1024
        encoder_ffn_dim: Optional[int] = 4096,  # 编码器前馈网络的维度，默认为 4096
        num_encoder_layers: Optional[int] = 12,  # 编码器层数，默认为 12
        num_encoder_attention_heads: Optional[int] = 16,  # 编码器注意力头数，默认为 16
        decoder_ffn_dim: Optional[int] = 4096,  # 解码器前馈网络的维度，默认为 4096
        num_decoder_layers: Optional[int] = 12,  # 解码器层数，默认为 12
        num_decoder_attention_heads: Optional[int] = 16,  # 解码器注意力头数，默认为 16
        attention_dropout: Optional[float] = 0.1,  # 注意力机制的 dropout 概率，默认为 0.1
        dropout: Optional[float] = 0.1,  # dropout 概率，默认为 0.1
        max_position_embeddings: Optional[int] = 512,  # 最大位置嵌入数，默认为 512
        init_std: Optional[float] = 0.02,  # 初始化标准差，默认为 0.02，用于初始化权重
        is_encoder_decoder: Optional[bool] = True,  # 是否是编码器-解码器结构，默认为 True
        add_cross_attention: Optional[bool] = True,  # 是否添加交叉注意力，默认为 True
        decoder_start_token_id: Optional[int] = 0,  # 解码器起始标记 ID，默认为 0
        ngram: Optional[int] = 2,  # N 元语法的 N，默认为 2
        num_buckets: Optional[int] = 32,  # 桶的数量，默认为 32
        relative_max_distance: Optional[int] = 128,  # 相对最大距离，默认为 128
        disable_ngram_loss: Optional[bool] = False,  # 是否禁用 N 元语法损失，默认为 False
        eps: Optional[float] = 0.0,  # 小数精度，默认为 0.0
        use_cache: Optional[bool] = True,  # 是否使用缓存，默认为 True
        pad_token_id: Optional[int] = 0,  # 填充标记 ID，默认为 0
        bos_token_id: Optional[int] = 1,  # 开始标记 ID，默认为 1
        eos_token_id: Optional[int] = 2,  # 结束标记 ID，默认为 2
        **kwargs,  # 其他参数
    ):
        self.vocab_size = vocab_size  # 设置词汇表大小
        self.hidden_size = hidden_size  # 设置隐藏层大小
        self.encoder_ffn_dim = encoder_ffn_dim  # 设置编码器前馈网络维度
        self.num_encoder_layers = num_encoder_layers  # 设置编码器层数
        self.num_encoder_attention_heads = num_encoder_attention_heads  # 设置编码器注意力头数
        self.decoder_ffn_dim = decoder_ffn_dim  # 设置解码器前馈网络维度
        self.num_decoder_layers = num_decoder_layers  # 设置解码器层数
        self.num_decoder_attention_heads = num_decoder_attention_heads  # 设置解码器注意力头数
        self.max_position_embeddings = max_position_embeddings  # 设置最大位置嵌入数
        self.init_std = init_std  # 设置初始化标准差，用于初始化权重
        self.activation_function = activation_function  # 设置激活函数类型

        # 用于 ProphetNet 的参数
        self.ngram = ngram  # 设置 N 元语法的 N
        self.num_buckets = num_buckets  # 设置桶的数量
        self.relative_max_distance = relative_max_distance  # 设置相对最大距离
        self.disable_ngram_loss = disable_ngram_loss  # 设置是否禁用 N 元语法损失
        self.eps = eps  # 设置小数精度

        # 三种类型的 Dropout
        self.attention_dropout = attention_dropout  # 设置注意力机制的 dropout 概率
        self.activation_dropout = activation_dropout  # 设置激活函数的 dropout 概率
        self.dropout = dropout  # 设置 dropout 概率

        self.use_cache = use_cache  # 设置是否使用缓存

        super().__init__(  # 调用父类构造函数
            pad_token_id=pad_token_id,  # 填充标记 ID
            bos_token_id=bos_token_id,  # 开始标记 ID
            eos_token_id=eos_token_id,  # 结束标记 ID
            is_encoder_decoder=is_encoder_decoder,  # 是否是编码器-解码器结构
            add_cross_attention=add_cross_attention,  # 是否添加交叉注意力
            decoder_start_token_id=decoder_start_token_id,  # 解码器起始标记 ID
            **kwargs,  # 其他参数
        )

    @property
    def num_hidden_layers(self) -> int:
        return self.num_encoder_layers + self.num_decoder_layers  # 返回隐藏层总数

    @num_hidden_layers.setter
    # 定义一个方法用于设置隐藏层数，但是抛出了未实现的错误，因为这个模型不支持直接设置隐藏层数。
    def num_hidden_layers(self, value):
        # 抛出未实现的错误，提醒用户不支持设置隐藏层数，应该分别设置编码器层数和解码器层数。
        raise NotImplementedError(
            "This model does not support the setting of `num_hidden_layers`. Please set `num_encoder_layers` and"
            " `num_decoder_layers`."
        )
```