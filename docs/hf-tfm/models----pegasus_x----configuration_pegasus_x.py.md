# `.\transformers\models\pegasus_x\configuration_pegasus_x.py`

```py
# coding=utf-8
# 版权 2022 年，Google 和 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版（“许可证”）获得许可;
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获得许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据“原样”分发，不提供任何担保或条件，明示或暗示。
# 有关详细信息，请参阅许可证。
""" PEGASUS-X 模型配置"""

# 从配置工具中导入预训练配置
from ...configuration_utils import PretrainedConfig
# 从工具中导入日志记录函数
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# PEGASUS-X 预训练配置文件的映射字典
PEGASUS_X_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/pegasus-x-base": "https://huggingface.co/google/pegasus-x-base/resolve/main/config.json",
    "google/pegasus-x-large": "https://huggingface.co/google/pegasus-x-large/resolve/main/config.json",
    # 查看所有 PEGASUS-X 模型，请访问 https://huggingface.co/models?filter=pegasus-x
}

# PEGASUS-X 配置类，用于存储 PEGASUS-X 模型的配置
class PegasusXConfig(PretrainedConfig):
    r"""
    这是用于存储 [`PegasusXModel`] 配置的配置类。它用于根据指定的参数实例化 PEGASUS-X 模型，定义模型架构。使用默认值实例化配置将产生与 PEGASUS-X [google/pegasus-x-large](https://huggingface.co/google/pegasus-x-large) 架构相似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。有关更多信息，请阅读 [`PretrainedConfig`] 的文档。


    示例:

    ```python
    >>> from transformers import PegasusXConfig, PegasusXModel

    >>> # 初始化 PEGASUS google/pegasus-x-large 风格的配置
    >>> configuration = PegasusXConfig()

    >>> # 从 google/pegasus-x-large 风格的配置初始化一个模型（带有随机权重）
    >>> model = PegasusXModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py"""
    # 模型类型为 PEGASUS-X
    model_type = "pegasus_x"
    # 推理时需要忽略的键列表
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射字典
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}
    # 初始化 Transformer 架构的参数
    def __init__(
        self,
        vocab_size=96103,  # 词汇表的大小，默认为96103
        max_position_embeddings=16384,  # 最大位置嵌入长度，默认为16384
        encoder_layers=16,  # 编码器层数，默认为16
        encoder_ffn_dim=4096,  # 编码器中全连接层的维度，默认为4096
        encoder_attention_heads=16,  # 编码器注意力头数，默认为16
        decoder_layers=16,  # 解码器层数，默认为16
        decoder_ffn_dim=4096,  # 解码器中全连接层的维度，默认为4096
        decoder_attention_heads=16,  # 解码器注意力头数，默认为16
        encoder_layerdrop=0.0,  # 编码器的dropout比例，默认为0.0
        decoder_layerdrop=0.0,  # 解码器的dropout比例，默认为0.0
        use_cache=True,  # 是否使用缓存，默认为True
        is_encoder_decoder=True,  # 是否为编码解码模型，默认为True
        activation_function="gelu",  # 激活函数类型，默认为"gelu"
        d_model=1024,  # 模型维度，默认为1024
        dropout=0.1,  # dropout比例，默认为0.1
        attention_dropout=0.0,  # 注意力机制dropout比例，默认为0.0
        activation_dropout=0.0,  # 激活函数dropout比例，默认为0.0
        init_std=0.02,  # 初始化的标准差，默认为0.02
        decoder_start_token_id=0,  # 解码器起始标记ID，默认为0
        scale_embedding=True,  # 是否对嵌入层进行缩放，默认为True，如果为True，缩放因子将为sqrt(d_model)
    
        pad_token_id=0,  # 填充标记ID，默认为0
        eos_token_id=1,  # 终止符标记ID，默认为1
        forced_eos_token_id=1,  # 强制终止符标记ID，默认为1
        num_global_tokens=32,  # 全局标记数目，默认为32
        block_size=512,  # 分块大小，默认为512
        stagger_local_blocks=True,  # 是否交错本地分块，默认为True
    
        **kwargs,
    ):
        # 初始化实例变量
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
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
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding
    
        self.num_global_tokens = num_global_tokens
        self.block_size = block_size
        self.stagger_local_blocks = stagger_local_blocks
    
        # 使用父类的初始化方法
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )
    
    @property
    def num_attention_heads(self) -> int:
        return self.encoder_attention_heads  # 返回编码器注意力头数
    
    @property
    def hidden_size(self) -> int:
        return self.d_model  # 返回模型的维度
```