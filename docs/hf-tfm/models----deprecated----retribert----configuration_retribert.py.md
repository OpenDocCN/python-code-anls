# `.\models\deprecated\retribert\configuration_retribert.py`

```py
# coding=utf-8
# 代码文件的编码格式为 UTF-8
# 版权声明，版权属于 HuggingFace Inc. 团队、Google AI 语言团队和 Facebook, Inc.
# 根据 Apache License, Version 2.0 许可，除非符合许可要求，否则禁止使用此文件
# 你可以在以下网址获取许可的副本：http://www.apache.org/licenses/LICENSE-2.0
# 根据适用法律的要求或书面同意，在 "AS IS" 基础上发布软件，不提供任何形式的担保或条件
# 有关更多信息，请参见许可，特定语言的描述，请参见许可文件

""" RetriBERT 模型配置"""

# 从配置工具导入 PretrainedConfig 类
from ....configuration_utils import PretrainedConfig
# 从工具包导入日志记录模块
from ....utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# TODO: upload to AWS
# 预训练配置文件的映射字典
RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "yjernite/retribert-base-uncased": (
        "https://huggingface.co/yjernite/retribert-base-uncased/resolve/main/config.json"
    ),
}


class RetriBertConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`RetriBertModel`] 的配置。它用于根据指定的参数实例化 RetriBertModel 模型，定义模型的架构。
    使用默认值实例化配置将产生类似于 RetriBERT [yjernite/retribert-base-uncased](https://huggingface.co/yjernite/retribert-base-uncased) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。有关更多信息，请阅读 [`PretrainedConfig`] 的文档。
    """
    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the RetriBERT model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`RetriBertModel`]
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the *token_type_ids* passed into [`BertModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        share_encoders (`bool`, *optional*, defaults to `True`):
            Whether or not to use the same Bert-type encoder for the queries and document
        projection_dim (`int`, *optional*, defaults to 128):
            Final dimension of the query and document representation after projection
    """

    model_type = "retribert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=8,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        share_encoders=True,
        projection_dim=128,
        pad_token_id=0,
        **kwargs,
    # 定义一个继承自特殊TensorFlow构建器类__init__的子类，并重写__init__方法
    def __init__(
        # 调用父类的初始化方法，并传入参数pad_token_id和kwargs
        super().__init__(pad_token_id=pad_token_id, **kwargs)
    
        # 定义词汇表大小
        self.vocab_size = vocab_size
        # 定义隐藏层大小
        self.hidden_size = hidden_size
        # 定义隐藏层数量
        self.num_hidden_layers = num_hidden_layers
        # 定义注意力头数量
        self.num_attention_heads = num_attention_heads
        # 定义隐藏层激活函数
        self.hidden_act = hidden_act
        # 定义中间层大小
        self.intermediate_size = intermediate_size
        # 定义隐藏层舍弃比例
        self.hidden_dropout_prob = hidden_dropout_prob
        # 定义注意力机制舍弃比例
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 定义最大位置嵌入大小
        self.max_position_embeddings = max_position_embeddings
        # 定义类型词汇表大小
        self.type_vocab_size = type_vocab_size
        # 定义初始化范围
        self.initializer_range = initializer_range
        # 定义层规范化的epsilon值
        self.layer_norm_eps = layer_norm_eps
        # 是否共享编码器
        self.share_encoders = share_encoders
        # 定义投影维度
        self.projection_dim = projection_dim
```