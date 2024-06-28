# `.\models\megatron_bert\configuration_megatron_bert.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 NVIDIA 公司和 HuggingFace Inc. 团队所有
# 根据 Apache 许可证版本 2.0 使用本文件，除非符合许可证的要求，否则不得使用本文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 根据适用法律或书面同意，本软件是基于“按原样”分发的，没有任何形式的担保或条件
# 请参阅许可证以获取详细的条款和条件信息

""" MEGATRON_BERT 模型配置"""

# 从 transformers 库中导入预训练配置类 PretrainedConfig
from ...configuration_utils import PretrainedConfig
# 从 transformers 库中导入日志记录工具 logging
from ...utils import logging

# 获取指定名称空间下的日志记录器
logger = logging.get_logger(__name__)

# MEGATRON_BERT 预训练配置文件存档映射，目前为空字典
MEGATRON_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    # 可以在 https://huggingface.co/models?filter=bert 查看所有 MEGATRON_BERT 模型
}


class MegatronBertConfig(PretrainedConfig):
    r"""
    这是用于存储 [`MegatronBertModel`] 配置的配置类。它用于根据指定的参数实例化一个 MEGATRON_BERT 模型，
    定义模型的架构。使用默认值实例化配置将产生类似于 MEGATRON_BERT 
    [nvidia/megatron-bert-uncased-345m](https://huggingface.co/nvidia/megatron-bert-uncased-345m) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。有关更多信息，请阅读 [`PretrainedConfig`] 的文档。

    Examples:

    ```python
    >>> from transformers import MegatronBertConfig, MegatronBertModel

    >>> # 初始化一个 MEGATRON_BERT google-bert/bert-base-uncased 风格的配置
    >>> configuration = MegatronBertConfig()

    >>> # 使用配置初始化一个（带有随机权重）从 google-bert/bert-base-uncased 风格配置的模型
    >>> model = MegatronBertModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```
    """

    # 模型类型为 "megatron-bert"
    model_type = "megatron-bert"

    def __init__(
        self,
        vocab_size=29056,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        **kwargs,
        ):
        # 调用父类的初始化方法，传递填充令牌 ID 和其他关键字参数
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        # 设置模型的词汇表大小
        self.vocab_size = vocab_size
        # 设置隐藏层的大小
        self.hidden_size = hidden_size
        # 设置隐藏层的数量
        self.num_hidden_layers = num_hidden_layers
        # 设置注意力头的数量
        self.num_attention_heads = num_attention_heads
        # 设置隐藏层激活函数的类型
        self.hidden_act = hidden_act
        # 设置中间层大小（即 Transformer 中的 feedforward 层大小）
        self.intermediate_size = intermediate_size
        # 设置隐藏层的 dropout 概率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 设置注意力概率 dropout 的概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 设置最大位置嵌入的大小
        self.max_position_embeddings = max_position_embeddings
        # 设置类型词汇表的大小（通常用于区分句子 A 和句子 B）
        self.type_vocab_size = type_vocab_size
        # 设置初始化范围（权重初始化的范围）
        self.initializer_range = initializer_range
        # 设置层归一化的 epsilon 值
        self.layer_norm_eps = layer_norm_eps
        # 设置位置嵌入的类型（绝对位置编码或相对位置编码）
        self.position_embedding_type = position_embedding_type
        # 设置是否使用缓存（用于缓存中间计算结果，提高效率）
        self.use_cache = use_cache
```