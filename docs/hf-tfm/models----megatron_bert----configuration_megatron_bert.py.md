# `.\transformers\models\megatron_bert\configuration_megatron_bert.py`

```
# 设置文件编码为 utf-8
# 版权声明，版权归 NVIDIA Corporation 和 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关特定语言的权限和限制

""" MEGATRON_BERT 模型配置"""

# 导入必要的模块和函数
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# MEGATRON_BERT 预训练配置映射
MEGATRON_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    # 查看所有 MEGATRON_BERT 模型 https://huggingface.co/models?filter=bert
}

# MegatronBertConfig 类，用于存储 MegatronBertModel 的配置
class MegatronBertConfig(PretrainedConfig):
    r"""
    这是用于存储 [`MegatronBertModel`] 配置的类。根据指定的参数实例化 MEGATRON_BERT 模型，定义模型架构。
    使用默认值实例化配置将产生类似于 MEGATRON_BERT [nvidia/megatron-bert-uncased-345m](https://huggingface.co/nvidia/megatron-bert-uncased-345m) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    示例:

    ```python
    >>> from transformers import MegatronBertConfig, MegatronBertModel

    >>> # 初始化一个 MEGATRON_BERT bert-base-uncased 风格的配置
    >>> configuration = MegatronBertConfig()

    >>> # 从 bert-base-uncased 风格的配置初始化一个模型（带有随机权重）
    >>> model = MegatronBertModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```"""

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
        # 调用父类的构造函数，初始化模型的填充标记 ID 和其他参数
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        # 设置模型的词汇表大小、隐藏层大小、隐藏层数量、注意力头数量等参数
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
```