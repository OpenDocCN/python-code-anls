# `.\models\dpr\configuration_dpr.py`

```py
# coding=utf-8
# 载入预训练配置类 PretrainedConfig 和 logging 工具
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取 logger 对象
logger = logging.get_logger(__name__)

# DPR 预训练配置文件的映射表，将模型名称映射到对应的配置文件地址
DPR_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/dpr-ctx_encoder-single-nq-base": (
        "https://huggingface.co/facebook/dpr-ctx_encoder-single-nq-base/resolve/main/config.json"
    ),
    "facebook/dpr-question_encoder-single-nq-base": (
        "https://huggingface.co/facebook/dpr-question_encoder-single-nq-base/resolve/main/config.json"
    ),
    "facebook/dpr-reader-single-nq-base": (
        "https://huggingface.co/facebook/dpr-reader-single-nq-base/resolve/main/config.json"
    ),
    "facebook/dpr-ctx_encoder-multiset-base": (
        "https://huggingface.co/facebook/dpr-ctx_encoder-multiset-base/resolve/main/config.json"
    ),
    "facebook/dpr-question_encoder-multiset-base": (
        "https://huggingface.co/facebook/dpr-question_encoder-multiset-base/resolve/main/config.json"
    ),
    "facebook/dpr-reader-multiset-base": (
        "https://huggingface.co/facebook/dpr-reader-multiset-base/resolve/main/config.json"
    ),
}

# DPRConfig 类，存储 DPRModel 的配置信息
class DPRConfig(PretrainedConfig):
    r"""
    [`DPRConfig`] 是用于存储 *DPRModel* 的配置信息的配置类。

    这是用于存储 [`DPRContextEncoder`]、[`DPRQuestionEncoder`] 或 [`DPRReader`] 的配置信息的配置类。
    该类用于根据指定的参数实例化 DPR 模型的组件，定义模型组件的架构。使用默认值实例化配置将产生类似于 DPRContextEncoder
    [facebook/dpr-ctx_encoder-single-nq-base](https://huggingface.co/facebook/dpr-ctx_encoder-single-nq-base)
    架构的配置。

    这个类是 [`BertConfig`] 的子类。请查看超类以获取所有 kwargs 的文档。

    示例:

    ```python
    >>> from transformers import DPRConfig, DPRContextEncoder

    >>> # 初始化一个 DPR facebook/dpr-ctx_encoder-single-nq-base 风格的配置
    >>> configuration = DPRConfig()

    >>> # 使用 facebook/dpr-ctx_encoder-single-nq-base 风格的配置初始化一个模型（随机权重）
    >>> model = DPRContextEncoder(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py"""
```  
    # 模型类型设为 "dpr"
    model_type = "dpr"

    # 初始化函数，设定模型参数
    def __init__(
        self,
        vocab_size=30522,  # 词汇表大小，默认为30522
        hidden_size=768,  # 隐藏层大小，默认为768
        num_hidden_layers=12,  # 隐藏层层数，默认为12
        num_attention_heads=12,  # 注意力头数，默认为12
        intermediate_size=3072,  # 中间层大小，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为gelu
        hidden_dropout_prob=0.1,  # 隐藏层dropout概率，默认为0.1
        attention_probs_dropout_prob=0.1,  # 注意力概率dropout概率，默认为0.1
        max_position_embeddings=512,  # 最大位置嵌入，默认为512
        type_vocab_size=2,  # 类型词汇表大小，默认为2
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-12,  # 层归一化epsilon值，默认为1e-12
        pad_token_id=0,  # 填充token的id，默认为0
        position_embedding_type="absolute",  # 位置嵌入类型，默认为绝对位置嵌入
        projection_dim: int = 0,  # 投影维度，默认为0
        **kwargs,
    ):
        # 调用父类的初始化函数，设定填充token的id
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        # 设置模型参数
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
        self.projection_dim = projection_dim
        self.position_embedding_type = position_embedding_type
```