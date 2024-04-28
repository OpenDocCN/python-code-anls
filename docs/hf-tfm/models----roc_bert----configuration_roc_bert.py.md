# `.\transformers\models\roc_bert\configuration_roc_bert.py`

```py
# 设定编码格式为 utf-8
# 版权声明
# 声明此代码版权属于 WeChatAI 和 The HuggingFace Inc. 团队，保留所有权利
# 根据 Apache 许可证 2.0 （“许可证”）授权
# 除非符合许可证的规定，否则您不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非符合适用法律规定或书面同意，根据许可证分发的软件是基于“原样”分发，不附带任何明示或默示的担保或条件
# 有关特定语言控制模型输出的许可证，并查看特定语言控制的文档
""" RoCBert 模型配置"""

# 导入预训练配置 abstract 和 logging 工具包
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取 logger
logger = logging.get_logger(__name__)

# 预训练配置的映射
ROC_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "weiweishi/roc-bert-base-zh": "https://huggingface.co/weiweishi/roc-bert-base-zh/resolve/main/config.json",
}

# RoCBert 模型配置类
class RoCBertConfig(PretrainedConfig):
    r"""
    这是用于存储 [`RoCBertModel`] 配置的配置类。 它用于实例化 RoCBert 模型，根据指定的参数定义模型架构。
    用默认值初始化配置将生成与 RoCBert [weiweishi/roc-bert-base-zh](https://huggingface.co/weiweishi/roc-bert-base-zh) 架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。 有关更多信息，请阅读 [`PretrainedConfig`] 的文档。

    
    ```python
    >>> from transformers import RoCBertModel, RoCBertConfig

    >>> # 初始化 RoCBert weiweishi/roc-bert-base-zh 样式配置
    >>> configuration = RoCBertConfig()

    >>> # 从 weiweishi/roc-bert-base-zh 样式配置初始化模型
    >>> model = RoCBertModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py"""

    # 模型类型为 "roc_bert"
    model_type = "roc_bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_cache=True,
        pad_token_id=0,
        position_embedding_type="absolute",
        classifier_dropout=None,
        enable_pronunciation=True,
        enable_shape=True,
        pronunciation_embed_dim=768,
        pronunciation_vocab_size=910,
        shape_embed_dim=512,
        shape_vocab_size=24858,
        concat_input=True,
        **kwargs,
        ):
        # 初始化 Transformer 模型的参数
        self.vocab_size = vocab_size  # 词汇表大小
        self.max_position_embeddings = max_position_embeddings  # 最大位置编码长度
        self.hidden_size = hidden_size  # 隐藏层大小
        self.num_hidden_layers = num_hidden_layers  # 隐藏层的数量
        self.num_attention_heads = num_attention_heads  # 注意力头的数量
        self.intermediate_size = intermediate_size  # 中间层大小
        self.hidden_act = hidden_act  # 激活函数类型
        self.hidden_dropout_prob = hidden_dropout_prob  # 隐藏层的 dropout 概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob  # 注意力权重的 dropout 概率
        self.initializer_range = initializer_range  # 参数初始化范围
        self.type_vocab_size = type_vocab_size  # 类型词汇表大小
        self.layer_norm_eps = layer_norm_eps  # LayerNormalization 的 epsilon 参数
        self.use_cache = use_cache  # 是否使用缓存
        self.enable_pronunciation = enable_pronunciation  # 是否启用发音特征
        self.enable_shape = enable_shape  # 是否启用形状特征
        self.pronunciation_embed_dim = pronunciation_embed_dim  # 发音特征嵌入维度
        self.pronunciation_vocab_size = pronunciation_vocab_size  # 发音特征词汇表大小
        self.shape_embed_dim = shape_embed_dim  # 形状特征嵌入维度
        self.shape_vocab_size = shape_vocab_size  # 形状特征词汇表大小
        self.concat_input = concat_input  # 是否将特征拼接到输入中
        self.position_embedding_type = position_embedding_type  # 位置编码类型
        self.classifier_dropout = classifier_dropout  # 分类器的 dropout 概率
        # 调用父类的初始化方法，传入 pad_token_id 参数以及其他可能的关键字参数
        super().__init__(pad_token_id=pad_token_id, **kwargs)
```