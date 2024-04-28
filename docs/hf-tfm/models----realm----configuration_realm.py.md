# `.\transformers\models\realm\configuration_realm.py`

```
# 设置文件编码为 utf-8
# 版权声明
#
# 遵循 Apache 2.0 许可协议
# 获取许可协议的地址
# https://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或经书面同意，否则不得使用本文件
# 分布的内容是"原样"的，没有任何明示或暗示的保修或条件
# 有关具体语言控制输出结果，以及限制协议下的权限和
# 条件，请参阅协议。
# REALM 模型的配置。

# 导入所需的模块
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# REALM 模型的预训练配置字典
REALM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/realm-cc-news-pretrained-embedder": (
        "https://huggingface.co/google/realm-cc-news-pretrained-embedder/resolve/main/config.json"
    ),
    "google/realm-cc-news-pretrained-encoder": (
        "https://huggingface.co/google/realm-cc-news-pretrained-encoder/resolve/main/config.json"
    ),
    "google/realm-cc-news-pretrained-scorer": (
        "https://huggingface.co/google/realm-cc-news-pretrained-scorer/resolve/main/config.json"
    ),
    "google/realm-cc-news-pretrained-openqa": (
        "https://huggingface.co/google/realm-cc-news-pretrained-openqa/aresolve/main/config.json"
    ),
    "google/realm-orqa-nq-openqa": "https://huggingface.co/google/realm-orqa-nq-openqa/resolve/main/config.json",
    "google/realm-orqa-nq-reader": "https://huggingface.co/google/realm-orqa-nq-reader/resolve/main/config.json",
    "google/realm-orqa-wq-openqa": "https://huggingface.co/google/realm-orqa-wq-openqa/resolve/main/config.json",
    "google/realm-orqa-wq-reader": "https://huggingface.co/google/realm-orqa-wq-reader/resolve/main/config.json",
    # 查看所有 REALM 模型 https://huggingface.co/models?filter=realm
}

# REALM 模型的配置类
class RealmConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of

    1. [`RealmEmbedder`]
    2. [`RealmScorer`]
    3. [`RealmKnowledgeAugEncoder`]
    4. [`RealmRetriever`]
    5. [`RealmReader`]
    6. [`RealmForOpenQA`]

    It is used to instantiate an REALM model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the REALM
    [google/realm-cc-news-pretrained-embedder](https://huggingface.co/google/realm-cc-news-pretrained-embedder)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:
    >>> from transformers import RealmConfig, RealmEmbedder
    >>> # 初始化 REALM realm-cc-news-pretrained-* 风格的配置
    >>> configuration = RealmConfig()
    # 使用 google/realm-cc-news-pretrained-embedder 风格的配置初始化一个模型，并赋予随机权重
    model = RealmEmbedder(configuration)
    
    # 访问模型的配置信息
    configuration = model.config
    
    # 定义模型类型为 "realm"
    model_type = "realm"
    
    def __init__(
        # 词汇表大小，默认为 30522
        self,
        vocab_size=30522,
        # 隐藏层大小，默认为 768
        hidden_size=768,
        # 检索器投影大小，默认为 128
        retriever_proj_size=128,
        # 隐藏层数，默认为 12
        num_hidden_layers=12,
        # 注意力头数，默认为 12
        num_attention_heads=12,
        # 候选项数量，默认为 8
        num_candidates=8,
        # 中间层大小，默认为 3072
        intermediate_size=3072,
        # 隐藏层激活函数，默认为 "gelu_new"
        hidden_act="gelu_new",
        # 隐藏层 dropout 概率，默认为 0.1
        hidden_dropout_prob=0.1,
        # 注意力概率 dropout 概率，默认为 0.1
        attention_probs_dropout_prob=0.1,
        # 最大位置嵌入长度，默认为 512
        max_position_embeddings=512,
        # 类型词汇表大小，默认为 2
        type_vocab_size=2,
        # 初始化范围，默认为 0.02
        initializer_range=0.02,
        # 层归一化 epsilon，默认为 1e-12
        layer_norm_eps=1e-12,
        # 跨度隐藏层大小，默认为 256
        span_hidden_size=256,
        # 最大跨度宽度，默认为 10
        max_span_width=10,
        # 阅读器层归一化 epsilon，默认为 1e-3
        reader_layer_norm_eps=1e-3,
        # 阅读器束搜索大小，默认为 5
        reader_beam_size=5,
        # 阅读器序列长度，默认为 320
        reader_seq_len=320,  # 288 + 32
        # 区块记录数量，默认为 13353718
        num_block_records=13353718,
        # 搜索器束搜索大小，默认为 5000
        searcher_beam_size=5000,
        # 填充标记 ID，默认为 1
        pad_token_id=1,
        # 开始标记 ID，默认为 0
        bos_token_id=0,
        # 结束标记 ID，默认为 2
        eos_token_id=2,
        # 其他关键字参数
        **kwargs,
    ):
        # 调用父类的初始化方法
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
    
        # 通用配置
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.retriever_proj_size = retriever_proj_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_candidates = num_candidates
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
    
        # 阅读器配置
        self.span_hidden_size = span_hidden_size
        self.max_span_width = max_span_width
        self.reader_layer_norm_eps = reader_layer_norm_eps
        self.reader_beam_size = reader_beam_size
        self.reader_seq_len = reader_seq_len
    
        # 检索配置
        self.num_block_records = num_block_records
        self.searcher_beam_size = searcher_beam_size
```