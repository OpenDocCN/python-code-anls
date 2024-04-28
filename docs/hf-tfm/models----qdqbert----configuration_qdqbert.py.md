# `.\transformers\models\qdqbert\configuration_qdqbert.py`

```
# 引入必要的模块和工具函数
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 设置日志记录器
logger = logging.get_logger(__name__)

# 定义一个字典，将BERT模型预训练配置文件映射到其对应的URL地址
QDQBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "bert-base-uncased": "https://huggingface.co/bert-base-uncased/resolve/main/config.json",
    # QDQBERT models can be loaded from any BERT checkpoint, available at https://huggingface.co/models?filter=bert
}

# 定义QDQBertConfig类，用于存储QDQBert模型的配置信息，并用于实例化QDQBERT模型
class QDQBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`QDQBertModel`]. It is used to instantiate an
    QDQBERT model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the BERT
    [bert-base-uncased](https://huggingface.co/bert-base-uncased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


# 以上代码是QDQBERT模型的配置文件，用于存储模型的配置信息和相关的URL地址映射。
    # 定义了 QDQBERT 模型的配置类 QDQBertConfig，以下是配置参数的说明
    
    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            QDQBERT 模型的词汇表大小，定义了在调用 [`QDQBertModel`] 时可以表示的不同标记数量。
        hidden_size (`int`, *optional*, defaults to 768):
            编码器层和池化层的维度。
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Transformer 编码器中的隐藏层数量。
        num_attention_heads (`int`, *optional*, defaults to 12):
            Transformer 编码器中每个注意力层的注意力头数量。
        intermediate_size (`int`, *optional*, defaults to 3072):
            Transformer 编码器中“中间”（即前馈）层的维度。
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            编码器和池化层中的非线性激活函数（函数或字符串）。如果是字符串，则支持 `"gelu"`、`"relu"`、`"selu"` 和 `"gelu_new"`。
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            在所有完全连接层中使用的 dropout 概率，包括嵌入层、编码器和池化器。
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            注意力概率的 dropout 比率。
        max_position_embeddings (`int`, *optional*, defaults to 512):
            此模型可能使用的最大序列长度。通常设置为一个较大的值（例如 512、1024 或 2048）。
        type_vocab_size (`int`, *optional*, defaults to 2):
            在调用 [`QDQBertModel`] 时传递的 `token_type_ids` 的词汇表大小。
        initializer_range (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态初始化器的标准偏差。
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            层归一化层使用的 epsilon。
        is_decoder (`bool`, *optional*, defaults to `False`):
            模型是否用作解码器。如果为 `False`，则模型用作编码器。
        use_cache (`bool`, *optional*, defaults to `True`):
            模型是否应返回最后的键/值注意力（不被所有模型使用）。仅在 `config.is_decoder=True` 时相关。
    
    Examples:
        
    
    >>> from transformers import QDQBertModel, QDQBertConfig
    
    >>> # 初始化一个 QDQBERT bert-base-uncased 风格的配置
    >>> configuration = QDQBertConfig()
    
    >>> # 从 bert-base-uncased 风格的配置初始化模型
    >>> model = QDQBertModel(configuration)
    
    >>> # 访问模型配置
    >>> configuration = model.config
    
    
    # 设置模型类型为 "qdqbert"
    model_type = "qdqbert"
    # 这是 GPT-3.5 模型的配置类的初始化函数
    def __init__(
        self,
        # 词表大小
        vocab_size=30522,
        # 隐藏层维度
        hidden_size=768, 
        # 隐藏层数量
        num_hidden_layers=12,
        # 注意力头数量
        num_attention_heads=12,
        # 前馈网络隐藏层大小
        intermediate_size=3072,
        # 激活函数类型
        hidden_act="gelu",
        # 隐藏层dropout概率
        hidden_dropout_prob=0.1, 
        # 注意力dropout概率
        attention_probs_dropout_prob=0.1,
        # 最大位置编码长度
        max_position_embeddings=512,
        # 句型 token 数量
        type_vocab_size=2,
        # 参数初始化范围
        initializer_range=0.02,
        # LayerNorm 的 epsilon 值
        layer_norm_eps=1e-12,
        # 是否使用 cache
        use_cache=True,
        # padding token id
        pad_token_id=1,
        # 开始token id
        bos_token_id=0,
        # 结束token id
        eos_token_id=2,
        **kwargs,
    ):
        # 调用父类的初始化函数
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
    
        # 赋值各种配置参数
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
```