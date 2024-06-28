# `.\models\distilbert\configuration_distilbert.py`

```py
# 导入必要的模块和函数
from collections import OrderedDict  # 导入OrderedDict，用于创建有序字典
from typing import Mapping  # 导入Mapping，用于类型提示

# 从相应的库中导入配置类和工具
from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...onnx import OnnxConfig  # 导入ONNX配置类
from ...utils import logging  # 导入日志工具

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义DistilBERT预训练配置文件的下载映射字典
DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "distilbert-base-uncased": "https://huggingface.co/distilbert-base-uncased/resolve/main/config.json",
    "distilbert-base-uncased-distilled-squad": (
        "https://huggingface.co/distilbert-base-uncased-distilled-squad/resolve/main/config.json"
    ),
    "distilbert-base-cased": "https://huggingface.co/distilbert-base-cased/resolve/main/config.json",
    "distilbert-base-cased-distilled-squad": (
        "https://huggingface.co/distilbert-base-cased-distilled-squad/resolve/main/config.json"
    ),
    "distilbert-base-german-cased": "https://huggingface.co/distilbert-base-german-cased/resolve/main/config.json",
    "distilbert-base-multilingual-cased": (
        "https://huggingface.co/distilbert-base-multilingual-cased/resolve/main/config.json"
    ),
    "distilbert-base-uncased-finetuned-sst-2-english": (
        "https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/config.json"
    ),
}

# 定义DistilBERT配置类，继承自PretrainedConfig
class DistilBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DistilBertModel`] or a [`TFDistilBertModel`]. It
    is used to instantiate a DistilBERT model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the DistilBERT
    [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    pass  # 此类目前为空，仅作为DistilBERT模型配置的基础类定义
    # 定义 DistilBERT 模型的配置类，用于初始化模型参数
    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            DistilBERT 模型的词汇表大小，定义了在调用 [`DistilBertModel`] 或 [`TFDistilBertModel`] 时可以表示的不同令牌数量。
        max_position_embeddings (`int`, *optional*, defaults to 512):
            模型可能使用的最大序列长度。通常设置为一个较大的值（例如 512、1024 或 2048）以防万一。
        sinusoidal_pos_embds (`boolean`, *optional*, defaults to `False`):
            是否使用正弦位置嵌入。
        n_layers (`int`, *optional*, defaults to 6):
            Transformer 编码器中隐藏层的数量。
        n_heads (`int`, *optional*, defaults to 12):
            Transformer 编码器中每个注意力层的注意头数。
        dim (`int`, *optional*, defaults to 768):
            编码器层和池化层的维度。
        hidden_dim (`int`, *optional*, defaults to 3072):
            Transformer 编码器中“中间”（通常称为前馈）层的大小。
        dropout (`float`, *optional*, defaults to 0.1):
            嵌入层、编码器和池化器中所有全连接层的 dropout 概率。
        attention_dropout (`float`, *optional*, defaults to 0.1):
            注意力概率的 dropout 比率。
        activation (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            编码器和池化器中的非线性激活函数（函数或字符串）。支持 "gelu"、"relu"、"silu" 和 "gelu_new"。
        initializer_range (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态初始化器的标准差。
        qa_dropout (`float`, *optional*, defaults to 0.1):
            用于问答模型 [`DistilBertForQuestionAnswering`] 中的 dropout 概率。
        seq_classif_dropout (`float`, *optional*, defaults to 0.2):
            用于序列分类和多选模型 [`DistilBertForSequenceClassification`] 中的 dropout 概率。

    Examples:

    ```
    >>> from transformers import DistilBertConfig, DistilBertModel

    >>> # 初始化一个 DistilBERT 配置
    >>> configuration = DistilBertConfig()

    >>> # 从配置初始化一个带有随机权重的模型
    >>> model = DistilBertModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```

    # 设置模型类型为 "distilbert"，并定义属性映射字典
    model_type = "distilbert"
    attribute_map = {
        "hidden_size": "dim",
        "num_attention_heads": "n_heads",
        "num_hidden_layers": "n_layers",
    }
    # 初始化函数，用于创建一个新的对象实例，并初始化其各项属性
    def __init__(
        self,
        vocab_size=30522,  # 设置词汇表大小，默认为30522
        max_position_embeddings=512,  # 设置最大位置编码长度，默认为512
        sinusoidal_pos_embds=False,  # 是否使用正弦位置编码，默认为False
        n_layers=6,  # Transformer 模型的层数，默认为6层
        n_heads=12,  # 每个多头注意力中的头数，默认为12个头
        dim=768,  # 模型中隐藏层的维度，默认为768
        hidden_dim=4 * 768,  # 隐藏层的维度，默认为4倍的768
        dropout=0.1,  # 模型的全连接层的dropout率，默认为0.1
        attention_dropout=0.1,  # 注意力机制中的dropout率，默认为0.1
        activation="gelu",  # 激活函数的类型，默认为GELU
        initializer_range=0.02,  # 参数初始化的范围，默认为0.02
        qa_dropout=0.1,  # 用于问答任务的dropout率，默认为0.1
        seq_classif_dropout=0.2,  # 序列分类任务中的dropout率，默认为0.2
        pad_token_id=0,  # 填充标记的ID，默认为0
        **kwargs,  # 其他未命名的参数，作为关键字参数传递
    ):
        # 初始化基类的构造函数，将额外参数传递给基类，并设置pad_token_id参数
        super().__init__(**kwargs, pad_token_id=pad_token_id)

        # 将传入的参数逐个赋值给对象的属性
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.sinusoidal_pos_embds = sinusoidal_pos_embds
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation = activation
        self.initializer_range = initializer_range
        self.qa_dropout = qa_dropout
        self.seq_classif_dropout = seq_classif_dropout
# 定义 DistilBertOnnxConfig 类，它继承自 OnnxConfig 类
class DistilBertOnnxConfig(OnnxConfig):

    # 定义 inputs 属性，返回一个映射结构，其中键为字符串，值为映射，映射的键是整数，值是字符串
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        
        # 如果任务为多选题（multiple-choice），则动态轴设置为三个维度的映射
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            # 否则动态轴设置为两个维度的映射
            dynamic_axis = {0: "batch", 1: "sequence"}
        
        # 返回一个有序字典，包含两个键值对，分别对应输入的 "input_ids" 和 "attention_mask"
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),       # input_ids 对应的动态轴
                ("attention_mask", dynamic_axis),  # attention_mask 对应的动态轴
            ]
        )
```