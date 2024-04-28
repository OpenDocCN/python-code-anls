# `.\transformers\models\camembert\configuration_camembert.py`

```
# 导入所需的库
from collections import OrderedDict
from typing import Mapping

# 导入预训练配置的基类
from ...configuration_utils import PretrainedConfig

# 导入ONNX配置
from ...onnx import OnnxConfig

# 导入日志记录工具
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义Camembert预训练模型配置文件的映射字典
CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "camembert-base": "https://huggingface.co/camembert-base/resolve/main/config.json",
    "umberto-commoncrawl-cased-v1": (
        "https://huggingface.co/Musixmatch/umberto-commoncrawl-cased-v1/resolve/main/config.json"
    ),
    "umberto-wikipedia-uncased-v1": (
        "https://huggingface.co/Musixmatch/umberto-wikipedia-uncased-v1/resolve/main/config.json"
    ),
}

# Camembert配置类，用于存储Camembert模型的配置
class CamembertConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`CamembertModel`] or a [`TFCamembertModel`]. It is
    used to instantiate a Camembert model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the Camembert
    [camembert-base](https://huggingface.co/camembert-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```python
    >>> from transformers import CamembertConfig, CamembertModel

    >>> # Initializing a Camembert camembert-base style configuration
    >>> configuration = CamembertConfig()

    >>> # Initializing a model (with random weights) from the camembert-base style configuration
    >>> model = CamembertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    
    # 定义模型类型
    model_type = "camembert"
    # 初始化函数，用于初始化 Transformer 模型的参数
    def __init__(
        self,
        vocab_size=30522,  # 词汇表大小，默认为 BERT 词汇表大小
        hidden_size=768,  # 隐藏层大小，默认为 BERT 隐藏层大小
        num_hidden_layers=12,  # 隐藏层数，默认为 BERT 的层数
        num_attention_heads=12,  # 注意力头数，默认为 BERT 的注意力头数
        intermediate_size=3072,  # 隐藏层中间层的大小，默认为 BERT 的中间层大小
        hidden_act="gelu",  # 隐藏层激活函数，默认为 GELU 激活函数
        hidden_dropout_prob=0.1,  # 隐藏层的 dropout 概率，默认为 0.1
        attention_probs_dropout_prob=0.1,  # 注意力层的 dropout 概率，默认为 0.1
        max_position_embeddings=512,  # 最大位置编码数，默认为 512
        type_vocab_size=2,  # 类型词汇表大小，默认为 2（用于区分句子的类型）
        initializer_range=0.02,  # 参数初始化范围，默认为 0.02
        layer_norm_eps=1e-12,  # LayerNorm 层的 epsilon 值，默认为 1e-12
        pad_token_id=1,  # 填充 token 的 id，默认为 1
        bos_token_id=0,  # 起始 token 的 id，默认为 0
        eos_token_id=2,  # 终止 token 的 id，默认为 2
        position_embedding_type="absolute",  # 位置编码的类型，默认为绝对位置编码
        use_cache=True,  # 是否使用缓存，默认为 True
        classifier_dropout=None,  # 分类器的 dropout，默认为 None
        **kwargs,  # 其他参数
    ):
        # 调用父类的初始化函数，设置填充、起始、终止 token 的 id
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 设置 Transformer 模型的各种参数
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
        self.classifier_dropout = classifier_dropout
```  
# 定义一个名为CamembertOnnxConfig的类，它继承自OnnxConfig类
class CamembertOnnxConfig(OnnxConfig):
    # 定义一个名为inputs的属性，返回一个映射，其中键是字符串，值是映射，映射的键是整数，值是字符串
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是多项选择
        if self.task == "multiple-choice":
            # 创建一个动态轴字典，其中0表示批次(batch)，1表示选择(choice)，2表示序列(sequence)
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            # 如果不是多项选择任务，创建一个动态轴字典，其中0表示批次(batch)，1表示序列(sequence)
            dynamic_axis = {0: "batch", 1: "sequence"}
        # 返回有序字典，包含输入名称到动态轴字典的映射
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),        # 输入名称为input_ids，对应的动态轴字典为dynamic_axis
                ("attention_mask", dynamic_axis),   # 输入名称为attention_mask，对应的动态轴字典为dynamic_axis
            ]
        )
```  
```