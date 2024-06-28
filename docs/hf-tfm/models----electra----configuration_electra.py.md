# `.\models\electra\configuration_electra.py`

```
# 引入必要的模块和类
from collections import OrderedDict  # 从 collections 模块中引入 OrderedDict 类
from typing import Mapping  # 从 typing 模块中引入 Mapping 类型

# 从相关的模块中导入必要的配置类和函数
from ...configuration_utils import PretrainedConfig  # 从 ...configuration_utils 模块导入 PretrainedConfig 类
from ...onnx import OnnxConfig  # 从 ...onnx 模块导入 OnnxConfig 类
from ...utils import logging  # 从 ...utils 模块导入 logging 工具

# 获取当前模块的 logger 对象
logger = logging.get_logger(__name__)

# 定义 ELECTRA 预训练模型配置文件的 URL 映射
ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/electra-small-generator": "https://huggingface.co/google/electra-small-generator/resolve/main/config.json",
    "google/electra-base-generator": "https://huggingface.co/google/electra-base-generator/resolve/main/config.json",
    "google/electra-large-generator": "https://huggingface.co/google/electra-large-generator/resolve/main/config.json",
    "google/electra-small-discriminator": (
        "https://huggingface.co/google/electra-small-discriminator/resolve/main/config.json"
    ),
    "google/electra-base-discriminator": (
        "https://huggingface.co/google/electra-base-discriminator/resolve/main/config.json"
    ),
    "google/electra-large-discriminator": (
        "https://huggingface.co/google/electra-large-discriminator/resolve/main/config.json"
    ),
}

# ElectraConfig 类，用于存储 ELECTRA 模型的配置信息，继承自 PretrainedConfig 类
class ElectraConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ElectraModel`] or a [`TFElectraModel`]. It is
    used to instantiate a ELECTRA model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the ELECTRA
    [google/electra-small-discriminator](https://huggingface.co/google/electra-small-discriminator) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Examples:

    ```python
    >>> from transformers import ElectraConfig, ElectraModel

    >>> # Initializing a ELECTRA electra-base-uncased style configuration
    >>> configuration = ElectraConfig()

    >>> # Initializing a model (with random weights) from the electra-base-uncased style configuration
    >>> model = ElectraModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "electra"  # 模型类型为 "electra"
    # 初始化函数，用于创建一个新的对象实例，设置各种参数和默认值
    def __init__(
        self,
        vocab_size=30522,  # 词汇表大小，默认为30522
        embedding_size=128,  # 嵌入大小，默认为128
        hidden_size=256,  # 隐藏层大小，默认为256
        num_hidden_layers=12,  # 隐藏层的数量，默认为12
        num_attention_heads=4,  # 注意力头的数量，默认为4
        intermediate_size=1024,  # 中间层大小，默认为1024
        hidden_act="gelu",  # 隐藏层激活函数，默认为GELU
        hidden_dropout_prob=0.1,  # 隐藏层的Dropout概率，默认为0.1
        attention_probs_dropout_prob=0.1,  # 注意力机制的Dropout概率，默认为0.1
        max_position_embeddings=512,  # 最大位置嵌入数，默认为512
        type_vocab_size=2,  # 类型词汇表大小，默认为2
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-12,  # 层归一化的ε值，默认为1e-12
        summary_type="first",  # 摘要类型，默认为"first"
        summary_use_proj=True,  # 是否使用投影进行摘要，默认为True
        summary_activation="gelu",  # 摘要激活函数，默认为GELU
        summary_last_dropout=0.1,  # 最后一层摘要的Dropout概率，默认为0.1
        pad_token_id=0,  # 填充标记的ID，默认为0
        position_embedding_type="absolute",  # 位置嵌入类型，默认为"absolute"
        use_cache=True,  # 是否使用缓存，默认为True
        classifier_dropout=None,  # 分类器的Dropout概率，默认为None
        **kwargs,
    ):
        # 调用父类的初始化方法，设置填充标记ID和其他可选参数
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        # 将参数值分配给对象的相应属性
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_last_dropout = summary_last_dropout
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
# 定义一个名为 ElectraOnnxConfig 的类，继承自 OnnxConfig 类
class ElectraOnnxConfig(OnnxConfig):
    
    # 定义一个 inputs 属性，返回一个映射，其键为字符串，值为映射类型，键为整数，值为字符串
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务类型是 "multiple-choice"
        if self.task == "multiple-choice":
            # 设置动态轴的映射，其中0对应 "batch"，1对应 "choice"，2对应 "sequence"
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            # 否则，设置动态轴的映射，其中0对应 "batch"，1对应 "sequence"
            dynamic_axis = {0: "batch", 1: "sequence"}
        
        # 返回一个有序字典，包含输入名称和相应的动态轴映射
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),         # 输入名称 "input_ids" 对应动态轴映射 dynamic_axis
                ("attention_mask", dynamic_axis),    # 输入名称 "attention_mask" 对应动态轴映射 dynamic_axis
                ("token_type_ids", dynamic_axis),    # 输入名称 "token_type_ids" 对应动态轴映射 dynamic_axis
            ]
        )
```