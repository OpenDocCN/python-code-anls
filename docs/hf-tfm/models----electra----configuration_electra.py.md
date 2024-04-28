# `.\models\electra\configuration_electra.py`

```
# 设置文件编码为utf-8
# 版权声明，包括作者和团队信息以及许可证信息
# 根据Apache License，版本2.0的许可证，除非符合许可证规定，否则不得使用此文件
# 可以在以下链接获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 根据适用法律或书面约定，分发的软件以"原样"分发，不提供任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制信息

""" ELECTRA model configuration"""

# 导入所需的模块
from collections import OrderedDict
from typing import Mapping
# 导入预训练配置相关的类和函数
from ...configuration_utils import PretrainedConfig
# 导入ONNX配置
from ...onnx import OnnxConfig
# 导入日志工具
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# ELECTRA预训练配置存档映射
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

# ELECTRA配置类，用于存储[`ElectraModel`]或[`TFElectraModel`]的配置
# 用指定的参数实例化一个ELECTRA模型，定义模型架构
# 使用默认设置实例化一个配置会产生类似google/electra-small-discriminator架构的配置
# 配置对象继承自[`PretrainedConfig`]，可用于控制模型输出，有关更多信息，请阅读来自[`PretrainedConfig`]的文档

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

    # 模型类型为"electra"
    model_type = "electra"
    # 初始化函数，为Transformer模型的参数设置默认数值
    def __init__(
        self,
        vocab_size=30522,  # 词汇表大小，默认为30522
        embedding_size=128,  # 嵌入向量的维度，默认为128
        hidden_size=256,  # 隐藏层的维度，默认为256
        num_hidden_layers=12,  # 隐藏层的数量，默认为12
        num_attention_heads=4,  # 注意力头的数量，默认为4
        intermediate_size=1024,  # 中间层的维度，默认为1024
        hidden_act="gelu",  # 隐藏层的激活函数，默认为gelu
        hidden_dropout_prob=0.1,  # 隐藏层的dropout比例，默认为0.1
        attention_probs_dropout_prob=0.1,  # 注意力概率的dropout比例，默认为0.1
        max_position_embeddings=512,  # 最大位置编码数，默认为512
        type_vocab_size=2,  # 类型词汇表的大小，默认为2
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-12,  # LayerNorm的epsilon值，默认为1e-12
        summary_type="first",  # 摘要类型，默认为first
        summary_use_proj=True,  # 是否使用摘要投影，默认为True
        summary_activation="gelu",  # 摘要激活函数，默认为gelu
        summary_last_dropout=0.1,  # 最后一层摘要的dropout比例，默认为0.1
        pad_token_id=0,  # 填充token的id，默认为0
        position_embedding_type="absolute",  # 位置编码类型，默认为absolute
        use_cache=True,  # 是否使用缓存，默认为True
        classifier_dropout=None,  # 分类器的dropout比例，默认为None
        **kwargs,  # 其他参数
    ):
        # 调用父类的初始化函数，设置填充token的id
        super().__init__(pad_token_id=pad_token_id, **kwargs)
    
        # 设置各个参数的数值
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
# 定义一个名为ElectraOnnxConfig的类，继承自OnnxConfig类
class ElectraOnnxConfig(OnnxConfig):
    # 定义名为inputs的属性，返回值为Mapping[str, Mapping[int, str>]类型
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务为"multiple-choice"，则设置动态轴为{0: "batch", 1: "choice", 2: "sequence"}，否则为{0: "batch", 1: "sequence"}
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        # 返回有序字典，包含input_ids、attention_mask和token_type_ids作为键，dynamic_axis作为值
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
                ("token_type_ids", dynamic_axis),
            ]
        )
``` 
```