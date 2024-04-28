# `.\transformers\models\roberta_prelayernorm\configuration_roberta_prelayernorm.py`

```
# 导入所需模块和库
from collections import OrderedDict
from typing import Mapping
# 导入自定义的配置类和函数
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 存储预训练模型及其配置文件的映射关系
ROBERTA_PRELAYERNORM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "andreasmadsen/efficient_mlm_m0.40": (
        "https://huggingface.co/andreasmadsen/efficient_mlm_m0.40/resolve/main/config.json"
    ),
}

# 这是一个配置类，用于存储RoBERTa-PreLayerNorm模型的配置信息
class RobertaPreLayerNormConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`RobertaPreLayerNormModel`] or a [`TFRobertaPreLayerNormModel`]. It is
    used to instantiate a RoBERTa-PreLayerNorm model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the RoBERTa-PreLayerNorm
    [andreasmadsen/efficient_mlm_m0.40](https://huggingface.co/andreasmadsen/efficient_mlm_m0.40) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Examples:

    ```python
    >>> from transformers import RobertaPreLayerNormConfig, RobertaPreLayerNormModel

    >>> # Initializing a RoBERTa-PreLayerNorm configuration
    >>> configuration = RobertaPreLayerNormConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = RobertaPreLayerNormModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    # 模型类型
    model_type = "roberta-prelayernorm"
    # 初始化函数，设置默认参数
    def __init__(
        self,
        vocab_size=50265, 
        # 词汇表大小，默认为50265
        hidden_size=768, 
        # 隐藏层大小，默认为768
        num_hidden_layers=12, 
        # 隐藏层层数，默认为12
        num_attention_heads=12, 
        # 注意力头数，默认为12
        intermediate_size=3072, 
        # 中间层大小，默认为3072
        hidden_act="gelu", 
        # 隐藏层激活函数，默认为gelu
        hidden_dropout_prob=0.1, 
        # 隐藏层dropout概率，默认为0.1
        attention_probs_dropout_prob=0.1, 
        # 注意力概率dropout概率，默认为0.1
        max_position_embeddings=512, 
        # 最大位置嵌入，默认为512
        type_vocab_size=2, 
        # 类型词汇表大小，默认为2
        initializer_range=0.02, 
        # 初始化范围，默认为0.02
        layer_norm_eps=1e-12, 
        # 层归一化epsilon，默认为1e-12
        pad_token_id=1, 
        # 填充标识id，默认为1
        bos_token_id=0, 
        # 开始标识id，默认为0
        eos_token_id=2, 
        # 结束标识id，默认为2
        position_embedding_type="absolute", 
        # 位置嵌入类型，默认为"absolute"
        use_cache=True, 
        # 是否使用缓存，默认为True
        classifier_dropout=None, 
        # 分类器dropout，初始为None
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 设置参数值
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
# 从 transformers.models.roberta.configuration_roberta.RobertaOnnxConfig 中复制代码，并将类名中的 "Roberta" 替换为 "RobertaPreLayerNorm"
class RobertaPreLayerNormOnnxConfig(OnnxConfig):
    # 定义 inputs 属性，返回输入的格式信息，以映射形式返回
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务为 "multiple-choice"，则定义动态轴的映射关系，包括 batch、choice 和 sequence
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        # 如果任务不是 "multiple-choice"，则定义动态轴的映射关系，包括 batch 和 sequence
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        # 返回一个有序字典，包含输入的名称和对应的动态轴映射关系
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),  # 输入的 token IDs，映射为动态轴
                ("attention_mask", dynamic_axis),  # 输入的注意力掩码，映射为动态轴
            ]
        )
```