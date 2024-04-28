# `.\transformers\models\roberta\configuration_roberta.py`

```
# coding=utf-8
# 版权信息

# 使用 OrderedDict 和 Mapping 模块
from collections import OrderedDict
from typing import Mapping

# 引入相关模块和类
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 RoBERTa 预训练模型的配置文件链接字典
ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "roberta-base": "https://huggingface.co/roberta-base/resolve/main/config.json",
    "roberta-large": "https://huggingface.co/roberta-large/resolve/main/config.json",
    "roberta-large-mnli": "https://huggingface.co/roberta-large-mnli/resolve/main/config.json",
    "distilroberta-base": "https://huggingface.co/distilroberta-base/resolve/main/config.json",
    "roberta-base-openai-detector": "https://huggingface.co/roberta-base-openai-detector/resolve/main/config.json",
    "roberta-large-openai-detector": "https://huggingface.co/roberta-large-openai-detector/resolve/main/config.json",
}

# RoBERTa 配置类，继承自 PretrainedConfig
class RobertaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`RobertaModel`] or a [`TFRobertaModel`]. It is
    used to instantiate a RoBERTa model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the RoBERTa
    [roberta-base](https://huggingface.co/roberta-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Examples:

    ```python
    >>> from transformers import RobertaConfig, RobertaModel

    >>> # Initializing a RoBERTa configuration
    >>> configuration = RobertaConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = RobertaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
	
    # 模型类型
    model_type = "roberta"
    # 初始化函数，用于创建一个新的Transformer模型实例
    def __init__(
        self,
        # 词汇表大小，默认为50265
        vocab_size=50265,
        # 隐藏层大小，默认为768
        hidden_size=768,
        # 隐藏层的数量，默认为12
        num_hidden_layers=12,
        # 注意力头的数量，默认为12
        num_attention_heads=12,
        # 中间层大小，默认为3072
        intermediate_size=3072,
        # 隐藏层激活函数，默认为"geli"
        hidden_act="gelu",
        # 隐藏层的dropout概率，默认为0.1
        hidden_dropout_prob=0.1,
        # 注意力机制的dropout概率，默认为0.1
        attention_probs_dropout_prob=0.1,
        # 最大位置嵌入大小，默认为512
        max_position_embeddings=512,
        # 类型词汇表大小，默认为2
        type_vocab_size=2,
        # 初始化范围，默认为0.02
        initializer_range=0.02,
        # 层归一化的epsilon值，默认为1e-12
        layer_norm_eps=1e-12,
        # 填充标记的ID，默认为1
        pad_token_id=1,
        # 开始标记的ID，默认为0
        bos_token_id=0,
        # 结束标记的ID，默认为2
        eos_token_id=2,
        # 位置嵌入类型，默认为"absolute"
        position_embedding_type="absolute",
        # 是否使用缓存，默认为True
        use_cache=True,
        # 分类器的dropout，默认为None
        classifier_dropout=None,
        # 其它参数
        **kwargs,
    ):
        # 调用父类初始化函数，设置特殊的标记ID
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 设置Transformer模型的参数
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
# RobertaOnnxConfig 是一个继承自 OnnxConfig 的类
class RobertaOnnxConfig(OnnxConfig):
    # 定义一个 property 属性 inputs，用于返回输入张量的信息
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务类型是 "multiple-choice"，则输入张量的动态轴为 batch、choice 和 sequence
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        # 否则，输入张量的动态轴为 batch 和 sequence
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        # 返回一个有序字典，包含输入张量的名称和对应的动态轴信息
        return OrderedDict(
            [
                # 输入 ID 张量
                ("input_ids", dynamic_axis),
                # 注意力掩码张量
                ("attention_mask", dynamic_axis),
            ]
        )
```