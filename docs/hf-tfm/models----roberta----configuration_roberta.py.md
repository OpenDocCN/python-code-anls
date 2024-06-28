# `.\models\roberta\configuration_roberta.py`

```
# 引入必要的模块和类
from collections import OrderedDict  # 导入 OrderedDict 类，用于有序字典操作
from typing import Mapping  # 导入 Mapping 类型提示，用于类型标注

# 从 transformers 包中导入预训练配置类和其他相关功能
from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...onnx import OnnxConfig  # 导入 ONNX 配置类
from ...utils import logging  # 导入日志工具

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 预训练模型配置的映射字典，将模型名称映射到配置文件的 URL
ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "FacebookAI/roberta-base": "https://huggingface.co/FacebookAI/roberta-base/resolve/main/config.json",
    "FacebookAI/roberta-large": "https://huggingface.co/FacebookAI/roberta-large/resolve/main/config.json",
    "FacebookAI/roberta-large-mnli": "https://huggingface.co/FacebookAI/roberta-large-mnli/resolve/main/config.json",
    "distilbert/distilroberta-base": "https://huggingface.co/distilbert/distilroberta-base/resolve/main/config.json",
    "openai-community/roberta-base-openai-detector": "https://huggingface.co/openai-community/roberta-base-openai-detector/resolve/main/config.json",
    "openai-community/roberta-large-openai-detector": "https://huggingface.co/openai-community/roberta-large-openai-detector/resolve/main/config.json",
}

# RoBERTa 的配置类，继承自 PretrainedConfig 类
class RobertaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`RobertaModel`] or a [`TFRobertaModel`]. It is
    used to instantiate a RoBERTa model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the RoBERTa
    [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base) architecture.

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
    ```
    """

    model_type = "roberta"  # 模型类型为 RoBERTa
    # 定义一个类的初始化方法，初始化 Transformer 模型的各种参数和选项
    def __init__(
        self,
        vocab_size=50265,  # 词汇表大小，默认为 50265
        hidden_size=768,   # 隐藏层大小，默认为 768
        num_hidden_layers=12,  # Transformer 模型的隐藏层层数，默认为 12
        num_attention_heads=12,  # 注意力头的数量，默认为 12
        intermediate_size=3072,  # 中间层大小，默认为 3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为 GELU
        hidden_dropout_prob=0.1,  # 隐藏层的 dropout 概率，默认为 0.1
        attention_probs_dropout_prob=0.1,  # 注意力概率的 dropout 概率，默认为 0.1
        max_position_embeddings=512,  # 最大位置嵌入大小，默认为 512
        type_vocab_size=2,  # 类型词汇表大小，默认为 2
        initializer_range=0.02,  # 初始化范围，默认为 0.02
        layer_norm_eps=1e-12,  # 层归一化的 epsilon，默认为 1e-12
        pad_token_id=1,  # 填充标记 ID，默认为 1
        bos_token_id=0,  # 开始序列标记 ID，默认为 0
        eos_token_id=2,  # 结束序列标记 ID，默认为 2
        position_embedding_type="absolute",  # 位置嵌入类型，默认为绝对位置编码
        use_cache=True,  # 是否使用缓存，默认为 True
        classifier_dropout=None,  # 分类器的 dropout，可选参数，默认为 None
        **kwargs,  # 其他关键字参数
    ):
        # 调用父类的初始化方法，设置填充、开始和结束序列标记 ID，以及其他传递的关键字参数
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
    
        # 初始化类的属性
        self.vocab_size = vocab_size  # 设置词汇表大小
        self.hidden_size = hidden_size  # 设置隐藏层大小
        self.num_hidden_layers = num_hidden_layers  # 设置隐藏层数
        self.num_attention_heads = num_attention_heads  # 设置注意力头数
        self.hidden_act = hidden_act  # 设置隐藏层激活函数
        self.intermediate_size = intermediate_size  # 设置中间层大小
        self.hidden_dropout_prob = hidden_dropout_prob  # 设置隐藏层 dropout 概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob  # 设置注意力 dropout 概率
        self.max_position_embeddings = max_position_embeddings  # 设置最大位置嵌入大小
        self.type_vocab_size = type_vocab_size  # 设置类型词汇表大小
        self.initializer_range = initializer_range  # 设置初始化范围
        self.layer_norm_eps = layer_norm_eps  # 设置层归一化的 epsilon
        self.position_embedding_type = position_embedding_type  # 设置位置嵌入类型
        self.use_cache = use_cache  # 设置是否使用缓存
        self.classifier_dropout = classifier_dropout  # 设置分类器的 dropout
# 定义一个继承自 OnnxConfig 的 RobertaOnnxConfig 类，用于配置 ROBERTA 模型的 ONNX 导出设置
class RobertaOnnxConfig(OnnxConfig):
    
    # inputs 属性，返回一个映射，描述了模型输入的结构
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是多项选择，则动态轴包含三个维度：batch、choice、sequence
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            # 否则动态轴只包含两个维度：batch、sequence
            dynamic_axis = {0: "batch", 1: "sequence"}
        
        # 返回一个有序字典，描述了模型输入的名称与对应的动态轴
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),        # 模型输入的 token IDs，使用 dynamic_axis 描述轴
                ("attention_mask", dynamic_axis),   # 模型输入的注意力掩码，使用 dynamic_axis 描述轴
            ]
        )
```