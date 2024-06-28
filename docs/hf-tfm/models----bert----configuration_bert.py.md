# `.\models\bert\configuration_bert.py`

```
# coding=utf-8
# 声明版权和许可信息

""" BERT模型配置 """
# 导入必要的模块
from collections import OrderedDict  # 导入OrderedDict类，用于创建有序字典
from typing import Mapping  # 导入Mapping类型提示

from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...onnx import OnnxConfig  # 导入ONNX配置
from ...utils import logging  # 导入日志工具

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# BERT预训练模型配置文件映射
BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google-bert/bert-base-uncased": "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/config.json",
    "google-bert/bert-large-uncased": "https://huggingface.co/google-bert/bert-large-uncased/resolve/main/config.json",
    "google-bert/bert-base-cased": "https://huggingface.co/google-bert/bert-base-cased/resolve/main/config.json",
    "google-bert/bert-large-cased": "https://huggingface.co/google-bert/bert-large-cased/resolve/main/config.json",
    "google-bert/bert-base-multilingual-uncased": "https://huggingface.co/google-bert/bert-base-multilingual-uncased/resolve/main/config.json",
    "google-bert/bert-base-multilingual-cased": "https://huggingface.co/google-bert/bert-base-multilingual-cased/resolve/main/config.json",
    "google-bert/bert-base-chinese": "https://huggingface.co/google-bert/bert-base-chinese/resolve/main/config.json",
    "google-bert/bert-base-german-cased": "https://huggingface.co/google-bert/bert-base-german-cased/resolve/main/config.json",
    "google-bert/bert-large-uncased-whole-word-masking": (
        "https://huggingface.co/google-bert/bert-large-uncased-whole-word-masking/resolve/main/config.json"
    ),
    "google-bert/bert-large-cased-whole-word-masking": (
        "https://huggingface.co/google-bert/bert-large-cased-whole-word-masking/resolve/main/config.json"
    ),
    "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad": (
        "https://huggingface.co/google-bert/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/config.json"
    ),
    "google-bert/bert-large-cased-whole-word-masking-finetuned-squad": (
        "https://huggingface.co/google-bert/bert-large-cased-whole-word-masking-finetuned-squad/resolve/main/config.json"
    ),
    "google-bert/bert-base-cased-finetuned-mrpc": "https://huggingface.co/google-bert/bert-base-cased-finetuned-mrpc/resolve/main/config.json",
    # 定义一个字典，将不同的BERT模型名称映射到其对应的配置文件URL
    "google-bert/bert-base-german-dbmdz-cased": "https://huggingface.co/google-bert/bert-base-german-dbmdz-cased/resolve/main/config.json",
    "google-bert/bert-base-german-dbmdz-uncased": "https://huggingface.co/google-bert/bert-base-german-dbmdz-uncased/resolve/main/config.json",
    "cl-tohoku/bert-base-japanese": "https://huggingface.co/cl-tohoku/bert-base-japanese/resolve/main/config.json",
    # 使用整词掩码技术的日语BERT模型的配置文件URL
    "cl-tohoku/bert-base-japanese-whole-word-masking": (
        "https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking/resolve/main/config.json"
    ),
    # 使用字符级整词掩码技术的日语BERT模型的配置文件URL
    "cl-tohoku/bert-base-japanese-char": (
        "https://huggingface.co/cl-tohoku/bert-base-japanese-char/resolve/main/config.json"
    ),
    # 使用字符级整词掩码技术的日语BERT模型的配置文件URL
    "cl-tohoku/bert-base-japanese-char-whole-word-masking": (
        "https://huggingface.co/cl-tohoku/bert-base-japanese-char-whole-word-masking/resolve/main/config.json"
    ),
    # 芬兰语大小写BERT模型的配置文件URL
    "TurkuNLP/bert-base-finnish-cased-v1": (
        "https://huggingface.co/TurkuNLP/bert-base-finnish-cased-v1/resolve/main/config.json"
    ),
    # 芬兰语小写BERT模型的配置文件URL
    "TurkuNLP/bert-base-finnish-uncased-v1": (
        "https://huggingface.co/TurkuNLP/bert-base-finnish-uncased-v1/resolve/main/config.json"
    ),
    # 荷兰语大小写BERT模型的配置文件URL
    "wietsedv/bert-base-dutch-cased": "https://huggingface.co/wietsedv/bert-base-dutch-cased/resolve/main/config.json",
    # 查看所有BERT模型的链接，可以在这里找到更多信息：https://huggingface.co/models?filter=bert
# 类定义：BertConfig，继承自PretrainedConfig，用于存储BERT模型的配置信息
class BertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BertModel`] or a [`TFBertModel`]. It is used to
    instantiate a BERT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the BERT
    [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    
    Examples:  # 示例代码

    ```python
    >>> from transformers import BertConfig, BertModel

    >>> # Initializing a BERT google-bert/bert-base-uncased style configuration
    >>> configuration = BertConfig()

    >>> # Initializing a model (with random weights) from the google-bert/bert-base-uncased style configuration
    >>> model = BertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "bert"  # 模型类型设置为"bert"

    # 初始化函数，设置Bert模型的各种配置参数
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
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)  # 调用父类的初始化函数

        # 设置BertConfig的各项配置参数
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


# 类定义：BertOnnxConfig，继承自OnnxConfig
class BertOnnxConfig(OnnxConfig):
    @property
    # 定义一个方法 inputs，返回一个字典结构
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务类型是多选题
        if self.task == "multiple-choice":
            # 定义动态轴的顺序，包括批次、选择和序列
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            # 否则，定义动态轴的顺序，包括批次和序列
            dynamic_axis = {0: "batch", 1: "sequence"}
        # 返回一个有序字典，包含输入数据的名称和对应的动态轴顺序
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),         # 输入的标识符 ID，使用动态轴顺序
                ("attention_mask", dynamic_axis),    # 注意力掩码，使用动态轴顺序
                ("token_type_ids", dynamic_axis),    # 令牌类型 ID，使用动态轴顺序
            ]
        )
```