# `.\transformers\models\xlm_roberta\configuration_xlm_roberta.py`

```
# 导入必要的模块和类
from collections import OrderedDict
from typing import Mapping

# 从其他模块导入必要的类和函数
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# XLM-RoBERTa 预训练配置文件映射
XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "xlm-roberta-base": "https://huggingface.co/xlm-roberta-base/resolve/main/config.json",
    "xlm-roberta-large": "https://huggingface.co/xlm-roberta-large/resolve/main/config.json",
    "xlm-roberta-large-finetuned-conll02-dutch": (
        "https://huggingface.co/xlm-roberta-large-finetuned-conll02-dutch/resolve/main/config.json"
    ),
    "xlm-roberta-large-finetuned-conll02-spanish": (
        "https://huggingface.co/xlm-roberta-large-finetuned-conll02-spanish/resolve/main/config.json"
    ),
    "xlm-roberta-large-finetuned-conll03-english": (
        "https://huggingface.co/xlm-roberta-large-finetuned-conll03-english/resolve/main/config.json"
    ),
    "xlm-roberta-large-finetuned-conll03-german": (
        "https://huggingface.co/xlm-roberta-large-finetuned-conll03-german/resolve/main/config.json"
    ),
}


class XLMRobertaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`XLMRobertaModel`] or a [`TFXLMRobertaModel`]. It
    is used to instantiate a XLM-RoBERTa model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the XLMRoBERTa
    [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Examples:

    ```python
    >>> from transformers import XLMRobertaConfig, XLMRobertaModel

    >>> # Initializing a XLM-RoBERTa xlm-roberta-base style configuration
    >>> configuration = XLMRobertaConfig()

    >>> # Initializing a model (with random weights) from the xlm-roberta-base style configuration
    >>> model = XLMRobertaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    # 模型类型
    model_type = "xlm-roberta"
    # 这是 BertConfig 类的构造方法，用于设置 BERT 模型的各种配置参数
    def __init__(
        self,
        # 词汇表大小
        vocab_size=30522,
        # 隐藏层大小
        hidden_size=768,
        # 隐藏层数量
        num_hidden_layers=12,
        # 注意力头的数量
        num_attention_heads=12,
        # 中间层大小
        intermediate_size=3072,
        # 激活函数
        hidden_act="gelu",
        # 隐藏层dropout比例
        hidden_dropout_prob=0.1,
        # 注意力权重dropout比例
        attention_probs_dropout_prob=0.1,
        # 位置embedding的最大序列长度
        max_position_embeddings=512,
        # 句类型embedding的种类数
        type_vocab_size=2,
        # 权重初始化范围
        initializer_range=0.02,
        # LayerNorm的epsilon值
        layer_norm_eps=1e-12,
        # 填充token ID
        pad_token_id=1,
        # 开始token ID
        bos_token_id=0,
        # 结束token ID
        eos_token_id=2,
        # 位置embedding类型
        position_embedding_type="absolute",
        # 是否使用cache
        use_cache=True,
        # 分类器dropout比例
        classifier_dropout=None,
        **kwargs,
    ):
        # 调用父类的构造方法
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        
        # 设置各种配置参数
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
# 从 transformers.models.roberta.configuration_roberta.RobertaOnnxConfig 复制代码，并将其中的 Roberta 替换为 XLMRoberta
class XLMRobertaOnnxConfig(OnnxConfig):
    @property
    # 定义 inputs 属性，返回一个映射对象，将字符串映射到映射对象，表示输入的维度
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务为多选，则动态轴包含批次、选择和序列
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            # 否则，动态轴包含批次和序列
            dynamic_axis = {0: "batch", 1: "sequence"}
        # 返回有序字典，包含 input_ids 和 attention_mask 作为键，对应动态轴作为值
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
            ]
        )
```