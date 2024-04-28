# `.\transformers\models\bert\configuration_bert.py`

```
# 导入必要的模块和函数
from collections import OrderedDict
from typing import Mapping

# 导入预训练配置的基类
from ...configuration_utils import PretrainedConfig
# 导入 ONNX 配置
from ...onnx import OnnxConfig
# 导入日志记录工具
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练 BERT 模型配置文件的存档映射，将模型名称映射到其配置文件的 URL
BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "bert-base-uncased": "https://huggingface.co/bert-base-uncased/resolve/main/config.json",
    "bert-large-uncased": "https://huggingface.co/bert-large-uncased/resolve/main/config.json",
    "bert-base-cased": "https://huggingface.co/bert-base-cased/resolve/main/config.json",
    "bert-large-cased": "https://huggingface.co/bert-large-cased/resolve/main/config.json",
    "bert-base-multilingual-uncased": "https://huggingface.co/bert-base-multilingual-uncased/resolve/main/config.json",
    "bert-base-multilingual-cased": "https://huggingface.co/bert-base-multilingual-cased/resolve/main/config.json",
    "bert-base-chinese": "https://huggingface.co/bert-base-chinese/resolve/main/config.json",
    "bert-base-german-cased": "https://huggingface.co/bert-base-german-cased/resolve/main/config.json",
    "bert-large-uncased-whole-word-masking": (
        "https://huggingface.co/bert-large-uncased-whole-word-masking/resolve/main/config.json"
    ),
    "bert-large-cased-whole-word-masking": (
        "https://huggingface.co/bert-large-cased-whole-word-masking/resolve/main/config.json"
    ),
    "bert-large-uncased-whole-word-masking-finetuned-squad": (
        "https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/config.json"
    ),
    "bert-large-cased-whole-word-masking-finetuned-squad": (
        "https://huggingface.co/bert-large-cased-whole-word-masking-finetuned-squad/resolve/main/config.json"
    ),
    "bert-base-cased-finetuned-mrpc": "https://huggingface.co/bert-base-cased-finetuned-mrpc/resolve/main/config.json",
    "bert-base-german-dbmdz-cased": "https://huggingface.co/bert-base-german-dbmdz-cased/resolve/main/config.json",
    "bert-base-german-dbmdz-uncased": "https://huggingface.co/bert-base-german-dbmdz-uncased/resolve/main/config.json",
    "cl-tohoku/bert-base-japanese": "https://huggingface.co/cl-tohoku/bert-base-japanese/resolve/main/config.json",
}
    "cl-tohoku/bert-base-japanese-whole-word-masking": (
        "https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking/resolve/main/config.json"
    ),
    # 日语 BERT 模型，使用整词掩码的配置文件链接
    "cl-tohoku/bert-base-japanese-char": (
        "https://huggingface.co/cl-tohoku/bert-base-japanese-char/resolve/main/config.json"
    ),
    # 日语 BERT 模型，使用字符级别的配置文件链接
    "cl-tohoku/bert-base-japanese-char-whole-word-masking": (
        "https://huggingface.co/cl-tohoku/bert-base-japanese-char-whole-word-masking/resolve/main/config.json"
    ),
    # 日语 BERT 模型，使用字符级别和整词掩码的配置文件链接
    "TurkuNLP/bert-base-finnish-cased-v1": (
        "https://huggingface.co/TurkuNLP/bert-base-finnish-cased-v1/resolve/main/config.json"
    ),
    # 芬兰语 BERT 模型，使用大小写的配置文件链接
    "TurkuNLP/bert-base-finnish-uncased-v1": (
        "https://huggingface.co/TurkuNLP/bert-base-finnish-uncased-v1/resolve/main/config.json"
    ),
    # 芬兰语 BERT 模型，不区分大小写的配置文件链接
    "wietsedv/bert-base-dutch-cased": "https://huggingface.co/wietsedv/bert-base-dutch-cased/resolve/main/config.json",
    # 荷兰语 BERT 模型，使用大小写的配置文件链接
    # 查看所有 BERT 模型，请访问 https://huggingface.co/models?filter=bert
# BertConfig 类用于存储 BERT 模型的配置信息，可以用来实例化 BERT 模型并定义其架构。
# 它是一个配置类，继承自 PretrainedConfig，用于控制模型的输出。
# 配置对象可以用于控制模型输出，可以阅读 PretrainedConfig 的文档获取更多信息。

class BertConfig(PretrainedConfig):
    # 模型类型为 "bert"
    model_type = "bert"

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
        # 调用父类 PretrainedConfig 的构造函数
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        # 设置配置参数
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

# BertOnnxConfig 类用于配置 ONNX 格式的 BERT 模型
class BertOnnxConfig(OnnxConfig):
    # inputs 属性定义了输入的格式
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是 "multiple-choice"，动态轴为 {0: "batch", 1: "choice", 2: "sequence"}，否则为 {0: "batch", 1: "sequence"}
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        # 返回输入格式的有序字典
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
                ("token_type_ids", dynamic_axis),
            ]
        )
```