# `.\transformers\models\mobilebert\configuration_mobilebert.py`

```py
# 设置编码方式为utf-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，你不得使用此文件，除非符合许可证的规定
# 你可以从以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或经书面同意，否则按"原样"分发的软件将按
# "不带任何担保或条件"基础分发，无论是明示还是暗示的
# 请查看许可证以获取特定语言的模型输出和限制信息
"""MobileBERT模型配置"""
from collections import OrderedDict  # 从collections模块导入OrderedDict类
from typing import Mapping  # 从typing模块导入Mapping类

from ...configuration_utils import PretrainedConfig  # 从...引入模块中的PretrainedConfig类
from ...onnx import OnnxConfig  # 从...引入模块中的OnnxConfig类
from ...utils import logging  # 从...引入模块中的logging类

# 获取logger实例，用于记录日志
logger = logging.get_logger(__name__)

# MobileBERT预训练配置存档映射
MOBILEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/mobilebert-uncased": "https://huggingface.co/google/mobilebert-uncased/resolve/main/config.json"
}


class MobileBertConfig(PretrainedConfig):
    r"""
    这是用于存储[`MobileBertModel`] 或 [`TFMobileBertModel`]配置的配置类。它用于根据指定的参数实例化MobileBERT模型，定义模型架构。
    使用默认值实例化配置将产生类似于MobileBERT[google/mobilebert-uncased](https://huggingface.co/google/mobilebert-uncased)架构的配置。

    配置对象继承自[`PretrainedConfig`]，可用于控制模型输出。阅读[`PretrainedConfig`]的文档以获取更多信息。

    示例：

    ```python
    >>> from transformers import MobileBertConfig, MobileBertModel

    >>> # 初始化一个MobileBERT配置
    >>> configuration = MobileBertConfig()

    >>> # 从上述配置初始化一个（具有随机权重）模型
    >>> model = MobileBertModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py

    属性：pretrained_config_archive_map（Dict[str, str]）：包含所有可用预训练检查点的字典。
    """

    pretrained_config_archive_map = MOBILEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP
    model_type = "mobilebert"
    # 这是一个初始化函数，用于设置模型的各种超参数
    def __init__(
        self,
        # 词汇表大小
        vocab_size=30522,
        # 隐藏层大小
        hidden_size=512,
        # 隐藏层数量
        num_hidden_layers=24,
        # 注意力头的数量
        num_attention_heads=4,
        # 中间层大小
        intermediate_size=512,
        # 隐藏层激活函数
        hidden_act="relu",
        # 隐藏层dropout概率
        hidden_dropout_prob=0.0,
        # 注意力层dropout概率
        attention_probs_dropout_prob=0.1,
        # 最大位置编码长度
        max_position_embeddings=512,
        # 类型 token 的数量
        type_vocab_size=2,
        # 初始化范围
        initializer_range=0.02,
        # Layer Norm 的 epsilon 值
        layer_norm_eps=1e-12,
        # Pad token ID
        pad_token_id=0,
        # 嵌入层大小
        embedding_size=128,
        # 是否使用三元输入
        trigram_input=True,
        # 是否使用bottleneck
        use_bottleneck=True,
        # bottleneck 层大小
        intra_bottleneck_size=128,
        # 是否在注意力层使用bottleneck
        use_bottleneck_attention=False,
        # 是否在 key 和 query 共享 bottleneck
        key_query_shared_bottleneck=True,
        # 前馈网络数量
        num_feedforward_networks=4,
        # 归一化类型
        normalization_type="no_norm",
        # 是否使用分类器激活
        classifier_activation=True,
        # 分类器dropout概率
        classifier_dropout=None,
        **kwargs,
    ):
        # 调用父类的初始化方法
        super().__init__(pad_token_id=pad_token_id, **kwargs)
    
        # 设置模型的各种超参数
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
        self.embedding_size = embedding_size
        self.trigram_input = trigram_input
        self.use_bottleneck = use_bottleneck
        self.intra_bottleneck_size = intra_bottleneck_size
        self.use_bottleneck_attention = use_bottleneck_attention
        self.key_query_shared_bottleneck = key_query_shared_bottleneck
        self.num_feedforward_networks = num_feedforward_networks
        self.normalization_type = normalization_type
        self.classifier_activation = classifier_activation
    
        # 根据是否使用bottleneck设置真实的隐藏层大小
        if self.use_bottleneck:
            self.true_hidden_size = intra_bottleneck_size
        else:
            self.true_hidden_size = hidden_size
    
        # 设置分类器dropout概率
        self.classifier_dropout = classifier_dropout
# 从transformers.models.bert.configuration_bert.BertOnnxConfig中拷贝代码并将Bert更改为MobileBert
class MobileBertOnnxConfig(OnnxConfig):
    # inputs方法的实现
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是"multiple-choice"，则设置dynamic_axis字典的键值对
        # 键为0，值为"batch"；键为1，值为"choice"；键为2，值为"sequence"
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        # 如果任务不是"multiple-choice"，则设置dynamic_axis字典的键值对
        # 键为0，值为"batch"；键为1，值为"sequence"
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        # 返回一个有序字典，包含键值对元组：("input_ids", dynamic_axis)、
        # ("attention_mask", dynamic_axis)和("token_type_ids", dynamic_axis)
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
                ("token_type_ids", dynamic_axis),
            ]
        )
```