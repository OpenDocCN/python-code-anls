# `.\models\convbert\configuration_convbert.py`

```
# coding=utf-8
# 定义了文件的编码格式为 UTF-8

# 版权声明和许可证信息，告知使用者此代码的版权信息和许可条件

# 导入必要的库和模块
from collections import OrderedDict
from typing import Mapping

# 从相关模块中导入必要的配置类和工具函数
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义了预训练模型名称到其配置文件地址的映射字典
CONVBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "YituTech/conv-bert-base": "https://huggingface.co/YituTech/conv-bert-base/resolve/main/config.json",
    "YituTech/conv-bert-medium-small": (
        "https://huggingface.co/YituTech/conv-bert-medium-small/resolve/main/config.json"
    ),
    "YituTech/conv-bert-small": "https://huggingface.co/YituTech/conv-bert-small/resolve/main/config.json",
    # 查看所有 ConvBERT 模型的地址：https://huggingface.co/models?filter=convbert
}

# 定义 ConvBertConfig 类，继承自 PretrainedConfig 类
class ConvBertConfig(PretrainedConfig):
    r"""
    这是存储 [`ConvBertModel`] 配置的类。用于根据指定的参数实例化 ConvBERT 模型，定义模型架构。使用默认参数实例化一个配置对象，
    可以得到与 ConvBERT [YituTech/conv-bert-base](https://huggingface.co/YituTech/conv-bert-base) 架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。阅读 [`PretrainedConfig`] 的文档获取更多信息。
    """
    # 定义 ConvBERT 模型的配置类，用于配置模型的各种参数和超参数
    class ConvBertConfig:
    
        # 初始化函数，设置模型的默认词汇大小为 30522
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
            head_ratio=2,
            num_groups=1,
            conv_kernel_size=9,
            classifier_dropout=None  # 分类头部的dropout比例，默认为None
        ):
            self.vocab_size = vocab_size
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
            self.head_ratio = head_ratio
            self.num_groups = num_groups
            self.conv_kernel_size = conv_kernel_size
            self.classifier_dropout = classifier_dropout
    
    # 引入 ConvBertConfig 和 ConvBertModel 类
    from transformers import ConvBertConfig, ConvBertModel
    
    # 创建一个 ConvBERT 模型的配置对象，使用默认配置
    configuration = ConvBertConfig()
    
    # 使用配置对象初始化一个 ConvBERT 模型，权重随机初始化
    model = ConvBertModel(configuration)
    # 访问模型配置信息
    configuration = model.config
# 从 transformers.models.bert.configuration_bert.BertOnnxConfig 复制过来的 ConvBertOnnxConfig 类
class ConvBertOnnxConfig(OnnxConfig):
    # 定义 inputs 属性，返回一个映射，表示模型的输入信息
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是多项选择（multiple-choice）
        if self.task == "multiple-choice":
            # 定义动态轴，包括 batch（批量）、choice（选择）、sequence（序列）
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            # 否则定义动态轴，包括 batch（批量）、sequence（序列）
            dynamic_axis = {0: "batch", 1: "sequence"}
        # 返回一个有序字典，包含输入名称到动态轴映射的信息
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),         # 输入：input_ids，使用动态轴
                ("attention_mask", dynamic_axis),    # 输入：attention_mask，使用动态轴
                ("token_type_ids", dynamic_axis),    # 输入：token_type_ids，使用动态轴
            ]
        )
```