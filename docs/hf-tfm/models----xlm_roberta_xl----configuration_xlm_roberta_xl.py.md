# `.\transformers\models\xlm_roberta_xl\configuration_xlm_roberta_xl.py`

```py
# coding=utf-8
# 版权 2022 年 HuggingFace Inc. 团队所有。
#
# 根据 Apache 许可 2.0 版本（“许可”）获得许可;
# 除非符合许可，否则不得使用此文件。
# 您可以在以下网址获取许可的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 按“原样”分发，不附带任何明示或暗示的保证
# 或特定目的的条件。详细信息请参阅许可证。
""" XLM_ROBERTa_XL 配置"""

from collections import OrderedDict
from typing import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)

# 定义预训练配置文件的映射
XLM_ROBERTA_XL_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/xlm-roberta-xl": "https://huggingface.co/facebook/xlm-roberta-xl/resolve/main/config.json",
    "facebook/xlm-roberta-xxl": "https://huggingface.co/facebook/xlm-roberta-xxl/resolve/main/config.json",
    # 查看所有 XLM-RoBERTa-XL 模型 https://huggingface.co/models?filter=xlm-roberta-xl
}


class XLMRobertaXLConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`XLMRobertaXLModel`] 或 [`TFXLMRobertaXLModel`] 的配置。
    它用于根据指定的参数实例化 XLM_ROBERTA_XL 模型，定义模型架构。
    使用默认值实例化配置将产生与 XLM_ROBERTA_XL [facebook/xlm-roberta-xl](https://huggingface.co/facebook/xlm-roberta-xl) 架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。有关更多信息，请阅读 [`PretrainedConfig`] 的文档。


    示例:

    ```python
    >>> from transformers import XLMRobertaXLConfig, XLMRobertaXLModel

    >>> # 初始化一个 XLM_ROBERTA_XL bert-base-uncased 风格的配置
    >>> configuration = XLMRobertaXLConfig()

    >>> # 从 bert-base-uncased 风格的配置初始化一个（带有随机权重的）模型
    >>> model = XLMRobertaXLModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py"""

    model_type = "xlm-roberta-xl"
    # 初始化函数，设置模型的各种参数
    def __init__(
        self,
        vocab_size=250880,  # 词汇表大小，默认为250880
        hidden_size=2560,   # 隐藏层大小，默认为2560
        num_hidden_layers=36,   # 隐藏层的数量，默认为36
        num_attention_heads=32,   # 注意力头的数量，默认为32
        intermediate_size=10240,   # 中间层大小，默认为10240
        hidden_act="gelu",   # 隐藏层激活函数，默认为gelu
        hidden_dropout_prob=0.1,   # 隐藏层的丢弃概率，默认为0.1
        attention_probs_dropout_prob=0.1,   # 注意力概率的丢弃概率，默认为0.1
        max_position_embeddings=514,   # 最大位置嵌入，默认为514
        type_vocab_size=1,   # 类型词汇表大小，默认为1
        initializer_range=0.02,   # 初始化范围，默认为0.02
        layer_norm_eps=1e-05,   # 层归一化的epsilon值，默认为1e-05
        pad_token_id=1,   # 填充标记的ID，默认为1
        bos_token_id=0,   # 开始标记的ID，默认为0
        eos_token_id=2,   # 结束标记的ID，默认为2
        position_embedding_type="absolute",   # 位置嵌入类型，默认为绝对位置嵌入
        use_cache=True,   # 是否使用缓存，默认为True
        classifier_dropout=None,   # 分类器的丢弃概率，默认为None
        **kwargs,   # 其余关键字参数
    ):
        # 调用父类的初始化函数，设置填充标记、开始标记、结束标记
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size  # 设置模型的词汇表大小
        self.hidden_size = hidden_size  # 设置模型的隐藏层大小
        self.num_hidden_layers = num_hidden_layers  # 设置模型的隐藏层数量
        self.num_attention_heads = num_attention_heads   # 设置模型的注意力头数量
        self.hidden_act = hidden_act   # 设置模型的隐藏层激活函数
        self.intermediate_size = intermediate_size   # 设置模型的中间层大小
        self.hidden_dropout_prob = hidden_dropout_prob   # 设置模型的隐藏层的丢弃概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob   # 设置模型的注意力概率的丢弃概率
        self.max_position_embeddings = max_position_embeddings   # 设置模型的最大位置嵌入
        self.type_vocab_size = type_vocab_size   # 设置模型的类型词汇表大小
        self.initializer_range = initializer_range   # 设置模型的初始化范围
        self.layer_norm_eps = layer_norm_eps   # 设置模型的层归一化的epsilon值
        self.position_embedding_type = position_embedding_type   # 设置模型的位置嵌入类型
        self.use_cache = use_cache   # 设置模型是否使用缓存
        self.classifier_dropout = classifier_dropout   # 设置模型的分类器的丢弃概率
# 从transformers.models.roberta.configuration_roberta.RobertaOnnxConfig复制代码，将Roberta更改为XLMRobertaXL
class XLMRobertaXLOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是多项选择，则动态轴为{0: "batch", 1: "choice", 2: "sequence"}
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            # 否则动态轴为{0: "batch", 1: "sequence"}
            dynamic_axis = {0: "batch", 1: "sequence"}
        # 返回有序字典，包含输入特征及其对应的动态轴
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
            ]
        )
```  
```