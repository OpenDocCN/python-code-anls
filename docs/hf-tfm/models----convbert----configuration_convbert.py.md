# `.\models\convbert\configuration_convbert.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 版权所有，由 HuggingFace 团队保留所有权利。
#
# 根据 Apache 许可证 2.0 版（“许可证”）获得许可;
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获得许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件根据“原样”基础提供，
# 不提供任何明示或暗示的担保或条件。
# 请参阅许可证了解具体语言的权限和限制。
""" ConvBERT 模型配置"""

# 导入必要的库
from collections import OrderedDict
from typing import Mapping

# 导入预训练配置类和 Onnx 配置
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练配置文件映射
CONVBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "YituTech/conv-bert-base": "https://huggingface.co/YituTech/conv-bert-base/resolve/main/config.json",
    "YituTech/conv-bert-medium-small": (
        "https://huggingface.co/YituTech/conv-bert-medium-small/resolve/main/config.json"
    ),
    "YituTech/conv-bert-small": "https://huggingface.co/YituTech/conv-bert-small/resolve/main/config.json",
    # 查看所有 ConvBERT 模型 https://huggingface.co/models?filter=convbert
}

# ConvBERT 配置类
class ConvBertConfig(PretrainedConfig):
    r"""
    这是一个用于存储 [`ConvBertModel`] 配置的配置类。它用于根据指定的参数实例化 ConvBERT 模型，
    定义模型架构。使用默认值实例化配置将产生与 ConvBERT
    [YituTech/conv-bert-base](https://huggingface.co/YituTech/conv-bert-base) 架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。请阅读
    [`PretrainedConfig`] 的文档以获取更多信息。
    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            ConvBERT 模型的词汇表大小。定义了在调用 [`ConvBertModel`] 或 [`TFConvBertModel`] 时可以表示的不同标记数量，传递给 `inputs_ids`。
        hidden_size (`int`, *optional*, defaults to 768):
            编码器层和池化层的维度。
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Transformer 编码器中的隐藏层数量。
        num_attention_heads (`int`, *optional*, defaults to 12):
            Transformer 编码器中每个注意力层的注意力头数量。
        intermediate_size (`int`, *optional*, defaults to 3072):
            Transformer 编码器中“中间”（即前馈）层的维度。
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            编码器和池化器中的非线性激活函数（函数或字符串）。如果是字符串，支持 `"gelu"`, `"relu"`, `"selu"` 和 `"gelu_new"`。
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            在嵌入层、编码器和池化器中所有全连接层的丢弃概率。
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            注意力概率的丢弃比例。
        max_position_embeddings (`int`, *optional*, defaults to 512):
            此模型可能会使用的最大序列长度。通常设置为较大的值（例如 512、1024 或 2048）。
        type_vocab_size (`int`, *optional*, defaults to 2):
            在调用 [`ConvBertModel`] 或 [`TFConvBertModel`] 时传递给 `token_type_ids` 的词汇表大小。
        initializer_range (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态初始化器的标准偏差。
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            层归一化层使用的 epsilon。
        head_ratio (`int`, *optional*, defaults to 2):
            用于减少注意力头数量的比例 gamma。
        num_groups (`int`, *optional*, defaults to 1):
            ConvBert 模型的分组线性层的组数。
        conv_kernel_size (`int`, *optional*, defaults to 9):
            卷积核的大小。
        classifier_dropout (`float`, *optional*):
            分类头的丢弃比率。

    Example:

    ```python
    >>> from transformers import ConvBertConfig, ConvBertModel

    >>> # 初始化一个 ConvBERT convbert-base-uncased 风格的配置
    >>> configuration = ConvBertConfig()

    >>> # 从 convbert-base-uncased 风格的配置初始化一个（带随机权重）模型
    >>> model = ConvBertModel(configuration)
    # 访问模型配置
    configuration = model.config
    # 设置模型类型为 "convbert"，默认值为 "convbert"
    model_type = "convbert"
    
    # 初始化函数
    def __init__(
        self,
        vocab_size=30522,  # 词汇表大小，默认值为 30522
        hidden_size=768,  # 隐藏层大小，默认值为 768
        num_hidden_layers=12,  # 隐藏层层数，默认值为 12
        num_attention_heads=12,  # 注意力头数，默认值为 12
        intermediate_size=3072,  # 中间层大小，默认值为 3072
        hidden_act="gelu",  # 激活函数，默认值为 "gelu"
        hidden_dropout_prob=0.1,  # 隐藏层丢弃概率，默认值为 0.1
        attention_probs_dropout_prob=0.1,  # 注意力层丢弃概率，默认值为 0.1
        max_position_embeddings=512,  # 最大位置嵌入数，默认值为 512
        type_vocab_size=2,  # 类型词表大小，默认值为 2
        initializer_range=0.02,  # 初始化范围，默认值为 0.02
        layer_norm_eps=1e-12,  # 层归一化 epsilon，默认值为 1e-12
        pad_token_id=1,  # 填充标记 ID，默认值为 1
        bos_token_id=0,  # 开始标记 ID，默认值为 0
        eos_token_id=2,  # 结束标记 ID，默认值为 2
        embedding_size=768,  # 嵌入大小，默认值为 768
        head_ratio=2,  # 头比率，默认值为 2
        conv_kernel_size=9,  # 卷积核大小，默认值为 9
        num_groups=1,  # 分组数量，默认值为 1
        classifier_dropout=None,  # 分类器丢弃概率，默认值为 None
        **kwargs,  # 其余关键字参数
    ):
        # 继承父类的初始化函数
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,  # 使用剩余的关键字参数
        )
    
        # 设置模型的各项参数
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
        self.embedding_size = embedding_size
        self.head_ratio = head_ratio
        self.conv_kernel_size = conv_kernel_size
        self.num_groups = num_groups
        self.classifier_dropout = classifier_dropout
# 从transformers.models.bert.configuration_bert.BertOnnxConfig复制而来的ConvBertOnnxConfig类
class ConvBertOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是多选，则动态轴包括batch、choice、sequence
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        # 如果任务不是多选，则动态轴包括batch、sequence
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        # 返回一个有序的字典，包括input_ids、attention_mask、token_type_ids作为键，对应的动态轴作为值
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
                ("token_type_ids", dynamic_axis),
            ]
        )
```