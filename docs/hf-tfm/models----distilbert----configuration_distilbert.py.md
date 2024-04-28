# `.\models\distilbert\configuration_distilbert.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 HuggingFace Inc. 团队、Google AI Language Team 和 Facebook, Inc. 所有
# 根据 Apache 许可证 2.0 版本使用此文件
# 除非符合许可证要求，否则禁止使用此文件
# 可以在下面链接获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
# 如果没有适用法律要求或书面同意，软件将按"原样"分发
# 没有明示或暗示的担保，包括但不限于适销性、特定用途适用性和非侵权性的担保
# 有关更多信息，请参阅许可证

# 从 collections 模块导入 OrderedDict 类
# 从 typing 模块导入 Mapping 类型
# 从配置工具模块导入 PretrainedConfig 类
# 从 ONNX 模块导入 OnnxConfig 类
# 从工具模块导入日志记录器
from collections import OrderedDict
from typing import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging

# 获取 logger 对象，用于记录日志
logger = logging.get_logger(__name__)

# 定义 DistilBERT 预训练配置文件映射字典，键为模型名称，值为配置文件的下载链接
DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "distilbert-base-uncased": "https://huggingface.co/distilbert-base-uncased/resolve/main/config.json",
    "distilbert-base-uncased-distilled-squad": (
        "https://huggingface.co/distilbert-base-uncased-distilled-squad/resolve/main/config.json"
    ),
    "distilbert-base-cased": "https://huggingface.co/distilbert-base-cased/resolve/main/config.json",
    "distilbert-base-cased-distilled-squad": (
        "https://huggingface.co/distilbert-base-cased-distilled-squad/resolve/main/config.json"
    ),
    "distilbert-base-german-cased": "https://huggingface.co/distilbert-base-german-cased/resolve/main/config.json",
    "distilbert-base-multilingual-cased": (
        "https://huggingface.co/distilbert-base-multilingual-cased/resolve/main/config.json"
    ),
    "distilbert-base-uncased-finetuned-sst-2-english": (
        "https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/config.json"
    ),
}

# 定义 DistilBERT 配置类，继承自 PretrainedConfig 类
# 用于存储 DistilBERT 模型的配置信息
class DistilBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DistilBertModel`] or a [`TFDistilBertModel`]. It
    is used to instantiate a DistilBERT model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the DistilBERT
    [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # 定义 DistilBERT 模型的配置参数类，用于初始化 DistilBERT 模型
    class DistilBertConfig:
        def __init__(
            self,
            vocab_size: int = 30522,  # DistilBERT 模型的词汇表大小，默认为 30522
            max_position_embeddings: int = 512,  # 最大序列长度，默认为 512
            sinusoidal_pos_embds: bool = False,  # 是否使用正弦位置编码，默认为 False
            n_layers: int = 6,  # Transformer 编码器中的隐藏层层数，默认为 6
            n_heads: int = 12,  # Transformer 编码器中每个注意力层的注意头数，默认为 12
            dim: int = 768,  # 编码器层和池化层的维度，默认为 768
            hidden_dim: int = 3072,  # Transformer 编码器中 "中间" 层的大小，默认为 3072
            dropout: float = 0.1,  # 所有全连接层中的 dropout 概率，默认为 0.1
            attention_dropout: float = 0.1,  # 注意力概率的 dropout 比例，默认为 0.1
            activation: str = "gelu",  # 编码器和池化器中的非线性激活函数，默认为 "gelu"
            initializer_range: float = 0.02,  # 用于初始化所有权重矩阵的截断正态分布标准差，默认为 0.02
            qa_dropout: float = 0.1,  # 用于问答模型中的 dropout 概率，默认为 0.1
            seq_classif_dropout: float = 0.2,  # 用于序列分类和多选模型中的 dropout 概率，默认为 0.2
        ):
            self.vocab_size = vocab_size
            self.max_position_embeddings = max_position_embeddings
            self.sinusoidal_pos_embds = sinusoidal_pos_embds
            self.n_layers = n_layers
            self.n_heads = n_heads
            self.dim = dim
            self.hidden_dim = hidden_dim
            self.dropout = dropout
            self.attention_dropout = attention_dropout
            self.activation = activation
            self.initializer_range = initializer_range
            self.qa_dropout = qa_dropout
            self.seq_classif_dropout = seq_classif_dropout
    
            self.model_type = "distilbert"  # 模型类型为 DistilBERT
            self.attribute_map = {
                "hidden_size": "dim",  # 对应 Transformer 编码器中的维度
                "num_attention_heads": "n_heads",  # 对应 Transformer 编码器中的注意力头数
                "num_hidden_layers": "n_layers",  # 对应 Transformer 编码器中的隐藏层层数
            }
    # 初始化函数，设置默认参数
    def __init__(
        self,
        vocab_size=30522,  # 词汇表大小，默认为30522
        max_position_embeddings=512,  # 最大位置嵌入数，默认为512
        sinusoidal_pos_embds=False,  # 是否使用正弦位置嵌入，默认为False
        n_layers=6,  # 层数，默认为6
        n_heads=12,  # 头数，默认为12
        dim=768,  # 维度，默认为768
        hidden_dim=4 * 768,  # 隐藏维度，默认为4 * 768
        dropout=0.1,  # 丢弃率，默认为0.1
        attention_dropout=0.1,  # 注意力丢弃率，默认为0.1
        activation="gelu",  # 激活函数，默认为gelu
        initializer_range=0.02,  # 初始化范围，默认为0.02
        qa_dropout=0.1,  # QA层丢弃率，默认为0.1
        seq_classif_dropout=0.2,  # 序列分类层丢弃率，默认为0.2
        pad_token_id=0,  # 填充标记ID，默认为0
        **kwargs,  # 其他参数
    ):
        # 设置属性值
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.sinusoidal_pos_embds = sinusoidal_pos_embds
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation = activation
        self.initializer_range = initializer_range
        self.qa_dropout = qa_dropout
        self.seq_classif_dropout = seq_classif_dropout
        # 调用父类初始化函数，并传入其他参数和填充标记ID
        super().__init__(**kwargs, pad_token_id=pad_token_id)
# 定义一个名为 DistilBertOnnxConfig 的类，继承自 OnnxConfig 类
class DistilBertOnnxConfig(OnnxConfig):
    # 定义 inputs 属性，用于指定模型输入
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是多选题，则动态轴包含 batch、choice 和 sequence
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        # 否则，动态轴包含 batch 和 sequence
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        # 返回一个有序字典，包含输入名称和对应的动态轴
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),        # 输入名称为 input_ids，对应的动态轴为 dynamic_axis 中指定的轴
                ("attention_mask", dynamic_axis),   # 输入名称为 attention_mask，对应的动态轴为 dynamic_axis 中指定的轴
            ]
        )
```