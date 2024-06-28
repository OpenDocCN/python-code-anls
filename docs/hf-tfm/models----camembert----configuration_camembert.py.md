# `.\models\camembert\configuration_camembert.py`

```
# 指定文件编码为 UTF-8
# 版权声明，包括谷歌 AI 语言团队和 HuggingFace 公司
# 版权声明，包括 NVIDIA 公司
#
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证规定，否则不得使用此文件
# 可在以下链接获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发本软件，无任何明示或暗示的保证或条件
# 详见许可证，获取更多信息
""" CamemBERT configuration"""

# 导入 OrderedDict 类型和 Mapping 接口
from collections import OrderedDict
from typing import Mapping

# 导入预训练配置类 PretrainedConfig
from ...configuration_utils import PretrainedConfig
# 导入 OnnxConfig 类
from ...onnx import OnnxConfig
# 导入日志记录工具
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义预训练模型配置文件的映射字典
CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "almanach/camembert-base": "https://huggingface.co/almanach/camembert-base/resolve/main/config.json",
    "umberto-commoncrawl-cased-v1": (
        "https://huggingface.co/Musixmatch/umberto-commoncrawl-cased-v1/resolve/main/config.json"
    ),
    "umberto-wikipedia-uncased-v1": (
        "https://huggingface.co/Musixmatch/umberto-wikipedia-uncased-v1/resolve/main/config.json"
    ),
}

# CamembertConfig 类继承自 PretrainedConfig 类
class CamembertConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`CamembertModel`] or a [`TFCamembertModel`]. It is
    used to instantiate a Camembert model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the Camembert
    [almanach/camembert-base](https://huggingface.co/almanach/camembert-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```python
    >>> from transformers import CamembertConfig, CamembertModel

    >>> # Initializing a Camembert almanach/camembert-base style configuration
    >>> configuration = CamembertConfig()

    >>> # Initializing a model (with random weights) from the almanach/camembert-base style configuration
    >>> model = CamembertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # 模型类型标识为 "camembert"
    model_type = "camembert"
    # 初始化函数，用于初始化一个 Transformer 模型的配置
    def __init__(
        self,
        vocab_size=30522,                            # 词汇表大小，默认为30522
        hidden_size=768,                             # 隐藏层大小，默认为768
        num_hidden_layers=12,                        # Transformer 模型中的隐藏层层数，默认为12
        num_attention_heads=12,                      # 注意力头的数量，默认为12
        intermediate_size=3072,                      # Feedforward 层的中间大小，默认为3072
        hidden_act="gelu",                           # 隐藏层激活函数，默认为 GELU
        hidden_dropout_prob=0.1,                     # 隐藏层的 dropout 概率，默认为0.1
        attention_probs_dropout_prob=0.1,             # 注意力概率的 dropout 概率，默认为0.1
        max_position_embeddings=512,                 # 最大位置编码的长度，默认为512
        type_vocab_size=2,                           # 类型词汇表的大小，默认为2
        initializer_range=0.02,                      # 参数初始化范围，默认为0.02
        layer_norm_eps=1e-12,                        # 层归一化的 epsilon 值，默认为1e-12
        pad_token_id=1,                              # 填充 token 的 id，默认为1
        bos_token_id=0,                              # 开始 token 的 id，默认为0
        eos_token_id=2,                              # 结束 token 的 id，默认为2
        position_embedding_type="absolute",          # 位置编码类型，默认为绝对位置编码
        use_cache=True,                              # 是否使用缓存，默认为True
        classifier_dropout=None,                     # 分类器的 dropout 概率，默认为None
        **kwargs,                                    # 其他未命名参数
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size                 # 设置模型的词汇表大小
        self.hidden_size = hidden_size               # 设置模型的隐藏层大小
        self.num_hidden_layers = num_hidden_layers   # 设置模型的隐藏层层数
        self.num_attention_heads = num_attention_heads  # 设置模型的注意力头数
        self.hidden_act = hidden_act                 # 设置模型的隐藏层激活函数
        self.intermediate_size = intermediate_size   # 设置模型的中间层大小
        self.hidden_dropout_prob = hidden_dropout_prob  # 设置模型的隐藏层 dropout 概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob  # 设置模型的注意力 dropout 概率
        self.max_position_embeddings = max_position_embeddings  # 设置模型的最大位置编码长度
        self.type_vocab_size = type_vocab_size       # 设置模型的类型词汇表大小
        self.initializer_range = initializer_range   # 设置模型参数初始化范围
        self.layer_norm_eps = layer_norm_eps         # 设置模型的层归一化 epsilon 值
        self.position_embedding_type = position_embedding_type  # 设置模型的位置编码类型
        self.use_cache = use_cache                   # 设置模型是否使用缓存
        self.classifier_dropout = classifier_dropout  # 设置模型的分类器 dropout 概率
# 定义 CamembertOnnxConfig 类，继承自 OnnxConfig 类
class CamembertOnnxConfig(OnnxConfig):
    
    # 定义 inputs 属性，返回一个映射，表示输入数据的结构
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是多项选择（multiple-choice）
        if self.task == "multiple-choice":
            # 定义动态轴的顺序，分别为 batch、choice、sequence
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            # 否则，定义动态轴的顺序，分别为 batch、sequence
            dynamic_axis = {0: "batch", 1: "sequence"}
        
        # 返回有序字典，表示输入的名称和对应的动态轴
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),         # 输入数据的标识符
                ("attention_mask", dynamic_axis),    # 输入数据的注意力掩码
            ]
        )
```