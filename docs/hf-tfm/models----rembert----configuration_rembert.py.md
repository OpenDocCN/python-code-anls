# `.\transformers\models\rembert\configuration_rembert.py`

```
# 设置编码为 UTF-8
# 版权声明，保留所有权利
# 根据 Apache 许可证 2.0 进行许可
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则不得分发此软件
# 此软件按“原样”提供，不提供任何明示或暗示的保证或条件
# 请参阅许可证获取特定语言的权限
""" RemBERT 模型配置 """
# 导入所需的模块
from collections import OrderedDict
from typing import Mapping

# 导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 导入 ONNX 配置
from ...onnx import OnnxConfig
# 导入日志记录工具
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# RemBERT 预训练配置文件存档映射
REMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/rembert": "https://huggingface.co/google/rembert/resolve/main/config.json",
    # 查看所有 RemBERT 模型，请访问 https://huggingface.co/models?filter=rembert
}


# RemBERT 配置类，继承自 PretrainedConfig
class RemBertConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`RemBertModel`] 的配置。它用于根据指定的参数实例化一个 RemBERT 模型，
    定义了模型的架构。使用默认值初始化配置将产生与 RemBERT
    [google/rembert](https://huggingface.co/google/rembert) 架构相似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。有关更多信息，请阅读
    [`PretrainedConfig`] 的文档。

    示例:

    ```python
    >>> from transformers import RemBertModel, RemBertConfig

    >>> # 初始化 RemBERT 风格的配置
    >>> configuration = RemBertConfig()

    >>> # 从 RemBERT 风格的配置初始化模型
    >>> model = RemBertModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```"""

    # 模型类型为 "rembert"
    model_type = "rembert"

    # 初始化方法
    def __init__(
        self,
        vocab_size=250300,
        hidden_size=1152,
        num_hidden_layers=32,
        num_attention_heads=18,
        input_embedding_size=256,
        output_embedding_size=1664,
        intermediate_size=4608,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=312,
        eos_token_id=313,
        **kwargs,
    # 调用父类的初始化方法，初始化模型的参数，包括填充、开始和结束标记的 token ID 等
    ):
        # 调用父类的初始化方法，传入填充、开始和结束标记的 token ID，以及其他关键字参数
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 设置词汇表大小
        self.vocab_size = vocab_size
        # 设置输入嵌入大小
        self.input_embedding_size = input_embedding_size
        # 设置输出嵌入大小
        self.output_embedding_size = output_embedding_size
        # 设置最大位置嵌入数
        self.max_position_embeddings = max_position_embeddings
        # 设置隐藏层大小
        self.hidden_size = hidden_size
        # 设置隐藏层数量
        self.num_hidden_layers = num_hidden_layers
        # 设置注意力头数量
        self.num_attention_heads = num_attention_heads
        # 设置中间层大小
        self.intermediate_size = intermediate_size
        # 设置隐藏层激活函数
        self.hidden_act = hidden_act
        # 设置隐藏层 dropout 概率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 设置注意力机制 dropout 概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 设置分类器 dropout 概率
        self.classifier_dropout_prob = classifier_dropout_prob
        # 设置初始化范围
        self.initializer_range = initializer_range
        # 设置类型词汇表大小
        self.type_vocab_size = type_vocab_size
        # 设置层归一化 epsilon
        self.layer_norm_eps = layer_norm_eps
        # 设置是否使用缓存
        self.use_cache = use_cache
        # 设置是否绑定词嵌入
        self.tie_word_embeddings = False
# 定义一个自定义的 RemBertOnnxConfig 类，继承自 OnnxConfig 类
class RemBertOnnxConfig(OnnxConfig):

    # 定义 inputs 属性，返回一个字符串到整数到字符串的映射的字典
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]:
        
        # 如果任务是多项选择的话，定义动态轴包含索引 0: "batch", 1: "choice", 2: "sequence"
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:  # 否则，定义动态轴包含索引 0: "batch", 1: "sequence"
            dynamic_axis = {0: "batch", 1: "sequence"}
        
        # 返回有序字典，包含三个键值对，键为输入名称，值为动态轴字典
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
                ("token_type_ids", dynamic_axis),
            ]
        )

    # 定义 atol_for_validation 属性，返回一个浮点数，用于验证的阈值
    @property
    def atol_for_validation(self) -> float:
        
        # 返回 1e-4 作为验证的阈值
        return 1e-4
```