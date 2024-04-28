# `.\transformers\models\albert\configuration_albert.py`

```
# 设置文件编码为 utf-8
# 版权声明，包括作者和团队信息
# 版权声明，版权所有，保留所有权利
# 根据 Apache 许可证 2.0 版本，只有在遵守许可证的情况下才能使用此文件
# 可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
""" ALBERT 模型配置 """
# 导入必要的库
from collections import OrderedDict
from typing import Mapping
# 导入预训练配置和 ONNX 配置
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig

# ALBERT 预训练配置映射
ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "albert-base-v1": "https://huggingface.co/albert-base-v1/resolve/main/config.json",
    "albert-large-v1": "https://huggingface.co/albert-large-v1/resolve/main/config.json",
    "albert-xlarge-v1": "https://huggingface.co/albert-xlarge-v1/resolve/main/config.json",
    "albert-xxlarge-v1": "https://huggingface.co/albert-xxlarge-v1/resolve/main/config.json",
    "albert-base-v2": "https://huggingface.co/albert-base-v2/resolve/main/config.json",
    "albert-large-v2": "https://huggingface.co/albert-large-v2/resolve/main/config.json",
    "albert-xlarge-v2": "https://huggingface.co/albert-xlarge-v2/resolve/main/config.json",
    "albert-xxlarge-v2": "https://huggingface.co/albert-xxlarge-v2/resolve/main/config.json",
}

# ALBERT 配置类，继承自预训练配置类
class AlbertConfig(PretrainedConfig):
    r"""
    这是用于存储 [`AlbertModel`] 或 [`TFAlbertModel`] 配置的配置类。根据指定的参数实例化 ALBERT 模型，定义模型架构。
    使用默认值实例化配置将产生类似于 ALBERT [albert-xxlarge-v2](https://huggingface.co/albert-xxlarge-v2) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    示例:

    ```python
    >>> from transformers import AlbertConfig, AlbertModel

    >>> # 初始化一个 ALBERT-xxlarge 风格的配置
    >>> albert_xxlarge_configuration = AlbertConfig()

    >>> # 初始化一个 ALBERT-base 风格的配置
    >>> albert_base_configuration = AlbertConfig(
    ...     hidden_size=768,
    ...     num_attention_heads=12,
    ...     intermediate_size=3072,
    ... )

    >>> # 从 ALBERT-base 风格的配置初始化一个模型（���有随机权重）
    >>> model = AlbertModel(albert_xxlarge_configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```
    """
    model_type = "albert"

    # 定义 ALBERT 模型的初始化方法，设置各种参数
    def __init__(
        self,
        vocab_size=30000,  # 词汇表大小，默认为 30000
        embedding_size=128,  # 嵌入层维度，默认为 128
        hidden_size=4096,  # 隐藏层大小，默认为 4096
        num_hidden_layers=12,  # 隐藏层的数量，默认为 12
        num_hidden_groups=1,  # 隐藏层的组数，默认为 1
        num_attention_heads=64,  # 注意力头的数量，默认为 64
        intermediate_size=16384,  # 中间层大小，默认为 16384
        inner_group_num=1,  # 内部组数量，默认为 1
        hidden_act="gelu_new",  # 隐藏层激活函数，默认为 "gelu_new"
        hidden_dropout_prob=0,  # 隐藏层的 dropout 概率，默认为 0
        attention_probs_dropout_prob=0,  # 注意力概率 dropout 概率，默认为 0
        max_position_embeddings=512,  # 最大位置嵌入，默认为 512
        type_vocab_size=2,  # 类型词汇表大小，默认为 2
        initializer_range=0.02,  # 初始化范围，默认为 0.02
        layer_norm_eps=1e-12,  # 层归一化的 epsilon，默认为 1e-12
        classifier_dropout_prob=0.1,  # 分类器的 dropout 概率，默认为 0.1
        position_embedding_type="absolute",  # 位置嵌入类型，默认为 "absolute"
        pad_token_id=0,  # 填充 token 的 id，默认为 0
        bos_token_id=2,  # 开始 token 的 id，默认为 2
        eos_token_id=3,  # 结束 token 的 id，默认为 3
        **kwargs,
    ):
        # 调用父类的初始化方法，设置填充、开始和结束 token 的 id
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 设置模型的各种参数
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_groups = num_hidden_groups
        self.num_attention_heads = num_attention_heads
        self.inner_group_num = inner_group_num
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.classifier_dropout_prob = classifier_dropout_prob
        self.position_embedding_type = position_embedding_type
# 从transformers.models.bert.configuration_bert.BertOnnxConfig复制并将Robert替换为Albert
class AlbertOnnxConfig(OnnxConfig):
    # 定义inputs属性，返回输入的映射关系
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是多项选择，则动态轴为{0: "batch", 1: "choice", 2: "sequence"}
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        # 否则动态轴为{0: "batch", 1: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        # 返回有序字典，包含输入名称和对应的动态轴
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
                ("token_type_ids", dynamic_axis),
            ]
        )
```