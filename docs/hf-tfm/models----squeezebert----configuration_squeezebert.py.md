# `.\transformers\models\squeezebert\configuration_squeezebert.py`

```
# 设置文件编码为 utf-8
# 版权声明
# 许可证详细信息
# 仅在遵循许可证的情况下使用此文件
# 请查看许可证的详细信息
# 在适用法律要求或书面同意的情况下，未经许可证允许的情况下分发软件将按"原样"分布，
# 没有任何明示或暗示的保证或条件
# 查看许可证以获取特定语言的功能和限制
""" SqueezeBERT 模型配置 """

# 导入必要的库
from collections import OrderedDict
from typing import Mapping
# 从配置工具中导入预训练配置
from ...configuration_utils import PretrainedConfig
# 导入 Onnx 配置 
from ...onnx import OnnxConfig
# 导入日志工具
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练配置文件地址映射
SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "squeezebert/squeezebert-uncased": (
        "https://huggingface.co/squeezebert/squeezebert-uncased/resolve/main/config.json"
    ),
    "squeezebert/squeezebert-mnli": "https://huggingface.co/squeezebert/squeezebert-mnli/resolve/main/config.json",
    "squeezebert/squeezebert-mnli-headless": (
        "https://huggingface.co/squeezebert/squeezebert-mnli-headless/resolve/main/config.json"
    ),
}

# 定义 SqueezeBERT 配置类，继承自 PretrainedConfig
class SqueezeBertConfig(PretrainedConfig):
    r"""
    这是用于存储 [`SqueezeBertModel`] 配置的配置类。用于根据指定参数实例化 SqueezeBERT 模型，定义模型架构。使用默认配置实例化一个配置将产生类似于 SqueezeBERT [squeezebert/squeezebert-uncased](https://huggingface.co/squeezebert/squeezebert-uncased) 架构的配置。

    配置对象继承自[`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    用法示例:

    ```python
    >>> from transformers import SqueezeBertConfig, SqueezeBertModel

    >>> # 初始化 SqueezeBERT 配置
    >>> configuration = SqueezeBertConfig()

    >>> # 根据上述配置初始化一个模型（带有随机权重）
    >>> model = SqueezeBertModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```

    属性: 
    pretrained_config_archive_map (Dict[str, str]): 包含所有可用预训练检查点的字典。
    """

    # 预训练配置文件地址映射
    pretrained_config_archive_map = SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP
    # 模型类型
    model_type = "squeezebert"
    # 初始化函数，用于创建一个新的BertConfig对象
    def __init__(
        self,
        vocab_size=30522,  # 词汇表的大小，默认为30522
        hidden_size=768,   # 隐藏层大小，默认为768
        num_hidden_layers=12,  # 隐藏层数，默认为12
        num_attention_heads=12,  # 注意力头数，默认为12
        intermediate_size=3072,  # 中间层大小，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为gelu
        hidden_dropout_prob=0.1,  # 隐藏层dropout概率，默认为0.1
        attention_probs_dropout_prob=0.1,  # 注意力机制dropout概率，默认为0.1
        max_position_embeddings=512,  # 最大位置编码长度，默认为512
        type_vocab_size=2,  # 类型词汇表大小，默认为2
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-12,  # 层归一化的epsilon值，默认为1e-12
        pad_token_id=0,  # 填充token的id，默认为0
        embedding_size=768,  # 嵌入层大小，默认为768
        q_groups=4,  # 查询向量组数，默认为4
        k_groups=4,  # 键向量组数，默认为4
        v_groups=4,  # 值向量组数，默认为4
        post_attention_groups=1,  # 注意力输出后组数，默认为1
        intermediate_groups=4,  # 中间层组数，默认为4
        output_groups=4,  # 输出层组数，默认为4
        **kwargs,
    ):
        # 调用父类的初始化函数
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        # 初始化对象属性
        self.vocab_size = vocab_size  # 词汇表大小
        self.hidden_size = hidden_size  # 隐藏层大小
        self.num_hidden_layers = num_hidden_layers  # 隐藏层数
        self.num_attention_heads = num_attention_heads  # 注意力头数
        self.hidden_act = hidden_act  # 隐藏层激活函数
        self.intermediate_size = intermediate_size  # 中间层大小
        self.hidden_dropout_prob = hidden_dropout_prob  # 隐藏层dropout概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob  # 注意力机制dropout概率
        self.max_position_embeddings = max_position_embeddings  # 最大位置编码长度
        self.type_vocab_size = type_vocab_size  # 类型词汇表大小
        self.initializer_range = initializer_range  # 初始化范围
        self.layer_norm_eps = layer_norm_eps  # 层归一化的epsilon值
        self.embedding_size = embedding_size  # 嵌入层大小
        self.q_groups = q_groups  # 查询向量组数
        self.k_groups = k_groups  # 键向量组数
        self.v_groups = v_groups  # 值向量组数
        self.post_attention_groups = post_attention_groups  # 注意力输出后组数
        self.intermediate_groups = intermediate_groups  # 中间层组数
        self.output_groups = output_groups  # 输出层组数
``` 
# 从transformers.models.bert.configuration_bert.BertOnnxConfig复制并修改为SqueezeBertOnnxConfig类
class SqueezeBertOnnxConfig(OnnxConfig):
    # 定义inputs属性，返回输入张量名称到动态轴的映射关系字典
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是多项选择，则设置动态轴为{0: "batch", 1: "choice", 2: "sequence"}
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        # 如果任务不是多项选择，则设置动态轴为{0: "batch", 1: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        # 返回有序字典，包含输入张量名称到动态轴的映射关系
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),  # 输入的词元 ID 张量对应的动态轴
                ("attention_mask", dynamic_axis),  # 输入的注意力掩码张量对应的动态轴
                ("token_type_ids", dynamic_axis),  # 输入的词元类型 ID 张量对应的动态轴
            ]
        )
```