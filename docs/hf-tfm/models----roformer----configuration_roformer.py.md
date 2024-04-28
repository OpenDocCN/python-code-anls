# `.\transformers\models\roformer\configuration_roformer.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# Apache 2.0 许可证
# RoFormer 模型配置

# 导入所需模块
from collections import OrderedDict
from typing import Mapping

# 导入自定义模块
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# RoFormer 预训练配置文件映射
ROFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "junnyu/roformer_chinese_small": "https://huggingface.co/junnyu/roformer_chinese_small/resolve/main/config.json",
    "junnyu/roformer_chinese_base": "https://huggingface.co/junnyu/roformer_chinese_base/resolve/main/config.json",
    "junnyu/roformer_chinese_char_small": (
        "https://huggingface.co/junnyu/roformer_chinese_char_small/resolve/main/config.json"
    ),
    "junnyu/roformer_chinese_char_base": (
        "https://huggingface.co/junnyu/roformer_chinese_char_base/resolve/main/config.json"
    ),
    "junnyu/roformer_small_discriminator": (
        "https://huggingface.co/junnyu/roformer_small_discriminator/resolve/main/config.json"
    ),
    "junnyu/roformer_small_generator": (
        "https://huggingface.co/junnyu/roformer_small_generator/resolve/main/config.json"
    ),
    # 查看所有 RoFormer 模型 https://huggingface.co/models?filter=roformer
}


class RoFormerConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`RoFormerModel`] 的配置。根据指定的参数，用于实例化 RoFormer 模型，定义模型架构。
    使用默认值实例化配置将产生类似 RoFormer [junnyu/roformer_chinese_base](https://huggingface.co/junnyu/roformer_chinese_base) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。
    Args:
        vocab_size (`int`, *optional*, defaults to 50000):
            RoFormer模型的词汇表大小。定义了在调用`RoFormerModel`或`TFRoFormerModel`时可以表示的不同token数量。
        embedding_size (`int`, *optional*, defaults to None):
            编码器层和池化层的维度。如果未提供，则默认为`hidden_size`。
        hidden_size (`int`, *optional*, defaults to 768):
            编码器层和池化层的维度。
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Transformer编码器中的隐藏层数量。
        num_attention_heads (`int`, *optional*, defaults to 12):
            Transformer编码器中每个注意力层的注意力头数量。
        intermediate_size (`int`, *optional*, defaults to 3072):
            Transformer编码器中"intermediate"（即前馈）层的维度。
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            编码器和池化层中的非线性激活函数（函数或字符串）。如果是字符串，则支持`"gelu"`, `"relu"`, `"selu"`和`"gelu_new"`。
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            嵌入层、编码器和池化层中所有全连接层的dropout概率。
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            注意力概率的dropout比率。
        max_position_embeddings (`int`, *optional*, defaults to 1536):
            此模型可能使用的最大序列长度。通常将其设置为较大以防万一（例如，512、1024或1536）。
        type_vocab_size (`int`, *optional*, defaults to 2):
            在调用`RoFormerModel`或`TFRoFormerModel`时传递的`token_type_ids`的词汇表大小。
        initializer_range (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态初始化器的标准差。
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            层归一化层使用的epsilon。
        is_decoder (`bool`, *optional*, defaults to `False`):
            模型是否作为解码器使用。如果`False`，则模型将作为编码器使用。
        use_cache (`bool`, *optional*, defaults to `True`):
            模型是否应返回最后的key/values注意力（并非所有模型都使用）。仅在`config.is_decoder=True`时相关。
        rotary_value (`bool`, *optional*, defaults to `False`):
            是否在值层上应用rotary位置嵌入。

    Example:

    ```python
    >>> from transformers import RoFormerModel, RoFormerConfig
    # 初始化一个 RoFormer junnyu/roformer_chinese_base 风格的配置
    >>> configuration = RoFormerConfig()
    
    # 从上述配置初始化一个模型（权重随机初始化）
    >>> model = RoFormerModel(configuration) 
    
    # 访问模型的配置信息
    >>> configuration = model.config
    
    
    # RoFormerConfig 类的初始化
    def __init__(
        self,
        vocab_size=50000,
        embedding_size=None,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1536,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        rotary_value=False,
        use_cache=True,
        **kwargs,
    ):
        # 调用父类 (PretrainedConfig) 的初始化方法
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        
        # 设置 RoFormerConfig 的属性
        self.vocab_size = vocab_size
        self.embedding_size = hidden_size if embedding_size is None else embedding_size
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
        self.rotary_value = rotary_value
        self.use_cache = use_cache
class RoFormerOnnxConfig(OnnxConfig):
    # 定义 RoFormerOnnxConfig 类，继承自 OnnxConfig 类
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是多项选择
        if self.task == "multiple-choice":
            # 定义动态轴，将维度索引映射到轴的名称
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            # 如果任务不是多项选择，定义动态轴，将维度索引映射到轴的名称
            dynamic_axis = {0: "batch", 1: "sequence"}
        # 重新定义动态轴，将维度索引映射到轴的名称
        dynamic_axis = {0: "batch", 1: "sequence"}
        # 返回输入的有序字典，键是输入名称，值是动态轴
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
                ("token_type_ids", dynamic_axis),
            ]
        )
```