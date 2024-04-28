# `.\transformers\models\biogpt\configuration_biogpt.py`

```py
# 设置文件编码为 UTF-8
# 版权声明和许可信息
# 定义了 BioGPT 模型的配置类
# 导入预训练配置的工具函数
from ...configuration_utils import PretrainedConfig
# 导入日志记录工具
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练配置与模型名称映射的字典
BIOGPT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    # BioGPT 模型名称及其对应的配置文件地址
    "microsoft/biogpt": "https://huggingface.co/microsoft/biogpt/resolve/main/config.json",
    # 查看所有 BioGPT 模型的地址
    # https://huggingface.co/models?filter=biogpt
}


# 定义了 BioGPT 模型的配置类
class BioGptConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BioGptModel`]. It is used to instantiate an
    BioGPT model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the BioGPT
    [microsoft/biogpt](https://huggingface.co/microsoft/biogpt) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # 定义 BioGPT 模型的配置类，用于配置模型参数
    Args:
        # 词汇表大小，默认为 42384，定义了在调用 BioGptModel 时传入的 input_ids 可表示的不同 token 数量
        vocab_size (`int`, *optional*, defaults to 42384):
        # 编码器层和池化层的维度，默认为 1024
        hidden_size (`int`, *optional*, defaults to 1024):
        # Transformer 编码器中隐藏层的数量，默认为 24
        num_hidden_layers (`int`, *optional*, defaults to 24):
        # Transformer 编码器中每个注意力层的注意力头数，默认为 16
        num_attention_heads (`int`, *optional*, defaults to 16):
        # Transformer 编码器中“中间”（即前馈）层的维度，默认为 4096
        intermediate_size (`int`, *optional*, defaults to 4096):
        # 编码器和池化器中的非线性激活函数（函数或字符串），默认为 "gelu"
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
        # 嵌入、编码器和池化器中所有全连接层的 dropout 概率，默认为 0.1
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        # 注意力概率的 dropout 比率，默认为 0.1
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        # 此模型可能与之一起使用的最大序列长度，默认设置为较大值（例如，512、1024 或 2048）
        max_position_embeddings (`int`, *optional*, defaults to 1024):
        # 初始化所有权重矩阵的截断正态分布的标准差，默认为 0.02
        initializer_range (`float`, *optional*, defaults to 0.02):
        # 层归一化层使用的 epsilon，默认为 1e-12
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        # 是否通过除以 sqrt(d_model) 来对嵌入进行缩放，默认为 True
        scale_embedding (`bool`, *optional*, defaults to `True`):
        # 模型是否应该返回最后一个键/值注意力（不是所有模型都使用），仅在 `config.is_decoder=True` 时相关
        use_cache (`bool`, *optional*, defaults to `True`):
        # LayerDrop 参数，请参阅有关 LayerDrop 的论文：https://arxiv.org/abs/1909.11556 以获取更多细节，默认为 0.0
        layerdrop (`float`, *optional*, defaults to 0.0):
        # 全连接层内部激活的 dropout 比率，默认为 0.0
        activation_dropout (`float`, *optional*, defaults to 0.0):
        # 填充标记的标识符，默认为 1
        pad_token_id (`int`, *optional*, defaults to 1):
        # 流的开始标记的标识符，默认为 0
        bos_token_id (`int`, *optional*, defaults to 0):
        # 流的结束标记的标识符，默认为 2
        eos_token_id (`int`, *optional*, defaults to 2):

    # 示例代码
    Example:

    ```python
    # 导入 BioGptModel 和 BioGptConfig 类
    >>> from transformers import BioGptModel, BioGptConfig
    ```py
    # 初始化一个 BioGPT 配置，采用 microsoft/biogpt 风格
    >>> configuration = BioGptConfig()

    # 从 microsoft/biogpt 风格的配置初始化一个模型
    >>> model = BioGptModel(configuration)

    # 访问模型配置
    >>> configuration = model.config
    ```
    # 设置模型类型为 "biogpt"
    model_type = "biogpt"

    # 初始化函数，设置模型的各项参数
    def __init__(
        self,
        vocab_size=42384,  # 词汇表大小，默认为42384
        hidden_size=1024,  # 隐藏层大小，默认为1024
        num_hidden_layers=24,  # 隐藏层层数，默认为24
        num_attention_heads=16,  # 注意力头数，默认为16
        intermediate_size=4096,  # 中间层大小，默认为4096
        hidden_act="gelu",  # 隐藏层激活函数，默认为 gelu
        hidden_dropout_prob=0.1,  # 隐藏层dropout概率，默认为0.1
        attention_probs_dropout_prob=0.1,  # 注意力概率dropout概率，默认为0.1
        max_position_embeddings=1024,  # 最大位置嵌入，默认为1024
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-12,  # LayerNorm的epsilon，默认为1e-12
        scale_embedding=True,  # 是否缩放嵌入，默认为True
        use_cache=True,  # 是否使用缓存，默认为True
        layerdrop=0.0,  # LayerDrop概率，默认为0.0
        activation_dropout=0.0,  # 激活函数的dropout概率，默认为0.0
        pad_token_id=1,  # 填充标记ID，默认为1
        bos_token_id=0,  # 起始标记ID，默认为0
        eos_token_id=2,  # 终止标记ID，默认为2
        **kwargs,
    ):
        # 设置模型的各项参数
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.scale_embedding = scale_embedding
        self.use_cache = use_cache
        self.layerdrop = layerdrop
        self.activation_dropout = activation_dropout
        # 调用父类的初始化函数，设置模型的特殊token ID
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
```