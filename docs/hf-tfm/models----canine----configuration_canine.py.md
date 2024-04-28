# `.\transformers\models\canine\configuration_canine.py`

```
# 导入所需模块和函数
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 定义 CANINE 预训练配置文件映射字典
CANINE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/canine-s": "https://huggingface.co/google/canine-s/resolve/main/config.json",
    # 可以在 https://huggingface.co/models?filter=canine 查看所有 CANINE 模型
}

# 定义 CanineConfig 类，用于存储 CANINE 模型的配置信息
class CanineConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CanineModel`]. It is used to instantiate an
    CANINE model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the CANINE
    [google/canine-s](https://huggingface.co/google/canine-s) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```python
    >>> from transformers import CanineConfig, CanineModel

    >>> # Initializing a CANINE google/canine-s style configuration
    >>> configuration = CanineConfig()

    >>> # Initializing a model (with random weights) from the google/canine-s style configuration
    >>> model = CanineModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    # 指定模型类型为 "canine"
    model_type = "canine"

    # 初始化函数，用于设置 CANINE 模型的各种配置参数
    def __init__(
        self,
        hidden_size=768,  # 隐藏层大小，默认为768
        num_hidden_layers=12,  # 隐藏层层数，默认为12
        num_attention_heads=12,  # 注意力头数，默认为12
        intermediate_size=3072,  # 中间层大小，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为 "gelu"
        hidden_dropout_prob=0.1,  # 隐藏层的 dropout 概率，默认为0.1
        attention_probs_dropout_prob=0.1,  # 注意力概率的 dropout 概率，默认为0.1
        max_position_embeddings=16384,  # 最大位置嵌入数，默认为16384
        type_vocab_size=16,  # 类型词汇表大小，默认为16
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-12,  # 层标准化的 epsilon，默认为1e-12
        pad_token_id=0,  # 填充标记的 ID，默认为0
        bos_token_id=0xE000,  # 起始标记的 ID，默认为0xE000
        eos_token_id=0xE001,  # 结束标记的 ID，默认为0xE001
        downsampling_rate=4,  # 下采样率，默认为4
        upsampling_kernel_size=4,  # 上采样核大小，默认为4
        num_hash_functions=8,  # 哈希函数数量，默认为8
        num_hash_buckets=16384,  # 哈希桶数量，默认为16384
        local_transformer_stride=128,  # 本地 Transformer 步长，默认为128，适合 TPU/XLA 内存对齐。
        **kwargs,
        # 调用父类的构造函数，初始化模型参数
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 设置模型的最大位置嵌入数
        self.max_position_embeddings = max_position_embeddings
        # 设置模型的隐藏层大小
        self.hidden_size = hidden_size
        # 设置模型的隐藏层数
        self.num_hidden_layers = num_hidden_layers
        # 设置模型的注意力头数
        self.num_attention_heads = num_attention_heads
        # 设置模型的中间层大小
        self.intermediate_size = intermediate_size
        # 设置模型的隐藏层激活函数
        self.hidden_act = hidden_act
        # 设置模型的隐藏层 dropout 概率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 设置模型的注意力 dropout 概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 设置模型的初始化范围
        self.initializer_range = initializer_range
        # 设置模型的类型词汇表大小
        self.type_vocab_size = type_vocab_size
        # 设置模型的层归一化 epsilon 值
        self.layer_norm_eps = layer_norm_eps

        # 字符配置：
        # 设置字符下采样率
        self.downsampling_rate = downsampling_rate
        # 设置字符上采样卷积核大小
        self.upsampling_kernel_size = upsampling_kernel_size
        # 设置哈希函数的数量
        self.num_hash_functions = num_hash_functions
        # 设置哈希桶的数量
        self.num_hash_buckets = num_hash_buckets
        # 设置局部 transformer 步长
        self.local_transformer_stride = local_transformer_stride
```