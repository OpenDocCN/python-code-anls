# `.\models\canine\configuration_canine.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，指出代码的所有权和授权信息
# 根据 Apache 2.0 许可证使用本代码
# 可以在符合许可证的前提下使用本代码
# 可以通过许可证链接获取许可证副本
# 如果适用法律要求或书面同意，本软件以"原样"分发，不提供任何明示或暗示的保证或条件
# 详见许可证获取更多信息
""" CANINE 模型配置"""

# 从配置工具中导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 从工具集中导入日志记录器
from ...utils import logging

# 获取 logger 对象用于记录日志
logger = logging.get_logger(__name__)

# CANINE 预训练模型配置文件映射字典，将模型名称映射到配置文件的 URL 地址
CANINE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/canine-s": "https://huggingface.co/google/canine-s/resolve/main/config.json",
    # 查看所有 CANINE 模型的列表 https://huggingface.co/models?filter=canine
}

# CanineConfig 类继承自 PretrainedConfig，用于存储 CANINE 模型的配置信息
class CanineConfig(PretrainedConfig):
    r"""
    这是配置类，用于存储 [`CanineModel`] 的配置信息。根据指定的参数实例化 CANINE 模型，定义模型架构。
    使用默认配置实例化将会产生类似于 CANINE [google/canine-s](https://huggingface.co/google/canine-s) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。阅读 [`PretrainedConfig`] 的文档获取更多信息。

    Example:

    ```
    >>> from transformers import CanineConfig, CanineModel

    >>> # 初始化一个 CANINE google/canine-s 风格的配置
    >>> configuration = CanineConfig()

    >>> # 使用该配置初始化一个（随机权重）模型，使用 google/canine-s 风格的配置
    >>> model = CanineModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```
    """

    # 模型类型为 "canine"
    model_type = "canine"

    # 构造函数，用于初始化 CANINE 模型的配置参数
    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=16384,
        type_vocab_size=16,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        bos_token_id=0xE000,
        eos_token_id=0xE001,
        downsampling_rate=4,
        upsampling_kernel_size=4,
        num_hash_functions=8,
        num_hash_buckets=16384,
        local_transformer_stride=128,  # 适合 TPU/XLA 内存对齐的良好值
        **kwargs,
        ):
        # 调用父类的构造方法，初始化模型的参数，包括填充、开头和结尾的特殊token id等
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 设置模型的最大位置嵌入数
        self.max_position_embeddings = max_position_embeddings
        # 设置模型的隐藏层大小
        self.hidden_size = hidden_size
        # 设置模型的隐藏层数量
        self.num_hidden_layers = num_hidden_layers
        # 设置模型的注意力头数量
        self.num_attention_heads = num_attention_heads
        # 设置模型的中间层大小
        self.intermediate_size = intermediate_size
        # 设置模型的隐藏层激活函数类型
        self.hidden_act = hidden_act
        # 设置模型的隐藏层dropout概率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 设置模型的注意力层dropout概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 设置模型的初始化范围
        self.initializer_range = initializer_range
        # 设置模型的类型词汇大小
        self.type_vocab_size = type_vocab_size
        # 设置模型的层归一化epsilon值
        self.layer_norm_eps = layer_norm_eps

        # 字符特征配置:
        # 设置字符特征的下采样率
        self.downsampling_rate = downsampling_rate
        # 设置字符特征的上采样卷积核大小
        self.upsampling_kernel_size = upsampling_kernel_size
        # 设置哈希函数的数量
        self.num_hash_functions = num_hash_functions
        # 设置哈希桶的数量
        self.num_hash_buckets = num_hash_buckets
        # 设置本地transformer的步长
        self.local_transformer_stride = local_transformer_stride
```