# `.\transformers\models\mra\configuration_mra.py`

```py
# 设置编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，此文件使用许可
# 除非符合许可，否则不能使用此文件
# 可在以下链接获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 根据适用法律或书面同意，分发的软件是基于“按原样”分发
# 没有任何担保或条件，无论是明示还是暗示
# 有关权限和限制，请参阅许可证
""" MRA 模型配置"""

# 导入 PretrainedConfig 类和 logging 工具
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# MRA 预训练配置映射
MRA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "uw-madison/mra-base-512-4": "https://huggingface.co/uw-madison/mra-base-512-4/resolve/main/config.json",
}

# MraConfig 类继承自 PretrainedConfig 类
class MraConfig(PretrainedConfig):
    r"""
    这是用于存储 [`MraModel`] 配置的配置类。用于根据指定参数实例化 MRA 模型，定义模型架构。使用默认值实例化配置将产生类似于 MRA
    [uw-madison/mra-base-512-4](https://huggingface.co/uw-madison/mra-base-512-4) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读来自 [`PretrainedConfig`] 的文档获取更多信息。
    Args:
        vocab_size (`int`, *optional*, defaults to 50265):
            Mra 模型的词汇大小。定义了调用 [`MraModel`] 时可以表示的不同令牌数量。
        hidden_size (`int`, *optional*, defaults to 768):
            编码器层和池层的维度。
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Transformer 编码器中的隐藏层数量。
        num_attention_heads (`int`, *optional*, defaults to 12):
            Transformer 编码器中每个注意力层的注意力头数。
        intermediate_size (`int`, *optional*, defaults to 3072):
            Transformer 编码器中“中间”（即前馈）层的维度。
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            编码器和池层的非线性激活函数（函数或字符串）。如果是字符串，支持 `"gelu"`, `"relu"`, `"selu"` 和 `"gelu_new"`。
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            嵌入层、编码器和池层中所有全连接层的丢弃概率。
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            注意力概率的丢弃比率。
        max_position_embeddings (`int`, *optional*, defaults to 512):
            该模型可能与之使用的最大序列长度。通常设置为一个较大的值（例如，512 或 1024 或 2048）。
        type_vocab_size (`int`, *optional*, defaults to 1):
            调用 [`MraModel`] 时传递的 `token_type_ids` 的字典大小。
        initializer_range (`float`, *optional*, defaults to 0.02):
            初始化所有权重矩阵的切割正态初始化器的标准差。
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            层归一化层使用的 epsilon。
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            位置嵌入的类型。选择其中之一 `"absolute"`, `"relative_key"`, `"relative_key_query"`.
        block_per_row (`int`, *optional*, defaults to 4):
            用于设置高分辨率尺度的预算。
        approx_mode (`str`, *optional*, defaults to `"full"`):
            控制是否同时使用低分辨率和高分辨率的近似。设置为 `"full"` 表示同时使用低和高分辨率，设置为 `"sparse"` 表示仅使用低分辨率。
        initial_prior_first_n_blocks (`int`, *optional*, defaults to 0):
            高分辨率所用的初始块数量。
        initial_prior_diagonal_n_blocks (`int`, *optional*, defaults to 0):
            高分辨率使用的对角块数。

    Example:

    ```python
    # 从transformers库中导入MraConfig和MraModel类
    >>> from transformers import MraConfig, MraModel

    # 初始化一个Mra uw-madison/mra-base-512-4风格的配置
    >>> configuration = MraConfig()

    # 使用uw-madison/mra-base-512-4风格的配置初始化一个模型（具有随机权重）
    >>> model = MraModel(configuration)

    # 获取模型的配置信息
    >>> configuration = model.config
    ```py"""

    # 定义一个模型类型为"mra"的类
    model_type = "mra"

    # 定义一个初始化方法，包括一系列参数
    def __init__(
        self,
        vocab_size=50265,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=1,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        position_embedding_type="absolute",
        block_per_row=4,
        approx_mode="full",
        initial_prior_first_n_blocks=0,
        initial_prior_diagonal_n_blocks=0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs,
    ):
        # 调用父类的初始化方法，传入一些参数
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 设置各个参数的值
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
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.block_per_row = block_per_row
        self.approx_mode = approx_mode
        self.initial_prior_first_n_blocks = initial_prior_first_n_blocks
        self.initial_prior_diagonal_n_blocks = initial_prior_diagonal_n_blocks
```