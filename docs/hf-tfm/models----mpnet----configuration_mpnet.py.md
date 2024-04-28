# `.\transformers\models\mpnet\configuration_mpnet.py`

```
# 导入必要的模块和类
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# MPNet 预训练模型配置的 URL 映射字典
MPNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/mpnet-base": "https://huggingface.co/microsoft/mpnet-base/resolve/main/config.json",
}

# MPNet 模型配置类，继承自 PretrainedConfig
class MPNetConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`MPNetModel`] 或 [`TFMPNetModel`] 的配置。根据指定的参数实例化一个 MPNet 模型，定义模型架构。
    使用默认值实例化一个配置会产生与 MPNet [microsoft/mpnet-base](https://huggingface.co/microsoft/mpnet-base) 架构相似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。
    ```
    Args:
        vocab_size (`int`, *optional*, defaults to 30527):
            Vocabulary size of the MPNet model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`MPNetModel`] or [`TFMPNetModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        relative_attention_num_buckets (`int`, *optional*, defaults to 32):
            The number of buckets to use for each attention layer.

    Examples:

    ```python
    >>> from transformers import MPNetModel, MPNetConfig

    >>> # Initializing a MPNet mpnet-base style configuration
    >>> configuration = MPNetConfig()

    >>> # Initializing a model from the mpnet-base style configuration
    >>> model = MPNetModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    设置模型类型为 "mpnet"
    model_type = "mpnet"
    # 定义一个类的初始化方法，设置了一系列默认参数值
    def __init__(
        self,
        vocab_size=30527,  # 词汇表大小，默认为30527
        hidden_size=768,  # 隐藏层大小，默认为768
        num_hidden_layers=12,  # 隐藏层层数，默认为12
        num_attention_heads=12,  # 注意力头数，默认为12
        intermediate_size=3072,  # 中间层大小，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为"gelu"
        hidden_dropout_prob=0.1,  # 隐藏层的 dropout 概率，默认为0.1
        attention_probs_dropout_prob=0.1,  # 注意力概率 dropout 概率，默认为0.1
        max_position_embeddings=512,  # 最大位置编码，默认为512
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-12,  # 层归一化 epsilon，默认为1e-12
        relative_attention_num_buckets=32,  # 相对注意力的桶数，默认为32
        pad_token_id=1,  # 填充 token ID，默认为1
        bos_token_id=0,  # 开始 token ID，默认为0
        eos_token_id=2,  # 结束 token ID，默认为2
        **kwargs,  # 其他关键字参数
    ):
        # 调用父类的初始化方法，传入填充、开始和结束 token 的 ID，以及其他关键字参数
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
    
        # 设置类的属性，初始化各种参数
        self.vocab_size = vocab_size  # 词汇表大小
        self.hidden_size = hidden_size  # 隐藏层大小
        self.num_hidden_layers = num_hidden_layers  # 隐藏层层数
        self.num_attention_heads = num_attention_heads  # 注意力头数
        self.hidden_act = hidden_act  # 隐藏层激活函数
        self.intermediate_size = intermediate_size  # 中间层大小
        self.hidden_dropout_prob = hidden_dropout_prob  # 隐藏层 dropout 概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob  # 注意力 dropout 概率
        self.max_position_embeddings = max_position_embeddings  # 最大位置编码
        self.initializer_range = initializer_range  # 初始化范围
        self.layer_norm_eps = layer_norm_eps  # 层归一化 epsilon
        self.relative_attention_num_buckets = relative_attention_num_buckets  # 相对注意力的桶数
```