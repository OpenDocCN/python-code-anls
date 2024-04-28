# `.\transformers\models\splinter\configuration_splinter.py`

```
# 设置文件编码为utf-8
# 版权声明及许可协议
# 导入所需的库和模块
""" Splinter model configuration"""

# 从...模块中导入PretrainedConfig类
from ...configuration_utils import PretrainedConfig
# 从...模块中导入logging模块
from ...utils import logging

# 获取logger对象
logger = logging.get_logger(__name__)

# 预训练配置存档映射
SPLINTER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "tau/splinter-base": "https://huggingface.co/tau/splinter-base/resolve/main/config.json",
    "tau/splinter-base-qass": "https://huggingface.co/tau/splinter-base-qass/resolve/main/config.json",
    "tau/splinter-large": "https://huggingface.co/tau/splinter-large/resolve/main/config.json",
    "tau/splinter-large-qass": "https://huggingface.co/tau/splinter-large-qass/resolve/main/config.json",
    # 查看所有Splinter模型：https://huggingface.co/models?filter=splinter
}

# Splinter配置类，用于存储SplinterModel的配置
class SplinterConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SplinterModel`]. It is used to instantiate an
    Splinter model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Splinter
    [tau/splinter-base](https://huggingface.co/tau/splinter-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the Splinter model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`SplinterModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`SplinterModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        question_token_id (`int`, *optional*, defaults to 104):
            The id of the `[QUESTION]` token.

    Example:

    ```python
    >>> from transformers import SplinterModel, SplinterConfig

    >>> # Initializing a Splinter tau/splinter-base style configuration
    >>> configuration = SplinterConfig()

    >>> # Initializing a model from the tau/splinter-base style configuration
    >>> model = SplinterModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "splinter"
    def __init__(
        self,
        vocab_size=30522,  # 设置词汇表大小，默认为30522
        hidden_size=768,  # 设置隐藏层大小，默认为768
        num_hidden_layers=12,  # 设置隐藏层层数，默认为12
        num_attention_heads=12,  # 设置注意力头数，默认为12
        intermediate_size=3072,  # 设置中间层大小，默认为3072
        hidden_act="gelu",  # 设置隐藏层激活函数，默认为gelu
        hidden_dropout_prob=0.1,  # 设置隐藏层dropout概率，默认为0.1
        attention_probs_dropout_prob=0.1,  # 设置注意力dropout概率，默认为0.1
        max_position_embeddings=512,  # 设置最大位置嵌入长度，默认为512
        type_vocab_size=2,  # 设置类型词汇表大小，默认为2
        initializer_range=0.02,  # 设置初始化范围，默认为0.02
        layer_norm_eps=1e-12,  # 设置层归一化epsilon值，默认为1e-12
        use_cache=True,  # 是否使用缓存，默认为True
        pad_token_id=0,  # 设置填充标记ID，默认为0
        question_token_id=104,  # 设置问题标记ID，默认为104
        **kwargs,  # 允许传入任意其他关键字参数
    ):
        # 调用父类的初始化方法，并传入填充标记ID和其他关键字参数
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        # 设置各种属性值
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
        self.use_cache = use_cache
        self.question_token_id = question_token_id
```