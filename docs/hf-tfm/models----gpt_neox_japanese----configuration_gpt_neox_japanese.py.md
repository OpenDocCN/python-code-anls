# `.\models\gpt_neox_japanese\configuration_gpt_neox_japanese.py`

```py
# 导入所需的模块和工具函数
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 映射预训练模型配置文件的路径
GPT_NEOX_JAPANESE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "abeja/gpt-neox-japanese-2.7b": "https://huggingface.co/abeja/gpt-neox-japanese-2.7b/resolve/main/config.json",
}

# 定义 GPTNeoXJapaneseConfig 类，用于存储 GPTNeoX 日语模型的配置信息
class GPTNeoXJapaneseConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GPTNeoXModelJapanese`]. It is used to instantiate
    a GPTNeoX model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the GPTNeoXJapanese
    [abeja/gpt-neox-japanese-2.7b](https://huggingface.co/abeja/gpt-neox-japanese-2.7b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information. Default configs is set as 2.7B model
```  
    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the GPTNeoXJapanese model. Defines the number of different tokens that can be
            represented by the `inputs_ids` passed when calling [`GPTNeoXJapanese`].
        hidden_size (`int`, *optional*, defaults to 2560):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_multiple_size (`int`, *optional*, defaults to 4):
            Dimension of the "intermediate" layer in the Transformer encoder is calculated by hidden_size *
            intermediate_multiple_size.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
        rotary_pct (`float`, *optional*, defaults to 1.00):
            percentage of hidden dimensions to allocate to rotary embeddings
        rotary_emb_base (`int`, *optional*, defaults to 10000)
            base for computing rotary embeddings frequency
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        hidden_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the hidden layer.
        Example:

    ```py
    >>> from transformers import GPTNeoXJapaneseConfig, GPTNeoXJapaneseModel

    >>> # Initializing a GPTNeoXJapanese gpt-neox-japanese-2.7b style configuration
    >>> configuration = GPTNeoXJapaneseConfig()

    >>> # Initializing a model (with random weights) from the gpt-neox-japanese-2.7b style configuration
    >>> model = GPTNeoXJapaneseModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    # 设置模型类型为"gpt_neox_japanese"
    model_type = "gpt_neox_japanese"
```py  
    # 初始化 Transformer 模型的参数
    def __init__(
        self,
        vocab_size=32000,  # 词汇表大小，默认为32000
        hidden_size=2560,  # 隐藏层尺寸，默认为2560
        num_hidden_layers=32,  # 隐藏层层数，默认为32
        num_attention_heads=32,  # 注意力头数，默认为32
        intermediate_multiple_size=4,  # 中间层尺寸倍数，默认为4
        hidden_act="gelu",  # 隐藏层激活函数，默认为gelu
        rotary_pct=1.00,  # 旋转嵌入比例，默认为1.00
        rotary_emb_base=10000,  # 旋转嵌入基数，默认为10000
        max_position_embeddings=2048,  # 最大位置嵌入数，默认为2048
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-5,  # 层归一化的ε值，默认为1e-5
        use_cache=True,  # 是否使用缓存，默认为True
        bos_token_id=31996,  # 开始标记的 token ID，默认为31996
        eos_token_id=31999,  # 结束标记的 token ID，默认为31999
        attention_dropout=0.1,  # 注意力层的 dropout，默认为0.1
        hidden_dropout=0.0,  # 隐藏层的 dropout，默认为0.0
        **kwargs,  # 其他额外参数
    ):
        # 调用父类构造函数，设置开始标记和结束标记的 token ID
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        # 设置词汇表大小
        self.vocab_size = vocab_size
        # 设置最大位置嵌入数
        self.max_position_embeddings = max_position_embeddings
        # 设置隐藏层尺寸
        self.hidden_size = hidden_size
        # 设置隐藏层层数
        self.num_hidden_layers = num_hidden_layers
        # 设置注意力头数
        self.num_attention_heads = num_attention_heads
        # 设置中间层尺寸倍数
        self.intermediate_multiple_size = intermediate_multiple_size
        # 设置隐藏层激活函数
        self.hidden_act = hidden_act
        # 设置旋转嵌入比例
        self.rotary_pct = rotary_pct
        # 设置旋转嵌入基数
        self.rotary_emb_base = rotary_emb_base
        # 设置初始化范围
        self.initializer_range = initializer_range
        # 设置层归一化的ε值
        self.layer_norm_eps = layer_norm_eps
        # 设置是否使用缓存
        self.use_cache = use_cache
        # 设置注意力层的 dropout
        self.attention_dropout = attention_dropout
        # 设置隐藏层的 dropout
        self.hidden_dropout = hidden_dropout
```