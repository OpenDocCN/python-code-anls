# `.\transformers\models\pix2struct\configuration_pix2struct.py`

```py
# 此文件包含 Pix2StructTextConfig 类的定义，该类用于存储 Pix2StructTextModel 的配置信息
# 包含了导入必要的模块以及定义了 Pix2StructTextConfig 类

# 引入 Python 标准库中的 os 模块和 typing 模块中的 Union 类型
import os
from typing import Union

# 引入 huggingface 库中的 PretrainedConfig 类和 logging 模块
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取当前模块的 logger 对象
logger = logging.get_logger(__name__)

# 定义 Pix2Struct 预训练模型配置存档映射字典
# 其中包含了 google/pix2struct-textcaps-base 模型的配置 JSON 文件的下载地址
PIX2STRUCT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/pix2struct-textcaps-base": (
        "https://huggingface.co/google/pix2struct-textcaps-base/resolve/main/config.json"
    ),
}

# 定义 Pix2StructTextConfig 类，继承自 PretrainedConfig 类
# 该类用于存储 Pix2StructTextModel 的配置信息
class Pix2StructTextConfig(PretrainedConfig):
    r"""
    这是用于存储 Pix2StructTextModel 配置的类。
    它用于根据指定的参数实例化一个 Pix2Struct 文本模型，定义模型架构。
    使用默认参数实例化该配置将产生与 google/pix2struct-base 架构中使用的 Pix2Struct 文本解码器类似的配置。

    配置对象继承自 PretrainedConfig 类，可用于控制模型输出。
    更多信息请参阅 PretrainedConfig 类的文档。
    """
    # 定义一个名为 `Pix2Struct` 的文本模型的配置类
    class Pix2StructTextConfig:
        # 初始化函数，设定模型的词汇表大小，默认为 50244
        def __init__(
            self,
            vocab_size: int = 50244,
            # 编码器层和池化层的维度
            hidden_size: int = 768,
            # 每个注意力头中键、查询、值投影的维度
            d_kv: int = 64,
            # Transformer 编码器中“中间”（即前馈）层的维度
            d_ff: int = 2048,
            # Transformer 编码器中隐藏层的数量
            num_layers: int = 12,
            # Transformer 编码器中每个注意力层的注意力头数量
            num_heads: int = 12,
            # 每个注意力层使用的相对注意力的桶的数量
            relative_attention_num_buckets: int = 32,
            # 用于桶分割的较长序列的最大距离
            relative_attention_max_distance: int = 128,
            # 嵌入层、编码器和池化层中所有全连接层的丢弃率
            dropout_rate: float = 0.1,
            # 层归一化层使用的 epsilon 值
            layer_norm_epsilon: float = 1e-6,
            # 用于初始化所有权重矩阵的因子（应保持为1，用于初始化测试）
            initializer_factor: float = 1.0,
            # 非线性激活函数（函数或字符串）
            dense_act_fn: Union[Callable, str] = "gelu_new",
            # 解码器起始令牌的 id
            decoder_start_token_id: int = 0,
            # 模型是否应返回最后一个键/值注意力（并非所有模型都使用）
            use_cache: bool = False,
            # 填充令牌的 id
            pad_token_id: int = 0,
            # 终止序列的 id
            eos_token_id: int = 1,
        ):
            pass
    
    # 模型类型
    model_type = "pix2struct_text_model"
    # 在推理过程中需要忽略的键列表
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射，用于将类的属性名映射到实际参数名
    attribute_map = {
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        vocab_size=50244,
        hidden_size=768,
        d_kv=64,
        d_ff=2048,
        num_layers=12,
        num_heads=12,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        dense_act_fn="gelu_new",
        decoder_start_token_id=0,
        use_cache=False,
        pad_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
        is_decoder=True,
        **kwargs,
    ):
        # 初始化对象属性
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.use_cache = use_cache

        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id

        # for backwards compatibility
        self.dense_act_fn = dense_act_fn

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            tie_word_embeddings=tie_word_embeddings,
            is_decoder=is_decoder,
            **kwargs,
        )

    @classmethod
    # 从预训练模型中加载配置的类方法
    def from_pretrained(
        cls, pretrainehidden_size_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        # 获取配置字典和其他参数
        config_dict, kwargs = cls.get_config_dict(pretrainehidden_size_name_or_path, **kwargs)

        # 如果是从 Pix2StructConfig 加载模型，则获取文本配置字典
        if config_dict.get("model_type") == "pix2struct":
            config_dict = config_dict["text_config"]

        # 检查要加载的模型类型与当前类定义的模型类型是否一致
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)
# 定义 Pix2StructVisionConfig 类，用于存储 Pix2StructVisionModel 的配置信息
class Pix2StructVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Pix2StructVisionModel`]. It is used to
    instantiate a Pix2Struct vision model according to the specified arguments, defining the model architecture.
    Instantiating a configuration defaults will yield a similar configuration to that of the Pix2Struct-base
    [google/pix2struct-base](https://huggingface.co/google/pix2struct-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        patch_embed_hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the input patch_embedding layer in the Transformer encoder.
        d_ff (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        d_kv (`int`, *optional*, defaults to 64):
            Dimensionality of the key, query, value projections per attention head.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        dense_act_fn (`str` or `function`, *optional*, defaults to `"gelu_new"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        dropout_rate (`float`, *optional*, defaults to 0.0):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 1e-10):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        seq_len (`int`, *optional*, defaults to 4096):
            Maximum sequence length (here number of patches) supported by the model.
        relative_attention_num_buckets (`int`, *optional*, defaults to 32):
            The number of buckets to use for each attention layer.
        relative_attention_max_distance (`int`, *optional*, defaults to 128):
            The maximum distance (in tokens) to use for each attention layer.

    Example:

    ```python
    >>> from transformers import Pix2StructVisionConfig, Pix2StructVisionModel

    >>> # Initializing a Pix2StructVisionConfig with google/pix2struct-base style configuration
    >>> configuration = Pix2StructVisionConfig()

    >>> # Initializing a Pix2StructVisionModel (with random weights) from the google/pix2struct-base style configuration
    >>> model = Pix2StructVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```py"""

    # 定义模型类型为 "pix2struct_vision_model"
    model_type = "pix2struct_vision_model"
    # 初始化函数，设置模型的各种参数
    def __init__(
        self,
        hidden_size=768,  # 隐藏层大小，默认为768
        patch_embed_hidden_size=768,  # 补丁嵌入隐藏层大小，默认为768
        d_ff=2048,  # 前馈网络中间层大小，默认为2048
        d_kv=64,  # 键值大小，默认为64
        num_hidden_layers=12,  # 隐藏层层数，默认为12
        num_attention_heads=12,  # 注意力头数，默认为12
        dense_act_fn="gelu_new",  # 密集层激活函数，默认为gelu_new
        layer_norm_eps=1e-6,  # 层归一化的epsilon值，默认为1e-6
        dropout_rate=0.0,  # 丢弃率，默认为0.0
        attention_dropout=0.0,  # 注意力丢弃率，默认为0.0
        initializer_range=1e-10,  # 初始化范围，默认为1e-10
        initializer_factor=1.0,  # 初始化因子，默认为1.0
        seq_len=4096,  # 序列长度，默认为4096
        relative_attention_num_buckets=32,  # 相对注意力桶数，默认为32
        relative_attention_max_distance=128,  # 相对注意力最大距离，默认为128
        **kwargs,  # 其他关键字参数
    ):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 设置各个参数的值
        self.hidden_size = hidden_size
        self.patch_embed_hidden_size = patch_embed_hidden_size
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.dense_act_fn = dense_act_fn
        self.seq_len = seq_len
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.d_kv = d_kv

    # 从预训练模型加载配置
    @classmethod
    def from_pretrained(
        cls, pretrainehidden_size_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> "PretrainedConfig":
        # 设置token在kwargs中
        cls._set_token_in_kwargs(kwargs)

        # 获取配置字典和kwargs
        config_dict, kwargs = cls.get_config_dict(pretrainehidden_size_name_or_path, **kwargs)

        # 如果加载自Pix2StructConfig，则获取视觉配置字典
        if config_dict.get("model_type") == "pix2struct":
            config_dict = config_dict["vision_config"]

        # 如果配置字典中包含model_type，并且cls有model_type属性且不等于配置字典中的model_type，则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从配置字典和kwargs中创建实例
        return cls.from_dict(config_dict, **kwargs)
# 定义 Pix2StructConfig 类，用于存储 Pix2StructForConditionalGeneration 模型的配置信息
class Pix2StructConfig(PretrainedConfig):
    r"""
    [`Pix2StructConfig`] is the configuration class to store the configuration of a
    [`Pix2StructForConditionalGeneration`]. It is used to instantiate a Pix2Struct model according to the specified
    arguments, defining the text model and vision model configs. Instantiating a configuration with the defaults will
    yield a similar configuration to that of the Pix2Struct-base
    [google/pix2struct-base](https://huggingface.co/google/pix2struct-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional`):
            Dictionary of configuration options used to initialize [`Pix2StructTextConfig`].
        vision_config (`dict`, *optional`):
            Dictionary of configuration options used to initialize [`Pix2StructVisionConfig`].
        initializer_factor (`float`, *optional`, defaults to 1.0):
            Factor to multiply the initialization range with.
        initializer_range (`float`, *optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        is_vqa (`bool`, *optional`, defaults to `False`):
            Whether the model has been fine-tuned for VQA or not.
        kwargs (*optional`):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import Pix2StructConfig, Pix2StructForConditionalGeneration

    >>> # Initializing a Pix2StructConfig with google/pix2struct-base style configuration
    >>> configuration = Pix2StructConfig()

    >>> # Initializing a Pix2StructForConditionalGeneration (with random weights) from the google/pix2struct-base style configuration
    >>> model = Pix2StructForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a Pix2StructConfig from a Pix2StructTextConfig and a Pix2StructVisionConfig

    >>> # Initializing a Pix2Struct text and Pix2Struct vision configuration
    >>> config_text = Pix2StructTextConfig()
    >>> config_vision = Pix2StructVisionConfig()

    >>> config = Pix2StructConfig.from_text_vision_configs(config_text, config_vision)
    ```py"""

    # 模型类型为 "pix2struct"
    model_type = "pix2struct"

    # 初始化方法，接受多个参数和关键字参数
    def __init__(
        self,
        text_config=None,
        vision_config=None,
        initializer_factor=1.0,
        initializer_range=0.02,
        is_vqa=False,
        tie_word_embeddings=False,
        is_encoder_decoder=True,
        **kwargs,
    # 调用父类的构造函数，初始化Pix2StructConfig对象
    def __init__(
        self, tie_word_embeddings=tie_word_embeddings, is_encoder_decoder=is_encoder_decoder, **kwargs
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, is_encoder_decoder=is_encoder_decoder, **kwargs)

        # 如果text_config为None，则使用空字典初始化
        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the Pix2StructTextConfig with default values.")

        # 如果vision_config为None，则使用空字典初始化
        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. Initializing the Pix2StructVisionConfig with default values.")

        # 使用text_config和vision_config初始化Pix2StructTextConfig和Pix2StructVisionConfig对象
        self.text_config = Pix2StructTextConfig(**text_config)
        self.vision_config = Pix2StructVisionConfig(**vision_config)

        # 初始化decoder_start_token_id, pad_token_id, eos_token_id
        self.decoder_start_token_id = self.text_config.decoder_start_token_id
        self.pad_token_id = self.text_config.pad_token_id
        self.eos_token_id = self.text_config.eos_token_id

        # 初始化initializer_factor和initializer_range
        self.initializer_factor = initializer_factor
        self.initializer_range = initializer_range

        # 将initializer_range赋值给text_config和vision_config的initializer_range
        self.text_config.initializer_range = self.initializer_range
        self.vision_config.initializer_range = self.initializer_range

        # 初始化is_vqa
        self.is_vqa = is_vqa

    # 类方法，从text_config和vision_config实例化Pix2StructConfig对象
    @classmethod
    def from_text_vision_configs(
        cls, text_config: Pix2StructTextConfig, vision_config: Pix2StructVisionConfig, **kwargs
    ):
        r"""
        Instantiate a [`Pix2StructConfig`] (or a derived class) from pix2struct text model configuration and pix2struct
        vision model configuration.

        Returns:
            [`Pix2StructConfig`]: An instance of a configuration object
        """

        # 使用text_config和vision_config的字典形式初始化Pix2StructConfig对象
        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)
```