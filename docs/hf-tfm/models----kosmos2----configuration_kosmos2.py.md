# `.\models\kosmos2\configuration_kosmos2.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 Microsoft Research 和 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本授权使用此文件
# 只有在遵守许可证的情况下才能使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以了解特定语言的权限和限制
""" KOSMOS-2 模型配置"""

# 导入必要的库
import os
from typing import Union

# 导入预训练配置和日志记录工具
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练配置文件映射
KOSMOS2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/kosmos-2-patch14-224": (
        "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/config.json"
    ),
    # 查看所有 KOSMOS-2 模型：https://huggingface.co/models?filter=kosmos-2
}

# KOSMOS-2 文本配置类，用于存储 KOSMOS-2 文本模型的配置
class Kosmos2TextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Kosmos2TextModel`]. It is used to instantiate a
    KOSMOS-2 text decoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the text decoder of the KOSMOS-2
    [microsoft/kosmos-2-patch14-224](https://huggingface.co/microsoft/kosmos-2-patch14-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 65037):
            Vocabulary size of the Kosmos2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Kosmos2Model`].
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        embed_dim (`int`, *optional*, defaults to 2048):
            Dimensionality of the layers and the pooler layer.
        layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        ffn_dim (`int`, *optional*, defaults to 8192):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_embedding (`bool`, *optional*, defaults to `True`):
            Scale embeddings by diving by sqrt(embed_dim).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
    """"

    model_type = "kosmos_2_text_model"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "attention_heads",
        "hidden_size": "embed_dim",
        "num_hidden_layers": "layers",
    }
    # 初始化函数，设置模型的各种参数
    def __init__(
        self,
        vocab_size=65037,  # 词汇表大小，默认为65037
        max_position_embeddings=2048,  # 最大位置编码长度，默认为2048
        embed_dim=2048,  # 嵌入维度，默认为2048
        layers=24,  # 层数，默认为24
        ffn_dim=8192,  # FeedForward网络维度，默认为8192
        attention_heads=32,  # 注意力头数，默认为32
        activation_function="gelu",  # 激活函数，默认为gelu
        dropout=0.1,  # 普通dropout概率，默认为0.1
        attention_dropout=0.1,  # 注意力dropout概率，默认为0.1
        activation_dropout=0.0,  # 激活函数dropout概率，默认为0.0
        layerdrop=0.0,  # 层dropout概率，默认为0.0
        layer_norm_eps=1e-5,  # LayerNorm的epsilon值，默认为1e-5
        init_std=0.02,  # 初始化标准差，默认为0.02
        scale_embedding=True,  # 是否对嵌入进行缩放，默认为True
        use_cache=True,  # 是否使用缓存，默认为True
        pad_token_id=1,  # 填充token的id，默认为1
        bos_token_id=0,  # 起始token的id，默认为0
        eos_token_id=2,  # 结束token的id，默认为2
        **kwargs,  # 其他参数
    ):
        # 调用父类的初始化函数，设置填充、起始、结束token的id
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        # 设置模型的各种参数
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.embed_dim = embed_dim
        self.layers = layers
        self.ffn_dim = ffn_dim
        self.attention_heads = attention_heads
        self.activation_function = activation_function
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.layerdrop = layerdrop
        self.layer_norm_eps = layer_norm_eps
        self.init_std = init_std
        self.scale_embedding = scale_embedding
        self.use_cache = use_cache

    # 从预训练模型加载配置
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 设置token相关参数到kwargs中
        cls._set_token_in_kwargs(kwargs)

        # 获取配置字典和kwargs
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果从Kosmos2Config加载，获取文本配置字典
        if config_dict.get("model_type") == "kosmos-2":
            config_dict = config_dict["text_config"]

        # 如果配置字典中包含模型类型，且与当前类的模型类型不一致，发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 根据配置字典和kwargs创建实例
        return cls.from_dict(config_dict, **kwargs)
class Kosmos2VisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Kosmos2VisionModel`]. It is used to instantiate a
    KOSMOS-2 vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the KOSMOS-2
    [microsoft/kosmos-2-patch14-224](https://huggingface.co/microsoft/kosmos-2-patch14-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
    """```

    model_type = "kosmos_2_vision_model"

    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_channels=3,
        image_size=224,
        patch_size=14,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        **kwargs,
    # 初始化函数，调用父类初始化方法，并传入关键字参数
    ):
        super().__init__(**kwargs)

        # 设定隐藏层大小
        self.hidden_size = hidden_size
        # 设定中间层大小
        self.intermediate_size = intermediate_size
        # 设定隐藏层数量
        self.num_hidden_layers = num_hidden_layers
        # 设定注意力头数量
        self.num_attention_heads = num_attention_heads
        # 设定通道数量
        self.num_channels = num_channels
        # 设定图像块大小
        self.patch_size = patch_size
        # 设定图像大小
        self.image_size = image_size
        # 设定初始化范围
        self.initializer_range = initializer_range
        # 设定初始化因子
        self.initializer_factor = initializer_factor
        # 设定注意力机制的丢弃率
        self.attention_dropout = attention_dropout
        # 设定层归一化的 epsilon
        self.layer_norm_eps = layer_norm_eps
        # 设定隐藏层激活函数
        self.hidden_act = hidden_act

    # 从预训练模型加载配置参数
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 在关键字参数中设置 token
        cls._set_token_in_kwargs(kwargs)

        # 获取配置字典和剩余关键字参数
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 若从 Kosmos2Config 加载，则获取视觉配置字典
        if config_dict.get("model_type") == "kosmos-2":
            config_dict = config_dict["vision_config"]

        # 若配置字典中存在模型类型且与当前类的模型类型不同，发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从配置字典和剩余关键字参数构建配置对象
        return cls.from_dict(config_dict, **kwargs)
class Kosmos2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Kosmos2Model`]. It is used to instantiate a
    KOSMOS-2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the KOSMOS-2
    [microsoft/kosmos-2-patch14-224](https://huggingface.co/microsoft/kosmos-2-patch14-224) architecture.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Kosmos2TextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Kosmos2VisionConfig`].
        latent_query_num (`int`, *optional*, defaults to 64):
            The number of latent query tokens that represent the image features used in the text decoder component.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```py
    >>> from transformers import Kosmos2Config, Kosmos2Model

    >>> # Initializing a Kosmos-2 kosmos-2-patch14-224 style configuration
    >>> configuration = Kosmos2Config()

    >>> # Initializing a model (with random weights) from the kosmos-2-patch14-224 style configuration
    >>> model = Kosmos2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "kosmos-2"
    is_composition = True

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        latent_query_num=64,
        **kwargs,
    ):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 如果 text_config 为 None，使用默认值初始化 Kosmos2TextConfig
        if text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `Kosmos2TextConfig` with default values.")

        # 如果 vision_config 为 None，使用默认值初始化 Kosmos2VisionConfig
        if vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. Initializing the `Kosmos2VisionConfig` with default values.")

        # 使用 text_config 初始化 Kosmos2TextConfig
        self.text_config = Kosmos2TextConfig(**text_config)
        # 使用 vision_config 初始化 Kosmos2VisionConfig
        self.vision_config = Kosmos2VisionConfig(**vision_config)

        # 设置 latent_query_num 属性
        self.latent_query_num = latent_query_num
```