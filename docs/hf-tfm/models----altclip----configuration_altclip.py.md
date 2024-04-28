# `.\transformers\models\altclip\configuration_altclip.py`

```
# 设置文件编码为 UTF-8
# 版权声明，包括对应作者和 HuggingFace Inc. 团队的版权说明
# 使用 Apache License, Version 2.0 授权，详见链接
# 如果不符合许可要求，则不能使用此文件
""" AltCLIP 模型配置"""
# 导入所需的库
import os
from typing import Union

# 从 transformers 库中导入预训练配置类和日志记录工具
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# AltCLIP 预训练模型配置文件的映射字典
ALTCLIP_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "BAAI/AltCLIP": "https://huggingface.co/BAAI/AltCLIP/resolve/main/config.json",
    # 查看所有 AltCLIP 模型：https://huggingface.co/models?filter=altclip
}

# AltCLIP 文本模型配置类，继承自预训练配置类
class AltCLIPTextConfig(PretrainedConfig):
    r"""
    这是用于存储 [`AltCLIPTextModel`] 配置的配置类。根据指定的参数，定义了模型的架构。使用默认参数实例化配置将生成类似于 AltCLIP
    [BAAI/AltCLIP](https://huggingface.co/BAAI/AltCLIP) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 文档获取更多信息。

    Examples:

    ```python
    >>> from transformers import AltCLIPTextModel, AltCLIPTextConfig

    >>> # 使用 BAAI/AltCLIP 风格配置初始化 AltCLIPTextConfig
    >>> configuration = AltCLIPTextConfig()

    >>> # 从 BAAI/AltCLIP 风格配置初始化 AltCLIPTextModel（带有随机权重）
    >>> model = AltCLIPTextModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```"""

    # 模型类型
    model_type = "altclip_text_model"

    # 初始化方法
    def __init__(
        self,
        vocab_size=250002,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=514,
        type_vocab_size=1,
        initializer_range=0.02,
        initializer_factor=0.02,
        layer_norm_eps=1e-05,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        position_embedding_type="absolute",
        use_cache=True,
        project_dim=768,
        **kwargs,
        # 调用父类的构造函数，初始化模型的各种参数，包括填充、起始和结束标记的 token id
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 设置词汇表大小
        self.vocab_size = vocab_size
        # 设置隐藏层大小
        self.hidden_size = hidden_size
        # 设置隐藏层的数量
        self.num_hidden_layers = num_hidden_layers
        # 设置注意力头的数量
        self.num_attention_heads = num_attention_heads
        # 设置隐藏层的激活函数
        self.hidden_act = hidden_act
        # 设置中间层大小
        self.intermediate_size = intermediate_size
        # 设置隐藏层的 dropout 概率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 设置注意力概率的 dropout 概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 设置最大位置编码的长度
        self.max_position_embeddings = max_position_embeddings
        # 设置类型词汇表大小
        self.type_vocab_size = type_vocab_size
        # 设置初始化范围
        self.initializer_range = initializer_range
        # 设置初始化因子
        self.initializer_factor = initializer_factor
        # 设置层归一化的 epsilon 值
        self.layer_norm_eps = layer_norm_eps
        # 设置位置编码的类型
        self.position_embedding_type = position_embedding_type
        # 设置是否使用缓存
        self.use_cache = use_cache
        # 设置项目维度
        self.project_dim = project_dim
class AltCLIPVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`AltCLIPModel`]. It is used to instantiate an
    AltCLIP model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the AltCLIP
    [BAAI/AltCLIP](https://huggingface.co/BAAI/AltCLIP) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and vision projection layers.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 32):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).

    Example:

    ```python
    >>> from transformers import AltCLIPVisionConfig, AltCLIPVisionModel

    >>> # Initializing a AltCLIPVisionConfig with BAAI/AltCLIP style configuration
    >>> configuration = AltCLIPVisionConfig()

    >>> # Initializing a AltCLIPVisionModel (with random weights) from the BAAI/AltCLIP style configuration
    >>> model = AltCLIPVisionModel(configuration)

    >>> # Accessing the model configuration
    # 获取模型的配置信息
    configuration = model.config



    # 定义模型类型为 "altclip_vision_model"
    model_type = "altclip_vision_model"



    # 初始化函数，设置模型各项参数
    def __init__(
        self,
        hidden_size=768,  # 隐藏层大小，默认为 768
        intermediate_size=3072,  # 中间层大小，默认为 3072
        projection_dim=512,  # 投影维度，默认为 512
        num_hidden_layers=12,  # 隐藏层数，默认为 12
        num_attention_heads=12,  # 注意力头数，默认为 12
        num_channels=3,  # 图像通道数，默认为 3
        image_size=224,  # 图像大小，默认为 224
        patch_size=32,  # 图像分块大小，默认为 32
        hidden_act="quick_gelu",  # 隐藏层激活函数，默认为 "quick_gelu"
        layer_norm_eps=1e-5,  # 层归一化 epsilon，默认为 1e-5
        attention_dropout=0.0,  # 注意力层 dropout，默认为 0.0
        initializer_range=0.02,  # 初始化范围，默认为 0.02
        initializer_factor=1.0,  # 初始化因子，默认为 1.0
        **kwargs,  # 其他参数
    ):
        # 调用父类构造函数
        super().__init__(**kwargs)

        # 设置模型各项参数
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act



    # 从预训练模型中加载配置信息
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 设置 token 参数到 kwargs 中
        cls._set_token_in_kwargs(kwargs)

        # 获取配置字典和 kwargs
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典中的模型类型为 "altclip"，则获取其视觉配置字典
        if config_dict.get("model_type") == "altclip":
            config_dict = config_dict["vision_config"]

        # 如果配置字典中包含模型类型，并且类中有 model_type 属性，且两者不匹配，发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 根据配置字典和 kwargs 创建预训练配置对象并返回
        return cls.from_dict(config_dict, **kwargs)
class AltCLIPConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`AltCLIPModel`]. It is used to instantiate an
    AltCLIP model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the AltCLIP
    [BAAI/AltCLIP](https://huggingface.co/BAAI/AltCLIP) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`AltCLIPTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`AltCLIPVisionConfig`].
        projection_dim (`int`, *optional`, defaults to 768):
            Dimentionality of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* paramter. Default is used as per the original CLIP implementation.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import AltCLIPConfig, AltCLIPModel

    >>> # Initializing a AltCLIPConfig with BAAI/AltCLIP style configuration
    >>> configuration = AltCLIPConfig()

    >>> # Initializing a AltCLIPModel (with random weights) from the BAAI/AltCLIP style configuration
    >>> model = AltCLIPModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a AltCLIPConfig from a AltCLIPTextConfig and a AltCLIPVisionConfig

    >>> # Initializing a AltCLIPText and AltCLIPVision configuration
    >>> config_text = AltCLIPTextConfig()
    >>> config_vision = AltCLIPVisionConfig()

    >>> config = AltCLIPConfig.from_text_vision_configs(config_text, config_vision)
    ```"""

    model_type = "altclip"

    def __init__(
        self, text_config=None, vision_config=None, projection_dim=768, logit_scale_init_value=2.6592, **kwargs
    # Class method to create an instance of AltCLIPConfig from AltCLIPTextConfig and AltCLIPVisionConfig
    @classmethod
    def from_text_vision_configs(cls, text_config: AltCLIPTextConfig, vision_config: AltCLIPVisionConfig, **kwargs):
        r"""
        Instantiate a [`AltCLIPConfig`] (or a derived class) from altclip text model configuration and altclip vision
        model configuration.

        Returns:
            [`AltCLIPConfig`]: An instance of a configuration object
        """

        # Create an instance of AltCLIPConfig using the text and vision configurations as dictionaries
        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)
```