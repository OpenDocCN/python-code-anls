# `.\models\bridgetower\configuration_bridgetower.py`

```
# coding=utf-8
# 设置模块的版权声明和许可信息

""" BridgeTower model configuration"""
# 引入必要的库和模块
import os
from typing import Union

# 从相对路径引入配置工具和日志模块
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练模型配置映射，将模型名称映射到其配置文件的下载链接
BRIDGETOWER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "BridgeTower/bridgetower-base": "https://huggingface.co/BridgeTower/bridgetower-base/blob/main/config.json",
    "BridgeTower/bridgetower-base-itm-mlm": (
        "https://huggingface.co/BridgeTower/bridgetower-base-itm-mlm/blob/main/config.json"
    ),
}

# 定义一个配置类 BridgeTowerVisionConfig，用于存储视觉编码器的配置信息
class BridgeTowerVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the vision configuration of a [`BridgeTowerModel`]. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the bridgetower-base
    [BridgeTower/bridgetower-base](https://huggingface.co/BridgeTower/bridgetower-base/) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in visual encoder model.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        image_size (`int`, *optional*, defaults to 288):
            The size (resolution) of each image.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        stop_gradient (`bool`, *optional*, defaults to `False`):
            Whether to stop gradient for training.
        share_layernorm (`bool`, *optional*, defaults to `True`):
            Whether LayerNorm layers are shared.
        remove_last_layer (`bool`, *optional*, defaults to `False`):
            Whether to remove the last layer from the vision encoder.
    """

    # 初始化函数，设置各种可选参数的默认值
    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        patch_size=16,
        image_size=288,
        initializer_factor=1.0,
        layer_norm_eps=1e-05,
        stop_gradient=False,
        share_layernorm=True,
        remove_last_layer=False,
        **kwargs
    ):
        # 调用父类的初始化函数，传递配置参数
        super().__init__(**kwargs)
        # 设置实例变量，存储每个参数的值
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_factor = initializer_factor
        self.layer_norm_eps = layer_norm_eps
        self.stop_gradient = stop_gradient
        self.share_layernorm = share_layernorm
        self.remove_last_layer = remove_last_layer
    >>> from transformers import BridgeTowerVisionConfig

    # 导入 BridgeTowerVisionConfig 类

    >>> # Initializing a BridgeTower BridgeTower/bridgetower-base style configuration for the vision model
    # 初始化一个 BridgeTower 风格的视觉模型配置，使用 BridgeTower/bridgetower-base 风格

    >>> configuration = BridgeTowerVisionConfig()

    # 创建一个 BridgeTowerVisionConfig 的实例，用于配置视觉模型

    >>> # Accessing the configuration
    # 访问配置实例
    >>> configuration
    ```"""

    model_type = "bridgetower_vision_model"

    # 设置模型类型为 "bridgetower_vision_model"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_channels=3,
        patch_size=16,
        image_size=288,
        initializer_factor=1,
        layer_norm_eps=1e-05,
        stop_gradient=False,
        share_layernorm=True,
        remove_last_layer=False,
        **kwargs,
    ):
        # 初始化方法，接受多个参数用于配置模型的各个属性
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_factor = initializer_factor
        self.layer_norm_eps = layer_norm_eps
        self.stop_gradient = stop_gradient
        self.share_layernorm = share_layernorm
        self.remove_last_layer = remove_last_layer

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 类方法，从预训练模型加载配置
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if config_dict.get("model_type") == "bridgetower":
            config_dict = config_dict["text_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)
# BridgeTowerTextConfig 类继承自 PretrainedConfig，用于存储文本模型的配置信息
class BridgeTowerTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the text configuration of a [`BridgeTowerModel`]. The default values here
    are copied from RoBERTa. Instantiating a configuration with the defaults will yield a similar configuration to that
    of the bridgetower-base [BridegTower/bridgetower-base](https://huggingface.co/BridgeTower/bridgetower-base/)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```python
    >>> from transformers import BridgeTowerTextConfig

    >>> # Initializing a BridgeTower BridgeTower/bridgetower-base style configuration for the text model
    >>> configuration = BridgeTowerTextConfig()

    >>> # Accessing the configuration
    >>> configuration
    ```"""

    # 模型类型为 "bridgetower_text_model"
    model_type = "bridgetower_text_model"

    # 初始化方法，设置各种模型参数
    def __init__(
        self,
        vocab_size=50265,  # 词汇表大小，默认为 50265
        hidden_size=768,   # 隐藏层大小，默认为 768
        num_hidden_layers=12,  # 隐藏层数，默认为 12
        num_attention_heads=12,  # 注意力头数，默认为 12
        initializer_factor=1,    # 初始化因子，默认为 1
        intermediate_size=3072,  # 中间层大小，默认为 3072
        hidden_act="gelu",        # 隐藏层激活函数，默认为 "gelu"
        hidden_dropout_prob=0.1,  # 隐藏层 dropout 概率，默认为 0.1
        attention_probs_dropout_prob=0.1,  # 注意力 dropout 概率，默认为 0.1
        max_position_embeddings=514,       # 最大位置嵌入数，默认为 514
        type_vocab_size=1,                 # 类型词汇表大小，默认为 1
        layer_norm_eps=1e-05,              # 层归一化 epsilon，默认为 1e-05
        pad_token_id=1,                    # 填充 token 的 id，默认为 1
        bos_token_id=0,                    # 开始 token 的 id，默认为 0
        eos_token_id=2,                    # 结束 token 的 id，默认为 2
        position_embedding_type="absolute",  # 位置嵌入类型，默认为 "absolute"
        use_cache=True,                     # 是否使用缓存，默认为 True
        **kwargs,
    ):
        super().__init__(**kwargs)  # 调用父类 PretrainedConfig 的初始化方法

        # 设置各个参数
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_factor = initializer_factor
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    @classmethod
    # 根据预训练模型名称或路径获取配置字典和额外的关键字参数
    config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

    # 如果配置字典中的模型类型是 "bridgetower"，则将配置字典更新为其"text_config"字段的内容
    if config_dict.get("model_type") == "bridgetower":
        config_dict = config_dict["text_config"]

    # 如果配置字典中包含"model_type"字段，并且类(cls)具有"model_type"属性，并且配置字典中的模型类型与类的模型类型不匹配，
    # 则发出警告，因为这种情况下并非所有模型配置都支持，可能导致错误
    if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
        logger.warning(
            f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
            f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
        )

    # 根据配置字典创建并返回预训练配置对象
    return cls.from_dict(config_dict, **kwargs)
# BridgeTowerConfig 类，用于存储 BridgeTowerModel 的配置信息
class BridgeTowerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BridgeTowerModel`]. It is used to instantiate a
    BridgeTower model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the bridgetower-base
    [BridgeTower/bridgetower-base](https://huggingface.co/BridgeTower/bridgetower-base/) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        share_cross_modal_transformer_layers (`bool`, *optional*, defaults to `True`):
            Whether cross modal transformer layers are shared.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        share_link_tower_layers (`bool`, *optional*, defaults to `False`):
            Whether the bride/link tower layers are shared.
        link_tower_type (`str`, *optional*, defaults to `"add"`):
            Type of the bridge/link layer.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input and output embeddings.
        init_layernorm_from_vision_encoder (`bool`, *optional*, defaults to `False`):
            Whether to init LayerNorm from the vision encoder.
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`BridgeTowerTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`BridgeTowerVisionConfig`].

    Example:

    ```python
    >>> from transformers import BridgeTowerModel, BridgeTowerConfig

    >>> # Initializing a BridgeTower BridgeTower/bridgetower-base style configuration
    >>> configuration = BridgeTowerConfig()

    >>> # Initializing a model from the BridgeTower/bridgetower-base style configuration
    >>> model = BridgeTowerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    # 设置模型类型为 "bridgetower"
    model_type = "bridgetower"
    
    # 定义一个名为 BridgeTowerConfig 的类，用于配置 Bridgetower 模型的参数
    class BridgeTowerConfig:
        # 初始化方法，用于设置 BridgetowerConfig 实例的各种参数
        def __init__(
            self,
            share_cross_modal_transformer_layers=True,  # 是否共享跨模态 Transformer 层
            hidden_act="gelu",  # 隐藏层激活函数类型，默认为 gelu
            hidden_size=768,  # 隐藏层大小，默认为 768
            initializer_factor=1,  # 初始化因子，默认为 1
            layer_norm_eps=1e-05,  # LayerNormalization 中的 epsilon，默认为 1e-05
            share_link_tower_layers=False,  # 是否共享链接塔层
            link_tower_type="add",  # 链接塔类型，默认为 add
            num_attention_heads=12,  # 注意力头数目，默认为 12
            num_hidden_layers=6,  # 隐藏层层数，默认为 6
            tie_word_embeddings=False,  # 是否绑定单词嵌入
            init_layernorm_from_vision_encoder=False,  # 是否从视觉编码器初始化层归一化
            text_config=None,  # 文本配置，如果为 None 则使用默认值
            vision_config=None,  # 视觉配置，如果为 None 则使用默认值
            **kwargs,  # 其他参数
        ):
            # TODO: remove this once the Hub files are updated.
            _ = kwargs.pop("text_config_dict", None)  # 从 kwargs 中移除 "text_config_dict" 键的值
            _ = kwargs.pop("vision_config_dict", None)  # 从 kwargs 中移除 "vision_config_dict" 键的值
    
            super().__init__(**kwargs)  # 调用父类的初始化方法，传入剩余的关键字参数
    
            # 设置类的实例变量
            self.share_cross_modal_transformer_layers = share_cross_modal_transformer_layers
            self.hidden_act = hidden_act
            self.hidden_size = hidden_size
            self.initializer_factor = initializer_factor
            self.layer_norm_eps = layer_norm_eps
            self.share_link_tower_layers = share_link_tower_layers
            self.link_tower_type = link_tower_type
            self.num_attention_heads = num_attention_heads
            self.num_hidden_layers = num_hidden_layers
            self.tie_word_embeddings = tie_word_embeddings
            self.init_layernorm_from_vision_encoder = init_layernorm_from_vision_encoder
    
            # 如果 text_config 为 None，则使用默认空字典，并记录日志消息
            if text_config is None:
                text_config = {}
                logger.info("`text_config` is `None`. Initializing the `BridgeTowerTextConfig` with default values.")
    
            # 如果 vision_config 为 None，则使用默认空字典，并记录日志消息
            if vision_config is None:
                vision_config = {}
                logger.info("`vision_config` is `None`. Initializing the `BridgeTowerVisionConfig` with default values.")
    
            # 根据给定的 text_config 创建 BridgeTowerTextConfig 的实例，并赋值给 self.text_config
            self.text_config = BridgeTowerTextConfig(**text_config)
    
            # 根据给定的 vision_config 创建 BridgeTowerVisionConfig 的实例，并赋值给 self.vision_config
            self.vision_config = BridgeTowerVisionConfig(**vision_config)
    
        @classmethod
        # 类方法，从 text_config 和 vision_config 创建 BridgeTowerConfig 的实例
        def from_text_vision_configs(
            cls, text_config: BridgeTowerTextConfig, vision_config: BridgeTowerVisionConfig, **kwargs
        ):
            r"""
            从 BridgeTower 文本模型配置实例化一个 [`BridgeTowerConfig`]（或其派生类）。返回：
                [`BridgeTowerConfig`]: 配置对象的一个实例
            """
    
            # 调用类的构造函数，传入 text_config 和 vision_config 的字典表示，以及其他关键字参数
            return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)
```