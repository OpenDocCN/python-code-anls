# `.\transformers\models\bridgetower\configuration_bridgetower.py`

```
# 导入所需模块和类
import os
from typing import Union

# 导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 导入日志记录工具
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义预训练模型的配置文件映射字典
BRIDGETOWER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "BridgeTower/bridgetower-base": "https://huggingface.co/BridgeTower/bridgetower-base/blob/main/config.json",
    "BridgeTower/bridgetower-base-itm-mlm": (
        "https://huggingface.co/BridgeTower/bridgetower-base-itm-mlm/blob/main/config.json"
    ),
}

# 定义视觉配置类，继承自预训练配置类
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


    Example:

    ```python
    from transformers import BridgeTowerVisionConfig
    
    # 初始化一个 BridgeTower/bridgetower-base 风格的视觉模型配置
    configuration = BridgeTowerVisionConfig()
    
    # 访问配置对象
    configuration
    
    
    
    # 定义模型类型为 bridgetower_vision_model
    model_type = "bridgetower_vision_model"
    
    # 初始化方法，设置模型各项参数
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
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 设置模型的隐藏层大小
        self.hidden_size = hidden_size
        # 设置模型的隐藏层数量
        self.num_hidden_layers = num_hidden_layers
        # 设置输入图像的通道数
        self.num_channels = num_channels
        # 设置图像分块大小
        self.patch_size = patch_size
        # 设置输入图像的大小
        self.image_size = image_size
        # 设置初始化因子
        self.initializer_factor = initializer_factor
        # 设置层归一化的 epsilon 值
        self.layer_norm_eps = layer_norm_eps
        # 设置是否停止梯度传播
        self.stop_gradient = stop_gradient
        # 设置是否共享层归一化
        self.share_layernorm = share_layernorm
        # 设置是否移除最后一层
        self.remove_last_layer = remove_last_layer
    
    # 从预训练模型加载配置
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 获取配置字典
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
    
        # 如果配置字典中的模型类型是 bridgetower，获取其中的文本配置
        if config_dict.get("model_type") == "bridgetower":
            config_dict = config_dict["text_config"]
    
        # 如果配置字典中包含模型类型，并且该类型与当前类的模型类型不匹配，发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )
    
        # 从配置字典创建配置对象
        return cls.from_dict(config_dict, **kwargs)
# 定义一个 BridgeTowerTextConfig 类，用于存储文本配置信息，继承自 PretrainedConfig 类
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

    # 模型类型为 bridgetower_text_model
    model_type = "bridgetower_text_model"

    # 初始化方法，设置各种配置参数
    def __init__(
        self,
        vocab_size=50265,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        initializer_factor=1,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=514,
        type_vocab_size=1,
        layer_norm_eps=1e-05,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        position_embedding_type="absolute",
        use_cache=True,
        **kwargs,
    ):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 设置各个配置参数
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

    # 类方法
    @classmethod
    # 从预训练模型名或路径加载配置，并返回配置字典及可能更新的额外参数
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 从预训练模型名或路径获取配置字典和可能更新的额外参数
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典中的模型类型为 "bridgetower"，则更新配置字典为文本配置
        if config_dict.get("model_type") == "bridgetower":
            config_dict = config_dict["text_config"]

        # 如果配置字典中包含 "model_type" 键且当前类有 "model_type" 属性，
        # 并且配置字典中的模型类型与当前类的模型类型不同，发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 根据配置字典创建预训练配置对象，并返回
        return cls.from_dict(config_dict, **kwargs)
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
            Whether cross modal transformer layers are shared. 跨模态 transformer 层是否共享
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. 编码器和池化器中的非线性激活函数
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer. 编码器层和池化层的维度
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing). 初始化所有权重矩阵的因子（应保持为1，用于内部初始化测试）
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers. 层归一化层使用的 epsilon
        share_link_tower_layers (`bool`, *optional*, defaults to `False`):
            Whether the bride/link tower layers are shared. 桥/链接塔层是否共享
        link_tower_type (`str`, *optional*, defaults to `"add"`):
            Type of the bridge/link layer. 桥/链接层的类型
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder. Transformer 编码器中每个注意力层的注意力头数
        num_hidden_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder. Transformer 编码器中的隐藏层数
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input and output embeddings. 是否绑定输入和输出嵌入
        init_layernorm_from_vision_encoder (`bool`, *optional*, defaults to `False`):
            Whether to init LayerNorm from the vision encoder. 是否从视觉编码器初始化 LayerNorm
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`BridgeTowerTextConfig`]. 用于初始化 [`BridgeTowerTextConfig`] 的配置选项字典
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`BridgeTowerVisionConfig`]. 用于初始化 [`BridgeTowerVisionConfig`] 的配置选项字典

    Example:

    ```python
    >>> from transformers import BridgeTowerModel, BridgeTowerConfig

    >>> # Initializing a BridgeTower BridgeTower/bridgetower-base style configuration
    >>> configuration = BridgeTowerConfig()

    >>> # Initializing a model from the BridgeTower/bridgetower-base style configuration
    >>> model = BridgeTowerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    # 指定模型类型为“bridgetower”
    model_type = "bridgetower"
    
    # 定义初始化函数，初始化 BridgetowerConfig 类
    def __init__(
        self,
        share_cross_modal_transformer_layers=True,  # 是否共享跨模态 Transformer 层
        hidden_act="gelu",  # 隐藏层激活函数，默认为 GELU
        hidden_size=768,  # 隐藏层大小，默认为 768
        initializer_factor=1,  # 初始化因子，默认为 1
        layer_norm_eps=1e-05,  # 层归一化的 epsilon，默认为 1e-05
        share_link_tower_layers=False,  # 是否共享链接塔层
        link_tower_type="add",  # 链接塔类型，默认为加法
        num_attention_heads=12,  # 注意力头的数量，默认为 12
        num_hidden_layers=6,  # 隐藏层的数量，默认为 6
        tie_word_embeddings=False,  # 是否绑定词嵌入
        init_layernorm_from_vision_encoder=False,  # 是否从视觉编码器初始化层归一化
        text_config=None,  # 文本配置，默认为 None
        vision_config=None,  # 视觉配置，默认为 None
        **kwargs,  # 其他关键字参数
    ):
        # TODO: remove this once the Hub files are updated.
        # 弹出 "text_config_dict" 和 "vision_config_dict" 键值对，一旦 Hub 文件更新，这部分应该移除
        _ = kwargs.pop("text_config_dict", None)
        _ = kwargs.pop("vision_config_dict", None)
    
        # 调用父类初始化函数
        super().__init__(**kwargs)
    
        # 设置属性
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
    
        # 如果文本配置为 None，则使用默认值初始化 BridgeTowerTextConfig 对象，并记录日志信息
        if text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `BridgeTowerTextConfig` with default values.")
    
        # 如果视觉配置为 None，则使用默认值初始化 BridgeTowerVisionConfig 对象，并记录日志信息
        if vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. Initializing the `BridgeTowerVisionConfig` with default values.")
    
        # 使用文本配置和视觉配置初始化 text_config 和 vision_config 属性
        self.text_config = BridgeTowerTextConfig(**text_config)
        self.vision_config = BridgeTowerVisionConfig(**vision_config)
    
    # 类方法：从文本和视觉配置实例化 BridgeTowerConfig 对象
    @classmethod
    def from_text_vision_configs(
        cls, text_config: BridgeTowerTextConfig, vision_config: BridgeTowerVisionConfig, **kwargs
    ):
        r"""
        从 BridgeTower 文本模型配置实例化 [`BridgeTowerConfig`]（或派生类）。返回：
            [`BridgeTowerConfig`]: 配置对象的一个实例
        """
    
        # 返回 BridgeTowerConfig 对象，传入文本配置和视觉配置的字典形式，以及其他关键字参数
        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)
```