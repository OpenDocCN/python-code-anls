# `.\transformers\models\siglip\configuration_siglip.py`

```
# 设置文件编码为 utf-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授权使用此文件
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的具体语言
""" Siglip 模型配置"""

# 导入必要的库
import os
from typing import Union

# 导入预训练配置类和日志记录工具
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练配置文件映射
SIGLIP_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/siglip-base-patch16-224": "https://huggingface.co/google/siglip-base-patch16-224/resolve/main/config.json",
}

# Siglip 文本配置类，继承自预训练配置类
class SiglipTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SiglipTextModel`]. It is used to instantiate a
    Siglip text encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the text encoder of the Siglip
    [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Siglip text model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`SiglipModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        max_position_embeddings (`int`, *optional*, defaults to 64):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        pad_token_id (`int`, *optional*, defaults to 1):
            The id of the padding token in the vocabulary.
        bos_token_id (`int`, *optional*, defaults to 49406):
            The id of the beginning-of-sequence token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 49407):
            The id of the end-of-sequence token in the vocabulary.

    Example:

    ```python
    >>> from transformers import SiglipTextConfig, SiglipTextModel

    >>> # Initializing a SiglipTextConfig with google/siglip-base-patch16-224 style configuration
    >>> configuration = SiglipTextConfig()

    >>> # Initializing a SiglipTextModel (with random weights) from the google/siglip-base-patch16-224 style configuration
    >>> model = SiglipTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    # 定义模型类型为 "siglip_text_model"
    model_type = "siglip_text_model"
    # 初始化函数，设置模型的各种参数
    def __init__(
        self,
        vocab_size=32000,  # 词汇表大小，默认为32000
        hidden_size=768,  # 隐藏层大小，默认为768
        intermediate_size=3072,  # 中间层大小，默认为3072
        num_hidden_layers=12,  # 隐藏层层数，默认为12
        num_attention_heads=12,  # 注意力头数，默认为12
        max_position_embeddings=64,  # 最大位置编码，默认为64
        hidden_act="gelu_pytorch_tanh",  # 隐藏层激活函数，默认为gelu_pytorch_tanh
        layer_norm_eps=1e-6,  # 层归一化的 epsilon，默认为1e-6
        attention_dropout=0.0,  # 注意力机制的 dropout，默认为0.0
        # 与 `CLIPTokenizer` 的默认值和 openai/siglip 不同
        # 参考 https://github.com/huggingface/transformers/pull/24773#issuecomment-1632287538
        pad_token_id=1,  # 填充标记的 ID，默认为1
        bos_token_id=49406,  # 起始标记的 ID，默认为49406
        eos_token_id=49407,  # 结束标记的 ID，默认为49407
        **kwargs,  # 其他参数
    ):
        # 调用父类的初始化函数，设置填充、起始和结束标记的 ID
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 设置模型的各种参数
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout

    # 从预训练模型加载配置
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 设置 token 参数到 kwargs 中
        cls._set_token_in_kwargs(kwargs)

        # 获取配置字典和 kwargs
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果加载的模型类型为 "siglip"，则获取文本配置字典
        if config_dict.get("model_type") == "siglip":
            config_dict = config_dict["text_config"]

        # 如果配置字典中包含模型类型，并且当前类的模型类型与配置不一致，发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 根据配置字典和 kwargs 创建实例
        return cls.from_dict(config_dict, **kwargs)
# 定义一个配置类，用于存储 [`SiglipVisionModel`] 的配置信息。该类用于实例化一个 Siglip 视觉编码器，根据指定的参数定义模型架构。
# 使用默认配置实例化一个配置对象将产生与 Siglip [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224) 架构的视觉编码器类似的配置。
# 配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

class SiglipVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SiglipVisionModel`]. It is used to instantiate a
    Siglip vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the Siglip
    [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input images.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    Example:

    ```python
    >>> from transformers import SiglipVisionConfig, SiglipVisionModel

    >>> # Initializing a SiglipVisionConfig with google/siglip-base-patch16-224 style configuration
    >>> configuration = SiglipVisionConfig()

    >>> # Initializing a SiglipVisionModel (with random weights) from the google/siglip-base-patch16-224 style configuration
    >>> model = SiglipVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    # 模型类型
    model_type = "siglip_vision_model"
    # 初始化函数，设置模型的各种参数
    def __init__(
        self,
        hidden_size=768,  # 隐藏层大小，默认为768
        intermediate_size=3072,  # 中间层大小，默认为3072
        num_hidden_layers=12,  # 隐藏层层数，默认为12
        num_attention_heads=12,  # 注意力头数，默认为12
        num_channels=3,  # 通道数，默认为3
        image_size=224,  # 图像大小，默认为224
        patch_size=16,  # 补丁大小，默认为16
        hidden_act="gelu_pytorch_tanh",  # 隐藏层激活函数，默认为gelu_pytorch_tanh
        layer_norm_eps=1e-6,  # 层归一化的 epsilon，默认为1e-6
        attention_dropout=0.0,  # 注意力机制的 dropout，默认为0.0
        **kwargs,  # 其他参数
    ):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 设置模型的各种参数
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act

    # 从预训练模型加载配置
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 设置 token 参数
        cls._set_token_in_kwargs(kwargs)

        # 获取配置字典和其他参数
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果加载的模型类型为 "siglip"，则获取视觉配置字典
        if config_dict.get("model_type") == "siglip":
            config_dict = config_dict["vision_config"]

        # 如果配置字典中包含模型类型，并且模型类型不等于当前类的模型类型，则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 根据配置字典和其他参数创建实例
        return cls.from_dict(config_dict, **kwargs)
# 定义 SiglipConfig 类，用于存储 SiglipModel 的配置信息
class SiglipConfig(PretrainedConfig):
    r"""
    [`SiglipConfig`] 是用于存储 [`SiglipModel`] 配置的类。它用于根据指定的参数实例化 Siglip 模型，定义文本模型和视觉模型配置。
    使用默认值实例化配置将产生类似于 Siglip [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    Args:
        text_config (`dict`, *optional*):
            用于初始化 [`SiglipTextConfig`] 的配置选项字典。
        vision_config (`dict`, *optional*):
            用于初始化 [`SiglipVisionConfig`] 的配置选项字典。
        kwargs (*optional*):
            关键字参数字典。

    Example:

    ```python
    >>> from transformers import SiglipConfig, SiglipModel

    >>> # 使用 google/siglip-base-patch16-224 风格配置初始化 SiglipConfig
    >>> configuration = SiglipConfig()

    >>> # 使用 google/siglip-base-patch16-224 风格配置初始化 SiglipModel（带有随机权重）
    >>> model = SiglipModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config

    >>> # 也可以从 SiglipTextConfig 和 SiglipVisionConfig 初始化 SiglipConfig
    >>> from transformers import SiglipTextConfig, SiglipVisionConfig

    >>> # 初始化 SiglipText 和 SiglipVision 配置
    >>> config_text = SiglipTextConfig()
    >>> config_vision = SiglipVisionConfig()

    >>> config = SiglipConfig.from_text_vision_configs(config_text, config_vision)
    ```"""

    # 模型类型为 "siglip"
    model_type = "siglip"

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        # 调用父类的构造函数
        super().__init__(**kwargs)

        # 如果 text_config 为 None，则使用默认值初始化 SiglipTextConfig
        if text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `SiglipTextConfig` with default values.")

        # 如果 vision_config 为 None，则使用默认值初始化 SiglipVisionConfig
        if vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. initializing the `SiglipVisionConfig` with default values.")

        # 使用 text_config 初始化 SiglipTextConfig
        self.text_config = SiglipTextConfig(**text_config)
        # 使用 vision_config 初始化 SiglipVisionConfig
        self.vision_config = SiglipVisionConfig(**vision_config)

        # 初始化因子为 1.0
        self.initializer_factor = 1.0

    @classmethod
    # 从文本模型配置和视觉模型配置实例化一个 SiglipConfig（或其派生类）
    def from_text_vision_configs(cls, text_config: SiglipTextConfig, vision_config: SiglipVisionConfig, **kwargs):
        r"""
        从 Siglip 文本模型配置和 Siglip 视觉模型配置实例化一个 SiglipConfig（或其派生类）。

        Returns:
            [`SiglipConfig`]: 一个配置对象的实例
        """

        # 从文本模型配置和视觉模型配置的字典形式实例化一个 SiglipConfig 对象
        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)
```