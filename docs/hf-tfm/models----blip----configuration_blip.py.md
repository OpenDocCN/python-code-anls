# `.\models\blip\configuration_blip.py`

```py
# 导入操作系统和Union类型
import os
from typing import Union

# 从configuration_utils中导入PretrainedConfig类
from ...configuration_utils import PretrainedConfig
# 从utils中导入logging函数
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义一个字典，映射预训练模型名称到其配置文件的URL
BLIP_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Salesforce/blip-vqa-base": "https://huggingface.co/Salesforce/blip-vqa-base/resolve/main/config.json",
    "Salesforce/blip-vqa-capfit-large": (
        "https://huggingface.co/Salesforce/blip-vqa-base-capfit/resolve/main/config.json"
    ),
    "Salesforce/blip-image-captioning-base": (
        "https://huggingface.co/Salesforce/blip-image-captioning-base/resolve/main/config.json"
    ),
    "Salesforce/blip-image-captioning-large": (
        "https://huggingface.co/Salesforce/blip-image-captioning-large/resolve/main/config.json"
    ),
    "Salesforce/blip-itm-base-coco": "https://huggingface.co/Salesforce/blip-itm-base-coco/resolve/main/config.json",
    "Salesforce/blip-itm-large-coco": "https://huggingface.co/Salesforce/blip-itm-large-coco/resolve/main/config.json",
    "Salesforce/blip-itm-base-flikr": "https://huggingface.co/Salesforce/blip-itm-base-flikr/resolve/main/config.json",
    "Salesforce/blip-itm-large-flikr": (
        "https://huggingface.co/Salesforce/blip-itm-large-flikr/resolve/main/config.json"
    ),
}

# 定义BlipTextConfig类，继承自PretrainedConfig类
class BlipTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BlipTextModel`]. It is used to instantiate a BLIP
    text model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the `BlipText` used by the [base
    architectures](https://huggingface.co/Salesforce/blip-vqa-base).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```
    >>> from transformers import BlipTextConfig, BlipTextModel

    >>> # Initializing a BlipTextConfig with Salesforce/blip-vqa-base style configuration
    >>> configuration = BlipTextConfig()

    >>> # Initializing a BlipTextModel (with random weights) from the Salesforce/blip-vqa-base style configuration
    >>> model = BlipTextModel(configuration)

    >>> # Accessing the model configuration

    """
    >>> configuration = model.config
    ```"""
    # 获取模型的配置信息
    configuration = model.config

    model_type = "blip_text_model"
    # 设置模型类型为文本模型

    def __init__(
        self,
        vocab_size=30524,
        hidden_size=768,
        encoder_hidden_size=768,
        intermediate_size=3072,
        projection_dim=768,
        num_hidden_layers=12,
        num_attention_heads=8,
        max_position_embeddings=512,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        bos_token_id=30522,
        eos_token_id=2,
        pad_token_id=0,
        sep_token_id=102,
        is_decoder=True,
        use_cache=True,
        label_smoothing=0.0,
        **kwargs,
    ):
        # 调用父类初始化方法，设置特殊标记的ID，并传入额外的关键字参数
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            sep_token_id=sep_token_id,
            **kwargs,
        )

        # 设置模型配置的各种参数
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.hidden_dropout_prob = hidden_dropout_prob
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.is_decoder = is_decoder
        self.use_cache = use_cache
        self.label_smoothing = label_smoothing

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 设置关键字参数中的token
        cls._set_token_in_kwargs(kwargs)

        # 获取配置字典和处理后的关键字参数
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果从BlipConfig加载，获取文本配置字典
        if config_dict.get("model_type") == "blip":
            config_dict = config_dict["text_config"]

        # 如果配置字典中包含model_type，并且不同于当前类的model_type，发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从配置字典和关键字参数创建实例
        return cls.from_dict(config_dict, **kwargs)
# 定义 BlipVisionConfig 类，用于存储 [`BlipVisionModel`] 的配置信息。该类用于实例化 BLIP 视觉模型，
# 根据指定参数定义模型架构。默认情况下，配置实例化将产生与 Blip-base 架构类似的配置。
class BlipVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BlipVisionModel`]. It is used to instantiate a
    BLIP vision model according to the specified arguments, defining the model architecture. Instantiating a
    configuration defaults will yield a similar configuration to that of the Blip-base
    [Salesforce/blip-vqa-base](https://huggingface.co/Salesforce/blip-vqa-base) architecture.

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
        image_size (`int`, *optional*, defaults to 384):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 1e-10):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:

    ```
    >>> from transformers import BlipVisionConfig, BlipVisionModel

    >>> # Initializing a BlipVisionConfig with Salesforce/blip-vqa-base style configuration
    >>> configuration = BlipVisionConfig()

    >>> # Initializing a BlipVisionModel (with random weights) from the Salesforce/blip-vqa-base style configuration
    >>> model = BlipVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

    model_type = "blip_vision_model"
    # 初始化方法，设置模型的各种参数和超参数
    def __init__(
        self,
        hidden_size=768,                  # 隐藏层的大小，默认为768
        intermediate_size=3072,           # 中间层的大小，默认为3072
        projection_dim=512,               # 投影维度，默认为512
        num_hidden_layers=12,             # 隐藏层数，默认为12
        num_attention_heads=12,           # 注意力头的数量，默认为12
        image_size=384,                   # 图像大小，默认为384
        patch_size=16,                    # 图像块大小，默认为16
        hidden_act="gelu",                # 隐藏层激活函数，默认为gelu
        layer_norm_eps=1e-5,              # 层归一化的 epsilon，默认为1e-5
        attention_dropout=0.0,            # 注意力层的 dropout，默认为0.0
        initializer_range=1e-10,          # 初始化范围，默认为1e-10
        **kwargs,                         # 其他关键字参数
    ):
        super().__init__(**kwargs)        # 调用父类的初始化方法，传入额外的关键字参数

        self.hidden_size = hidden_size    # 设置对象的隐藏层大小属性
        self.intermediate_size = intermediate_size  # 设置对象的中间层大小属性
        self.projection_dim = projection_dim        # 设置对象的投影维度属性
        self.num_hidden_layers = num_hidden_layers  # 设置对象的隐藏层数属性
        self.num_attention_heads = num_attention_heads  # 设置对象的注意力头数量属性
        self.patch_size = patch_size                # 设置对象的图像块大小属性
        self.image_size = image_size                # 设置对象的图像大小属性
        self.initializer_range = initializer_range  # 设置对象的初始化范围属性
        self.attention_dropout = attention_dropout  # 设置对象的注意力 dropout 属性
        self.layer_norm_eps = layer_norm_eps        # 设置对象的层归一化 epsilon 属性
        self.hidden_act = hidden_act                # 设置对象的隐藏层激活函数属性

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)  # 调用类方法设置 kwargs 中的 token

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)  # 获得配置字典和更新的关键字参数

        # 如果配置字典的模型类型是 "blip"，则使用视觉配置字典
        if config_dict.get("model_type") == "blip":
            config_dict = config_dict["vision_config"]

        # 如果配置字典中包含模型类型，并且类属性中的 model_type 不同，发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从配置字典和关键字参数创建对象
        return cls.from_dict(config_dict, **kwargs)
# BlipConfig 类继承自 PretrainedConfig 类，用于存储 BLIP 模型的配置信息。
class BlipConfig(PretrainedConfig):
    r"""
    [`BlipConfig`] 是一个配置类，用于存储 [`BlipModel`] 的配置信息。它用于根据指定的参数实例化一个 BLIP 模型，
    定义文本模型和视觉模型的配置。使用默认参数实例化一个配置对象将得到类似于 BLIP-base
    [Salesforce/blip-vqa-base](https://huggingface.co/Salesforce/blip-vqa-base) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。更多信息请参阅 [`PretrainedConfig`] 的文档。

    Args:
        text_config (`dict`, *optional*):
            用于初始化 [`BlipTextConfig`] 的配置选项字典。
        vision_config (`dict`, *optional*):
            用于初始化 [`BlipVisionConfig`] 的配置选项字典。
        projection_dim (`int`, *optional*, defaults to 512):
            文本和视觉投影层的维度。
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            *logit_scale* 参数的初始值。默认使用原始 BLIP 实现中的值。
        image_text_hidden_size (`int`, *optional*, defaults to 256):
            图像文本融合层隐藏状态的维度。
        label_smoothing (float, optional, *optional*, defaults to 0.0):
            在计算损失时的平滑度，取值范围为 [0.0, 1.0]，其中 0.0 表示无平滑。目标成为原始标签和均匀分布的混合体，详细描述见
            `Rethinking the Inception Architecture for Computer Vision <https://arxiv.org/abs/1512.00567>`__。默认值: :math:`0.0`。
        kwargs (*optional*):
            关键字参数字典。

    Example:

    ```
    >>> from transformers import BlipConfig, BlipModel

    >>> # 使用 Salesforce/blip-vqa-base 风格的配置初始化 BlipConfig
    >>> configuration = BlipConfig()

    >>> # 使用 Salesforce/blip-vqa-base 风格的配置初始化 BlipModel（随机权重）
    >>> model = BlipModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config

    >>> # 也可以从 BlipTextConfig 和 BlipVisionConfig 初始化 BlipConfig

    >>> # 初始化 BLIPText 和 BLIPVision 配置
    >>> config_text = BlipTextConfig()
    >>> config_vision = BlipVisionConfig()

    >>> config = BlipConfig.from_text_vision_configs(config_text, config_vision)
    ```
    """

    # 模型类型标识符
    model_type = "blip"

    # 初始化方法
    def __init__(
        self,
        text_config=None,
        vision_config=None,
        projection_dim=512,
        logit_scale_init_value=2.6592,
        image_text_hidden_size=256,
        label_smoothing=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if text_config is None:
            text_config = {}
            # 如果未提供 `text_config` 参数，则使用默认配置，并记录日志
            logger.info("`text_config` is `None`. Initializing the `BlipTextConfig` with default values.")

        if vision_config is None:
            vision_config = {}
            # 如果未提供 `vision_config` 参数，则使用默认配置，并记录日志
            logger.info("`vision_config` is `None`. Initializing the `BlipVisionConfig` with default values.")

        # 使用提供的 `text_config` 和 `vision_config` 创建 `BlipTextConfig` 和 `BlipVisionConfig` 的实例
        self.text_config = BlipTextConfig(**text_config)
        self.vision_config = BlipVisionConfig(**vision_config)

        # 设置文本编码器的隐藏层大小为视觉模型的隐藏层大小
        self.text_config.encoder_hidden_size = self.vision_config.hidden_size

        # 设置投影维度、logit 缩放初始值、初始化因子、初始化范围、图像文本隐藏层大小和标签平滑度参数
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0
        self.initializer_range = 0.02
        self.image_text_hidden_size = image_text_hidden_size
        self.label_smoothing = label_smoothing

    @classmethod
    def from_text_vision_configs(cls, text_config: BlipTextConfig, vision_config: BlipVisionConfig, **kwargs):
        r"""
        从 Blip 文本模型配置和 Blip 视觉模型配置实例化一个 [`BlipConfig`]（或其派生类）。

        Returns:
            [`BlipConfig`]: 一个配置对象的实例
        """

        # 使用文本配置和视觉配置的字典表示，以及其他可能的关键字参数实例化类
        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)
```