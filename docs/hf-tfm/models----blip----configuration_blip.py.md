# `.\transformers\models\blip\configuration_blip.py`

```py
# 设置编码格式为 utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
# 版权声明，声明代码版权归 HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按"原样"提供，不提供任何明示或暗示的担保或条件。
# 有关更多信息，请参阅许可证

# 导入必要的模块
import os
from typing import Union

# 从 huggingface 库中导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 导入日志记录工具
from ...utils import logging

# 获取 logger 对象用于记录日志
logger = logging.get_logger(__name__)

# 定义预训练模型名称到配置文件地址的映射字典
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

# 定义 BLIP 文本模型的配置类，继承自预训练配置类 PretrainedConfig
class BlipTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BlipTextModel`]. It is used to instantiate a BLIP
    text model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the `BlipText` used by the [base
    architectures](https://huggingface.co/Salesforce/blip-vqa-base).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    
    这是用于存储 [`BlipTextModel`] 配置的配置类。根据指定的参数实例化 BLIP 文本模型，定义模型架构。使用默认配置实例化配置对象将产生类似于 [基础架构](https://huggingface.co/Salesforce/blip-vqa-base) 使用的 `BlipText` 的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。请阅读 [`PretrainedConfig`] 的文档以获取更多信息。
    Args:
        vocab_size (`int`, *optional*, defaults to 30524):
            定义`Blip`文本模型的词汇表大小。定义了在调用[`BlipModel`]时可以表示的不同标记数量。
        hidden_size (`int`, *optional*, defaults to 768):
            编码器层和池化层的维度。
        encoder_hidden_size (`int`, *optional*, defaults to 768):
            来自视觉模型的编码器层的维度。
        intermediate_size (`int`, *optional*, defaults to 3072):
            Transformer编码器中“中间”（即前馈）层的维度。
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Transformer编码器中的隐藏层数量。
        num_attention_heads (`int`, *optional*, defaults to 8):
            Transformer编码器中每个注意力层的注意力头数量。
        max_position_embeddings (`int`, *optional*, defaults to 512):
            此模型可能使用的最大序列长度。通常设置为较大的值（例如512、1024或2048）以防万一。
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            编码器和池化器中的非线性激活函数（函数或字符串）。如果是字符串，支持`"gelu"`、`"relu"`、`"selu"`和`"gelu_new"` `"gelu"`。
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            层归一化层使用的epsilon。
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            嵌入层、编码器和池化器中所有全连接层的丢失概率。
        attention_dropout (`float`, *optional*, defaults to 0.0):
            注意力概率的丢失比率。
        initializer_range (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态初始化器的标准差。
        bos_token_id (`int`, *optional*, defaults to 30522):
            `序列开始`标记的ID。
        eos_token_id (`int`, *optional*, defaults to 2):
            `序列结束`标记的ID。
        pad_token_id (`int`, *optional*, defaults to 0):
            `填充`标记的ID。
        sep_token_id (`int`, *optional*, defaults to 102):
            `分隔符`标记的ID。
        is_decoder (`bool`, *optional*, defaults to `True`):
            模型是否用作解码器。
        use_cache (`bool`, *optional*, defaults to `True`):
            模型是否应返回最后一个键/值注意力（并非所有模型都使用）。

    Example:

    ```python
    >>> from transformers import BlipTextConfig, BlipTextModel

    >>> # 使用Salesforce/blip-vqa-base风格配置初始化BlipTextConfig
    >>> configuration = BlipTextConfig()

    >>> # 从 Salesforce/blip-vqa-base 风格的配置初始化 BlipTextModel（带有随机权重）
    >>> model = BlipTextModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    """"

    # 模型类型
    model_type = "blip_text_model"

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
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            sep_token_id=sep_token_id,
            **kwargs,
        )

        # 初始化 BlipTextModel 类的实例
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

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        # 从预训练模型的名称或路径加载配置字典
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果从 BlipConfig 加载，则获取文本配置字典
        if config_dict.get("model_type") == "blip":
            config_dict = config_dict["text_config"]

        # 如果配置的模型类型与类的模型类型不匹配，则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从配置字典创建 BlipTextConfig 类的实例
        return cls.from_dict(config_dict, **kwargs)
# 定义 BLIP 视觉模型的配置类
class BlipVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BlipVisionModel`]. It is used to instantiate a
    BLIP vision model according to the specified arguments, defining the model architecture. Instantiating a
    configuration defaults will yield a similar configuration to that of the Blip-base
    [Salesforce/blip-vqa-base](https://huggingface.co/Salesforce/blip-vqa-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    # 初始化 BLIP 视觉模型的配置对象

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

    # 参数说明：

    Example:

    ```py
    >>> from transformers import BlipVisionConfig, BlipVisionModel

    >>> # Initializing a BlipVisionConfig with Salesforce/blip-vqa-base style configuration
    >>> configuration = BlipVisionConfig()

    >>> # Initializing a BlipVisionModel (with random weights) from the Salesforce/blip-vqa-base style configuration
    >>> model = BlipVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    # 指定模型类型为 "blip_vision_model"
    model_type = "blip_vision_model"
    # 初始化函数，设置模型的各种参数
    def __init__(
        self,
        hidden_size=768,  # 隐藏层的大小，默认为768
        intermediate_size=3072,  # 中间层的大小，默认为3072
        projection_dim=512,  # 投影维度，默认为512
        num_hidden_layers=12,  # 隐藏层的数量，默认为12
        num_attention_heads=12,  # 注意力头的数量，默认为12
        image_size=384,  # 图像大小，默认为384
        patch_size=16,  # 补丁大小，默认为16
        hidden_act="gelu",  # 隐藏层激活函数，默认为gelu
        layer_norm_eps=1e-5,  # 层归一化的 epsilon，默认为1e-5
        attention_dropout=0.0,  # 注意力机制的 dropout，默认为0.0
        initializer_range=1e-10,  # 初始化范围，默认为1e-10
        **kwargs,  # 其他参数
    ):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 设置模型的各种参数
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act

    # 从预训练模型中加载配置信息
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 设置 token 参数
        cls._set_token_in_kwargs(kwargs)

        # 获取配置字典和其他参数
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果是从 BlipConfig 加载，获取视觉配置字典
        if config_dict.get("model_type") == "blip":
            config_dict = config_dict["vision_config"]

        # 如果配置字典中包含模型类型，并且模型类型不匹配当前类的模型类型，发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 根据配置字典和其他参数创建实例
        return cls.from_dict(config_dict, **kwargs)
# 定义 BlipConfig 类，用于存储 BlipModel 的配置信息。该类用于实例化 BLIP 模型，根据指定的参数定义文本模型和视觉模型配置。
# 实例化一个具有默认设置的配置将产生类似于 BLIP-base [Salesforce/blip-vqa-base](https://huggingface.co/Salesforce/blip-vqa-base) 架构的配置。
# 配置对象继承自 PretrainedConfig，可用于控制模型输出。有关更多信息，请阅读 PretrainedConfig 的文档。

class BlipConfig(PretrainedConfig):

    # 模型类型
    model_type = "blip"

    # 初始化方法，接受多个参数，其中有些是可选的
    def __init__(
        self,
        text_config=None, # 文本配置，用于初始化 BlipTextConfig 的配置选项的字典
        vision_config=None, # 视觉配置，用于初始化 BlipVisionConfig 的配置选项的字典
        projection_dim=512, # 文本和视觉投影层的维度，默认为512
        logit_scale_init_value=2.6592, # logit_scale 参数的初始值，默认为2.6592，按照原始 BLIP 实现使用默认值
        image_text_hidden_size=256, # 图像文本融合层隐藏状态的维度，默认为256
        **kwargs, # 可选的关键字参数
    ):
        # 调用父类的构造函数，传入关键字参数
        super().__init__(**kwargs)

        # 如果文本配置为空，则初始化为一个空字典
        if text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `BlipTextConfig` with default values.")

        # 如果视觉配置为空，则初始化为一个空字典
        if vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. Initializing the `BlipVisionConfig` with default values.")

        # 使用文本配置和视觉配置初始化 BlipTextConfig 和 BlipVisionConfig 对象
        self.text_config = BlipTextConfig(**text_config)
        self.vision_config = BlipVisionConfig(**vision_config)

        # 将编码器隐藏层大小设置为视觉配置的隐藏层大小
        self.text_config.encoder_hidden_size = self.vision_config.hidden_size

        # 初始化投影维度、logit 缩放初始值、初始化因子、初始化范围和图像文本隐藏层大小
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0
        self.initializer_range = 0.02
        self.image_text_hidden_size = image_text_hidden_size

    @classmethod
    def from_text_vision_configs(cls, text_config: BlipTextConfig, vision_config: BlipVisionConfig, **kwargs):
        r"""
        从 blip 文本模型配置和 blip 视觉模型配置实例化一个 [`BlipConfig`]（或其派生类）。

        返回:
            [`BlipConfig`]: 配置对象的一个实例
        """

        # 从文本配置和视觉配置的字典形式实例化一个 BlipConfig 对象
        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)
```