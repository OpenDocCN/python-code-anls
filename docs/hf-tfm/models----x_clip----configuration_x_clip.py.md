# `.\transformers\models\x_clip\configuration_x_clip.py`

```py
# 设置文件编码为utf-8
# 版权声明
# 根据Apache License, Version 2.0许可证，你可以在符合许可证的情况下使用该文件
# 你可以在以下网址获取该许可证的副本: http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意, 否则在“AS IS”基础上分发软件
# 没有任何明示或暗示的保证或条件
# 关于特定语言的特定语言执行权限和限制请阅读许可证
""" X-CLIP模型配置"""

# 导入模块
import os
from typing import Union
# 导入必要的模块和类
# PretrainedConfig用于存储和控制预训练模型的配置
from ...configuration_utils import PretrainedConfig
# 日志记录模块
from ...utils import logging

# 获取logger对象
logger = logging.get_logger(__name__)

# 预训练配置映射
XCLIP_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/xclip-base-patch32": "https://huggingface.co/microsoft/xclip-base-patch32/resolve/main/config.json",
}

# XCLIPTextConfig类，用于存储X-CLIP模型的配置
class XCLIPTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`XCLIPModel`]. It is used to instantiate an X-CLIP
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the X-CLIP
    [microsoft/xclip-base-patch32](https://huggingface.co/microsoft/xclip-base-patch32) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 49408):
            Vocabulary size of the X-CLIP text model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`XCLIPModel`].
        hidden_size (`int`, *optional*, defaults to 512):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        max_position_embeddings (`int`, *optional*, defaults to 77):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
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

    Example:

    ```python
    >>> from transformers import XCLIPTextModel, XCLIPTextConfig

    >>> # Initializing a XCLIPTextModel with microsoft/xclip-base-patch32 style configuration
    >>> configuration = XCLIPTextConfig()

    >>> # Initializing a XCLIPTextConfig from the microsoft/xclip-base-patch32 style configuration
    >>> model = XCLIPTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```py"""

    model_type = "xclip_text_model"

    def __init__(
        self,
        vocab_size=49408,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=8,
        max_position_embeddings=77,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs,
    ):  # 定义一个初始化方法，设置一些参数
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)  # 调用父类的初始化方法，设置特殊的标记ID和额外的关键字参数

        self.vocab_size = vocab_size  # 设置词汇表大小
        self.hidden_size = hidden_size  # 设置隐藏层大小
        self.intermediate_size = intermediate_size  # 设置中间层大小
        self.num_hidden_layers = num_hidden_layers  # 设置隐藏层的数量
        self.num_attention_heads = num_attention_heads  # 设置注意力头的数量
        self.max_position_embeddings = max_position_embeddings  # 设置最大位置嵌入
        self.layer_norm_eps = layer_norm_eps  # 设置层归一化数值修正项
        self.hidden_act = hidden_act  # 设置隐藏层激活函数
        self.initializer_range = initializer_range  # 设置初始化范围
        self.initializer_factor = initializer_factor  # 设置初始化因子
        self.attention_dropout = attention_dropout  # 设置注意力丢弃率

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":  # 定义从预训练模型加载参数的类方法，返回预训练配置
        cls._set_token_in_kwargs(kwargs)  # 设置关键字参数中的令牌

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)  # 获取配置字典和关键字参数

        # 如果从XCLIPConfig加载，获取文本配置字典
        if config_dict.get("model_type") == "xclip":
            config_dict = config_dict["text_config"]

        # 检查模型类型是否匹配
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)  # 返回根据字典和关键字参数实例化的配置
# 定义 XCLIPVisionConfig 类，用于存储 X-CLIP 模型的配置信息
class XCLIPVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`XCLIPModel`]. It is used to instantiate an X-CLIP
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the X-CLIP
    [microsoft/xclip-base-patch32](https://huggingface.co/microsoft/xclip-base-patch32) architecture.

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
            mit_hidden_size (`int`, *optional*, defaults to 512):
                Dimensionality of the encoder layers of the Multiframe Integration Transformer (MIT).
            mit_intermediate_size (`int`, *optional*, defaults to 2048):
                Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Multiframe Integration Transformer
                (MIT).
            mit_num_hidden_layers (`int`, *optional*, defaults to 1):
                Number of hidden layers in the Multiframe Integration Transformer (MIT).
            mit_num_attention_heads (`int`, *optional*, defaults to 8):
                Number of attention heads for each attention layer in the Multiframe Integration Transformer (MIT).
            image_size (`int`, *optional*, defaults to 224):
                The size (resolution) of each image.
            patch_size (`int`, *optional*, defaults to 32):
                The size (resolution) of each patch.
            hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
                The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
                `"relu"`, `"selu"`, `"gelu_new"` and ``"quick_gelu"` are supported.
            layer_norm_eps (`float`, *optional*, defaults to 1e-5):
                The epsilon used by the layer normalization layers.
            attention_dropout (`float`, *optional*, defaults to 0.0):
                The dropout ratio for the attention probabilities.
            initializer_range (`float`, *optional*, defaults to 0.02):
                The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            initializer_factor (`float`, *optional*, defaults to 1):
                A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
                testing).
            drop_path_rate (`float`, *optional*, defaults to 0.0):
                Stochastic depth rate.

        Example:

        ```python
        >>> from transformers import XCLIPVisionModel, XCLIPVisionConfig

        >>> # Initializing a XCLIPVisionModel with microsoft/xclip-base-patch32 style configuration
        >>> configuration = XCLIPVisionConfig()

        >>> # Initializing a XCLIPVisionModel model from the microsoft/xclip-base-patch32 style configuration
        >>> model = XCLIPVisionModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    # 设置模型类型为"xclip_vision_model"
    model_type = "xclip_vision_model"

    # 初始化函数，接收各种参数用于配置模型
    def __init__(
        self,
        hidden_size=768,  # 隐藏层大小
        intermediate_size=3072,  # 中间层大小
        num_hidden_layers=12,  # 隐藏层层数
        num_attention_heads=12,  # 注意力头数
        mit_hidden_size=512,  # MIT隐藏层大小
        mit_intermediate_size=2048,  # MIT中间层大小
        mit_num_hidden_layers=1,  # MIT隐藏层层数
        mit_num_attention_heads=8,  # MIT注意力头数
        num_channels=3,  # 通道数
        image_size=224,  # 图像尺寸
        patch_size=32,  # 补丁尺寸
        num_frames=8,  # 帧数
        hidden_act="quick_gelu",  # 隐藏层激活函数
        layer_norm_eps=1e-5,  # LayerNormalization的epsilon值
        attention_dropout=0.0,  # 注意力层的dropout率
        initializer_range=0.02,  # 初始化范围
        initializer_factor=1.0,  # 初始化因子
        drop_path_rate=0.0,  # DropPath率
        **kwargs,  # 其他参数
    ):
        super().__init__(**kwargs)  # 调用父类的初始化函数

        # 设置各种参数
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mit_hidden_size = mit_hidden_size
        self.mit_intermediate_size = mit_intermediate_size
        self.mit_num_hidden_layers = mit_num_hidden_layers
        self.mit_num_attention_heads = mit_num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.drop_path_rate = drop_path_rate

    # 从预训练模型加载配置，参数为预训练模型名称或路径
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)  # 设置kwargs中的token

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)  # 获取配置字典

        # 如果是从XCLIPConfig加载，获取视觉配置字典
        if config_dict.get("model_type") == "xclip":
            config_dict = config_dict["vision_config"]

        # 如果配置字典中存在"model_type"并且模型类型不匹配，则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 根据配置字典创建实例
        return cls.from_dict(config_dict, **kwargs)
# XCLIPConfig 类是用于存储 [`XCLIPModel`] 的配置的配置类。
# 它用于根据指定的参数实例化 X-CLIP 模型，定义文本模型和视觉模型的配置。
# 使用默认值实例化配置将产生与 X-CLIP [microsoft/xclip-base-patch32](https://huggingface.co/microsoft/xclip-base-patch32) 架构相似的配置。
# 配置对象继承自 [`PretrainedConfig`] 并可用于控制模型输出。有关更多信息，请参阅 [`PretrainedConfig`] 的文档。
class XCLIPConfig(PretrainedConfig):

    model_type = "xclip"

    # 初始化方法，用于创建 XCLIPConfig 实例
    def __init__(
        self,
        text_config=None,  # 文本模型配置的字典，默认为 None
        vision_config=None,  # 视觉模型配置的字典，默认为 None
        projection_dim=512,  # 文本和视觉投影层的维度，默认为 512
        prompt_layers=2,  # 视频特定提示生成器中的层数，默认为 2
        prompt_alpha=0.1,  # 视频特定提示生成器中的 alpha 值，默认为 0.1
        prompt_hidden_act="quick_gelu",  # 视频特定提示生成器中的非线性激活函数，默认为 "quick_gelu"
        prompt_num_attention_heads=8,  # 视频特定提示生成器中的跨注意力头数，默认为 8
        prompt_attention_dropout=0.0,  # 视频特定提示生成器中的注意力层的 dropout 概率，默认为 0.0
        prompt_projection_dropout=0.0,  # 视频特定提示生成器中的投影层的 dropout 概率，默认为 0.0
        logit_scale_init_value=2.6592,  # *logit_scale* 参数的初始值，默认为 2.6592，与原始 XCLIP 实现一致
        **kwargs,  # 可变数量的关键字参数
    @classmethod
    def from_text_vision_configs(cls, text_config: XCLIPTextConfig, vision_config: XCLIPVisionConfig, **kwargs):
        r"""
        从xclip文本模型配置和xclip视觉模型配置实例化一个[`XCLIPConfig`]（或派生类）。
    
        返回：
            [`XCLIPConfig`]：配置对象的实例
        """
    
        # 从文本模型配置和视觉模型配置实例化一个配置对象，并返回
        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)
```