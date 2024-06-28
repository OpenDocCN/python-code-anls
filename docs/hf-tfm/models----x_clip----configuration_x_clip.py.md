# `.\models\x_clip\configuration_x_clip.py`

```py
# 设置文件编码为 UTF-8，确保代码中文本正确解析
# 版权声明和许可信息，指明此代码的使用权限和限制
# 导入必要的模块和函数，包括 PretrainedConfig 和 logging
import os
from typing import Union

from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取当前模块的日志记录器对象
logger = logging.get_logger(__name__)

# XCLIP_PRETRAINED_CONFIG_ARCHIVE_MAP 是一个映射表，将模型名称映射到其预训练配置文件的 URL
XCLIP_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/xclip-base-patch32": "https://huggingface.co/microsoft/xclip-base-patch32/resolve/main/config.json",
}

# XCLIPTextConfig 是一个配置类，用于存储 X-CLIP 模型的配置信息
class XCLIPTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`XCLIPModel`]. It is used to instantiate an X-CLIP
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the X-CLIP
    [microsoft/xclip-base-patch32](https://huggingface.co/microsoft/xclip-base-patch32) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # 模型类型为文本X-CLIP模型
    model_type = "xclip_text_model"

    # 初始化函数，设置X-CLIP文本模型的各种参数
    def __init__(
        self,
        vocab_size=49408,  # 词汇表大小，默认49408，定义可表示的不同token数量
        hidden_size=512,   # 编码器层和池化器层的维度
        intermediate_size=2048,  # Transformer编码器中"intermediate"（即前馈）层的维度
        num_hidden_layers=12,     # Transformer编码器中的隐藏层层数
        num_attention_heads=8,    # Transformer编码器中每个注意力层的注意头数
        max_position_embeddings=77,  # 模型可能使用的最大序列长度，一般设置为较大值，如512、1024或2048
        hidden_act="quick_gelu",  # 编码器和池化器中的非线性激活函数，支持"gelu"、"relu"、"selu"和"quick_gelu"
        layer_norm_eps=1e-5,      # 层归一化层使用的epsilon值
        attention_dropout=0.0,    # 注意力概率的dropout比率
        initializer_range=0.02,   # 用于初始化所有权重矩阵的截断正态分布的标准差
        initializer_factor=1.0,   # 初始化所有权重矩阵的因子（内部初始化测试时应保持为1）
        pad_token_id=1,           # 填充token的ID
        bos_token_id=0,           # 开始token的ID
        eos_token_id=2,           # 结束token的ID
        **kwargs,                 # 其他关键字参数
    ):
    ):
        # 调用父类的初始化方法，设置Transformer模型的各种参数，包括填充、起始和结束标记的ID
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 设置当前对象的词汇表大小、隐藏层大小、中间层大小、隐藏层的数量、注意力头的数量、最大位置嵌入、层归一化epsilon值、隐藏层激活函数、初始化范围、初始化因子以及注意力机制的dropout率
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 将token相关的参数设置到kwargs中
        cls._set_token_in_kwargs(kwargs)

        # 从预训练模型名称或路径中获取配置字典和更新后的kwargs
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典中的模型类型是"xclip"，则使用其文本配置字典
        if config_dict.get("model_type") == "xclip":
            config_dict = config_dict["text_config"]

        # 如果配置字典中存在"model_type"属性，并且当前类定义了"model_type"属性，且它们不相等，发出警告信息
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 根据配置字典和kwargs创建当前类的实例
        return cls.from_dict(config_dict, **kwargs)
# XCLIPVisionConfig 类，继承自 PretrainedConfig，用于存储 X-CLIP 模型的配置信息
class XCLIPVisionConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`XCLIPModel`] 的配置。根据指定的参数实例化 X-CLIP 模型，定义模型架构。使用默认配置实例化将得到与 X-CLIP
    [microsoft/xclip-base-patch32](https://huggingface.co/microsoft/xclip-base-patch32) 架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。详细信息请参阅 [`PretrainedConfig`] 的文档。
    """
    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            # 编码器层和池化层的维度大小
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            # Transformer 编码器中“中间”（即前馈）层的维度大小
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            # Transformer 编码器中的隐藏层数量
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            # Transformer 编码器中每个注意力层的注意力头数量
            Number of attention heads for each attention layer in the Transformer encoder.
        mit_hidden_size (`int`, *optional*, defaults to 512):
            # Multiframe Integration Transformer（MIT）中编码器层的维度大小
            Dimensionality of the encoder layers of the Multiframe Integration Transformer (MIT).
        mit_intermediate_size (`int`, *optional*, defaults to 2048):
            # Multiframe Integration Transformer（MIT）中“中间”（即前馈）层的维度大小
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Multiframe Integration Transformer (MIT).
        mit_num_hidden_layers (`int`, *optional*, defaults to 1):
            # Multiframe Integration Transformer（MIT）中的隐藏层数量
            Number of hidden layers in the Multiframe Integration Transformer (MIT).
        mit_num_attention_heads (`int`, *optional*, defaults to 8):
            # Multiframe Integration Transformer（MIT）中每个注意力层的注意力头数量
            Number of attention heads for each attention layer in the Multiframe Integration Transformer (MIT).
        image_size (`int`, *optional*, defaults to 224):
            # 每个图像的分辨率大小
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 32):
            # 每个图像块（patch）的分辨率大小
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            # 编码器和池化层中的非线性激活函数
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"`, `"gelu_new"` and ``"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            # 层归一化层使用的 epsilon 值
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            # 注意力概率的 dropout 比率
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            # 用于初始化所有权重矩阵的截断正态分布的标准差
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1):
            # 用于初始化所有权重矩阵的因子（内部测试时保持为1）
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            # 随机深度的比率
            Stochastic depth rate.

    Example:

    ```
    >>> from transformers import XCLIPVisionModel, XCLIPVisionConfig

    >>> # 使用 microsoft/xclip-base-patch32 风格的配置初始化 XCLIPVisionModel
    >>> configuration = XCLIPVisionConfig()

    >>> # 使用 microsoft/xclip-base-patch32 风格的配置初始化 XCLIPVisionModel 模型
    >>> model = XCLIPVisionModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```
    # 设定模型类型为"xclip_vision_model"
    model_type = "xclip_vision_model"

    # 定义初始化方法，设置模型各种参数
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        mit_hidden_size=512,
        mit_intermediate_size=2048,
        mit_num_hidden_layers=1,
        mit_num_attention_heads=8,
        num_channels=3,
        image_size=224,
        patch_size=32,
        num_frames=8,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        drop_path_rate=0.0,
        **kwargs,
    ):
        # 调用父类的初始化方法，传递额外参数
        super().__init__(**kwargs)

        # 设置模型各项参数
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

    @classmethod
    # 从预训练模型加载配置，返回预训练配置的实例
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 设置Token到kwargs中
        cls._set_token_in_kwargs(kwargs)

        # 获取配置字典和更新后的kwargs
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典的模型类型是"xclip"，则使用视觉配置字典
        if config_dict.get("model_type") == "xclip":
            config_dict = config_dict["vision_config"]

        # 如果配置字典中包含模型类型，并且模型类型与当前类的model_type不一致，发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从配置字典创建实例，并传入更新后的kwargs
        return cls.from_dict(config_dict, **kwargs)
# XCLIPConfig 类继承自 PretrainedConfig，用于存储 X-CLIP 模型的配置信息。
# 该配置类包含了初始化 X-CLIP 模型所需的各种参数，定义了文本模型和视觉模型的配置。
# 使用默认参数实例化一个配置对象将会得到与 microsoft/xclip-base-patch32 架构类似的配置。
# 配置对象继承自 PretrainedConfig，并可用于控制模型的输出。详细信息请参阅 PretrainedConfig 的文档。

class XCLIPConfig(PretrainedConfig):
    r"""
    [`XCLIPConfig`] is the configuration class to store the configuration of a [`XCLIPModel`]. It is used to
    instantiate X-CLIP model according to the specified arguments, defining the text model and vision model configs.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the X-CLIP
    [microsoft/xclip-base-patch32](https://huggingface.co/microsoft/xclip-base-patch32) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`XCLIPTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`XCLIPVisionConfig`].
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and vision projection layers.
        prompt_layers (`int`, *optional*, defaults to 2):
            Number of layers in the video specific prompt generator.
        prompt_alpha (`float`, *optional*, defaults to 0.1):
            Alpha value to use in the video specific prompt generator.
        prompt_hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the video specific prompt generator. If string,
            `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        prompt_num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the cross-attention of the video specific prompt generator.
        prompt_attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for the attention layers in the video specific prompt generator.
        prompt_projection_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for the projection layers in the video specific prompt generator.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* parameter. Default is used as per the original XCLIP implementation.
        kwargs (*optional*):
            Dictionary of keyword arguments.
    """

    model_type = "xclip"

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        projection_dim=512,
        prompt_layers=2,
        prompt_alpha=0.1,
        prompt_hidden_act="quick_gelu",
        prompt_num_attention_heads=8,
        prompt_attention_dropout=0.0,
        prompt_projection_dropout=0.0,
        logit_scale_init_value=2.6592,
        **kwargs,
    ):
    # 从文本模型配置和视觉模型配置实例化一个 XCLIPConfig（或其派生类）对象
    def from_text_vision_configs(cls, text_config: XCLIPTextConfig, vision_config: XCLIPVisionConfig, **kwargs):
        r"""
        从 XCLIP 文本模型配置和 XCLIP 视觉模型配置实例化一个 [`XCLIPConfig`]（或其派生类）对象。

        Returns:
            [`XCLIPConfig`]: 配置对象的一个实例
        """

        # 使用 text_config 的字典表示和 vision_config 的字典表示实例化一个配置对象，并传递额外的关键字参数
        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)
```