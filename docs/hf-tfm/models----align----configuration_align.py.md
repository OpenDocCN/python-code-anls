# `.\transformers\models\align\configuration_align.py`

```
# 设置文件编码为 UTF-8
# 版权声明：2023 年由 HuggingFace Inc. 团队所有。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）授权；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件，
# 没有任何明示或暗示的保证或条件。
# 有关详细信息，请参阅许可证。
""" ALIGN 模型配置"""

# 导入必要的库
import os
from typing import TYPE_CHECKING, List, Union

# 如果是类型检查阶段，执行以下代码
if TYPE_CHECKING:
    pass

# 导入配置工具和日志记录工具
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练模型配置映射
ALIGN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "kakaobrain/align-base": "https://huggingface.co/kakaobrain/align-base/resolve/main/config.json",
}

# AlignTextConfig 类，用于存储 [`AlignTextModel`] 的配置
class AlignTextConfig(PretrainedConfig):
    r"""
    这是用于存储 [`AlignTextModel`] 配置的配置类。根据指定的参数，它用于实例化一个 ALIGN 文本编码器，定义模型架构。
    使用默认值实例化配置将产生与 ALIGN [kakaobrain/align-base](https://huggingface.co/kakaobrain/align-base) 架构的文本编码器类似的配置。
    这里的默认值是从 BERT 复制过来的。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    示例：

    ```python
    >>> from transformers import AlignTextConfig, AlignTextModel

    >>> # 使用 kakaobrain/align-base 样式配置初始化 AlignTextConfig
    >>> configuration = AlignTextConfig()

    >>> # 从 kakaobrain/align-base 样式配置初始化一个 AlignTextModel（带有随机权重）
    >>> model = AlignTextModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```"""

    # 模型类型
    model_type = "align_text_model"

    # 初始化方法
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        **kwargs,
    # 初始化函数，接受一系列参数并调用父类的初始化方法
    ):
        super().__init__(**kwargs)

        # 设置词汇表大小
        self.vocab_size = vocab_size
        # 设置隐藏层大小
        self.hidden_size = hidden_size
        # 设置隐藏层数量
        self.num_hidden_layers = num_hidden_layers
        # 设置注意力头的数量
        self.num_attention_heads = num_attention_heads
        # 设置隐藏层激活函数
        self.hidden_act = hidden_act
        # 设置中间层大小
        self.intermediate_size = intermediate_size
        # 设置隐藏层的丢弃概率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 设置注意力概率的丢弃概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 设置最大位置嵌入
        self.max_position_embeddings = max_position_embeddings
        # 设置类型词汇表大小
        self.type_vocab_size = type_vocab_size
        # 设置初始化范围
        self.initializer_range = initializer_range
        # 设置层归一化的 epsilon 值
        self.layer_norm_eps = layer_norm_eps
        # 设置位置嵌入类型
        self.position_embedding_type = position_embedding_type
        # 设置是否使用缓存
        self.use_cache = use_cache
        # 设置填充 token 的 id
        self.pad_token_id = pad_token_id

    @classmethod
    # 从预训练模型加载配置
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 在 kwargs 中设置 token
        cls._set_token_in_kwargs(kwargs)

        # 获取配置字典和可能存在的额外参数
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果从 AlignConfig 加载，则获取文本配置字典
        if config_dict.get("model_type") == "align":
            config_dict = config_dict["text_config"]

        # 如果配置字典中包含模型类型，并且类有 model_type 属性，并且配置字典中的模型类型与类的模型类型不同，则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从配置字典创建实例并返回
        return cls.from_dict(config_dict, **kwargs)
```  
class AlignVisionConfig(PretrainedConfig):
    r"""
    这是配置类，用于存储 [`AlignVisionModel`] 的配置。它用于根据指定的参数实例化一个 ALIGN 视觉编码器，定义模型架构。使用默认值实例化一个配置会产生类似于 ALIGN [kakaobrain/align-base](https://huggingface.co/kakaobrain/align-base) 架构的视觉编码器的配置。默认值是从 EfficientNet (efficientnet-b7) 复制过来的。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。
    """
    Args:
        num_channels (`int`, *optional*, defaults to 3):
            输入通道的数量。
        image_size (`int`, *optional*, defaults to 600):
            输入图像的大小。
        width_coefficient (`float`, *optional*, defaults to 2.0):
            每个阶段网络宽度的缩放系数。
        depth_coefficient (`float`, *optional*, defaults to 3.1):
            每个阶段网络深度的缩放系数。
        depth_divisor `int`, *optional*, defaults to 8):
            网络宽度的一个单位。
        kernel_sizes (`List[int]`, *optional*, defaults to `[3, 3, 5, 3, 5, 5, 3]`):
            每个块中要使用的内核大小列表。
        in_channels (`List[int]`, *optional*, defaults to `[32, 16, 24, 40, 80, 112, 192]`):
            用于卷积层中每个块中要使用的输入通道大小列表。
        out_channels (`List[int]`, *optional*, defaults to `[16, 24, 40, 80, 112, 192, 320]`):
            用于卷积层中每个块中要使用的输出通道大小列表。
        depthwise_padding (`List[int]`, *optional*, defaults to `[]`):
            具有方形填充的块索引列表。
        strides (`List[int]`, *optional*, defaults to `[1, 2, 2, 2, 1, 2, 1]`):
            每个块中要使用的步幅大小列表。
        num_block_repeats (`List[int]`, *optional*, defaults to `[1, 2, 2, 3, 3, 4, 1]`):
            每个块要重复的次数列表。
        expand_ratios (`List[int]`, *optional*, defaults to `[1, 6, 6, 6, 6, 6, 6]`):
            每个块的缩放系数列表。
        squeeze_expansion_ratio (`float`, *optional*, defaults to 0.25):
            挤压扩展比率。
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            每个块中的非线性激活函数（函数或字符串）。如果是字符串，支持 `"gelu"`, `"relu"`, `"selu"`, `"gelu_new"`, `"silu"` 和 `"mish"`。
        hiddem_dim (`int`, *optional*, defaults to 1280):
            分类头之前层的隐藏维度。
        pooling_type (`str` or `function`, *optional*, defaults to `"mean"`):
            应用于密集分类头之前的最终池化类型。可用选项为 [`"mean"`, `"max"`]。
        initializer_range (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态初始化器的标准差。
        batch_norm_eps (`float`, *optional*, defaults to 1e-3):
            批量归一化层使用的 epsilon。
        batch_norm_momentum (`float`, *optional*, defaults to 0.99):
            批量归一化层使用的动量。
        drop_connect_rate (`float`, *optional*, defaults to 0.2):
            跳过连接的丢弃率。

    Example:

    ```python
    # 导入 AlignVisionConfig 和 AlignVisionModel 类
    >>> from transformers import AlignVisionConfig, AlignVisionModel

    # 使用 kakaobrain/align-base 风格的配置初始化 AlignVisionConfig
    >>> configuration = AlignVisionConfig()

    # 使用 kakaobrain/align-base 风格的配置初始化一个具有随机权重的 AlignVisionModel
    >>> model = AlignVisionModel(configuration)

    # 获取模型的配置信息
    >>> configuration = model.config
    ```


    # 定义模型类型为 "align_vision_model"
    model_type = "align_vision_model"

    # 定义模型类的初始化方法，包括一系列参数，用于初始化模型对象
    def __init__(
        self,
        num_channels: int = 3,
        image_size: int = 600,
        width_coefficient: float = 2.0,
        depth_coefficient: float = 3.1,
        depth_divisor: int = 8,
        kernel_sizes: List[int] = [3, 3, 5, 3, 5, 5, 3],
        in_channels: List[int] = [32, 16, 24, 40, 80, 112, 192],
        out_channels: List[int] = [16, 24, 40, 80, 112, 192, 320],
        depthwise_padding: List[int] = [],
        strides: List[int] = [1, 2, 2, 2, 1, 2, 1],
        num_block_repeats: List[int] = [1, 2, 2, 3, 3, 4, 1],
        expand_ratios: List[int] = [1, 6, 6, 6, 6, 6, 6],
        squeeze_expansion_ratio: float = 0.25,
        hidden_act: str = "swish",
        hidden_dim: int = 2560,
        pooling_type: str = "mean",
        initializer_range: float = 0.02,
        batch_norm_eps: float = 0.001,
        batch_norm_momentum: float = 0.99,
        drop_connect_rate: float = 0.2,
        **kwargs,
    ):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 初始化模型参数
        self.num_channels = num_channels
        self.image_size = image_size
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.depth_divisor = depth_divisor
        self.kernel_sizes = kernel_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depthwise_padding = depthwise_padding
        self.strides = strides
        self.num_block_repeats = num_block_repeats
        self.expand_ratios = expand_ratios
        self.squeeze_expansion_ratio = squeeze_expansion_ratio
        self.hidden_act = hidden_act
        self.hidden_dim = hidden_dim
        self.pooling_type = pooling_type
        self.initializer_range = initializer_range
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum
        self.drop_connect_rate = drop_connect_rate
        self.num_hidden_layers = sum(num_block_repeats) * 4

    # 类方法
    @classmethod
    # 从预训练模型名称或路径创建一个预训练配置对象
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 在关键字参数中设置令牌
        cls._set_token_in_kwargs(kwargs)

        # 获取预训练模型的配置字典和更新后的关键字参数
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典中的模型类型为"align"，则获取视觉配置字典
        if config_dict.get("model_type") == "align":
            config_dict = config_dict["vision_config"]

        # 如果配置字典中存在"model_type"并且类中有"model_type"属性，并且配置字典中的模型类型与类的模型类型不匹配，则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 根据配置字典创建一个预训练配置对象
        return cls.from_dict(config_dict, **kwargs)
# 定义 AlignConfig 类，用于存储 AlignModel 的配置信息
class AlignConfig(PretrainedConfig):
    r"""
    [`AlignConfig`] is the configuration class to store the configuration of a [`AlignModel`]. It is used to
    instantiate a ALIGN model according to the specified arguments, defining the text model and vision model configs.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the ALIGN
    [kakaobrain/align-base](https://huggingface.co/kakaobrain/align-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional`):
            Dictionary of configuration options used to initialize [`AlignTextConfig`].
        vision_config (`dict`, *optional`):
            Dictionary of configuration options used to initialize [`AlignVisionConfig`].
        projection_dim (`int`, *optional`, defaults to 640):
            Dimentionality of text and vision projection layers.
        temperature_init_value (`float`, *optional`, defaults to 1.0):
            The inital value of the *temperature* paramter. Default is used as per the original ALIGN implementation.
        initializer_range (`float`, *optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        kwargs (*optional`):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import AlignConfig, AlignModel

    >>> # Initializing a AlignConfig with kakaobrain/align-base style configuration
    >>> configuration = AlignConfig()

    >>> # Initializing a AlignModel (with random weights) from the kakaobrain/align-base style configuration
    >>> model = AlignModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a AlignConfig from a AlignTextConfig and a AlignVisionConfig
    >>> from transformers import AlignTextConfig, AlignVisionConfig

    >>> # Initializing ALIGN Text and Vision configurations
    >>> config_text = AlignTextConfig()
    >>> config_vision = AlignVisionConfig()

    >>> config = AlignConfig.from_text_vision_configs(config_text, config_vision)
    ```"""

    # 模型类型为 "align"
    model_type = "align"

    # 初始化方法，接受多个参数和关键字参数
    def __init__(
        self,
        text_config=None,
        vision_config=None,
        projection_dim=640,
        temperature_init_value=1.0,
        initializer_range=0.02,
        **kwargs,
    # 调用父类构造函数，并传递关键字参数
    ):
        super().__init__(**kwargs)

        # 如果文本配置为None，则使用默认空字典，并记录日志信息
        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the AlignTextConfig with default values.")

        # 如果视觉配置为None，则使用默认空字典，并记录日志信息
        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. Initializing the AlignVisionConfig with default values.")

        # 根据文本配置和视觉配置实例化AlignTextConfig和AlignVisionConfig对象
        self.text_config = AlignTextConfig(**text_config)
        self.vision_config = AlignVisionConfig(**vision_config)

        # 设置投影维度、温度初始值和初始化范围
        self.projection_dim = projection_dim
        self.temperature_init_value = temperature_init_value
        self.initializer_range = initializer_range

    @classmethod
    def from_text_vision_configs(cls, text_config: AlignTextConfig, vision_config: AlignVisionConfig, **kwargs):
        r"""
        从文本模型配置和视觉模型配置实例化一个[`AlignConfig`]（或其派生类）。

        Args:
            text_config (AlignTextConfig): 文本模型配置对象
            vision_config (AlignVisionConfig): 视觉模型配置对象
            **kwargs: 其他关键字参数

        Returns:
            [`AlignConfig`]: 配置对象的实例
        """

        # 使用文本配置和视觉配置的字典形式来初始化一个新的配置对象
        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)

```  
```