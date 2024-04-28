# `.\transformers\models\owlv2\configuration_owlv2.py`

```
# coding=utf-8
# 版权声明：此代码版权归 The HuggingFace Inc. 团队所有
#
# 根据 Apache License, Version 2.0 ("License") 授权许可；
# 你可以在符合许可的情况下使用此文件，但是请注意：
# 除非有适用法律要求或书面同意，软件分发基于"原样"的方式，不提供任何明示或暗示的保证或条件。
# 请阅读许可的具体内容：http://www.apache.org/licenses/LICENSE-2.0
#
# 假如需要检查模型和输入类型是否支持，使用 TYPE_CHECKING 对其进行进行判断
# 我们没有提供 TYPE_CHECKING 任何的代码
from typing import TYPE_CHECKING, Dict, Union

# 使用以下import语句导入必要的模块、类和函数
# 如果TYPE_CHECKING是True，skip以下import
if TYPE_CHECKING:
    pass

# 使用以下import语句导入必要的模块、类和函数
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 使用loging模块设置logger对象
logger = logging.get_logger(__name__)

# 创建字典OWLV2_PRETRAINED_CONFIG_ARCHIVE_MAP，用于存储模型的预训练配置
OWLV2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/owlv2-base-patch16": "https://huggingface.co/google/owlv2-base-patch16/resolve/main/config.json",
}

# 定义Owlv2TextConfig类，继承PretrainedConfig类
# 该类用于存储`Owlv2TextModel`的配置信息
# 通过指定的参数来初始化Owlv2文本编码器，以定义模型架构。
# 使用默认参数实例化一个配置将生成与Owlv2 ['google/owlv2-base-patch16'](https://huggingface.co/google/owlv2-base-patch16)架构类似的配置。
# 配置对象继承自`PretrainedConfig`，可以用来控制模型的输出。
# 阅读 `PretrainedConfig` 的文档以获得更多信息。
class Owlv2TextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`Owlv2TextModel`]. It is used to instantiate an
    Owlv2 text encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Owlv2
    [google/owlv2-base-patch16](https://huggingface.co/google/owlv2-base-patch16) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # 定义 Owlv2TextConfig 类的参数说明
    Args:
        # 文本模型词汇表大小，默认为 49408
        vocab_size (`int`, *optional*, defaults to 49408):
            Vocabulary size of the OWLv2 text model. Defines the number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`Owlv2TextModel`].
        # 编码器层和池化层的维度，默认为 512
        hidden_size (`int`, *optional*, defaults to 512):
            Dimensionality of the encoder layers and the pooler layer.
        # 中间层的维度，默认为 2048
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        # 编码器层数，默认为 12
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        # 注意力头数，默认为 8
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        # 最大序列长度，默认为 16
        max_position_embeddings (`int`, *optional*, defaults to 16):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        # 激活函数，默认为 "quick_gelu"
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        # 层归一化的 epsilon 值，默认为 1e-05
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        # 注意力权重的丢弃率，默认为 0.0
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        # 权重初始化的标准差，默认为 0.02
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        # 权重初始化的缩放因子，默认为 1.0
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        # 填充令牌 ID，默认为 0
        pad_token_id (`int`, *optional*, defaults to 0):
            The id of the padding token in the input sequences.
        # 开始令牌 ID，默认为 49406
        bos_token_id (`int`, *optional*, defaults to 49406):
            The id of the beginning-of-sequence token in the input sequences.
        # 结束令牌 ID，默认为 49407
        eos_token_id (`int`, *optional*, defaults to 49407):
            The id of the end-of-sequence token in the input sequences.
    
    # 示例用法
    Example:
    
    
    >>> from transformers import Owlv2TextConfig, Owlv2TextModel
    
    >>> # Initializing a Owlv2TextModel with google/owlv2-base-patch16 style configuration
    >>> configuration = Owlv2TextConfig()
    
    >>> # Initializing a Owlv2TextConfig from the google/owlv2-base-patch16 style configuration
    >>> model = Owlv2TextModel(configuration)
    
    >>> # Accessing the model configuration
    >>> configuration = model.config
    
    
    # 设置模型类型
    model_type = "owlv2_text_model"
    # 初始化函数，设置各种模型参数的默认数值
    def __init__(
        self,
        vocab_size=49408,  # 词汇表大小，默认49408
        hidden_size=512,  # 隐藏层大小，默认512
        intermediate_size=2048,  # 中间层大小，默认2048
        num_hidden_layers=12,  # 隐藏层层数，默认12
        num_attention_heads=8,  # 注意力头数，默认8
        max_position_embeddings=16,  # 最大位置嵌入数，默认16
        hidden_act="quick_gelu",  # 隐藏层激活函数，默认快速GELU
        layer_norm_eps=1e-5,  # 层归一化的epsilon值，默认1e-5
        attention_dropout=0.0,  # 注意力层的dropout率，默认0.0
        initializer_range=0.02,  # 初始化范围，默认0.02
        initializer_factor=1.0,  # 初始化因子，默认1.0
        pad_token_id=0,  # 填充标记的ID，默认0
        bos_token_id=49406,  # 起始标记的ID，默认49406
        eos_token_id=49407,  # 结束标记的ID，默认49407
        **kwargs,
    ):
        # 调用父类的初始化函数，传入填充标记、起始标记、结束标记等参数
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 设置模型参数
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 设置kwargs中的token参数
        cls._set_token_in_kwargs(kwargs)

        # 调用get_config_dict函数获取配置字典和更新后的kwargs
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果是从Owlv2Config加载的，则获取文本配置字典
        if config_dict.get("model_type") == "owlv2":
            config_dict = config_dict["text_config"]

        # 如果配置字典中包含模型类型，并且当前类的模型类型与配置字典中的模型类型不匹配，则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 返回从配置字典中创建的实例对象
        return cls.from_dict(config_dict, **kwargs)
# 从 transformers.models.owlvit.configuration_owlvit.OwlViTVisionConfig 复制过来，将 OwlViT->Owlv2, owlvit-base-patch32->owlv2-base-patch16, owlvit->owlv2, OWL-ViT->OWLv2, 32->16
class Owlv2VisionConfig(PretrainedConfig):
    r"""
    这是用于存储 [`Owlv2VisionModel`] 配置的配置类。它用于根据指定的参数实例化 OWLv2 图像编码器，定义模型架构。使用默认参数实例化配置将产生与 OWLv2 [google/owlv2-base-patch16](https://huggingface.co/google/owlv2-base-patch16) 架构类似的配置。

    配置对象继承自 [`PretrainedConfig`] 并可用于控制模型输出。有关更多信息，请阅读 [`PretrainedConfig`] 的文档。

    Args:
        hidden_size (`int`, *optional*, 默认为 768):
            编码器层和汇聚层的维度。
        intermediate_size (`int`, *optional*, 默认为 3072):
            Transformer 编码器中的 "intermediate"（即，前馈）层的维度。
        num_hidden_layers (`int`, *optional*, 默认为 12):
            Transformer 编码器中的隐藏层数量。
        num_attention_heads (`int`, *optional*, 默认为 12):
            Transformer 编码器中每个注意力层的注意力头数量。
        num_channels (`int`, *optional*, 默认为 3):
            输入图像中的通道数量。
        image_size (`int`, *optional*, 默认为 768):
            每个图像的大小（分辨率）。
        patch_size (`int`, *optional*, 默认为 16):
            每个补丁的大小（分辨率）。
        hidden_act (`str` or `function`, *optional*, 默认为 `"quick_gelu"`):
            编码器和池化器中的非线性激活函数（函数或字符串）。如果是字符串，则支持 `"gelu"`, `"relu"`, `"selu"` 和 `"gelu_new"` 以及 `"quick_gelu"`。
        layer_norm_eps (`float`, *optional*, 默认为 1e-05):
            层归一化层使用的 epsilon。
        attention_dropout (`float`, *optional*, 默认为 0.0):
            注意力概率的丢弃比率。
        initializer_range (`float`, *optional*, 默认为 0.02):
            用于初始化所有权重矩阵的截断正态初始化器的标准差。
        initializer_factor (`float`, *optional*, 默认为 1.0):
            用于初始化所有权重矩阵的因子（应保持为 1，内部用于初始化测试）。

    Example:

    ```python
    >>> from transformers import Owlv2VisionConfig, Owlv2VisionModel

    >>> # 使用 google/owlv2-base-patch16 风格的配置初始化 Owlv2VisionModel
    >>> configuration = Owlv2VisionConfig()

    >>> # 从 google/owlv2-base-patch16 风格的配置初始化 Owlv2VisionModel 模型
    # 创建 Owlv2VisionModel 的实例
    >>> model = Owlv2VisionModel(configuration)
    
    # 访问模型配置信息
    >>> configuration = model.config
    
    # Owlv2VisionModel 类定义
    class Owlv2VisionModel(PreTrainedModel):
        # 模型类型为 'owlv2_vision_model'
        model_type = "owlv2_vision_model"
    
        def __init__(
            self,
            # 隐藏层大小
            hidden_size=768,
            # 中间层大小
            intermediate_size=3072, 
            # 隐藏层数量
            num_hidden_layers=12,
            # 注意力头数量
            num_attention_heads=12,
            # 输入图像通道数
            num_channels=3,
            # 输入图像尺寸
            image_size=768,
            # 图像分块大小
            patch_size=16,
            # 激活函数
            hidden_act="quick_gelu",
            # Layer Norm 参数
            layer_norm_eps=1e-5,
            # Attention Dropout
            attention_dropout=0.0,
            # 权重初始化范围
            initializer_range=0.02,
            # 初始化因子
            initializer_factor=1.0,
            **kwargs,
        ):
            # 调用父类的初始化方法
            super().__init__(**kwargs)
    
            # 设置各属性值
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.num_channels = num_channels
            self.image_size = image_size
            self.patch_size = patch_size
            self.hidden_act = hidden_act
            self.layer_norm_eps = layer_norm_eps
            self.attention_dropout = attention_dropout
            self.initializer_range = initializer_range
            self.initializer_factor = initializer_factor
    
        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
            # 设置token相关参数
            cls._set_token_in_kwargs(kwargs)
    
            # 获取预训练模型配置信息
            config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
    
            # 如果是从 Owlv2Config 加载的，提取 vision_config
            if config_dict.get("model_type") == "owlv2":
                config_dict = config_dict["vision_config"]
    
            # 如果模型类型不匹配，给出警告
            if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
                logger.warning(
                    f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                    f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
                )
    
            # 从字典创建配置实例
            return cls.from_dict(config_dict, **kwargs)
# 定义 OWLv2 模型的配置类 Owlv2Config，继承自 PretrainedConfig 类
class Owlv2Config(PretrainedConfig):
    r"""
    [`Owlv2Config`] 是用来存储 [`Owlv2Model`] 的配置的类。它被用来根据指定的参数实例化一个 OWLv2 模型，
    定义了文本模型和视觉模型的配置。使用默认参数实例化一个配置对象将会产生一个类似 OWLv2
    [google/owlv2-base-patch16](https://huggingface.co/google/owlv2-base-patch16) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    Args:
        text_config (`dict`, *optional*):
            用于初始化 [`Owlv2TextConfig`] 的配置选项字典。
        vision_config (`dict`, *optional*):
            用于初始化 [`Owlv2VisionConfig`] 的配置选项字典。
        projection_dim (`int`, *optional*, 默认为 512):
            文本和视觉投影层的维度。
        logit_scale_init_value (`float`, *optional*, 默认为 2.6592):
            *logit_scale* 参数的初始值。默认值按照原始 OWLv2 实现使用。
        return_dict (`bool`, *optional*, 默认为 `True`):
            模型是否应该返回一个字典。如果为 `False`，则返回一个元组。
        kwargs (*optional*):
            关键字参数字典。
    """

    model_type = "owlv2"

    # 初始化方法
    def __init__(
        self,
        text_config=None,
        vision_config=None,
        projection_dim=512,
        logit_scale_init_value=2.6592,
        return_dict=True,
        **kwargs,
    ):
        # 调用父类 PretrainedConfig 的初始化方法
        super().__init__(**kwargs)

        # 如果 text_config 为空，则使用默认配置并记录日志
        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the Owlv2TextConfig with default values.")

        # 如果 vision_config 为空，则使用默认配置并记录日志
        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. initializing the Owlv2VisionConfig with default values.")

        # 使用指定的配置选项初始化 Owlv2TextConfig 和 Owlv2VisionConfig 对象
        self.text_config = Owlv2TextConfig(**text_config)
        self.vision_config = Owlv2VisionConfig(**vision_config)

        # 设置投影层维度、logit_scale 初始值和返回字典的标志
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.return_dict = return_dict
        self.initializer_factor = 1.0

    @classmethod
    # 从预训练模型名称或路径创建一个预训练配置对象
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 设置kwargs中的token
        cls._set_token_in_kwargs(kwargs)

        # 获取预训练模型的配置字典和剩余的kwargs参数
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典中包含"model_type"属性，并且类中定义了"model_type"属性且不同于配置字典中的值，则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从配置字典创建一个配置对象并返回
        return cls.from_dict(config_dict, **kwargs)

    # 从文本模型配置和视觉模型配置创建一个owlv2配置对象
    @classmethod
    def from_text_vision_configs(cls, text_config: Dict, vision_config: Dict, **kwargs):
        r"""
        Instantiate a [`Owlv2Config`] (or a derived class) from owlv2 text model configuration and owlv2 vision
        model configuration.

        Returns:
            [`Owlv2Config`]: An instance of a configuration object
        """
        # 创建一个空的配置字典，并将文本和视觉配置添加到其中
        config_dict = {}
        config_dict["text_config"] = text_config
        config_dict["vision_config"] = vision_config

        # 从配置字典创建一个配置对象并返回
        return cls.from_dict(config_dict, **kwargs)
```