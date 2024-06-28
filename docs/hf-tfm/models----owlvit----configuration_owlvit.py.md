# `.\models\owlvit\configuration_owlvit.py`

```py
# 指定编码方式为 UTF-8

# 版权声明和许可条款，声明代码版权和使用许可
# 根据 Apache License, Version 2.0 许可，使用该文件需要遵守该许可协议

# 导入标准库中的 os 模块
import os
# 导入 collections 模块中的 OrderedDict 类
from collections import OrderedDict
# 导入类型检查相关模块
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Union

# 如果 TYPE_CHECKING 为 True，则导入处理工具的混合器和张量类型
if TYPE_CHECKING:
    from ...processing_utils import ProcessorMixin
    from ...utils import TensorType

# 导入配置工具中的预训练配置类 PretrainedConfig
from ...configuration_utils import PretrainedConfig
# 导入 ONNX 配置类 OnnxConfig
from ...onnx import OnnxConfig
# 导入日志记录工具
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# OWLVIT_PRETRAINED_CONFIG_ARCHIVE_MAP 字典，映射了预训练模型名到配置文件 URL 的对应关系
OWLVIT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/owlvit-base-patch32": "https://huggingface.co/google/owlvit-base-patch32/resolve/main/config.json",
    "google/owlvit-base-patch16": "https://huggingface.co/google/owlvit-base-patch16/resolve/main/config.json",
    "google/owlvit-large-patch14": "https://huggingface.co/google/owlvit-large-patch14/resolve/main/config.json",
}


class OwlViTTextConfig(PretrainedConfig):
    r"""
    这是配置类，用于存储 [`OwlViTTextModel`] 的配置信息。它被用来实例化 OwlViT 文本编码器，根据指定的参数定义模型架构。
    使用默认参数实例化一个配置对象将会产生与 OwlViT [google/owlvit-base-patch32] 架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，可以用来控制模型的输出。详细信息请参阅 [`PretrainedConfig`] 的文档。
    """
    Args:
        vocab_size (`int`, *optional*, defaults to 49408):
            Vocabulary size of the OWL-ViT text model. Defines the number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`OwlViTTextModel`].
        hidden_size (`int`, *optional*, defaults to 512):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        max_position_embeddings (`int`, *optional*, defaults to 16):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
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
        pad_token_id (`int`, *optional*, defaults to 0):
            The id of the padding token in the input sequences.
        bos_token_id (`int`, *optional*, defaults to 49406):
            The id of the beginning-of-sequence token in the input sequences.
        eos_token_id (`int`, *optional*, defaults to 49407):
            The id of the end-of-sequence token in the input sequences.

    Example:

    ```
    >>> from transformers import OwlViTTextConfig, OwlViTTextModel

    >>> # Initializing a OwlViTTextModel with google/owlvit-base-patch32 style configuration
    >>> configuration = OwlViTTextConfig()

    >>> # Initializing a OwlViTTextConfig from the google/owlvit-base-patch32 style configuration
    >>> model = OwlViTTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```



    # 定义模型类型为 "owlvit_text_model"
    model_type = "owlvit_text_model"
    # 初始化函数，用于创建一个新的配置对象实例
    def __init__(
        self,
        vocab_size=49408,                    # 词汇表大小，默认为 49408
        hidden_size=512,                     # 隐藏层大小，默认为 512
        intermediate_size=2048,              # 中间层大小，默认为 2048
        num_hidden_layers=12,                # 隐藏层数，默认为 12
        num_attention_heads=8,               # 注意力头数，默认为 8
        max_position_embeddings=16,          # 最大位置编码数，默认为 16
        hidden_act="quick_gelu",             # 隐藏层激活函数，默认为 "quick_gelu"
        layer_norm_eps=1e-5,                 # LayerNormalization 中的 epsilon，默认为 1e-5
        attention_dropout=0.0,               # 注意力机制中的 dropout 概率，默认为 0.0
        initializer_range=0.02,              # 初始化范围，默认为 0.02
        initializer_factor=1.0,              # 初始化因子，默认为 1.0
        pad_token_id=0,                      # 填充标记的 ID，默认为 0
        bos_token_id=49406,                  # 起始标记的 ID，默认为 49406
        eos_token_id=49407,                  # 结束标记的 ID，默认为 49407
        **kwargs,                            # 其他关键字参数
    ):
        # 调用父类的初始化方法，设置填充、起始和结束标记的 ID，以及其他参数
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 设置对象的属性值
        self.vocab_size = vocab_size                     # 设置词汇表大小
        self.hidden_size = hidden_size                   # 设置隐藏层大小
        self.intermediate_size = intermediate_size       # 设置中间层大小
        self.num_hidden_layers = num_hidden_layers       # 设置隐藏层数
        self.num_attention_heads = num_attention_heads   # 设置注意力头数
        self.max_position_embeddings = max_position_embeddings  # 设置最大位置编码数
        self.hidden_act = hidden_act                     # 设置隐藏层激活函数
        self.layer_norm_eps = layer_norm_eps             # 设置 LayerNormalization 中的 epsilon
        self.attention_dropout = attention_dropout       # 设置注意力机制中的 dropout 概率
        self.initializer_range = initializer_range       # 设置初始化范围
        self.initializer_factor = initializer_factor     # 设置初始化因子

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 设置 token 相关的参数到 kwargs 中
        cls._set_token_in_kwargs(kwargs)

        # 获取预训练模型的配置字典和更新后的 kwargs
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典中的模型类型为 "owlvit"，则从 "text_config" 中获取文本配置字典
        if config_dict.get("model_type") == "owlvit":
            config_dict = config_dict["text_config"]

        # 如果配置字典中包含 "model_type" 并且类本身有 "model_type" 属性，并且两者不同，发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 使用配置字典和 kwargs 创建一个新的对象实例
        return cls.from_dict(config_dict, **kwargs)
# 定义 OwlViTVisionConfig 类，继承自 PretrainedConfig 类
class OwlViTVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`OwlViTVisionModel`]. It is used to instantiate
    an OWL-ViT image encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the OWL-ViT
    [google/owlvit-base-patch32](https://huggingface.co/google/owlvit-base-patch32) architecture.

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
        image_size (`int`, *optional*, defaults to 768):
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

    ```
    >>> from transformers import OwlViTVisionConfig, OwlViTVisionModel

    >>> # Initializing a OwlViTVisionModel with google/owlvit-base-patch32 style configuration
    >>> configuration = OwlViTVisionConfig()

    >>> # Initializing a OwlViTVisionModel model from the google/owlvit-base-patch32 style configuration
    >>> model = OwlViTVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """
    # 定义模型类型字符串常量
    model_type = "owlvit_vision_model"
    
    # 定义模型配置类
    class PretrainedConfig:
    
        # 初始化方法，设置模型的各种参数
        def __init__(
            self,
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_channels=3,
            image_size=768,
            patch_size=32,
            hidden_act="quick_gelu",
            layer_norm_eps=1e-5,
            attention_dropout=0.0,
            initializer_range=0.02,
            initializer_factor=1.0,
            **kwargs,
        ):
            # 调用父类的初始化方法
            super().__init__(**kwargs)
    
            # 设置对象的各个属性
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
    
        # 类方法，从预训练模型加载配置
        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
            # 在 kwargs 中设置 token 相关参数
            cls._set_token_in_kwargs(kwargs)
    
            # 调用 get_config_dict 方法获取配置字典和更新后的 kwargs
            config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
    
            # 如果配置字典中的模型类型是 owlviT，则使用其中的 vision_config
            if config_dict.get("model_type") == "owlvit":
                config_dict = config_dict["vision_config"]
    
            # 如果配置字典中有 model_type 属性，并且与当前类中的 model_type 不一致，输出警告信息
            if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
                logger.warning(
                    f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                    f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
                )
    
            # 调用 from_dict 方法，从配置字典创建对象
            return cls.from_dict(config_dict, **kwargs)
# `OwlViTConfig` 是存储 `OwlViTModel` 配置的类。
# 该类用于实例化 OWL-ViT 模型，根据指定参数定义文本模型和视觉模型配置。
# 通过使用默认参数实例化配置对象将生成类似于 OWL-ViT [google/owlvit-base-patch32] 架构的配置。
class OwlViTConfig(PretrainedConfig):
    r"""
    [`OwlViTConfig`] 是用于存储 [`OwlViTModel`] 配置的类。它用于根据指定的参数实例化 OWL-ViT 模型，
    定义文本模型和视觉模型的配置。使用默认参数实例化配置对象将生成类似于 OWL-ViT
    [google/owlvit-base-patch32](https://huggingface.co/google/owlvit-base-patch32) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    Args:
        text_config (`dict`, *optional*):
            用于初始化 [`OwlViTTextConfig`] 的配置选项字典。
        vision_config (`dict`, *optional*):
            用于初始化 [`OwlViTVisionConfig`] 的配置选项字典。
        projection_dim (`int`, *optional*, defaults to 512):
            文本和视觉投影层的维度。
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            *logit_scale* 参数的初始值。默认值按照原始 OWL-ViT 实现使用。
        return_dict (`bool`, *optional*, defaults to `True`):
            模型是否应返回一个字典。如果为 `False`，则返回一个元组。
        kwargs (*optional*):
            关键字参数字典。
    """

    # 模型类型为 "owlvit"
    model_type = "owlvit"

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        projection_dim=512,
        logit_scale_init_value=2.6592,
        return_dict=True,
        **kwargs,
    ):
        # 调用父类的构造方法
        super().__init__(**kwargs)

        # 如果 text_config 为 None，则使用默认值初始化 OwlViTTextConfig，并记录日志
        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the OwlViTTextConfig with default values.")

        # 如果 vision_config 为 None，则使用默认值初始化 OwlViTVisionConfig，并记录日志
        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. initializing the OwlViTVisionConfig with default values.")

        # 使用 text_config 初始化 self.text_config
        self.text_config = OwlViTTextConfig(**text_config)
        # 使用 vision_config 初始化 self.vision_config
        self.vision_config = OwlViTVisionConfig(**vision_config)

        # 设置 projection_dim 属性为传入的参数 projection_dim
        self.projection_dim = projection_dim
        # 设置 logit_scale_init_value 属性为传入的参数 logit_scale_init_value
        self.logit_scale_init_value = logit_scale_init_value
        # 设置 return_dict 属性为传入的参数 return_dict
        self.return_dict = return_dict
        # 设置 initializer_factor 属性为 1.0
        self.initializer_factor = 1.0

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)  # 调用类方法设置关键字参数中的 token

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        # 获取预训练模型的配置字典和更新后的关键字参数

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            # 如果配置字典中包含 "model_type" 键，并且类中有 "model_type" 属性，并且它们不相等
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )
            # 发出警告，指出正在使用不同类型的模型进行实例化，可能导致错误

        return cls.from_dict(config_dict, **kwargs)
        # 使用配置字典和关键字参数实例化类对象，并返回该对象

    @classmethod
    def from_text_vision_configs(cls, text_config: Dict, vision_config: Dict, **kwargs):
        r"""
        Instantiate a [`OwlViTConfig`] (or a derived class) from owlvit text model configuration and owlvit vision
        model configuration.

        Returns:
            [`OwlViTConfig`]: An instance of a configuration object
        """
        config_dict = {}
        config_dict["text_config"] = text_config  # 将文本模型配置存储到配置字典中
        config_dict["vision_config"] = vision_config  # 将视觉模型配置存储到配置字典中

        return cls.from_dict(config_dict, **kwargs)
        # 使用配置字典和关键字参数实例化类对象，并返回该对象
# 定义一个继承自OnnxConfig的OwlViTOnnxConfig类，用于配置Owl Vision Transformer模型的ONNX导出设置

@property
def inputs(self) -> Mapping[str, Mapping[int, str]]:
    # 定义模型输入的顺序字典，指定每个输入的名称和对应的维度标识
    return OrderedDict(
        [
            ("input_ids", {0: "batch", 1: "sequence"}),  # input_ids输入的维度说明
            ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),  # pixel_values输入的维度说明
            ("attention_mask", {0: "batch", 1: "sequence"}),  # attention_mask输入的维度说明
        ]
    )

@property
def outputs(self) -> Mapping[str, Mapping[int, str]]:
    # 定义模型输出的顺序字典，指定每个输出的名称和对应的维度标识
    return OrderedDict(
        [
            ("logits_per_image", {0: "batch"}),  # logits_per_image输出的维度说明
            ("logits_per_text", {0: "batch"}),  # logits_per_text输出的维度说明
            ("text_embeds", {0: "batch"}),  # text_embeds输出的维度说明
            ("image_embeds", {0: "batch"}),  # image_embeds输出的维度说明
        ]
    )

@property
def atol_for_validation(self) -> float:
    # 定义用于验证的绝对误差容限
    return 1e-4

def generate_dummy_inputs(
    self,
    processor: "ProcessorMixin",
    batch_size: int = -1,
    seq_length: int = -1,
    framework: Optional["TensorType"] = None,
) -> Mapping[str, Any]:
    # 生成用于模型推理的虚拟输入数据，包括文本和图像的虚拟输入
    text_input_dict = super().generate_dummy_inputs(
        processor.tokenizer, batch_size=batch_size, seq_length=seq_length, framework=framework
    )
    image_input_dict = super().generate_dummy_inputs(
        processor.image_processor, batch_size=batch_size, framework=framework
    )
    return {**text_input_dict, **image_input_dict}

@property
def default_onnx_opset(self) -> int:
    # 定义默认的ONNX操作集版本
    return 14
```