# `.\transformers\models\clip\configuration_clip.py`

```py
# 设置文件编码为 UTF-8
# 版权声明及许可证信息
# 导入所需模块和类
import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union

# 如果当前是类型检查模式，则导入类型检查所需的模块
if TYPE_CHECKING:
    from ...processing_utils import ProcessorMixin
    from ...utils import TensorType

# 导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 导入 ONNX 配置类
from ...onnx import OnnxConfig
# 导入日志记录工具
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# CLIP 预训练模型配置文件的映射字典
CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "openai/clip-vit-base-patch32": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/config.json",
    # 查看所有 CLIP 模型：https://huggingface.co/models?filter=clip
}

# CLIP 文本编码模型配置类，继承自预训练配置类
class CLIPTextConfig(PretrainedConfig):
    r"""
    这是用于存储 [`CLIPTextModel`] 配置的配置类。根据指定的参数实例化一个 CLIP 文本编码器，定义模型架构。
    使用默认参数实例化一个配置将产生类似于 CLIP [openai/clip-vit-base-patch32]
    (https://huggingface.co/openai/clip-vit-base-patch32) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。有关更多信息，请阅读 [`PretrainedConfig`] 的文档。
    """
    Args:
        vocab_size (`int`, *optional*, defaults to 49408):
            Vocabulary size of the CLIP text model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`CLIPModel`].
        hidden_size (`int`, *optional*, defaults to 512):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and vision projection layers.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        max_position_embeddings (`int`, *optional*, defaults to 77):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        pad_token_id (`int`, *optional*, defaults to 1):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 49406):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 49407):
            End of stream token id.

    Example:

    ```python
    >>> from transformers import CLIPTextConfig, CLIPTextModel

    >>> # Initializing a CLIPTextConfig with openai/clip-vit-base-patch32 style configuration
    >>> configuration = CLIPTextConfig()

    >>> # Initializing a CLIPTextModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
    >>> model = CLIPTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```py

    # 定义模型类型为 "clip_text_model"
    model_type = "clip_text_model"
    # 初始化函数，用于创建一个新的配置对象
    def __init__(
        self,
        vocab_size=49408,  # 词汇表大小，默认为49408
        hidden_size=512,  # 隐藏层大小，默认为512
        intermediate_size=2048,  # 中间层大小，默认为2048
        projection_dim=512,  # 投影维度，默认为512
        num_hidden_layers=12,  # 隐藏层数，默认为12
        num_attention_heads=8,  # 注意力头数，默认为8
        max_position_embeddings=77,  # 最大位置嵌入，默认为77
        hidden_act="quick_gelu",  # 隐藏层激活函数，默认为"quick_gelu"
        layer_norm_eps=1e-5,  # 层归一化的 epsilon，默认为1e-5
        attention_dropout=0.0,  # 注意力机制的 dropout，默认为0.0
        initializer_range=0.02,  # 初始化范围，默认为0.02
        initializer_factor=1.0,  # 初始化因子，默认为1.0
        # 这与`CLIPTokenizer`的默认值不同，也不同于openai/clip
        # 参见https://github.com/huggingface/transformers/pull/24773#issuecomment-1632287538
        pad_token_id=1,  # 填充标记的ID，默认为1
        bos_token_id=49406,  # 起始标记的ID，默认为49406
        eos_token_id=49407,  # 终止标记的ID，默认为49407
        **kwargs,  # 其他关键字参数
    ):
        # 调用父类的初始化函数，设置填充、起始和终止标记的ID
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 设置配置对象的属性值
        self.vocab_size = vocab_size  # 词汇表大小
        self.hidden_size = hidden_size  # 隐藏层大小
        self.intermediate_size = intermediate_size  # 中间层大小
        self.projection_dim = projection_dim  # 投影维度
        self.num_hidden_layers = num_hidden_layers  # 隐藏层数
        self.num_attention_heads = num_attention_heads  # 注意力头数
        self.max_position_embeddings = max_position_embeddings  # 最大位置嵌入
        self.layer_norm_eps = layer_norm_eps  # 层归一化的 epsilon
        self.hidden_act = hidden_act  # 隐藏层激活函数
        self.initializer_range = initializer_range  # 初始化范围
        self.initializer_factor = initializer_factor  # 初始化因子
        self.attention_dropout = attention_dropout  # 注意力机制的 dropout

    # 类方法，用于从预训练模型加载配置
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 将token信息设置到kwargs中
        cls._set_token_in_kwargs(kwargs)

        # 获取配置字典和其他kwargs
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果加载的模型类型为clip，则获取文本配置字典
        if config_dict.get("model_type") == "clip":
            config_dict = config_dict["text_config"]

        # 如果配置字典中包含模型类型，并且当前类的模型类型与其不匹配，则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从配置字典和其他kwargs中创建配置对象
        return cls.from_dict(config_dict, **kwargs)
# 定义 CLIPVisionConfig 类，用于存储 CLIPVisionModel 的配置信息
class CLIPVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CLIPVisionModel`]. It is used to instantiate a
    CLIP vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the CLIP
    [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and vision projection layers.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`int`, *optional*, defaults to 224):
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

    ```python
    >>> from transformers import CLIPVisionConfig, CLIPVisionModel

    >>> # Initializing a CLIPVisionConfig with openai/clip-vit-base-patch32 style configuration
    >>> configuration = CLIPVisionConfig()

    >>> # Initializing a CLIPVisionModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
    # 创建一个 CLIP 视觉模型实例，使用给定的配置
    model = CLIPVisionModel(configuration)

    # 访问模型配置信息
    # 将模型配置信息赋值给 configuration
    configuration = model.config
    ```py

    # 定义 CLIP 视觉模型类型
    model_type = "clip_vision_model"

    # 初始化 CLIP 视觉模型类
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        projection_dim=512,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
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

        # 设置模型的各种参数
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act

    # 从预训练模型加载 CLIP 视觉模型的配置
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 设置参数中的 token
        cls._set_token_in_kwargs(kwargs)

        # 获取预训练模型的配置字典和参数字典
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典中的模型类型是 "clip"，则取其视觉配置字典
        if config_dict.get("model_type") == "clip":
            config_dict = config_dict["vision_config"]

        # 如果配置字典中包含模型类型，并且当前类的模型类型与之不同，发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 根据配置字典创建实例
        return cls.from_dict(config_dict, **kwargs)
class CLIPConfig(PretrainedConfig):
    r"""
    [`CLIPConfig`] is the configuration class to store the configuration of a [`CLIPModel`]. It is used to instantiate
    a CLIP model according to the specified arguments, defining the text model and vision model configs. Instantiating
    a configuration with the defaults will yield a similar configuration to that of the CLIP
    [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`CLIPTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`CLIPVisionConfig`].
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* paramter. Default is used as per the original CLIP implementation.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import CLIPConfig, CLIPModel

    >>> # Initializing a CLIPConfig with openai/clip-vit-base-patch32 style configuration
    >>> configuration = CLIPConfig()

    >>> # Initializing a CLIPModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
    >>> model = CLIPModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a CLIPConfig from a CLIPTextConfig and a CLIPVisionConfig
    >>> from transformers import CLIPTextConfig, CLIPVisionConfig

    >>> # Initializing a CLIPText and CLIPVision configuration
    >>> config_text = CLIPTextConfig()
    >>> config_vision = CLIPVisionConfig()

    >>> config = CLIPConfig.from_text_vision_configs(config_text, config_vision)
    ```py"""

    model_type = "clip"

    def __init__(
        self, text_config=None, vision_config=None, projection_dim=512, logit_scale_init_value=2.6592, **kwargs
    ):
        # 继承父类构造方法，初始化CLIPConfig对象
        super().__init__(**kwargs)
        # 设定模型类型为"clip"
        self.model_type = "clip"
        # 设定文本配置，默认为空字典
        self.text_config = text_config if text_config is not None else {}
        # 设定视觉配置，默认为空字典
        self.vision_config = vision_config if vision_config is not None else {}
        # 设定投影维度，默认为512
        self.projection_dim = projection_dim
        # 设定logit_scale参数的初始值，默认为2.6592
        self.logit_scale_init_value = logit_scale_init_value

    @classmethod
    def from_text_vision_configs(cls, text_config: CLIPTextConfig, vision_config: CLIPVisionConfig, **kwargs):
        r"""
        Instantiate a [`CLIPConfig`] (or a derived class) from clip text model configuration and clip vision model
        configuration.

        Returns:
            [`CLIPConfig`]: An instance of a configuration object
        """

        # 从文本配置和视觉配置实例化一个CLIPConfig对象
        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)


class CLIPOnnxConfig(OnnxConfig):
    @property
    # 定义一个方法，返回输入的规格
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 返回一个有序字典，描述输入的数据结构，包含 input_ids、pixel_values、attention_mask 三个键
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),  # input_ids: batch, sequence
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),  # pixel_values: batch, num_channels, height, width
                ("attention_mask", {0: "batch", 1: "sequence"}),  # attention_mask: batch, sequence
            ]
        )

    # 定义一个只读属性，返回输出的规格
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        # 返回一个有序字典，描述输出的数据结构，包含 logits_per_image、logits_per_text、text_embeds、image_embeds 四个键
        return OrderedDict(
            [
                ("logits_per_image", {0: "batch"}),  # logits_per_image: batch
                ("logits_per_text", {0: "batch"}),  # logits_per_text: batch
                ("text_embeds", {0: "batch"}),  # text_embeds: batch
                ("image_embeds", {0: "batch"}),  # image_embeds: batch
            ]
        )

    # 定义一个属性，返回用于验证时的绝对误差容差
    @property
    def atol_for_validation(self) -> float:
        # 返回一个浮点数，表示绝对误差容差
        return 1e-4

    # 定义一个方法，生成虚拟输入数据
    def generate_dummy_inputs(
        self,
        processor: "ProcessorMixin",
        batch_size: int = -1,
        seq_length: int = -1,
        framework: Optional["TensorType"] = None,
    ) -> Mapping[str, Any]:
        # 生成文本输入的虚拟数据
        text_input_dict = super().generate_dummy_inputs(
            processor.tokenizer, batch_size=batch_size, seq_length=seq_length, framework=framework
        )
        # 生成图像输入的虚拟数据
        image_input_dict = super().generate_dummy_inputs(
            processor.image_processor, batch_size=batch_size, framework=framework
        )
        # 合并文本和图像输入的虚拟数据，返回一个字典
        return {**text_input_dict, **image_input_dict}

    # 定义一个只读属性，返回默认的 ONNX 操作集版本号
    @property
    def default_onnx_opset(self) -> int:
        # 返回一个整数，表示默认的 ONNX 操作集版本号
        return 14
```  
```