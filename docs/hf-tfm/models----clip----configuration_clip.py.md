# `.\models\clip\configuration_clip.py`

```
# 设置文件编码为 UTF-8
# 版权声明和许可信息
# 根据 Apache 许可证 2.0 版本，许可文件的链接
# 如果符合许可证的条件，可以使用该文件，否则禁止使用
""" CLIP 模型配置"""

# 导入标准库和模块
import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 导入处理工具混合类和张量类型
    from ...processing_utils import ProcessorMixin
    from ...utils import TensorType

# 导入配置工具类和 ONNX 配置
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# CLIP 预训练配置映射字典
CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "openai/clip-vit-base-patch32": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/config.json",
    # 查看所有 CLIP 模型：https://huggingface.co/models?filter=clip
}

# CLIPTextConfig 类，继承自 PretrainedConfig
class CLIPTextConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`CLIPTextModel`] 的配置。根据指定的参数实例化 CLIP 文本编码器，定义模型架构。
    使用默认配置实例化将得到类似于 CLIP [openai/clip-vit-base-patch32] 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。
    """
    # 定义模型类型为 CLIP 文本模型
    model_type = "clip_text_model"
    def __init__(
        self,
        vocab_size=49408,
        hidden_size=512,
        intermediate_size=2048,
        projection_dim=512,
        num_hidden_layers=12,
        num_attention_heads=8,
        max_position_embeddings=77,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        # This differs from `CLIPTokenizer`'s default and from openai/clip
        # See https://github.com/huggingface/transformers/pull/24773#issuecomment-1632287538
        pad_token_id=1,
        bos_token_id=49406,
        eos_token_id=49407,
        **kwargs,
    ):
        # 调用父类的初始化方法，设置特殊的标记符号的ID，并传递其他参数
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 设置模型的各种超参数
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
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
        # 调用内部方法，设置token相关的kwargs参数
        cls._set_token_in_kwargs(kwargs)

        # 获取预训练模型的配置字典和更新后的kwargs
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典中的模型类型是"clip"，则获取其文本配置字典
        if config_dict.get("model_type") == "clip":
            config_dict = config_dict["text_config"]

        # 如果配置字典中包含模型类型且与当前类的模型类型不同，发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 根据配置字典和kwargs创建模型配置对象并返回
        return cls.from_dict(config_dict, **kwargs)
# 定义 CLIPVisionConfig 类，继承自 PretrainedConfig，用于存储 CLIPVisionModel 的配置信息
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
    >>> model = CLIPVisionModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config


    model_type = "clip_vision_model"

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
        # 调用父类的构造方法，初始化基类的属性
        super().__init__(**kwargs)

        # 初始化模型的各种参数
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

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        # 获取预训练模型的配置字典和额外的关键字参数
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果从 CLIPConfig 加载，获取视觉配置字典
        if config_dict.get("model_type") == "clip":
            config_dict = config_dict["vision_config"]

        # 如果配置字典中存在模型类型，且与当前类的模型类型不匹配，发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 使用配置字典和额外参数创建类的实例
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
    ```"""

    model_type = "clip"

    def __init__(
        self, text_config=None, vision_config=None, projection_dim=512, logit_scale_init_value=2.6592, **kwargs
    ):
        # 调用父类的初始化方法，初始化基类的配置
        super().__init__(**kwargs)
        # 设定文本配置
        self.text_config = text_config
        # 设定视觉配置
        self.vision_config = vision_config
        # 设定投影维度
        self.projection_dim = projection_dim
        # 设定logit_scale参数的初始值
        self.logit_scale_init_value = logit_scale_init_value

    @classmethod
    def from_text_vision_configs(cls, text_config: CLIPTextConfig, vision_config: CLIPVisionConfig, **kwargs):
        r"""
        Instantiate a [`CLIPConfig`] (or a derived class) from clip text model configuration and clip vision model
        configuration.

        Returns:
            [`CLIPConfig`]: An instance of a configuration object
        """
        # 从文本配置和视觉配置创建一个新的 `CLIPConfig` 实例
        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)


class CLIPOnnxConfig(OnnxConfig):
    @property
    # 定义一个方法 `inputs`，返回一个有序字典，描述了输入数据的结构
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 返回一个有序字典，包含三个键值对，每个键值对描述了不同输入的维度信息
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),  # 表示 input_ids 维度为 [batch, sequence]
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),  # pixel_values 维度为 [batch, num_channels, height, width]
                ("attention_mask", {0: "batch", 1: "sequence"}),  # attention_mask 维度为 [batch, sequence]
            ]
        )

    # 定义一个只读属性 `outputs`，返回一个有序字典，描述了输出数据的结构
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        # 返回一个有序字典，包含四个键值对，每个键值对描述了不同输出的维度信息
        return OrderedDict(
            [
                ("logits_per_image", {0: "batch"}),  # logits_per_image 维度为 [batch]
                ("logits_per_text", {0: "batch"}),   # logits_per_text 维度为 [batch]
                ("text_embeds", {0: "batch"}),       # text_embeds 维度为 [batch]
                ("image_embeds", {0: "batch"}),      # image_embeds 维度为 [batch]
            ]
        )

    # 定义一个方法 `atol_for_validation`，返回浮点数值，表示验证中的绝对容差
    @property
    def atol_for_validation(self) -> float:
        return 1e-4

    # 定义一个方法 `generate_dummy_inputs`，生成虚拟输入数据的字典
    def generate_dummy_inputs(
        self,
        processor: "ProcessorMixin",
        batch_size: int = -1,
        seq_length: int = -1,
        framework: Optional["TensorType"] = None,
    ) -> Mapping[str, Any]:
        # 使用父类的方法生成文本输入的虚拟数据字典
        text_input_dict = super().generate_dummy_inputs(
            processor.tokenizer, batch_size=batch_size, seq_length=seq_length, framework=framework
        )
        # 使用父类的方法生成图像输入的虚拟数据字典
        image_input_dict = super().generate_dummy_inputs(
            processor.image_processor, batch_size=batch_size, framework=framework
        )
        # 返回合并了文本和图像输入数据字典的结果
        return {**text_input_dict, **image_input_dict}

    # 定义一个只读属性 `default_onnx_opset`，返回整数值，表示默认的 ONNX 运算集版本
    @property
    def default_onnx_opset(self) -> int:
        return 14
```