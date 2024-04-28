# `.\transformers\models\owlvit\configuration_owlvit.py`

```
# 设置文件编码为 utf-8
# 版权声明
#
# OWL-ViT 模型配置

# 引入所需的库
import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Union

# 如果类型检查开启，则引入相关模块
if TYPE_CHECKING:
    from ...processing_utils import ProcessorMixin
    from ...utils import TensorType

# 引入配置管理工具
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging

# 初始化日志
logger = logging.get_logger(__name__)

# 预训练配置的映射
OWLVIT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/owlvit-base-patch32": "https://huggingface.co/google/owlvit-base-patch32/resolve/main/config.json",
    "google/owlvit-base-patch16": "https://huggingface.co/google/owlvit-base-patch16/resolve/main/config.json",
    "google/owlvit-large-patch14": "https://huggingface.co/google/owlvit-large-patch14/resolve/main/config.json",
}

# OwlViT 文本配置类
class OwlViTTextConfig(PretrainedConfig):
    r"""
    此类存储 [`OwlViTTextModel`] 的配置，并用于实例化 OwlViT 文本编码器，
    根据指定的参数定义模型架构。使用默认配置实例化一个配置将产生与 OwlViT [google/owlvit-base-patch32]
    (https://huggingface.co/google/owlvit-base-patch32) 架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。
    ```
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

    ```python
    >>> from transformers import OwlViTTextConfig, OwlViTTextModel

    >>> # Initializing a OwlViTTextModel with google/owlvit-base-patch32 style configuration
    >>> configuration = OwlViTTextConfig()

    >>> # Initializing a OwlViTTextConfig from the google/owlvit-base-patch32 style configuration
    >>> model = OwlViTTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
"""
    # 设定模型类型为 "owlvit_text_model"
    model_type = "owlvit_text_model"
    # 初始化方法，用于创建一个新的配置对象
    def __init__(
        self,
        vocab_size=49408,  # 词汇表大小，默认为 49408
        hidden_size=512,  # 隐藏层大小，默认为 512
        intermediate_size=2048,  # 中间层大小，默认为 2048
        num_hidden_layers=12,  # 隐藏层数，默认为 12
        num_attention_heads=8,  # 注意力头数，默认为 8
        max_position_embeddings=16,  # 最大位置嵌入数，默认为 16
        hidden_act="quick_gelu",  # 隐藏层激活函数，默认为 "quick_gelu"
        layer_norm_eps=1e-5,  # 层归一化 epsilon，默认为 1e-5
        attention_dropout=0.0,  # 注意力 dropout，默认为 0.0
        initializer_range=0.02,  # 初始化范围，默认为 0.02
        initializer_factor=1.0,  # 初始化因子，默认为 1.0
        pad_token_id=0,  # 填充符 token id，默认为 0
        bos_token_id=49406,  # 开始符 token id，默认为 49406
        eos_token_id=49407,  # 结束符 token id，默认为 49407
        **kwargs,  # 其他关键字参数
    ):
        # 调用父类的初始化方法，传入填充符、开始符、结束符 token id 以及其他关键字参数
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 设置对象的属性
        self.vocab_size = vocab_size  # 词汇表大小
        self.hidden_size = hidden_size  # 隐藏层大小
        self.intermediate_size = intermediate_size  # 中间层大小
        self.num_hidden_layers = num_hidden_layers  # 隐藏层数
        self.num_attention_heads = num_attention_heads  # 注意力头数
        self.max_position_embeddings = max_position_embeddings  # 最大位置嵌入数
        self.hidden_act = hidden_act  # 隐藏层激活函数
        self.layer_norm_eps = layer_norm_eps  # 层归一化 epsilon
        self.attention_dropout = attention_dropout  # 注意力 dropout
        self.initializer_range = initializer_range  # 初始化范围
        self.initializer_factor = initializer_factor  # 初始化因子

    # 从预训练模型中加载配置对象的类方法
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 在 kwargs 中设置 token 参数
        cls._set_token_in_kwargs(kwargs)

        # 获取配置字典和 kwargs
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果加载自 OwlViTConfig，则获取文本配置字典
        if config_dict.get("model_type") == "owlvit":
            config_dict = config_dict["text_config"]

        # 如果配置字典中包含模型类型，并且该模型类型与当前类的模型类型不匹配，则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从配置字典中创建配置对象并返回
        return cls.from_dict(config_dict, **kwargs)
# 定义 OWL-ViT 图像编码器的配置类
class OwlViTVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`OwlViTVisionModel`]. It is used to instantiate
    an OWL-ViT image encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the OWL-ViT
    [google/owlvit-base-patch32](https://huggingface.co/google/owlvit-base-patch32) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        # 编码器层和汇集层的维度
        hidden_size (`int`, *optional*, defaults to 768):
        # Transformer 编码器中的"中间"（即前馈）层的维度
        intermediate_size (`int`, *optional*, defaults to 3072):
        # Transformer 编码器中的隐藏层数量
        num_hidden_layers (`int`, *optional*, defaults to 12):
        # Transformer 编码器中每个注意力层的注意力头数量
        num_attention_heads (`int`, *optional*, defaults to 12):
        # 输入图像中的通道数
        num_channels (`int`, *optional*, defaults to 3):
        # 每个图像的大小（分辨率）
        image_size (`int`, *optional*, defaults to 768):
        # 每个补丁的大小（分辨率）
        patch_size (`int`, *optional*, defaults to 32):
        # 编码器和汇集层中的非线性激活函数
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
        # 层归一化层使用的 epsilon 值
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
        # 注意力概率的 dropout 比率
        attention_dropout (`float`, *optional*, defaults to 0.0):
        # 用于初始化所有权重矩阵的截断正态分布的标准差
        initializer_range (`float`, *optional*, defaults to 0.02):
        # 用于初始化所有权重矩阵的因子（应保持为1，内部用于初始化测试）
        initializer_factor (`float`, *optional*, defaults to 1.0):

    Example:

    ```python
    >>> from transformers import OwlViTVisionConfig, OwlViTVisionModel

    >>> # 使用 google/owlvit-base-patch32 风格配置初始化 OwlViTVisionModel
    >>> configuration = OwlViTVisionConfig()

    >>> # 使用 google/owlvit-base-patch32 风格配置初始化 OwlViTVisionModel 模型
    >>> model = OwlViTVisionModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```
    # 定义模型类型为 'owlvit_vision_model'
    model_type = "owlvit_vision_model"
    
    # 定义模型的各种参数
    def __init__(
        self,
        # 隐藏层大小
        hidden_size=768,
        # 中间层大小
        intermediate_size=3072,
        # 隐藏层层数
        num_hidden_layers=12,
        # 注意力头的数量
        num_attention_heads=12,
        # 输入图像通道数
        num_channels=3,
        # 输入图像尺寸
        image_size=768,
        # 图像分块大小
        patch_size=32,
        # 激活函数
        hidden_act="quick_gelu",
        # 层归一化 epsilon 值
        layer_norm_eps=1e-5,
        # 注意力dropout
        attention_dropout=0.0,
        # 初始化范围
        initializer_range=0.02,
        # 初始化因子
        initializer_factor=1.0,
        **kwargs,
    ):
        # 调用父类构造函数
        super().__init__(**kwargs)
        # 初始化各种参数
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
    
    # 从预训练模型加载配置
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 设置 token 参数
        cls._set_token_in_kwargs(kwargs)
        # 获取配置字典和其他参数
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        # 如果是 OwlViTConfig 类型, 取 vision_config 部分
        if config_dict.get("model_type") == "owlvit":
            config_dict = config_dict["vision_config"]
        # 如果模型类型不匹配, 输出警告信息
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )
        # 根据字典创建配置实例
        return cls.from_dict(config_dict, **kwargs)
class OwlViTConfig(PretrainedConfig):
    r"""
    [`OwlViTConfig`] is the configuration class to store the configuration of an [`OwlViTModel`]. It is used to
    instantiate an OWL-ViT model according to the specified arguments, defining the text model and vision model
    configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the OWL-ViT
    [google/owlvit-base-patch32](https://huggingface.co/google/owlvit-base-patch32) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`OwlViTTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`OwlViTVisionConfig`].
        projection_dim (`int`, *optional*, defaults to 512):
            Dimensionality of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The initial value of the *logit_scale* parameter. Default is used as per the original OWL-ViT
            implementation.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return a dictionary. If `False`, returns a tuple.
        kwargs (*optional*):
            Dictionary of keyword arguments.
    """

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
        # Call the constructor of the parent class to initialize inherited attributes
        super().__init__(**kwargs)

        # If text_config is not provided, initialize it with an empty dictionary
        if text_config is None:
            text_config = {}
            # Log that text_config is None and initialize OwlViTTextConfig with default values
            logger.info("text_config is None. Initializing the OwlViTTextConfig with default values.")

        # If vision_config is not provided, initialize it with an empty dictionary
        if vision_config is None:
            vision_config = {}
            # Log that vision_config is None and initialize OwlViTVisionConfig with default values
            logger.info("vision_config is None. initializing the OwlViTVisionConfig with default values.")

        # Initialize text_config and vision_config with OwlViTTextConfig and OwlViTVisionConfig instances
        self.text_config = OwlViTTextConfig(**text_config)
        self.vision_config = OwlViTVisionConfig(**vision_config)

        # Set projection_dim, logit_scale_init_value, return_dict, and initializer_factor attributes
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.return_dict = return_dict
        self.initializer_factor = 1.0

    @classmethod
    # 从预训练模型名或路径创建配置对象的类方法
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 在 kwargs 中设置 token
        cls._set_token_in_kwargs(kwargs)

        # 获取预训练模型的配置字典及其他参数
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典中包含“model_type”字段并且类具有“model_type”属性，并且它们不相等，则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 根据配置字典创建配置对象并返回
        return cls.from_dict(config_dict, **kwargs)

    # 从文本模型配置和视觉模型配置实例化 OwlViTConfig（或其派生类）的类方法
    @classmethod
    def from_text_vision_configs(cls, text_config: Dict, vision_config: Dict, **kwargs):
        r"""
        从 owlvit 文本模型配置和 owlvit 视觉模型配置实例化 [`OwlViTConfig`]（或其派生类）。

        返回：
            [`OwlViTConfig`]：配置对象的实例
        """
        # 创建一个空的配置字典，并将文本和视觉模型配置添加到其中
        config_dict = {}
        config_dict["text_config"] = text_config
        config_dict["vision_config"] = vision_config

        # 根据配置字典创建配置对象并返回
        return cls.from_dict(config_dict, **kwargs)
# OwlViTOnnxConfig 类是 OnnxConfig 的子类
class OwlViTOnnxConfig(OnnxConfig):
    # inputs 属性返回一个有序字典，描述模型输入的形状
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                # "input_ids" 输入的形状是 (batch, sequence)
                ("input_ids", {0: "batch", 1: "sequence"}),
                # "pixel_values" 输入的形状是 (batch, num_channels, height, width)
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
                # "attention_mask" 输入的形状是 (batch, sequence)
                ("attention_mask", {0: "batch", 1: "sequence"}),
            ]
        )

    # outputs 属性返回一个有序字典，描述模型输出的形状
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                # "logits_per_image" 输出的形状是 (batch,)
                ("logits_per_image", {0: "batch"}),
                # "logits_per_text" 输出的形状是 (batch,)
                ("logits_per_text", {0: "batch"}),
                # "text_embeds" 输出的形状是 (batch,)
                ("text_embeds", {0: "batch"}),
                # "image_embeds" 输出的形状是 (batch,)
                ("image_embeds", {0: "batch"}),
            ]
        )

    # atol_for_validation 属性返回 ONNX 模型验证时允许的绝对误差
    @property
    def atol_for_validation(self) -> float:
        return 1e-4

    # generate_dummy_inputs 方法用于生成模型的虚拟输入数据
    def generate_dummy_inputs(
        self,
        processor: "ProcessorMixin",
        batch_size: int = -1,
        seq_length: int = -1,
        framework: Optional["TensorType"] = None,
    ) -> Mapping[str, Any]:
        # 生成文本输入数据
        text_input_dict = super().generate_dummy_inputs(
            processor.tokenizer, batch_size=batch_size, seq_length=seq_length, framework=framework
        )
        # 生成图像输入数据
        image_input_dict = super().generate_dummy_inputs(
            processor.image_processor, batch_size=batch_size, framework=framework
        )
        # 合并文本输入和图像输入数据
        return {**text_input_dict, **image_input_dict}

    # default_onnx_opset 属性返回默认的 ONNX 算子集版本
    @property
    def default_onnx_opset(self) -> int:
        return 14
```