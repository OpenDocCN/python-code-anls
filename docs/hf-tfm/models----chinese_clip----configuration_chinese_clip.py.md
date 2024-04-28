# `.\transformers\models\chinese_clip\configuration_chinese_clip.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 版权所有（c）2022年OFA-Sys团队作者和HuggingFace团队。保留所有权利。
#
# 根据Apache许可证2.0版（“许可证”）许可;
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件是基于“按原样”提供的，
# 没有任何明示或暗示的担保或条件。
# 请查看许可证了解特定的语言管辖权下的权限和限制。
""" Chinese-CLIP 模型配置"""

# 导入必要的库
import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union

# 检查类型
if TYPE_CHECKING:
    from ...processing_utils import ProcessorMixin
    from ...utils import TensorType

# 导入所需的配置类和日志记录工具
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练模型配置文件映射
CHINESE_CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "OFA-Sys/chinese-clip-vit-base-patch16": (
        "https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16/resolve/main/config.json"
    ),
}

# 定义 ChineseCLIPTextConfig 类，用于存储 Chinese CLIP 模型的配置信息
class ChineseCLIPTextConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`ChineseCLIPModel`] 的配置。根据指定的参数，它用于实例化一个 Chinese CLIP 模型，定义模型的架构。
    使用默认值实例化一个配置将产生与 Chinese CLIP [OFA-Sys/chinese-clip-vit-base-patch16](https:
        //huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16) 架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。


    示例:

    ```python
    >>> from transformers import ChineseCLIPTextConfig, ChineseCLIPTextModel

    >>> # 使用 OFA-Sys/chinese-clip-vit-base-patch16 风格的配置初始化 ChineseCLIPTextConfig
    >>> configuration = ChineseCLIPTextConfig()

    >>> # 使用 OFA-Sys/chinese-clip-vit-base-patch16 风格的配置初始化 ChineseCLIPTextModel（带有随机权重）
    >>> model = ChineseCLIPTextModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```"""

    model_type = "chinese_clip_text_model"
    # 初始化方法，用于创建一个新的配置对象
    def __init__(
        self,
        vocab_size=30522,  # 词汇表大小，默认为30522
        hidden_size=768,   # 隐藏层大小，默认为768
        num_hidden_layers=12,   # 隐藏层层数，默认为12
        num_attention_heads=12, # 注意力头数，默认为12
        intermediate_size=3072, # 中间层大小，默认为3072
        hidden_act="gelu",      # 隐藏层激活函数，默认为gelu
        hidden_dropout_prob=0.1,    # 隐藏层dropout概率，默认为0.1
        attention_probs_dropout_prob=0.1,  # 注意力矩阵dropout概率，默认为0.1
        max_position_embeddings=512,    # 最大位置嵌入，默认为512
        type_vocab_size=2,  # 类型词汇表大小，默认为2
        initializer_range=0.02,    # 初始化范围，默认为0.02
        initializer_factor=1.0,    # 初始化因子，默认为1.0
        layer_norm_eps=1e-12,   # 层归一化epsilon值，默认为1e-12
        pad_token_id=0, # 填充标记ID，默认为0
        position_embedding_type="absolute", # 位置嵌入类型，默认为"absolute"
        use_cache=True, # 是否使用缓存，默认为True
        **kwargs,
    ):
        # 调用父类初始化方法
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        # 设置配置参数
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache

    # 从预训练模型中加载配置
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 在kwargs中设置token
        cls._set_token_in_kwargs(kwargs)

        # 获取配置字典和剩余的kwargs
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典中的模型类型为"chinese_clip"，则获取视觉配置字典
        if config_dict.get("model_type") == "chinese_clip":
            config_dict = config_dict["text_config"]

        # 如果配置字典中有模型类型，并且该类型与当前类的模型类型不同，则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从配置字典创建配置对象，并返回
        return cls.from_dict(config_dict, **kwargs)
# 定义一个用于存储 ChineseCLIPModel 配置的类
class ChineseCLIPVisionConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`ChineseCLIPModel`] 的配置。它用于根据指定的参数实例化一个 ChineseCLIP 模型，定义模型架构。
    使用默认参数实例化一个配置将会产生类似于 ChineseCLIP [OFA-Sys/chinese-clip-vit-base-patch16](https:
        //huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16) 架构的配置。

    配置对象继承自 [`PretrainedConfig`] 并可用于控制模型的输出。阅读 [`PretrainedConfig`] 的文档获取更多信息。


    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            编码器层和池化器层的维度。
        intermediate_size (`int`, *optional*, defaults to 3072):
            Transformer 编码器中“中间”（即前馈）层的维度。
        projection_dim (`int`, *optional*, defaults to 512):
            文本和视觉投影层的维度。
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Transformer 编码器中的隐藏层数。
        num_attention_heads (`int`, *optional*, defaults to 12):
            Transformer 编码器中每个注意力层的注意力头数。
        num_channels (`int`, *optional*, defaults to 3):
            输入通道数。
        image_size (`int`, *optional*, defaults to 224):
            每个图像的大小（分辨率）。
        patch_size (`int`, *optional*, defaults to 32):
            每个补丁的大小（分辨率）。
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            编码器和池化器中的非线性激活函数（函数或字符串）。如果是字符串，支持 `"gelu"`, `"relu"`, `"selu"` 和 `"gelu_new"` `"quick_gelu"`。
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            层归一化层使用的 epsilon。
        attention_dropout (`float`, *optional*, defaults to 0.0):
            注意力概率的丢弃率。
        initializer_range (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态初始化器的标准差。
        initializer_factor (`float`, *optional*, defaults to 1.0):
            初始化所有权重矩阵的因子（应保持为 1，内部用于初始化测试）。
    Example:
    ```python
    >>> from transformers import ChineseCLIPVisionConfig, ChineseCLIPVisionModel

    >>> # 使用 OFA-Sys/chinese-clip-vit-base-patch16 风格配置初始化一个 ChineseCLIPVisionConfig
    >>> configuration = ChineseCLIPVisionConfig()
    # 使用 OFA-Sys/chinese-clip-vit-base-patch16 风格的配置初始化一个 ChineseCLIPVisionModel，权重随机初始化
    model = ChineseCLIPVisionModel(configuration)
    
    # 访问模型的配置信息
    configuration = model.config
# 定义一个配置类，用于存储 ChineseCLIPModel 的配置信息
class ChineseCLIPConfig(PretrainedConfig):
    r"""
    [`ChineseCLIPConfig`] 是用于存储 [`ChineseCLIPModel`] 的配置信息的配置类。它用于根据指定的参数实例化 Chinese-CLIP 模型，
    定义了文本模型和视觉模型的配置。使用默认值实例化一个配置将产生类似于
    Chinese-CLIP [OFA-Sys/chinese-clip-vit-base-patch16](https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16)
    架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。更多信息请阅读 [`PretrainedConfig`] 的文档。

    Args:
        text_config (`dict`, *optional*):
            用于初始化 [`ChineseCLIPTextConfig`] 的配置选项字典。
        vision_config (`dict`, *optional*):
            用于初始化 [`ChineseCLIPVisionConfig`] 的配置选项字典。
        projection_dim (`int`, *optional*, 默认为 512):
            文本和视觉投影层的维度。
        logit_scale_init_value (`float`, *optional*, 默认为 2.6592):
            *logit_scale* 参数的初始值。默认值根据原始 ChineseCLIP 实现使用。
        kwargs (*optional*):
            关键字参数的字典。

    Example:

    ```python
    >>> from transformers import ChineseCLIPConfig, ChineseCLIPModel

    >>> # 使用 OFA-Sys/chinese-clip-vit-base-patch16 风格配置初始化 ChineseCLIPConfig
    >>> configuration = ChineseCLIPConfig()

    >>> # 使用 OFA-Sys/chinese-clip-vit-base-patch16 风格配置初始化 ChineseCLIPModel（带有随机权重）
    >>> model = ChineseCLIPModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config

    >>> # 我们也可以从 ChineseCLIPTextConfig 和 ChineseCLIPVisionConfig 初始化 ChineseCLIPConfig

    >>> # 初始化 ChineseCLIPTextConfig 和 ChineseCLIPVisionConfig 配置
    >>> config_text = ChineseCLIPTextConfig()
    >>> config_vision = ChineseCLIPVisionConfig()

    >>> config = ChineseCLIPConfig.from_text_vision_configs(config_text, config_vision)
    ```"""
    # 模型类型为 "chinese_clip"
    model_type = "chinese_clip"

    # 初始化方法
    def __init__(
        self, text_config=None, vision_config=None, projection_dim=512, logit_scale_init_value=2.6592, **kwargs
    # 从文本和视觉配置中创建 ChineseCLIPConfig 的类方法
    @classmethod
    def from_text_vision_configs(
        cls, text_config: ChineseCLIPTextConfig, vision_config: ChineseCLIPVisionConfig, **kwargs
        ):
        r"""
        实例化一个[`ChineseCLIPConfig`]（或其派生类）从Chinese-CLIP文本模型配置和Chinese-CLIP视觉模型配置。 返回：
            [`ChineseCLIPConfig`]：配置对象的一个实例
        """

        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)
# 定义一个中文 CLIP 的 ONNX 配置类，继承自 OnnxConfig 类
class ChineseCLIPOnnxConfig(OnnxConfig):
    # 返回输入的映射关系，包含输入名称和维度的映射
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
                ("attention_mask", {0: "batch", 1: "sequence"}),
            ]
        )

    # 返回输出的映射关系，包含输出名称和维度的映射
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("logits_per_image", {0: "batch"}),
                ("logits_per_text", {0: "batch"}),
                ("text_embeds", {0: "batch"}),
                ("image_embeds", {0: "batch"}),
            ]
        )

    # 返回用于验证的绝对误差容限
    @property
    def atol_for_validation(self) -> float:
        return 1e-4

    # 生成虚拟输入数据，包括文本和图像输入
    def generate_dummy_inputs(
        self,
        processor: "ProcessorMixin",
        batch_size: int = -1,
        seq_length: int = -1,
        framework: Optional["TensorType"] = None,
    ) -> Mapping[str, Any]:
        text_input_dict = super().generate_dummy_inputs(
            processor.tokenizer, batch_size=batch_size, seq_length=seq_length, framework=framework
        )
        image_input_dict = super().generate_dummy_inputs(
            processor.image_processor, batch_size=batch_size, framework=framework
        )
        return {**text_input_dict, **image_input_dict}

    # 返回默认的 ONNX 操作集版本
    @property
    def default_onnx_opset(self) -> int:
        return 14
```