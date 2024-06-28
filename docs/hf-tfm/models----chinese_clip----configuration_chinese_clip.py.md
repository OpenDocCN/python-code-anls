# `.\models\chinese_clip\configuration_chinese_clip.py`

```
# coding=utf-8
# Copyright 2022 The OFA-Sys Team Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Chinese-CLIP model configuration"""

import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union

# 如果 TYPE_CHECKING 为 True，则导入以下模块
if TYPE_CHECKING:
    from ...processing_utils import ProcessorMixin
    from ...utils import TensorType

# 导入 Transformers 库中的预训练配置类和 ONNX 配置类
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging

# 获取模块专用的日志记录器
logger = logging.get_logger(__name__)

# 定义预训练模型名称到配置文件链接的映射字典
CHINESE_CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "OFA-Sys/chinese-clip-vit-base-patch16": (
        "https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16/resolve/main/config.json"
    ),
}

# 定义 ChineseCLIPTextConfig 类，继承自 PretrainedConfig 类
class ChineseCLIPTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ChineseCLIPModel`]. It is used to instantiate a
    Chinese CLIP model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Chinese CLIP
    [OFA-Sys/chinese-clip-vit-base-patch16](https:
        //huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```python
    >>> from transformers import ChineseCLIPTextConfig, ChineseCLIPTextModel

    >>> # Initializing a ChineseCLIPTextConfig with OFA-Sys/chinese-clip-vit-base-patch16 style configuration
    >>> configuration = ChineseCLIPTextConfig()

    >>> # Initializing a ChineseCLIPTextModel (with random weights) from the OFA-Sys/chinese-clip-vit-base-patch16 style configuration
    >>> model = ChineseCLIPTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # 模型类型标识符
    model_type = "chinese_clip_text_model"
    # 初始化函数，设置模型配置参数
    def __init__(
        self,
        vocab_size=30522,  # 词汇表大小，默认为30522
        hidden_size=768,   # 隐藏层大小，默认为768
        num_hidden_layers=12,  # 隐藏层层数，默认为12
        num_attention_heads=12,  # 注意力头数，默认为12
        intermediate_size=3072,  # 中间层大小，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为GELU
        hidden_dropout_prob=0.1,  # 隐藏层dropout概率，默认为0.1
        attention_probs_dropout_prob=0.1,  # 注意力机制的dropout概率，默认为0.1
        max_position_embeddings=512,  # 最大位置嵌入数，默认为512
        type_vocab_size=2,  # 类型词汇表大小，默认为2
        initializer_range=0.02,  # 初始化范围，默认为0.02
        initializer_factor=1.0,  # 初始化因子，默认为1.0
        layer_norm_eps=1e-12,  # 层归一化的epsilon值，默认为1e-12
        pad_token_id=0,  # 填充标记ID，默认为0
        position_embedding_type="absolute",  # 位置嵌入类型，默认为绝对位置嵌入
        use_cache=True,  # 是否使用缓存，默认为True
        **kwargs,  # 其他关键字参数
    ):
        # 调用父类的初始化方法，设置填充标记ID和其他关键字参数
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        # 初始化各个模型配置参数
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

    @classmethod
    # 从预训练模型加载配置参数
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 设置kwargs中的token
        cls._set_token_in_kwargs(kwargs)

        # 获取配置字典和更新后的kwargs
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典的模型类型为"chinese_clip"，则获取文本配置字典
        if config_dict.get("model_type") == "chinese_clip":
            config_dict = config_dict["text_config"]

        # 如果配置字典中包含模型类型，并且类具有model_type属性，并且配置字典的模型类型与类的模型类型不同，发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从配置字典和kwargs创建类的实例
        return cls.from_dict(config_dict, **kwargs)
# 定义一个配置类 ChineseCLIPVisionConfig，继承自 PretrainedConfig 类
class ChineseCLIPVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ChineseCLIPModel`]. It is used to instantiate an
    ChineseCLIP model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the ChineseCLIP
    [OFA-Sys/chinese-clip-vit-base-patch16](https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    
    # 构造函数，初始化 ChineseCLIPModel 的配置参数
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
    >>> from transformers import ChineseCLIPVisionConfig, ChineseCLIPVisionModel

    >>> # Initializing a ChineseCLIPVisionConfig with OFA-Sys/chinese-clip-vit-base-patch16 style configuration
    >>> configuration = ChineseCLIPVisionConfig()
    # 设置模型类型为 "chinese_clip_vision_model"
    model_type = "chinese_clip_vision_model"

    # 定义 ChineseCLIPVisionModel 类，继承自父类
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
        # 调用父类构造函数
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
        # 从预训练模型名称或路径中获取配置信息
        cls._set_token_in_kwargs(kwargs)

        # 获取配置字典和额外的关键字参数
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典的模型类型为 "chinese_clip"，则获取视觉配置字典
        if config_dict.get("model_type") == "chinese_clip":
            config_dict = config_dict["vision_config"]

        # 如果配置字典中包含模型类型，并且模型类型不等于当前类的模型类型，则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从配置字典和关键字参数中创建类的实例
        return cls.from_dict(config_dict, **kwargs)
# 定义一个用于存储 ChineseCLIPModel 配置的类，继承自 PretrainedConfig 类
class ChineseCLIPConfig(PretrainedConfig):
    r"""
    [`ChineseCLIPConfig`] 是用来存储 [`ChineseCLIPModel`] 的配置信息的类。它用于根据指定的参数实例化
    Chinese-CLIP 模型，定义了文本模型和视觉模型的配置。使用默认参数实例化配置将生成与
    Chinese-CLIP [OFA-Sys/chinese-clip-vit-base-patch16](https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16)
    架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档获取更多信息。

    Args:
        text_config (`dict`, *optional*):
            用于初始化 [`ChineseCLIPTextConfig`] 的配置选项字典。
        vision_config (`dict`, *optional*):
            用于初始化 [`ChineseCLIPVisionConfig`] 的配置选项字典。
        projection_dim (`int`, *optional*, 默认为 512):
            文本和视觉投影层的维度。
        logit_scale_init_value (`float`, *optional*, 默认为 2.6592):
            *logit_scale* 参数的初始值。根据原始的 ChineseCLIP 实现使用默认值。
        kwargs (*optional*):
            关键字参数的字典。

    Example:

    ```python
    >>> from transformers import ChineseCLIPConfig, ChineseCLIPModel

    >>> # 使用 OFA-Sys/chinese-clip-vit-base-patch16 风格的配置初始化 ChineseCLIPConfig
    >>> configuration = ChineseCLIPConfig()

    >>> # 使用 OFA-Sys/chinese-clip-vit-base-patch16 风格的配置初始化一个具有随机权重的 ChineseCLIPModel
    >>> model = ChineseCLIPModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config

    >>> # 也可以从 ChineseCLIPTextConfig 和 ChineseCLIPVisionConfig 初始化 ChineseCLIPConfig

    >>> # 初始化 ChineseCLIPTextConfig 和 ChineseCLIPVisionConfig 配置
    >>> config_text = ChineseCLIPTextConfig()
    >>> config_vision = ChineseCLIPVisionConfig()

    >>> config = ChineseCLIPConfig.from_text_vision_configs(config_text, config_vision)
    ```"""
    
    # 类变量，指定模型类型为 "chinese_clip"
    model_type = "chinese_clip"

    # 构造方法，初始化配置
    def __init__(
        self, text_config=None, vision_config=None, projection_dim=512, logit_scale_init_value=2.6592, **kwargs
    ):
        # 父类构造方法，使用传入的参数初始化配置
        super().__init__(**kwargs)

    @classmethod
    def from_text_vision_configs(
        cls, text_config: ChineseCLIPTextConfig, vision_config: ChineseCLIPVisionConfig, **kwargs
    ):
        # 类方法，从文本和视觉配置初始化 ChineseCLIPConfig 实例
        pass
        ):
        r"""
        Instantiate a [`ChineseCLIPConfig`] (or a derived class) from Chinese-CLIP text model configuration and
        Chinese-CLIP vision model configuration. Returns:
            [`ChineseCLIPConfig`]: An instance of a configuration object
        """

        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)


注释：


# 定义一个类方法，用于从给定的中文 CLIP 文本模型配置和视觉模型配置实例化一个 [`ChineseCLIPConfig`] 或其派生类的对象。
# 返回一个 [`ChineseCLIPConfig`] 的实例化对象。
# 定义一个名为 ChineseCLIPOnnxConfig 的类，继承自 OnnxConfig 类
class ChineseCLIPOnnxConfig(OnnxConfig):
    
    # 返回一个有序字典，描述模型的输入规格
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),  # 输入的文本序列
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),  # 输入的图像像素值
                ("attention_mask", {0: "batch", 1: "sequence"}),  # 输入的注意力掩码
            ]
        )
    
    # 返回一个有序字典，描述模型的输出规格
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("logits_per_image", {0: "batch"}),  # 图像输出的逻辑回归
                ("logits_per_text", {0: "batch"}),   # 文本输出的逻辑回归
                ("text_embeds", {0: "batch"}),       # 文本的嵌入表示
                ("image_embeds", {0: "batch"}),      # 图像的嵌入表示
            ]
        )
    
    # 返回用于验证时的绝对误差容限
    @property
    def atol_for_validation(self) -> float:
        return 1e-4
    
    # 生成模型的虚拟输入，包括文本和图像输入
    def generate_dummy_inputs(
        self,
        processor: "ProcessorMixin",
        batch_size: int = -1,
        seq_length: int = -1,
        framework: Optional["TensorType"] = None,
    ) -> Mapping[str, Any]:
        # 调用父类的方法生成文本输入
        text_input_dict = super().generate_dummy_inputs(
            processor.tokenizer, batch_size=batch_size, seq_length=seq_length, framework=framework
        )
        # 调用父类的方法生成图像输入
        image_input_dict = super().generate_dummy_inputs(
            processor.image_processor, batch_size=batch_size, framework=framework
        )
        # 合并文本和图像输入字典并返回
        return {**text_input_dict, **image_input_dict}
    
    # 返回默认的 ONNX 操作集版本
    @property
    def default_onnx_opset(self) -> int:
        return 14
```