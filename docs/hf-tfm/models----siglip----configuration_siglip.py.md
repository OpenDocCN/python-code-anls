# `.\models\siglip\configuration_siglip.py`

```
# coding=utf-8
# 上方声明文件编码为 UTF-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# 版权声明，版权归 HuggingFace Inc. 团队所有

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
# 根据 Apache License, Version 2.0 进行许可，详细信息可查看指定的链接

""" Siglip model configuration"""
# 文档字符串，指定此文件是关于 Siglip 模型配置的

import os
from typing import Union

# 从其他模块导入必要的类和函数
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取模块的日志记录器
logger = logging.get_logger(__name__)

# 定义预训练配置文件映射，将模型名称映射到预训练配置文件的 URL
SIGLIP_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/siglip-base-patch16-224": "https://huggingface.co/google/siglip-base-patch16-224/resolve/main/config.json",
}


class SiglipTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SiglipTextModel`]. It is used to instantiate a
    Siglip text encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the text encoder of the Siglip
    [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # SiglipTextConfig 类继承自 PretrainedConfig，用于存储 SiglipTextModel 的配置信息
    # 根据指定的参数实例化 Siglip 文本编码器，定义模型架构
    # 使用默认配置实例化将产生与 Siglip google/siglip-base-patch16-224 模型架构相似的配置

    # 详细文档字符串描述了此类的作用和用法，以及与 PretrainedConfig 类的关系，用于控制模型输出
    # 可以查阅 PretrainedConfig 的文档获取更多信息
    # 定义模型类型为 Siglip 文本模型
    model_type = "siglip_text_model"
    # 初始化方法，用于创建一个新的配置对象
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=64,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        # 这个参数与 `CLIPTokenizer` 的默认值以及 openai/siglip 的默认值不同
        # 参考 https://github.com/huggingface/transformers/pull/24773#issuecomment-1632287538
        pad_token_id=1,
        bos_token_id=49406,
        eos_token_id=49407,
        **kwargs,
    ):
        # 调用父类的初始化方法，设置特殊标记的 ID 和其余的关键字参数
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 设置配置对象的各种属性
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout

    @classmethod
    # 从预训练模型名称或路径创建一个预训练配置对象的类方法
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 在关键字参数中设置 token 相关的参数
        cls._set_token_in_kwargs(kwargs)

        # 调用 get_config_dict 方法获取配置字典和更新后的关键字参数
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典的 model_type 为 "siglip"，则使用其 text_config 配置
        if config_dict.get("model_type") == "siglip":
            config_dict = config_dict["text_config"]

        # 如果配置字典中有 model_type，且与当前类的 model_type 属性不匹配，输出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从配置字典和关键字参数创建一个新的类对象
        return cls.from_dict(config_dict, **kwargs)
class SiglipVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SiglipVisionModel`]. It is used to instantiate a
    Siglip vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the Siglip
    [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224) architecture.

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
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    Example:

    ```python
    >>> from transformers import SiglipVisionConfig, SiglipVisionModel

    >>> # Initializing a SiglipVisionConfig with google/siglip-base-patch16-224 style configuration
    >>> configuration = SiglipVisionConfig()

    >>> # Initializing a SiglipVisionModel (with random weights) from the google/siglip-base-patch16-224 style configuration
    >>> model = SiglipVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    # 设置模型类型标识符为 "siglip_vision_model"
    model_type = "siglip_vision_model"
    # 初始化函数，用于创建一个新的配置对象
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        **kwargs,
    ):
        # 调用父类的初始化方法，传递所有未命名的参数
        super().__init__(**kwargs)

        # 设置配置对象的属性值
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 在 kwargs 中设置 token
        cls._set_token_in_kwargs(kwargs)

        # 获取配置字典和可能更新后的 kwargs
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典中的 model_type 是 "siglip"，则使用 vision_config 作为新的配置字典
        if config_dict.get("model_type") == "siglip":
            config_dict = config_dict["vision_config"]

        # 如果配置字典中包含 model_type，并且类有 model_type 属性，且它们不相同，发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 使用配置字典和 kwargs 创建新的配置对象
        return cls.from_dict(config_dict, **kwargs)
# 定义了一个继承自PretrainedConfig的配置类SiglipConfig，用于存储SiglipModel的配置信息
class SiglipConfig(PretrainedConfig):
    r"""
    [`SiglipConfig`] 是用来存储 [`SiglipModel`] 配置的类。它用于根据指定的参数实例化一个Siglip模型，
    定义了文本模型和视觉模型的配置。使用默认值实例化配置将产生类似于Siglip [google/siglip-base-patch16-224]
    架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可以用来控制模型的输出。详细信息请参阅 [`PretrainedConfig`] 的文档。

    Args:
        text_config (`dict`, *optional*):
            用于初始化 [`SiglipTextConfig`] 的配置选项字典。
        vision_config (`dict`, *optional*):
            用于初始化 [`SiglipVisionConfig`] 的配置选项字典。
        kwargs (*optional*):
            关键字参数字典。

    Example:

    ```python
    >>> from transformers import SiglipConfig, SiglipModel

    >>> # 使用google/siglip-base-patch16-224风格的配置初始化SiglipConfig
    >>> configuration = SiglipConfig()

    >>> # 使用google/siglip-base-patch16-224风格的配置初始化SiglipModel（随机权重）
    >>> model = SiglipModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config

    >>> # 我们还可以从SiglipTextConfig和SiglipVisionConfig初始化SiglipConfig
    >>> from transformers import SiglipTextConfig, SiglipVisionConfig

    >>> # 初始化SiglipText和SiglipVision配置
    >>> config_text = SiglipTextConfig()
    >>> config_vision = SiglipVisionConfig()

    >>> config = SiglipConfig.from_text_vision_configs(config_text, config_vision)
    ```"""

    # 模型类型标识为"siglip"
    model_type = "siglip"

    # 初始化方法，接受text_config和vision_config作为可选参数
    def __init__(self, text_config=None, vision_config=None, **kwargs):
        # 调用父类PretrainedConfig的初始化方法
        super().__init__(**kwargs)

        # 如果text_config为None，则使用默认空字典并记录日志
        if text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `SiglipTextConfig` with default values.")

        # 如果vision_config为None，则使用默认空字典并记录日志
        if vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. initializing the `SiglipVisionConfig` with default values.")

        # 使用text_config初始化SiglipTextConfig对象，并赋值给self.text_config
        self.text_config = SiglipTextConfig(**text_config)
        # 使用vision_config初始化SiglipVisionConfig对象，并赋值给self.vision_config
        self.vision_config = SiglipVisionConfig(**vision_config)

        # 设置初始化因子为1.0
        self.initializer_factor = 1.0

    @classmethod
    # 定义一个类方法，用于从 Siglip 文本模型配置和 Siglip 视觉模型配置实例化一个 SiglipConfig 对象或其派生类。
    @classmethod
    # 方法参数包括文本配置对象 text_config、视觉配置对象 vision_config，以及任意额外的关键字参数 kwargs
    def from_text_vision_configs(cls, text_config: SiglipTextConfig, vision_config: SiglipVisionConfig, **kwargs):
        r"""
        Instantiate a [`SiglipConfig`] (or a derived class) from siglip text model configuration and siglip vision
        model configuration.

        Returns:
            [`SiglipConfig`]: An instance of a configuration object
        """
        # 使用 text_config 的字典形式和 vision_config 的字典形式，以及任意额外的关键字参数，实例化一个 SiglipConfig 对象并返回
        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)
```