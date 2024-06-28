# `.\models\altclip\configuration_altclip.py`

```py
# coding=utf-8
# Copyright 2022 WenXiang ZhongzhiCheng LedellWu LiuGuang BoWenZhang and The HuggingFace Inc. team. All rights reserved.
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
"""
AltCLIP model configuration
"""
import os  # 导入标准库中的os模块，用于与操作系统交互
from typing import Union  # 从typing模块导入Union类型，用于类型提示

from ...configuration_utils import PretrainedConfig  # 从本地的配置工具中导入预训练配置类
from ...utils import logging  # 从本地的工具模块中导入日志记录工具

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器实例

# 定义AltCLIP预训练模型配置文件的下载映射字典
ALTCLIP_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "BAAI/AltCLIP": "https://huggingface.co/BAAI/AltCLIP/resolve/main/config.json",
    # 查看所有AltCLIP模型的下载地址：https://huggingface.co/models?filter=altclip
}


class AltCLIPTextConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`AltCLIPTextModel`]. It is used to instantiate a
    AltCLIP text model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the AltCLIP
    [BAAI/AltCLIP](https://huggingface.co/BAAI/AltCLIP) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Examples:

    ```
    >>> from transformers import AltCLIPTextModel, AltCLIPTextConfig

    >>> # Initializing a AltCLIPTextConfig with BAAI/AltCLIP style configuration
    >>> configuration = AltCLIPTextConfig()

    >>> # Initializing a AltCLIPTextModel (with random weights) from the BAAI/AltCLIP style configuration
    >>> model = AltCLIPTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "altclip_text_model"

    def __init__(
        self,
        vocab_size=250002,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=514,
        type_vocab_size=1,
        initializer_range=0.02,
        initializer_factor=0.02,
        layer_norm_eps=1e-05,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        position_embedding_type="absolute",
        use_cache=True,
        project_dim=768,
        **kwargs,
        # 调用父类的初始化方法，设置模型参数，包括特殊的 token ID 和其他关键字参数
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 初始化模型的词汇表大小
        self.vocab_size = vocab_size
        # 初始化模型的隐藏层大小
        self.hidden_size = hidden_size
        # 初始化模型的隐藏层数量
        self.num_hidden_layers = num_hidden_layers
        # 初始化模型的注意力头数量
        self.num_attention_heads = num_attention_heads
        # 初始化模型的隐藏层激活函数
        self.hidden_act = hidden_act
        # 初始化模型的中间层大小
        self.intermediate_size = intermediate_size
        # 初始化模型的隐藏层 dropout 概率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 初始化模型的注意力机制 dropout 概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 初始化模型的最大位置嵌入长度
        self.max_position_embeddings = max_position_embeddings
        # 初始化模型的类型词汇表大小
        self.type_vocab_size = type_vocab_size
        # 初始化模型的参数初始化范围
        self.initializer_range = initializer_range
        # 初始化模型的初始化因子
        self.initializer_factor = initializer_factor
        # 初始化模型的层归一化 epsilon
        self.layer_norm_eps = layer_norm_eps
        # 初始化模型的位置嵌入类型
        self.position_embedding_type = position_embedding_type
        # 初始化模型是否使用缓存
        self.use_cache = use_cache
        # 初始化模型的投影维度
        self.project_dim = project_dim
# 定义 AltCLIPVisionConfig 类，继承自 PretrainedConfig 类，用于存储 AltCLIPModel 的配置信息
class AltCLIPVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`AltCLIPModel`]. It is used to instantiate an
    AltCLIP model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the AltCLIP
    [BAAI/AltCLIP](https://huggingface.co/BAAI/AltCLIP) architecture.

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

    ```
    >>> from transformers import AltCLIPVisionConfig, AltCLIPVisionModel

    >>> # Initializing a AltCLIPVisionConfig with BAAI/AltCLIP style configuration
    >>> configuration = AltCLIPVisionConfig()

    >>> # Initializing a AltCLIPVisionModel (with random weights) from the BAAI/AltCLIP style configuration
    >>> model = AltCLIPVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    # 将 model 对象的配置信息存储在 configuration 变量中
    configuration = model.config

    model_type = "altclip_vision_model"

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
        # 调用父类的构造函数，初始化对象
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

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 在 kwargs 中设置 token
        cls._set_token_in_kwargs(kwargs)

        # 获取预训练模型的配置字典和额外的 kwargs
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果从 AltCLIPConfig 加载，则获取视觉配置字典
        if config_dict.get("model_type") == "altclip":
            config_dict = config_dict["vision_config"]

        # 检查模型类型是否匹配，并生成警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从配置字典创建模型实例
        return cls.from_dict(config_dict, **kwargs)
class AltCLIPConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`AltCLIPModel`]. It is used to instantiate an
    AltCLIP model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the AltCLIP
    [BAAI/AltCLIP](https://huggingface.co/BAAI/AltCLIP) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`AltCLIPTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`AltCLIPVisionConfig`].
        projection_dim (`int`, *optional*, defaults to 768):
            Dimentionality of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* paramter. Default is used as per the original CLIP implementation.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```
    >>> from transformers import AltCLIPConfig, AltCLIPModel

    >>> # Initializing a AltCLIPConfig with BAAI/AltCLIP style configuration
    >>> configuration = AltCLIPConfig()

    >>> # Initializing a AltCLIPModel (with random weights) from the BAAI/AltCLIP style configuration
    >>> model = AltCLIPModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a AltCLIPConfig from a AltCLIPTextConfig and a AltCLIPVisionConfig

    >>> # Initializing a AltCLIPText and AltCLIPVision configuration
    >>> config_text = AltCLIPTextConfig()
    >>> config_vision = AltCLIPVisionConfig()

    >>> config = AltCLIPConfig.from_text_vision_configs(config_text, config_vision)
    ```"""

    model_type = "altclip"

    def __init__(
        self, text_config=None, vision_config=None, projection_dim=768, logit_scale_init_value=2.6592, **kwargs
    ):
        # 初始化方法，用于设置 AltCLIPConfig 对象的属性
        super().__init__(**kwargs)
        # 设置 text_config 属性，用于存储文本配置的字典
        self.text_config = text_config
        # 设置 vision_config 属性，用于存储视觉配置的字典
        self.vision_config = vision_config
        # 设置 projection_dim 属性，用于存储投影层的维度，默认为 768
        self.projection_dim = projection_dim
        # 设置 logit_scale_init_value 属性，用于存储 logit_scale 参数的初始值，默认为 2.6592
        self.logit_scale_init_value = logit_scale_init_value

    @classmethod
    def from_text_vision_configs(cls, text_config: AltCLIPTextConfig, vision_config: AltCLIPVisionConfig, **kwargs):
        r"""
        Instantiate a [`AltCLIPConfig`] (or a derived class) from altclip text model configuration and altclip vision
        model configuration.

        Returns:
            [`AltCLIPConfig`]: An instance of a configuration object
        """
        # 从给定的 AltCLIPTextConfig 和 AltCLIPVisionConfig 对象实例化一个 AltCLIPConfig 对象
        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)
```