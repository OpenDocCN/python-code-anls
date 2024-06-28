# `.\models\owlv2\configuration_owlv2.py`

```py
# coding=utf-8
# 文件编码声明，指定使用UTF-8编码
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
# 版权声明，标明代码版权归HuggingFace Inc.团队所有

# Licensed under the Apache License, Version 2.0 (the "License");
# 指定代码采用Apache License, Version 2.0许可证发布
# you may not use this file except in compliance with the License.
# 除非符合许可证条件，否则不得使用此文件
# You may obtain a copy of the License at
# 可在上述链接获取许可证的副本

#     http://www.apache.org/licenses/LICENSE-2.0
# 许可证链接

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 根据许可证规定，软件按“原样”分发，不提供任何形式的保证或条件

""" OWLv2 model configuration"""
# 代码文件说明，指定为OWLv2模型配置

import os
# 导入操作系统模块
from typing import TYPE_CHECKING, Dict, Union
# 导入类型检查相关模块

if TYPE_CHECKING:
    pass
# 如果类型检查为真，则执行相应操作（此处为空）

from ...configuration_utils import PretrainedConfig
# 导入预训练配置类
from ...utils import logging
# 导入日志模块

logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器

OWLV2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/owlv2-base-patch16": "https://huggingface.co/google/owlv2-base-patch16/resolve/main/config.json",
}
# OWLv2预训练配置存档映射，将模型名称映射到其配置文件的URL

# Copied from transformers.models.owlvit.configuration_owlvit.OwlViTTextConfig with OwlViT->Owlv2, owlvit-base-patch32->owlv2-base-patch16, owlvit->owlv2, OWL-ViT->OWLv2
# 从transformers.models.owlvit.configuration_owlvit.OwlViTTextConfig复制而来，修改了OwlViT为Owlv2，owlvit-base-patch32为owlv2-base-patch16，owlvit为owlv2，OWL-ViT为OWLv2
class Owlv2TextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`Owlv2TextModel`]. It is used to instantiate an
    Owlv2 text encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Owlv2
    [google/owlv2-base-patch16](https://huggingface.co/google/owlv2-base-patch16) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # Owlv2TextConfig类，用于存储Owlv2TextModel的配置，根据指定参数实例化Owlv2文本编码器，定义模型架构。
    # 使用默认配置实例化将产生类似于Owlv2 [google/owlv2-base-patch16] 架构的配置。

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # 配置对象继承自[`PretrainedConfig`]，可用于控制模型输出。阅读[`PretrainedConfig`]文档以获取更多信息。
    # 定义默认的 OWLv2 文本模型的词汇表大小
    vocab_size (`int`, *optional*, defaults to 49408):
        Vocabulary size of the OWLv2 text model. Defines the number of different tokens that can be represented
        by the `inputs_ids` passed when calling [`Owlv2TextModel`].
    
    # 定义编码器层和池化层的维度大小
    hidden_size (`int`, *optional*, defaults to 512):
        Dimensionality of the encoder layers and the pooler layer.
    
    # 定义 Transformer 编码器中“中间”（即前馈）层的维度大小
    intermediate_size (`int`, *optional*, defaults to 2048):
        Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
    
    # 定义 Transformer 编码器中隐藏层的数量
    num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    
    # 定义 Transformer 编码器中每个注意力层的注意力头数量
    num_attention_heads (`int`, *optional*, defaults to 8):
        Number of attention heads for each attention layer in the Transformer encoder.
    
    # 设置该模型可能使用的最大序列长度，通常设为一个较大的值
    max_position_embeddings (`int`, *optional*, defaults to 16):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    
    # 定义编码器和池化器中的非线性激活函数
    hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
    
    # 定义层归一化层使用的 epsilon 值
    layer_norm_eps (`float`, *optional*, defaults to 1e-05):
        The epsilon used by the layer normalization layers.
    
    # 定义注意力概率的 dropout 比率
    attention_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the attention probabilities.
    
    # 初始化所有权重矩阵的截断正态分布的标准差
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    
    # 用于初始化所有权重矩阵的因子（通常保持为 1，仅在初始化测试中使用）
    initializer_factor (`float`, *optional*, defaults to 1.0):
        A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
        testing).
    
    # 输入序列中填充标记的 id
    pad_token_id (`int`, *optional*, defaults to 0):
        The id of the padding token in the input sequences.
    
    # 输入序列中开始序列的标记 id
    bos_token_id (`int`, *optional*, defaults to 49406):
        The id of the beginning-of-sequence token in the input sequences.
    
    # 输入序列中结束序列的标记 id
    eos_token_id (`int`, *optional*, defaults to 49407):
        The id of the end-of-sequence token in the input sequences.

Example:


>>> from transformers import Owlv2TextConfig, Owlv2TextModel

>>> # Initializing a Owlv2TextModel with google/owlv2-base-patch16 style configuration
>>> configuration = Owlv2TextConfig()

>>> # Initializing a Owlv2TextConfig from the google/owlv2-base-patch16 style configuration
>>> model = Owlv2TextModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config


model_type = "owlv2_text_model"
    # 初始化函数，用于创建一个新的配置对象实例
    def __init__(
        self,
        vocab_size=49408,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=8,
        max_position_embeddings=16,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        pad_token_id=0,
        bos_token_id=49406,
        eos_token_id=49407,
        **kwargs,
    ):
        # 调用父类的初始化函数，传递相关参数
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 初始化对象的属性，设置配置的默认值
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
        # 设置关键字参数中的特殊标记
        cls._set_token_in_kwargs(kwargs)

        # 获取预训练模型的配置字典及更新后的关键字参数
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典中的模型类型为 "owlv2"，则使用其文本配置字典
        if config_dict.get("model_type") == "owlv2":
            config_dict = config_dict["text_config"]

        # 如果类定义了模型类型，并且配置字典中的模型类型与类中的不同，发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从配置字典及更新后的关键字参数创建配置对象实例并返回
        return cls.from_dict(config_dict, **kwargs)
# 从 transformers.models.owlvit.configuration_owlvit.OwlViTVisionConfig 复制而来的配置类 Owlv2VisionConfig，对 OWLv2 图像编码器的配置进行存储。
# 用于根据指定参数实例化一个 OWLv2 图像编码器，定义模型架构。使用默认配置实例化该配置将得到类似于 OWLv2 google/owlv2-base-patch16 架构的配置。
class Owlv2VisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`Owlv2VisionModel`]. It is used to instantiate
    an OWLv2 image encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the OWLv2
    [google/owlv2-base-patch16](https://huggingface.co/google/owlv2-base-patch16) architecture.

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
        patch_size (`int`, *optional*, defaults to 16):
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
    >>> from transformers import Owlv2VisionConfig, Owlv2VisionModel

    >>> # Initializing a Owlv2VisionModel with google/owlv2-base-patch16 style configuration
    >>> configuration = Owlv2VisionConfig()

    >>> # Initializing a Owlv2VisionModel model from the google/owlv2-base-patch16 style configuration
    >>> model = Owlv2VisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

    model_type = "owlv2_vision_model"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=768,
        patch_size=16,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # 设置模型的各种参数
        self.hidden_size = hidden_size  # 隐藏层大小
        self.intermediate_size = intermediate_size  # 中间层大小
        self.num_hidden_layers = num_hidden_layers  # 隐藏层数量
        self.num_attention_heads = num_attention_heads  # 注意力头数量
        self.num_channels = num_channels  # 图像通道数
        self.image_size = image_size  # 图像大小
        self.patch_size = patch_size  # 图像分块大小
        self.hidden_act = hidden_act  # 隐藏层激活函数
        self.layer_norm_eps = layer_norm_eps  # 层归一化 epsilon 参数
        self.attention_dropout = attention_dropout  # 注意力机制的 dropout 概率
        self.initializer_range = initializer_range  # 初始化范围
        self.initializer_factor = initializer_factor  # 初始化因子

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        # 从预训练模型中加载配置字典和额外的关键字参数
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典中的模型类型是 "owlv2"，则使用视觉配置字典
        if config_dict.get("model_type") == "owlv2":
            config_dict = config_dict["vision_config"]

        # 检查模型类型是否与类属性中指定的模型类型匹配，如果不匹配则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从配置字典创建类的实例
        return cls.from_dict(config_dict, **kwargs)
# 从 transformers.models.owlvit.configuration_owlvit.OwlViTConfig 复制过来，将 OwlViT 替换为 Owlv2，owlvit-base-patch32 替换为 owlv2-base-patch16，owlvit 替换为 owlv2，OWL-ViT 替换为 OWLv2
class Owlv2Config(PretrainedConfig):
    r"""
    [`Owlv2Config`] 是用来存储 [`Owlv2Model`] 配置的类。它用于根据指定的参数实例化一个 OWLv2 模型，定义文本模型和视觉模型的配置。
    使用默认参数实例化配置将产生与 OWLv2 [google/owlv2-base-patch16](https://huggingface.co/google/owlv2-base-patch16) 架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。更多信息请参阅 [`PretrainedConfig`] 的文档。

    Args:
        text_config (`dict`, *optional*):
            用于初始化 [`Owlv2TextConfig`] 的配置选项字典。
        vision_config (`dict`, *optional*):
            用于初始化 [`Owlv2VisionConfig`] 的配置选项字典。
        projection_dim (`int`, *optional*, defaults to 512):
            文本和视觉投影层的维度。
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            *logit_scale* 参数的初始值。默认值与原始 OWLv2 实现相同。
        return_dict (`bool`, *optional*, defaults to `True`):
            模型是否应返回字典。如果为 `False`，返回一个元组。
        kwargs (*optional*):
            关键字参数的字典。
    """

    model_type = "owlv2"

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        projection_dim=512,
        logit_scale_init_value=2.6592,
        return_dict=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the Owlv2TextConfig with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. initializing the Owlv2VisionConfig with default values.")

        # 使用给定的文本配置和视觉配置初始化 Owlv2TextConfig 和 Owlv2VisionConfig 对象
        self.text_config = Owlv2TextConfig(**text_config)
        self.vision_config = Owlv2VisionConfig(**vision_config)

        # 设置投影维度、logit_scale 初始值和返回字典选项
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.return_dict = return_dict
        self.initializer_factor = 1.0

    @classmethod
    # 类方法：从预训练模型名称或路径加载配置，并返回预训练配置对象
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 在关键字参数中设置 token
        cls._set_token_in_kwargs(kwargs)

        # 调用类方法获取预训练模型的配置字典和更新后的关键字参数
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典中包含 "model_type" 键且类有 "model_type" 属性，并且它们不一致，发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 使用配置字典创建配置对象并返回
        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def from_text_vision_configs(cls, text_config: Dict, vision_config: Dict, **kwargs):
        r"""
        从 owlv2 文本模型配置和 owlv2 视觉模型配置实例化一个 [`Owlv2Config`]（或其派生类）。

        返回：
            [`Owlv2Config`]: 配置对象的一个实例
        """
        # 创建一个空的配置字典，存储文本配置和视觉配置
        config_dict = {}
        config_dict["text_config"] = text_config
        config_dict["vision_config"] = vision_config

        # 使用配置字典创建配置对象并返回
        return cls.from_dict(config_dict, **kwargs)
```