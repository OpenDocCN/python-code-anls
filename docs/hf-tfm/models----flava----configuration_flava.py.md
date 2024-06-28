# `.\models\flava\configuration_flava.py`

```py
# coding=utf-8
# 上面的声明指定了代码文件的编码格式为 UTF-8，确保支持中文等非ASCII字符
# Copyright 2022 Meta Platforms authors and The HuggingFace Team. All rights reserved.
# 版权声明，保留所有权利，指出代码的版权归 Meta Platforms 和 The HuggingFace Team 所有
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License, Version 2.0 许可证授权，可以自由使用本代码，遵循许可证的条件
# you may not use this file except in compliance with the License.
# 除非符合许可证的条件，否则不得使用此文件
# You may obtain a copy of the License at
# 可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非适用法律要求或书面同意，否则按“原样”分发的软件没有任何明示或暗示的保证或条件
# See the License for the specific language governing permissions and
# limitations under the License.
# 详细了解许可证，包括特定语言控制权限和限制，请参阅许可证
""" FLAVA model configurations"""
# FLAVA 模型的配置信息

import os
# 导入操作系统相关模块
from typing import Any, Dict, Union
# 导入用于类型提示的模块

from ...configuration_utils import PretrainedConfig
# 从配置工具中导入预训练配置类
from ...utils import logging
# 导入日志记录相关的模块

logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器

FLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/flava-full": "https://huggingface.co/facebook/flava-full/resolve/main/config.json",
}
# FLAVA 预训练模型的配置文件映射表，指定了模型名称及其对应的配置文件地址

class FlavaImageConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FlavaImageModel`]. It is used to instantiate an
    FLAVA model according to the specified arguments, defining the model architecture.
    # 这是用于存储 [`FlavaImageModel`] 配置的配置类，用于根据指定的参数实例化 FLAVA 模型，定义模型架构。

    Instantiating a configuration with the defaults will yield a similar configuration to that of the FLAVA
    [facebook/flava-full](https://huggingface.co/facebook/flava-full) architecture.
    # 使用默认值实例化配置将产生类似于 FLAVA [facebook/flava-full](https://huggingface.co/facebook/flava-full) 架构的配置。

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # 配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。更多信息，请阅读 [`PretrainedConfig`] 的文档。
    # 定义模型类型为 FLAVA 图像模型
    model_type = "flava_image_model"
    # 初始化函数，设置模型配置参数
    def __init__(
        self,
        hidden_size: int = 768,                             # 隐藏层大小，默认为768
        num_hidden_layers: int = 12,                        # 隐藏层数，默认为12
        num_attention_heads: int = 12,                      # 注意力头数，默认为12
        intermediate_size: int = 3072,                      # 中间层大小，默认为3072
        hidden_act: int = "gelu",                           # 隐藏层激活函数，默认为"gelu"
        hidden_dropout_prob: float = 0.0,                   # 隐藏层dropout概率，默认为0.0
        attention_probs_dropout_prob: float = 0.0,          # 注意力概率dropout概率，默认为0.0
        initializer_range: float = 0.02,                    # 初始化范围，默认为0.02
        layer_norm_eps: float = 1e-12,                      # LayerNorm的epsilon，默认为1e-12
        image_size: int = 224,                              # 图像大小，默认为224
        patch_size: int = 16,                               # 图像块大小，默认为16
        num_channels: int = 3,                              # 图像通道数，默认为3
        qkv_bias: bool = True,                              # 是否在QKV中使用偏置，默认为True
        mask_token: bool = True,                            # 是否使用掩码token，默认为True
        vocab_size: int = 8192,                             # 词汇表大小，默认为8192
        **kwargs,                                           # 其他关键字参数
    ):
        super().__init__(**kwargs)                          # 调用父类初始化方法

        self.hidden_size = hidden_size                      # 设置隐藏层大小
        self.num_hidden_layers = num_hidden_layers          # 设置隐藏层数
        self.num_attention_heads = num_attention_heads      # 设置注意力头数
        self.intermediate_size = intermediate_size          # 设置中间层大小
        self.hidden_act = hidden_act                        # 设置隐藏层激活函数
        self.hidden_dropout_prob = hidden_dropout_prob      # 设置隐藏层dropout概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob  # 设置注意力概率dropout概率
        self.initializer_range = initializer_range          # 设置初始化范围
        self.layer_norm_eps = layer_norm_eps                # 设置LayerNorm的epsilon
        self.image_size = image_size                        # 设置图像大小
        self.patch_size = patch_size                        # 设置图像块大小
        self.num_channels = num_channels                    # 设置图像通道数
        self.qkv_bias = qkv_bias                            # 设置是否在QKV中使用偏置
        self.mask_token = mask_token                        # 设置是否使用掩码token
        self.vocab_size = vocab_size                        # 设置词汇表大小

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)                    # 设置kwargs中的token

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)  # 获取配置字典和更新后的kwargs

        # 如果从FlavaConfig加载，获取图像配置字典
        if config_dict.get("model_type") == "flava":
            config_dict = config_dict["image_config"]

        # 如果配置字典中包含model_type，并且cls有model_type属性，并且不匹配时，发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)          # 从配置字典和kwargs创建实例
class FlavaTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FlavaTextModel`]. It is used to instantiate an
    FLAVA model according to the specified arguments, defining the model architecture.

    Instantiating a configuration with the defaults will yield a similar configuration to that of the FLAVA
    [facebook/flava-full](https://huggingface.co/facebook/flava-full) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```
    >>> from transformers import FlavaTextConfig, FlavaTextModel

    >>> # Initializing a FlavaTextModel with  style configuration
    >>> configuration = FlavaTextConfig()

    >>> # Initializing a FlavaTextModel model (with random weights) from the style configuration
    >>> model = FlavaTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "flava_text_model"

    def __init__(
        self,
        vocab_size: int = 30522,
        type_vocab_size: int = 2,
        max_position_embeddings: int = 512,
        position_embedding_type: str = "absolute",
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        qkv_bias: bool = True,
        **kwargs,
    ):
        # 调用父类构造函数，初始化继承自 PretrainedConfig 的属性
        super().__init__(**kwargs)

        # 初始化配置参数
        self.vocab_size = vocab_size                      # 词汇表大小
        self.type_vocab_size = type_vocab_size            # 类型词汇表大小
        self.max_position_embeddings = max_position_embeddings  # 最大位置嵌入长度
        self.position_embedding_type = position_embedding_type  # 位置嵌入类型
        self.hidden_size = hidden_size                    # 隐藏层大小
        self.num_hidden_layers = num_hidden_layers        # 隐藏层层数
        self.num_attention_heads = num_attention_heads    # 注意力头数
        self.intermediate_size = intermediate_size        # 中间层大小
        self.hidden_act = hidden_act                      # 隐藏层激活函数
        self.hidden_dropout_prob = hidden_dropout_prob    # 隐藏层 dropout 概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob  # 注意力 dropout 概率
        self.initializer_range = initializer_range        # 初始化范围
        self.layer_norm_eps = layer_norm_eps              # 层归一化 epsilon
        self.qkv_bias = qkv_bias                          # 是否使用 QKV 偏置
        self.pad_token_id = pad_token_id                  # 填充 token 的 ID

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 在kwargs中设置token
        cls._set_token_in_kwargs(kwargs)

        # 获取预训练模型的配置字典和更新后的kwargs
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典指定的模型类型是"flava"，则使用其text_config作为配置字典
        if config_dict.get("model_type") == "flava":
            config_dict = config_dict["text_config"]

        # 如果配置字典中有"model_type"字段，并且类(cls)具有"model_type"属性，
        # 且配置字典中的模型类型不等于类的模型类型，则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 使用配置字典和kwargs创建实例
        return cls.from_dict(config_dict, **kwargs)
# 定义一个配置类，用于存储 FlavaMultimodalModel 的配置信息。该类继承自 PretrainedConfig。
class FlavaMultimodalConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FlavaMultimodalModel`]. It is used to instantiate
    an FLAVA model according to the specified arguments, defining the model architecture.

    Instantiating a configuration with the defaults will yield a similar configuration to that of the FLAVA
    [facebook/flava-full](https://huggingface.co/facebook/flava-full) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        use_cls_token (`bool`, *optional*, defaults to `True`):
            Whether to use an extra CLS token for multimodal settings. Usually needed by the FLAVA model.

    Example:

    ```
    >>> from transformers import FlavaMultimodalConfig, FlavaMultimodalModel

    >>> # Initializing a FlavaMultimodalModel with  style configuration
    >>> configuration = FlavaMultimodalConfig()

    >>> # Initializing a FlavaMultimodalModel model (with random weights) from the style configuration
    >>> model = FlavaMultimodalModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    
    # 类属性，指定模型类型为 flava_multimodal_model
    model_type = "flava_multimodal_model"
    # 初始化方法，用于初始化模型配置参数
    def __init__(
        self,
        hidden_size: int = 768,                      # 隐藏层大小，默认为768
        num_hidden_layers: int = 6,                  # 隐藏层数，默认为6
        num_attention_heads: int = 12,               # 注意力头的数量，默认为12
        intermediate_size: int = 3072,               # 中间层大小，默认为3072
        hidden_act: int = "gelu",                    # 隐藏层激活函数，默认为gelu
        hidden_dropout_prob: int = 0.0,              # 隐藏层的dropout概率，默认为0.0
        attention_probs_dropout_prob: int = 0.0,     # 注意力概率的dropout概率，默认为0.0
        initializer_range: float = 0.02,             # 初始化范围，默认为0.02
        layer_norm_eps: float = 1e-12,               # 层归一化的epsilon值，默认为1e-12
        qkv_bias: bool = True,                       # 是否在QKV中使用偏置，默认为True
        use_cls_token: bool = True,                  # 是否使用CLS令牌，默认为True
        **kwargs,                                    # 其余关键字参数
    ):
        # 调用父类的初始化方法，传入其余的关键字参数
        super().__init__(**kwargs)

        # 设置模型的各种配置参数
        self.hidden_size = hidden_size                 # 设置隐藏层大小
        self.num_hidden_layers = num_hidden_layers     # 设置隐藏层数
        self.num_attention_heads = num_attention_heads # 设置注意力头的数量
        self.intermediate_size = intermediate_size     # 设置中间层大小
        self.hidden_act = hidden_act                   # 设置隐藏层激活函数
        self.hidden_dropout_prob = hidden_dropout_prob # 设置隐藏层的dropout概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob  # 设置注意力概率的dropout概率
        self.initializer_range = initializer_range     # 设置初始化范围
        self.layer_norm_eps = layer_norm_eps           # 设置层归一化的epsilon值
        self.qkv_bias = qkv_bias                       # 设置是否在QKV中使用偏置
        self.use_cls_token = use_cls_token             # 设置是否使用CLS令牌

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 在kwargs中设置token
        cls._set_token_in_kwargs(kwargs)

        # 获取模型的配置字典和剩余的kwargs参数
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典中的模型类型是"flava"，则使用多模态配置字典
        if config_dict.get("model_type") == "flava":
            config_dict = config_dict["multimodal_config"]

        # 如果配置字典中包含"model_type"且模型类型与当前类的模型类型不匹配，发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 使用配置字典和kwargs参数创建一个新的实例
        return cls.from_dict(config_dict, **kwargs)
class FlavaImageCodebookConfig(PretrainedConfig):
    model_type = "flava_image_codebook"

    r"""
    [`FlavaImageCodebookConfig`] is the configuration class to store the configuration of a [`FlavaImageCodebook`]. It
    is used to instantiate an FLAVA model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the FLAVA
    [facebook/flava-image-codebook](https://huggingface.co/facebook/flava-image-codebook) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_groups (`int`, defaults to 4):
            Number of groups to be created. This parameter as of now doesn't affect the model and is used for some
            internal calculation and estimations.
        input_channels (`int`, defaults to 3):
            Number of channels in the image to be passed.
        num_blocks_per_group (`int`, defaults to 2):
            Number of conv-based blocks per group.
        hidden_size (`int`, defaults to 256):
            Size of hidden dim for the blocks.
        vocab_size (`int`, defaults to 8192):
            Size of the output vocabulary for the codebook.
        freeze (`bool`, defaults to `True`):
            Whether to freeze the weights of the model.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```
    >>> from transformers import FlavaImageCodebookConfig, FlavaImageCodebook

    >>> # Initializing a FlavaImageCodebook with style configuration
    >>> configuration = FlavaImageCodebookConfig()

    >>> # Initializing a FlavaImageCodebook model (with random weights) from the style configuration
    >>> model = FlavaImageCodebook(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    def __init__(
        self,
        num_groups: int = 4,
        input_channels: int = 3,
        num_blocks_per_group: int = 2,
        hidden_size: int = 256,
        vocab_size: int = 8192,
        freeze: int = True,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        # 调用父类的初始化方法，传入所有额外的关键字参数
        super().__init__(**kwargs)
        # 设置对象的各项配置参数
        self.num_groups = num_groups  # 设置创建的组数
        self.input_channels = input_channels  # 设置传递的图像通道数
        self.num_blocks_per_group = num_blocks_per_group  # 设置每组的卷积块数
        self.hidden_size = hidden_size  # 设置块的隐藏维度大小
        self.vocab_size = vocab_size  # 设置代码本输出词汇表的大小
        self.freeze = freeze  # 设置是否冻结模型权重
        self.initializer_range = initializer_range  # 设置权重矩阵初始化的标准差范围

    @classmethod
    # 根据预训练模型名称或路径及额外参数创建一个预训练配置对象
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 在 kwargs 中设置 token
        cls._set_token_in_kwargs(kwargs)

        # 获取预训练模型的配置字典和更新后的 kwargs
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典指定模型类型为 "flava"，则从中获取图像代码本配置字典
        if config_dict.get("model_type") == "flava":
            config_dict = config_dict["image_codebook_config"]

        # 如果配置字典中包含 "model_type"，并且类有 "model_type" 属性，并且配置的模型类型与类中定义的不同，
        # 则发出警告提示，因为这可能会导致不同配置的模型出现错误。
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 根据配置字典和 kwargs 创建预训练配置对象并返回
        return cls.from_dict(config_dict, **kwargs)
# `FlavaConfig` 是一个配置类，继承自 `PretrainedConfig`
class FlavaConfig(PretrainedConfig):
    r"""
    [`FlavaConfig`] is the configuration class to store the configuration of a [`FlavaModel`]. It is used to
    instantiate FLAVA model according to the specified arguments, defining the text model, image model, image codebook
    and multimodal model configs. Instantiating a configuration with the defaults will yield a similar configuration to
    that of the FLAVA [facebook/flava-full](https://huggingface.co/facebook/flava-full) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        text_config (`dict`, *optional*):
            文本配置选项的字典，用于初始化 FlavaTextConfig。
        image_config (`dict`, *optional*):
            图像配置选项的字典，用于初始化 FlavaImageConfig。
        multimodal_config (`dict`, *optional*):
            多模态配置选项的字典，用于初始化 FlavaMultimodalConfig。
        hidden_size (`int`, *optional*, defaults to 768):
            编码器层和汇聚层的维度。
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            层归一化层使用的 epsilon 值。
        projection_dim (`int`, *optional*, defaults to 512):
            文本和图像投影层的维度。
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            *logit_scale* 参数的初始值，默认按照 FLAVA/CLIP 实现使用的默认值。
        initializer_range (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态分布的标准差。
        ce_ignore_index (`int`, *optional*, defaults to -100):
            交叉熵忽略的索引。
        mim_weight (`float`, *optional*, defaults to 1.0):
            分配给 MIM（Masked Image Modeling）单模态损失的权重。
        mlm_weight (`float`, *optional*, defaults to 1.0):
            分配给 MLM（Masked Language Modeling）单模态损失的权重。
        global_contrastive_weight (`float`, *optional*, defaults to 1.0):
            分配给全局对比度交叉对齐损失的权重。
        itm_weight (`float`, *optional*, defaults to 1.0):
            分配给图像-文本匹配多模态损失的权重。
        mmm_image_weight (`float`, *optional*, defaults to 1.0):
            分配给 MMM 损失的图像部分的权重。
        mmm_text_weight (`float`, *optional*, defaults to 1.0):
            分配给 MMM 损失的文本部分的权重。
        global_backprop_contrastive (`bool`, *optional*, defaults to `True`):
            是否在对比度损失中通过所有工作者进行全局反向传播。
        skip_unmasked_multimodal_encoder (`bool`, *optional*, defaults to `True`):
            是否跳过运行未屏蔽的多模态编码器，其输出不被 FLAVA 损失使用。
        return_loss (`bool`, *optional*, defaults to `True`):
            是否返回损失值。

        kwargs (*optional*):
            关键字参数的字典。

    Example:

    ```
    >>> from transformers import FlavaConfig, FlavaModel, FlavaForPreTraining

    >>> # 使用风格配置初始化 FlavaConfig
    >>> configuration = FlavaConfig()

    >>> # 使用风格配置初始化 FlavaModel 和 FlavaForPreTraining 模型（带有随机权重）
    ```
    >>> model = FlavaModel(configuration)
    >>> model_pre = FlavaForPreTraining(configuration)


    # 实例化 FlavaModel 和 FlavaForPreTraining 对象，使用给定的 configuration 配置
    >>> configuration = model.config
    >>> configuration_pre = model_pre.config


    model_type = "flava"

    def __init__(
        self,
        image_config: Dict[str, Any] = None,
        text_config: Dict[str, Any] = None,
        multimodal_config: Dict[str, Any] = None,
        image_codebook_config: Dict[str, Any] = None,
        hidden_size: int = 768,
        layer_norm_eps: float = 1e-12,
        projection_dim: int = 768,
        init_codebook: bool = True,
        logit_scale_init_value: float = 2.6592,
        initializer_range: float = 0.02,
        ce_ignore_index: int = -100,
        mim_weight: float = 1.0,
        mlm_weight: float = 1.0,
        global_contrastive_weight: float = 1.0,
        itm_weight: float = 1.0,
        mmm_image_weight: float = 1.0,
        mmm_text_weight: float = 1.0,
        global_backprop_contrastive: bool = True,
        skip_unmasked_multimodal_encoder: bool = True,
        return_loss: bool = True,
        **kwargs,
    ):


        r"""
        Instantiate a [`FlavaConfig`] (or a derived class) from flava text model configuration, flava image model
        configuration, flava multimodal model and flava codebook model configuration.

        Returns:
            [`FlavaConfig`]: An instance of a configuration object
        """

        # 使用给定的配置参数初始化 FlavaConfig 类的实例
        return cls(
            image_config=image_config.to_dict(),
            text_config=text_config.to_dict(),
            multimodal_config=multimodal_config.to_dict(),
            image_codebook_config=image_codebook_config.to_dict(),
            **kwargs,
        )
```