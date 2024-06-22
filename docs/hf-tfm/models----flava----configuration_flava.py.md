# `.\models\flava\configuration_flava.py`

```py
# coding=utf-8
# 指定文件编码为 UTF-8
# Copyright 2022 Meta Platforms authors and The HuggingFace Team. All rights reserved.
# 版权声明：版权为Meta Platforms作者和The HuggingFace团队所有
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 该文件遵循Apache License 2.0（许可证）的条款
# you may not use this file except in compliance with the License.
# 除非符合许可证规定，否则不得使用此文件
# You may obtain a copy of the License at
# 可以在以下URL获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#     许可证URL
#
# Unless required by applicable law or agreed to in writing, software
# 除非适用法律要求或书面同意，否则
# distributed under the License is distributed on an "AS IS" BASIS,
# 根据许可证发布的软件将是“按原样”提供，
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 不包含任何形式的明示或隐含的担保或条件。
# See the License for the specific language governing permissions and
# 参阅许可证了解管理权限和
# limitations under the License.
# 许可证下的限制。
""" FLAVA model configurations"""
# FLAVA模型配置的文件说明

import os
# 导入os模块，用于处理操作系统功能，如文件路径等
from typing import Any, Dict, Union
# 导入类型注解模块，用于代码类型标注

from ...configuration_utils import PretrainedConfig
# 从上级目录导入PretrainedConfig类，用于处理预训练模型的配置
from ...utils import logging
# 从上级目录导入logging模块，用于日志管理

logger = logging.get_logger(__name__)
# 创建一个logger对象，用于记录日志，__name__表示当前模块名

FLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/flava-full": "https://huggingface.co/facebook/flava-full/resolve/main/config.json",
}
# 定义一个字典，映射模型的名称到其预训练配置文件的URL

class FlavaImageConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FlavaImageModel`]. It is used to instantiate an
    FLAVA model according to the specified arguments, defining the model architecture.
    # 这是一个配置类，用于存储`FlavaImageModel`的配置。它用于根据指定的参数实例化FLAVA模型，定义模型的架构。

    Instantiating a configuration with the defaults will yield a similar configuration to that of the FLAVA
    [facebook/flava-full](https://huggingface.co/facebook/flava-full) architecture.
    # 使用默认配置实例化将产生一个与FLAVA [facebook/flava-full]架构类似的配置。

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # 配置对象继承自`PretrainedConfig`，可以用来控制模型输出。有关更多信息，请阅读`PretrainedConfig`的文档。
    # 设置隐藏层大小，默认为768
    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
    # 设置Transformer编码器中隐藏层的数量，默认为12
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
    # 设置Transformer编码器中每个注意力层的注意力头数，默认为12
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
    # 设置Transformer编码器中"中间"（即，前馈）层的维度，默认为3072
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
    # 设置编码器和池化层中的非线性激活函数，默认为"gelu"
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
    # 设置嵌入层、编码器和池化层中所有全连接层的丢弃概率，默认为0.0
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
    # 设置注意力概率的丢弃比率，默认为0.0
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
    # 初始化所有权重矩阵的截断正态分布标准差，默认为0.02
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    # 层归一化层使用的epsilon，默认为1e-12
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
    # 每个图像的大小（分辨率），默认为224
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
    # 每个补丁的大小（分辨率），默认为16
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
    # 输入通道的数量，默认为3
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
    # 是否对查询、键和值添加偏置，默认为True
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
    # 是否使用掩码令牌，默认为True。在FLAVA中用于MIM（Masked Image Modeling）损失
        mask_token (`bool`, *optional*, defaults to `True`):
            Whether to use a mask token or not. Used in MIM (Masked Image Modeling) loss for FLAVA.
    # 与[`FlavaImageModel`]结合使用的 [`FlavaImageCodebook`] 的词汇大小，用于FLAVA的MIM（Masked Image Modeling）损失，默认为8192
        vocab_size (`int`, *optional*, defaults to 8192):
            Vocabulary size of the [`FlavaImageCodebook`] used in conjunction with [`FlavaImageModel`] for MIM (Masked Image Modeling) loss for FLAVA.

    # 示例代码
    Example:

    ```python
    >>> from transformers import FlavaImageConfig, FlavaImageModel

    >>> # 使用style配置初始化FlavaImageModel
    >>> configuration = FlavaImageConfig()

    >>> # 使用style配置初始化随机权重的FlavaImageModel模型
    >>> model = FlavaImageModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py"""
    
    # 设置模型类型为"flava_image_model"
    model_type = "flava_image_model"
    # 初始化函数，设置各种默认参数
    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: int = "gelu",
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        qkv_bias: bool = True,
        mask_token: bool = True,
        vocab_size: int = 8192,
        **kwargs,
    ):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 设置各种参数值
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.mask_token = mask_token
        self.vocab_size = vocab_size

    @classmethod
    # 从预训练模型中加载配置
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 设置token到kwargs中
        cls._set_token_in_kwargs(kwargs)

        # 获取配置字典
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果从FlavaConfig加载，获取image配置字典
        if config_dict.get("model_type") == "flava":
            config_dict = config_dict["image_config"]

        # 如果配置字典中包含model_type，并且类有model_type属性，并且model_type不匹配，发出警告信息
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 返回配置类实例
        return cls.from_dict(config_dict, **kwargs)
# 定义了一个名为FlavaTextConfig的类，该类继承自PretrainedConfig
class FlavaTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FlavaTextModel`]. It is used to instantiate an
    FLAVA model according to the specified arguments, defining the model architecture.

    Instantiating a configuration with the defaults will yield a similar configuration to that of the FLAVA
    [facebook/flava-full](https://huggingface.co/facebook/flava-full) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```python
    >>> from transformers import FlavaTextConfig, FlavaTextModel

    >>> # Initializing a FlavaTextModel with  style configuration
    >>> configuration = FlavaTextConfig()

    >>> # Initializing a FlavaTextModel model (with random weights) from the style configuration
    >>> model = FlavaTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```py
    """

    # 定义了类属性model_type，并赋值为"flava_text_model"
    model_type = "flava_text_model"

    # 定义了初始化方法__init__，接收各种参数
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
        # 调用父类PretrainedConfig的__init__方法
        super().__init__(**kwargs)

        # 初始化各个实例属性，并赋值为相应的参数值
        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.position_embedding_type = position_embedding_type
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
        self.pad_token_id = pad_token_id

    # 定义了一个classmethod
    @classmethod
    # 定义一个类方法，用于从预训练模型名称或路径加载模型配置
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 在传递的参数中设置 token
        cls._set_token_in_kwargs(kwargs)

        # 获取配置字典和更新后的 kwargs
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典中的模型类型为 "flava"，则获取文本配置字典
        if config_dict.get("model_type") == "flava":
            config_dict = config_dict["text_config"]

        # 检查配置字典中的模型类型是否与当前类的模型类型匹配，如果不匹配则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 根据配置字典和参数实例化一个模型配置对象并返回
        return cls.from_dict(config_dict, **kwargs)
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
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
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

    ```python
    >>> from transformers import FlavaMultimodalConfig, FlavaMultimodalModel

    >>> # Initializing a FlavaMultimodalModel with  style configuration
    >>> configuration = FlavaMultimodalConfig()

    >>> # Initializing a FlavaMultimodalModel model (with random weights) from the style configuration
    >>> model = FlavaMultimodalModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```py"""

    # 指定模型类型为 "flava_multimodal_model"
    model_type = "flava_multimodal_model"
    # 初始化函数，设置模型的各项参数
    def __init__(
        self,
        hidden_size: int = 768,                              # 隐藏层的大小
        num_hidden_layers: int = 6,                           # 隐藏层数
        num_attention_heads: int = 12,                        # 注意力头的数量
        intermediate_size: int = 3072,                        # 中间层的大小
        hidden_act: int = "gelu",                              # 隐藏层激活函数
        hidden_dropout_prob: int = 0.0,                        # 隐藏层的丢弃概率
        attention_probs_dropout_prob: int = 0.0,               # 注意力概率的丢弃概率
        initializer_range: float = 0.02,                        # 初始化范围
        layer_norm_eps: float = 1e-12,                           # 层归一化的epsilon值
        qkv_bias: bool = True,                                      # 是否使用qkv偏置
        use_cls_token: bool = True,                                   # 是否使用CLS标记
        **kwargs,                                                        # 其他参数
    ):
        super().__init__(**kwargs)                                    # 调用父类初始化函数

        # 设置模型的各项参数
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
        self.use_cls_token = use_cls_token

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)                                    # 调用内部方法，设置kwargs

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)    # 获取配置字典和kwargs

        # 如果加载自FlavaConfig，则获取多模态配置字典
        if config_dict.get("model_type") == "flava":
            config_dict = config_dict["multimodal_config"]

        # 如果配置字典中存在模型类型并且是不同于当前模型类型，则产生警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)    # 从配置字典和kwargs创建实例
class FlavaImageCodebookConfig(PretrainedConfig):
    # 定义模型类型为 "flava_image_codebook"
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

    ```python
    >>> from transformers import FlavaImageCodebookConfig, FlavaImageCodebook

    >>> # Initializing a FlavaImageCodebook with style configuration
    >>> configuration = FlavaImageCodebookConfig()

    >>> # Initializing a FlavaImageCodebook model (with random weights) from the style configuration
    >>> model = FlavaImageCodebook(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```py
    """

    # 初始化方法
    def __init__(
        self,
        num_groups: int = 4,  # 创建的组数，默认为4
        input_channels: int = 3,  # 传递的图像通道数，默认为3
        num_blocks_per_group: int = 2,  # 每个组的基于卷积的块的数量，默认为2
        hidden_size: int = 256,  # 块的隐藏维度大小，默认为256
        vocab_size: int = 8192,  # 用于代码书的输出词汇表的大小，默认为8192
        freeze: int = True,  # 是否冻结模型的权重，默认为True
        initializer_range: float = 0.02,  # 用于初始化所有权重矩阵的截断正态初始化器的标准差，默认为0.02
        **kwargs,  # 可选的其他参数
    ):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 设置各个参数
        self.num_groups = num_groups
        self.input_channels = input_channels
        self.num_blocks_per_group = num_blocks_per_group
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.freeze = freeze
        self.initializer_range = initializer_range

    # 类方法
    @classmethod
        # 从预训练模型的名称或路径创建一个预训练配置对象
        def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
            # 在kwargs中设置令牌
            cls._set_token_in_kwargs(kwargs)

            # 获取预训练模型的配置字典和kwargs
            config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

            # 如果我们正在从FlavaConfig加载，则获取图像码书配置字典
            if config_dict.get("model_type") == "flava":
                config_dict = config_dict["image_codebook_config"]

            # 如果配置字典中存在"model_type"，并且类中存在"model_type"属性，并且配置字典中的"model_type"不等于类的model_type
            if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
                logger.warning(
                    f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                    f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
                )

            # 从配置字典创建一个实例化对象
            return cls.from_dict(config_dict, **kwargs)
class FlavaConfig(PretrainedConfig):
    r"""
    [`FlavaConfig`] is the configuration class to store the configuration of a [`FlavaModel`]. It is used to
    instantiate FLAVA model according to the specified arguments, defining the text model, image model, image codebook
    and multimodal model configs. Instantiating a configuration with the defaults will yield a similar configuration to
    that of the FLAVA [facebook/flava-full](https://huggingface.co/facebook/flava-full) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


# 定义了一个名为FlavaConfig的类，继承自PretrainedConfig类，用于存储FlavaModel的配置。
# 该类用于根据指定的参数实例化FLAVA模型，定义文本模型、图像模型、图像代码本和多模型配置。
# 使用默认参数实例化一个配置将会产生与FLAVA facebook/flava-full 架构类似的配置。
# 配置对象继承自PretrainedConfig，并可用于控制模型输出。阅读PretrainedConfig的文档以获取更多信息。
    # 定义函数，用于创建 FLAVA 模型的配置
    Args:
        text_config (`dict`, *optional*):
            用于初始化 [`FlavaTextConfig`] 的配置选项字典。
        image_config (`dict`, *optional*):
            用于初始化 [`FlavaImageConfig`] 的配置选项字典。
        multimodal_config (`dict`, *optional*):
            用于初始化 [`FlavaMultimodalConfig`] 的配置选项字典。
        hidden_size (`int`, *optional*, 默认为 768):
            编码器层和汇合层的维度。
        layer_norm_eps (`float`, *optional*, 默认为 1e-12):
            层归一化层使用的 epsilon 值。
        projection_dim (`int`, *optional*, 默认为 512):
            文本和图像投影层的维度。
        logit_scale_init_value (`float`, *optional*, 默认为 2.6592):
            *logit_scale* 参数的初始值。默认值是根据原始 FLAVA/CLIP 实现而使用的。
        initializer_range (`float`, *optional*, 默认为 0.02):
            用于初始化所有权重矩阵的截断正态分布的标准偏差。
        ce_ignore_index (`int`, *optional*, 默认为 -100):
            要忽略的交叉熵索引。
        mim_weight (`float`, *optional*, 默认为 1.0):
            分配给 MIM（Masked Image Modeling）单模态损失的权重。
        mlm_weight (`float`, *optional*, 默认为 1.0):
            分配给 MLM（Masked Language Modeling）单模态损失的权重。
        global_contrastive_weight (`float`, *optional*, 默认为 1.0):
            分配给全局对比度跨模态对齐损失的权重。
        itm_weight (`float`, *optional*, 默认为 1.0):
            分配给图像-文本匹配多模态损失的权重。
        mmm_image_weight (`float`, *optional*, 默认为 1.0):
            分配给 MMM 损失的图像部分的权重。
        mmm_text_weight (`float`, *optional*, 默认为 1.0):
            分配给 MMM 损失的文本部分的权重。
        global_backprop_contrastive (`bool`, *optional*, 默认为 `True`):
            是否在对比损失中通过所有工作器进行全局反向传播。
        skip_unmasked_multimodal_encoder (`bool`, *optional*, 默认为 `True`):
            是否跳过运行未掩盖的多模态编码器，其输出未被 FLAVA 损失使用。
        return_loss (`bool`, *optional*, 默认为 `True`):
            是否返回损失。

        kwargs (*optional*):
            关键字参数字典。

    Example:

    ```python
    >>> from transformers import FlavaConfig, FlavaModel, FlavaForPreTraining

    >>> # 使用样式配置初始化 FlavaConfig
    >>> configuration = FlavaConfig()

    >>> # 使用样式配置初始化 FlavaModel 和 FlavaForPreTraining 模型（带有随机权重）
    # 创建一个名为model的FlavaModel对象，使用给定配置
    model = FlavaModel(configuration)
    # 创建一个名为model_pre的FlavaForPreTraining对象，使用给定配置
    model_pre = FlavaForPreTraining(configuration)

    # 访问模型配置
    # 获取model对象的配置并赋值给configuration
    configuration = model.config
    # 获取model_pre对象的配置并赋值给configuration_pre
    configuration_pre = model_pre.config

    # 设置模型类型为"flava"
    model_type = "flava"

    # 定义FlavaConfig类
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
    # 从给定的配置创建一个FlavaConfig实例
    @classmethod
    def from_configs(
        cls,
        image_config: FlavaImageConfig,
        text_config: FlavaTextConfig,
        multimodal_config: FlavaMultimodalConfig,
        image_codebook_config: FlavaImageCodebookConfig,
        **kwargs,
    ):
        r"""
        从flava文本模型配置、flava图片模型配置、flava多模态模型配置和flava码书模型配置实例化一个`FlavaConfig`（或派生类）。

        返回:
            [`FlavaConfig`]: 配置对象的一个实例
        """

        # 从给定的配置创建一个FlavaConfig实例
        return cls(
            image_config=image_config.to_dict(),
            text_config=text_config.to_dict(),
            multimodal_config=multimodal_config.to_dict(),
            image_codebook_config=image_codebook_config.to_dict(),
            **kwargs,
        )
```