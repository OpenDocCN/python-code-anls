# `.\models\align\configuration_align.py`

```
# coding=utf-8
# 声明文件的编码格式为UTF-8

# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
# 版权声明，保留所有权利

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache 许可证版本 2.0 许可，除非符合许可协议要求，否则禁止使用该文件
# You may obtain a copy of the License at
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 除非适用法律要求或书面同意，本软件按"原样"提供，不提供任何形式的明示或暗示保证或条件。
# 请参阅许可证了解更多信息。

""" ALIGN model configuration"""
# 模型配置的文档字符串注释

import os
# 导入标准库os

from typing import TYPE_CHECKING, List, Union
# 导入类型提示相关模块

if TYPE_CHECKING:
    pass
# 如果在类型检查环境下，不执行任何操作

from ...configuration_utils import PretrainedConfig
# 导入预训练配置工具中的PretrainedConfig类

from ...utils import logging
# 导入工具包中的logging模块

logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器对象

ALIGN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "kakaobrain/align-base": "https://huggingface.co/kakaobrain/align-base/resolve/main/config.json",
}
# 预训练模型名称到配置文件地址的映射字典

class AlignTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`AlignTextModel`]. It is used to instantiate a
    ALIGN text encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the text encoder of the ALIGN
    [kakaobrain/align-base](https://huggingface.co/kakaobrain/align-base) architecture. The default values here are
    copied from BERT.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```python
    >>> from transformers import AlignTextConfig, AlignTextModel

    >>> # Initializing a AlignTextConfig with kakaobrain/align-base style configuration
    >>> configuration = AlignTextConfig()

    >>> # Initializing a AlignTextModel (with random weights) from the kakaobrain/align-base style configuration
    >>> model = AlignTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """
    # AlignTextConfig类的文档字符串注释，描述了如何配置和使用该类

    model_type = "align_text_model"
    # 模型类型为align_text_model

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        **kwargs,
    ):
        # 初始化函数，用于设置模型的各项配置参数

        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            position_embedding_type=position_embedding_type,
            use_cache=use_cache,
            **kwargs,
        )
        # 调用父类的初始化函数，设置模型配置参数
    ):
        super().__init__(**kwargs)
        # 调用父类的初始化方法，传入所有的关键字参数

        self.vocab_size = vocab_size
        # 设定词汇表大小

        self.hidden_size = hidden_size
        # 设定隐藏层大小

        self.num_hidden_layers = num_hidden_layers
        # 设定隐藏层数量

        self.num_attention_heads = num_attention_heads
        # 设定注意力头的数量

        self.hidden_act = hidden_act
        # 设定隐藏层激活函数

        self.intermediate_size = intermediate_size
        # 设定中间层大小

        self.hidden_dropout_prob = hidden_dropout_prob
        # 设定隐藏层的Dropout概率

        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 设定注意力概率的Dropout概率

        self.max_position_embeddings = max_position_embeddings
        # 设定最大位置嵌入数量

        self.type_vocab_size = type_vocab_size
        # 设定类型词汇表大小

        self.initializer_range = initializer_range
        # 设定初始化范围

        self.layer_norm_eps = layer_norm_eps
        # 设定LayerNormalization的epsilon值

        self.position_embedding_type = position_embedding_type
        # 设定位置嵌入类型

        self.use_cache = use_cache
        # 设定是否使用缓存

        self.pad_token_id = pad_token_id
        # 设定填充token的ID

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)
        # 在关键字参数中设置token

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        # 获取预训练模型的配置字典和更新后的关键字参数

        # 如果从AlignConfig加载，则获取文本配置字典
        if config_dict.get("model_type") == "align":
            config_dict = config_dict["text_config"]
        # 如果配置字典中包含"model_type"且类中有"model_type"属性，并且它们不相等，发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)
        # 使用配置字典和关键字参数构建并返回预训练配置类实例
# 定义一个配置类 `AlignVisionConfig`，继承自 `PretrainedConfig`，用于存储 [`AlignVisionModel`] 的配置信息。
# 该类用于实例化一个 ALIGN 视觉编码器，根据指定的参数定义模型架构。
# 实例化一个带有默认值的配置将产生与 ALIGN 架构的视觉编码器类似的配置。
# 默认值来自 EfficientNet (efficientnet-b7)。
# 
# 配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。更多信息请参阅 [`PretrainedConfig`] 的文档。
class AlignVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`AlignVisionModel`]. It is used to instantiate a
    ALIGN vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the ALIGN
    [kakaobrain/align-base](https://huggingface.co/kakaobrain/align-base) architecture. The default values are copied
    from EfficientNet (efficientnet-b7)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`int`, *optional*, defaults to 600):
            The input image size.
        width_coefficient (`float`, *optional*, defaults to 2.0):
            Scaling coefficient for network width at each stage.
        depth_coefficient (`float`, *optional*, defaults to 3.1):
            Scaling coefficient for network depth at each stage.
        depth_divisor (`int`, *optional*, defaults to 8):
            A unit of network width.
        kernel_sizes (`List[int]`, *optional*, defaults to `[3, 3, 5, 3, 5, 5, 3]`):
            List of kernel sizes to be used in each block.
        in_channels (`List[int]`, *optional*, defaults to `[32, 16, 24, 40, 80, 112, 192]`):
            List of input channel sizes to be used in each block for convolutional layers.
        out_channels (`List[int]`, *optional*, defaults to `[16, 24, 40, 80, 112, 192, 320]`):
            List of output channel sizes to be used in each block for convolutional layers.
        depthwise_padding (`List[int]`, *optional*, defaults to `[]`):
            List of block indices with square padding.
        strides (`List[int]`, *optional*, defaults to `[1, 2, 2, 2, 1, 2, 1]`):
            List of stride sizes to be used in each block for convolutional layers.
        num_block_repeats (`List[int]`, *optional*, defaults to `[1, 2, 2, 3, 3, 4, 1]`):
            List of the number of times each block is to be repeated.
        expand_ratios (`List[int]`, *optional*, defaults to `[1, 6, 6, 6, 6, 6, 6]`):
            List of scaling coefficients for each block.
        squeeze_expansion_ratio (`float`, *optional*, defaults to 0.25):
            Squeeze expansion ratio.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in each block. Supported options are
            `"gelu"`, `"relu"`, `"selu"`, `"gelu_new"`, `"silu"`, and `"mish"`.
        hidden_dim (`int`, *optional*, defaults to 1280):
            The hidden dimension of the layer before the classification head.
        pooling_type (`str` or `function`, *optional*, defaults to `"mean"`):
            Type of final pooling to be applied before the dense classification head. Options are `"mean"` or `"max"`.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing weight matrices.
        batch_norm_eps (`float`, *optional*, defaults to 1e-3):
            Epsilon value used by batch normalization layers.
        batch_norm_momentum (`float`, *optional*, defaults to 0.99):
            Momentum value used by batch normalization layers.
        drop_connect_rate (`float`, *optional*, defaults to 0.2):
            The drop rate for skip connections.

    Example:

    ```python
    >>> from transformers import AlignVisionConfig, AlignVisionModel

    >>> # 使用 kakaobrain/align-base 风格的配置初始化 AlignVisionConfig
    >>> configuration = AlignVisionConfig()

    >>> # 使用 kakaobrain/align-base 风格的配置初始化一个带有随机权重的 AlignVisionModel
    >>> model = AlignVisionModel(configuration)

    >>> # 访问模型的配置信息
    >>> configuration = model.config



    model_type = "align_vision_model"

    def __init__(
        self,
        num_channels: int = 3,
        image_size: int = 600,
        width_coefficient: float = 2.0,
        depth_coefficient: float = 3.1,
        depth_divisor: int = 8,
        kernel_sizes: List[int] = [3, 3, 5, 3, 5, 5, 3],
        in_channels: List[int] = [32, 16, 24, 40, 80, 112, 192],
        out_channels: List[int] = [16, 24, 40, 80, 112, 192, 320],
        depthwise_padding: List[int] = [],
        strides: List[int] = [1, 2, 2, 2, 1, 2, 1],
        num_block_repeats: List[int] = [1, 2, 2, 3, 3, 4, 1],
        expand_ratios: List[int] = [1, 6, 6, 6, 6, 6, 6],
        squeeze_expansion_ratio: float = 0.25,
        hidden_act: str = "swish",
        hidden_dim: int = 2560,
        pooling_type: str = "mean",
        initializer_range: float = 0.02,
        batch_norm_eps: float = 0.001,
        batch_norm_momentum: float = 0.99,
        drop_connect_rate: float = 0.2,
        **kwargs,
    ):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 初始化模型的各种参数
        self.num_channels = num_channels
        self.image_size = image_size
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.depth_divisor = depth_divisor
        self.kernel_sizes = kernel_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depthwise_padding = depthwise_padding
        self.strides = strides
        self.num_block_repeats = num_block_repeats
        self.expand_ratios = expand_ratios
        self.squeeze_expansion_ratio = squeeze_expansion_ratio
        self.hidden_act = hidden_act
        self.hidden_dim = hidden_dim
        self.pooling_type = pooling_type
        self.initializer_range = initializer_range
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum
        self.drop_connect_rate = drop_connect_rate
        self.num_hidden_layers = sum(num_block_repeats) * 4

    @classmethod
    # 类方法，用于从预训练模型名称或路径加载预训练配置，返回一个预训练配置对象
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 调用内部方法，将token设置到kwargs中
        cls._set_token_in_kwargs(kwargs)

        # 调用类方法获取配置字典和更新后的kwargs
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典中的模型类型为"align"，则从中获取视觉配置字典
        if config_dict.get("model_type") == "align":
            config_dict = config_dict["vision_config"]

        # 如果配置字典中有"model_type"键且当前类有"model_type"属性，并且二者不相等，发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 调用类方法，根据配置字典创建预训练配置对象并返回
        return cls.from_dict(config_dict, **kwargs)
# `AlignConfig` 是用来存储 [`AlignModel`] 的配置信息的类。
class AlignConfig(PretrainedConfig):
    r"""
    [`AlignConfig`] is the configuration class to store the configuration of a [`AlignModel`]. It is used to
    instantiate a ALIGN model according to the specified arguments, defining the text model and vision model configs.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the ALIGN
    [kakaobrain/align-base](https://huggingface.co/kakaobrain/align-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`AlignTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`AlignVisionConfig`].
        projection_dim (`int`, *optional*, defaults to 640):
            Dimentionality of text and vision projection layers.
        temperature_init_value (`float`, *optional*, defaults to 1.0):
            The inital value of the *temperature* paramter. Default is used as per the original ALIGN implementation.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import AlignConfig, AlignModel

    >>> # Initializing a AlignConfig with kakaobrain/align-base style configuration
    >>> configuration = AlignConfig()

    >>> # Initializing a AlignModel (with random weights) from the kakaobrain/align-base style configuration
    >>> model = AlignModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a AlignConfig from a AlignTextConfig and a AlignVisionConfig
    >>> from transformers import AlignTextConfig, AlignVisionConfig

    >>> # Initializing ALIGN Text and Vision configurations
    >>> config_text = AlignTextConfig()
    >>> config_vision = AlignVisionConfig()

    >>> config = AlignConfig.from_text_vision_configs(config_text, config_vision)
    ```"""
    
    # 类属性，标识模型类型为 "align"
    model_type = "align"

    # 初始化方法，接收多个参数来配置模型
    def __init__(
        self,
        text_config=None,               # 文本配置的字典，用于初始化 `AlignTextConfig`
        vision_config=None,             # 视觉配置的字典，用于初始化 `AlignVisionConfig`
        projection_dim=640,             # 文本和视觉投影层的维度
        temperature_init_value=1.0,     # 温度参数的初始值，默认为 1.0
        initializer_range=0.02,         # 所有权重矩阵初始化的截断正态分布标准差
        **kwargs,                       # 其他关键字参数
    ):
        super().__init__(**kwargs)
        # 调用父类的初始化方法，传入所有的关键字参数

        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the AlignTextConfig with default values.")
            # 如果文本配置为空，使用空字典，并记录日志，使用默认值初始化 AlignTextConfig

        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. Initializing the AlignVisionConfig with default values.")
            # 如果视觉配置为空，使用空字典，并记录日志，使用默认值初始化 AlignVisionConfig

        self.text_config = AlignTextConfig(**text_config)
        # 使用传入的文本配置参数初始化 AlignTextConfig 对象，并赋值给实例变量 self.text_config

        self.vision_config = AlignVisionConfig(**vision_config)
        # 使用传入的视觉配置参数初始化 AlignVisionConfig 对象，并赋值给实例变量 self.vision_config

        self.projection_dim = projection_dim
        # 将传入的 projection_dim 参数赋值给实例变量 self.projection_dim

        self.temperature_init_value = temperature_init_value
        # 将传入的 temperature_init_value 参数赋值给实例变量 self.temperature_init_value

        self.initializer_range = initializer_range
        # 将传入的 initializer_range 参数赋值给实例变量 self.initializer_range

    @classmethod
    def from_text_vision_configs(cls, text_config: AlignTextConfig, vision_config: AlignVisionConfig, **kwargs):
        r"""
        Instantiate a [`AlignConfig`] (or a derived class) from align text model configuration and align vision model
        configuration.

        Returns:
            [`AlignConfig`]: An instance of a configuration object
        """
        # 类方法：根据文本和视觉配置实例化一个 AlignConfig（或其派生类）对象

        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)
        # 返回使用文本和视觉配置对象的字典形式初始化的 AlignConfig 对象，同时传入其他关键字参数
```