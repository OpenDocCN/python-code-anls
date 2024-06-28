# `.\models\blip_2\configuration_blip_2.py`

```py
# coding=utf-8
# 定义文件编码格式为 UTF-8

# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
# 版权声明，版权归 HuggingFace 公司所有

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache 许可证 2.0 版本进行许可

# you may not use this file except in compliance with the License.
# 除非遵守许可证，否则不得使用此文件

# You may obtain a copy of the License at
# 您可以在以下网址获取许可证副本

#     http://www.apache.org/licenses/LICENSE-2.0
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非适用法律要求或书面同意，否则依据“现状”分发软件，无论是明示的还是暗示的保证或条件

# See the License for the specific language governing permissions and
# limitations under the License.
# 请查阅许可证了解特定语言的权限和限制

""" BLIP-2 model configuration"""
# BLIP-2 模型配置信息

import os
# 导入操作系统相关模块

from typing import Union
# 导入 Union 类型提示

from ...configuration_utils import PretrainedConfig
# 从配置工具中导入 PretrainedConfig 类

from ...models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
# 从自动化模型中导入模型映射名称

from ...utils import logging
# 从工具包中导入日志记录模块

from ..auto import CONFIG_MAPPING
# 从自动化模块中导入配置映射

logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器

BLIP_2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "salesforce/blip2-opt-2.7b": "https://huggingface.co/salesforce/blip2-opt-2.7b/resolve/main/config.json",
}
# BLIP-2 预训练配置文件映射，指定了模型的配置文件路径和名称
# 键是模型的名称，值是配置文件的 URL 地址

class Blip2VisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Blip2VisionModel`]. It is used to instantiate a
    BLIP-2 vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration defaults will yield a similar configuration to that of the BLIP-2
    [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # Blip2VisionConfig 类，用于存储 Blip2VisionModel 的配置信息
    # 此类通过指定的参数实例化 BLIP-2 视觉编码器，定义模型架构
    # 实例化配置的默认值将产生与 BLIP-2 [Salesforce/blip2-opt-2.7b] 架构相似的配置
    # 配置对象继承自 PretrainedConfig，并可用于控制模型输出

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 调用父类构造函数初始化配置对象
    # 定义模`
    # 参数说明
    Args:
        hidden_size (`int`, *optional*, defaults to 1408):
            # 编码器层和池化层的维度。
        intermediate_size (`int`, *optional*, defaults to 6144):
            # Transformer 编码器中“中间”层（即前馈层）的维度。
        num_hidden_layers (`int`, *optional*, defaults to 39):
            # Transformer 编码器中的隐藏层数量。
        num_attention_heads (`int`, *optional*, defaults to 16):
            # 每个注意力层中的注意力头数量。
        image_size (`int`, *optional*, defaults to 224):
            # 每个图像的大小（分辨率）。
        patch_size (`int`, *optional*, defaults to 14):
            # 每个 patch 的大小（分辨率）。
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            # 编码器和池化器中的非线性激活函数（函数或字符串）。如果是字符串，支持 `"gelu"`、`"relu"`、`"selu"` 和 `"gelu_new"`。
            # layer_norm_eps (`float`, *optional*, defaults to 1e-5): 层归一化层使用的 epsilon 值。
        attention_dropout (`float`, *optional*, defaults to 0.0):
            # 注意力概率的丢弃比率。
        initializer_range (`float`, *optional*, defaults to 0.02):
            # 用于初始化所有权重矩阵的 truncated_normal_initializer 的标准差。
        qkv_bias (`bool`, *optional*, defaults to `True`):
            # 是否在自注意力层中的查询和值添加偏置。

    Example:
    ```
    >>> from transformers import Blip2VisionConfig, Blip2VisionModel

    >>> # 初始化 Blip2VisionConfig，使用 Salesforce/blip2-opt-2.7b 样式配置
    >>> configuration = Blip2VisionConfig()

    >>> # 从 Salesforce/blip2-opt-2.7b 样式配置初始化 Blip2VisionModel（使用随机权重）
    >>> model = Blip2VisionModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```"""
    model_type = "blip_2_vision_model"

    def __init__(
        self,
        hidden_size=1408,  # 编码器层和池化层的维度，默认为 1408
        intermediate_size=6144,  # Transformer 编码器中前馈层的维度，默认为 6144
        num_hidden_layers=39,  # Transformer 编码器的隐藏层数量，默认为 39
        num_attention_heads=16,  # 每个注意力层的注意力头数量，默认为 16
        image_size=224,  # 每个图像的大小（分辨率），默认为 224
        patch_size=14,  # 每个 patch 的大小（分辨率），默认为 14
        hidden_act="gelu",  # 编码器和池化器的非线性激活函数，默认为 "gelu"
        layer_norm_eps=1e-6,  # 层归一化层使用的 epsilon 值，默认为 1e-6
        attention_dropout=0.0,  # 注意力概率的丢弃比率，默认为 0.0
        initializer_range=1e-10,  # 初始化所有权重矩阵的 truncated_normal_initializer 的标准差，默认为 1e-10
        qkv_bias=True,  # 是否在自注意力层的查询和值添加偏置，默认为 True
        **kwargs,  # 接受其他关键字参数
    ):
        # 调用父类的构造方法，传递所有关键字参数
        super().__init__(**kwargs)

        # 设置隐藏层大小
        self.hidden_size = hidden_size
        # 设置中间层大小
        self.intermediate_size = intermediate_size
        # 设置隐藏层数量
        self.num_hidden_layers = num_hidden_layers
        # 设置注意力头的数量
        self.num_attention_heads = num_attention_heads
        # 设置补丁大小
        self.patch_size = patch_size
        # 设置图像大小
        self.image_size = image_size
        # 设置初始化范围
        self.initializer_range = initializer_range
        # 设置注意力机制的dropout率
        self.attention_dropout = attention_dropout
        # 设置层归一化的epsilon值
        self.layer_norm_eps = layer_norm_eps
        # 设置隐藏层激活函数
        self.hidden_act = hidden_act
        # 设置注意力机制中的QKV偏置
        self.qkv_bias = qkv_bias

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 在关键字参数中设置token
        cls._set_token_in_kwargs(kwargs)

        # 获取预训练模型的配置字典和更新后的关键字参数
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典指定的模型类型是"blip-2"，则使用视觉配置字典
        if config_dict.get("model_type") == "blip-2":
            config_dict = config_dict["vision_config"]

        # 如果配置字典中存在"model_type"且与当前类的模型类型不同，发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从配置字典和关键字参数中创建类的实例
        return cls.from_dict(config_dict, **kwargs)
# 定义 Blip2QFormerConfig 类，继承自 PretrainedConfig 类
class Blip2QFormerConfig(PretrainedConfig):
    r"""
    这是配置类，用于存储 [`Blip2QFormerModel`] 的配置信息。它被用来根据指定的参数实例化一个 BLIP-2 Querying Transformer (Q-Former) 模型，
    定义模型的架构。使用默认参数实例化一个配置对象会产生与 BLIP-2 [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b) 架构
    类似的配置。配置对象继承自 [`PretrainedConfig`]，可以用来控制模型的输出。阅读 [`PretrainedConfig`] 的文档获取更多信息。

    注意，[`Blip2QFormerModel`] 与 [`BertLMHeadModel`] 非常相似，具有交错的跨注意力机制。
    ```
    # 定义 Q-Former 模型的配置类，包括模型的各种参数设置
    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Q-Former 模型的词汇表大小，定义了在调用模型时 `inputs_ids` 可以表示的不同标记数量。
        hidden_size (`int`, *optional*, defaults to 768):
            编码器层和池化层的维度大小。
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Transformer 编码器中隐藏层的数量。
        num_attention_heads (`int`, *optional*, defaults to 12):
            Transformer 编码器中每个注意力层的注意力头数量。
        intermediate_size (`int`, *optional*, defaults to 3072):
            Transformer 编码器中“中间”（常称为前馈）层的维度。
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            编码器和池化器中的非线性激活函数（函数或字符串）。支持的字符串有："gelu"、"relu"、"silu" 和 "gelu_new"。
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            嵌入层、编码器和池化器中所有全连接层的 dropout 概率。
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            注意力概率的 dropout 比率。
        max_position_embeddings (`int`, *optional*, defaults to 512):
            模型可能使用的最大序列长度。通常设置为一个较大的值（例如 512、1024 或 2048）。
        initializer_range (`float`, *optional*, defaults to 0.02):
            初始化所有权重矩阵的截断正态分布的标准差。
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            层归一化层使用的 epsilon 值。
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            位置嵌入的类型。选择 `"absolute"`、`"relative_key"` 或 `"relative_key_query"` 之一。关于 `"relative_key"` 的更多信息，请参考
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155)。
            关于 `"relative_key_query"` 的更多信息，请参考
            [Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658) 中的 *Method 4*。
        cross_attention_frequency (`int`, *optional*, defaults to 2):
            在 Transformer 层中添加跨注意力的频率。
        encoder_hidden_size (`int`, *optional*, defaults to 1408):
            用于跨注意力的隐藏状态的隐藏大小。

    Examples:

    ```
    >>> from transformers import Blip2QFormerConfig, Blip2QFormerModel
    >>> # 设置模型类型为 "blip_2_qformer"
    model_type = "blip_2_qformer"

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
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        cross_attention_frequency=2,
        encoder_hidden_size=1408,
        **kwargs,
    ):
        # 调用父类的构造函数，初始化模型配置
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        # 初始化模型配置的各个参数
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.cross_attention_frequency = cross_attention_frequency
        self.encoder_hidden_size = encoder_hidden_size

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 设置关键字参数中的 token_id
        cls._set_token_in_kwargs(kwargs)

        # 获取预训练模型的配置字典及其他参数
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典中指定了模型类型为 "blip-2"，则使用其 qformer_config 配置
        if config_dict.get("model_type") == "blip-2":
            config_dict = config_dict["qformer_config"]

        # 检查模型类型是否匹配，如果不匹配则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从配置字典创建一个实例
        return cls.from_dict(config_dict, **kwargs)
# 定义 Blip2Config 类，用于存储 Blip2ForConditionalGeneration 模型的配置信息
class Blip2Config(PretrainedConfig):
    # Blip2Config 是用于存储 Blip2ForConditionalGeneration 模型配置的类。它用于根据指定参数实例化 BLIP-2 模型，
    # 定义了视觉模型、Q-Former 模型和语言模型的配置。使用默认配置实例化将得到与 BLIP-2 架构
    # Salesforce/blip2-opt-2.7b 相似的配置。

    # 配置对象继承自 PretrainedConfig，可用于控制模型输出。有关更多信息，请阅读 PretrainedConfig 的文档。

    # Args:
    #     vision_config (`dict`, *optional*):
    #         用于初始化 Blip2VisionConfig 的配置选项字典。
    #     qformer_config (`dict`, *optional*):
    #         用于初始化 Blip2QFormerConfig 的配置选项字典。
    #     text_config (`dict`, *optional*):
    #         用于初始化任何 PretrainedConfig 的配置选项字典。
    #     num_query_tokens (`int`, *optional*, 默认为 32):
    #         通过 Transformer 传递的查询令牌数量。
    # 
    #     kwargs (*optional*):
    #         关键字参数字典。

    # 示例:
    # 
    # ```
    # >>> from transformers import (
    # ...     Blip2VisionConfig,
    # ...     Blip2QFormerConfig,
    # ...     OPTConfig,
    # ...     Blip2Config,
    # ...     Blip2ForConditionalGeneration,
    # ... )
    # 
    # >>> # 使用 Salesforce/blip2-opt-2.7b 风格配置初始化 Blip2Config
    # >>> configuration = Blip2Config()
    # 
    # >>> # 使用 Salesforce/blip2-opt-2.7b 风格配置初始化 Blip2ForConditionalGeneration（随机权重）
    # >>> model = Blip2ForConditionalGeneration(configuration)
    # 
    # >>> # 访问模型配置
    # >>> configuration = model.config
    # 
    # >>> # 也可以从 Blip2VisionConfig、Blip2QFormerConfig 和任何 PretrainedConfig 初始化 Blip2Config
    # 
    # >>> # 初始化 BLIP-2 视觉、BLIP-2 Q-Former 和语言模型配置
    # >>> vision_config = Blip2VisionConfig()
    # >>> qformer_config = Blip2QFormerConfig()
    # >>> text_config = OPTConfig()
    # 
    # >>> config = Blip2Config.from_text_vision_configs(vision_config, qformer_config, text_config)
    # ```
    model_type = "blip-2"
    # 定义模型类型为 "blip-2"
    def __init__(self, vision_config=None, qformer_config=None, text_config=None, num_query_tokens=32, **kwargs):
        super().__init__(**kwargs)  # 调用父类的构造函数，传递任意额外的关键字参数

        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. initializing the Blip2VisionConfig with default values.")
        
        if qformer_config is None:
            qformer_config = {}
            logger.info("qformer_config is None. Initializing the Blip2QFormerConfig with default values.")

        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the text config with default values (`OPTConfig`).")

        self.vision_config = Blip2VisionConfig(**vision_config)  # 根据提供的视觉配置初始化 Blip2VisionConfig 对象
        self.qformer_config = Blip2QFormerConfig(**qformer_config)  # 根据提供的 Q-Former 配置初始化 Blip2QFormerConfig 对象
        text_model_type = text_config["model_type"] if "model_type" in text_config else "opt"
        self.text_config = CONFIG_MAPPING[text_model_type](**text_config)  # 根据文本配置的模型类型选择相应的配置对象进行初始化

        self.tie_word_embeddings = self.text_config.tie_word_embeddings  # 设置是否共享词嵌入参数
        self.is_encoder_decoder = self.text_config.is_encoder_decoder  # 设置是否为编码器-解码器模型

        self.num_query_tokens = num_query_tokens  # 设置查询令牌的数量
        self.qformer_config.encoder_hidden_size = self.vision_config.hidden_size  # 将视觉配置的隐藏大小设置为 Q-Former 配置的编码器隐藏大小
        self.use_decoder_only_language_model = self.text_config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES  # 设置是否仅使用解码器语言模型
        self.initializer_factor = 1.0  # 初始化因子设为1.0
        self.initializer_range = 0.02  # 初始化范围设为0.02

    @classmethod
    def from_vision_qformer_text_configs(
        cls,
        vision_config: Blip2VisionConfig,
        qformer_config: Blip2QFormerConfig,
        text_config: PretrainedConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`Blip2Config`] (or a derived class) from a BLIP-2 vision model, Q-Former and language model
        configurations.

        Returns:
            [`Blip2Config`]: An instance of a configuration object
        """

        return cls(
            vision_config=vision_config.to_dict(),  # 将视觉配置对象转换为字典形式传递给构造函数
            qformer_config=qformer_config.to_dict(),  # 将 Q-Former 配置对象转换为字典形式传递给构造函数
            text_config=text_config.to_dict(),  # 将语言模型配置对象转换为字典形式传递给构造函数
            **kwargs,  # 传递额外的关键字参数
        )
```