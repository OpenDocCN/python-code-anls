# `.\models\clipseg\configuration_clipseg.py`

```py
# coding=utf-8
# 上面这行声明了文件的编码格式为 UTF-8，确保可以正确处理中文等特殊字符
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
# 版权声明，版权归 HuggingFace Inc. 团队所有，保留所有权利
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 授权协议声明，使用 Apache License, Version 2.0，允许在符合许可的情况下使用该文件
# you may not use this file except in compliance with the License.
# 除非符合许可，否则不得使用本文件
# You may obtain a copy of the License at
# 可以通过上述链接获取许可协议的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非法律要求或书面同意，本软件按"原样"分发，不附带任何形式的担保或条件
# See the License for the specific language governing permissions and
# limitations under the License.
# 查看许可协议以了解具体的语言控制权限和限制
""" CLIPSeg model configuration"""
# 说明这是 CLIPSeg 模型的配置文件

import os
# 导入操作系统相关功能
from typing import Union
# 导入 Union 类型，用于类型注解

from ...configuration_utils import PretrainedConfig
# 导入 PretrainedConfig 类，用于模型预训练配置
from ...utils import logging
# 导入 logging 工具，用于日志记录

logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器

CLIPSEG_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "CIDAS/clipseg-rd64": "https://huggingface.co/CIDAS/clipseg-rd64/resolve/main/config.json",
}
# 定义一个预训练模型配置文件映射，包含模型名称和其对应的配置文件链接

class CLIPSegTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CLIPSegModel`]. It is used to instantiate an
    CLIPSeg model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the CLIPSeg
    [CIDAS/clipseg-rd64](https://huggingface.co/CIDAS/clipseg-rd64) architecture.
    """
    # CLIPSegTextConfig 类，用于存储 CLIPSegModel 的配置信息

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # 配置对象继承自 PretrainedConfig，并可用于控制模型输出。详细信息请阅读 PretrainedConfig 的文档。
    # 定义 CLIPSeg 文本模型的配置类，设置各种参数的默认值
    Args:
        vocab_size (`int`, *optional*, defaults to 49408):
            CLIPSeg 文本模型的词汇表大小，定义了在调用 `CLIPSegModel` 时 `inputs_ids` 可表示的不同标记数量。
        hidden_size (`int`, *optional*, defaults to 512):
            编码器层和池化层的维度。
        intermediate_size (`int`, *optional*, defaults to 2048):
            Transformer 编码器中“中间”（即前馈）层的维度。
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Transformer 编码器中的隐藏层数量。
        num_attention_heads (`int`, *optional*, defaults to 8):
            Transformer 编码器中每个注意力层的注意头数量。
        max_position_embeddings (`int`, *optional*, defaults to 77):
            可能用于该模型的最大序列长度。通常设置为较大的值（例如 512、1024 或 2048）。
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            编码器和池化层中的非线性激活函数（函数或字符串）。如果是字符串，支持 `"gelu"`, `"relu"`, `"selu"` 和 `"gelu_new"` 
            `"quick_gelu"`。
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            层归一化层使用的 epsilon。
        attention_dropout (`float`, *optional*, defaults to 0.0):
            注意力概率的 dropout 比率。
        initializer_range (`float`, *optional*, defaults to 0.02):
            初始化所有权重矩阵的截断正态初始化器的标准差。
        initializer_factor (`float`, *optional*, defaults to 1.0):
            初始化所有权重矩阵的因子（内部用于初始化测试应保持为 1）。
        pad_token_id (`int`, *optional*, defaults to 1):
            填充标记 id。
        bos_token_id (`int`, *optional*, defaults to 49406):
            流的开始标记 id。
        eos_token_id (`int`, *optional*, defaults to 49407):
            流的结束标记 id。

    Example:

    ```
    >>> from transformers import CLIPSegTextConfig, CLIPSegTextModel

    >>> # 使用 CIDAS/clipseg-rd64 风格配置初始化 CLIPSegTextConfig
    >>> configuration = CLIPSegTextConfig()

    >>> # 使用 CIDAS/clipseg-rd64 风格配置初始化随机权重的 CLIPSegTextModel
    >>> model = CLIPSegTextModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```
    
    # 模型类型设置为 "clipseg_text_model"
    model_type = "clipseg_text_model"
    # 初始化函数，设置模型配置参数
    def __init__(
        self,
        vocab_size=49408,                      # 词汇表大小，默认为 49408
        hidden_size=512,                       # 隐藏层大小，默认为 512
        intermediate_size=2048,                # 中间层大小，默认为 2048
        num_hidden_layers=12,                  # 隐藏层数，默认为 12
        num_attention_heads=8,                 # 注意力头数，默认为 8
        max_position_embeddings=77,            # 最大位置嵌入长度，默认为 77
        hidden_act="quick_gelu",               # 隐藏层激活函数，默认为 quick_gelu
        layer_norm_eps=1e-5,                   # 层归一化 epsilon，默认为 1e-5
        attention_dropout=0.0,                 # 注意力机制的 dropout 率，默认为 0.0
        initializer_range=0.02,                # 初始化范围，默认为 0.02
        initializer_factor=1.0,                # 初始化因子，默认为 1.0
        pad_token_id=1,                        # 填充标记的 ID，默认为 1
        bos_token_id=49406,                    # 开始标记的 ID，默认为 49406
        eos_token_id=49407,                    # 结束标记的 ID，默认为 49407
        **kwargs,
    ):
        # 调用父类构造函数，传入填充、开始和结束标记的 ID，以及其他关键字参数
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 设置模型配置参数
        self.vocab_size = vocab_size                # 设置词汇表大小
        self.hidden_size = hidden_size              # 设置隐藏层大小
        self.intermediate_size = intermediate_size  # 设置中间层大小
        self.num_hidden_layers = num_hidden_layers  # 设置隐藏层数
        self.num_attention_heads = num_attention_heads  # 设置注意力头数
        self.max_position_embeddings = max_position_embeddings  # 设置最大位置嵌入长度
        self.layer_norm_eps = layer_norm_eps        # 设置层归一化 epsilon
        self.hidden_act = hidden_act                # 设置隐藏层激活函数
        self.initializer_range = initializer_range  # 设置初始化范围
        self.initializer_factor = initializer_factor  # 设置初始化因子
        self.attention_dropout = attention_dropout  # 设置注意力机制的 dropout 率

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 设置 token 参数到 kwargs 中
        cls._set_token_in_kwargs(kwargs)

        # 获取配置字典和剩余的 kwargs
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果从 CLIPSegConfig 加载，获取文本配置字典
        if config_dict.get("model_type") == "clipseg":
            config_dict = config_dict["text_config"]

        # 如果模型类型不匹配且不是所有配置的模型都支持的情况下，给出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从配置字典创建模型实例并返回
        return cls.from_dict(config_dict, **kwargs)
# CLIPSegVisionConfig 是一个配置类，用于存储 CLIPSegModel 的配置信息。
# 这个配置类定义了 CLIPSeg 模型的架构，根据指定的参数实例化一个模型。
# 当使用默认参数实例化时，会生成与 CIDAS/clipseg-rd64 架构类似的配置。
# 配置对象继承自 PretrainedConfig，可以用来控制模型的输出。详细信息请参阅 PretrainedConfig 的文档。

class CLIPSegVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CLIPSegModel`]. It is used to instantiate an
    CLIPSeg model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the CLIPSeg
    [CIDAS/clipseg-rd64](https://huggingface.co/CIDAS/clipseg-rd64) architecture.

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
    >>> from transformers import CLIPSegVisionConfig, CLIPSegVisionModel

    >>> # Initializing a CLIPSegVisionConfig with CIDAS/clipseg-rd64 style configuration
    >>> configuration = CLIPSegVisionConfig()

    >>> # Initializing a CLIPSegVisionModel (with random weights) from the CIDAS/clipseg-rd64 style configuration
    >>> model = CLIPSegVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    # 模型类型的标识符，用于指示这是 CLIPSeg 视觉模型的配置
    model_type = "clipseg_vision_model"
    # 初始化方法，设置模型配置参数
    def __init__(
        self,
        hidden_size=768,                 # 隐藏层大小，默认为768
        intermediate_size=3072,          # 中间层大小，默认为3072
        num_hidden_layers=12,            # 隐藏层数，默认为12
        num_attention_heads=12,          # 注意力头数，默认为12
        num_channels=3,                  # 图像通道数，默认为3
        image_size=224,                  # 图像大小，默认为224
        patch_size=32,                   # 图像分块大小，默认为32
        hidden_act="quick_gelu",         # 隐藏层激活函数，默认为"quick_gelu"
        layer_norm_eps=1e-5,             # Layer Normalization 的 epsilon，默认为1e-5
        attention_dropout=0.0,           # 注意力机制的dropout率，默认为0.0
        initializer_range=0.02,          # 初始化权重范围，默认为0.02
        initializer_factor=1.0,          # 初始化因子，默认为1.0
        **kwargs,
    ):
        super().__init__(**kwargs)       # 调用父类的初始化方法，并传入其他参数

        self.hidden_size = hidden_size   # 设置隐藏层大小
        self.intermediate_size = intermediate_size   # 设置中间层大小
        self.num_hidden_layers = num_hidden_layers   # 设置隐藏层数
        self.num_attention_heads = num_attention_heads   # 设置注意力头数
        self.num_channels = num_channels   # 设置图像通道数
        self.patch_size = patch_size     # 设置图像分块大小
        self.image_size = image_size     # 设置图像大小
        self.initializer_range = initializer_range   # 设置初始化权重范围
        self.initializer_factor = initializer_factor   # 设置初始化因子
        self.attention_dropout = attention_dropout   # 设置注意力dropout率
        self.layer_norm_eps = layer_norm_eps   # 设置Layer Normalization的epsilon
        self.hidden_act = hidden_act       # 设置隐藏层激活函数

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)   # 调用类方法，设置kwargs中的token参数

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)   # 调用类方法获取配置字典和更新后的kwargs

        # 如果配置字典中的模型类型是"clipseg"，则使用其视觉配置
        if config_dict.get("model_type") == "clipseg":
            config_dict = config_dict["vision_config"]

        # 如果配置字典中包含"model_type"键，并且当前类的model_type属性存在且不同于配置字典中的类型，发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 根据配置字典和kwargs创建配置对象
        return cls.from_dict(config_dict, **kwargs)
# 定义 `CLIPSegConfig` 类，继承自 `PretrainedConfig` 类，用于存储 `CLIPSegModel` 的配置信息。
# 该类用于实例化 CLIPSeg 模型，根据指定的参数定义文本模型和视觉模型的配置。
# 使用默认参数实例化配置对象将产生与 `CIDAS/clipseg-rd64` 架构相似的配置。
class CLIPSegConfig(PretrainedConfig):
    r"""
    [`CLIPSegConfig`] is the configuration class to store the configuration of a [`CLIPSegModel`]. It is used to
    instantiate a CLIPSeg model according to the specified arguments, defining the text model and vision model configs.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the CLIPSeg
    [CIDAS/clipseg-rd64](https://huggingface.co/CIDAS/clipseg-rd64) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`CLIPSegTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`CLIPSegVisionConfig`].
        projection_dim (`int`, *optional*, defaults to 512):
            Dimensionality of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* paramter. Default is used as per the original CLIPSeg implementation.
        extract_layers (`List[int]`, *optional*, defaults to `[3, 6, 9]`):
            Layers to extract when forwarding the query image through the frozen visual backbone of CLIP.
        reduce_dim (`int`, *optional*, defaults to 64):
            Dimensionality to reduce the CLIP vision embedding.
        decoder_num_attention_heads (`int`, *optional*, defaults to 4):
            Number of attention heads in the decoder of CLIPSeg.
        decoder_attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        decoder_hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        decoder_intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layers in the Transformer decoder.
        conditional_layer (`int`, *optional*, defaults to 0):
            The layer to use of the Transformer encoder whose activations will be combined with the condition
            embeddings using FiLM (Feature-wise Linear Modulation). If 0, the last layer is used.
        use_complex_transposed_convolution (`bool`, *optional*, defaults to `False`):
            Whether to use a more complex transposed convolution in the decoder, enabling more fine-grained
            segmentation.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```
    >>> from transformers import CLIPSegConfig, CLIPSegModel
    # 初始化一个 CLIPSegConfig，使用 CIDAS/clipseg-rd64 风格的配置
    >>> configuration = CLIPSegConfig()

    # 使用 CIDAS/clipseg-rd64 风格的配置初始化一个 CLIPSegModel（带有随机权重）
    >>> model = CLIPSegModel(configuration)

    # 访问模型的配置信息
    >>> configuration = model.config

    # 我们也可以从 CLIPSegTextConfig 和 CLIPSegVisionConfig 初始化一个 CLIPSegConfig

    # 初始化一个 CLIPSegTextConfig 和 CLIPSegVisionConfig
    >>> config_text = CLIPSegTextConfig()
    >>> config_vision = CLIPSegVisionConfig()

    # 使用 CLIPSegTextConfig 和 CLIPSegVisionConfig 初始化一个 CLIPSegConfig 对象
    >>> config = CLIPSegConfig.from_text_vision_configs(config_text, config_vision)
```