# `.\transformers\models\clipseg\configuration_clipseg.py`

```py
# coding=utf-8
# 代码文件的编码声明，指定为 UTF-8 编码
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
# 版权声明，版权归 HuggingFace Inc. 团队所有，保留所有权利
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
""" CLIPSeg model configuration"""
# 导入所需的模块和类型
import os
from typing import Union

# 从配置工具模块中导入 PretrainedConfig 类
from ...configuration_utils import PretrainedConfig
# 从工具模块中导入日志记录器
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练模型配置文件的映射字典，映射模型名称到配置文件的 URL
CLIPSEG_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "CIDAS/clipseg-rd64": "https://huggingface.co/CIDAS/clipseg-rd64/resolve/main/config.json",
}

# CLIPSegTextConfig 类，继承自 PretrainedConfig 类
class CLIPSegTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CLIPSegModel`]. It is used to instantiate an
    CLIPSeg model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the CLIPSeg
    [CIDAS/clipseg-rd64](https://huggingface.co/CIDAS/clipseg-rd64) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 49408):
            Vocabulary size of the CLIPSeg text model. Defines the number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`CLIPSegModel`].
        hidden_size (`int`, *optional*, defaults to 512):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        max_position_embeddings (`int`, *optional*, defaults to 77):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
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
        pad_token_id (`int`, *optional*, defaults to 1):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 49406):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 49407):
            End of stream token id.

    Example:

    ```python
    >>> from transformers import CLIPSegTextConfig, CLIPSegTextModel

    >>> # Initializing a CLIPSegTextConfig with CIDAS/clipseg-rd64 style configuration
    >>> configuration = CLIPSegTextConfig()

    >>> # Initializing a CLIPSegTextModel (with random weights) from the CIDAS/clipseg-rd64 style configuration
    >>> model = CLIPSegTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```py"""

    # 定义模型类型为 "clipseg_text_model"
    model_type = "clipseg_text_model"
    # 初始化方法，用于创建一个新的配置对象
    def __init__(
        self,
        vocab_size=49408,  # 词汇表大小，默认为 49408
        hidden_size=512,  # 隐藏层大小，默认为 512
        intermediate_size=2048,  # 中间层大小，默认为 2048
        num_hidden_layers=12,  # 隐藏层数量，默认为 12
        num_attention_heads=8,  # 注意力头数量，默认为 8
        max_position_embeddings=77,  # 最大位置编码数量，默认为 77
        hidden_act="quick_gelu",  # 隐藏层激活函数，默认为 "quick_gelu"
        layer_norm_eps=1e-5,  # 层归一化的 epsilon，默认为 1e-5
        attention_dropout=0.0,  # 注意力头的 dropout 概率，默认为 0.0
        initializer_range=0.02,  # 初始化范围，默认为 0.02
        initializer_factor=1.0,  # 初始化因子，默认为 1.0
        pad_token_id=1,  # 填充 token 的 ID，默认为 1
        bos_token_id=49406,  # 起始 token 的 ID，默认为 49406
        eos_token_id=49407,  # 结束 token 的 ID，默认为 49407
        **kwargs,  # 其它参数，作为关键字参数传递
    ):
        # 调用父类的初始化方法，传递填充、起始和结束 token 的 ID，以及其它参数
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 设置配置对象的属性值
        self.vocab_size = vocab_size  # 词汇表大小
        self.hidden_size = hidden_size  # 隐藏层大小
        self.intermediate_size = intermediate_size  # 中间层大小
        self.num_hidden_layers = num_hidden_layers  # 隐藏层数量
        self.num_attention_heads = num_attention_heads  # 注意力头数量
        self.max_position_embeddings = max_position_embeddings  # 最大位置编码数量
        self.layer_norm_eps = layer_norm_eps  # 层归一化的 epsilon
        self.hidden_act = hidden_act  # 隐藏层激活函数
        self.initializer_range = initializer_range  # 初始化范围
        self.initializer_factor = initializer_factor  # 初始化因子
        self.attention_dropout = attention_dropout  # 注意力头的 dropout 概率

    # 从预训练模型加载配置对象的类方法
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 设置关键字参数中的 token ID
        cls._set_token_in_kwargs(kwargs)

        # 获取配置字典和更新后的关键字参数
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典中的模型类型为 "clipseg"，则获取文本配置字典
        if config_dict.get("model_type") == "clipseg":
            config_dict = config_dict["text_config"]

        # 如果配置字典中包含模型类型且与当前类的模型类型不匹配，则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 根据配置字典创建配置对象，并传入更新后的关键字参数
        return cls.from_dict(config_dict, **kwargs)
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

    ```python
    >>> from transformers import CLIPSegVisionConfig, CLIPSegVisionModel

    >>> # Initializing a CLIPSegVisionConfig with CIDAS/clipseg-rd64 style configuration
    >>> configuration = CLIPSegVisionConfig()

    >>> # Initializing a CLIPSegVisionModel (with random weights) from the CIDAS/clipseg-rd64 style configuration
    >>> model = CLIPSegVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```py"""

    # 定义类属性 `model_type` 为 "clipseg_vision_model"
    model_type = "clipseg_vision_model"
    # 初始化函数，用于创建一个新的配置对象
    def __init__(
        self,
        hidden_size=768,  # 隐藏层大小，默认为768
        intermediate_size=3072,  # 中间层大小，默认为3072
        num_hidden_layers=12,  # 隐藏层数，默认为12
        num_attention_heads=12,  # 注意力头数，默认为12
        num_channels=3,  # 图像通道数，默认为3
        image_size=224,  # 图像大小，默认为224
        patch_size=32,  # 补丁大小，默认为32
        hidden_act="quick_gelu",  # 隐藏层激活函数，默认为"quick_gelu"
        layer_norm_eps=1e-5,  # 层归一化的 epsilon，默认为1e-5
        attention_dropout=0.0,  # 注意力机制的 dropout，默认为0.0
        initializer_range=0.02,  # 初始化范围，默认为0.02
        initializer_factor=1.0,  # 初始化因子，默认为1.0
        **kwargs,  # 其他参数
    ):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 设置配置的各种参数
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
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

    # 从预训练模型加载配置
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 设置 token 到参数中
        cls._set_token_in_kwargs(kwargs)

        # 获取配置字典和参数
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果加载自 CLIPSegConfig，则获取视觉配置字典
        if config_dict.get("model_type") == "clipseg":
            config_dict = config_dict["vision_config"]

        # 如果配置字典中存在模型类型，并且当前类的模型类型与之不匹配，发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从配置字典中创建配置对象并返回
        return cls.from_dict(config_dict, **kwargs)
# CLIPSegConfig 是用来存储 CLIPSegModel 的配置信息的类
class CLIPSegConfig(PretrainedConfig):
    r"""
    [`CLIPSegConfig`] 是用来存储 [`CLIPSegModel`] 的配置信息的类。它被用来根据指定的参数实例化一个 CLIPSeg 模型，
    定义文本模型和视觉模型的配置。使用默认参数实例化一个配置对象将会产生与 CLIPSeg [CIDAS/clipseg-rd64](https://huggingface.co/CIDAS/clipseg-rd64) 架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，并且可以用来控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    Args:
        text_config (`dict`, *optional*):
            用于初始化 [`CLIPSegTextConfig`] 的配置选项字典。
        vision_config (`dict`, *optional*):
            用于初始化 [`CLIPSegVisionConfig`] 的配置选项字典。
        projection_dim (`int`, *optional*, defaults to 512):
            文本和视觉投影层的维度。
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            *logit_scale* 参数的初始值。默认值按照原始 CLIPSeg 实现使用。
        extract_layers (`List[int]`, *optional*, defaults to `[3, 6, 9]`):
            在通过 CLIP 的冻结视觉主干传递查询图像时要提取的层。
        reduce_dim (`int`, *optional*, defaults to 64):
            要减少 CLIP 视觉嵌入的维度。
        decoder_num_attention_heads (`int`, *optional*, defaults to 4):
            CLIPSeg 解码器中的注意力头数。
        decoder_attention_dropout (`float`, *optional*, defaults to 0.0):
            注意力概率的 dropout 比率。
        decoder_hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            编码器和池化器中的非线性激活函数（函数或字符串）。如果是字符串，支持 `"gelu"`, `"relu"`, `"selu"` 和 `"gelu_new"` `"quick_gelu"`。
        decoder_intermediate_size (`int`, *optional*, defaults to 2048):
            Transformer 解码器中“中间”（即前馈）层的维度。
        conditional_layer (`int`, *optional*, defaults to 0):
            将被使用的 Transformer 编码器层，其激活将使用 FiLM（Feature-wise Linear Modulation）与条件嵌入进行组合。如果为 0，则使用最后一层。
        use_complex_transposed_convolution (`bool`, *optional*, defaults to `False`):
            是否在解码器中使用更复杂的转置卷积，以实现更精细的分割。
        kwargs (*optional*):
            关键字参数字典。

    Example:

    ```python
    >>> from transformers import CLIPSegConfig, CLIPSegModel
    ```py
    # 使用CIDAS/clipseg-rd64风格的配置初始化一个CLIPSegConfig对象
    configuration = CLIPSegConfig()
    
    # 使用CIDAS/clipseg-rd64风格的配置初始化一个带有随机权重的CLIPSegModel对象
    model = CLIPSegModel(configuration)
    
    # 获取模型的配置信息
    configuration = model.config
    
    # 也可以从CLIPSegTextConfig和CLIPSegVisionConfig初始化一个CLIPSegConfig对象
    
    # 初始化CLIPSegText和CLIPSegVision配置
    config_text = CLIPSegTextConfig()
    config_vision = CLIPSegVisionConfig()
    
    # 从CLIPSegTextConfig和CLIPSegVisionConfig初始化一个CLIPSegConfig对象
    config = CLIPSegConfig.from_text_vision_configs(config_text, config_vision)
    
    # 模型类型设为"clipseg"
    model_type = "clipseg"
    
    # CLIPSegConfig类的构造函数
    def __init__(
        self,
        text_config=None,
        vision_config=None,
        projection_dim=512,
        logit_scale_init_value=2.6592,
        extract_layers=[3, 6, 9],
        reduce_dim=64,
        decoder_num_attention_heads=4,
        decoder_attention_dropout=0.0,
        decoder_hidden_act="quick_gelu",
        decoder_intermediate_size=2048,
        conditional_layer=0,
        use_complex_transposed_convolution=False,
        **kwargs,
    ):
    
    # 从CLIPSegTextConfig和CLIPSegVisionConfig对象实例化一个CLIPSegConfig对象
    @classmethod
    def from_text_vision_configs(cls, text_config: CLIPSegTextConfig, vision_config: CLIPSegVisionConfig, **kwargs):
        r"""
        从clipseg文本模型配置和clipseg视觉模型配置实例化一个CLIPSegConfig（或派生类）对象。
    
        返回：
            [`CLIPSegConfig`]: 一个配置对象的实例
        """
    
        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)
```