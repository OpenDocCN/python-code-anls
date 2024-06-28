# `.\models\kosmos2\configuration_kosmos2.py`

```py
# coding=utf-8
# 设置文件编码为UTF-8，确保支持多语言字符集

# 版权声明及许可证信息
# 版权所有 2023 Microsoft Research 和 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件
# 没有任何明示或暗示的保证或条件。
# 有关详细信息，请参阅许可证。

""" KOSMOS-2 模型配置"""

import os
from typing import Union

# 导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 导入日志记录工具
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练配置存档映射字典，映射模型名称到其配置文件的下载链接
KOSMOS2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/kosmos-2-patch14-224": (
        "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/config.json"
    ),
    # 查看所有 KOSMOS-2 模型的列表：https://huggingface.co/models?filter=kosmos-2
}


class Kosmos2TextConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`Kosmos2TextModel`] 的配置信息。根据指定的参数实例化 KOSMOS-2 文本解码器，
    定义模型架构。使用默认参数实例化配置对象将产生类似于 KOSMOS-2 文本解码器
    [microsoft/kosmos-2-patch14-224](https://huggingface.co/microsoft/kosmos-2-patch14-224) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。
    ```
    # 定义 Kosmos2 模型的参数和默认值

    # 模型的类型，用于标识 Kosmos2 文本模型
    model_type = "kosmos_2_text_model"

    # 推断阶段需要忽略的键列表，这些键不会在推断时使用
    keys_to_ignore_at_inference = ["past_key_values"]

    # 属性映射字典，将模型参数的名称映射到 Kosmos2 模型期望的名称
    attribute_map = {
        "num_attention_heads": "attention_heads",  # 注意力头的数量
        "hidden_size": "embed_dim",  # 隐藏层的维度
        "num_hidden_layers": "layers",  # Transformer 编码器中的隐藏层数量
    }
    # 初始化函数，用于创建一个新的配置对象
    def __init__(
        self,
        vocab_size=65037,                    # 词汇表大小，默认为65037
        max_position_embeddings=2048,        # 最大位置嵌入数量，默认为2048
        embed_dim=2048,                      # 嵌入维度，默认为2048
        layers=24,                           # 层数，默认为24
        ffn_dim=8192,                        # 前馈神经网络维度，默认为8192
        attention_heads=32,                  # 注意力头数，默认为32
        activation_function="gelu",          # 激活函数，默认为"gelu"
        dropout=0.1,                         # 普通层级dropout概率，默认为0.1
        attention_dropout=0.1,               # 注意力模块dropout概率，默认为0.1
        activation_dropout=0.0,              # 激活函数dropout概率，默认为0.0
        layerdrop=0.0,                       # 层级dropout概率，默认为0.0
        layer_norm_eps=1e-5,                 # 层归一化的epsilon，默认为1e-5
        init_std=0.02,                       # 初始化标准差，默认为0.02
        scale_embedding=True,                # 是否缩放嵌入，默认为True
        use_cache=True,                      # 是否使用缓存，默认为True
        pad_token_id=1,                      # 填充标记ID，默认为1
        bos_token_id=0,                      # 开始序列标记ID，默认为0
        eos_token_id=2,                      # 结束序列标记ID，默认为2
        **kwargs,                            # 其他关键字参数
    ):
        # 调用父类的初始化方法，设置填充、开始、结束标记ID等参数
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        # 初始化配置对象的各个属性
        self.vocab_size = vocab_size                         # 设置词汇表大小属性
        self.max_position_embeddings = max_position_embeddings  # 设置最大位置嵌入数量属性
        self.embed_dim = embed_dim                           # 设置嵌入维度属性
        self.layers = layers                                 # 设置层数属性
        self.ffn_dim = ffn_dim                               # 设置前馈神经网络维度属性
        self.attention_heads = attention_heads               # 设置注意力头数属性
        self.activation_function = activation_function       # 设置激活函数属性
        self.dropout = dropout                               # 设置普通层级dropout概率属性
        self.attention_dropout = attention_dropout           # 设置注意力模块dropout概率属性
        self.activation_dropout = activation_dropout         # 设置激活函数dropout概率属性
        self.layerdrop = layerdrop                           # 设置层级dropout概率属性
        self.layer_norm_eps = layer_norm_eps                 # 设置层归一化的epsilon属性
        self.init_std = init_std                             # 设置初始化标准差属性
        self.scale_embedding = scale_embedding               # 设置是否缩放嵌入属性
        self.use_cache = use_cache                           # 设置是否使用缓存属性

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 将token相关参数添加到kwargs中
        cls._set_token_in_kwargs(kwargs)

        # 获取配置字典和更新后的kwargs
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果加载自Kosmos2Config，则获取文本配置字典
        if config_dict.get("model_type") == "kosmos-2":
            config_dict = config_dict["text_config"]

        # 如果配置字典中存在model_type，并且与类的model_type不同，发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从配置字典和kwargs创建类的实例
        return cls.from_dict(config_dict, **kwargs)
# 定义 `Kosmos2VisionConfig` 类，用于存储 `Kosmos2VisionModel` 的配置信息。
# 继承自 `PretrainedConfig`，用于控制模型的输出。详细信息请参考 `PretrainedConfig` 的文档。

class Kosmos2VisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Kosmos2VisionModel`]. It is used to instantiate a
    KOSMOS-2 vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the KOSMOS-2
    [microsoft/kosmos-2-patch14-224](https://huggingface.co/microsoft/kosmos-2-patch14-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
    """
    ):
        super().__init__(**kwargs)
        # 调用父类的初始化方法，传入关键字参数

        self.hidden_size = hidden_size
        # 设置隐藏层大小

        self.intermediate_size = intermediate_size
        # 设置中间层大小

        self.num_hidden_layers = num_hidden_layers
        # 设置隐藏层数量

        self.num_attention_heads = num_attention_heads
        # 设置注意力头数量

        self.num_channels = num_channels
        # 设置通道数量

        self.patch_size = patch_size
        # 设置图像块大小

        self.image_size = image_size
        # 设置图像大小

        self.initializer_range = initializer_range
        # 设置初始化范围

        self.initializer_factor = initializer_factor
        # 设置初始化因子

        self.attention_dropout = attention_dropout
        # 设置注意力丢弃率

        self.layer_norm_eps = layer_norm_eps
        # 设置层归一化的 epsilon 值

        self.hidden_act = hidden_act
        # 设置隐藏层激活函数

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)
        # 调用类方法 _set_token_in_kwargs，设置关键字参数中的 token

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        # 调用类方法 get_config_dict，获取预训练模型的配置字典和更新后的关键字参数

        # 如果从 Kosmos2Config 加载，则获取视觉配置字典
        if config_dict.get("model_type") == "kosmos-2":
            config_dict = config_dict["vision_config"]

        # 如果配置字典中存在 "model_type" 并且类具有 "model_type" 属性，并且它们不相同，发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从配置字典创建类的实例，并返回
        return cls.from_dict(config_dict, **kwargs)
class Kosmos2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Kosmos2Model`]. It is used to instantiate a
    KOSMOS-2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the KOSMOS-2
    [microsoft/kosmos-2-patch14-224](https://huggingface.co/microsoft/kosmos-2-patch14-224) architecture.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Kosmos2TextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Kosmos2VisionConfig`].
        latent_query_num (`int`, *optional*, defaults to 64):
            The number of latent query tokens that represent the image features used in the text decoder component.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```
    >>> from transformers import Kosmos2Config, Kosmos2Model

    >>> # Initializing a Kosmos-2 kosmos-2-patch14-224 style configuration
    >>> configuration = Kosmos2Config()

    >>> # Initializing a model (with random weights) from the kosmos-2-patch14-224 style configuration
    >>> model = Kosmos2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    # 设定模型类型为 "kosmos-2"
    model_type = "kosmos-2"
    # 标志这个配置类是由多个部分组成
    is_composition = True

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        latent_query_num=64,
        **kwargs,
    ):
        # 调用父类构造函数，传入所有额外的关键字参数
        super().__init__(**kwargs)

        # 如果文本配置为空，使用默认空字典并记录日志
        if text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `Kosmos2TextConfig` with default values.")

        # 如果视觉配置为空，使用默认空字典并记录日志
        if vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. Initializing the `Kosmos2VisionConfig` with default values.")

        # 根据传入的文本配置初始化 `Kosmos2TextConfig` 对象
        self.text_config = Kosmos2TextConfig(**text_config)
        # 根据传入的视觉配置初始化 `Kosmos2VisionConfig` 对象
        self.vision_config = Kosmos2VisionConfig(**vision_config)

        # 设置 latent_query_num 属性，表示在文本解码器组件中用于表示图像特征的潜在查询标记数目
        self.latent_query_num = latent_query_num
```