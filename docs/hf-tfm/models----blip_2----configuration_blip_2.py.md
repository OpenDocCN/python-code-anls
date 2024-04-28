# `.\transformers\models\blip_2\configuration_blip_2.py`

```py
# 指定编码格式为 UTF-8
# 版权声明
# 版权所有 © 2023 年 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）进行许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获得许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按“原样”分发，
# 无任何形式的明示或暗示的保证或条件。
# 请查阅许可证了解更多信息。
""" BLIP-2 模型配置"""

# 导入必要的模块
import os
from typing import Union

# 导入配置相关的模块和类
from ...configuration_utils import PretrainedConfig
from ...models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from ...utils import logging
from ..auto import CONFIG_MAPPING

# 获取日志记录器
logger = logging.get_logger(__name__)

# BLIP-2 预训练配置存档映射
BLIP_2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "salesforce/blip2-opt-2.7b": "https://huggingface.co/salesforce/blip2-opt-2.7b/resolve/main/config.json",
}


class Blip2VisionConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`Blip2VisionModel`] 的配置。它用于根据指定的参数实例化一个 BLIP-2 视觉编码器，
    定义模型架构。实例化默认配置将产生与 BLIP-2 [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b)
    架构相似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。
    Args:
        hidden_size (`int`, *optional*, defaults to 1408):
            编码器层和池化层的维度。
        intermediate_size (`int`, *optional*, defaults to 6144):
            Transformer 编码器中“中间”（即前馈）层的维度。
        num_hidden_layers (`int`, *optional*, defaults to 39):
            Transformer 编码器中的隐藏层数量。
        num_attention_heads (`int`, *optional*, defaults to 16):
            Transformer 编码器中每个注意力层的注意力头数。
        image_size (`int`, *optional*, defaults to 224):
            每个图像的大小（分辨率）。
        patch_size (`int`, *optional*, defaults to 14):
            每个补丁的大小（分辨率）。
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            编码器和池化器中的非线性激活函数（函数或字符串）。如果是字符串，则支持 `"gelu"`、`"relu"`、`"selu"` 和 `"gelu_new"`。
        layer_norm_eps (`float`, *optional*, defaults to 1e-5): 
            层标准化层使用的 epsilon。
        attention_dropout (`float`, *optional*, defaults to 0.0):
            注意力概率的 dropout 比率。
        initializer_range (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态初始化器的标准差。
        qkv_bias (`bool`, *optional*, defaults to `True`):
            是否向自注意力层的查询和值中添加偏置。

    Example:

    ```python
    >>> from transformers import Blip2VisionConfig, Blip2VisionModel

    >>> # 使用 Salesforce/blip2-opt-2.7b 风格的配置初始化 Blip2VisionConfig
    >>> configuration = Blip2VisionConfig()

    >>> # 使用 Salesforce/blip2-opt-2.7b 风格的配置初始化 Blip2VisionModel（带有随机权重）
    >>> model = Blip2VisionModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py

    model_type = "blip_2_vision_model"

    def __init__(
        self,
        hidden_size=1408,
        intermediate_size=6144,
        num_hidden_layers=39,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        initializer_range=1e-10,
        qkv_bias=True,
        **kwargs,
    # 初始化方法，接受参数并调用父类初始化方法
    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        patch_size: int = 16,
        image_size: int = 224,
        initializer_range: float = 0.02,
        attention_dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        hidden_act: str = "gelu",
        qkv_bias: bool = False,
        **kwargs
    ):
        # 调用父类的初始化方法，传入额外的参数
        super().__init__(**kwargs)

        # 设置类的属性值为传入的参数值
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.qkv_bias = qkv_bias

    # 从预训练模型中加载配置信息
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 在传入参数中设置 token，将其从 kwargs 中剔除
        cls._set_token_in_kwargs(kwargs)

        # 获取配置字典和剩余的 kwargs 参数
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典中的模型类型是 "blip-2"，则获取视觉配置字典
        if config_dict.get("model_type") == "blip-2":
            config_dict = config_dict["vision_config"]

        # 如果配置字典中包含 "model_type" 并且当前类的模型类型不匹配，则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从配置字典和剩余的 kwargs 参数中创建配置对象并返回
        return cls.from_dict(config_dict, **kwargs)
# 定义 Blip2QFormerModel 的配置类，用于存储 BLIP-2 Querying Transformer 模型的配置信息
class Blip2QFormerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Blip2QFormerModel`]. It is used to instantiate a
    BLIP-2 Querying Transformer (Q-Former) model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the BLIP-2
    [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b) architecture. Configuration objects
    inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the documentation from
    [`PretrainedConfig`] for more information.

    Note that [`Blip2QFormerModel`] is very similar to [`BertLMHeadModel`] with interleaved cross-attention.
    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Q-Former 模型的词汇表大小。定义了在调用模型时可以表示的不同标记的数量。
        hidden_size (`int`, *optional*, defaults to 768):
            编码器层和池化层的维度。
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Transformer 编码器中隐藏层的数量。
        num_attention_heads (`int`, *optional*, defaults to 12):
            Transformer 编码器中每个注意力层的注意力头数。
        intermediate_size (`int`, *optional*, defaults to 3072):
            Transformer 编码器中“中间”（通常称为前馈）层的维度。
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            编码器和池化器中的非线性激活函数（函数或字符串）。如果是字符串，支持 `"gelu"`, `"relu"`, `"silu"` 和 `"gelu_new"`。
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            嵌入层、编码器和池化器中所有全连接层的丢弃概率。
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            注意力概率的丢弃比率。
        max_position_embeddings (`int`, *optional*, defaults to 512):
            模型可能使用的最大序列长度。通常将其设置为较大的值（例如 512、1024 或 2048）以防万一。
        initializer_range (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态初始化器的标准差。
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            层归一化层使用的 epsilon。
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            位置嵌入的类型。选择 `"absolute"`、`"relative_key"` 或 `"relative_key_query"` 之一。有关 `"relative_key"` 的更多信息，请参阅 [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155)。有关 `"relative_key_query"` 的更多信息，请参阅 [Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658) 中的 *Method 4*。
        cross_attention_frequency (`int`, *optional*, defaults to 2):
            向 Transformer 层添加交叉注意力的频率。
        encoder_hidden_size (`int`, *optional*, defaults to 1408):
            用于交叉注意力的隐藏状态的隐藏大小。

    Examples:

    ```python
    >>> from transformers import Blip2QFormerConfig, Blip2QFormerModel
    你的代码已经全部注释好了。
# 定义 Blip2Config 类，用于存储 Blip2ForConditionalGeneration 模型的配置信息
class Blip2Config(PretrainedConfig):
    r"""
    [`Blip2Config`] 是用于存储 [`Blip2ForConditionalGeneration`] 配置的类。它用于根据指定的参数实例化 BLIP-2 模型，
    定义了视觉模型、Q-Former 模型和语言模型的配置。使用默认配置实例化一个配置对象将产生类似于 BLIP-2 [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    Args:
        vision_config (`dict`, *optional*):
            用于初始化 [`Blip2VisionConfig`] 的配置选项字典。
        qformer_config (`dict`, *optional*):
            用于初始化 [`Blip2QFormerConfig`] 的配置选项字典。
        text_config (`dict`, *optional*):
            用于初始化任何 [`PretrainedConfig`] 的配置选项字典。
        num_query_tokens (`int`, *optional*, defaults to 32):
            通过 Transformer 传递的查询令牌数量。

        kwargs (*optional*):
            关键字参数字典。

    Example:

    ```py
    >>> from transformers import (
    ...     Blip2VisionConfig,
    ...     Blip2QFormerConfig,
    ...     OPTConfig,
    ...     Blip2Config,
    ...     Blip2ForConditionalGeneration,
    ... )

    >>> # 使用 Salesforce/blip2-opt-2.7b 风格配置初始化 Blip2Config
    >>> configuration = Blip2Config()

    >>> # 使用 Salesforce/blip2-opt-2.7b 风格配置初始化 Blip2ForConditionalGeneration（带有随机权重）
    >>> model = Blip2ForConditionalGeneration(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config

    >>> # 我们还可以从 Blip2VisionConfig、Blip2QFormerConfig 和任何 PretrainedConfig 初始化 Blip2Config

    >>> # 初始化 BLIP-2 视觉、BLIP-2 Q-Former 和语言模型配置
    >>> vision_config = Blip2VisionConfig()
    >>> qformer_config = Blip2QFormerConfig()
    >>> text_config = OPTConfig()

    >>> config = Blip2Config.from_text_vision_configs(vision_config, qformer_config, text_config)
    ```"""

    # 模型类型为 "blip-2"
    model_type = "blip-2"
    def __init__(self, vision_config=None, qformer_config=None, text_config=None, num_query_tokens=32, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 如果视觉配置为空，则使用默认配置，并记录日志
        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. initializing the Blip2VisionConfig with default values.")

        # 如果 Q-Former 配置为空，则使用默认配置，并记录日志
        if qformer_config is None:
            qformer_config = {}
            logger.info("qformer_config is None. Initializing the Blip2QFormerConfig with default values.")

        # 如果文本配置为空，则使用默认配置，并记录日志
        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the text config with default values (`OPTConfig`).")

        # 创建视觉配置对象
        self.vision_config = Blip2VisionConfig(**vision_config)
        # 创建 Q-Former 配置对象
        self.qformer_config = Blip2QFormerConfig(**qformer_config)
        # 获取文本模型类型，如果未指定则默认为 "opt"
        text_model_type = text_config["model_type"] if "model_type" in text_config else "opt"
        # 根据文本模型类型创建相应的文本配置对象
        self.text_config = CONFIG_MAPPING[text_model_type](**text_config)

        # 将是否绑定词嵌入的属性设置为文本配置中的值
        self.tie_word_embeddings = self.text_config.tie_word_embeddings
        # 将是否为编码器-解码器模型的属性设置为文本配置中的值
        self.is_encoder_decoder = self.text_config.is_encoder_decoder

        # 设置查询令牌的数量
        self.num_query_tokens = num_query_tokens
        # 将 Q-Former 配置中的编码器隐藏层大小设置为视觉配置中的隐藏层大小
        self.qformer_config.encoder_hidden_size = self.vision_config.hidden_size
        # 如果文本模型类型属于序列生成模型，则设置为仅使用解码器的语言模型
        self.use_decoder_only_language_model = self.text_config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        # 设置初始化因子和初始化范围
        self.initializer_factor = 1.0
        self.initializer_range = 0.02

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

        # 从 BLIP-2 视觉模型、Q-Former 和语言模型的配置实例化一个 Blip2Config 对象
        return cls(
            vision_config=vision_config.to_dict(),
            qformer_config=qformer_config.to_dict(),
            text_config=text_config.to_dict(),
            **kwargs,
        )
```