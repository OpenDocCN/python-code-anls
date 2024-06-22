# `.\models\instructblip\configuration_instructblip.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，禁止未经许可使用此文件
# 可以在遵守许可证的情况下使用此文件
# 可以在以下链接获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
""" InstructBLIP 模型配置"""

# 导入所需的库
import os
from typing import Union

# 导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 导入模型映射名称
from ...models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
# 导入日志记录工具
from ...utils import logging
# 导入自动配置映射
from ..auto import CONFIG_MAPPING

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练配置文件映射
INSTRUCTBLIP_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Salesforce/instruct-blip-flan-t5": "https://huggingface.co/Salesforce/instruct-blip-flan-t5/resolve/main/config.json",
}

# InstructBlipVisionConfig 类，用于存储 InstructBlipVisionModel 的配置
class InstructBlipVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`InstructBlipVisionModel`]. It is used to
    instantiate a InstructBLIP vision encoder according to the specified arguments, defining the model architecture.
    Instantiating a configuration defaults will yield a similar configuration to that of the InstructBLIP
    [Salesforce/instruct-blip-flan-t5](https://huggingface.co/Salesforce/instruct-blip-flan-t5) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # 定义一个类 InstructBlipVisionConfig，用于配置 InstructBlipVisionModel
    Args:
        # 编码器层和池化层的维度
        hidden_size (`int`, *optional*, defaults to 1408):
        # Transformer 编码器中“中间”（即前馈）层的维度
        intermediate_size (`int`, *optional*, defaults to 6144):
        # Transformer 编码器中隐藏层的数量
        num_hidden_layers (`int`, *optional*, defaults to 39):
        # Transformer 编码器中每个注意力层的注意力头数量
        num_attention_heads (`int`, *optional*, defaults to 16):
        # 每个图像的大小（分辨率）
        image_size (`int`, *optional*, defaults to 224):
        # 每个补丁的大小（分辨率）
        patch_size (`int`, *optional*, defaults to 14):
        # 编码器和池化层中的非线性激活函数
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
        # 层归一化层使用的 epsilon
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
        # 注意力概率的 dropout 比率
        attention_dropout (`float`, *optional*, defaults to 0.0):
        # 用于初始化所有权重矩阵的截断正态初始化器的标准差
        initializer_range (`float`, *optional*, defaults to 1e-10):
        # 是否在自注意力层中为查询和值添加偏置
        qkv_bias (`bool`, *optional*, defaults to `True`):

    Example:

    ```python
    >>> from transformers import InstructBlipVisionConfig, InstructBlipVisionModel

    >>> # 使用 Salesforce/instruct-blip-flan-t5 风格配置初始化 InstructBlipVisionConfig
    >>> configuration = InstructBlipVisionConfig()

    >>> # 使用 Salesforce/instruct-blip-flan-t5 风格配置初始化 InstructBlipVisionModel（带有随机权重）
    >>> model = InstructBlipVisionModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py"""

    # 模型类型为 instructblip_vision_model
    model_type = "instructblip_vision_model"

    # 初始化方法，设置各种参数
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
    # 初始化函数，继承父类的初始化方法，并传入关键字参数
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
        # 调用父类的初始化方法，传入关键字参数
        super().__init__(**kwargs)

        # 初始化模型的隐藏层大小
        self.hidden_size = hidden_size
        # 初始化模型的中间层大小
        self.intermediate_size = intermediate_size
        # 初始化模型的隐藏层数
        self.num_hidden_layers = num_hidden_layers
        # 初始化模型的注意力头数
        self.num_attention_heads = num_attention_heads
        # 初始化模型的补丁大小
        self.patch_size = patch_size
        # 初始化模型的图像大小
        self.image_size = image_size
        # 初始化模型的初始化范围
        self.initializer_range = initializer_range
        # 初始化模型的注意力丢弃率
        self.attention_dropout = attention_dropout
        # 初始化模型的层归一化 epsilon 值
        self.layer_norm_eps = layer_norm_eps
        # 初始化模型的隐藏层激活函数
        self.hidden_act = hidden_act
        # 初始化模型的 qkv 偏置
        self.qkv_bias = qkv_bias

    # 类方法，从预训练模型中加载配置
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 设置 token 在关键字参数中
        cls._set_token_in_kwargs(kwargs)

        # 获取配置字典和关键字参数
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典中的模型类型为 instructblip，则获取视觉配置字典
        if config_dict.get("model_type") == "instructblip":
            config_dict = config_dict["vision_config"]

        # 如果配置字典中存在模型类型，并且类中有 model_type 属性，并且配置字典中的模型类型不等于类的模型类型，则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从配置字典和关键字参数中创建实例
        return cls.from_dict(config_dict, **kwargs)
class InstructBlipQFormerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`InstructBlipQFormerModel`]. It is used to
    instantiate a InstructBLIP Querying Transformer (Q-Former) model according to the specified arguments, defining the
    model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the InstructBLIP [Salesforce/instruct-blip-flan-t5](https://huggingface.co/Salesforce/instruct-blip-flan-t5)
    architecture. Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs.
    Read the documentation from [`PretrainedConfig`] for more information.

    Note that [`InstructBlipQFormerModel`] is very similar to [`BertLMHeadModel`] with interleaved cross-attention.
    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Q-Former 模型的词汇表大小。定义了在调用模型时可以表示的不同标记数量。
        hidden_size (`int`, *optional*, defaults to 768):
            编码器层和池化层的维度。
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Transformer 编码器中的隐藏层数量。
        num_attention_heads (`int`, *optional*, defaults to 12):
            Transformer 编码器中每个注意力层的注意力头数量。
        intermediate_size (`int`, *optional*, defaults to 3072):
            Transformer 编码器中“中间”（通常称为前馈）层的维度。
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            编码器和池化器中的非线性激活函数（函数或字符串）。如果是字符串，支持 `"gelu"`, `"relu"`, `"silu"` 和 `"gelu_new"`。
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            嵌入层、编码器和池化器中所有全连接层的丢弃概率。
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            注意力概率的丢弃比率。
        max_position_embeddings (`int`, *optional*, defaults to 512):
            模型可能使用的最大序列长度。通常设置为一个较大的值（例如，512、1024或2048）。
        initializer_range (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态初始化器的标准差。
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            层归一化层使用的 epsilon。
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            位置嵌入的类型。选择 `"absolute"`, `"relative_key"`, `"relative_key_query"` 中的一个。对于位置嵌入使用 `"absolute"`。
            有关 `"relative_key"` 的更多信息，请参考 [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155)。
            有关 `"relative_key_query"` 的更多信息，请参考 [Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658) 中的 *Method 4*。
        cross_attention_frequency (`int`, *optional*, defaults to 2):
            向 Transformer 层添加交叉注意力的频率。
        encoder_hidden_size (`int`, *optional*, defaults to 1408):
            交叉注意力的隐藏状态的隐藏大小。

    Examples:

    ```python
    >>> from transformers import InstructBlipQFormerConfig, InstructBlipQFormerModel
    >>> # 初始化一个 InstructBLIP Salesforce/instruct-blip-flan-t5 风格的配置
    >>> configuration = InstructBlipQFormerConfig()

    >>> # 使用 Salesforce/instruct-blip-flan-t5 风格的配置初始化一个模型（带有随机权重）
    >>> model = InstructBlipQFormerModel(configuration)
    >>> # 访问模型配置
    >>> configuration = model.config
    ```py"""

    # 模型类型为 "instructblip_qformer"
    model_type = "instructblip_qformer"

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
        # 调用父类的初始化方法
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        # 设置各种配置参数
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
        # 设置 token 参数
        cls._set_token_in_kwargs(kwargs)

        # 获取预训练模型的配置字典和参数
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果从 InstructBlipConfig 加载，则获取 qformer 配置字典
        if config_dict.get("model_type") == "instructblip":
            config_dict = config_dict["qformer_config"]

        # 如果配置字典中包含模型类型并且与当前模型类型不匹配，则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)
# 定义 InstructBlipConfig 类，用于存储 InstructBlipForConditionalGeneration 模型的配置信息
class InstructBlipConfig(PretrainedConfig):
    r"""
    [`InstructBlipConfig`] 是用于存储 [`InstructBlipForConditionalGeneration`] 模型配置的类。它用于根据指定的参数实例化一个 InstructBLIP 模型，定义了视觉模型、Q-Former 模型和语言模型的配置。使用默认配置实例化一个配置对象将产生类似于 InstructBLIP [Salesforce/instruct-blip-flan-t5](https://huggingface.co/Salesforce/instruct-blip-flan-t5) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    Args:
        vision_config (`dict`, *optional*):
            用于初始化 [`InstructBlipVisionConfig`] 的配置选项字典。
        qformer_config (`dict`, *optional*):
            用于初始化 [`InstructBlipQFormerConfig`] 的配置选项字典。
        text_config (`dict`, *optional*):
            用于初始化任何 [`PretrainedConfig`] 的配置选项字典。
        num_query_tokens (`int`, *optional*, 默认为 32):
            通过 Transformer 传递的查询令牌数量。

        kwargs (*optional*):
            关键字参数字典。

    Example:

    ```python
    >>> from transformers import (
    ...     InstructBlipVisionConfig,
    ...     InstructBlipQFormerConfig,
    ...     OPTConfig,
    ...     InstructBlipConfig,
    ...     InstructBlipForConditionalGeneration,
    ... )

    >>> # 使用 Salesforce/instruct-blip-flan-t5 风格配置初始化 InstructBlipConfig
    >>> configuration = InstructBlipConfig()

    >>> # 使用 Salesforce/instruct-blip-flan-t5 风格配置初始化 InstructBlipForConditionalGeneration（带有随机权重）
    >>> model = InstructBlipForConditionalGeneration(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config

    >>> # 我们还可以从 InstructBlipVisionConfig、InstructBlipQFormerConfig 和任何 PretrainedConfig 初始化 InstructBlipConfig

    >>> # 初始化 InstructBLIP 视觉、InstructBLIP Q-Former 和语言模型配置
    >>> vision_config = InstructBlipVisionConfig()
    >>> qformer_config = InstructBlipQFormerConfig()
    >>> text_config = OPTConfig()

    >>> config = InstructBlipConfig.from_text_vision_configs(vision_config, qformer_config, text_config)
    ```py"""

    # 模型类型为 instructblip
    model_type = "instructblip"
    # 初始化函数，接受视觉配置、查询形式器配置、文本配置和其他参数
    def __init__(self, vision_config=None, qformer_config=None, text_config=None, num_query_tokens=32, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 如果视觉配置为空，则初始化为空字典，并记录日志
        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. initializing the InstructBlipVisionConfig with default values.")

        # 如果查询形式器配置为空，则初始化为空字典，并记录日志
        if qformer_config is None:
            qformer_config = {}
            logger.info("qformer_config is None. Initializing the InstructBlipQFormerConfig with default values.")

        # 如果文本配置为空，则初始化为空字典，并记录日志
        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the text config with default values (`OPTConfig`).")

        # 使用视觉配置初始化 InstructBlipVisionConfig 对象
        self.vision_config = InstructBlipVisionConfig(**vision_config)
        # 使用查询形式器配置初始化 InstructBlipQFormerConfig 对象
        self.qformer_config = InstructBlipQFormerConfig(**qformer_config)
        # 获取文本模型类型，如果不存在则默认为 "opt"
        text_model_type = text_config["model_type"] if "model_type" in text_config else "opt"
        # 使用文本配置初始化对应的配置对象
        self.text_config = CONFIG_MAPPING[text_model_type](**text_config)

        # 设置是否共享词嵌入
        self.tie_word_embeddings = self.text_config.tie_word_embeddings
        # 设置是否为编码器-解码器模型
        self.is_encoder_decoder = self.text_config.is_encoder_decoder

        # 设置查询标记数量
        self.num_query_tokens = num_query_tokens
        # 将查询形式器的编码器隐藏大小设置为视觉配置的隐藏大小
        self.qformer_config.encoder_hidden_size = self.vision_config.hidden_size
        # 根据文本模型类型判断是否仅使用解码器的语言模型
        self.use_decoder_only_language_model = self.text_config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        # 初始化因子为1.0
        self.initializer_factor = 1.0
        # 初始化范围为0.02
        self.initializer_range = 0.02

    # 类方法，从视觉配置、查询形式器配置和文本配置实例化 InstructBlipConfig 对象
    @classmethod
    def from_vision_qformer_text_configs(
        cls,
        vision_config: InstructBlipVisionConfig,
        qformer_config: InstructBlipQFormerConfig,
        text_config: PretrainedConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`InstructBlipConfig`] (or a derived class) from a InstructBLIP vision model, Q-Former and
        language model configurations.

        Returns:
            [`InstructBlipConfig`]: An instance of a configuration object
        """

        # 返回一个根据给定配置实例化的 InstructBlipConfig 对象
        return cls(
            vision_config=vision_config.to_dict(),
            qformer_config=qformer_config.to_dict(),
            text_config=text_config.to_dict(),
            **kwargs,
        )
```