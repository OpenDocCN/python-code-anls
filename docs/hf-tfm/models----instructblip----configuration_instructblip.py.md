# `.\models\instructblip\configuration_instructblip.py`

```
# coding=utf-8
# 定义脚本编码格式为 UTF-8

# 版权声明，此代码版权归 HuggingFace Inc. 团队所有，保留所有权利
# 根据 Apache License, Version 2.0 许可证使用此文件，除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律有明确规定或书面同意，否则依据此许可证分发的软件是基于“原样”提供的，无任何明示或暗示的担保或条件
# 请查看许可证以获取特定语言的详细信息

""" InstructBLIP model configuration"""

# 导入操作系统模块
import os
# 导入 Union 类型提示
from typing import Union

# 导入配置工具函数
from ...configuration_utils import PretrainedConfig
# 导入模型映射名称
from ...models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
# 导入日志工具
from ...utils import logging
# 导入自动配置映射
from ..auto import CONFIG_MAPPING

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义预训练配置存档映射字典
INSTRUCTBLIP_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Salesforce/instruct-blip-flan-t5": "https://huggingface.co/Salesforce/instruct-blip-flan-t5/resolve/main/config.json",
}

# 定义 InstructBlipVisionConfig 类，继承自 PretrainedConfig 类
class InstructBlipVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`InstructBlipVisionModel`]. It is used to
    instantiate a InstructBLIP vision encoder according to the specified arguments, defining the model architecture.
    Instantiating a configuration defaults will yield a similar configuration to that of the InstructBLIP
    [Salesforce/instruct-blip-flan-t5](https://huggingface.co/Salesforce/instruct-blip-flan-t5) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # 模型类型字符串，用于指示模型类型
    model_type = "instructblip_vision_model"
    
    # 初始化方法，用于创建一个新的InstructBlipVisionConfig对象
    def __init__(
        self,
        # 编码器层和池化层的维度大小，默认为1408
        hidden_size=1408,
        # Transformer编码器中“中间”（即前馈）层的维度大小，默认为6144
        intermediate_size=6144,
        # Transformer编码器中隐藏层的数量，默认为39
        num_hidden_layers=39,
        # Transformer编码器中每个注意力层的注意力头数量，默认为16
        num_attention_heads=16,
        # 每个图像的分辨率大小，默认为224
        image_size=224,
        # 每个图像分块的分辨率大小，默认为14
        patch_size=14,
        # 编码器和池化器中的非线性激活函数（函数或字符串），支持"gelu"、"relu"、"selu"和"gelu_new"，默认为"gelu"
        hidden_act="gelu",
        # 层归一化层使用的epsilon值，默认为1e-6
        layer_norm_eps=1e-6,
        # 注意力概率的dropout比率，默认为0.0
        attention_dropout=0.0,
        # 用于初始化所有权重矩阵的截断正态分布的标准差，默认为1e-10
        initializer_range=1e-10,
        # 是否在自注意力层中添加查询和值的偏置，默认为True
        qkv_bias=True,
        **kwargs,
    ):
        # 调用父类的初始化方法，传递所有的关键字参数
        super().__init__(**kwargs)

        # 设置隐藏层大小
        self.hidden_size = hidden_size
        # 设置中间层大小
        self.intermediate_size = intermediate_size
        # 设置隐藏层的数量
        self.num_hidden_layers = num_hidden_layers
        # 设置注意力头的数量
        self.num_attention_heads = num_attention_heads
        # 设置图像块的大小
        self.patch_size = patch_size
        # 设置图像的总体大小
        self.image_size = image_size
        # 设置初始化范围
        self.initializer_range = initializer_range
        # 设置注意力机制的dropout率
        self.attention_dropout = attention_dropout
        # 设置层归一化的epsilon值
        self.layer_norm_eps = layer_norm_eps
        # 设置隐藏层激活函数
        self.hidden_act = hidden_act
        # 设置查询/键/值矩阵的偏置
        self.qkv_bias = qkv_bias

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 调用内部方法，将token信息设置到关键字参数中
        cls._set_token_in_kwargs(kwargs)

        # 从预训练模型名称或路径获取配置字典和剩余的关键字参数
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典中的模型类型是"instructblip"，则使用其视觉配置字典
        if config_dict.get("model_type") == "instructblip":
            config_dict = config_dict["vision_config"]

        # 如果配置字典中指定了模型类型，并且该类有模型类型属性，并且两者不匹配，则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 使用配置字典和剩余的关键字参数构建一个新的实例
        return cls.from_dict(config_dict, **kwargs)
# 定义配置类 `InstructBlipQFormerConfig`，继承自 `PretrainedConfig`，用于存储 `InstructBlipQFormerModel` 的配置信息。
class InstructBlipQFormerConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`InstructBlipQFormerModel`] 的配置信息。它用于实例化一个 InstructBLIP Querying Transformer (Q-Former) 模型，
    根据指定的参数定义模型架构。使用默认参数实例化配置将产生与 InstructBLIP [Salesforce/instruct-blip-flan-t5](https://huggingface.co/Salesforce/instruct-blip-flan-t5)
    架构类似的配置。配置对象继承自 [`PretrainedConfig`]，可以用于控制模型的输出。阅读 [`PretrainedConfig`] 的文档获取更多信息。

    注意，[`InstructBlipQFormerModel`] 与 [`BertLMHeadModel`] 非常相似，具有交织的交叉注意力。
    """
    # 定义 Q-Former 模型配置的类，包含了模型的各种参数设置
    class InstructBlipQFormerConfig:
        # 词汇表大小，默认为 30522，定义了输入 `inputs_ids` 中可以表示的不同令牌数量
        def __init__(self,
                     vocab_size=30522,
                     # 编码器层和汇聚层的维度
                     hidden_size=768,
                     # Transformer 编码器中的隐藏层数量
                     num_hidden_layers=12,
                     # 每个注意力层中的注意力头数
                     num_attention_heads=12,
                     # Transformer 编码器中 "中间"（通常称为前馈）层的维度
                     intermediate_size=3072,
                     # 编码器和汇聚层中的非线性激活函数
                     hidden_act="gelu",
                     # 嵌入层、编码器和汇聚层中所有全连接层的 dropout 概率
                     hidden_dropout_prob=0.1,
                     # 注意力概率的 dropout 比例
                     attention_probs_dropout_prob=0.1,
                     # 可能用于模型的最大序列长度
                     max_position_embeddings=512,
                     # 初始化所有权重矩阵的截断正态初始化器的标准差
                     initializer_range=0.02,
                     # 层归一化层使用的 epsilon
                     layer_norm_eps=1e-12,
                     # 位置嵌入类型，可以选择 "absolute"、"relative_key" 或 "relative_key_query"
                     position_embedding_type="absolute",
                     # 在 Transformer 层中添加交叉注意力的频率
                     cross_attention_frequency=2,
                     # 交叉注意力中隐藏状态的隐藏大小
                     encoder_hidden_size=1408):
            # 将所有参数赋值给对应的实例变量
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.intermediate_size = intermediate_size
            self.hidden_act = hidden_act
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
            self.position_embedding_type = position_embedding_type
            self.cross_attention_frequency = cross_attention_frequency
            self.encoder_hidden_size = encoder_hidden_size
    # 设定模型类型为 "instructblip_qformer"
    model_type = "instructblip_qformer"

    # 初始化方法，定义模型配置的各个参数
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
        # 调用父类的初始化方法，设置 pad_token_id 参数
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        # 初始化对象的各个属性，以提供模型配置的详细信息
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
        # 设置关键字参数中的 token
        cls._set_token_in_kwargs(kwargs)

        # 获取预训练模型的配置字典和可能的额外参数
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典中的模型类型为 "instructblip"，则从中提取 qformer_config
        if config_dict.get("model_type") == "instructblip":
            config_dict = config_dict["qformer_config"]

        # 如果配置字典中的模型类型与当前类的模型类型不匹配，发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 使用配置字典和关键字参数创建类的实例
        return cls.from_dict(config_dict, **kwargs)
class InstructBlipConfig(PretrainedConfig):
    r"""
    [`InstructBlipConfig`] is the configuration class to store the configuration of a
    [`InstructBlipForConditionalGeneration`]. It is used to instantiate a InstructBLIP model according to the specified
    arguments, defining the vision model, Q-Former model and language model configs. Instantiating a configuration with
    the defaults will yield a similar configuration to that of the InstructBLIP
    [Salesforce/instruct-blip-flan-t5](https://huggingface.co/Salesforce/instruct-blip-flan-t5) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`InstructBlipVisionConfig`].
        qformer_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`InstructBlipQFormerConfig`].
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize any [`PretrainedConfig`].
        num_query_tokens (`int`, *optional*, defaults to 32):
            The number of query tokens passed through the Transformer.

        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import (
    ...     InstructBlipVisionConfig,
    ...     InstructBlipQFormerConfig,
    ...     OPTConfig,
    ...     InstructBlipConfig,
    ...     InstructBlipForConditionalGeneration,
    ... )

    >>> # Initializing a InstructBlipConfig with Salesforce/instruct-blip-flan-t5 style configuration
    >>> configuration = InstructBlipConfig()

    >>> # Initializing a InstructBlipForConditionalGeneration (with random weights) from the Salesforce/instruct-blip-flan-t5 style configuration
    >>> model = InstructBlipForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a InstructBlipConfig from a InstructBlipVisionConfig, InstructBlipQFormerConfig and any PretrainedConfig

    >>> # Initializing InstructBLIP vision, InstructBLIP Q-Former and language model configurations
    >>> vision_config = InstructBlipVisionConfig()
    >>> qformer_config = InstructBlipQFormerConfig()
    >>> text_config = OPTConfig()

    >>> config = InstructBlipConfig.from_text_vision_configs(vision_config, qformer_config, text_config)
    ```"""

    # 设置模型类型为 "instructblip"
    model_type = "instructblip"
    # 定义类的初始化方法，接受多个配置参数和其他关键字参数
    def __init__(self, vision_config=None, qformer_config=None, text_config=None, num_query_tokens=32, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 如果视觉配置为空，则使用空字典，并记录日志信息
        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. initializing the InstructBlipVisionConfig with default values.")

        # 如果Q-Former配置为空，则使用空字典，并记录日志信息
        if qformer_config is None:
            qformer_config = {}
            logger.info("qformer_config is None. Initializing the InstructBlipQFormerConfig with default values.")

        # 如果文本配置为空，则使用空字典，并记录日志信息
        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the text config with default values (`OPTConfig`).")

        # 根据传入的配置参数实例化视觉配置对象
        self.vision_config = InstructBlipVisionConfig(**vision_config)
        # 根据传入的配置参数实例化Q-Former配置对象
        self.qformer_config = InstructBlipQFormerConfig(**qformer_config)
        
        # 获取文本模型的类型，若未指定则默认为"opt"
        text_model_type = text_config["model_type"] if "model_type" in text_config else "opt"
        # 根据文本模型类型选择对应的配置类实例化文本配置对象
        self.text_config = CONFIG_MAPPING[text_model_type](**text_config)

        # 设置词嵌入是否共享的标志
        self.tie_word_embeddings = self.text_config.tie_word_embeddings
        # 设置是否为编码器-解码器模型的标志
        self.is_encoder_decoder = self.text_config.is_encoder_decoder

        # 设置查询令牌的数量
        self.num_query_tokens = num_query_tokens
        # 将Q-Former的编码器隐藏层大小设置为视觉配置的隐藏层大小
        self.qformer_config.encoder_hidden_size = self.vision_config.hidden_size
        # 根据文本模型类型判断是否只使用解码器作为语言模型
        self.use_decoder_only_language_model = self.text_config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        # 初始化因子
        self.initializer_factor = 1.0
        # 初始化范围
        self.initializer_range = 0.02

    @classmethod
    # 类方法：从给定的视觉配置、Q-Former配置和文本配置参数实例化一个InstructBlipConfig（或其派生类）对象
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

        # 调用类的构造函数，传入各配置对象的字典形式和其他关键字参数
        return cls(
            vision_config=vision_config.to_dict(),
            qformer_config=qformer_config.to_dict(),
            text_config=text_config.to_dict(),
            **kwargs,
        )
```