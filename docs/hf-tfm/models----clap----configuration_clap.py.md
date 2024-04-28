# `.\transformers\models\clap\configuration_clap.py`

```
# 设置文件编码为 UTF-8

# 版权声明和许可证信息

# 导入必要的模块
import os  # 导入操作系统模块
from typing import Union  # 导入 Union 类型，用于类型注解

# 导入配置相关的工具函数和类
from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...utils import logging  # 导入日志工具

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 预训练模型的存档列表，包含模型名称和对应的配置文件 URL
CLAP_PRETRAINED_MODEL_ARCHIVE_LIST = {
    "laion/clap-htsat-fused": "https://huggingface.co/laion/clap-htsat-fused/resolve/main/config.json",
    "laion/clap-htsat-unfused": "https://huggingface.co/laion/clap-htsat-unfused/resolve/main/config.json",
}

# CLAP 文本模型的配置类，继承自 PretrainedConfig
class ClapTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ClapTextModel`]. It is used to instantiate a CLAP
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the CLAP
    [calp-hsat-fused](https://huggingface.co/laion/clap-hsat-fused) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Examples:

    ```python
    >>> from transformers import ClapTextConfig, ClapTextModel

    >>> # Initializing a CLAP text configuration
    >>> configuration = ClapTextConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = ClapTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    # 模型类型
    model_type = "clap_text_model"

    # 初始化函数，用于设置模型配置的各种参数
    def __init__(
        self,
        vocab_size=50265,  # 词汇表大小，默认为 50265
        hidden_size=768,  # 隐藏层大小，默认为 768
        num_hidden_layers=12,  # 隐藏层的数量，默认为 12
        num_attention_heads=12,  # 注意力头的数量，默认为 12
        intermediate_size=3072,  # 中间层大小，默认为 3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为 "gelu"
        hidden_dropout_prob=0.1,  # 隐藏层的 dropout 概率，默认为 0.1
        attention_probs_dropout_prob=0.1,  # 注意力概率的 dropout 概率，默认为 0.1
        max_position_embeddings=514,  # 最大位置嵌入数，默认为 514
        type_vocab_size=1,  # 类型词汇表大小，默认为 1
        initializer_factor=1.0,  # 初始化因子，默认为 1.0
        layer_norm_eps=1e-12,  # 层归一化的 epsilon，默认为 1e-12
        projection_dim=512,  # 投影维度，默认为 512
        pad_token_id=1,  # 填充标记的 ID，默认为 1
        bos_token_id=0,  # 起始标记的 ID，默认为 0
        eos_token_id=2,  # 结束标记的 ID，默认为 2
        position_embedding_type="absolute",  # 位置嵌入类型，默认为 "absolute"
        use_cache=True,  # 是否使用缓存，默认为 True
        projection_hidden_act="relu",  # 投影隐藏层的激活函数，默认为 "relu"
        **kwargs,  # 其他参数
    # 初始化函数，继承父类的初始化方法，设置填充、起始和结束标记的标识符
    def __init__(
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 设置词汇表大小、隐藏层大小、隐藏层数、注意力头数、隐藏层激活函数、中间层大小、隐藏层丢弃概率、注意力概率丢弃概率、最大位置嵌入、类型词汇表大小、初始化因子、层归一化 epsilon、位置嵌入类型、是否使用缓存、投影隐藏层激活函数、投影维度
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_factor = initializer_factor
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.projection_hidden_act = projection_hidden_act
        self.projection_dim = projection_dim

    # 类方法，从预训练模型中加载配置
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 设置 token 参数
        cls._set_token_in_kwargs(kwargs)

        # 获取配置字典和参数
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果从 ClapConfig 加载，则获取文本配置字典
        if config_dict.get("model_type") == "clap":
            config_dict = config_dict["text_config"]

        # 如果配置字典中存在模型类型，并且当前类的模型类型与配置字典中的模型类型不一致，则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从配置字典创建实例
        return cls.from_dict(config_dict, **kwargs)
class ClapAudioConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ClapAudioModel`]. It is used to instantiate a
    CLAP audio encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the audio encoder of the CLAP
    [laion/clap-htsat-fused](https://huggingface.co/laion/clap-htsat-fused) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```python
    >>> from transformers import ClapAudioConfig, ClapAudioModel

    >>> # Initializing a ClapAudioConfig with laion/clap-htsat-fused style configuration
    >>> configuration = ClapAudioConfig()

    >>> # Initializing a ClapAudioModel (with random weights) from the laion/clap-htsat-fused style configuration
    >>> model = ClapAudioModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    # 定义模型类型为 CLAP 音频模型
    model_type = "clap_audio_model"

    # 初始化函数，用于设置模型的各种参数
    def __init__(
        self,
        # 窗口大小，用于提取特征的窗口大小，默认为 8
        window_size=8,
        # 梅尔频谱的频带数，默认为 64
        num_mel_bins=64,
        # 语谱图的大小，默认为 256
        spec_size=256,
        # 隐藏层激活函数，默认为 gelu
        hidden_act="gelu",
        # 补丁大小，默认为 4
        patch_size=4,
        # 补丁的步长，默认为 [4, 4]
        patch_stride=[4, 4],
        # 类别数量，默认为 527
        num_classes=527,
        # 隐藏层大小，默认为 768
        hidden_size=768,
        # 投影维度，默认为 512
        projection_dim=512,
        # 不同深度的 Transformer 层数量，默认为 [2, 2, 6, 2]
        depths=[2, 2, 6, 2],
        # 不同注意力头的数量，默认为 [4, 8, 16, 32]
        num_attention_heads=[4, 8, 16, 32],
        # 是否启用融合，默认为 False
        enable_fusion=False,
        # 隐藏层 dropout 概率，默认为 0.1
        hidden_dropout_prob=0.1,
        # 融合类型，默认为 None
        fusion_type=None,
        # 补丁嵌入层输入通道数，默认为 1
        patch_embed_input_channels=1,
        # 是否展平补丁嵌入，默认为 True
        flatten_patch_embeds=True,
        # 补丁嵌入的隐藏层大小，默认为 96
        patch_embeds_hidden_size=96,
        # 是否启用补丁层归一化，默认为 True
        enable_patch_layer_norm=True,
        # dropout 路径丢弃率，默认为 0.0
        drop_path_rate=0.0,
        # 注意力概率的 dropout 概率，默认为 0.0
        attention_probs_dropout_prob=0.0,
        # QKV 是否包含偏置，默认为 True
        qkv_bias=True,
        # MLP 的隐藏层与输入层的维度比例，默认为 4.0
        mlp_ratio=4.0,
        # 关注块的半径，默认为 4
        aff_block_r=4,
        # 隐藏层的数量，默认为 4
        num_hidden_layers=4,
        # 投影层的隐藏层激活函数，默认为 relu
        projection_hidden_act="relu",
        # 层归一化的 epsilon，默认为 1e-5
        layer_norm_eps=1e-5,
        # 初始化因子，默认为 1.0
        initializer_factor=1.0,
        **kwargs,
        ):
        # 调用父类的构造函数，传入关键字参数
        super().__init__(**kwargs)
        # 初始化窗口大小
        self.window_size = window_size
        # 初始化梅尔频谱的数量
        self.num_mel_bins = num_mel_bins
        # 初始化规范大小
        self.spec_size = spec_size
        # 初始化补丁大小
        self.patch_size = patch_size
        # 初始化补丁步幅
        self.patch_stride = patch_stride
        # 初始化类别数量
        self.num_classes = num_classes
        # 初始化隐藏层大小
        self.hidden_size = hidden_size
        # 初始化深度
        self.depths = depths
        # 初始化隐藏层数量
        self.num_hidden_layers = num_hidden_layers
        # 初始化注意力头数量
        self.num_attention_heads = num_attention_heads
        # 初始化窗口大小
        self.window_size = window_size
        # 初始化启用融合
        self.enable_fusion = enable_fusion
        # 初始化融合类型
        self.fusion_type = fusion_type
        # 初始化隐藏层激活函数
        self.hidden_act = hidden_act
        # 初始化隐藏层丢弃概率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 初始化投影维度
        self.projection_dim = projection_dim
        # 初始化展平补丁嵌入
        self.flatten_patch_embeds = flatten_patch_embeds
        # 初始化补丁嵌入隐藏层大小
        self.patch_embeds_hidden_size = patch_embeds_hidden_size
        # 初始化启用补丁层归一化
        self.enable_patch_layer_norm = enable_patch_layer_norm
        # 初始化丢弃路径率
        self.drop_path_rate = drop_path_rate
        # 初始化注意力概率丢弃概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 初始化查询键值偏置
        self.qkv_bias = qkv_bias
        # 初始化MLP比率
        self.mlp_ratio = mlp_ratio
        # 初始化补丁嵌入输入通道数
        self.patch_embed_input_channels = patch_embed_input_channels
        # 初始化关联块半径
        self.aff_block_r = aff_block_r
        # 初始化层归一化epsilon
        self.layer_norm_eps = layer_norm_eps
        # 初始化初始化因子
        self.initializer_factor = initializer_factor
        # 初始化投影隐藏层激活函数
        self.projection_hidden_act = projection_hidden_act

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 设置令牌在关键字参数中
        cls._set_token_in_kwargs(kwargs)

        # 获取预训练模型的配置字典和关键字参数
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果加载自ClapConfig，则获取音频配置字典
        if config_dict.get("model_type") == "clap":
            config_dict = config_dict["audio_config"]

        # 如果配置字典中存在"model_type"并且类具有"model_type"属性且不等于cls.model_type，则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从配置字典和关键字参数创建实例
        return cls.from_dict(config_dict, **kwargs)
class ClapConfig(PretrainedConfig):
    r"""
    [`ClapConfig`] is the configuration class to store the configuration of a [`ClapModel`]. It is used to instantiate
    a CLAP model according to the specified arguments, defining the text model and audio model configs. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the CLAP
    [laion/clap-htsat-fused](https://huggingface.co/laion/clap-htsat-fused) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`ClapTextConfig`].
        audio_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`ClapAudioConfig`].
        logit_scale_init_value (`float`, *optional*, defaults to 14.29):
            The inital value of the *logit_scale* parameter. Default is used as per the original CLAP implementation.
        projection_dim (`int`, *optional*, defaults to 512):
            Dimensionality of text and audio projection layers.
        projection_hidden_act (`str`, *optional*, defaults to `"relu"`):
            Activation function for the projection layers.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            Factor to scale the initialization of the model weights.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import ClapConfig, ClapModel

    >>> # Initializing a ClapConfig with laion-ai/base style configuration
    >>> configuration = ClapConfig()

    >>> # Initializing a ClapModel (with random weights) from the laion-ai/base style configuration
    >>> model = ClapModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a ClapConfig from a ClapTextConfig and a ClapAudioConfig
    >>> from transformers import ClapTextConfig, ClapAudioConfig

    >>> # Initializing a ClapText and ClapAudioConfig configuration
    >>> config_text = ClapTextConfig()
    >>> config_audio = ClapAudioConfig()

    >>> config = ClapConfig.from_text_audio_configs(config_text, config_audio)
    ```"""

    model_type = "clap"

    def __init__(
        self,
        text_config=None,
        audio_config=None,
        logit_scale_init_value=(1 / 0.07),  # 初始化 *logit_scale* 参数的初始值，默认使用原始 CLAP 实现中的值
        projection_dim=512,  # 文本和音频投影层的维度
        projection_hidden_act="relu",  # 投影层的激活函数
        initializer_factor=1.0,  # 用于缩放模型权重初始化的因子
        **kwargs,  # 其他关键字参数
```  
    ):
        # 调用父类的构造函数，传入关键字参数
        super().__init__(**kwargs)

        # 如果文本配置为空，则使用默认配置，并记录日志
        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the ClapTextConfig with default values.")

        # 如果音频配置为空，则使用默认配置，并记录日志
        if audio_config is None:
            audio_config = {}
            logger.info("audio_config is None. initializing the ClapAudioConfig with default values.")

        # 使用文本配置和音频配置初始化ClapTextConfig和ClapAudioConfig对象，并设置投影维度
        self.text_config = ClapTextConfig(**text_config)
        self.audio_config = ClapAudioConfig(**audio_config)
        self.text_config.projection_dim = projection_dim
        self.audio_config.projection_dim = projection_dim

        # 设置投影隐藏层激活函数和隐藏层大小
        self.text_config.projection_hidden_act = projection_hidden_act
        self.audio_config.projection_hidden_act = projection_hidden_act
        self.projection_dim = projection_dim
        self.projection_hidden_act = projection_hidden_act
        self.hidden_size = self.text_config.hidden_size

        # 初始化logit缩放初始值、初始化因子和隐藏层数量
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = initializer_factor
        self.num_hidden_layers = self.text_config.num_hidden_layers + len(self.audio_config.depths)

    @classmethod
    def from_text_audio_configs(cls, text_config: ClapTextConfig, audio_config: ClapAudioConfig, **kwargs):
        r"""
        从Clap文本模型配置和Clap音频模型配置实例化一个ClapConfig对象。

        Returns:
            [`ClapConfig`]: 配置对象的一个实例
        """

        # 从文本和音频配置实例化一个ClapConfig对象，并传入额外的关键字参数
        return cls(text_config=text_config.to_dict(), audio_config=audio_config.to_dict(), **kwargs)
```