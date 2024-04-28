# `.\transformers\models\clvp\configuration_clvp.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，禁止未经许可使用该文件
# 可以在以下链接获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制

""" CLVP 模型配置"""

# 导入必要的库
import os
from typing import TYPE_CHECKING, Union

# 如果是类型检查，则执行以下代码
if TYPE_CHECKING:
    pass

# 导入配置工具和日志工具
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# CLVP 预训练配置文件映射
CLVP_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "susnato/clvp_dev": "https://huggingface.co/susnato/clvp_dev/resolve/main/config.json",
}

# CLVP 编码器配置类，用于存储 CLVP 编码器的配置
class ClvpEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ClvpEncoder`]. It is used to instantiate a CLVP
    text or CLVP speech encoder according to the specified arguments. Instantiating a configuration with the defaults
    will yield a similar configuration to that of the encoder of the CLVP
    [susnato/clvp_dev](https://huggingface.co/susnato/clvp_dev) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 256):
            Vocabulary size of the CLVP Encoder model.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 1536):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        projection_dim (`int`, *optional*, defaults to 768):
            Dimensionality of the projection vector.
        num_hidden_layers (`int`, *optional*, defaults to 20):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the feed-forward layers in [`ClvpEncoderMLP`].
        use_rotary_embedding (`bool`, *optional*, defaults to `True`):
            Whether to use rotary_embedding or not.
        use_attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in Query, Key and Value layers during self attention.
        summary_type (`str`, *optional*, defaults to `"mean"`):
            What strategy to use to get pooler_output from the last_hidden_state. `"last"`, `"first"`, `"mean"` and
            `"cls_index"` are supported.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1.0, used internally for initialization
            testing).
        bos_token_id (`int`, *optional*, defaults to 255):
            Beginning of sequence token id.
        eos_token_id (`int`, *optional*, defaults to 0):
            End of sequence token id.

    Example:

    ```python
    >>> from transformers import ClvpEncoderConfig, ClvpEncoder

    >>> # Initializing a ClvpEncoderConfig with susnato/clvp_dev style configuration
    >>> encoder_configuration = ClvpEncoderConfig()

    >>> # Initializing a ClvpEncoder (with random weights) from the susnato/clvp_dev style configuration
    >>> model = ClvpEncoder(encoder_configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```py"""

    # 设定模型类型为 "clvp_encoder"
    model_type = "clvp_encoder"
    # 初始化函数，设置模型的各种参数
    def __init__(
        self,
        vocab_size=256,
        hidden_size=768,
        intermediate_size=1536,
        projection_dim=768,
        num_hidden_layers=20,
        num_attention_heads=12,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.1,
        dropout=0.1,
        use_rotary_embedding=True,
        use_attention_bias=False,
        summary_type="mean",
        initializer_factor=1.0,
        bos_token_id=255,
        eos_token_id=0,
        **kwargs,
    ):
        # 设置模型的各种参数
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.use_rotary_embedding = use_rotary_embedding
        self.use_attention_bias = use_attention_bias
        self.summary_type = summary_type
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # 调用父类的初始化函数
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

    # 从预训练模型加载配置
    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], config_type: str = "text_config", **kwargs
    ) -> "PretrainedConfig":
        # 设置token参数
        cls._set_token_in_kwargs(kwargs)

        # 获取预训练模型的配置字典和其他参数
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 确保config_type为"text_config"或"speech_config"
        if config_type not in ["text_config", "speech_config"]:
            raise ValueError(
                f"We can only load either 'text_config' or 'speech_config' but you are trying to load" f"{config_type}"
            )

        # 如果配置字典中的model_type为"clvp"，则获取对应的配置字典
        if config_dict.get("model_type") == "clvp":
            config_dict = config_dict[config_type]

        # 如果配置字典中存在"model_type"，并且与当前类的model_type不同，发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 根据配置字典和其他参数创建实例
        return cls.from_dict(config_dict, **kwargs)
# 定义一个配置类，用于存储 CLVP 解码器的配置，用于实例化 CLVP 解码器模型
class ClvpDecoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ClvpDecoder`]. It is used to instantiate a CLVP
    Decoder Model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Decoder part of the CLVP
    [susnato/clvp_dev](https://huggingface.co/susnato/clvp_dev) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    The architecture is similar to GPT2.

    Example:

    ```python
    >>> from transformers import ClvpDecoderConfig, ClvpDecoder

    >>> # Initializing a ClvpDecoderConfig with susnato/clvp_dev style configuration
    >>> decoder_configuration = ClvpDecoderConfig()

    >>> # Initializing a ClvpDecoder (with random weights) from the susnato/clvp_dev style configuration
    >>> model = ClvpDecoder(decoder_configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```py"""

    # 模型类型为 CLVP 解码器
    model_type = "clvp_decoder"

    # 初始化方法，设置 CLVP 解码器的各种配置参数
    def __init__(
        self,
        vocab_size=8194,
        max_position_embeddings=608,
        max_text_tokens=404,
        hidden_size=1024,
        num_hidden_layers=30,
        num_attention_heads=16,
        n_inner=None,
        num_mel_attn_blocks=6,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attention_dropout=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        use_cache=True,
        bos_token_id=8192,
        eos_token_id=8193,
        feature_size=80,
        use_attention_bias=True,
        initializer_factor=1.0,
        decoder_fixing_codes=[83, 45, 45, 248],
        **kwargs,
        self.vocab_size = vocab_size
        # 设置词汇表大小
        self.max_position_embeddings = max_position_embeddings
        # 设置最大位置嵌入
        self.max_text_tokens = max_text_tokens
        # 设置最大文本标记数
        self.hidden_size = hidden_size
        # 设置隐藏层大小
        self.num_hidden_layers = num_hidden_layers
        # 设置隐藏层数量
        self.num_attention_heads = num_attention_heads
        # 设置注意力头数量
        self.n_inner = n_inner
        # 设置内部层数量
        self.num_mel_attn_blocks = num_mel_attn_blocks
        # 设置 mel 注意力块数量
        self.activation_function = activation_function
        # 设置激活函数
        self.resid_pdrop = resid_pdrop
        # 设置残差丢弃率
        self.embd_pdrop = embd_pdrop
        # 设置嵌入丢弃率
        self.attention_dropout = attention_dropout
        # 设置注意力丢弃率
        self.layer_norm_epsilon = layer_norm_epsilon
        # 设置层归一化 epsilon
        self.initializer_range = initializer_range
        # 设置初始化范围
        self.summary_type = summary_type
        # 设置摘要类型
        self.summary_use_proj = summary_use_proj
        # 设置是否使用摘要投影
        self.summary_activation = summary_activation
        # 设置摘要激活函数
        self.summary_first_dropout = summary_first_dropout
        # 设置摘要首次丢弃率
        self.summary_proj_to_labels = summary_proj_to_labels
        # 设置摘要投影到标签
        self.use_cache = use_cache
        # 设置是否使用缓存
        self.feature_size = feature_size
        # 设置特征大小
        self.use_attention_bias = use_attention_bias
        # 设置是否使用注意力偏置
        self.initializer_factor = initializer_factor
        # 设置初始化因子
        self.decoder_fixing_codes = decoder_fixing_codes
        # 设置解码器修复代码

        self.bos_token_id = bos_token_id
        # 设置开始标记 ID
        self.eos_token_id = eos_token_id
        # 设置结束标记 ID

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        # 调用父类初始化方法，传入开始标记 ID、结束标记 ID 和其他关键字参数

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 从预训练模型名称或路径创建配置类实例，返回预训练配置类

        cls._set_token_in_kwargs(kwargs)
        # 在关键字参数中设置令牌

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        # 获取配置字典和关键字参数

        # 如果从 ClvpConfig 加载，则获取语音配置字典
        if config_dict.get("model_type") == "clvp":
            config_dict = config_dict["decoder_config"]

        # 如果配置字典中存在 "model_type"，并且类中有 "model_type" 属性，并且配置字典中的 "model_type" 不等于类的 model_type
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从字典创建配置类实例，传入关键字参数
        return cls.from_dict(config_dict, **kwargs)
# 定义 ClvpConfig 类，用于存储 ClvpModelForConditionalGeneration 的配置信息
class ClvpConfig(PretrainedConfig):
    r"""
    [`ClvpConfig`] is the configuration class to store the configuration of a [`ClvpModelForConditionalGeneration`]. It
    is used to instantiate a CLVP model according to the specified arguments, defining the text model, speech model and
    decoder model configs. Instantiating a configuration with the defaults will yield a similar configuration to that
    of the CLVP [susnato/clvp_dev](https://huggingface.co/susnato/clvp_dev) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize the CLVP text encoder.
        speech_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize CLVP speech encoder.
        decoder_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`ClvpDecoderConfig`].
        projection_dim (`int`, *optional*, defaults to 768):
            Dimentionality of text and speech projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* paramter. Default is used as per the original CLVP implementation.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1.0, used internally for initialization
            testing).
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import ClvpConfig, ClvpModelForConditionalGeneration

    >>> # Initializing a ClvpConfig with susnato/clvp_dev style configuration
    >>> configuration = ClvpConfig()

    >>> # Initializing a ClvpModelForConditionalGeneration (with random weights) from the susnato/clvp_dev style configuration
    >>> model = ClvpModelForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a CLVPConfig from a CLVPTextConfig, CLVPSpeechConfig and a CLVPAutoRegressiveConfig
    >>> from transformers import ClvpEncoderConfig, ClvpDecoderConfig

    >>> # Initializing a CLVP text, CLVP speech and CLVP decoder configuration
    >>> config_text = ClvpEncoderConfig()
    >>> config_speech = ClvpEncoderConfig()
    >>> decoder_config = ClvpDecoderConfig()

    >>> config = ClvpConfig.from_sub_model_configs(config_text, config_speech, decoder_config)
    ```py"""

    # 模型类型为 CLVP
    model_type = "clvp"
    # 表示该配置是由多个子配置组合而成
    is_composition = True

    def __init__(
        self,
        text_config=None,
        speech_config=None,
        decoder_config=None,
        projection_dim=768,
        logit_scale_init_value=2.6592,
        initializer_factor=1.0,
        **kwargs,
    ):
        # 调用父类的构造函数
        super().__init__(**kwargs)

        # 如果文本配置为空，则将其初始化为空字典，并记录日志
        if text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `ClvpEncoderConfig` with default values.")

        # 如果语音配置为空，则将其初始化为空字典，并记录日志
        if speech_config is None:
            speech_config = {}
            logger.info("`speech_config` is `None`. initializing the `ClvpEncoderConfig` with default values.")

        # 如果解码器配置为空，则将其初始化为空字典，并记录日志
        if decoder_config is None:
            decoder_config = {}
            logger.info("`decoder_config` is `None`. initializing the `ClvpDecoderConfig` with default values.")

        # 使用文本、语音和解码器配置来实例化相应的配置对象
        self.text_config = ClvpEncoderConfig(**text_config)
        self.speech_config = ClvpEncoderConfig(**speech_config)
        self.decoder_config = ClvpDecoderConfig(**decoder_config)

        # 设置投影维度、logit 缩放初始值和初始化因子
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = initializer_factor

    @classmethod
    def from_sub_model_configs(
        cls,
        text_config: ClvpEncoderConfig,
        speech_config: ClvpEncoderConfig,
        decoder_config: ClvpDecoderConfig,
        **kwargs,
    ):
        r"""
        从 CLVP 文本模型配置、CLVP 语音模型配置和 CLVP 解码器模型配置实例化一个 [`ClvpConfig`]（或其派生类）。

        Args:
            text_config (`ClvpEncoderConfig`):
                类型为 [`ClvpEncoderConfig`] 的文本模型配置。
            speech_config (`ClvpEncoderConfig`):
                类型为 [`ClvpEncoderConfig`] 的语音模型配置。
            decoder_config (`ClvpDecoderConfig`):
                类型为 [`ClvpDecoderConfig`] 的解码器模型配置。

        Returns:
            [`ClvpConfig`]: 配置对象的一个实例
        """

        # 使用文本、语音和解码器配置的字典形式来实例化一个配置对象
        return cls(
            text_config=text_config.to_dict(),
            speech_config=speech_config.to_dict(),
            decoder_config=decoder_config.to_dict(),
            **kwargs,
        )
```