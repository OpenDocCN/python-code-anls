# `.\models\clvp\configuration_clvp.py`

```py
# 设置文件编码格式为 utf-8
# 版权声明及许可协议说明
# 这是 CLVP 模型的配置文件

# 导入操作系统模块和类型检查模块
import os
from typing import TYPE_CHECKING, Union

# 如果类型检查为真，则执行代码块
if TYPE_CHECKING:
    pass

# 从 HuggingFace 函数库中导入预训练配置和日志工具
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# CLVP 预训练配置存档
CLVP_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "susnato/clvp_dev": "https://huggingface.co/susnato/clvp_dev/resolve/main/config.json",
}

# CLVP 编码器配置类，用于存储 CLVP 编码器的配置信息
class ClvpEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ClvpEncoder`]. It is used to instantiate a CLVP
    text or CLVP speech encoder according to the specified arguments. Instantiating a configuration with the defaults
    will yield a similar configuration to that of the encoder of the CLVP
    [susnato/clvp_dev](https://huggingface.co/susnato/clvp_dev) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # 设置模型类型为 "clvp_encoder"
    model_type = "clvp_encoder"
    # 初始化函数，用于设置模型的
# 定义 CLVP 解码器配置类，继承自预训练配置类 PretrainedConfig
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

    ```
    >>> from transformers import ClvpDecoderConfig, ClvpDecoder

    >>> # Initializing a ClvpDecoderConfig with susnato/clvp_dev style configuration
    >>> decoder_configuration = ClvpDecoderConfig()

    >>> # Initializing a ClvpDecoder (with random weights) from the susnato/clvp_dev style configuration
    >>> model = ClvpDecoder(decoder_configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    # 模型类型为 "clvp_decoder"
    model_type = "clvp_decoder"

    # 初始化函数，定义 CLVP 解码器配置的各项参数
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
    ):
        # 调用父类的初始化函数，传递所有参数
        super().__init__(**kwargs)
        # 定义 CLVP 解码器特有的参数
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.max_text_tokens = max_text_tokens
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_inner = n_inner
        self.num_mel_attn_blocks = num_mel_attn_blocks
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attention_dropout = attention_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_proj_to_labels = summary_proj_to_labels
        self.summary_first_dropout = summary_first_dropout
        self.use_cache = use_cache
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.feature_size = feature_size
        self.use_attention_bias = use_attention_bias
        self.initializer_factor = initializer_factor
        self.decoder_fixing_codes = decoder_fixing_codes
        # 接受并处理未定义的额外参数
        self.update_from_kwargs(kwargs)
        ):
        # 初始化方法，接收多个参数来配置模型的各种属性
        self.vocab_size = vocab_size
        # 设置词汇表大小
        self.max_position_embeddings = max_position_embeddings
        # 设置最大位置编码长度
        self.max_text_tokens = max_text_tokens
        # 设置最大文本标记数
        self.hidden_size = hidden_size
        # 设置隐藏层大小
        self.num_hidden_layers = num_hidden_layers
        # 设置隐藏层数量
        self.num_attention_heads = num_attention_heads
        # 设置注意力头数
        self.n_inner = n_inner
        # 设置内部层大小
        self.num_mel_attn_blocks = num_mel_attn_blocks
        # 设置 MEL 注意力块数量
        self.activation_function = activation_function
        # 设置激活函数
        self.resid_pdrop = resid_pdrop
        # 设置残差连接丢弃率
        self.embd_pdrop = embd_pdrop
        # 设置嵌入层丢弃率
        self.attention_dropout = attention_dropout
        # 设置注意力丢弃率
        self.layer_norm_epsilon = layer_norm_epsilon
        # 设置层归一化的 epsilon 参数
        self.initializer_range = initializer_range
        # 设置初始化范围
        self.summary_type = summary_type
        # 设置摘要类型
        self.summary_use_proj = summary_use_proj
        # 设置是否使用摘要投影
        self.summary_activation = summary_activation
        # 设置摘要激活函数
        self.summary_first_dropout = summary_first_dropout
        # 设置摘要的首次丢弃率
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
        # 设置解码器修复码

        self.bos_token_id = bos_token_id
        # 设置起始标记 ID
        self.eos_token_id = eos_token_id
        # 设置结束标记 ID

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        # 调用父类初始化方法，传入起始和结束标记 ID 以及其他参数

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 类方法：从预训练模型名或路径创建配置对象，返回预训练配置对象

        cls._set_token_in_kwargs(kwargs)
        # 将 token 设置到 kwargs 中

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        # 获取配置字典和更新后的 kwargs 参数

        # 如果从 ClvpConfig 加载，则获取语音配置字典
        if config_dict.get("model_type") == "clvp":
            config_dict = config_dict["decoder_config"]

        # 如果配置字典中有模型类型，并且类具有 model_type 属性，并且模型类型不等于 cls.model_type，则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)
        # 从配置字典和 kwargs 参数创建配置对象并返回
# `ClvpConfig` 是存储 [`ClvpModelForConditionalGeneration`] 配置的类。
# 该配置类用于实例化 CLVP 模型，定义文本模型、语音模型和解码器模型的配置。
# 使用默认参数实例化配置对象将生成类似于 CLVP [susnato/clvp_dev](https://huggingface.co/susnato/clvp_dev) 架构的配置。
# 配置对象继承自 [`PretrainedConfig`]，用于控制模型输出。更多信息请参阅 [`PretrainedConfig`] 的文档。

class ClvpConfig(PretrainedConfig):
    model_type = "clvp"
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
        super().__init__(**kwargs)

        if text_config is None:
            text_config = {}
            # 如果未提供text_config参数，则使用默认空字典
            logger.info("`text_config` is `None`. Initializing the `ClvpEncoderConfig` with default values.")

        if speech_config is None:
            speech_config = {}
            # 如果未提供speech_config参数，则使用默认空字典
            logger.info("`speech_config` is `None`. initializing the `ClvpEncoderConfig` with default values.")

        if decoder_config is None:
            decoder_config = {}
            # 如果未提供decoder_config参数，则使用默认空字典
            logger.info("`decoder_config` is `None`. initializing the `ClvpDecoderConfig` with default values.")

        self.text_config = ClvpEncoderConfig(**text_config)
        # 初始化self.text_config，使用ClvpEncoderConfig类及其参数
        self.speech_config = ClvpEncoderConfig(**speech_config)
        # 初始化self.speech_config，使用ClvpEncoderConfig类及其参数
        self.decoder_config = ClvpDecoderConfig(**decoder_config)
        # 初始化self.decoder_config，使用ClvpDecoderConfig类及其参数

        self.projection_dim = projection_dim
        # 设置投影维度
        self.logit_scale_init_value = logit_scale_init_value
        # 设置logit缩放初始值
        self.initializer_factor = initializer_factor
        # 设置初始化因子

    @classmethod
    def from_sub_model_configs(
        cls,
        text_config: ClvpEncoderConfig,
        speech_config: ClvpEncoderConfig,
        decoder_config: ClvpDecoderConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`ClvpConfig`] (or a derived class) from CLVP text model configuration, CLVP speech model
        configuration and CLVP decoder model configuration.

        Args:
            text_config (`ClvpEncoderConfig`):
                Text model configuration of type [`ClvpEncoderConfig`].
            speech_config (`ClvpEncoderConfig`):
                Speech model configuration of type [`ClvpEncoderConfig`].
            decoder_config (`ClvpDecoderConfig`):
                Decoder model configuration of type [`ClvpDecoderConfig`].

        Returns:
            [`ClvpConfig`]: An instance of a configuration object
        """

        return cls(
            text_config=text_config.to_dict(),
            # 将text_config转换为字典形式传递给cls构造函数
            speech_config=speech_config.to_dict(),
            # 将speech_config转换为字典形式传递给cls构造函数
            decoder_config=decoder_config.to_dict(),
            # 将decoder_config转换为字典形式传递给cls构造函数
            **kwargs,
        )
```