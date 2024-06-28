# `.\models\mvp\configuration_mvp.py`

```py
# coding=utf-8
# 代码文件的版权声明和许可证信息

""" MVP model configuration"""
# 导入警告模块
import warnings

# 导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 导入日志记录工具
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# MVP预训练配置文件映射
MVP_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "RUCAIBox/mvp": "https://huggingface.co/RUCAIBox/mvp/resolve/main/config.json",
}

# MVP配置类，继承自预训练配置类
class MvpConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MvpModel`]. It is used to instantiate a MVP model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the MVP [RUCAIBox/mvp](https://huggingface.co/RUCAIBox/mvp)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```
    >>> from transformers import MvpConfig, MvpModel

    >>> # Initializing a MVP RUCAIBox/mvp style configuration
    >>> configuration = MvpConfig()

    >>> # Initializing a model (with random weights) from the RUCAIBox/mvp style configuration
    >>> model = MvpModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # 模型类型为MVP
    model_type = "mvp"
    # 推断阶段忽略的键
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射字典
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    def __init__(
        self,
        vocab_size=50267,
        max_position_embeddings=1024,
        encoder_layers=12,
        encoder_ffn_dim=4096,
        encoder_attention_heads=16,
        decoder_layers=12,
        decoder_ffn_dim=4096,
        decoder_attention_heads=16,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        activation_function="gelu",
        d_model=1024,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        classifier_dropout=0.0,
        scale_embedding=False,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        is_encoder_decoder=True,
        decoder_start_token_id=2,
        forced_eos_token_id=2,
        use_prompt=False,
        prompt_length=100,
        prompt_mid_dim=800,
        **kwargs,
    ):
        # 初始化方法，用于设置配置参数
        # 调用父类的初始化方法
        super().__init__(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            encoder_layers=encoder_layers,
            encoder_ffn_dim=encoder_ffn_dim,
            encoder_attention_heads=encoder_attention_heads,
            decoder_layers=decoder_layers,
            decoder_ffn_dim=decoder_ffn_dim,
            decoder_attention_heads=decoder_attention_heads,
            encoder_layerdrop=encoder_layerdrop,
            decoder_layerdrop=decoder_layerdrop,
            activation_function=activation_function,
            d_model=d_model,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            init_std=init_std,
            classifier_dropout=classifier_dropout,
            scale_embedding=scale_embedding,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            use_prompt=use_prompt,
            prompt_length=prompt_length,
            prompt_mid_dim=prompt_mid_dim,
            **kwargs,
        )
        ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding  # 如果为 True，则缩放因子为 sqrt(d_model)
        self.use_prompt = use_prompt
        self.prompt_length = prompt_length
        self.prompt_mid_dim = prompt_mid_dim

        # 调用父类的初始化方法，传入配置参数
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )

        # 如果 forced_bos_token_id 未设置且 force_bos_token_to_be_generated 在 kwargs 中为 True，则使用 bos_token_id
        if self.forced_bos_token_id is None and kwargs.get("force_bos_token_to_be_generated", False):
            self.forced_bos_token_id = self.bos_token_id
            # 发出警告，提醒将来版本需要在配置中包含 forced_bos_token_id
            warnings.warn(
                f"Please make sure the config includes `forced_bos_token_id={self.bos_token_id}` in future versions. "
                "The config can simply be saved and uploaded again to be fixed."
            )
```