# `.\models\idefics\configuration_idefics.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 EleutherAI 和 HuggingFace Inc. 团队所有
# 该代码基于 EleutherAI 的 GPT-NeoX 库以及该库中的 GPT-NeoX 和 OPT 实现。已经根据与 Meta AI 团队训练模型时的微小架构差异进行了修改
# 根据 Apache 许可证版本 2.0 进行许可
# 除非符合许可证要求或经书面同意，否则不得使用此文件
# 可以在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证副本
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的，没有任何形式的担保或条件，无论是明示的还是暗示的
# 请查看许可证以获取有关特定语言的权限和限制
""" Idefics model configuration"""

# 导入必要的模块和类
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义预训练配置文件映射
IDEFICS_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "HuggingFaceM4/idefics-9b": "https://huggingface.co/HuggingFaceM4/idefics-9b/blob/main/config.json",
    "HuggingFaceM4/idefics-80b": "https://huggingface.co/HuggingFaceM4/idefics-80b/blob/main/config.json",
}

# 定义 IdeficsVisionConfig 类，用于存储 IdeficsModel 的配置
class IdeficsVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`IdeficsModel`]. It is used to instantiate an
    Idefics model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Idefics-9B.

    e.g. [HuggingFaceM4/idefics-9b](https://huggingface.co/HuggingFaceM4/idefics-9b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer. (elsewhere referred to as `hidden_size`)
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        intermediate_size (`int`, *optional*, defaults to 5120):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        image_num_channels (`int`, *optional*, defaults to `3`):
            Number of image channels.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1.0, used internally for initialization
            testing).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    """

    model_type = "idefics"
    attribute_map = {
        "hidden_size": "embed_dim",
    }

    # 初始化类的构造函数
    def __init__(
        self,
        embed_dim=768,
        image_size=224,
        intermediate_size=5120,
        patch_size=14,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_channels=3,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        **kwargs,
        ):
        # 初始化模型的各种参数
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.intermediate_size = intermediate_size
        self.patch_size = patch_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.hidden_act = hidden_act

        # 调用父类的初始化方法
        super().__init__(**kwargs)
# 定义一个配置类，用于存储 IdeficsModel 的配置信息，用于实例化一个 Idefics 模型
class IdeficsPerceiverConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`IdeficsModel`]. It is used to instantiate an
    Idefics model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Idefics-9B.

    e.g. [HuggingFaceM4/idefics-9b](https://huggingface.co/HuggingFaceM4/idefics-9b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        use_resampler (`bool`, *optional*, defaults to `False`):
            Whether or not to use the resampler
        resampler_n_latents (`int`, *optional*, defaults to ):
            Number of latent embeddings to resample ("compress") the input sequence to (usually < 128).
        resampler_depth (`int`, *optional*, defaults to 6):
            Depth of the Perceiver Resampler (Transformer w/ cross attention). Should be shallow (< 3).
        resampler_n_heads (`int`, *optional*, defaults to 16):
            Number of heads in each Transformer block (for multi-headed self-attention).
        resampler_head_dim (`int`, *optional*, defaults to 96):
            Dimensionality of each head projection in the Transformer block.
        qk_layer_norms_perceiver (`bool`, *optional*, defaults to `False`):
            Whether or not to use qk layer norms in perceiver
    """

    model_type = "idefics"

    # 初始化方法，接受一系列参数
    def __init__(
        self,
        use_resampler=False,
        resampler_n_latents=64,
        resampler_depth=6,
        resampler_n_heads=16,
        resampler_head_dim=96,
        qk_layer_norms_perceiver=False,
        **kwargs,
    ):
        # 将参数赋值给对象的属性
        self.use_resampler = use_resampler
        self.resampler_n_latents = resampler_n_latents
        self.resampler_depth = resampler_depth
        self.resampler_n_heads = resampler_n_heads
        self.resampler_head_dim = resampler_head_dim
        self.qk_layer_norms_perceiver = qk_layer_norms_perceiver

        # 调用父类的初始化方法
        super().__init__(**kwargs)


# 定义一个配置类，用于存储 IdeficsModel 的配置信息，用于实例化一个 Idefics 模型
class IdeficsConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`IdeficsModel`]. It is used to instantiate an
    Idefics model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Idefics-9B.

    e.g. [HuggingFaceM4/idefics-9b](https://huggingface.co/HuggingFaceM4/idefics-9b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```python
    >>> from transformers import IdeficsModel, IdeficsConfig

    >>> # Initializing a Idefics idefics-9b style configuration
    >>> configuration = IdeficsConfig()
    # 从 idefics-9b 风格的配置初始化一个模型
    model = IdeficsModel(configuration)

    # 访问模型配置
    configuration = model.config
    ```py

    # 定义模型类型为 "idefics"，不是组合模型
    model_type = "idefics"
    is_composition = False

    # 初始化函数，设置模型的各种参数
    def __init__(
        self,
        vocab_size=32000,
        additional_vocab_size=0,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        dropout=0.0,
        hidden_act="silu",
        initializer_range=0.02,
        alpha_initializer="zeros",
        alphas_initializer_range=0.0,
        alpha_type="float",
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        cross_layer_interval=1,
        qk_layer_norms=False,
        freeze_text_layers=True,
        freeze_text_module_exceptions=[],
        freeze_lm_head=False,
        freeze_vision_layers=True,
        freeze_vision_module_exceptions=[],
        use_resampler=False,
        vision_config=None,
        perceiver_config=None,
        **kwargs,
        self.vocab_size = vocab_size
        self.additional_vocab_size = additional_vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.alpha_initializer = alpha_initializer
        self.alphas_initializer_range = alphas_initializer_range
        self.alpha_type = alpha_type
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache

        self.cross_layer_interval = cross_layer_interval
        self.qk_layer_norms = qk_layer_norms
        self.freeze_vision_layers = freeze_vision_layers

        self.freeze_text_layers = freeze_text_layers
        self.freeze_text_module_exceptions = freeze_text_module_exceptions
        self.freeze_vision_module_exceptions = freeze_vision_module_exceptions
        self.freeze_lm_head = freeze_lm_head

        self.use_resampler

        # 如果 perceiver_config 为 None，则使用默认配置
        if perceiver_config is None:
            self.perceiver_config = IdeficsPerceiverConfig()
        # 如果 perceiver_config 是字典类型，则使用给定的配置
        elif isinstance(perceiver_config, dict):
            self.perceiver_config = IdeficsPerceiverConfig(**perceiver_config)
        # 如果 perceiver_config 是 IdeficsPerceiverConfig 类型，则直接使用
        elif isinstance(perceiver_config, IdeficsPerceiverConfig):
            self.perceiver_config = perceiver_config

        # 如果 vision_config 为 None，则使用默认配置
        if vision_config is None:
            self.vision_config = IdeficsVisionConfig()
        # 如果 vision_config 是字典类型，则使用给定的配置
        elif isinstance(vision_config, dict):
            self.vision_config = IdeficsVisionConfig(**vision_config)
        # 如果 vision_config 是 IdeficsVisionConfig 类型，则直接使用
        elif isinstance(vision_config, IdeficsVisionConfig):
            self.vision_config = vision_config

        # 调用父类的初始化方法，传入参数
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # 重要提示：不要在构造函数中进行任何基于 __init__ 参数的检查，因为 PretrainedConfig.from_dict 首先使用配置字典实例化类，
        # 然后才使用 from_pretrained 中的 `kwargs` 更新配置对象，因此在实例化此对象时，许多属性具有默认值，尚未被覆盖。
        # 在运行父类的 `from_pretrained` 后，在 `from_pretrained` 内部执行任何必要的检查。
```