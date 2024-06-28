# `.\models\sam\configuration_sam.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，指出版权归 HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本授权使用本文件，除非遵守许可证的条款，否则不得使用此文件
# 可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件是基于“原样”分发的，不提供任何形式的担保或条件，无论是明示的还是隐含的
# 请参阅许可证了解具体的法律规定
""" SAM 模型配置"""


# 从配置工具中导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 从工具包中导入日志记录模块
from ...utils import logging


# 获取名为 __name__ 的日志记录器
logger = logging.get_logger(__name__)

# 定义 SAM 预训练配置文件映射字典
SAM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/sam-vit-huge": "https://huggingface.co/facebook/sam-vit-huge/resolve/main/config.json",
    "facebook/sam-vit-large": "https://huggingface.co/facebook/sam-vit-large/resolve/main/config.json",
    "facebook/sam-vit-base": "https://huggingface.co/facebook/sam-vit-base/resolve/main/config.json",
}


# 定义 SamPromptEncoderConfig 类，继承自 PretrainedConfig
class SamPromptEncoderConfig(PretrainedConfig):
    r"""
    这是用于存储 [`SamPromptEncoder`] 配置的配置类。[`SamPromptEncoder`] 模块用于编码输入的 2D 点和边界框。
    实例化配置默认将生成与 SAM-vit-h
    [facebook/sam-vit-huge](https://huggingface.co/facebook/sam-vit-huge) 架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。有关更多信息，请阅读 [`PretrainedConfig`] 的文档。

    Args:
        hidden_size (`int`, *optional*, 默认为 256):
            隐藏状态的维度。
        image_size (`int`, *optional*, 默认为 1024):
            图像的预期输出分辨率。
        patch_size (`int`, *optional*, 默认为 16):
            每个补丁的大小（分辨率）。
        mask_input_channels (`int`, *optional*, 默认为 16):
            要馈送到 `MaskDecoder` 模块的通道数。
        num_point_embeddings (`int`, *optional*, 默认为 4):
            要使用的点嵌入数量。
        hidden_act (`str`, *optional*, 默认为 `"gelu"`):
            编码器和池化器中的非线性激活函数。
    """

    def __init__(
        self,
        hidden_size=256,
        image_size=1024,
        patch_size=16,
        mask_input_channels=16,
        num_point_embeddings=4,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        **kwargs,
        ):
        # 调用父类的构造函数，传递所有的关键字参数
        super().__init__(**kwargs)
        # 设置隐藏层大小
        self.hidden_size = hidden_size
        # 设置图像大小
        self.image_size = image_size
        # 设置补丁大小
        self.patch_size = patch_size
        # 计算图像嵌入大小，等于图像大小除以补丁大小
        self.image_embedding_size = image_size // patch_size
        # 设置掩码输入通道数
        self.mask_input_channels = mask_input_channels
        # 设置点嵌入数量
        self.num_point_embeddings = num_point_embeddings
        # 设置隐藏层激活函数
        self.hidden_act = hidden_act
        # 设置层归一化的 epsilon 值
        self.layer_norm_eps = layer_norm_eps
# `SamMaskDecoderConfig` 类，用于存储 `SamMaskDecoder` 的配置信息。
# 继承自 `PretrainedConfig`，用于控制模型输出。
# 该配置类用于实例化一个 `SamMaskDecoder`，定义模型的架构。
# 默认情况下，实例化配置类将生成类似于 `facebook/sam-vit-huge` 架构的配置。

class SamMaskDecoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SamMaskDecoder`]. It is used to instantiate a SAM
    mask decoder to the specified arguments, defining the model architecture. Instantiating a configuration defaults
    will yield a similar configuration to that of the SAM-vit-h
    [facebook/sam-vit-huge](https://huggingface.co/facebook/sam-vit-huge) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the hidden states.
        hidden_act (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function used inside the `SamMaskDecoder` module.
        mlp_dim (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 2):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        attention_downsample_rate (`int`, *optional*, defaults to 2):
            The downsampling rate of the attention layer.
        num_multimask_outputs (`int`, *optional*, defaults to 3):
            The number of outputs from the `SamMaskDecoder` module. In the Segment Anything paper, this is set to 3.
        iou_head_depth (`int`, *optional*, defaults to 3):
            The number of layers in the IoU head module.
        iou_head_hidden_dim (`int`, *optional*, defaults to 256):
            The dimensionality of the hidden states in the IoU head module.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.

    """
    
    # 初始化方法，用于设置配置参数
    def __init__(
        self,
        hidden_size=256,
        hidden_act="relu",
        mlp_dim=2048,
        num_hidden_layers=2,
        num_attention_heads=8,
        attention_downsample_rate=2,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        layer_norm_eps=1e-6,
        **kwargs,
    ):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 设置各个配置参数
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.mlp_dim = mlp_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_downsample_rate = attention_downsample_rate
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_head_depth = iou_head_depth
        self.iou_head_hidden_dim = iou_head_hidden_dim
        self.layer_norm_eps = layer_norm_eps


class SamVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SamVisionModel`]. It is used to instantiate a SAM
    vision encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
    defaults will yield a similar configuration to that of the SAM ViT-h
    [facebook/sam-vit-huge](https://huggingface.co/facebook/sam-vit-huge) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # 设置编码器层和池化层的维度大小，默认为768
    hidden_size (`int`, *optional*, defaults to 768):
    # Patch Encoder 中输出通道的维度大小，默认为256
    output_channels (`int`, *optional*, defaults to 256):
    # Transformer 编码器中隐藏层的数量，默认为12
    num_hidden_layers (`int`, *optional*, defaults to 12):
    # Transformer 编码器中每个注意力层的注意力头数，默认为12
    num_attention_heads (`int`, *optional*, defaults to 12):
    # 输入图像的通道数，默认为3
    num_channels (`int`, *optional*, defaults to 3):
    # 期望的输入图像分辨率，默认为1024
    image_size (`int`, *optional*, defaults to 1024):
    # 从输入图像中提取的补丁大小，默认为16
    patch_size (`int`, *optional*, defaults to 16):
    # 非线性激活函数的类型，默认为 "gelu"
    hidden_act (`str`, *optional*, defaults to `"gelu"`):
    # 层归一化层中使用的 epsilon 值，默认为 1e-06
    layer_norm_eps (`float`, *optional*, defaults to 1e-06):
    # 注意力概率的 dropout 比率，默认为0.0（不使用 dropout）
    attention_dropout (`float`, *optional*, defaults to 0.0):
    # 初始化所有权重矩阵的截断正态分布的标准差，默认为1e-10
    initializer_range (`float`, *optional*, defaults to 1e-10):
    # 是否向查询、键、值的投影中添加偏置，默认为 True
    qkv_bias (`bool`, *optional*, defaults to `True`):
    # MLP 隐藏层维度与嵌入维度之比，默认为4.0
    mlp_ratio (`float`, *optional*, defaults to 4.0):
    # 是否使用绝对位置编码，默认为 True
    use_abs_pos (`bool`, *optional*, defaults to `True`):
    # 是否使用相对位置编码，默认为 True
    use_rel_pos (`bool`, *optional*, defaults to `True`):
    # 相对位置的窗口大小，默认为14
    window_size (`int`, *optional*, defaults to 14):
    # 全局注意力层的索引列表，默认为 `[2, 5, 8, 11]`
    global_attn_indexes (`List[int]`, *optional*, defaults to `[2, 5, 8, 11]`):
    # 位置嵌入的维度大小，默认为128
    num_pos_feats (`int`, *optional*, defaults to 128):
    # Transformer 编码器中 MLP 层的维度大小。如果为 `None`，则默认为 `mlp_ratio * hidden_size`
    mlp_dim (`int`, *optional*):
    # 初始化函数，设置Transformer模型的各项参数
    def __init__(
        self,
        hidden_size=768,                  # 隐藏层大小，默认为768
        output_channels=256,              # 输出通道数，默认为256
        num_hidden_layers=12,             # 隐藏层的数量，默认为12
        num_attention_heads=12,           # 注意力头的数量，默认为12
        num_channels=3,                   # 输入图像的通道数，默认为3（RGB）
        image_size=1024,                  # 输入图像的大小，默认为1024x1024像素
        patch_size=16,                    # 图像分块的大小，默认为16x16像素
        hidden_act="gelu",                # 隐藏层激活函数，默认为GELU
        layer_norm_eps=1e-06,             # Layer Normalization的epsilon，默认为1e-06
        attention_dropout=0.0,            # 注意力机制的dropout率，默认为0.0（不使用dropout）
        initializer_range=1e-10,          # 参数初始化的范围，默认为1e-10
        qkv_bias=True,                    # 是否在QKV矩阵中使用偏置，默认为True
        mlp_ratio=4.0,                    # MLP的维度扩展比例，默认为4.0
        use_abs_pos=True,                 # 是否使用绝对位置编码，默认为True
        use_rel_pos=True,                 # 是否使用相对位置编码，默认为True
        window_size=14,                   # 局部注意力窗口大小，默认为14
        global_attn_indexes=[2, 5, 8, 11], # 全局注意力层的索引，默认为[2, 5, 8, 11]
        num_pos_feats=128,                # 位置特征的数量，默认为128
        mlp_dim=None,                     # MLP的维度，默认为hidden_size * mlp_ratio，若给定mlp_dim则使用给定值
        **kwargs,                         # 其他未指定的参数
    ):
        super().__init__(**kwargs)        # 调用父类的初始化方法
    
        self.hidden_size = hidden_size    # 设置隐藏层大小属性
        self.output_channels = output_channels  # 设置输出通道数属性
        self.num_hidden_layers = num_hidden_layers  # 设置隐藏层数量属性
        self.num_attention_heads = num_attention_heads  # 设置注意力头数量属性
        self.num_channels = num_channels  # 设置输入图像通道数属性
        self.image_size = image_size      # 设置输入图像大小属性
        self.patch_size = patch_size      # 设置图像分块大小属性
        self.hidden_act = hidden_act      # 设置隐藏层激活函数属性
        self.layer_norm_eps = layer_norm_eps  # 设置Layer Normalization的epsilon属性
        self.attention_dropout = attention_dropout  # 设置注意力dropout率属性
        self.initializer_range = initializer_range  # 设置参数初始化范围属性
        self.qkv_bias = qkv_bias          # 设置是否使用QKV偏置属性
        self.mlp_ratio = mlp_ratio        # 设置MLP维度扩展比例属性
        self.use_abs_pos = use_abs_pos    # 设置是否使用绝对位置编码属性
        self.use_rel_pos = use_rel_pos    # 设置是否使用相对位置编码属性
        self.window_size = window_size    # 设置局部注意力窗口大小属性
        self.global_attn_indexes = global_attn_indexes  # 设置全局注意力层的索引属性
        self.num_pos_feats = num_pos_feats  # 设置位置特征数量属性
        self.mlp_dim = int(hidden_size * mlp_ratio) if mlp_dim is None else mlp_dim  # 设置MLP的维度属性，如果mlp_dim未指定则计算为hidden_size * mlp_ratio
# 定义 `SamConfig` 类，用于存储 `SamModel` 的配置信息，继承自 `PretrainedConfig`。
class SamConfig(PretrainedConfig):
    # 文档字符串，描述了 `SamConfig` 的作用和用法，以及如何实例化 SAM 模型的相关参数。
    r"""
    [`SamConfig`] is the configuration class to store the configuration of a [`SamModel`]. It is used to instantiate a
    SAM model according to the specified arguments, defining the vision model, prompt-encoder model and mask decoder
    configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the
    SAM-ViT-H [facebook/sam-vit-huge](https://huggingface.co/facebook/sam-vit-huge) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (Union[`dict`, `SamVisionConfig`], *optional*):
            Dictionary of configuration options used to initialize [`SamVisionConfig`].
        prompt_encoder_config (Union[`dict`, `SamPromptEncoderConfig`], *optional*):
            Dictionary of configuration options used to initialize [`SamPromptEncoderConfig`].
        mask_decoder_config (Union[`dict`, `SamMaskDecoderConfig`], *optional*):
            Dictionary of configuration options used to initialize [`SamMaskDecoderConfig`].

        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```
    >>> from transformers import (
    ...     SamVisionConfig,
    ...     SamPromptEncoderConfig,
    ...     SamMaskDecoderConfig,
    ...     SamModel,
    ... )

    >>> # Initializing a SamConfig with `"facebook/sam-vit-huge"` style configuration
    >>> configuration = SamConfig()

    >>> # Initializing a SamModel (with random weights) from the `"facebook/sam-vit-huge"` style configuration
    >>> model = SamModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a SamConfig from a SamVisionConfig, SamPromptEncoderConfig, and SamMaskDecoderConfig

    >>> # Initializing SAM vision, SAM Q-Former and language model configurations
    >>> vision_config = SamVisionConfig()
    >>> prompt_encoder_config = SamPromptEncoderConfig()
    >>> mask_decoder_config = SamMaskDecoderConfig()

    >>> config = SamConfig(vision_config, prompt_encoder_config, mask_decoder_config)
    ```"""

    # 类属性 `model_type`，指定模型类型为 "sam"。
    model_type = "sam"

    # 构造函数 `__init__`，用于初始化 `SamConfig` 类的实例。
    def __init__(
        self,
        vision_config=None,
        prompt_encoder_config=None,
        mask_decoder_config=None,
        initializer_range=0.02,
        **kwargs,
        ):
            # 调用父类的构造方法，传递所有的关键字参数
            super().__init__(**kwargs)
            # 如果 vision_config 不为 None，则使用其值；否则使用空字典
            vision_config = vision_config if vision_config is not None else {}
            # 如果 prompt_encoder_config 不为 None，则使用其值；否则使用空字典
            prompt_encoder_config = prompt_encoder_config if prompt_encoder_config is not None else {}
            # 如果 mask_decoder_config 不为 None，则使用其值；否则使用空字典
            mask_decoder_config = mask_decoder_config if mask_decoder_config is not None else {}

            # 如果 vision_config 是 SamVisionConfig 类的实例，则转换为字典
            if isinstance(vision_config, SamVisionConfig):
                vision_config = vision_config.to_dict()
            # 如果 prompt_encoder_config 是 SamPromptEncoderConfig 类的实例，则转换为字典
            if isinstance(prompt_encoder_config, SamPromptEncoderConfig):
                prompt_encoder_config = prompt_encoder_config.to_dict()
            # 如果 mask_decoder_config 是 SamMaskDecoderConfig 类的实例，则转换为字典
            if isinstance(mask_decoder_config, SamMaskDecoderConfig):
                mask_decoder_config = mask_decoder_config.to_dict()

            # 使用 vision_config 字典创建 SamVisionConfig 对象
            self.vision_config = SamVisionConfig(**vision_config)
            # 使用 prompt_encoder_config 字典创建 SamPromptEncoderConfig 对象
            self.prompt_encoder_config = SamPromptEncoderConfig(**prompt_encoder_config)
            # 使用 mask_decoder_config 字典创建 SamMaskDecoderConfig 对象
            self.mask_decoder_config = SamMaskDecoderConfig(**mask_decoder_config)
            # 设置 initializer_range 实例变量
            self.initializer_range = initializer_range
```