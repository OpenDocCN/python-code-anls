# `.\transformers\models\sam\configuration_sam.py`

```
# 设置文件编码格式为utf-8
# 版权声明
# 根据Apache 2.0许可协议，除非符合许可协议要求或经书面同意，否则不得使用此文件
# 可以在以下链接获取许可协议的副本：http://www.apache.org/licenses/LICENSE-2.0
# 在适用法律要求或书面同意的情况下，本软件按“原样”分发，无任何明示或暗示的担保或条件
# 请参阅许可协议以了解特定语言的权限和限制
""" SAM模型配置 """

# 导入所需的模块
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# SAM预训练配置映射表
SAM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/sam-vit-huge": "https://huggingface.co/facebook/sam-vit-huge/resolve/main/config.json",
    "facebook/sam-vit-large": "https://huggingface.co/facebook/sam-vit-large/resolve/main/config.json",
    "facebook/sam-vit-base": "https://huggingface.co/facebook/sam-vit-base/resolve/main/config.json",
}

class SamPromptEncoderConfig(PretrainedConfig):
    # SAM Prompt编码器配置类
    # 用于存储SamPromptEncoder的配置
    # 参数将影响模型输出，可以查看PretrainedConfig的文档了解更多信息
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
        # 调用父类的构造方法，并传入参数
        super().__init__(**kwargs)
        # 初始化隐藏层大小
        self.hidden_size = hidden_size
        # 初始化图像大小
        self.image_size = image_size
        # 初始化补丁大小
        self.patch_size = patch_size
        # 初始化图像嵌入大小
        self.image_embedding_size = image_size // patch_size
        # 初始化掩码输入通道数
        self.mask_input_channels = mask_input_channels
        # 初始化点嵌入数量
        self.num_point_embeddings = num_point_embeddings
        # 初始化隐藏层激活函数
        self.hidden_act = hidden_act
        # 初始化层归一化 epsilon
        self.layer_norm_eps = layer_norm_eps
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

    # 定义 SamMaskDecoderConfig 类，用于存储 SamMaskDecoder 的配置信息
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

    # 初始化方法，设置配置参数的默认值
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
        # 设置各个配置参数的值
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
    # 这是用于存储 [`SamVisionModel`] 的配置的类。它用于根据指定参数实例化 SAM 视觉编码器，定义模型架构。实例化配置默认将产生与 SAM ViT-h [facebook/sam-vit-huge](https://huggingface.co/facebook/sam-vit-huge) 架构类似的配置。
    
    # 配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档获取更多信息。
    # 定义一个函数参数列表，描述了 Patch Transformer 模型的各种配置参数
    
    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            编码器层和汇聚层的维度。
        output_channels (`int`, *optional*, defaults to 256):
            Patch Encoder 输出通道的维度。
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Transformer 编码器中隐藏层的数量。
        num_attention_heads (`int`, *optional*, defaults to 12):
            Transformer 编码器中每个注意力层的注意头数。
        num_channels (`int`, *optional*, defaults to 3):
            输入图像的通道数。
        image_size (`int`, *optional*, defaults to 1024):
            期望的分辨率，调整后的输入图像的目标大小。
        patch_size (`int`, *optional*, defaults to 16):
            从输入图像中提取的补丁的大小。
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            非线性激活函数（函数或字符串）。
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            层归一化层使用的 epsilon。
        attention_dropout (`float`, *optional*, defaults to 0.0):
            注意力概率的 dropout 比率。
        initializer_range (`float`, *optional*, defaults to 1e-10):
            用于初始化所有权重矩阵的截断正态初始化器的标准差。
        qkv_bias (`bool`, *optional*, defaults to `True`):
            是否为查询、键、值投影添加偏置。
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            mlp 隐藏维度与嵌入维度的比率。
        use_abs_pos (`bool`, *optional*, defaults to `True`):
            是否使用绝对位置嵌入。
        use_rel_pos (`bool`, *optional*, defaults to `True`):
            是否使用相对位置嵌入。
        window_size (`int`, *optional*, defaults to 14):
            相对位置的窗口大小。
        global_attn_indexes (`List[int]`, *optional*, defaults to `[2, 5, 8, 11]`):
            全局注意力层的索引。
        num_pos_feats (`int`, *optional*, defaults to 128):
            位置嵌入的维度。
        mlp_dim (`int`, *optional*):
            Transformer 编码器中 MLP 层的维度。如果为 `None`，则默认为 `mlp_ratio * hidden_size`。
    ```  
    # 初始化函数，设置模型的各种参数
    def __init__(
        self,
        hidden_size=768,                     # 隐藏层的大小，默认为768
        output_channels=256,                 # 输出通道数，默认为256
        num_hidden_layers=12,                # 隐藏层的层数，默认为12
        num_attention_heads=12,              # 注意力头的数量，默认为12
        num_channels=3,                      # 通道数，默认为3
        image_size=1024,                     # 图像大小，默认为1024
        patch_size=16,                       # 图像块大小，默认为16
        hidden_act="gelu",                   # 隐藏层激活函数，默认为GELU
        layer_norm_eps=1e-06,                # Layer normalization 的 epsilon，默认为1e-06
        attention_dropout=0.0,               # 注意力层的dropout比例，默认为0.0
        initializer_range=1e-10,             # 初始化范围，默认为1e-10
        qkv_bias=True,                       # 是否在QKV矩阵中添加偏置，默认为True
        mlp_ratio=4.0,                       # MLP的隐藏层大小相对于hidden_size的倍数，默认为4.0
        use_abs_pos=True,                    # 是否使用绝对位置编码，默认为True
        use_rel_pos=True,                    # 是否使用相对位置编码，默认为True
        window_size=14,                      # 窗口大小，默认为14
        global_attn_indexes=[2, 5, 8, 11],   # 全局注意力层的索引，默认为[2, 5, 8, 11]
        num_pos_feats=128,                   # 位置编码的特征数量，默认为128
        mlp_dim=None,                        # MLP隐藏层大小，默认为None
        **kwargs,                            # 允许接受任意额外的关键字参数
    ):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 设置模型各项参数
        self.hidden_size = hidden_size
        self.output_channels = output_channels
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.qkv_bias = qkv_bias
        self.mlp_ratio = mlp_ratio
        self.use_abs_pos = use_abs_pos
        self.use_rel_pos = use_rel_pos
        self.window_size = window_size
        self.global_attn_indexes = global_attn_indexes
        self.num_pos_feats = num_pos_feats
        # 如果未指定MLP隐藏层大小，则设置为hidden_size乘以mlp_ratio
        self.mlp_dim = int(hidden_size * mlp_ratio) if mlp_dim is None else mlp_dim
class SamConfig(PretrainedConfig):
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
        prompt_encoder_config (Union[`dict`, `SamPromptEncoderConfig`], *optional`):
            Dictionary of configuration options used to initialize [`SamPromptEncoderConfig`].
        mask_decoder_config (Union[`dict`, `SamMaskDecoderConfig`], *optional`):
            Dictionary of configuration options used to initialize [`SamMaskDecoderConfig`].

        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
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

    # Define the type of model as "sam"
    model_type = "sam"

    # Constructor to initialize the configuration object
    def __init__(
        self,
        vision_config=None,  
        prompt_encoder_config=None, 
        mask_decoder_config=None, 
        initializer_range=0.02,  # Default value for the weight initializer range
        **kwargs,  # Additional optional keyword arguments
        ):
        # 调用父类的构造函数，传入关键字参数
        super().__init__(**kwargs)
        # 如果 vision_config 不为 None，则使用其值；否则使用空字典
        vision_config = vision_config if vision_config is not None else {}
        # 如果 prompt_encoder_config 不为 None，则使用其值；否则使用空字典
        prompt_encoder_config = prompt_encoder_config if prompt_encoder_config is not None else {}
        # 如果 mask_decoder_config 不为 None，则使用其值；否则使用空字典
        mask_decoder_config = mask_decoder_config if mask_decoder_config is not None else {}

        # 如果 vision_config 是 SamVisionConfig 的实例，则将其转换为字典
        if isinstance(vision_config, SamVisionConfig):
            vision_config = vision_config.to_dict()
        # 如果 prompt_encoder_config 是 SamPromptEncoderConfig 的实例，则将其转换为字典
        if isinstance(prompt_encoder_config, SamPromptEncoderConfig):
            prompt_encoder_config = prompt_encoder_config.to_dict()
        # 如果 mask_decoder_config 是 SamMaskDecoderConfig 的实例，则将其转换为字典
        if isinstance(mask_decoder_config, SamMaskDecoderConfig):
            mask_decoder_config = mask_decoder_config.to_dict()

        # 使用给定的 vision_config 创建 SamVisionConfig 实例
        self.vision_config = SamVisionConfig(**vision_config)
        # 使用给定的 prompt_encoder_config 创建 SamPromptEncoderConfig 实例
        self.prompt_encoder_config = SamPromptEncoderConfig(**prompt_encoder_config)
        # 使用给定的 mask_decoder_config 创建 SamMaskDecoderConfig 实例
        self.mask_decoder_config = SamMaskDecoderConfig(**mask_decoder_config)
        # 设置初始化范围
        self.initializer_range = initializer_range
```