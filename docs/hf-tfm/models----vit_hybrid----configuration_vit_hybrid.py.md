# `.\transformers\models\vit_hybrid\configuration_vit_hybrid.py`

```py
# coding=utf-8
# 以上为 Python 文件编码声明和版权信息

# 引入配置基类 PretrainedConfig 和日志工具 logging
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 从自动配置模块导入配置映射和 BitConfig 类
from ..auto.configuration_auto import CONFIG_MAPPING
from ..bit import BitConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# ViT Hybrid 预训练配置文件映射表
VIT_HYBRID_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/vit-hybrid-base-bit-384": "https://huggingface.co/vit-hybrid-base-bit-384/resolve/main/config.json",
    # 查看所有 ViT 混合模型，请访问 https://huggingface.co/models?filter=vit
}

# ViT Hybrid 模型配置类
class ViTHybridConfig(PretrainedConfig):
    r"""
    这是用于存储 [`ViTHybridModel`] 配置的配置类。它用于根据指定的参数实例化一个 ViT Hybrid 模型，定义模型架构。
    使用默认值实例化一个配置将产生类似于 ViT Hybrid [google/vit-hybrid-base-bit-384](https://huggingface.co/google/vit-hybrid-base-bit-384) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。请阅读 [`PretrainedConfig`] 的文档以获取更多信息。
    """
    pass
    Args:
        backbone_config (`Union[Dict[str, Any], PretrainedConfig]`, *optional*):
            The configuration of the backbone in a dictionary or the config object of the backbone.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 1):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        backbone_featmap_shape (`List[int]`, *optional*, defaults to `[1, 1024, 24, 24]`):
            Used only for the `hybrid` embedding type. The shape of the feature maps of the backbone.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.

    Example:

    ```python
    >>> from transformers import ViTHybridConfig, ViTHybridModel

    >>> # Initializing a ViT Hybrid vit-hybrid-base-bit-384 style configuration
    >>> configuration = ViTHybridConfig()

    >>> # Initializing a model (with random weights) from the vit-hybrid-base-bit-384 style configuration
    >>> model = ViTHybridModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```py"""

    # 设置模型类型为 "vit-hybrid"
    model_type = "vit-hybrid"
    # 初始化函数，设置模型各种配置参数
    def __init__(
        self,
        backbone_config=None,  # 模型的骨干网络配置
        hidden_size=768,  # 隐藏层的大小
        num_hidden_layers=12,  # 隐藏层数量
        num_attention_heads=12,  # 多头注意力的头数
        intermediate_size=3072,  # 中间层的大小
        hidden_act="gelu",  # 隐藏层激活函数
        hidden_dropout_prob=0.0,  # 隐藏层的Dropout概率
        attention_probs_dropout_prob=0.0,  # 注意力的Dropout概率
        initializer_range=0.02,  # 初始化的范围
        layer_norm_eps=1e-12,  # LayerNormalization层的epsilon值
        image_size=224,  # 图片的大小
        patch_size=1,  # Patch的大小
        num_channels=3,  # 图片的通道数
        backbone_featmap_shape=[1, 1024, 24, 24],  # 骨干网络的特征图形状
        qkv_bias=True,  # Query、Key和Value是否包含偏置
        **kwargs,  # 其他参数
    ):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 如果没有给定骨干网络配置，则使用默认的BiT骨干网络配置
        if backbone_config is None:
            logger.info("`backbone_config` is `None`. Initializing the config with a `BiT` backbone.")
            backbone_config = {
                "global_padding": "same",
                "layer_type": "bottleneck",
                "depths": [3, 4, 9],
                "out_features": ["stage3"],
                "embedding_dynamic_padding": True,
            }

        # 如果给定的骨干网络配置是字典类型
        if isinstance(backbone_config, dict):
            # 如果字典中包含"model_type"键
            if "model_type" in backbone_config:
                backbone_config_class = CONFIG_MAPPING[backbone_config["model_type"]]
            else:
                logger.info(
                    "`model_type` is not found in `backbone_config`. Use `Bit` as the backbone configuration class."
                )
                backbone_config_class = BitConfig
            # 使用骨干网络配置类初始化骨干网络配置
            backbone_config = backbone_config_class(**backbone_config)

        # 设置各个参数值
        self.backbone_featmap_shape = backbone_featmap_shape
        self.backbone_config = backbone_config
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
```