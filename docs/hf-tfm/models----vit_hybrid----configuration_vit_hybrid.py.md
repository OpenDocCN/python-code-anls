# `.\models\vit_hybrid\configuration_vit_hybrid.py`

```py
# 设置文件编码为 UTF-8
# 版权声明及许可条款
# 根据 Apache 许可证 2.0 版本使用此文件
# 可以在符合许可证条件的情况下使用该文件
# 许可证详细信息请参见 http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按"原样"分发本软件
# 没有任何形式的明示或暗示担保或条件
# 请参阅许可证了解具体的语言权限以及限制条件
""" ViT Hybrid model configuration"""

# 从相关模块导入必要的类和函数
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import CONFIG_MAPPING
from ..bit import BitConfig

# 获取一个用于记录日志的 logger 对象
logger = logging.get_logger(__name__)

# 预训练配置文件的映射，指定了各种预训练模型的下载地址
VIT_HYBRID_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/vit-hybrid-base-bit-384": "https://huggingface.co/vit-hybrid-base-bit-384/resolve/main/config.json",
    # 查看所有 ViT 混合模型的完整列表请访问 https://huggingface.co/models?filter=vit
}

# ViT Hybrid 配置类，继承自 PretrainedConfig 类
class ViTHybridConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ViTHybridModel`]. It is used to instantiate a ViT
    Hybrid model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the ViT Hybrid
    [google/vit-hybrid-base-bit-384](https://huggingface.co/google/vit-hybrid-base-bit-384) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```
    >>> from transformers import ViTHybridConfig, ViTHybridModel

    >>> # Initializing a ViT Hybrid vit-hybrid-base-bit-384 style configuration
    >>> configuration = ViTHybridConfig()

    >>> # Initializing a model (with random weights) from the vit-hybrid-base-bit-384 style configuration
    >>> model = ViTHybridModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    # 指定模型类型为 "vit-hybrid"
    model_type = "vit-hybrid"

    # 初始化方法，定义了该配置类的各种参数
    def __init__(
        self,
        backbone_config=None,
        backbone=None,
        use_pretrained_backbone=False,
        use_timm_backbone=False,
        backbone_kwargs=None,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=224,
        patch_size=1,
        num_channels=3,
        backbone_featmap_shape=[1, 1024, 24, 24],
        qkv_bias=True,
        **kwargs,
        ):
        super().__init__(**kwargs)
        # 如果使用预训练的主干网络，则抛出异常
        if use_pretrained_backbone:
            raise ValueError("Pretrained backbones are not supported yet.")

        # 如果同时指定了 `backbone` 和 `backbone_config`，则抛出异常
        if backbone_config is not None and backbone is not None:
            raise ValueError("You can't specify both `backbone` and `backbone_config`.")

        # 如果 `backbone_config` 和 `backbone` 都未指定，则使用默认的 `BiT` 主干网络配置
        if backbone_config is None and backbone is None:
            logger.info("`backbone_config` is `None`. Initializing the config with a `BiT` backbone.")
            backbone_config = {
                "global_padding": "same",
                "layer_type": "bottleneck",
                "depths": [3, 4, 9],
                "out_features": ["stage3"],
                "embedding_dynamic_padding": True,
            }

        # 如果同时指定了 `backbone_kwargs` 和 `backbone_config`，则抛出异常
        if backbone_kwargs is not None and backbone_kwargs and backbone_config is not None:
            raise ValueError("You can't specify both `backbone_kwargs` and `backbone_config`.")

        # 如果 `backbone_config` 是一个字典，则根据 `model_type` 创建对应的主干网络配置类
        if isinstance(backbone_config, dict):
            if "model_type" in backbone_config:
                backbone_config_class = CONFIG_MAPPING[backbone_config["model_type"]]
            else:
                logger.info(
                    "`model_type` is not found in `backbone_config`. Use `Bit` as the backbone configuration class."
                )
                backbone_config_class = BitConfig
            backbone_config = backbone_config_class(**backbone_config)

        # 设置类的属性值
        self.backbone_featmap_shape = backbone_featmap_shape
        self.backbone_config = backbone_config
        self.backbone = backbone
        self.use_pretrained_backbone = use_pretrained_backbone
        self.use_timm_backbone = use_timm_backbone
        self.backbone_kwargs = backbone_kwargs
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