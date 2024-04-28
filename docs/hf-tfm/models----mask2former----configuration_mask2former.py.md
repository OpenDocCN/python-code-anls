# `.\transformers\models\mask2former\configuration_mask2former.py`

```py
# coding=utf-8
# 声明编码格式和版权信息
# 版权归 Meta Platforms, Inc. 和 The HuggingFace Inc. 团队所有
# 根据 Apache License, Version 2.0 (Apache 2.0) 进行许可
# 这是一个 Mask2Former 模型配置文件

# 导入类型提示和其他依赖
from typing import Dict, List, Optional
# 从 transformers 库中导入配置和日志功能
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING

# Mask2Former 预训练模型配置文件的映射
MASK2FORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/mask2former-swin-small-coco-instance": (
        "https://huggingface.co/facebook/mask2former-swin-small-coco-instance/blob/main/config.json"
    )
    # 查看所有 Mask2Former 模型：https://huggingface.co/models?filter=mask2former
}

# 获取日志记录器
logger = logging.get_logger(__name__)

# Mask2Former 配置类，用于存储模型架构的配置
class Mask2FormerConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`Mask2FormerModel`] 模型的配置。它用于根据指定的参数实例化 Mask2Former 模型，并定义模型的架构。
    使用默认值实例化配置将会得到与 Mask2Former 模型
    [facebook/mask2former-swin-small-coco-instance](https://huggingface.co/facebook/mask2former-swin-small-coco-instance)
    架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，可以用于控制模型的输出。请阅读
    [`PretrainedConfig`] 的文档了解更多信息。

    目前，Mask2Former 仅支持 [Swin Transformer](swin)，作为其主干 (backbone)。

    用法示例:

    ```python
    >>> from transformers import Mask2FormerConfig, Mask2FormerModel

    >>> # 初始化一个 Mask2Former facebook/mask2former-swin-small-coco-instance 配置
    >>> configuration = Mask2FormerConfig()

    >>> # 从 facebook/mask2former-swin-small-coco-instance 风格的配置初始化一个（带有随机权重的）模型
    >>> model = Mask2FormerModel(configuration)

    >>> # 访问模型的配置
    >>> configuration = model.config
    ```py

    """

    model_type = "mask2former"
    # 支持的主干 (backbones) 类型列表
    backbones_supported = ["swin"]
    # 属性映射字典
    attribute_map = {"hidden_size": "hidden_dim"}
    # 初始化函数，用于创建一个新的实例
    def __init__(
        # backbone_config: Optional[Dict] = None，Backbone 模型的配置参数，默认为空字典
        self,
        backbone_config: Optional[Dict] = None,
        # 特征大小，默认为 256
        feature_size: int = 256,
        # 掩膜特征大小，默认为 256
        mask_feature_size: int = 256,
        # 隐藏层维度，默认为 256
        hidden_dim: int = 256,
        # 编码器前馈网络维度，默认为 1024
        encoder_feedforward_dim: int = 1024,
        # 激活函数，默认为 "relu"
        activation_function: str = "relu",
        # 编码器层数，默认为 6
        encoder_layers: int = 6,
        # 解码器层数，默认为 10
        decoder_layers: int = 10,
        # 注意力头数，默认为 8
        num_attention_heads: int = 8,
        # Dropout 概率，默认为 0.0
        dropout: float = 0.0,
        # 前馈网络维度，默认为 2048
        dim_feedforward: int = 2048,
        # 是否使用预层归一化，默认为 False
        pre_norm: bool = False,
        # 是否强制输入投影，默认为 False
        enforce_input_projection: bool = False,
        # 公共步长，默认为 4
        common_stride: int = 4,
        # 忽略值，默认为 255
        ignore_value: int = 255,
        # 查询数量，默认为 100
        num_queries: int = 100,
        # 无物体权重，默认为 0.1
        no_object_weight: float = 0.1,
        # 类别权重，默认为 2.0
        class_weight: float = 2.0,
        # 掩膜权重，默认为 5.0
        mask_weight: float = 5.0,
        # Dice 损失权重，默认为 5.0
        dice_weight: float = 5.0,
        # 训练点数量，默认为 12544
        train_num_points: int = 12544,
        # 过采样比例，默认为 3.0
        oversample_ratio: float = 3.0,
        # 重要性采样比例，默认为 0.75
        importance_sample_ratio: float = 0.75,
        # 初始化标准差，默认为 0.02
        init_std: float = 0.02,
        # 初始化 Xavier 标准差，默认为 1.0
        init_xavier_std: float = 1.0,
        # 是否使用辅助损失，默认为 True
        use_auxiliary_loss: bool = True,
        # 特征步长列表，默认为 [4, 8, 16, 32]
        feature_strides: List[int] = [4, 8, 16, 32],
        # 是否输出辅助日志，默认为 None
        output_auxiliary_logits: bool = None,
        # **kwargs，其他未指定参数的关键字参数
        **kwargs,
        # 如果没有传入backbone_config，则使用默认的'Swin'骨干网络配置
        if backbone_config is None:
            logger.info("`backbone_config` is `None`. Initializing the config with the default `Swin` backbone.")
            backbone_config = CONFIG_MAPPING["swin"](
                image_size=224,
                in_channels=3,
                patch_size=4,
                embed_dim=96,
                depths=[2, 2, 18, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                drop_path_rate=0.3,
                use_absolute_embeddings=False,
                out_features=["stage1", "stage2", "stage3", "stage4"],
            )

        # 如果backbone_config是一个字典，则将其转换为对应的配置类对象
        if isinstance(backbone_config, dict):
            backbone_model_type = backbone_config.pop("model_type")
            config_class = CONFIG_MAPPING[backbone_model_type]
            backbone_config = config_class.from_dict(backbone_config)

        # 验证骨干网络是否受支持
        if backbone_config.model_type not in self.backbones_supported:
            logger.warning_once(
                f"Backbone {backbone_config.model_type} is not a supported model and may not be compatible with Mask2Former. "
                f"Supported model types: {','.join(self.backbones_supported)}"
            )
        
        # 设置各种属性
        self.backbone_config = backbone_config
        self.feature_size = feature_size
        self.mask_feature_size = mask_feature_size
        self.hidden_dim = hidden_dim
        self.encoder_feedforward_dim = encoder_feedforward_dim
        self.activation_function = activation_function
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward
        self.pre_norm = pre_norm
        self.enforce_input_projection = enforce_input_projection
        self.common_stride = common_stride
        self.ignore_value = ignore_value
        self.num_queries = num_queries
        self.no_object_weight = no_object_weight
        self.class_weight = class_weight
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        self.train_num_points = train_num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.init_std = init_std
        self.init_xavier_std = init_xavier_std
        self.use_auxiliary_loss = use_auxiliary_loss
        self.feature_strides = feature_strides
        self.output_auxiliary_logits = output_auxiliary_logits
        self.num_hidden_layers = decoder_layers

        # 调用父类的构造函数
        super().__init__(**kwargs)

    # 类方法
    @classmethod
    def from_backbone_config(cls, backbone_config: PretrainedConfig, **kwargs):
        """从预训练的骨干模型配置实例化一个 `Mask2FormerConfig`（或其派生类）配置对象。

        Args:
            backbone_config ([`PretrainedConfig`]):
                骨干模型的配置对象。

        Returns:
            [`Mask2FormerConfig`]: 配置对象的一个实例
        """
        # 使用给定的骨干模型配置实例化一个 `Mask2FormerConfig` 对象
        return cls(
            backbone_config=backbone_config,
            **kwargs,
        )
```