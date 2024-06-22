# `.\models\deformable_detr\configuration_deformable_detr.py`

```py
# coding=utf-8
# 声明文件编码为 UTF-8
# 版权声明
# SenseTime 和 The HuggingFace Inc. 团队保留所有权利
# 在 Apache 许可证，版本 2.0 下授权
# 除非遵守许可证，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按“原样”分发的软件
# 没有任何形式的担保或条件，无论是明示的还是暗示的
# 查看特定语言的许可证权限和限制
# Deformable DETR 模型配置

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING

# 获取日志记录器
logger = logging.get_logger(__name__)

# Deformable DETR 预训练配置档案映射
DEFORMABLE_DETR_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "SenseTime/deformable-detr": "https://huggingface.co/sensetime/deformable-detr/resolve/main/config.json",
    # 查看所有 Deformable DETR 模型 https://huggingface.co/models?filter=deformable-detr
}

# DeformableDetrConfig 类继承自 PretrainedConfig
class DeformableDetrConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DeformableDetrModel`]. It is used to instantiate
    a Deformable DETR model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Deformable DETR
    [SenseTime/deformable-detr](https://huggingface.co/SenseTime/deformable-detr) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Examples:

    ```python
    >>> from transformers import DeformableDetrConfig, DeformableDetrModel

    >>> # Initializing a Deformable DETR SenseTime/deformable-detr style configuration
    >>> configuration = DeformableDetrConfig()

    >>> # Initializing a model (with random weights) from the SenseTime/deformable-detr style configuration
    >>> model = DeformableDetrModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```py"""

    # 模型类型
    model_type = "deformable_detr"
    # 属性映射关系
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
    }
    # 初始化函数，设置模型参数和参数默认值
    def __init__(
        self,
        use_timm_backbone=True,  # 是否使用timm模块提供的backbone，默认为True
        backbone_config=None,  # backbone配置，默认为None
        num_channels=3,  # 输入通道数，默认为3
        num_queries=300,  # 查询数量，默认为300
        max_position_embeddings=1024,  # 最大位置嵌入数量，默认为1024
        encoder_layers=6,  # 编码器层数，默认为6
        encoder_ffn_dim=1024,  # 编码器前馈网络神经元数目，默认为1024
        encoder_attention_heads=8,  # 编码器注意力头数，默认为8
        decoder_layers=6,  # 解码器层数，默认为6
        decoder_ffn_dim=1024,  # 解码器前馈网络神经元数目，默认为1024
        decoder_attention_heads=8,  # 解码器注意力头数，默认为8
        encoder_layerdrop=0.0,  # 编码器层丢弃概率，默认为0.0
        is_encoder_decoder=True,  # 是否是编码-解码模型，默认为True
        activation_function="relu",  # 激活函数，默认为"relu"
        d_model=256,  # 模型维度，默认为256
        dropout=0.1,  # 失活概率，默认为0.1
        attention_dropout=0.0,  # 注意力失活概率，默认为0.0
        activation_dropout=0.0,  # 激活函数失活概率，默认为0.0
        init_std=0.02,  # 初始化标准差，默认为0.02
        init_xavier_std=1.0,  # 初始化Xavier标准差，默认为1.0
        return_intermediate=True,  # 是否返回中间结果，默认为True
        auxiliary_loss=False,  # 是否辅助损失，默认为False
        position_embedding_type="sine",  # 位置嵌入类型，默认为"sine"
        backbone="resnet50",  # backbone类型，默认为"resnet50"
        use_pretrained_backbone=True,  # 是否使用预训练的backbone，默认为True
        dilation=False,  # 是否膨胀，默认为False
        num_feature_levels=4,  # 特征级数目，默认为4
        encoder_n_points=4,  # 编码器特征点数目，默认为4
        decoder_n_points=4,  # 解码器特征点数目，默认为4
        two_stage=False,  # 是否两阶段，默认为False
        two_stage_num_proposals=300,  # 两阶段提议数量，默认为300
        with_box_refine=False,  # 是否有盒子细化，默认为False
        class_cost=1,  # 分类损失系数，默认为1
        bbox_cost=5,  # 边界盒损失系数，默认为5
        giou_cost=2,  # giou损失系数，默认为2
        mask_loss_coefficient=1,  # 掩码损失系数，默认为1
        dice_loss_coefficient=1,  # dice损失系数，默认为1
        bbox_loss_coefficient=5,  # 边界盒损失系数，默认为5
        giou_loss_coefficient=2,  # giou损失系数，默认为2
        eos_coefficient=0.1,  # eos损失系数，默认为0.1
        focal_alpha=0.25,  # 焦点α，默认为0.25
        disable_custom_kernels=False,  # 是否禁用自定义内核，默认为False
        **kwargs,  # 其他参数
    ):

        # 检查是否同时指定了 `backbone_config` 和 `use_timm_backbone`
        if backbone_config is not None and use_timm_backbone:
            raise ValueError("You can't specify both `backbone_config` and `use_timm_backbone`.")

        # 如果不使用 timm backbone
        if not use_timm_backbone:

            # 如果没有指定 backbone_config，则使用默认的 ResNet backbone
            if backbone_config is None:
                logger.info("`backbone_config` is `None`. Initializing the config with the default `ResNet` backbone.")
                backbone_config = CONFIG_MAPPING["resnet"](out_features=["stage4"])
            
            # 如果 backbone_config 是字典类型，根据 model_type 获得对应的 config_class，并根据字典初始化 backbone_config
            elif isinstance(backbone_config, dict):
                backbone_model_type = backbone_config.get("model_type")
                config_class = CONFIG_MAPPING[backbone_model_type]
                backbone_config = config_class.from_dict(backbone_config)
        
        # 设置各个属性
        self.use_timm_backbone = use_timm_backbone
        self.backbone_config = backbone_config
        self.num_channels = num_channels
        self.num_queries = num_queries
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
        self.init_xavier_std = init_xavier_std
        self.encoder_layerdrop = encoder_layerdrop
        self.auxiliary_loss = auxiliary_loss
        self.position_embedding_type = position_embedding_type
        self.backbone = backbone
        self.use_pretrained_backbone = use_pretrained_backbone
        self.dilation = dilation

        # 变形属性
        self.num_feature_levels = num_feature_levels
        self.encoder_n_points = encoder_n_points
        self.decoder_n_points = decoder_n_points
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.with_box_refine = with_box_refine

        # 如果 two_stage 为 True，但 with_box_refine 为 False，则报错
        if two_stage is True and with_box_refine is False:
            raise ValueError("If two_stage is True, with_box_refine must be True.")

        # Hungarian matcher
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost

        # Loss 系数
        self.mask_loss_coefficient = mask_loss_coefficient
        self.dice_loss_coefficient = dice_loss_coefficient
        self.bbox_loss_coefficient = bbox_loss_coefficient
        self.giou_loss_coefficient = giou_loss_coefficient
        self.eos_coefficient = eos_coefficient
        self.focal_alpha = focal_alpha
        self.disable_custom_kernels = disable_custom_kernels

        # 调用父类的初始化函数
        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)

    @property
    # 返回编码器的注意力头数
    def num_attention_heads(self) -> int:
        # 返回编码器注意力头数
        return self.encoder_attention_heads

    # 返回隐藏层的大小
    @property
    def hidden_size(self) -> int:
        # 返回模型的维度大小
        return self.d_model
```