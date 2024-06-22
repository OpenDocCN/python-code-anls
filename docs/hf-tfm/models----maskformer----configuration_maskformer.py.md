# `.\transformers\models\maskformer\configuration_maskformer.py`

```py
# 设置编码格式为 utf-8
# 版权声明
# 根据 Apache 2.0 许可，可以使用此文件，但需要遵守许可的相关规定
# 获取许可的具体链接
# 根据适用法律或书面协议要求，本软件是基于"原样"分发，不提供任何明示或默示的保证或条件。
# 请参阅许可协议以获取有关许可的详细信息，包括权限
""" MaskFormer model configuration"""
# 导入所需模块
from typing import Dict, Optional
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING
from ..detr import DetrConfig
from ..swin import SwinConfig

# 预训练配置文件的映射
MASKFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/maskformer-swin-base-ade": (
        "https://huggingface.co/facebook/maskformer-swin-base-ade/blob/main/config.json"
    )
    # 查看所有的 MaskFormer 模型：https://huggingface.co/models?filter=maskformer
}

# 获取日志记录器
logger = logging.get_logger(__name__)

# MaskFormerConfig 类，用于存储 MaskFormerModel 的配置
class MaskFormerConfig(PretrainedConfig):
    r"""
    这是用于存储[`MaskFormerModel`]配置的配置类。根据指定的参数实例化 MaskFormer 模型，定义模型体系结构。使用默认值实例化配置将产生类似于在 [facebook/maskformer-swin-base-ade](https://huggingface.co/facebook/maskformer-swin-base-ade) 上训练的配置，训练集为 [ADE20k-150](https://huggingface.co/datasets/scene_parse_150)。

    配置对象继承自[`PretrainedConfig`]，可用于控制模型的输出。阅读 [`PretrainedConfig`] 文档以获取更多信息。

    目前，MaskFormer 只支持 [Swin Transformer](swin) 作为骨干网络。
    Args:
        mask_feature_size (`int`, *optional*, defaults to 256):
            The masks' features size, this value will also be used to specify the Feature Pyramid Network features'
            size.
        no_object_weight (`float`, *optional*, defaults to 0.1):
            Weight to apply to the null (no object) class.
        use_auxiliary_loss(`bool`, *optional*, defaults to `False`):
            If `True` [`MaskFormerForInstanceSegmentationOutput`] will contain the auxiliary losses computed using the
            logits from each decoder's stage.
        backbone_config (`Dict`, *optional*):
            The configuration passed to the backbone, if unset, the configuration corresponding to
            `swin-base-patch4-window12-384` will be used.
        decoder_config (`Dict`, *optional`):
            The configuration passed to the transformer decoder model, if unset the base config for `detr-resnet-50`
            will be used.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        init_xavier_std (`float`, *optional*, defaults to 1):
            The scaling factor used for the Xavier initialization gain in the HM Attention map module.
        dice_weight (`float`, *optional*, defaults to 1.0):
            The weight for the dice loss.
        cross_entropy_weight (`float`, *optional*, defaults to 1.0):
            The weight for the cross entropy loss.
        mask_weight (`float`, *optional*, defaults to 20.0):
            The weight for the mask loss.
        output_auxiliary_logits (`bool`, *optional*):
            Should the model output its `auxiliary_logits` or not.

    Raises:
        `ValueError`:
            Raised if the backbone model type selected is not in `["swin"]` or the decoder model type selected is not
            in `["detr"]`

    Examples:

    ```python
    >>> from transformers import MaskFormerConfig, MaskFormerModel

    >>> # Initializing a MaskFormer facebook/maskformer-swin-base-ade configuration
    >>> configuration = MaskFormerConfig()

    >>> # Initializing a model (with random weights) from the facebook/maskformer-swin-base-ade style configuration
    >>> model = MaskFormerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```py

    """

    # 定义模型类型为 "maskformer"
    model_type = "maskformer"
    # 定义属性映射字典，将 "hidden_size" 映射为 "mask_feature_size"
    attribute_map = {"hidden_size": "mask_feature_size"}
    # 支持的主干(backbone)模型类型列表
    backbones_supported = ["resnet", "swin"]
    # 支持的解码器(decoder)模型类型列表
    decoders_supported = ["detr"]
    # 初始化函数，设置默认参数数值
    def __init__(
        # 设置特征大小，默认为256
        self,
        fpn_feature_size: int = 256,
        # 设置掩码特征大小，默认为256
        mask_feature_size: int = 256,
        # 设置无对象权重，默认为0.1
        no_object_weight: float = 0.1,
        # 是否使用辅助损失，默认为False
        use_auxiliary_loss: bool = False,
        # 设置骨干网络配置参数，默认为空
        backbone_config: Optional[Dict] = None,
        # 设置解码器配置参数，默认为空
        decoder_config: Optional[Dict] = None,
        # 初始化标准差，默认为0.02
        init_std: float = 0.02,
        # 初始化Xavier标准差，默认为1.0
        init_xavier_std: float = 1.0,
        # 设置Dice损失权重，默认为1.0
        dice_weight: float = 1.0,
        # 设置交叉熵损失权重，默认为1.0
        cross_entropy_weight: float = 1.0,
        # 设置掩码权重，默认为20.0
        mask_weight: float = 20.0,
        # 输出辅助Logits，默认为None
        output_auxiliary_logits: Optional[bool] = None,
        # 其他参数，以键值对形式接受
        **kwargs,
        ):
            # 如果未提供backbone_config，则回退到使用默认的Swin Transformer配置
            if backbone_config is None:
                # 回退到 https://huggingface.co/microsoft/swin-base-patch4-window12-384-in22k
                backbone_config = SwinConfig(
                    image_size=384,
                    in_channels=3,
                    patch_size=4,
                    embed_dim=128,
                    depths=[2, 2, 18, 2],
                    num_heads=[4, 8, 16, 32],
                    window_size=12,
                    drop_path_rate=0.3,
                    out_features=["stage1", "stage2", "stage3", "stage4"],
                )

            # 如果backbone_config是一个字典，则从字典中提取backbone_model_type并创建对应的config_class
            if isinstance(backbone_config, dict):
                backbone_model_type = backbone_config.pop("model_type")
                config_class = CONFIG_MAPPING[backbone_model_type]
                backbone_config = config_class.from_dict(backbone_config)

            # 确认backbone是否受支持
            if backbone_config.model_type not in self.backbones_supported:
                logger.warning_once(
                    f"Backbone {backbone_config.model_type} is not a supported model and may not be compatible with MaskFormer. "
                    f"Supported model types: {','.join(self.backbones_supported)}"
                )

            # 如果未提供decoder_config，则回退到使用默认的DETR配置
            if decoder_config is None:
                # 回退到 https://huggingface.co/facebook/detr-resnet-50
                decoder_config = DetrConfig()
            else:
                # 确认decoder是否受支持
                decoder_type = (
                    decoder_config.pop("model_type") if isinstance(decoder_config, dict) else decoder_config.model_type
                )
                if decoder_type not in self.decoders_supported:
                    raise ValueError(
                        f"Transformer Decoder {decoder_type} not supported, please use one of"
                        f" {','.join(self.decoders_supported)}"
                    )
                if isinstance(decoder_config, dict):
                    config_class = CONFIG_MAPPING[decoder_type]
                    decoder_config = config_class.from_dict(decoder_config)

            # 设置backbone_config和decoder_config
            self.backbone_config = backbone_config
            self.decoder_config = decoder_config
            # 模型的主要特征维度
            self.fpn_feature_size = fpn_feature_size
            self.mask_feature_size = mask_feature_size
            # 初始化器
            self.init_std = init_std
            self.init_xavier_std = init_xavier_std
            # Hungarian matcher && loss
            self.cross_entropy_weight = cross_entropy_weight
            self.dice_weight = dice_weight
            self.mask_weight = mask_weight
            self.use_auxiliary_loss = use_auxiliary_loss
            self.no_object_weight = no_object_weight
            self.output_auxiliary_logits = output_auxiliary_logits

            # 设置注意力头数和隐藏层数
            self.num_attention_heads = self.decoder_config.encoder_attention_heads
            self.num_hidden_layers = self.decoder_config.num_hidden_layers
            # 调用父类的初始化方法
            super().__init__(**kwargs)

        @classmethod
        def from_backbone_and_decoder_configs(
            cls, backbone_config: PretrainedConfig, decoder_config: PretrainedConfig, **kwargs
        """从预训练的骨干模型配置和DETR模型配置实例化一个[`MaskFormerConfig`]（或派生类）。

        Args:
            backbone_config ([`PretrainedConfig`]):
                骨干模型的配置。
            decoder_config ([`PretrainedConfig`]):
                要使用的变压器解码器配置。

        Returns:
            [`MaskFormerConfig`]: 配置对象的实例
        """
        # 从给定的配置实例化一个[`MaskFormerConfig`]对象
        return cls(
            backbone_config=backbone_config,
            decoder_config=decoder_config,
            **kwargs,  # 将额外的关键字参数传递给构造函数
        )
```