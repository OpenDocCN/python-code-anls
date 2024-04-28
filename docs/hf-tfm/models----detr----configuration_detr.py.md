# `.\models\detr\configuration_detr.py`

```py
# 设置文件编码为utf-8
# 版权声明
# 根据Apache License, Version 2.0，除非符合许可证要求或书面同意，在不提供任何保证或条件的情况下分发软件
# 请查看许可证以获取更多信息：http://www.apache.org/licenses/LICENSE-2.0
# DETR模型配置

# 导入需要的库
from collections import OrderedDict
from typing import Mapping
from packaging import version
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING

# 获取日志记录器
logger = logging.get_logger(__name__)

# DETR预训练配置档案映射
DETR_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/detr-resnet-50": "https://huggingface.co/facebook/detr-resnet-50/resolve/main/config.json",
    # 查看所有DETR模型：https://huggingface.co/models?filter=detr
}

# DetrConfig类继承自PretrainedConfig
# 用于存储DetrModel的配置，并可根据指定参数实例化DETR模型架构
# 使用默认值实例化配置将产生类似于DETR[facebook/detr-resnet-50]架构的配置
# 配置对象继承自PretrainedConfig，并可用于控制模型输出
# 有关更多信息，请阅读来自PretrainedConfig的文档
# 示例
class DetrConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`DetrModel`]. It is used to instantiate a DETR
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the DETR
    [facebook/detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Examples:

    ```python
    >>> from transformers import DetrConfig, DetrModel

    >>> # Initializing a DETR facebook/detr-resnet-50 style configuration
    >>> configuration = DetrConfig()

    >>> # Initializing a model (with random weights) from the facebook/detr-resnet-50 style configuration
    >>> model = DetrModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```py
    """
    # 模型类型为"detr"
    model_type = "detr"
    # 推断时要忽略的键
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
    }
    # 初始化函数，用于创建一个新的模型实例
    def __init__(
        self,
        use_timm_backbone=True,  # 是否使用timm库提供的预训练骨干网络，默认为True
        backbone_config=None,    # 骨干网络配置参数，用于指定骨干网络的设置，可以为None
        num_channels=3,          # 输入图像通道数，默认为3（RGB图像）
        num_queries=100,         # 解码器中查询的数量，默认为100
        encoder_layers=6,        # 编码器的层数，默认为6层
        encoder_ffn_dim=2048,    # 编码器中全连接层的维度，默认为2048
        encoder_attention_heads=8,  # 编码器中的注意力头数，默认为8个
        decoder_layers=6,        # 解码器的层数，默认为6层
        decoder_ffn_dim=2048,    # 解码器中全连接层的维度，默认为2048
        decoder_attention_heads=8,  # 解码器中的注意力头数，默认为8个
        encoder_layerdrop=0.0,    # 编码器的层丢弃率，默认为0.0（不丢弃）
        decoder_layerdrop=0.0,    # 解码器的层丢弃率，默认为0.0（不丢弃）
        is_encoder_decoder=True,  # 是否使用编码-解码结构，默认为True
        activation_function="relu",  # 激活函数的类型，默认为ReLU
        d_model=256,              # 模型维度，默认为256
        dropout=0.1,              # 全连接层的丢弃率，默认为0.1
        attention_dropout=0.0,    # 注意力层的丢弃率，默认为0.0
        activation_dropout=0.0,   # 激活函数的丢弃率，默认为0.0
        init_std=0.02,            # 参数初始化的标准差，默认为0.02
        init_xavier_std=1.0,      # Xavier初始化的标准差，默认为1.0
        auxiliary_loss=False,     # 是否使用辅助损失，默认为False
        position_embedding_type="sine",  # 位置嵌入的类型，默认为正弦位置嵌入
        backbone="resnet50",      # 使用的骨干网络类型，默认为ResNet-50
        use_pretrained_backbone=True,  # 是否使用预训练的骨干网络，默认为True
        dilation=False,           # 是否使用扩张卷积，默认为False
        class_cost=1,             # 类别损失的权重，默认为1
        bbox_cost=5,              # 边界框损失的权重，默认为5
        giou_cost=2,              # GIoU损失的权重，默认为2
        mask_loss_coefficient=1,  # 掩模损失的系数，默认为1
        dice_loss_coefficient=1,  # Dice损失的系数，默认为1
        bbox_loss_coefficient=5,  # 边界框损失的系数，默认为5
        giou_loss_coefficient=2,  # GIoU损失的系数，默认为2
        eos_coefficient=0.1,      # EOS（终止符）损失的系数，默认为0.1
        **kwargs,                 # 其他关键字参数，用于灵活传递额外的参数
        ):
            # 如果同时指定了 `backbone_config` 和 `use_timm_backbone`，则抛出 ValueError
            if backbone_config is not None and use_timm_backbone:
                raise ValueError("You can't specify both `backbone_config` and `use_timm_backbone`.")

            # 如果不使用 timm 的 backbone
            if not use_timm_backbone:
                # 如果未指定 backbone_config，则初始化配置为默认的 ResNet backbone
                if backbone_config is None:
                    logger.info("`backbone_config` is `None`. Initializing the config with the default `ResNet` backbone.")
                    backbone_config = CONFIG_MAPPING["resnet"](out_features=["stage4"])
                # 如果 backbone_config 是字典类型
                elif isinstance(backbone_config, dict):
                    # 从字典中获取 backbone_model_type
                    backbone_model_type = backbone_config.get("model_type")
                    # 根据 backbone_model_type 获取对应的配置类
                    config_class = CONFIG_MAPPING[backbone_model_type]
                    # 根据字典创建配置对象
                    backbone_config = config_class.from_dict(backbone_config)
                # 设置 timm 相关属性为 None
                dilation, backbone, use_pretrained_backbone = None, None, None

            # 设置类的属性
            self.use_timm_backbone = use_timm_backbone
            self.backbone_config = backbone_config
            self.num_channels = num_channels
            self.num_queries = num_queries
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
            self.decoder_layerdrop = decoder_layerdrop
            self.num_hidden_layers = encoder_layers
            self.auxiliary_loss = auxiliary_loss
            self.position_embedding_type = position_embedding_type
            self.backbone = backbone
            self.use_pretrained_backbone = use_pretrained_backbone
            self.dilation = dilation
            # Hungarian matcher
            self.class_cost = class_cost
            self.bbox_cost = bbox_cost
            self.giou_cost = giou_cost
            # Loss coefficients
            self.mask_loss_coefficient = mask_loss_coefficient
            self.dice_loss_coefficient = dice_loss_coefficient
            self.bbox_loss_coefficient = bbox_loss_coefficient
            self.giou_loss_coefficient = giou_loss_coefficient
            self.eos_coefficient = eos_coefficient
            # 调用父类的构造函数
            super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)

        @property
        def num_attention_heads(self) -> int:
            # 返回编码器注意力头的数量
            return self.encoder_attention_heads

        @property
        def hidden_size(self) -> int:
            # 返回隐藏层大小
            return self.d_model

        @classmethod
    # 从预训练的骨干模型配置实例化一个`DetrConfig`（或派生类）

    def from_backbone_config(cls, backbone_config: PretrainedConfig, **kwargs):
        """Instantiate a [`DetrConfig`] (or a derived class) from a pre-trained backbone model configuration.

        Args:
            backbone_config ([`PretrainedConfig`]):
                The backbone configuration. # 骨干配置

        Returns:
            [`DetrConfig`]: An instance of a configuration object # 返回一个配置对象的实例
        """
        # 返回使用给定的骨干配置和其他传入参数的类实例
        return cls(backbone_config=backbone_config, **kwargs)
# 定义一个名为 DetrOnnxConfig 的类，继承自 OnnxConfig
class DetrOnnxConfig(OnnxConfig):
    # 设置 torch_onnx_minimum_version 属性为解析后的版本号 1.11
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义 inputs 属性，返回一个有序字典，包含输入张量的名称及其维度索引映射
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
                ("pixel_mask", {0: "batch"}),
            ]
        )

    # 定义 atol_for_validation 属性，返回用于验证的绝对误差阈值
    @property
    def atol_for_validation(self) -> float:
        return 1e-5

    # 定义 default_onnx_opset 属性，返回默认的 ONNX 运算集版本号
    @property
    def default_onnx_opset(self) -> int:
        return 12
```