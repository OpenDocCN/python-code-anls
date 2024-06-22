# `.\models\conditional_detr\configuration_conditional_detr.py`

```py
# 设置文件编码为 utf-8

# 版权声明
# 版权归 The HuggingFace Inc. 团队所有

# 根据 Apache License, Version 2.0 许可协议提供授权
# 访问 http://www.apache.org/licenses/LICENSE-2.0 可以获取许可协议的副本

# 条款：除非适用法律要求或书面同意，否则按"原样"分发软件，
# 没有任何明示或暗示的担保或条件。
# 参见许可协议以了解特定语言的权限和限制

""" Conditional DETR 模型配置"""
# 从 collections 模块中导入 OrderedDict 类
from collections import OrderedDict
# 从 typing 模块中导入 Mapping 类
from typing import Mapping

# 从 packaging 模块中导入 version 类
from packaging import version

# 从..模块中导入 PretrainedConfig 类、OnnxConfig 类和 logging 工具
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging
# 从..auto 模块中导入 CONFIG_MAPPING
from ..auto import CONFIG_MAPPING

# 获取日志记录器
logger = logging.get_logger(__name__)

# 给定条件下的 DETR 预训练模型配置文件映射
CONDITIONAL_DETR_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/conditional-detr-resnet-50": (
        "https://huggingface.co/microsoft/conditional-detr-resnet-50/resolve/main/config.json"
    ),
}

# ConditionalDetrConfig 继承自 PretrainedConfig 类
class ConditionalDetrConfig(PretrainedConfig):
    r"""
    这是用于存储 [`ConditionalDetrModel`] 配置的配置类。它用于根据指定的参数实例化具有指定模型架构的条件 DETR 模型。使用默认值实例化配置将产生类似 Conditional DETR [microsoft/conditional-detr-resnet-50](https://huggingface.co/microsoft/conditional-detr-resnet-50) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    示例:

    ```python
    >>> from transformers import ConditionalDetrConfig, ConditionalDetrModel

    >>> # 初始化一个 Conditional DETR microsoft/conditional-detr-resnet-50 风格的配置
    >>> configuration = ConditionalDetrConfig()

    >>> # 使用 microsoft/conditional-detr-resnet-50 风格的配置初始化一个模型（带有随机权重）
    >>> model = ConditionalDetrModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py"""

    # 模型类型为 conditional_detr
    model_type = "conditional_detr"
    # 推理时无需关注的键
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
    }
    # 初始化函数，用于创建一个新的对象实例
    def __init__(
        # 是否使用 timm 的骨干网络，默认为 True
        self,
        use_timm_backbone=True,
        # 骨干网络配置参数
        backbone_config=None,
        # 输入图像的通道数，默认为 3（RGB 图像）
        num_channels=3,
        # 查询数量，默认为 300
        num_queries=300,
        # 编码器层数，默认为 6
        encoder_layers=6,
        # 编码器中全连接层的维度，默认为 2048
        encoder_ffn_dim=2048,
        # 编码器中注意力头的数量，默认为 8
        encoder_attention_heads=8,
        # 解码器层数，默认为 6
        decoder_layers=6,
        # 解码器中全连接层的维度，默认为 2048
        decoder_ffn_dim=2048,
        # 解码器中注意力头的数量，默认为 8
        decoder_attention_heads=8,
        # 编码器层的丢弃率，默认为 0.0
        encoder_layerdrop=0.0,
        # 解码器层的丢弃率，默认为 0.0
        decoder_layerdrop=0.0,
        # 是否为编码器-解码器模型，默认为 True
        is_encoder_decoder=True,
        # 激活函数，默认为 "relu"
        activation_function="relu",
        # 模型维度，默认为 256
        d_model=256,
        # 普通丢弃率，默认为 0.1
        dropout=0.1,
        # 注意力机制的丢弃率，默认为 0.0
        attention_dropout=0.0,
        # 激活函数的丢弃率，默认为 0.0
        activation_dropout=0.0,
        # 初始化的标准差，默认为 0.02
        init_std=0.02,
        # Xavier 初始化的标准差，默认为 1.0
        init_xavier_std=1.0,
        # 是否使用辅助损失，默认为 False
        auxiliary_loss=False,
        # 位置嵌入的类型，默认为 "sine"
        position_embedding_type="sine",
        # 骨干网络类型，默认为 "resnet50"
        backbone="resnet50",
        # 是否使用预训练的骨干网络，默认为 True
        use_pretrained_backbone=True,
        # 是否进行扩张，默认为 False
        dilation=False,
        # 分类损失的权重，默认为 2
        class_cost=2,
        # 包围框损失的权重，默认为 5
        bbox_cost=5,
        # giou 损失的权重，默认为 2
        giou_cost=2,
        # 掩码损失的系数，默认为 1
        mask_loss_coefficient=1,
        # dice 损失的系数，默认为 1
        dice_loss_coefficient=1,
        # 分类损失的系数，默认为 2
        cls_loss_coefficient=2,
        # 包围框损失的系数，默认为 5
        bbox_loss_coefficient=5,
        # giou 损失的系数，默认为 2
        giou_loss_coefficient=2,
        # focal 损失的 alpha 参数，默认为 0.25
        focal_alpha=0.25,
        **kwargs,
        # 其他参数
        ):
            # 如果给定了backbone_config并且使用了timm的backbone，则抛出数值错误
            if backbone_config is not None and use_timm_backbone:
                raise ValueError("You can't specify both `backbone_config` and `use_timm_backbone`.")

            # 如果没有使用timm的backbone
            if not use_timm_backbone:
                # 如果没有给定backbone_config，则使用默认的ResNet配置
                if backbone_config is None:
                    logger.info("`backbone_config` is `None`. Initializing the config with the default `ResNet` backbone.")
                    backbone_config = CONFIG_MAPPING["resnet"](out_features=["stage4"])
                # 如果backbone_config是字典，则根据模型类型选择对应的配置，并将字典转换成相应的配置类
                elif isinstance(backbone_config, dict):
                    backbone_model_type = backbone_config.get("model_type")
                    config_class = CONFIG_MAPPING[backbone_model_type]
                    backbone_config = config_class.from_dict(backbone_config)

        # 初始化实例变量
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
        # 设置Hungarian matcher参数
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        # 设置Loss系数
        self.mask_loss_coefficient = mask_loss_coefficient
        self.dice_loss_coefficient = dice_loss_coefficient
        self.cls_loss_coefficient = cls_loss_coefficient
        self.bbox_loss_coefficient = bbox_loss_coefficient
        self.giou_loss_coefficient = giou_loss_coefficient
        self.focal_alpha = focal_alpha
        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)

    @property
    def num_attention_heads(self) -> int:
        # 返回encoder的attention头数
        return self.encoder_attention_heads

    @property
    def hidden_size(self) -> int:
        # 返回d_model作为hidden_size
        return self.d_model
# 创建一个名为ConditionalDetrOnnxConfig的类，继承自OnnxConfig类
class ConditionalDetrOnnxConfig(OnnxConfig):
    # 设置torch_onnx_minimum_version为1.11的版本
    torch_onnx_minimum_version = version.parse("1.11")

    # 设置inputs属性，返回一个有序字典，包含两个键值对，分别为"pixel_values"和"pixel_mask"
    # 每个键值对中的值是一个包含多个键值对的字典，表示每个输入的维度
    # 例如"pixel_values"对应的字典表示输入的维度分别为batch、num_channels、height、width
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
                ("pixel_mask", {0: "batch"}),
            ]
        )

    # 设置atol_for_validation属性，返回一个浮点数，表示验证时的绝对容差
    @property
    def atol_for_validation(self) -> float:
        return 1e-5

    # 设置default_onnx_opset属性，返回一个整数，表示默认的ONNX操作集的版本
    @property
    def default_onnx_opset(self) -> int:
        return 12
```