# `.\transformers\models\table_transformer\configuration_table_transformer.py`

```
# 设置文件编码为 utf-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证的条件，否则您不得使用此文件。
# 您可以在以下网址获得许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则不得根据许可协议分发软件，软件是根据“原样”分发的，没有任何保证或条件，无论明示或暗示。
# 请参阅许可证以获取有关特定语言的限制和限制的权限。
""" Table Transformer 模型配置 """
# 从 collections 模块中导入 OrderedDict 、从 typing 模块中导入 Mapping
from collections import OrderedDict
from typing import Mapping
# 从 packaging 模块中导入 version
from packaging import version
# 从 ... 中导入相关模块
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging
# 从 ..auto 中导入 CONFIG_MAPPING
from ..auto import CONFIG_MAPPING

# 获取日志记录器
logger = logging.get_logger(__name__)

# 设置 TABLE_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP 字典
TABLE_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/table-transformer-detection": (
        "https://huggingface.co/microsoft/table-transformer-detection/resolve/main/config.json"
    ),
}

# 定义 TableTransformerConfig 类，继承自 PretrainedConfig 类
class TableTransformerConfig(PretrainedConfig):
    # TableTransformerConfig 类的文档字符串
    r"""
    This is the configuration class to store the configuration of a [`TableTransformerModel`]. It is used to
    instantiate a Table Transformer model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the Table Transformer
    [microsoft/table-transformer-detection](https://huggingface.co/microsoft/table-transformer-detection) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Examples:

    ```python
    >>> from transformers import TableTransformerModel, TableTransformerConfig

    >>> # Initializing a Table Transformer microsoft/table-transformer-detection style configuration
    >>> configuration = TableTransformerConfig()

    >>> # Initializing a model from the microsoft/table-transformer-detection style configuration
    >>> model = TableTransformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    # 设置 model_type 属性为 "table-transformer"
    model_type = "table-transformer"
    # 设置 keys_to_ignore_at_inference 属性为 ["past_key_values"]
    keys_to_ignore_at_inference = ["past_key_values"]
    # 设置 attribute_map 属性
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
    }
    # 从 transformers.models.detr.configuration_detr.DetrConfig.__init__ 复制
    # 初始化函数，用于创建一个新的对象
    def __init__(
        # 是否使用timm的backbone作为编码器的特征提取器，默认为True
        self,
        use_timm_backbone=True,
        # 编码器的配置参数，默认为None
        backbone_config=None,
        # 输入图像的通道数，默认为3
        num_channels=3,
        # 查询向量的数量，默认为100
        num_queries=100,
        # 编码器层数，默认为6
        encoder_layers=6,
        # 编码器中全连接层的维度，默认为2048
        encoder_ffn_dim=2048,
        # 编码器中注意力头的数量，默认为8
        encoder_attention_heads=8,
        # 解码器层数，默认为6
        decoder_layers=6,
        # 解码器中全连接层的维度，默认为2048
        decoder_ffn_dim=2048,
        # 解码器中注意力头的数量，默认为8
        decoder_attention_heads=8,
        # 编码器中层的随机丢弃比例，默认为0.0
        encoder_layerdrop=0.0,
        # 解码器中层的随机丢弃比例，默认为0.0
        decoder_layerdrop=0.0,
        # 是否是编码-解码结构，默认为True
        is_encoder_decoder=True,
        # 激活函数的类型，默认为"relu"
        activation_function="relu",
        # 模型的维度，默认为256
        d_model=256,
        # 模型的丢弃比例，默认为0.1
        dropout=0.1,
        # 注意力层的丢弃比例，默认为0.0
        attention_dropout=0.0,
        # 激活函数的丢弃比例，默认为0.0
        activation_dropout=0.0,
        # 初始化权重的标准差，默认为0.02
        init_std=0.02,
        # Xavier初始化的标准差，默认为1.0
        init_xavier_std=1.0,
        # 是否使用辅助损失，默认为False
        auxiliary_loss=False,
        # 位置编码的类型，默认为"sine"
        position_embedding_type="sine",
        # 使用的backbone类型，默认为"resnet50"
        backbone="resnet50",
        # 是否使用预训练的backbone，默认为True
        use_pretrained_backbone=True,
        # 是否使用膨胀卷积，默认为False
        dilation=False,
        # 分类损失的权重，默认为1
        class_cost=1,
        # 边界框损失的权重，默认为5
        bbox_cost=5,
        # giou损失的权重，默认为2
        giou_cost=2,
        # 掩膜损失系数，默认为1
        mask_loss_coefficient=1,
        # dice损失系数，默认为1
        dice_loss_coefficient=1,
        # 边界框损失系数，默认为5
        bbox_loss_coefficient=5,
        # giou损失系数，默认为2
        giou_loss_coefficient=2,
        # 结束符损失系数，默认为0.1
        eos_coefficient=0.1,
        # 其他参数，以字典形式传入
        **kwargs,
        ):
        # 检查是否同时指定了`backbone_config`和`use_timm_backbone`，若是则抛出数值错误
        if backbone_config is not None and use_timm_backbone:
            raise ValueError("You can't specify both `backbone_config` and `use_timm_backbone`.")

        # 如果不使用 tmm backbone
        if not use_timm_backbone:
            # 如果没有指定`backbone_config`
            if backbone_config is None:
                # 输出日志信息，并使用默认的`ResNet` backbone初始化配置
                logger.info("`backbone_config` is `None`. Initializing the config with the default `ResNet` backbone.")
                backbone_config = CONFIG_MAPPING["resnet"](out_features=["stage4"])
            # 如果`backbone_config`是字典类型
            elif isinstance(backbone_config, dict):
                # 获取`model_type`，并根据其值创建相应的配置类
                backbone_model_type = backbone_config.get("model_type")
                config_class = CONFIG_MAPPING[backbone_model_type]
                backbone_config = config_class.from_dict(backbone_config)
            # 将tmm的属性设为None
            dilation, backbone, use_pretrained_backbone = None, None, None

        # 设置实例属性
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
        # 匈牙利匹配器
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        # 损失系数
        self.mask_loss_coefficient = mask_loss_coefficient
        self.dice_loss_coefficient = dice_loss_coefficient
        self.bbox_loss_coefficient = bbox_loss_coefficient
        self.giou_loss_coefficient = giou_loss_coefficient
        self.eos_coefficient = eos_coefficient
        # 调用父类初始化方法
        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)

    @property
    def num_attention_heads(self) -> int:
        # 返回编码器关注头的数量
        return self.encoder_attention_heads

    @property
    def hidden_size(self) -> int:
        # 返回隐藏层的大小
        return self.d_model
# 从transformers.models.detr.configuration_detr.DetrOnnxConfig中复制代码
class TableTransformerOnnxConfig(OnnxConfig):
    # 定义torch_onnx_minimum_version属性，赋值为1.11
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义inputs属性，返回有序字典，包含像素值和像素掩码的输入维度信息
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
                ("pixel_mask", {0: "batch"}),
            ]
        )

    # 定义atol_for_validation属性，返回用于验证的绝对误差值
    @property
    def atol_for_validation(self) -> float:
        return 1e-5

    # 定义default_onnx_opset属性，返回默认的ONNX操作集版本
    @property
    def default_onnx_opset(self) -> int:
        return 12
```