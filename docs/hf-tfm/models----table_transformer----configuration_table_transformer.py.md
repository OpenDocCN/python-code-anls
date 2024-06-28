# `.\models\table_transformer\configuration_table_transformer.py`

```
# coding=utf-8
# 指定文件编码格式为 UTF-8

# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证版本 2.0 进行许可
# 除非符合许可证要求，否则不得使用本文件
# 您可以在以下网址获取许可证的副本：
# http://www.apache.org/licenses/LICENSE-2.0

# 导入必要的模块
""" Table Transformer 模型配置"""
from collections import OrderedDict  # 导入 OrderedDict 类
from typing import Mapping  # 导入 Mapping 类型

from packaging import version  # 导入 version 函数

# 导入配置相关的模块和类
from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...onnx import OnnxConfig  # 导入 Onnx 配置类
from ...utils import logging  # 导入日志工具
from ..auto import CONFIG_MAPPING  # 导入自动配置映射

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义预训练模型与配置文件映射
TABLE_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/table-transformer-detection": (
        "https://huggingface.co/microsoft/table-transformer-detection/resolve/main/config.json"
    ),
}

# TableTransformerConfig 类，继承自 PretrainedConfig 类
class TableTransformerConfig(PretrainedConfig):
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
    ```

    """
    
    # 模型类型设为 "table-transformer"
    model_type = "table-transformer"
    # 推理时忽略的关键字列表
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射字典，用于配置转换
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
    }

    # 以下内容是从 transformers.models.detr.configuration_detr.DetrConfig.__init__ 中复制而来
    # 定义一个类的初始化方法，初始化对象的各个属性
    def __init__(
        self,
        use_timm_backbone=True,  # 是否使用timm的骨干网络，默认为True
        backbone_config=None,  # 骨干网络配置参数，默认为None
        num_channels=3,  # 输入通道数，默认为3
        num_queries=100,  # 查询的数量，默认为100
        encoder_layers=6,  # 编码器层数，默认为6
        encoder_ffn_dim=2048,  # 编码器中FFN层的维度，默认为2048
        encoder_attention_heads=8,  # 编码器中注意力头的数量，默认为8
        decoder_layers=6,  # 解码器层数，默认为6
        decoder_ffn_dim=2048,  # 解码器中FFN层的维度，默认为2048
        decoder_attention_heads=8,  # 解码器中注意力头的数量，默认为8
        encoder_layerdrop=0.0,  # 编码器层dropout比率，默认为0.0
        decoder_layerdrop=0.0,  # 解码器层dropout比率，默认为0.0
        is_encoder_decoder=True,  # 是否为编码-解码结构，默认为True
        activation_function="relu",  # 激活函数类型，默认为"relu"
        d_model=256,  # 模型维度，默认为256
        dropout=0.1,  # 全局dropout比率，默认为0.1
        attention_dropout=0.0,  # 注意力机制的dropout比率，默认为0.0
        activation_dropout=0.0,  # 激活函数的dropout比率，默认为0.0
        init_std=0.02,  # 初始化的标准差，默认为0.02
        init_xavier_std=1.0,  # Xavier初始化的标准差，默认为1.0
        auxiliary_loss=False,  # 是否使用辅助损失，默认为False
        position_embedding_type="sine",  # 位置嵌入类型，默认为"sine"
        backbone="resnet50",  # 骨干网络类型，默认为"resnet50"
        use_pretrained_backbone=True,  # 是否使用预训练的骨干网络，默认为True
        backbone_kwargs=None,  # 骨干网络的其他关键字参数，默认为None
        dilation=False,  # 是否使用扩张卷积，默认为False
        class_cost=1,  # 分类损失的系数，默认为1
        bbox_cost=5,  # 边界框损失的系数，默认为5
        giou_cost=2,  # GIoU损失的系数，默认为2
        mask_loss_coefficient=1,  # 掩膜损失的系数，默认为1
        dice_loss_coefficient=1,  # Dice损失的系数，默认为1
        bbox_loss_coefficient=5,  # 边界框损失的系数，默认为5
        giou_loss_coefficient=2,  # GIoU损失的系数，默认为2
        eos_coefficient=0.1,  # EOS损失的系数，默认为0.1
        **kwargs,  # 其他可选关键字参数
    ):
    @property
    # 返回编码器中的注意力头数量
    def num_attention_heads(self) -> int:
        return self.encoder_attention_heads
    
    @property
    # 返回模型的隐藏层大小（维度）
    def hidden_size(self) -> int:
        return self.d_model
# Copied from transformers.models.detr.configuration_detr.DetrOnnxConfig
# 从 transformers.models.detr.configuration_detr.DetrOnnxConfig 中复制而来

class TableTransformerOnnxConfig(OnnxConfig):
    # 定义 torch_onnx_minimum_version 属性，指定最低的 Torch 版本要求为 1.11
    torch_onnx_minimum_version = version.parse("1.11")

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 返回一个有序字典，描述模型输入的名称与维度索引的映射关系
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
                ("pixel_mask", {0: "batch"}),
            ]
        )

    @property
    def atol_for_validation(self) -> float:
        # 返回一个浮点数，表示在验证时使用的绝对容差值
        return 1e-5

    @property
    def default_onnx_opset(self) -> int:
        # 返回一个整数，表示默认的 ONNX 运算集版本号
        return 12
```