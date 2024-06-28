# `.\models\conditional_detr\configuration_conditional_detr.py`

```
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Conditional DETR model configuration"""

# 导入 OrderedDict 和 Mapping 类型
from collections import OrderedDict
from typing import Mapping

# 导入 version 函数从 packaging 模块中
from packaging import version

# 从相对路径中导入配置相关的类和函数
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING

# 获取全局日志记录器
logger = logging.get_logger(__name__)

# 定义预训练模型配置文件的映射字典
CONDITIONAL_DETR_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/conditional-detr-resnet-50": (
        "https://huggingface.co/microsoft/conditional-detr-resnet-50/resolve/main/config.json"
    ),
}

# ConditionalDetrConfig 类，继承自 PretrainedConfig 类
class ConditionalDetrConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ConditionalDetrModel`]. It is used to instantiate
    a Conditional DETR model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Conditional DETR
    [microsoft/conditional-detr-resnet-50](https://huggingface.co/microsoft/conditional-detr-resnet-50) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Examples:

    ```python
    >>> from transformers import ConditionalDetrConfig, ConditionalDetrModel

    >>> # Initializing a Conditional DETR microsoft/conditional-detr-resnet-50 style configuration
    >>> configuration = ConditionalDetrConfig()

    >>> # Initializing a model (with random weights) from the microsoft/conditional-detr-resnet-50 style configuration
    >>> model = ConditionalDetrModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    # 模型类型为 conditional_detr
    model_type = "conditional_detr"
    
    # 推断阶段要忽略的键列表
    keys_to_ignore_at_inference = ["past_key_values"]
    
    # 属性映射字典，用于配置属性名的转换
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
    }
    # 初始化方法，用于创建一个新的对象实例
    def __init__(
        self,
        use_timm_backbone=True,  # 是否使用timm作为backbone，默认为True
        backbone_config=None,  # backbone的配置参数，默认为None
        num_channels=3,  # 输入图像的通道数，默认为3
        num_queries=300,  # 查询向量的数量，默认为300
        encoder_layers=6,  # 编码器层数，默认为6层
        encoder_ffn_dim=2048,  # 编码器中FFN层的维度，默认为2048
        encoder_attention_heads=8,  # 编码器注意力头的数量，默认为8
        decoder_layers=6,  # 解码器层数，默认为6层
        decoder_ffn_dim=2048,  # 解码器中FFN层的维度，默认为2048
        decoder_attention_heads=8,  # 解码器注意力头的数量，默认为8
        encoder_layerdrop=0.0,  # 编码器层dropout的概率，默认为0.0
        decoder_layerdrop=0.0,  # 解码器层dropout的概率，默认为0.0
        is_encoder_decoder=True,  # 是否是编码器-解码器结构，默认为True
        activation_function="relu",  # 激活函数的类型，默认为ReLU
        d_model=256,  # 模型的维度，默认为256
        dropout=0.1,  # 全局dropout的概率，默认为0.1
        attention_dropout=0.0,  # 注意力层dropout的概率，默认为0.0
        activation_dropout=0.0,  # 激活函数dropout的概率，默认为0.0
        init_std=0.02,  # 权重初始化的标准差，默认为0.02
        init_xavier_std=1.0,  # Xavier初始化的标准差，默认为1.0
        auxiliary_loss=False,  # 是否使用辅助损失，默认为False
        position_embedding_type="sine",  # 位置编码的类型，默认为"sine"
        backbone="resnet50",  # 使用的backbone网络，默认为"resnet50"
        use_pretrained_backbone=True,  # 是否使用预训练的backbone，默认为True
        backbone_kwargs=None,  # backbone网络的额外参数，默认为None
        dilation=False,  # 是否使用扩张卷积，默认为False
        class_cost=2,  # 类别损失的权重，默认为2
        bbox_cost=5,  # 边界框损失的权重，默认为5
        giou_cost=2,  # GIoU损失的权重，默认为2
        mask_loss_coefficient=1,  # 掩码损失的系数，默认为1
        dice_loss_coefficient=1,  # Dice损失的系数，默认为1
        cls_loss_coefficient=2,  # 类别损失的系数，默认为2
        bbox_loss_coefficient=5,  # 边界框损失的系数，默认为5
        giou_loss_coefficient=2,  # GIoU损失的系数，默认为2
        focal_alpha=0.25,  # Focal损失的alpha参数，默认为0.25
        **kwargs,  # 其他未列出的关键字参数
    ):
        # 编码器的注意力头数量，从类属性中获取
        @property
        def num_attention_heads(self) -> int:
            return self.encoder_attention_heads

        # 隐藏层大小，从类属性中获取
        @property
        def hidden_size(self) -> int:
            return self.d_model
# 定义一个名为 ConditionalDetrOnnxConfig 的类，继承自 OnnxConfig 类
class ConditionalDetrOnnxConfig(OnnxConfig):
    # 设定 torch_onnx_minimum_version 属性为版本号 1.11
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义一个 inputs 的属性方法，返回一个有序字典，描述了模型的输入信息
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
                ("pixel_mask", {0: "batch"}),
            ]
        )

    # 定义一个 atol_for_validation 的属性方法，返回一个用于验证的浮点数容差值
    @property
    def atol_for_validation(self) -> float:
        return 1e-5

    # 定义一个 default_onnx_opset 的属性方法，返回默认的 ONNX 操作集版本号
    @property
    def default_onnx_opset(self) -> int:
        return 12
```