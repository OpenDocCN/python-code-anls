# `.\models\detr\configuration_detr.py`

```
# coding=utf-8
# Copyright 2021 Facebook AI Research and The HuggingFace Inc. team. All rights reserved.
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
""" DETR model configuration"""

# 从 collections 模块中导入 OrderedDict 类
from collections import OrderedDict
# 导入 Mapping 类型
from typing import Mapping

# 从 packaging 模块中导入 version 函数
from packaging import version

# 从 configuration_utils.py 文件中导入 PretrainedConfig 类
from ...configuration_utils import PretrainedConfig
# 从 onnx.py 文件中导入 OnnxConfig 类
from ...onnx import OnnxConfig
# 从 utils.py 文件中导入 logging 函数
from ...utils import logging
# 从 auto.py 文件中导入 CONFIG_MAPPING 变量
from ..auto import CONFIG_MAPPING

# 获取 logger 对象
logger = logging.get_logger(__name__)

# DETR 预训练配置文件映射表，指定了每个预训练模型对应的配置文件 URL
DETR_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/detr-resnet-50": "https://huggingface.co/facebook/detr-resnet-50/resolve/main/config.json",
    # 查看所有 DETR 模型的列表链接：https://huggingface.co/models?filter=detr
}

# DetrConfig 类，继承自 PretrainedConfig 类，用于存储 DETR 模型的配置信息
class DetrConfig(PretrainedConfig):
    r"""
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
    ```
    """

    # 模型类型为 "detr"
    model_type = "detr"
    # 推断过程中需要忽略的键列表
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射，将配置中的属性名映射到 DETR 架构中对应的名称
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
    }
    # 初始化函数，用于实例化一个配置对象
    def __init__(
        self,
        use_timm_backbone=True,  # 是否使用timm的backbone模型，默认为True
        backbone_config=None,  # backbone模型的配置参数，默认为None
        num_channels=3,  # 输入图片的通道数，默认为3（RGB）
        num_queries=100,  # 查询向量的数量，默认为100
        encoder_layers=6,  # 编码器的层数，默认为6层
        encoder_ffn_dim=2048,  # 编码器中FFN层的维度，默认为2048
        encoder_attention_heads=8,  # 编码器的注意力头数，默认为8个头
        decoder_layers=6,  # 解码器的层数，默认为6层
        decoder_ffn_dim=2048,  # 解码器中FFN层的维度，默认为2048
        decoder_attention_heads=8,  # 解码器的注意力头数，默认为8个头
        encoder_layerdrop=0.0,  # 编码器的层丢弃率，默认为0.0
        decoder_layerdrop=0.0,  # 解码器的层丢弃率，默认为0.0
        is_encoder_decoder=True,  # 是否为编码解码模型，默认为True
        activation_function="relu",  # 激活函数，默认为ReLU
        d_model=256,  # 模型的维度，默认为256
        dropout=0.1,  # 全局dropout率，默认为0.1
        attention_dropout=0.0,  # 注意力机制的dropout率，默认为0.0
        activation_dropout=0.0,  # 激活函数的dropout率，默认为0.0
        init_std=0.02,  # 参数初始化的标准差，默认为0.02
        init_xavier_std=1.0,  # Xavier初始化的标准差，默认为1.0
        auxiliary_loss=False,  # 是否使用辅助损失，默认为False
        position_embedding_type="sine",  # 位置编码类型，默认为正弦位置编码
        backbone="resnet50",  # 使用的backbone模型，默认为ResNet-50
        use_pretrained_backbone=True,  # 是否使用预训练的backbone，默认为True
        backbone_kwargs=None,  # backbone模型的额外参数，默认为None
        dilation=False,  # 是否使用扩张卷积，默认为False
        class_cost=1,  # 分类损失的权重，默认为1
        bbox_cost=5,  # 边界框损失的权重，默认为5
        giou_cost=2,  # GIoU损失的权重，默认为2
        mask_loss_coefficient=1,  # 掩码损失的系数，默认为1
        dice_loss_coefficient=1,  # Dice损失的系数，默认为1
        bbox_loss_coefficient=5,  # 边界框损失的系数，默认为5
        giou_loss_coefficient=2,  # GIoU损失的系数，默认为2
        eos_coefficient=0.1,  # 结束标记的损失权重，默认为0.1
        **kwargs,  # 其他关键字参数，用于接收未指定的参数
    ):
        pass

    @property
    def num_attention_heads(self) -> int:
        # 返回编码器中的注意力头数
        return self.encoder_attention_heads

    @property
    def hidden_size(self) -> int:
        # 返回模型的隐藏层大小
        return self.d_model

    @classmethod
    def from_backbone_config(cls, backbone_config: PretrainedConfig, **kwargs):
        """从预训练的backbone模型配置中实例化一个DetrConfig（或其派生类）对象。

        Args:
            backbone_config ([PretrainedConfig]): 
                预训练的backbone模型的配置对象。

        Returns:
            [DetrConfig]: DetrConfig对象的一个实例
        """
        return cls(backbone_config=backbone_config, **kwargs)
# 定义一个名为 DetrOnnxConfig 的类，它继承自 OnnxConfig 类
class DetrOnnxConfig(OnnxConfig):
    # 设定 torch_onnx_minimum_version 属性为版本号 1.11
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义一个 inputs 属性，返回一个有序字典，描述了模型的输入
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
                ("pixel_mask", {0: "batch"}),
            ]
        )

    # 定义一个 atol_for_validation 属性，返回一个用于验证的绝对容差值
    @property
    def atol_for_validation(self) -> float:
        return 1e-5

    # 定义一个 default_onnx_opset 属性，返回默认的 ONNX 操作集版本号
    @property
    def default_onnx_opset(self) -> int:
        return 12
```