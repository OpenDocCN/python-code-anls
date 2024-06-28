# `.\models\deit\configuration_deit.py`

```
# coding=utf-8
# Copyright 2021 Facebook AI Research (FAIR) and The HuggingFace Inc. team. All rights reserved.
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
""" DeiT model configuration"""

# 从 collections 模块导入 OrderedDict，用于有序字典的支持
from collections import OrderedDict
# 导入 Mapping 用于类型提示
from typing import Mapping

# 从 packaging 模块导入 version，用于版本处理
from packaging import version

# 导入预训练配置的基类 PretrainedConfig
from ...configuration_utils import PretrainedConfig
# 导入 OnnxConfig 用于 ONNX 格式配置
from ...onnx import OnnxConfig
# 导入 logging 模块中的 get_logger 函数
from ...utils import logging

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 预训练模型的配置文件映射，包含模型名称及其对应的配置文件 URL
DEIT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/deit-base-distilled-patch16-224": (
        "https://huggingface.co/facebook/deit-base-patch16-224/resolve/main/config.json"
    ),
    # 查看所有 DeiT 模型的列表：https://huggingface.co/models?filter=deit
}


# DeiT 模型的配置类，继承自 PretrainedConfig
class DeiTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DeiTModel`]. It is used to instantiate an DeiT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the DeiT
    [facebook/deit-base-distilled-patch16-224](https://huggingface.co/facebook/deit-base-distilled-patch16-224)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # 模型类型设定为 "deit"
    model_type = "deit"

    # 初始化函数，定义了模型的各种配置参数
    def __init__(
        self,
        # 编码器层和池化层的维度
        hidden_size=768,
        # Transformer 编码器中隐藏层的数量
        num_hidden_layers=12,
        # Transformer 编码器中每个注意力层的注意力头数
        num_attention_heads=12,
        # Transformer 编码器中"中间"（即前馈）层的维度
        intermediate_size=3072,
        # 编码器和池化层中的非线性激活函数
        hidden_act="gelu",
        # 嵌入层、编码器和池化层中所有全连接层的 dropout 概率
        hidden_dropout_prob=0.0,
        # 注意力概率的 dropout 比例
        attention_probs_dropout_prob=0.0,
        # 初始化所有权重矩阵的截断正态初始化器的标准差
        initializer_range=0.02,
        # 层归一化层使用的 epsilon 值
        layer_norm_eps=1e-12,
        # 每个图像的大小（分辨率）
        image_size=224,
        # 每个图像块（patch）的大小（分辨率）
        patch_size=16,
        # 输入通道的数量
        num_channels=3,
        # 是否为查询、键和值添加偏置
        qkv_bias=True,
        # 解码器头部中用于掩蔽图像建模的空间分辨率增加因子
        encoder_stride=16,
        **kwargs,
        ):
        # 调用父类的初始化方法，传递所有关键字参数
        super().__init__(**kwargs)

        # 初始化模型的隐藏层大小
        self.hidden_size = hidden_size
        # 设置模型的隐藏层数量
        self.num_hidden_layers = num_hidden_layers
        # 设置注意力头的数量
        self.num_attention_heads = num_attention_heads
        # 设置中间层的大小
        self.intermediate_size = intermediate_size
        # 激活函数类型
        self.hidden_act = hidden_act
        # 隐藏层的 dropout 概率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 注意力概率的 dropout 概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 初始化范围
        self.initializer_range = initializer_range
        # 层标准化的 epsilon 值
        self.layer_norm_eps = layer_norm_eps
        # 图像大小
        self.image_size = image_size
        # 图像分块大小
        self.patch_size = patch_size
        # 图像通道数
        self.num_channels = num_channels
        # 是否使用 QKV 偏置
        self.qkv_bias = qkv_bias
        # 编码器步长
        self.encoder_stride = encoder_stride
class DeiTOnnxConfig(OnnxConfig):
    # 定义一个新的配置类 DeiTOnnxConfig，继承自 OnnxConfig 类

    torch_onnx_minimum_version = version.parse("1.11")
    # 设置 torch 和 ONNX 的最低兼容版本为 1.11

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 定义 inputs 属性，返回一个有序字典，表示输入数据的结构
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    @property
    def atol_for_validation(self) -> float:
        # 定义 atol_for_validation 属性，返回一个浮点数，表示验证时的容差值
        return 1e-4
```