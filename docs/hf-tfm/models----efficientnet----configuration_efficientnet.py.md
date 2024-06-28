# `.\models\efficientnet\configuration_efficientnet.py`

```py
# coding=utf-8
# Copyright 2023 Google Research, Inc. and The HuggingFace Inc. team. All rights reserved.
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
""" EfficientNet model configuration"""

# 导入 OrderedDict 和 Mapping 类型
from collections import OrderedDict
from typing import List, Mapping

# 导入版本控制的模块
from packaging import version

# 导入预训练配置和 ONNX 配置
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
# 导入日志工具
from ...utils import logging

# 获取记录器
logger = logging.get_logger(__name__)

# 预训练模型的配置文件映射字典
EFFICIENTNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/efficientnet-b7": "https://huggingface.co/google/efficientnet-b7/resolve/main/config.json",
}


# EfficientNet 配置类，继承自 PretrainedConfig
class EfficientNetConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`EfficientNetModel`]. It is used to instantiate an
    EfficientNet model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the EfficientNet
    [google/efficientnet-b7](https://huggingface.co/google/efficientnet-b7) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:
    ```
    >>> from transformers import EfficientNetConfig, EfficientNetModel

    >>> # Initializing a EfficientNet efficientnet-b7 style configuration
    >>> configuration = EfficientNetConfig()

    >>> # Initializing a model (with random weights) from the efficientnet-b7 style configuration
    >>> model = EfficientNetModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    
    # 模型类型定义为 efficientnet
    model_type = "efficientnet"
    def __init__(
        self,
        num_channels: int = 3,
        image_size: int = 600,
        width_coefficient: float = 2.0,
        depth_coefficient: float = 3.1,
        depth_divisor: int = 8,
        kernel_sizes: List[int] = [3, 3, 5, 3, 5, 5, 3],
        in_channels: List[int] = [32, 16, 24, 40, 80, 112, 192],
        out_channels: List[int] = [16, 24, 40, 80, 112, 192, 320],
        depthwise_padding: List[int] = [],
        strides: List[int] = [1, 2, 2, 2, 1, 2, 1],
        num_block_repeats: List[int] = [1, 2, 2, 3, 3, 4, 1],
        expand_ratios: List[int] = [1, 6, 6, 6, 6, 6, 6],
        squeeze_expansion_ratio: float = 0.25,
        hidden_act: str = "swish",
        hidden_dim: int = 2560,
        pooling_type: str = "mean",
        initializer_range: float = 0.02,
        batch_norm_eps: float = 0.001,
        batch_norm_momentum: float = 0.99,
        dropout_rate: float = 0.5,
        drop_connect_rate: float = 0.2,
        **kwargs,
    ):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 设置模型的各种超参数
        self.num_channels = num_channels  # 图像通道数
        self.image_size = image_size  # 图像尺寸
        self.width_coefficient = width_coefficient  # 宽度系数
        self.depth_coefficient = depth_coefficient  # 深度系数
        self.depth_divisor = depth_divisor  # 深度除数
        self.kernel_sizes = kernel_sizes  # 卷积核尺寸列表
        self.in_channels = in_channels  # 输入通道数列表
        self.out_channels = out_channels  # 输出通道数列表
        self.depthwise_padding = depthwise_padding  # 深度卷积填充列表
        self.strides = strides  # 步长列表
        self.num_block_repeats = num_block_repeats  # 每个块的重复次数列表
        self.expand_ratios = expand_ratios  # 扩展比率列表
        self.squeeze_expansion_ratio = squeeze_expansion_ratio  # 压缩扩展比率
        self.hidden_act = hidden_act  # 隐藏层激活函数类型
        self.hidden_dim = hidden_dim  # 隐藏层维度
        self.pooling_type = pooling_type  # 池化类型
        self.initializer_range = initializer_range  # 初始化范围
        self.batch_norm_eps = batch_norm_eps  # 批归一化 epsilon
        self.batch_norm_momentum = batch_norm_momentum  # 批归一化动量
        self.dropout_rate = dropout_rate  # Dropout 比率
        self.drop_connect_rate = drop_connect_rate  # DropConnect 比率
        self.num_hidden_layers = sum(num_block_repeats) * 4  # 计算总隐藏层数
# 定义一个 EfficientNetOnnxConfig 类，继承自 OnnxConfig 类
class EfficientNetOnnxConfig(OnnxConfig):
    # 定义一个类变量 torch_onnx_minimum_version，指定最小版本为 1.11
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义一个 inputs 属性，返回一个有序字典，描述输入的结构
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                # 指定输入的像素值结构，包括批次、通道数、高度、宽度
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    # 定义一个 atol_for_validation 属性，返回一个浮点数，表示验证时的容差
    @property
    def atol_for_validation(self) -> float:
        return 1e-5
```