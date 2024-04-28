# `.\transformers\models\mobilenet_v2\configuration_mobilenet_v2.py`

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
""" MobileNetV2 model configuration"""

# 从 collections 模块中导入 OrderedDict 类
from collections import OrderedDict
# 导入 Mapping 类型的类型提示
from typing import Mapping
# 导入 packaging 模块的 version 函数
from packaging import version
# 从上级目录中导入 configuration_utils 模块的 PretrainedConfig 类
from ...configuration_utils import PretrainedConfig
# 从上级目录中导入 onnx 模块的 OnnxConfig 类
from ...onnx import OnnxConfig
# 从上级目录中导入 utils 模块的 logging 函数
from ...utils import logging

# 获取 logger 对象
logger = logging.get_logger(__name__)

# MobileNetV2 预训练配置文件的映射表
MOBILENET_V2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/mobilenet_v2_1.4_224": "https://huggingface.co/google/mobilenet_v2_1.4_224/resolve/main/config.json",
    "google/mobilenet_v2_1.0_224": "https://huggingface.co/google/mobilenet_v2_1.0_224/resolve/main/config.json",
    "google/mobilenet_v2_0.75_160": "https://huggingface.co/google/mobilenet_v2_0.75_160/resolve/main/config.json",
    "google/mobilenet_v2_0.35_96": "https://huggingface.co/google/mobilenet_v2_0.35_96/resolve/main/config.json",
    # See all MobileNetV2 models at https://huggingface.co/models?filter=mobilenet_v2
}

# MobileNetV2 配置类，继承自 PretrainedConfig
class MobileNetV2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MobileNetV2Model`]. It is used to instantiate a
    MobileNetV2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MobileNetV2
    [google/mobilenet_v2_1.0_224](https://huggingface.co/google/mobilenet_v2_1.0_224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # 定义函数参数及默认值说明
    Args:
        num_channels (`int`, *optional*, defaults to 3):
            输入通道的数量，默认值为3
        image_size (`int`, *optional*, defaults to 224):
            每个图像的大小（分辨率），默认值为224
        depth_multiplier (`float`, *optional*, defaults to 1.0):
            收缩或扩展每一层中通道数量的倍数。默认值为1.0，即网络从32个通道开始。有时也称为“alpha”或“宽度倍增器”。
        depth_divisible_by (`int`, *optional*, defaults to 8):
            每一层中的通道数量将始终是此数字的倍数，默认值为8
        min_depth (`int`, *optional*, defaults to 8):
            所有层将至少具有此数量的通道，默认值为8
        expand_ratio (`float`, *optional*, defaults to 6.0):
            每个块中第一层的输出通道数量是输入通道乘以扩展比。默认值为6.0
        output_stride (`int`, *optional*, defaults to 32):
            输入特征图和输出特征图的空间分辨率之间的比率。默认情况下，模型将输入维度减小32倍。如果`output_stride`是8或16，则模型会在深度层上使用扩张卷积，而不是常规卷积，从而使特征图永远不会比输入图像小8倍或16倍。
        first_layer_is_expansion (`bool`, *optional*, defaults to `True`):
            如果非常第一个卷积层也是第一个扩展块的扩展层，则为True，默认值为True。
        finegrained_output (`bool`, *optional*, defaults to `True`):
            如果为true，则即使`depth_multiplier`小于1，最终卷积层中的输出通道数量将保持较大（1280），默认为True。
        hidden_act (`str` or `function`, *optional*, defaults to `"relu6"`):
            在变压器编码器和卷积层中的非线性激活函数（函数或字符串）。默认值为"relu6"。
        tf_padding (`bool`, *optional*, defaults to `True`):
            是否在卷积层上使用TensorFlow填充规则，默认值为True。
        classifier_dropout_prob (`float`, *optional*, defaults to 0.8):
            附加分类器的丢失率，默认值为0.8。
        initializer_range (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态初始化器的标准差，默认值为0.02。
        layer_norm_eps (`float`, *optional*, defaults to 0.001):
            层规范化层使用的ε，默认值为0.001。
        semantic_loss_ignore_index (`int`, *optional*, defaults to 255):
            语义分割模型的丢失函数忽略的索引，默认值为255。
    
    Example:
    
        ```python
        >>> from transformers import MobileNetV2Config, MobileNetV2Model
    
        >>> # Initializing a "mobilenet_v2_1.0_224" style configuration
        >>> configuration = MobileNetV2Config()
    # 从“mobilenet_v2_1.0_224”样式配置初始化模型
    model = MobileNetV2Model(configuration)
    
    # 访问模型配置
    configuration = model.config
# 定义一个 MobileNetV2OnnxConfig 类，它继承自 OnnxConfig 类
class MobileNetV2OnnxConfig(OnnxConfig):
    # 定义 torch_onnx_minimum_version 属性，设置为版本号 1.11
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义 inputs 属性，返回一个有序字典，键为字符串 "pixel_values"，值为一个字典，包含键值对 {0: "batch"}
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict([("pixel_values", {0: "batch"})])

    # 定义 outputs 属性，返回一个有序字典，如果任务为 "image-classification"，则输出键 "logits"，值为一个字典 {0: "batch"}，否则输出键 "last_hidden_state" 和 "pooler_output"，值为字典 {0: "batch"}
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "image-classification":
            return OrderedDict([("logits", {0: "batch"})])
        else:
            return OrderedDict([("last_hidden_state", {0: "batch"}), ("pooler_output", {0: "batch"})])

    # 定义 atol_for_validation 属性，返回一个浮点数 1e-4，用于验证
    @property
    def atol_for_validation(self) -> float:
        return 1e-4
```