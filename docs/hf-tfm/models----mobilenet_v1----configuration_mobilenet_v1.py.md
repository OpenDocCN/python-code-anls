# `.\models\mobilenet_v1\configuration_mobilenet_v1.py`

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
""" MobileNetV1 model configuration"""

# 引入 OrderedDict 用于有序字典，Mapping 用于类型提示
from collections import OrderedDict
from typing import Mapping

# 引入 version 函数从 packaging 模块中
from packaging import version

# 从相应的路径导入所需的配置类和工具
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging

# 获取 logger 对象
logger = logging.get_logger(__name__)

# MobileNetV1 预训练配置文件的映射，每个模型映射到其配置文件的 URL
MOBILENET_V1_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/mobilenet_v1_1.0_224": "https://huggingface.co/google/mobilenet_v1_1.0_224/resolve/main/config.json",
    "google/mobilenet_v1_0.75_192": "https://huggingface.co/google/mobilenet_v1_0.75_192/resolve/main/config.json",
    # 查看所有 MobileNetV1 模型请访问 https://huggingface.co/models?filter=mobilenet_v1
}

# MobileNetV1 配置类，继承自 PretrainedConfig 类
class MobileNetV1Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MobileNetV1Model`]. It is used to instantiate a
    MobileNetV1 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MobileNetV1
    [google/mobilenet_v1_1.0_224](https://huggingface.co/google/mobilenet_v1_1.0_224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # 空的类定义，用于后续填充 MobileNetV1 的具体配置参数
    pass
    # 定义 MobileNetV1Config 类型的模型配置
    model_type = "mobilenet_v1"
    
    # MobileNetV1Config 类的构造函数，设置模型的各种参数和选项
    def __init__(
        self,
        num_channels=3,  # 输入通道数，默认为3
        image_size=224,  # 图像大小（分辨率），默认为224
        depth_multiplier=1.0,  # 层的通道数乘数因子，默认为1.0
        min_depth=8,  # 所有层至少有的通道数，默认为8
        hidden_act="relu6",  # 在 Transformer 编码器和卷积层中使用的非线性激活函数，默认为 "relu6"
        tf_padding=True,  # 是否在卷积层中使用 TensorFlow 的填充规则，默认为 True
        classifier_dropout_prob=0.999,  # 附加分类器的 dropout 比例，默认为0.999
        initializer_range=0.02,  # 初始化所有权重矩阵的截断正态分布的标准差，默认为0.02
        layer_norm_eps=0.001,  # 层归一化层使用的 epsilon 值，默认为0.001
        **kwargs,  # 其他参数
    ):
        # 调用父类的构造函数，并传递其他参数
        super().__init__(**kwargs)
    
        # 如果 depth_multiplier 小于等于 0，则抛出 ValueError
        if depth_multiplier <= 0:
            raise ValueError("depth_multiplier must be greater than zero.")
    
        # 设置模型的属性
        self.num_channels = num_channels
        self.image_size = image_size
        self.depth_multiplier = depth_multiplier
        self.min_depth = min_depth
        self.hidden_act = hidden_act
        self.tf_padding = tf_padding
        self.classifier_dropout_prob = classifier_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
# 定义 MobileNetV1OnnxConfig 类，继承自 OnnxConfig 类
class MobileNetV1OnnxConfig(OnnxConfig):
    
    # 设定 torch_onnx_minimum_version 属性为 1.11 的版本对象
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义 inputs 属性，返回一个有序字典，包含输入名称到维度映射的字典
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict([("pixel_values", {0: "batch"})])

    # 定义 outputs 属性，根据任务类型返回不同的有序字典，包含输出名称到维度映射的字典
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "image-classification":
            return OrderedDict([("logits", {0: "batch"})])
        else:
            return OrderedDict([("last_hidden_state", {0: "batch"}), ("pooler_output", {0: "batch"})])

    # 定义 atol_for_validation 属性，返回用于验证的绝对误差阈值
    @property
    def atol_for_validation(self) -> float:
        return 1e-4
```