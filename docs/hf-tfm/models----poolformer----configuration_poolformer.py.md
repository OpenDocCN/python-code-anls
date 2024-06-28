# `.\models\poolformer\configuration_poolformer.py`

```
# coding=utf-8
# 声明文件编码格式为 UTF-8

# Copyright 2022 Sea AI Labs and The HuggingFace Inc. team. All rights reserved.
# 版权声明，保留所有权利

# Licensed under the Apache License, Version 2.0 (the "License");
# 授权许可声明，使用 Apache License, Version 2.0

# you may not use this file except in compliance with the License.
# 您除非遵守许可证，否则不得使用此文件。

# You may obtain a copy of the License at
# 您可以在以下网址获取许可证副本

#     http://www.apache.org/licenses/LICENSE-2.0
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and
# limitations under the License.
# 请查阅许可证以获取详细的权限和限制信息。

""" PoolFormer model configuration"""
# 模型配置的说明文档

from collections import OrderedDict
# 导入 OrderedDict 类，用于创建有序字典

from typing import Mapping
# 导入 Mapping 类型提示，用于类型注解

from packaging import version
# 导入 version 模块，用于处理版本号

from ...configuration_utils import PretrainedConfig
# 导入预训练配置类

from ...onnx import OnnxConfig
# 导入 ONNX 配置类

from ...utils import logging
# 导入日志工具模块

logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器

POOLFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "sail/poolformer_s12": "https://huggingface.co/sail/poolformer_s12/resolve/main/config.json",
    # 定义预训练模型名称和对应的配置文件 URL
    # 可在 https://huggingface.co/models?filter=poolformer 查看所有 PoolFormer 模型
}


class PoolFormerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of [`PoolFormerModel`]. It is used to instantiate a
    PoolFormer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the PoolFormer
    [sail/poolformer_s12](https://huggingface.co/sail/poolformer_s12) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # PoolFormer 模型的配置类，用于存储 PoolFormerModel 的配置信息
    # 可根据指定参数实例化 PoolFormer 模型，定义模型架构
    # 使用默认参数实例化配置对象将得到与 PoolFormer sail/poolformer_s12 架构相似的配置
    # 配置对象继承自 PretrainedConfig，可用于控制模型输出。详细信息请阅读 PretrainedConfig 的文档。
    # 定义模型类型为 "poolformer"
    model_type = "poolformer"
    
    # 定义 PoolFormerModel 类，用于创建 PoolFormer 模型的配置和实例化
    def __init__(
        self,
        num_channels=3,  # 输入图像的通道数，默认为 3
        patch_size=16,   # 输入补丁的大小，默认为 16
        stride=16,       # 输入补丁的步长，默认为 16
        pool_size=3,     # 池化窗口的大小，默认为 3
        mlp_ratio=4.0,   # MLP 输出通道数与输入通道数的比率，默认为 4.0
        depths=[2, 2, 6, 2],           # 每个编码器块的深度，默认为 `[2, 2, 6, 2]`
        hidden_sizes=[64, 128, 320, 512],  # 每个编码器块的隐藏层大小，默认为 `[64, 128, 320, 512]`
        patch_sizes=[7, 3, 3, 3],      # 每个编码器块的输入补丁大小，默认为 `[7, 3, 3, 3]`
        strides=[4, 2, 2, 2],          # 每个编码器块的输入补丁步长，默认为 `[4, 2, 2, 2]`
        padding=[2, 1, 1, 1],          # 每个编码器块的输入补丁填充，默认为 `[2, 1, 1, 1]`
        num_encoder_blocks=4,          # 编码器块的数量，默认为 4
        drop_path_rate=0.0,            # 用于丢弃层的丢弃率，默认为 0.0
        hidden_act="gelu",             # 隐藏层的激活函数，默认为 "gelu"
        use_layer_scale=True,          # 是否使用层尺度，默认为 True
        layer_scale_init_value=1e-5,   # 层尺度的初始值，默认为 1e-5
        initializer_range=0.02,        # 权重的初始化范围，默认为 0.02
        **kwargs,
    ):
        ):
        # 初始化函数，设置各个参数并调用父类的初始化方法
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.pool_size = pool_size
        self.hidden_sizes = hidden_sizes
        self.mlp_ratio = mlp_ratio
        self.depths = depths
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.num_encoder_blocks = num_encoder_blocks
        self.drop_path_rate = drop_path_rate
        self.hidden_act = hidden_act
        self.use_layer_scale = use_layer_scale
        self.layer_scale_init_value = layer_scale_init_value
        self.initializer_range = initializer_range
        # 调用父类的初始化方法
        super().__init__(**kwargs)
class PoolFormerOnnxConfig(OnnxConfig):
    # 定义 PoolFormerOnnxConfig 类，继承自 OnnxConfig 类
    
    torch_onnx_minimum_version = version.parse("1.11")
    # 设置 torch_onnx_minimum_version 属性为 1.11 的版本对象

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 定义 inputs 属性作为 property 方法，返回一个有序字典
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )
        # 返回一个字典，键为 "pixel_values"，值为另一个字典，映射关系为索引到字符串描述

    @property
    def atol_for_validation(self) -> float:
        # 定义 atol_for_validation 属性作为 property 方法，返回一个浮点数
        return 2e-3
        # 返回浮点数 0.002，用于验证的绝对容差限制
```