# `.\models\resnet\configuration_resnet.py`

```
# coding=utf-8
# 文件编码声明，指定使用 UTF-8 编码格式
# Copyright 2022 Microsoft Research, Inc. and The HuggingFace Inc. team. All rights reserved.
# 版权声明，版权归 Microsoft Research, Inc. 和 HuggingFace Inc. 团队所有
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License, Version 2.0 许可协议授权，除非符合许可协议要求，否则不得使用此文件
# you may not use this file except in compliance with the License.
# 未经许可，不得使用此文件
# You may obtain a copy of the License at
# 可以在以下链接获取许可协议的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非适用法律要求或书面同意，否则按“原样”分发本软件，不附带任何明示或暗示的担保或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 不提供任何类型的担保或条件
# See the License for the specific language governing permissions and
# limitations under the License.
# 查看许可协议，了解权限和限制
""" ResNet model configuration"""
# ResNet 模型配置说明文档

from collections import OrderedDict
# 导入 OrderedDict 类，用于创建有序字典
from typing import Mapping
# 导入 Mapping 类型，用于类型提示

from packaging import version
# 导入 version 模块，用于版本管理

from ...configuration_utils import PretrainedConfig
# 从上级目录中导入 PretrainedConfig 类
from ...onnx import OnnxConfig
# 从上级目录中导入 OnnxConfig 类
from ...utils import logging
# 从上级目录中导入 logging 模块
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices
# 从上级目录中导入 BackboneConfigMixin 类和 get_aligned_output_features_output_indices 函数

logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器

RESNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/resnet-50": "https://huggingface.co/microsoft/resnet-50/blob/main/config.json",
}
# 定义 RESNET_PRETRAINED_CONFIG_ARCHIVE_MAP 字典，映射预训练模型名到其配置文件的 URL
# 用于提供预训练模型的配置信息

class ResNetConfig(BackboneConfigMixin, PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ResNetModel`]. It is used to instantiate an
    ResNet model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the ResNet
    [microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # ResNetConfig 类，用于存储 ResNetModel 的配置信息
    # 继承自 BackboneConfigMixin 和 PretrainedConfig 类

    def __init__(self, **kwargs):
        # 初始化方法，接受任意关键字参数

        super().__init__(**kwargs)
        # 调用父类 PretrainedConfig 的初始化方法，传入关键字参数
    # 定义了模型类型为 "resnet"
    model_type = "resnet"
    
    # 定义了可选的层类型列表，包括 "basic" 和 "bottleneck"
    layer_types = ["basic", "bottleneck"]
    # 初始化函数，用于创建一个新的神经网络模型对象
    def __init__(
        self,
        num_channels=3,  # 输入数据的通道数，默认为3（RGB图像）
        embedding_size=64,  # 嵌入向量的大小，默认为64
        hidden_sizes=[256, 512, 1024, 2048],  # 每个阶段的隐藏层大小列表
        depths=[3, 4, 6, 3],  # 每个阶段的残差块数量列表
        layer_type="bottleneck",  # 残差块类型，默认为瓶颈块
        hidden_act="relu",  # 隐藏层激活函数，默认为ReLU
        downsample_in_first_stage=False,  # 是否在第一个阶段下采样，默认为False
        downsample_in_bottleneck=False,  # 是否在瓶颈块中进行下采样，默认为False
        out_features=None,  # 输出特征的列表或字典，默认为None
        out_indices=None,  # 输出索引的列表或字典，默认为None
        **kwargs,  # 其他未命名参数
    ):
        super().__init__(**kwargs)  # 调用父类的初始化方法
        if layer_type not in self.layer_types:  # 如果给定的层类型不在允许的层类型列表中
            raise ValueError(f"layer_type={layer_type} is not one of {','.join(self.layer_types)}")
        self.num_channels = num_channels  # 设置输入数据的通道数
        self.embedding_size = embedding_size  # 设置嵌入向量的大小
        self.hidden_sizes = hidden_sizes  # 设置每个阶段的隐藏层大小列表
        self.depths = depths  # 设置每个阶段的残差块数量列表
        self.layer_type = layer_type  # 设置残差块的类型
        self.hidden_act = hidden_act  # 设置隐藏层的激活函数
        self.downsample_in_first_stage = downsample_in_first_stage  # 设置是否在第一个阶段进行下采样
        self.downsample_in_bottleneck = downsample_in_bottleneck  # 设置是否在瓶颈块中进行下采样
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(depths) + 1)]  # 设置阶段的名称列表
        # 调用函数获取对齐的输出特征和输出索引
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )
class ResNetOnnxConfig(OnnxConfig):
    # 定义一个 ResNet 的 ONNX 配置类，继承自 OnnxConfig 类

    # 定义 torch_onnx_minimum_version 属性，指定最低支持的 Torch 版本为 1.11
    torch_onnx_minimum_version = version.parse("1.11")

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 返回一个有序字典，包含输入的名称到索引及其描述的映射关系
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    @property
    def atol_for_validation(self) -> float:
        # 返回用于验证的绝对容差值，设定为 1e-3
        return 1e-3
```