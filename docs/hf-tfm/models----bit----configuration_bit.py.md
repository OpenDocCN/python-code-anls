# `.\models\bit\configuration_bit.py`

```
# coding=utf-8
# 上面的行指定了文件的编码格式为 UTF-8，确保文件中的中文等特殊字符能正确解析
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
# 版权声明，标明此文件版权归 HuggingFace 公司所有，保留所有权利
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache 2.0 许可证授权，允许在符合许可证条件的情况下使用本文件
# you may not use this file except in compliance with the License.
# 除非遵守许可证规定，否则不得使用本文件
# You may obtain a copy of the License at
# 可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# 除非适用法律要求或书面同意，否则在法律允许的范围内，软件
# distributed under the License is distributed on an "AS IS" BASIS,
# 根据许可证以“原样”分发
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 没有任何形式的明示或暗示保证和条件
# See the License for the specific language governing permissions and
# 请查看许可证了解特定语言的权限和限制
# limitations under the License.
""" BiT model configuration"""
# BiT 模型配置

from ...configuration_utils import PretrainedConfig
# 导入预训练配置类
from ...utils import logging
# 导入日志工具
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices
# 导入 Backbone 配置混合类和获取对齐输出特征输出索引的工具函数


logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器

BIT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/bit-50": "https://huggingface.co/google/bit-50/resolve/main/config.json",
}
# BiT 预训练模型的配置映射，将模型名称映射到配置文件的 URL
# 当需要获取特定模型的配置时，可以通过模型名称在这里查找对应的配置 URL


class BitConfig(BackboneConfigMixin, PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BitModel`]. It is used to instantiate an BiT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the BiT
    [google/bit-50](https://huggingface.co/google/bit-50) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # BitConfig 类，用于存储 BitModel 的配置信息
    # 继承自 BackboneConfigMixin 和 PretrainedConfig，用于控制模型输出
    # 模型类型设定为 "bit"
    model_type = "bit"
    # 定义神经网络层类型列表，包括“preactivation”和“bottleneck”
    layer_types = ["preactivation", "bottleneck"]
    # 定义支持的填充策略列表，包括“SAME”和“VALID”
    supported_padding = ["SAME", "VALID"]

    # 初始化函数，用于设置神经网络模型的各种参数和属性
    def __init__(
        self,
        num_channels=3,
        embedding_size=64,
        hidden_sizes=[256, 512, 1024, 2048],
        depths=[3, 4, 6, 3],
        layer_type="preactivation",  # 神经网络层类型，默认为“preactivation”
        hidden_act="relu",  # 隐藏层激活函数，默认为ReLU
        global_padding=None,  # 全局填充策略，默认为None
        num_groups=32,  # 分组数，默认为32
        drop_path_rate=0.0,  # DropPath率，默认为0.0
        embedding_dynamic_padding=False,  # 是否使用动态填充embedding，默认为False
        output_stride=32,  # 输出步幅，默认为32
        width_factor=1,  # 宽度因子，默认为1
        out_features=None,
        out_indices=None,
        **kwargs,
    ):
        super().__init__(**kwargs)  # 调用父类的初始化方法

        # 检查传入的神经网络层类型是否在支持的层类型列表中，否则抛出数值错误
        if layer_type not in self.layer_types:
            raise ValueError(f"layer_type={layer_type} is not one of {','.join(self.layer_types)}")
        
        # 如果全局填充策略不为None，则检查其是否在支持的填充策略列表中，否则抛出数值错误
        if global_padding is not None:
            if global_padding.upper() in self.supported_padding:
                global_padding = global_padding.upper()
            else:
                raise ValueError(f"Padding strategy {global_padding} not supported")
        
        # 将所有传入的参数赋值给对象的属性
        self.num_channels = num_channels
        self.embedding_size = embedding_size
        self.hidden_sizes = hidden_sizes
        self.depths = depths
        self.layer_type = layer_type
        self.hidden_act = hidden_act
        self.global_padding = global_padding
        self.num_groups = num_groups
        self.drop_path_rate = drop_path_rate
        self.embedding_dynamic_padding = embedding_dynamic_padding
        self.output_stride = output_stride
        self.width_factor = width_factor

        # 设置阶段（stage）名称列表，包括“stem”和“stage1”到“stage4”（或更多阶段）
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(depths) + 1)]
        
        # 调用函数获取对齐的输出特征和输出索引，用于后续计算
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )
```