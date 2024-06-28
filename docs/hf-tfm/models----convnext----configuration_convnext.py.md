# `.\models\convnext\configuration_convnext.py`

```py
# coding=utf-8
# Copyright 2022 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
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
""" ConvNeXT model configuration"""

# 导入所需模块
from collections import OrderedDict
from typing import Mapping

from packaging import version

# 导入配置工具和相关模块
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练模型的配置文件映射表
CONVNEXT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/convnext-tiny-224": "https://huggingface.co/facebook/convnext-tiny-224/resolve/main/config.json",
    # 查看所有 ConvNeXT 模型请访问 https://huggingface.co/models?filter=convnext
}


class ConvNextConfig(BackboneConfigMixin, PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ConvNextModel`]. It is used to instantiate an
    ConvNeXT model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the ConvNeXT
    [facebook/convnext-tiny-224](https://huggingface.co/facebook/convnext-tiny-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # 定义变量 `model_type` 并赋值为字符串 "convnext"
    model_type = "convnext"
    # 初始化函数，用于初始化一个自定义类的实例
    def __init__(
        self,
        num_channels=3,                              # 设置默认值为3的通道数
        patch_size=4,                                # 设置默认值为4的补丁大小
        num_stages=4,                                # 设置默认值为4的阶段数
        hidden_sizes=None,                           # 隐藏层大小列表，默认为[96, 192, 384, 768]
        depths=None,                                 # 每个阶段的深度列表，默认为[3, 3, 9, 3]
        hidden_act="gelu",                           # 隐藏层激活函数，默认为GELU
        initializer_range=0.02,                      # 初始化器范围，默认为0.02
        layer_norm_eps=1e-12,                        # 层归一化的epsilon值，默认为1e-12
        layer_scale_init_value=1e-6,                 # 层缩放初始化值，默认为1e-6
        drop_path_rate=0.0,                          # DropPath的比率，默认为0.0
        image_size=224,                              # 图像尺寸，默认为224
        out_features=None,                           # 输出特征的名称列表，默认为None
        out_indices=None,                            # 输出特征的索引列表，默认为None
        **kwargs,                                    # 其他关键字参数
    ):
        # 调用父类的初始化方法，传入所有未显式命名的关键字参数
        super().__init__(**kwargs)

        # 初始化各个属性值
        self.num_channels = num_channels              # 设置实例的通道数属性
        self.patch_size = patch_size                  # 设置实例的补丁大小属性
        self.num_stages = num_stages                  # 设置实例的阶段数属性
        self.hidden_sizes = [96, 192, 384, 768] if hidden_sizes is None else hidden_sizes
                                                     # 如果隐藏层大小列表为空，则使用默认值
        self.depths = [3, 3, 9, 3] if depths is None else depths
                                                     # 如果深度列表为空，则使用默认值
        self.hidden_act = hidden_act                  # 设置实例的隐藏层激活函数属性
        self.initializer_range = initializer_range    # 设置实例的初始化器范围属性
        self.layer_norm_eps = layer_norm_eps          # 设置实例的层归一化epsilon属性
        self.layer_scale_init_value = layer_scale_init_value
                                                     # 设置实例的层缩放初始化值属性
        self.drop_path_rate = drop_path_rate          # 设置实例的DropPath比率属性
        self.image_size = image_size                  # 设置实例的图像尺寸属性

        # 定义阶段名称列表，包括"stem"和从"stage1"到"stageN"的命名
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.depths) + 1)]

        # 调用辅助函数获取对齐的输出特征和输出索引
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features,                # 输出特征名称列表
            out_indices=out_indices,                  # 输出特征索引列表
            stage_names=self.stage_names               # 阶段名称列表
        )
# 定义一个名为 ConvNextOnnxConfig 的类，继承自 OnnxConfig 类
class ConvNextOnnxConfig(OnnxConfig):
    
    # 定义类属性 torch_onnx_minimum_version，并赋值为解析后的版本号 "1.11"
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义一个 inputs 的属性方法，返回一个有序字典，表示输入的映射关系
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    # 定义一个 atol_for_validation 的属性方法，返回一个浮点数，表示验证时的允许误差
    @property
    def atol_for_validation(self) -> float:
        return 1e-5
```