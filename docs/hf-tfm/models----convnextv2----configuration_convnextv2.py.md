# `.\models\convnextv2\configuration_convnextv2.py`

```py
# coding=utf-8
# Copyright 2023 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""
ConvNeXTV2 model configuration
"""

from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...utils import logging  # 导入日志工具
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices  # 导入背骨配置混合类和获取对齐输出特征输出索引的方法

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

CONVNEXTV2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/convnextv2-tiny-1k-224": "https://huggingface.co/facebook/convnextv2-tiny-1k-224/resolve/main/config.json",
}

class ConvNextV2Config(BackboneConfigMixin, PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ConvNextV2Model`]. It is used to instantiate an
    ConvNeXTV2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the ConvNeXTV2
    [facebook/convnextv2-tiny-1k-224](https://huggingface.co/facebook/convnextv2-tiny-1k-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # 设置模型类型为 "convnextv2"
    model_type = "convnextv2"
    # 初始化函数，用于初始化一个类的实例
    def __init__(
        self,
        num_channels=3,                   # 图像输入通道数，默认为3
        patch_size=4,                     # 图像分块大小，默认为4
        num_stages=4,                     # 模型阶段数，默认为4
        hidden_sizes=None,                # 每个阶段的隐藏层大小列表，默认为[96, 192, 384, 768]
        depths=None,                      # 每个阶段的层数列表，默认为[3, 3, 9, 3]
        hidden_act="gelu",                # 隐藏层激活函数，默认为'gelu'
        initializer_range=0.02,           # 参数初始化范围，默认为0.02
        layer_norm_eps=1e-12,             # Layer Normalization 的 epsilon 值，默认为1e-12
        drop_path_rate=0.0,               # Drop Path 比率，默认为0.0
        image_size=224,                   # 输入图像大小，默认为224
        out_features=None,                # 输出特征列表，用于对齐模型输出特征
        out_indices=None,                 # 输出特征索引列表，用于对齐模型输出特征
        **kwargs,
    ):
        super().__init__(**kwargs)        # 调用父类的初始化函数

        self.num_channels = num_channels  # 设置图像输入通道数
        self.patch_size = patch_size      # 设置图像分块大小
        self.num_stages = num_stages      # 设置模型阶段数
        self.hidden_sizes = [96, 192, 384, 768] if hidden_sizes is None else hidden_sizes  # 设置每个阶段的隐藏层大小列表
        self.depths = [3, 3, 9, 3] if depths is None else depths  # 设置每个阶段的层数列表
        self.hidden_act = hidden_act      # 设置隐藏层激活函数
        self.initializer_range = initializer_range  # 设置参数初始化范围
        self.layer_norm_eps = layer_norm_eps        # 设置 Layer Normalization 的 epsilon 值
        self.drop_path_rate = drop_path_rate        # 设置 Drop Path 比率
        self.image_size = image_size        # 设置输入图像大小
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.depths) + 1)]
        # 设置模型阶段的名称列表，包括 'stem' 和 'stage1', 'stage2', ...
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )
        # 调用函数获取对齐后的输出特征和输出特征索引
```