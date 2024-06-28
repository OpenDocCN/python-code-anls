# `.\models\levit\configuration_levit.py`

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
"""
LeViT model configuration
"""

# 导入需要的库
from collections import OrderedDict  # 导入有序字典模块
from typing import Mapping  # 导入 Mapping 类型提示

from packaging import version  # 导入版本相关的模块

# 导入配置相关的工具函数和类
from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...onnx import OnnxConfig  # 导入 ONNX 配置类
from ...utils import logging  # 导入日志相关的工具函数

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 预训练模型名称与其配置文件的映射字典
LEVIT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/levit-128S": "https://huggingface.co/facebook/levit-128S/resolve/main/config.json",
    # 查看所有 LeViT 模型的列表：https://huggingface.co/models?filter=levit
}


class LevitConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LevitModel`]. It is used to instantiate a LeViT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LeViT
    [facebook/levit-128S](https://huggingface.co/facebook/levit-128S) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # 设定模型类型为 "levit"
    model_type = "levit"

    # 初始化函数，设置模型的各项参数
    def __init__(
        self,
        image_size=224,  # 输入图像的尺寸，默认为224
        num_channels=3,  # 输入图像的通道数，默认为3
        kernel_size=3,   # 初始卷积层的卷积核大小，默认为3
        stride=2,        # 初始卷积层的步长大小，默认为2
        padding=1,       # 初始卷积层的填充大小，默认为1
        patch_size=16,   # 嵌入的补丁大小，默认为16
        hidden_sizes=[128, 256, 384],     # 每个编码器块的隐藏层维度，默认为[128, 256, 384]
        num_attention_heads=[4, 8, 12],   # 每个Transformer编码器块中注意力层的注意力头数，默认为[4, 8, 12]
        depths=[4, 4, 4],                 # 每个编码器块中的层的数量，默认为[4, 4, 4]
        key_dim=[16, 16, 16],             # 每个编码器块中键的大小，默认为[16, 16, 16]
        drop_path_rate=0,                 # 用于随机深度中的dropout概率，默认为0
        mlp_ratio=[2, 2, 2],              # Mix FFNs中隐藏层大小与输入层大小的比例，默认为[2, 2, 2]
        attention_ratio=[2, 2, 2],        # 注意力层输出维度与输入维度的比例，默认为[2, 2, 2]
        initializer_range=0.02,           # 初始化所有权重矩阵的截断正态分布标准差，默认为0.02
        **kwargs,                         # 其他参数，使用关键字参数方式接收
    ):
        ):
            # 调用父类的初始化方法，传入所有的关键字参数
            super().__init__(**kwargs)
            # 设置图像大小
            self.image_size = image_size
            # 设置通道数
            self.num_channels = num_channels
            # 设置卷积核大小
            self.kernel_size = kernel_size
            # 设置步长
            self.stride = stride
            # 设置填充
            self.padding = padding
            # 设置隐藏层大小
            self.hidden_sizes = hidden_sizes
            # 设置注意力头数目
            self.num_attention_heads = num_attention_heads
            # 设置深度
            self.depths = depths
            # 设置键的维度
            self.key_dim = key_dim
            # 设置丢弃路径的比率
            self.drop_path_rate = drop_path_rate
            # 设置补丁大小
            self.patch_size = patch_size
            # 设置注意力比率
            self.attention_ratio = attention_ratio
            # 设置MLP比率
            self.mlp_ratio = mlp_ratio
            # 设置初始化器范围
            self.initializer_range = initializer_range
            # 设置下采样操作列表
            self.down_ops = [
                # 第一个下采样操作
                ["Subsample", key_dim[0], hidden_sizes[0] // key_dim[0], 4, 2, 2],
                # 第二个下采样操作
                ["Subsample", key_dim[0], hidden_sizes[1] // key_dim[0], 4, 2, 2],
            ]
# 从transformers.models.vit.configuration_vit.ViTOnnxConfig复制而来的LevitOnnxConfig类
class LevitOnnxConfig(OnnxConfig):
    # 定义torch_onnx_minimum_version属性为1.11版本
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义inputs属性为一个OrderedDict，包含映射关系
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                # 输入映射，将输入通道名称映射到索引位置
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    # 定义atol_for_validation属性为浮点数1e-4，用于验证时的绝对误差容忍度
    @property
    def atol_for_validation(self) -> float:
        return 1e-4
```