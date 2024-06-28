# `.\models\vit\configuration_vit.py`

```
# coding=utf-8
# Copyright 2021 Google AI and The HuggingFace Inc. team. All rights reserved.
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
ViT model configuration
"""

from collections import OrderedDict  # 导入有序字典模块
from typing import Mapping  # 导入类型提示模块 Mapping

from packaging import version  # 导入版本控制模块

from ...configuration_utils import PretrainedConfig  # 导入预训练模型配置工具
from ...onnx import OnnxConfig  # 导入ONNX配置
from ...utils import logging  # 导入日志工具

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

VIT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/vit-base-patch16-224": "https://huggingface.co/vit-base-patch16-224/resolve/main/config.json",
    # See all ViT models at https://huggingface.co/models?filter=vit
}


class ViTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ViTModel`]. It is used to instantiate an ViT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ViT
    [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
        # 定义 ViT 模型的配置类
        Args:
            hidden_size (`int`, *optional*, defaults to 768):
                编码器层和池化层的维度。
            num_hidden_layers (`int`, *optional*, defaults to 12):
                Transformer 编码器中的隐藏层数量。
            num_attention_heads (`int`, *optional*, defaults to 12):
                Transformer 编码器中每个注意力层的注意力头数量。
            intermediate_size (`int`, *optional*, defaults to 3072):
                Transformer 编码器中“中间”（即前馈）层的维度。
            hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
                编码器和池化器中的非线性激活函数（函数或字符串）。如果是字符串，支持 "gelu"、"relu"、"selu" 和 "gelu_new"。
            hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
                嵌入、编码器和池化层中所有全连接层的 dropout 概率。
            attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
                注意力概率的 dropout 比例。
            initializer_range (`float`, *optional*, defaults to 0.02):
                用于初始化所有权重矩阵的截断正态初始化器的标准差。
            layer_norm_eps (`float`, *optional*, defaults to 1e-12):
                层规范化层使用的 epsilon。
            image_size (`int`, *optional*, defaults to 224):
                每个图像的大小（分辨率）。
            patch_size (`int`, *optional*, defaults to 16):
                每个图块的大小（分辨率）。
            num_channels (`int`, *optional*, defaults to 3):
                输入通道的数量。
            qkv_bias (`bool`, *optional*, defaults to `True`):
                是否为查询、键和值添加偏置。
            encoder_stride (`int`, *optional*, defaults to 16):
                用于遮蔽图像建模中解码器头部中空间分辨率的增加因子。

        Example:

        ```python
        >>> from transformers import ViTConfig, ViTModel

        >>> # 初始化一个 ViT vit-base-patch16-224 风格的配置
        >>> configuration = ViTConfig()

        >>> # 根据 vit-base-patch16-224 风格的配置初始化一个模型（使用随机权重）
        >>> model = ViTModel(configuration)

        >>> # 访问模型配置
        >>> configuration = model.config
        ```"""
        # 调用父类的初始化方法，传递所有的关键字参数
        super().__init__(**kwargs)

        # 设置隐藏层的大小
        self.hidden_size = hidden_size
        # 设置隐藏层的数量
        self.num_hidden_layers = num_hidden_layers
        # 设置注意力头的数量
        self.num_attention_heads = num_attention_heads
        # 设置中间层的大小
        self.intermediate_size = intermediate_size
        # 设置隐藏层的激活函数类型
        self.hidden_act = hidden_act
        # 设置隐藏层的dropout概率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 设置注意力概率的dropout概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 设置初始化范围
        self.initializer_range = initializer_range
        # 设置层归一化的epsilon值
        self.layer_norm_eps = layer_norm_eps
        # 设置图像的大小
        self.image_size = image_size
        # 设置图像块的大小
        self.patch_size = patch_size
        # 设置通道的数量
        self.num_channels = num_channels
        # 设置查询-键-值的偏置
        self.qkv_bias = qkv_bias
        # 设置编码器的步长
        self.encoder_stride = encoder_stride
class ViTOnnxConfig(OnnxConfig):
    # 定义一个继承自OnnxConfig的类ViTOnnxConfig

    # 设置torch_onnx_minimum_version属性为最低要求的版本号为1.11
    torch_onnx_minimum_version = version.parse("1.11")

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 定义一个属性inputs，返回一个OrderedDict，其中包含输入数据的描述
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
                # 描述像素值输入的结构，使用字典表示各维度的含义
            ]
        )

    @property
    def atol_for_validation(self) -> float:
        # 定义一个属性atol_for_validation，返回用于验证的绝对误差阈值为1e-4
        return 1e-4
```