# `.\models\pvt\configuration_pvt.py`

```
# coding=utf-8
# 上面是设置文件编码为UTF-8，确保可以处理各种语言字符
# Copyright 2023 Authors: Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan,
# Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao and The HuggingFace Inc. team.
# 版权声明，列出了作者和HuggingFace团队的版权信息
# All rights reserved.
# 版权声明，保留所有权利
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 依据Apache License, Version 2.0许可证授权
# you may not use this file except in compliance with the License.
# 除非符合许可证的规定，否则不得使用此文件
# You may obtain a copy of the License at
# 可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 除非适用法律要求或书面同意，否则依据许可证分发的软件都是基于"AS IS"的基础上分发
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 没有明示或暗示的担保或条件
# See the License for the specific language governing permissions and
# 请查看许可证，了解具体的语言控制和
# limitations under the License.
# 许可证下的限制
""" Pvt model configuration"""
# 此处是模型配置的文档字符串说明

from collections import OrderedDict
# 导入OrderedDict用于创建有序字典
from typing import Callable, List, Mapping
# 导入类型提示模块，用于声明函数类型、列表类型和映射类型

from packaging import version
# 导入版本包装模块，用于处理版本信息

from ...configuration_utils import PretrainedConfig
# 导入预训练配置工具模块中的PretrainedConfig类
from ...onnx import OnnxConfig
# 导入ONNX配置模块中的OnnxConfig类
from ...utils import logging
# 导入工具包中的日志模块

logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器对象

PVT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "pvt-tiny-224": "https://huggingface.co/Zetatech/pvt-tiny-224",
    # Pvt预训练模型的名称映射到其存档URL
    # 可以在https://huggingface.co/models?filter=pvt查看所有PVT模型
}


class PvtConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PvtModel`]. It is used to instantiate an Pvt
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Pvt
    [Xrenya/pvt-tiny-224](https://huggingface.co/Xrenya/pvt-tiny-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # PvtConfig类继承自PretrainedConfig，用于存储PvtModel的配置信息
    # 通过指定的参数实例化一个Pvt模型，定义模型架构
    # 使用默认参数实例化配置对象将产生与Pvt [Xrenya/pvt-tiny-224]架构类似的配置
    # 定义 PVT 模型配置类，用于初始化 PVT 模型的参数
    Args:
        image_size (`int`, *optional*, defaults to 224):
            输入图像的大小，默认为224
        num_channels (`int`, *optional*, defaults to 3):
            输入通道的数量，默认为3
        num_encoder_blocks (`int`, *optional*, defaults to 4):
            编码器块的数量（Mix Transformer 编码器中的阶段数），默认为4
        depths (`List[int]`, *optional*, defaults to `[2, 2, 2, 2]`):
            每个编码器块中的层数，默认为 `[2, 2, 2, 2]`
        sequence_reduction_ratios (`List[int]`, *optional*, defaults to `[8, 4, 2, 1]`):
            每个编码器块中的序列减少比例，默认为 `[8, 4, 2, 1]`
        hidden_sizes (`List[int]`, *optional*, defaults to `[64, 128, 320, 512]`):
            每个编码器块的维度，默认为 `[64, 128, 320, 512]`
        patch_sizes (`List[int]`, *optional*, defaults to `[4, 2, 2, 2]`):
            每个编码器块之前的补丁大小，默认为 `[4, 2, 2, 2]`
        strides (`List[int]`, *optional*, defaults to `[4, 2, 2, 2]`):
            每个编码器块之前的步长，默认为 `[4, 2, 2, 2]`
        num_attention_heads (`List[int]`, *optional*, defaults to `[1, 2, 5, 8]`):
            每个 Transformer 编码器块中每个注意力层的注意力头数，默认为 `[1, 2, 5, 8]`
        mlp_ratios (`List[int]`, *optional*, defaults to `[8, 8, 4, 4]`):
            Mix FFNs 中隐藏层大小与输入层大小的比例，默认为 `[8, 8, 4, 4]`
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            编码器和池化器中的非线性激活函数（函数或字符串），支持 `"gelu"`, `"relu"`, `"selu"` 和 `"gelu_new"`, 默认为 `"gelu"`
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            嵌入层、编码器和池化器中所有全连接层的 dropout 概率，默认为 0.0
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            注意力概率的 dropout 比率，默认为 0.0
        initializer_range (`float`, *optional*, defaults to 0.02):
            初始化所有权重矩阵的截断正态分布的标准差，默认为 0.02
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            用于随机深度的 dropout 概率，在 Transformer 编码器的块中使用，默认为 0.0
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            层归一化层使用的 epsilon，默认为 1e-06
        qkv_bias (`bool`, *optional*, defaults to `True`):
            是否为查询、键和值添加可学习偏置，默认为 True
        num_labels ('int', *optional*, defaults to 1000):
            类别数量，默认为 1000
    Example:

    ```python
    >>> from transformers import PvtModel, PvtConfig

    >>> # Initializing a PVT Xrenya/pvt-tiny-224 style configuration
    >>> configuration = PvtConfig()
    ```
    >>> model = PvtModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```
    
    # 定义一个类名为PvtModel，表示使用PVT模型
    model_type = "pvt"

    # 初始化函数，设置PvtModel类的各种属性
    def __init__(
        self,
        image_size: int = 224,  # 图像大小，默认为224
        num_channels: int = 3,  # 图像通道数，默认为3
        num_encoder_blocks: int = 4,  # 编码器块的数量，默认为4
        depths: List[int] = [2, 2, 2, 2],  # 每个阶段的深度列表，默认为[2, 2, 2, 2]
        sequence_reduction_ratios: List[int] = [8, 4, 2, 1],  # 序列减少比例列表，默认为[8, 4, 2, 1]
        hidden_sizes: List[int] = [64, 128, 320, 512],  # 隐藏层大小列表，默认为[64, 128, 320, 512]
        patch_sizes: List[int] = [4, 2, 2, 2],  # 补丁大小列表，默认为[4, 2, 2, 2]
        strides: List[int] = [4, 2, 2, 2],  # 步幅列表，默认为[4, 2, 2, 2]
        num_attention_heads: List[int] = [1, 2, 5, 8],  # 注意力头的数量列表，默认为[1, 2, 5, 8]
        mlp_ratios: List[int] = [8, 8, 4, 4],  # MLP比率列表，默认为[8, 8, 4, 4]
        hidden_act: Mapping[str, Callable] = "gelu",  # 隐藏层激活函数，默认为'gelu'
        hidden_dropout_prob: float = 0.0,  # 隐藏层dropout概率，默认为0.0
        attention_probs_dropout_prob: float = 0.0,  # 注意力概率dropout概率，默认为0.0
        initializer_range: float = 0.02,  # 初始化范围，默认为0.02
        drop_path_rate: float = 0.0,  # drop path率，默认为0.0
        layer_norm_eps: float = 1e-6,  # 层归一化epsilon值，默认为1e-6
        qkv_bias: bool = True,  # 是否使用QKV偏置，默认为True
        num_labels: int = 1000,  # 标签数量，默认为1000
        **kwargs,
    ):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 设置对象的各种属性
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_encoder_blocks = num_encoder_blocks
        self.depths = depths
        self.sequence_reduction_ratios = sequence_reduction_ratios
        self.hidden_sizes = hidden_sizes
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.mlp_ratios = mlp_ratios
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.drop_path_rate = drop_path_rate
        self.layer_norm_eps = layer_norm_eps
        self.num_labels = num_labels
        self.qkv_bias = qkv_bias
# 定义一个私有的 OnnxConfig 类，继承自 OnnxConfig 类
class PvtOnnxConfig(OnnxConfig):
    # 设定 torch_onnx_minimum_version 属性为版本号 "1.11"
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义 inputs 属性为一个字典，表示模型输入的结构
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                # 定义输入的像素值及其维度顺序
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    # 定义 atol_for_validation 属性，表示用于验证的绝对容差
    @property
    def atol_for_validation(self) -> float:
        return 1e-4

    # 定义 default_onnx_opset 属性，表示默认的 ONNX 运算集版本
    @property
    def default_onnx_opset(self) -> int:
        return 12
```