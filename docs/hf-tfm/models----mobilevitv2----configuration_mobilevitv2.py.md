# `.\models\mobilevitv2\configuration_mobilevitv2.py`

```
# coding=utf-8
# 文件编码声明，指定使用UTF-8编码格式

# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
# 版权声明，版权归HuggingFace Inc.团队所有

# Licensed under the Apache License, Version 2.0 (the "License");
# 按照Apache许可证版本2.0授权许可

# you may not use this file except in compliance with the License.
# 除非符合许可证要求，否则不得使用本文件

# You may obtain a copy of the License at
# 您可以在以下网址获取许可证副本

#     http://www.apache.org/licenses/LICENSE-2.0
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# 除非适用法律要求或书面同意，否则软件

# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 根据"原样"分发许可证，无论是明示还是暗示，不带任何形式的担保或条件

# See the License for the specific language governing permissions and
# 请查阅许可证，获取详细的权限说明及限制条款

# limitations under the License.
# 许可证下的限制条件

""" MobileViTV2 model configuration"""
# 文档字符串，指明本代码是关于MobileViTV2模型配置的

from collections import OrderedDict
# 导入OrderedDict类，用于创建有序字典

from typing import Mapping
# 导入Mapping类型提示，用于类型注解

from packaging import version
# 导入version模块，用于处理版本信息

from ...configuration_utils import PretrainedConfig
# 从...configuration_utils中导入PretrainedConfig类，用于继承模型配置

from ...onnx import OnnxConfig
# 从...onnx中导入OnnxConfig类，用于处理ONNX配置

from ...utils import logging
# 从...utils中导入logging模块，用于日志记录

logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器

MOBILEVITV2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "apple/mobilevitv2-1.0": "https://huggingface.co/apple/mobilevitv2-1.0/resolve/main/config.json",
}
# MobileViTV2预训练模型配置存档映射，指定模型名称及其对应的配置文件URL

class MobileViTV2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MobileViTV2Model`]. It is used to instantiate a
    MobileViTV2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MobileViTV2
    [apple/mobilevitv2-1.0](https://huggingface.co/apple/mobilevitv2-1.0) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # MobileViTV2Config类，用于存储MobileViTV2模型的配置信息

    def __init__(self, **kwargs):
        # 构造方法，初始化MobileViTV2Config对象
        super().__init__(**kwargs)
        # 调用父类构造方法初始化配置对象
    Args:
        num_channels (`int`, *optional*, defaults to 3):
            输入通道数，默认为3。
        image_size (`int`, *optional*, defaults to 256):
            每张图片的分辨率大小，默认为256像素。
        patch_size (`int`, *optional*, defaults to 2):
            每个图块的分辨率大小，默认为2像素。
        expand_ratio (`float`, *optional*, defaults to 2.0):
            MobileNetv2层的扩展因子，默认为2.0。
        hidden_act (`str` or `function`, *optional*, defaults to `"swish"`):
            Transformer编码器和卷积层中的非线性激活函数（函数或字符串），默认为"swish"。
        conv_kernel_size (`int`, *optional*, defaults to 3):
            MobileViTV2层中卷积核的大小，默认为3。
        output_stride (`int`, *optional*, defaults to 32):
            输出空间分辨率与输入图像分辨率之比，默认为32。
        classifier_dropout_prob (`float`, *optional*, defaults to 0.1):
            附加分类器的dropout比率，默认为0.1。
        initializer_range (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态初始化器的标准差，默认为0.02。
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            层归一化层使用的epsilon值，默认为1e-05。
        aspp_out_channels (`int`, *optional*, defaults to 512):
            语义分割中ASPP层使用的输出通道数，默认为512。
        atrous_rates (`List[int]`, *optional*, defaults to `[6, 12, 18]`):
            语义分割中ASPP层使用的扩张（空洞）率列表，默认为`[6, 12, 18]`。
        aspp_dropout_prob (`float`, *optional*, defaults to 0.1):
            语义分割中ASPP层的dropout比率，默认为0.1。
        semantic_loss_ignore_index (`int`, *optional*, defaults to 255):
            语义分割模型损失函数中被忽略的索引，默认为255。
        n_attn_blocks (`List[int]`, *optional*, defaults to `[2, 4, 3]`):
            每个MobileViTV2Layer中注意力块的数量列表，默认为`[2, 4, 3]`。
        base_attn_unit_dims (`List[int]`, *optional*, defaults to `[128, 192, 256]`):
            每个MobileViTV2Layer中注意力块维度的基础乘数列表，默认为`[128, 192, 256]`。
        width_multiplier (`float`, *optional*, defaults to 1.0):
            MobileViTV2的宽度乘数，默认为1.0。
        ffn_multiplier (`int`, *optional*, defaults to 2):
            MobileViTV2的FFN乘数，默认为2。
        attn_dropout (`float`, *optional*, defaults to 0.0):
            注意力层中的dropout比率，默认为0.0。
        ffn_dropout (`float`, *optional*, defaults to 0.0):
            FFN层之间的dropout比率，默认为0.0。

    Example:

    ```python
    >>> from transformers import MobileViTV2Config, MobileViTV2Model

    >>> # Initializing a mobilevitv2-small style configuration
    >>> configuration = MobileViTV2Config()
    # 初始化一个 MobileViTV2Model 模型，使用给定的配置信息进行配置
    model = MobileViTV2Model(configuration)

    # 访问模型的配置信息
    configuration = model.config
# 定义一个 MobileViTV2OnnxConfig 类，继承自 OnnxConfig 类
class MobileViTV2OnnxConfig(OnnxConfig):
    
    # 定义类变量 torch_onnx_minimum_version，指定最小的 Torch ONNX 版本为 1.11
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义 inputs 属性，返回一个有序字典，描述模型输入的维度顺序和名称
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict([("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"})])

    # 定义 outputs 属性，返回一个有序字典，根据任务类型返回不同的输出结构描述
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "image-classification":
            return OrderedDict([("logits", {0: "batch"})])
        else:
            return OrderedDict([("last_hidden_state", {0: "batch"}), ("pooler_output", {0: "batch"})])

    # 定义 atol_for_validation 属性，返回一个浮点数，指定验证过程中的容差阈值
    @property
    def atol_for_validation(self) -> float:
        return 1e-4
```