# `.\models\efficientnet\configuration_efficientnet.py`

```
# coding=utf-8
# 版权声明
# 版权所有2023年 Google Research, Inc. 和 The HuggingFace Inc. team. 保留所有权利。
#
# 根据 Apache 许可证 2.0 版本进行授权。
# 除非符合许可证的规定，否则您不能使用此文件。
# 您可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或经书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，没有任何明示或暗示的担保或条件。
# 请查看许可证以了解特定语言的权限和限制。
""" EfficientNet model configuration"""

# 导入必要的包
from collections import OrderedDict
from typing import List, Mapping
from packaging import version
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging

# 获取logger对象
logger = logging.get_logger(__name__)

# EfficientNet 预训练配置映射，包括模型名和配置文件下载链接
EFFICIENTNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/efficientnet-b7": "https://huggingface.co/google/efficientnet-b7/resolve/main/config.json",
}


# EfficientNet 配置类，用于存储 [`EfficientNetModel`] 的配置
class EfficientNetConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`EfficientNetModel`]. It is used to instantiate an
    EfficientNet model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the EfficientNet
    [google/efficientnet-b7](https://huggingface.co/google/efficientnet-b7) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:
    ```python
    >>> from transformers import EfficientNetConfig, EfficientNetModel

    >>> # Initializing a EfficientNet efficientnet-b7 style configuration
    >>> configuration = EfficientNetConfig()

    >>> # Initializing a model (with random weights) from the efficientnet-b7 style configuration
    >>> model = EfficientNetModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    # 模型类型
    model_type = "efficientnet"
    # 初始化函数，用于初始化神经网络模型的参数
    def __init__(
        self,
        # 输入通道数，默认为3
        num_channels: int = 3,
        # 图像大小，默认为600
        image_size: int = 600,
        # 宽度系数，默认为2.0
        width_coefficient: float = 2.0,
        # 深度系数，默认为3.1
        depth_coefficient: float = 3.1,
        # 深度除数，默认为8
        depth_divisor: int = 8,
        # 卷积核尺寸列表，默认为[3, 3, 5, 3, 5, 5, 3]
        kernel_sizes: List[int] = [3, 3, 5, 3, 5, 5, 3],
        # 输入通道数列表，默认为[32, 16, 24, 40, 80, 112, 192]
        in_channels: List[int] = [32, 16, 24, 40, 80, 112, 192],
        # 输出通道数列表，默认为[16, 24, 40, 80, 112, 192, 320]
        out_channels: List[int] = [16, 24, 40, 80, 112, 192, 320],
        # 深度可分离卷积的填充列表，默认为空
        depthwise_padding: List[int] = [],
        # 步长列表，默认为[1, 2, 2, 2, 1, 2, 1]
        strides: List[int] = [1, 2, 2, 2, 1, 2, 1],
        # 每个模块的重复次数列表，默认为[1, 2, 2, 3, 3, 4, 1]
        num_block_repeats: List[int] = [1, 2, 2, 3, 3, 4, 1],
        # 扩张比例列表，默认为[1, 6, 6, 6, 6, 6, 6]
        expand_ratios: List[int] = [1, 6, 6, 6, 6, 6, 6],
        # Squeeze-and-Excitation模块的压缩比例，默认为0.25
        squeeze_expansion_ratio: float = 0.25,
        # 隐藏层激活函数，默认为"swish"
        hidden_act: str = "swish",
        # 隐藏层维度，默认为2560
        hidden_dim: int = 2560,
        # 池化类型，默认为"mean"
        pooling_type: str = "mean",
        # 初始化范围，默认为0.02
        initializer_range: float = 0.02,
        # 批归一化的epsilon，默认为0.001
        batch_norm_eps: float = 0.001,
        # 批归一化的动量，默认为0.99
        batch_norm_momentum: float = 0.99,
        # Dropout率，默认为0.5
        dropout_rate: float = 0.5,
        # DropConnect率，默认为0.2
        drop_connect_rate: float = 0.2,
        # 其他关键字参数
        **kwargs,
    ):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 初始化网络模型参数
        self.num_channels = num_channels
        self.image_size = image_size
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.depth_divisor = depth_divisor
        self.kernel_sizes = kernel_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depthwise_padding = depthwise_padding
        self.strides = strides
        self.num_block_repeats = num_block_repeats
        self.expand_ratios = expand_ratios
        self.squeeze_expansion_ratio = squeeze_expansion_ratio
        self.hidden_act = hidden_act
        self.hidden_dim = hidden_dim
        self.pooling_type = pooling_type
        self.initializer_range = initializer_range
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum
        self.dropout_rate = dropout_rate
        self.drop_connect_rate = drop_connect_rate
        # 计算总的隐藏层数
        self.num_hidden_layers = sum(num_block_repeats) * 4
# 定义一个用于配置EfficientNet ONNX模型的类，继承自OnnxConfig类
class EfficientNetOnnxConfig(OnnxConfig):
    # 设置torch_onnx_minimum_version属性为1.11版本
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义inputs属性，返回一个有序字典，表示模型的输入信息
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                # 设置模型的输入名称及其对应的维度信息
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    # 定义atol_for_validation属性，返回用于验证的绝对容差值
    @property
    def atol_for_validation(self) -> float:
        return 1e-5
```