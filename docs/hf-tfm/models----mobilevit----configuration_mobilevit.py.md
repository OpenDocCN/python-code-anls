# `.\models\mobilevit\configuration_mobilevit.py`

```
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
# 上面是版权声明和编码声明

# 导入所需的库和模块
from collections import OrderedDict  # 导入有序字典模块
from typing import Mapping  # 导入 Mapping 类型的声明

from packaging import version  # 导入版本信息的包

from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...onnx import OnnxConfig  # 导入 ONNX 配置类
from ...utils import logging  # 导入日志工具模块

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义预训练模型配置文件的映射字典
MOBILEVIT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "apple/mobilevit-small": "https://huggingface.co/apple/mobilevit-small/resolve/main/config.json",
    "apple/mobilevit-x-small": "https://huggingface.co/apple/mobilevit-x-small/resolve/main/config.json",
    "apple/mobilevit-xx-small": "https://huggingface.co/apple/mobilevit-xx-small/resolve/main/config.json",
    "apple/deeplabv3-mobilevit-small": (
        "https://huggingface.co/apple/deeplabv3-mobilevit-small/resolve/main/config.json"
    ),
    "apple/deeplabv3-mobilevit-x-small": (
        "https://huggingface.co/apple/deeplabv3-mobilevit-x-small/resolve/main/config.json"
    ),
    "apple/deeplabv3-mobilevit-xx-small": (
        "https://huggingface.co/apple/deeplabv3-mobilevit-xx-small/resolve/main/config.json"
    ),
    # 查看所有 MobileViT 模型 https://huggingface.co/models?filter=mobilevit
}

# MobileViTConfig 类，继承自 PretrainedConfig，用于存储 MobileViT 模型的配置信息
class MobileViTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MobileViTModel`]. It is used to instantiate a
    MobileViT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MobileViT
    [apple/mobilevit-small](https://huggingface.co/apple/mobilevit-small) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```python
    >>> from transformers import MobileViTConfig, MobileViTModel

    >>> # Initializing a mobilevit-small style configuration
    >>> configuration = MobileViTConfig()

    >>> # Initializing a model from the mobilevit-small style configuration
    >>> model = MobileViTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """
    
    model_type = "mobilevit"  # 模型类型为 "mobilevit"
    # 定义一个初始化方法，用于初始化神经网络模型的各种参数和属性
    def __init__(
        self,
        num_channels=3,  # 图像的通道数，默认为3（RGB）
        image_size=256,  # 图像尺寸，默认为256x256像素
        patch_size=2,  # 感受野(patch)的大小，默认为2x2像素
        hidden_sizes=[144, 192, 240],  # 编码器和解码器中隐藏层的大小列表
        neck_hidden_sizes=[16, 32, 64, 96, 128, 160, 640],  # 语义分割网络中颈部的隐藏层大小列表
        num_attention_heads=4,  # 注意力头的数量，默认为4
        mlp_ratio=2.0,  # MLP扩展比例，默认为2.0
        expand_ratio=4.0,  # 扩展比例，默认为4.0
        hidden_act="silu",  # 隐藏层激活函数，默认为SILU（Sigmoid-weighted Linear Unit）
        conv_kernel_size=3,  # 卷积核大小，默认为3x3
        output_stride=32,  # 输出步长，默认为32
        hidden_dropout_prob=0.1,  # 隐藏层的Dropout概率，默认为0.1
        attention_probs_dropout_prob=0.0,  # 注意力概率的Dropout概率，默认为0.0
        classifier_dropout_prob=0.1,  # 分类器的Dropout概率，默认为0.1
        initializer_range=0.02,  # 初始化器范围，默认为0.02
        layer_norm_eps=1e-5,  # Layer Normalization的epsilon值，默认为1e-5
        qkv_bias=True,  # 是否在QKV注意力机制中使用偏置，默认为True
    
        # 语义分割网络中ASPP（空洞空间金字塔池化）模块的输出通道数
        aspp_out_channels=256,
        # ASPP模块中不同尺度空洞率的列表
        atrous_rates=[6, 12, 18],
        aspp_dropout_prob=0.1,  # ASPP模块的Dropout概率，默认为0.1
        semantic_loss_ignore_index=255,  # 语义损失函数中的忽略索引，默认为255
        **kwargs,  # 其他可能传递的参数
    ):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
    
        # 将传入的参数赋值给对象的属性
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_sizes = hidden_sizes
        self.neck_hidden_sizes = neck_hidden_sizes
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.expand_ratio = expand_ratio
        self.hidden_act = hidden_act
        self.conv_kernel_size = conv_kernel_size
        self.output_stride = output_stride
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.classifier_dropout_prob = classifier_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
    
        # 语义分割头部的属性设置
        self.aspp_out_channels = aspp_out_channels
        self.atrous_rates = atrous_rates
        self.aspp_dropout_prob = aspp_dropout_prob
        self.semantic_loss_ignore_index = semantic_loss_ignore_index
# 定义一个名为 MobileViTOnnxConfig 的类，它继承自 OnnxConfig 类
class MobileViTOnnxConfig(OnnxConfig):
    
    # 设置 torch_onnx_minimum_version 属性为解析后的版本号 "1.11"
    torch_onnx_minimum_version = version.parse("1.11")
    
    # 定义一个 inputs 属性，返回一个有序字典，描述模型输入的结构
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict([("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"})])
    
    # 定义一个 outputs 属性，返回一个有序字典，描述模型输出的结构
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "image-classification":
            return OrderedDict([("logits", {0: "batch"})])
        else:
            return OrderedDict([("last_hidden_state", {0: "batch"}), ("pooler_output", {0: "batch"})])
    
    # 定义一个 atol_for_validation 属性，返回浮点数 1e-4，表示验证时的绝对误差容限
    @property
    def atol_for_validation(self) -> float:
        return 1e-4
```