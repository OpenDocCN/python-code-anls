# `.\transformers\models\resnet\configuration_resnet.py`

```py
# coding=utf-8
# Copyright 2022 Microsoft Research, Inc. and The HuggingFace Inc. team. All rights reserved.
# 版权声明和许可证信息

# 导入必要的库
from collections import OrderedDict  # 导入OrderedDict类，用于创建有序字典
from typing import Mapping  # 导入Mapping类型，用于类型提示

# 导入版本相关的库
from packaging import version  # 导入version模块，用于处理版本信息

# 导入配置相关的库和模块
from ...configuration_utils import PretrainedConfig  # 导入PretrainedConfig类，用于存储预训练模型的配置
from ...onnx import OnnxConfig  # 导入OnnxConfig类，用于存储ONNX模型的配置
from ...utils import logging  # 导入logging模块，用于日志记录
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices  # 导入BackboneConfigMixin类和get_aligned_output_features_output_indices函数，用于处理骨干网络的配置信息

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练的 ResNet 模型的配置文件映射
RESNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/resnet-50": "https://huggingface.co/microsoft/resnet-50/blob/main/config.json",
}

# ResNet 模型配置类，继承自 BackboneConfigMixin 和 PretrainedConfig
class ResNetConfig(BackboneConfigMixin, PretrainedConfig):
    r"""
    这是用于存储 [`ResNetModel`] 配置的配置类。它用于根据指定的参数实例化一个 ResNet 模型，定义模型的架构。
    使用默认值实例化一个配置将会产生一个类似于 ResNet [microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可以用于控制模型的输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。
    """
    # ResNetConfig 类的参数说明
    Args:
        # 输入通道数，默认为 3
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        # 嵌入层的特征维度，默认为 64
        embedding_size (`int`, *optional*, defaults to 64):
            Dimensionality (hidden size) for the embedding layer.
        # 每个阶段的隐藏层大小，默认为 [256, 512, 1024, 2048]
        hidden_sizes (`List[int]`, *optional*, defaults to `[256, 512, 1024, 2048]`):
            Dimensionality (hidden size) at each stage.
        # 每个阶段的层数，默认为 [3, 4, 6, 3]
        depths (`List[int]`, *optional*, defaults to `[3, 4, 6, 3]`):
            Depth (number of layers) for each stage.
        # 层类型，可以是 "basic" 或 "bottleneck"，默认为 "bottleneck"
        layer_type (`str`, *optional*, defaults to `"bottleneck"`):
            The layer to use, it can be either `"basic"` (used for smaller models, like resnet-18 or resnet-34) or
            `"bottleneck"` (used for larger models like resnet-50 and above).
        # 非线性激活函数，默认为 "relu"
        hidden_act (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function in each block. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"`
            are supported.
        # 是否在第一个阶段进行下采样，默认为 False
        downsample_in_first_stage (`bool`, *optional*, defaults to `False`):
            If `True`, the first stage will downsample the inputs using a `stride` of 2.
        # 是否在瓶颈层进行下采样，默认为 False
        downsample_in_bottleneck (`bool`, *optional*, defaults to `False`):
            If `True`, the first conv 1x1 in ResNetBottleNeckLayer will downsample the inputs using a `stride` of 2.
        # 如果作为 backbone 使用，指定输出的特征层，默认为最后一个阶段
        out_features (`List[str]`, *optional*):
            If used as backbone, list of features to output. Can be any of `"stem"`, `"stage1"`, `"stage2"`, etc.
            (depending on how many stages the model has). If unset and `out_indices` is set, will default to the
            corresponding stages. If unset and `out_indices` is unset, will default to the last stage. Must be in the
            same order as defined in the `stage_names` attribute.
        # 如果作为 backbone 使用，指定输出特征层的索引，默认为最后一个阶段
        out_indices (`List[int]`, *optional*):
            If used as backbone, list of indices of features to output. Can be any of 0, 1, 2, etc. (depending on how
            many stages the model has). If unset and `out_features` is set, will default to the corresponding stages.
            If unset and `out_features` is unset, will default to the last stage. Must be in the
            same order as defined in the `stage_names` attribute.
    
    Example:
    
    >>> from transformers import ResNetConfig, ResNetModel
    
    >>> # Initializing a ResNet resnet-50 style configuration
    >>> configuration = ResNetConfig()
    
    >>> # Initializing a model (with random weights) from the resnet-50 style configuration
    >>> model = ResNetModel(configuration)
    
    >>> # Accessing the model configuration
    >>> configuration = model.config
    
    
    # ResNetConfig 类信息
    model_type = "resnet"
    layer_types = ["basic", "bottleneck"]
    def __init__(
        self,
        num_channels=3,
        embedding_size=64,
        hidden_sizes=[256, 512, 1024, 2048],
        depths=[3, 4, 6, 3],
        layer_type="bottleneck",
        hidden_act="relu",
        downsample_in_first_stage=False,
        downsample_in_bottleneck=False,
        out_features=None,
        out_indices=None,
        **kwargs,
    ):
        # 调用父类初始化方法
        super().__init__(**kwargs)
        # 检查 layer_type 是否合法
        if layer_type not in self.layer_types:
            raise ValueError(f"layer_type={layer_type} is not one of {','.join(self.layer_types)}")
        # 初始化网络参数
        self.num_channels = num_channels
        self.embedding_size = embedding_size
        self.hidden_sizes = hidden_sizes
        self.depths = depths
        self.layer_type = layer_type
        self.hidden_act = hidden_act
        self.downsample_in_first_stage = downsample_in_first_stage
        self.downsample_in_bottleneck = downsample_in_bottleneck
        # 初始化网络各个阶段名称
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(depths) + 1)]
        # 获取对齐的输出特征和输出索引
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )
# 定义一个名为 ResNetOnnxConfig 的类，它是 OnnxConfig 类的子类
class ResNetOnnxConfig(OnnxConfig):
    # 设置 torch_onnx_minimum_version 属性为 1.11，指定了 Torch 和 ONNX 的最小版本要求
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义 inputs 属性为一个映射，表示输入数据的结构
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 返回一个有序字典，表示输入数据的排列顺序和名称
        return OrderedDict(
            [
                # 键为 "pixel_values"，值为字典，表示对应的维度名称和位置
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    # 定义 atol_for_validation 属性，表示用于验证的绝对容差
    @property
    def atol_for_validation(self) -> float:
        # 返回验证用的绝对容差值
        return 1e-3
```