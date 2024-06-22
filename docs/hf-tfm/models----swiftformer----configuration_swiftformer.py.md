# `.\transformers\models\swiftformer\configuration_swiftformer.py`

```py
# 设置编码格式为 UTF-8
# 版权声明
# Copyright 2023 MBZUAI and The HuggingFace Inc. team. All rights reserved.
#
# 根据 Apache 许可证 2.0 版本使用本文件，除非你遵循许可证规定，否则你不能使用这个文件。
# 你可以在以下网址获得许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按"原样"提供，不提供任何形式的担保或条件。
# 请查阅许可证了解详细信息。

""" SwiftFormer 模型配置"""

# 导入所需模块
from collections import OrderedDict
from typing import Mapping

from packaging import version

# 从 HuggingFace 库导入相关工具
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练模型配置文件存档映射
SWIFTFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "MBZUAI/swiftformer-xs": "https://huggingface.co/MBZUAI/swiftformer-xs/resolve/main/config.json",
}


class SwiftFormerConfig(PretrainedConfig):
    r"""
    这是用于存储[`SwiftFormerModel`]配置的配置类。它用于根据指定的参数实例化SwiftFormer模型，定义模型架构。
    使用默认值实例化配置将产生类似于SwiftFormer [MBZUAI/swiftformer-xs](https://huggingface.co/MBZUAI/swiftformer-xs)
    架构的配置。

    配置对象继承自[`PretrainedConfig`]，可用于控制模型输出。阅读[`PretrainedConfig`]的文档以获取更多信息。
    """
    # 参数设置部分
    Args:
        # 输入通道数，默认为3
        num_channels (`int`, *optional*, defaults to 3):
        # 每个阶段的深度，默认为[3, 3, 6, 4]
        depths (`List[int]`, *optional*, defaults to `[3, 3, 6, 4]`):
        # 每个阶段的嵌入维度，默认为[48, 56, 112, 220]
        embed_dims (`List[int]`, *optional*, defaults to `[48, 56, 112, 220]`):
        # MLP隐藏维度与输入维度之间的比例，默认为4
        mlp_ratio (`int`, *optional*, defaults to 4):
        # 是否在两个阶段之间对输入进行下采样，默认为[True, True, True, True]
        downsamples (`List[bool]`, *optional*, defaults to `[True, True, True, True]`):
        # 隐藏层激活函数，默认为"gelu"
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
        # 下采样层中的补丁大小，默认为3
        down_patch_size (`int`, *optional*, defaults to 3):
        # 下采样层中卷积核的步长，默认为2
        down_stride (`int`, *optional*, defaults to 2):
        # 下采样层中的填充，默认为1
        down_pad (`int`, *optional*, defaults to 1):
        # DropPath中增加的丢弃概率，默认为0.0
        drop_path_rate (`float`, *optional*, defaults to 0.0):
        # 是否从令牌混合器中缩放输出，默认为True
        use_layer_scale (`bool`, *optional*, defaults to `True`):
        # 从令牌混合器中缩放输出的初始化值，默认为1e-5
        layer_scale_init_value (`float`, *optional*, defaults to 1e-05):
        # 批量归一化层使用的epsilon，默认为1e-5
        batch_norm_eps (`float`, *optional*, defaults to 1e-05):

    # 示例代码
    Example:

    ```python
    >>> from transformers import SwiftFormerConfig, SwiftFormerModel

    >>> # 初始化一个SwiftFormer swiftformer-base-patch16-224风格的配置
    >>> configuration = SwiftFormerConfig()

    >>> # 用swiftformer-base-patch16-224风格的配置初始化一个具有随机权重的模型
    >>> model = SwiftFormerModel(configuration)

    >>> # 获取模型配置
    >>> configuration = model.config
    ```py"""

    # 模型类型
    model_type = "swiftformer"

    def __init__(
        self,
        num_channels=3,
        depths=[3, 3, 6, 4],
        embed_dims=[48, 56, 112, 220],
        mlp_ratio=4,
        downsamples=[True, True, True, True],
        hidden_act="gelu",
        down_patch_size=3,
        down_stride=2,
        down_pad=1,
        drop_path_rate=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        batch_norm_eps=1e-5,
        **kwargs,
        ):
        # 调用父类的构造函数，传入关键字参数
        super().__init__(**kwargs)
        # 设置神经网络的通道数
        self.num_channels = num_channels
        # 设置深度信息
        self.depths = depths
        # 设置嵌入维度
        self.embed_dims = embed_dims
        # 设置MLP比率
        self.mlp_ratio = mlp_ratio
        # 设置下采样信息
        self.downsamples = downsamples
        # 设置隐藏层激活函数类型
        self.hidden_act = hidden_act
        # 设置下采样的路径大小
        self.down_patch_size = down_patch_size
        # 设置下采样的步进大小
        self.down_stride = down_stride
        # 设置下采样的填充大小
        self.down_pad = down_pad
        # 设置丢弃路径的比率
        self.drop_path_rate = drop_path_rate
        # 是否使用层规模
        self.use_layer_scale = use_layer_scale
        # 设置层规模初始值
        self.layer_scale_init_value = layer_scale_init_value
        # 批次规范化的 epsilon 值
        self.batch_norm_eps = batch_norm_eps
# 创建 SwiftFormerOnnxConfig 类，继承自 OnnxConfig
class SwiftFormerOnnxConfig(OnnxConfig):
    # 设置 torch_onnx_minimum_version 属性为最低版本号1.11
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义 inputs 属性，返回一个有序字典，表示输入数据的结构
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    # 定义 atol_for_validation 属性，返回用于验证的容差值
    @property
    def atol_for_validation(self) -> float:
        return 1e-4
```