# `.\models\swiftformer\configuration_swiftformer.py`

```
# coding=utf-8
# 版权所有 2023 MBZUAI 和 The HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件按“原样”分发，
# 没有任何明示或暗示的保证或条件。
# 有关特定语言的权限，请参阅许可证。
""" SwiftFormer 模型配置 """

from collections import OrderedDict  # 导入有序字典类
from typing import Mapping  # 导入映射类型

from packaging import version  # 导入版本包

from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...onnx import OnnxConfig  # 导入ONNX配置类
from ...utils import logging  # 导入日志工具

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

# SwiftFormer 预训练配置文件映射，指定了模型的预训练配置文件
SWIFTFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "MBZUAI/swiftformer-xs": "https://huggingface.co/MBZUAI/swiftformer-xs/resolve/main/config.json",
}


class SwiftFormerConfig(PretrainedConfig):
    r"""
    这是配置类，用于存储 [`SwiftFormerModel`] 的配置。根据指定的参数实例化一个 SwiftFormer 模型，定义模型的体系结构。
    使用默认值实例化配置将产生与 SwiftFormer [MBZUAI/swiftformer-xs](https://huggingface.co/MBZUAI/swiftformer-xs) 架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。有关更多信息，请阅读 [`PretrainedConfig`] 的文档。
    """
    # 定义模型类型为 "swiftformer"
    model_type = "swiftformer"
    
    # 初始化 SwiftFormerConfig 类，设定模型的各项配置参数
    def __init__(
        self,
        num_channels=3,  # 输入通道数，默认为 3
        depths=[3, 3, 6, 4],  # 每个阶段的深度列表，默认为 [3, 3, 6, 4]
        embed_dims=[48, 56, 112, 220],  # 每个阶段的嵌入维度列表，默认为 [48, 56, 112, 220]
        mlp_ratio=4,  # MLP 隐藏层维度与输入维度之比，默认为 4
        downsamples=[True, True, True, True],  # 每个阶段是否进行下采样的布尔列表，默认为 [True, True, True, True]
        hidden_act="gelu",  # 非线性激活函数类型，默认为 "gelu"
        down_patch_size=3,  # 下采样层的补丁大小，默认为 3
        down_stride=2,  # 下采样层卷积核的步幅，默认为 2
        down_pad=1,  # 下采样层的填充大小，默认为 1
        drop_path_rate=0.0,  # DropPath 增加 dropout 概率的比率，默认为 0.0
        use_layer_scale=True,  # 是否对来自令牌混合器的输出进行缩放，默认为 True
        layer_scale_init_value=1e-5,  # 令牌混合器输出缩放的初始值，默认为 1e-5
        batch_norm_eps=1e-5,  # 批量归一化层使用的 epsilon 值，默认为 1e-5
        **kwargs,  # 其他未命名参数，用于接收额外的配置参数
    ):
        ):
        # 调用父类的初始化方法，传入所有的关键字参数
        super().__init__(**kwargs)
        # 设置当前对象的通道数
        self.num_channels = num_channels
        # 设置深度信息
        self.depths = depths
        # 设置嵌入维度信息
        self.embed_dims = embed_dims
        # 设置MLP的比率
        self.mlp_ratio = mlp_ratio
        # 设置下采样信息
        self.downsamples = downsamples
        # 设置隐藏层激活函数
        self.hidden_act = hidden_act
        # 设置下采样的补丁大小
        self.down_patch_size = down_patch_size
        # 设置下采样的步长
        self.down_stride = down_stride
        # 设置下采样的填充
        self.down_pad = down_pad
        # 设置丢弃路径率
        self.drop_path_rate = drop_path_rate
        # 是否使用层尺度
        self.use_layer_scale = use_layer_scale
        # 初始化层尺度的值
        self.layer_scale_init_value = layer_scale_init_value
        # 设置批归一化的eps值
        self.batch_norm_eps = batch_norm_eps
# 定义 SwiftFormerOnnxConfig 类，继承自 OnnxConfig 类
class SwiftFormerOnnxConfig(OnnxConfig):
    # 设定 torch_onnx_minimum_version 属性，要求最低版本为 1.11
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义 inputs 属性，返回一个有序字典，用于描述模型输入的结构
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                # 描述模型输入的具体信息，包括像素值和对应的维度顺序
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    # 定义 atol_for_validation 属性，返回一个浮点数，表示验证过程中的绝对误差容忍度
    @property
    def atol_for_validation(self) -> float:
        return 1e-4
```