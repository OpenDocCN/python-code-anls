# `.\transformers\models\pvt\configuration_pvt.py`

```
# coding=utf-8
# 版权 2023 作者：Wenhai Wang、Enze Xie、Xiang Li、Deng-Ping Fan、
# Kaitao Song、Ding Liang、Tong Lu、Ping Luo、Ling Shao 和 HuggingFace Inc. 团队。
# 保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（"许可证"）进行许可；
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按"原样"分发，
# 不附带任何明示或暗示的保证或条件。
# 有关特定语言的权限，请参阅许可证。
""" Pvt 模型配置"""

# 从 collections 模块中导入 OrderedDict 类
from collections import OrderedDict
# 从 typing 模块中导入 Callable、List、Mapping 类
from typing import Callable, List, Mapping

# 从 packaging 模块中导入 version 函数
from packaging import version

# 从父目录中导入 configuration_utils 模块
from ...configuration_utils import PretrainedConfig
# 从父目录中导入 OnnxConfig 类
from ...onnx import OnnxConfig
# 从父目录中导入 logging 函数
from ...utils import logging

# 获取当前模块的 logger
logger = logging.get_logger(__name__)

# 定义 PVT 预训练配置和存档映射字典
PVT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "pvt-tiny-224": "https://huggingface.co/Zetatech/pvt-tiny-224",
    # 查看所有 PVT 模型 https://huggingface.co/models?filter=pvt
}


class PvtConfig(PretrainedConfig):
    r"""
    这是存储 [`PvtModel`] 配置的配置类。它用于根据指定的参数实例化 Pvt 模型，定义模型架构。使用默认值实例化配置将产生类似于 Pvt
    [Xrenya/pvt-tiny-224](https://huggingface.co/Xrenya/pvt-tiny-224) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。有关更多信息，请阅读来自 [`PretrainedConfig`] 的文档。
    # 定义函数 read_zip，参数如下：
    # - image_size：输入图像尺寸，默认为 224
    # - num_channels：输入通道数，默认为 3
    # - num_encoder_blocks：编码器块的数量，默认为 4
    # - depths：每个编码器块中层的数量，默认为 `[2, 2, 2, 2]`
    # - sequence_reduction_ratios：每个编码器块中的序列缩减比率，默认为 `[8, 4, 2, 1]`
    # - hidden_sizes：每个编码器块的维度，默认为 `[64, 128, 320, 512]`
    # - patch_sizes：每个编码器块之前的补丁大小，默认为 `[4, 2, 2, 2]`
    # - strides：每个编码器块之前的步幅，默认为 `[4, 2, 2, 2]`
    # - num_attention_heads：每个块中注意力层的注意头数，默认为 `[1, 2, 5, 8]`
    # - mlp_ratios：Mix FFNs 中隐藏层大小与输入层大小的比率，默认为 `[8, 8, 4, 4]`
    # - hidden_act：编码器和池化器中的非线性激活函数，默认为 "gelu"
    # - hidden_dropout_prob：嵌入、编码器和池化器中所有全连接层的丢失概率，默认为 0.0
    # - attention_probs_dropout_prob：注意力概率的丢失比例，默认为 0.0
    # - initializer_range：初始化所有权重矩阵的截断正态初始化器的标准差，默认为 0.02
    # - drop_path_rate：随机深度中块的丢失概率，默认为 0.0
    # - layer_norm_eps：层归一化层使用的 epsilon，默认为 1e-06
    # - qkv_bias：是否应为查询、键和值添加可学习的偏置，默认为 True
    # - num_labels：类的数量，默认为 1000
    # 示例使用示例
    # - 从 transformers 库导入 PvtModel、PvtConfig
    # - 初始化一个 PVT Xrenya/pvt-tiny-224 风格的配置
    # - 从 Xrenya/pvt-tiny-224 风格的配置初始化一个模型
    # 创建一个PvtModel实例，使用给定的配置信息
    model = PvtModel(configuration)

    # 访问模型的配置信息
    configuration = model.config
# 创建一个私有的 OnnxConfig 类，它继承自 OnnxConfig
class PvtOnnxConfig(OnnxConfig):
    # 设置 torch_onnx_minimum_version 属性为版本 1.11，用于指定 Torch 版本的最小要求
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义 inputs 属性，指定模型输入的名称和维度顺序
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 返回一个有序字典，包含模型输入名称和对应的维度信息
        return OrderedDict(
            [
                # 定义模型输入为 "pixel_values"，并指定其维度顺序为 (batch, num_channels, height, width)
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    # 定义 atol_for_validation 属性，指定用于验证的绝对误差容差值
    @property
    def atol_for_validation(self) -> float:
        # 返回绝对误差容差值为 1e-4
        return 1e-4

    # 定义 default_onnx_opset 属性，指定默认的 ONNX 操作集版本
    @property
    def default_onnx_opset(self) -> int:
        # 返回默认的 ONNX 操作集版本为 12
        return 12
```