# `.\transformers\models\mobilevitv2\configuration_mobilevitv2.py`

```
# 设置文件编码为 UTF-8
# 版权声明及许可信息
#
# 版权 2023 年由 HuggingFace Inc. 团队保留。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（"许可证"）获得许可;
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下网址获得许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按"原样"分发，
# 无任何明示或暗示的保证或条件。
# 请参阅许可证以获取特定于许可证的语言和权限。
""" MobileViTV2 模型配置"""

# 从 collections 模块导入 OrderedDict 类
# 从 typing 模块导入 Mapping 类型
from collections import OrderedDict
from typing import Mapping

# 从 packaging 模块导入 version 函数
from packaging import version

# 从 ... 目录下的 configuration_utils 模块导入 PretrainedConfig 类
# 从 ... 目录下的 onnx 模块导入 OnnxConfig 类
# 从 ... 目录下的 utils 模块导入 logging 函数
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging

# 获取名为 __name__ 的 logger 对象
logger = logging.get_logger(__name__)

# 定义 MobileViTV2 预训练配置文件的存档映射字典
MOBILEVITV2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "apple/mobilevitv2-1.0": "https://huggingface.co/apple/mobilevitv2-1.0/resolve/main/config.json",
}

# 定义 MobileViTV2Config 类，继承自 PretrainedConfig 类
class MobileViTV2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MobileViTV2Model`]. It is used to instantiate a
    MobileViTV2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MobileViTV2
    [apple/mobilevitv2-1.0](https://huggingface.co/apple/mobilevitv2-1.0) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        num_channels (`int`, *optional*, defaults to 3):
            输入通道的数量，默认为3。
        image_size (`int`, *optional*, defaults to 256):
            每张图片的大小，默认为256。
        patch_size (`int`, *optional*, defaults to 2):
            每个patch的大小，默认为2。
        expand_ratio (`float`, *optional*, defaults to 2.0):
            MobileNetv2层的扩张因子，默认为2.0。
        hidden_act (`str` or `function`, *optional*, defaults to `"swish"`):
            Transformer编码器和卷积层中的非线性激活函数（函数或字符串），默认为"swish"。
        conv_kernel_size (`int`, *optional*, defaults to 3):
            MobileViTV2层中卷积核的大小，默认为3。
        output_stride (`int`, *optional*, defaults to 32):
            输出空间分辨率与输入图像分辨率之间的比率，默认为32。
        classifier_dropout_prob (`float`, *optional*, defaults to 0.1):
            附加分类器的dropout比率，默认为0.1。
        initializer_range (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态分布的标准差，默认为0.02。
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            层标准化层使用的epsilon值，默认为1e-05。
        aspp_out_channels (`int`, *optional*, defaults to 512):
            用于语义分割的ASPP层的输出通道数量，默认为512。
        atrous_rates (`List[int]`, *optional*, defaults to `[6, 12, 18]`):
            用于语义分割的ASPP层中的膨胀（atrous）因子，默认为`[6, 12, 18]`。
        aspp_dropout_prob (`float`, *optional*, defaults to 0.1):
            用于语义分割的ASPP层的dropout比率，默认为0.1。
        semantic_loss_ignore_index (`int`, *optional*, defaults to 255):
            语义分割模型的损失函数中被忽略的索引，默认为255。
        n_attn_blocks (`List[int]`, *optional*, defaults to `[2, 4, 3]`):
            每个MobileViTV2Layer中的注意力块的数量，默认为`[2, 4, 3]`。
        base_attn_unit_dims (`List[int]`, *optional*, defaults to `[128, 192, 256]`):
            每个MobileViTV2Layer中注意力块维度的基础乘法因子，默认为`[128, 192, 256]`。
        width_multiplier (`float`, *optional*, defaults to 1.0):
            MobileViTV2的宽度乘法因子，默认为1.0。
        ffn_multiplier (`int`, *optional*, defaults to 2):
            MobileViTV2的FFN乘法因子，默认为2。
        attn_dropout (`float`, *optional*, defaults to 0.0):
            注意力层中的dropout率，默认为0.0。
        ffn_dropout (`float`, *optional*, defaults to 0.0):
            FFN层之间的dropout率，默认为0.0。

    Example:

    ```python
    >>> from transformers import MobileViTV2Config, MobileViTV2Model

    >>> # 初始化一个mobilevitv2-small风格的配置
    >>> configuration = MobileViTV2Config()
    # 从 mobilevitv2-small 风格的配置中初始化一个模型
    model = MobileViTV2Model(configuration)
    
    # 访问模型配置
    configuration = model.config
# 定义 MobileViTV2OnnxConfig 类，继承自 OnnxConfig 类
class MobileViTV2OnnxConfig(OnnxConfig):
    # 设置支持的最小 PyTorch ONNX 版本为 1.11
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义输入张量的属性
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 返回一个有序字典，包含输入张量 "pixel_values" 的维度信息
        return OrderedDict([("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"})])

    # 定义输出张量的属性
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        # 根据任务类型返回不同的输出张量信息
        if self.task == "image-classification":
            # 对于图像分类任务，输出张量为 "logits"，其第一个维度为 "batch"
            return OrderedDict([("logits", {0: "batch"})])
        else:
            # 对于其他任务，输出张量为 "last_hidden_state" 和 "pooler_output"，第一个维度为 "batch"
            return OrderedDict([("last_hidden_state", {0: "batch"}), ("pooler_output", {0: "batch"})])

    # 定义验证过程中允许的绝对误差
    @property
    def atol_for_validation(self) -> float:
        # 设置绝对误差阈值为 1e-4
        return 1e-4
```