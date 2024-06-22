# `.\transformers\models\mobilenet_v1\configuration_mobilenet_v1.py`

```py
# 设定文件编码为 UTF-8
# 版权声明
# 版权 2022 年 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件，
# 无任何明示或暗示的保证或条件。
# 有关特定语言的权限，请参阅许可证。
""" MobileNetV1 模型配置"""

# 导入所需库
from collections import OrderedDict
from typing import Mapping

from packaging import version

# 导入配置相关的类和函数
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练 MobileNetV1 模型配置文件映射字典
MOBILENET_V1_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/mobilenet_v1_1.0_224": "https://huggingface.co/google/mobilenet_v1_1.0_224/resolve/main/config.json",
    "google/mobilenet_v1_0.75_192": "https://huggingface.co/google/mobilenet_v1_0.75_192/resolve/main/config.json",
    # 在 https://huggingface.co/models?filter=mobilenet_v1 查看所有 MobileNetV1 模型
}

# MobileNetV1 配置类
class MobileNetV1Config(PretrainedConfig):
    r"""
    这是用于存储 [`MobileNetV1Model`] 配置的配置类。它用于根据指定的参数实例化 MobileNetV1 模型，定义模型架构。
    使用默认值实例化配置将产生类似于 MobileNetV1 [google/mobilenet_v1_1.0_224](https://huggingface.co/google/mobilenet_v1_1.0_224) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。
    # 定义一个配置类，用于配置 MobileNetV1Model 模型
    class MobileNetV1Config(Config):
    
        # 初始化一个 "mobilenet_v1" 类型的模型
        model_type = "mobilenet_v1"
    
        # 初始化方法，用于设置模型的参数
        def __init__(
            self,
            num_channels=3,  # 输入通道数，默认为 3
            image_size=224,  # 图像大小，默认为 224
            depth_multiplier=1.0,  # 每层通道数的缩放系数，默认为 1.0
            min_depth=8,  # 所有层至少有的通道数，默认为 8
            hidden_act="relu6",  # 隐藏层激活函数，默认为 "relu6"
            tf_padding=True,  # 是否使用 TensorFlow 的填充规则，默认为 True
            classifier_dropout_prob=0.999,  # 附加分类器的 dropout 比率，默认为 0.999
            initializer_range=0.02,  # 用于初始化所有权重矩阵的截断正态分布的标准差，默认为 0.02
            layer_norm_eps=0.001,  # 层归一化层使用的 epsilon，默认为 0.001
            **kwargs,  # 其他关键字参数
        ):
            # 调用父类的初始化方法
            super().__init__(**kwargs)
    
            # 如果深度缩放系数小于等于 0，抛出 ValueError
            if depth_multiplier <= 0:
                raise ValueError("depth_multiplier must be greater than zero.")
    
            # 将参数赋值给模型对象
            self.num_channels = num_channels
            self.image_size = image_size
            self.depth_multiplier = depth_multiplier
            self.min_depth = min_depth
            self.hidden_act = hidden_act
            self.tf_padding = tf_padding
            self.classifier_dropout_prob = classifier_dropout_prob
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
# 定义一个继承自OnnxConfig的MobileNetV1OnnxConfig类
class MobileNetV1OnnxConfig(OnnxConfig):
    
    # 定义torch_onnx_minimum_version属性为版本1.11
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义inputs属性，返回一个有序字典，包含键pixel_values和值{0: "batch"}
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict([("pixel_values", {0: "batch"})])

    # 定义outputs属性，根据任务类型返回不同的字典，
    # 如果任务是图片分类，返回键logits和值{0: "batch"}
    # 如果任务不是图片分类，返回键last_hidden_state和值{0: "batch"}，以及键pooler_output和值{0: "batch"}
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "image-classification":
            return OrderedDict([("logits", {0: "batch"})])
        else:
            return OrderedDict([("last_hidden_state", {0: "batch"}), ("pooler_output", {0: "batch"})])

    # 定义atol_for_validation属性，返回浮点数1e-4
    @property
    def atol_for_validation(self) -> float:
        return 1e-4
```  
```