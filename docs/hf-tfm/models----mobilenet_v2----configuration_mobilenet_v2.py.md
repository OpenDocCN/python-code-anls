# `.\models\mobilenet_v2\configuration_mobilenet_v2.py`

```py
# 导入必要的模块和类
from collections import OrderedDict
from typing import Mapping

# 导入版本管理模块
from packaging import version

# 导入配置基类
from ...configuration_utils import PretrainedConfig

# 导入ONNX配置模块
from ...onnx import OnnxConfig

# 导入日志记录工具
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 预训练模型配置文件映射表，映射了不同预训练模型的名称和对应的配置文件链接
MOBILENET_V2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/mobilenet_v2_1.4_224": "https://huggingface.co/google/mobilenet_v2_1.4_224/resolve/main/config.json",
    "google/mobilenet_v2_1.0_224": "https://huggingface.co/google/mobilenet_v2_1.0_224/resolve/main/config.json",
    "google/mobilenet_v2_0.75_160": "https://huggingface.co/google/mobilenet_v2_0.75_160/resolve/main/config.json",
    "google/mobilenet_v2_0.35_96": "https://huggingface.co/google/mobilenet_v2_0.35_96/resolve/main/config.json",
    # 查看所有MobileNetV2模型：https://huggingface.co/models?filter=mobilenet_v2
}


class MobileNetV2Config(PretrainedConfig):
    r"""
    这是一个配置类，用于存储[`MobileNetV2Model`]的配置。根据指定的参数实例化MobileNetV2模型，定义模型架构。
    使用默认参数实例化配置将产生与MobileNetV2 [google/mobilenet_v2_1.0_224]架构相似的配置。

    配置对象继承自[`PretrainedConfig`]，可用于控制模型输出。阅读[`PretrainedConfig`]的文档获取更多信息。
    """
    
    def __init__(self, **kwargs):
        # 调用父类的初始化方法，传递参数给父类构造函数
        super().__init__(**kwargs)

# 不完整的代码截断，无法提供完整的类定义
    Args:
        num_channels (`int`, *optional*, defaults to 3):
            输入图像的通道数，默认为3。
        image_size (`int`, *optional*, defaults to 224):
            每张图像的分辨率大小，默认为224。
        depth_multiplier (`float`, *optional*, defaults to 1.0):
            每层中通道数的缩放倍数。默认为1.0，表示网络从32个通道开始。有时也称为“alpha”或“宽度倍增器”。
        depth_divisible_by (`int`, *optional*, defaults to 8):
            每层的通道数始终是此数的倍数，默认为8。
        min_depth (`int`, *optional*, defaults to 8):
            所有层至少具有的通道数，默认为8。
        expand_ratio (`float`, *optional*, defaults to 6.0):
            每个块中第一层的输出通道数是输入通道数乘以扩展比例。
        output_stride (`int`, *optional*, defaults to 32):
            输入和输出特征图之间的空间分辨率比例。默认情况下，模型将输入尺寸减少32倍。
            如果 `output_stride` 是8或16，模型会在深度wise层上使用扩张卷积，以确保特征图不会比输入图像小超过8倍或16倍。
        first_layer_is_expansion (`bool`, *optional*, defaults to `True`):
            如果第一个卷积层也是第一个扩展块的扩展层，则为True。
        finegrained_output (`bool`, *optional*, defaults to `True`):
            如果为True，则最终卷积层中的输出通道数将保持较大值（1280），即使 `depth_multiplier` 小于1。
        hidden_act (`str` or `function`, *optional*, defaults to `"relu6"`):
            在Transformer编码器和卷积层中使用的非线性激活函数（函数或字符串）。
        tf_padding (`bool`, *optional*, defaults to `True`):
            是否在卷积层中使用TensorFlow的填充规则。
        classifier_dropout_prob (`float`, *optional*, defaults to 0.8):
            附加分类器的dropout比率。
        initializer_range (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态初始化器的标准差。
        layer_norm_eps (`float`, *optional*, defaults to 0.001):
            层归一化层使用的epsilon值。
        semantic_loss_ignore_index (`int`, *optional*, defaults to 255):
            语义分割模型损失函数中忽略的索引。
    Example:

    ```
    >>> from transformers import MobileNetV2Config, MobileNetV2Model

    >>> # Initializing a "mobilenet_v2_1.0_224" style configuration
    >>> configuration = MobileNetV2Config()
    # 定义一个字符串变量，表示模型类型为 MobileNetV2
    model_type = "mobilenet_v2"
    
    # 定义 MobileNetV2Model 类，继承自某个父类（未显示出来）
    class MobileNetV2Model:
    
        # 初始化方法，设置模型的各项参数和超参数
        def __init__(
            self,
            num_channels=3,  # 输入图像的通道数，默认为3（RGB图像）
            image_size=224,  # 输入图像的尺寸，默认为224x224像素
            depth_multiplier=1.0,  # 深度乘数，控制模型的宽度，默认为1.0
            depth_divisible_by=8,  # 深度可被这个数整除，默认为8
            min_depth=8,  # 最小深度，默认为8
            expand_ratio=6.0,  # 扩展比率，默认为6.0
            output_stride=32,  # 输出步长，默认为32
            first_layer_is_expansion=True,  # 第一层是否是扩展层，默认为True
            finegrained_output=True,  # 是否输出细粒度特征，默认为True
            hidden_act="relu6",  # 隐藏层激活函数，默认为 relu6
            tf_padding=True,  # 是否使用 TensorFlow 的填充方式，默认为True
            classifier_dropout_prob=0.8,  # 分类器的 dropout 概率，默认为0.8
            initializer_range=0.02,  # 初始化范围，默认为0.02
            layer_norm_eps=0.001,  # Layer Normalization 的 epsilon 参数，默认为0.001
            semantic_loss_ignore_index=255,  # 语义损失函数中的忽略索引，默认为255
            **kwargs,  # 其他参数
        ):
            # 调用父类的初始化方法，传入其他关键字参数
            super().__init__(**kwargs)
    
            # 如果 depth_multiplier 小于等于0，抛出数值错误异常
            if depth_multiplier <= 0:
                raise ValueError("depth_multiplier must be greater than zero.")
    
            # 设置模型对象的各项属性
            self.num_channels = num_channels
            self.image_size = image_size
            self.depth_multiplier = depth_multiplier
            self.depth_divisible_by = depth_divisible_by
            self.min_depth = min_depth
            self.expand_ratio = expand_ratio
            self.output_stride = output_stride
            self.first_layer_is_expansion = first_layer_is_expansion
            self.finegrained_output = finegrained_output
            self.hidden_act = hidden_act
            self.tf_padding = tf_padding
            self.classifier_dropout_prob = classifier_dropout_prob
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
            self.semantic_loss_ignore_index = semantic_loss_ignore_index
# 定义一个 MobileNetV2OnnxConfig 类，继承自 OnnxConfig 类
class MobileNetV2OnnxConfig(OnnxConfig):
    # 设置 torch 转换为 ONNX 的最低版本要求为 1.11
    torch_onnx_minimum_version = version.parse("1.11")

    # 返回模型输入的描述信息，使用有序字典来指定每个输入的名称及其维度信息
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict([("pixel_values", {0: "batch"})])

    # 根据任务类型返回模型输出的描述信息，有条件地返回分类器的逻辑输出或者特征提取器的输出
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "image-classification":
            return OrderedDict([("logits", {0: "batch"})])
        else:
            return OrderedDict([("last_hidden_state", {0: "batch"}), ("pooler_output", {0: "batch"})])

    # 返回用于验证时的绝对误差容限
    @property
    def atol_for_validation(self) -> float:
        return 1e-4
```