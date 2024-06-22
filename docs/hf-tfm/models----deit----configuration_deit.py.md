# `.\models\deit\configuration_deit.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 遵循 Apache License, Version 2.0
# 获取许可证的链接
# 根据适用法律或书面同意的要求，按"原样"分布的软件，没有任何保证或条件，无论是明示的还是暗示的。
# 有特定语言的特定许可证可以验证权限和限制
# DeiT 模型配置
# 导入所需模块
from collections import OrderedDict
from typing import Mapping
from packaging import version

# 导入父类 PretrainedConfig
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# DeiT 预训练配置档案映射表
DEIT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/deit-base-distilled-patch16-224": (
        "https://huggingface.co/facebook/deit-base-patch16-224/resolve/main/config.json"
    ),
    # 查看所有 DeiT 模型的链接
}
# DeiT 的配置类，继承自 PretrainedConfig
class DeiTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DeiTModel`]. It is used to instantiate an DeiT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the DeiT
    [facebook/deit-base-distilled-patch16-224](https://huggingface.co/facebook/deit-base-distilled-patch16-224)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # 类型为DeiT
    model_type = "deit"

    # 初始化函数
    def __init__(
        # 隐藏状态的维度，默认为768
        hidden_size=768,
        # Transformer编码器中隐藏层的数量，默认为12
        num_hidden_layers=12,
        # Transformer编码器中每个注意力层的注意力头数量，默认为12
        num_attention_heads=12,
        # Transformer编码器中“中间”（即前馈）层的维度，默认为3072
        intermediate_size=3072,
        # 编码器和池化层中的非线性激活函数（函数或字符串），默认为"gelu"，支持"gelu"、"relu"、"selu"和"gelu_new"
        hidden_act="gelu",
        # 嵌入层、编码器和池化层中所有全连接层的dropout概率，默认为0.0
        hidden_dropout_prob=0.0,
        # 注意力概率的dropout比例，默认为0.0
        attention_probs_dropout_prob=0.0,
        # 用于初始化所有权重矩阵的截断正态初始化器的标准差，默认为0.02
        initializer_range=0.02,
        # 层归一化层使用的epsilon，默认为1e-12
        layer_norm_eps=1e-12,
        # 每个图像的大小（分辨率），默认为224
        image_size=224,
        # 每个补丁的大小（分辨率），默认为16
        patch_size=16,
        # 输入通道的数量，默认为3
        num_channels=3,
        # 是否向查询、键和值添加偏置，默认为True
        qkv_bias=True,
        # 用于掩码图像建模中解码器头部中增加空间分辨率的因子，默认为16
        encoder_stride=16,
        # 其他参数
        **kwargs,
        # 调用父类的构造函数
        super().__init__(**kwargs)

        # 设置隐藏层的大小
        self.hidden_size = hidden_size
        # 设置隐藏层的数量
        self.num_hidden_layers = num_hidden_layers
        # 设置注意力头的数量
        self.num_attention_heads = num_attention_heads
        # 设置中间层的大小
        self.intermediate_size = intermediate_size
        # 设置隐藏层的激活函数
        self.hidden_act = hidden_act
        # 设置隐藏层的丢弃概率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 设置注意力矩阵的丢弃概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 设置初始化范围
        self.initializer_range = initializer_range
        # 设置层归一化的小值偏移量
        self.layer_norm_eps = layer_norm_eps
        # 设置图像的大小
        self.image_size = image_size
        # 设置图像的patch大小
        self.patch_size = patch_size
        # 设置通道的数量
        self.num_channels = num_channels
        # 设置Query、Key和Value的偏置
        self.qkv_bias = qkv_bias
        # 设置编码器的步长
        self.encoder_stride = encoder_stride
# 创建一个名为DeiTOnnxConfig的类，并继承自OnnxConfig类
class DeiTOnnxConfig(OnnxConfig):
    # 设置torch_onnx_minimum_version属性为1.11
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义一个名为inputs的属性，并返回一个有序字典
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                # 键为"pixel_values"，值为一个字典，表示输入的各维度的名称
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    # 定义一个名为atol_for_validation的属性，并返回一个浮点数
    @property
    def atol_for_validation(self) -> float:
        return 1e-4
```