# `.\models\yolos\configuration_yolos.py`

```
# 设置编码为 UTF-8
# 版权声明 2022 年由 HuggingFace Inc. 团队所有
# 根据 Apache 许可证版本 2.0 许可，除非符合许可证条款，否则不得使用此文件
# 您可以在以下链接处获得许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按“原样”分发软件，不附带任何明示或暗示的保证或条件
# 请参阅许可证以了解详细信息

""" YOLOS 模型配置"""

# 导入必要的库
from collections import OrderedDict  # 导入 OrderedDict 类
from typing import Mapping  # 导入 Mapping 类型提示

from packaging import version  # 导入 version 模块

# 导入配置工具类和 OnnxConfig
from ...configuration_utils import PretrainedConfig  
from ...onnx import OnnxConfig  
from ...utils import logging  # 导入 logging 模块

# 获取 logger 实例
logger = logging.get_logger(__name__)

# YOLOS 预训练模型配置存档映射
YOLOS_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "hustvl/yolos-small": "https://huggingface.co/hustvl/yolos-small/resolve/main/config.json",
    # 可在 https://huggingface.co/models?filter=yolos 查看所有 YOLOS 模型
}

# YolosConfig 类，继承自 PretrainedConfig 类
class YolosConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`YolosModel`] 的配置。它用于根据指定的参数实例化 YOLOS 模型，定义模型架构。使用默认值实例化配置将产生类似 YOLOS [hustvl/yolos-base](https://huggingface.co/hustvl/yolos-base) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。有关更多信息，请阅读 [`PretrainedConfig`] 的文档。

    示例:

    ```python
    >>> from transformers import YolosConfig, YolosModel

    >>> # 初始化一个 YOLOS hustvl/yolos-base 风格的配置
    >>> configuration = YolosConfig()

    >>> # 使用配置初始化一个（具有随机权重）hustvl/yolos-base 风格的模型
    >>> model = YolosModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```
    """

    model_type = "yolos"

    # 初始化函数，定义了 YolosConfig 的各种配置参数
    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=[512, 864],
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        num_detection_tokens=100,
        use_mid_position_embeddings=True,
        auxiliary_loss=False,
        class_cost=1,
        bbox_cost=5,
        giou_cost=2,
        bbox_loss_coefficient=5,
        giou_loss_coefficient=2,
        eos_coefficient=0.1,
        **kwargs,
    ):
        # 调用父类的初始化方法，传递所有关键字参数
        super().__init__(**kwargs)

        # 设置隐藏层大小
        self.hidden_size = hidden_size
        # 设置隐藏层数量
        self.num_hidden_layers = num_hidden_layers
        # 设置注意力头的数量
        self.num_attention_heads = num_attention_heads
        # 设置中间层大小
        self.intermediate_size = intermediate_size
        # 设置隐藏层激活函数
        self.hidden_act = hidden_act
        # 设置隐藏层的丢弃率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 设置注意力概率的丢弃率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 设置初始化范围
        self.initializer_range = initializer_range
        # 设置层归一化的 epsilon 值
        self.layer_norm_eps = layer_norm_eps
        # 设置图像大小
        self.image_size = image_size
        # 设置图像的补丁大小
        self.patch_size = patch_size
        # 设置图像的通道数
        self.num_channels = num_channels
        # 是否使用 QKV 偏置
        self.qkv_bias = qkv_bias
        # 检测令牌的数量
        self.num_detection_tokens = num_detection_tokens
        # 是否使用中间位置嵌入
        self.use_mid_position_embeddings = use_mid_position_embeddings
        # 是否使用辅助损失
        self.auxiliary_loss = auxiliary_loss
        # 设置类别损失的成本
        self.class_cost = class_cost
        # 设置边界框损失的成本
        self.bbox_cost = bbox_cost
        # 设置GIoU损失的成本
        self.giou_cost = giou_cost
        # 设置边界框损失系数
        self.bbox_loss_coefficient = bbox_loss_coefficient
        # 设置GIoU损失系数
        self.giou_loss_coefficient = giou_loss_coefficient
        # 设置EOS损失系数
        self.eos_coefficient = eos_coefficient
# 定义一个自定义的 YolosOnnxConfig 类，继承自 OnnxConfig 类
class YolosOnnxConfig(OnnxConfig):
    # 设置 torch_onnx_minimum_version 属性为版本号 1.11
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义 inputs 属性作为属性方法，返回一个有序字典
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 返回一个有序字典，包含 pixel_values 键和相应的字典值
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    # 定义 atol_for_validation 属性作为属性方法，返回浮点数 1e-4
    @property
    def atol_for_validation(self) -> float:
        # 返回用于验证的绝对误差容限值
        return 1e-4

    # 定义 default_onnx_opset 属性作为属性方法，返回整数 12
    @property
    def default_onnx_opset(self) -> int:
        # 返回默认的 ONNX 运算集版本号
        return 12
```