# `.\transformers\models\yolos\configuration_yolos.py`

```py
# 导入所需的库和模块
from collections import OrderedDict
from typing import Mapping

from packaging import version

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging

# 获取日志器
logger = logging.get_logger(__name__)

# 定义预训练模型配置文件的映射
YOLOS_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "hustvl/yolos-small": "https://huggingface.co/hustvl/yolos-small/resolve/main/config.json",
    # See all YOLOS models at https://huggingface.co/models?filter=yolos
}

# 定义YOLOS模型的配置类
class YolosConfig(PretrainedConfig):
    """
    这是用于存储 YolosModel 配置的配置类。它用于根据指定的参数实例化一个 YOLOS 模型,定义模型架构。
    使用默认值实例化配置将产生与 YOLOS [hustvl/yolos-base] 架构类似的配置。

    配置对象继承自 PretrainedConfig,可用于控制模型输出。请阅读 PretrainedConfig 的文档以了解更多信息。

    示例:
    ```python
    from transformers import YolosConfig, YolosModel

    # 初始化一个 YOLOS hustvl/yolos-base 样式的配置
    configuration = YolosConfig()

    # 从 hustvl/yolos-base 样式的配置初始化一个模型(使用随机权重)
    model = YolosModel(configuration)

    # 访问模型配置
    configuration = model.config
    ```py
    """

    model_type = "yolos"

    def __init__(
        self,
        hidden_size=768,  # 隐藏层大小
        num_hidden_layers=12,  # 隐藏层数量
        num_attention_heads=12,  # 注意力头的数量
        intermediate_size=3072,  # 前馈网络中间层大小
        hidden_act="gelu",  # 隐藏层激活函数
        hidden_dropout_prob=0.0,  # 隐藏层dropout概率
        attention_probs_dropout_prob=0.0,  # 注意力头dropout概率
        initializer_range=0.02,  # 权重初始化范围
        layer_norm_eps=1e-12,  # LayerNorm的epsilon值
        image_size=[512, 864],  # 输入图像尺寸
        patch_size=16,  # Patch大小
        num_channels=3,  # 输入图像通道数
        qkv_bias=True,  # Q、K、V是否有偏置
        num_detection_tokens=100,  # 检测tokens数量
        use_mid_position_embeddings=True,  # 是否使用中间位置编码
        auxiliary_loss=False,  # 是否使用辅助损失
        class_cost=1,  # 类别损失权重
        bbox_cost=5,  # 边界框损失权重
        giou_cost=2,  # GIoU损失权重
        bbox_loss_coefficient=5,  # 边界框损失系数
        giou_loss_coefficient=2,  # GIoU损失系数
        eos_coefficient=0.1,  # EOS损失系数
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.num_detection_tokens = num_detection_tokens
        self.use_mid_position_embeddings = use_mid_position_embeddings
        self.auxiliary_loss = auxiliary_loss
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        self.bbox_loss_coefficient = bbox_loss_coefficient
        self.giou_loss_coefficient = giou_loss_coefficient
        self.eos_coefficient = eos_coefficient
        # 调用父类的构造函数，传入关键字参数
        super().__init__(**kwargs)

        # 隐藏层的尺寸
        self.hidden_size = hidden_size
        # 隐藏层的数量
        self.num_hidden_layers = num_hidden_layers
        # 注意力头的数量
        self.num_attention_heads = num_attention_heads
        # 中间层的尺寸
        self.intermediate_size = intermediate_size
        # 隐藏层的激活函数
        self.hidden_act = hidden_act
        # 隐藏层的丢弃概率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 注意力层的丢弃概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 初始化范围
        self.initializer_range = initializer_range
        # 层标准化的 epsilon
        self.layer_norm_eps = layer_norm_eps
        # 图像尺寸
        self.image_size = image_size
        # 补丁尺寸
        self.patch_size = patch_size
        # 通道数量
        self.num_channels = num_channels
        # Query、Key、Value 是否使用偏置
        self.qkv_bias = qkv_bias
        # 检测令牌的数量
        self.num_detection_tokens = num_detection_tokens
        # 是否使用中间位置嵌入
        self.use_mid_position_embeddings = use_mid_position_embeddings
        # 辅助损失
        self.auxiliary_loss = auxiliary_loss
        # 匈牙利匹配器相关参数
        # 类别损失
        self.class_cost = class_cost
        # 边界框损失
        self.bbox_cost = bbox_cost
        # Giou 损失
        self.giou_cost = giou_cost
        # 损失系数
        # 边界框损失系数
        self.bbox_loss_coefficient = bbox_loss_coefficient
        # Giou 损失系数
        self.giou_loss_coefficient = giou_loss_coefficient
        # EOS（End Of Sequence）损失系数
        self.eos_coefficient = eos_coefficient
# 定义 YolosOnnxConfig 类，继承自 OnnxConfig 类
class YolosOnnxConfig(OnnxConfig):
    # 设定 torch_onnx_minimum_version 属性值为版本号 1.11
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义 inputs 属性，返回一个有序字典
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 返回包含像素值的输入信息，以字典的形式返回
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    # 定义 atol_for_validation 属性，返回浮点数值 1e-4
    @property
    def atol_for_validation(self) -> float:
        # 返回用于验证的绝对误差容忍值
        return 1e-4

    # 定义 default_onnx_opset 属性，返回整数值 12
    @property
    def default_onnx_opset(self) -> int:
        # 返回默认的 ONNX 操作集版本号
        return 12
```