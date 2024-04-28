# `.\transformers\models\mobilevit\configuration_mobilevit.py`

```
# 导入必要的库和模块
import collections
from typing import Mapping
from packaging import version
from transformers.configuration_utils import PretrainedConfig
from transformers.onnx import OnnxConfig
from transformers.utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 MobileViT 预训练模型的配置档案映射
MOBILEVIT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    # 定义多个预训练模型的配置文件地址
    "apple/mobilevit-small": "https://huggingface.co/apple/mobilevit-small/resolve/main/config.json",
    "apple/mobilevit-x-small": "https://huggingface.co/apple/mobilevit-x-small/resolve/main/config.json",
    "apple/mobilevit-xx-small": "https://huggingface.co/apple/mobilevit-xx-small/resolve/main/config.json",
    "apple/deeplabv3-mobilevit-small": "https://huggingface.co/apple/deeplabv3-mobilevit-small/resolve/main/config.json",
    "apple/deeplabv3-mobilevit-x-small": "https://huggingface.co/apple/deeplabv3-mobilevit-x-small/resolve/main/config.json",
    "apple/deeplabv3-mobilevit-xx-small": "https://huggingface.co/apple/deeplabv3-mobilevit-xx-small/resolve/main/config.json",
    # 查看更多 MobileViT 模型可访问的网址
}

# 定义 MobileViTConfig 类，继承自 PretrainedConfig
class MobileViTConfig(PretrainedConfig):
    """
    这是存储 MobileViTModel 配置的类。它用于根据指定的参数实例化一个 MobileViT 模型,定义模型架构。
    使用默认值实例化配置会产生一个类似于 apple/mobilevit-small 架构的配置。

    配置对象继承自 PretrainedConfig,可用于控制模型输出。
    更多信息请参考 PretrainedConfig 的文档。

    示例:
    ```python
    from transformers import MobileViTConfig, MobileViTModel
    
    # 初始化一个 mobilevit-small 风格的配置
    configuration = MobileViTConfig()

    # 根据 mobilevit-small 风格的配置初始化一个模型
    model = MobileViTModel(configuration)

    # 访问模型配置
    configuration = model.config
    ```
    """

    model_type = "mobilevit"
    # 初始化函数，设置各种参数的默认值
    def __init__(
        self,
        num_channels=3,  # 输入图像的通道数，默认为3
        image_size=256,  # 输入图像的大小，默认为256
        patch_size=2,  # 每个图像补丁的大小，默认为2
        hidden_sizes=[144, 192, 240],  # 隐藏层的神经元数量，默认值为[144, 192, 240]
        neck_hidden_sizes=[16, 32, 64, 96, 128, 160, 640],  # 颈部隐藏层的神经元数量，默认值为[16, 32, 64, 96, 128, 160, 640]
        num_attention_heads=4,  # 注意力头的数量，默认为4
        mlp_ratio=2.0,  # MLP层宽度相对于嵌入维度的倍数，默认为2.0
        expand_ratio=4.0,  # 展开比率，默认为4.0
        hidden_act="silu",  # 隐藏层的激活函数，默认为'silu'
        conv_kernel_size=3,  # 卷积层核大小，默认为3
        output_stride=32,  # 输出步幅，默认为32
        hidden_dropout_prob=0.1,  # 隐藏层的Dropout概率，默认为0.1
        attention_probs_dropout_prob=0.0,  # 注意力概率的Dropout概率，默认为0.0
        classifier_dropout_prob=0.1,  # 分类器的Dropout概率，默认为0.1
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-5,  # 层归一化的常数值，默认为1e-5
        qkv_bias=True,  # Query、Key和Value的偏置，默认为True

        # 用于语义分割的解码头属性
        aspp_out_channels=256,  # ASPP输出通道数，默认为256
        atrous_rates=[6, 12, 18],  # 穿透率列表，默认为[6, 12, 18]
        aspp_dropout_prob=0.1,  # ASPP的Dropout概率，默认为0.1
        semantic_loss_ignore_index=255,  # 语义损失的忽略索引，默认为255
        **kwargs,  # 其他关键字参数
    ):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 初始化各个参数
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_sizes = hidden_sizes
        self.neck_hidden_sizes = neck_hidden_sizes
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.expand_ratio = expand_ratio
        self.hidden_act = hidden_act
        self.conv_kernel_size = conv_kernel_size
        self.output_stride = output_stride
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.classifier_dropout_prob = classifier_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias

        # 用于语义分割的解码头属性赋值
        self.aspp_out_channels = aspp_out_channels
        self.atrous_rates = atrous_rates
        self.aspp_dropout_prob = aspp_dropout_prob
        self.semantic_loss_ignore_index = semantic_loss_ignore_index
# 定义 MobileViTOnnxConfig 类，继承自 OnnxConfig 类
class MobileViTOnnxConfig(OnnxConfig):
    # 定义 torch_onnx_minimum_version 属性，设置其值为版本解析为 1.11 的对象
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义 inputs 属性，返回一个有序字典，表示输入节点的名称和维度信息
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict([("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"})])

    # 定义 outputs 属性，根据任务类型返回输出节点的名称和维度信息
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "image-classification":
            return OrderedDict([("logits", {0: "batch"})])
        else:
            return OrderedDict([("last_hidden_state", {0: "batch"}), ("pooler_output", {0: "batch"})])

    # 定义 atol_for_validation 属性，返回用于验证的公差值
    @property
    def atol_for_validation(self) -> float:
        return 1e-4
```