# `.\transformers\models\vit\configuration_vit.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 版权所有 2021 年 Google AI 和 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，
# 没有任何明示或暗示的担保或条件。
# 请查看许可证以获取有关权限和限制的具体语言。
""" ViT 模型配置"""

# 导入所需的库
from collections import OrderedDict
from typing import Mapping

from packaging import version

# 导入自定义的模块
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练 ViT 模型配置文件映射
VIT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/vit-base-patch16-224": "https://huggingface.co/vit-base-patch16-224/resolve/main/config.json",
    # 查看所有 ViT 模型 https://huggingface.co/models?filter=vit
}

# ViT 模型配置类，继承自 PretrainedConfig
class ViTConfig(PretrainedConfig):
    r"""
    这是用于存储 [`ViTModel`] 配置的配置类。根据指定的参数实例化 ViT 模型，定义模型架构。
    使用默认值实例化配置将产生类似于 ViT [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。
    # 定义一个类，表示ViT模型的配置
    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            编码器层和池化层的维度。
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Transformer编码器中的隐藏层数量。
        num_attention_heads (`int`, *optional*, defaults to 12):
            Transformer编码器中每个注意力层的注意力头数量。
        intermediate_size (`int`, *optional*, defaults to 3072):
            Transformer编码器中“中间”（即前馈）层的维度。
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            编码器和池化器中的非线性激活函数（函数或字符串）。如果是字符串，支持`"gelu"`, `"relu"`, `"selu"` 和 `"gelu_new"`。
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            嵌入层、编码器和池化器中所有全连接层的dropout概率。
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            注意力概率的dropout比率。
        initializer_range (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态初始化器的标准差。
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            层归一化层使用的epsilon。
        image_size (`int`, *optional*, defaults to 224):
            每个图像的大小（分辨率）。
        patch_size (`int`, *optional*, defaults to 16):
            每个补丁的大小（分辨率）。
        num_channels (`int`, *optional*, defaults to 3):
            输入通道的数量。
        qkv_bias (`bool`, *optional*, defaults to `True`):
            是否为查询、键和值添加偏置。
        encoder_stride (`int`, *optional*, defaults to 16):
            用于掩码图像建模中解码器头部中增加空间分辨率的因子。

    Example:

    ```python
    >>> from transformers import ViTConfig, ViTModel

    >>> # 初始化一个ViT vit-base-patch16-224风格的配置
    >>> configuration = ViTConfig()

    >>> # 从vit-base-patch16-224风格的配置初始化一个模型（带有随机权重）
    >>> model = ViTModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py"""

    # 模型类型为"vit"
    model_type = "vit"

    # 初始化方法，设置模型的各种配置参数
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
        image_size=224,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        encoder_stride=16,
        **kwargs,
        ):
        # 调用父类的构造函数，传入关键字参数
        super().__init__(**kwargs)

        # 初始化隐藏层大小
        self.hidden_size = hidden_size
        # 初始化隐藏层数量
        self.num_hidden_layers = num_hidden_layers
        # 初始化注意力头数量
        self.num_attention_heads = num_attention_heads
        # 初始化中间层大小
        self.intermediate_size = intermediate_size
        # 初始化隐藏层激活函数
        self.hidden_act = hidden_act
        # 初始化隐藏层丢弃概率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 初始化注意力概率丢弃概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 初始化初始化器范围
        self.initializer_range = initializer_range
        # 初始化层归一化 epsilon
        self.layer_norm_eps = layer_norm_eps
        # 初始化图像大小
        self.image_size = image_size
        # 初始化块大小
        self.patch_size = patch_size
        # 初始化通道数量
        self.num_channels = num_channels
        # 初始化查询键值偏置
        self.qkv_bias = qkv_bias
        # 初始化编码器步长
        self.encoder_stride = encoder_stride
# 定义一个继承自OnnxConfig的ViTOnnxConfig类
class ViTOnnxConfig(OnnxConfig):
    # 定义torch_onnx_minimum_version属性为1.11版本
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义inputs属性，返回一个有序字典，包含像素值的输入信息
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    # 定义atol_for_validation属性，返回一个用于验证的绝对误差阈值
    @property
    def atol_for_validation(self) -> float:
        return 1e-4
```