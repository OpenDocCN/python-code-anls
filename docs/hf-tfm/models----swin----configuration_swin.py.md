# `.\transformers\models\swin\configuration_swin.py`

```py
# 设置字符编码为utf-8
# 版权声明
# 根据Apache许可证2.0版（“许可证”）的规定，在使用此文件时必须遵守许可证的规定
# 可以在以下链接获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除了适用法律规定或经书面同意外，根据许可证分发的软件是在“原样”的基础上分发的，没有任何担保或条件，不管是明示还是暗示的
# 请查看许可证以获取有关权限和限制的具体语言
"""Swin Transformer模型配置"""

# 从集合模块中导入有序字典
from collections import OrderedDict
# 导入Mapping类型
from typing import Mapping
# 从packaging模块导入version
from packaging import version
# 从...模块中导入PretrainedConfig
from ...configuration_utils import PretrainedConfig
# 从...onnx模块中导入OnnxConfig
from ...onnx import OnnxConfig
# 从...utils模块导入logging
from ...utils import logging
# 从...utils.backbone_utils模块中导入BackboneConfigMixin, get_aligned_output_features_output_indices函数
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices

# 获取名为__name__的logger
logger = logging.get_logger(__name__)

# Swin预训练配置文件映射
SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/swin-tiny-patch4-window7-224": (
        "https://huggingface.co/microsoft/swin-tiny-patch4-window7-224/resolve/main/config.json"
    ),
    # 在https://huggingface.co/models?filter=swin中查看全部Swin模型
}

# Swin配置类，包含了骨干网络配置混合和预训练配置
class SwinConfig(BackboneConfigMixin, PretrainedConfig):
    r"""
    这是用于存储[`SwinModel`]配置的配置类。它用于根据指定的参数实例化一个Swin模型，定义模型架构。使用默认配置实例化配置类将产生与Swin [microsoft/swin-tiny-patch4-window7-224] (https://huggingface.co/microsoft/swin-tiny-patch4-window7-224) 架构类似的配置。

    配置对象继承自[`PretrainedConfig`]，可用于控制模型输出。阅读来自[`PretrainedConfig`]的文档以获取更多信息。

    示例：

    ```python
    >>> from transformers import SwinConfig, SwinModel

    >>> # 初始化一个Swin microsoft/swin-tiny-patch4-window7-224风格的配置
    >>> configuration = SwinConfig()

    >>> # 从microsoft/swin-tiny-patch4-window7-224风格的配置初始化一个模型（带有随机权重）
    >>> model = SwinModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py"""
    # 模型类型为“swin”
    model_type = "swin"

    # 属性映射
    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }
    # 初始化函数，用于初始化Swin Transformer模型的参数
    def __init__(
        self,
        image_size=224,  # 图像大小，默认为224
        patch_size=4,  # 图像切片大小，默认为4
        num_channels=3,  # 图像通道数，默认为3
        embed_dim=96,  # 嵌入维度，默认为96
        depths=[2, 2, 6, 2],  # 不同阶段的Transformer块数量，默认为[2, 2, 6, 2]
        num_heads=[3, 6, 12, 24],  # 不同阶段的注意力头数，默认为[3, 6, 12, 24]
        window_size=7,  # 窗口大小，默认为7
        mlp_ratio=4.0,  # MLP扩展比率，默认为4.0
        qkv_bias=True,  # 是否使用查询、键、值的偏置，默认为True
        hidden_dropout_prob=0.0,  # 隐藏层dropout概率，默认为0.0
        attention_probs_dropout_prob=0.0,  # 注意力概率dropout概率，默认为0.0
        drop_path_rate=0.1,  # DropPath概率，默认为0.1
        hidden_act="gelu",  # 隐藏层激活函数，默认为gelu
        use_absolute_embeddings=False,  # 是否使用绝对位置编码，默认为False
        initializer_range=0.02,  # 参数初始化范围，默认为0.02
        layer_norm_eps=1e-5,  # LayerNorm层的epsilon值，默认为1e-5
        encoder_stride=32,  # 编码器步幅，默认为32
        out_features=None,  # 输出特征，默认为None
        out_indices=None,  # 输出索引，默认为None
        **kwargs,  # 其他参数
    ):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 设置各种参数
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_layers = len(depths)
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.hidden_act = hidden_act
        self.use_absolute_embeddings = use_absolute_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.encoder_stride = encoder_stride
        # 设置隐藏大小属性，以使Swin与VisionEncoderDecoderModel配合工作
        # 这表示模型最后一个阶段之后的通道维度
        self.hidden_size = int(embed_dim * 2 ** (len(depths) - 1))
        # 设置阶段名称列表
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(depths) + 1)]
        # 获取对齐的输出特征和输出索引
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )
# 定义一个名为SwinOnnxConfig的类，继承自OnnxConfig类
class SwinOnnxConfig(OnnxConfig):
    # 设置torch_onnx_minimum_version属性值为1.11，表示需要最低版本为1.11的torch进行ONNX模型转换
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义inputs属性，返回一个有序字典，包含像素值的输入映射
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    # 定义atol_for_validation属性，返回用于验证的绝对误差值
    @property
    def atol_for_validation(self) -> float:
        return 1e-4
```