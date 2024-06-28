# `.\models\swin\configuration_swin.py`

```py
# coding=utf-8
# 版权 2022 年 HuggingFace Inc. 团队保留所有权利。
#
# 根据 Apache 许可证版本 2.0 许可，您可以不使用此文件，除非遵守许可。
# 您可以在以下地址获取许可的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按"原样"分发，
# 没有任何明示或暗示的保证或条件。
# 请查阅许可证获取更多信息。
""" Swin Transformer 模型配置"""

from collections import OrderedDict  # 导入有序字典类
from typing import Mapping  # 导入映射类型

from packaging import version  # 导入版本控制库

from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...onnx import OnnxConfig  # 导入ONNX配置类
from ...utils import logging  # 导入日志工具
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices  # 导入背骨结构工具函数


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/swin-tiny-patch4-window7-224": (
        "https://huggingface.co/microsoft/swin-tiny-patch4-window7-224/resolve/main/config.json"
    ),
    # 查看所有 Swin 模型，请访问 https://huggingface.co/models?filter=swin
}


class SwinConfig(BackboneConfigMixin, PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`SwinModel`] 的配置。它用于根据指定的参数实例化 Swin 模型，定义模型的架构。
    使用默认值实例化配置将产生类似于 Swin [microsoft/swin-tiny-patch4-window7-224](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224)
    架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。有关更多信息，请阅读 [`PretrainedConfig`] 的文档。

    示例:

    ```
    >>> from transformers import SwinConfig, SwinModel
    >>> # 初始化一个 Swin microsoft/swin-tiny-patch4-window7-224 风格的配置
    >>> configuration = SwinConfig()
    >>> # 从 microsoft/swin-tiny-patch4-window7-224 风格的配置初始化一个模型（带有随机权重）
    >>> model = SwinModel(configuration)
    >>> # 访问模型配置
    >>> configuration = model.config
    ```
    """

    model_type = "swin"

    attribute_map = {
        "num_attention_heads": "num_heads",  # 属性映射，注意力头数映射到 num_heads
        "num_hidden_layers": "num_layers",   # 属性映射，隐藏层数映射到 num_layers
    }
    # 定义一个初始化函数，用于初始化Swin Transformer模型的各种参数
    def __init__(
        self,
        image_size=224,  # 图像尺寸，默认为224
        patch_size=4,  # 每个patch的大小，默认为4
        num_channels=3,  # 输入图像的通道数，默认为3（RGB图像）
        embed_dim=96,  # 嵌入维度，默认为96
        depths=[2, 2, 6, 2],  # 各个阶段的深度列表，默认为[2, 2, 6, 2]
        num_heads=[3, 6, 12, 24],  # 各个阶段的注意力头数列表，默认为[3, 6, 12, 24]
        window_size=7,  # 窗口大小，默认为7
        mlp_ratio=4.0,  # MLP扩展比例，默认为4.0
        qkv_bias=True,  # 是否使用QKV偏置，默认为True
        hidden_dropout_prob=0.0,  # 隐藏层Dropout概率，默认为0.0
        attention_probs_dropout_prob=0.0,  # 注意力概率Dropout概率，默认为0.0
        drop_path_rate=0.1,  # DropPath概率，默认为0.1
        hidden_act="gelu",  # 隐藏层激活函数，默认为GELU
        use_absolute_embeddings=False,  # 是否使用绝对位置嵌入，默认为False
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-5,  # Layer Norm的epsilon值，默认为1e-5
        encoder_stride=32,  # 编码器的步长，默认为32
        out_features=None,  # 输出特征列表，用于对齐特征，默认为None
        out_indices=None,  # 输出索引列表，用于对齐索引，默认为None
        **kwargs,  # 其他关键字参数
    ):
        super().__init__(**kwargs)  # 调用父类的初始化函数
    
        self.image_size = image_size  # 初始化图像尺寸属性
        self.patch_size = patch_size  # 初始化patch大小属性
        self.num_channels = num_channels  # 初始化通道数属性
        self.embed_dim = embed_dim  # 初始化嵌入维度属性
        self.depths = depths  # 初始化深度列表属性
        self.num_layers = len(depths)  # 计算层数并初始化属性
        self.num_heads = num_heads  # 初始化注意力头数列表属性
        self.window_size = window_size  # 初始化窗口大小属性
        self.mlp_ratio = mlp_ratio  # 初始化MLP扩展比例属性
        self.qkv_bias = qkv_bias  # 初始化QKV偏置属性
        self.hidden_dropout_prob = hidden_dropout_prob  # 初始化隐藏层Dropout概率属性
        self.attention_probs_dropout_prob = attention_probs_dropout_prob  # 初始化注意力概率Dropout概率属性
        self.drop_path_rate = drop_path_rate  # 初始化DropPath概率属性
        self.hidden_act = hidden_act  # 初始化隐藏层激活函数属性
        self.use_absolute_embeddings = use_absolute_embeddings  # 初始化使用绝对位置嵌入属性
        self.layer_norm_eps = layer_norm_eps  # 初始化Layer Norm的epsilon值属性
        self.initializer_range = initializer_range  # 初始化初始化范围属性
        self.encoder_stride = encoder_stride  # 初始化编码器步长属性
    
        # 设置隐藏大小属性，以使Swin与VisionEncoderDecoderModel配合工作
        # 这指示模型最后一个阶段之后的通道维度
        self.hidden_size = int(embed_dim * 2 ** (len(depths) - 1))
    
        # 设置阶段名称列表，包括stem和各个阶段（例如stage1、stage2等）
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(depths) + 1)]
    
        # 获取对齐的输出特征和输出索引，用于与给定的阶段名称对齐
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )
# 定义 SwinOnnxConfig 类，继承自 OnnxConfig 类
class SwinOnnxConfig(OnnxConfig):
    # 设定 torch_onnx_minimum_version 属性为版本号 1.11
    torch_onnx_minimum_version = version.parse("1.11")

    # inputs 属性，返回一个有序字典，描述输入数据的结构
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    # atol_for_validation 属性，返回用于验证的绝对容差值
    @property
    def atol_for_validation(self) -> float:
        return 1e-4
```