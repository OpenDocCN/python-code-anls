# `.\models\dinov2\configuration_dinov2.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，声明此文件版权归 HuggingFace Inc. 团队所有
# 根据 Apache License, Version 2.0 许可证使用本文件，除非符合许可证的条件，否则不得使用本文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按"原样"分发本软件
# 本软件不提供任何明示或暗示的保证或条件，包括但不限于特定目的适销性和适用性的隐含保证或条件
# 有关详细信息，请参阅许可证

""" DINOv2 模型配置 """

# 从 collections 模块导入 OrderedDict 类
# 从 typing 模块导入 Mapping 类型
from collections import OrderedDict
from typing import Mapping

# 从 packaging 模块导入 version 函数
from packaging import version

# 从配置文件工具中导入 PretrainedConfig 类
from ...configuration_utils import PretrainedConfig
# 从 ONNX 配置中导入 OnnxConfig 类
from ...onnx import OnnxConfig
# 从工具集中导入日志记录器
from ...utils import logging
# 从 Backbone 工具集中导入 BackboneConfigMixin 类和 get_aligned_output_features_output_indices 函数
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# DINOV2 预训练配置文件的映射字典，指定预训练模型名称和配置文件地址
DINOV2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/dinov2-base": "https://huggingface.co/facebook/dinov2-base/resolve/main/config.json",
}

# Dinov2Config 类，继承自 BackboneConfigMixin 和 PretrainedConfig 类
class Dinov2Config(BackboneConfigMixin, PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`Dinov2Model`] 的配置。根据指定的参数实例化一个 Dinov2 模型，定义模型架构。
    使用默认值实例化配置将产生与 Dinov2 [google/dinov2-base-patch16-224] 架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。有关更多信息，请阅读 [`PretrainedConfig`] 的文档。

    示例：

    ```
    >>> from transformers import Dinov2Config, Dinov2Model

    >>> # 初始化一个 Dinov2 dinov2-base-patch16-224 风格的配置
    >>> configuration = Dinov2Config()

    >>> # 使用 Dinov2 dinov2-base-patch16-224 风格的配置初始化一个模型（带有随机权重）
    >>> model = Dinov2Model(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```
    """

    # 模型类型标识为 "dinov2"
    model_type = "dinov2"

    # 构造函数，定义 Dinov2Config 的各种参数和默认值
    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        mlp_ratio=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        image_size=224,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        layerscale_value=1.0,
        drop_path_rate=0.0,
        use_swiglu_ffn=False,
        out_features=None,
        out_indices=None,
        apply_layernorm=True,
        reshape_hidden_states=True,
        **kwargs,
        ):
            super().__init__(**kwargs)
        # 调用父类的初始化方法，并传入所有关键字参数

        self.hidden_size = hidden_size
        # 设置模型的隐藏层大小

        self.num_hidden_layers = num_hidden_layers
        # 设置模型的隐藏层数量

        self.num_attention_heads = num_attention_heads
        # 设置注意力头的数量

        self.mlp_ratio = mlp_ratio
        # 设置MLP（多层感知机）部分的大小与隐藏层大小之比

        self.hidden_act = hidden_act
        # 设置隐藏层的激活函数类型

        self.hidden_dropout_prob = hidden_dropout_prob
        # 设置隐藏层的dropout概率

        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 设置注意力概率矩阵的dropout概率

        self.initializer_range = initializer_range
        # 设置初始化权重的范围

        self.layer_norm_eps = layer_norm_eps
        # 设置层归一化操作中的epsilon值

        self.image_size = image_size
        # 设置输入图像的尺寸

        self.patch_size = patch_size
        # 设置图像切片的尺寸

        self.num_channels = num_channels
        # 设置输入图像的通道数

        self.qkv_bias = qkv_bias
        # 设置是否使用查询、键、值的偏置项

        self.layerscale_value = layerscale_value
        # 设置层标度值

        self.drop_path_rate = drop_path_rate
        # 设置DropPath操作的比例

        self.use_swiglu_ffn = use_swiglu_ffn
        # 设置是否使用Swish-Gated Gated Linear Unit (SwiGLU)作为FFN层的激活函数

        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, num_hidden_layers + 1)]
        # 创建阶段名称列表，包括“stem”和从“stage1”到“stageN”的阶段

        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )
        # 调用函数获取对齐的输出特征和输出索引，存储在self._out_features和self._out_indices中

        self.apply_layernorm = apply_layernorm
        # 设置是否应用层归一化操作

        self.reshape_hidden_states = reshape_hidden_states
        # 设置是否需要对隐藏状态进行重塑
# 定义一个新的类 Dinov2OnnxConfig，继承自 OnnxConfig 类
class Dinov2OnnxConfig(OnnxConfig):
    # 设置 torch_onnx_minimum_version 属性为版本号 1.11
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义 inputs 属性，返回一个有序字典，描述模型输入的维度顺序和名称
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                # 定义输入的像素值对应的维度顺序和名称
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    # 定义 atol_for_validation 属性，返回一个浮点数，表示验证时的容差
    @property
    def atol_for_validation(self) -> float:
        return 1e-4
```