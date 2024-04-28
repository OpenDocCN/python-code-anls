# `.\models\convnextv2\configuration_convnextv2.py`

```
# 设置文件编码为 UTF-8
# 版权声明以及许可协议信息
# 作者：Meta Platforms, Inc. 和 The HuggingFace Inc. team
# 版权所有
# 根据 Apache 许可证 2.0 版本授权
# 在遵守许可协议的情况下才能使用此文件
# 您可以在以下链接找到许可协议的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则不得使用此文件
# 根据许可协议分发的软件是基于"AS IS"基础进行分发的，没有任何明示或暗示的保证或条件
# 请查看许可协议，了解对模型输出的具体语言，权限和限制
""" ConvNeXTV2 model configuration"""

# 从相关模块中导入所需的库
# 配置模块
# 日志记录模块
# 背景骨干网络配置混合模块
# 获取对齐输出特征输出指数的工具函数
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices

# 获取模块的记录器
logger = logging.get_logger(__name__)

# 预训练配置映射表
CONVNEXTV2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/convnextv2-tiny-1k-224": "https://huggingface.co/facebook/convnextv2-tiny-1k-224/resolve/main/config.json",
}

# ConvNextV2配置类，继承自PretrainedConfig和BackboneConfigMixin
class ConvNextV2Config(BackboneConfigMixin, PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ConvNextV2Model`]. It is used to instantiate an
    ConvNeXTV2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the ConvNeXTV2
    [facebook/convnextv2-tiny-1k-224](https://huggingface.co/facebook/convnextv2-tiny-1k-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        num_channels (`int`, *optional*, defaults to 3):
            输入通道数。
        patch_size (`int`, optional, defaults to 4):
            在patch embedding层中使用的patch大小。
        num_stages (`int`, optional, defaults to 4):
            模型中阶段的数量。
        hidden_sizes (`List[int]`, *optional*, defaults to `[96, 192, 384, 768]`):
            每个阶段的隐藏层大小。
        depths (`List[int]`, *optional*, defaults to `[3, 3, 9, 3]`):
            每个阶段的深度（块的数量）。
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            每个块中的非线性激活函数（函数或字符串）。支持的字符串值有："gelu"、"relu"、"selu"和"gelu_new"。
        initializer_range (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态初始化器的标准偏差。
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            层归一化层使用的epsilon。
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            随机深度的dropout率。
        out_features (`List[str]`, *optional*):
            如果用作主干网络，则输出的特征列表。可以是任意特征（“stem”，“stage1”，“stage2”等等），
            取决于模型有多少阶段。如果未设置并且设置了`out_indices`，则默认为相应的阶段。
            如果未设置并且`out_indices`未设置，则默认为最后一个阶段。必须与`stage_names`属性中的顺序相同。
        out_indices (`List[int]`, *optional*):
            如果用作主干网络，则输出的特征索引列表。可以是任意索引（0、1、2等），取决于模型有多少阶段。
            如果未设置并且设置了`out_features`，则默认为相应的阶段。
            如果未设置并且`out_features`未设置，则默认为最后一个阶段。必须与`stage_names`属性中的顺序相同。

    Example:
    ```python
    >>> from transformers import ConvNeXTV2Config, ConvNextV2Model

    >>> # 根据convnextv2-tiny-1k-224样式的配置初始化ConvNeXTV2Config
    >>> configuration = ConvNeXTV2Config()

    >>> # 根据convnextv2-tiny-1k-224样式的配置初始化具有随机权重的模型
    >>> model = ConvNextV2Model(configuration)

    >>> # 访问模型的配置
    >>> configuration = model.config
    ```"""

    # 模型类型
    model_type = "convnextv2"
    # 初始化函数，用于创建一个新的实例
    def __init__(
        self,
        num_channels=3,                          # 输入图像的通道数，默认为 3
        patch_size=4,                            # 每个 patch 的大小，默认为 4
        num_stages=4,                            # Transformer 模型的阶段数，默认为 4
        hidden_sizes=None,                       # 每个阶段的隐藏层大小，默认为 [96, 192, 384, 768]
        depths=None,                             # 每个阶段的 Transformer 层深度，默认为 [3, 3, 9, 3]
        hidden_act="gelu",                       # 隐藏层激活函数，默认为 GELU
        initializer_range=0.02,                  # 参数初始化范围，默认为 0.02
        layer_norm_eps=1e-12,                    # LayerNorm 层的 epsilon 参数，默认为 1e-12
        drop_path_rate=0.0,                      # DropPath 的概率，默认为 0.0
        image_size=224,                          # 输入图像的大小，默认为 224
        out_features=None,                       # 输出特征的名称，默认为 None
        out_indices=None,                        # 输出特征的索引，默认为 None
        **kwargs,
    ):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 设置实例变量
        self.num_channels = num_channels         # 输入图像的通道数
        self.patch_size = patch_size             # 每个 patch 的大小
        self.num_stages = num_stages             # Transformer 模型的阶段数
        # 每个阶段的隐藏层大小，如果未提供则使用默认值
        self.hidden_sizes = [96, 192, 384, 768] if hidden_sizes is None else hidden_sizes
        # 每个阶段的 Transformer 层深度，如果未提供则使用默认值
        self.depths = [3, 3, 9, 3] if depths is None else depths
        self.hidden_act = hidden_act             # 隐藏层激活函数
        self.initializer_range = initializer_range  # 参数初始化范围
        self.layer_norm_eps = layer_norm_eps     # LayerNorm 层的 epsilon 参数
        self.drop_path_rate = drop_path_rate     # DropPath 的概率
        self.image_size = image_size             # 输入图像的大小
        # 每个阶段的名称列表，包括 "stem" 和 stage1 至 stageN
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.depths) + 1)]
        # 获得对齐的输出特征和输出索引
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )
```