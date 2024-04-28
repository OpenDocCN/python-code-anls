# `.\models\convnext\configuration_convnext.py`

```py
# 设置编码格式为UTF-8
# 版权声明，版权归Meta Platforms，Inc.和The HuggingFace Inc. team所有
# 根据Apache License, Version 2.0许可。非许可下禁止使用此文件
# 可以在以下链接获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律有要求或书面同意，否则在许可下分发的软件是基于"AS IS"的基础上分发的，没有任何明示或暗示的担保或条件
# 请查看特定语言控制权限和许可限制
""" ConvNeXT model configuration"""

# 导入所需模块和类
from collections import OrderedDict
from typing import Mapping
# 导入版本包
from packaging import version
# 从相关模块中导入所需的类和函数
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices
# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 预先训练的ConvNext模型配置文件map
CONVNEXT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/convnext-tiny-224": "https://huggingface.co/facebook/convnext-tiny-224/resolve/main/config.json",
    # 在https://huggingface.co/models?filter=convnext中查看所有ConvNeXT模型
}

# ConvNext模型配置类，继承自BackboneConfigMixin和PretrainedConfig
class ConvNextConfig(BackboneConfigMixin, PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ConvNextModel`]. It is used to instantiate an
    ConvNeXT model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the ConvNeXT
    [facebook/convnext-tiny-224](https://huggingface.co/facebook/convnext-tiny-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        num_channels (`int`, *optional*, defaults to 3):
            输入通道的数量，默认为 3。
        patch_size (`int`, optional, defaults to 4):
            在补丁嵌入层中使用的补丁大小。
        num_stages (`int`, optional, defaults to 4):
            模型中的阶段数量。
        hidden_sizes (`List[int]`, *optional*, defaults to [96, 192, 384, 768]):
            每个阶段的隐藏维度（隐藏大小）。
        depths (`List[int]`, *optional*, defaults to [3, 3, 9, 3]):
            每个阶段的深度（块的数量）。
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            每个块中的非线性激活函数（函数或字符串）。如果是字符串，支持 `"gelu"`、`"relu"`、`"selu"` 和 `"gelu_new"`。
        initializer_range (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态初始化器的标准差。
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            层归一化层使用的 epsilon 值。
        layer_scale_init_value (`float`, *optional*, defaults to 1e-6):
            层尺度的初始值。
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            随机深度的丢弃率。
        out_features (`List[str]`, *optional*):
            如果作为主干结构使用，要输出的特征列表。可以是 `"stem"`、`"stage1"`、`"stage2"` 等任何一个（取决于模型有多少阶段）。
            如果未设置且设置了 `out_indices`，将默认为对应的阶段。如果未设置且未设置 `out_indices`，将默认为最后一个阶段。
            必须按照 `stage_names` 属性中定义的顺序排列。
        out_indices (`List[int]`, *optional*):
            如果作为主干结构使用，要输出的功能索引列表。可以是 0、1、2 等任何一个（取决于模型有多少阶段）。
            如果未设置且设置了 `out_features`，将默认为对应的阶段。如果未设置且未设置 `out_features`，将默认为最后一个阶段。
            必须按照 `stage_names` 属性中定义的顺序排列。

    Example:
    ```python
    >>> from transformers import ConvNextConfig, ConvNextModel

    >>> # Initializing a ConvNext convnext-tiny-224 style configuration
    >>> configuration = ConvNextConfig()

    >>> # Initializing a model (with random weights) from the convnext-tiny-224 style configuration
    >>> model = ConvNextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```py"""

    # 定义模型类型为 "convnext"
    model_type = "convnext"
    # 初始化方法，设置各种默认参数
    def __init__(
        self,
        num_channels=3, # 设置输入通道数，默认为3
        patch_size=4, # 设置图像分块大小，默认为4
        num_stages=4, # 设置模型阶段数量，默认为4
        hidden_sizes=None, # 设置隐藏层大小，默认为[96, 192, 384, 768]
        depths=None, # 设置深度，默认为[3, 3, 9, 3]
        hidden_act="gelu", # 设置隐藏层激活函数，默认为"glove"
        initializer_range=0.02, # 设置初始化范围，默认为0.02
        layer_norm_eps=1e-12, # 设置层归一化的小数值，默认为1e-12
        layer_scale_init_value=1e-6, # 设置层缩放的初始化值，默认为1e-6
        drop_path_rate=0.0, # 设置丢弃路径的比例，默认为0.0
        image_size=224, # 设置图像尺寸，默认为224
        out_features=None, # 设置输出特征，默认为None
        out_indices=None, # 设置输出索引，默认为None
        **kwargs, # 设置其它关键字参数
    ):
        super().__init__(**kwargs) # 调用父类初始化方法

        self.num_channels = num_channels # 设置实例的输入通道数
        self.patch_size = patch_size # 设置实例的图像分块大小
        self.num_stages = num_stages # 设置实例的模型阶段数量
        self.hidden_sizes = [96, 192, 384, 768] if hidden_sizes is None else hidden_sizes # 设置实例的隐藏层大小
        self.depths = [3, 3, 9, 3] if depths is None else depths # 设置实例的深度
        self.hidden_act = hidden_act # 设置实例的隐藏层激活函数
        self.initializer_range = initializer_range # 设置实例的初始化范围
        self.layer_norm_eps = layer_norm_eps # 设置实例的层归一化的小数值
        self.layer_scale_init_value = layer_scale_init_value # 设置实例的层缩放的初始化值
        self.drop_path_rate = drop_path_rate # 设置实例的丢弃路径的比例
        self.image_size = image_size # 设置实例的图像尺寸
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.depths) + 1)] # 设置实例的阶段名称
        self._out_features, self._out_indices = get_aligned_output_features_output_indices( # 获取对齐的输出特征和输出索引
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )
# 定义一个名为ConvNextOnnxConfig的类，继承自OnnxConfig
class ConvNextOnnxConfig(OnnxConfig):
    # 定义一个名为torch_onnx_minimum_version的类属性，并赋值为version.parse("1.11")
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义一个名为inputs的属性，返回一个有序字典
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                # 定义键为"pixel_values"，值为一个字典，键为int类型，值为str类型
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    # 定义一个名为atol_for_validation的属性，返回一个float类型的值
    @property
    def atol_for_validation(self) -> float:
        return 1e-5
```