# `.\transformers\models\poolformer\configuration_poolformer.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，禁止未经许可使用此文件
# 可以在遵守许可证的情况下使用此文件
# 可以在以下链接获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础，没有任何形式的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
""" PoolFormer 模型配置"""
# 导入所需的库
from collections import OrderedDict
from typing import Mapping

from packaging import version

# 导入必要的类和函数
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练 PoolFormer 模型配置文件映射
POOLFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "sail/poolformer_s12": "https://huggingface.co/sail/poolformer_s12/resolve/main/config.json",
    # 查看所有 PoolFormer 模型：https://huggingface.co/models?filter=poolformer
}

# PoolFormer 配置类，用于存储 PoolFormerModel 的配置
class PoolFormerConfig(PretrainedConfig):
    r"""
    这是用于存储 [`PoolFormerModel`] 配置的配置类。根据指定的参数实例化 PoolFormer 模型，定义模型架构。
    使用默认值实例化配置将产生类似于 PoolFormer [sail/poolformer_s12](https://huggingface.co/sail/poolformer_s12) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。
    # 定义 PoolFormerConfig 类，用于配置 PoolFormerModel 模型
    Args:
        num_channels (`int`, *optional*, defaults to 3):
            输入图像的通道数。
        patch_size (`int`, *optional*, defaults to 16):
            输入补丁的大小。
        stride (`int`, *optional*, defaults to 16):
            输入补丁的步幅。
        pool_size (`int`, *optional*, defaults to 3):
            池化窗口的大小。
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            MLP 输出通道数与输入通道数的比率。
        depths (`list`, *optional*, defaults to `[2, 2, 6, 2]`):
            每个编码器块的深度。
        hidden_sizes (`list`, *optional*, defaults to `[64, 128, 320, 512]`):
            每个编码器块的隐藏层大小。
        patch_sizes (`list`, *optional*, defaults to `[7, 3, 3, 3]`):
            每个编码器块的输入补丁大小。
        strides (`list`, *optional*, defaults to `[4, 2, 2, 2]`):
            每个编码器块的输入补丁步幅。
        padding (`list`, *optional*, defaults to `[2, 1, 1, 1]`):
            每个编码器块的输入补丁填充。
        num_encoder_blocks (`int`, *optional*, defaults to 4):
            编码器块的数量。
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            丢弃层的丢弃率。
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            隐藏层的激活函数。
        use_layer_scale (`bool`, *optional*, defaults to `True`):
            是否使用层缩放。
        layer_scale_init_value (`float`, *optional*, defaults to 1e-05):
            层缩放的初始值。
        initializer_range (`float`, *optional*, defaults to 0.02):
            权重的初始化范围。

    Example:

    ```python
    >>> from transformers import PoolFormerConfig, PoolFormerModel

    >>> # 初始化一个 PoolFormer sail/poolformer_s12 风格的配置
    >>> configuration = PoolFormerConfig()

    >>> # 从 sail/poolformer_s12 风格的配置初始化一个模型（带有随机权重）
    >>> model = PoolFormerModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py
    """

    # 模型类型为 "poolformer"
    model_type = "poolformer"

    # 初始化 PoolFormerModel 类
    def __init__(
        self,
        num_channels=3,
        patch_size=16,
        stride=16,
        pool_size=3,
        mlp_ratio=4.0,
        depths=[2, 2, 6, 2],
        hidden_sizes=[64, 128, 320, 512],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        padding=[2, 1, 1, 1],
        num_encoder_blocks=4,
        drop_path_rate=0.0,
        hidden_act="gelu",
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        initializer_range=0.02,
        **kwargs,
        # 初始化卷积神经网络的参数
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.pool_size = pool_size
        self.hidden_sizes = hidden_sizes
        self.mlp_ratio = mlp_ratio
        self.depths = depths
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.num_encoder_blocks = num_encoder_blocks
        self.drop_path_rate = drop_path_rate
        self.hidden_act = hidden_act
        self.use_layer_scale = use_layer_scale
        self.layer_scale_init_value = layer_scale_init_value
        self.initializer_range = initializer_range
        # 调用父类的初始化方法
        super().__init__(**kwargs)
# 定义 PoolFormerOnnxConfig 类，继承自 OnnxConfig 类
class PoolFormerOnnxConfig(OnnxConfig):
    # 定义 torch_onnx_minimum_version 属性，值为版本号 1.11
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义 inputs 属性，返回一个有序字典，包含像素值的输入信息
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    # 定义 atol_for_validation 属性，返回一个浮点数，表示验证时的绝对误差容限
    @property
    def atol_for_validation(self) -> float:
        return 2e-3
```