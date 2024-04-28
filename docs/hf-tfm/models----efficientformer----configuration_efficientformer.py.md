# `.\models\efficientformer\configuration_efficientformer.py`

```py
# 设置编码格式为 utf-8
# 版权声明
# 在 Apache 许可证 2.0 下，根据许可证规定，除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证副本
# 在适用法律或书面同意的情况下，根据许可证分发的软件是基于“按原样”分发的，不附带任何明示或暗示的担保或条件
# 有关特定语言的特定语言的许可证的详细信息，请参见许可证
# 限制许可证下的特定语言的特定语言的权限与限制
""" EfficientFormer model configuration"""

from typing import List

# 导入预训练配置
from ...configuration_utils import PretrainedConfig
# 导入日志工具
from ...utils import logging

# 获取日志器
logger = logging.get_logger(__name__)

# EfficientFormer 预训练模型配置存档映射
EFFICIENTFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "snap-research/efficientformer-l1-300": (
        "https://huggingface.co/snap-research/efficientformer-l1-300/resolve/main/config.json"
    ),
}

# EfficientFormer 配置类
class EfficientFormerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`EfficientFormerModel`]. It is used to
    instantiate an EfficientFormer model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the EfficientFormer
    [snap-research/efficientformer-l1](https://huggingface.co/snap-research/efficientformer-l1) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```python
    >>> from transformers import EfficientFormerConfig, EfficientFormerModel

    >>> # Initializing a EfficientFormer efficientformer-l1 style configuration
    >>> configuration = EfficientFormerConfig()

    >>> # Initializing a EfficientFormerModel (with random weights) from the efficientformer-l3 style configuration
    >>> model = EfficientFormerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```py"""
    
    # 模型类型为 efficientformer
    model_type = "efficientformer"
    # 初始化函数，用于初始化模型参数
    def __init__(
        self,
        # 设置深度列表，默认为 [3, 2, 6, 4]
        depths: List[int] = [3, 2, 6, 4],
        # 设置隐藏层大小列表，默认为 [48, 96, 224, 448]
        hidden_sizes: List[int] = [48, 96, 224, 448],
        # 设置下采样标志列表，默认为 [True, True, True, True]
        downsamples: List[bool] = [True, True, True, True],
        # 设置维度，默认为 448
        dim: int = 448,
        # 设置关键维度，默认为 32
        key_dim: int = 32,
        # 设置注意力比率，默认为 4
        attention_ratio: int = 4,
        # 设置分辨率，默认为 7
        resolution: int = 7,
        # 设置隐藏层数量，默认为 5
        num_hidden_layers: int = 5,
        # 设置注意力头数量，默认为 8
        num_attention_heads: int = 8,
        # 设置 MLP 扩展比率，默认为 4
        mlp_expansion_ratio: int = 4,
        # 设置隐藏层 dropout 概率，默认为 0.0
        hidden_dropout_prob: float = 0.0,
        # 设置图块大小，默认为 16
        patch_size: int = 16,
        # 设置通道数，默认为 3
        num_channels: int = 3,
        # 设置池化大小，默认为 3
        pool_size: int = 3,
        # 设置下采样图块大小，默认为 3
        downsample_patch_size: int = 3,
        # 设置下采样步长，默认为 2
        downsample_stride: int = 2,
        # 设置下采样填充，默认为 1
        downsample_pad: int = 1,
        # 设置 drop path 比率，默认为 0.0
        drop_path_rate: float = 0.0,
        # 设置 3D 元块数量，默认为 1
        num_meta3d_blocks: int = 1,
        # 设置蒸馏标志，默认为 True
        distillation: bool = True,
        # 设置是否使用层尺度，默认为 True
        use_layer_scale: bool = True,
        # 设置层尺度初始值，默认为 1e-5
        layer_scale_init_value: float = 1e-5,
        # 设置隐藏层激活函数，默认为 "gelu"
        hidden_act: str = "gelu",
        # 设置初始化范围，默认为 0.02
        initializer_range: float = 0.02,
        # 设置层归一化 eps，默认为 1e-12
        layer_norm_eps: float = 1e-12,
        # 设置图像大小，默认为 224
        image_size: int = 224,
        # 设置批归一化 eps，默认为 1e-05
        batch_norm_eps: float = 1e-05,
        # 接收额外的参数并传递给基类的初始化函数
        **kwargs,
    ) -> None:
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 将参数赋值给对应的实例变量
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_sizes = hidden_sizes
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.depths = depths
        self.mlp_expansion_ratio = mlp_expansion_ratio
        self.downsamples = downsamples
        self.dim = dim
        self.key_dim = key_dim
        self.attention_ratio = attention_ratio
        self.resolution = resolution
        self.pool_size = pool_size
        self.downsample_patch_size = downsample_patch_size
        self.downsample_stride = downsample_stride
        self.downsample_pad = downsample_pad
        self.drop_path_rate = drop_path_rate
        self.num_meta3d_blocks = num_meta3d_blocks
        self.distillation = distillation
        self.use_layer_scale = use_layer_scale
        self.layer_scale_init_value = layer_scale_init_value
        self.image_size = image_size
        self.batch_norm_eps = batch_norm_eps
```