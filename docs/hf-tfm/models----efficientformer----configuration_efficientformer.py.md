# `.\models\efficientformer\configuration_efficientformer.py`

```
# 设置文件编码为 UTF-8
# 版权声明：2022 年由 HuggingFace Inc. 团队保留所有权利
#
# 根据 Apache 许可证 2.0 版本授权，除非遵守许可证的要求，否则不得使用此文件
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件根据“原样”的基础分发，
# 不提供任何形式的明示或暗示担保或条件
# 有关更多信息，请参阅许可证
""" EfficientFormer 模型配置 """

from typing import List

# 从 transformers 库中导入预训练配置类 PretrainedConfig
from ...configuration_utils import PretrainedConfig
# 从 transformers 库中导入日志工具 logging
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义 EfficientFormer 预训练模型配置文件的映射字典，指定模型名称及其配置文件的 URL
EFFICIENTFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "snap-research/efficientformer-l1-300": (
        "https://huggingface.co/snap-research/efficientformer-l1-300/resolve/main/config.json"
    ),
}


class EfficientFormerConfig(PretrainedConfig):
    r"""
    这是配置类，用于存储 [`EfficientFormerModel`] 的配置信息。根据指定的参数实例化 EfficientFormer 模型，
    定义模型架构。使用默认值实例化配置将产生类似于 EfficientFormer
    [snap-research/efficientformer-l1](https://huggingface.co/snap-research/efficientformer-l1) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。阅读 [`PretrainedConfig`] 的文档获取更多信息。

    示例：

    ```python
    >>> from transformers import EfficientFormerConfig, EfficientFormerModel

    >>> # 初始化 EfficientFormer efficientformer-l1 风格的配置
    >>> configuration = EfficientFormerConfig()

    >>> # 从 efficientformer-l3 风格的配置初始化 EfficientFormerModel（具有随机权重）
    >>> model = EfficientFormerModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```
    """

    # 模型类型标识符为 "efficientformer"
    model_type = "efficientformer"
    # 定义一个构造函数，初始化一个新的对象
    def __init__(
        self,
        depths: List[int] = [3, 2, 6, 4],  # 设置深度列表，默认为 [3, 2, 6, 4]
        hidden_sizes: List[int] = [48, 96, 224, 448],  # 设置隐藏层大小列表，默认为 [48, 96, 224, 448]
        downsamples: List[bool] = [True, True, True, True],  # 设置是否下采样列表，默认为 [True, True, True, True]
        dim: int = 448,  # 设置维度大小，默认为 448
        key_dim: int = 32,  # 设置键维度大小，默认为 32
        attention_ratio: int = 4,  # 设置注意力比例，默认为 4
        resolution: int = 7,  # 设置分辨率大小，默认为 7
        num_hidden_layers: int = 5,  # 设置隐藏层数量，默认为 5
        num_attention_heads: int = 8,  # 设置注意力头数量，默认为 8
        mlp_expansion_ratio: int = 4,  # 设置MLP扩展比率，默认为 4
        hidden_dropout_prob: float = 0.0,  # 设置隐藏层dropout概率，默认为 0.0
        patch_size: int = 16,  # 设置补丁大小，默认为 16
        num_channels: int = 3,  # 设置通道数量，默认为 3
        pool_size: int = 3,  # 设置池化大小，默认为 3
        downsample_patch_size: int = 3,  # 设置下采样补丁大小，默认为 3
        downsample_stride: int = 2,  # 设置下采样步长，默认为 2
        downsample_pad: int = 1,  # 设置下采样填充大小，默认为 1
        drop_path_rate: float = 0.0,  # 设置DropPath概率，默认为 0.0
        num_meta3d_blocks: int = 1,  # 设置Meta3D块数量，默认为 1
        distillation: bool = True,  # 是否进行蒸馏，默认为 True
        use_layer_scale: bool = True,  # 是否使用层比例，默认为 True
        layer_scale_init_value: float = 1e-5,  # 层比例初始化值，默认为 1e-5
        hidden_act: str = "gelu",  # 隐藏层激活函数，默认为 "gelu"
        initializer_range: float = 0.02,  # 初始化范围，默认为 0.02
        layer_norm_eps: float = 1e-12,  # 层归一化epsilon值，默认为 1e-12
        image_size: int = 224,  # 图像大小，默认为 224
        batch_norm_eps: float = 1e-05,  # 批归一化epsilon值，默认为 1e-05
        **kwargs,  # 接收任意额外的关键字参数
    ) -> None:  # 返回值为 None
        super().__init__(**kwargs)  # 调用父类的构造函数
    
        self.hidden_act = hidden_act  # 将隐藏层激活函数赋值给对象属性
        self.hidden_dropout_prob = hidden_dropout_prob  # 将隐藏层dropout概率赋值给对象属性
        self.hidden_sizes = hidden_sizes  # 将隐藏层大小列表赋值给对象属性
        self.num_hidden_layers = num_hidden_layers  # 将隐藏层数量赋值给对象属性
        self.num_attention_heads = num_attention_heads  # 将注意力头数量赋值给对象属性
        self.initializer_range = initializer_range  # 将初始化范围赋值给对象属性
        self.layer_norm_eps = layer_norm_eps  # 将层归一化epsilon值赋值给对象属性
        self.patch_size = patch_size  # 将补丁大小赋值给对象属性
        self.num_channels = num_channels  # 将通道数量赋值给对象属性
        self.depths = depths  # 将深度列表赋值给对象属性
        self.mlp_expansion_ratio = mlp_expansion_ratio  # 将MLP扩展比率赋值给对象属性
        self.downsamples = downsamples  # 将是否下采样列表赋值给对象属性
        self.dim = dim  # 将维度大小赋值给对象属性
        self.key_dim = key_dim  # 将键维度大小赋值给对象属性
        self.attention_ratio = attention_ratio  # 将注意力比例赋值给对象属性
        self.resolution = resolution  # 将分辨率大小赋值给对象属性
        self.pool_size = pool_size  # 将池化大小赋值给对象属性
        self.downsample_patch_size = downsample_patch_size  # 将下采样补丁大小赋值给对象属性
        self.downsample_stride = downsample_stride  # 将下采样步长赋值给对象属性
        self.downsample_pad = downsample_pad  # 将下采样填充大小赋值给对象属性
        self.drop_path_rate = drop_path_rate  # 将DropPath概率赋值给对象属性
        self.num_meta3d_blocks = num_meta3d_blocks  # 将Meta3D块数量赋值给对象属性
        self.distillation = distillation  # 将是否进行蒸馏赋值给对象属性
        self.use_layer_scale = use_layer_scale  # 将是否使用层比例赋值给对象属性
        self.layer_scale_init_value = layer_scale_init_value  # 将层比例初始化值赋值给对象属性
        self.image_size = image_size  # 将图像大小赋值给对象属性
        self.batch_norm_eps = batch_norm_eps  # 将批归一化epsilon值赋值给对象属性
```