# `.\models\deprecated\van\configuration_van.py`

```
# 设置编码为 utf-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 根据 Apache License, Version 2.0 授权，除非遵守许可证，否则不得使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 在适用法律要求或书面同意的情况下，依据许可证分发的软件基于“原样”分发，不附带任何担保或条件，无论是明示还是暗示的
# 请查看许可证以获取有关特定语言的权限和限制
""" VAN model configuration"""

# 导入所需模块和函数
from ....configuration_utils import PretrainedConfig
from ....utils import logging

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 预训练模型配置归档映射
VAN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Visual-Attention-Network/van-base": (
        "https://huggingface.co/Visual-Attention-Network/van-base/blob/main/config.json"
    ),
}

# 定义 VanConfig 类，继承自 PretrainedConfig 类
class VanConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VanModel`]. It is used to instantiate a VAN model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the VAN
    [Visual-Attention-Network/van-base](https://huggingface.co/Visual-Attention-Network/van-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # VAN模型的配置参数类
    Args:
        # 图像的大小（分辨率）
        image_size (`int`, *optional*, defaults to 224):
        # 输入通道的数量
        num_channels (`int`, *optional*, defaults to 3):
        # 每个阶段嵌入层使用的补丁大小
        patch_sizes (`List[int]`, *optional*, defaults to `[7, 3, 3, 3]`):
        # 每个阶段嵌入层用于下采样输入的步幅大小
        strides (`List[int]`, *optional*, defaults to `[4, 2, 2, 2]`):
        # 每个阶段的隐藏层维度（隐藏大小）
        hidden_sizes (`List[int]`, *optional*, defaults to `[64, 128, 320, 512]`):
        # 每个阶段的网络深度（层数）
        depths (`List[int]`, *optional*, defaults to `[3, 3, 12, 3]`):
        # 每个阶段MLP层的扩展比率
        mlp_ratios (`List[int]`, *optional*, defaults to `[8, 8, 4, 4]`):
        # 每一层的非线性激活函数
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
        # 用于初始化所有权重矩阵的截断正态初始化器的标准偏差
        initializer_range (`float`, *optional*, defaults to 0.02):
        # 层标准化层使用的 epsilon 值
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
        # 层缩放的初始值
        layer_scale_init_value (`float`, *optional*, defaults to 0.01):
        # 随机深度的dropout概率
        drop_path_rate (`float`, *optional*, defaults to 0.0):
        # 用于dropout的概率
        dropout_rate (`float`, *optional*, defaults to 0.0):

    Example:
    # 示例
    ```python
    >>> from transformers import VanModel, VanConfig

    >>> # 初始化一个VAN van-base样式的配置
    >>> configuration = VanConfig()
    >>> # 从van-base样式的配置初始化一个模型
    >>> model = VanModel(configuration)
    >>> # 访问模型配置
    >>> configuration = model.config
    ```"""

    # 模型类型为"van"
    model_type = "van"

    # 初始化方法
    def __init__(
        # 图像的大小（分辨率）
        self,
        image_size=224,
        # 输入通道的数量
        num_channels=3,
        # 每个阶段嵌入层使用的补丁大小
        patch_sizes=[7, 3, 3, 3],
        # 每个阶段嵌入层用于下采样输入的步幅大小
        strides=[4, 2, 2, 2],
        # 每个阶段的隐藏层维度（隐藏大小）
        hidden_sizes=[64, 128, 320, 512],
        # 每个阶段的网络深度（层数）
        depths=[3, 3, 12, 3],
        # 每个阶段MLP层的扩展比率
        mlp_ratios=[8, 8, 4, 4],
        # 每一层的非线性激活函数
        hidden_act="gelu",
        # 用于初始化所有权重矩阵的截断正态初始化器的标准偏差
        initializer_range=0.02,
        # 层标准化层使用的 epsilon 值
        layer_norm_eps=1e-6,
        # 层缩放的初始值
        layer_scale_init_value=1e-2,
        # 随机深度的dropout概率
        drop_path_rate=0.0,
        # 用于dropout的概率
        dropout_rate=0.0,
        **kwargs,
    # 调用父类的构造函数并传入关键字参数
        super().__init__(**kwargs)
        
        # 设置图像大小
        self.image_size = image_size
        
        # 设置通道数量
        self.num_channels = num_channels
        
        # 设置补丁的大小
        self.patch_sizes = patch_sizes
        
        # 设置步幅
        self.strides = strides
        
        # 设置隐藏层的大小
        self.hidden_sizes = hidden_sizes
        
        # 设置深度
        self.depths = depths
        
        # 设置多层感知机的比例
        self.mlp_ratios = mlp_ratios
        
        # 设置隐藏层的激活函数
        self.hidden_act = hidden_act
        
        # 设置初始化范围
        self.initializer_range = initializer_range
        
        # 设置层归一化的 epsilon 值
        self.layer_norm_eps = layer_norm_eps
        
        # 设置层缩放初始化值
        self.layer_scale_init_value = layer_scale_init_value
        
        # 设置丢弃路径比率
        self.drop_path_rate = drop_path_rate
        
        # 设置丢失比率
        self.dropout_rate = dropout_rate
```