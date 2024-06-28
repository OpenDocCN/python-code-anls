# `.\models\deprecated\van\configuration_van.py`

```py
# coding=utf-8
# 版权所有 2022 年 HuggingFace Inc. 团队保留所有权利。
#
# 根据 Apache 许可证版本 2.0 授权;
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证副本:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件是基于"原样"提供的，
# 没有任何明示或暗示的保证或条件。
# 有关更多信息，请参阅许可证。

""" VAN 模型配置"""

from ....configuration_utils import PretrainedConfig  # 导入预训练配置类
from ....utils import logging  # 导入日志工具

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

VAN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Visual-Attention-Network/van-base": (
        "https://huggingface.co/Visual-Attention-Network/van-base/blob/main/config.json"
    ),
}

class VanConfig(PretrainedConfig):
    r"""
    这是配置类，用于存储 [`VanModel`] 的配置。根据指定的参数实例化 VAN 模型，
    定义模型架构。使用默认值实例化配置将生成与 VAN
    [Visual-Attention-Network/van-base](https://huggingface.co/Visual-Attention-Network/van-base)
    架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读来自 [`PretrainedConfig`] 的文档获取更多信息。
    """
    Args:
        image_size (`int`, *optional*, defaults to 224):
            每个图像的大小（分辨率）。
        num_channels (`int`, *optional*, defaults to 3):
            输入通道的数量。
        patch_sizes (`List[int]`, *optional*, defaults to `[7, 3, 3, 3]`):
            每个阶段嵌入层使用的补丁大小。
        strides (`List[int]`, *optional*, defaults to `[4, 2, 2, 2]`):
            每个阶段嵌入层用来降采样输入的步幅大小。
        hidden_sizes (`List[int]`, *optional*, defaults to `[64, 128, 320, 512]`):
            每个阶段的隐藏层维度。
        depths (`List[int]`, *optional*, defaults to `[3, 3, 12, 3]`):
            每个阶段的层数。
        mlp_ratios (`List[int]`, *optional*, defaults to `[8, 8, 4, 4]`):
            每个阶段MLP层的扩展比率。
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            每一层的非线性激活函数（可以是函数或字符串）。支持的字符串有 "gelu", "relu", "selu" 和 "gelu_new"。
        initializer_range (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态初始化器的标准差。
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            层归一化层使用的 epsilon。
        layer_scale_init_value (`float`, *optional*, defaults to 0.01):
            层缩放的初始值。
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            随机深度（stochastic depth）的丢弃概率。
        dropout_rate (`float`, *optional*, defaults to 0.0):
            丢弃的概率。

    Example:
    ```
    >>> from transformers import VanModel, VanConfig

    >>> # Initializing a VAN van-base style configuration
    >>> configuration = VanConfig()
    >>> # Initializing a model from the van-base style configuration
    >>> model = VanModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    ):
        # 调用父类的初始化方法，传递所有的关键字参数
        super().__init__(**kwargs)
        # 设置图像大小
        self.image_size = image_size
        # 设置通道数
        self.num_channels = num_channels
        # 设置每个补丁的大小
        self.patch_sizes = patch_sizes
        # 设置每个补丁的步长
        self.strides = strides
        # 设置隐藏层的大小
        self.hidden_sizes = hidden_sizes
        # 设置深度
        self.depths = depths
        # 设置多层感知机（MLP）的比率
        self.mlp_ratios = mlp_ratios
        # 设置隐藏层的激活函数
        self.hidden_act = hidden_act
        # 设置初始化范围
        self.initializer_range = initializer_range
        # 设置层归一化的 epsilon 值
        self.layer_norm_eps = layer_norm_eps
        # 设置层尺度初始化值
        self.layer_scale_init_value = layer_scale_init_value
        # 设置丢弃路径的比率
        self.drop_path_rate = drop_path_rate
        # 设置丢弃率
        self.dropout_rate = dropout_rate
```