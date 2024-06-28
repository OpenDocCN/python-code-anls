# `.\models\pvt_v2\configuration_pvt_v2.py`

```py
# coding=utf-8
# 版权 2024 作者: Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan,
# Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao 和 HuggingFace 公司团队。
# 保留所有权利。
#
# 根据 Apache 许可证版本 2.0 许可；
# 除非符合许可证的要求，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件根据“原样”分发，
# 不附带任何明示或暗示的担保或条件。
# 有关更多信息，请参见许可证。
"""Pvt V2 模型配置"""

from typing import Callable, List, Tuple, Union

# 导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 导入日志工具
from ...utils import logging
# 导入骨干网络配置混合类和获取对齐输出特征输出索引函数
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices

# 获取全局日志记录器
logger = logging.get_logger(__name__)

# Pvt V2 预训练模型配置映射表，指定不同模型的预训练地址
PVT_V2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "pvt_v2_b0": "https://huggingface.co/OpenGVLab/pvt_v2_b0",
    "pvt_v2_b1": "https://huggingface.co/OpenGVLab/pvt_v2_b1",
    "pvt_v2_b2": "https://huggingface.co/OpenGVLab/pvt_v2_b2",
    "pvt_v2_b2_linear": "https://huggingface.co/OpenGVLab/pvt_v2_b2_linear",
    "pvt_v2_b3": "https://huggingface.co/OpenGVLab/pvt_v2_b3",
    "pvt_v2_b4": "https://huggingface.co/OpenGVLab/pvt_v2_b4",
    "pvt_v2_b5": "https://huggingface.co/OpenGVLab/pvt_v2_b5",
}

# Pvt V2 模型配置类，继承自骨干网络配置混合类和预训练配置类
class PvtV2Config(BackboneConfigMixin, PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`PvtV2Model`] 的配置信息。根据指定的参数实例化 Pvt V2 模型，
    定义模型的架构。使用默认配置进行实例化将产生类似于 Pvt V2 B0
    [OpenGVLab/pvt_v2_b0](https://huggingface.co/OpenGVLab/pvt_v2_b0) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读
    [`PretrainedConfig`] 的文档以获取更多信息。

    示例：

    ```
    >>> from transformers import PvtV2Model, PvtV2Config

    >>> # 初始化一个 pvt_v2_b0 风格的配置
    >>> configuration = PvtV2Config()

    >>> # 从 OpenGVLab/pvt_v2_b0 风格的配置初始化一个模型
    >>> model = PvtV2Model(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```
    """

    # 模型类型
    model_type = "pvt_v2"
    # 初始化函数，用于创建一个新的对象实例
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 224,  # 图像尺寸，可以是单个整数或元组表示的宽高
        num_channels: int = 3,  # 图像通道数，默认为3（RGB）
        num_encoder_blocks: int = 4,  # 编码器块的数量，默认为4
        depths: List[int] = [2, 2, 2, 2],  # 每个阶段的编码器块的深度列表，默认为每阶段2个块
        sr_ratios: List[int] = [8, 4, 2, 1],  # 每个阶段的空间分辨率缩放比例列表，默认为递减的倍数
        hidden_sizes: List[int] = [32, 64, 160, 256],  # 每个阶段的隐藏层大小列表，默认为递增
        patch_sizes: List[int] = [7, 3, 3, 3],  # 每个阶段的图像块大小列表，默认为不同的尺寸
        strides: List[int] = [4, 2, 2, 2],  # 每个阶段的步长列表，默认为不同的步幅
        num_attention_heads: List[int] = [1, 2, 5, 8],  # 每个阶段的注意力头数列表，默认为不同的数量
        mlp_ratios: List[int] = [8, 8, 4, 4],  # 每个阶段的MLP层的扩展比例列表，默认为不同的倍数
        hidden_act: Union[str, Callable] = "gelu",  # 隐藏层激活函数，默认为GELU函数
        hidden_dropout_prob: float = 0.0,  # 隐藏层的dropout概率，默认为0.0，即无dropout
        attention_probs_dropout_prob: float = 0.0,  # 注意力概率的dropout概率，默认为0.0，即无dropout
        initializer_range: float = 0.02,  # 初始化范围，默认为0.02
        drop_path_rate: float = 0.0,  # 丢弃路径的比率，默认为0.0，即无丢弃
        layer_norm_eps: float = 1e-6,  # 层归一化的epsilon值，默认为1e-6
        qkv_bias: bool = True,  # 是否在QKV层使用偏置，默认为True
        linear_attention: bool = False,  # 是否使用线性注意力，默认为False
        out_features=None,  # 输出特征，用于描述模型输出的特征信息，默认为None
        out_indices=None,  # 输出索引，用于描述模型输出的索引信息，默认为None
        **kwargs,  # 其他关键字参数，用于接收除上述参数外的其他参数
    ):
        super().__init__(**kwargs)  # 调用父类的初始化方法

        image_size = (image_size, image_size) if isinstance(image_size, int) else image_size  # 如果image_size是整数，则转换为元组形式

        self.image_size = image_size  # 设置对象实例的图像尺寸属性
        self.num_channels = num_channels  # 设置对象实例的图像通道数属性
        self.num_encoder_blocks = num_encoder_blocks  # 设置对象实例的编码器块数量属性
        self.depths = depths  # 设置对象实例的深度列表属性
        self.sr_ratios = sr_ratios  # 设置对象实例的空间分辨率缩放比例列表属性
        self.hidden_sizes = hidden_sizes  # 设置对象实例的隐藏层大小列表属性
        self.patch_sizes = patch_sizes  # 设置对象实例的图像块大小列表属性
        self.strides = strides  # 设置对象实例的步长列表属性
        self.mlp_ratios = mlp_ratios  # 设置对象实例的MLP层扩展比例列表属性
        self.num_attention_heads = num_attention_heads  # 设置对象实例的注意力头数列表属性
        self.hidden_act = hidden_act  # 设置对象实例的隐藏层激活函数属性
        self.hidden_dropout_prob = hidden_dropout_prob  # 设置对象实例的隐藏层dropout概率属性
        self.attention_probs_dropout_prob = attention_probs_dropout_prob  # 设置对象实例的注意力dropout概率属性
        self.initializer_range = initializer_range  # 设置对象实例的初始化范围属性
        self.drop_path_rate = drop_path_rate  # 设置对象实例的丢弃路径比率属性
        self.layer_norm_eps = layer_norm_eps  # 设置对象实例的层归一化epsilon值属性
        self.qkv_bias = qkv_bias  # 设置对象实例的QKV层是否使用偏置属性
        self.linear_attention = linear_attention  # 设置对象实例的是否使用线性注意力属性
        self.stage_names = [f"stage{idx}" for idx in range(1, len(depths) + 1)]  # 设置对象实例的阶段名称列表属性
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )  # 调用函数获取对齐的输出特征和输出索引，设置对象实例的相关属性
```