# `.\transformers\models\bit\configuration_bit.py`

```
# coding=utf-8
# 版权所有 2022 年 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版（“许可证”）授权；
# 除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件，
# 没有任何明示或暗示的保证或条件。
# 有关许可证的详细信息，请参阅许可证。
""" BiT 模型配置"""

# 从配置工具中导入预训练配置
from ...configuration_utils import PretrainedConfig
# 导入日志记录工具
from ...utils import logging
# 导入 Backbone 相关配置混合类和输出特征对齐工具函数
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices

# 获取日志记录器
logger = logging.get_logger(__name__)

# BiT 预训练配置文件的映射字典
BIT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/bit-50": "https://huggingface.co/google/bit-50/resolve/main/config.json",
}


# BiT 配置类，继承了 BackboneConfigMixin 和 PretrainedConfig
class BitConfig(BackboneConfigMixin, PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`BitModel`] 的配置。根据指定的参数来实例化 BiT 模型，定义模型架构。
    使用默认值实例化配置将产生与 BiT [google/bit-50](https://huggingface.co/google/bit-50) 架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。有关更多信息，请阅读 [`PretrainedConfig`] 的文档。
    # 定义一个函数，用于创建 BiT 模型的配置
    Args:
        num_channels (`int`, *optional*, defaults to 3):
            输入通道的数量，默认为 3。
        embedding_size (`int`, *optional*, defaults to 64):
            嵌入层的维度（隐藏大小），默认为 64。
        hidden_sizes (`List[int]`, *optional*, defaults to `[256, 512, 1024, 2048]`):
            每个阶段的隐藏大小（层的维度），默认为 `[256, 512, 1024, 2048]`。
        depths (`List[int]`, *optional*, defaults to `[3, 4, 6, 3]`):
            每个阶段的深度（层数），默认为 `[3, 4, 6, 3]`。
        layer_type (`str`, *optional*, defaults to `"preactivation"`):
            要使用的层类型，可以是 `"preactivation"` 或 `"bottleneck"`，默认为 `"preactivation"`。
        hidden_act (`str`, *optional*, defaults to `"relu"`):
            每个块中的非线性激活函数。如果是字符串，支持 `"gelu"`、`"relu"`、`"selu"` 和 `"gelu_new"`，默认为 `"relu"`。
        global_padding (`str`, *optional*):
            用于卷积层的填充策略。可以是 `"valid"`、`"same"` 或 `None`，默认为 `None`。
        num_groups (`int`, *optional*, defaults to 32):
            用于 `BitGroupNormActivation` 层的组数，默认为 32。
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            随机深度的丢弃路径率，默认为 0.0。
        embedding_dynamic_padding (`bool`, *optional*, defaults to `False`):
            是否使用动态填充来进行嵌入层的填充，默认为 `False`。
        output_stride (`int`, *optional*, defaults to 32):
            模型的输出步幅，默认为 32。
        width_factor (`int`, *optional*, defaults to 1):
            模型的宽度因子，默认为 1。
        out_features (`List[str]`, *optional*):
            如果作为骨干网络使用，要输出的特征列表。可以是 `"stem"`、`"stage1"`、`"stage2"` 等（取决于模型有多少阶段）。
            如果未设置且设置了 `out_indices`，将默认为相应的阶段。如果未设置且未设置 `out_indices`，将默认为最后一个阶段。
            必须与 `stage_names` 属性中定义的顺序相同。
        out_indices (`List[int]`, *optional*):
            如果作为骨干网络使用，要输出的特征的索引列表。可以是 0、1、2 等（取决于模型有多少阶段）。
            如果未设置且设置了 `out_features`，将默认为相应的阶段。如果未设置且未设置 `out_features`，将默认为最后一个阶段。
            必须与 `stage_names` 属性中定义的顺序相同。
    
    Example:
    
    >>> from transformers import BitConfig, BitModel
    
    >>> # 初始化一个 BiT bit-50 风格的配置
    >>> configuration = BitConfig()
    
    >>> # 从 bit-50 风格的配置中初始化一个（带有随机权重）模型
    >>> model = BitModel(configuration)
    
    >>> # 访问模型配置
    >>> configuration = model.config
    
    
    model_type = "bit"
    # 定义支持的层类型
    layer_types = ["preactivation", "bottleneck"]
    # 定义支持的填充方式
    supported_padding = ["SAME", "VALID"]
    
    # 初始化函数，用于创建一个深度神经网络模型
    def __init__(
        self,
        num_channels=3,  # 输入通道数，默认为3
        embedding_size=64,  # 嵌入尺寸，默认为64
        hidden_sizes=[256, 512, 1024, 2048],  # 隐藏层的尺寸列表，默认为[256, 512, 1024, 2048]
        depths=[3, 4, 6, 3],  # 各个阶段的深度列表，默认为[3, 4, 6, 3]
        layer_type="preactivation",  # 网络层类型，默认为"preactivation"
        hidden_act="relu",  # 隐藏层的激活函数，默认为"relu"
        global_padding=None,  # 全局填充策略，默认为None
        num_groups=32,  # 卷积分组数，默认为32
        drop_path_rate=0.0,  # DropPath率，默认为0.0
        embedding_dynamic_padding=False,  # 是否使用嵌入动态填充，默认为False
        output_stride=32,  # 输出步幅，默认为32
        width_factor=1,  # 宽度因子，默认为1
        out_features=None,  # 输出特征数，默认为None
        out_indices=None,  # 输出索引，默认为None
        **kwargs,  # 其他关键字参数
    ):
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 检查层类型是否在支持的层类型列表中，如果不在则抛出异常
        if layer_type not in self.layer_types:
            raise ValueError(f"layer_type={layer_type} is not one of {','.join(self.layer_types)}")
        # 检查全局填充策略是否被支持，如果不支持则抛出异常
        if global_padding is not None:
            if global_padding.upper() in self.supported_padding:
                global_padding = global_padding.upper()
            else:
                raise ValueError(f"Padding strategy {global_padding} not supported")
        # 初始化各个参数
        self.num_channels = num_channels
        self.embedding_size = embedding_size
        self.hidden_sizes = hidden_sizes
        self.depths = depths
        self.layer_type = layer_type
        self.hidden_act = hidden_act
        self.global_padding = global_padding
        self.num_groups = num_groups
        self.drop_path_rate = drop_path_rate
        self.embedding_dynamic_padding = embedding_dynamic_padding
        self.output_stride = output_stride
        self.width_factor = width_factor
    
        # 设置阶段名称列表，包括' stem '（干细胞）和阶段编号，从1到深度列表的长度加1
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(depths) + 1)]
        # 获得对齐的输出特征和输出索引
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )
```