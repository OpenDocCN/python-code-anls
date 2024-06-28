# `.\models\data2vec\configuration_data2vec_vision.py`

```py
# 设置文件编码为 UTF-8
# 版权声明：Meta Platforms 和 The HuggingFace Inc. 团队保留所有权利
#
# 根据 Apache 许可证 2.0 版本授权使用本文件
# 除非遵守许可证，否则不得使用本文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件基于"原样"分发，无任何明示或暗示的担保或条件
# 有关详细信息，请参阅许可证

""" Data2VecVision 模型配置 """

# 导入所需的模块
from collections import OrderedDict  # 导入有序字典模块
from typing import Mapping  # 导入类型提示 Mapping

from packaging import version  # 导入版本管理模块

# 导入配置工具函数和类
from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...onnx import OnnxConfig  # 导入 ONNX 配置类
from ...utils import logging  # 导入日志工具模块

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 预训练配置存档映射表
DATA2VEC_VISION_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/data2vec-vision-base-ft": (
        "https://huggingface.co/facebook/data2vec-vision-base-ft/resolve/main/config.json"
    ),
}

class Data2VecVisionConfig(PretrainedConfig):
    r"""
    这是用于存储 [`Data2VecVisionModel`] 配置的配置类。根据指定的参数实例化 Data2VecVision 模型，定义模型架构。
    使用默认值实例化配置将产生类似于 Data2VecVision [facebook/data2vec-vision-base](https://huggingface.co/facebook/data2vec-vision-base) 架构的配置。

    示例:

    ```
    >>> from transformers import Data2VecVisionConfig, Data2VecVisionModel

    >>> # 初始化一个 Data2VecVision data2vec_vision-base-patch16-224-in22k 风格的配置
    >>> configuration = Data2VecVisionConfig()

    >>> # 从上述配置初始化一个（带有随机权重）模型
    >>> model = Data2VecVisionModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```
    """

    model_type = "data2vec-vision"
    # 初始化函数，用于设置Transformer模型的各项参数
    def __init__(
        self,
        hidden_size=768,  # Transformer中隐藏层的大小，默认为768
        num_hidden_layers=12,  # Transformer中的隐藏层数，默认为12
        num_attention_heads=12,  # 每个注意力头的数量，默认为12
        intermediate_size=3072,  # Feedforward层的中间大小，默认为3072
        hidden_act="gelu",  # 激活函数类型，默认为GELU
        hidden_dropout_prob=0.0,  # 隐藏层的Dropout概率，默认为0.0（无Dropout）
        attention_probs_dropout_prob=0.0,  # 注意力层的Dropout概率，默认为0.0（无Dropout）
        initializer_range=0.02,  # 参数初始化的范围，默认为0.02
        layer_norm_eps=1e-12,  # Layer normalization的epsilon值，默认为1e-12
        image_size=224,  # 输入图像的大小，默认为224
        patch_size=16,  # 每个patch的大小，默认为16
        num_channels=3,  # 输入图像的通道数，默认为3（RGB）
        use_mask_token=False,  # 是否使用Mask Token，默认为False
        use_absolute_position_embeddings=False,  # 是否使用绝对位置编码，默认为False
        use_relative_position_bias=False,  # 是否使用相对位置偏置，默认为False
        use_shared_relative_position_bias=False,  # 是否使用共享的相对位置偏置，默认为False
        layer_scale_init_value=0.1,  # 层次标度初始化值，默认为0.1
        drop_path_rate=0.1,  # Drop Path的概率，默认为0.1
        use_mean_pooling=True,  # 是否使用均值池化，默认为True
        out_indices=[3, 5, 7, 11],  # 输出索引的列表（用于解码头），默认为[3, 5, 7, 11]
        pool_scales=[1, 2, 3, 6],  # 池化的尺度列表（用于解码头），默认为[1, 2, 3, 6]
        use_auxiliary_head=True,  # 是否使用辅助解码头，默认为True
        auxiliary_loss_weight=0.4,  # 辅助解码头的损失权重，默认为0.4
        auxiliary_channels=256,  # 辅助解码头的通道数，默认为256
        auxiliary_num_convs=1,  # 辅助解码头中的卷积层数，默认为1
        auxiliary_concat_input=False,  # 辅助解码头中是否将输入进行拼接，默认为False
        semantic_loss_ignore_index=255,  # 语义损失中要忽略的索引，默认为255
        **kwargs,  # 其他未指定参数，以字典形式接收
    ):
        super().__init__(**kwargs)
    
        # 设置Transformer模型的各项参数
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
    
        # 图像相关的参数设置
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.use_mask_token = use_mask_token
        self.use_absolute_position_embeddings = use_absolute_position_embeddings
        self.use_relative_position_bias = use_relative_position_bias
        self.use_shared_relative_position_bias = use_shared_relative_position_bias
        self.layer_scale_init_value = layer_scale_init_value
        self.drop_path_rate = drop_path_rate
        self.use_mean_pooling = use_mean_pooling
    
        # 解码头相关的参数设置（语义分割）
        self.out_indices = out_indices
        self.pool_scales = pool_scales
    
        # 辅助解码头相关的参数设置（语义分割）
        self.use_auxiliary_head = use_auxiliary_head
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.auxiliary_channels = auxiliary_channels
        self.auxiliary_num_convs = auxiliary_num_convs
        self.auxiliary_concat_input = auxiliary_concat_input
        self.semantic_loss_ignore_index = semantic_loss_ignore_index
# 从transformers.models.vit.configuration_vit.ViTOnnxConfig复制而来的类Data2VecVisionOnnxConfig，继承自OnnxConfig类
class Data2VecVisionOnnxConfig(OnnxConfig):
    # 设定torch_onnx_minimum_version属性为1.11版本
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义inputs属性为一个有序字典，描述模型输入的名称及其对应的维度顺序
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    # 定义atol_for_validation属性，返回浮点数1e-4，用作验证时的绝对误差容差
    @property
    def atol_for_validation(self) -> float:
        return 1e-4
```