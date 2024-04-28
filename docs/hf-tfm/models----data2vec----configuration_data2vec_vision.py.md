# `.\models\data2vec\configuration_data2vec_vision.py`

```py
# 设置文件编码格式为 utf-8
# 版权声明和许可证信息
#
# 版权所有 Meta Platforms 和 HuggingFace Inc. 团队。
# 根据 Apache 许可证 2.0 版本（“许可证”）进行许可。
# 您不能在不符合许可证的情况下使用此文件。
# 您可以在以下链接获取许可证的副本：
# http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律需要或书面同意，否则依照许可证分发的软件将基于“现状”分发，
# 没有任何明示或暗示的担保或条件。
# 请查看许可证以获取特定语言的权限和限制。
""" Data2VecVision 模型配置"""
# 导入必要的库和模块
from collections import OrderedDict
from typing import Mapping
from packaging import version
# 导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 导入 ONNX 配置
from ...onnx import OnnxConfig
# 导入日志工具
from ...utils import logging

# 获取 logger 实例
logger = logging.get_logger(__name__)

# 预训练配置文件的映射表
DATA2VEC_VISION_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/data2vec-vision-base-ft": (
        "https://huggingface.co/facebook/data2vec-vision-base-ft/resolve/main/config.json"
    ),
}

# Data2VecVision 配置类，继承自 PretrainedConfig
class Data2VecVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Data2VecVisionModel`]. It is used to instantiate
    an Data2VecVision model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Data2VecVision
    [facebook/data2vec-vision-base](https://huggingface.co/facebook/data2vec-vision-base) architecture.

    Example:

    ```python
    >>> from transformers import Data2VecVisionConfig, Data2VecVisionModel

    >>> # Initializing a Data2VecVision data2vec_vision-base-patch16-224-in22k style configuration
    >>> configuration = Data2VecVisionConfig()

    >>> # Initializing a model (with random weights) from the data2vec_vision-base-patch16-224-in22k style configuration
    >>> model = Data2VecVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```py"""
    # 模型类型为 data2vec-vision
    model_type = "data2vec-vision"
    # 初始化函数，设置模型的各种参数和属性
    def __init__(
        self,
        hidden_size=768,  # 隐藏层的大小，默认为768
        num_hidden_layers=12,  # 隐藏层的层数，默认为12
        num_attention_heads=12,  # 注意力头的数量，默认为12
        intermediate_size=3072,  # 中间层的大小，默认为3072
        hidden_act="gelu",  # 隐藏层的激活函数，默认为gelu
        hidden_dropout_prob=0.0,  # 隐藏层的dropout概率，默认为0.0
        attention_probs_dropout_prob=0.0,  # 注意力概率的dropout概率，默认为0.0
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-12,  # 层标准化的epsilon值，默认为1e-12
        image_size=224,  # 图像的大小，默认为224
        patch_size=16,  # 补丁的大小，默认为16
        num_channels=3,  # 通道的数量，默认为3
        use_mask_token=False,  # 是否使用掩码令牌，默认为False
        use_absolute_position_embeddings=False,  # 是否使用绝对位置嵌入，默认为False
        use_relative_position_bias=False,  # 是否使用相对位置偏差，默认为False
        use_shared_relative_position_bias=False,  # 是否使用共享的相对位置偏差，默认为False
        layer_scale_init_value=0.1,  # 层缩放的初始值，默认为0.1
        drop_path_rate=0.1,  # 丢弃路径的比率，默认为0.1
        use_mean_pooling=True,  # 是否使用均值池化，默认为True
        out_indices=[3, 5, 7, 11],  # 输出索引，默认为[3, 5, 7, 11]
        pool_scales=[1, 2, 3, 6],  # 池化的尺度，默认为[1, 2, 3, 6]
        use_auxiliary_head=True,  # 是否使用辅助头，默认为True
        auxiliary_loss_weight=0.4,  # 辅助损失的权重，默认为0.4
        auxiliary_channels=256,  # 辅助通道的数量，默认为256
        auxiliary_num_convs=1,  # 辅助卷积的数量，默认为1
        auxiliary_concat_input=False,  # 是否连接输入和辅助，默认为False
        semantic_loss_ignore_index=255,  # 语义损失忽略的索引，默认为255
        **kwargs,  # 其他关键字参数
    ):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 设置模型的各种属性
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

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
        # 解码头属性（语义分割）
        self.out_indices = out_indices
        self.pool_scales = pool_scales
        # 辅助头属性（语义分割）
        self.use_auxiliary_head = use_auxiliary_head
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.auxiliary_channels = auxiliary_channels
        self.auxiliary_num_convs = auxiliary_num_convs
        self.auxiliary_concat_input = auxiliary_concat_input
        self.semantic_loss_ignore_index = semantic_loss_ignore_index
# 从transformers.models.vit.configuration_vit.ViTOnnxConfig中复制代码，定义Data2VecVisionOnnxConfig类
class Data2VecVisionOnnxConfig(OnnxConfig):
    # 定义torch_onnx_minimum_version属性，赋值为1.11
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义inputs属性，返回有序字典，包含像素值的键值对
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    # 定义atol_for_validation属性，返回浮点数1e-4
    @property
    def atol_for_validation(self) -> float:
        return 1e-4
```