# `.\transformers\models\beit\configuration_beit.py`

```
# 导入所需模块和类
from collections import OrderedDict
from typing import Mapping

# 导入版本管理模块
from packaging import version

# 导入预训练配置类和相关模块
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义预训练模型配置文件的下载地址映射
BEIT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/beit-base-patch16-224-pt22k": (
        "https://huggingface.co/microsoft/beit-base-patch16-224-pt22k/resolve/main/config.json"
    ),
    # 查看所有 BEiT 模型的下载地址: https://huggingface.co/models?filter=beit
}

# 定义 BEiT 模型配置类
class BeitConfig(BackboneConfigMixin, PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BeitModel`]. It is used to instantiate an BEiT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the BEiT
    [microsoft/beit-base-patch16-224-pt22k](https://huggingface.co/microsoft/beit-base-patch16-224-pt22k) architecture.

    Example:

    ```python
    >>> from transformers import BeitConfig, BeitModel

    >>> # Initializing a BEiT beit-base-patch16-224-pt22k style configuration
    >>> configuration = BeitConfig()

    >>> # Initializing a model (with random weights) from the beit-base-patch16-224-pt22k style configuration
    >>> model = BeitModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    # 设定模型类型
    model_type = "beit"
    # 初始化函数，设置模型的各种参数
    def __init__(
        self,
        vocab_size=8192,  # 词汇表大小，默认为8192
        hidden_size=768,  # 隐藏层大小，默认为768
        num_hidden_layers=12,  # 隐藏层层数，默认为12
        num_attention_heads=12,  # 注意力头数，默认为12
        intermediate_size=3072,  # 中间层大小，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为gelu
        hidden_dropout_prob=0.0,  # 隐藏层dropout概率，默认为0.0
        attention_probs_dropout_prob=0.0,  # 注意力概率dropout概率，默认为0.0
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-12,  # 层归一化epsilon，默认为1e-12
        image_size=224,  # 图像大小，默认为224
        patch_size=16,  # 图像块大小，默认为16
        num_channels=3,  # 图像通道数，默认为3
        use_mask_token=False,  # 是否使用mask token，默认为False
        use_absolute_position_embeddings=False,  # 是否使用绝对位置嵌入，默认为False
        use_relative_position_bias=False,  # 是否使用相对位置偏置，默认为False
        use_shared_relative_position_bias=False,  # 是否使用共享相对位置偏置，默认为False
        layer_scale_init_value=0.1,  # 层缩放初始化值，默认为0.1
        drop_path_rate=0.1,  # drop path率，默认为0.1
        use_mean_pooling=True,  # 是否使用平均池化，默认为True
        pool_scales=[1, 2, 3, 6],  # 池化尺度列表，默认为[1, 2, 3, 6]
        use_auxiliary_head=True,  # 是否使用辅助头，默认为True
        auxiliary_loss_weight=0.4,  # 辅助损失权重，默认为0.4
        auxiliary_channels=256,  # 辅助头通道数，默认为256
        auxiliary_num_convs=1,  # 辅助头卷积层数，默认为1
        auxiliary_concat_input=False,  # 是否将输入与辅助头连接，默认为False
        semantic_loss_ignore_index=255,  # 语义损失忽略索引，默认为255
        out_features=None,  # 输出特征，默认为None
        out_indices=None,  # 输出索引，默认为None
        add_fpn=False,  # 是否添加特征金字塔网络，默认为False
        reshape_hidden_states=True,  # 是否重塑隐藏状态，默认为True
        **kwargs,  # 其他参数
        # 调用父类的构造函数，初始化参数
        super().__init__(**kwargs)

        # 设置模型的词汇表大小
        self.vocab_size = vocab_size
        # 设置隐藏层大小
        self.hidden_size = hidden_size
        # 设置隐藏层数量
        self.num_hidden_layers = num_hidden_layers
        # 设置注意力头的数量
        self.num_attention_heads = num_attention_heads
        # 设置中间层大小
        self.intermediate_size = intermediate_size
        # 设置隐藏层激活函数
        self.hidden_act = hidden_act
        # 设置隐藏层的丢弃率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 设置注意力概率的丢弃率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 设置初始化范围
        self.initializer_range = initializer_range
        # 设置层标准化的 epsilon
        self.layer_norm_eps = layer_norm_eps

        # 设置图像大小
        self.image_size = image_size
        # 设置图像块大小
        self.patch_size = patch_size
        # 设置通道数量
        self.num_channels = num_channels
        # 是否使用掩码令牌
        self.use_mask_token = use_mask_token
        # 是否使用绝对位置嵌入
        self.use_absolute_position_embeddings = use_absolute_position_embeddings
        # 是否使用相对位置偏差
        self.use_relative_position_bias = use_relative_position_bias
        # 是否使用共享的相对位置偏差
        self.use_shared_relative_position_bias = use_shared_relative_position_bias
        # 层比例初始化值
        self.layer_scale_init_value = layer_scale_init_value
        # 丢弃路径率
        self.drop_path_rate = drop_path_rate
        # 是否使用平均池化
        self.use_mean_pooling = use_mean_pooling
        # 解码头属性（语义分割）
        self.pool_scales = pool_scales
        # 辅助头属性（语义分割）
        self.use_auxiliary_head = use_auxiliary_head
        # 辅助损失权重
        self.auxiliary_loss_weight = auxiliary_loss_weight
        # 辅助头通道数量
        self.auxiliary_channels = auxiliary_channels
        # 辅助头卷积数量
        self.auxiliary_num_convs = auxiliary_num_convs
        # 辅助头输入连接
        self.auxiliary_concat_input = auxiliary_concat_input
        # 语义损失忽略索引
        self.semantic_loss_ignore_index = semantic_loss_ignore_index

        # 处理向后兼容性
        # 如果参数中包含 "segmentation_indices"，发出警告，并将其替换为 "out_indices"
        if "segmentation_indices" in kwargs:
            logger.warning(
                "The `segmentation_indices` argument is deprecated and will be removed in a future version, use `out_indices` instead.",
                FutureWarning,
            )
            out_indices = kwargs.pop("segmentation_indices")

        # 设置骨干网络属性
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, self.num_hidden_layers + 1)]
        # 获取对齐的输出特征和输出索引
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )
        # 是否添加特征金字塔网络（FPN）
        self.add_fpn = add_fpn
        # 是否重塑隐藏状态
        self.reshape_hidden_states = reshape_hidden_states
# 从transformers.models.vit.configuration_vit.ViTOnnxConfig中复制代码，定义BeitOnnxConfig类，继承自OnnxConfig类
class BeitOnnxConfig(OnnxConfig):
    # 定义torch_onnx_minimum_version属性，指定Torch与ONNX的最低版本要求为1.11
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义inputs属性，用于指定模型输入的维度信息，返回一个有序字典
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                # 指定模型输入的名称为"pixel_values"，对应的维度顺序为(batch, num_channels, height, width)
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    # 定义atol_for_validation属性，指定用于验证的绝对误差阈值为1e-4
    @property
    def atol_for_validation(self) -> float:
        return 1e-4
```