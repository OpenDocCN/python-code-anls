# `.\transformers\models\vitdet\configuration_vitdet.py`

```
# 设置文件编码为 utf-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关特定语言的权限和限制
""" VitDet 模型配置"""

# 导入必要的模块和类
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练配置映射
VITDET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/vit-det-base": "https://huggingface.co/facebook/vit-det-base/resolve/main/config.json",
}

# VitDet 配置类，继承自 BackboneConfigMixin 和 PretrainedConfig
class VitDetConfig(BackboneConfigMixin, PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VitDetModel`]. It is used to instantiate an
    VitDet model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the VitDet
    [google/vitdet-base-patch16-224](https://huggingface.co/google/vitdet-base-patch16-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```python
    >>> from transformers import VitDetConfig, VitDetModel

    >>> # Initializing a VitDet configuration
    >>> configuration = VitDetConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = VitDetModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    # 模型类型为 "vitdet"
    model_type = "vitdet"

    # 初始化方法，设置各种配置参数
    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        mlp_ratio=4,
        hidden_act="gelu",
        dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        image_size=224,
        pretrain_image_size=224,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        drop_path_rate=0.0,
        window_block_indices=[],
        residual_block_indices=[],
        use_absolute_position_embeddings=True,
        use_relative_position_embeddings=False,
        window_size=0,
        out_features=None,
        out_indices=None,
        **kwargs,
        # 调用父类的构造函数，传入关键字参数
        super().__init__(**kwargs)

        # 初始化模型的隐藏层大小
        self.hidden_size = hidden_size
        # 初始化模型的隐藏层数量
        self.num_hidden_layers = num_hidden_layers
        # 初始化模型的注意力头数量
        self.num_attention_heads = num_attention_heads
        # 初始化MLP的比例
        self.mlp_ratio = mlp_ratio
        # 初始化隐藏层的激活函数
        self.hidden_act = hidden_act
        # 初始化模型的dropout概率
        self.dropout_prob = dropout_prob
        # 初始化模型的初始化范围
        self.initializer_range = initializer_range
        # 初始化模型的层归一化epsilon值
        self.layer_norm_eps = layer_norm_eps
        # 初始化输入图像的大小
        self.image_size = image_size
        # 初始化预训练图像的大小
        self.pretrain_image_size = pretrain_image_size
        # 初始化图像的patch大小
        self.patch_size = patch_size
        # 初始化图像的通道数量
        self.num_channels = num_channels
        # 初始化是否使用qkv偏置
        self.qkv_bias = qkv_bias
        # 初始化drop path的概率
        self.drop_path_rate = drop_path_rate
        # 初始化窗口块的索引
        self.window_block_indices = window_block_indices
        # 初始化残差块的索引
        self.residual_block_indices = residual_block_indices
        # 初始化是否使用绝对位置嵌入
        self.use_absolute_position_embeddings = use_absolute_position_embeddings
        # 初始化是否使用相对位置嵌入
        self.use_relative_position_embeddings = use_relative_position_embeddings
        # 初始化窗口大小
        self.window_size = window_size

        # 初始化阶段名称列表，包括"stem"和"stage1"到"stageN"
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, self.num_hidden_layers + 1)]
        # 获取对齐的输出特征和输出索引
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )
```