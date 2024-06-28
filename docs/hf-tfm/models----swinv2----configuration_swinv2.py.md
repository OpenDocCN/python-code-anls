# `.\models\swinv2\configuration_swinv2.py`

```py
# 设置编码格式为 UTF-8

# 版权声明和许可证，声明代码版权归 HuggingFace Inc. 团队所有，遵循 Apache License 2.0 版本
# 只有在遵守许可证的情况下才能使用此文件。您可以在以下网址获取许可证的副本：
# http://www.apache.org/licenses/LICENSE-2.0

# 如果适用法律要求或书面同意，本软件按"原样"分发，不提供任何明示或暗示的担保或条件。
# 有关详细信息，请参阅许可证。

""" Swinv2 Transformer model configuration"""

# 导入必要的模块和类
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices

# 获取日志记录器
logger = logging.get_logger(__name__)

# Swinv2 模型预训练配置文件映射，指定模型的预训练配置文件位置
SWINV2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/swinv2-tiny-patch4-window8-256": (
        "https://huggingface.co/microsoft/swinv2-tiny-patch4-window8-256/resolve/main/config.json"
    ),
}

# Swinv2Config 类，用于存储 Swinv2 模型的配置信息
class Swinv2Config(BackboneConfigMixin, PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Swinv2Model`]. It is used to instantiate a Swin
    Transformer v2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Swin Transformer v2
    [microsoft/swinv2-tiny-patch4-window8-256](https://huggingface.co/microsoft/swinv2-tiny-patch4-window8-256)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```
    >>> from transformers import Swinv2Config, Swinv2Model

    >>> # Initializing a Swinv2 microsoft/swinv2-tiny-patch4-window8-256 style configuration
    >>> configuration = Swinv2Config()

    >>> # Initializing a model (with random weights) from the microsoft/swinv2-tiny-patch4-window8-256 style configuration
    >>> model = Swinv2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    
    """

    # 模型类型为 Swinv2
    model_type = "swinv2"

    # 属性映射表，将一些属性名映射为另一些属性名
    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }
    # 初始化函数，用于初始化一个Swing Transformer模型的参数
    def __init__(
        self,
        image_size=224,  # 图像尺寸，默认为224
        patch_size=4,  # 每个patch的大小，默认为4
        num_channels=3,  # 输入图像的通道数，默认为3（RGB图像）
        embed_dim=96,  # 嵌入维度，默认为96
        depths=[2, 2, 6, 2],  # 各个阶段的深度列表，默认为[2, 2, 6, 2]
        num_heads=[3, 6, 12, 24],  # 各个阶段的注意力头数列表，默认为[3, 6, 12, 24]
        window_size=7,  # 窗口大小，默认为7
        pretrained_window_sizes=[0, 0, 0, 0],  # 预训练窗口大小列表，默认为[0, 0, 0, 0]
        mlp_ratio=4.0,  # MLP放大比例，默认为4.0
        qkv_bias=True,  # 是否使用注意力的查询、键、值偏置，默认为True
        hidden_dropout_prob=0.0,  # 隐藏层dropout概率，默认为0.0（无dropout）
        attention_probs_dropout_prob=0.0,  # 注意力概率dropout概率，默认为0.0（无dropout）
        drop_path_rate=0.1,  # drop path的概率，默认为0.1
        hidden_act="gelu",  # 隐藏层激活函数，默认为gelu
        use_absolute_embeddings=False,  # 是否使用绝对位置嵌入，默认为False
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-5,  # LayerNorm的epsilon，默认为1e-5
        encoder_stride=32,  # 编码器步长，默认为32
        out_features=None,  # 输出特征列表，默认为None
        out_indices=None,  # 输出索引列表，默认为None
        **kwargs,  # 其他关键字参数
    ):
        super().__init__(**kwargs)
    
        # 设置各种参数到对象的属性中
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_layers = len(depths)  # 设置阶段的数量为depths列表的长度
        self.num_heads = num_heads
        self.window_size = window_size
        self.pretrained_window_sizes = pretrained_window_sizes
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.hidden_act = hidden_act
        self.use_absolute_embeddings = use_absolute_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.encoder_stride = encoder_stride
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(depths) + 1)]
        # 获取对齐的输出特征和输出索引，以便与VisionEncoderDecoderModel兼容
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )
        # 设置hidden_size属性，表示模型最后一个阶段之后的通道维度
        self.hidden_size = int(embed_dim * 2 ** (len(depths) - 1))
```