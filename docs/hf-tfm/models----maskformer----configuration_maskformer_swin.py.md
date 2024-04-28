# `.\transformers\models\maskformer\configuration_maskformer_swin.py`

```
# 定义了一个编码声明，指定文件编码为 UTF-8

# 引入必要的模块和类
# configuration_utils 模块用于配置相关的实用函数
# logging 模块用于记录日志信息
# BackboneConfigMixin 用于处理与骨干网络相关的配置
# get_aligned_output_features_output_indices 函数用于获取对齐的输出特征和输出索引
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义 MaskFormerSwinConfig 类，用于存储 MaskFormerSwinModel 的配置
class MaskFormerSwinConfig(BackboneConfigMixin, PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MaskFormerSwinModel`]. It is used to instantiate
    a Donut model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Swin
    [microsoft/swin-tiny-patch4-window7-224](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```python
    >>> from transformers import MaskFormerSwinConfig, MaskFormerSwinModel

    >>> # Initializing a microsoft/swin-tiny-patch4-window7-224 style configuration
    >>> configuration = MaskFormerSwinConfig()

    >>> # Initializing a model (with random weights) from the microsoft/swin-tiny-patch4-window7-224 style configuration
    >>> model = MaskFormerSwinModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

    model_type = "maskformer-swin"

    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    # 初始化函数，用于设置模型的各种参数
    def __init__(
        self,
        image_size=224,  # 图像大小，默认为 224
        patch_size=4,  # 补丁大小，默认为 4
        num_channels=3,  # 图像通道数，默认为 3
        embed_dim=96,  # 嵌入维度，默认为 96
        depths=[2, 2, 6, 2],  # 不同阶段的深度列表，默认为 [2, 2, 6, 2]
        num_heads=[3, 6, 12, 24],  # 不同阶段的注意力头数列表，默认为 [3, 6, 12, 24]
        window_size=7,  # 窗口大小，默认为 7
        mlp_ratio=4.0,  # MLP 层扩展倍率，默认为 4.0
        qkv_bias=True,  # 是否在 QKV 线性映射中使用偏置，默认为 True
        hidden_dropout_prob=0.0,  # 隐藏层的 dropout 概率，默认为 0.0
        attention_probs_dropout_prob=0.0,  # 注意力概率的 dropout 概率，默认为 0.0
        drop_path_rate=0.1,  # drop path 的概率，默认为 0.1
        hidden_act="gelu",  # 隐藏层激活函数，默认为 gelu
        use_absolute_embeddings=False,  # 是否使用绝对位置嵌入，默认为 False
        initializer_range=0.02,  # 初始化范围，默认为 0.02
        layer_norm_eps=1e-5,  # 层归一化的 epsilon，默认为 1e-5
        out_features=None,  # 输出特征，默认为 None
        out_indices=None,  # 输出索引，默认为 None
        **kwargs,  # 其他关键字参数
    # 调用父类的初始化方法，传入任意关键字参数
    ):
        super().__init__(**kwargs)

        # 设置图像大小
        self.image_size = image_size
        # 设置补丁大小
        self.patch_size = patch_size
        # 设置通道数
        self.num_channels = num_channels
        # 设置嵌入维度
        self.embed_dim = embed_dim
        # 设置每个阶段的深度列表
        self.depths = depths
        # 计算阶段的数量
        self.num_layers = len(depths)
        # 设置注意力头的数量
        self.num_heads = num_heads
        # 设置窗口大小
        self.window_size = window_size
        # 设置多层感知机的比例
        self.mlp_ratio = mlp_ratio
        # 设置查询、键、值是否包含偏置
        self.qkv_bias = qkv_bias
        # 设置隐藏层的丢弃率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 设置注意力概率的丢弃率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 设置路径丢弃率
        self.drop_path_rate = drop_path_rate
        # 设置隐藏层激活函数
        self.hidden_act = hidden_act
        # 设置是否使用绝对位置嵌入
        self.use_absolute_embeddings = use_absolute_embeddings
        # 设置层归一化的 epsilon
        self.layer_norm_eps = layer_norm_eps
        # 设置初始化范围
        self.initializer_range = initializer_range
        # 设置隐藏大小属性，以使 Swin 与 VisionEncoderDecoderModel 兼容
        # 这指示模型最后一个阶段之后的通道维度
        self.hidden_size = int(embed_dim * 2 ** (len(depths) - 1))
        # 设置阶段名称列表，包括 "stem" 和从 1 到深度数量的阶段索引
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(depths) + 1)]
        # 获取对齐的输出特征和输出索引，以便使其与 VisionEncoderDecoderModel 兼容
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )
```