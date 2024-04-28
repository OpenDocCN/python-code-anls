# `.\transformers\models\swinv2\configuration_swinv2.py`

```
# 设置文件编码为 utf-8
# 版权声明
# 根据 Apache 许可证，遵循相关规定
# 获取 Apache 许可证的链接
# 在适用法律或书面同意的情况下，根据许可证，以“原样”分发软件，没有任何明示或暗示的保证或条件
# 参见许可证以获取特定语言的应用程序和限制

# 导入必要的模块
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# Swinv2 预训练配置文件映射
SWINV2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/swinv2-tiny-patch4-window8-256": (
        "https://huggingface.co/microsoft/swinv2-tiny-patch4-window8-256/resolve/main/config.json"
    ),
}

# Swinv2 配置类，继承了 BackboneConfigMixin 和 PretrainedConfig
class Swinv2Config(BackboneConfigMixin, PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`Swinv2Model`] 的配置。根据指定的参数来实例化 Swin Transformer v2 模型，
    定义模型架构。使用默认值初始化配置将生成类似于 Swin Transformer v2
    [microsoft/swinv2-tiny-patch4-window8-256](https://huggingface.co/microsoft/swinv2-tiny-patch4-window8-256)
    架构的配置。

    配置对象继承自[`PretrainedConfig`]，可用于控制模型的输出。阅读[`PretrainedConfig`]的文档获取更多信息。

    示例:

    ```python
    >>> from transformers import Swinv2Config, Swinv2Model

    >>> # 初始化一个 Swinv2 microsoft/swinv2-tiny-patch4-window8-256 风格的配置
    >>> configuration = Swinv2Config()

    >>> # 初始化一个带有随机权重的模型，使用 microsoft/swinv2-tiny-patch4-window8-256 风格的配置
    >>> model = Swinv2Model(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```"""
    
    # 模型类型为 "swinv2"
    model_type = "swinv2"

    # 属性映射
    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }
    # 初始化函数，用于初始化 Swin Transformer 模型的各项参数
    def __init__(
        self,
        # 输入图片的大小，默认为 224
        image_size=224,
        # 切分图片为小块的大小，默认为 4
        patch_size=4,
        # 输入图片的通道数，默认为 3（RGB 图片）
        num_channels=3,
        # 嵌入维度，默认为 96
        embed_dim=96,
        # 每个阶段的层数，默认为 [2, 2, 6, 2]
        depths=[2, 2, 6, 2],
        # 每个注意力头的数量，默认为 [3, 6, 12, 24]
        num_heads=[3, 6, 12, 24],
        # 滑动窗口大小，默认为 7
        window_size=7,
        # 预训练时的窗口大小，默认为 [0, 0, 0, 0]
        pretrained_window_sizes=[0, 0, 0, 0],
        # MLP 层中隐藏层的维度扩展倍率，默认为 4.0
        mlp_ratio=4.0,
        # 查询、键、值是否使用偏置，默认为 True
        qkv_bias=True,
        # 隐藏层的 dropout 概率，默认为 0.0
        hidden_dropout_prob=0.0,
        # 注意力矩阵中的 dropout 概率，默认为 0.0
        attention_probs_dropout_prob=0.0,
        # 路径丢弃率，默认为 0.1
        drop_path_rate=0.1,
        # 隐藏层激活函数，默认为 "gelu"
        hidden_act="gelu",
        # 是否使用绝对位置编码，默认为 False
        use_absolute_embeddings=False,
        # 初始化范围，默认为 0.02
        initializer_range=0.02,
        # LayerNorm 层的 epsilon 参数，默认为 1e-5
        layer_norm_eps=1e-5,
        # 编码器的步长，默认为 32
        encoder_stride=32,
        # 输出特征的维度，默认为 None
        out_features=None,
        # 输出特征的索引，默认为 None
        out_indices=None,
        **kwargs,
    ):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 设置各项参数
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_layers = len(depths)
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
        # 生成各个阶段的名称，包括 "stem" 和 "stage1" 到 "stageN"
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(depths) + 1)]
        # 获取对齐的输出特征维度和输出特征索引
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )
        # 设置 hidden_size 属性，表示模型最后一个阶段后的通道维度
        self.hidden_size = int(embed_dim * 2 ** (len(depths) - 1))
```