# `.\models\vivit\configuration_vivit.py`

```
# coding=utf-8
# 定义模块的版权信息和编码格式

# 导入必要的模块和类
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义预训练模型配置文件的映射字典，将模型名称映射到其配置文件的 URL
VIVIT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/vivit-b-16x2-kinetics400": (
        "https://huggingface.co/google/vivit-b-16x2-kinetics400/resolve/main/config.json"
    ),
    # 可在此处查看所有 ViViT 模型：https://huggingface.co/models?filter=vivit
}


class VivitConfig(PretrainedConfig):
    r"""
    这是用于存储 [`VivitModel`] 配置的配置类。根据指定的参数实例化配置对象，定义模型架构。
    使用默认参数实例化配置对象将产生类似于 ViViT [google/vivit-b-16x2-kinetics400]
    (https://huggingface.co/google/vivit-b-16x2-kinetics400) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。
    """
    pass
    # 定义模型类型为 "vivit"
    model_type = "vivit"
    
    # 初始化函数，设置模型的各项配置参数
    def __init__(
        self,
        image_size=224,  # 图像尺寸，默认为 224
        num_frames=32,  # 每个视频的帧数，默认为 32
        tubelet_size=[2, 16, 16],  # 每个 tubelet 的尺寸，默认为 [2, 16, 16]
        num_channels=3,  # 输入通道数，默认为 3
        hidden_size=768,  # 编码器层和池化层的维度，默认为 768
        num_hidden_layers=12,  # Transformer 编码器中的隐藏层层数，默认为 12
        num_attention_heads=12,  # 每个注意力层中的注意力头数，默认为 12
        intermediate_size=3072,  # Transformer 编码器中“中间”（即前馈）层的维度，默认为 3072
        hidden_act="gelu_fast",  # 编码器和池化器中的非线性激活函数，默认为 "gelu_fast"
        hidden_dropout_prob=0.0,  # 嵌入层、编码器和池化器中全连接层的 dropout 概率，默认为 0.0
        attention_probs_dropout_prob=0.0,  # 注意力概率的 dropout 比率，默认为 0.0
        initializer_range=0.02,  # 初始化所有权重矩阵的截断正态分布的标准差，默认为 0.02
        layer_norm_eps=1e-06,  # 层归一化层使用的 epsilon，默认为 1e-06
        qkv_bias=True,  # 是否为查询、键和值添加偏置，默认为 True
        **kwargs,  # 其他关键字参数
    ):
        ):
        # 初始化模型参数
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        # 初始化视频特征提取器的参数
        self.image_size = image_size  # 图像大小
        self.num_frames = num_frames  # 视频帧数
        self.tubelet_size = tubelet_size  # 视频片段大小
        self.num_channels = num_channels  # 视频通道数
        self.qkv_bias = qkv_bias  # 查询、键、值的偏置项

        # 调用父类的初始化方法
        super().__init__(**kwargs)
```