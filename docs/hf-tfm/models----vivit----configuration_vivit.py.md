# `.\transformers\models\vivit\configuration_vivit.py`

```
# 设置编码格式为 utf-8

# 版权声明和许可信息

# 引入预训练配置和日志工具
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练配置与存档映射
VIVIT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/vivit-b-16x2-kinetics400": (
        "https://huggingface.co/google/vivit-b-16x2-kinetics400/resolve/main/config.json"
    ),
    # 查看所有 ViViT 模型 https://huggingface.co/models?filter=vivit
}

# ViViT 模型配置类，用于存储 ViViT 模型的配置，根据指定参数实例化 ViViT 模型
class VivitConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VivitModel`]. It is used to instantiate a ViViT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ViViT
    [google/vivit-b-16x2-kinetics400](https://huggingface.co/google/vivit-b-16x2-kinetics400) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # 定义一个类，用于ViViT模型的配置
    class VivitConfig:
    
        # 初始化函数，设置ViViT模型的配置参数
        def __init__(
            self,
            image_size=224,  # 图像尺寸，默认为224
            num_frames=32,  # 每个视频中的帧数，默认为32
            tubelet_size=[2, 16, 16],  # 每个tubelet的尺寸，默认为[2, 16, 16]
            num_channels=3,  # 输入通道数，默认为3
            hidden_size=768,  # 编码器层和pooler层的维度，默认为768
            num_hidden_layers=12,  # Transformer编码器中隐藏层的数量，默认为12
            num_attention_heads=12,  # Transformer编码器中每个注意力层的注意头数量，默认为12
            intermediate_size=3072,  # Transformer编码器中"中间"（即前馈）层的维度，默认为3072
            hidden_act="gelu_fast",  # 编码器和pooler中的非线性激活函数，默认为"gelu_fast"
            hidden_dropout_prob=0.0,  # 嵌入层、编码器和pooler中所有全连接层的dropout概率，默认为0.0
            attention_probs_dropout_prob=0.0,  # 注意力概率的dropout比例，默认为0.0
            initializer_range=0.02,  # 初始化所有权重矩阵的截断正态分布标准差，默认为0.02
            layer_norm_eps=1e-06,  # 层归一化层使用的epsilon值，默认为1e-06
            qkv_bias=True,  # 是否为查询、键和值添加偏置，默认为True
            **kwargs,  # 其他参数
    
        model_type = "vivit"  # 模型类型为"vivit"
    
        # 示例代码
        def example():
            from transformers import VivitConfig, VivitModel
    
            # 初始化一个ViViT google/vivit-b-16x2-kinetics400风格的配置
            configuration = VivitConfig()
    
            # 用google/vivit-b-16x2-kinetics400风格的配置初始化一个具有随机权重的模型
            model = VivitModel(configuration)
    
            # 访问模型配置
            configuration = model.config
        ):
        # 初始化 Transformer 模型的参数
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        # 设置图像处理的参数
        self.image_size = image_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias

        # 调用父类的初始化方法
        super().__init__(**kwargs)
```