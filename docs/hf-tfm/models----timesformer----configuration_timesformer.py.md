# `.\models\timesformer\configuration_timesformer.py`

```
# coding=utf-8
# 定义编码方式为 UTF-8

# 导入预训练配置类 PretrainedConfig 和日志工具 logging
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义预训练模型配置文件的映射字典，指定了模型名称和其对应的配置文件链接
TIMESFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/timesformer": "https://huggingface.co/facebook/timesformer/resolve/main/config.json",
}

# TimesformerConfig 类，继承自 PretrainedConfig 类，用于存储 TimeSformer 模型的配置信息
class TimesformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`TimesformerModel`]. It is used to instantiate a
    TimeSformer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the TimeSformer
    [facebook/timesformer-base-finetuned-k600](https://huggingface.co/facebook/timesformer-base-finetuned-k600)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # 设置默认参数：图像尺寸为224像素，每个补丁(patch)的尺寸为16像素，输入通道数为3
    # 每个视频包含8帧，编码器和池化器层的隐藏层大小为768维
    # Transformer编码器中隐藏层的数量为12，每个注意力层中的注意力头数为12
    # Transformer编码器中"中间"（即前馈）层的维度为3072
    # 编码器和池化器中的非线性激活函数为GELU
    # 嵌入层、编码器和池化器中所有全连接层的丢弃概率为0.0
    # 注意力概率的丢弃比例为0.0
    # 初始化所有权重矩阵的截断正态分布标准差为0.02
    # 层归一化层使用的epsilon为1e-06
    # 是否向查询、键和值添加偏置
    # 使用的注意力类型为"divided_space_time"
    # 随机深度的丢弃比率为0

    # 将模型类型设置为"timesformer"
    model_type = "timesformer"
    # 初始化函数，用于初始化一个自定义的神经网络模型
    def __init__(
        self,
        image_size=224,  # 图像输入大小，默认为224
        patch_size=16,  # 每个patch的大小，默认为16
        num_channels=3,  # 输入图像的通道数，默认为3（RGB）
        num_frames=8,  # 输入视频帧数，默认为8
        hidden_size=768,  # Transformer模型中隐藏层的大小，默认为768
        num_hidden_layers=12,  # Transformer模型中隐藏层的数量，默认为12
        num_attention_heads=12,  # Transformer模型中注意力头的数量，默认为12
        intermediate_size=3072,  # Transformer模型中Feedforward层的中间大小，默认为3072
        hidden_act="gelu",  # 隐藏层的激活函数，默认为GELU
        hidden_dropout_prob=0.0,  # 隐藏层的dropout概率，默认为0（无dropout）
        attention_probs_dropout_prob=0.0,  # 注意力层的dropout概率，默认为0（无dropout）
        initializer_range=0.02,  # 初始化权重的范围，默认为0.02
        layer_norm_eps=1e-6,  # LayerNorm层的epsilon值，默认为1e-6
        qkv_bias=True,  # 是否在QKV（查询、键、值）矩阵中使用偏置项，默认为True
        attention_type="divided_space_time",  # 注意力机制的类型，默认为“divided_space_time”
        drop_path_rate=0,  # DropPath层的drop率，默认为0（无drop）
        **kwargs,
    ):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 设置对象的属性值
        self.image_size = image_size  # 图像输入大小
        self.patch_size = patch_size  # 每个patch的大小
        self.num_channels = num_channels  # 输入图像的通道数
        self.num_frames = num_frames  # 输入视频帧数

        self.hidden_size = hidden_size  # Transformer模型中隐藏层的大小
        self.num_hidden_layers = num_hidden_layers  # Transformer模型中隐藏层的数量
        self.num_attention_heads = num_attention_heads  # Transformer模型中注意力头的数量
        self.intermediate_size = intermediate_size  # Transformer模型中Feedforward层的中间大小
        self.hidden_act = hidden_act  # 隐藏层的激活函数
        self.hidden_dropout_prob = hidden_dropout_prob  # 隐藏层的dropout概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob  # 注意力层的dropout概率
        self.initializer_range = initializer_range  # 初始化权重的范围
        self.layer_norm_eps = layer_norm_eps  # LayerNorm层的epsilon值
        self.qkv_bias = qkv_bias  # 是否在QKV（查询、键、值）矩阵中使用偏置项

        self.attention_type = attention_type  # 注意力机制的类型
        self.drop_path_rate = drop_path_rate  # DropPath层的drop率
```