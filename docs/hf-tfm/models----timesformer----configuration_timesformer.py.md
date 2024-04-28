# `.\transformers\models\timesformer\configuration_timesformer.py`

```
# 设置编码为utf-8
# 版权声明
# 根据Apache License, Version 2.0许可，可以使用此文件
# 请遵守许可的规定
# 可以从以下网址获得许可的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则不得使用此文件
# 根据"AS IS"基础分发软件，并没有任何种类的保证或条件，无论是明示的还是默示的
# 请查阅许可协议以获取更多关于特定语言控制输出和限制的信息
"""TimeSformer模型配置"""

# 从导入的包中引入预训练配置和日志功能
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 从日志中获取logger
logger = logging.get_logger(__name__)

# 预训练配置映射到配置档案
TIMESFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/timesformer": "https://huggingface.co/facebook/timesformer/resolve/main/config.json",
}

# Timesformer配置类，用于存储TimesformerModel的配置
# 用于根据指定参数实例化TimeSformer模型，定义模型体系结构
# 使用默认参数实例化配置将产生类似于TimeSformer的配置
# [facebook/timesformer-base-finetuned-k600](https://huggingface.co/facebook/timesformer-base-finetuned-k600) 架构
# 配置对象继承自PretrainedConfig，可用于控制模型输出。阅读PretrainedConfig的文档以获取更多信息。
    Args:
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        num_frames (`int`, *optional*, defaults to 8):
            The number of frames in each video.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        attention_type (`str`, *optional*, defaults to `"divided_space_time"`):
            The attention type to use. Must be one of `"divided_space_time"`, `"space_only"`, `"joint_space_time"`.
        drop_path_rate (`float`, *optional*, defaults to 0):
            The dropout ratio for stochastic depth.

    Example:

    ```python
    >>> from transformers import TimesformerConfig, TimesformerModel

    >>> # Initializing a TimeSformer timesformer-base style configuration
    >>> configuration = TimesformerConfig()

    >>> # Initializing a model from the configuration
    >>> model = TimesformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    # 设定模型类型为 "timesformer"
    model_type = "timesformer"
    # 初始化函数，设置模型的各种参数
    def __init__(
        self,
        image_size=224,  # 图像大小，默认为224像素
        patch_size=16,  # 分块大小，默认为16像素
        num_channels=3,  # 图像通道数，默认为3（RGB）
        num_frames=8,  # 帧数，默认为8帧
        hidden_size=768,  # 隐藏层大小，默认为768
        num_hidden_layers=12,  # 隐藏层层数，默认为12层
        num_attention_heads=12,  # 注意力头数，默认为12个
        intermediate_size=3072,  # 中间层大小，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为gelu
        hidden_dropout_prob=0.0,  # 隐藏层丢弃率，默认为0
        attention_probs_dropout_prob=0.0,  # 注意力机制丢弃率，默认为0
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-6,  # 层归一化的 epsilon 值，默认为1e-6
        qkv_bias=True,  # 是否在 QKV 线性层添加偏置，默认为True
        attention_type="divided_space_time",  # 注意力类型，默认为"divided_space_time"
        drop_path_rate=0,  # 路径丢弃率，默认为0
        **kwargs,
    ):
        
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 设置模型的各个参数值
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_frames = num_frames

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias

        self.attention_type = attention_type
        self.drop_path_rate = drop_path_rate
```