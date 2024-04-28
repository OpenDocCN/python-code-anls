# `.\transformers\models\audio_spectrogram_transformer\configuration_audio_spectrogram_transformer.py`

```py
# 设置编码格式为 UTF-8
# 版权声明，声明了 Google AI 和 HuggingFace Inc. 团队对该代码的版权
# 该代码遵循 Apache 2.0 许可证，可以在符合许可证的条件下使用
# 可以在上述链接获取完整的许可证文本
# 除非适用法律要求或书面同意，否则按"原样"提供本软件，不提供任何明示或暗示的保证或条件
# 请参阅许可证了解更多信息
# 导入 AST 模型的配置所需的工具和库
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# AST 模型的预训练配置存档映射，将预训练模型名称映射到其对应的配置文件 URL
AUDIO_SPECTROGRAM_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "MIT/ast-finetuned-audioset-10-10-0.4593": (
        "https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593/resolve/main/config.json"
    ),
}

# ASTConfig 类，用于存储 AST 模型的配置信息
class ASTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ASTModel`]. It is used to instantiate an AST
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the AST
    [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
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
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        frequency_stride (`int`, *optional*, defaults to 10):
            Frequency stride to use when patchifying the spectrograms.
        time_stride (`int`, *optional*, defaults to 10):
            Temporal stride to use when patchifying the spectrograms.
        max_length (`int`, *optional*, defaults to 1024):
            Temporal dimension of the spectrograms.
        num_mel_bins (`int`, *optional*, defaults to 128):
            Frequency dimension of the spectrograms (number of Mel-frequency bins).

    Example:

    ```python
    >>> from transformers import ASTConfig, ASTModel

    >>> # Initializing a AST MIT/ast-finetuned-audioset-10-10-0.4593 style configuration
    >>> configuration = ASTConfig()

    >>> # Initializing a model (with random weights) from the MIT/ast-finetuned-audioset-10-10-0.4593 style configuration
    >>> model = ASTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```py"""

    # 设置模型类型为 "audio-spectrogram-transformer"
    model_type = "audio-spectrogram-transformer"
    # 初始化函数，设置模型的各项参数
    def __init__(
        self,
        hidden_size=768,  # 隐藏层大小，默认为768
        num_hidden_layers=12,  # 隐藏层数，默认为12
        num_attention_heads=12,  # 注意力头数，默认为12
        intermediate_size=3072,  # 中间层大小，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为gelu
        hidden_dropout_prob=0.0,  # 隐藏层丢弃概率，默认为0.0
        attention_probs_dropout_prob=0.0,  # 注意力机制丢弃概率，默认为0.0
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-12,  # 层归一化的epsilon，默认为1e-12
        patch_size=16,  # 图片切片大小，默认为16
        qkv_bias=True,  # 是否包含Q、K、V的偏置，默认为True
        frequency_stride=10,  # 频率方向的步长，默认为10
        time_stride=10,  # 时间方向的步长，默认为10
        max_length=1024,  # 最大序列长度，默认为1024
        num_mel_bins=128,  # 梅尔频谱的频道数，默认为128
        **kwargs,
    ):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 设置模型的各项参数
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.patch_size = patch_size
        self.qkv_bias = qkv_bias
        self.frequency_stride = frequency_stride
        self.time_stride = time_stride
        self.max_length = max_length
        self.num_mel_bins = num_mel_bins
```