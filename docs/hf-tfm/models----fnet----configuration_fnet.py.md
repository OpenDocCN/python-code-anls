# `.\models\fnet\configuration_fnet.py`

```py
# 设置编码格式为 utf-8
# 版权声明
# 版权所属：2021年 Google AI 和 HuggingFace Inc.团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）授权;
# 你不得使用本文件，除非符合许可证的规定。
# 你可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面同意，否则根据许可证分发的软件属于“按原样”基础分发，
# 没有任何明示或暗示的保证或条件。
# 有关特定语言控制模型输出的更多信息，请阅读[`PretrainedConfig`]的文档。

# 导入依赖库
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 预训练配置对应的存档映射
FNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/fnet-base": "https://huggingface.co/google/fnet-base/resolve/main/config.json",
    "google/fnet-large": "https://huggingface.co/google/fnet-large/resolve/main/config.json",
    # 查看所有 FNet 模型 https://huggingface.co/models?filter=fnet
}

# FNet 配置类，用于存储 FNetModel 的配置，根据指定的参数实例化 FNet 模型，定义模型架构
class FNetConfig(PretrainedConfig):
    r"""
    这是用于存储 [`FNetModel`] 的配置类。它用于根据指定的参数实例化 FNet 模型，定义模型架构。
    使用默认值实例化配置将生成类似于 FNet [google/fnet-base](https://huggingface.co/google/fnet-base) 架构的配置。
    
    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读[`PretrainedConfig`]中的文档获取更多信息。
    ```
    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the FNet model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`FNetModel`] or [`TFFNetModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_new"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 4):
            The vocabulary size of the `token_type_ids` passed when calling [`FNetModel`] or [`TFFNetModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_tpu_fourier_optimizations (`bool`, *optional*, defaults to `False`):
            Determines whether to use TPU optimized FFTs. If `True`, the model will favor axis-wise FFTs transforms.
            Set to `False` for GPU/CPU hardware, in which case n-dimensional FFTs are used.
        tpu_short_seq_length (`int`, *optional*, defaults to 512):
            The sequence length that is expected by the model when using TPUs. This will be used to initialize the DFT
            matrix only when *use_tpu_fourier_optimizations* is set to `True` and the input sequence is shorter than or
            equal to 4096 tokens.

    Example:

    ```py
    >>> from transformers import FNetConfig, FNetModel

    >>> # Initializing a FNet fnet-base style configuration
    >>> configuration = FNetConfig()

    >>> # Initializing a model (with random weights) from the fnet-base style configuration
    >>> model = FNetModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

    # 设置模型类型为 "fnet"
    model_type = "fnet"
    # 初始化函数，设置模型参数
    def __init__(
        self,
        vocab_size=32000,  # 词汇表大小，默认为32000
        hidden_size=768,  # 隐藏层大小，默认为768
        num_hidden_layers=12,  # 隐藏层数量，默认为12
        intermediate_size=3072,  # 中间层大小，默认为3072
        hidden_act="gelu_new",  # 激活函数，默认为gelu_new
        hidden_dropout_prob=0.1,  # 隐藏层dropout概率，默认为0.1
        max_position_embeddings=512,  # 最大位置编码，默认为512
        type_vocab_size=4,  # 类型词典大小，默认为4
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-12,  # Layer Norm的epsilon值，默认为1e-12
        use_tpu_fourier_optimizations=False,  # 是否使用TPU的傅里叶优化，默认为False
        tpu_short_seq_length=512,  # TPU短序列长度，默认为512
        pad_token_id=3,  # 填充标记的id，默认为3
        bos_token_id=1,  # 起始标记的id，默认为1
        eos_token_id=2,  # 结束标记的id，默认为2
        **kwargs,  # 其他参数
    ):
        # 调用父类的初始化函数，并设置相关变量
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
    
        # 设置模型参数
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.use_tpu_fourier_optimizations = use_tpu_fourier_optimizations
        self.tpu_short_seq_length = tpu_short_seq_length
    ```py  
```