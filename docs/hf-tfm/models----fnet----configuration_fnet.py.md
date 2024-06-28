# `.\models\fnet\configuration_fnet.py`

```py
"""
FNet model configuration

This module defines the configuration for the FNet model, specifying how to instantiate
different variants of the model architecture based on provided arguments.

"""

# Import necessary modules from Hugging Face library
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# Get logger instance for logging messages related to this module
logger = logging.get_logger(__name__)

# Map of pretrained FNet model configurations with their respective URLs
FNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/fnet-base": "https://huggingface.co/google/fnet-base/resolve/main/config.json",
    "google/fnet-large": "https://huggingface.co/google/fnet-large/resolve/main/config.json",
    # Additional models can be found at the provided URL
}

class FNetConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FNetModel`]. It is used to instantiate an FNet
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the FNet
    [google/fnet-base](https://huggingface.co/google/fnet-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
"""
    # 模型类型设定为 "fnet"
    model_type = "fnet"
    # 初始化函数，用于创建一个新的对象实例
    def __init__(
        self,
        vocab_size=32000,  # 设置词汇表大小，默认为32000
        hidden_size=768,  # 设置隐藏层大小，默认为768
        num_hidden_layers=12,  # 设置隐藏层数，默认为12
        intermediate_size=3072,  # 设置中间层大小，默认为3072
        hidden_act="gelu_new",  # 设置隐藏层激活函数，默认为"gelu_new"
        hidden_dropout_prob=0.1,  # 设置隐藏层dropout概率，默认为0.1
        max_position_embeddings=512,  # 设置最大位置嵌入大小，默认为512
        type_vocab_size=4,  # 设置类型词汇表大小，默认为4
        initializer_range=0.02,  # 设置初始化范围，默认为0.02
        layer_norm_eps=1e-12,  # 设置层归一化的epsilon值，默认为1e-12
        use_tpu_fourier_optimizations=False,  # 是否使用TPU Fourier优化，默认为False
        tpu_short_seq_length=512,  # TPU短序列长度，默认为512
        pad_token_id=3,  # PAD标记的token id，默认为3
        bos_token_id=1,  # 开始序列标记的token id，默认为1
        eos_token_id=2,  # 结束序列标记的token id，默认为2
        **kwargs,
    ):
        # 调用父类的初始化方法，传入PAD、BOS、EOS标记的token id以及其他关键字参数
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
    
        # 初始化对象的各个属性，用传入的参数或者默认值
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
```