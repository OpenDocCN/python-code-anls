# `.\transformers\models\pegasus\configuration_pegasus.py`

```py
# 设置编码为 UTF-8
# 版权声明，版权归 Google 和 The HuggingFace Inc. 团队所有
# 根据 Apache License, Version 2.0 许可使用该文件
# 你只能依照此许可使用该文件，详细信息请查看许可证链接
# https://www.apache.org/licenses/LICENSE-2.0
# 除非法律要求或书面同意，否则根据许可分发的软件是基于"AS IS"的，没有任何明示或暗示的担保
# 请查看许可证以获取详细的授权和限制

""" PEGASUS model configuration"""

# 从父目录的 configuration_utils 导入 PretrainedConfig 类
# 从父目录的 utils 导入 logging 模块
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取 logger 对象，该对象用于记录日志
logger = logging.get_logger(__name__)

# PEGASUS 预训练配置文件映射
# 键为模型名称，值为配置文件地址
PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/pegasus-large": "https://huggingface.co/google/pegasus-large/resolve/main/config.json",
    # 查看所有 PEGASUS 模型请访问 https://huggingface.co/models?filter=pegasus
}

# PegasusConfig 类继承 PretrainedConfig 类
class PegasusConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PegasusModel`]. It is used to instantiate an
    PEGASUS model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the PEGASUS
    [google/pegasus-large](https://huggingface.co/google/pegasus-large) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```python
    >>> from transformers import PegasusConfig, PegasusModel

    >>> # Initializing a PEGASUS google/pegasus-large style configuration
    >>> configuration = PegasusConfig()

    >>> # Initializing a model (with random weights) from the google/pegasus-large style configuration
    >>> model = PegasusModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```py"""

    # 模型类型为 PEGASUS
    model_type = "pegasus"
    # 推断时要忽略的键
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射，将指定键映射到其他键
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}
    # 这是一个 Transformer 模型的配置类，包含了模型的各种超参数
    def __init__(
        self,
        # 词汇表大小
        vocab_size=50265,
        # 最大位置编码长度
        max_position_embeddings=1024,
        # 编码器层数
        encoder_layers=12,
        # 编码器前馈网络维度
        encoder_ffn_dim=4096,
        # 编码器注意力头数
        encoder_attention_heads=16,
        # 解码器层数
        decoder_layers=12,
        # 解码器前馈网络维度
        decoder_ffn_dim=4096,
        # 解码器注意力头数
        decoder_attention_heads=16,
        # 编码器 layer dropout 比例
        encoder_layerdrop=0.0,
        # 解码器 layer dropout 比例
        decoder_layerdrop=0.0,
        # 是否使用缓存
        use_cache=True,
        # 是否为编码器-解码器模型
        is_encoder_decoder=True,
        # 激活函数类型
        activation_function="gelu",
        # 模型维度
        d_model=1024,
        # 常规 dropout 比例
        dropout=0.1,
        # 注意力 dropout 比例
        attention_dropout=0.0,
        # 激活函数 dropout 比例
        activation_dropout=0.0,
        # 参数初始化标准差
        init_std=0.02,
        # 解码器起始 token id
        decoder_start_token_id=0,
        # 是否缩放词嵌入
        scale_embedding=False,
        # pad token id
        pad_token_id=0,
        # 结束 token id
        eos_token_id=1,
        # 强制结束 token id
        forced_eos_token_id=1,
        **kwargs,
    ):
        # 设置各种超参数
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding
        
        # 调用父类的初始化方法
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )
    
    # 获取编码器注意力头数
    @property
    def num_attention_heads(self) -> int:
        return self.encoder_attention_heads
    
    # 获取模型隐藏层维度
    @property
    def hidden_size(self) -> int:
        return self.d_model
```