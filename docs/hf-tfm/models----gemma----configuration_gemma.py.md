# `.\models\gemma\configuration_gemma.py`

```py
# coding=utf-8
# 声明版权和许可信息

# 导入所需的模块和函数
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 初始化一个空字典，用于存储预训练配置的归档映射
GEMMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {}

# GemmaConfig 类，用于存储 GemmaModel 的配置信息，继承自 PretrainedConfig 类
class GemmaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GemmaModel`]. It is used to instantiate an Gemma
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Gemma-7B.

    e.g. [google/gemma-7b](https://huggingface.co/google/gemma-7b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    ```
    >>> from transformers import GemmaModel, GemmaConfig

    >>> # Initializing a Gemma gemma-7b style configuration
    >>> configuration = GemmaConfig()

    >>> # Initializing a model from the gemma-7b style configuration
    >>> model = GemmaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # 模型类型标识为 "gemma"
    model_type = "gemma"

    # 在推理过程中要忽略的键列表
    keys_to_ignore_at_inference = ["past_key_values"]

    # GemmaConfig 类的初始化方法
    def __init__(
        self,
        vocab_size=256000,
        hidden_size=3072,
        intermediate_size=24576,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=256,
        hidden_act="gelu_pytorch_tanh",
        hidden_activation=None,
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        # 调用父类 PretrainedConfig 的初始化方法
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            hidden_act=hidden_act,
            hidden_activation=hidden_activation,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.hidden_activation = hidden_activation
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout



        # 初始化模型参数
        self.vocab_size = vocab_size  # 词汇表大小
        self.max_position_embeddings = max_position_embeddings  # 最大位置嵌入大小
        self.hidden_size = hidden_size  # 隐藏层大小
        self.intermediate_size = intermediate_size  # 中间层大小
        self.num_hidden_layers = num_hidden_layers  # 隐藏层数量
        self.num_attention_heads = num_attention_heads  # 注意力头数量
        self.head_dim = head_dim  # 注意力头维度
        self.num_key_value_heads = num_key_value_heads  # 键值头数量
        self.hidden_act = hidden_act  # 隐藏层激活函数
        self.hidden_activation = hidden_activation  # 隐藏层激活函数（备用）
        self.initializer_range = initializer_range  # 初始化范围
        self.rms_norm_eps = rms_norm_eps  # RMS 归一化的 epsilon 值
        self.use_cache = use_cache  # 是否使用缓存
        self.rope_theta = rope_theta  # ROPE 参数
        self.attention_bias = attention_bias  # 注意力偏置
        self.attention_dropout = attention_dropout  # 注意力丢弃率



        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )



        # 调用父类初始化方法，传入特定参数和关键字参数
        super().__init__(
            pad_token_id=pad_token_id,  # 填充符号 ID
            bos_token_id=bos_token_id,  # 起始符号 ID
            eos_token_id=eos_token_id,  # 结束符号 ID
            tie_word_embeddings=tie_word_embeddings,  # 是否共享词嵌入
            **kwargs,  # 其他未命名参数
        )
```