# `.\models\persimmon\configuration_persimmon.py`

```
# coding=utf-8
# 代码文件的版权信息和许可证声明

""" Persimmon model configuration"""
# 模型配置文件的简短描述

from ...configuration_utils import PretrainedConfig
from ...utils import logging
# 导入必要的模块和函数

logger = logging.get_logger(__name__)
# 获取与当前模块相关的日志记录器

PERSIMMON_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "adept/persimmon-8b-base": "https://huggingface.co/adept/persimmon-8b-base/resolve/main/config.json",
}
# 定义预训练模型的配置映射表，将模型名称映射到其配置文件的下载链接

class PersimmonConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PersimmonModel`]. It is used to instantiate an
    Persimmon model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the
    [adept/persimmon-8b-base](https://huggingface.co/adept/persimmon-8b-base).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    ```python
    >>> from transformers import PersimmonModel, PersimmonConfig

    >>> # Initializing a Persimmon persimmon-7b style configuration
    >>> configuration = PersimmonConfig()
    ```
    """
    # PersimmonConfig 类的说明文档，描述如何使用该类配置 Persimmon 模型

    model_type = "persimmon"
    # 模型类型为 "persimmon"

    keys_to_ignore_at_inference = ["past_key_values"]
    # 推断过程中忽略的键列表，这里包含 "past_key_values"

    def __init__(
        self,
        vocab_size=262144,
        hidden_size=4096,
        intermediate_size=16384,
        num_hidden_layers=36,
        num_attention_heads=64,
        hidden_act="relu2",
        max_position_embeddings=16384,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=25000.0,
        rope_scaling=None,
        qk_layernorm=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        partial_rotary_factor=0.5,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        # PersimmonConfig 的初始化函数，用于设置模型的各项配置参数
        pass
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.qk_layernorm = qk_layernorm
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.partial_rotary_factor = partial_rotary_factor
        self._rope_scaling_validation()

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


# 构造函数，初始化模型配置参数和调用验证函数
def __init__(
    vocab_size,  # 词汇表大小
    max_position_embeddings,  # 最大位置编码长度
    hidden_size,  # 隐藏层大小
    intermediate_size,  # 中间层大小
    num_hidden_layers,  # 隐藏层层数
    num_attention_heads,  # 注意力头的数量
    hidden_act,  # 隐藏层激活函数
    initializer_range,  # 参数初始化范围
    layer_norm_eps,  # 层归一化 epsilon 参数
    use_cache,  # 是否使用缓存
    rope_theta,  # 绳子模型 theta 参数
    rope_scaling,  # 绳子模型缩放参数
    qk_layernorm,  # QK 归一化参数
    hidden_dropout,  # 隐藏层 dropout 概率
    attention_dropout,  # 注意力机制 dropout 概率
    partial_rotary_factor,  # 部分旋转因子
    pad_token_id=None,  # 填充 token ID
    bos_token_id=None,  # 开始 token ID
    eos_token_id=None,  # 结束 token ID
    tie_word_embeddings=False,  # 是否共享词嵌入
    **kwargs,  # 其他参数
):
    # 初始化模型配置参数
    self.vocab_size = vocab_size
    self.max_position_embeddings = max_position_embeddings
    self.hidden_size = hidden_size
    self.intermediate_size = intermediate_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.initializer_range = initializer_range
    self.layer_norm_eps = layer_norm_eps
    self.use_cache = use_cache
    self.rope_theta = rope_theta
    self.rope_scaling = rope_scaling
    self.qk_layernorm = qk_layernorm
    self.hidden_dropout = hidden_dropout
    self.attention_dropout = attention_dropout
    self.partial_rotary_factor = partial_rotary_factor
    # 调用私有方法验证绳子模型缩放参数的有效性
    self._rope_scaling_validation()

    # 调用父类的初始化方法，传递必要的参数和其他关键字参数
    super().__init__(
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        tie_word_embeddings=tie_word_embeddings,
        **kwargs,
    )
```