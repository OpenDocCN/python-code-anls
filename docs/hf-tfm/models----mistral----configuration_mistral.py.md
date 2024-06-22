# `.\transformers\models\mistral\configuration_mistral.py`

```py
# 设置文件编码为utf-8
# 版权声明
# 根据Apache授权许可规定，使用者除非合规或通过许可，否则不得使用该文件
# 可以在下面网址获取授权许可的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，根据授权许可分发的软件是基于"AS IS"的基础，不带有任何保证或条件，无论是明示还是默示的
# 查看授权许可以获取特定语言控制和限制的权限
""" Mistral模型配置"""

# 导入所需的类和函数
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取logger对象
logger = logging.get_logger(__name__)

# 配置Mistral预训练配置存档映射
MISTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "mistralai/Mistral-7B-v0.1": "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/config.json",
    "mistralai/Mistral-7B-Instruct-v0.1": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/resolve/main/config.json",
}


class MistralConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储[`MistralModel`]的配置。根据指定的参数实例化Mistral模型，定义模型架构。使用默认值实例化配置将产生类似Mistral-7B-v0.1或Mistral-7B-Instruct-v0.1的配置。

    [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
    [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)

    配置对象继承自[`PretrainedConfig`]，可用于控制模型的输出。阅读[`PretrainedConfig`]的文档以获取更多信息。

    ```python
    >>> from transformers import MistralModel, MistralConfig

    >>> # 初始化一个Mistral 7B风格配置
    >>> configuration = MistralConfig()

    >>> # 使用Mistral 7B风格配置初始化一个模型
    >>> model = MistralModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py"""

    model_type = "mistral"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=4096 * 32,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        sliding_window=4096,
        attention_dropout=0.0,
        **kwargs,
        ):
        # 初始化器设置词汇表大小
        self.vocab_size = vocab_size
        # 初始化器设置最大位置嵌入
        self.max_position_embeddings = max_position_embeddings
        # 初始化器设置隐藏层大小
        self.hidden_size = hidden_size
        # 初始化器设置中间层大小
        self.intermediate_size = intermediate_size
        # 初始化器设置隐藏层数量
        self.num_hidden_layers = num_hidden_layers
        # 初始化器设置注意力头的数量
        self.num_attention_heads = num_attention_heads
        # 初始化器设置滑动窗口
        self.sliding_window = sliding_window

        # 为了向后兼容性
        # 如果 num_key_value_heads 为空，则设置为 num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        # 初始化器设置关键值头的数量
        self.num_key_value_heads = num_key_value_heads
        # 初始化器设置隐藏层激活函数
        self.hidden_act = hidden_act
        # 初始化器设置初始化器范围
        self.initializer_range = initializer_range
        # 初始化器设置 RMS 规范 Epsilon
        self.rms_norm_eps = rms_norm_eps
        # 初始化器设置使用缓存
        self.use_cache = use_cache
        # 初始化器设置绳（Rope）Theta
        self.rope_theta = rope_theta
        # 初始化器设置注意力丢失
        self.attention_dropout = attention_dropout

        # 调用父类的初始化函数，并设置参数
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
```