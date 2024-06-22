# `.\transformers\models\qwen2\configuration_qwen2.py`

```py
# 设定编码格式为utf-8
# 版权声明及许可协议
# 输出日志工具
""" Qwen2 模型配置"""

# 从自定义的配置类继承
from ...configuration_utils import PretrainedConfig
# 导入日志记录工具
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# Qwen2 预训练配置文件映射
QWEN2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Qwen/Qwen2-7B-beta": "https://huggingface.co/Qwen/Qwen2-7B-beta/resolve/main/config.json",
}

# Qwen2 模型配置类
class Qwen2Config(PretrainedConfig):
    r"""
    这是用于存储 [`Qwen2Model`] 配置信息的配置类。根据指定的参数实例化 Qwen2 模型，定义模型架构。
    使用默认值实例化配置将产生类似于 Qwen2-7B-beta [Qwen/Qwen2-7B-beta](https://huggingface.co/Qwen/Qwen2-7B-beta) 的配置。

    配置对象继承自 [`PretrainedConfig`]，可以用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    ```python
    >>> from transformers import Qwen2Model, Qwen2Config

    >>> # 初始化一个 Qwen2 风格的配置
    >>> configuration = Qwen2Config()

    >>> # 从 Qwen2-7B 风格的配置初始化一个模型
    >>> model = Qwen2Model(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py"""

    model_type = "qwen2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        **kwargs,
        # 设置词汇表的大小
        self.vocab_size = vocab_size
        # 设置位置编码的最大长度
        self.max_position_embeddings = max_position_embeddings
        # 设置隐藏层的大小
        self.hidden_size = hidden_size
        # 设置中间层的大小
        self.intermediate_size = intermediate_size
        # 设置隐藏层的数量
        self.num_hidden_layers = num_hidden_layers
        # 设置注意力头的数量
        self.num_attention_heads = num_attention_heads
        # 设置是否使用滑动窗口
        self.use_sliding_window = use_sliding_window
        # 设置滑动窗口的大小
        self.sliding_window = sliding_window
        # 设置最大滑动窗口的层数
        self.max_window_layers = max_window_layers

        # 用于向后兼容
        # 如果num_key_value_heads为None，则将其设置为num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        # 设置关键值头的数量
        self.num_key_value_heads = num_key_value_heads
        # 设置隐藏层的激活函数
        self.hidden_act = hidden_act
        # 设置初始化范围
        self.initializer_range = initializer_range
        # 设置rms归一化的ε值
        self.rms_norm_eps = rms_norm_eps
        # 设置是否使用缓存
        self.use_cache = use_cache
        # 设置rope_theta值
        self.rope_theta = rope_theta
        # 设置注意力的dropout率
        self.attention_dropout = attention_dropout

        # 调用父类的初始化方法，设置词嵌入是否共享参数等
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
```