# `.\transformers\models\mixtral\configuration_mixtral.py`

```
# 设置文件编码为utf-8
# 版权声明
# 根据Apache许可证进行许可
# 查看许可证信息：http://www.apache.org/licenses/LICENSE-2.0
# 软件按"AS IS"基础分发，不带任何明示或暗示的担保和条件
# 查看限制和特定语言的许可证
""" Mixtral 模型配置"""

# 导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 导入日志模块
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练配置文件映射
MIXTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "mistral-ai/Mixtral-8x7B": "https://huggingface.co/mistral-ai/Mixtral-8x7B/resolve/main/config.json",
}

# Mixtral 配置类继承自预训练配置类
class MixtralConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`MixtralModel`] 的配置。它被用于根据指定的参数实例化Mixtral模型，定义模型架构。
    使用默认值实例化配置将产生类似于Mixtral-7B-v0.1或Mixtral-7B-Instruct-v0.1的配置。

    [mixtralai/Mixtral-8x7B](https://huggingface.co/mixtralai/Mixtral-8x7B)
    [mixtralai/Mixtral-7B-Instruct-v0.1](https://huggingface.co/mixtralai/Mixtral-7B-Instruct-v0.1)

    配置对象继承自[`PretrainedConfig`]，可用于控制模型输出。阅读[`PretrainedConfig`]的文档以获取更多信息。

    ```python
    >>> from transformers import MixtralModel, MixtralConfig

    >>> # 初始化一个Mixtral 7B风格的配置
    >>> configuration = MixtralConfig()

    >>> # 使用Mixtral 7B风格的配置初始化一个模型
    >>> model = MixtralModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```"""

    # 模型类型为mixtral
    model_type = "mixtral"
    # 推理时要忽略的键
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
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=1e6,
        sliding_window=None,
        attention_dropout=0.0,
        num_experts_per_tok=2,
        num_local_experts=8,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        **kwargs,
        # 初始化Transformer模型的参数
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window

        # 为了向后兼容性
        # 如果未指定num_key_value_heads，则设为num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout

        # 设定Transformer模型的专家数目参数
        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef

        # 调用父类的构造函数，设定一些通用参数
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
```