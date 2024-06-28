# `.\models\mixtral\configuration_mixtral.py`

```py
# 设置编码格式为 UTF-8
# 版权声明，包括公司和团队
# 在 Apache License 2.0 下授权使用该文件
# 可以在指定许可证下使用此文件，详见链接
# 如果不符合条件，则不能使用此文件
# 根据法律要求或书面同意，分发的软件以“原样”分发
# 没有任何明示或暗示的保证或条件
# 详见许可证以了解特定的语言权限

""" Mixtral model configuration"""

# 从 transformers 库中导入预训练配置类 PretrainedConfig
from ...configuration_utils import PretrainedConfig
# 导入 logging 模块
from ...utils import logging

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 预训练配置文件映射字典，指定 Mixtral 预训练模型和其配置文件的链接
MIXTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "mistral-ai/Mixtral-8x7B": "https://huggingface.co/mistral-ai/Mixtral-8x7B/resolve/main/config.json",
}

# MixtralConfig 类，继承自 PretrainedConfig 类
class MixtralConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MixtralModel`]. It is used to instantiate an
    Mixtral model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Mixtral-7B-v0.1 or Mixtral-7B-Instruct-v0.1.

    [mixtralai/Mixtral-8x7B](https://huggingface.co/mixtralai/Mixtral-8x7B)
    [mixtralai/Mixtral-7B-Instruct-v0.1](https://huggingface.co/mixtralai/Mixtral-7B-Instruct-v0.1)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    ```
    >>> from transformers import MixtralModel, MixtralConfig

    >>> # Initializing a Mixtral 7B style configuration
    >>> configuration = MixtralConfig()

    >>> # Initializing a model from the Mixtral 7B style configuration
    >>> model = MixtralModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # 模型类型为 mixtral
    model_type = "mixtral"
    # 推断时忽略的键
    keys_to_ignore_at_inference = ["past_key_values"]

    # 构造函数，定义 MixtralConfig 的各项配置参数
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
    ):
        # 调用父类 PretrainedConfig 的构造函数
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            sliding_window=sliding_window,
            attention_dropout=attention_dropout,
            num_experts_per_tok=num_experts_per_tok,
            num_local_experts=num_local_experts,
            output_router_logits=output_router_logits,
            router_aux_loss_coef=router_aux_loss_coef,
            **kwargs,
        )
        ):
            # 初始化模型参数：词汇表大小、最大位置嵌入、隐藏层大小、中间层大小、隐藏层数量、注意力头数量、滑动窗口大小
            self.vocab_size = vocab_size
            self.max_position_embeddings = max_position_embeddings
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.sliding_window = sliding_window

            # 为了向后兼容性
            # 如果未提供 num_key_value_heads，则将其设置为 num_attention_heads
            if num_key_value_heads is None:
                num_key_value_heads = num_attention_heads

            # 设置 key-value 头的数量
            self.num_key_value_heads = num_key_value_heads
            # 设置隐藏层激活函数
            self.hidden_act = hidden_act
            # 设置初始化范围
            self.initializer_range = initializer_range
            # RMS 归一化的 epsilon 值
            self.rms_norm_eps = rms_norm_eps
            # 是否使用缓存
            self.use_cache = use_cache
            # ROPE 损失函数参数
            self.rope_theta = rope_theta
            # 注意力机制的 dropout 概率
            self.attention_dropout = attention_dropout

            # 每个 token 的专家数量
            self.num_experts_per_tok = num_experts_per_tok
            # 本地专家的数量
            self.num_local_experts = num_local_experts
            # 是否输出路由器的 logits
            self.output_router_logits = output_router_logits
            # 路由器辅助损失系数
            self.router_aux_loss_coef = router_aux_loss_coef

            # 调用父类的初始化方法，设置模型的特殊标记 ID 和其他参数
            super().__init__(
                pad_token_id=pad_token_id,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                tie_word_embeddings=tie_word_embeddings,
                **kwargs,
            )
```