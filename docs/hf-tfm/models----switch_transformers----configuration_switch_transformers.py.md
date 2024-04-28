# `.\transformers\models\switch_transformers\configuration_switch_transformers.py`

```
# 设置文件编码格式为utf-8
# 版权声明
# 根据Apache License, Version 2.0，如未遵守协议，不得使用此文件
# 可在http://www.apache.org/licenses/LICENSE-2.0获取协议副本
# 根据适用法律或书面同意，软件分发基于“原样”基础，无论明示或暗示，均无任何保证或条件
# 请参阅特定语言下的许可证以获取权限和限制
# Switch Transformers 模型配置信息
# 从公共库导入预先配置的类
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练配置存档映射
SWITCH_TRANSFORMERS_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/switch-base-8": "https://huggingface.co/google/switch-base-8/blob/main/config.json",
}

# SwitchTransformers配置类
class SwitchTransformersConfig(PretrainedConfig):
    r"""
    这是用于存储 [`SwitchTransformersModel`] 配置的配置类。它用于根据指定参数实例化SwitchTransformers模型，
    定义模型体系结构。使用默认值实例化配置将产生与SwitchTransformers 
    [google/switch-base-8](https://huggingface.co/google/switch-base-8)结构类似的配置。

    配置对象继承自 [`PretrainedConfig`] 可用于控制模型输出。阅读来自 [`PretrainedConfig`] 的文档以获取更多信息。

    """

    # 模型类型为 "switch_transformers"
    model_type = "switch_transformers"
    # 推断时忽略的键（past_key_values）
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射
    attribute_map = {"hidden_size": "d_model", "num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}

    # 初始化函数
    def __init__(
        self,
        vocab_size=32128,
        d_model=768,
        d_kv=64,
        d_ff=2048,
        expert_capacity=64,
        num_layers=12,
        num_sparse_encoder_layers=3,
        num_decoder_layers=12,
        num_sparse_decoder_layers=3,
        num_heads=12,
        num_experts=8,
        router_bias=False,
        router_jitter_noise=0.01,
        router_dtype="float32",
        router_ignore_padding_tokens=False,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        router_z_loss_coef=0.001,
        router_aux_loss_coef=0.001,
        initializer_factor=1.0,
        dense_act_fn="relu",
        is_encoder_decoder=True,
        add_router_probs=False,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        **kwargs,
        ):
        # 初始化模型的参数：词汇表大小、模型维度、键值对维度、前馈网络维度
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff

        # 编码器层数、解码器层数、稀疏编码器层数、稀疏解码器层数
        self.num_sparse_encoder_layers = num_sparse_encoder_layers
        self.num_layers = num_layers
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else self.num_layers
        )  # 默认与编码器对称
        self.num_sparse_decoder_layers = num_sparse_decoder_layers

        # 计算每几层编码器设置一个稀疏层
        if self.num_sparse_encoder_layers > 0:
            self.encoder_sparse_step = self.num_layers // self.num_sparse_encoder_layers
        else:
            self.encoder_sparse_step = self.num_layers  # HACK: 这将创建0个稀疏层

        # 计算每几层解码器设置一个稀疏层
        if self.num_sparse_decoder_layers > 0:
            self.decoder_sparse_step = self.num_decoder_layers // self.num_sparse_decoder_layers
        else:
            self.decoder_sparse_step = self.num_decoder_layers  # HACK: 这将创建0个稀疏层

        # 头数、专家数、专家容量、路由器偏差、路由器抖动噪声、路由器数据类型
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.router_bias = router_bias
        self.router_jitter_noise = router_jitter_noise
        if router_dtype not in ["float32", "float16", "bfloat16"]:
            raise ValueError(f"`router_dtype` must be one of 'float32', 'float16' or 'bfloat16', got {router_dtype}")
        self.router_dtype = router_dtype

        # 是否忽略填充令牌、相对注意力桶数、相对注意力最大距离
        self.router_ignore_padding_tokens = router_ignore_padding_tokens
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance

        # 丢弃率、层规范化epsilon、初始化因子、使用缓存、添加路由器概率
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.use_cache = use_cache
        self.add_router_probs = add_router_probs

        # 路由器z损失系数、路由器辅助损失系数、稠密激活函数
        self.router_z_loss_coef = router_z_loss_coef
        self.router_aux_loss_coef = router_aux_loss_coef
        self.dense_act_fn = dense_act_fn

        # 调用父类的初始化方法，传入填充令牌ID、结束令牌ID、是否是编码器解码器模型、其他参数
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )
```