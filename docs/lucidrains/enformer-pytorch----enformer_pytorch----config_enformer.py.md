# `.\lucidrains\enformer-pytorch\enformer_pytorch\config_enformer.py`

```py
# 导入预训练配置类 PretrainedConfig 从 transformers 模块
from transformers import PretrainedConfig

# 创建 EnformerConfig 类，继承自 PretrainedConfig 类
class EnformerConfig(PretrainedConfig):
    # 模型类型为 "enformer"
    model_type = "enformer"

    # 初始化函数，接受多个参数
    def __init__(
        self,
        dim = 1536,  # 维度为 1536
        depth = 11,  # 深度为 11
        heads = 8,   # 头数为 8
        output_heads = dict(human = 5313, mouse= 1643),  # 输出头数为人类 5313，老鼠 1643
        target_length = 896,  # 目标长度为 896
        attn_dim_key = 64,    # 注意力维度为 64
        dropout_rate = 0.4,   # 丢弃率为 0.4
        attn_dropout = 0.05,  # 注意力丢弃率为 0.05
        pos_dropout = 0.01,   # 位置丢弃率为 0.01
        use_checkpointing = False,  # 是否使用检查点为 False
        use_convnext = False,       # 是否使用卷积为 False
        num_downsamples = 7,        # 下采样次数为 7，默认 Enformer 下采样 2 ** 7 == 128 倍，可以更改以获得更高分辨率
        dim_divisible_by = 128,     # 维度可被 128 整除
        use_tf_gamma = False,       # 是否使用 TensorFlow Gamma 为 False
        **kwargs,  # 其他关键字参数
    ):
        # 初始化各个参数
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.output_heads = output_heads
        self.target_length = target_length
        self.attn_dim_key = attn_dim_key
        self.dropout_rate = dropout_rate
        self.attn_dropout = attn_dropout
        self.pos_dropout = pos_dropout
        self.use_checkpointing = use_checkpointing
        self.num_downsamples = num_downsamples
        self.dim_divisible_by = dim_divisible_by
        self.use_tf_gamma = use_tf_gamma

        # 调用父类的初始化函数
        super().__init__(**kwargs)
```