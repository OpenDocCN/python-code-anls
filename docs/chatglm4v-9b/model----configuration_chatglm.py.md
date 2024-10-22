# `.\chatglm4v-9b\configuration_chatglm.py`

```
# 从 transformers 库导入预训练配置类
from transformers import PretrainedConfig


# 定义 ChatGLMConfig 类，继承自 PretrainedConfig
class ChatGLMConfig(PretrainedConfig):
    # 设置模型类型为 "chatglm"
    model_type = "chatglm"

    # 初始化方法，设置模型的各种参数
    def __init__(
            # 定义模型层数，默认为 28
            num_layers=28,
            # 定义填充后的词汇表大小，默认为 65024
            padded_vocab_size=65024,
            # 定义隐藏层的大小，默认为 4096
            hidden_size=4096,
            # 定义前馈网络隐藏层的大小，默认为 13696
            ffn_hidden_size=13696,
            # 定义键值通道的数量，默认为 128
            kv_channels=128,
            # 定义注意力头的数量，默认为 32
            num_attention_heads=32,
            # 定义序列长度，默认为 2048
            seq_length=2048,
            # 定义隐藏层的 dropout 比例，默认为 0.0
            hidden_dropout=0.0,
            # 定义分类器的 dropout 比例，默认为 None
            classifier_dropout=None,
            # 定义注意力层的 dropout 比例，默认为 0.0
            attention_dropout=0.0,
            # 定义 layernorm 的 epsilon 值，默认为 1e-5
            layernorm_epsilon=1e-5,
            # 定义是否使用 rmsnorm，默认为 True
            rmsnorm=True,
            # 定义是否在 layernorm 后应用残差连接，默认为 False
            apply_residual_connection_post_layernorm=False,
            # 定义是否使用后层归一化，默认为 True
            post_layer_norm=True,
            # 定义是否添加线性偏置，默认为 False
            add_bias_linear=False,
            # 定义是否添加 QKV 偏置，默认为 False
            add_qkv_bias=False,
            # 定义是否进行偏置 dropout 融合，默认为 True
            bias_dropout_fusion=True,
            # 定义是否使用多查询注意力，默认为 False
            multi_query_attention=False,
            # 定义多查询组的数量，默认为 1
            multi_query_group_num=1,
            # 定义 ROPE 比例，默认为 1
            rope_ratio=1,
            # 定义是否应用查询-键层缩放，默认为 True
            apply_query_key_layer_scaling=True,
            # 定义是否在 FP32 中进行注意力 softmax，默认为 True
            attention_softmax_in_fp32=True,
            # 定义是否使用 FP32 残差连接，默认为 False
            fp32_residual_connection=False,
            # 定义前序列长度，默认为 None
            pre_seq_len=None,
            # 定义是否使用前缀投影，默认为 False
            prefix_projection=False,
            # 定义 BOI token 的 ID，默认为 None
            boi_token_id=None,
            # 定义 EOI token 的 ID，默认为 None
            eoi_token_id=None,
            # 其他参数，允许扩展
            **kwargs
    ):
        # 将 num_layers 参数赋值给实例属性
        self.num_layers = num_layers
        # 将词汇表大小赋值给实例属性
        self.vocab_size = padded_vocab_size
        # 将填充后的词汇表大小赋值给实例属性
        self.padded_vocab_size = padded_vocab_size
        # 将隐藏层大小赋值给实例属性
        self.hidden_size = hidden_size
        # 将前馈网络隐藏层大小赋值给实例属性
        self.ffn_hidden_size = ffn_hidden_size
        # 将键值通道数量赋值给实例属性
        self.kv_channels = kv_channels
        # 将注意力头数量赋值给实例属性
        self.num_attention_heads = num_attention_heads
        # 将序列长度赋值给实例属性
        self.seq_length = seq_length
        # 将隐藏层 dropout 赋值给实例属性
        self.hidden_dropout = hidden_dropout
        # 将分类器 dropout 赋值给实例属性
        self.classifier_dropout = classifier_dropout
        # 将注意力 dropout 赋值给实例属性
        self.attention_dropout = attention_dropout
        # 将 layernorm epsilon 赋值给实例属性
        self.layernorm_epsilon = layernorm_epsilon
        # 将 rmsnorm 赋值给实例属性
        self.rmsnorm = rmsnorm
        # 将是否应用残差连接后的 layernorm 赋值给实例属性
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        # 将后层归一化赋值给实例属性
        self.post_layer_norm = post_layer_norm
        # 将是否添加线性偏置赋值给实例属性
        self.add_bias_linear = add_bias_linear
        # 将是否添加 QKV 偏置赋值给实例属性
        self.add_qkv_bias = add_qkv_bias
        # 将偏置 dropout 融合赋值给实例属性
        self.bias_dropout_fusion = bias_dropout_fusion
        # 将多查询注意力赋值给实例属性
        self.multi_query_attention = multi_query_attention
        # 将多查询组的数量赋值给实例属性
        self.multi_query_group_num = multi_query_group_num
        # 将 ROPE 比例赋值给实例属性
        self.rope_ratio = rope_ratio
        # 将查询-键层缩放赋值给实例属性
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        # 将注意力 softmax 在 FP32 中的设置赋值给实例属性
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        # 将 FP32 残差连接的设置赋值给实例属性
        self.fp32_residual_connection = fp32_residual_connection
        # 将前序列长度赋值给实例属性
        self.pre_seq_len = pre_seq_len
        # 将前缀投影赋值给实例属性
        self.prefix_projection = prefix_projection
        # 将 BOI token ID 赋值给实例属性
        self.boi_token_id = boi_token_id
        # 将 EOI token ID 赋值给实例属性
        self.eoi_token_id = eoi_token_id
        # 调用父类的初始化方法，传递其他参数
        super().__init__(**kwargs)
```