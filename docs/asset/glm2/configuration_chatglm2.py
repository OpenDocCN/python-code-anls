from transformers import PretrainedConfig


class ChatGLMConfig(PretrainedConfig):
    model_type = "chatglm"
    def __init__(
        self,
        num_layers=28,
        padded_vocab_size=65024,
        hidden_size=4096,
        ffn_hidden_size=13696,
        kv_channels=128,
        num_attention_heads=32,
        seq_length=2048,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        layernorm_epsilon=1e-5,
        rmsnorm=True,
        apply_residual_connection_post_layernorm=False,
        post_layer_norm=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        bias_dropout_fusion=True,
        multi_query_attention=False,
        multi_query_group_num=1,
        apply_query_key_layer_scaling=True,
        attention_softmax_in_fp32=True,
        fp32_residual_connection=False,
        quantization_bit=0,
        pre_seq_len=None,
        prefix_projection=False,
        **kwargs
    ):
        # LayerCount，层数
        self.num_layers = num_layers
        # VocabSize，词表大小，也就是词嵌入数量
        self.vocab_size = padded_vocab_size
        # 这个和上面一样
        self.padded_vocab_size = padded_vocab_size
        # HidSize，词嵌入的长度
        self.hidden_size = hidden_size
        # FFS，FFN 层的中间那个嵌入的长度，一般是 4 * HidSize
        self.ffn_hidden_size = ffn_hidden_size
        # HeadSize，每个头的长度
        self.kv_channels = kv_channels
        # HC，头的个数，HeadSize 和 HC 乘起来是投影大小 PS，并非ES
        self.num_attention_heads = num_attention_heads
        # SeqLen，单词最大长度
        self.seq_length = seq_length
        # FFN 层的dropout大小
        self.hidden_dropout = hidden_dropout
        # 注意力层的dropout大小
        self.attention_dropout = attention_dropout
        # LN 层归一化的时候分母的容差，防止分母为零
        self.layernorm_epsilon = layernorm_epsilon
        # 是否将 LN 改成RMS
        self.rmsnorm = rmsnorm
        # TFBlock 中，残差项在 LN1 之前还是之后
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        # 是否在最后一个 TFBlock 之后添加 LN
        self.post_layer_norm = post_layer_norm
        # nn.Linear 是否添加偏置
        self.add_bias_linear = add_bias_linear
        # 计算QKV的线性层（LLQKV）是否添加偏置
        self.add_qkv_bias = add_qkv_bias
        self.bias_dropout_fusion = bias_dropout_fusion
        # 是否开启 MQA（给注意力头分组，每组共享一个KV）
        self.multi_query_attention = multi_query_attention
        # 分组的数量
        self.multi_query_group_num = multi_query_group_num
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        # FP32 残差连接
        self.fp32_residual_connection = fp32_residual_connection
        # 量化位数
        self.quantization_bit = quantization_bit
        # 
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection
        super().__init__(**kwargs)