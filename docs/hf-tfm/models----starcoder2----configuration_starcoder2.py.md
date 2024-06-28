# `.\models\starcoder2\configuration_starcoder2.py`

```py
# 定义 Starcoder2Config 类，用于存储 Starcoder2 模型的配置信息
class Starcoder2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Starcoder2Model`]. It is used to instantiate a
    Starcoder2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the [bigcode/starcoder2-7b_16k](https://huggingface.co/bigcode/starcoder2-7b_16k) model.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    ```
    >>> from transformers import Starcoder2Model, Starcoder2Config

    >>> # Initializing a Starcoder2 7B style configuration
    >>> configuration = Starcoder2Config()

    >>> # Initializing a model from the Starcoder2 7B style configuration
    >>> model = Starcoder2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

    model_type = "starcoder2"  # 指定模型类型为 "starcoder2"
    keys_to_ignore_at_inference = ["past_key_values"]  # 推理过程中需要忽略的键列表

    def __init__(
        self,
        vocab_size=49152,  # 词汇表大小，默认为 49152
        hidden_size=3072,  # 隐藏层大小，默认为 3072
        intermediate_size=12288,  # 中间层大小，默认为 12288
        num_hidden_layers=30,  # 隐藏层层数，默认为 30
        num_attention_heads=24,  # 注意力头数，默认为 24
        num_key_value_heads=2,  # 键值头数，默认为 2
        hidden_act="gelu_pytorch_tanh",  # 隐藏层激活函数，默认为 "gelu_pytorch_tanh"
        max_position_embeddings=4096,  # 最大位置嵌入数，默认为 4096
        initializer_range=0.018042,  # 初始化范围，默认为 0.018042
        norm_epsilon=1e-5,  # 归一化过程中的 epsilon 值，默认为 1e-5
        use_cache=True,  # 是否使用缓存，默认为 True
        bos_token_id=50256,  # 开始标记的 token ID，默认为 50256
        eos_token_id=50256,  # 结束标记的 token ID，默认为 50256
        rope_theta=10000.0,  # 绳索 theta 参数，默认为 10000.0
        sliding_window=None,  # 滑动窗口大小，默认为 None
        attention_dropout=0.0,  # 注意力部分的 dropout 率，默认为 0.0
        residual_dropout=0.0,  # 残差连接的 dropout 率，默认为 0.0
        embedding_dropout=0.0,  # 嵌入层的 dropout 率，默认为 0.0
        use_bias=True,  # 是否使用偏置，默认为 True
        **kwargs,  # 其他参数
    ):
        # 调用父类 PretrainedConfig 的构造函数，初始化配置信息
        super().__init__(**kwargs)
        # 初始化Transformer模型的各种超参数
        self.vocab_size = vocab_size  # 词汇表大小
        self.max_position_embeddings = max_position_embeddings  # 最大位置编码数
        self.hidden_size = hidden_size  # 隐藏层大小
        self.intermediate_size = intermediate_size  # 中间层大小
        self.num_hidden_layers = num_hidden_layers  # 隐藏层层数
        self.num_attention_heads = num_attention_heads  # 注意力头数
        self.sliding_window = sliding_window  # 滑动窗口大小
        self.use_bias = use_bias  # 是否使用偏置项
        self.num_key_value_heads = num_key_value_heads  # 键值头数
        self.hidden_act = hidden_act  # 隐藏层激活函数
        self.initializer_range = initializer_range  # 初始化范围
        self.norm_epsilon = norm_epsilon  # 归一化操作中的epsilon值
        self.use_cache = use_cache  # 是否使用缓存
        self.rope_theta = rope_theta  # ROPE模型的theta参数
        self.attention_dropout = attention_dropout  # 注意力机制的dropout率
        self.residual_dropout = residual_dropout  # 残差连接的dropout率
        self.embedding_dropout = embedding_dropout  # 嵌入层的dropout率

        # 调用父类构造函数，初始化Transformer模型的基本设置
        super().__init__(
            bos_token_id=bos_token_id,  # 开始标记的ID
            eos_token_id=eos_token_id,  # 结束标记的ID
            **kwargs,  # 其他可能的参数
        )
```