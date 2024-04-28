# `.\transformers\models\rwkv\configuration_rwkv.py`

```
# 设置文件编码为 utf-8
# 版权声明
# 版权许可协议
# 当前配置类用于存储 [`RwkvModel`] 的配置信息。根据指定参数实例化 RWKV 模型，定义模型架构。使用默认配置实例化将得到类似 RWVK-4 [RWKV/rwkv-4-169m-pile](https://huggingface.co/RWKV/rwkv-4-169m-pile) 架构的配置。
# 配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。
    # 设置参数说明
    Args:
        vocab_size (`int`, *optional*, defaults to 50277):
            RWKV 模型的词汇表大小。定义了在调用 [`RwkvModel`] 时可以表示的不同标记的数量。
        context_length (`int`, *optional*, defaults to 1024):
            此模型可以在单个前向传播中使用的最大序列长度（以 RNN 模式使用时，可使用任何序列长度）。
        hidden_size (`int`, *optional*, defaults to 4096):
            嵌入和隐藏状态的维度。
        num_hidden_layers (`int`, *optional*, defaults to 32):
            模型中的隐藏层数。
        attention_hidden_size (`int`, *optional*):
            注意力隐藏状态的维度。如果未设置，将默认为 `hidden_size`。
        intermediate_size (`int`, *optional*):
            内部前馈层的维度。如果未设置，将默认为 `hidden_size` 的 4 倍。
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
            在层归一化层中使用的 epsilon。
        bos_token_id (`int`, *optional*, defaults to 0):
            词汇表中句子开始标记的 id。由于 RWKV 使用与 GPTNeoX 相同的分词器，默认为 0。
        eos_token_id (`int`, *optional*, defaults to 0):
            词汇表中句子结束标记的 id。由于 RWKV 使用与 GPTNeoX 相同的分词器，默认为 0。
        rescale_every (`int`, *optional*, defaults to 6):
            在推断时，隐藏状态（以及相应输出层的权重）在每 `rescale_every` 层后除以 2。如果设置为 0 或负数，则不进行重新缩放。
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            是否将词嵌入与输入令牌嵌入进行绑定。
        use_cache (`bool`, *optional*, defaults to `True`):
            模型是否应返回最后的状态。

    # 示例代码
    Example:

    ```python
    >>> from transformers import RwkvConfig, RwkvModel

    >>> # 初始化 Rwkv 配置
    >>> configuration = RwkvConfig()

    >>> # 从配置初始化模型（具有随机权重）
    >>> model = RwkvModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```"""

    # 定义模型类型
    model_type = "rwkv"
    # 属性映射
    attribute_map = {"max_position_embeddings": "context_length"}

    # 初始化函数
    def __init__(
        self,
        vocab_size=50277,
        context_length=1024,
        hidden_size=4096,
        num_hidden_layers=32,
        attention_hidden_size=None,
        intermediate_size=None,
        layer_norm_epsilon=1e-5,
        bos_token_id=0,
        eos_token_id=0,
        rescale_every=6,
        tie_word_embeddings=False,
        use_cache=True,
        **kwargs,
        # 初始化 Transformer 模型的参数，包括词汇量大小、上下文长度、隐藏层大小、隐藏层层数、注意力隐藏层大小、中间层大小、LayerNormalization 的 epsilon 值、重新缩放频率、是否使用缓存
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.attention_hidden_size = attention_hidden_size if attention_hidden_size is not None else hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else 4 * hidden_size
        self.layer_norm_epsilon = layer_norm_epsilon
        self.rescale_every = rescale_every
        self.use_cache = use_cache

        # 初始化起始符号和结束符号的 token ID
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # 调用父类的初始化方法，传递参数并绑定词嵌入权重（如果需要）
        super().__init__(
            tie_word_embeddings=tie_word_embeddings, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs
        )
```