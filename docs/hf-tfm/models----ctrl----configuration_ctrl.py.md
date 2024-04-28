# `.\models\ctrl\configuration_ctrl.py`

```
# 设置文件编码为utf-8
# 版权声明
# 根据Apache License, Version 2.0进行许可，禁止在非遵守许可情况下使用该文件
# 可以在上述链接获取许可副本
# 在适用法律要求或书面同意的情况下，根据许可分发的软件是基于“按原样”分发的，没有任何种类的保证或条件，不管是明示还是暗示
# 请参阅许可中关于特定语言的输出和限制的相关说明
# 导入必要的库和模块
# 定义全局日志记录器对象
# 在以下字典数据结构中，键为"Salesforce/ctrl"，值为对应的预训练配置文件的URL
# 定义CTRLConfig类，继承自PretrainedConfig类
# 用于实例化CTRL模型，根据指定的参数定义模型架构
# 默认情况下实例化配置将产生类似于销售部/ctrl架构的配置
# 配置对象继承自PretrainedConfig，用于控制模型输出，阅读PretrainedConfig的文档以获取更多信息
    Args:
        vocab_size (`int`, *optional*, defaults to 246534):
            Vocabulary size of the CTRL model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`CTRLModel`] or [`TFCTRLModel`].
        n_positions (`int`, *optional*, defaults to 256):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_embd (`int`, *optional*, defaults to 1280):
            Dimensionality of the embeddings and hidden states.
        dff (`int`, *optional*, defaults to 8192):
            Dimensionality of the inner dimension of the feed forward networks (FFN).
        n_layer (`int`, *optional*, defaults to 48):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`int`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-06):
            The epsilon to use in the layer normalization layers
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).


    Examples:

    ```python
    >>> from transformers import CTRLConfig, CTRLModel

    >>> # Initializing a CTRL configuration
    >>> configuration = CTRLConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = CTRLModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    # 设置模型类型为 "ctrl"
    model_type = "ctrl"
    # 推断时需要忽略的键列表
    keys_to_ignore_at_inference = ["past_key_values"]
    # 映射属性名到初始化时的参数名
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        # 初始化方法，设置默认参数值
        self,
        vocab_size=246534,
        n_positions=256,
        n_embd=1280,
        dff=8192,
        n_layer=48,
        n_head=16,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        layer_norm_epsilon=1e-6,
        initializer_range=0.02,
        use_cache=True,
        **kwargs,
    # 定义初始化函数，用于设置模型的超参数
    def __init__(
        self,
        vocab_size,                 # 词表大小
        n_positions,                # 位置编码的最大长度
        n_embd,                     # 词向量的维度
        n_layer,                    # Transformer 层数
        n_head,                     # 多头自注意力头数
        dff,                        # FeedForward 层隐藏层维度
        resid_pdrop,                # 残差连接的 dropout 概率
        embd_pdrop,                 # 词嵌入 dropout 概率
        layer_norm_epsilon,         # Layer Norm 层 epsilon 值
        initializer_range,          # 参数初始化的范围
        use_cache=False,            # 是否使用缓存
        **kwargs
    ):
        # 将超参数赋值给对象的属性
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.dff = dff
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range

        # 设置是否使用缓存的属性
        self.use_cache = use_cache

        # 调用父类的初始化函数
        super().__init__(**kwargs)
```