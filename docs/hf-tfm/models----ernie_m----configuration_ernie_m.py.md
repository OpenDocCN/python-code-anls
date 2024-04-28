# `.\models\ernie_m\configuration_ernie_m.py`

```
# 设置文件编码为UTF-8
# 版权声明
# 版权声明说明了使用该文件的限制和条件
# 授权许可
# 通过该链接获取许可证的副本
# 
# 此处引用了原 PaddleNLP 代码库中的 ErnieM 模型配置
# 导入所需的类型注解
# 导入预训练配置
# 指定预训练模型配置文件的映射关系
# 预训练模型配置文件的 URL 映射
# ErnieM 模型配置类，用于存储 [`ErnieMModel`] 的配置信息
# 该类用于根据指定参数实例化Ernie-M模型，定义模型架构
# 使用默认值实例化配置对象将生成类似于 `Ernie-M` [susnato/ernie-m-base_pytorch](https://huggingface.co/susnato/ernie-m-base_pytorch) 架构的配置
# 配置对象从 [`PretrainedConfig`] 继承，并可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息
    Args:
        vocab_size (`int`, *optional*, defaults to 250002):
            Vocabulary size of `inputs_ids` in [`ErnieMModel`]. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling
            [`ErnieMModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the embedding layer, encoder layers and pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the feed-forward (ff) layer in the encoder. Input tensors to feed-forward layers are
            firstly projected from hidden_size to intermediate_size, and then projected back to hidden_size. Typically
            intermediate_size is larger than hidden_size.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function in the feed-forward layer. `"gelu"`, `"relu"` and any other torch
            supported activation functions are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings and encoder.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability used in `MultiHeadAttention` in all encoder layers to drop some attention target.
        max_position_embeddings (`int`, *optional*, defaults to 514):
            The maximum value of the dimensionality of position encoding, which dictates the maximum supported length
            of an input sequence.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the normal initializer for initializing all weight matrices. The index of padding
            token in the token vocabulary.
        pad_token_id (`int`, *optional*, defaults to 1):
            Padding token id.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
        act_dropout (`float`, *optional*, defaults to 0.0):
            This dropout probability is used in `ErnieMEncoderLayer` after activation.

    A normal_initializer initializes weight matrices as normal distributions. See
    `ErnieMPretrainedModel._init_weights()` for how weights are initialized in `ErnieMModel`.
    """

    # 定义模型类型为 "ernie_m"
    model_type = "ernie_m"
    # 定义属性映射字典，将 "dropout" 映射为 "classifier_dropout"，将 "num_classes" 映射为 "num_labels"
    attribute_map: Dict[str, str] = {"dropout": "classifier_dropout", "num_classes": "num_labels"}
    # 初始化函数，定义了一个名为__init__的类构造函数
    def __init__(
        self,
        vocab_size: int = 250002,  # 词汇表大小，默认为250002
        hidden_size: int = 768,  # 隐藏层大小，默认为768
        num_hidden_layers: int = 12,  # 隐藏层数，默认为12
        num_attention_heads: int = 12,  # 注意力头数，默认为12
        intermediate_size: int = 3072,  # 中间层大小，默认为3072
        hidden_act: str = "gelu",  # 隐藏层激活函数，默认为'gelu'
        hidden_dropout_prob: float = 0.1,  # 隐藏层丢弃率，默认为0.1
        attention_probs_dropout_prob: float = 0.1,  # 注意力概率丢弃率，默认为0.1
        max_position_embeddings: int = 514,  # 最大位置嵌入，默认为514
        initializer_range: float = 0.02,  # 初始化范围，默认为0.02
        pad_token_id: int = 1,  # 填充符的token id，默认为1
        layer_norm_eps: float = 1e-05,  # 层归一化的epsilon值，默认为1e-05
        classifier_dropout=None,  # 分类器丢弃率，默认为None
        act_dropout=0.0,  # 激活函数丢弃率，默认为0.0
        **kwargs,  # 接收额外的关键字参数
    ):
        # 调用父类的初始化方法，设置填充符的token id和额外的关键字参数
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        # 初始化模型参数
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.classifier_dropout = classifier_dropout
        self.act_dropout = act_dropout
```