# `.\transformers\models\nezha\configuration_nezha.py`

```
# 从预训练配置中导入PretrainedConfig类
from ... import PretrainedConfig

# 定义NEZHA预训练配置的映射，将模型名称映射到对应的配置文件链接
NEZHA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "sijunhe/nezha-cn-base": "https://huggingface.co/sijunhe/nezha-cn-base/resolve/main/config.json",
}

# 创建NezhaConfig类，用于存储NezhaModel的配置
class NezhaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`NezhaModel`]. It is used to instantiate an Nezha
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Nezha
    [sijunhe/nezha-cn-base](https://huggingface.co/sijunhe/nezha-cn-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # 这段代码定义了 NezhaConfig 类的参数及其默认值
    Args:
        # 词汇表的大小，定义了可以表示的不同 token
        vocab_size (`int`, optional, defaults to 21128): 
            Vocabulary size of the NEZHA model. Defines the different tokens that can be represented by the
            *inputs_ids* passed to the forward method of [`NezhaModel`].
        # 编码层和池化层的维度
        hidden_size (`int`, optional, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        # Transformer 编码器的隐藏层数量
        num_hidden_layers (`int`, optional, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        # 每个注意力层的注意力头的数量
        num_attention_heads (`int`, optional, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        # 前馈网络层的维度
        intermediate_size (`int`, optional, defaults to 3072):
            The dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        # 编码器和池化层使用的激活函数
        hidden_act (`str` or `function`, optional, defaults to "gelu"):
            The non-linear activation function (function or string) in the encoder and pooler.
        # 在嵌入、编码器和池化层的所有全连接层中使用的dropout概率
        hidden_dropout_prob (`float`, optional, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        # 注意力概率的dropout概率
        attention_probs_dropout_prob (`float`, optional, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        # 模型能处理的最大序列长度
        max_position_embeddings (`int`, optional, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            (e.g., 512 or 1024 or 2048).
        # token_type_ids 的词汇表大小
        type_vocab_size (`int`, optional, defaults to 2):
            The vocabulary size of the *token_type_ids* passed into [`NezhaModel`].
        # 权重矩阵初始化的标准差
        initializer_range (`float`, optional, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        # 层归一化的epsilon值
        layer_norm_eps (`float`, optional, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        # 分类器的dropout概率
        classifier_dropout (`float`, optional, defaults to 0.1):
            The dropout ratio for attached classifiers.
        # 是否用作解码器
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
    
    Example:
    
    
    >>> from transformers import NezhaConfig, NezhaModel
    
    >>> # Initializing an Nezha configuration
    >>> configuration = NezhaConfig()
    
    >>> # Initializing a model (with random weights) from the Nezha-base style configuration model
    >>> model = NezhaModel(configuration)
    
    >>> # Accessing the model configuration
    >>> configuration = model.config
    # 初始化函数，用于初始化 Transformer 模型的参数
    def __init__(
        self,
        vocab_size=21128,  # 词汇表大小，默认为 21128
        hidden_size=768,   # 隐藏层大小，默认为 768
        num_hidden_layers=12,  # Transformer 模型中的隐藏层数，默认为 12
        num_attention_heads=12,  # 注意力头的数量，默认为 12
        intermediate_size=3072,  # 中间层大小，默认为 3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为 GELU
        hidden_dropout_prob=0.1,  # 隐藏层的 dropout 概率，默认为 0.1
        attention_probs_dropout_prob=0.1,  # 注意力层的 dropout 概率，默认为 0.1
        max_position_embeddings=512,  # 最大位置编码，默认为 512
        max_relative_position=64,  # 最大相对位置编码，默认为 64
        type_vocab_size=2,  # 类型词汇表大小，默认为 2
        initializer_range=0.02,  # 参数初始化范围，默认为 0.02
        layer_norm_eps=1e-12,  # LayerNormalization 的 epsilon，默认为 1e-12
        classifier_dropout=0.1,  # 分类器 dropout 概率，默认为 0.1
        pad_token_id=0,  # 填充 token 的 id，默认为 0
        bos_token_id=2,  # 起始 token 的 id，默认为 2
        eos_token_id=3,  # 结束 token 的 id，默认为 3
        use_cache=True,  # 是否使用缓存，默认为 True
        **kwargs,  # 其他参数
    ):
        # 调用父类的初始化函数，传入填充、起始和结束 token 的 id
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
    
        # 设置模型参数
        self.vocab_size = vocab_size  # 词汇表大小
        self.hidden_size = hidden_size  # 隐藏层大小
        self.num_hidden_layers = num_hidden_layers  # 隐藏层数量
        self.num_attention_heads = num_attention_heads  # 注意力头的数量
        self.hidden_act = hidden_act  # 隐藏层激活函数
        self.intermediate_size = intermediate_size  # 中间层大小
        self.hidden_dropout_prob = hidden_dropout_prob  # 隐藏层的 dropout 概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob  # 注意力层的 dropout 概率
        self.max_position_embeddings = max_position_embeddings  # 最大位置编码
        self.max_relative_position = max_relative_position  # 最大相对位置编码
        self.type_vocab_size = type_vocab_size  # 类型词汇表大小
        self.initializer_range = initializer_range  # 参数初始化范围
        self.layer_norm_eps = layer_norm_eps  # LayerNormalization 的 epsilon
        self.classifier_dropout = classifier_dropout  # 分类器 dropout 概率
        self.use_cache = use_cache  # 是否使用缓存
    ```  
```