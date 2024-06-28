# `.\models\nezha\configuration_nezha.py`

```py
# 导入PretrainedConfig类
from ... import PretrainedConfig

# NEZHA_PRETRAINED_CONFIG_ARCHIVE_MAP定义了一个字典，映射了预训练模型名称到其配置文件的URL
NEZHA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "sijunhe/nezha-cn-base": "https://huggingface.co/sijunhe/nezha-cn-base/resolve/main/config.json",
}

# NezhaConfig类继承自PretrainedConfig类，用于存储Nezha模型的配置信息
class NezhaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`NezhaModel`]. It is used to instantiate an Nezha
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Nezha
    [sijunhe/nezha-cn-base](https://huggingface.co/sijunhe/nezha-cn-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # NEZHA 模型的配置类，用于定义模型的各种参数和超参数
    Args:
        vocab_size (`int`, optional, defaults to 21128):
            NEZHA 模型的词汇表大小，定义了可以被输入到 `NezhaModel` 的 `input_ids` 中的不同标记。
        hidden_size (`int`, optional, defaults to 768):
            编码器层和池化层的维度。
        num_hidden_layers (`int`, optional, defaults to 12):
            Transformer 编码器中的隐藏层数量。
        num_attention_heads (`int`, optional, defaults to 12):
            Transformer 编码器中每个注意力层的注意头数量。
        intermediate_size (`int`, optional, defaults to 3072):
            Transformer 编码器中“中间”（即前馈）层的维度。
        hidden_act (`str` or `function`, optional, defaults to "gelu"):
            编码器和池化层中的非线性激活函数。
        hidden_dropout_prob (`float`, optional, defaults to 0.1):
            嵌入层、编码器和池化层中所有全连接层的 dropout 概率。
        attention_probs_dropout_prob (`float`, optional, defaults to 0.1):
            注意力概率的 dropout 比率。
        max_position_embeddings (`int`, optional, defaults to 512):
            模型可能使用的最大序列长度。通常设置为一个较大的值（例如 512、1024 或 2048）。
        type_vocab_size (`int`, optional, defaults to 2):
            传递给 `NezhaModel` 的 `token_type_ids` 的词汇表大小。
        initializer_range (`float`, optional, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态初始化器的标准差。
        layer_norm_eps (`float`, optional, defaults to 1e-12):
            层归一化层使用的 epsilon 值。
        classifier_dropout (`float`, optional, defaults to 0.1):
            附加分类器的 dropout 比率。
        is_decoder (`bool`, *optional*, defaults to `False`):
            模型是否作为解码器使用。如果为 `False`，则模型作为编码器使用。

    Example:

    ```
    >>> from transformers import NezhaConfig, NezhaModel

    >>> # Initializing an Nezha configuration
    >>> configuration = NezhaConfig()

    >>> # Initializing a model (with random weights) from the Nezha-base style configuration model
    >>> model = NezhaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

    # NEZHA 预训练模型配置文件的存档映射
    pretrained_config_archive_map = NEZHA_PRETRAINED_CONFIG_ARCHIVE_MAP

    # 模型类型为 NEZHA
    model_type = "nezha"
    # 初始化函数，用于创建一个新的实例对象
    def __init__(
        self,
        vocab_size=21128,  # 词汇表大小，默认为21128
        hidden_size=768,   # 隐藏层大小，默认为768
        num_hidden_layers=12,  # 隐藏层的数量，默认为12
        num_attention_heads=12,  # 注意力头的数量，默认为12
        intermediate_size=3072,  # 中间层大小，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为GELU
        hidden_dropout_prob=0.1,  # 隐藏层的Dropout概率，默认为0.1
        attention_probs_dropout_prob=0.1,  # 注意力概率的Dropout概率，默认为0.1
        max_position_embeddings=512,  # 最大位置编码数，默认为512
        max_relative_position=64,  # 最大相对位置数，默认为64
        type_vocab_size=2,   # 类型词汇表大小，默认为2
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-12,   # 层归一化的epsilon，默认为1e-12
        classifier_dropout=0.1,   # 分类器的Dropout概率，默认为0.1
        pad_token_id=0,   # 填充标记ID，默认为0
        bos_token_id=2,   # 起始标记ID，默认为2
        eos_token_id=3,   # 结束标记ID，默认为3
        use_cache=True,   # 是否使用缓存，默认为True
        **kwargs,   # 其他关键字参数
    ):
        # 调用父类的初始化函数，传递填充标记ID、起始标记ID、结束标记ID以及其他关键字参数
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 设置实例对象的属性
        self.vocab_size = vocab_size   # 设置词汇表大小属性
        self.hidden_size = hidden_size   # 设置隐藏层大小属性
        self.num_hidden_layers = num_hidden_layers   # 设置隐藏层数量属性
        self.num_attention_heads = num_attention_heads   # 设置注意力头数量属性
        self.hidden_act = hidden_act   # 设置隐藏层激活函数属性
        self.intermediate_size = intermediate_size   # 设置中间层大小属性
        self.hidden_dropout_prob = hidden_dropout_prob   # 设置隐藏层的Dropout概率属性
        self.attention_probs_dropout_prob = attention_probs_dropout_prob   # 设置注意力概率的Dropout概率属性
        self.max_position_embeddings = max_position_embeddings   # 设置最大位置编码数属性
        self.max_relative_position = max_relative_position   # 设置最大相对位置数属性
        self.type_vocab_size = type_vocab_size   # 设置类型词汇表大小属性
        self.initializer_range = initializer_range   # 设置初始化范围属性
        self.layer_norm_eps = layer_norm_eps   # 设置层归一化的epsilon属性
        self.classifier_dropout = classifier_dropout   # 设置分类器的Dropout概率属性
        self.use_cache = use_cache   # 设置是否使用缓存属性
```