# `.\models\bros\configuration_bros.py`

```py
# 导入所需模块和类
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取 logger 对象用于记录日志
logger = logging.get_logger(__name__)

# 预训练配置与 URL 映射表，用于不同的 Bros 模型
BROS_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "jinho8345/bros-base-uncased": "https://huggingface.co/jinho8345/bros-base-uncased/blob/main/config.json",
    "jinho8345/bros-large-uncased": "https://huggingface.co/jinho8345/bros-large-uncased/blob/main/config.json",
}

# BrosConfig 类，继承自 PretrainedConfig，用于存储 Bros 模型的配置信息
class BrosConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BrosModel`] or a [`TFBrosModel`]. It is used to
    instantiate a Bros model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Bros
    [jinho8345/bros-base-uncased](https://huggingface.co/jinho8345/bros-base-uncased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # 定义 Bros 模型的配置类 BrosConfig，用于设置模型参数
    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Bros 模型的词汇表大小，定义了在调用 `BrosModel` 或 `TFBrosModel` 时可以表示的不同 token 数量。
        hidden_size (`int`, *optional*, defaults to 768):
            编码器层和池化层的维度大小。
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Transformer 编码器中的隐藏层数量。
        num_attention_heads (`int`, *optional*, defaults to 12):
            Transformer 编码器中每个注意力层的注意力头数量。
        intermediate_size (`int`, *optional*, defaults to 3072):
            Transformer 编码器中“中间层”（通常称为前馈层）的维度大小。
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            编码器和池化器中的非线性激活函数（函数或字符串）。支持的字符串有 `"gelu"`, `"relu"`, `"silu"` 和 `"gelu_new"`。
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            嵌入层、编码器和池化器中所有全连接层的 dropout 概率。
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            注意力概率的 dropout 比率。
        max_position_embeddings (`int`, *optional*, defaults to 512):
            此模型可能使用的最大序列长度。通常设置为较大的值（例如 512、1024 或 2048）以防万一。
        type_vocab_size (`int`, *optional*, defaults to 2):
            在调用 `BrosModel` 或 `TFBrosModel` 时传递的 `token_type_ids` 的词汇表大小。
        initializer_range (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态初始化器的标准差。
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            层归一化层使用的 epsilon 值。
        pad_token_id (`int`, *optional*, defaults to 0):
            词汇表中填充 token 的索引。
        dim_bbox (`int`, *optional*, defaults to 8):
            边界框坐标的维度大小。 (x0, y1, x1, y0, x1, y1, x0, y1)
        bbox_scale (`float`, *optional*, defaults to 100.0):
            边界框坐标的缩放因子。
        n_relations (`int`, *optional*, defaults to 1):
            SpadeEE（实体提取）、SpadeEL（实体链接）头部的关系数量。
        classifier_dropout_prob (`float`, *optional*, defaults to 0.1):
            分类器头部的 dropout 比率。
    
    Examples:
    
    ```
    >>> from transformers import BrosConfig, BrosModel
    
    >>> # Initializing a BROS jinho8345/bros-base-uncased style configuration
    >>> configuration = BrosConfig()
    
    
    # 创建一个BrosConfig的实例对象并赋值给configuration变量
    configuration = BrosConfig()
    
    
    
    >>> # 使用jinho8345/bros-base-uncased风格配置初始化一个模型
    >>> model = BrosModel(configuration)
    
    
    # 使用BrosConfig实例对象configuration初始化一个BrosModel模型
    model = BrosModel(configuration)
    
    
    
    >>> # 获取模型的配置信息
    >>> configuration = model.config
    
    
    # 获取模型model的配置信息，并赋值给configuration变量
    configuration = model.config
    
    
    
    model_type = "bros"
    
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        dim_bbox=8,
        bbox_scale=100.0,
        n_relations=1,
        classifier_dropout_prob=0.1,
        **kwargs,
    ):
    
    
    # 设置模型的类型为"bros"
    model_type = "bros"
    
    # 初始化函数，用于创建BrosConfig的实例
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        dim_bbox=8,
        bbox_scale=100.0,
        n_relations=1,
        classifier_dropout_prob=0.1,
        **kwargs,
    ):
    
    
    
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            **kwargs,
        )
    
    
    # 调用父类的初始化方法，初始化模型的各种参数
    super().__init__(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=type_vocab_size,
        initializer_range=initializer_range,
        layer_norm_eps=layer_norm_eps,
        pad_token_id=pad_token_id,
        **kwargs,
    )
    
    
    
        self.dim_bbox = dim_bbox
        self.bbox_scale = bbox_scale
        self.n_relations = n_relations
        self.dim_bbox_sinusoid_emb_2d = self.hidden_size // 4
        self.dim_bbox_sinusoid_emb_1d = self.dim_bbox_sinusoid_emb_2d // self.dim_bbox
        self.dim_bbox_projection = self.hidden_size // self.num_attention_heads
        self.classifier_dropout_prob = classifier_dropout_prob
    
    
    # 初始化模型的特定属性和超参数
    self.dim_bbox = dim_bbox
    self.bbox_scale = bbox_scale
    self.n_relations = n_relations
    self.dim_bbox_sinusoid_emb_2d = self.hidden_size // 4
    self.dim_bbox_sinusoid_emb_1d = self.dim_bbox_sinusoid_emb_2d // self.dim_bbox
    self.dim_bbox_projection = self.hidden_size // self.num_attention_heads
    self.classifier_dropout_prob = classifier_dropout_prob
```