# `.\transformers\models\xglm\configuration_xglm.py`

```py
# 导入必要的模块和函数
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# XGLM 模型的预训练配置映射
XGLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/xglm-564M": "https://huggingface.co/facebook/xglm-564M/resolve/main/config.json",
    # 在 https://huggingface.co/models?filter=xglm 查看所有 XGLM 模型
}

# XGLM 模型的配置类，用于存储模型的配置和控制模型输出
class XGLMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`XGLMModel`]. It is used to instantiate an XGLM
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the XGLM
    [facebook/xglm-564M](https://huggingface.co/facebook/xglm-564M) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # 定义一个函数，该函数用于初始化XGLM模型的配置
    Args:
        vocab_size (`int`, *optional*, defaults to 256008):
            XGLM模型的词汇表大小。定义在调用[`XGLMModel`]或[`FlaxXGLMModel`]时传入的`inputs_ids`可以表示的不同令牌数量。
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            此模型可能使用的最大序列长度。通常设置为一个较大的值（例如512、1024或2048）以防万一。
        d_model (`int`, *optional*, defaults to 1024):
            层和池化层的维度。
        ffn_dim (`int`, *optional*, defaults to 4096):
            解码器中“中间”（通常称为前馈）层的维度。
        num_layers (`int`, *optional*, defaults to 24):
            Transformer解码器中的隐藏层数量。
        attention_heads (`int`, *optional*, defaults to 16):
            Transformer解码器中每个注意力层的注意力头数量。
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            编码器和池化层中的非线性激活函数（函数或字符串）。如果是字符串，支持`"gelu"`、`"relu"`、`"silu"`和`"gelu_new"`。
        dropout (`float`, *optional*, defaults to 0.1):
            所有全连接层中的丢弃概率（嵌入层、解码器和池化层）。
        attention_dropout (`float`, *optional*, defaults to 0.1):
            注意力概率的丢弃比例。
        activation_dropout (`float`, *optional*, defaults to 0.0):
            全连接层内部激活的丢弃比例。
        layerdrop (`float`, *optional*, defaults to 0.0):
            编码器的LayerDrop概率。更多细节可参考[LayerDrop paper](see https://arxiv.org/abs/1909.11556)。
        init_std (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态初始化器的标准差。
        scale_embedding (`bool`, *optional*, defaults to `True`):
            通过d_model的平方根进行嵌入缩放。
        use_cache (`bool`, *optional*, defaults to `True`):
            模型是否应返回最后的键/值注意力（并非所有模型都使用）。

    Example:

    ```python
    >>> from transformers import XGLMModel, XGLMConfig

    >>> # 初始化一个XGLM facebook/xglm-564M风格的配置
    >>> configuration = XGLMConfig()

    >>> # 从facebook/xglm-564M风格的配置初始化一个模型
    >>> model = XGLMModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py"""
    
    # 定义模型类型为"xglm"
    model_type = "xglm"
    
    # 在推理时需要忽略的键列表
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射，将类的初始化参数名映射到内部属性名
    attribute_map = {
        "num_attention_heads": "attention_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "num_layers",
    }
    
    # 类的初始化方法，设置默认参数值
    def __init__(
        self,
        vocab_size=256008,  # 词汇表大小，默认为256008
        max_position_embeddings=2048,  # 最大位置嵌入，默认为2048
        d_model=1024,  # 隐藏层大小，默认为1024
        ffn_dim=4096,  # 前馈神经网络维度，默认为4096
        num_layers=24,  # 隐藏层层数，默认为24
        attention_heads=16,  # 注意力头数，默认为16
        activation_function="gelu",  # 激活函数，默认为gelu
        dropout=0.1,  # 丢弃率，默认为0.1
        attention_dropout=0.1,  # 注意力丢弃率，默认为0.1
        activation_dropout=0.0,  # 激活函数丢弃率，默认为0.0
        layerdrop=0.0,  # 层丢弃率，默认为0.0
        init_std=0.02,  # 初始化标准差，默认为0.02
        scale_embedding=True,  # 是否缩放嵌入，默认为True
        use_cache=True,  # 是否使用缓存，默认为True
        decoder_start_token_id=2,  # 解码器起始标记ID，默认为2
        pad_token_id=1,  # 填充标记ID，默认为1
        bos_token_id=0,  # 起始标记ID，默认为0
        eos_token_id=2,  # 结束标记ID，默认为2
        **kwargs,  # 其他参数
    ):
        self.vocab_size = vocab_size  # 设置词汇表大小
        self.max_position_embeddings = max_position_embeddings  # 设置最大位置嵌入
        self.d_model = d_model  # 设置隐藏层大小
        self.ffn_dim = ffn_dim  # 设置前馈神经网络维度
        self.num_layers = num_layers  # 设置隐藏层层数
        self.attention_heads = attention_heads  # 设置注意力头数
        self.activation_function = activation_function  # 设置激活函数
        self.dropout = dropout  # 设置丢弃率
        self.attention_dropout = attention_dropout  # 设置注意力丢弃率
        self.activation_dropout = activation_dropout  # 设置激活函数丢弃率
        self.layerdrop = layerdrop  # 设置层丢弃率
        self.init_std = init_std  # 设置初始化标准差
        self.scale_embedding = scale_embedding  # 设置是否缩放嵌入
        self.use_cache = use_cache  # 设置是否使用缓存
    
        super().__init__(  # 调用父类的初始化方法
            pad_token_id=pad_token_id,  # 设置填充标记ID
            bos_token_id=bos_token_id,  # 设置起始标记ID
            eos_token_id=eos_token_id,  # 设置结束标记ID
            decoder_start_token_id=decoder_start_token_id,  # 设置解码器起始标记ID
            **kwargs,  # 其他参数
        )
```