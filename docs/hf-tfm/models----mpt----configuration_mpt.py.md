# `.\transformers\models\mpt\configuration_mpt.py`

```py
# 设置文件编码格式为utf-8

# 版权声明

# 只有在遵守许可证的情况下才能使用该文件

# 如果需要，可以在此处获取许可证的副本

# 除非软件受适用法律规定或以书面形式同意，否则按"原样"提供软件

# 没有任何保证或条件，无论是明示还是默示

# 有关特定语言控制的权限和限制，请参阅许可证

"""
Mpt配置
"""

# 导入必要的模块
from typing import TYPE_CHECKING, Optional, Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# MPT预训练配置档案映射
MPT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "mosaicml/mpt-7b": "https://huggingface.co/mosaicml/mpt-7b/resolve/main/config.json",
}

# MptAttentionConfig类，用于存储MptAttention类的配置
class MptAttentionConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`MptAttention`] class. It is used to instantiate
    attention layers according to the specified arguments, defining the layers architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MPT
    [mosaicml/mpt-7b](https://huggingface.co/mosaicml/mpt-7b) architecture. Most of the arguments are kept for backward
    compatibility with previous MPT models that are hosted on the Hub (previously with `trust_remote_code=True`).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        attn_type (`str`, *optional*, defaults to `"multihead_attention"`):
            type of attention to use. Options: `"multihead_attention"`, `"multiquery_attention"`.
        attn_pdrop (`float`, *optional*, defaults to 0.0):
            The dropout probability for the attention layers.
        attn_impl (`str`, *optional*, defaults to `"torch"`):
            The attention implementation to use. One of `"torch"`, `"flash"`, or `"triton"`.
        clip_qkv (`float`, *optional*):
            If not `None`, clip the queries, keys, and values in the attention layer to this value.
        softmax_scale (`float`, *optional*, defaults to `None`):
            If not `None`, scale the softmax in the attention layer by this value. If `None`, will default to
            `1/sqrt(hidden_size)`.
        prefix_lm (`bool`, *optional*, defaults to `False`)):
            Whether the model should operate as a Prefix LM. This requires passing an extra `prefix_mask` argument
            which indicates which tokens belong to the prefix. Tokens in the prefix can attend to one another
            bi-directionally. Tokens outside the prefix use causal attention.
        qk_ln (`bool`, *optional*, defaults to `False`):
            Whether to apply layer normalization to the queries and keys in the attention layer.
        attn_uses_sequence_id (`bool`, *optional*, defaults to `False`)):
            Whether to restrict attention to tokens that have the same token_type_ids. When the model is in `train`
            mode, this requires passing an extra *token_type_ids* argument which indicates which sub-sequence each
            token belongs to. Defaults to `False` meaning any provided *token_type_ids* will be ignored.
        alibi (`bool`, *optional*, defaults to `True`):
            Whether or not to use the alibi bias instead of positional embedding.
        alibi_bias_max (`int`, *optional*, defaults to 8):
            The maximum value of the alibi bias.
    """

    def __init__(
        self,
        attn_type="multihead_attention",
        attn_pdrop=0,
        attn_impl="torch",
        clip_qkv=None,
        softmax_scale=None,
        prefix_lm=False,
        qk_ln=False,
        attn_uses_sequence_id=False,
        alibi=True,
        alibi_bias_max=8,
        **kwargs,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 设置注意力机制类型
        self.attn_type = attn_type
        # 设置注意力层的丢弃概率
        self.attn_pdrop = attn_pdrop
        # 设置注意力实现方式
        self.attn_impl = attn_impl
        # 设置是否对注意力层的查询、键和值进行剪裁
        self.clip_qkv = clip_qkv
        # 设置softmax缩放因子
        self.softmax_scale = softmax_scale
        # 设置是否作为前缀语言模型运行
        self.prefix_lm = prefix_lm
        # 设置是否对注意力层的查询和键应用层归一化
        self.qk_ln = qk_ln
        # 设置是否使用序列ID来限制注意力
        self.attn_uses_sequence_id = attn_uses_sequence_id
        # 设置是否使用 alibi 偏置而不是位置嵌入
        self.alibi = alibi
        # 设置 alibi 偏置的最大值
        self.alibi_bias_max = alibi_bias_max

        # 检查注意力类型是否有效
        if attn_type not in ["multihead_attention", "multiquery_attention"]:
            raise ValueError(
                f"`attn_type` has to be either `multihead_attention` or `multiquery_attention`. Received: {attn_type}"
            )

    @classmethod
    # 从预训练的模型名称或路径中加载预训练配置信息并返回对应的配置对象
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs) -> "PretrainedConfig":
        # 设置 token 到参数中
        cls._set_token_in_kwargs(kwargs)

        # 获取预训练模型的配置字典和参数
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典中的模型类型为 "mpt"，则将配置字典更新为其 "attn_config" 部分
        if config_dict.get("model_type") == "mpt":
            config_dict = config_dict["attn_config"]

        # 如果配置字典中存在 "model_type" 并且当前类有 "model_type" 属性，且二者不相等，则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 根据配置字典创建对应类的对象并返回
        return cls.from_dict(config_dict, **kwargs)
# 创建MptConfig类，它继承自PretrainedConfig类
class MptConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`MptModel`]. It is used to instantiate a Mpt model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to the Mpt-7b architecture
    [mosaicml/mpt-7b](https://huggingface.co/mosaicml/mpt-7b).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # 定义一个函数，用于初始化 MPT 模型的参数
    Args:
        # 设置嵌入和隐藏状态的维度，默认为 2048
        d_model (`int`, *optional*, defaults to 2048):
        # 每个注意力层中的注意力头的数量，默认为 16
        n_heads (`int`, *optional*, defaults to 16):
        # Transformer 编码器中隐藏层的数量，默认为 24
        n_layers (`int`, *optional*, defaults to 24):
        # MLP 中上/下比例的倍数，默认为 4
        expansion_ratio (`int`, *optional*, defaults to 4):
        # 模型的最大序列长度，默认为 2048
        max_seq_len (`int`, *optional*, defaults to 2048):
        # MPT 模型的词汇表大小，默认为 50368
        vocab_size (`int`, *optional*, defaults to 50368):
            # 定义`MptModel`调用时`inputs_ids`传递的最大不同标记数量
            # 详细信息参见[此处的讨论](https://huggingface.co/bigscience/mpt/discussions/120#633d28389addb8530b406c2a)
        resid_pdrop (`float`, *optional*, defaults to 0.0):
            # 合并残差之前应用于注意力输出的丢弃概率
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
            # 在层规范化层中使用的 epsilon
        emb_pdrop (`float`, *optional*, defaults to 0.0):
            # 嵌入层的丢弃概率
        learned_pos_emb (`bool`, *optional*, defaults to `True`):
            # 是否使用学习的位置嵌入
        attn_config (`dict`, *optional*):
            # 用于配置模型注意模块的字典
        init_device (`str`, *optional*, defaults to `"cpu"`):
            # 用于参数初始化的设备。出于向后兼容性而定义
        logit_scale (`float`, *optional*):
            # 如果不是 None，则按该值缩放对数值
        no_bias (`bool`, *optional*, defaults to `True`):
            # 是否在所有线性层中使用偏置
        verbose (`int`, *optional*, defaults to 0):
            # 用于日志记录的详细级别。在以前的 MPT 模型版本中用于日志记录。此参数已弃用
        embedding_fraction (`float`, *optional*, defaults to 1.0):
            # 用于按比例扩展嵌入层梯度的分数
        norm_type (`str`, *optional*, defaults to `"low_precision_layernorm"`):
            # 要使用的层规范化类型。所有 MPT 模型使用相同的层规范化实现。出于向后兼容性考虑
        use_cache (`bool`, *optional*, defaults to `False`):
            # 模型是否应返回最后的键/值注意力（不是所有模型都使用）
        initializer_range (`float`, *optional*, defaults to 0.02):
            # 用于初始化所有权重矩阵的截断正态初始化器的标准偏差
    Example:
    # 示例
    ```python
    # 导入 MptConfig 和 MptModel 类
    >>> from transformers import MptConfig, MptModel

    # 初始化一个 Mpt 配置对象
    >>> # Initializing a Mpt configuration
    >>> configuration = MptConfig()

    # 用配置初始化一个模型（带有随机权重）
    >>> # Initializing a model (with random weights) from the configuration
    >>> model = MptModel(configuration)

    # 获取模型的配置信息
    >>> # Accessing the model configuration
    >>> configuration = model.config
    """
    
    # 定义模型类型
    model_type = "mpt"
    # 属性映射字典，用于转换配置参数
    attribute_map = {
        "num_attention_heads": "n_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "n_layers",
    }

    # 初始化 MptModel 类
    def __init__(
        self,
        d_model: int = 2048,  # 隐藏层维度，默认为 2048
        n_heads: int = 16,    # 注意力头数，默认为 16
        n_layers: int = 24,   # 隐藏层层数，默认为 24
        expansion_ratio: int = 4,  # 扩张比例，默认为 4
        max_seq_len: int = 2048,   # 最大序列长度，默认为 2048
        vocab_size: int = 50368,   # 词汇表大小，默认为 50368
        resid_pdrop: float = 0.0,  # 残差连接概率，默认为 0.0
        layer_norm_epsilon: float = 1e-5,  # 层归一化 epsilon，默认为 1e-5
        emb_pdrop: float = 0.0,   # 嵌入层 dropout 概率，默认为 0.0
        learned_pos_emb: bool = True,  # 是否使用学习位置嵌入，默认为 True
        attn_config: MptAttentionConfig = None,  # 注意力配置，默认为 None
        init_device: str = "cpu",   # 初始化设备，默认为 "cpu"
        logit_scale: Optional[Union[float, str]] = None,  # 对数尺度，默认为 None
        no_bias: bool = True,       # 是否禁用偏置，默认为 True
        verbose: int = 0,           # 冗余输出级别，默认为 0
        embedding_fraction: float = 1.0,  # 嵌入层权重占比，默认为 1.0
        norm_type: str = "low_precision_layernorm",  # 归一化类型，默认为 "low_precision_layernorm"
        use_cache: bool = False,    # 是否使用缓存，默认为 False
        initializer_range=0.02,     # 初始化范围，默认为 0.02
        **kwargs,                   # 其他参数
    ):
        # 如果未提供注意力配置，则创建一个新的配置对象
        if attn_config is None:
            self.attn_config = MptAttentionConfig()
        # 如果提供的是字典形式的注意力配置，则使用它创建一个新的配置对象
        elif isinstance(attn_config, dict):
            self.attn_config = MptAttentionConfig(**attn_config)
        # 否则，直接使用提供的注意力配置对象
        else:
            self.attn_config = attn_config
        # 设置模型的各种属性
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.expansion_ratio = expansion_ratio
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.resid_pdrop = resid_pdrop
        self.emb_pdrop = emb_pdrop
        self.learned_pos_emb = learned_pos_emb
        self.init_device = init_device
        self.logit_scale = logit_scale
        self.no_bias = no_bias
        self.verbose = verbose
        self.embedding_fraction = embedding_fraction
        self.norm_type = norm_type
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        # 调用父类的初始化方法
        super().__init__(**kwargs)
```