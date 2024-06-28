# `.\models\idefics\configuration_idefics.py`

```py
# coding=utf-8
# 声明文件编码格式为UTF-8

# 版权声明和许可证信息，说明此代码基于EleutherAI的GPT-NeoX库，经过修改以适应与Meta AI团队训练的模型的轻微架构差异
# 详细说明了代码的版权信息和许可证，允许在Apache License, Version 2.0下使用此文件

# 导入必要的模块和库
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取与当前模块关联的日志记录器对象
logger = logging.get_logger(__name__)

# 预训练模型配置文件与存档映射表
IDEFICS_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "HuggingFaceM4/idefics-9b": "https://huggingface.co/HuggingFaceM4/idefics-9b/blob/main/config.json",
    "HuggingFaceM4/idefics-80b": "https://huggingface.co/HuggingFaceM4/idefics-80b/blob/main/config.json",
}

# IdeficsVisionConfig类，继承自PretrainedConfig类，用于存储Idefics模型的配置信息
class IdeficsVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`IdeficsModel`]. It is used to instantiate an
    Idefics model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Idefics-9B.

    e.g. [HuggingFaceM4/idefics-9b](https://huggingface.co/HuggingFaceM4/idefics-9b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # 定义模型类型为 "idefics"
    model_type = "idefics"
    
    # 创建属性映射字典，将 "hidden_size" 映射为 "embed_dim"
    attribute_map = {
        "hidden_size": "embed_dim",
    }
    
    # 初始化函数，定义了模型的参数和默认取值
    def __init__(
        self,
        embed_dim=768,  # 编码器层和池化层的维度，默认为 768
        image_size=224,  # 每个图像的分辨率大小，默认为 224
        intermediate_size=5120,  # Transformer 编码器中"中间"（即前馈）层的维度，默认为 5120
        patch_size=14,  # 每个补丁的大小（分辨率），默认为 14
        num_hidden_layers=32,  # Transformer 编码器中的隐藏层数量，默认为 32
        num_attention_heads=16,  # 每个注意力层中的注意力头数，默认为 16
        num_channels=3,  # 图像通道数，默认为 3
        hidden_act="gelu",  # 编码器和池化器中的非线性激活函数，默认为 "gelu"
        layer_norm_eps=1e-5,  # 层归一化层使用的 epsilon，默认为 1e-5
        attention_dropout=0.0,  # 注意力概率的 dropout 比率，默认为 0.0
        initializer_range=0.02,  # 用于初始化所有权重矩阵的截断正态分布的标准差，默认为 0.02
        initializer_factor=1.0,  # 用于初始化权重矩阵的因子（通常保持为 1.0，仅用于初始化测试中）
        **kwargs,  # 其他参数，未指定的参数会被捕获在这里
        ):
            # 设置嵌入维度
            self.embed_dim = embed_dim
            # 设置图像尺寸
            self.image_size = image_size
            # 设置中间层大小
            self.intermediate_size = intermediate_size
            # 设置patch大小
            self.patch_size = patch_size
            # 设置隐藏层数量
            self.num_hidden_layers = num_hidden_layers
            # 设置注意力头数量
            self.num_attention_heads = num_attention_heads
            # 设置通道数量
            self.num_channels = num_channels
            # 设置层归一化 epsilon 值
            self.layer_norm_eps = layer_norm_eps
            # 设置注意力机制的 dropout 率
            self.attention_dropout = attention_dropout
            # 设置初始化范围
            self.initializer_range = initializer_range
            # 设置初始化因子
            self.initializer_factor = initializer_factor
            # 设置隐藏层激活函数
            self.hidden_act = hidden_act

            # 调用父类的初始化方法
            super().__init__(**kwargs)
class IdeficsConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`IdeficsModel`]. It is used to instantiate an
    Idefics model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Idefics-9B.

    e.g. [HuggingFaceM4/idefics-9b](https://huggingface.co/HuggingFaceM4/idefics-9b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```
    >>> from transformers import IdeficsModel, IdeficsConfig

    >>> # Initializing a Idefics idefics-9b style configuration
    >>> configuration = IdeficsConfig()
    ```

    注释：
    声明一个名为 IdeficsConfig 的配置类，用于存储 `IdeficsModel` 的配置信息。
    该配置类根据指定的参数实例化一个 Idefics 模型，定义模型的架构。
    使用默认参数实例化配置类会产生类似 Idefics-9B 模型的配置。
    例如，[HuggingFaceM4/idefics-9b](https://huggingface.co/HuggingFaceM4/idefics-9b) 提供了相关的预训练模型。

    Configuration objects 继承自 [`PretrainedConfig`]，可用于控制模型的输出。详细信息请参阅 [`PretrainedConfig`] 的文档。
    ```
    >>> # 从 idefics-9b 风格的配置中初始化一个模型
    >>> model = IdeficsModel(configuration)
    
    >>> # 访问模型的配置信息
    >>> configuration = model.config
        ):
        # 初始化函数，设置模型的各项参数
        self.vocab_size = vocab_size
        # 额外词汇表大小
        self.additional_vocab_size = additional_vocab_size
        # 隐藏层大小
        self.hidden_size = hidden_size
        # 中间层大小
        self.intermediate_size = intermediate_size
        # 隐藏层的数量
        self.num_hidden_layers = num_hidden_layers
        # 注意力头的数量
        self.num_attention_heads = num_attention_heads
        # dropout 概率
        self.dropout = dropout
        # 隐藏层激活函数
        self.hidden_act = hidden_act
        # 初始化范围
        self.initializer_range = initializer_range
        # alpha 初始化器
        self.alpha_initializer = alpha_initializer
        # alpha 初始化范围
        self.alphas_initializer_range = alphas_initializer_range
        # alpha 类型
        self.alpha_type = alpha_type
        # RMS 规范化的 epsilon
        self.rms_norm_eps = rms_norm_eps
        # 是否使用缓存
        self.use_cache = use_cache

        # 交叉层间隔
        self.cross_layer_interval = cross_layer_interval
        # qk 层归一化
        self.qk_layer_norms = qk_layer_norms
        # 冻结视觉层
        self.freeze_vision_layers = freeze_vision_layers

        # 冻结文本层
        self.freeze_text_layers = freeze_text_layers
        # 冻结文本模块例外
        self.freeze_text_module_exceptions = freeze_text_module_exceptions
        # 冻结视觉模块例外
        self.freeze_vision_module_exceptions = freeze_vision_module_exceptions
        # 冻结 LM 头部
        self.freeze_lm_head = freeze_lm_head

        # 是否使用重采样器
        self.use_resampler = use_resampler

        # 如果 perceiver_config 为 None，则使用默认配置
        if perceiver_config is None:
            self.perceiver_config = IdeficsPerceiverConfig()
        # 如果 perceiver_config 是字典类型，则使用给定的配置创建 IdeficsPerceiverConfig 对象
        elif isinstance(perceiver_config, dict):
            self.perceiver_config = IdeficsPerceiverConfig(**perceiver_config)
        # 如果 perceiver_config 已经是 IdeficsPerceiverConfig 类型，则直接使用它
        elif isinstance(perceiver_config, IdeficsPerceiverConfig):
            self.perceiver_config = perceiver_config

        # 如果 vision_config 为 None，则使用默认配置
        if vision_config is None:
            self.vision_config = IdeficsVisionConfig()
        # 如果 vision_config 是字典类型，则使用给定的配置创建 IdeficsVisionConfig 对象
        elif isinstance(vision_config, dict):
            self.vision_config = IdeficsVisionConfig(**vision_config)
        # 如果 vision_config 已经是 IdeficsVisionConfig 类型，则直接使用它
        elif isinstance(vision_config, IdeficsVisionConfig):
            self.vision_config = vision_config

        # 调用父类的初始化方法，设置特殊标记的 token ID 和其他参数
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # 注意：不要在构造函数中进行任何基于 __init__ 参数的检查，
        # 因为 PretrainedConfig.from_dict 首先使用配置字典实例化类，然后
        # 仅在 from_pretrained 中使用 from_pretrained 的 kwargs 更新配置对象，
        # 所以在实例化此对象时，许多属性具有默认值，尚未被覆盖。
        # 请在运行父类的 from_pretrained 后，在 from_pretrained 中执行任何必要的检查。
```