# `.\transformers\models\autoformer\configuration_autoformer.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 根据适用法律的要求或书面同意，本软件是基于“按原样”基础分发的，没有任何担保或条件
# 请参阅许可证以获取更多信息

"""Autoformer 模型配置"""

# 导入类型提示
from typing import List, Optional

# 导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 导入日志记录工具
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练模型配置文件的映射，将模型名称映射到其配置文件的 URL
AUTOFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "huggingface/autoformer-tourism-monthly": "https://huggingface.co/huggingface/autoformer-tourism-monthly/resolve/main/config.json",
}

# Autoformer 模型配置类，继承自预训练配置类
class AutoformerConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`AutoformerModel`] 的配置。根据指定的参数实例化 Autoformer 模型，
    定义模型架构。使用默认参数实例化配置将产生类似于 Autoformer
    [huggingface/autoformer-tourism-monthly](https://huggingface.co/huggingface/autoformer-tourism-monthly)
    架构的配置。

    继承自 [`PretrainedConfig`] 的配置对象可用于控制模型的输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    ```python
    >>> from transformers import AutoformerConfig, AutoformerModel

    >>> # 初始化默认的 Autoformer 配置
    >>> configuration = AutoformerConfig()

    >>> # 随机初始化一个模型（具有随机权重）从配置中
    >>> model = AutoformerModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py"""

    # 模型类型
    model_type = "autoformer"
    # 属性映射，将 Autoformer 的配置属性映射到内部使用的属性名称
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
        "num_hidden_layers": "encoder_layers",
    }
    # 初始化函数，设置模型的参数
    def __init__(
        self,
        prediction_length: Optional[int] = None,  # 预测长度，默认为None
        context_length: Optional[int] = None,  # 上下文长度，默认为None
        distribution_output: str = "student_t",  # 分布输出类型，默认为"student_t"
        loss: str = "nll",  # 损失函数类型，默认为"nll"
        input_size: int = 1,  # 输入大小，默认为1
        lags_sequence: List[int] = [1, 2, 3, 4, 5, 6, 7],  # 滞后序列，默认为[1, 2, 3, 4, 5, 6, 7]
        scaling: bool = True,  # 是否进行缩放，默认为True
        num_time_features: int = 0,  # 时间特征数量，默认为0
        num_dynamic_real_features: int = 0,  # 动态实数特征数量，默认为0
        num_static_categorical_features: int = 0,  # 静态分类特征数量，默认为0
        num_static_real_features: int = 0,  # 静态实数特征数量，默认为0
        cardinality: Optional[List[int]] = None,  # 类别数量列表，默认为None
        embedding_dimension: Optional[List[int]] = None,  # 嵌入维度列表，默认为None
        d_model: int = 64,  # 模型维度，默认为64
        encoder_attention_heads: int = 2,  # 编码器注意力头数，默认为2
        decoder_attention_heads: int = 2,  # 解码器注意力头数，默认为2
        encoder_layers: int = 2,  # 编码器层数，默认为2
        decoder_layers: int = 2,  # 解码器层数，默认为2
        encoder_ffn_dim: int = 32,  # 编码器前馈网络维度，默认为32
        decoder_ffn_dim: int = 32,  # 解码器前馈网络维度，默认为32
        activation_function: str = "gelu",  # 激活函数类型，默认为"gelu"
        dropout: float = 0.1,  # 丢弃率，默认为0.1
        encoder_layerdrop: float = 0.1,  # 编码器层丢弃率，默认为0.1
        decoder_layerdrop: float = 0.1,  # 解码器层丢弃率，默认为0.1
        attention_dropout: float = 0.1,  # 注意力丢弃率，默认为0.1
        activation_dropout: float = 0.1,  # 激活丢弃率，默认为0.1
        num_parallel_samples: int = 100,  # 并行采样数量，默认为100
        init_std: float = 0.02,  # 初始化标准差，默认为0.02
        use_cache: bool = True,  # 是否使用缓存，默认为True
        is_encoder_decoder=True,  # 是否为编码器-解码器模型，默认为True
        # Autoformer参数
        label_length: int = 10,  # 标签长度，默认为10
        moving_average: int = 25,  # 移动平均值，默认为25
        autocorrelation_factor: int = 3,  # 自相关因子，���认为3
        **kwargs,  # 其他参数
        # 设置时间序列特定的配置参数
        self.prediction_length = prediction_length
        self.context_length = context_length if context_length is not None else prediction_length
        self.distribution_output = distribution_output
        self.loss = loss
        self.input_size = input_size
        self.num_time_features = num_time_features
        self.lags_sequence = lags_sequence
        self.scaling = scaling
        self.num_dynamic_real_features = num_dynamic_real_features
        self.num_static_real_features = num_static_real_features
        self.num_static_categorical_features = num_static_categorical_features
        
        # 检查并设置静态分类特征的基数和嵌入维度
        if cardinality is not None and num_static_categorical_features > 0:
            if len(cardinality) != num_static_categorical_features:
                raise ValueError(
                    "The cardinality should be a list of the same length as `num_static_categorical_features`"
                )
            self.cardinality = cardinality
        else:
            self.cardinality = [0]
        
        if embedding_dimension is not None and num_static_categorical_features > 0:
            if len(embedding_dimension) != num_static_categorical_features:
                raise ValueError(
                    "The embedding dimension should be a list of the same length as `num_static_categorical_features`"
                )
            self.embedding_dimension = embedding_dimension
        else:
            self.embedding_dimension = [min(50, (cat + 1) // 2) for cat in self.cardinality]
        
        self.num_parallel_samples = num_parallel_samples

        # 设置Transformer架构的配置参数
        self.feature_size = input_size * len(self.lags_sequence) + self._number_of_features
        self.d_model = d_model
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_attention_heads = decoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.decoder_ffn_dim = decoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers

        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop

        self.activation_function = activation_function
        self.init_std = init_std

        self.use_cache = use_cache

        # 设置Autoformer相关参数
        self.label_length = label_length
        self.moving_average = moving_average
        self.autocorrelation_factor = autocorrelation_factor

        # 调用父类的初始化方法
        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)
    # 定义一个私有方法，用于计算特征的数量，返回一个整数值
    def _number_of_features(self) -> int:
        # 计算嵌入维度列表中所有元素的总和
        return (
            sum(self.embedding_dimension)
            # 加上动态实数特征的数量
            + self.num_dynamic_real_features
            # 加上时间特征的数量
            + self.num_time_features
            # 加上静态实数特征的数量
            + self.num_static_real_features
            # 加上输入大小的两倍，即 log1p(abs(loc)) 和 log(scale) 特征的数量
            + self.input_size * 2  # the log1p(abs(loc)) and log(scale) features
        )
```