# `.\models\time_series_transformer\configuration_time_series_transformer.py`

```py
# 设置文件编码为 UTF-8

# 版权声明和许可信息，指定此代码的使用条款和条件
# 根据 Apache License, Version 2.0 许可，除非符合许可条件，否则不得使用此文件
# 可以在下面的链接获取完整的许可证文本：http://www.apache.org/licenses/LICENSE-2.0

# 引入必要的模块和类
from typing import List, Optional, Union

# 从 Transformers 库中导入预训练配置类 PretrainedConfig
from ...configuration_utils import PretrainedConfig
# 从 Transformers 库中导入日志记录工具
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义一个映射，将预训练模型名称映射到其配置文件的 URL
TIME_SERIES_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "huggingface/time-series-transformer-tourism-monthly": (
        "https://huggingface.co/huggingface/time-series-transformer-tourism-monthly/resolve/main/config.json"
    ),
    # 更多 TimeSeriesTransformer 模型可以在 https://huggingface.co/models?filter=time_series_transformer 查看
}

# TimeSeriesTransformerConfig 类，继承自 PretrainedConfig 类，用于存储时间序列 Transformer 模型的配置信息
class TimeSeriesTransformerConfig(PretrainedConfig):
    r"""
    这是用于存储 [`TimeSeriesTransformerModel`] 配置的类。根据指定的参数实例化一个 Time Series Transformer 模型的配置，
    定义模型架构。使用默认配置实例化将产生类似于 Time Series Transformer
    [huggingface/time-series-transformer-tourism-monthly](https://huggingface.co/huggingface/time-series-transformer-tourism-monthly)
    架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可以用于控制模型的输出。查阅 [`PretrainedConfig`] 的文档以获取更多信息。

    ```
    >>> from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel

    >>> # 使用 12 个时间步进行预测初始化 Time Series Transformer 配置
    >>> configuration = TimeSeriesTransformerConfig(prediction_length=12)

    >>> # 从配置中随机初始化一个模型（带有随机权重）
    >>> model = TimeSeriesTransformerModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```
    """

    # 模型类型标识为 "time_series_transformer"
    model_type = "time_series_transformer"

    # 属性映射字典，将配置属性名映射到模型参数名
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
        "num_hidden_layers": "encoder_layers",
    }
    # 初始化函数，用于设置模型的参数和配置
    def __init__(
        self,
        prediction_length: Optional[int] = None,  # 预测长度，可以为None
        context_length: Optional[int] = None,  # 上下文长度，可以为None
        distribution_output: str = "student_t",  # 输出分布类型，默认为学生t分布
        loss: str = "nll",  # 损失函数类型，默认为负对数似然
        input_size: int = 1,  # 输入数据的维度，默认为1
        lags_sequence: List[int] = [1, 2, 3, 4, 5, 6, 7],  # 滞后序列，用于特征提取，默认包含1到7的整数
        scaling: Optional[Union[str, bool]] = "mean",  # 数据缩放方式，默认为均值缩放
        num_dynamic_real_features: int = 0,  # 动态实数特征数量，默认为0
        num_static_categorical_features: int = 0,  # 静态分类特征数量，默认为0
        num_static_real_features: int = 0,  # 静态实数特征数量，默认为0
        num_time_features: int = 0,  # 时间特征数量，默认为0
        cardinality: Optional[List[int]] = None,  # 分类特征的基数列表，可以为None
        embedding_dimension: Optional[List[int]] = None,  # 嵌入层的维度列表，可以为None
        encoder_ffn_dim: int = 32,  # 编码器前馈神经网络的维度，默认为32
        decoder_ffn_dim: int = 32,  # 解码器前馈神经网络的维度，默认为32
        encoder_attention_heads: int = 2,  # 编码器注意力头数，默认为2
        decoder_attention_heads: int = 2,  # 解码器注意力头数，默认为2
        encoder_layers: int = 2,  # 编码器层数，默认为2
        decoder_layers: int = 2,  # 解码器层数，默认为2
        is_encoder_decoder: bool = True,  # 是否是编码解码结构，默认为True
        activation_function: str = "gelu",  # 激活函数类型，默认为GELU
        d_model: int = 64,  # 模型维度，默认为64
        dropout: float = 0.1,  # dropout率，默认为0.1
        encoder_layerdrop: float = 0.1,  # 编码器层dropout率，默认为0.1
        decoder_layerdrop: float = 0.1,  # 解码器层dropout率，默认为0.1
        attention_dropout: float = 0.1,  # 注意力机制的dropout率，默认为0.1
        activation_dropout: float = 0.1,  # 激活函数的dropout率，默认为0.1
        num_parallel_samples: int = 100,  # 并行采样的样本数量，默认为100
        init_std: float = 0.02,  # 初始化的标准差，默认为0.02
        use_cache=True,  # 是否使用缓存，默认为True
        **kwargs,  # 其他可选参数
        # time series specific configuration
        self.prediction_length = prediction_length
        # 设置预测长度，用于时间序列预测模型
        self.context_length = context_length or prediction_length
        # 设置上下文长度，若未提供则默认与预测长度相同
        self.distribution_output = distribution_output
        # 分布输出配置，指定预测分布的类型（如正态分布、负二项分布等）
        self.loss = loss
        # 损失函数配置，用于模型训练时的优化目标
        self.input_size = input_size
        # 输入特征的大小
        self.num_time_features = num_time_features
        # 时间特征的数量
        self.lags_sequence = lags_sequence
        # 滞后序列的配置，用于时间序列模型的输入
        self.scaling = scaling
        # 是否进行数据缩放的标志
        self.num_dynamic_real_features = num_dynamic_real_features
        # 动态实数特征的数量
        self.num_static_real_features = num_static_real_features
        # 静态实数特征的数量
        self.num_static_categorical_features = num_static_categorical_features
        # 静态分类特征的数量
        if cardinality and num_static_categorical_features > 0:
            if len(cardinality) != num_static_categorical_features:
                raise ValueError(
                    "The cardinality should be a list of the same length as `num_static_categorical_features`"
                )
            self.cardinality = cardinality
        else:
            self.cardinality = [0]
        # 静态分类特征的基数，用于嵌入编码
        if embedding_dimension and num_static_categorical_features > 0:
            if len(embedding_dimension) != num_static_categorical_features:
                raise ValueError(
                    "The embedding dimension should be a list of the same length as `num_static_categorical_features`"
                )
            self.embedding_dimension = embedding_dimension
        else:
            self.embedding_dimension = [min(50, (cat + 1) // 2) for cat in self.cardinality]
        # 静态分类特征的嵌入维度配置
        self.num_parallel_samples = num_parallel_samples
        # 并行采样的数量

        # Transformer architecture configuration
        self.feature_size = input_size * len(lags_sequence) + self._number_of_features
        # 特征向量的大小，由输入特征、滞后序列长度和额外特征共同决定
        self.d_model = d_model
        # Transformer 模型的隐藏层维度
        self.encoder_attention_heads = encoder_attention_heads
        # 编码器注意力头的数量
        self.decoder_attention_heads = decoder_attention_heads
        # 解码器注意力头的数量
        self.encoder_ffn_dim = encoder_ffn_dim
        # 编码器前馈神经网络的维度
        self.decoder_ffn_dim = decoder_ffn_dim
        # 解码器前馈神经网络的维度
        self.encoder_layers = encoder_layers
        # 编码器层数
        self.decoder_layers = decoder_layers
        # 解码器层数

        self.dropout = dropout
        # 总体丢弃率
        self.attention_dropout = attention_dropout
        # 注意力机制的丢弃率
        self.activation_dropout = activation_dropout
        # 激活函数的丢弃率
        self.encoder_layerdrop = encoder_layerdrop
        # 编码器层的丢弃率
        self.decoder_layerdrop = decoder_layerdrop
        # 解码器层的丢弃率

        self.activation_function = activation_function
        # 激活函数的选择
        self.init_std = init_std
        # 初始化标准差

        self.use_cache = use_cache
        # 是否使用缓存

        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)
        # 调用父类初始化方法，设置是否为编码-解码模型等参数

    @property
    def _number_of_features(self) -> int:
        return (
            sum(self.embedding_dimension)
            + self.num_dynamic_real_features
            + self.num_time_features
            + self.num_static_real_features
            + self.input_size * 2  # the log1p(abs(loc)) and log(scale) features
        )
        # 计算特征数量，包括嵌入维度、动态实数特征、时间特征、静态实数特征和额外特征
```