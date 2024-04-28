# `.\transformers\models\time_series_transformer\configuration_time_series_transformer.py`

```
# coding=utf-8
# 版权声明

# 从 typing 模块引入 List、Optional 和 Union 类型
from typing import List, Optional, Union

# 从父级目录中导入 PretrainedConfig 类
from ...configuration_utils import PretrainedConfig
# 从工具类中导入 logging 函数
from ...utils import logging

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 预训练配置文件的映射，包含了时间序列变压器模型的预训练配置文件链接
TIME_SERIES_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "huggingface/time-series-transformer-tourism-monthly": (
        "https://huggingface.co/huggingface/time-series-transformer-tourism-monthly/resolve/main/config.json"
    ),
    # 查看所有时间序列变压器模型：https://huggingface.co/models?filter=time_series_transformer
}

# 时间序列变压器模型配置类，继承自 PretrainedConfig 类
class TimeSeriesTransformerConfig(PretrainedConfig):
    r"""
    这是用于存储 [`TimeSeriesTransformerModel`] 的配置的配置类。用于根据指定的参数实例化时间序列变压器模型，定义模型架构。
    使用默认值实例化配置将产生类似于时间序列变压器
    [huggingface/time-series-transformer-tourism-monthly](https://huggingface.co/huggingface/time-series-transformer-tourism-monthly)
    架构的配置。

    继承自 [`PretrainedConfig`] 的配置对象可用于控制模型输出。阅读来自 [`PretrainedConfig`] 的文档以获取更多信息。

    ```python
    >>> from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel

    >>> # 初始化具有12个时间步长的预测时间序列变压器配置
    >>> configuration = TimeSeriesTransformerConfig(prediction_length=12)

    >>> # 从配置随机初始化模型（带有随机权重）
    >>> model = TimeSeriesTransformerModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```"""
    
    # 模型类型为"time_series_transformer"
    model_type = "time_series_transformer"
    # 属性映射
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
        "num_hidden_layers": "encoder_layers",
    }
    # 初始化函数，用于初始化模型的参数
    def __init__(
        # 设置预测长度的可选参数，默认值为None
        self,
        prediction_length: Optional[int] = None,
        # 设置上下文长度的可选参数，默认值为None
        context_length: Optional[int] = None,
        # 设置分布输出的类型，默认为学生t分布
        distribution_output: str = "student_t",
        # 设置损失函数的类型，默认为负对数似然损失
        loss: str = "nll",
        # 设置输入尺寸，默认为1
        input_size: int = 1,
        # 设置时间序列的滞后值，默认为[1, 2, 3, 4, 5, 6, 7]
        lags_sequence: List[int] = [1, 2, 3, 4, 5, 6, 7],
        # 设置归一化方式的可选参数，默认为平均值
        scaling: Optional[Union[str, bool]] = "mean",
        # 设置动态实数特征的数量，默认为0
        num_dynamic_real_features: int = 0,
        # 设置静态分类特征的数量，默认为0
        num_static_categorical_features: int = 0,
        # 设置静态实数特征的数量，默认为0
        num_static_real_features: int = 0,
        # 设置时间特征的数量，默认为0
        num_time_features: int = 0,
        # 设置类别的基数，可选参数条件下为None
        cardinality: Optional[List[int]] = None,
        # 设置嵌入维度的列表，可选参数条件下为None
        embedding_dimension: Optional[List[int]] = None,
        # 设置编码器前馈神经网络层的维度，默认为32
        encoder_ffn_dim: int = 32,
        # 设置解码器前馈神经网络层的维度，默认为32
        decoder_ffn_dim: int = 32,
        # 设置编码器注意力头的数量，默认为2
        encoder_attention_heads: int = 2,
        # 设置解码器注意力头的数量，默认为2
        decoder_attention_heads: int = 2,
        # 设置编码器层的数量，默认为2
        encoder_layers: int = 2,
        # 设置解码器层的数量，默认为2
        decoder_layers: int = 2,
        # 设置是否是编码器解码器模型的布尔类型参数，默认为True
        is_encoder_decoder: bool = True,
        # 设置激活函数的类型，默认为GELU
        activation_function: str = "gelu",
        # 设置模型维度的整数值，默认为64
        d_model: int = 64,
        # 设置丢弃率，默认为0.1
        dropout: float = 0.1,
        # 设置编码器层丢弃率，默认为0.1
        encoder_layerdrop: float = 0.1,
        # 设置解码器层丢弃率，默认为0.1
        decoder_layerdrop: float = 0.1,
        # 设置注意力丢弃率，默认为0.1
        attention_dropout: float = 0.1,
        # 设置激活函数丢弃率，默认为0.1
        activation_dropout: float = 0.1,
        # 设置并行采样的数量，默认为100
        num_parallel_samples: int = 100,
        # 设置初始化标准差的值，默认为0.02
        init_std: float = 0.02,
        # 使用缓存的布尔类型参数，默认为True
        use_cache=True,
        # 对其他关键字参数进行传递
        **kwargs,
        # 初始化方法，设置时间序列特定配置
        # 设置预测长度
        self.prediction_length = prediction_length
        # 设置上下文长度，默认为预测长度
        self.context_length = context_length or prediction_length
        # 设置输出分布
        self.distribution_output = distribution_output
        # 设置损失函数
        self.loss = loss
        # 设置输入大小
        self.input_size = input_size
        # 设置时间特征数量
        self.num_time_features = num_time_features
        # 设置滞后序列
        self.lags_sequence = lags_sequence
        # 设置缩放
        self.scaling = scaling
        # 设置动态实数特征数量
        self.num_dynamic_real_features = num_dynamic_real_features
        # 设置静态实数特征数量
        self.num_static_real_features = num_static_real_features
        # 设置静态分类特征数量
        self.num_static_categorical_features = num_static_categorical_features
        # 如果存在基数和静态分类特征数量大于0
        if cardinality and num_static_categorical_features > 0:
            # 如果基数长度与静态分类特征数量不一致，则抛出异常
            if len(cardinality) != num_static_categorical_features:
                raise ValueError(
                    "The cardinality should be a list of the same length as `num_static_categorical_features`"
                )
            self.cardinality = cardinality
        else:
            self.cardinality = [0]
        # 如果存在嵌入维度和静态分类特征数量大于0
        if embedding_dimension and num_static_categorical_features > 0:
            # 如果嵌入维度长度与静态分类特征数量不一致，则抛出异常
            if len(embedding_dimension) != num_static_categorical_features:
                raise ValueError(
                    "The embedding dimension should be a list of the same length as `num_static_categorical_features`"
                )
            self.embedding_dimension = embedding_dimension
        else:
            # 计算默认嵌入维度：最小为50和 (类别数+1)//2 的最小值
            self.embedding_dimension = [min(50, (cat + 1) // 2) for cat in self.cardinality]
        # 设置并行采样数量
        self.num_parallel_samples = num_parallel_samples

        # Transformer 架构配置
        # 计算特征大小
        self.feature_size = input_size * len(lags_sequence) + self._number_of_features
        # 设置模型维度
        self.d_model = d_model
        # 设置编码器注意力头数
        self.encoder_attention_heads = encoder_attention_heads
        # 设置解码器注意力头数
        self.decoder_attention_heads = decoder_attention_heads
        # 设置编码器前馈神经网络维度
        self.encoder_ffn_dim = encoder_ffn_dim
        # 设置解码器前馈神经网络维度
        self.decoder_ffn_dim = decoder_ffn_dim
        # 设置编码器层数
        self.encoder_layers = encoder_layers
        # 设置解码器层数
        self.decoder_layers = decoder_layers

        # 设置丢弃率
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop

        # 设置激活函数
        self.activation_function = activation_function
        # 设置初始化标准差
        self.init_std = init_std

        # 使用缓存
        self.use_cache = use_cache

        # 调用父类的初始化方法
        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)

    @property
    def _number_of_features(self) -> int:
        # 计算特征数量：嵌入维度之和 + 动态实数特征数量 + 时间特征数量 + 静态实数特征数量 + 2倍的输入大小（log1p(abs(loc)) 和 log(scale) 特征）
        return (
            sum(self.embedding_dimension)
            + self.num_dynamic_real_features
            + self.num_time_features
            + self.num_static_real_features
            + self.input_size * 2  # the log1p(abs(loc)) and log(scale) features
        )
```