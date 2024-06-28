# `.\models\informer\configuration_informer.py`

```
# coding=utf-8
# 定义模型配置的文件，声明版权信息和许可证信息

"""Informer model configuration"""
# 引入必要的库和模块
from typing import List, Optional, Union

from ...configuration_utils import PretrainedConfig  # 导入预训练配置基类
from ...utils import logging  # 导入日志记录工具


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器对象

# 预训练配置文件映射字典，包含预训练模型的名称和对应的配置文件URL
INFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "huggingface/informer-tourism-monthly": (
        "https://huggingface.co/huggingface/informer-tourism-monthly/resolve/main/config.json"
    ),
    # 查看所有 Infromer 模型的信息，地址：https://huggingface.co/models?filter=informer
}


class InformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`InformerModel`]. It is used to instantiate an
    Informer model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Informer
    [huggingface/informer-tourism-monthly](https://huggingface.co/huggingface/informer-tourism-monthly) architecture.

    Configuration objects inherit from [`PretrainedConfig`] can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```python
    >>> from transformers import InformerConfig, InformerModel

    >>> # Initializing an Informer configuration with 12 time steps for prediction
    >>> configuration = InformerConfig(prediction_length=12)

    >>> # Randomly initializing a model (with random weights) from the configuration
    >>> model = InformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """
    # 模型类型标识
    model_type = "informer"
    # 属性映射字典，将配置参数转换为模型需要的参数名
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
        "num_hidden_layers": "encoder_layers",
    }
    # 定义一个初始化方法，用于初始化模型的各种参数和设置
    def __init__(
        self,
        # 预测长度，可选参数，用于指定模型的预测长度
        prediction_length: Optional[int] = None,
        # 上下文长度，可选参数，用于指定模型的上下文长度
        context_length: Optional[int] = None,
        # 分布输出类型，字符串参数，默认为学生 t 分布
        distribution_output: str = "student_t",
        # 损失函数类型，字符串参数，默认为负对数似然损失
        loss: str = "nll",
        # 输入尺寸，整数参数，默认为 1
        input_size: int = 1,
        # 滞后序列，整数列表参数，可选参数，用于指定滞后序列
        lags_sequence: List[int] = None,
        # 缩放方法，字符串或布尔类型参数，默认为均值
        scaling: Optional[Union[str, bool]] = "mean",
        # 动态实数特征数，整数参数，默认为 0
        num_dynamic_real_features: int = 0,
        # 静态实数特征数，整数参数，默认为 0
        num_static_real_features: int = 0,
        # 静态分类特征数，整数参数，默认为 0
        num_static_categorical_features: int = 0,
        # 时间特征数，整数参数，默认为 0
        num_time_features: int = 0,
        # 基数，整数列表参数，可选参数，用于指定分类特征的基数
        cardinality: Optional[List[int]] = None,
        # 嵌入维度，整数列表参数，可选参数，用于指定嵌入特征的维度
        embedding_dimension: Optional[List[int]] = None,
        # 编码器模型的维度，整数参数，默认为 64
        d_model: int = 64,
        # 编码器前馈神经网络的维度，整数参数，默认为 32
        encoder_ffn_dim: int = 32,
        # 解码器前馈神经网络的维度，整数参数，默认为 32
        decoder_ffn_dim: int = 32,
        # 编码器注意力头数，整数参数，默认为 2
        encoder_attention_heads: int = 2,
        # 解码器注意力头数，整数参数，默认为 2
        decoder_attention_heads: int = 2,
        # 编码器层数，整数参数，默认为 2
        encoder_layers: int = 2,
        # 解码器层数，整数参数，默认为 2
        decoder_layers: int = 2,
        # 是否是编码器-解码器结构，布尔类型参数，默认为 True
        is_encoder_decoder: bool = True,
        # 激活函数类型，字符串参数，默认为 GELU
        activation_function: str = "gelu",
        # 普通 dropout 概率，浮点数参数，默认为 0.05
        dropout: float = 0.05,
        # 编码器层 dropout 概率，浮点数参数，默认为 0.1
        encoder_layerdrop: float = 0.1,
        # 解码器层 dropout 概率，浮点数参数，默认为 0.1
        decoder_layerdrop: float = 0.1,
        # 注意力 dropout 概率，浮点数参数，默认为 0.1
        attention_dropout: float = 0.1,
        # 激活函数 dropout 概率，浮点数参数，默认为 0.1
        activation_dropout: float = 0.1,
        # 并行采样数，整数参数，默认为 100
        num_parallel_samples: int = 100,
        # 初始化标准差，浮点数参数，默认为 0.02
        init_std: float = 0.02,
        # 是否使用缓存，布尔类型参数，默认为 True
        use_cache=True,
        # Informer 模型特有参数
        # 注意类型，字符串参数，默认为 "prob"
        attention_type: str = "prob",
        # 采样因子，整数参数，默认为 5
        sampling_factor: int = 5,
        # 是否蒸馏，布尔类型参数，默认为 True
        distil: bool = True,
        # 其他参数，字典参数，用于接收额外的关键字参数
        **kwargs,
        # time series specific configuration
        self.prediction_length = prediction_length  # 设置预测长度
        self.context_length = context_length or prediction_length  # 设置上下文长度，如果未提供则默认与预测长度相同
        self.distribution_output = distribution_output  # 设置分布输出类型
        self.loss = loss  # 设置损失函数类型
        self.input_size = input_size  # 设置输入特征的大小
        self.num_time_features = num_time_features  # 设置时间特征的数量
        self.lags_sequence = lags_sequence if lags_sequence is not None else [1, 2, 3, 4, 5, 6, 7]  # 设置滞后序列，如果未提供则默认为指定的序列
        self.scaling = scaling  # 设置是否进行数据缩放
        self.num_dynamic_real_features = num_dynamic_real_features  # 设置动态实数特征的数量
        self.num_static_real_features = num_static_real_features  # 设置静态实数特征的数量
        self.num_static_categorical_features = num_static_categorical_features  # 设置静态分类特征的数量

        # set cardinality
        if cardinality and num_static_categorical_features > 0:
            if len(cardinality) != num_static_categorical_features:
                raise ValueError(
                    "The cardinality should be a list of the same length as `num_static_categorical_features`"
                )
            self.cardinality = cardinality  # 设置分类特征的基数（类别数量）
        else:
            self.cardinality = [0]  # 如果未提供分类特征或数量为零，则设置基数为零

        # set embedding_dimension
        if embedding_dimension and num_static_categorical_features > 0:
            if len(embedding_dimension) != num_static_categorical_features:
                raise ValueError(
                    "The embedding dimension should be a list of the same length as `num_static_categorical_features`"
                )
            self.embedding_dimension = embedding_dimension  # 设置嵌入维度
        else:
            # 计算默认嵌入维度，每个分类特征的嵌入维度最大为50，最小为其基数加1再除以2
            self.embedding_dimension = [min(50, (cat + 1) // 2) for cat in self.cardinality]

        self.num_parallel_samples = num_parallel_samples  # 设置并行采样的数量

        # Transformer architecture configuration
        self.feature_size = input_size * len(self.lags_sequence) + self._number_of_features  # 计算特征大小
        self.d_model = d_model  # 设置Transformer模型的维度大小
        self.encoder_attention_heads = encoder_attention_heads  # 设置编码器注意力头数
        self.decoder_attention_heads = decoder_attention_heads  # 设置解码器注意力头数
        self.encoder_ffn_dim = encoder_ffn_dim  # 设置编码器前馈神经网络的维度
        self.decoder_ffn_dim = decoder_ffn_dim  # 设置解码器前馈神经网络的维度
        self.encoder_layers = encoder_layers  # 设置编码器层数
        self.decoder_layers = decoder_layers  # 设置解码器层数

        self.dropout = dropout  # 设置全局的dropout比率
        self.attention_dropout = attention_dropout  # 设置注意力层的dropout比率
        self.activation_dropout = activation_dropout  # 设置激活函数的dropout比率
        self.encoder_layerdrop = encoder_layerdrop  # 设置编码器层的随机丢弃比率
        self.decoder_layerdrop = decoder_layerdrop  # 设置解码器层的随机丢弃比率

        self.activation_function = activation_function  # 设置激活函数类型
        self.init_std = init_std  # 设置初始化标准差

        self.use_cache = use_cache  # 设置是否使用缓存

        # Informer
        self.attention_type = attention_type  # 设置注意力机制的类型
        self.sampling_factor = sampling_factor  # 设置采样因子
        self.distil = distil  # 设置是否使用蒸馏技术

        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)  # 调用父类的初始化方法
    # 定义一个私有方法，用于计算特征的总数并返回整数类型的结果
    def _number_of_features(self) -> int:
        # 计算嵌入维度列表中所有元素的总和
        return (
            sum(self.embedding_dimension)  # 加上嵌入维度的总和
            + self.num_dynamic_real_features  # 加上动态实数特征的数量
            + self.num_time_features  # 加上时间特征的数量
            + self.num_static_real_features  # 加上静态实数特征的数量
            + self.input_size * 2  # 加上输入大小乘以2，代表 log1p(abs(loc)) 和 log(scale) 特征
        )
```