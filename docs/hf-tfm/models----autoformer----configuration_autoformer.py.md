# `.\models\autoformer\configuration_autoformer.py`

```
# 设置编码格式为 UTF-8

# 版权声明，声明此代码的版权归 HuggingFace Inc. 团队所有，保留所有权利。
# 根据 Apache License, Version 2.0 许可证使用本文件。您可以在符合许可证的情况下使用此文件，
# 您可以获取许可证的副本，具体网址在 http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则本软件根据“原样”分发，无任何明示或暗示的担保或条件。
# 有关更多信息，请参见许可证文档。

""" Autoformer model configuration"""

# 引入必要的模块
from typing import List, Optional
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义预训练配置文件的映射
AUTOFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "huggingface/autoformer-tourism-monthly": "https://huggingface.co/huggingface/autoformer-tourism-monthly/resolve/main/config.json",
}

# Autoformer 配置类，继承自 PretrainedConfig 类
class AutoformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`AutoformerModel`]. It is used to instantiate an
    Autoformer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Autoformer
    [huggingface/autoformer-tourism-monthly](https://huggingface.co/huggingface/autoformer-tourism-monthly)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    ```python
    >>> from transformers import AutoformerConfig, AutoformerModel

    >>> # Initializing a default Autoformer configuration
    >>> configuration = AutoformerConfig()

    >>> # Randomly initializing a model (with random weights) from the configuration
    >>> model = AutoformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # 模型类型为 "autoformer"
    model_type = "autoformer"

    # 属性映射字典，将 AutoformerConfig 类的属性映射到预训练模型的配置中
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
        "num_hidden_layers": "encoder_layers",
    }
    # 初始化函数，用于设置模型的各种参数和默认值
    def __init__(
        self,
        prediction_length: Optional[int] = None,  # 预测长度，可选参数，默认为 None
        context_length: Optional[int] = None,     # 上下文长度，可选参数，默认为 None
        distribution_output: str = "student_t",   # 分布输出类型，默认为 "student_t"
        loss: str = "nll",                        # 损失函数类型，默认为 "nll"
        input_size: int = 1,                      # 输入数据的维度，默认为 1
        lags_sequence: List[int] = [1, 2, 3, 4, 5, 6, 7],  # 滞后序列，列表，默认为 [1, 2, 3, 4, 5, 6, 7]
        scaling: bool = True,                     # 是否进行数据缩放，默认为 True
        num_time_features: int = 0,               # 时间特征的数量，默认为 0
        num_dynamic_real_features: int = 0,       # 动态实数特征的数量，默认为 0
        num_static_categorical_features: int = 0, # 静态分类特征的数量，默认为 0
        num_static_real_features: int = 0,        # 静态实数特征的数量，默认为 0
        cardinality: Optional[List[int]] = None,  # 分类特征的基数，可选参数，默认为 None
        embedding_dimension: Optional[List[int]] = None,  # 嵌入维度，可选参数，默认为 None
        d_model: int = 64,                        # 模型的维度，默认为 64
        encoder_attention_heads: int = 2,         # 编码器注意力头的数量，默认为 2
        decoder_attention_heads: int = 2,         # 解码器注意力头的数量，默认为 2
        encoder_layers: int = 2,                  # 编码器层数，默认为 2
        decoder_layers: int = 2,                  # 解码器层数，默认为 2
        encoder_ffn_dim: int = 32,                # 编码器中 FFN 层的维度，默认为 32
        decoder_ffn_dim: int = 32,                # 解码器中 FFN 层的维度，默认为 32
        activation_function: str = "gelu",        # 激活函数类型，默认为 "gelu"
        dropout: float = 0.1,                     # 通用的 dropout 比例，默认为 0.1
        encoder_layerdrop: float = 0.1,           # 编码器层 dropout 比例，默认为 0.1
        decoder_layerdrop: float = 0.1,           # 解码器层 dropout 比例，默认为 0.1
        attention_dropout: float = 0.1,           # 注意力机制的 dropout 比例，默认为 0.1
        activation_dropout: float = 0.1,          # 激活函数的 dropout 比例，默认为 0.1
        num_parallel_samples: int = 100,          # 并行采样数量，默认为 100
        init_std: float = 0.02,                   # 初始化标准差，默认为 0.02
        use_cache: bool = True,                   # 是否使用缓存，默认为 True
        is_encoder_decoder=True,                  # 是否是编码器-解码器结构，默认为 True
        # Autoformer 参数
        label_length: int = 10,                   # 标签长度，默认为 10
        moving_average: int = 25,                 # 移动平均窗口大小，默认为 25
        autocorrelation_factor: int = 3,          # 自相关因子，默认为 3
        **kwargs,                                 # 其他未指定的参数，作为字典接收
        # 时间序列特定配置
        self.prediction_length = prediction_length  # 设置预测长度
        self.context_length = context_length if context_length is not None else prediction_length  # 设置上下文长度，默认为预测长度
        self.distribution_output = distribution_output  # 分布输出配置
        self.loss = loss  # 损失函数配置
        self.input_size = input_size  # 输入尺寸
        self.num_time_features = num_time_features  # 时间特征数量
        self.lags_sequence = lags_sequence  # 滞后序列配置
        self.scaling = scaling  # 是否进行缩放处理
        self.num_dynamic_real_features = num_dynamic_real_features  # 动态实数特征数量
        self.num_static_real_features = num_static_real_features  # 静态实数特征数量
        self.num_static_categorical_features = num_static_categorical_features  # 静态分类特征数量
        if cardinality is not None and num_static_categorical_features > 0:
            if len(cardinality) != num_static_categorical_features:
                raise ValueError(
                    "The cardinality should be a list of the same length as `num_static_categorical_features`"
                )
            self.cardinality = cardinality  # 静态分类特征的基数列表
        else:
            self.cardinality = [0]  # 默认基数为0，表示无静态分类特征
        if embedding_dimension is not None and num_static_categorical_features > 0:
            if len(embedding_dimension) != num_static_categorical_features:
                raise ValueError(
                    "The embedding dimension should be a list of the same length as `num_static_categorical_features`"
                )
            self.embedding_dimension = embedding_dimension  # 静态分类特征的嵌入维度列表
        else:
            self.embedding_dimension = [min(50, (cat + 1) // 2) for cat in self.cardinality]  # 默认嵌入维度计算
        self.num_parallel_samples = num_parallel_samples  # 并行采样数量设置

        # Transformer 架构配置
        self.feature_size = input_size * len(self.lags_sequence) + self._number_of_features  # 特征大小计算
        self.d_model = d_model  # Transformer 模型的维度
        self.encoder_attention_heads = encoder_attention_heads  # 编码器注意力头数
        self.decoder_attention_heads = decoder_attention_heads  # 解码器注意力头数
        self.encoder_ffn_dim = encoder_ffn_dim  # 编码器前馈网络维度
        self.decoder_ffn_dim = decoder_ffn_dim  # 解码器前馈网络维度
        self.encoder_layers = encoder_layers  # 编码器层数
        self.decoder_layers = decoder_layers  # 解码器层数

        self.dropout = dropout  # 普通的dropout率
        self.attention_dropout = attention_dropout  # 注意力机制中的dropout率
        self.activation_dropout = activation_dropout  # 激活函数的dropout率
        self.encoder_layerdrop = encoder_layerdrop  # 编码器层级dropout率
        self.decoder_layerdrop = decoder_layerdrop  # 解码器层级dropout率

        self.activation_function = activation_function  # 激活函数类型
        self.init_std = init_std  # 初始化标准差

        self.use_cache = use_cache  # 是否使用缓存

        # Autoformer
        self.label_length = label_length  # 标签长度
        self.moving_average = moving_average  # 移动平均配置
        self.autocorrelation_factor = autocorrelation_factor  # 自相关因子配置

        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)  # 调用父类初始化函数
    # 定义一个私有方法 `_number_of_features`，返回整数类型的值
    def _number_of_features(self) -> int:
        # 计算所有嵌入维度的总和
        return (
            sum(self.embedding_dimension)
            # 加上动态实数特征的数量
            + self.num_dynamic_real_features
            # 加上时间特征的数量
            + self.num_time_features
            # 加上静态实数特征的数量
            + self.num_static_real_features
            # 加上输入大小的两倍，代表 log1p(abs(loc)) 和 log(scale) 特征
            + self.input_size * 2
        )
```