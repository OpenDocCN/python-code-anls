# `.\models\informer\configuration_informer.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制

"""Informer 模型配置"""

from typing import List, Optional, Union

# 导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 导入日志工具
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# Informer 预训练配置存档映射
INFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "huggingface/informer-tourism-monthly": (
        "https://huggingface.co/huggingface/informer-tourism-monthly/resolve/main/config.json"
    ),
    # 查看所有 Informer 模型 https://huggingface.co/models?filter=informer
}

# Informer 配置类，继承自预训练配置类
class InformerConfig(PretrainedConfig):
    r"""
    这是用于存储 [`InformerModel`] 配置的配置类。根据指定的参数实例化 Informer 模型，定义模型架构。
    使用默认值实例化配置将产生类似 Informer [huggingface/informer-tourism-monthly](https://huggingface.co/huggingface/informer-tourism-monthly) 架构的配置。

    继承自 [`PretrainedConfig`] 的配置对象可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    示例:

    ```python
    >>> from transformers import InformerConfig, InformerModel

    >>> # 初始化一个具有 12 个时间步长的预测 Informer 配置
    >>> configuration = InformerConfig(prediction_length=12)

    >>> # 从配置随机初始化一个模型（具有随机权重）
    >>> model = InformerModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```"""

    model_type = "informer"
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
        "num_hidden_layers": "encoder_layers",
    }
    # 初始化函数，设置模型的各种参数
    def __init__(
        self,
        # 预测长度，默认为 None
        prediction_length: Optional[int] = None,
        # 上下文长度，默认为 None
        context_length: Optional[int] = None,
        # 分布输出类型，默认为 "student_t"
        distribution_output: str = "student_t",
        # 损失函数，默认为 "nll"
        loss: str = "nll",
        # 输入大小，默认为 1
        input_size: int = 1,
        # 滞后序列，默认为 None
        lags_sequence: List[int] = None,
        # 缩放方式，默认为 "mean"
        scaling: Optional[Union[str, bool]] = "mean",
        # 动态实数特征数量，默认为 0
        num_dynamic_real_features: int = 0,
        # 静态实数特征数量，默认为 0
        num_static_real_features: int = 0,
        # 静态分类特征数量，默认为 0
        num_static_categorical_features: int = 0,
        # 时间特征数量，默认为 0
        num_time_features: int = 0,
        # 类别数量，默认为 None
        cardinality: Optional[List[int]] = None,
        # 嵌入维度，默认为 None
        embedding_dimension: Optional[List[int]] = None,
        # 编码器模型大小，默认为 64
        d_model: int = 64,
        # 编码器前馈网络维度，默认为 32
        encoder_ffn_dim: int = 32,
        # 解码器前馈网络维度，默认为 32
        decoder_ffn_dim: int = 32,
        # 编码器注意力头数，默认为 2
        encoder_attention_heads: int = 2,
        # 解码器注意力头数，默认为 2
        decoder_attention_heads: int = 2,
        # 编码器层数，默认为 2
        encoder_layers: int = 2,
        # 解码器层数，默认为 2
        decoder_layers: int = 2,
        # 是否为编码器-解码器模型，默认为 True
        is_encoder_decoder: bool = True,
        # 激活函数，默认为 "gelu"
        activation_function: str = "gelu",
        # 丢弃率，默认为 0.05
        dropout: float = 0.05,
        # 编码器层丢弃率，默认为 0.1
        encoder_layerdrop: float = 0.1,
        # 解码器层丢弃率，默认为 0.1
        decoder_layerdrop: float = 0.1,
        # 注意力丢弃率，默认为 0.1
        attention_dropout: float = 0.1,
        # 激活丢弃率，默认为 0.1
        activation_dropout: float = 0.1,
        # 并行采样数量，默认为 100
        num_parallel_samples: int = 100,
        # 初始化标准差，默认为 0.02
        init_std: float = 0.02,
        # 是否使用缓存，��认为 True
        use_cache=True,
        # Informer 参数
        # 注意力类型，默认为 "prob"
        attention_type: str = "prob",
        # 采样因子，默认为 5
        sampling_factor: int = 5,
        # 是否蒸馏，默认为 True
        distil: bool = True,
        # 其他参数
        **kwargs,
        # 设置时间序列特定配置
        self.prediction_length = prediction_length
        self.context_length = context_length or prediction_length
        self.distribution_output = distribution_output
        self.loss = loss
        self.input_size = input_size
        self.num_time_features = num_time_features
        self.lags_sequence = lags_sequence if lags_sequence is not None else [1, 2, 3, 4, 5, 6, 7]
        self.scaling = scaling
        self.num_dynamic_real_features = num_dynamic_real_features
        self.num_static_real_features = num_static_real_features
        self.num_static_categorical_features = num_static_categorical_features

        # 设置基数
        if cardinality and num_static_categorical_features > 0:
            if len(cardinality) != num_static_categorical_features:
                raise ValueError(
                    "The cardinality should be a list of the same length as `num_static_categorical_features`"
                )
            self.cardinality = cardinality
        else:
            self.cardinality = [0]

        # 设置嵌入维度
        if embedding_dimension and num_static_categorical_features > 0:
            if len(embedding_dimension) != num_static_categorical_features:
                raise ValueError(
                    "The embedding dimension should be a list of the same length as `num_static_categorical_features`"
                )
            self.embedding_dimension = embedding_dimension
        else:
            self.embedding_dimension = [min(50, (cat + 1) // 2) for cat in self.cardinality]

        self.num_parallel_samples = num_parallel_samples

        # 设置Transformer架构配置
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

        # 设置Informer
        self.attention_type = attention_type
        self.sampling_factor = sampling_factor
        self.distil = distil

        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)
    # 返回特征的数量，包括嵌入维度之和、动态实数特征数量、时间特征数量、静态实数特征数量、以及输入大小乘以2（log1p(abs(loc))和log(scale)特征）
    def _number_of_features(self) -> int:
        return (
            sum(self.embedding_dimension)  # 计算嵌入维度之和
            + self.num_dynamic_real_features  # 加上动态实数特征数量
            + self.num_time_features  # 加上时间特征数量
            + self.num_static_real_features  # 加上静态实数特征数量
            + self.input_size * 2  # 加上输入大小乘以2（log1p(abs(loc))和log(scale)特征）
        )
```