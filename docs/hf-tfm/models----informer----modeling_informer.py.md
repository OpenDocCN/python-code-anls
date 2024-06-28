# `.\models\informer\modeling_informer.py`

```py
# coding=utf-8
# Copyright 2023 Amazon and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Informer model."""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    SampleTSPredictionOutput,
    Seq2SeqTSModelOutput,
    Seq2SeqTSPredictionOutput,
)
from ...modeling_utils import PreTrainedModel
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_informer import InformerConfig

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 文档中使用的配置名
_CONFIG_FOR_DOC = "InformerConfig"

# 预训练模型的存档列表
INFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "huggingface/informer-tourism-monthly",
    # See all Informer models at https://huggingface.co/models?filter=informer
]


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesFeatureEmbedder with TimeSeries->Informer
class InformerFeatureEmbedder(nn.Module):
    """
    Embed a sequence of categorical features.

    Args:
        cardinalities (`list[int]`):
            List of cardinalities of the categorical features.
        embedding_dims (`list[int]`):
            List of embedding dimensions of the categorical features.
    """

    def __init__(self, cardinalities: List[int], embedding_dims: List[int]) -> None:
        super().__init__()

        # 计算输入的分类特征数量
        self.num_features = len(cardinalities)
        # 使用 nn.ModuleList 创建嵌入层列表，每个分类特征对应一个 Embedding 层
        self.embedders = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(cardinalities, embedding_dims)])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.num_features > 1:
            # 如果有多个特征，按最后一个维度切片特征张量
            cat_feature_slices = torch.chunk(features, self.num_features, dim=-1)
        else:
            cat_feature_slices = [features]

        # 将每个分类特征片段经过对应的 Embedding 层嵌入，并在最后一个维度上连接它们
        return torch.cat(
            [
                embed(cat_feature_slice.squeeze(-1))
                for embed, cat_feature_slice in zip(self.embedders, cat_feature_slices)
            ],
            dim=-1,
        )
# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesStdScaler with TimeSeriesTransformer->Informer,TimeSeries->Informer
class InformerStdScaler(nn.Module):
    """
    Standardize features by calculating the mean and scaling along the first dimension, and then normalizes it by
    subtracting from the mean and dividing by the standard deviation.
    """

    def __init__(self, config: InformerConfig):
        super().__init__()
        # 初始化标准化器对象，从配置中获取标准化维度，默认为第一维
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        # 确定是否保持维度大小不变，默认为True
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True
        # 最小缩放值，用于避免除以零或接近零的情况，默认为1e-5
        self.minimum_scale = config.minimum_scale if hasattr(config, "minimum_scale") else 1e-5

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
            observed_indicator (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Calculating the scale on the observed indicator.
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        # 计算分母，即观察指示器的和，用于标准差计算，保持维度不变
        denominator = observed_indicator.sum(self.dim, keepdim=self.keepdim)
        # 确保分母不小于1.0，避免除以零或接近零的情况
        denominator = denominator.clamp_min(1.0)
        # 计算数据的均值，仅在观察指示器为真时考虑数据贡献
        loc = (data * observed_indicator).sum(self.dim, keepdim=self.keepdim) / denominator

        # 计算方差，仅在观察指示器为真时考虑数据贡献，用于计算标准差
        variance = (((data - loc) * observed_indicator) ** 2).sum(self.dim, keepdim=self.keepdim) / denominator
        # 计算标准差，并加上最小缩放值以确保数值稳定性
        scale = torch.sqrt(variance + self.minimum_scale)
        # 标准化数据并返回标准化后的数据、均值和标准差
        return (data - loc) / scale, loc, scale


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesMeanScaler with TimeSeriesTransformer->Informer,TimeSeries->Informer
class InformerMeanScaler(nn.Module):
    """
    Computes a scaling factor as the weighted average absolute value along the first dimension, and scales the data
    accordingly.
    """

    def __init__(self, config: InformerConfig):
        super().__init__()
        # 初始化均值标准化器对象，从配置中获取标准化维度，默认为第一维
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        # 确定是否保持维度大小不变，默认为True
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True
        # 最小缩放值，用于避免除以零或接近零的情况，默认为1e-10
        self.minimum_scale = config.minimum_scale if hasattr(config, "minimum_scale") else 1e-10
        # 默认的缩放值，如果配置中未指定，默认为None
        self.default_scale = config.default_scale if hasattr(config, "default_scale") else None

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
            observed_indicator (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Calculating the scale on the observed indicator.
        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`

        """
        # 返回数据经过均值标准化处理后的结果
        return data / (data.abs().mean(self.dim, keepdim=self.keepdim) + self.minimum_scale)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
            observed_indicator (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Calculating the scale on the observed indicator.
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        # Calculate the sum of absolute values of data multiplied by observed_indicator,
        # along the specified dimension `self.dim`, keeping the dimensionality.
        ts_sum = (data * observed_indicator).abs().sum(self.dim, keepdim=True)

        # Count the number of observed elements (True values) in observed_indicator
        # along dimension `self.dim`, keeping the dimensionality.
        num_observed = observed_indicator.sum(self.dim, keepdim=True)

        # Compute the scale factor by dividing ts_sum by num_observed, clamping
        # num_observed to a minimum value of 1 to avoid division by zero.
        scale = ts_sum / torch.clamp(num_observed, min=1)

        # If `default_scale` is not provided, calculate it using the batch sum of ts_sum
        # and batch_observations, otherwise use the provided `default_scale`.
        if self.default_scale is None:
            batch_sum = ts_sum.sum(dim=0)
            batch_observations = torch.clamp(num_observed.sum(0), min=1)
            default_scale = torch.squeeze(batch_sum / batch_observations)
        else:
            default_scale = self.default_scale * torch.ones_like(scale)

        # Apply default_scale where num_observed is greater than zero, otherwise use scale.
        scale = torch.where(num_observed > 0, scale, default_scale)

        # Ensure the scale values are at least `self.minimum_scale`.
        scale = torch.clamp(scale, min=self.minimum_scale)

        # Normalize data by dividing each element by its corresponding scale.
        scaled_data = data / scale

        # If keepdim is False, squeeze the scale tensor along dimension `self.dim`.
        if not self.keepdim:
            scale = scale.squeeze(dim=self.dim)

        # Return scaled_data, a tensor of zeros with the same shape as scale, and scale itself.
        return scaled_data, torch.zeros_like(scale), scale
# 从 transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesNOPScaler 复制而来，改名为 InformerNOPScaler，
# 仅对名字进行了变更，并未改动其功能或逻辑
class InformerNOPScaler(nn.Module):
    """
    Assigns a scaling factor equal to 1 along the first dimension, and therefore applies no scaling to the input data.
    """

    def __init__(self, config: InformerConfig):
        super().__init__()
        # 根据配置文件，确定缩放的维度，默认为第一维
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        # 根据配置文件，确定是否保持维度，默认为 True
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        # 计算数据的缩放因子，沿着指定维度取均值，得到缩放因子
        scale = torch.ones_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        # 计算数据的均值，沿着指定维度取均值，得到均值
        loc = torch.zeros_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        # 返回原始数据、均值和缩放因子
        return data, loc, scale


# 从 transformers.models.time_series_transformer.modeling_time_series_transformer.weighted_average 复制而来
def weighted_average(input_tensor: torch.Tensor, weights: Optional[torch.Tensor] = None, dim=None) -> torch.Tensor:
    """
    Computes the weighted average of a given tensor across a given `dim`, masking values associated with weight zero,
    meaning instead of `nan * 0 = nan` you will get `0 * 0 = 0`.

    Args:
        input_tensor (`torch.FloatTensor`):
            Input tensor, of which the average must be computed.
        weights (`torch.FloatTensor`, *optional*):
            Weights tensor, of the same shape as `input_tensor`.
        dim (`int`, *optional*):
            The dim along which to average `input_tensor`.

    Returns:
        `torch.FloatTensor`: The tensor with values averaged along the specified `dim`.
    """
    if weights is not None:
        # 计算加权平均值，避免权重为零时的 NaN 问题，使用 torch.where 处理
        weighted_tensor = torch.where(weights != 0, input_tensor * weights, torch.zeros_like(input_tensor))
        # 计算非零权重的和，并进行最小值约束
        sum_weights = torch.clamp(weights.sum(dim=dim) if dim else weights.sum(), min=1.0)
        # 返回加权平均值
        return (weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()) / sum_weights
    else:
        # 若没有提供权重，则计算普通的平均值
        return input_tensor.mean(dim=dim)


# 从 transformers.models.time_series_transformer.modeling_time_series_transformer.nll 复制而来
def nll(input: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the negative log likelihood loss from input distribution with respect to target.
    """
    # 计算输入分布相对于目标的负对数似然损失
    return -input.log_prob(target)
# 从 transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding 复制并修改为 InformerSinusoidalPositionalEmbedding
class InformerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None) -> None:
        super().__init__(num_positions, embedding_dim)
        # 初始化权重
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """
        与 XLM 中 create_sinusoidal_embeddings 函数相同，但特征未交错。cos 特征在向量的第二半部分 [dim // 2:]。
        """
        n_pos, dim = out.shape
        # 创建位置编码矩阵
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        # 在早期设置为不需要梯度，以避免在 pytorch-1.8+ 中出现错误
        out.requires_grad = False
        # 确定分界线位置
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        # 填充正弦和余弦值
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0) -> torch.Tensor:
        """`input_ids_shape` 期望是 [bsz x seqlen]。"""
        bsz, seq_len = input_ids_shape[:2]
        # 生成位置索引
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions)


# 从 transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesValueEmbedding 复制并修改为 InformerValueEmbedding
class InformerValueEmbedding(nn.Module):
    def __init__(self, feature_size, d_model):
        super().__init__()
        # 值投影层，将输入特征映射到模型维度
        self.value_projection = nn.Linear(in_features=feature_size, out_features=d_model, bias=False)

    def forward(self, x):
        return self.value_projection(x)


# 从 transformers.models.bart.modeling_bart.BartAttention 复制并修改为 InformerAttention
class InformerAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[InformerConfig] = None,
    # 继承父类的初始化方法，初始化注意力机制的参数
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 设置嵌入维度
        self.embed_dim = embed_dim
        # 设置注意力头的数量
        self.num_heads = num_heads
        # 设置dropout概率
        self.dropout = dropout
        # 计算每个注意力头的维度
        self.head_dim = embed_dim // num_heads
        # 设置配置参数
        self.config = config

        # 检查嵌入维度是否能够被注意力头的数量整除
        if (self.head_dim * num_heads) != self.embed_dim:
            # 如果不能整除，抛出数值错误异常
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        # 设置缩放因子为头维度的倒数平方根
        self.scaling = self.head_dim**-0.5
        # 设置是否为解码器
        self.is_decoder = is_decoder
        # 设置是否为因果模型
        self.is_causal = is_causal

        # 初始化键、值、查询、输出的线性投影层
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # 定义形状变换函数，用于将张量变形成适合注意力计算的形状
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传播函数，实现注意力机制的计算过程
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        key_value_states: Optional[torch.Tensor] = None,  # 键值状态张量（可选）
        past_key_value: Optional[Tuple[torch.Tensor]] = None,  # 过去的键值状态元组（可选）
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码张量（可选）
        layer_head_mask: Optional[torch.Tensor] = None,  # 层级头掩码张量（可选）
        output_attentions: bool = False,  # 是否输出注意力权重（默认为否）
# 定义了一个名为InformerProbSparseAttention的PyTorch模型类，实现了概率注意力机制来选择“活跃”查询，而不是“懒惰”查询，并提供了稀疏Transformer以减少传统注意力机制的二次计算和内存需求。

class InformerProbSparseAttention(nn.Module):
    """Probabilistic Attention mechanism to select the "active"
    queries rather than the "lazy" queries and provides a sparse Transformer thus mitigating the quadratic compute and
    memory requirements of vanilla attention"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        sampling_factor: int = 5,
        bias: bool = True,
    ):
        super().__init__()
        # 设置采样因子
        self.factor = sampling_factor
        # 嵌入维度
        self.embed_dim = embed_dim
        # 注意力头的数量
        self.num_heads = num_heads
        # dropout概率
        self.dropout = dropout
        # 每个头的维度
        self.head_dim = embed_dim // num_heads

        # 如果头维度乘以头的数量不等于嵌入维度，抛出数值错误
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        # 缩放因子
        self.scaling = self.head_dim**-0.5
        # 是否为解码器
        self.is_decoder = is_decoder

        # 初始化线性变换层
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # 重新塑形张量的辅助函数，用于变换张量形状以适应注意力机制
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传播函数，定义了模型的计算流程和数据处理
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 此处实现了模型的具体计算流程，具体内容可以查看源代码链接
        pass

# InformerConvLayer类的定义，实现了一维卷积层、批归一化、ELU激活函数和最大池化操作，用于模型的卷积处理

class InformerConvLayer(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        # 一维卷积层定义，包括输入通道数、输出通道数、卷积核大小和填充方式
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        # 批归一化层
        self.norm = nn.BatchNorm1d(c_in)
        # ELU激活函数
        self.activation = nn.ELU()
        # 最大池化层
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    # 前向传播函数，定义了数据在卷积层中的处理流程
    def forward(self, x):
        # 数据的维度变换，将数据的通道维度调整到卷积层期望的格式
        x = self.downConv(x.permute(0, 2, 1))
        # 数据的批归一化处理
        x = self.norm(x)
        # 数据的ELU激活函数处理
        x = self.activation(x)
        # 数据的最大池化处理
        x = self.maxPool(x)
        # 数据维度的还原，恢复到原来的维度格式
        x = x.transpose(1, 2)
        return x

# InformerEncoderLayer类的定义，可以继续添加注释
    # 初始化函数，接受一个 InformedConfig 类型的配置对象作为参数
    def __init__(self, config: InformerConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 设置嵌入维度为配置对象中的 d_model 属性
        self.embed_dim = config.d_model
        # 根据配置对象中的 attention_type 属性选择不同类型的注意力机制
        if config.attention_type == "prob":
            # 如果 attention_type 是 "prob"，使用稀疏注意力机制
            self.self_attn = InformerProbSparseAttention(
                embed_dim=self.embed_dim,  # 设置嵌入维度
                num_heads=config.encoder_attention_heads,  # 设置注意力头数
                dropout=config.attention_dropout,  # 设置注意力机制的 dropout 率
                sampling_factor=config.sampling_factor,  # 设置采样因子
            )
        else:
            # 否则，使用标准的注意力机制
            self.self_attn = InformerAttention(
                embed_dim=self.embed_dim,  # 设置嵌入维度
                num_heads=config.encoder_attention_heads,  # 设置注意力头数
                dropout=config.attention_dropout,  # 设置注意力机制的 dropout 率
            )
        # 对自注意力机制的输出进行层归一化
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 设置 dropout 率
        self.dropout = config.dropout
        # 根据配置对象中的 activation_function 属性选择激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 设置激活函数的 dropout 率
        self.activation_dropout = config.activation_dropout
        # 设置第一个全连接层，将嵌入维度映射到编码器前馈网络的维度
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        # 设置第二个全连接层，将编码器前馈网络的维度映射回嵌入维度
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        # 对最终输出进行层归一化
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states  # 保存输入 hidden_states 的原始值，用于残差连接
        hidden_states, attn_weights, _ = self.self_attn(  # 使用 self-attention 层处理输入
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)  # 对 self-attention 的输出进行 dropout
        hidden_states = residual + hidden_states  # 残差连接，将 dropout 后的结果与原始输入相加
        hidden_states = self.self_attn_layer_norm(hidden_states)  # 使用 LayerNorm 进行归一化处理

        residual = hidden_states  # 保存处理后的 hidden_states，用于残差连接
        hidden_states = self.activation_fn(self.fc1(hidden_states))  # 通过线性层和激活函数 fc1 处理 hidden_states
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)  # 对 fc1 的输出进行 dropout
        hidden_states = self.fc2(hidden_states)  # 通过线性层 fc2 处理 hidden_states
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)  # 对 fc2 的输出进行 dropout
        hidden_states = residual + hidden_states  # 残差连接，将 dropout 后的结果与之前保存的 hidden_states 相加
        hidden_states = self.final_layer_norm(hidden_states)  # 使用 LayerNorm 进行归一化处理

        if hidden_states.dtype == torch.float16 and (  # 如果 hidden_states 的数据类型为 float16，且包含无穷大或 NaN 值
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000  # 获取当前数据类型的最大值，用于 clamp 操作
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)  # 对 hidden_states 进行 clamp 操作

        outputs = (hidden_states,)  # 将处理后的 hidden_states 存入 outputs 中

        if output_attentions:  # 如果指定要返回 attention tensors
            outputs += (attn_weights,)  # 将 attention weights 也加入 outputs 中

        return outputs  # 返回最终的输出结果
class InformerDecoderLayer(nn.Module):
    # Informer 解码器层的定义，继承自 nn.Module
    def __init__(self, config: InformerConfig):
        super().__init__()
        self.embed_dim = config.d_model
        # 根据配置选择不同类型的注意力机制
        if config.attention_type == "prob":
            # 使用稀疏概率注意力机制
            self.self_attn = InformerProbSparseAttention(
                embed_dim=self.embed_dim,
                num_heads=config.decoder_attention_heads,
                dropout=config.attention_dropout,
                sampling_factor=config.sampling_factor,
                is_decoder=True,
            )
        else:
            # 使用普通的注意力机制
            self.self_attn = InformerAttention(
                embed_dim=self.embed_dim,
                num_heads=config.decoder_attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
            )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        # 解码器自注意力层归一化
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 解码器与编码器之间的注意力
        self.encoder_attn = InformerAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        # 编码器注意力层归一化
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 解码器的前馈神经网络的第一层
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        # 解码器的前馈神经网络的第二层
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        # 最终层的归一化
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    # 前向传播函数，接受多个参数和可选的参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,



class InformerPreTrainedModel(PreTrainedModel):
    # Informer 预训练模型，继承自 PreTrainedModel 类
    config_class = InformerConfig
    base_model_prefix = "model"
    main_input_name = "past_values"
    supports_gradient_checkpointing = True

    # 初始化权重函数，根据模块类型不同进行不同的初始化
    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


INFORMER_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    # 这个模型也是一个 PyTorch 的 torch.nn.Module 子类。
    # 可以像使用普通的 PyTorch 模块一样使用它，关于一般使用和行为的所有问题，请参考 PyTorch 的文档。

    Parameters:
        # config 参数接受一个 TimeSeriesTransformerConfig 类的实例。
        # 这个模型的配置类包含所有的模型参数。使用配置文件初始化模型时，并不会加载与模型相关的权重，仅加载配置。
        # 若要加载模型的权重，请参考 PreTrainedModel.from_pretrained 方法。
"""
INFORMER_INPUTS_DOCSTRING = r"""
"""


class InformerEncoder(InformerPreTrainedModel):
    """
    Informer encoder consisting of *config.encoder_layers* self attention layers with distillation layers. Each
    attention layer is an [`InformerEncoderLayer`].

    Args:
        config: InformerConfig
    """

    def __init__(self, config: InformerConfig):
        super().__init__(config)

        self.dropout = config.dropout  # 从配置中获取 dropout 率
        self.layerdrop = config.encoder_layerdrop  # 从配置中获取 encoder 层的 dropout 率
        self.gradient_checkpointing = False  # 初始化梯度检查点标志为 False
        if config.prediction_length is None:
            raise ValueError("The `prediction_length` config needs to be specified.")

        self.value_embedding = InformerValueEmbedding(feature_size=config.feature_size, d_model=config.d_model)
        self.embed_positions = InformerSinusoidalPositionalEmbedding(
            config.context_length + config.prediction_length, config.d_model
        )  # 创建位置嵌入对象，用于处理输入序列位置信息
        self.layers = nn.ModuleList([InformerEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)  # 创建输入嵌入的 LayerNorm 层

        if config.distil:
            self.conv_layers = nn.ModuleList(
                [InformerConvLayer(config.d_model) for _ in range(config.encoder_layers - 1)]
            )
            self.conv_layers.append(None)
        else:
            self.conv_layers = [None] * config.encoder_layers  # 根据 distil 配置初始化卷积层列表

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass for the InformerEncoder.

        Args:
            attention_mask: Optional mask for attention layers.
            head_mask: Optional mask for attention heads.
            inputs_embeds: Optional input embeddings.
            output_attentions: Optional flag to output attentions.
            output_hidden_states: Optional flag to output hidden states.
            return_dict: Optional flag to return a dictionary as output.

        Returns:
            Depending on `return_dict`, either a tuple or a dictionary with different outputs.
        """
        # Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesTransformerDecoder with TimeSeriesTransformer->Informer,TimeSeriesTransformerConfig->InformerConfig,time-series-transformer->informer,Transformer->Informer,TimeSeries->Informer


class InformerDecoder(InformerPreTrainedModel):
    """
    Informer decoder consisting of *config.decoder_layers* layers. Each layer is a
    [`InformerDecoderLayer`]

    Args:
        config: InformerConfig
    """
    # 初始化函数，接受一个 InfromerConfig 类型的配置对象作为参数
    def __init__(self, config: InformerConfig):
        # 调用父类的初始化方法，传入配置对象
        super().__init__(config)
        # 设置类的dropout属性为配置对象中的dropout值
        self.dropout = config.dropout
        # 设置类的layerdrop属性为配置对象中的decoder_layerdrop值
        self.layerdrop = config.decoder_layerdrop
        # 如果配置对象中的prediction_length为None，则抛出值错误异常
        if config.prediction_length is None:
            raise ValueError("The `prediction_length` config needs to be specified.")

        # 创建一个 InfromerValueEmbedding 实例，传入配置对象中的feature_size和d_model作为参数
        self.value_embedding = InformerValueEmbedding(feature_size=config.feature_size, d_model=config.d_model)
        # 创建一个 InfromerSinusoidalPositionalEmbedding 实例，传入context_length + prediction_length和d_model作为参数
        self.embed_positions = InformerSinusoidalPositionalEmbedding(
            config.context_length + config.prediction_length, config.d_model
        )
        # 创建一个由多个 InfromerDecoderLayer 实例组成的 ModuleList，数量为配置对象中的decoder_layers
        self.layers = nn.ModuleList([InformerDecoderLayer(config) for _ in range(config.decoder_layers)])
        # 创建一个 LayerNorm 层，归一化维度为配置对象中的d_model
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        # 初始化梯度检查点为False
        self.gradient_checkpointing = False
        # 执行额外的初始化操作和最终处理
        self.post_init()

    # 前向传播函数
    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
@add_start_docstrings(
    "The bare Informer Model outputting raw hidden-states without any specific head on top.",
    INFORMER_START_DOCSTRING,
)
# 从transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesTransformerModel复制而来，将TimeSeriesTransformer->Informer，TIME_SERIES_TRANSFORMER->INFORMER，time-series-transformer->informer，TimeSeries->Informer
class InformerModel(InformerPreTrainedModel):
    def __init__(self, config: InformerConfig):
        super().__init__(config)

        # 根据配置选择合适的缩放器
        if config.scaling == "mean" or config.scaling is True:
            self.scaler = InformerMeanScaler(config)
        elif config.scaling == "std":
            self.scaler = InformerStdScaler(config)
        else:
            self.scaler = InformerNOPScaler(config)

        # 如果存在静态分类特征，创建特征嵌入器
        if config.num_static_categorical_features > 0:
            self.embedder = InformerFeatureEmbedder(
                cardinalities=config.cardinality,
                embedding_dims=config.embedding_dimension,
            )

        # 初始化变压器编码器和解码器，以及掩码初始化器
        self.encoder = InformerEncoder(config)
        self.decoder = InformerDecoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    @property
    def _past_length(self) -> int:
        # 返回过去历史长度，包括最大的滞后长度
        return self.config.context_length + max(self.config.lags_sequence)

    def get_lagged_subsequences(
        self, sequence: torch.Tensor, subsequences_length: int, shift: int = 0
    ) -> torch.Tensor:
        """
        返回给定序列的滞后子序列。返回的张量形状为(N, S, C, I)，其中S为子序列长度，I为indices的长度，包含滞后的子序列。
        具体而言，lagged[i, j, :, k] = sequence[i, -indices[k]-S+j, :].

        Args:
            sequence: Tensor
                要提取滞后子序列的序列。形状为(N, T, C).
            subsequences_length : int
                要提取的子序列长度。
            shift: int
                向后移动滞后的量。
        """
        sequence_length = sequence.shape[1]
        indices = [lag - shift for lag in self.config.lags_sequence]

        if max(indices) + subsequences_length > sequence_length:
            raise ValueError(
                f"lags不能超过历史长度，发现lag {max(indices)}，而历史长度仅为{sequence_length}"
            )

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...])
        return torch.stack(lagged_values, dim=-1)
    # 创建神经网络模型的输入数据
    def create_network_inputs(
        self,
        # 过去的值，作为模型输入的时间序列数据
        past_values: torch.Tensor,
        # 过去时间特征，如日期、时间等
        past_time_features: torch.Tensor,
        # 静态分类特征，如类别信息（可选）
        static_categorical_features: Optional[torch.Tensor] = None,
        # 静态实数特征，如价格、数量等（可选）
        static_real_features: Optional[torch.Tensor] = None,
        # 过去观测的掩码，标记观测是否存在（可选）
        past_observed_mask: Optional[torch.Tensor] = None,
        # 未来的目标值（可选）
        future_values: Optional[torch.Tensor] = None,
        # 未来时间特征（可选）
        future_time_features: Optional[torch.Tensor] = None,
        ):
            # time feature
            time_feat = (
                torch.cat(
                    (
                        past_time_features[:, self._past_length - self.config.context_length :, ...],
                        future_time_features,
                    ),
                    dim=1,
                )
                if future_values is not None
                else past_time_features[:, self._past_length - self.config.context_length :, ...]
            )

            # target
            if past_observed_mask is None:
                past_observed_mask = torch.ones_like(past_values)

            # Extract context from past values
            context = past_values[:, -self.config.context_length :]
            observed_context = past_observed_mask[:, -self.config.context_length :]

            # Normalize context using the scaler
            _, loc, scale = self.scaler(context, observed_context)
            
            # Prepare inputs for the model, applying normalization
            inputs = (
                (torch.cat((past_values, future_values), dim=1) - loc) / scale
                if future_values is not None
                else (past_values - loc) / scale
            )

            # Calculate static features
            log_abs_loc = loc.abs().log1p() if self.config.input_size == 1 else loc.squeeze(1).abs().log1p()
            log_scale = scale.log() if self.config.input_size == 1 else scale.squeeze(1).log()
            static_feat = torch.cat((log_abs_loc, log_scale), dim=1)

            # Incorporate additional static real features if available
            if static_real_features is not None:
                static_feat = torch.cat((static_real_features, static_feat), dim=1)
            
            # Incorporate additional static categorical features if available
            if static_categorical_features is not None:
                embedded_cat = self.embedder(static_categorical_features)
                static_feat = torch.cat((embedded_cat, static_feat), dim=1)
            
            # Expand static features to align with time feature shape
            expanded_static_feat = static_feat.unsqueeze(1).expand(-1, time_feat.shape[1], -1)

            # Combine static and time features
            features = torch.cat((expanded_static_feat, time_feat), dim=-1)

            # Generate lagged sequences of inputs
            subsequences_length = (
                self.config.context_length + self.config.prediction_length
                if future_values is not None
                else self.config.context_length
            )
            lagged_sequence = self.get_lagged_subsequences(sequence=inputs, subsequences_length=subsequences_length)
            lags_shape = lagged_sequence.shape
            reshaped_lagged_sequence = lagged_sequence.reshape(lags_shape[0], lags_shape[1], -1)

            # Check if the length of reshaped lagged sequence matches time feature length
            if reshaped_lagged_sequence.shape[1] != time_feat.shape[1]:
                raise ValueError(
                    f"input length {reshaped_lagged_sequence.shape[1]} and time feature lengths {time_feat.shape[1]} does not match"
                )

            # Prepare inputs for transformer, combining reshaped lagged sequence and features
            transformer_inputs = torch.cat((reshaped_lagged_sequence, features), dim=-1)

            return transformer_inputs, loc, scale, static_feat

        def get_encoder(self):
            return self.encoder

        def get_decoder(self):
            return self.decoder

        @add_start_docstrings_to_model_forward(INFORMER_INPUTS_DOCSTRING)
        @replace_return_docstrings(output_type=Seq2SeqTSModelOutput, config_class=_CONFIG_FOR_DOC)
    # 定义神经网络模型中的前向传播方法，处理输入和输出
    def forward(
        self,
        # 过去的数值状态，形状为 [batch_size, seq_length, num_features]
        past_values: torch.Tensor,
        # 过去的时间特征，形状为 [batch_size, seq_length, num_time_features]
        past_time_features: torch.Tensor,
        # 过去观察到的掩码，形状为 [batch_size, seq_length]
        past_observed_mask: torch.Tensor,
        # 静态分类特征，形状为 [batch_size, num_static_cat_features]
        static_categorical_features: Optional[torch.Tensor] = None,
        # 静态实数特征，形状为 [batch_size, num_static_real_features]
        static_real_features: Optional[torch.Tensor] = None,
        # 未来的数值状态，形状为 [batch_size, seq_length, num_future_features]
        future_values: Optional[torch.Tensor] = None,
        # 未来的时间特征，形状为 [batch_size, seq_length, num_time_features]
        future_time_features: Optional[torch.Tensor] = None,
        # 解码器的注意力掩码，形状为 [batch_size, seq_length]
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        # 多头注意力掩码，形状为 [num_heads, seq_length, seq_length]
        head_mask: Optional[torch.Tensor] = None,
        # 解码器头部的掩码，形状为 [num_layers, num_heads, seq_length, seq_length]
        decoder_head_mask: Optional[torch.Tensor] = None,
        # 交叉注意力头部的掩码，形状为 [num_layers, num_heads, seq_length, seq_length]
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        # 编码器输出列表，每个元素形状为 [batch_size, seq_length, hidden_size]
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        # 过去的键值对列表，每个元素形状为 [batch_size, seq_length, hidden_size]
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # 输出隐藏状态的标志
        output_hidden_states: Optional[bool] = None,
        # 输出注意力权重的标志
        output_attentions: Optional[bool] = None,
        # 是否使用缓存的标志
        use_cache: Optional[bool] = None,
        # 是否返回字典格式的结果
        return_dict: Optional[bool] = None,
        ):
# 为 InfornerForPrediction 类添加文档字符串，描述其作为基于时间序列预测的模型的分布式头部
# 参考自 transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesTransformerForPrediction，将 TimeSeriesTransformer->Informer，TIME_SERIES_TRANSFORMER->INFORMER，time-series-transformer->informer 进行替换
@add_start_docstrings(
    "The Informer Model with a distribution head on top for time-series forecasting.",
    INFORMER_START_DOCSTRING,
)
class InformerForPrediction(InformerPreTrainedModel):
    def __init__(self, config: InformerConfig):
        super().__init__(config)
        # 使用给定的配置初始化 InfornerModel
        self.model = InformerModel(config)
        # 根据配置选择分布输出类型，如 student_t, normal, negative_binomial
        if config.distribution_output == "student_t":
            self.distribution_output = StudentTOutput(dim=config.input_size)
        elif config.distribution_output == "normal":
            self.distribution_output = NormalOutput(dim=config.input_size)
        elif config.distribution_output == "negative_binomial":
            self.distribution_output = NegativeBinomialOutput(dim=config.input_size)
        else:
            raise ValueError(f"Unknown distribution output {config.distribution_output}")

        # 使用 distribution_output 的参数投影初始化 parameter_projection
        self.parameter_projection = self.distribution_output.get_parameter_projection(self.model.config.d_model)
        # 设置 target_shape 为 distribution_output 的事件形状
        self.target_shape = self.distribution_output.event_shape

        # 根据配置选择损失函数，如 nll
        if config.loss == "nll":
            self.loss = nll
        else:
            raise ValueError(f"Unknown loss function {config.loss}")

        # 初始化 distribution_output 的权重并应用最终处理
        self.post_init()

    # 输出参数投影的方法，接受解码器输出 dec_output
    def output_params(self, dec_output):
        return self.parameter_projection(dec_output)

    # 获取编码器的方法
    def get_encoder(self):
        return self.model.get_encoder()

    # 获取解码器的方法
    def get_decoder(self):
        return self.model.get_decoder()

    # 忽略此方法的 Torch JIT 编译
    @torch.jit.ignore
    def output_distribution(self, params, loc=None, scale=None, trailing_n=None) -> torch.distributions.Distribution:
        # 如果 trailing_n 不为 None，则对 params 进行切片操作
        sliced_params = params
        if trailing_n is not None:
            sliced_params = [p[:, -trailing_n:] for p in params]
        # 返回 distribution_output 的分布对象，传入切片后的参数、loc 和 scale
        return self.distribution_output.distribution(sliced_params, loc=loc, scale=scale)

    # 添加 INFORMER_INPUTS_DOCSTRING 到 model_forward 方法的文档字符串
    @add_start_docstrings_to_model_forward(INFORMER_INPUTS_DOCSTRING)
    # 替换输出类型文档字符串为 Seq2SeqTSModelOutput，使用 _CONFIG_FOR_DOC 作为配置类
    @replace_return_docstrings(output_type=Seq2SeqTSModelOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个方法用于向前传播，接收多个输入参数和可选的参数，返回模型预测结果
    def forward(
        self,
        past_values: torch.Tensor,  # 过去的数值数据，类型为 Torch 张量
        past_time_features: torch.Tensor,  # 过去的时间特征，类型为 Torch 张量
        past_observed_mask: torch.Tensor,  # 过去观察掩码，类型为 Torch 张量
        static_categorical_features: Optional[torch.Tensor] = None,  # 静态分类特征，可选的 Torch 张量
        static_real_features: Optional[torch.Tensor] = None,  # 静态实数特征，可选的 Torch 张量
        future_values: Optional[torch.Tensor] = None,  # 未来的数值数据，可选的 Torch 张量
        future_time_features: Optional[torch.Tensor] = None,  # 未来的时间特征，可选的 Torch 张量
        future_observed_mask: Optional[torch.Tensor] = None,  # 未来观察掩码，可选的 Torch 张量
        decoder_attention_mask: Optional[torch.LongTensor] = None,  # 解码器注意力掩码，可选的 LongTensor
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码，可选的 Torch 张量
        decoder_head_mask: Optional[torch.Tensor] = None,  # 解码器头部掩码，可选的 Torch 张量
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 交叉注意力头部掩码，可选的 Torch 张量
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,  # 编码器输出列表，每个元素为 Torch 浮点张量，可选的
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 过去键值列表，每个元素为 Torch 浮点张量，可选的
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选的布尔值
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选的布尔值
        use_cache: Optional[bool] = None,  # 是否使用缓存，可选的布尔值
        return_dict: Optional[bool] = None,  # 是否返回字典格式的结果，可选的布尔值
    ):
        # 在不计算梯度的上下文中执行方法体
        @torch.no_grad()
        def generate(
            self,
            past_values: torch.Tensor,  # 过去的数值数据，类型为 Torch 张量
            past_time_features: torch.Tensor,  # 过去的时间特征，类型为 Torch 张量
            future_time_features: torch.Tensor,  # 未来的时间特征，类型为 Torch 张量
            past_observed_mask: Optional[torch.Tensor] = None,  # 过去观察掩码，可选的 Torch 张量
            static_categorical_features: Optional[torch.Tensor] = None,  # 静态分类特征，可选的 Torch 张量
            static_real_features: Optional[torch.Tensor] = None,  # 静态实数特征，可选的 Torch 张量
            output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选的布尔值
            output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选的布尔值
        ):
```