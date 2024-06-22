# `.\transformers\models\time_series_transformer\modeling_time_series_transformer.py`

```py
# 设定 UTF-8 编码格式
# 版权声明
# 根据Apache License, Version 2.0许可协议，除非符合许可协议使用，否则不得使用此文件
# 您可以在以下网址获取许可协议的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 未经适用法律或书面约定，根据许可协议分发的软件以“原样”分发
# 没有任何明示或暗示的保证或条件，包括但不限于适销性和特定用途适用性
# 请参阅许可协议，了解特定语言管理权限和限制
""" PyTorch时间序列变压器模型."""

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
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_time_series_transformer import TimeSeriesTransformerConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "TimeSeriesTransformerConfig"

# 预训练模型列表
TIME_SERIES_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "huggingface/time-series-transformer-tourism-monthly",
    # 查看所有的TimeSeriesTransformer模型 https://huggingface.co/models?filter=time_series_transformer
]

class TimeSeriesFeatureEmbedder(nn.Module):
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

        # 特征数量
        self.num_features = len(cardinalities)
        # 初始化嵌入器
        self.embedders = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(cardinalities, embedding_dims)])
    # 定义一个前向传播的方法，输入是特征张量，输出也是特征张量
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 判断特征张量的最后一个维度是否大于1
        if self.num_features > 1:
            # 如果大于1，将最后一维度切片，得到长度为self.num_features的数组，形状为(N,T)或(N)
            cat_feature_slices = torch.chunk(features, self.num_features, dim=-1)
        else:
            # 如果不大于1，将特征张量放入列表中
            cat_feature_slices = [features]

        # 对切片后的特征进行嵌入操作，并将结果拼接在一起，沿着最后一个维度
        return torch.cat(
            [
                embed(cat_feature_slice.squeeze(-1))
                for embed, cat_feature_slice in zip(self.embedders, cat_feature_slices)
            ],
            dim=-1,
        )
class TimeSeriesStdScaler(nn.Module):
    """
    根据均值和标准差对特征进行标准化，先求均值和标准差，然后通过减去均值并除以标准差进行归一化。
    """

    def __init__(self, config: TimeSeriesTransformerConfig):
        super().__init__()
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1  # 如果配置中存在缩放维度，则使用配置中的值，否则默认为1
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True  # 如果配置中存在keepdim参数，则使用配置中的值，否则默认为True
        self.minimum_scale = config.minimum_scale if hasattr(config, "minimum_scale") else 1e-5  # 如果配置中存在最小标度参数，则使用配置中的值，否则默认为1e-5

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        参数:
            data (形状为 `(batch_size, sequence_length, num_input_channels)` 的 `torch.Tensor`):
                用于批量标准化计算的输入
            observed_indicator (形状为 `(batch_size, sequence_length, num_input_channels)` 的 `torch.BoolTensor`):
                根据观察指示器计算标度。
        返回:
            形状为(`(batch_size, sequence_length, num_input_channels)`, `(batch_size, 1, num_input_channels)`,
            `(batch_size, 1, num_input_channels)`)的torch.Tensor元组
        """
        # 对观察指示器求和，作为分母，保持维度与指示器一致
        denominator = observed_indicator.sum(self.dim, keepdim=self.keepdim)
        # 将分母限制在最小值1.0以上
        denominator = denominator.clamp_min(1.0)
        # 计算平均值
        loc = (data * observed_indicator).sum(self.dim, keepdim=self.keepdim) / denominator

        # 计算方差
        variance = (((data - loc) * observed_indicator) ** 2).sum(self.dim, keepdim=self.keepdim) / denominator
        # 计算标度（标准差）
        scale = torch.sqrt(variance + self.minimum_scale)
        return (data - loc) / scale, loc, scale


class TimeSeriesMeanScaler(nn.Module):
    """
    计算缩放因子作为沿第一个维度的加权平均绝对值，并相应地缩放数据。
    """

    def __init__(self, config: TimeSeriesTransformerConfig):
        super().__init__()
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1  # 如果配置中存在缩放维度，则使用配置中的值，否则默认为1
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True  # 如果配置中存在keepdim参数，则使用配置中的值，否则默认为True
        self.minimum_scale = config.minimum_scale if hasattr(config, "minimum_scale") else 1e-10  # 如果配置中存在最小标度参数，则使用配置中的值，否则默认为1e-10
        self.default_scale = config.default_scale if hasattr(config, "default_scale") else None  # 如果配置中存在默认标度参数，则使用配置中的值，否则默认为None

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                输入用于批归一化计算的数据
            observed_indicator (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                计算观测指标上的尺度
        Returns:
            返回值为元组，包含3个 `torch.Tensor` 对象的形状，
            依次为 `(batch_size, sequence_length, num_input_channels)`, `(batch_size, 1, num_input_channels)`,
            `(batch_size, 1, num_input_channels)`
        """

        # 计算在观察指标上的时间序列之和，并保持维度
        ts_sum = (data * observed_indicator).abs().sum(self.dim, keepdim=True)
        # 计算观察指标的数量，并保持维度
        num_observed = observed_indicator.sum(self.dim, keepdim=True)

        # 计算尺度
        scale = ts_sum / torch.clamp(num_observed, min=1)

        # 如果提供了 `default_scale`，则使用它，否则使用批次的尺度
        if self.default_scale is None:
            # 计算批次时间序列之和
            batch_sum = ts_sum.sum(dim=0)
            # 计算批次观察数量
            batch_observations = torch.clamp(num_observed.sum(0), min=1)
            # 计算默认尺度
            default_scale = torch.squeeze(batch_sum / batch_observations)
        else:
            default_scale = self.default_scale * torch.ones_like(scale)

        # 在没有观察到数据的地方应用默认尺度
        scale = torch.where(num_observed > 0, scale, default_scale)

        # 确保尺度至少为 `self.minimum_scale`
        scale = torch.clamp(scale, min=self.minimum_scale)
        # 缩放数据
        scaled_data = data / scale

        # 如果不保持维度，则压缩尺度
        if not self.keepdim:
            scale = scale.squeeze(dim=self.dim)

        # 返回缩放后的数据、与尺度相同形状的零张量、和尺度
        return scaled_data, torch.zeros_like(scale), scale
class TimeSeriesNOPScaler(nn.Module):
    """
    Assigns a scaling factor equal to 1 along the first dimension, and therefore applies no scaling to the input data.
    """

    def __init__(self, config: TimeSeriesTransformerConfig):
        # 初始化函数，设置初始参数
        super().__init__()
        # 确定维度，默认为1
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        # 确定保留维度，默认为True
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
        # 计算数据在指定维度上的平均值
        scale = torch.ones_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        # 计算数据在指定维度上的平均值
        loc = torch.zeros_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        # 返回数据、位置和缩放比例
        return data, loc, scale


def nll(input: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the negative log likelihood loss from input distribution with respect to target.
    """
    # 计算负的对数似然损失
    return -input.log_prob(target)


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
        # 计算加权平均值，将权重为零的值替换为0
        weighted_tensor = torch.where(weights != 0, input_tensor * weights, torch.zeros_like(input_tensor))
        # 限制权重总和至少为1
        sum_weights = torch.clamp(weights.sum(dim=dim) if dim else weights.sum(), min=1.0)
        # 返回加权平均值
        return (weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()) / sum_weights
    else:
        # 计算未加权平均值
        return input_tensor.mean(dim=dim)


# Copied from transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding with Marian->TimeSeries
class TimeSeriesSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""
    # 初始化函数
    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None) -> None:
        # 调用父类的初始化函数
        super().__init__(num_positions, embedding_dim)
        # 初始化权重
        self.weight = self._init_weight(self.weight)

    # 初始化权重函数，返回初始化后的权重
    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        # 获取权重的维度
        n_pos, dim = out.shape
        # 创建位置编码矩阵
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        # 设置权重的梯度不进行计算
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        # 计算第一个分界线
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        # 使用正弦函数填充第一部分的权重
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        # 使用余弦函数填充第二部分的权重
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        # 取消权重与梯度的关联
        out.detach_()
        # 返回初始化后的权重
        return out

    # 前向传播函数，返回位置编码的张量
    @torch.no_grad()
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0) -> torch.Tensor:
        """
        `input_ids_shape` is expected to be [bsz x seqlen].
        """
        # 获取输入张量的形状
        bsz, seq_len = input_ids_shape[:2]
        # 计算位置编码
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        # 调用父类的前向传播函数，返回位置编码的张量
        return super().forward(positions)
# 定义时间序列数值嵌入模块
class TimeSeriesValueEmbedding(nn.Module):
    # 初始化方法，接受特征大小和模型大小作为参数
    def __init__(self, feature_size, d_model):
        super().__init__()
        # 使用线性层将输入特征投影到模型大小的维度
        self.value_projection = nn.Linear(in_features=feature_size, out_features=d_model, bias=False)

    # 前向传播方法，输入参数 x，返回经过投影后的结果
    def forward(self, x):
        return self.value_projection(x)


# 从 transformers.models.bart.modeling_bart.BartAttention 复制而来，将 Bart 替换为 TimeSeriesTransformer
class TimeSeriesTransformerAttention(nn.Module):
    """来自“Attention Is All You Need”论文的多头注意力"""

    # 初始化方法，接受嵌入维度、头的数量、dropout概率等参数
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[TimeSeriesTransformerConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        # 检查 embed_dim 是否可以被 num_heads 整除，否则抛出错误
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        # 初始化线性层，用于处理输入的 key、value、query 和输出
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # 辅助方法，用于重塑输入张量的形状
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传播方法，接受隐藏状态、键值状态等参数，执行多头注意力计算
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,

``` 
# 从 transformers.models.bart.modeling_bart.BartEncoderLayer 复制而来，将 Bart 替换为 TimeSeriesTransformer，将 BART 替换为 TIME_SERIES_TRANSFORMER
class TimeSeriesTransformerEncoderLayer(nn.Module):
    # 初始化函数，接受配置作为参数
    def __init__(self, config: TimeSeriesTransformerConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 设置嵌入维度为配置中的模型维度
        self.embed_dim = config.d_model

        # 创建自注意力机制对象
        self.self_attn = TIME_SERIES_TRANSFORMER_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )
        # 创建自注意力机制的层归一化
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 设置丢弃概率
        self.dropout = config.dropout
        # 设置激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 设置激活函数的丢弃概率
        self.activation_dropout = config.activation_dropout
        # 创建全连接层1
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        # 创建全连接层2
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        # 创建最终的层归一化
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        layer_head_mask: torch.FloatTensor,
        output_attentions: Optional[bool] = False,
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
        # 保存输入的hidden_states作为残差连接的输入
        residual = hidden_states
        # 通过 self-attention 层处理 hidden_states，获取输出的hidden_states、注意力权重attn_weights以及空参数 _
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 对 hidden_states 进行 dropout 操作
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 添加残差连接
        hidden_states = residual + hidden_states
        # 对添加残差连接后的 hidden_states 进行层归一化操作
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 保存新的输入hidden_states作为残差连接的输入
        residual = hidden_states
        # 通过激活函数处理 hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对 hidden_states 进行 activation dropout 操作
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # 通过全连接层 fc2 处理 hidden_states
        hidden_states = self.fc2(hidden_states)
        # 再次进行 dropout 操作
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 添加残差连接
        hidden_states = residual + hidden_states
        # 对添加残差连接后的 hidden_states 进行层归一化操作
        hidden_states = self.final_layer_norm(hidden_states)

        # 如果 hidden_states 数据类型为 torch.float16 且包含 inf 或 nan 值，进行限值处理
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # 构建返回的输出元组
        outputs = (hidden_states,)

        # 如果需要返回 attentions，则在 outputs 中添加 attn_weights
        if output_attentions:
            outputs += (attn_weights,)

        return outputs
# TODO: 实现TimeSeriesTransformer的SDPA（自注意力机制）。
# 时间序列Transformer的注意力类
TIME_SERIES_TRANSFORMER_ATTENTION_CLASSES = {
    "eager": TimeSeriesTransformerAttention,
}


# 从transformers.models.bart.modeling_bart.BartDecoderLayer中复制过来，将Bart->TimeSeriesTransformer，将BART->TIME_SERIES_TRANSFORMER
class TimeSeriesTransformerDecoderLayer(nn.Module):
    def __init__(self, config: TimeSeriesTransformerConfig):
        super().__init__()
        self.embed_dim = config.d_model

        # 通过字典选择合适的注意力类，并进行初始化
        self.self_attn = TIME_SERIES_TRANSFORMER_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
        )
        # 使用config中的dropout参数
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        # 初始化LayerNorm
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 选取合适的注意力类，并进行初始化
        self.encoder_attn = TIME_SERIES_TRANSFORMER_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        # 初始化LayerNorm
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 全连接层
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        # 初始化LayerNorm
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

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
class TimeSeriesTransformerPreTrainedModel(PreTrainedModel):
    config_class = TimeSeriesTransformerConfig
    base_model_prefix = "model"
    main_input_name = "past_values"
    supports_gradient_checkpointing = True

    # 初始化权重
    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, TimeSeriesSinusoidalPositionalEmbedding):
            pass
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# TimeSeriesTransformer的文档字符串起始
TIME_SERIES_TRANSFORMER_START_DOCSTRING = r"""
    # 这个模型继承自 [`PreTrainedModel`]。检查父类文档中实现的通用方法,如下载、保存、调整输入嵌入、修剪头部等。
    # 这个模型也是 PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) 的子类。
    # 把它当作一个标准的 PyTorch Module 使用,并参考 PyTorch 文档以获取有关一般用法和行为的所有信息。
    
    # 参数:
    #     config ([`TimeSeriesTransformerConfig`]):
    #         包含模型所有参数的模型配置类。使用配置文件进行初始化不会加载与模型关联的权重,只会加载配置。
    #         查看 [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
"""

TIME_SERIES_TRANSFORMER_INPUTS_DOCSTRING = r"""
"""

# 定义时间序列Transformer编码器类，包括多个自注意力层
class TimeSeriesTransformerEncoder(TimeSeriesTransformerPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`TimeSeriesTransformerEncoderLayer`].

    Args:
        config: TimeSeriesTransformerConfig
    """

    def __init__(self, config: TimeSeriesTransformerConfig):
        super().__init__(config)

        # 初始化dropout和layerdrop
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        # 检查预测长度是否已指定
        if config.prediction_length is None:
            raise ValueError("The `prediction_length` config needs to be specified.")

        # 初始化数值嵌入和时间位置嵌入
        self.value_embedding = TimeSeriesValueEmbedding(feature_size=config.feature_size, d_model=config.d_model)
        self.embed_positions = TimeSeriesSinusoidalPositionalEmbedding(
            config.context_length + config.prediction_length, config.d_model
        )
        # 初始化多个编码器层
        self.layers = nn.ModuleList([TimeSeriesTransformerEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
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

# 定义时间序列Transformer解码器类，包括多个解码层
class TimeSeriesTransformerDecoder(TimeSeriesTransformerPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a
    [`TimeSeriesTransformerDecoderLayer`]

    Args:
        config: TimeSeriesTransformerConfig
    """

    def __init__(self, config: TimeSeriesTransformerConfig):
        super().__init__(config)
        # 初始化dropout和layerdrop
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        # 检查预测长度是否已指定
        if config.prediction_length is None:
            raise ValueError("The `prediction_length` config needs to be specified.")

        # 初始化数值嵌入和时间位置嵌入
        self.value_embedding = TimeSeriesValueEmbedding(feature_size=config.feature_size, d_model=config.d_model)
        self.embed_positions = TimeSeriesSinusoidalPositionalEmbedding(
            config.context_length + config.prediction_length, config.d_model
        )
        # 初始化多个解码器层
        self.layers = nn.ModuleList([TimeSeriesTransformerDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
    # 定义 forward 方法，用于模型的前向传播
    def forward(
        # 输入参数 attention_mask: 可选的注意力遮罩张量
        attention_mask: Optional[torch.Tensor] = None,
        # 输入参数 encoder_hidden_states: 可选的编码器隐藏状态张量
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        # 输入参数 encoder_attention_mask: 可选的编码器注意力遮罩张量
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        # 输入参数 head_mask: 可选的注意力头遮罩张量
        head_mask: Optional[torch.Tensor] = None,
        # 输入参数 cross_attn_head_mask: 可选的交叉注意力头遮罩张量
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        # 输入参数 past_key_values: 可选的历史键值对列表
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # 输入参数 inputs_embeds: 可选的输入嵌入张量
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 输入参数 use_cache: 可选的是否使用缓存标志
        use_cache: Optional[bool] = None,
        # 输入参数 output_attentions: 可选的是否输出注意力权重标志
        output_attentions: Optional[bool] = None,
        # 输入参数 output_hidden_states: 可选的是否输出隐藏状态标志
        output_hidden_states: Optional[bool] = None,
        # 输入参数 return_dict: 可选的是否返回字典结果标志
        return_dict: Optional[bool] = None,
# 添加文档字符串，描述 Time Series Transformer 模型输出原始隐藏状态，不带任何具体头部
# 并且包含时间序列转换器模型的开始文档字符串
@add_start_docstrings(
    "The bare Time Series Transformer Model outputting raw hidden-states without any specific head on top.",
    TIME_SERIES_TRANSFORMER_START_DOCSTRING,
)
class TimeSeriesTransformerModel(TimeSeriesTransformerPreTrainedModel):
    def __init__(self, config: TimeSeriesTransformerConfig):
        # 调用基类的初始化函数
        super().__init__(config)

        # 根据配置参数初始化缩放器
        if config.scaling == "mean" or config.scaling is True:
            self.scaler = TimeSeriesMeanScaler(config)
        elif config.scaling == "std":
            self.scaler = TimeSeriesStdScaler(config)
        else:
            self.scaler = TimeSeriesNOPScaler(config)

        # 如果静态分类特征数量大于0，则初始化特征嵌入器
        if config.num_static_categorical_features > 0:
            self.embedder = TimeSeriesFeatureEmbedder(
                cardinalities=config.cardinality,
                embedding_dims=config.embedding_dimension,
            )

        # 初始化 Transformer 编码器和解码器
        self.encoder = TimeSeriesTransformerEncoder(config)
        self.decoder = TimeSeriesTransformerDecoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    @property
    def _past_length(self) -> int:
        # 返回过去长度，为上下文长度加上最大滞后序列
        return self.config.context_length + max(self.config.lags_sequence)

    def get_lagged_subsequences(
        self, sequence: torch.Tensor, subsequences_length: int, shift: int = 0
    ) -> torch.Tensor:
        """
        获取给定序列的滞后子序列。返回形状为 (N, S, C, I) 的张量，其中 S = subsequences_length，I = len(indices)，包含滞后子序列。
        具体而言，lagged[i, j, :, k] = sequence[i, -indices[k]-S+j, :]。

        Args:
            sequence: Tensor
                应该从中提取滞后子序列的序列。形状：(N, T, C)。
            subsequences_length : int
                要提取的子序列的长度。
            shift: int
                向后移动这个数量的滞后。
        """
        sequence_length = sequence.shape[1]
        indices = [lag - shift for lag in self.config.lags_sequence]

        if max(indices) + subsequences_length > sequence_length:
            # 抛出值错误，指示滞后不能超出历史长度
            raise ValueError(
                f"lags cannot go further than history length, found lag {max(indices)} "
                f"while history length is only {sequence_length}"
            )

        # 存储滞后值
        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...])
        return torch.stack(lagged_values, dim=-1)
    # 创建用于神经网络输入的函数
    def create_network_inputs(
        # 过去值的张量
        past_values: torch.Tensor,
        # 过去时间特征的张量
        past_time_features: torch.Tensor,
        # 静态分类特征的张量（可选）
        static_categorical_features: Optional[torch.Tensor] = None,
        # 静态实数特征的张量（可选）
        static_real_features: Optional[torch.Tensor] = None,
        # 过去观察掩码的张量（可选）
        past_observed_mask: Optional[torch.Tensor] = None,
        # 未来值的张量（可选）
        future_values: Optional[torch.Tensor] = None,
        # 未来时间特征的张量（可选）
        future_time_features: Optional[torch.Tensor] = None,
```py  
    # 准备时间特征
    # 如果有未来值，则将过去时间特征和未来时间特征拼接在一起
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
    
    # 准备目标值
    # 如果没有提供过去观察掩码，则默认为全1
    if past_observed_mask is None:
        past_observed_mask = torch.ones_like(past_values)
    
    # 提取上下文
    context = past_values[:, -self.config.context_length :]
    observed_context = past_observed_mask[:, -self.config.context_length :]
    # 使用缩放器对上下文进行标准化
    _, loc, scale = self.scaler(context, observed_context)
    
    # 准备输入
    # 如果有未来值，则将过去值和未来值拼接在一起标准化
    # 否则只标准化过去值
    inputs = (
        (torch.cat((past_values, future_values), dim=1) - loc) / scale
        if future_values is not None
        else (past_values - loc) / scale
    )
    
    # 准备静态特征
    # 对平均绝对值和标准差取对数
    log_abs_loc = loc.abs().log1p() if self.config.input_size == 1 else loc.squeeze(1).abs().log1p()
    log_scale = scale.log() if self.config.input_size == 1 else scale.squeeze(1).log()
    static_feat = torch.cat((log_abs_loc, log_scale), dim=1)
    
    # 如果有其他静态特征，则拼接在一起
    if static_real_features is not None:
        static_feat = torch.cat((static_real_features, static_feat), dim=1)
    if static_categorical_features is not None:
        embedded_cat = self.embedder(static_categorical_features)
        static_feat = torch.cat((embedded_cat, static_feat), dim=1)
    
    # 扩展静态特征以匹配时间特征的维度
    expanded_static_feat = static_feat.unsqueeze(1).expand(-1, time_feat.shape[1], -1)
    
    # 将静态特征和时间特征拼接在一起作为最终特征
    features = torch.cat((expanded_static_feat, time_feat), dim=-1)
    
    # 准备滞后特征
    # 计算需要的滞后序列长度
    subsequences_length = (
        self.config.context_length + self.config.prediction_length
        if future_values is not None
        else self.config.context_length
    )
    # 获取滞后序列
    lagged_sequence = self.get_lagged_subsequences(sequence=inputs, subsequences_length=subsequences_length)
    lags_shape = lagged_sequence.shape
    reshaped_lagged_sequence = lagged_sequence.reshape(lags_shape[0], lags_shape[1], -1)
    
    # 检查滞后特征维度是否与时间特征维度一致
    if reshaped_lagged_sequence.shape[1] != time_feat.shape[1]:
        raise ValueError(
            f"input length {reshaped_lagged_sequence.shape[1]} and time feature lengths {time_feat.shape[1]} does not match"
        )
    
    # 将滞后特征和其他特征拼接在一起作为Transformer的输入
    transformer_inputs = torch.cat((reshaped_lagged_sequence, features), dim=-1)
    
    return transformer_inputs, loc, scale, static_feat
    # 定义一个前向传播函数
    def forward(
        # 过去的值序列
        self,
        past_values: torch.Tensor,
        # 过去的时间特征序列
        past_time_features: torch.Tensor,
        # 过去观察值的掩码序列
        past_observed_mask: torch.Tensor,
        # 可选的静态类别特征
        static_categorical_features: Optional[torch.Tensor] = None,
        # 可选的静态实数特征
        static_real_features: Optional[torch.Tensor] = None,
        # 可选的未来值序列
        future_values: Optional[torch.Tensor] = None,
        # 可选的未来时间特征序列
        future_time_features: Optional[torch.Tensor] = None,
        # 可选的解码器注意力掩码
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        # 可选的头部掩码
        head_mask: Optional[torch.Tensor] = None,
        # 可选的解码器头部掩码
        decoder_head_mask: Optional[torch.Tensor] = None,
        # 可选的交叉注意力头部掩码
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        # 可选的编码器输出
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        # 可选的过去的键值对
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # 是否输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 是否输出注意力权重
        output_attentions: Optional[bool] = None,
        # 是否使用缓存
        use_cache: Optional[bool] = None,
        # 是否返回字典格式的输出
        return_dict: Optional[bool] = None,
    ):
# 为时间序列预测的变压器模型添加分布头
class TimeSeriesTransformerForPrediction(TimeSeriesTransformerPreTrainedModel):
    # 初始化方法
    def __init__(self, config: TimeSeriesTransformerConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建时间序列变压器模型对象
        self.model = TimeSeriesTransformerModel(config)
        # 根据配置选择分布输出类型
        if config.distribution_output == "student_t":
            self.distribution_output = StudentTOutput(dim=config.input_size)
        elif config.distribution_output == "normal":
            self.distribution_output = NormalOutput(dim=config.input_size)
        elif config.distribution_output == "negative_binomial":
            self.distribution_output = NegativeBinomialOutput(dim=config.input_size)
        else:
            raise ValueError(f"Unknown distribution output {config.distribution_output}")

        # 根据模型配置获取参数投影
        self.parameter_projection = self.distribution_output.get_parameter_projection(self.model.config.d_model)
        # 设置目标形状
        self.target_shape = self.distribution_output.event_shape

        # 根据配置选择损失函数
        if config.loss == "nll":
            self.loss = nll
        else:
            raise ValueError(f"Unknown loss function {config.loss}")

        # 初始化分布输出权重并应用最终处理
        self.post_init()

    # 输出参数方法
    def output_params(self, dec_output):
        return self.parameter_projection(dec_output)

    # 获取编码器方法
    def get_encoder(self):
        return self.model.get_encoder()

    # 获取解码器方法
    def get_decoder(self):
        return self.model.get_decoder()

    # 忽略 torch.jit 注释的方法
    @torch.jit.ignore
    def output_distribution(self, params, loc=None, scale=None, trailing_n=None) -> torch.distributions.Distribution:
        sliced_params = params
        # 如果存在 trailing_n，对参数进行切片
        if trailing_n is not None:
            sliced_params = [p[:, -trailing_n:] for p in params]
        # 返回分布输出
        return self.distribution_output.distribution(sliced_params, loc=loc, scale=scale)

    # 为模型前向方法添加注释
    @add_start_docstrings_to_model_forward(TIME_SERIES_TRANSFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqTSModelOutput, config_class=_CONFIG_FOR_DOC)
    # 前向传播函数，用于模型的前向推理
    def forward(
        self,
        # 过去的数值值
        past_values: torch.Tensor,
        # 过去的时间特征
        past_time_features: torch.Tensor,
        # 过去的观测掩码
        past_observed_mask: torch.Tensor,
        # 静态分类特征，可选
        static_categorical_features: Optional[torch.Tensor] = None,
        # 静态实数特征，可选
        static_real_features: Optional[torch.Tensor] = None,
        # 未来的数值值，可选
        future_values: Optional[torch.Tensor] = None,
        # 未来的时间特征，可选
        future_time_features: Optional[torch.Tensor] = None,
        # 未来的观测掩码，可选
        future_observed_mask: Optional[torch.Tensor] = None,
        # 解码器注意力掩码，可选
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        # 头部掩码，可选
        head_mask: Optional[torch.Tensor] = None,
        # 解码器头部掩码，可选
        decoder_head_mask: Optional[torch.Tensor] = None,
        # 交叉注意力头部掩码，可选
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        # 编码器输出，可选
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        # 过去的关键值，可选
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # 是否输出隐藏状态，可选
        output_hidden_states: Optional[bool] = None,
        # 是否输出注意力，可选
        output_attentions: Optional[bool] = None,
        # 是否使用缓存，可选
        use_cache: Optional[bool] = None,
        # 是否返回字典，可选
        return_dict: Optional[bool] = None,
    # 禁用梯度
    @torch.no_grad()
    # 生成函数，用于模型的生成任务
    def generate(
        self,
        # 过去的数值值
        past_values: torch.Tensor,
        # 过去的时间特征
        past_time_features: torch.Tensor,
        # 未来的时间特征
        future_time_features: torch.Tensor,
        # 过去的观测掩码，可选
        past_observed_mask: Optional[torch.Tensor] = None,
        # 静态分类特征，可选
        static_categorical_features: Optional[torch.Tensor] = None,
        # 静态实数特征，可选
        static_real_features: Optional[torch.Tensor] = None,
        # 是否输出注意力，可选
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，可选
        output_hidden_states: Optional[bool] = None,
```