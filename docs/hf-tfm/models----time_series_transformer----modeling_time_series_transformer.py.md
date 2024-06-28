# `.\models\time_series_transformer\modeling_time_series_transformer.py`

```
# 设置编码格式为 UTF-8
# 版权声明，指出本代码版权归 HuggingFace Inc. 和 Amazon.com, Inc. 所有
# 根据 Apache 许可证 2.0 版本授权使用本代码
# 除非符合许可证的要求，否则禁止使用本文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 根据适用法律或书面同意，本软件根据"原样"分发，不提供任何明示或暗示的担保或条件。
# 有关详细信息，请参阅许可证。

""" PyTorch 时间序列 Transformer 模型。"""

from typing import List, Optional, Tuple, Union

import numpy as np  # 引入 NumPy 库
import torch  # 引入 PyTorch 库
from torch import nn  # 引入 PyTorch 的 nn 模块

# 从其他模块引入相关内容
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
from .configuration_time_series_transformer import TimeSeriesTransformerConfig  # 从本地模块导入特定配置

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

_CONFIG_FOR_DOC = "TimeSeriesTransformerConfig"  # 针对文档的配置信息

# 预训练模型存档列表，包含 HuggingFace 的时间序列 Transformer 模型
TIME_SERIES_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "huggingface/time-series-transformer-tourism-monthly",
    # 查看所有 TimeSeriesTransformer 模型：https://huggingface.co/models?filter=time_series_transformer
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

        self.num_features = len(cardinalities)  # 特征数量为分类特征的个数
        self.embedders = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(cardinalities, embedding_dims)])
        # 创建一个 nn.ModuleList，其中包含多个嵌入层，每个嵌入层对应一个分类特征的不同嵌入维度
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 如果特征数大于1，则将最后一个维度切片，得到一个长度为self.num_features的数组，形状为(N,T)或(N)
        if self.num_features > 1:
            cat_feature_slices = torch.chunk(features, self.num_features, dim=-1)
        else:
            cat_feature_slices = [features]

        # 对每个切片进行嵌入操作，然后在最后一个维度上进行连接
        return torch.cat(
            [
                embed(cat_feature_slice.squeeze(-1))
                for embed, cat_feature_slice in zip(self.embedders, cat_feature_slices)
            ],
            dim=-1,
        )
class TimeSeriesStdScaler(nn.Module):
    """
    Standardize features by calculating the mean and scaling along the first dimension, and then normalizes it by
    subtracting from the mean and dividing by the standard deviation.
    """

    def __init__(self, config: TimeSeriesTransformerConfig):
        super().__init__()
        # 初始化标准化器，从配置中获取相关参数，如果不存在则使用默认值
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True
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
        # 计算每个特征维度的均值
        denominator = observed_indicator.sum(self.dim, keepdim=self.keepdim)
        # 确保分母至少为1，避免除零错误
        denominator = denominator.clamp_min(1.0)
        # 计算均值
        loc = (data * observed_indicator).sum(self.dim, keepdim=self.keepdim) / denominator

        # 计算方差
        variance = (((data - loc) * observed_indicator) ** 2).sum(self.dim, keepdim=self.keepdim) / denominator
        # 计算标准差，加上最小缩放值以确保数值稳定性
        scale = torch.sqrt(variance + self.minimum_scale)
        # 标准化数据并返回标准化后的数据、均值和标准差
        return (data - loc) / scale, loc, scale


class TimeSeriesMeanScaler(nn.Module):
    """
    Computes a scaling factor as the weighted average absolute value along the first dimension, and scales the data
    accordingly.
    """

    def __init__(self, config: TimeSeriesTransformerConfig):
        super().__init__()
        # 初始化均值标准化器，从配置中获取相关参数，如果不存在则使用默认值
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True
        self.minimum_scale = config.minimum_scale if hasattr(config, "minimum_scale") else 1e-10
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
            `torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`:
                Normalized data.
        """
        # 计算每个特征维度的加权平均绝对值
        scale = (torch.abs(data) * observed_indicator).sum(self.dim, keepdim=self.keepdim)
        # 加上最小缩放值以确保数值稳定性
        scale = scale.clamp_min(self.minimum_scale)
        
        # 如果存在默认缩放值，则用默认缩放值代替
        if self.default_scale is not None:
            scale = torch.where(scale == 0, torch.tensor(self.default_scale).to(scale.device), scale)
        
        # 标准化数据并返回
        return data / scale
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
        # Calculate the sum of absolute values of `data` multiplied by `observed_indicator`,
        # summed across the specified dimension (`self.dim`).
        ts_sum = (data * observed_indicator).abs().sum(self.dim, keepdim=True)
        
        # Count the number of True values in `observed_indicator` across the specified dimension (`self.dim`).
        num_observed = observed_indicator.sum(self.dim, keepdim=True)

        # Compute the scale as the element-wise division of `ts_sum` by `num_observed`, ensuring `num_observed`
        # is clamped to a minimum of 1 to avoid division by zero.
        scale = ts_sum / torch.clamp(num_observed, min=1)

        # If `default_scale` is not provided, calculate it as the sum of `ts_sum` across the batches divided by
        # the sum of `num_observed` across the batches, clamping the latter to a minimum of 1.
        if self.default_scale is None:
            batch_sum = ts_sum.sum(dim=0)
            batch_observations = torch.clamp(num_observed.sum(0), min=1)
            default_scale = torch.squeeze(batch_sum / batch_observations)
        else:
            # If `default_scale` is provided, use it across all elements of `scale`.
            default_scale = self.default_scale * torch.ones_like(scale)

        # Apply `default_scale` where `num_observed` is greater than 0; otherwise, use `scale`.
        scale = torch.where(num_observed > 0, scale, default_scale)

        # Ensure that `scale` is not less than `self.minimum_scale`.
        scale = torch.clamp(scale, min=self.minimum_scale)

        # Normalize `data` by dividing it element-wise by `scale`.
        scaled_data = data / scale

        # If `self.keepdim` is False, squeeze `scale` along the specified dimension (`self.dim`).
        if not self.keepdim:
            scale = scale.squeeze(dim=self.dim)

        # Return `scaled_data`, a tensor of zeros like `scale`, and `scale` itself.
        return scaled_data, torch.zeros_like(scale), scale
class TimeSeriesNOPScaler(nn.Module):
    """
    Assigns a scaling factor equal to 1 along the first dimension, and therefore applies no scaling to the input data.
    """

    def __init__(self, config: TimeSeriesTransformerConfig):
        super().__init__()
        # 初始化时根据配置设置维度和是否保持维度信息
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
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
        # 计算沿指定维度的均值，作为数据的缩放因子
        scale = torch.ones_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        # 计算沿指定维度的均值，作为数据的位置偏移
        loc = torch.zeros_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        # 返回原始数据、位置偏移和缩放因子
        return data, loc, scale


def nll(input: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the negative log likelihood loss from input distribution with respect to target.
    """
    # 计算输入分布相对于目标的负对数似然损失
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
        # 根据权重计算加权平均值，并处理权重为零的情况
        weighted_tensor = torch.where(weights != 0, input_tensor * weights, torch.zeros_like(input_tensor))
        # 对权重求和，确保不为零
        sum_weights = torch.clamp(weights.sum(dim=dim) if dim else weights.sum(), min=1.0)
        # 返回加权平均值
        return (weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()) / sum_weights
    else:
        # 若未提供权重，则计算简单的平均值
        return input_tensor.mean(dim=dim)


# Copied from transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding with Marian->TimeSeries
class TimeSeriesSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""
    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None) -> None:
        # 调用父类的初始化方法，传入位置数量和嵌入维度
        super().__init__(num_positions, embedding_dim)
        # 初始化权重矩阵，并将结果赋给self.weight
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """
        与 XLM 的 create_sinusoidal_embeddings 方法相同，不同之处在于特征没有交错。
        余弦特征位于向量的第二半部分 [dim // 2:]
        """
        # 获取输出张量的维度信息
        n_pos, dim = out.shape
        # 创建位置编码数组，使用正弦和余弦函数填充
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        # 设置梯度计算为False，以避免在pytorch-1.8+版本中的错误
        out.requires_grad = False
        # 根据维度是否为偶数，确定sentinel的值，用于分割正弦和余弦编码
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        # 将正弦编码部分赋值给out的前半部分
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        # 将余弦编码部分赋值给out的后半部分
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        # 分离out张量，使其不再追踪计算梯度
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0) -> torch.Tensor:
        """`input_ids_shape` 期望是 [bsz x seqlen]。"""
        # 解析输入形状，bsz是批量大小，seq_len是序列长度
        bsz, seq_len = input_ids_shape[:2]
        # 根据序列长度生成位置张量，加上past_key_values_length作为起始位置
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        # 调用父类的forward方法，传入位置张量，并返回结果张量
        return super().forward(positions)
class TimeSeriesValueEmbedding(nn.Module):
    def __init__(self, feature_size, d_model):
        super().__init__()
        # 初始化线性层，用于将特征大小 feature_size 的输入投影到维度为 d_model 的输出空间，无偏置
        self.value_projection = nn.Linear(in_features=feature_size, out_features=d_model, bias=False)

    def forward(self, x):
        # 前向传播函数，将输入 x 投影到 d_model 维度的空间
        return self.value_projection(x)


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->TimeSeriesTransformer
class TimeSeriesTransformerAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

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
        # 初始化注意力层参数
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        # 检查 embed_dim 是否能被 num_heads 整除
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        # 缩放因子，用于缩放注意力分数
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        # 初始化投影层：键（k_proj）、值（v_proj）、查询（q_proj）和输出（out_proj）
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 重新整形张量以适应多头注意力计算的形状
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 实现多头注意力机制的前向传播
        pass  # 实际实现未包含在提供的代码片段中

# Copied from transformers.models.bart.modeling_bart.BartEncoderLayer with Bart->TimeSeriesTransformer, BART->TIME_SERIES_TRANSFORMER
class TimeSeriesTransformerEncoderLayer(nn.Module):
    # 此类定义编码器层的结构，与 TimeSeriesTransformer 模型相关
    # 初始化函数，接受一个时间序列转换器的配置对象作为参数
    def __init__(self, config: TimeSeriesTransformerConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 设置嵌入维度等于配置中的模型维度
        self.embed_dim = config.d_model
    
        # 创建自注意力机制，根据配置选择不同的实现类
        self.self_attn = TIME_SERIES_TRANSFORMER_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )
        # 自注意力层归一化
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        # 设置丢弃率
        self.dropout = config.dropout
        # 激活函数选择，根据配置中的激活函数名从预定义映射中选择
        self.activation_fn = ACT2FN[config.activation_function]
        # 激活函数的丢弃率
        self.activation_dropout = config.activation_dropout
        
        # 第一个全连接层，将嵌入维度映射到编码器前馈神经网络维度
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        # 第二个全连接层，将编码器前馈神经网络维度映射回嵌入维度
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        # 最终的层归一化
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
        residual = hidden_states  # 保存输入的残差连接
        # 使用 self_attn 层进行自注意力计算
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 应用 dropout，防止过拟合
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 添加残差连接
        hidden_states = residual + hidden_states
        # 对输出应用 layer normalization
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states  # 再次保存残差连接
        # 应用激活函数和线性变换 fc1
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 应用 dropout，防止过拟合
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # 应用线性变换 fc2
        hidden_states = self.fc2(hidden_states)
        # 应用 dropout，防止过拟合
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 添加残差连接
        hidden_states = residual + hidden_states
        # 对输出应用 layer normalization
        hidden_states = self.final_layer_norm(hidden_states)

        # 如果 hidden_states 的数据类型为 torch.float16 并且包含无穷大或 NaN 值，则进行截断处理
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)  # 输出为元组形式，包含最终的隐藏状态

        # 如果需要返回注意力权重 tensors，则将 attn_weights 添加到输出中
        if output_attentions:
            outputs += (attn_weights,)  # 将注意力权重 tensors 添加到输出元组中

        return outputs
# 定义一个字典，将字符串 "eager" 映射到 TimeSeriesTransformerAttention 类
TIME_SERIES_TRANSFORMER_ATTENTION_CLASSES = {
    "eager": TimeSeriesTransformerAttention,
}


# 从 transformers.models.bart.modeling_bart.BartDecoderLayer 复制而来，将 Bart 替换为 TimeSeriesTransformer，BART 替换为 TIME_SERIES_TRANSFORMER
class TimeSeriesTransformerDecoderLayer(nn.Module):
    def __init__(self, config: TimeSeriesTransformerConfig):
        super().__init__()
        self.embed_dim = config.d_model  # 初始化 embedding 维度为 config 中的 d_model

        # 初始化自注意力机制，使用预先选择的注意力类
        self.self_attn = TIME_SERIES_TRANSFORMER_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
        )

        self.dropout = config.dropout  # 初始化 dropout 概率
        self.activation_fn = ACT2FN[config.activation_function]  # 激活函数根据配置选择
        self.activation_dropout = config.activation_dropout  # 激活函数的 dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)  # LayerNorm 层，标准化自注意力输出

        # 初始化编码器-解码器注意力机制，使用预先选择的注意力类
        self.encoder_attn = TIME_SERIES_TRANSFORMER_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)  # LayerNorm 层，标准化编码器-解码器注意力输出

        # 第一个全连接层，用于多头自注意力和编码器-解码器注意力之后的线性变换
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        # 第二个全连接层，用于上述变换后的再次线性变换
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)

        self.final_layer_norm = nn.LayerNorm(self.embed_dim)  # 最终的 LayerNorm 层，标准化最终输出

    # 前向传播函数，处理各个层的输入输出以及相关参数
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
    config_class = TimeSeriesTransformerConfig  # 配置类的引用为 TimeSeriesTransformerConfig
    base_model_prefix = "model"  # 基础模型前缀为 "model"
    main_input_name = "past_values"  # 主输入名称为 "past_values"
    supports_gradient_checkpointing = True  # 支持梯度检查点

    # 初始化权重函数，根据模块类型不同，应用不同的初始化方法
    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)  # 线性层权重初始化为正态分布
            if module.bias is not None:
                module.bias.data.zero_()  # 如果有偏置，初始化为零
        elif isinstance(module, TimeSeriesSinusoidalPositionalEmbedding):
            pass  # 对于特定类型的模块，不进行任何初始化
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)  # Embedding 层权重初始化为正态分布
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()  # 如果有 padding_idx，将其对应的权重初始化为零

# 开始模型文档字符串，用于 TimeSeriesTransformer
TIME_SERIES_TRANSFORMER_START_DOCSTRING = r"""
    # 该模型继承自PreTrainedModel。查看超类文档，了解库实现的所有模型的通用方法（例如下载或保存、调整输入嵌入、剪枝头等）。
    # 该模型也是PyTorch的torch.nn.Module子类。将其视为常规的PyTorch模块，并查阅PyTorch文档，了解与一般用法和行为有关的所有事项。
    
    # 参数:
    # config ([TimeSeriesTransformerConfig]):
    # 模型配置类，包含模型的所有参数。使用配置文件进行初始化不会加载与模型关联的权重，只会加载配置。查看PreTrainedModel.from_pretrained方法以加载模型权重。
"""
TIME_SERIES_TRANSFORMER_INPUTS_DOCSTRING = r"""
"""

class TimeSeriesTransformerEncoder(TimeSeriesTransformerPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`TimeSeriesTransformerEncoderLayer`].

    Args:
        config: TimeSeriesTransformerConfig
    """

    def __init__(self, config: TimeSeriesTransformerConfig):
        super().__init__(config)

        self.dropout = config.dropout  # 设置 dropout 比率
        self.layerdrop = config.encoder_layerdrop  # 设置 encoder 层的 dropout 比率
        if config.prediction_length is None:
            raise ValueError("The `prediction_length` config needs to be specified.")

        self.value_embedding = TimeSeriesValueEmbedding(feature_size=config.feature_size, d_model=config.d_model)  # 定义时间序列数值嵌入
        self.embed_positions = TimeSeriesSinusoidalPositionalEmbedding(
            config.context_length + config.prediction_length, config.d_model
        )  # 定义位置嵌入，使用正弦函数
        self.layers = nn.ModuleList([TimeSeriesTransformerEncoderLayer(config) for _ in range(config.encoder_layers)])  # 创建 encoder 层列表
        self.layernorm_embedding = nn.LayerNorm(config.d_model)  # 对嵌入进行 layer normalization

        self.gradient_checkpointing = False  # 是否使用梯度检查点
        # 初始化权重并进行最终处理
        self.post_init()

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):  # 前向传播函数定义
        """
        在输入的时间序列数据上执行前向传播。

        Args:
            attention_mask: 可选的注意力遮罩张量
            head_mask: 可选的注意力头部遮罩张量
            inputs_embeds: 可选的输入嵌入张量
            output_attentions: 可选的是否输出注意力张量
            output_hidden_states: 可选的是否输出隐藏状态张量
            return_dict: 可选的是否返回字典形式的输出

        Returns:
            输出字典或元组，根据 return_dict 参数决定
        """
        pass  # 实际代码会在这里完成

class TimeSeriesTransformerDecoder(TimeSeriesTransformerPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a
    [`TimeSeriesTransformerDecoderLayer`]

    Args:
        config: TimeSeriesTransformerConfig
    """

    def __init__(self, config: TimeSeriesTransformerConfig):
        super().__init__(config)
        self.dropout = config.dropout  # 设置 dropout 比率
        self.layerdrop = config.decoder_layerdrop  # 设置 decoder 层的 dropout 比率
        if config.prediction_length is None:
            raise ValueError("The `prediction_length` config needs to be specified.")

        self.value_embedding = TimeSeriesValueEmbedding(feature_size=config.feature_size, d_model=config.d_model)  # 定义时间序列数值嵌入
        self.embed_positions = TimeSeriesSinusoidalPositionalEmbedding(
            config.context_length + config.prediction_length, config.d_model
        )  # 定义位置嵌入，使用正弦函数
        self.layers = nn.ModuleList([TimeSeriesTransformerDecoderLayer(config) for _ in range(config.decoder_layers)])  # 创建 decoder 层列表
        self.layernorm_embedding = nn.LayerNorm(config.d_model)  # 对嵌入进行 layer normalization

        self.gradient_checkpointing = False  # 是否使用梯度检查点
        # 初始化权重并进行最终处理
        self.post_init()
    # 定义一个方法用于执行前向传播，通常用于模型推理或训练过程中的前向计算
    def forward(
        self,
        # 可选参数：用于注意力机制的掩码，指定哪些位置是padding的，哪些是有效的
        attention_mask: Optional[torch.Tensor] = None,
        # 可选参数：编码器的隐藏状态，用于跨层注意力机制或连接不同模型的情况
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        # 可选参数：编码器的注意力掩码，指定哪些位置是padding的，哪些是有效的
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        # 可选参数：用于掩盖指定的注意力头，以便控制每个头的重要性
        head_mask: Optional[torch.Tensor] = None,
        # 可选参数：用于跨层注意力机制中掩盖指定的注意力头
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        # 可选参数：过去的键值对，用于自回归生成过程中保存先前计算的键值对
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # 可选参数：用于指定输入的嵌入表示，覆盖模型内部嵌入层的输入
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 可选参数：是否使用缓存来加快计算，适用于需要多次调用的场景
        use_cache: Optional[bool] = None,
        # 可选参数：是否输出注意力权重
        output_attentions: Optional[bool] = None,
        # 可选参数：是否输出所有隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 可选参数：是否返回一个字典格式的输出
        return_dict: Optional[bool] = None,
# 添加文档字符串注释，描述该类作为不带特定顶部头部的裸时间序列Transformer模型的输出
@add_start_docstrings(
    "The bare Time Series Transformer Model outputting raw hidden-states without any specific head on top.",
    TIME_SERIES_TRANSFORMER_START_DOCSTRING,
)
class TimeSeriesTransformerModel(TimeSeriesTransformerPreTrainedModel):
    
    # 初始化方法，接收一个TimeSeriesTransformerConfig类型的参数config
    def __init__(self, config: TimeSeriesTransformerConfig):
        super().__init__(config)

        # 根据配置选择合适的数据缩放器
        if config.scaling == "mean" or config.scaling is True:
            self.scaler = TimeSeriesMeanScaler(config)
        elif config.scaling == "std":
            self.scaler = TimeSeriesStdScaler(config)
        else:
            self.scaler = TimeSeriesNOPScaler(config)

        # 如果存在静态分类特征，则初始化时间序列特征嵌入器
        if config.num_static_categorical_features > 0:
            self.embedder = TimeSeriesFeatureEmbedder(
                cardinalities=config.cardinality,
                embedding_dims=config.embedding_dimension,
            )

        # 初始化Transformer编码器和解码器
        self.encoder = TimeSeriesTransformerEncoder(config)
        self.decoder = TimeSeriesTransformerDecoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 属性方法，返回过去长度，即上下文长度加上最大滞后序列长度
    @property
    def _past_length(self) -> int:
        return self.config.context_length + max(self.config.lags_sequence)

    # 方法用于获取给定序列的滞后子序列
    def get_lagged_subsequences(
        self, sequence: torch.Tensor, subsequences_length: int, shift: int = 0
    ) -> torch.Tensor:
        """
        Returns lagged subsequences of a given sequence. Returns a tensor of shape (N, S, C, I),
            where S = subsequences_length and I = len(indices), containing lagged subsequences. Specifically, lagged[i,
            j, :, k] = sequence[i, -indices[k]-S+j, :].

        Args:
            sequence: Tensor
                The sequence from which lagged subsequences should be extracted. Shape: (N, T, C).
            subsequences_length : int
                Length of the subsequences to be extracted.
            shift: int
                Shift the lags by this amount back.
        """
        # 获取输入序列的长度
        sequence_length = sequence.shape[1]
        
        # 根据配置中的滞后序列计算滞后的索引
        indices = [lag - shift for lag in self.config.lags_sequence]

        # 检查滞后的最大索引加上子序列长度是否超过序列的长度，如果超过则抛出异常
        if max(indices) + subsequences_length > sequence_length:
            raise ValueError(
                f"lags cannot go further than history length, found lag {max(indices)} "
                f"while history length is only {sequence_length}"
            )

        # 初始化一个列表，用于存储滞后的值
        lagged_values = []
        
        # 遍历每个滞后索引，提取对应的滞后子序列并存储在列表中
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...])
        
        # 将所有滞后子序列堆叠成一个张量并返回，最后一个维度表示滞后的数量
        return torch.stack(lagged_values, dim=-1)
    # 创建网络输入的方法，用于组装神经网络所需的输入数据
    def create_network_inputs(
        self,
        # 过去的数值数据，作为神经网络输入的一部分
        past_values: torch.Tensor,
        # 过去的时间特征数据，用于神经网络输入
        past_time_features: torch.Tensor,
        # 可选参数：静态分类特征数据，如果存在的话
        static_categorical_features: Optional[torch.Tensor] = None,
        # 可选参数：静态实数特征数据，如果存在的话
        static_real_features: Optional[torch.Tensor] = None,
        # 可选参数：过去观测掩码，如果存在的话
        past_observed_mask: Optional[torch.Tensor] = None,
        # 可选参数：未来的数值数据，如果存在的话
        future_values: Optional[torch.Tensor] = None,
        # 可选参数：未来的时间特征数据，如果存在的话
        future_time_features: Optional[torch.Tensor] = None,
        # time feature
        # 按照指定的上下文长度将过去时间特征和未来时间特征连接起来
        time_feat = (
            torch.cat(
                (
                    past_time_features[:, self._past_length - self.config.context_length :, ...],
                    future_time_features,
                ),
                dim=1,
            )
            if future_values is not None  # 如果存在未来数值，则连接未来时间特征
            else past_time_features[:, self._past_length - self.config.context_length :, ...]  # 否则仅使用过去时间特征
        )

        # target
        # 如果过去观察掩码为空，则将其初始化为与过去数值形状相同的全1张量
        if past_observed_mask is None:
            past_observed_mask = torch.ones_like(past_values)

        # 获取当前上下文的数值
        context = past_values[:, -self.config.context_length :]
        # 获取当前上下文的观察掩码
        observed_context = past_observed_mask[:, -self.config.context_length :]
        # 使用规模器对象处理上下文数据，返回位置、缩放参数
        _, loc, scale = self.scaler(context, observed_context)

        # 构建模型输入数据
        inputs = (
            (torch.cat((past_values, future_values), dim=1) - loc) / scale
            if future_values is not None  # 如果存在未来数值，则将过去和未来数值归一化
            else (past_values - loc) / scale  # 否则仅归一化过去数值
        )

        # static features
        # 计算位置绝对值的对数并添加到静态特征中
        log_abs_loc = loc.abs().log1p() if self.config.input_size == 1 else loc.squeeze(1).abs().log1p()
        # 计算缩放参数的对数并添加到静态特征中
        log_scale = scale.log() if self.config.input_size == 1 else scale.squeeze(1).log()
        # 合并位置和缩放的对数作为静态特征
        static_feat = torch.cat((log_abs_loc, log_scale), dim=1)

        # 如果存在实数类型的静态特征，则将其添加到静态特征中
        if static_real_features is not None:
            static_feat = torch.cat((static_real_features, static_feat), dim=1)
        # 如果存在分类类型的静态特征，则将其嵌入后添加到静态特征中
        if static_categorical_features is not None:
            embedded_cat = self.embedder(static_categorical_features)
            static_feat = torch.cat((embedded_cat, static_feat), dim=1)
        # 将静态特征扩展以匹配时间特征的长度
        expanded_static_feat = static_feat.unsqueeze(1).expand(-1, time_feat.shape[1], -1)

        # all features
        # 将静态特征和时间特征连接成为模型的所有输入特征
        features = torch.cat((expanded_static_feat, time_feat), dim=-1)

        # lagged features
        # 计算滞后序列的长度
        subsequences_length = (
            self.config.context_length + self.config.prediction_length
            if future_values is not None  # 如果存在未来数值，则将预测长度加入上下文长度
            else self.config.context_length  # 否则只使用上下文长度
        )
        # 获取滞后的子序列
        lagged_sequence = self.get_lagged_subsequences(sequence=inputs, subsequences_length=subsequences_length)
        lags_shape = lagged_sequence.shape
        # 将滞后序列重塑为三维张量
        reshaped_lagged_sequence = lagged_sequence.reshape(lags_shape[0], lags_shape[1], -1)

        # 如果重塑后的滞后序列长度与时间特征长度不匹配，则引发数值错误
        if reshaped_lagged_sequence.shape[1] != time_feat.shape[1]:
            raise ValueError(
                f"input length {reshaped_lagged_sequence.shape[1]} and time feature lengths {time_feat.shape[1]} does not match"
            )

        # transformer inputs
        # 将重塑后的滞后序列和所有特征连接成为变换器的输入
        transformer_inputs = torch.cat((reshaped_lagged_sequence, features), dim=-1)

        # 返回变换器的输入、位置、缩放参数和静态特征
        return transformer_inputs, loc, scale, static_feat
    # 定义一个方法 `forward`，用于模型的前向传播
    def forward(
        self,
        # 过去时间步的值，类型为 Torch 张量
        past_values: torch.Tensor,
        # 过去时间步的时间特征，类型为 Torch 张量
        past_time_features: torch.Tensor,
        # 过去时间步的观察掩码，类型为 Torch 张量
        past_observed_mask: torch.Tensor,
        # 静态分类特征，可选的 Torch 张量
        static_categorical_features: Optional[torch.Tensor] = None,
        # 静态实数特征，可选的 Torch 张量
        static_real_features: Optional[torch.Tensor] = None,
        # 未来时间步的值，可选的 Torch 张量
        future_values: Optional[torch.Tensor] = None,
        # 未来时间步的时间特征，可选的 Torch 张量
        future_time_features: Optional[torch.Tensor] = None,
        # 解码器注意力掩码，可选的 Torch 长整型张量
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        # 头部掩码，可选的 Torch 张量
        head_mask: Optional[torch.Tensor] = None,
        # 解码器头部掩码，可选的 Torch 张量
        decoder_head_mask: Optional[torch.Tensor] = None,
        # 交叉注意力头部掩码，可选的 Torch 张量
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        # 编码器输出列表，可选的浮点数 Torch 列表
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        # 过去关键值列表，可选的浮点数 Torch 列表
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # 输出隐藏状态的标志，可选的布尔值
        output_hidden_states: Optional[bool] = None,
        # 输出注意力的标志，可选的布尔值
        output_attentions: Optional[bool] = None,
        # 是否使用缓存的标志，可选的布尔值
        use_cache: Optional[bool] = None,
        # 是否返回字典的标志，可选的布尔值
        return_dict: Optional[bool] = None,
# 给 TimeSeriesTransformerForPrediction 类添加文档字符串，描述其作为基于时间序列的变压器模型预测模型的用途和结构
@add_start_docstrings(
    "The Time Series Transformer Model with a distribution head on top for time-series forecasting.",
    TIME_SERIES_TRANSFORMER_START_DOCSTRING,
)
# 定义 TimeSeriesTransformerForPrediction 类，继承自 TimeSeriesTransformerPreTrainedModel
class TimeSeriesTransformerForPrediction(TimeSeriesTransformerPreTrainedModel):
    
    # 初始化方法，接受一个 TimeSeriesTransformerConfig 类型的 config 参数
    def __init__(self, config: TimeSeriesTransformerConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        
        # 创建 TimeSeriesTransformerModel 模型实例
        self.model = TimeSeriesTransformerModel(config)
        
        # 根据 config 中的 distribution_output 参数选择合适的输出分布
        if config.distribution_output == "student_t":
            self.distribution_output = StudentTOutput(dim=config.input_size)
        elif config.distribution_output == "normal":
            self.distribution_output = NormalOutput(dim=config.input_size)
        elif config.distribution_output == "negative_binomial":
            self.distribution_output = NegativeBinomialOutput(dim=config.input_size)
        else:
            raise ValueError(f"Unknown distribution output {config.distribution_output}")
        
        # 根据模型配置的维度，初始化参数投影
        self.parameter_projection = self.distribution_output.get_parameter_projection(self.model.config.d_model)
        
        # 获取分布输出的事件形状
        self.target_shape = self.distribution_output.event_shape
        
        # 根据 config 中的 loss 参数选择损失函数
        if config.loss == "nll":
            self.loss = nll
        else:
            raise ValueError(f"Unknown loss function {config.loss}")
        
        # 调用后处理初始化方法
        self.post_init()

    # 输出参数的方法，接受解码器输出作为参数，返回参数投影后的结果
    def output_params(self, dec_output):
        return self.parameter_projection(dec_output)

    # 获取编码器的方法，返回模型的编码器部分
    def get_encoder(self):
        return self.model.get_encoder()

    # 获取解码器的方法，返回模型的解码器部分
    def get_decoder(self):
        return self.model.get_decoder()

    # 输出分布的方法，接受参数 params、loc、scale 和 trailing_n，返回分布对象
    @torch.jit.ignore
    def output_distribution(self, params, loc=None, scale=None, trailing_n=None) -> torch.distributions.Distribution:
        # 如果 trailing_n 不为 None，则对 params 进行切片操作
        sliced_params = params
        if trailing_n is not None:
            sliced_params = [p[:, -trailing_n:] for p in params]
        # 调用 distribution_output 对象的 distribution 方法生成分布对象
        return self.distribution_output.distribution(sliced_params, loc=loc, scale=scale)

    # 将 TIME_SERIES_TRANSFORMER_INPUTS_DOCSTRING 和 _CONFIG_FOR_DOC 添加到模型前向方法的文档字符串中
    @add_start_docstrings_to_model_forward(TIME_SERIES_TRANSFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqTSModelOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个方法 `forward`，用于模型的前向传播
    def forward(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        past_observed_mask: torch.Tensor,
        static_categorical_features: Optional[torch.Tensor] = None,
        static_real_features: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        future_time_features: Optional[torch.Tensor] = None,
        future_observed_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 标记这个方法不需要梯度计算
        @torch.no_grad()
        # 定义一个方法 `generate`，用于模型的生成
        def generate(
            self,
            past_values: torch.Tensor,
            past_time_features: torch.Tensor,
            future_time_features: torch.Tensor,
            past_observed_mask: Optional[torch.Tensor] = None,
            static_categorical_features: Optional[torch.Tensor] = None,
            static_real_features: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
        ):
```