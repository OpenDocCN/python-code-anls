# `.\models\informer\modeling_informer.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 Amazon 和 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
""" PyTorch Informer 模型。"""

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

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 用于文档的配置信息
_CONFIG_FOR_DOC = "InformerConfig"

# 预训练模型的存档列表
INFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "huggingface/informer-tourism-monthly",
    # 查看所有 Informer 模型 https://huggingface.co/models?filter=informer
]

# 从 transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesFeatureEmbedder 复制并修改为 InformerFeatureEmbedder
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

        # 计算特征数量
        self.num_features = len(cardinalities)
        # 创建嵌入层列表，每个嵌入层对应一个分类特征
        self.embedders = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(cardinalities, embedding_dims)])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.num_features > 1:
            # 将最后一个维度切片，得到一个长度为 self.num_features 的数组，形状为 (N, T) 或 (N)
            cat_feature_slices = torch.chunk(features, self.num_features, dim=-1)
        else:
            cat_feature_slices = [features]

        # 将嵌入后的特征拼接在一起
        return torch.cat(
            [
                embed(cat_feature_slice.squeeze(-1))
                for embed, cat_feature_slice in zip(self.embedders, cat_feature_slices)
            ],
            dim=-1,
        )
# 定义一个类InformerStdScaler，用于标准化特征数据
class InformerStdScaler(nn.Module):
    """
    Standardize features by calculating the mean and scaling along the first dimension, and then normalizes it by
    subtracting from the mean and dividing by the standard deviation.
    """

    def __init__(self, config: InformerConfig):
        super().__init__()
        # 设置标准化维度，默认为1
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        # 是否保持维度，默认为True
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True
        # 最小缩放值，默认为1e-5
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
        # 计算观察指示器的和
        denominator = observed_indicator.sum(self.dim, keepdim=self.keepdim)
        # 将和值限制在最小值为1.0
        denominator = denominator.clamp_min(1.0)
        # 计算均值
        loc = (data * observed_indicator).sum(self.dim, keepdim=self.keepdim) / denominator

        # 计算方差
        variance = (((data - loc) * observed_indicator) ** 2).sum(self.dim, keepdim=self.keepdim) / denominator
        # 计算标准差
        scale = torch.sqrt(variance + self.minimum_scale)
        return (data - loc) / scale, loc, scale


# 定义一个类InformerMeanScaler，用于计算加权平均绝对值并相应地缩放数据
class InformerMeanScaler(nn.Module):
    """
    Computes a scaling factor as the weighted average absolute value along the first dimension, and scales the data
    accordingly.
    """

    def __init__(self, config: InformerConfig):
        super().__init__()
        # 设置缩放维度，默认为1
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        # 是否保持维度��默认为True
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True
        # 最小缩放值，默认为1e-10
        self.minimum_scale = config.minimum_scale if hasattr(config, "minimum_scale") else 1e-10
        # 默认缩放值，默认为None
        self.default_scale = config.default_scale if hasattr(config, "default_scale") else None

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    def forward(self, data: torch.Tensor, observed_indicator: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                输入用于批量归一化计算的数据
            observed_indicator (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                用于计算观察指标的规模。
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        # 计算数据乘以观察指标的绝对值之和
        ts_sum = (data * observed_indicator).abs().sum(self.dim, keepdim=True)
        # 计算观察指标的总数
        num_observed = observed_indicator.sum(self.dim, keepdim=True)

        # 计算规模，避免除以0
        scale = ts_sum / torch.clamp(num_observed, min=1)

        # 如果提供了`default_scale`，则使用它，否则使用批次的规模
        if self.default_scale is None:
            # 计算批次总和
            batch_sum = ts_sum.sum(dim=0)
            # 计算批次观察数
            batch_observations = torch.clamp(num_observed.sum(0), min=1)
            default_scale = torch.squeeze(batch_sum / batch_observations)
        else:
            default_scale = self.default_scale * torch.ones_like(scale)

        # 在没有观察到的地方应用默认规模
        scale = torch.where(num_observed > 0, scale, default_scale)

        # 确保规模至少为`self.minimum_scale`
        scale = torch.clamp(scale, min=self.minimum_scale)
        # 对数据进行缩放
        scaled_data = data / scale

        # 如果不保持维度，则压缩规模
        if not self.keepdim:
            scale = scale.squeeze(dim=self.dim)

        return scaled_data, torch.zeros_like(scale), scale
# 从transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesNOPScaler复制而来，用于Informer，TimeSeries->Informer
class InformerNOPScaler(nn.Module):
    """
    为第一维度分配一个等于1的缩放因子，因此对输入数据不进行缩放。
    """

    def __init__(self, config: InformerConfig):
        super().__init__()
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        参数:
            data (`torch.Tensor`，形状为`(batch_size, sequence_length, num_input_channels)`):
                用于批量归一化计算的输入
        返回:
            形状为(`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
            `(batch_size, 1, num_input_channels)`)的`torch.Tensor`元组
        """
        scale = torch.ones_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        loc = torch.zeros_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        return data, loc, scale


# 从transformers.models.time_series_transformer.modeling_time_series_transformer.weighted_average复制而来
def weighted_average(input_tensor: torch.Tensor, weights: Optional[torch.Tensor] = None, dim=None) -> torch.Tensor:
    """
    计算给定张量在给定`dim`上的加权平均值，屏蔽与权重为零相关的值，意味着不是`nan * 0 = nan`，而是`0 * 0 = 0`。

    参数:
        input_tensor (`torch.FloatTensor`):
            输入张量，需要计算平均值。
        weights (`torch.FloatTensor`，*可选*):
            权重张量，与`input_tensor`形状相同。
        dim (`int`，*可选*):
            沿着哪个维度对`input_tensor`进行平均。

    返回:
        `torch.FloatTensor`: 沿着指定`dim`平均值的张量。
    """
    if weights is not None:
        weighted_tensor = torch.where(weights != 0, input_tensor * weights, torch.zeros_like(input_tensor))
        sum_weights = torch.clamp(weights.sum(dim=dim) if dim else weights.sum(), min=1.0)
        return (weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()) / sum_weights
    else:
        return input_tensor.mean(dim=dim)


# 从transformers.models.time_series_transformer.modeling_time_series_transformer.nll复制而来
def nll(input: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor:
    """
    根据目标计算输入分布的负对数似然损失。
    """
    return -input.log_prob(target)
# 从transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding复制并修改为InformerSinusoidalPositionalEmbedding
class InformerSinusoidalPositionalEmbedding(nn.Embedding):
    """该模块生成任意长度的正弦位置嵌入。"""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None) -> None:
        # 调用父类的初始化方法
        super().__init__(num_positions, embedding_dim)
        # 初始化权重
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """
        与XLM create_sinusoidal_embeddings相同，除了特征不是交错的。cos特征在向量的第二半部分。[dim // 2:]
        """
        n_pos, dim = out.shape
        # 创建正弦位置编码
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False  # 提前设置以避免在pytorch-1.8+中出错
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        # 将正弦和余弦值赋给权重
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2])
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0) -> torch.Tensor:
        """`input_ids_shape`预期为[bsz x seqlen]。"""
        bsz, seq_len = input_ids_shape[:2]
        # 生成位置编码
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions)


# 从transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesValueEmbedding复制并修改为InformerValueEmbedding
class InformerValueEmbedding(nn.Module):
    def __init__(self, feature_size, d_model):
        # 调用父类的初始化方法
        super().__init__()
        # 值投影层
        self.value_projection = nn.Linear(in_features=feature_size, out_features=d_model, bias=False)

    def forward(self, x):
        return self.value_projection(x)


# 从transformers.models.bart.modeling_bart.BartAttention复制并修改为InformerAttention
class InformerAttention(nn.Module):
    """来自'Attention Is All You Need'论文的多头注意力机制"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[InformerConfig] = None,
    # 初始化 Transformer 的 MultiheadAttention 层
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        is_decoder: bool = False,
        is_causal: bool = False,
        config: Optional[Config] = None
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 设置嵌入维度、头数、dropout率、头维度、配置信息等属性
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        # 检查头维度乘以头数是否等于嵌入维度，如果不等则抛出异常
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        # 计算缩放因子
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        # 初始化线性变换层，用于计算 Q、K、V 和输出
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # 重塑张量形状的辅助方法
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传播方法，接收隐藏状态、键值状态、过去的键值、注意力掩码、层头掩码等参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
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
        # 初始化函数，设置模型参数
        super().__init__()
        self.factor = sampling_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            # 检查 embed_dim 是否可以被 num_heads 整除
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        # 初始化线性变换层
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 重塑张量形状
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        # 模型前向传播函数



class InformerConvLayer(nn.Module):
    def __init__(self, c_in):
        # 初始化函数，设置卷积层参数
        super().__init__()
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # 前向传播函数
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class InformerEncoderLayer(nn.Module):
    # InformerEncoderLayer 类定义
    # 初始化函数，接受一个 InformedConfig 类型的参数
    def __init__(self, config: InformerConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 设置嵌入维度为配置中的模型维度
        self.embed_dim = config.d_model
        # 根据配置中的注意力类型选择不同的注意力机制
        if config.attention_type == "prob":
            # 如果是概率注意力类型，则使用 InformedProbSparseAttention 类
            self.self_attn = InformerProbSparseAttention(
                embed_dim=self.embed_dim,
                num_heads=config.encoder_attention_heads,
                dropout=config.attention_dropout,
                sampling_factor=config.sampling_factor,
            )
        else:
            # 否则使用 InformedAttention 类
            self.self_attn = InformerAttention(
                embed_dim=self.embed_dim,
                num_heads=config.encoder_attention_heads,
                dropout=config.attention_dropout,
            )
        # 初始化自注意力层归一化
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 设置丢弃率为配置中的丢弃率
        self.dropout = config.dropout
        # 设置激活函数为配置中指定的激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 设置激活丢弃率为配置中的激活丢弃率
        self.activation_dropout = config.activation_dropout
        # 第一个全连接层，输入维度为嵌入维度，输出维度为配置中的编码器前馈网络维度
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        # 第二个全连接层，输入维度为配置中的编码器前馈网络维度，输出维度为嵌入维度
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        # 最终层归一化
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    # 前向传播函数，接受隐藏状态、注意力掩码、层头掩码和是否输出注意力矩阵等参数
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
        # 保存输入 hidden_states 作为残差连接的基准
        residual = hidden_states
        # 使用 self_attn 层处理 hidden_states，得到输出 hidden_states 和注意力权重 attn_weights
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 对 hidden_states 进行 dropout 操作
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 将残差连接的结果与当前 hidden_states 相加
        hidden_states = residual + hidden_states
        # 对相加后的 hidden_states 进行 LayerNorm 处理
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 保存当前 hidden_states 作为残差连接的基准
        residual = hidden_states
        # 使用激活函数 activation_fn 处理 hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对处理后的 hidden_states 进行 dropout 操作
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # 使用第二个全连接层 fc2 处理 hidden_states
        hidden_states = self.fc2(hidden_states)
        # 对处理后的 hidden_states 进行 dropout 操作
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 将残差连接的结果与当前 hidden_states 相加
        hidden_states = residual + hidden_states
        # 对相加后的 hidden_states 进行 LayerNorm 处理
        hidden_states = self.final_layer_norm(hidden_states)

        # 如果 hidden_states 的数据类型为 torch.float16，并且包含无穷大或 NaN 值，则进行截断处理
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # 构建输出元组，包含处理后的 hidden_states
        outputs = (hidden_states,)

        # 如果需要返回 attentions，则将 attn_weights 加入输出元组
        if output_attentions:
            outputs += (attn_weights,)

        return outputs
class InformerDecoderLayer(nn.Module):
    # 定义InformerDecoderLayer类，继承自nn.Module
    def __init__(self, config: InformerConfig):
        # 初始化函数，接受一个InformerConfig类型的参数config
        super().__init__()
        # 调用父类的初始化函数
        self.embed_dim = config.d_model
        # 设置embed_dim为config中的d_model值

        if config.attention_type == "prob":
            # 如果config中的attention_type为"prob"
            self.self_attn = InformerProbSparseAttention(
                embed_dim=self.embed_dim,
                num_heads=config.decoder_attention_heads,
                dropout=config.attention_dropout,
                sampling_factor=config.sampling_factor,
                is_decoder=True,
            )
            # 创建InformerProbSparseAttention对象并赋值给self.self_attn
        else:
            self.self_attn = InformerAttention(
                embed_dim=self.embed_dim,
                num_heads=config.decoder_attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
            )
            # 创建InformerAttention对象并赋值给self.self_attn
        self.dropout = config.dropout
        # 设置dropout为config中的dropout值
        self.activation_fn = ACT2FN[config.activation_function]
        # 设置activation_fn为ACT2FN字典中对应config.activation_function的值
        self.activation_dropout = config.activation_dropout
        # 设置activation_dropout为config中的activation_dropout值

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 创建LayerNorm对象并赋值给self.self_attn_layer_norm
        self.encoder_attn = InformerAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        # 创建InformerAttention对象并赋值给self.encoder_attn
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 创建LayerNorm对象并赋值给self.encoder_attn_layer_norm
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        # 创建Linear对象并赋值给self.fc1
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        # 创建Linear对象并赋值给self.fc2
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        # 创建LayerNorm对象并赋值给self.final_layer_norm

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
        # 定义前向传播函数，接受多个参数
class InformerPreTrainedModel(PreTrainedModel):
    # 定义InformerPreTrainedModel类，继承自PreTrainedModel
    config_class = InformerConfig
    # 设置config_class为InformerConfig
    base_model_prefix = "model"
    # 设置base_model_prefix为"model"
    main_input_name = "past_values"
    # 设置main_input_name为"past_values"
    supports_gradient_checkpointing = True
    # 设置supports_gradient_checkpointing为True

    def _init_weights(self, module):
        # 定义_init_weights函数，接受一个module参数
        std = self.config.init_std
        # 设置std为self.config中的init_std值
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # 如果module是nn.Linear或nn.Conv1d类型
            module.weight.data.normal_(mean=0.0, std=std)
            # 将module的权重数据初始化为正态分布
            if module.bias is not None:
                module.bias.data.zero_()
                # 如果module有偏置项，将其初始化为0
        elif isinstance(module, nn.Embedding):
            # 如果module是nn.Embedding类型
            module.weight.data.normal_(mean=0.0, std=std)
            # 将module的权重数据初始化为正态分布
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
                # 如果module有padding_idx，将其对应的权重初始化为0

INFORMER_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    # 设置INFORMER_START_DOCSTRING为一段文档字符串
    # 这个模型也是 PyTorch 的 torch.nn.Module 子类。
    # 可以像普通的 PyTorch 模块一样使用，并参考 PyTorch 文档了解与一般用法和行为相关的所有事项。

    # 参数:
    # config ([`TimeSeriesTransformerConfig`]):
    #     包含模型所有参数的模型配置类。使用配置文件初始化不会加载与模型相关的权重，只会加载配置。查看
    #     `~PreTrainedModel.from_pretrained` 方法以加载模型权重。
"""

# 定义 INFORMER_INPUTS_DOCSTRING 为空字符串
INFORMER_INPUTS_DOCSTRING = r"""
"""

# 定义 InformerEncoder 类，继承自 InformerPreTrainedModel
class InformerEncoder(InformerPreTrainedModel):
    """
    Informer encoder consisting of *config.encoder_layers* self attention layers with distillation layers. Each
    attention layer is an [`InformerEncoderLayer`].

    Args:
        config: InformerConfig
    """

    def __init__(self, config: InformerConfig):
        super().__init__(config)

        # 初始化 dropout、layerdrop 和 gradient_checkpointing
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        self.gradient_checkpointing = False
        # 如果 prediction_length 为 None，则抛出 ValueError
        if config.prediction_length is None:
            raise ValueError("The `prediction_length` config needs to be specified.")

        # 初始化 value_embedding、embed_positions、layers 和 layernorm_embedding
        self.value_embedding = InformerValueEmbedding(feature_size=config.feature_size, d_model=config.d_model)
        self.embed_positions = InformerSinusoidalPositionalEmbedding(
            config.context_length + config.prediction_length, config.d_model
        )
        self.layers = nn.ModuleList([InformerEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        # 如果 distil 为 True，则初始化 conv_layers 为 InformerConvLayer 的 ModuleList，否则初始化为 None 的列表
        if config.distil:
            self.conv_layers = nn.ModuleList(
                [InformerConvLayer(config.d_model) for _ in range(config.encoder_layers - 1)]
            )
            self.conv_layers.append(None)
        else:
            self.conv_layers = [None] * config.encoder_layers

        # 初始化权重并应用最终处理
        self.post_init()

    # 定义 forward 方法
    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesTransformerDecoder with TimeSeriesTransformer->Informer,TimeSeriesTransformerConfig->InformerConfig,time-series-transformer->informer,Transformer->Informer,TimeSeries->Informer
class InformerDecoder(InformerPreTrainedModel):
    """
    Informer decoder consisting of *config.decoder_layers* layers. Each layer is a
    [`InformerDecoderLayer`]

    Args:
        config: InformerConfig
    """
    # 初始化函数，接受一个InformerConfig对象作为参数
    def __init__(self, config: InformerConfig):
        # 调用父类的初始化函数
        super().__init__(config)
        # 从config中获取dropout参数并赋值给self.dropout
        self.dropout = config.dropout
        # 从config中获取decoder_layerdrop参数并赋值给self.layerdrop
        self.layerdrop = config.decoder_layerdrop
        # 如果config中的prediction_length为None，则抛出数值错误异常
        if config.prediction_length is None:
            raise ValueError("The `prediction_length` config needs to be specified.")

        # 创建InformerValueEmbedding对象并赋值给self.value_embedding
        self.value_embedding = InformerValueEmbedding(feature_size=config.feature_size, d_model=config.d_model)
        # 创建InformerSinusoidalPositionalEmbedding对象并赋值给self.embed_positions
        self.embed_positions = InformerSinusoidalPositionalEmbedding(
            config.context_length + config.prediction_length, config.d_model
        )
        # 创建包含多个InformerDecoderLayer对象的ModuleList并赋值给self.layers
        self.layers = nn.ModuleList([InformerDecoderLayer(config) for _ in range(config.decoder_layers)])
        # 创建LayerNorm对象并赋值给self.layernorm_embedding
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        # 初始化梯度检查点为False
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数，接受多个可选参数
    def forward(
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
# 添加模型文档字符串，描述该模型输出原始隐藏状态而没有特定的头部
# 从transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesTransformerModel复制而来，将TimeSeriesTransformer->Informer,TIME_SERIES_TRANSFORMER->INFORMER,time-series-transformer->informer,TimeSeries->Informer
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

        # 初始化transformer编码器和解码器以及掩码初始化器
        self.encoder = InformerEncoder(config)
        self.decoder = InformerDecoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    @property
    def _past_length(self) -> int:
        return self.config.context_length + max(self.config.lags_sequence)

    def get_lagged_subsequences(
        self, sequence: torch.Tensor, subsequences_length: int, shift: int = 0
    ) -> torch.Tensor:
        """
        返回给定序列的滞后子序列。返回形状为(N, S, C, I)的张量，其中S = subsequences_length，I = len(indices)，包含滞后子序列。
        具体来说，lagged[i, j, :, k] = sequence[i, -indices[k]-S+j, :]。

        Args:
            sequence: Tensor
                应从中提取滞后子序列的序列。形状：(N, T, C)。
            subsequences_length : int
                要提取的子序列的长度。
            shift: int
                将滞后值向后移动这个数量。
        """
        sequence_length = sequence.shape[1]
        indices = [lag - shift for lag in self.config.lags_sequence]

        if max(indices) + subsequences_length > sequence_length:
            raise ValueError(
                f"lags不能超过历史长度，发现滞后值 {max(indices)}，而历史长度仅为 {sequence_length}"
            )

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...])
        return torch.stack(lagged_values, dim=-1)
    # 创建神经网络的输入数据
    # self参数表示这是一个类方法
    # past_values: 过去的数值数据，torch.Tensor类型
    # past_time_features: 过去的时间特征数据，torch.Tensor类型
    # static_categorical_features: 静态的分类特征数据，torch.Tensor类型，可选
    # static_real_features: 静态的实数特征数据，torch.Tensor类型，可选
    # past_observed_mask: 过去的观测掩码，torch.Tensor类型，可选
    # future_values: 未来的数值数据，torch.Tensor类型，可选
    # future_time_features: 未来的时间特征数据，torch.Tensor类型，可选
        # time feature
        # 时间特征拼接，将过去时间特征和未来时间特征拼接成一个整体
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
        # 目标数值的处理
        if past_observed_mask is None:
            past_observed_mask = torch.ones_like(past_values)

        context = past_values[:, -self.config.context_length :]
        observed_context = past_observed_mask[:, -self.config.context_length :]
        _, loc, scale = self.scaler(context, observed_context)

        inputs = (
            (torch.cat((past_values, future_values), dim=1) - loc) / scale
            if future_values is not None
            else (past_values - loc) / scale
        )

        # static features
        # 静态特征处理
        log_abs_loc = loc.abs().log1p() if self.config.input_size == 1 else loc.squeeze(1).abs().log1p()
        log_scale = scale.log() if self.config.input_size == 1 else scale.squeeze(1).log()
        static_feat = torch.cat((log_abs_loc, log_scale), dim=1)

        if static_real_features is not None:
            static_feat = torch.cat((static_real_features, static_feat), dim=1)
        if static_categorical_features is not None:
            embedded_cat = self.embedder(static_categorical_features)
            static_feat = torch.cat((embedded_cat, static_feat), dim=1)
        expanded_static_feat = static_feat.unsqueeze(1).expand(-1, time_feat.shape[1], -1)

        # all features
        # 所有特征组合
        features = torch.cat((expanded_static_feat, time_feat), dim=-1)

        # lagged features
        # 滞后特征处理
        subsequences_length = (
            self.config.context_length + self.config.prediction_length
            if future_values is not None
            else self.config.context_length
        )
        lagged_sequence = self.get_lagged_subsequences(sequence=inputs, subsequences_length=subsequences_length)
        lags_shape = lagged_sequence.shape
        reshaped_lagged_sequence = lagged_sequence.reshape(lags_shape[0], lags_shape[1], -1)

        # 若滞后特征长度不等于时间特征长度，则抛出数值错误
        if reshaped_lagged_sequence.shape[1] != time_feat.shape[1]:
            raise ValueError(
                f"input length {reshaped_lagged_sequence.shape[1]} and time feature lengths {time_feat.shape[1]} does not match"
            )

        # transformer inputs
        # 计算 transformer 的输入
        transformer_inputs = torch.cat((reshaped_lagged_sequence, features), dim=-1)

        return transformer_inputs, loc, scale, static_feat

    # 获取编码器
    def get_encoder(self):
        return self.encoder

    # 获取解码器
    def get_decoder(self):
        return self.decoder

    # 将模型输入/输出类型和配置添加到模型前向方法的注释中
    @add_start_docstrings_to_model_forward(INFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqTSModelOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个方法，用于进行前向传播
    def forward(
        # 过去的数值，torch.Tensor 类型
        past_values: torch.Tensor,
        # 过去的时间特征，torch.Tensor 类型
        past_time_features: torch.Tensor,
        # 过去的观察掩码，torch.Tensor 类型
        past_observed_mask: torch.Tensor,
        # 静态类别特征，可选 torch.Tensor 类型
        static_categorical_features: Optional[torch.Tensor] = None,
        # 静态实数特征，可选 torch.Tensor 类型
        static_real_features: Optional[torch.Tensor] = None,
        # 未来的数值，可选 torch.Tensor 类型
        future_values: Optional[torch.Tensor] = None,
        # 未来的时间特征，可选 torch.Tensor 类型
        future_time_features: Optional[torch.Tensor] = None,
        # 解码器注意力掩码，可选 torch.LongTensor 类型
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        # 头部掩码，可选 torch.Tensor 类型
        head_mask: Optional[torch.Tensor] = None,
        # 解码器头部掩码，可选 torch.Tensor 类型
        decoder_head_mask: Optional[torch.Tensor] = None,
        # 交叉注意力头部掩码，可选 torch.Tensor 类型
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        # 编码器输出列表，可选包含 torch.FloatTensor 类型
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        # 过去键值列表，可选包含 torch.FloatTensor 类型
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # 是否输出隐藏状态，可选布尔值
        output_hidden_states: Optional[bool] = None,
        # 是否输出注意力权重，可选布尔值
        output_attentions: Optional[bool] = None,
        # 是否使用缓存，可选布尔值
        use_cache: Optional[bool] = None,
        # 返回字典，可选布尔值
        return_dict: Optional[bool] = None,
# 添加模型描述和文档字符串开始部分到模型类InformerForPrediction
# 模型类继承自InformerPreTrainedModel，并在原有TimeSeriesTransformerForPrediction基础上调整为Informer
class InformerForPrediction(InformerPreTrainedModel):
    # 初始化模型类
    def __init__(self, config: InformerConfig):
        # 调用父类初始化函数
        super().__init__(config)
        # 创建InformerModel对象
        self.model = InformerModel(config)
        
        # 根据配置选择输出分布
        if config.distribution_output == "student_t":
            self.distribution_output = StudentTOutput(dim=config.input_size)
        elif config.distribution_output == "normal":
            self.distribution_output = NormalOutput(dim=config.input_size)
        elif config.distribution_output == "negative_binomial":
            self.distribution_output = NegativeBinomialOutput(dim=config.input_size)
        else:
            raise ValueError(f"Unknown distribution output {config.distribution_output}")
        
        # 根据模型配置和选择的输出分布初始化参数投影和目标形状
        self.parameter_projection = self.distribution_output.get_parameter_projection(self.model.config.d_model)
        self.target_shape = self.distribution_output.event_shape

        # 根据配置选择损失函数
        if config.loss == "nll":
            self.loss = nll
        else:
            raise ValueError(f"Unknown loss function {config.loss}")

        # 初始化输出参数
        self.post_init()
    
    # 获取输出参数
    def output_params(self, dec_output):
        return self.parameter_projection(dec_output)

    # 获取编码器
    def get_encoder(self):
        return self.model.get_encoder()

    # 获取解码器
    def get_decoder(self):
        return self.model.get_decoder()

    # 忽略torch jit注解
    @torch.jit.ignore
    def output_distribution(self, params, loc=None, scale=None, trailing_n=None) -> torch.distributions.Distribution:
        sliced_params = params
        if trailing_n is not None:
            sliced_params = [p[:, -trailing_n:] for p in params]
        return self.distribution_output.distribution(sliced_params, loc=loc, scale=scale)

    # 添加模型输入文档字符串到模型前向方法
    @add_start_docstrings_to_model_forward(INFORMER_INPUTS_DOCSTRING)
    # 替换模型前向方法返回文档字符串
    @replace_return_docstrings(output_type=Seq2SeqTSModelOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个类方法用于前向传播，接收多个输入参数
    def forward(
        self,
        past_values: torch.Tensor,  # 过去值的张量
        past_time_features: torch.Tensor,  # 过去时间特征的张量
        past_observed_mask: torch.Tensor,  # 过去观察掩码的张量
        static_categorical_features: Optional[torch.Tensor] = None,  # 静态分类特征的张量（可选）
        static_real_features: Optional[torch.Tensor] = None,  # 静态实数特征的张量（可选）
        future_values: Optional[torch.Tensor] = None,  # 未来值的张量（可选）
        future_time_features: Optional[torch.Tensor] = None,  # 未来时间特征的张量（可选）
        future_observed_mask: Optional[torch.Tensor] = None,  # 未来观察掩码的张量（可选）
        decoder_attention_mask: Optional[torch.LongTensor] = None,  # 解码器注意力掩码的长整型张量（可选）
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码的张量（可选）
        decoder_head_mask: Optional[torch.Tensor] = None,  # 解码器头部掩码的张量（可选）
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 交叉注意力头部掩码的张量（可选）
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,  # 编码器输出的张量列表（可选）
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 过去键值的张量列表（可选）
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态（可选）
        output_attentions: Optional[bool] = None,  # 是否输出注意力（可选）
        use_cache: Optional[bool] = None,  # 是否使用缓存（可选）
        return_dict: Optional[bool] = None,  # 是否返回字典（可选）
    
    # 定义一个装饰器，表示下面的方法是不需要梯度计算的
    @torch.no_grad()
    # 定义一个类方法用于生成，接收多个输入参数
    def generate(
        self,
        past_values: torch.Tensor,  # 过去值的张量
        past_time_features: torch.Tensor,  # 过去时间特征的张量
        future_time_features: torch.Tensor,  # 未来时间特征的张量
        past_observed_mask: Optional[torch.Tensor] = None,  # 过去观察掩码的张量（可选）
        static_categorical_features: Optional[torch.Tensor] = None,  # 静态分类特征的张量（可选）
        static_real_features: Optional[torch.Tensor] = None,  # 静态实数特征的张量（可选）
        output_attentions: Optional[bool] = None,  # 是否输出注意力（可选）
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态（可选）
```