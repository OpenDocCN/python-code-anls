# `.\transformers\models\autoformer\modeling_autoformer.py`

```
# 导入必要的库和模块
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn

# 导入相关的函数和类
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import (
    BaseModelOutput,
    ModelOutput,
    SampleTSPredictionOutput,
    Seq2SeqTSPredictionOutput,
)
from ...modeling_utils import PreTrainedModel
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_autoformer import AutoformerConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置名称
_CONFIG_FOR_DOC = "AutoformerConfig"

# 定义 AutoFormerDecoderOutput 类，用于模型输出
@dataclass
class AutoFormerDecoderOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        trend (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Trend tensor for each time series.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """

    # Define input tensors with their respective types and shapes
    last_hidden_state: torch.FloatTensor = None
    trend: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义了两个可选类型的变量 attentions 和 cross_attentions，并初始化为 None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
import torch.nn as nn

# 使用dataclass装饰器定义AutoformerModelOutput类，继承自ModelOutput类
@dataclass
class AutoformerModelOutput(ModelOutput):
    """
    Autoformer model output that contains the additional trend output.
    自动形态模型输出，包含额外的趋势输出。
    """

    last_hidden_state: torch.FloatTensor = None
    trend: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    loc: Optional[torch.FloatTensor] = None
    scale: Optional[torch.FloatTensor] = None
    static_features: Optional[torch.FloatTensor] = None


# 预训练模型列表
AUTOFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "huggingface/autoformer-tourism-monthly",
    # See all Autoformer models at https://huggingface.co/models?filter=autoformer
]

# 从TimeSeriesFeatureEmbedder复制而来，修改为AutoformerFeatureEmbedder
# 将一系列的分类特征嵌入到连续的向量空间中
class AutoformerFeatureEmbedder(nn.Module):
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

        # 记录特征的数量
        self.num_features = len(cardinalities)
        # 创建嵌入层列表，每个分类特征对应一个嵌入层
        self.embedders = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(cardinalities, embedding_dims)])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.num_features > 1:
            # 将最后一个维度切片，得到一个长度为self.num_features的数组，形状为(N,T)或(N)
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


# 从TimeSeriesStdScaler复制而来，修改为AutoformerStdScaler
# 标准化特征，计算均值并沿第一个维度进行缩放，然后通过减去均值并除以标准差进行归一化。
class AutoformerStdScaler(nn.Module):
    """
    Standardize features by calculating the mean and scaling along the first dimension, and then normalizes it by
    subtracting from the mean and dividing by the standard deviation.
    """
    def __init__(self, config: AutoformerConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 如果配置对象有scaling_dim属性，则将其值赋给self.dim，否则默认为1
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        # 如果配置对象有keepdim属性，则将其值赋给self.keepdim，否则默认为True
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True
        # 如果配置对象有minimum_scale属性，则将其值赋给self.minimum_scale，否则默认为1e-5
        self.minimum_scale = config.minimum_scale if hasattr(config, "minimum_scale") else 1e-5

    def forward(
        self,  torch.Tensor, observed_indicator: torch.Tensor
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
        # 计算观测指示器在指定维度上的和，如果keepdim为True，则保持维度不变
        denominator = observed_indicator.sum(self.dim, keepdim=self.keepdim)
        # 将分母限制在最小值为1.0，避免除零错误
        denominator = denominator.clamp_min(1.0)
        # 计算均值，先对数据与观测指示器进行元素级乘法，然后在指定维度上求和，保持维度不变
        loc = (data * observed_indicator).sum(self.dim, keepdim=self.keepdim) / denominator

        # 计算方差，先对数据减去均值，再乘以观测指示器，再平方，然后在指定维度上求和，保持维度不变
        variance = (((data - loc) * observed_indicator) ** 2).sum(self.dim, keepdim=self.keepdim) / denominator
        # 计算标准差，对方差加上最小标量然后取平方根
        scale = torch.sqrt(variance + self.minimum_scale)
        # 返回标准化后的数据，均值和标准差
        return (data - loc) / scale, loc, scale
# 从transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesMeanScaler复制而来，将TimeSeriesTransformer->Autoformer，TimeSeries->Autoformer
# 定义AutoformerMeanScaler类，用于计算加权平均绝对值作为缩放因子，并相应地缩放数据
class AutoformerMeanScaler(nn.Module):
    """
    Computes a scaling factor as the weighted average absolute value along the first dimension, and scales the data
    accordingly.
    """

    def __init__(self, config: AutoformerConfig):
        super().__init__()
        # 如果config中有scaling_dim属性，则使用该值，否则默认为1
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        # 如果config中有keepdim属性，则使用该值，否则默认为True
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True
        # 如果config中有minimum_scale属性，则使用该值，否则默认为1e-10
        self.minimum_scale = config.minimum_scale if hasattr(config, "minimum_scale") else 1e-10
        # 如果config中有default_scale属性，则使用该值，否则默认为None
        self.default_scale = config.default_scale if hasattr(config, "default_scale") else None

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
        # 计算加权平均绝对值作为缩放因子
        ts_sum = (data * observed_indicator).abs().sum(self.dim, keepdim=True)
        num_observed = observed_indicator.sum(self.dim, keepdim=True)

        scale = ts_sum / torch.clamp(num_observed, min=1)

        # 如果提供了`default_scale`，则使用它，否则使用批次的规模
        if self.default_scale is None:
            batch_sum = ts_sum.sum(dim=0)
            batch_observations = torch.clamp(num_observed.sum(0), min=1)
            default_scale = torch.squeeze(batch_sum / batch_observations)
        else:
            default_scale = self.default_scale * torch.ones_like(scale)

        # 在没有观测到的地方应用默认规模
        scale = torch.where(num_observed > 0, scale, default_scale)

        # 确保规模至少为`self.minimum_scale`
        scale = torch.clamp(scale, min=self.minimum_scale)
        scaled_data = data / scale

        if not self.keepdim:
            scale = scale.squeeze(dim=self.dim)

        return scaled_data, torch.zeros_like(scale), scale


# 从transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesNOPScaler复制而来，将TimeSeriesTransformer->Autoformer，TimeSeries->Autoformer
# 定义AutoformerNOPScaler类，将第一维度的缩放因子设置为1，因此对输入数据不进行缩放
class AutoformerNOPScaler(nn.Module):
    """
    Assigns a scaling factor equal to 1 along the first dimension, and therefore applies no scaling to the input data.
    """
    # 初始化方法，接受一个 AutoformerConfig 对象作为参数
    def __init__(self, config: AutoformerConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 如果 config 对象有 scaling_dim 属性，则将其赋值给 self.dim，否则赋值为 1
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        # 如果 config 对象有 keepdim 属性，则将其赋值给 self.keepdim，否则赋值为 True
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True

    # 前向传播方法
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
        # 计算 data 在指定维度上的均值，并创建一个与 data 相同形状的张量，值为 1
        scale = torch.ones_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        # 创建一个与 data 相同形状的张量，值为 0
        loc = torch.zeros_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        # 返回 data、loc、scale 三个张量组成的元组
        return data, loc, scale
# 从 transformers.models.time_series_transformer.modeling_time_series_transformer.weighted_average 复制而来
def weighted_average(input_tensor: torch.Tensor, weights: Optional[torch.Tensor] = None, dim=None) -> torch.Tensor:
    """
    计算给定张量在给定 `dim` 上的加权平均值，掩盖与权重零相关联的值，
    这意味着你会得到 `nan * 0 = nan` 而不是 `0 * 0 = 0`。

    Args:
        input_tensor (`torch.FloatTensor`):
            输入张量，必须计算其平均值。
        weights (`torch.FloatTensor`, *可选*):
            权重张量，与 `input_tensor` 形状相同。
        dim (`int`, *可选*):
            要沿其对 `input_tensor` 进行平均的维度。

    Returns:
        `torch.FloatTensor`: 沿指定 `dim` 平均值的张量。
    """
    if weights is not None:
        weighted_tensor = torch.where(weights != 0, input_tensor * weights, torch.zeros_like(input_tensor))
        sum_weights = torch.clamp(weights.sum(dim=dim) if dim else weights.sum(), min=1.0)
        return (weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()) / sum_weights
    else:
        return input_tensor.mean(dim=dim)


# 从 transformers.models.time_series_transformer.modeling_time_series_transformer.nll 复制而来
def nll(input: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor:
    """
    根据输入分布计算与目标相关的负对数似然损失。
    """
    return -input.log_prob(target)


# 从 transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding 复制而来，将 Marian->Autoformer
class AutoformerSinusoidalPositionalEmbedding(nn.Embedding):
    """此模块生成任意长度的正弦位置嵌入。"""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None) -> None:
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """
        与 XLM create_sinusoidal_embeddings 相同，除了特征不是交错的。
        cos 特征位于向量的第二半部分。[dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False  # 早期设置以避免在 pytorch-1.8+ 中出错
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
```  
    # 定义了一个函数 forward，用于生成位置编码
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        # 从 input_ids_shape 中获取 batch size 和 sequence length
        bsz, seq_len = input_ids_shape[:2]
        # 生成位置编码的位置信息，从 past_key_values_length 到 past_key_values_length + seq_len
        # 使用 torch.arange 函数生成一个从 past_key_values_length 到 past_key_values_length + seq_len 的序列
        # dtype 设置为 torch.long，表示数据类型为长整型
        # device 设置为 self.weight.device，即与 self.weight 张量相同的设备
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        # 调用父类的 forward 方法，传入位置信息张量 positions，生成位置编码张量
        return super().forward(positions)
# 从transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesValueEmbedding复制代码，用Autoformer替代TimeSeries
class AutoformerValueEmbedding(nn.Module):
    def __init__(self, feature_size, d_model):
        super().__init__()
        # 创建一个线性层，用于将特征向量投影到模型维度
        self.value_projection = nn.Linear(in_features=feature_size, out_features=d_model, bias=False)

    def forward(self, x):
        # 将输入数据投影到模型维度并返回
        return self.value_projection(x)


# 基于https://github.com/thuml/Autoformer/blob/c6a0694ff484753f2d986cc0bb1f99ee850fc1a8/layers/Autoformer_EncDec.py#L39的类，其中AutoformerSeriesDecompositionLayer是series_decomp + moving_average
class AutoformerSeriesDecompositionLayer(nn.Module):
    """
    返回时间序列的趋势和季节性部分。计算方法为：

        x_trend = AvgPool(Padding(X)) and x_seasonal = X - x_trend
    """

    def __init__(self, config: AutoformerConfig):
        super().__init__()
        # 设置移动平均滑动窗口大小
        self.kernel_size = config.moving_average
        # 创建一个平均池化层，用于计算移动平均
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)

    def forward(self, x):
        """输入形状: Batch x Time x EMBED_DIM"""
        # 在时间序列两端进行填充
        num_of_pads = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, num_of_pads, 1)
        end = x[:, -1:, :].repeat(1, num_of_pads, 1)
        x_padded = torch.cat([front, x, end], dim=1)

        # 计算时间序列的趋势和季节性部分
        x_trend = self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
        x_seasonal = x - x_trend
        return x_seasonal, x_trend


# 基于https://github.com/thuml/Autoformer/blob/c6a0694ff484753f2d986cc0bb1f99ee850fc1a8/layers/Autoformer_EncDec.py#L6的类，其中AutoformerLayernorm是my_Layernorm
class AutoformerLayernorm(nn.Module):
    """
    专门设计的层归一化用于季节性部分，计算方法为: AutoformerLayernorm(x) = nn.LayerNorm(x)
    - torch.mean(nn.LayerNorm(x))
    """

    def __init__(self, config: AutoformerConfig):
        super().__init__()
        # 创建一个层归一化层，用于对季节性部分进行归一化
        self.layernorm = nn.LayerNorm(config.d_model)

    def forward(self, x):
        # 对输入进行层归一化
        x_hat = self.layernorm(x)
        # 计算归一化后的均值，并在第二个维度上进行扩展
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class AutoformerAttention(nn.Module):
    """
    具有以下两个阶段的自相关机制：
        (1) 基于周期的依赖性发现 (2) 时间延迟聚合
    该块替换了规范的自注意力机制。
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        autocorrelation_factor: int = 3,
    # 初始化函数，设置注意力机制的参数
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 设置嵌入维度
        self.embed_dim = embed_dim
        # 设置注意力头的数量
        self.num_heads = num_heads
        # 设置dropout概率
        self.dropout = dropout
        # 计算每个注意力头的维度
        self.head_dim = embed_dim // num_heads

        # 检查embed_dim是否可以被num_heads整除
        if (self.head_dim * num_heads) != self.embed_dim:
            # 如果不能整除，抛出数值错误
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        # 缩放因子，用于缩放注意力分数
        self.scaling = self.head_dim**-0.5
        # 是否是解码器的标志
        self.is_decoder = is_decoder

        # 将输入进行线性变换以获取查询、键和值
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # 将注意力头的输出进行线性变换以恢复原始维度
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # 自相关因子，用于自相关注意力
        self.autocorrelation_factor = autocorrelation_factor

    # 将张量重塑成适当的形状
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
# 定义 AutoformerEncoderLayer 类，继承自 nn.Module
class AutoformerEncoderLayer(nn.Module):
    # 初始化函数，接受一个 AutoformerConfig 类型的参数 config
    def __init__(self, config: AutoformerConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 设置 embed_dim 为 config 中的 d_model
        self.embed_dim = config.d_model
        # 创建自注意力层 AutoformerAttention 对象
        self.self_attn = AutoformerAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            autocorrelation_factor=config.autocorrelation_factor,
        )
        # 创建自注意力层的 LayerNorm 层
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 设置 dropout 为 config 中的 dropout
        self.dropout = config.dropout
        # 设置激活函数为 config 中指定的激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 设置激活函数的 dropout 为 config 中的 activation_dropout
        self.activation_dropout = config.activation_dropout
        # 创建全连接层 fc1，输入维度为 embed_dim，输出维度为 config 中的 encoder_ffn_dim
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        # 创建全连接层 fc2，输入维度为 config 中的 encoder_ffn_dim，输出维度为 embed_dim
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        # 创建最终的 LayerNorm 层 AutoformerLayernorm
        self.final_layer_norm = AutoformerLayernorm(config)
        # 创建 AutoformerSeriesDecompositionLayer 对象 decomp1
        self.decomp1 = AutoformerSeriesDecompositionLayer(config)
        # 创建 AutoformerSeriesDecompositionLayer 对象 decomp2
        self.decomp2 = AutoformerSeriesDecompositionLayer(config)

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
        # 添加层归一化操作
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 使用 decomp1 层处理 hidden_states
        hidden_states, _ = self.decomp1(hidden_states)

        # 保存输入 hidden_states 作为残差连接的基准
        residual = hidden_states
        # 使用激活函数 activation_fn 处理 hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对 hidden_states 进行 dropout 操作
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # 使用 fc2 层处理 hidden_states
        hidden_states = self.fc2(hidden_states)
        # 对 hidden_states 进行 dropout 操作
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 将残差连接的结果与当前 hidden_states 相加
        hidden_states = residual + hidden_states
        # 使用 decomp2 层处理 hidden_states
        hidden_states, _ = self.decomp2(hidden_states)
        # 使用 final_layer_norm 层进行最终的归一化操作
        hidden_states = self.final_layer_norm(hidden_states)

        # 如果 hidden_states 的数据类型为 torch.float16，并且包含无穷大或 NaN 值，则进行截断处理
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # 构建输出元组
        outputs = (hidden_states,)

        # 如果需要返回注意力权重，则将 attn_weights 加入输出元组
        if output_attentions:
            outputs += (attn_weights,)

        return outputs
        # 这里是 forward 函数的定义，用于实现自动形态解码器层的前向传播逻辑
        # hidden_states: 解码器的隐藏状态张量
        # attention_mask: 注意力掩码，用于指示哪些位置需要被掩盖
        # encoder_hidden_states: 编码器的隐藏状态张量
        # encoder_attention_mask: 编码器的注意力掩码
        # layer_head_mask: 头部掩码，用于控制每个注意力头部的作用
        # cross_attn_layer_head_mask: 交叉注意力头部掩码，用于控制解码器与编码器注意力的头部作用
        # past_key_value: 上一步的键值对，用于实现循环解码器
        # output_attentions: 是否输出注意力权重
        # use_cache: 是否使用缓存以加速解码器的计算
        # 此处缺少部分代码，需补全
    # 初始化神经网络模块的权重
    def _init_weights(self, module):
        # 从配置中获取初始化标准差
        std = self.config.init_std
        # 如果模块是线性层或一维卷积层
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在偏置，则将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是 AutoformerSinusoidalPositionalEmbedding 类的实例
        elif isinstance(module, AutoformerSinusoidalPositionalEmbedding):
            # 不进行任何操作
            pass
        # 如果模块是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在填充索引，则将其对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
# AUTOFORMER_START_DOCSTRING 是一个字符串变量，包含 Autoformer 模型的文档字符串的起始部分
AUTOFORMER_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`AutoformerConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# AUTOFORMER_INPUTS_DOCSTRING 是一个字符串变量，包含 Autoformer 模型的输入文档字符串
AUTOFORMER_INPUTS_DOCSTRING = r"""
"""


# AutoformerEncoder 类，继承自 AutoformerPreTrainedModel 类
# 该类是 Autoformer 模型的编码器部分，由多个 AutoformerEncoderLayer 组成
class AutoformerEncoder(AutoformerPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`AutoformerEncoderLayer`].

    Args:
        config: AutoformerConfig
    """

    # 初始化函数
    def __init__(self, config: AutoformerConfig):
        # 调用父类的初始化函数
        super().__init__(config)

        # 初始化 dropout 参数
        self.dropout = config.dropout
        # 初始化 layerdrop 参数
        self.layerdrop = config.encoder_layerdrop
        # 检查是否指定了 prediction_length 参数
        if config.prediction_length is None:
            raise ValueError("The `prediction_length` config needs to be specified.")

        # 初始化 value_embedding，使用 AutoformerValueEmbedding 类
        self.value_embedding = AutoformerValueEmbedding(feature_size=config.feature_size, d_model=config.d_model)
        # 初始化 embed_positions，使用 AutoformerSinusoidalPositionalEmbedding 类
        self.embed_positions = AutoformerSinusoidalPositionalEmbedding(
            config.context_length + config.prediction_length, config.d_model
        )
        # 初始化 layers，包含多个 AutoformerEncoderLayer，数量由 config.encoder_layers 决定
        self.layers = nn.ModuleList([AutoformerEncoderLayer(config) for _ in range(config.encoder_layers)])
        # 初始化 layernorm_embedding，进行层标准化
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        # 初始化 gradient_checkpointing 参数
        self.gradient_checkpointing = False
        # 调用后处理函数，用于初始化权重和应用最终处理
        self.post_init()

    # 前向传播函数
    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
class AutoformerDecoder(AutoformerPreTrainedModel):
    """
    Transformer decoder consisting of `config.decoder_layers` layers. Each layer is a [`AutoformerDecoderLayer`]

    Args:
        config: AutoformerConfig
    """
    # 初始化函数，接受一个 AutoformerConfig 对象作为参数
    def __init__(self, config: AutoformerConfig):
        # 调用父类的初始化函数
        super().__init__(config)
        # 从配置中获取 dropout 参数
        self.dropout = config.dropout
        # 从配置中获取 decoder_layerdrop 参数
        self.layerdrop = config.decoder_layerdrop
        # 如果配置中未指定 prediction_length，则抛出数值错误
        if config.prediction_length is None:
            raise ValueError("The `prediction_length` config needs to be specified.")

        # 创建值嵌入对象
        self.value_embedding = AutoformerValueEmbedding(feature_size=config.feature_size, d_model=config.d_model)
        # 创建位置嵌入对象
        self.embed_positions = AutoformerSinusoidalPositionalEmbedding(
            config.context_length + config.prediction_length, config.d_model
        )
        # 创建多个 AutoformerDecoderLayer 层，并组成模块列表
        self.layers = nn.ModuleList([AutoformerDecoderLayer(config) for _ in range(config.decoder_layers)])
        # 创建 LayerNorm 层用于嵌入
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        # 创建线性层用于季节性投影
        self.seasonality_projection = nn.Linear(config.d_model, config.feature_size)

        # 初始化梯度检查点为 False
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数，接受多个输入参数
    def forward(
        self,
        trend: Optional[torch.Tensor] = None,
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
# 添加自动编码器模型的文档字符串，描述该模型输出原始隐藏状态而没有特定的顶层头部
@add_start_docstrings(
    "The bare Autoformer Model outputting raw hidden-states without any specific head on top.",
    AUTOFORMER_START_DOCSTRING,
)
class AutoformerModel(AutoformerPreTrainedModel):
    # 初始化方法，接受 AutoformerConfig 类型的配置参数
    def __init__(self, config: AutoformerConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 根据配置参数中的缩放选项选择不同的缩放器
        if config.scaling == "mean" or config.scaling is True:
            self.scaler = AutoformerMeanScaler(config)
        elif config.scaling == "std":
            self.scaler = AutoformerStdScaler(config)
        else:
            self.scaler = AutoformerNOPScaler(config)

        # 如果配置参数中存在静态分类特征，则创建特征嵌入器
        if config.num_static_categorical_features > 0:
            self.embedder = AutoformerFeatureEmbedder(
                cardinalities=config.cardinality, embedding_dims=config.embedding_dimension
            )

        # 创建自动编码器的编码器和解码器
        self.encoder = AutoformerEncoder(config)
        self.decoder = AutoformerDecoder(config)

        # 用于解码器季节性和趋势初始化的分解层
        self.decomposition_layer = AutoformerSeriesDecompositionLayer(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 计算过去长度的属性方法
    @property
    def _past_length(self) -> int:
        return self.config.context_length + max(self.config.lags_sequence)

    # 获取滞后子序列的方法
    def get_lagged_subsequences(
        self, sequence: torch.Tensor, subsequences_length: int, shift: int = 0
    ) -> torch.Tensor:
        """
        Returns lagged subsequences of a given sequence. Returns a tensor of shape (batch_size, subsequences_length,
        feature_size, indices_length), containing lagged subsequences. Specifically, lagged[i, j, :, k] = sequence[i,
        -indices[k]-subsequences_length+j, :].

        Args:
            sequence (`torch.Tensor` or shape `(batch_size, context_length,
                feature_size)`): The sequence from which lagged subsequences should be extracted.
            subsequences_length (`int`):
                Length of the subsequences to be extracted.
            shift (`int`, *optional* defaults to 0):
                Shift the lags by this amount back in the time index.
        """

        # calculates the indices of the lags by subtracting the shift value from the given lags_sequence
        indices = [lag - shift for lag in self.config.lags_sequence]

        # checks if the maximum lag plus the length of the subsequences exceeds the length of the input sequence
        sequence_length = sequence.shape[1]
        if max(indices) + subsequences_length > sequence_length:
            raise ValueError(
                f"lags cannot go further than history length, found lag {max(indices)} "
                f"while history length is only {sequence_length}"
            )

        # extracts the lagged subsequences from the input sequence using the calculated indices
        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...])

        # return as stacked tensor in the feature dimension
        return torch.stack(lagged_values, dim=-1)

    def create_network_inputs(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        static_categorical_features: Optional[torch.Tensor] = None,
        static_real_features: Optional[torch.Tensor] = None,
        past_observed_mask: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        future_time_features: Optional[torch.Tensor] = None,
    ):
        """
        Constructs inputs for the network based on past and future values and features.

        Args:
            past_values (`torch.Tensor`): Tensor containing past values of shape `(batch_size, context_length,
                feature_size)`.
            past_time_features (`torch.Tensor`): Tensor containing time features of the past values of shape
                `(batch_size, context_length, num_features)`.
            static_categorical_features (`Optional[torch.Tensor]`, *optional*): Tensor containing static
                categorical features of shape `(batch_size, num_features)`.
            static_real_features (`Optional[torch.Tensor]`, *optional*): Tensor containing static real-valued
                features of shape `(batch_size, num_features)`.
            past_observed_mask (`Optional[torch.Tensor]`, *optional*): Tensor containing observed mask for the
                past values of shape `(batch_size, context_length)`.
            future_values (`Optional[torch.Tensor]`, *optional*): Tensor containing future values of shape
                `(batch_size, prediction_length, feature_size)`.
            future_time_features (`Optional[torch.Tensor]`, *optional*): Tensor containing time features of the
                future values of shape `(batch_size, prediction_length, num_features)`.

        Returns:
            Tuple: A tuple containing network inputs:
                - `inputs`: Input tensor of shape `(batch_size, total_length, num_features)`.
                - `target`: Target tensor of shape `(batch_size, prediction_length, feature_size)`.
                - `observed_mask`: Observed mask tensor of shape `(batch_size, total_length)`.
        """
        # Combine past and future values and features to construct inputs for the network
        inputs = torch.cat([past_values, future_values], dim=1)
        time_features = torch.cat([past_time_features, future_time_features], dim=1)

        # If available, concatenate static categorical and real-valued features
        if static_categorical_features is not None:
            inputs = torch.cat([inputs, static_categorical_features.unsqueeze(1).repeat(1, inputs.shape[1], 1)], dim=-1)
        if static_real_features is not None:
            inputs = torch.cat([inputs, static_real_features.unsqueeze(1).repeat(1, inputs.shape[1], 1)], dim=-1)

        # If available, concatenate past observed mask
        if past_observed_mask is not None:
            observed_mask = torch.cat([past_observed_mask, torch.ones_like(future_values[..., 0])], dim=1)
        else:
            observed_mask = torch.cat([torch.ones_like(past_values[..., 0]), torch.ones_like(future_values[..., 0])], dim=1)

        return inputs, future_values, observed_mask

    def get_encoder(self):
        """
        Returns the encoder module of the autoformer model.

        Returns:
            nn.Module: The encoder module.
        """
        return self.encoder

    def get_decoder(self):
        """
        Returns the decoder module of the autoformer model.

        Returns:
            nn.Module: The decoder module.
        """
        return self.decoder

    @add_start_docstrings_to_model_forward(AUTOFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=AutoformerModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, *args, **kwargs):
        """
        Refer to the superclass for the full documentation.
        """
        # This method is annotated by `add_start_docstrings_to_model_forward` and `replace_return_docstrings`.
        # Refer to the superclass for detailed documentation.
        return super().forward(*args, **kwargs)
    # 定义一个方法用于模型的前向传播
    def forward(
        # 过去的数值信息，类型为 torch.Tensor
        past_values: torch.Tensor,
        # 过去的时间特征，类型为 torch.Tensor
        past_time_features: torch.Tensor,
        # 过去的观测掩码，类型为 torch.Tensor
        past_observed_mask: torch.Tensor,
        # 静态分类特征，可选参数，类型为 torch.Tensor
        static_categorical_features: Optional[torch.Tensor] = None,
        # 静态实数特征，可选参数，类型为 torch.Tensor
        static_real_features: Optional[torch.Tensor] = None,
        # 未来的数值信息，可选参数，类型为 torch.Tensor
        future_values: Optional[torch.Tensor] = None,
        # 未来的时间特征，可选参数，类型为 torch.Tensor
        future_time_features: Optional[torch.Tensor] = None,
        # 解码器注意力掩码，可选参数，类型为 torch.LongTensor
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        # 头掩码，可选参数，类型为 torch.Tensor
        head_mask: Optional[torch.Tensor] = None,
        # 解码器头掩码，可选参数，类型为 torch.Tensor
        decoder_head_mask: Optional[torch.Tensor] = None,
        # 交叉注意力头掩码，可选参数，类型为 torch.Tensor
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        # 编码器输出，可选参数，类型为 List[torch.FloatTensor]
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        # 过去的键值对，可选参数，类型为 List[torch.FloatTensor]
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # 输出隐藏状态，可选参数，类型为 bool
        output_hidden_states: Optional[bool] = None,
        # 输出注意力，可选参数，类型为 bool
        output_attentions: Optional[bool] = None,
        # 使用缓存，可选参数，类型为 bool
        use_cache: Optional[bool] = None,
        # 返回字典，可选参数，类型为 bool
        return_dict: Optional[bool] = None,
# 为时间序列预测添加一个分布头的 Autoformer 模型
class AutoformerForPrediction(AutoformerPreTrainedModel):
    def __init__(self, config: AutoformerConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建 Autoformer 模型
        self.model = AutoformerModel(config)
        # 根据配置选择不同的分布输出
        if config.distribution_output == "student_t":
            self.distribution_output = StudentTOutput(dim=config.input_size)
        elif config.distribution_output == "normal":
            self.distribution_output = NormalOutput(dim=config.input_size)
        elif config.distribution_output == "negative_binomial":
            self.distribution_output = NegativeBinomialOutput(dim=config.input_size)
        else:
            raise ValueError(f"Unknown distribution output {config.distribution_output}")

        # 获取参数投影
        self.parameter_projection = self.distribution_output.get_parameter_projection(self.model.config.feature_size)
        # 获取目标形状
        self.target_shape = self.distribution_output.event_shape

        # 根据配置选择损失函数
        if config.loss == "nll":
            self.loss = nll
        else:
            raise ValueError(f"Unknown loss function {config.loss}")

        # 初始化分布输出的权重并应用最终处理
        self.post_init()

    # 输出参数
    def output_params(self, decoder_output):
        return self.parameter_projection(decoder_output[:, -self.config.prediction_length :, :])

    # 获取编码器
    def get_encoder(self):
        return self.model.get_encoder()

    # 获取解码器
    def get_decoder(self):
        return self.model.get_decoder()

    # 输出分布
    @torch.jit.ignore
    def output_distribution(self, params, loc=None, scale=None, trailing_n=None) -> torch.distributions.Distribution:
        sliced_params = params
        if trailing_n is not None:
            sliced_params = [p[:, -trailing_n:] for p in params]
        return self.distribution_output.distribution(sliced_params, loc=loc, scale=scale)

    # 添加模型前向传播的文档字符串
    @add_start_docstrings_to_model_forward(AUTOFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqTSPredictionOutput, config_class=_CONFIG_FOR_DOC)
    # 此方法用于模型的前向传播
    def forward(
        self,
        # 过去的数值特征，类型为 torch.Tensor
        past_values: torch.Tensor,
        # 过去的时间特征，类型为 torch.Tensor
        past_time_features: torch.Tensor,
        # 过去的观测掩码，类型为 torch.Tensor
        past_observed_mask: torch.Tensor,
        # 静态分类特征，可选，类型为 torch.Tensor
        static_categorical_features: Optional[torch.Tensor] = None,
        # 静态实数特征，可选，类型为 torch.Tensor
        static_real_features: Optional[torch.Tensor] = None,
        # 未来的数值特征，可选，类型为 torch.Tensor
        future_values: Optional[torch.Tensor] = None,
        # 未来的时间特征，可选，类型为 torch.Tensor
        future_time_features: Optional[torch.Tensor] = None,
        # 未来的观测掩码，可选，类型为 torch.Tensor
        future_observed_mask: Optional[torch.Tensor] = None,
        # 解码器注意力掩码，可选，类型为 torch.LongTensor
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        # 头部掩码，可选，类型为 torch.Tensor
        head_mask: Optional[torch.Tensor] = None,
        # 解码器头部掩码，可选，类型为 torch.Tensor
        decoder_head_mask: Optional[torch.Tensor] = None,
        # 交叉注意力头部掩码，可选，类型为 torch.Tensor
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        # 编码器输出，可选，类型为 List[torch.FloatTensor]
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        # 过去的键值对，可选，类型为 List[torch.FloatTensor]
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # 是否输出隐藏状态，可选，类型为 bool
        output_hidden_states: Optional[bool] = None,
        # 是否输出注意力权重，可选，类型为 bool
        output_attentions: Optional[bool] = None,
        # 是否使用缓存，可选，类型为 bool
        use_cache: Optional[bool] = None,
        # 是否返回字典，可选，类型为 bool
        return_dict: Optional[bool] = None,
    ):
    # 用于禁用梯度计算的装饰器
    @torch.no_grad()
    # 此方法用于生成
    def generate(
        # 过去的数值特征，类型为 torch.Tensor
        self,
        past_values: torch.Tensor,
        # 过去的时间特征，类型为 torch.Tensor
        past_time_features: torch.Tensor,
        # 未来的时间特征，类型为 torch.Tensor
        future_time_features: torch.Tensor,
        # 过去的观测掩码，可选，类型为 torch.Tensor
        past_observed_mask: Optional[torch.Tensor] = None,
        # 静态分类特征，可选，类型为 torch.Tensor
        static_categorical_features: Optional[torch.Tensor] = None,
        # 静态实数特征，可选，类型为 torch.Tensor
        static_real_features: Optional[torch.Tensor] = None,
        # 是否输出注意力权重，可选，类型为 bool
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，可选，类型为 bool
        output_hidden_states: Optional[bool] = None,
```