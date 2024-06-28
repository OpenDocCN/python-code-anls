# `.\models\autoformer\modeling_autoformer.py`

```
# 设置编码格式为 UTF-8，确保代码中可以正确处理各种字符
# 版权声明，这些代码的版权归清华大学 THUML、亚马逊公司及其关联公司以及HuggingFace团队所有
# 根据 Apache 许可证 2.0 版本，你可以在遵守许可证的情况下使用这些代码
# 访问 http://www.apache.org/licenses/LICENSE-2.0 查看许可证的详细信息

""" PyTorch Autoformer model. """

# 导入必要的库和模块
import math  # 导入数学库
from dataclasses import dataclass  # 导入数据类
from typing import List, Optional, Tuple, Union  # 导入类型提示相关模块

import numpy as np  # 导入 NumPy 库
import torch  # 导入 PyTorch 库
import torch.utils.checkpoint  # 导入 PyTorch 中的 checkpoint 功能
from torch import nn  # 从 PyTorch 中导入神经网络模块

# 导入额外的自定义模块和函数
from ...activations import ACT2FN  # 从模型中导入激活函数 ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask  # 导入注意力掩码相关函数
from ...modeling_outputs import (  # 导入模型输出相关类
    BaseModelOutput,
    ModelOutput,
    SampleTSPredictionOutput,
    Seq2SeqTSPredictionOutput,
)
from ...modeling_utils import PreTrainedModel  # 导入预训练模型相关工具函数
from ...time_series_utils import (  # 导入时间序列相关输出
    NegativeBinomialOutput,
    NormalOutput,
    StudentTOutput,
)
from ...utils import (  # 导入通用工具函数
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_autoformer import AutoformerConfig  # 导入 Autoformer 的配置文件

# 获取 logger 对象用于记录日志
logger = logging.get_logger(__name__)

# 用于文档的配置名称
_CONFIG_FOR_DOC = "AutoformerConfig"


@dataclass
class AutoFormerDecoderOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).
    """
    pass  # AutoFormerDecoderOutput 类的基类，用于模型输出，可能包含过去的键/值以加速顺序解码
    # 最后一层模型的隐藏状态，形状为 `(batch_size, sequence_length, hidden_size)`
    last_hidden_state: torch.FloatTensor = None
    
    # 每个时间序列的趋势张量，形状为 `(batch_size, sequence_length, hidden_size)`
    trend: torch.FloatTensor = None
    
    # 如果使用了缓存 (`use_cache=True` 或 `config.use_cache=True`)，则返回的预计算密钥和值
    # 是一个元组，包含长度为 `config.n_layers` 的元组，每个元组包含两个形状为
    # `(batch_size, num_heads, sequence_length, embed_size_per_head)` 的张量。如果
    # `config.is_encoder_decoder=True`，还包括两个额外的张量，形状为
    # `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`。
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    
    # 如果输出隐藏状态 (`output_hidden_states=True` 或 `config.output_hidden_states=True`)，
    # 则返回的隐藏状态是一个元组，包含以下两个张量：
    # 1. 形状为 `(batch_size, sequence_length, hidden_size)` 的模型每一层的输出隐藏状态；
    # 2. 如果模型有嵌入层，则包括形状为 `(batch_size, sequence_length, hidden_size)` 的初始嵌入输出。
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    
    # 如果输出注意力权重 (`output_attentions=True` 或 `config.output_attentions=True`)，
    # 则返回的注意力权重是一个元组，包含每一层的注意力权重张量：
    # 形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    
    # 如果输出交叉注意力权重 (`output_attentions=True` 且 `config.add_cross_attention=True` 或 `config.output_attentions=True`)，
    # 则返回的交叉注意力权重是一个元组，包含每一层的交叉注意力权重张量：
    # 形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 定义一个名为 attentions 的可选类型变量，用于存储一个包含 torch.FloatTensor 类型对象的元组或者为 None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    
    # 定义一个名为 cross_attentions 的可选类型变量，用于存储一个包含 torch.FloatTensor 类型对象的元组或者为 None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
# 定义一个数据类，用于存储Autoformer模型的输出，包括最后隐藏状态、趋势、过去的键值、解码器隐藏状态、
# 解码器注意力、交叉注意力、编码器最后隐藏状态、编码器隐藏状态、编码器注意力、位置和规模以及静态特征
@dataclass
class AutoformerModelOutput(ModelOutput):
    """
    Autoformer model output that contains the additional trend output.
    """

    last_hidden_state: torch.FloatTensor = None  # 最后隐藏状态
    trend: torch.FloatTensor = None  # 趋势
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # 过去的键值
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 解码器隐藏状态
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None  # 解码器注意力
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None  # 交叉注意力
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None  # 编码器最后隐藏状态
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 编码器隐藏状态
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None  # 编码器注意力
    loc: Optional[torch.FloatTensor] = None  # 位置
    scale: Optional[torch.FloatTensor] = None  # 规模
    static_features: Optional[torch.FloatTensor] = None  # 静态特征


AUTOFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "huggingface/autoformer-tourism-monthly",
    # 查看所有Autoformer模型的列表链接
    # See all Autoformer models at https://huggingface.co/models?filter=autoformer
]


# 从transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesFeatureEmbedder复制而来，
# 更名为AutoformerFeatureEmbedder，用于嵌入序列的分类特征
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

        # 计算分类特征的数量
        self.num_features = len(cardinalities)
        # 创建嵌入层列表，每个分类特征对应一个嵌入层
        self.embedders = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(cardinalities, embedding_dims)])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.num_features > 1:
            # 切分最后一个维度，得到一个形状为(N, T)或者(N)的长度为self.num_features的数组
            cat_feature_slices = torch.chunk(features, self.num_features, dim=-1)
        else:
            cat_feature_slices = [features]

        # 将每个切片通过对应的嵌入层嵌入，并在最后一个维度上拼接起来
        return torch.cat(
            [
                embed(cat_feature_slice.squeeze(-1))
                for embed, cat_feature_slice in zip(self.embedders, cat_feature_slices)
            ],
            dim=-1,
        )


# 从transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesStdScaler复制而来，
# 更名为AutoformerStdScaler，用于标准化特征
class AutoformerStdScaler(nn.Module):
    """
    Standardize features by calculating the mean and scaling along the first dimension, and then normalizes it by
    subtracting from the mean and dividing by the standard deviation.
    """
    # 初始化方法，接受一个 AutoformerConfig 对象作为参数
    def __init__(self, config: AutoformerConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 如果 config 对象有 scaling_dim 属性，则将其赋值给 self.dim；否则默认为 1
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        # 如果 config 对象有 keepdim 属性，则将其赋值给 self.keepdim；否则默认为 True
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True
        # 如果 config 对象有 minimum_scale 属性，则将其赋值给 self.minimum_scale；否则默认为 1e-5
        self.minimum_scale = config.minimum_scale if hasattr(config, "minimum_scale") else 1e-5

    # 前向传播方法，接受两个参数并返回三个 Tensor 对象的元组
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
        # 计算 observed_indicator 在指定维度上的和，根据 keepdim 参数决定是否保持维度
        denominator = observed_indicator.sum(self.dim, keepdim=self.keepdim)
        # 将 denominator 中的值限制下限为 1.0
        denominator = denominator.clamp_min(1.0)
        # 计算 loc（均值），使用 data 和 observed_indicator 的乘积，并在指定维度上求和
        loc = (data * observed_indicator).sum(self.dim, keepdim=self.keepdim) / denominator

        # 计算方差，使用 data、loc 和 observed_indicator 计算
        variance = (((data - loc) * observed_indicator) ** 2).sum(self.dim, keepdim=self.keepdim) / denominator
        # 计算 scale（标准差），在方差的基础上加上 minimum_scale，然后取平方根
        scale = torch.sqrt(variance + self.minimum_scale)
        # 返回标准化后的 data、loc 和 scale
        return (data - loc) / scale, loc, scale
# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesMeanScaler with TimeSeriesTransformer->Autoformer,TimeSeries->Autoformer
class AutoformerMeanScaler(nn.Module):
    """
    Computes a scaling factor as the weighted average absolute value along the first dimension, and scales the data
    accordingly.
    """

    def __init__(self, config: AutoformerConfig):
        super().__init__()
        # 初始化时从配置中获取缩放的维度，默认为第一维度
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        # 是否保持维度，默认为True
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True
        # 最小缩放值，默认为1e-10
        self.minimum_scale = config.minimum_scale if hasattr(config, "minimum_scale") else 1e-10
        # 默认缩放值，如果配置中有指定则使用，否则为None
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
        # 计算加权平均绝对值，以第一维度为基础
        ts_sum = (data * observed_indicator).abs().sum(self.dim, keepdim=True)
        # 计算观测指标的数量
        num_observed = observed_indicator.sum(self.dim, keepdim=True)

        # 计算缩放比例，确保不会除以零
        scale = ts_sum / torch.clamp(num_observed, min=1)

        # 如果未提供 `default_scale`，则使用批次的缩放比例，否则使用指定的缩放比例
        if self.default_scale is None:
            batch_sum = ts_sum.sum(dim=0)
            batch_observations = torch.clamp(num_observed.sum(0), min=1)
            default_scale = torch.squeeze(batch_sum / batch_observations)
        else:
            default_scale = self.default_scale * torch.ones_like(scale)

        # 应用默认缩放比例到没有观测到的地方
        scale = torch.where(num_observed > 0, scale, default_scale)

        # 确保缩放比例至少为 `self.minimum_scale`
        scale = torch.clamp(scale, min=self.minimum_scale)
        # 应用缩放到数据上
        scaled_data = data / scale

        # 如果不保持维度，则去除对应维度的缩放比例
        if not self.keepdim:
            scale = scale.squeeze(dim=self.dim)

        return scaled_data, torch.zeros_like(scale), scale


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesNOPScaler with TimeSeriesTransformer->Autoformer,TimeSeries->Autoformer
class AutoformerNOPScaler(nn.Module):
    """
    Assigns a scaling factor equal to 1 along the first dimension, and therefore applies no scaling to the input data.
    """
    # 初始化方法，接受一个配置参数 `config`，类型为 `AutoformerConfig`
    def __init__(self, config: AutoformerConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 如果配置对象 `config` 中有 `scaling_dim` 属性，则将其赋值给 `self.dim`，否则设为 1
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        # 如果配置对象 `config` 中有 `keepdim` 属性，则将其赋值给 `self.keepdim`，否则设为 True
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True

    # 前向传播方法，接受输入数据 `data` 和可选的观察指示器 `observed_indicator`
    # 返回值是一个元组，包含三个 `torch.Tensor` 类型的张量
    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 计算 `data` 张量每个维度的均值，并创建与 `data` 相同形状的全为 1 的张量 `scale`
        scale = torch.ones_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        # 创建与 `data` 相同形状的全为 0 的张量 `loc`
        loc = torch.zeros_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        # 返回原始输入 `data`，以及计算得到的 `loc` 和 `scale`
        return data, loc, scale
# 从transformers.models.time_series_transformer.modeling_time_series_transformer.weighted_average中复制过来的函数
def weighted_average(input_tensor: torch.Tensor, weights: Optional[torch.Tensor] = None, dim=None) -> torch.Tensor:
    """
    计算给定维度上张量的加权平均值，并对与权重为零相关的值进行掩码处理，
    这意味着你将得到`nan * 0 = nan`的替代值`0 * 0 = 0`。

    Args:
        input_tensor (`torch.FloatTensor`):
            输入张量，需要计算平均值。
        weights (`torch.FloatTensor`, *可选*):
            权重张量，与`input_tensor`形状相同。
        dim (`int`, *可选*):
            沿着哪个维度对`input_tensor`进行平均。

    Returns:
        `torch.FloatTensor`: 沿指定`dim`平均值的张量。
    """
    if weights is not None:
        weighted_tensor = torch.where(weights != 0, input_tensor * weights, torch.zeros_like(input_tensor))
        sum_weights = torch.clamp(weights.sum(dim=dim) if dim else weights.sum(), min=1.0)
        return (weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()) / sum_weights
    else:
        return input_tensor.mean(dim=dim)


# 从transformers.models.time_series_transformer.modeling_time_series_transformer.nll中复制过来的函数
def nll(input: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor:
    """
    从输入分布计算与目标相关的负对数似然损失。
    """
    return -input.log_prob(target)


# 从transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding复制到Autoformer
class AutoformerSinusoidalPositionalEmbedding(nn.Embedding):
    """该模块产生任意长度的正弦位置嵌入。"""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None) -> None:
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """
        与XLM create_sinusoidal_embeddings相同，除了特征不是交错的。余弦特征在向量的第二半部分。[dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False  # 设置早以避免在pytorch-1.8+中出错
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        # 从 `input_ids_shape` 中获取 batch size (bsz) 和 sequence length (seq_len)
        bsz, seq_len = input_ids_shape[:2]
        # 根据 past_key_values_length 和 seq_len 创建位置编码的张量，设备为 self.weight 的设备
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        # 调用父类的 forward 方法，传入位置编码张量，返回结果张量
        return super().forward(positions)
# 从transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesValueEmbedding复制到Autoformer
class AutoformerValueEmbedding(nn.Module):
    def __init__(self, feature_size, d_model):
        super().__init__()
        # 定义线性投影层，将输入特征大小映射到模型维度大小，无偏置
        self.value_projection = nn.Linear(in_features=feature_size, out_features=d_model, bias=False)

    def forward(self, x):
        # 前向传播函数，将输入数据进行线性投影
        return self.value_projection(x)


# 基于以下链接的类
# https://github.com/thuml/Autoformer/blob/c6a0694ff484753f2d986cc0bb1f99ee850fc1a8/layers/Autoformer_EncDec.py#L39
# 其中AutoformerSeriesDecompositionLayer是series_decomp + moving_average
class AutoformerSeriesDecompositionLayer(nn.Module):
    """
    返回时间序列的趋势和季节部分。计算方式为:

        x_trend = AvgPool(Padding(X)) and x_seasonal = X - x_trend
    """

    def __init__(self, config: AutoformerConfig):
        super().__init__()
        # 设置移动平均的内核大小
        self.kernel_size = config.moving_average
        # 定义一维平均池化层，用于计算移动平均
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)

    def forward(self, x):
        """输入形状: Batch x Time x EMBED_DIM"""
        # 在时间序列的两端进行填充
        num_of_pads = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, num_of_pads, 1)
        end = x[:, -1:, :].repeat(1, num_of_pads, 1)
        x_padded = torch.cat([front, x, end], dim=1)

        # 计算时间序列的趋势和季节部分
        x_trend = self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
        x_seasonal = x - x_trend
        return x_seasonal, x_trend


# 基于以下链接的类
# https://github.com/thuml/Autoformer/blob/c6a0694ff484753f2d986cc0bb1f99ee850fc1a8/layers/Autoformer_EncDec.py#L6
# 其中AutoformerLayernorm是my_Layernorm
class AutoformerLayernorm(nn.Module):
    """
    为季节部分设计的特殊层归一化，计算方式为: AutoformerLayernorm(x) = nn.LayerNorm(x)
    - torch.mean(nn.LayerNorm(x))
    """

    def __init__(self, config: AutoformerConfig):
        super().__init__()
        # 定义LayerNorm层，将模型维度归一化
        self.layernorm = nn.LayerNorm(config.d_model)

    def forward(self, x):
        # 对输入数据进行LayerNorm
        x_hat = self.layernorm(x)
        # 计算偏置，对LayerNorm的输出进行均值操作
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class AutoformerAttention(nn.Module):
    """
    自相关机制，包含以下两个阶段:
        (1) 基于周期的依赖发现 (2) 时间延迟聚合
    该模块替代了传统的自注意力机制。
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        autocorrelation_factor: int = 3,
        # 省略了后续的初始化参数说明
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        # 初始化线性层，用于查询、键、值和输出的投影
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # 自动相关因子，用于注意力计算
        self.autocorrelation_factor = autocorrelation_factor

    # 重新塑造张量形状，用于多头注意力的计算
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传播函数，实现注意力机制的计算
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
# AutoformerEncoderLayer 类定义，继承自 nn.Module，表示这是一个 PyTorch 模型层
class AutoformerEncoderLayer(nn.Module):
    # 初始化函数，接受一个 AutoformerConfig 对象作为参数
    def __init__(self, config: AutoformerConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 设置 embed_dim 为配置中的 d_model，表示嵌入维度
        self.embed_dim = config.d_model
        # self_attn 属性，使用 AutoformerAttention 自定义注意力层
        self.self_attn = AutoformerAttention(
            embed_dim=self.embed_dim,  # 设置注意力层的嵌入维度
            num_heads=config.encoder_attention_heads,  # 注意力头的数量
            dropout=config.attention_dropout,  # 注意力层的dropout率
            autocorrelation_factor=config.autocorrelation_factor,  # 自相关因子
        )
        # self_attn_layer_norm 属性，LayerNorm 层，用于规范化注意力层的输出
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # dropout 属性，全局的dropout率
        self.dropout = config.dropout
        # activation_fn 属性，激活函数，根据配置选择对应的激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # activation_dropout 属性，激活函数的dropout率
        self.activation_dropout = config.activation_dropout
        # fc1 属性，全连接层1，输入维度为 embed_dim，输出维度为配置中的 encoder_ffn_dim
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        # fc2 属性，全连接层2，输入维度为配置中的 encoder_ffn_dim，输出维度为 embed_dim
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        # final_layer_norm 属性，最终输出的 LayerNorm 层
        self.final_layer_norm = AutoformerLayernorm(config)
        # decomp1 和 decomp2 属性，使用 AutoformerSeriesDecompositionLayer 进行时间序列分解
        self.decomp1 = AutoformerSeriesDecompositionLayer(config)
        self.decomp2 = AutoformerSeriesDecompositionLayer(config)

    # forward 方法，定义了模型层的前向传播逻辑
    def forward(
        self,
        hidden_states: torch.FloatTensor,  # 输入的隐藏状态张量
        attention_mask: torch.FloatTensor,  # 注意力掩码张量
        layer_head_mask: torch.FloatTensor,  # 层头掩码张量
        output_attentions: Optional[bool] = False,  # 是否输出注意力权重，默认为 False
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
        # 保存输入的原始值，用于残差连接
        residual = hidden_states
        # 调用自注意力机制层进行计算
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 使用 dropout 进行正则化
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 残差连接
        hidden_states = residual + hidden_states
        # 在此处添加层归一化以改进模型性能
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 经过第一个线性层和激活函数
        hidden_states, _ = self.decomp1(hidden_states)

        # 保存输入的原始值，用于残差连接
        residual = hidden_states
        # 经过第二个线性层和激活函数
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 使用 dropout 进行正则化
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # 经过第三个线性层
        hidden_states = self.fc2(hidden_states)
        # 使用 dropout 进行正则化
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 残差连接
        hidden_states = residual + hidden_states
        # 经过第二个分解层
        hidden_states, _ = self.decomp2(hidden_states)
        # 最终层归一化
        hidden_states = self.final_layer_norm(hidden_states)

        # 如果隐藏状态的数据类型为 torch.float16 并且存在无穷大或 NaN 的情况，进行数值截断
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # 构建输出元组
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则添加到输出元组中
        if output_attentions:
            outputs += (attn_weights,)

        return outputs
# 定义 AutoformerDecoderLayer 类，继承自 nn.Module
class AutoformerDecoderLayer(nn.Module):
    # 初始化方法，接受一个 AutoformerConfig 类型的 config 参数
    def __init__(self, config: AutoformerConfig):
        # 调用父类 nn.Module 的初始化方法
        super().__init__()
        # 设置 embed_dim 属性为 config.d_model，即模型的维度
        self.embed_dim = config.d_model

        # 初始化自注意力层 self_attn，使用 AutoformerAttention 类
        self.self_attn = AutoformerAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            autocorrelation_factor=config.autocorrelation_factor,
        )

        # 设置 dropout 属性为 config.dropout，用于网络的随机失活
        self.dropout = config.dropout
        # 设置 activation_fn 属性为 config.activation_function 对应的激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 设置 activation_dropout 属性为 config.activation_dropout，用于激活函数的随机失活

        self.activation_dropout = config.activation_dropout

        # 初始化自注意力层后的 LayerNorm 层 self_attn_layer_norm
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 初始化编码器注意力层 encoder_attn，使用 AutoformerAttention 类
        self.encoder_attn = AutoformerAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            autocorrelation_factor=config.autocorrelation_factor,
        )

        # 初始化编码器注意力层后的 LayerNorm 层 encoder_attn_layer_norm
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 初始化全连接层 fc1，输入维度为 self.embed_dim，输出维度为 config.decoder_ffn_dim
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        # 初始化全连接层 fc2，输入维度为 config.decoder_ffn_dim，输出维度为 self.embed_dim
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)

        # 初始化最终的 LayerNorm 层 final_layer_norm，使用 AutoformerLayernorm 类
        self.final_layer_norm = AutoformerLayernorm(config)

        # 初始化 AutoformerSeriesDecompositionLayer 类的实例 decomp1, decomp2, decomp3
        self.decomp1 = AutoformerSeriesDecompositionLayer(config)
        self.decomp2 = AutoformerSeriesDecompositionLayer(config)
        self.decomp3 = AutoformerSeriesDecompositionLayer(config)

        # 初始化趋势投影层 trend_projection，使用 nn.Conv1d 类
        # 设置输入通道数为 self.embed_dim，输出通道数为 config.feature_size
        # 使用 kernel_size=3 的卷积核，步长为 1，padding 方式为 circular，无偏置项
        self.trend_projection = nn.Conv1d(
            in_channels=self.embed_dim,
            out_channels=config.feature_size,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=False,
        )

    # 前向传播方法定义，接受多个参数，包括隐藏状态、注意力掩码等
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
    ):
        # 这里可以添加具体的前向传播逻辑，但不在注释范围内
        pass

# 定义 AutoformerPreTrainedModel 类，继承自 PreTrainedModel
class AutoformerPreTrainedModel(PreTrainedModel):
    # 设置配置类为 AutoformerConfig
    config_class = AutoformerConfig
    # 设置基础模型前缀为 "model"
    base_model_prefix = "model"
    # 设置主输入名称为 "past_values"
    main_input_name = "past_values"
    # 支持梯度检查点技术
    supports_gradient_checkpointing = True
    # 初始化神经网络模块的权重
    def _init_weights(self, module):
        std = self.config.init_std
        # 如果模块是线性层或一维卷积层
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # 使用正态分布初始化权重，均值为0，标准差为config中指定的值std
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在偏置项，则将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是AutoformerSinusoidalPositionalEmbedding类型的，不进行任何操作
        elif isinstance(module, AutoformerSinusoidalPositionalEmbedding):
            pass
        # 如果模块是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为0，标准差为config中指定的值std
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果指定了padding_idx，则将对应位置的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
# AUTOFORMER_START_DOCSTRING 变量，包含了关于 Autoformer 模型的详细文档字符串，介绍了模型的继承关系和参数说明
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

# AUTOFORMER_INPUTS_DOCSTRING 变量，当前为空字符串，用于添加输入参数的文档字符串
AUTOFORMER_INPUTS_DOCSTRING = r"""
"""


# AutoformerEncoder 类定义，继承自 AutoformerPreTrainedModel，代表了 Autoformer 模型的编码器部分
class AutoformerEncoder(AutoformerPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`AutoformerEncoderLayer`].

    Args:
        config: AutoformerConfig
    """

    def __init__(self, config: AutoformerConfig):
        # 调用父类构造函数初始化模型
        super().__init__(config)

        # 初始化类成员变量
        self.dropout = config.dropout  # 设置模型的 dropout 率
        self.layerdrop = config.encoder_layerdrop  # 设置编码器层级的 dropout 率
        if config.prediction_length is None:
            raise ValueError("The `prediction_length` config needs to be specified.")

        # 初始化模型的值嵌入和位置嵌入
        self.value_embedding = AutoformerValueEmbedding(feature_size=config.feature_size, d_model=config.d_model)
        self.embed_positions = AutoformerSinusoidalPositionalEmbedding(
            config.context_length + config.prediction_length, config.d_model
        )
        
        # 使用 AutoformerEncoderLayer 初始化编码器的层，并组成层的列表
        self.layers = nn.ModuleList([AutoformerEncoderLayer(config) for _ in range(config.encoder_layers)])
        
        # 应用层归一化到嵌入
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # 初始化权重并进行最终处理
        self.post_init()

    # 前向传播函数，接受多个可选的输入参数，并返回模型的输出
    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 初始化函数，接受一个 AutoformerConfig 类型的参数 config
    def __init__(self, config: AutoformerConfig):
        # 调用父类的初始化函数，传入 config 参数
        super().__init__(config)
        # 设置 dropout 参数为 config 中的 dropout 设置
        self.dropout = config.dropout
        # 设置 layerdrop 参数为 config 中的 decoder_layerdrop 设置
        self.layerdrop = config.decoder_layerdrop
        # 如果 config 中的 prediction_length 参数为 None，则抛出数值错误异常
        if config.prediction_length is None:
            raise ValueError("The `prediction_length` config needs to be specified.")

        # 初始化 AutoformerValueEmbedding 对象，使用 config 中的 feature_size 和 d_model 参数
        self.value_embedding = AutoformerValueEmbedding(feature_size=config.feature_size, d_model=config.d_model)
        # 初始化 AutoformerSinusoidalPositionalEmbedding 对象，使用 config 中的 context_length、prediction_length 和 d_model 参数
        self.embed_positions = AutoformerSinusoidalPositionalEmbedding(
            config.context_length + config.prediction_length, config.d_model
        )
        # 使用列表推导式初始化 nn.ModuleList，包含 config.decoder_layers 个 AutoformerDecoderLayer 对象
        self.layers = nn.ModuleList([AutoformerDecoderLayer(config) for _ in range(config.decoder_layers)])
        # 初始化 nn.LayerNorm 对象，使用 config 中的 d_model 参数
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        # 使用 nn.Linear 初始化 seasonality_projection 属性，将 d_model 映射到 feature_size
        self.seasonality_projection = nn.Linear(config.d_model, config.feature_size)

        # 设置 gradient_checkpointing 属性为 False
        self.gradient_checkpointing = False
        # 执行初始化函数 post_init，用于初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数，接受多个可选参数并返回结果
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
# 使用自动形态编码器的基类进行模型定义，输出原始隐藏状态，没有特定的顶部头部。
# 继承自AutoformerPreTrainedModel类
class AutoformerModel(AutoformerPreTrainedModel):
    
    def __init__(self, config: AutoformerConfig):
        super().__init__(config)

        # 根据配置选择合适的数据缩放器
        if config.scaling == "mean" or config.scaling is True:
            self.scaler = AutoformerMeanScaler(config)
        elif config.scaling == "std":
            self.scaler = AutoformerStdScaler(config)
        else:
            self.scaler = AutoformerNOPScaler(config)

        # 如果有静态分类特征，则初始化特征嵌入器
        if config.num_static_categorical_features > 0:
            self.embedder = AutoformerFeatureEmbedder(
                cardinalities=config.cardinality, embedding_dims=config.embedding_dimension
            )

        # 初始化编码器和解码器部分
        self.encoder = AutoformerEncoder(config)  # 自动形态编码器的编码器部分
        self.decoder = AutoformerDecoder(config)  # 自动形态编码器的解码器部分

        # 用于解码器季节性和趋势初始化的分解层
        self.decomposition_layer = AutoformerSeriesDecompositionLayer(config)

        # 初始化权重并应用最终处理
        self.post_init()

    @property
    def _past_length(self) -> int:
        # 返回上下文长度和滞后序列中的最大值之和，作为过去观察长度
        return self.config.context_length + max(self.config.lags_sequence)

    def get_lagged_subsequences(
        self, sequence: torch.Tensor, subsequences_length: int, shift: int = 0
        # 根据给定序列获取滞后子序列，指定子序列长度和偏移量
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
        Creates inputs for the network by combining past values, time features, and optional static features.

        Args:
            past_values (`torch.Tensor`): Tensor containing past values of shape (batch_size, context_length, feature_size).
            past_time_features (`torch.Tensor`): Tensor containing time features for the past values.
            static_categorical_features (`Optional[torch.Tensor]`, *optional*):
                Tensor containing static categorical features.
            static_real_features (`Optional[torch.Tensor]`, *optional*):
                Tensor containing static real-valued features.
            past_observed_mask (`Optional[torch.Tensor]`, *optional*):
                Mask indicating which past values are observed.
            future_values (`Optional[torch.Tensor]`, *optional*):
                Tensor containing future values if available.
            future_time_features (`Optional[torch.Tensor]`, *optional*):
                Tensor containing time features for the future values.
        """
        return NotImplementedError

    def get_encoder(self):
        """
        Returns the encoder object associated with this model.
        """
        return self.encoder

    def get_decoder(self):
        """
        Returns the decoder object associated with this model.
        """
        return self.decoder

    @add_start_docstrings_to_model_forward(AUTOFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=AutoformerModelOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个方法 `forward`，用于执行模型的前向传播过程，接受多个参数作为输入
    def forward(
        self,
        # 过去的值，作为模型的输入之一，是一个 Tensor
        past_values: torch.Tensor,
        # 过去的时间特征，也是模型输入的一部分，是一个 Tensor
        past_time_features: torch.Tensor,
        # 过去观测的遮罩，用于指示哪些观测值在过去是可见的，是一个 Tensor
        past_observed_mask: torch.Tensor,
        # 静态的分类特征，可选输入，如果有的话是一个 Tensor
        static_categorical_features: Optional[torch.Tensor] = None,
        # 静态的实数特征，可选输入，如果有的话是一个 Tensor
        static_real_features: Optional[torch.Tensor] = None,
        # 未来的值，可选输入，如果有的话是一个 Tensor
        future_values: Optional[torch.Tensor] = None,
        # 未来的时间特征，可选输入，如果有的话是一个 Tensor
        future_time_features: Optional[torch.Tensor] = None,
        # 解码器注意力遮罩，可选输入，如果有的话是一个 LongTensor
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        # 头部遮罩，可选输入，如果有的话是一个 Tensor
        head_mask: Optional[torch.Tensor] = None,
        # 解码器头部遮罩，可选输入，如果有的话是一个 Tensor
        decoder_head_mask: Optional[torch.Tensor] = None,
        # 交叉注意力头部遮罩，可选输入，如果有的话是一个 Tensor
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        # 编码器输出，可选输入，如果有的话是一个浮点数列表
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        # 过去的键值对，可选输入，如果有的话是一个浮点数列表
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # 输出隐藏状态，可选参数，如果设置为 True 则输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 输出注意力权重，可选参数，如果设置为 True 则输出注意力权重
        output_attentions: Optional[bool] = None,
        # 使用缓存，可选参数，如果设置为 True 则使用缓存
        use_cache: Optional[bool] = None,
        # 返回字典，可选参数，如果设置为 True 则返回字典
        return_dict: Optional[bool] = None,
# 使用装饰器为该类添加文档字符串，描述了该类是基于 Autoformer 模型的时间序列预测模型，带有一个分布输出头部
# 以用于时间序列预测。
@add_start_docstrings(
    "The Autoformer Model with a distribution head on top for time-series forecasting.",
    AUTOFORMER_START_DOCSTRING,
)
class AutoformerForPrediction(AutoformerPreTrainedModel):
    def __init__(self, config: AutoformerConfig):
        # 调用父类构造函数，传入配置对象来初始化
        super().__init__(config)
        # 使用给定配置初始化 AutoformerModel 模型
        self.model = AutoformerModel(config)
        
        # 根据配置选择分布输出类型，并初始化相应的分布输出对象
        if config.distribution_output == "student_t":
            self.distribution_output = StudentTOutput(dim=config.input_size)
        elif config.distribution_output == "normal":
            self.distribution_output = NormalOutput(dim=config.input_size)
        elif config.distribution_output == "negative_binomial":
            self.distribution_output = NegativeBinomialOutput(dim=config.input_size)
        else:
            # 如果配置中指定的分布输出类型未知，则引发值错误异常
            raise ValueError(f"Unknown distribution output {config.distribution_output}")

        # 根据分布输出对象的特征大小获取参数投影方法
        self.parameter_projection = self.distribution_output.get_parameter_projection(self.model.config.feature_size)
        # 设置目标形状为分布输出对象的事件形状
        self.target_shape = self.distribution_output.event_shape

        # 根据配置选择损失函数，如果未知则引发值错误异常
        if config.loss == "nll":
            self.loss = nll
        else:
            raise ValueError(f"Unknown loss function {config.loss}")

        # 初始化分布输出对象的权重并应用最终处理
        self.post_init()

    # 返回解码器输出的参数投影
    def output_params(self, decoder_output):
        return self.parameter_projection(decoder_output[:, -self.config.prediction_length :, :])

    # 获取编码器部分
    def get_encoder(self):
        return self.model.get_encoder()

    # 获取解码器部分
    def get_decoder(self):
        return self.model.get_decoder()

    # 使用 torch.jit.ignore 装饰器，指示编译时忽略该方法，该方法用于生成分布对象
    @torch.jit.ignore
    def output_distribution(self, params, loc=None, scale=None, trailing_n=None) -> torch.distributions.Distribution:
        # 如果指定了 trailing_n，则对参数进行切片
        sliced_params = params
        if trailing_n is not None:
            sliced_params = [p[:, -trailing_n:] for p in params]
        # 调用分布输出对象的 distribution 方法生成分布对象
        return self.distribution_output.distribution(sliced_params, loc=loc, scale=scale)

    # 使用装饰器为模型的前向传播方法添加文档字符串，文档字符串包含了输入的详细说明
    @add_start_docstrings_to_model_forward(AUTOFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqTSPredictionOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个方法 `forward`，用于模型的前向传播
    def forward(
        self,
        # 过去的值作为输入，类型为 torch.Tensor
        past_values: torch.Tensor,
        # 过去的时间特征作为输入，类型为 torch.Tensor
        past_time_features: torch.Tensor,
        # 过去观察到的掩码，类型为 torch.Tensor
        past_observed_mask: torch.Tensor,
        # 可选的静态分类特征，类型为 Optional[torch.Tensor]
        static_categorical_features: Optional[torch.Tensor] = None,
        # 可选的静态实数特征，类型为 Optional[torch.Tensor]
        static_real_features: Optional[torch.Tensor] = None,
        # 可选的未来的值，类型为 Optional[torch.Tensor]
        future_values: Optional[torch.Tensor] = None,
        # 可选的未来时间特征，类型为 Optional[torch.Tensor]
        future_time_features: Optional[torch.Tensor] = None,
        # 可选的未来观察到的掩码，类型为 Optional[torch.Tensor]
        future_observed_mask: Optional[torch.Tensor] = None,
        # 可选的解码器注意力掩码，类型为 Optional[torch.LongTensor]
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        # 可选的头掩码，类型为 Optional[torch.Tensor]
        head_mask: Optional[torch.Tensor] = None,
        # 可选的解码器头部掩码，类型为 Optional[torch.Tensor]
        decoder_head_mask: Optional[torch.Tensor] = None,
        # 可选的交叉注意力头部掩码，类型为 Optional[torch.Tensor]
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        # 可选的编码器输出列表，类型为 Optional[List[torch.FloatTensor]]
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        # 可选的过去关键值列表，类型为 Optional[List[torch.FloatTensor]]
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # 是否输出隐藏状态的标志，类型为 Optional[bool]
        output_hidden_states: Optional[bool] = None,
        # 是否输出注意力权重的标志，类型为 Optional[bool]
        output_attentions: Optional[bool] = None,
        # 是否使用缓存的标志，类型为 Optional[bool]
        use_cache: Optional[bool] = None,
        # 是否返回字典格式的结果，类型为 Optional[bool]
        return_dict: Optional[bool] = None,
    # 使用 @torch.no_grad() 装饰器，确保在生成过程中不计算梯度
    @torch.no_grad()
    # 定义一个方法 `generate`，用于生成过程
    def generate(
        # 过去的值作为输入，类型为 torch.Tensor
        self,
        past_values: torch.Tensor,
        # 过去的时间特征作为输入，类型为 torch.Tensor
        past_time_features: torch.Tensor,
        # 未来的时间特征作为输入，类型为 torch.Tensor
        future_time_features: torch.Tensor,
        # 可选的过去观察到的掩码，类型为 Optional[torch.Tensor]
        past_observed_mask: Optional[torch.Tensor] = None,
        # 可选的静态分类特征，类型为 Optional[torch.Tensor]
        static_categorical_features: Optional[torch.Tensor] = None,
        # 可选的静态实数特征，类型为 Optional[torch.Tensor]
        static_real_features: Optional[torch.Tensor] = None,
        # 是否输出注意力权重的标志，类型为 Optional[bool]
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态的标志，类型为 Optional[bool]
        output_hidden_states: Optional[bool] = None,
```