# `.\transformers\models\mega\modeling_mega.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 The Mega Authors 和 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”基础分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关特定语言的权限和限制
"""PyTorch MEGA model."""

# 导入所需的库
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入自定义的模块和类
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

# 从配置文件中导入 MegaConfig 类
from .configuration_mega import MegaConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "mnaylor/mega-base-wikitext"
_CONFIG_FOR_DOC = "MegaConfig"

# 预训练模型的存档列表
MEGA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "mnaylor/mega-base-wikitext",
    # 查看所有 Mega 模型 https://huggingface.co/models?filter=mega
]


class MegaEmbeddings(nn.Module):
    """
    Mega 的基本实现不包含令牌类型嵌入，因此这是 RoBERTa 嵌入的简化版本，可选择包含令牌类型
    """

    def __init__(self, config: MegaConfig):
        super().__init__()
        # 创建词嵌入层
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 是否使用令牌类型嵌入
        self.use_token_types = config.add_token_type_embeddings
        if self.use_token_types:
            # 创建令牌类型嵌入层
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
            # 注册一个缓冲区，允许在不传递可���令牌类型 ID 时进行模型跟踪
            # 更多信息请参考 transformers 问题 #5664
            self.register_buffer(
                "token_type_ids", torch.zeros(config.max_positions, dtype=torch.long).expand((1, -1)), persistent=False
            )

        self.padding_idx = config.pad_token_id
    # 定义一个前向传播函数，接受输入的input_ids、token_type_ids和inputs_embeds
    def forward(self, input_ids=None, token_type_ids=None, inputs_embeds=None):
        # 如果既没有提供input_ids也没有提供inputs_embeds，则抛出数值错误
        if (input_ids is None) and (inputs_embeds is None):
            raise ValueError("Must provide one of input_ids or inputs_embeds")
        # 如果提供了input_ids
        elif input_ids is not None:
            # 获取input_ids的形状
            input_shape = input_ids.size()
            # 获取input_ids所在设备
            device = input_ids.device

            # 如果只提供了IDs，则获取单词嵌入
            inputs_embeds = self.word_embeddings(input_ids)
        else:
            # 获取inputs_embeds的形状，去掉最后一个维度
            input_shape = inputs_embeds.size()[:-1]
            # 获取inputs_embeds所在设备
            device = inputs_embeds.device

        # 原始的Mega实现不包括token type embeddings，所以我们添加一个选项来使用它们；如果提供了embeddings并且没有提供token type IDs，则使用注册的缓冲区（有助于跟踪）
        if self.use_token_types:
            # 如果token_type_ids没有提供
            if token_type_ids is None:
                # 如果模型有token_type_ids属性
                if hasattr(self, "token_type_ids"):
                    # 获取缓冲区中的token_type_ids，并截取到与input_shape[1]相同的长度
                    buffered_token_type_ids = self.token_type_ids[:, : input_shape[1]]
                    # 将截取后的token_type_ids进行扩展，使其与input_shape相同
                    buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], input_shape[1])
                    token_type_ids = buffered_token_type_ids_expanded
                else:
                    # 如果模型没有token_type_ids属性，则创建一个全零的token_type_ids张量
                    token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

            # 获取token type embeddings
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            # 将token type embeddings添加到单词嵌入中
            embeddings = inputs_embeds + token_type_embeddings
        else:
            # 如果不使用token type embeddings，则直接使用inputs_embeds
            embeddings = inputs_embeds
        # 返回最终的嵌入结果
        return embeddings
class MegaSimpleRelativePositionalBias(nn.Module):
    """
    Simple relative positional embeddings copied from the Mega repo; renamed variables for better readability
    """

    def __init__(self, config: MegaConfig):
        # 初始化函数，接受一个 MegaConfig 类型的参数
        super().__init__()
        # 保存传入的配置参数
        self.config = config
        # 根据配置参数确定最大位置，如果 chunk_size 小于 0，则使用 max_positions，否则使用 chunk_size
        self.max_positions = self.config.max_positions if self.config.chunk_size < 0 else self.config.chunk_size
        # 创建一个可学习的参数，用于存储相对位置偏置
        self.rel_pos_bias = nn.Parameter(torch.Tensor(2 * config.max_positions - 1))

    def forward(self, seq_len):
        # 前向传播函数，接受序列长度作为参数
        if seq_len > self.max_positions:
            # 如果序列长度超过最大位置，抛出异常
            raise ValueError("Sequence length {} going beyond max length {}".format(seq_len, self.max_positions))

        # 计算偏置的范围
        bias = self.rel_pos_bias[(self.max_positions - seq_len) : (self.max_positions + seq_len - 1)]
        # 对偏置进行填充
        tile = F.pad(bias, (0, seq_len))
        # 对填充后的偏置进行复制
        tile = torch.tile(tile, (seq_len,))
        tile = tile[:-seq_len]
        # 调整复制后的偏置的形状
        tile = tile.view(seq_len, 3 * seq_len - 2)
        start = (2 * seq_len - 1) // 2
        end = tile.size(1) - start
        tile = tile[:, start:end]
        return tile


class MegaRotaryRelativePositionalBias(nn.Module):
    """
    Rotary relative bias for positional information; similar in concept to RoPE (i.e. RoFormer) but taken from the Mega
    repo due to differences in implementation.

    When initialized, produces a positional bias which ranges from position 0 to config.max_positions, but can
    extrapolate to longer sequences. Can be indexed according to input position IDs
    """

    def __init__(self, config: MegaConfig):
        # 初始化函数，接受一个 MegaConfig 类型的参数
        super().__init__()
        # 检查 hidden_size 是否为 2 的倍数
        if config.hidden_size % 2 != 0:
            raise RuntimeError("Rotary positional bias requires `hidden_size` to be a multiple of 2")
        # 保存传入的配置参数
        self.config = config
        # 保存共享表示大小
        self.embed_dim = config.shared_representation_size
        # 根据配置参数确��最大位置，如果 chunk_size 小于 0，则使用 max_positions，否则使用 chunk_size
        self.max_positions = self.config.max_positions if self.config.chunk_size < 0 else self.config.chunk_size
        # 获取正弦和余弦的嵌入
        self.sine, self.cosine = MegaRotaryRelativePositionalBias.get_sinusoid_embeddings(
            config.max_positions, self.embed_dim
        )
        # 创建可学习的参数 alpha 和 b_param，用于存储旋转偏置
        self.alpha = nn.Parameter(torch.Tensor(1, self.embed_dim))
        self.b_param = nn.Parameter(torch.Tensor(1, self.embed_dim))
        # 注册一个缓冲张量
        self.register_buffer("_float_tensor", torch.FloatTensor([0.0]))

    @staticmethod
    def get_sinusoid_embeddings(max_positions: int, embedding_dim: int):
        # 静态方法，用于获取正弦和余弦的嵌入
        half_dim = embedding_dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(max_positions, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        return torch.sin(emb), torch.cos(emb)
    # 定义一个旋转函数，用于对输入进行旋转操作
    def rotary(self, input):
        # 获取输入的序列长度和嵌入维度
        seq_len, embed_dim = input.size()
        # 将输入按照最后一个维度分成两个部分
        chunk_1, chunk_2 = torch.chunk(input, 2, dim=-1)
        # 如果正弦和余弦值为空或者序列长度大于正弦值的长度，则重新获取正弦和余弦值
        if self.sine is None or seq_len > self.sine.size(0):
            self.sine, self.cosine = MegaRotaryRelativePositionalBias.get_sinusoid_embeddings(seq_len, embed_dim)
            self.max_positions = seq_len
        # 将正弦和余弦值转换为指定类型的张量
        self.sine = self.sine.to(self._float_tensor)
        self.cosine = self.cosine.to(self._float_tensor)

        # 获取当前序列长度范围内的正弦和余弦值
        sin = self.sine[:seq_len]
        cos = self.cosine[:seq_len]
        # 返回旋转后的结果，将两部分按照一定规则组合在一起
        return torch.cat([chunk_1 * cos - chunk_2 * sin, chunk_2 * cos + chunk_1 * sin], dim=1)

    # 定义前向传播函数，用于计算旋转后的偏置
    def forward(self, seq_len):
        # 对 alpha 参数进行旋转
        rotary_alpha = self.rotary(self.alpha.expand(seq_len, self.embed_dim))
        # 对 beta 参数进行旋转
        rotary_beta = self.rotary(self.b_param.expand(seq_len, self.embed_dim))
        # 使用 einsum 函数计算旋转后的偏置
        bias = torch.einsum("mk,nk->mn", rotary_alpha, rotary_beta)
        # 返回计算得到的偏置
        return bias
class MegaDropout(nn.Module):
    """
    A unified class for standard dropout functionality and featurewise dropout.

    The original fairseq Mega repo used 2 classes for these, which included some unnecessary handling of training logic
    and an unused `inplace` option. The original implementation used torch.nn.functional instead of submodules, which
    is retained here as well.
    """

    def __init__(self, dropout_probability, is_featurewise=False):
        # 初始化 MegaDropout 类，设置 dropout 概率和是否为 featurewise dropout
        super().__init__()
        self.dropout_probability = dropout_probability
        self.is_featurewise = is_featurewise

    def forward(self, input, batch_first: bool = False):
        # 前向传播函数，根据 is_featurewise 参数选择不同的处理方式
        if self.is_featurewise:
            if batch_first:
                # 如果 batch_first 为 True，对输入进行维度转换和 dropout 操作
                # (batch_size X sequence_length X feature_dimension)
                # -> (batch_size X feature_dimension X sequence_length)
                # -> (batch_size X sequence_length X feature_dimension)
                return F.dropout2d(
                    input.transpose(-1, -2), p=self.dropout_probability, training=self.training
                ).transpose(-1, -2)
            else:
                if input.dim() != 3:
                    # 如果输入维度不为 3，抛出数值错误
                    raise ValueError(
                        "Feature dropout inputs must be exactly 3-dimensional if inputs are ordered [sequence length, batch size, hidden dimension]"
                    )
                # 如果 batch_first 为 False，对输入进行维度转换和 dropout 操作
                # (sequence_length X batch_size X feature_dimension)
                # -> (batch_size X feature_dimension X sequence_length)
                # -> (sequence_length X batch_size X feature_dimension)
                return F.dropout2d(input.permute(1, 2, 0), p=self.dropout_probability, training=self.training).permute(
                    2, 0, 1
                )
        else:
            # 如果不是 featurewise dropout，直接对输入进行 dropout 操作
            return F.dropout(input, p=self.dropout_probability, training=self.training)


class MegaRMSNorm(nn.Module):
    """
    RMSNorm used in Mega implementation. Differs from T5's RMSNorm by applying the weight prior to taking the square
    root (as opposed to after in T5)
    """

    def __init__(self, number_features, eps=1e-6, affine=True):
        # 初始化 MegaRMSNorm 类，设置特征数量、eps 和是否可仿射
        super().__init__()
        self.num_features = number_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.Tensor(self.num_features))
        else:
            self.register_parameter("weight", None)

    def forward(self, input):
        # 前向传播函数，计算均方和，根据权重对输入进行处理
        mean_square = torch.mean(torch.square(input), dim=-1, keepdim=True)
        if self.weight is not None:
            input = input * self.weight

        input * torch.rsqrt(mean_square + self.eps)
        return input


class MegaScaleNorm(nn.Module):
    """
    Scale normalization introduced in MEGA which is similar to RMSNorm, but uses a single parameter for scalar
    multiplication instead of a vector, and applies over a specified dimension
    """
    # 初始化 BatchNorm1d 类的实例对象，设置维度、epsilon值和是否进行仿射变换
    def __init__(self, dim, eps=1e-6, affine=True):
        # 调用父类的初始化方法
        super().__init__()
        # 保存传入的维度、epsilon值和是否进行仿射变换
        self.dim = dim
        self.eps = eps
        self.affine = affine
        # 如果需要进行仿射变换
        if affine:
            # 创建一个可学习的参数scalar，用于仿射变换
            self.scalar = nn.Parameter(torch.Tensor(1))
        else:
            # 如果不需要进行仿射变换，则参数scalar为空
            self.register_parameter("scalar", None)
    
    # 前向传播函数，对传入的input进行归一化处理
    def forward(self, input):
        # 计算input在指定维度上的平方取平均值
        mean_square = torch.mean(torch.square(input), dim=self.dim, keepdim=True)
        # 若仿射变换参数scalar不为空
        if self.scalar is not None:
            # 对input进行仿射变换
            input = self.scalar * input
    
        # 对input进行归一化处理，并乘以1/sqrt(平方平均值+epsilon)
        output = input * torch.rsqrt(mean_square + self.eps)
        return output
class MegaSequenceNorm(nn.Module):
    """
    A wrapper class for various layer normalization options used in Mega. Used to handle differences in expectations on
    input axis locations for different normalization methods.
    """

    def __init__(self, norm_type, embedding_dim, eps=1e-5, affine=True, export=False):
        # 初始化函数，接受规范化类型、嵌入维度、eps、是否要仿射变换、是否要导出的参数
        super().__init__()
        # 调用父类的初始化函数
        if norm_type == "layernorm":
            # 如果规范化类型为"layernorm"
            self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine=affine)
            # 创建 LayerNorm 对象
        elif norm_type == "scalenorm":
            # 如果规范化类型为"scalenorm"
            self.norm = MegaScaleNorm(dim=-1, eps=eps, affine=affine)
            # 创建 MegaScaleNorm 对象
        elif norm_type == "rmsnorm":
            # 如果规范化类型为"rmsnorm"
            self.norm = MegaRMSNorm(embedding_dim, eps=eps, affine=affine)
            # 创建 MegaRMSNorm 对象
        elif norm_type == "batchnorm":
            # 如果规范化类型为"batchnorm"
            self.norm = nn.BatchNorm1d(embedding_dim, eps=eps, affine=affine)
            # 创建 BatchNorm1d 对象
        elif norm_type == "syncbatchnorm":
            # 如果规范化类型为"syncbatchnorm"
            self.norm = nn.SyncBatchNorm(embedding_dim, eps=eps, affine=affine)
            # 创建 SyncBatchNorm 对象
        else:
            raise ValueError("Unknown norm type: {}".format(norm_type))
            # 抛出异常，未知的规范化类型

    def forward(self, input):
        # 前向传播函数
        if isinstance(self.norm, nn.modules.batchnorm._BatchNorm):
            # 如果规范化类型是 BatchNorm 类型
            if input.dim() != 3:
                # 如果输入数据维度不是3
                raise ValueError("BatchNorm inputs must be exactly 3-dimensional")
                # 抛出异常，BatchNorm 的输入必须是3维的
            input = input.permute(1, 2, 0)
            # 将输入数据维度进行转置
            input = self.norm(input)
            # 规范化输入数据
            return input.permute(2, 0, 1)
            # 将结果进行转置
        else:
            return self.norm(input)
            # 返回规范化后的结果


# add this layernorm class to ALL_LAYERNORM_LAYERS
ALL_LAYERNORM_LAYERS.append(MegaSequenceNorm)
# 将 MegaSequenceNorm 类添加到 ALL_LAYERNORM_LAYERS 列表中


class MegaMultiDimensionDampedEma(nn.Module):
    """
    Mega's Exponential Moving Average layer, largely left unmodified from the original repo with the exception of
    variable names and moving away from the stateful representation of incremental decoding state. See
    "https://arxiv.org/abs/2209.10655" for more details.
    """
    # 初始化函数，接受一个 MegaConfig 对象作为参数
    def __init__(self, config: MegaConfig):
        # 调用父类的初始化函数
        super().__init__()

        # 将传入的配置信息保存在对象中
        self.config = config

        # 初始化嵌入维度为隐藏大小
        self.embed_dim = config.hidden_size
        # 初始化嵌入维度为 EMA 投影大小
        self.ndim = config.ema_projection_size
        # 初始化双向标记
        self.bidirectional = config.bidirectional
        # 初始化截断大小
        self.truncation = config.truncation
        # 初始化尺度为嵌入维度的倒数的平方根
        self.scale = math.sqrt(1.0 / self.ndim)

        # 计算卷积核的维度
        kernel_dim = 2 * config.hidden_size if self.bidirectional else config.hidden_size
        # 重新命名阻尼因子和衰减因子，更能描述这些参数的作用
        self.damping_factor = nn.Parameter(torch.Tensor(kernel_dim, self.ndim, 1))
        self.decay_factor = nn.Parameter(torch.Tensor(kernel_dim, self.ndim, 1))
        # 重新命名 gamma 和 beta 分别为 EMA 扩展矩阵和核投影矩阵，避免与 HF 的重新命名混淆，并与论文中描述这些参数的行为保持一致
        self.ema_expansion_matrix = nn.Parameter(torch.Tensor(kernel_dim, self.ndim, 1))
        self.kernel_projection_matrix = nn.Parameter(torch.Tensor(kernel_dim, self.ndim))
        # 重新命名 omega 为残差权重，描述其作用
        self.residual_weight = nn.Parameter(torch.Tensor(config.hidden_size))
        self._kernel = None
        self._coeffs = None

    # 计算 EMA 系数
    def _compute_ema_coefficients(self):
        self._coeffs = None
        # 将 alpha 和 delta 参数（kernel_dim x EMA projection size x 1）转换为 [0, 1] 区间，使用 sigmoid 函数
        damping_factor = torch.sigmoid(self.damping_factor)
        decay_factor = torch.sigmoid(self.decay_factor)
        previous_timestep_weight = 1.0 - damping_factor * decay_factor
        return damping_factor, previous_timestep_weight

    # 计算用于高效阻尼的 EMA 核
    def _compute_efficient_ema_kernel(self, length: int):
        # 计算用于通过 FFT 卷积应用的高效阻尼的 EMA 核
        self._kernel = None
        # p 和 q 的形状为 (kernel_dim x ema_projection_size x 1)
        damping_factor, previous_timestep_weight = self._compute_ema_coefficients()
        # 将核扩展为 (kernel_dim x ema_projection_size x sequence_length)，并将 q 乘以顺序整数直到序列长度
        vander = torch.arange(length).to(damping_factor).view(1, 1, length) * torch.log(previous_timestep_weight)
        kernel = (damping_factor * self.ema_expansion_matrix) * torch.exp(vander)
        # (kernel_dim x ema_projection_size x sequence_length) -> (kernel_dim, sequence_length)
        return torch.einsum("dnl,dn->dl", kernel, self.kernel_projection_matrix * self.scale)

    # 获取 EMA 系数
    def get_ema_coefficients(self):
        if self.training:
            return self._compute_ema_coefficients()
        else:
            if self._coeffs is None:
                self._coeffs = self._compute_ema_coefficients()
            return self._coeffs
    # 定义一个获取指数移动平均（EMA）卷积核的方法，接受长度参数
    def get_ema_kernel(self, length: int):
        # 计算卷积核大小，如果截断大小不为None，则取截断大小和长度的最小值
        kernel_size = length if self.truncation is None else min(self.truncation, length)
        # 如果处于训练状态
        if self.training:
            # 调用方法计算高效的EMA卷积核
            return self._compute_efficient_ema_kernel(kernel_size)
        else:
            # 如果卷积核为空或者卷积核最后一个维度小于kernel_size
            if self._kernel is None or self._kernel.size(-1) < kernel_size:
                # 计算高效的EMA卷积核
                self._kernel = self._compute_efficient_ema_kernel(kernel_size)
            # 返回卷积核中前kernel_size个元素
            return self._kernel[..., :kernel_size]

    # 定义一个使用FFT卷积进行重复计算EMA的包装方法
    def fft_convolution(self, inputs, kernel, length):
        # 对输入进行FFT变换，n=2*length表示双倍长度
        inputs_fft = torch.fft.rfft(inputs.float(), n=2 * length)
        # 对卷积核进行FFT变换，n=2*length表示双倍长度
        kernel_fft = torch.fft.rfft(kernel.float(), n=2 * length)
        # 使用FFT卷积计算卷积序列
        convolved_sequence = torch.fft.irfft(inputs_fft * kernel_fft, n=2 * length)
        # 返回卷积序列
        return convolved_sequence
    # 对指数移动平均进行一步计算，考虑输入、长度和之前状态
    def ema_step(self, inputs, length, past_state=None):
        # 如果长度为1，直接调用一步计算函数返回结果
        if length == 1:
            return self.one_ema_step(inputs, past_state=past_state)

        # 计算阻尼因子和上一个时间步的权重
        damping_factor, previous_timestep_weight = self.get_ema_coefficients()
        
        # 创建Vandermonde矩阵
        vander = torch.arange(length + 1).to(damping_factor).view(1, 1, length + 1) * torch.log(previous_timestep_weight)
        vander = torch.exp(vander)
        
        # 如果有过去状态
        if past_state is not None:
            # 计算过去EMA投影
            past_ema_proj = vander[:, :, 1:] * (self.kernel_projection_matrix * self.scale).unsqueeze(-1)
            past_ema_state = torch.einsum("bdn,dnl->bdl", past_state, past_ema_proj)
            past_vandermonde = vander[:, :, -1] * past_state
        else:
            past_ema_state = None
            past_vandermonde = None

        # 更新Vandermonde矩阵
        vander = vander[:, :, :-1]
        
        # 计算卷积核
        kernel = (damping_factor * self.ema_expansion_matrix) * vander
        kernel_proj = torch.einsum("dnl,dn->dl", kernel, self.kernel_projection_matrix * self.scale)

        # 进行FFT卷积操作
        ema_output = self.fft_convolution(inputs, kernel_proj, length=length)[..., 0:length]
        ema_output = ema_output.type_as(inputs)
        
        # 若有过去EMA状态，加上该状态
        if past_ema_state is not None:
            ema_output = ema_output + past_ema_state

        # 更新隐藏状态
        updated_hidden_state = torch.einsum("bdl,dnl->bdn", inputs, torch.flip(kernel, dims=[2]))
        
        # 若有过去Vandermonde矩阵，将其加到隐藏状态中
        if past_vandermonde is not None:
            updated_hidden_state = updated_hidden_state + past_vandermonde
        
        # 返回包含EMA输出和更新后隐藏状态的元组
        return ema_output.permute(2, 0, 1), updated_hidden_state

    # 单步计算指数移动平均
    def one_ema_step(self, inputs, past_state=None):
        damping_factor, previous_timestep_weight = self.get_ema_coefficients()
        
        # 更新状态
        updated_state = (damping_factor * self.ema_expansion_matrix).squeeze(-1) * inputs
        
        if past_state is not None:
            updated_state = updated_state + previous_timestep_weight.squeeze(-1) * past_state
        
        # 计算输出
        out = torch.einsum("bdn,dn->bd", updated_state, self.kernel_projection_matrix * self.scale)
        
        # 返回包含输出和更新后状态的元组
        return out.unsqueeze(0), updated_state
    # 定义神经网络的前向传播方法
        def forward(
            # 神经网络的输入张量
            self,
            inputs,
            # 可选的注意力掩码，用于遮蔽无效的输入
            attention_mask: Optional[torch.Tensor] = None,
            # 可选的前一状态，用于维持 RNN 或 Transformer 等结构的状态
            prev_state: Optional[torch.Tensor] = None,
            # 是否使用缓存，如果为 True，可能会使用缓存来加速推理
            use_cache: bool = False,
# 这个类实现了一种 Mega Gated Cross Attention 机制,用于编码器-解码器模型中。
# 它是根据 Mega 论文提出的机制进行修改实现的,主要改动有变量名称、删除了一些不必要的参数,以及对增量式解码器状态的表示方式。
class MegaGatedCrossAttention(nn.Module):
    # 初始化方法,接受一个 MegaConfig 配置对象
    def __init__(self, config: MegaConfig):
        super().__init__()
        # 保存配置对象
        self.config = config
        # 根据配置创建激活函数
        self.activation = ACT2FN[self.config.activation]
        # 保存注意力机制的激活函数
        self.attention_activation = self.config.attention_activation
        # 如果使用 softmax 注意力,则计算缩放因子
        self.scaling = self.config.shared_representation_size**-0.5 if self.attention_activation == "softmax" else None

        # 创建特征级别dropout
        self.dropout = MegaDropout(self.config.dropout_prob, is_featurewise=self.config.use_feature_dropout)
        # 创建隐藏层dropout
        self.hidden_dropout = MegaDropout(
            self.config.hidden_dropout_prob, is_featurewise=self.config.use_feature_dropout
        )
        # 创建注意力权重dropout
        self.attention_dropout = MegaDropout(self.config.attention_probs_dropout_prob, is_featurewise=False)

        # 是否在注意力之前进行归一化
        self.prenorm = self.config.normalize_before_mega
        # 创建序列层归一化层
        self.norm = MegaSequenceNorm(
            self.config.normalization_type, self.config.hidden_size, affine=self.config.norm_affine
        )

        # 创建键、值、查询映射层
        self.k_proj = nn.Linear(self.config.hidden_size, self.config.shared_representation_size)
        self.v_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.q_proj = nn.Linear(
            self.config.hidden_size, 2 * self.config.hidden_size + self.config.shared_representation_size
        )
        # 创建隐藏层映射层
        self.h_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)

        # 根据配置创建相对位置偏置
        if self.config.relative_positional_bias == "simple":
            self.rel_pos_bias = MegaSimpleRelativePositionalBias(config)
        elif self.config.relative_positional_bias == "rotary":
            self.rel_pos_bias = MegaRotaryRelativePositionalBias(config)
        else:
            raise ValueError("unknown relative position bias: {}".format(self.config.relative_positional_bias))

        # 创建 softmax 层
        self.softmax = nn.Softmax(dim=-1)
    # 计算注意力权重的方法
    def element_attention(self, query, key, key_padding_mask, pidx):
        # 获取 key 张量的 batch_size、源序列长度、特征维度
        bsz, src_len, _ = key.size()
        # 获取 query 张量的目标序列长度，如果提供了位置索引 pidx，则长度为 pidx+1
        tgt_len = query.size(1) if pidx is None else pidx + 1
        # 如果提供了 key_padding_mask，则计算有效序列长度
        if key_padding_mask is not None:
            # (batch_size X source_sequence_length) --> (batch_size X 1 X 1)
            lengths = key_padding_mask.sum(dim=-1).view(bsz, 1, 1)
        else:
            # 否则，有效序列长度即为源序列长度
            lengths = src_len
    
        # 计算相对位置偏移
        # (target_sequence_length X source_sequence_length)
        bias = self.rel_pos_bias(max(tgt_len, src_len))[:, :src_len]
        # 如果提供了位置索引，则只取对应的偏移
        if pidx is not None:
            if query.size(1) != 1:
                raise ValueError("Position offset provided with queries longer than 1 token")
            # source_sequence_length
            bias = bias[pidx]
        else:
            # (target_sequence_length X source_sequence_length)
            bias = bias[:tgt_len]
    
        # 计算注意力权重
        # (batch_size X target_sequence_length X source_sequence_length)
        qk = torch.bmm(query, key.transpose(1, 2)) / lengths + bias
    
        # 应用激活函数得到最终的注意力权重
        attn_weights = ACT2FN[self.attention_activation](qk).type_as(qk)
    
        # 如果提供了 key_padding_mask，则将无效位置的注意力权重置为 0
        if key_padding_mask is not None:
            attn_weights = attn_weights * key_padding_mask.unsqueeze(1)
    
        return attn_weights
    
    # 计算 Softmax 注意力权重的方法
    def softmax_attention(self, query, key, key_padding_mask, pidx):
        # 获取 key 张量的 batch_size、源序列长度、特征维度
        bsz, src_len, _ = key.size()
        # 获取 query 张量的目标序列长度，如果提供了位置索引 pidx，则长度为 pidx+1
        tgt_len = query.size(1) if pidx is None else pidx + 1
    
        # 计算相对位置偏移
        # (target_sequence_length X source_sequence_length)
        bias = self.rel_pos_bias(max(tgt_len, src_len))[:, :src_len]
        # 如果提供了位置索引，则只取对应的偏移
        if pidx is not None:
            if query.size(1) != 1:
                raise ValueError("Position offset provided with queries longer than 1 token")
            # source_sequence_length
            bias = bias[pidx]
        else:
            # (target_sequence_length X source_sequence_length)
            bias = bias[:tgt_len]
    
        # 对 query 进行缩放
        query = query * self.scaling
        # 计算注意力权重
        # (batch_size X target_sequence_length X source_sequence_length)
        qk = torch.bmm(query, key.transpose(1, 2)) + bias
    
        # 如果提供了 key_padding_mask，则将无效位置的注意力权重置为负无穷
        if key_padding_mask is not None:
            qk = qk.masked_fill((1 - key_padding_mask).unsqueeze(1).to(torch.bool), float("-inf"))
    
        # 应用 Softmax 函数得到最终的注意力权重
        attn_weights = self.softmax(qk).type_as(qk)
        return attn_weights
    
    # 注意力计算的前向传播方法
    def forward(
        self,
        query,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        # 根据输入参数调用 element_attention 或 softmax_attention 计算注意力权重
        if self.attention_type == 'element':
            attn_weights = self.element_attention(query, key, key_padding_mask, pidx=None)
        elif self.attention_type == 'softmax':
            attn_weights = self.softmax_attention(query, key, key_padding_mask, pidx=None)
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")
    
        # 根据注意力权重和值张量计算输出
        # ...
class MegaMovingAverageGatedAttention(nn.Module):
    """
    Pure PyTorch implementation of Mega block; see https://arxiv.org/abs/2209.10655 and original fairseq implementation
    at https://github.com/facebookresearch/mega (copyright Meta Research, licensed under MIT License)

    Differences from original implementation include hidden state refactor and fixed inconsistency with additive /
    multiplicative attention masks
    """

    def __init__(self, config: MegaConfig):
        # 初始化 MegaMovingAverageGatedAttention 类
        super().__init__()
        # 保存传入的配置信息
        self.config = config
        # 获取激活函数
        self.activation = ACT2FN[self.config.activation]
        # 计算缩放值（若使用 softmax 激活，则取 shared_representation_size 的倒数，否则为 None）
        self.scaling = (
            self.config.shared_representation_size**-0.5 if self.config.attention_activation == "softmax" else None
        )
        # 创建 dropout 层
        self.dropout = MegaDropout(self.config.dropout_prob, is_featurewise=self.config.use_feature_dropout)
        self.hidden_dropout = MegaDropout(
            self.config.hidden_dropout_prob, is_featurewise=self.config.use_feature_dropout
        )
        # 创建 attention dropout 层
        self.attention_dropout = MegaDropout(self.config.attention_probs_dropout_prob, is_featurewise=False)

        # 创建归一化层
        self.norm = MegaSequenceNorm(
            self.config.normalization_type, self.config.hidden_size, affine=self.config.norm_affine
        )
        # 创建 EMA（Exponential Moving Average）门控层
        self.ema_gate = MegaMultiDimensionDampedEma(config)

        # 三个线性变换层
        self.v_proj = nn.Linear(self.config.hidden_size, self.config.intermediate_size)
        self.mx_proj = nn.Linear(
            self.config.hidden_size,
            self.config.shared_representation_size + self.config.intermediate_size + 2 * self.config.hidden_size,
        )
        self.h_proj = nn.Linear(self.config.intermediate_size, self.config.hidden_size)

        # 生成参数
        self.qk_weight = nn.Parameter(torch.Tensor(2, self.config.shared_representation_size))
        self.qk_bias = nn.Parameter(torch.Tensor(2, self.config.shared_representation_size))

        # 根据配置选则不同的相对位置偏置模块
        if self.config.relative_positional_bias == "simple":
            self.rel_pos_bias = MegaSimpleRelativePositionalBias(config)
        elif self.config.relative_positional_bias == "rotary":
            self.rel_pos_bias = MegaRotaryRelativePositionalBias(config)
        else:
            raise ValueError(f"Unknown relative positional bias: {self.config.relative_positional_bias}")

        # 创建 softmax 激活函数
        self.softmax = nn.Softmax(dim=-1)
        # 根据配置选则不同的注意力计算方式
        self.attention_function = (
            self.softmax_attention if self.config.attention_activation == "softmax" else self.element_attention
        )
    def element_attention(self, query, key, padding_mask, causal_mask):
        """
        Apply element-wise attention via relu^2 or laplace. Same as original implementation but with standardized
        causal attention mask. Expects the Hugging Face standard attention mask paradigm: 1 for not masked, and 0 for
        masked.
        """
        # 获取序列长度
        seq_len = key.size(2)
        # 如果存在padding_mask，则计算每个样本的有效长度
        if padding_mask is not None:
            # (batch_size X number of chunks X 1)
            lengths = padding_mask.sum(-1, keepdim=True)
            # (batch_size X number of chunks X 1 X 1)
            lengths = lengths.clamp(min=1.0).unsqueeze(-1)
        else:
            lengths = seq_len

        # 如果存在causal_mask，则重新计算长度
        if causal_mask is not None:
            lengths = causal_mask.sum(dim=-1, keepdim=True)

        # 计算相对位置偏置
        # (sequence_length X sequence_length)
        bias = self.rel_pos_bias(seq_len)
        
        # 如果序列长度不等于查询向量长度，抛出错误
        if seq_len != query.size(2):
            if query.size(2) != 1:
                raise ValueError("Size mismatch between Q and K in element attention")
            # (1 X sequence_length)
            bias = bias[-1:]

        # 计算qk，即query和key的点积，除以指定长度，并加上偏置
        # (batch_size X number of chunks X sequence_length X sequence_length)
        qk = torch.matmul(query, key.transpose(2, 3)) / lengths + bias

        # 根据config中的attention_activation选择激活函数，并将qk转为相应类型
        attn_weights = ACT2FN[self.config.attention_activation](qk).type_as(qk)

        # 如果存在padding_mask，则将注意力权重乘以padding_mask
        if padding_mask is not None:
            attn_weights = attn_weights * padding_mask.unsqueeze(2)

        # 如果存在causal_mask，则将注意力权重乘以causal_mask
        if causal_mask is not None:
            attn_weights = attn_weights * causal_mask

        return attn_weights
    # 实现标准的softmax自注意力机制，与Transformer论文中的描述相符
    def softmax_attention(self, query, key, padding_mask, causal_mask):
        # 获取序列长度
        seq_len = key.size(2)
        # 为序列中每个位置计算相对位置编码的偏置，大小为sequence_length x sequence_length
        bias = self.rel_pos_bias(seq_len)
        # 如果Q和K的长度不匹配
        if seq_len != query.size(2):
            # 如果query的长度不是1，抛出异常
            if query.size(2) != 1:
                raise ValueError("Size mismatch between Q and K in softmax attention")
            # 获取bias中的最后一个序列，大小为1 x sequence_length
            bias = bias[-1:]

        # 进行缩放
        query = query * self.scaling

        # 计算qk相乘的结果，并加上偏置，大小为batch_size x number of chunks x chunk_size x chunk_size或batch_size x 1 x sequence_length x sequence_length
        qk = torch.matmul(query, key.transpose(2, 3)) + bias

        # 对causal_mask进行应用，假设为1或0表示未屏蔽/屏蔽
        # 加性操作，但转换为0/-inf（在Mega源代码中没有明确说明）
        if causal_mask is not None:
            # 创建与causal_mask相同大小的全0张量，并设置数据类型与qk相同
            additive_causal_mask = torch.zeros_like(causal_mask, dtype=qk.dtype)
            # 将未屏蔽部分的值设置为-inf
            additive_causal_mask = additive_causal_mask.masked_fill((1 - causal_mask).bool(), float("-inf"))
            qk = qk + additive_causal_mask

        if padding_mask is not None:
            # 在padding_mask中，1表示未屏蔽的标记，0表示屏蔽的标记
            # 将屏蔽的标记替换为-inf，以便softmax忽略它们
            # 需要将padding_mask取反，以匹配Mega原始代码的行为
            padding_mask = 1 - padding_mask
            # 取padding_mask的所有值为逻辑与，判断哪些位置需要进行屏蔽
            padding_mask_all = padding_mask.all(dim=-1, keepdim=True)
            # 将padding_mask与~padding_mask_all进行逻辑与操作，得到真正需要屏蔽的位置
            padding_mask = torch.logical_and(padding_mask, ~padding_mask_all)
            # 在qk中将标记为True的位置替换为-inf
            qk = qk.masked_fill(padding_mask.unsqueeze(2).to(torch.bool), float("-inf"))

        # 对qk进行softmax操作，并使其类型与qk相同
        attn_weights = self.softmax(qk).type_as(qk)
        return attn_weights

    def forward(
        self,
        input,
        padding_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions=False,
        use_cache=False,
class MegaNormalizedFeedForwardNetwork(nn.Module):
    """
    Normalized feed-forward network used in Mega blocks. Left as-is from original Mega repo aside from retrieving args
    from Hugging Face config
    """

    def __init__(self, config: MegaConfig):
        super().__init__()

        self.config = config
        self.hidden_dim = config.nffn_hidden_size
        self.act_fn = config.activation
        self.activation = ACT2FN[config.activation]

        self.dropout = MegaDropout(self.config.dropout_prob, is_featurewise=self.config.use_feature_dropout)
        self.hidden_dropout = MegaDropout(
            self.config.nffn_activation_dropout_prob, is_featurewise=self.config.use_feature_dropout
        )

        self.prenorm = self.config.normalize_before_ffn
        self.norm = MegaSequenceNorm(
            self.config.normalization_type, self.config.hidden_size, affine=self.config.norm_affine
        )

        self.fc1 = nn.Linear(self.config.hidden_size, self.config.nffn_hidden_size)
        self.fc2 = nn.Linear(self.config.nffn_hidden_size, self.config.hidden_size)

    def forward(self, inputs):
        residual = inputs

        if self.prenorm:
            inputs = self.norm(inputs)

        hidden = self.activation(self.fc1(inputs))
        hidden = self.hidden_dropout(hidden)
        output = self.fc2(hidden)
        output = self.dropout(output)
        output = output + residual

        if not self.prenorm:
            output = self.norm(output)

        return output


class MegaBlock(nn.Module):
    def __init__(self, config: MegaConfig):
        super().__init__()
        self.seq_len_dim = 1
        self.mega_layer = MegaMovingAverageGatedAttention(config)
        self.nffn = MegaNormalizedFeedForwardNetwork(config) if config.use_normalized_ffn else None
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.cross_attn = MegaGatedCrossAttention(config)
        else:
            self.cross_attn = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        causal_mask: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[torch.FloatTensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: bool = False,
# copied from transformers.models.roberta.modeling_roberta.RobertaPooler with Roberta->Mega
class MegaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
    # 定义一个前向传播函数，接受隐藏状态张量并返回一个张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过简单地选择与第一个标记对应的隐藏状态来"池化"模型
        first_token_tensor = hidden_states[:, 0]
        # 将第一个标记的隐藏状态传入全连接层
        pooled_output = self.dense(first_token_tensor)
        # 将全连接层输出应用激活函数
        pooled_output = self.activation(pooled_output)
        # 返回池化输出
        return pooled_output
# 定义一个名为 MegaPreTrainedModel 的类，继承自 PreTrainedModel 类
class MegaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    # 设置配置类为 MegaConfig
    config_class = MegaConfig
    # 设置基础模型前缀为 "mega"
    base_model_prefix = "mega"
    # 设置是否支持梯度检查点为 False
    supports_gradient_checkpointing = False
    # 设置不需要拆分的模块列表，这些模块不会被拆分
    _no_split_modules = ["MegaMovingAverageGatedAttention"]
    def _init_weights(self, module):
        """初始化权重"""
        如果 module 是 MegaMultiDimensionDampedEma 类型的实例：
            使用 torch.no_grad() 上下文管理器初始化 delta 和 alpha
            nn.init.normal_(module.damping_factor, mean=0.0, std=self.config.ema_delta_alpha_range)
            nn.init.normal_(module.decay_factor, mean=0.0, std=self.config.ema_delta_alpha_range)
            创建值为 [1, -1, 1, -1, ...] 的张量，如果 config.ema_projection_size 大于 1，则进行修改以使其更稳定
            val = torch.ones(self.config.ema_projection_size, 1)
            if self.config.ema_projection_size > 1:
                idx = torch.tensor(list(range(1, self.config.ema_projection_size, 2)))
                val.index_fill_(0, idx, -1.0)
            使用正态分布初始化 module.ema_expansion_matrix
            module.ema_expansion_matrix.normal_(mean=0.0, std=self.config.ema_beta_range).add_(val)
            使用正态分布初始化 gamma 和 omega
            nn.init.normal_(module.kernel_projection_matrix, mean=0.0, std=self.config.ema_gamma_omega_range)
            nn.init.normal_(module.residual_weight, mean=0.0, std=self.config.ema_gamma_omega_range)
            
        如果 module 是 MegaSimpleRelativePositionalBias 类型的实例：
            使用正态分布初始化 module.rel_pos_bias
            
        如果 module 是 MegaRotaryRelativePositionalBias 类型的实例：
            使用正态分布初始化 alpha 和 b_param
            
        如果 module 是 MegaScaleNorm 类型的实例：
            如果 norm_affine 为 True，则将 module.scalar 初始化为 1.0
            
        如果 module 是 MegaRMSNorm 类型的实例：
            如果 norm_affine 为 True，则将 module.weight 初始化为 1.0
            
        如果 module 是 MegaMovingAverageGatedAttention 类型的实例：
            对于 linear layers，使用正态分布初始化 module.qk_weight，并将 module.qk_bias 初始化为 0.0
            
        如果 module 是 nn.Linear 类型的实例：
            初始化整个网络中的所有线性层
            使用正态分布初始化 module.weight，如果存在偏置，则将其初始化为 0
            
        如果 module 是 nn.Embedding 类型的实例：
            使用正态分布初始化 module.weight，如果存在 padding_idx，则将其对应的权重初始化为 0
            
        如果 module 是 nn.LayerNorm 类型的实例：
            将 module.bias 初始化为 0
            将 module.weight 初始化为 1.0
# Mega 模型继承自 PreTrainedModel，查看其父类的文档了解库实现的通用方法（例如下载或保存、调整输入嵌入、剪枝等等）
# 这个模型同时也是一个 PyTorch 的 torch.nn.Module 子类
# 可以像使用普通的 PyTorch 模块一样使用它，并参考 PyTorch 文档了解一般用法和行为
# 参数：
#    config 传入 MegaConfig 类，包含模型的所有参数
#    初始化时，仅加载与模型关联的权重，不会加载模型本身的权重
#    可以使用~PreTrainedModel.from_pretrained 方法来加载模型的权重
def Mega(
    config
):
    ...


注释：这段代码是说明 Mega 模型的定义和使用方法。首先指出了这个模型继承自 PreTrainedModel，可以使用该父类的通用方法；其次提到了这个模型也是一个 PyTorch 的 torch.nn.Module 子类，可以像使用普通的 PyTorch 模块一样使用它；最后给出了初始化参数和加载权重的方法。
    # 这些是输入数据的相关说明
    Args:
        # input_ids 是输入序列中每个词汇在词典中的索引，是一个长度为 {0} 的张量
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.
    
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
    
            [What are input IDs?](../glossary#input-ids)
        # attention_mask 是一个长度为 {0} 的浮点张量，用于标记哪些 token 需要注意力计算
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
    
            [What are attention masks?](../glossary#attention-mask)
        # token_type_ids 是一个长度为 {0} 的整型张量，用于区分输入序列的不同部分
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,1]`:
    
            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.
            This parameter can only be used when the model is initialized with `add_token_type_embeddings` parameter
            set to `True`. All the value in this tensor should be always < config.type_vocab_size.
    
            [What are token type IDs?](../glossary#token-type-ids)
        # inputs_embeds 是一个大小为 ({0}, hidden_size) 的浮点张量，可以直接传入嵌入向量
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        # output_attentions 控制是否输出各注意力层的注意力张量
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        # output_hidden_states 控制是否输出所有层的隐藏状态
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        # return_dict 控制是否返回 ModelOutput 对象
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# 导入必要的库
from transformers import MegaPreTrainedModel, MegaConfig, MegaEmbeddings, MegaBlock, MegaPooler, BaseModelOutputWithPoolingAndCrossAttentions
import torch
from typing import Optional, List

# 定义 MEGA 模型类，作为一个没有特定头部输出原始隐藏状态的模型
@add_start_docstrings(
    "The bare MEGA Model transformer outputting raw hidden-states without any specific head on top.",
    MEGA_START_DOCSTRING,
)
class MegaModel(MegaPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added after self-attention, following the architecture described in *Mega: Moving Average
    Equipped Gated Attention*_ by Xuezhe Ma, Chunting Zhou, Xiang Kong, Junxian He, Liangke Gui, Graham Neubig,
    Jonathan May, and Luke Zettlemoyer

    To behave as a decoder the model needs to be initialized with the `is_decoder` argument of the configuration set to
    `True` and `bidirectional` set to `False`. To be used in a Seq2Seq model, the model needs to initialized with both
    `is_decoder=True` and `bidirectional=False` argument as well as `add_cross_attention` set to `True`; an
    `encoder_hidden_states` is then expected as an input to the forward pass.

    .. _*Mega: Moving Average Equipped Gated Attention*: https://arxiv.org/abs/2209.10655

    """

    # 初始化方法
    def __init__(self, config: MegaConfig, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # 初始化嵌入层
        self.embedding_layer = MegaEmbeddings(config)
        # 创建多个 MEGA 块
        self.layers = nn.ModuleList([MegaBlock(config) for _ in range(config.num_hidden_layers)])

        # 添加池化层
        self.pooler = MegaPooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.embedding_layer.word_embeddings

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.embedding_layer.word_embeddings = value

    # 前馈传播方法
    @add_start_docstrings_to_model_forward(MEGA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    )

# 定义带 `language modeling` 头部的 MEGA 模型类，用于 CLM 微调
@add_start_docstrings(
    """MEGA Model with a `language modeling` head on top for CLM fine-tuning.""", MEGA_START_DOCSTRING
)
class MegaForCausalLM(MegaPreTrainedModel):
    # 定义 tied weights 的键
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config: MegaConfig):
        # 调用父类的初始化方法，传入配置参数
        super().__init__(config)

        # 如果配置参数中不是解码器，发出警告提示
        if not config.is_decoder:
            logger.warning("If you want to use `MegaForCausalLM` as a standalone, add `is_decoder=True.`")

        # 初始化 MegaModel，不添加池化层
        self.mega = MegaModel(config, add_pooling_layer=False)

        # 如果配置参数中包含 LM 隐藏层的密集层
        if config.add_lm_hidden_dense_layer:
            # 初始化隐藏层的线性变换层
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            # 初始化隐藏层的激活函数为双曲正切函数
            self.hidden_activation = nn.Tanh()
        else:
            # 如果配置参数中不包含隐藏层的密集层，则将其置为空
            self.dense = None
            self.hidden_activation = None

        # 初始化 LM 的线性输出层
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 前向传播方法
    @add_start_docstrings_to_model_forward(MEGA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 准备生成输入的方法
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # 如果没有提供注意力掩码，则创建一个全为 1 的注意力掩码
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 如果使用了过去的键值对，则截取输入的编码器输入 ID
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # 返回输入字典
        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}

    # 重新排序缓存方法
    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                # 使用 beam_idx 重新排序每一层的过去键值对
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
# 带有"语言建模"头的 MEGA 模型
@add_start_docstrings("""MEGA Model with a `language modeling` head on top.""", MEGA_START_DOCSTRING)
class MegaForMaskedLM(MegaPreTrainedModel):
    # 权重绑定的键列表
    _tied_weights_keys = ["mlm_head.weight"]

    def __init__(self, config: MegaConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 如果模型被设置为解码器，给出警告
        if config.is_decoder:
            logger.warning(
                "If you want to use `MegaForMaskedLM`, set `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 初始化 MEGA 模型，不添加池化层
        self.mega = MegaModel(config, add_pooling_layer=False)
        
        # 如果需要添加隐藏层
        if config.add_lm_hidden_dense_layer:
            # 添加线性层
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            # 添加激活函数
            self.hidden_activation = nn.Tanh()
        # 否则不添加隐藏层
        else:
            self.dense = None
            self.hidden_activation = None
        
        # 添加用于语言模型的线性层
        self.mlm_head = nn.Linear(config.hidden_size, config.vocab_size)
        # 添加 dropout 层
        self.dropout = nn.Dropout(config.dropout_prob)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入层
    def get_output_embeddings(self):
        return self.mlm_head

    # 设置输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.mlm_head = new_embeddings

    # 前向传播
    @add_start_docstrings_to_model_forward(MEGA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<mask>",
        expected_output="' Paris'",
        expected_loss=0.1,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        # 设置是否返回字典形式的输出，默认与模型配置中的设置一致
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给 Mega 模型，并返回输出
        outputs = self.mega(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取模型输出的序列部分
        sequence_output = outputs[0]
        # 如果存在密集连接层，则对序列输出进行密集连接和激活函数处理
        if self.dense is not None:
            sequence_output = self.dense(sequence_output)
            sequence_output = self.hidden_activation(sequence_output)
        # 通过 Masked LM 头部预测下一个单词的概率分布
        prediction_scores = self.mlm_head(sequence_output)

        # 初始化 masked_lm_loss 为空
        masked_lm_loss = None
        # 如果存在标签，则计算 masked language modeling 损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # 计算交叉熵损失
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果不返回字典形式的输出，则返回输出元组
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 如果返回字典形式的输出，则构造 MaskedLMOutput 对象
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 带有顶部的序列分类/回归头的 MEGA 模型转换器（在池化输出的顶部有一个线性层），例如用于 GLUE 任务
@add_start_docstrings(
    """
    MEGA Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    MEGA_START_DOCSTRING,
)
class MegaForSequenceClassification(MegaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # 初始化 MEGA 模型，不添加池化层
        self.mega = MegaModel(config, add_pooling_layer=False)
        # 初始化分类头
        self.classifier = MegaClassificationHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(MEGA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```  
    # 该函数是用于序列分类任务的前向传播过程
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        # 如果没有指定 return_dict，则使用配置文件中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 将输入数据传入 self.mega 模型得到序列输出
        outputs = self.mega(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        
        # 将序列输出传入分类器得到分类logits
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            # 根据问题类型确定损失函数
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        # 根据 return_dict 决定返回值的格式
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
    
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 这个类是一个预训练的 MEGA 模型,具有多选分类头部(pooled output 上的线性层和 softmax)
@add_start_docstrings(
    """
    MEGA Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    MEGA_START_DOCSTRING,
)
class MegaForMultipleChoice(MegaPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建 MEGA 模型
        self.mega = MegaModel(config)
        # 创建dropout层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建分类器,输入为 hidden_size,输出为 1
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 添加输入文档字符串,定义输入格式
    @add_start_docstrings_to_model_forward(MEGA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    # 添加输出文档字符串,定义输出格式
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        # 输入ID序列,可选
        input_ids: Optional[torch.LongTensor] = None,
        # 输入的token类型ID,可选
        token_type_ids: Optional[torch.LongTensor] = None,
        # 输入的注意力掩码,可选
        attention_mask: Optional[torch.FloatTensor] = None,
        # 标签,可选
        labels: Optional[torch.LongTensor] = None,
        # 输入的嵌入,可选
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 是否输出注意力,可选
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态,可选
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典,可选
        return_dict: Optional[bool] = None,
    # 定义多项选择模型的前向传播函数
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            用于计算多项选择分类损失的标签。索引应该在`[0, ..., num_choices-1]`之间，其中`num_choices`是输入张量第二维的大小。 (参见上文的`input_ids`)
        """
        # 如果`return_dict`不为`None`，则将其赋给`return_dict`；否则使用模型配置中的`use_return_dict`
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果`input_ids`不为`None`，则获取第一维的大小作为`num_choices`，否则使用`inputs_embeds`的第二维大小
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 将输入`input_ids`展平为二维张量，如果`input_ids`为`None`，则返回`None`
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        # 将输入`token_type_ids`展平为二维张量，如果`token_type_ids`为`None`，则返回`None`
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # 将输入`attention_mask`展平为二维张量，如果`attention_mask`为`None`，则返回`None`
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # 将输入`inputs_embeds`展平为三维张量，如果`inputs_embeds`为`None`，则返回`None`
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 调用`mega`模型进行前向传播，获取输出
        outputs = self.mega(
            flat_input_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取汇总输出
        pooled_output = outputs[1]

        # 对汇总输出进行dropout操作
        pooled_output = self.dropout(pooled_output)
        # 将汇总输出传入分类器中，获取logits
        logits = self.classifier(pooled_output)
        # 调整logits的形状为二维张量
        reshaped_logits = logits.view(-1, num_choices)

        # 初始化损失为`None`
        loss = None
        # 如果标签不为`None`，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果`return_dict`为`False`，则返回输出
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回多项选择模型的输出对象
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 导入必要的库
@add_start_docstrings(
    """
    MEGA Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    MEGA_START_DOCSTRING,
)
# 定义 MegaForTokenClassification 类，用于在 MEGA 模型的基础上添加标记分类头部，例如用于命名实体识别（NER）任务
class MegaForTokenClassification(MegaPreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 记录标签数量
        self.num_labels = config.num_labels

        # 创建 MEGA 模型，不添加池化层
        self.mega = MegaModel(config, add_pooling_layer=False)
        # 计算分类器的 dropout
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 添加 dropout 层
        self.dropout = nn.Dropout(classifier_dropout)
        # 创建线性分类器
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
    @add_start_docstrings_to_model_forward(MEGA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 检查是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 MEGA 模型的前向传播方法
        outputs = self.mega(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出
        sequence_output = outputs[0]

        # 对序列输出进行 dropout 处理
        sequence_output = self.dropout(sequence_output)
        # 使用分类器进行分类
        logits = self.classifier(sequence_output)

        # 初始化损失值为 None
        loss = None
        # 如果有标签，则计算损失值
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不使用返回字典
        if not return_dict:
            # 组装输出
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 使用 TokenClassifierOutput 类组装输出
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 从transformers.models.roberta.modeling_roberta.RobertaClassificationHead复制并修改，更改为处理Mega模型的分类头
class MegaClassificationHead(nn.Module):
    """用于句子级分类任务的头部。"""

    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入维度为config.hidden_size，输出维度为config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 如果配置中classifier_dropout不为None，则使用该值，否则使用config.hidden_dropout_prob
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 使用指定的dropout概率进行dropout
        self.dropout = nn.Dropout(classifier_dropout)
        # 创建一个全连接层，输入维度为config.hidden_size，输出维度为config.num_labels
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # 取features的第一个token（等同于[CLS]）
        x = features[:, 0, :]
        # 对x进行dropout
        x = self.dropout(x)
        # 将x输入到全连接层
        x = self.dense(x)
        # 使用tanh激活函数
        x = torch.tanh(x)
        # 再次进行dropout
        x = self.dropout(x)
        # 将x输入到输出全连接层
        x = self.out_proj(x)
        return x


# MEGA模型的问题回答类，添加一个用于抽取式问答任务（如SQuAD）的跨度分类头部
class MegaForQuestionAnswering(MegaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 获取标签的数量
        self.num_labels = config.num_labels

        # 创建一个MEGA模型，关闭添加池化层的选项
        self.mega = MegaModel(config, add_pooling_layer=False)
        # 创建一个线性层，输入维度为config.hidden_size，输出维度为config.num_labels
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 对模型的前向传播进行定义
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        返回字典类型是否包含 ModelOutput；输入参数名 start_positions 的说明
        输出参数 start_positions 表示需要标记的文本起始位置的ground truth索引的标签，用于计算标记分类的损失
        输入参数名 end_positions 的说明
        输出参数 end_positions 表示需要标记的文本结束位置的ground truth索引的标签，用于计算标记分类的损失
        return_dict 接受返回参数字典类型；如果 return_dict 为 None，则使用 self.config.use_return_dict 的值
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        将 return_dict 的值赋给 return_dict 并判断是否为 None；是则将 self.config.use_return_dict 的值赋给 return_dict

        outputs = self.mega(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        将 input_ids, attention_mask, token_type_ids, inputs_embeds 等作为参数传入 self.mega 方法，得到输出 outputs

        sequence_output = outputs[0]
        将 outputs 第一个元素赋给 sequence_output

        logits = self.qa_outputs(sequence_output)
        利用 sequence_output 作为输入参数传入 qa_outputs 方法，得到逻辑回归的结果 logits

        start_logits, end_logits = logits.split(1, dim=-1)
        将 logits 按 dim=-1 进行切分，分别赋值给 start_logits 和 end_logits

        start_logits = start_logits.squeeze(-1).contiguous()
        压缩 start_logits 的最后一维（-1），并使得其在内存中连续分布

        end_logits = end_logits.squeeze(-1).contiguous()
        压缩 end_logits 的最后一维（-1），并使得其在内存中连续分布

        total_loss = None
        损失置为 None

        if start_positions is not None and end_positions is not None:
            判断 start_positions 和 end_positions 是否为 None

            # If we are on multi-GPU, split add a dimension
            如果是多GPU模式，添加一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            如果 start_positions 的维度数大于1，将最后一维进行压缩

            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            如果 end_positions 的维度数大于1，将最后一维进行压缩

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            有时候，起始/结束位置在模型输入范围之外，我们忽略这些位置
            
            ignored_index = start_logits.size(1)
            定义一个 ignored_index，起始位置（start_logits）大小的第二维度

            start_positions = start_positions.clamp(0, ignored_index)
            截断 start_positions，将超出 0, ignored_index 的位置设为这两个边界值

            end_positions = end_positions.clamp(0, ignored_index)
            截断 end_positions，将超出 0, ignored_index 的位置设为这两个边界值

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            使用 CrossEntropyLoss，传入一个 ignore_index（上面定义的 ignored_index）

            start_loss = loss_fct(start_logits, start_positions)
            计算起始位置的损失，输入参数为 start_logits、start_positions

            end_loss = loss_fct(end_logits, end_positions)
            计算结束位置的损失，输入参数为 end_logits、end_positions

            total_loss = (start_loss + end_loss) / 2
            计算平均损失

        if not return_dict:
            如果 return_dict 为 False

            output = (start_logits, end_logits) + outputs[2:]
            将 start_logits、end_logits 和 outputs 的第三个元素至最后一个元素拼接到 output 上

            return ((total_loss,) + output) if total_loss is not None else output
            返回 total_loss 和 output 的组合，如果 total_loss 不为 None，则返回 total_loss 和 output 的组合，否则返回 output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        返回一个 QuestionAnsweringModelOutput 对象，包含 loss、start_logits、end_logits、hidden_states 和 attentions 字段
```