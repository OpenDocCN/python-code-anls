# `.\models\mega\modeling_mega.py`

```py
# coding=utf-8
# Copyright 2023 The Mega Authors and The HuggingFace Inc. team.
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
"""
PyTorch MEGA model.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

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
from .configuration_mega import MegaConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "mnaylor/mega-base-wikitext"
_CONFIG_FOR_DOC = "MegaConfig"

MEGA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "mnaylor/mega-base-wikitext",
    # See all Mega models at https://huggingface.co/models?filter=mega
]


class MegaEmbeddings(nn.Module):
    """
    Mega's basic implementation does not incorporate token type embeddings, so this is a stripped-down version of
    RoBERTa's embeddings which optionally includes token types
    """

    def __init__(self, config: MegaConfig):
        super().__init__()
        # Word embeddings layer using nn.Embedding, initialized with MegaConfig parameters
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # Boolean flag indicating whether token type embeddings are used
        self.use_token_types = config.add_token_type_embeddings
        if self.use_token_types:
            # Token type embeddings layer using nn.Embedding, initialized with MegaConfig parameters
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
            # Registering a buffer for token type IDs to enable model tracing when optional IDs are not passed
            # More information at transformers issue #5664
            self.register_buffer(
                "token_type_ids", torch.zeros(config.max_positions, dtype=torch.long).expand((1, -1)), persistent=False
            )

        # Padding token index from MegaConfig
        self.padding_idx = config.pad_token_id
    # 定义模型的前向传播函数，接受输入的标识符、token类型标识符或嵌入向量
    def forward(self, input_ids=None, token_type_ids=None, inputs_embeds=None):
        # 如果既未提供input_ids也未提供inputs_embeds，则抛出数值错误
        if (input_ids is None) and (inputs_embeds is None):
            raise ValueError("Must provide one of input_ids or inputs_embeds")
        # 如果提供了input_ids
        elif input_ids is not None:
            # 获取input_ids的形状
            input_shape = input_ids.size()
            # 获取input_ids所在设备
            device = input_ids.device

            # 如果仅提供了input_ids，则从word_embeddings中获取词嵌入
            inputs_embeds = self.word_embeddings(input_ids)
        else:
            # 获取inputs_embeds的形状，去掉最后一维
            input_shape = inputs_embeds.size()[:-1]
            # 获取inputs_embeds所在设备
            device = inputs_embeds.device

        # 原始的Mega实现不包含token类型嵌入，因此我们添加了一个选项来使用它们
        if self.use_token_types:
            # 如果未提供token_type_ids
            if token_type_ids is None:
                # 如果模型具有"token_type_ids"属性，则使用已注册的缓冲区
                if hasattr(self, "token_type_ids"):
                    # 获取缓冲区的token_type_ids，并根据输入的形状进行截取或扩展
                    buffered_token_type_ids = self.token_type_ids[:, : input_shape[1]]
                    buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], input_shape[1])
                    token_type_ids = buffered_token_type_ids_expanded
                else:
                    # 否则创建一个全零的token_type_ids张量
                    token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

            # 获取token类型嵌入
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            # 将token类型嵌入添加到词嵌入中
            embeddings = inputs_embeds + token_type_embeddings
        else:
            # 如果不使用token类型嵌入，则直接使用inputs_embeds作为输出的嵌入向量
            embeddings = inputs_embeds
        # 返回最终的嵌入向量
        return embeddings
# 定义一个名为 MegaSimpleRelativePositionalBias 的类，继承自 nn.Module
class MegaSimpleRelativePositionalBias(nn.Module):
    """
    Simple relative positional embeddings copied from the Mega repo; renamed variables for better readability
    """

    # 初始化方法，接收一个 MegaConfig 类的实例作为参数
    def __init__(self, config: MegaConfig):
        super().__init__()
        # 将传入的配置对象保存为类的属性
        self.config = config
        # 根据配置中的 chunk_size 设置最大位置信息，若 chunk_size < 0 则使用 max_positions
        self.max_positions = self.config.max_positions if self.config.chunk_size < 0 else self.config.chunk_size
        # 创建一个可学习参数，表示相对位置偏置，长度为 2 * max_positions - 1
        self.rel_pos_bias = nn.Parameter(torch.Tensor(2 * config.max_positions - 1))

    # 前向传播方法，接收一个整数 seq_len 作为输入
    def forward(self, seq_len):
        # 若输入的序列长度超过了最大位置信息，则抛出 ValueError
        if seq_len > self.max_positions:
            raise ValueError("Sequence length {} going beyond max length {}".format(seq_len, self.max_positions))

        # 从 rel_pos_bias 中选择合适的偏置，长度为 seq_len * 2 - 1
        bias = self.rel_pos_bias[(self.max_positions - seq_len) : (self.max_positions + seq_len - 1)]
        # 对 bias 进行填充，向右填充 seq_len 个 0，使其长度变为 seq_len * 3 - 1
        tile = F.pad(bias, (0, seq_len))
        # 将 tile 复制 seq_len 次，得到长度为 (seq_len * 3 - 1) * seq_len 的张量
        tile = torch.tile(tile, (seq_len,))
        # 去除末尾多余的部分，使得最终的维度为 seq_len x (3 * seq_len - 2)
        tile = tile[:-seq_len]
        # 返回处理后的相对位置偏置张量
        return tile


# 定义一个名为 MegaRotaryRelativePositionalBias 的类，继承自 nn.Module
class MegaRotaryRelativePositionalBias(nn.Module):
    """
    Rotary relative bias for positional information; similar in concept to RoPE (i.e. RoFormer) but taken from the Mega
    repo due to differences in implementation.

    When initialized, produces a positional bias which ranges from position 0 to config.max_positions, but can
    extrapolate to longer sequences. Can be indexed according to input position IDs
    """

    # 初始化方法，接收一个 MegaConfig 类的实例作为参数
    def __init__(self, config: MegaConfig):
        super().__init__()
        # 如果 hidden_size 不是 2 的倍数，则抛出 RuntimeError
        if config.hidden_size % 2 != 0:
            raise RuntimeError("Rotary positional bias requires `hidden_size` to be a multiple of 2")
        # 将传入的配置对象保存为类的属性
        self.config = config
        # 设置嵌入维度为 shared_representation_size
        self.embed_dim = config.shared_representation_size
        # 根据 chunk_size 设置最大位置信息，若 chunk_size < 0 则使用 max_positions
        self.max_positions = self.config.max_positions if self.config.chunk_size < 0 else self.config.chunk_size
        # 调用静态方法 get_sinusoid_embeddings 生成正弦和余弦的嵌入
        self.sine, self.cosine = MegaRotaryRelativePositionalBias.get_sinusoid_embeddings(
            config.max_positions, self.embed_dim
        )
        # 创建两个可学习参数，分别表示 alpha 和 b_param（避免与 tf/flax 的权重处理冲突，将 b_param 重命名为 b_param）
        self.alpha = nn.Parameter(torch.Tensor(1, self.embed_dim))
        self.b_param = nn.Parameter(torch.Tensor(1, self.embed_dim))
        # 注册一个缓冲张量，值为 0.0
        self.register_buffer("_float_tensor", torch.FloatTensor([0.0]))

    # 静态方法，生成正弦和余弦的嵌入
    @staticmethod
    def get_sinusoid_embeddings(max_positions: int, embedding_dim: int):
        # 计算 embedding_dim 的一半
        half_dim = embedding_dim // 2
        # 计算指数衰减率
        emb = math.log(10000) / half_dim
        # 计算正弦和余弦的嵌入张量
        emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
        emb = torch.arange(max_positions, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        return torch.sin(emb), torch.cos(emb)
    # 定义一个函数 `rotary`，用于处理输入数据
    def rotary(self, input):
        # 获取输入张量的长度和嵌入维度
        seq_len, embed_dim = input.size()
        # 将输入张量按照最后一个维度分成两个块
        chunk_1, chunk_2 = torch.chunk(input, 2, dim=-1)
        
        # 如果 sine 或 cosine 是空的，或者序列长度超过当前的最大位置数
        if self.sine is None or seq_len > self.sine.size(0):
            # 生成新的 sine 和 cosine 位置嵌入
            self.sine, self.cosine = MegaRotaryRelativePositionalBias.get_sinusoid_embeddings(seq_len, embed_dim)
            # 更新最大的位置数
            self.max_positions = seq_len
        
        # 将 sine 和 cosine 转换为指定的浮点张量类型
        self.sine = self.sine.to(self._float_tensor)
        self.cosine = self.cosine.to(self._float_tensor)

        # 取出当前序列长度内的 sine 和 cosine
        sin = self.sine[:seq_len]
        cos = self.cosine[:seq_len]
        
        # 返回旋转后的张量结果，按照旋转矩阵的定义进行计算
        return torch.cat([chunk_1 * cos - chunk_2 * sin, chunk_2 * cos + chunk_1 * sin], dim=1)

    # 定义前向传播函数 `forward`
    def forward(self, seq_len):
        # 计算 alpha 的旋转结果
        rotary_alpha = self.rotary(self.alpha.expand(seq_len, self.embed_dim))
        # 计算 beta 的旋转结果
        rotary_beta = self.rotary(self.b_param.expand(seq_len, self.embed_dim))
        
        # 计算旋转后的偏置张量，使用 Einstein Summation Notation (einsum) 定义
        bias = torch.einsum("mk,nk->mn", rotary_alpha, rotary_beta)
        
        # 返回偏置张量作为前向传播的结果
        return bias
class MegaDropout(nn.Module):
    """
    A unified class for standard dropout functionality and featurewise dropout.

    The original fairseq Mega repo used 2 classes for these, which included some unnecessary handling of training logic
    and an unused `inplace` option. The original implementation used torch.nn.functional instead of submodules, which
    is retained here as well.
    """

    def __init__(self, dropout_probability, is_featurewise=False):
        super().__init__()
        self.dropout_probability = dropout_probability  # 设置 dropout 的概率
        self.is_featurewise = is_featurewise  # 是否使用特征级别的 dropout

    def forward(self, input, batch_first: bool = False):
        if self.is_featurewise:
            if batch_first:
                # 如果 batch_first 为 True，则进行维度转换：
                # (batch_size X sequence_length X feature_dimension)
                # -> (batch_size X feature_dimension X sequence_length)
                # -> (batch_size X sequence_length X feature_dimension)
                return F.dropout2d(
                    input.transpose(-1, -2), p=self.dropout_probability, training=self.training
                ).transpose(-1, -2)
            else:
                if input.dim() != 3:
                    raise ValueError(
                        "Feature dropout inputs must be exactly 3-dimensional if inputs are ordered [sequence length, batch size, hidden dimension]"
                    )
                # 如果 batch_first 为 False，并且输入不是 3 维的，抛出 ValueError
                # (sequence_length X batch_size X feature_dimension)
                # -> (batch_size X feature_dimension X sequence_length)
                # -> (sequence_length X batch_size X feature_dimension)
                return F.dropout2d(input.permute(1, 2, 0), p=self.dropout_probability, training=self.training).permute(
                    2, 0, 1
                )
        else:
            # 如果不是 featurewise dropout，直接应用标准的 dropout
            return F.dropout(input, p=self.dropout_probability, training=self.training)


class MegaRMSNorm(nn.Module):
    """
    RMSNorm used in Mega implementation. Differs from T5's RMSNorm by applying the weight prior to taking the square
    root (as opposed to after in T5)
    """

    def __init__(self, number_features, eps=1e-6, affine=True):
        super().__init__()
        self.num_features = number_features  # 特征的数量
        self.eps = eps  # epsilon 值，用于数值稳定性
        self.affine = affine  # 是否应用仿射变换
        if affine:
            self.weight = nn.Parameter(torch.Tensor(self.num_features))  # 如果 affine 为 True，则初始化权重参数
        else:
            self.register_parameter("weight", None)  # 如果 affine 为 False，则不使用权重参数

    def forward(self, input):
        mean_square = torch.mean(torch.square(input), dim=-1, keepdim=True)  # 计算输入的平方的均值
        if self.weight is not None:
            input = input * self.weight  # 如果有权重参数，将输入乘以权重

        input * torch.rsqrt(mean_square + self.eps)  # 应用 RMS 标准化
        return input


class MegaScaleNorm(nn.Module):
    """
    Scale normalization introduced in MEGA which is similar to RMSNorm, but uses a single parameter for scalar
    multiplication instead of a vector, and applies over a specified dimension
    """

    # 此处留空，未提供具体实现，仅有描述
    # 初始化函数，用于初始化 BatchNorm 自定义层
    def __init__(self, dim, eps=1e-6, affine=True):
        # 调用父类的初始化方法
        super().__init__()
        # 设置 BatchNorm 的维度
        self.dim = dim
        # 设置 BatchNorm 的 epsilon 值，用于数值稳定性
        self.eps = eps
        # 是否使用仿射变换
        self.affine = affine
        # 如果启用仿射变换
        if affine:
            # 创建一个可学习的标量参数 scalar
            self.scalar = nn.Parameter(torch.Tensor(1))
        else:
            # 如果不启用仿射变换，则注册一个空的参数 scalar
            self.register_parameter("scalar", None)

    # 前向传播函数，用于计算 BatchNorm 的输出
    def forward(self, input):
        # 计算输入张量的各维度上的平方后求平均值
        mean_square = torch.mean(torch.square(input), dim=self.dim, keepdim=True)
        # 如果存在仿射变换参数 scalar
        if self.scalar is not None:
            # 对输入张量进行仿射变换
            input = self.scalar * input

        # 根据 BatchNorm 公式，计算 BatchNorm 的输出
        output = input * torch.rsqrt(mean_square + self.eps)
        return output
# MegaSequenceNorm 类定义，用于包装 Mega 中使用的各种层归一化选项，处理不同归一化方法对输入轴位置的期望差异。

    """
    A wrapper class for various layer normalization options used in Mega. Used to handle differences in expectations on
    input axis locations for different normalization methods.
    """

    def __init__(self, norm_type, embedding_dim, eps=1e-5, affine=True, export=False):
        # 初始化函数，根据给定的归一化类型选择相应的归一化层
        super().__init__()
        if norm_type == "layernorm":
            # 如果是 layernorm，使用 PyTorch 的 LayerNorm 归一化层
            self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine=affine)
        elif norm_type == "scalenorm":
            # 如果是 scalenorm，使用 MegaScaleNorm 归一化层
            self.norm = MegaScaleNorm(dim=-1, eps=eps, affine=affine)
        elif norm_type == "rmsnorm":
            # 如果是 rmsnorm，使用 MegaRMSNorm 归一化层
            self.norm = MegaRMSNorm(embedding_dim, eps=eps, affine=affine)
        elif norm_type == "batchnorm":
            # 如果是 batchnorm，使用 PyTorch 的 BatchNorm1d 归一化层
            self.norm = nn.BatchNorm1d(embedding_dim, eps=eps, affine=affine)
        elif norm_type == "syncbatchnorm":
            # 如果是 syncbatchnorm，使用 PyTorch 的 SyncBatchNorm 归一化层
            self.norm = nn.SyncBatchNorm(embedding_dim, eps=eps, affine=affine)
        else:
            # 如果类型未知，则抛出 ValueError 异常
            raise ValueError("Unknown norm type: {}".format(norm_type))

    def forward(self, input):
        # 前向传播函数，根据归一化层类型执行相应的归一化操作
        if isinstance(self.norm, nn.modules.batchnorm._BatchNorm):
            # 如果当前归一化层是 BatchNorm 类型，则要求输入必须是三维的张量
            if input.dim() != 3:
                raise ValueError("BatchNorm inputs must be exactly 3-dimensional")
            # 将输入的维度顺序转换为 (batch_size, seq_len, embedding_dim)
            input = input.permute(1, 2, 0)
            # 应用归一化层
            input = self.norm(input)
            # 将输出维度顺序转换回来 (seq_len, batch_size, embedding_dim)
            return input.permute(2, 0, 1)
        else:
            # 对于其他类型的归一化层，直接应用归一化
            return self.norm(input)


# 将 MegaSequenceNorm 类添加到 ALL_LAYERNORM_LAYERS 列表中
ALL_LAYERNORM_LAYERS.append(MegaSequenceNorm)


class MegaMultiDimensionDampedEma(nn.Module):
    """
    Mega's Exponential Moving Average layer, largely left unmodified from the original repo with the exception of
    variable names and moving away from the stateful representation of incremental decoding state. See
    "https://arxiv.org/abs/2209.10655" for more details.
    """
    def __init__(self, config: MegaConfig):
        super().__init__()
        
        # 初始化函数，接收一个 MegaConfig 类型的配置参数对象
        self.config = config

        # 设置嵌入维度为配置中的隐藏大小
        self.embed_dim = config.hidden_size
        # 设置维度为配置中的EMA投影大小
        self.ndim = config.ema_projection_size
        # 设置是否双向
        self.bidirectional = config.bidirectional
        # 设置截断大小
        self.truncation = config.truncation
        # 设置比例为EMA投影大小的倒数平方根
        self.scale = math.sqrt(1.0 / self.ndim)

        # 计算卷积核维度，如果是双向则为隐藏大小的两倍
        kernel_dim = 2 * config.hidden_size if self.bidirectional else config.hidden_size
        
        # 重命名阻尼因子和衰减因子以更清晰地描述参数功能
        self.damping_factor = nn.Parameter(torch.Tensor(kernel_dim, self.ndim, 1))
        self.decay_factor = nn.Parameter(torch.Tensor(kernel_dim, self.ndim, 1))
        
        # 重命名EMA扩展矩阵和核投影矩阵以避免与HF重命名冲突，并与论文描述保持一致
        self.ema_expansion_matrix = nn.Parameter(torch.Tensor(kernel_dim, self.ndim, 1))
        self.kernel_projection_matrix = nn.Parameter(torch.Tensor(kernel_dim, self.ndim))
        
        # 将omega重命名为残差权重以描述其作用
        self.residual_weight = nn.Parameter(torch.Tensor(config.hidden_size))
        
        # 初始化私有变量
        self._kernel = None
        self._coeffs = None

    def _compute_ema_coefficients(self):
        # 计算EMA系数
        self._coeffs = None
        
        # 将阻尼因子和衰减因子（kernel_dim x EMA投影大小 x 1）转换为[0, 1]区间，使用sigmoid函数
        damping_factor = torch.sigmoid(self.damping_factor)
        decay_factor = torch.sigmoid(self.decay_factor)
        
        # 计算上一个时间步的权重
        previous_timestep_weight = 1.0 - damping_factor * decay_factor
        
        return damping_factor, previous_timestep_weight

    def _compute_efficient_ema_kernel(self, length: int):
        # 计算用于高效阻尼EMA的卷积核
        
        self._kernel = None
        
        # 计算EMA系数
        damping_factor, previous_timestep_weight = self._compute_ema_coefficients()
        
        # 创建Vandermonde矩阵，形状为(1, 1, length)，乘以对数化的上一个时间步权重
        vander = torch.arange(length).to(damping_factor).view(1, 1, length) * torch.log(previous_timestep_weight)
        
        # 计算卷积核，形状为(kernel_dim x EMA投影大小 x sequence_length)
        kernel = (damping_factor * self.ema_expansion_matrix) * torch.exp(vander)
        
        # 将卷积核从三维形状(kernel_dim x EMA投影大小 x sequence_length)压缩为二维形状(kernel_dim, sequence_length)
        return torch.einsum("dnl,dn->dl", kernel, self.kernel_projection_matrix * self.scale)

    def get_ema_coefficients(self):
        # 获取EMA系数
        
        if self.training:
            # 在训练模式下，重新计算EMA系数并返回
            return self._compute_ema_coefficients()
        else:
            # 在非训练模式下，如果系数尚未计算，则计算并存储，并返回
            if self._coeffs is None:
                self._coeffs = self._compute_ema_coefficients()
            return self._coeffs
    # 定义一个方法，用于获取指数移动平均（EMA）核函数
    def get_ema_kernel(self, length: int):
        # 确定核函数的大小，取决于给定的长度和截断值（如果有的话）
        kernel_size = length if self.truncation is None else min(self.truncation, length)
        
        # 如果处于训练模式下
        if self.training:
            # 调用计算高效EMA核函数的私有方法
            return self._compute_efficient_ema_kernel(kernel_size)
        else:
            # 如果核函数为空或者大小小于指定的长度
            if self._kernel is None or self._kernel.size(-1) < kernel_size:
                # 计算并缓存高效EMA核函数
                self._kernel = self._compute_efficient_ema_kernel(kernel_size)
            
            # 返回核函数的部分，截取到指定长度
            return self._kernel[..., :kernel_size]

    # 定义一个方法，实现使用FFT卷积进行重复计算EMA的包装器
    def fft_convolution(self, inputs, kernel, length):
        # 对输入进行FFT（快速傅里叶变换），扩展为两倍的长度
        inputs_fft = torch.fft.rfft(inputs.float(), n=2 * length)
        
        # 对核函数进行FFT，扩展为两倍的长度
        kernel_fft = torch.fft.rfft(kernel.float(), n=2 * length)
        
        # 执行FFT卷积，得到卷积序列
        convolved_sequence = torch.fft.irfft(inputs_fft * kernel_fft, n=2 * length)
        
        # 返回卷积序列作为结果
        return convolved_sequence
    # 计算指数移动平均 (EMA) 的一步更新
    def ema_step(self, inputs, length, past_state=None):
        # 如果长度为1，直接调用单步 EMA 更新函数
        if length == 1:
            return self.one_ema_step(inputs, past_state=past_state)

        # 获取当前时间步的阻尼系数和上一时间步权重
        damping_factor, previous_timestep_weight = self.get_ema_coefficients()

        # 构建范德蒙德矩阵
        vander = torch.arange(length + 1).to(damping_factor).view(1, 1, length + 1) * torch.log(
            previous_timestep_weight
        )
        vander = torch.exp(vander)

        # 如果有过去状态，计算过去的 EMA 投影和范德蒙德矩阵
        if past_state is not None:
            # 计算过去 EMA 投影
            past_ema_proj = vander[:, :, 1:] * (self.kernel_projection_matrix * self.scale).unsqueeze(-1)
            # 计算过去 EMA 状态
            past_ema_state = torch.einsum("bdn,dnl->bdl", past_state, past_ema_proj)
            # 计算过去范德蒙德矩阵
            past_vandermonde = vander[:, :, -1] * past_state
        else:
            past_ema_state = None
            past_vandermonde = None

        # 调整范德蒙德矩阵的维度
        vander = vander[:, :, :-1]

        # 计算卷积核
        kernel = (damping_factor * self.ema_expansion_matrix) * vander
        kernel_proj = torch.einsum("dnl,dn->dl", kernel, self.kernel_projection_matrix * self.scale)

        # 执行 FFT 卷积操作
        ema_output = self.fft_convolution(inputs, kernel_proj, length=length)[..., 0:length]
        ema_output = ema_output.type_as(inputs)

        # 如果有过去 EMA 状态，加上过去状态
        if past_ema_state is not None:
            ema_output = ema_output + past_ema_state

        # 更新隐藏状态
        updated_hidden_state = torch.einsum("bdl,dnl->bdn", inputs, torch.flip(kernel, dims=[2]))

        # 如果有过去范德蒙德矩阵，加上过去范德蒙德矩阵
        if past_vandermonde is not None:
            updated_hidden_state = updated_hidden_state + past_vandermonde

        # 返回结果，包括 EMA 输出和更新后的隐藏状态
        # 返回一个元组:
        # (sequence_length, batch_size, kernel_dim)
        # (batch_size, kernel_dim, ema_projection_size)
        return ema_output.permute(2, 0, 1), updated_hidden_state

    # 计算指数移动平均 (EMA) 的单步更新
    def one_ema_step(self, inputs, past_state=None):
        # 获取当前时间步的阻尼系数和上一时间步权重
        damping_factor, previous_timestep_weight = self.get_ema_coefficients()

        # 计算更新后的状态
        updated_state = (damping_factor * self.ema_expansion_matrix).squeeze(-1) * inputs

        # 如果有过去状态，加上过去状态的权重
        if past_state is not None:
            updated_state = updated_state + previous_timestep_weight.squeeze(-1) * past_state

        # 计算输出
        out = torch.einsum("bdn,dn->bd", updated_state, self.kernel_projection_matrix * self.scale)

        # 返回结果，包括输出和更新后的状态
        # 返回一个元组:
        # (1, batch_size, kernel_dim), (batch_size, kernel_dim, ema_projection_size)
        return out.unsqueeze(0), updated_state
    # 定义一个方法 `forward`，用于模型的前向传播
    # 参数 `self` 表示类的实例本身，`inputs` 是输入数据
    # `attention_mask` 是一个可选的张量，用于注意力掩码
    # `prev_state` 也是一个可选的张量，表示前一个状态的输出
    # `use_cache` 是一个布尔值，默认为 False，指示是否使用缓存
class MegaGatedCrossAttention(nn.Module):
    """
    Gated Structured State Attention for use in encoder-decoder model. See Mega paper for more details. Only
    modifications from original implementation are variable names, removing the unnecessary `before_attn_fn` and
    `static_kv` arguments, and the stateful representation of incremental decoder state.
    """

    def __init__(self, config: MegaConfig):
        super().__init__()

        self.config = config  # 存储传入的配置对象
        self.activation = ACT2FN[self.config.activation]  # 根据配置中的激活函数名称选择对应的激活函数
        self.attention_activation = self.config.attention_activation  # 存储注意力激活函数类型
        self.scaling = self.config.shared_representation_size**-0.5 if self.attention_activation == "softmax" else None  # 如果注意力激活函数是softmax，则设置缩放因子，否则为None

        self.dropout = MegaDropout(self.config.dropout_prob, is_featurewise=self.config.use_feature_dropout)  # 使用MegaDropout初始化普通的Dropout
        self.hidden_dropout = MegaDropout(
            self.config.hidden_dropout_prob, is_featurewise=self.config.use_feature_dropout
        )  # 使用MegaDropout初始化隐藏层Dropout
        # Attention dropout is standard dropout
        self.attention_dropout = MegaDropout(self.config.attention_probs_dropout_prob, is_featurewise=False)  # 使用MegaDropout初始化注意力Dropout

        self.prenorm = self.config.normalize_before_mega  # 是否在应用Mega之前进行归一化的标志
        self.norm = MegaSequenceNorm(
            self.config.normalization_type, self.config.hidden_size, affine=self.config.norm_affine
        )  # 使用MegaSequenceNorm初始化归一化层

        self.k_proj = nn.Linear(self.config.hidden_size, self.config.shared_representation_size)  # 创建线性层k_proj，用于映射隐藏状态到共享表示大小
        self.v_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)  # 创建线性层v_proj，用于映射隐藏状态到隐藏大小
        self.q_proj = nn.Linear(
            self.config.hidden_size, 2 * self.config.hidden_size + self.config.shared_representation_size
        )  # 创建线性层q_proj，用于映射隐藏状态到查询大小
        self.h_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)  # 创建线性层h_proj，用于映射隐藏状态到隐藏大小

        if self.config.relative_positional_bias == "simple":
            self.rel_pos_bias = MegaSimpleRelativePositionalBias(config)  # 如果相对位置偏置为简单类型，则创建简单相对位置偏置对象
        elif self.config.relative_positional_bias == "rotary":
            self.rel_pos_bias = MegaRotaryRelativePositionalBias(config)  # 如果相对位置偏置为旋转类型，则创建旋转相对位置偏置对象
        else:
            raise ValueError("unknown relative position bias: {}".format(self.config.relative_positional_bias))  # 如果相对位置偏置类型未知，则抛出异常

        self.softmax = nn.Softmax(dim=-1)  # 创建softmax层，沿着最后一个维度进行softmax操作
    def element_attention(self, query, key, key_padding_mask, pidx):
        # 获取 key 的尺寸信息
        bsz, src_len, _ = key.size()
        # 获取查询序列的长度，如果有位置索引 pidx，则使用 pidx+1，否则使用查询序列的长度
        tgt_len = query.size(1) if pidx is None else pidx + 1
        if key_padding_mask is not None:
            # 计算每个样本在源序列上的有效长度，并扩展为 (batch_size X 1 X 1) 的形状
            lengths = key_padding_mask.sum(dim=-1).view(bsz, 1, 1)
        else:
            # 如果没有提供 key_padding_mask，则使用源序列的长度作为有效长度
            lengths = src_len

        # 生成相对位置偏置矩阵，形状为 (target_sequence_length X source_sequence_length)
        bias = self.rel_pos_bias(max(tgt_len, src_len))[:, :src_len]
        if pidx is not None:
            if query.size(1) != 1:
                raise ValueError("Position offset provided with queries longer than 1 token")
            # 如果提供了位置索引 pidx，并且查询序列的长度不为 1，则引发异常
            bias = bias[pidx]
        else:
            # 如果没有提供位置索引 pidx，则截取相对位置偏置矩阵到目标序列长度 tgt_len
            bias = bias[:tgt_len]

        # 计算查询-键之间的点积注意力，除以有效长度并加上偏置
        qk = torch.bmm(query, key.transpose(1, 2)) / lengths + bias

        # 使用激活函数 ACT2FN[self.attention_activation] 处理注意力权重
        attn_weights = ACT2FN[self.attention_activation](qk).type_as(qk)

        if key_padding_mask is not None:
            # 如果存在 key_padding_mask，则将注意力权重乘以 key_padding_mask 的扩展形状
            attn_weights = attn_weights * key_padding_mask.unsqueeze(1)

        return attn_weights

    def softmax_attention(self, query, key, key_padding_mask, pidx):
        # 获取 key 的尺寸信息
        bsz, src_len, _ = key.size()
        # 获取查询序列的长度，如果有位置索引 pidx，则使用 pidx+1，否则使用查询序列的长度
        tgt_len = query.size(1) if pidx is None else pidx + 1

        # 生成相对位置偏置矩阵，形状为 (target_sequence_length X source_sequence_length)
        bias = self.rel_pos_bias(max(tgt_len, src_len))[:, :src_len]
        if pidx is not None:
            if query.size(1) != 1:
                raise ValueError("Position offset provided with queries longer than 1 token")
            # 如果提供了位置索引 pidx，并且查询序列的长度不为 1，则引发异常
            bias = bias[pidx]
        else:
            # 如果没有提供位置索引 pidx，则截取相对位置偏置矩阵到目标序列长度 tgt_len
            bias = bias[:tgt_len]

        # 对查询进行缩放
        query = query * self.scaling
        # 计算查询-键之间的点积注意力，并加上偏置
        qk = torch.bmm(query, key.transpose(1, 2)) + bias

        if key_padding_mask is not None:
            # 使用 key_padding_mask 屏蔽无效位置
            qk = qk.masked_fill((1 - key_padding_mask).unsqueeze(1).to(torch.bool), float("-inf"))

        # 对注意力权重进行 softmax 归一化
        attn_weights = self.softmax(qk).type_as(qk)
        return attn_weights

    def forward(
        self,
        query,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    """
    Pure PyTorch implementation of Mega block; see https://arxiv.org/abs/2209.10655 and original fairseq implementation
    at https://github.com/facebookresearch/mega (copyright Meta Research, licensed under MIT License)

    Differences from original implementation include hidden state refactor and fixed inconsistency with additive /
    multiplicative attention masks
    """

    def __init__(self, config: MegaConfig):
        super().__init__()
        self.config = config
        self.activation = ACT2FN[self.config.activation]  # 设置激活函数，根据配置选择相应的激活函数
        self.scaling = (
            self.config.shared_representation_size**-0.5 if self.config.attention_activation == "softmax" else None
        )  # 如果注意力激活函数为 softmax，则设置缩放因子为共享表示大小的倒数，否则为 None
        self.dropout = MegaDropout(self.config.dropout_prob, is_featurewise=self.config.use_feature_dropout)  # 初始化特征级别的 dropout
        self.hidden_dropout = MegaDropout(
            self.config.hidden_dropout_prob, is_featurewise=self.config.use_feature_dropout
        )  # 初始化隐藏层级别的 dropout
        # attention dropout is standard dropout
        self.attention_dropout = MegaDropout(self.config.attention_probs_dropout_prob, is_featurewise=False)  # 初始化注意力矩阵的 dropout

        self.norm = MegaSequenceNorm(
            self.config.normalization_type, self.config.hidden_size, affine=self.config.norm_affine
        )  # 初始化序列规范化层，根据配置设置归一化类型和是否仿射变换
        self.ema_gate = MegaMultiDimensionDampedEma(config)  # 初始化多维度阻尼 EMA（指数移动平均）

        self.v_proj = nn.Linear(self.config.hidden_size, self.config.intermediate_size)  # 初始化线性变换 v_proj
        self.mx_proj = nn.Linear(
            self.config.hidden_size,
            self.config.shared_representation_size + self.config.intermediate_size + 2 * self.config.hidden_size,
        )  # 初始化线性变换 mx_proj
        self.h_proj = nn.Linear(self.config.intermediate_size, self.config.hidden_size)  # 初始化线性变换 h_proj

        self.qk_weight = nn.Parameter(torch.Tensor(2, self.config.shared_representation_size))  # 初始化查询和键的权重参数
        self.qk_bias = nn.Parameter(torch.Tensor(2, self.config.shared_representation_size))  # 初始化查询和键的偏置参数

        if self.config.relative_positional_bias == "simple":
            self.rel_pos_bias = MegaSimpleRelativePositionalBias(config)  # 如果相对位置偏置为简单模式，则初始化简单相对位置偏置
        elif self.config.relative_positional_bias == "rotary":
            self.rel_pos_bias = MegaRotaryRelativePositionalBias(config)  # 如果相对位置偏置为旋转模式，则初始化旋转相对位置偏置
        else:
            raise ValueError(f"Unknown relative positional bias: {self.config.relative_positional_bias}")  # 抛出异常，未知的相对位置偏置类型

        self.softmax = nn.Softmax(dim=-1)  # 初始化 Softmax 函数，用于计算 softmax 注意力分布
        self.attention_function = (
            self.softmax_attention if self.config.attention_activation == "softmax" else self.element_attention
        )  # 根据配置选择注意力激活函数，softmax 或者元素级别的注意力
    def element_attention(self, query, key, padding_mask, causal_mask):
        """
        Apply element-wise attention via relu^2 or laplace. Same as original implementation but with standardized
        causal attention mask. Expects the Hugging Face standard attention mask paradigm: 1 for not masked, and 0 for
        masked.
        """
        # 获取序列长度
        seq_len = key.size(2)

        # 如果存在填充掩码
        if padding_mask is not None:
            # 计算每个样本的有效长度并扩展维度
            # (batch_size X number of chunks X 1)
            lengths = padding_mask.sum(-1, keepdim=True)
            # (batch_size X number of chunks X 1 X 1)
            lengths = lengths.clamp(min=1.0).unsqueeze(-1)
        else:
            # 如果没有填充掩码，则使用序列长度
            lengths = seq_len

        # 如果存在因果掩码
        if causal_mask is not None:
            # 计算因果掩码的和并扩展维度
            lengths = causal_mask.sum(dim=-1, keepdim=True)

        # 获取相对位置偏置
        # (sequence_length X sequence_length)
        bias = self.rel_pos_bias(seq_len)

        # 如果查询向量的长度不等于键向量的长度，则抛出异常
        if seq_len != query.size(2):
            if query.size(2) != 1:
                raise ValueError("Size mismatch between Q and K in element attention")
            # 只选择最后一个位置的偏置
            # (1 X sequence_length)
            bias = bias[-1:]

        # 计算查询-键之间的注意力分数
        # (batch_size X number of chunks X sequence_length X sequence_length)
        qk = torch.matmul(query, key.transpose(2, 3)) / lengths + bias

        # 应用激活函数到注意力分数
        attn_weights = ACT2FN[self.config.attention_activation](qk).type_as(qk)

        # 如果存在填充掩码，则应用填充掩码
        if padding_mask is not None:
            attn_weights = attn_weights * padding_mask.unsqueeze(2)

        # 如果存在因果掩码，则应用因果掩码
        if causal_mask is not None:
            attn_weights = attn_weights * causal_mask

        # 返回注意力权重
        return attn_weights
    def softmax_attention(self, query, key, padding_mask, causal_mask):
        "Standard softmax self-attention, as in the original Transformer paper"
        # 获取序列长度
        seq_len = key.size(2)
        
        # 生成相对位置偏置矩阵
        bias = self.rel_pos_bias(seq_len)
        
        # 如果 Q 和 K 的长度不匹配，进行异常处理
        if seq_len != query.size(2):
            if query.size(2) != 1:
                raise ValueError("Size mismatch between Q and K in softmax attention")
            # 如果长度不匹配，只取最后一行偏置矩阵
            bias = bias[-1:]

        # 缩放注意力权重
        query = query * self.scaling

        # 计算注意力矩阵 QK
        qk = torch.matmul(query, key.transpose(2, 3)) + bias
        
        # 应用因果遮蔽（假设为1/0表示未遮蔽/遮蔽）
        if causal_mask is not None:
            additive_causal_mask = torch.zeros_like(causal_mask, dtype=qk.dtype)
            additive_causal_mask = additive_causal_mask.masked_fill((1 - causal_mask).bool(), float("-inf"))
            qk = qk + additive_causal_mask

        # 应用填充遮蔽
        if padding_mask is not None:
            # 将填充遮蔽反转，以符合 Mega 源码的处理方式
            padding_mask = 1 - padding_mask
            padding_mask_all = padding_mask.all(dim=-1, keepdim=True)
            padding_mask = torch.logical_and(padding_mask, ~padding_mask_all)
            qk = qk.masked_fill(padding_mask.unsqueeze(2).to(torch.bool), float("-inf"))

        # 计算 softmax 权重并转换为与 QK 相同的数据类型
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
# 定义一个名为 MegaNormalizedFeedForwardNetwork 的类，继承自 nn.Module 类
class MegaNormalizedFeedForwardNetwork(nn.Module):
    """
    Normalized feed-forward network used in Mega blocks. Left as-is from original Mega repo aside from retrieving args
    from Hugging Face config
    """

    # 初始化方法，接受一个 MegaConfig 类型的参数 config
    def __init__(self, config: MegaConfig):
        super().__init__()

        # 将参数 config 存储在对象的属性中
        self.config = config
        # 初始化隐藏层维度为配置中的 nffn_hidden_size
        self.hidden_dim = config.nffn_hidden_size
        # 激活函数名称从配置中获取，并且将其映射为对应的激活函数
        self.act_fn = config.activation
        self.activation = ACT2FN[config.activation]

        # 初始化两个 MegaDropout 对象，分别用于不同的配置参数
        self.dropout = MegaDropout(self.config.dropout_prob, is_featurewise=self.config.use_feature_dropout)
        self.hidden_dropout = MegaDropout(
            self.config.nffn_activation_dropout_prob, is_featurewise=self.config.use_feature_dropout
        )

        # 根据配置参数决定是否在前馈网络之前进行归一化
        self.prenorm = self.config.normalize_before_ffn
        # 初始化 MegaSequenceNorm 对象，用于序列归一化
        self.norm = MegaSequenceNorm(
            self.config.normalization_type, self.config.hidden_size, affine=self.config.norm_affine
        )

        # 初始化两个线性层，分别为输入到隐藏层和隐藏层到输出层的线性变换
        self.fc1 = nn.Linear(self.config.hidden_size, self.config.nffn_hidden_size)
        self.fc2 = nn.Linear(self.config.nffn_hidden_size, self.config.hidden_size)

    # 前向传播方法，接受输入 inputs，并返回输出 output
    def forward(self, inputs):
        # 将输入保存为残差连接的一部分
        residual = inputs

        # 如果配置要求在前馈网络之前进行归一化，则对输入进行归一化处理
        if self.prenorm:
            inputs = self.norm(inputs)

        # 第一层前馈网络，使用激活函数后，应用 dropout
        hidden = self.activation(self.fc1(inputs))
        hidden = self.hidden_dropout(hidden)
        # 第二层前馈网络，无激活函数，但应用 dropout
        output = self.fc2(hidden)
        output = self.dropout(output)
        # 将输出与残差相加，实现残差连接
        output = output + residual

        # 如果配置要求在前馈网络之后进行归一化，则对输出进行归一化处理
        if not self.prenorm:
            output = self.norm(output)

        # 返回最终的输出
        return output


# 定义一个名为 MegaBlock 的类，继承自 nn.Module 类
class MegaBlock(nn.Module):
    # 初始化方法，接受一个 MegaConfig 类型的参数 config
    def __init__(self, config: MegaConfig):
        super().__init__()
        # 设置序列长度的维度为 1
        self.seq_len_dim = 1
        # 初始化 MegaMovingAverageGatedAttention 对象
        self.mega_layer = MegaMovingAverageGatedAttention(config)
        # 根据配置决定是否初始化 MegaNormalizedFeedForwardNetwork 对象
        self.nffn = MegaNormalizedFeedForwardNetwork(config) if config.use_normalized_ffn else None
        # 设置是否为解码器的标志
        self.is_decoder = config.is_decoder
        # 根据配置决定是否添加交叉注意力机制
        self.add_cross_attention = config.add_cross_attention
        # 如果添加了交叉注意力机制
        if self.add_cross_attention:
            # 如果不是解码器模型，则抛出 ValueError 异常
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 初始化 MegaGatedCrossAttention 对象
            self.cross_attn = MegaGatedCrossAttention(config)
        else:
            self.cross_attn = None

    # 前向传播方法，接受多个输入参数，返回输出 hidden_states
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
# 从 transformers.models.roberta.modeling_roberta.RobertaPooler 复制，将 Roberta 替换为 Mega
class MegaPooler(nn.Module):
    # 初始化方法，接受一个 config 参数
    def __init__(self, config):
        super().__init__()
        # 初始化线性层，输入和输出维度都为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 激活函数为双曲正切函数
        self.activation = nn.Tanh()
    # 定义前向传播方法，接受隐藏状态张量作为输入，并返回张量作为输出
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过获取第一个 token 对应的隐藏状态来“池化”模型
        first_token_tensor = hidden_states[:, 0]
        # 将第一个 token 的隐藏状态传递给全连接层进行线性变换
        pooled_output = self.dense(first_token_tensor)
        # 应用激活函数到线性变换的结果上
        pooled_output = self.activation(pooled_output)
        # 返回池化后的输出张量
        return pooled_output
class MegaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类，用于当前类的配置
    config_class = MegaConfig
    # 设置基础模型前缀，用于标识当前类的基础模型
    base_model_prefix = "mega"
    # 指示当前类不支持梯度检查点
    supports_gradient_checkpointing = False
    # 定义不需要拆分的模块列表，这些模块不会在模型参数分组中进行拆分
    _no_split_modules = ["MegaMovingAverageGatedAttention"]
    # 初始化模块的权重
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果模块是 MegaMultiDimensionDampedEma 类型
        if isinstance(module, MegaMultiDimensionDampedEma):
            # 使用 torch.no_grad() 上下文管理器，确保不计算梯度
            with torch.no_grad():
                # 初始化模块的阻尼因子和衰减因子
                nn.init.normal_(module.damping_factor, mean=0.0, std=self.config.ema_delta_alpha_range)
                nn.init.normal_(module.decay_factor, mean=0.0, std=self.config.ema_delta_alpha_range)
                # 初始化模块的扩展矩阵，其中特定索引位置的值为 -1，其余为 1
                val = torch.ones(self.config.ema_projection_size, 1)
                if self.config.ema_projection_size > 1:
                    idx = torch.tensor(list(range(1, self.config.ema_projection_size, 2)))
                    val.index_fill_(0, idx, -1.0)
                module.ema_expansion_matrix.normal_(mean=0.0, std=self.config.ema_beta_range).add_(val)
                # 初始化模块的核心投影矩阵和残余权重
                nn.init.normal_(module.kernel_projection_matrix, mean=0.0, std=self.config.ema_gamma_omega_range)
                nn.init.normal_(module.residual_weight, mean=0.0, std=self.config.ema_gamma_omega_range)
        # 如果模块是 MegaSimpleRelativePositionalBias 类型
        elif isinstance(module, MegaSimpleRelativePositionalBias):
            # 初始化相对位置偏置
            nn.init.normal_(module.rel_pos_bias, mean=0.0, std=self.config.initializer_range)
        # 如果模块是 MegaRotaryRelativePositionalBias 类型
        elif isinstance(module, MegaRotaryRelativePositionalBias):
            # 初始化旋转相对位置偏置的 alpha 和 b_param
            nn.init.normal_(module.alpha, mean=0.0, std=self.config.initializer_range)
            nn.init.normal_(module.b_param, mean=0.0, std=self.config.initializer_range)
        # 如果模块是 MegaScaleNorm 类型
        elif isinstance(module, MegaScaleNorm):
            # 如果配置中开启了归一化参数，初始化 scalar 参数为 1.0
            if self.config.norm_affine:
                nn.init.constant_(module.scalar, 1.0)
        # 如果模块是 MegaRMSNorm 类型
        elif isinstance(module, MegaRMSNorm):
            # 如果配置中开启了归一化参数，初始化 weight 参数为 1.0
            if self.config.norm_affine:
                nn.init.constant_(module.weight, 1.0)
        # 如果模块是 MegaMovingAverageGatedAttention 类型
        elif isinstance(module, MegaMovingAverageGatedAttention):
            # 初始化模块的 qk_weight 和 qk_bias
            # 线性层在下面的通用 nn.Linear 初始化中单独处理
            nn.init.normal_(module.qk_weight, mean=0.0, std=self.config.initializer_range)
            nn.init.constant_(module.qk_bias, 0.0)
        # 如果模块是 nn.Linear 类型
        elif isinstance(module, nn.Linear):
            # 初始化整个网络中所有线性层的权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是 nn.Embedding 类型
        elif isinstance(module, nn.Embedding):
            # 初始化嵌入层的权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果指定了填充索引，将其对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果模块是 nn.LayerNorm 类型
        elif isinstance(module, nn.LayerNorm):
            # 初始化层归一化的偏置为零，权重为 1.0
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
# MEGA_START_DOCSTRING 是一个长字符串，用来描述此模型的文档字符串。
# 文档字符串提供了从 PreTrainedModel 继承的信息，包括通用方法（如下载、保存、调整输入嵌入、修剪头等）。
# 此模型还是一个 PyTorch 的 torch.nn.Module 子类，可以像常规 PyTorch 模块一样使用，并且可以参考 PyTorch 文档以获取一般使用和行为相关的所有信息。
# 参数：
#   config（MegaConfig）：模型配置类，包含模型的所有参数。使用配置文件初始化时不会加载与模型关联的权重，只加载配置。
#   使用 ~PreTrainedModel.from_pretrained 方法加载模型权重。
MEGA_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MegaConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# MEGA_INPUTS_DOCSTRING 是另一个字符串变量，暂时为空字符串，可能用于描述此模型的输入。
MEGA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列标记的索引，对应词汇表中的标记

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 遮罩用于避免在填充的标记索引上执行注意力操作。遮罩值在 `[0, 1]` 范围内：

            - 1 表示 **未被遮罩** 的标记，
            - 0 表示 **被遮罩** 的标记。

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 段标记索引，用于指示输入的第一部分和第二部分。索引选择在 `[0,1]` 范围内：

            - 0 对应于 *句子A* 的标记，
            - 1 对应于 *句子B* 的标记。
            此参数仅在模型初始化时设置了 `add_token_type_embeddings` 参数为 `True` 时可用。此张量中的所有值应始终 < config.type_vocab_size。

            [What are token type IDs?](../glossary#token-type-ids)
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选地，您可以直接传递嵌入表示，而不是传递 `input_ids`。如果您想对如何将 `input_ids` 索引转换为关联向量有更多控制，这将非常有用，而不是使用模型内部的嵌入查找矩阵。

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回的张量中的 `attentions`。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参见返回的张量中的 `hidden_states`。

        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""
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

    def __init__(self, config: MegaConfig, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # Initialize embedding layer specific to MegaModel
        self.embedding_layer = MegaEmbeddings(config)
        
        # Create multiple MegaBlocks (transformer blocks)
        self.layers = nn.ModuleList([MegaBlock(config) for _ in range(config.num_hidden_layers)])

        # Optionally add a pooling layer
        self.pooler = MegaPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing (retained from RoBERTa code)
        self.post_init()

    def get_input_embeddings(self):
        # Retrieve the word embedding layer
        return self.embedding_layer.word_embeddings

    def set_input_embeddings(self, value):
        # Set the word embedding layer with a new value
        self.embedding_layer.word_embeddings = value

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
@add_start_docstrings(
    """MEGA Model with a `language modeling` head on top for CLM fine-tuning.""", MEGA_START_DOCSTRING
)
class MegaForCausalLM(MegaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config: MegaConfig):
        # 调用父类构造函数初始化对象
        super().__init__(config)

        # 如果配置不是解码器，则发出警告信息
        if not config.is_decoder:
            logger.warning("If you want to use `MegaForCausalLM` as a standalone, add `is_decoder=True.`")

        # 创建 MegaModel 对象，不添加池化层
        self.mega = MegaModel(config, add_pooling_layer=False)

        # 根据配置决定是否添加 LM 隐藏层稠密层和激活函数
        if config.add_lm_hidden_dense_layer:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.hidden_activation = nn.Tanh()
        else:
            self.dense = None
            self.hidden_activation = None

        # 创建 LM 头部线性层，输出大小为词汇表大小
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入层（LM 头部线性层）
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入层（LM 头部线性层）
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 调用装饰器添加模型前向方法的文档字符串
    # 调用装饰器替换返回文档字符串，输出类型为 CausalLMOutputWithCrossAttentions，配置类为 _CONFIG_FOR_DOC
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
        # 方法用于为生成准备输入
        input_shape = input_ids.shape
        
        # 如果注意力掩码为空，则创建与输入形状相同的全 1 矩阵作为注意力掩码
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 如果使用过去的键值对，则截取输入的最后一个标记
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # 返回包含输入标记、注意力掩码和过去键值对的字典
        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}

    # 重新排序缓存中的过去键值对
    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
# 使用装饰器为 MegaForMaskedLM 类添加文档字符串，描述其作为带有语言建模头部的 MEGA 模型
@add_start_docstrings("""MEGA Model with a `language modeling` head on top.""", MEGA_START_DOCSTRING)
class MegaForMaskedLM(MegaPreTrainedModel):
    # 指定共享权重的键名，这里是语言建模头部的权重
    _tied_weights_keys = ["mlm_head.weight"]

    def __init__(self, config: MegaConfig):
        super().__init__(config)

        # 如果配置为解码器模式，发出警告，因为 MegaForMaskedLM 适合使用单向自注意力
        if config.is_decoder:
            logger.warning(
                "If you want to use `MegaForMaskedLM`, set `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 初始化 MegaModel，根据配置决定是否添加语言建模隐藏层的稠密层
        self.mega = MegaModel(config, add_pooling_layer=False)
        if config.add_lm_hidden_dense_layer:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.hidden_activation = nn.Tanh()
        else:
            self.dense = None
            self.hidden_activation = None
        
        # 语言建模头部，线性层将隐藏状态映射到词汇表大小
        self.mlm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout_prob)

        # 初始化权重并进行最终处理
        self.post_init()

    # 返回语言建模头部的输出嵌入
    def get_output_embeddings(self):
        return self.mlm_head

    # 设置语言建模头部的新嵌入
    def set_output_embeddings(self, new_embeddings):
        self.mlm_head = new_embeddings

    # 使用装饰器为 forward 方法添加文档字符串，描述输入参数及其作用
    # 还包括代码示例的文档字符串，展示输入、输出、期望输出和损失
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
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        # Decide whether to use return_dict based on provided value or default from configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass input arguments to the mega model for processing
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
        # Extract the sequence output from the mega model's outputs
        sequence_output = outputs[0]

        # Apply additional dense layer transformation if defined
        if self.dense is not None:
            sequence_output = self.dense(sequence_output)
            sequence_output = self.hidden_activation(sequence_output)

        # Generate prediction scores using the MLM head on the processed sequence output
        prediction_scores = self.mlm_head(sequence_output)

        # Initialize masked language modeling loss as None
        masked_lm_loss = None

        # Compute masked LM loss if labels are provided
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # Prepare output based on whether return_dict is False
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]  # Include prediction scores and other outputs
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # Return MaskedLMOutput with relevant components
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 基于 MEGA 模型的序列分类/回归头部的模型定义，使用线性层作为池化输出之上的顶层，例如用于GLUE任务。
@add_start_docstrings(
    """
    MEGA Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    MEGA_START_DOCSTRING,
)
class MegaForSequenceClassification(MegaPreTrainedModel):
    def __init__(self, config):
        # 调用父类构造函数初始化模型配置
        super().__init__(config)
        # 记录标签数和配置
        self.num_labels = config.num_labels
        self.config = config

        # 初始化 MEGA 模型，不包含池化层
        self.mega = MegaModel(config, add_pooling_layer=False)
        # 初始化 MEGA 分类头部
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
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 不为 None，则使用它；否则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给模型 `self.mega`
        outputs = self.mega(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取模型输出的序列输出
        sequence_output = outputs[0]
        # 将序列输出传递给分类器 `self.classifier` 得到 logits
        logits = self.classifier(sequence_output)

        # 初始化损失为 None
        loss = None
        # 如果 labels 不为 None，则计算损失
        if labels is not None:
            # 如果问题类型未定义，则根据情况设置问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择相应的损失函数
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

        # 如果 return_dict 为 False，则返回不同的输出形式
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回包含损失、logits、隐藏状态和注意力权重的 SequenceClassifierOutput 对象
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用指定的文档字符串注释装饰器，描述了此类的作用和结构
@add_start_docstrings(
    """
    MEGA Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    MEGA_START_DOCSTRING,  # 引用了全局变量 MEGA_START_DOCSTRING，补充了更多文档内容
)
# 定义 MegaForMultipleChoice 类，继承自 MegaPreTrainedModel
class MegaForMultipleChoice(MegaPreTrainedModel):
    def __init__(self, config):
        # 调用父类 MegaPreTrainedModel 的初始化方法
        super().__init__(config)

        # 创建一个 MegaModel 实例，接收给定的配置信息
        self.mega = MegaModel(config)
        # 创建一个 dropout 层，使用给定的隐藏层 dropout 概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建一个线性层，将隐藏状态大小映射到 1（用于多选题的分类）
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 执行初始化权重和应用最终处理
        self.post_init()

    # 使用指定的文档字符串注释装饰器，描述了 forward 方法的输入参数
    @add_start_docstrings_to_model_forward(MEGA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    # 使用指定的代码示例文档字符串注释装饰器，提供了 forward 方法的示例用法和输出类型
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,  # 引用了全局变量 _CHECKPOINT_FOR_DOC，指示可以使用的检查点
        output_type=MultipleChoiceModelOutput,  # 指定了输出类型为 MultipleChoiceModelOutput
        config_class=_CONFIG_FOR_DOC,  # 引用了全局变量 _CONFIG_FOR_DOC，指示可以使用的配置类
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 根据输入的 return_dict 参数确定是否返回字典格式的结果
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取输入中选择项的数量
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 将输入张量展平，以便适应模型输入要求
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 将展平后的输入传递给模型进行处理，并获取输出
        outputs = self.mega(
            flat_input_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取模型输出中的汇总向量
        pooled_output = outputs[1]

        # 对汇总向量应用 dropout
        pooled_output = self.dropout(pooled_output)
        # 将汇总后的向量输入到分类器中得到 logits
        logits = self.classifier(pooled_output)
        # 将 logits 重塑为原始的多选项形状
        reshaped_logits = logits.view(-1, num_choices)

        # 如果提供了 labels，则计算交叉熵损失
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 根据 return_dict 参数确定返回的结果格式
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回多选模型的输出，包括损失、logits、隐藏状态和注意力权重
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
"""
MEGA Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
Named-Entity-Recognition (NER) tasks.
"""

@add_start_docstrings(
    """
    MEGA Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    MEGA_START_DOCSTRING,
)
class MegaForTokenClassification(MegaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # Initialize the MEGA model with pooling layer excluded
        self.mega = MegaModel(config, add_pooling_layer=False)
        
        # Determine dropout rate for the classifier, fallback to config's hidden dropout prob if not specified
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        
        # Linear layer for token classification, output size equals config's hidden size and number of labels
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and perform any post initialization steps
        self.post_init()

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
        # Determine if return_dict is explicitly provided, otherwise use model's default setting
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward pass through the MEGA model to get outputs
        outputs = self.mega(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extract the sequence output from the MEGA model's outputs
        sequence_output = outputs[0]

        # Apply dropout to the sequence output
        sequence_output = self.dropout(sequence_output)
        
        # Generate logits for token classification using a linear layer
        logits = self.classifier(sequence_output)

        # Compute the loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # Return different output formats based on return_dict flag
        if not return_dict:
            output = (logits,) + outputs[2:]  # Include hidden states and attentions if not using return_dict
            return ((loss,) + output) if loss is not None else output
        else:
            # Return TokenClassifierOutput with relevant outputs
            return TokenClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
# 从transformers.models.roberta.modeling_roberta.RobertaClassificationHead复制，将Roberta改为Mega
class MegaClassificationHead(nn.Module):
    """用于句子级分类任务的头部模块。"""

    def __init__(self, config):
        super().__init__()
        # 全连接层，输入和输出维度都是config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 分类器的Dropout率，如果config.classifier_dropout不为None，则使用该值，否则使用config.hidden_dropout_prob
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # Dropout层
        self.dropout = nn.Dropout(classifier_dropout)
        # 输出投影层，将config.hidden_size映射到config.num_labels
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # 取features的第一个token的隐藏状态（相当于[CLS] token）
        x = features[:, 0, :]
        # 应用Dropout
        x = self.dropout(x)
        # 全连接层
        x = self.dense(x)
        # 使用tanh激活函数
        x = torch.tanh(x)
        # 再次应用Dropout
        x = self.dropout(x)
        # 输出投影层
        x = self.out_proj(x)
        return x


@add_start_docstrings(
    """
    MEGA模型，顶部具有一个用于抽取式问答任务（例如SQuAD）的跨度分类头部模块（在隐藏状态输出的线性层之上计算`跨度起始logits`和`跨度终止logits`）。
    """,
    MEGA_START_DOCSTRING,
)
class MegaForQuestionAnswering(MegaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 类别数目
        self.num_labels = config.num_labels

        # 使用MegaModel构建mega模型，不添加池化层
        self.mega = MegaModel(config, add_pooling_layer=False)
        # QA输出层，全连接层将config.hidden_size映射到config.num_labels
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(MEGA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
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
        # 如果 `return_dict` 不为 None，则使用其值；否则使用模型配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用模型的前向传播方法 `mega`，传递各种输入和参数
        outputs = self.mega(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从模型输出中获取序列输出 `sequence_output`
        sequence_output = outputs[0]

        # 将序列输出传递给模型的 QA 输出层，获得开始和结束位置的 logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果 `start_positions` 或 `end_positions` 是多维的，在第一维上进行压缩
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 忽略超出模型输入长度的位置
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 定义交叉熵损失函数，忽略索引为 `ignored_index` 的位置
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        # 如果 `return_dict` 为 False，则按照非字典返回格式构造输出
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 如果 `return_dict` 为 True，则构造 `QuestionAnsweringModelOutput` 对象返回
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```