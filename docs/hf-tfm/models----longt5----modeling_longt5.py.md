# `.\models\longt5\modeling_longt5.py`

```
# coding=utf-8
# Copyright 2022 Google LLC., LongT5 Authors and HuggingFace Inc. team.
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
""" PyTorch LongT5 model."""


import copy
import math
import warnings
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    logging,
    replace_return_docstrings,
)
from .configuration_longt5 import LongT5Config


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LongT5Config"
_CHECKPOINT_FOR_DOC = "google/long-t5-local-base"

# TODO: Update before the merge
LONGT5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/long-t5-local-base",
    "google/long-t5-local-large",
    "google/long-t5-tglobal-base",
    "google/long-t5-tglobal-large",
]


def _pad_to_multiple(x: torch.Tensor, block_len: int, dim: int, pad_value: int = 0) -> torch.Tensor:
    """
    Pad a tensor so that a sequence length will be a multiple of `block_len`.

    Args:
        x (torch.Tensor): Input tensor to be padded.
        block_len (int): Desired block length for padding.
        dim (int): Dimension along which padding will be applied.
        pad_value (int, optional): Value used for padding. Defaults to 0.

    Returns:
        torch.Tensor: Padded tensor.
    """
    pad_len = -x.shape[dim] % block_len
    # Handle cases when an empty input sequence is given
    if not all(x.shape):
        new_shape = list(x.shape)
        new_shape[dim] += pad_len
        return torch.zeros(new_shape, dtype=x.dtype)

    pad = [(0, 0)] * x.ndim
    pad[dim] = (0, pad_len)
    pad = sum(pad[::-1], ())
    x = nn.functional.pad(x, pad=pad, mode="constant", value=pad_value)
    return x


def _split_into_blocks(x: torch.Tensor, block_len: int, dim: int) -> torch.Tensor:
    """
    Split an input tensor into blocks of a given `block_len` along the given `dim`.
    If the dimension length is not a multiple of `block_len`, it will be padded first
    with selected `pad_value`.

    Args:
        x (torch.Tensor): Input tensor to be split into blocks.
        block_len (int): Length of each block.
        dim (int): Dimension along which to split the tensor.

    Returns:
        torch.Tensor: Tensor split into blocks.
    """
    # pad tensor to multiple of block_len
    if x.shape[dim] % block_len != 0:
        x = _pad_to_multiple(x, block_len, dim, pad_value=0)
    num_blocks = x.shape[dim] // block_len
    output_shape = x.shape[:dim] + (num_blocks, block_len) + x.shape[(dim + 1) :]
    # 如果输出形状中包含0，由于与ONNX转换不兼容，无法应用reshape操作
    if 0 in output_shape:
        # 返回一个空的张量，形状与output_shape相同，数据类型与x相同，设备与x相同
        return torch.empty(output_shape, dtype=x.dtype, device=x.device)
    # 否则，对张量x进行reshape操作，将其形状调整为output_shape
    return x.reshape(output_shape)
def _concatenate_3_blocks(x: torch.Tensor, block_dim: int, sequence_dim: int, pad_value: int = 0) -> torch.Tensor:
    """Concatenate three consecutive blocks for each input block for local attentiont.

    For more information, see: https://arxiv.org/pdf/2112.07916.pdf.
    """
    num_blocks = x.shape[block_dim]

    # Calculate padding configuration to expand the tensor dimensions
    pad = [(0, 0)] * x.ndim
    pad[block_dim] = (1, 1)
    pad = sum(pad[::-1], ())  # Convert pad list to tuple for torch.nn.functional.pad
    # Pad the tensor along block_dim dimension with constant pad_value
    x = nn.functional.pad(x, pad=pad, mode="constant", value=pad_value)

    blocks_list: List[torch.Tensor] = []
    for i in range(3):
        # Slice tensor to extract blocks using specified block_dim and num_blocks
        indices = [slice(0, None)] * x.ndim
        indices[block_dim] = slice(i, i + num_blocks)
        indices = tuple(indices)
        blocks_list.append(x[indices])
    # Concatenate blocks along sequence_dim to form a single tensor
    return torch.cat(blocks_list, dim=sequence_dim)


def _make_3block_relative_position_ids(block_len: int) -> torch.Tensor:
    """Makes 3-blocked relative position ids for local attention."""
    # Generate position ids for 3-blocked structure
    position_ids = torch.arange(3 * block_len, dtype=torch.int32)
    # Extract center position ids for relative positioning
    center_position_ids = position_ids[block_len:-block_len]
    # Compute relative position ids based on center positions
    relative_position_ids = position_ids.unsqueeze(0) - center_position_ids.unsqueeze(1)
    return relative_position_ids


def _mask_local_attention_mask(local_attention_mask: torch.Tensor, block_len: int) -> torch.Tensor:
    """Mask local attention mask to enforce that tokens are not allowed to attend tokens farther than ``local_radius."""
    # Generate relative position ids for 3-blocked structure
    relative_position_ids = _make_3block_relative_position_ids(block_len)
    # Create a mask enforcing local attention radius
    locality_mask = torch.abs(relative_position_ids) < block_len
    locality_mask = locality_mask[None, None, :, :]
    locality_mask = locality_mask.to(local_attention_mask.device)
    # Combine input attention mask and local attention mask
    return torch.logical_and(local_attention_mask, locality_mask)


def _get_local_attention_mask(attention_mask: torch.Tensor, block_len: int, device: torch.device) -> torch.Tensor:
    """Prepare attention mask to be applied for a local attention."""
    # Split attention_mask into blocks along the specified dimension
    _blocked_attention_mask = _split_into_blocks(attention_mask, block_len, dim=1)
    # Concatenate 3 blocks to form extended attention mask
    _3blocked_attention_mask = _concatenate_3_blocks(_blocked_attention_mask, block_dim=1, sequence_dim=2)

    _blocked_attention_mask = _blocked_attention_mask.unsqueeze(-1)
    _3blocked_attention_mask = _3blocked_attention_mask.unsqueeze(-2)
    # Create a local attention mask using logical operation
    local_attention_mask = torch.logical_and(_blocked_attention_mask, _3blocked_attention_mask)
    # Apply masking to enforce local attention constraints
    local_attention_mask = _mask_local_attention_mask(local_attention_mask, block_len)
    # Expand dimension to match the required output shape
    return local_attention_mask.unsqueeze(1).to(device)
# 根据输入的注意力掩码和全局块大小生成固定块的全局ID，并返回两个张量组成的元组
def _make_global_fixed_block_ids(
    attention_mask: torch.Tensor, global_block_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """获取每个输入标记对应的“固定块”全局ID。

    这个实现是从以下地址采用的 Flaxformr 原始实现的简化版本：
    https://github.com/google/flaxformer/blob/main/flaxformer/architectures/longt5/long_attention.py.

    在我们的场景中，由于我们仅用于解码器，孤立的标记（即不组成整个固定块的标记）将被分配到前一个块。

    原始序列中的填充标记由 -1 表示。
    """
    batch_size, seq_len = attention_mask.shape[:2]

    def handle_orphan_tokens(block_ids: torch.Tensor) -> torch.Tensor:
        # 确定每个块的结束位置，若为全局块大小减一，则为块的结尾
        block_ends = (torch.arange(seq_len) % global_block_size) == global_block_size - 1
        block_ends = block_ends.to(block_ids.device)
        true_block_ends = torch.logical_and(block_ends, block_ids >= 0)
        full_blocks = true_block_ends.sum(-1).unsqueeze(-1).type(block_ids.dtype) - 1
        block_ids = torch.where(block_ids < full_blocks, block_ids, full_blocks)
        return block_ids

    # 创建一个与注意力掩码相同大小的全1张量，然后对其进行累加，最后减去自身，以生成固定块掩码
    fixed_block_mask = torch.ones_like(attention_mask, device=attention_mask.device) / global_block_size
    fixed_block_mask = torch.cumsum(fixed_block_mask, axis=1) - fixed_block_mask

    # 将注意力掩码中非零元素替换为1.0，零元素替换为-1000.0，并生成全局块ID张量
    mask = torch.where(attention_mask != 0.0, 1.0, -1000.0).type(attention_mask.dtype)
    global_block_ids = torch.floor(mask + fixed_block_mask - 1.0).type(attention_mask.dtype)

    # 创建一个下界张量，并将全局块ID限制在下界以上
    _global_block_ids_lower_bound = torch.tensor(-1, dtype=global_block_ids.dtype, device=global_block_ids.device)
    global_block_ids = torch.where(
        global_block_ids > _global_block_ids_lower_bound, global_block_ids, _global_block_ids_lower_bound
    )

    # 将填充标记设为-1
    global_block_ids = (global_block_ids * attention_mask) + (attention_mask - 1)

    # 处理孤立标记，将它们分配到前一个块中
    global_block_ids = handle_orphan_tokens(global_block_ids)

    # 计算序列中全局块的数量
    num_globals = seq_len // global_block_size

    # 如果全局块数量大于0，则生成全局段ID张量，否则生成一个空张量
    if num_globals > 0:
        _sequence_block_ids_max = torch.max(global_block_ids, dim=-1).values.repeat(num_globals, 1).transpose(0, 1)
    else:
        _sequence_block_ids_max = torch.zeros(
            batch_size, 0, dtype=global_block_ids.dtype, device=global_block_ids.device
        )

    # 生成全局段ID张量，表示每个全局块的ID
    global_segment_ids = torch.cumsum(torch.ones(batch_size, num_globals), dim=-1) - 1
    global_segment_ids = global_segment_ids.to(attention_mask.device)
    global_segment_ids = torch.where(global_segment_ids <= _sequence_block_ids_max, 1, 0)

    # 返回全局块ID和全局段ID的整型张量
    return global_block_ids.type(torch.int), global_segment_ids.type(torch.int)


# 根据输入的注意力掩码和全局块大小创建用于局部到全局注意力的相对位置ID张量
def _make_side_relative_position_ids(attention_mask: torch.Tensor, global_block_size: int) -> torch.Tensor:
    """Create the relative position tensor for local -> global attention."""
    # 调用函数 _make_global_fixed_block_ids，生成全局固定块的标识和全局段标识
    block_ids, global_segment_ids = _make_global_fixed_block_ids(attention_mask, global_block_size)
    
    # 获取全局段标识数组的最后一个维度的长度，即全局序列的长度
    global_seq_len = global_segment_ids.shape[-1]
    
    # 在设备上创建一个张量，包含从0到全局序列长度的整数，设备与block_ids相同
    global_positions = torch.arange(global_seq_len, device=block_ids.device)
    
    # 计算侧向相对位置，即全局位置与每个块标识的差值，扩展维度以适应广播操作
    side_relative_position = global_positions - block_ids[..., None]
    
    # 将侧向相对位置转换为 torch.int64 类型并返回结果
    return side_relative_position.type(torch.int64)
def _create_global_aggregates(
    hidden_states: torch.Tensor, block_ids: torch.Tensor, global_seq_len: int
) -> torch.Tensor:
    """Compute individual block aggregates by summing over individual blocks."""
    # 将 block_ids 中小于 0 的值替换为 global_seq_len
    block_ids = block_ids.where(
        block_ids >= 0, torch.tensor(global_seq_len, dtype=block_ids.dtype, device=block_ids.device)
    )
    # 将 block_ids 转换成 one-hot 编码，维度为 (batch..., seq_len, global_seq_len)
    one_hot_block_ids = nn.functional.one_hot(block_ids.type(torch.int64), global_seq_len + 1)[:, :, :-1]
    # 使用 einsum 计算全局聚合，维度为 (...gd)，其中 g 是 global_seq_len，d 是 hidden_state 的最后一个维度
    return torch.einsum("...nd,...ng->...gd", hidden_states, one_hot_block_ids.type(hidden_states.dtype))


# 从 transformers.models.t5.modeling_t5.T5LayerNorm 复制到 LongT5LayerNorm
class LongT5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the LongT5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        # 初始化权重参数，用于缩放
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # 设置方差的小量值
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # 计算隐藏状态的方差，不减去均值，使用 fp32 累积半精度输入
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        # 根据方差和小量值计算 layer_norm，并缩放权重
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # 如果权重参数是半精度的，则将隐藏状态转换为相应精度
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


try:
    from apex.normalization import FusedRMSNorm

    LongT5LayerNorm = FusedRMSNorm  # noqa

    # 如果成功导入 FusedRMSNorm，使用它并记录信息
    logger.info("Discovered apex.normalization.FusedRMSNorm - will use it instead of LongT5LayerNorm")
except ImportError:
    # 如果导入失败，继续使用 LongT5LayerNorm
    pass
except Exception:
    # 如果发生其他异常，记录警告信息并回退到 LongT5LayerNorm
    logger.warning("discovered apex but it failed to load, falling back to LongT5LayerNorm")
    pass

# 将 LongT5LayerNorm 添加到全局变量 ALL_LAYERNORM_LAYERS 中
ALL_LAYERNORM_LAYERS.append(LongT5LayerNorm)


# 从 transformers.models.t5.modeling_t5.T5DenseActDense 复制到 LongT5DenseActDense
class LongT5DenseActDense(nn.Module):
    def __init__(self, config: LongT5Config):
        super().__init__()
        # 初始化两个线性层和一个 dropout 层，用于密集连接和激活
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        # 设置激活函数
        self.act = ACT2FN[config.dense_act_fn]
    # 定义前向传播函数，接受隐藏状态作为输入参数
    def forward(self, hidden_states):
        # 使用权重矩阵 wi 对隐藏状态进行线性变换
        hidden_states = self.wi(hidden_states)
        # 对变换后的隐藏状态应用激活函数 act
        hidden_states = self.act(hidden_states)
        # 对激活后的隐藏状态应用 dropout 操作
        hidden_states = self.dropout(hidden_states)
        # 检查输出层权重是否是张量类型，并且隐藏状态的数据类型是否与输出层权重不同，且输出层权重不是 torch.int8 类型
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            # 将隐藏状态转换到与输出层权重相同的数据类型
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        # 使用权重矩阵 wo 对处理后的隐藏状态进行线性变换
        hidden_states = self.wo(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states
# 定义一个名为 LongT5DenseGatedActDense 的 nn.Module 类，用于实现 LongT5 模型的 FeedForward 层
class LongT5DenseGatedActDense(nn.Module):
    def __init__(self, config: LongT5Config):
        super().__init__()
        # 使用 nn.Linear 定义一个线性层 wi_0，输入维度为 config.d_model，输出维度为 config.d_ff，无偏置
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        # 使用 nn.Linear 定义另一个线性层 wi_1，输入维度同上，输出维度同样为 config.d_ff，无偏置
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        # 使用 nn.Linear 定义一个线性层 wo，输入维度为 config.d_ff，输出维度为 config.d_model，无偏置
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        # 定义一个 Dropout 层，使用 config.dropout_rate 作为 dropout 概率
        self.dropout = nn.Dropout(config.dropout_rate)
        # 从全局 ACT2FN 字典中选择激活函数，函数由 config.dense_act_fn 决定
        self.act = ACT2FN[config.dense_act_fn]

    # 定义前向传播函数，接受 hidden_states 作为输入
    def forward(self, hidden_states):
        # 计算激活函数后的输出，输入为 wi_0 对 hidden_states 的线性变换结果
        hidden_gelu = self.act(self.wi_0(hidden_states))
        # 计算 wi_1 对 hidden_states 的线性变换结果
        hidden_linear = self.wi_1(hidden_states)
        # 将激活后的结果与线性变换结果相乘，作为下一步的隐藏状态
        hidden_states = hidden_gelu * hidden_linear
        # 对 hidden_states 应用 Dropout
        hidden_states = self.dropout(hidden_states)
        # 使用 wo 对 hidden_states 进行线性变换
        hidden_states = self.wo(hidden_states)
        # 返回变换后的 hidden_states
        return hidden_states


# 从 transformers.models.t5.modeling_t5.T5LayerFF 复制并修改为 LongT5
class LongT5LayerFF(nn.Module):
    def __init__(self, config: LongT5Config):
        super().__init__()
        # 根据配置选择是否使用门控激活函数，实例化对应的 Dense 层
        if config.is_gated_act:
            self.DenseReluDense = LongT5DenseGatedActDense(config)
        else:
            self.DenseReluDense = LongT5DenseActDense(config)

        # 使用 LongT5LayerNorm 对象对输入进行归一化
        self.layer_norm = LongT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 定义一个 Dropout 层，使用 config.dropout_rate 作为 dropout 概率
        self.dropout = nn.Dropout(config.dropout_rate)

    # 定义前向传播函数，接受 hidden_states 作为输入
    def forward(self, hidden_states):
        # 对输入 hidden_states 进行 LayerNorm 归一化
        forwarded_states = self.layer_norm(hidden_states)
        # 使用 DenseReluDense 层对归一化后的 hidden_states 进行前向传播
        forwarded_states = self.DenseReluDense(forwarded_states)
        # 将原始输入 hidden_states 与 Dropout 后的 forwarded_states 相加，作为输出
        hidden_states = hidden_states + self.dropout(forwarded_states)
        # 返回计算结果
        return hidden_states


# 从 transformers.models.t5.modeling_t5.T5Attention 复制并修改为 LongT5
class LongT5Attention(nn.Module):
    def __init__(self, config: LongT5Config, has_relative_attention_bias=False):
        super().__init__()
        # 根据配置初始化注意力模块的参数
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # 使用 nn.Linear 定义查询、键、值、输出映射层
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        # 如果使用相对位置注意力偏置，初始化相对注意力偏置 Embedding
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = False
    # 剪枝操作，根据给定的头部列表来剪枝
    def prune_heads(self, heads):
        # 如果头部列表为空，则直接返回，不进行剪枝操作
        if len(heads) == 0:
            return
        # 调用函数查找可剪枝的头部和对应索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # 对四个线性层进行剪枝操作
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # 更新超参数：减少头部数量、重新计算内部维度、更新已剪枝头部集合
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor - represents the relative position between memory and query
            bidirectional: a boolean - indicates if attention is bidirectional or unidirectional
            num_buckets: an integer - number of buckets to divide the range of relative positions
            max_distance: an integer - maximum allowed distance for relative positions

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        # Adjust num_buckets and calculate relative_buckets based on bidirectional flag
        if bidirectional:
            num_buckets //= 2
            # Calculate relative_buckets based on whether relative_position is positive
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            # Ensure all relative_position values are non-positive
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

        # now relative_position is in the range [0, inf)
        
        # Determine if relative_position falls within small or large range
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # Calculate bucket index based on whether the position is small or large
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        
        # Cap relative_position_if_large to num_buckets - 1
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        # Determine final relative_buckets using a conditional selection
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets
    # 计算相对位置偏置
    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        # 如果设备未指定，则使用相对注意力偏置权重的设备
        if device is None:
            device = self.relative_attention_bias.weight.device
        # 创建一个形状为 (query_length, 1) 的张量，表示查询位置
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        # 创建一个形状为 (1, key_length) 的张量，表示记忆位置
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        # 计算相对位置矩阵，形状为 (query_length, key_length)
        relative_position = memory_position - context_position
        # 将相对位置矩阵映射到预定义数量的桶中，形状仍为 (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # 使用相对位置桶计算相对注意力偏置，形状为 (query_length, key_length, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)
        # 将结果转置并增加维度，形状变为 (1, num_heads, query_length, key_length)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        # 返回计算得到的相对注意力偏置
        return values

    # 前向传播方法
    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    # 定义了一个名为 LongT5LocalAttention 的类，继承自 nn.Module
    class LongT5LocalAttention(nn.Module):
        def __init__(self, config: LongT5Config, has_relative_attention_bias: bool = False) -> None:
            super().__init__()
            # 初始化类的属性
            self.is_decoder = config.is_decoder  # 是否为解码器
            self.has_relative_attention_bias = has_relative_attention_bias  # 是否具有相对注意力偏置
            self.relative_attention_num_buckets = config.relative_attention_num_buckets  # 相对注意力的桶数量
            self.relative_attention_max_distance = config.relative_attention_max_distance  # 相对注意力的最大距离
            self.d_model = config.d_model  # 模型的输入维度
            self.key_value_proj_dim = config.d_kv  # 键值投影维度
            self.n_heads = config.num_heads  # 注意力头的数量
            self.local_radius = config.local_radius  # 局部注意力的半径
            self.block_len = self.local_radius + 1  # 区块长度
            self.dropout = config.dropout_rate  # 丢弃率
            self.inner_dim = self.n_heads * self.key_value_proj_dim  # 内部维度

            # 使用线性层定义模型的参数
            self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)  # 查询向量的线性层
            self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)  # 键向量的线性层
            self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)  # 值向量的线性层
            self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)  # 输出向量的线性层

            if self.has_relative_attention_bias:
                # 如果具有相对注意力偏置，使用嵌入层定义相对注意力偏置
                self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
            self.pruned_heads = set()  # 初始化剪枝的注意力头集合
            self.gradient_checkpointing = False  # 梯度检查点设为 False

        # 定义了一个方法 prune_heads，用于剪枝注意力头
        # 该方法的实现是从 transformers 库中 T5Attention 类的 prune_heads 方法复制而来
        def prune_heads(self, heads):
            if len(heads) == 0:
                return
            heads, index = find_pruneable_heads_and_indices(
                heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
            )
            # 剪枝线性层
            self.q = prune_linear_layer(self.q, index)
            self.k = prune_linear_layer(self.k, index)
            self.v = prune_linear_layer(self.v, index)
            self.o = prune_linear_layer(self.o, index, dim=1)
            # 更新超参数
            self.n_heads = self.n_heads - len(heads)
            self.inner_dim = self.key_value_proj_dim * self.n_heads
            self.pruned_heads = self.pruned_heads.union(heads)

        @staticmethod
        # 定义了一个静态方法 _relative_position_bucket，用于计算相对位置桶
        # 该方法的实现是从 transformers 库中 T5Attention 类的 _relative_position_bucket 方法复制而来
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor - 相对位置，表示从当前位置到被关注位置的距离（记忆位置 - 查询位置）
            bidirectional: a boolean - 是否为双向注意力
            num_buckets: an integer - 桶的数量，用来分桶相对位置
            max_distance: an integer - 最大距离，超过此距离的相对位置都映射到同一个桶内

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
            返回一个形状与relative_position相同的张量，其中的值在区间[0, num_buckets)内，表示相对位置所属的桶编号
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            # 计算相对位置的正负性，并将其映射到不同的桶
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            # 如果是单向注意力，将所有相对位置映射为非正的值
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # 现在relative_position的范围为[0, inf)

        # 半数桶用于精确增量位置
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # 另一半桶用于对数级别的更大距离位置
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        # 将超过最大桶数的相对位置映射到最大桶内
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        # 根据相对位置大小，选择对应的桶编号
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets
    def compute_bias(self, block_length: int):
        """Compute binned relative position bias"""
        # 获取self.relative_attention_bias的设备，如果不是"meta"类型则使用其设备，否则设为None
        target_device = (
            self.relative_attention_bias.weight.device
            if self.relative_attention_bias.weight.device.type != "meta"
            else None
        )
        # 创建一个长为3 * block_length的长整型张量，设备为target_device
        memory_position = torch.arange(3 * block_length, dtype=torch.long, device=target_device)
        # 从memory_position中选取出中间部分的位置作为context_position
        context_position = memory_position[block_length:-block_length]

        # 计算相对位置矩阵 (block_length, 3 * block_length)
        relative_position = memory_position[None, :] - context_position[:, None]
        # 将相对位置矩阵映射到桶中，返回相对位置桶的张量
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # (block_length, 3 * block_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # 计算相对注意力偏置值，维度为(block_length, 3 * block_length, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)
        # 对结果进行维度变换，维度变为(1, 1, num_heads, block_length, 3 * block_length)
        values = values.permute([2, 0, 1]).unsqueeze(0).unsqueeze(0)
        # 返回计算的相对注意力偏置张量
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        position_bias=None,
        layer_head_mask=None,
        output_attentions=False,
    # 定义一个名为 LongT5TransientGlobalAttention 的新的 PyTorch 模型类，继承自 nn.Module
    class LongT5TransientGlobalAttention(nn.Module):
        # 初始化函数，接受 LongT5Config 类型的配置对象 config 和一个布尔类型的参数 has_relative_attention_bias
        def __init__(self, config: LongT5Config, has_relative_attention_bias: bool = False) -> None:
            super().__init__()
            # 根据配置对象 config 设置模型的各种属性
            self.is_decoder = config.is_decoder
            self.has_relative_attention_bias = has_relative_attention_bias
            self.relative_attention_num_buckets = config.relative_attention_num_buckets
            self.relative_attention_max_distance = config.relative_attention_max_distance
            self.d_model = config.d_model
            self.key_value_proj_dim = config.d_kv
            self.n_heads = config.num_heads
            self.local_radius = config.local_radius
            self.block_len = self.local_radius + 1
            self.global_block_size = config.global_block_size
            self.dropout = config.dropout_rate
            self.inner_dim = self.n_heads * self.key_value_proj_dim

            # 使用 nn.Linear 定义线性层，设置输入维度为 d_model，输出维度为 inner_dim，没有偏置
            self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
            self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
            self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
            self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

            # 如果需要相对注意力偏置，使用 nn.Embedding 定义一个嵌入层，维度为 relative_attention_num_buckets x n_heads
            if self.has_relative_attention_bias:
                self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
            self.pruned_heads = set()

            # 如果需要全局注意力的相对注意力偏置，也使用 nn.Embedding 定义一个嵌入层
            if self.has_relative_attention_bias:
                self.global_relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
            # 使用 LongT5LayerNorm 类初始化全局注意力的输入层规范化
            self.global_input_layer_norm = LongT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

        # 定义一个用于剪枝注意力头的静态方法，接受参数 heads
        def prune_heads(self, heads):
            # 如果没有要剪枝的头部，直接返回
            if len(heads) == 0:
                return
            # 调用 find_pruneable_heads_and_indices 函数找到要剪枝的头部和索引
            heads, index = find_pruneable_heads_and_indices(
                heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
            )
            # 对线性层进行剪枝操作
            self.q = prune_linear_layer(self.q, index)
            self.k = prune_linear_layer(self.k, index)
            self.v = prune_linear_layer(self.v, index)
            self.o = prune_linear_layer(self.o, index, dim=1)
            # 更新超参数
            self.n_heads = self.n_heads - len(heads)
            self.inner_dim = self.key_value_proj_dim * self.n_heads
            self.pruned_heads = self.pruned_heads.union(heads)

        # 静态方法，用于计算相对位置桶的索引
        @staticmethod
        def _relative_position_bucket():
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor - represents the relative position between memory and query
            bidirectional: a boolean - whether the attention is bidirectional or unidirectional
            num_buckets: an integer - number of buckets to quantize relative positions into
            max_distance: an integer - maximum distance that a relative position can have

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        # Initialize relative_buckets to 0
        relative_buckets = 0
        
        # Adjust num_buckets if bidirectional is True
        if bidirectional:
            num_buckets //= 2
            # Compute relative_buckets based on whether relative_position is positive
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            # Ensure relative_position is non-positive for unidirectional attention
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        
        # now relative_position is in the range [0, inf)

        # Determine if relative_position is small (less than max_exact)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # Calculate relative position if it is large (greater than or equal to max_exact)
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        
        # Cap relative_position_if_large to num_buckets - 1
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        # Combine relative_buckets based on whether relative_position is small or large
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        
        # Return the computed relative_buckets
        return relative_buckets
    def compute_bias(self, block_length: int):
        """Compute binned relative position bias"""
        # 确定目标设备，以便在设备类型不是"meta"时将其用于计算
        target_device = (
            self.relative_attention_bias.weight.device
            if self.relative_attention_bias.weight.device.type != "meta"
            else None
        )
        # 创建一个长为 3 * block_length 的长整型张量，用于表示内存位置
        memory_position = torch.arange(3 * block_length, dtype=torch.long, device=target_device)
        # 根据 block_length 计算上下文位置，排除边界块长度
        context_position = memory_position[block_length:-block_length]

        # 计算相对位置张量 (block_length, 3 * block_length)
        relative_position = memory_position[None, :] - context_position[:, None]
        # 将相对位置映射到桶中，以便后续使用
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # 使用相对位置桶计算相对注意力偏置值 (block_length, 3 * block_length, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)
        # 调整维度顺序以匹配模型期望的格式 (1, 1, num_heads, block_length, 3 * block_length)
        values = values.permute([2, 0, 1]).unsqueeze(0).unsqueeze(0)
        return values

    def compute_side_bias(self, mask: torch.Tensor, global_segment_ids: torch.Tensor) -> torch.Tensor:
        # 创建侧边注意力掩码 (batch_size, 1, seq_len, global_seq_len)
        side_attention_mask = torch.eq(mask[..., None], global_segment_ids[:, None, :])[:, None, ...]
        # 设置侧边注意力偏置，匹配掩码中的条件 (batch_size, 1, seq_len, global_seq_len)
        attention_side_bias = torch.where(side_attention_mask > 0, 0.0, -1e10)
        # 创建侧边相对位置张量 (batch_size, seq_len, global_seq_len)
        side_relative_position = _make_side_relative_position_ids(mask, self.global_block_size)
        # 将侧边相对位置映射到桶中，以备后续使用
        side_relative_position_bucket = self._relative_position_bucket(
            side_relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # 使用全局相对注意力偏置计算侧边注意力偏置 (batch_size, seq_len, global_seq_len, num_heads)
        side_bias = self.global_relative_attention_bias(side_relative_position_bucket)
        # 调整顺序以匹配模型期望的格式 (batch_size, num_heads, seq_len, global_seq_len)
        side_bias = side_bias.permute([0, 3, 1, 2])
        # 将侧边注意力偏置与注意力偏置相结合 (batch_size, num_heads, seq_len, global_seq_len)
        attention_side_bias = attention_side_bias + side_bias
        return attention_side_bias

    def forward(
        self,
        hidden_states,
        mask=None,
        position_bias=None,
        layer_head_mask=None,
        output_attentions=False,
# 从transformers.models.t5.modeling_t5.T5LayerSelfAttention复制，并将T5->LongT5
class LongT5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        # 初始化长序列Transformer的自注意力层
        self.SelfAttention = LongT5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        # 初始化层归一化
        self.layer_norm = LongT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 初始化dropout层
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        # 应用层归一化到隐藏状态
        normed_hidden_states = self.layer_norm(hidden_states)
        # 使用自注意力层处理归一化后的隐藏状态
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 将dropout后的注意力输出与原始隐藏状态相加
        hidden_states = hidden_states + self.dropout(attention_output[0])
        # 如果输出注意力权重，将它们添加到输出元组中
        outputs = (hidden_states,) + attention_output[1:]  # 如果输出注意力权重，则添加它们
        return outputs


class LongT5LayerLocalSelfAttention(nn.Module):
    """用于编码器的局部自注意力"""

    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        # 初始化长序列Transformer的局部自注意力层
        self.LocalSelfAttention = LongT5LocalAttention(config, has_relative_attention_bias=has_relative_attention_bias)
        # 初始化层归一化
        self.layer_norm = LongT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 初始化dropout层
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        output_attentions=False,
        **kwargs: Any,  # 接受past_key_value和use_cache等关键字参数
    ):
        # 应用层归一化到隐藏状态
        normed_hidden_states = self.layer_norm(hidden_states)
        # 使用局部自注意力层处理归一化后的隐藏状态
        attention_output = self.LocalSelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 将dropout后的注意力输出与原始隐藏状态相加
        hidden_states = hidden_states + self.dropout(attention_output[0])
        # 如果输出注意力权重，将它们添加到输出元组中
        outputs = (hidden_states,) + attention_output[1:]  # 如果输出注意力权重，则添加它们
        return outputs


class LongT5LayerTransientGlobalSelfAttention(nn.Module):
    """用于编码器的瞬时全局自注意力"""
    # 初始化函数，用于创建一个新的实例
    def __init__(self, config, has_relative_attention_bias=False):
        # 调用父类的初始化方法
        super().__init__()
        # 使用LongT5TransientGlobalAttention类初始化TransientGlobalSelfAttention属性，
        # 并传入config和has_relative_attention_bias参数
        self.TransientGlobalSelfAttention = LongT5TransientGlobalAttention(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        # 初始化layer_norm属性，使用LongT5LayerNorm类创建一个LayerNorm层，
        # 将d_model作为参数传入，同时设置eps为config中的layer_norm_epsilon值
        self.layer_norm = LongT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 初始化dropout属性，使用nn.Dropout类创建一个Dropout层，
        # 将dropout_rate作为参数传入，dropout_rate从config中获取
        self.dropout = nn.Dropout(config.dropout_rate)

    # 前向传播函数，定义了模型的数据流向
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        output_attentions=False,
        **kwargs: Any,  # 接受past_key_value和use_cache等其他关键字参数
    ):
        # 对输入的hidden_states进行LayerNorm归一化处理
        normed_hidden_states = self.layer_norm(hidden_states)
        # 将归一化后的hidden_states输入到TransientGlobalSelfAttention模块中进行处理，
        # 并传入attention_mask、position_bias、layer_head_mask和output_attentions等参数
        attention_output = self.TransientGlobalSelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 将原始的hidden_states与经Dropout处理后的attention_output[0]相加，得到更新后的hidden_states
        hidden_states = hidden_states + self.dropout(attention_output[0])
        # 将更新后的hidden_states作为第一个元素，同时保留attention_output的其余部分（如果有的话）
        # 组成输出的元组outputs
        outputs = (hidden_states,) + attention_output[1:]  # 如果有需要，可以添加注意力信息
        # 返回outputs作为前向传播的结果
        return outputs
# 从 transformers.models.t5.modeling_t5.T5LayerCrossAttention 复制而来，将 T5 替换为 LongT5
class LongT5LayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化 LongT5Attention 作为 Encoder-Decoder 注意力层
        self.EncDecAttention = LongT5Attention(config, has_relative_attention_bias=False)
        # 使用 LongT5LayerNorm 对隐藏状态进行层归一化
        self.layer_norm = LongT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 使用 nn.Dropout 进行 dropout 正则化
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        # 对隐藏状态进行层归一化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 进行 Encoder-Decoder 注意力计算
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        # 计算最终层输出，包括 dropout 正则化
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # 如果有需要，则添加注意力输出
        return outputs


class LongT5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        # 根据配置选择适当的注意力层
        if config.is_decoder:
            attention_layer = LongT5LayerSelfAttention
        elif config.encoder_attention_type == "local":
            attention_layer = LongT5LayerLocalSelfAttention
        elif config.encoder_attention_type == "transient-global":
            attention_layer = LongT5LayerTransientGlobalSelfAttention
        else:
            raise ValueError(
                "For encoder attention mechanism, either `local` or `transient-global` attention type is expected, "
                f"but got {config.encoder_attention_type}."
            )
        # 初始化 LongT5Block 的各层
        self.layer = nn.ModuleList()
        self.layer.append(attention_layer(config, has_relative_attention_bias=has_relative_attention_bias))
        # 如果是解码器，添加交叉注意力层
        if self.is_decoder:
            self.layer.append(LongT5LayerCrossAttention(config))

        # 添加前馈神经网络层
        self.layer.append(LongT5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):
        # 依次传递每一层的输入与参数，计算输出
        for layer_module in self.layer:
            hidden_states = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                return_dict=return_dict,
            )
        return hidden_states
    """
    模型的基类，LongT5PreTrainedModel，继承自父类 T5PreTrainedModel，并针对 LongT5 进行特定设置和调整。
    """

    # 指定模型配置类为 LongT5Config
    config_class = LongT5Config
    # 基础模型前缀设置为 "transformer"
    base_model_prefix = "transformer"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不需要拆分的模块列表，这里包含 "LongT5Block"
    _no_split_modules = ["LongT5Block"]

    @property
    # 从 transformers.models.t5.modeling_t5.T5PreTrainedModel.dummy_inputs 复制的属性和注释
    def dummy_inputs(self):
        # 创建一个包含虚拟输入的张量 input_ids 和 input_mask
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        # 构建虚拟输入字典 dummy_inputs
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    # 从 transformers.models.t5.modeling_t5.T5PreTrainedModel._shift_right 复制的方法和注释，但用 LongT5 替代了 T5
    def _shift_right(self, input_ids):
        # 获取解码器的起始标记 ID 和填充标记 ID
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        if decoder_start_token_id is None:
            # 如果未定义 decoder_start_token_id，则抛出 ValueError
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In LongT5 it is usually set to the pad_token_id. "
                "See LongT5 docs for more information."
            )

        # 将输入向右移动一位
        if is_torch_fx_proxy(input_ids):
            # 对于 TorchFX 代理，不支持原生的项目赋值操作
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            # 如果未定义 pad_token_id，则抛出 ValueError
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        
        # 将标签中可能存在的 -100 值替换为 pad_token_id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids
class LongT5Stack(LongT5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)  # 创建词嵌入层，大小为vocab_size × d_model
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight  # 如果传入了自定义的词嵌入，则使用这些权重
        self.is_decoder = config.is_decoder  # 是否为解码器模式的标志

        self.local_radius = config.local_radius  # 获取局部注意力的半径大小
        self.block_len = self.local_radius + 1  # 计算每个块的长度，包括中心位置和左右邻居

        self.block = nn.ModuleList(
            [LongT5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )  # 创建长T5块的列表，每个块都包含一个长T5块模型
        self.final_layer_norm = LongT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)  # 创建最终的层归一化层
        self.dropout = nn.Dropout(config.dropout_rate)  # 创建dropout层，用于正则化

        self.gradient_checkpointing = False  # 梯度检查点标志，默认为False，表示不使用梯度检查点

        # Initialize weights and apply final processing
        self.post_init()  # 执行后初始化方法，用于权重初始化和最终处理

    # Copied from transformers.models.t5.modeling_t5.T5Stack.get_input_embeddings
    def get_input_embeddings(self):
        return self.embed_tokens  # 返回输入的词嵌入层对象

    # Copied from transformers.models.t5.modeling_t5.T5Stack.set_input_embeddings
    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings  # 设置新的输入词嵌入层对象

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
LONGT5_START_DOCSTRING = r"""
    The LongT5 model was proposed in [LongT5: Efficient Text-To-Text Transformer for Long
    Sequences](https://arxiv.org/abs/2112.07916) by Mandy Guo, Joshua Ainslie, David Uthus, Santiago Ontanon, Jianmo
    Ni, Yun-Hsuan Sung and Yinfei Yang. It's an encoder-decoder transformer pre-trained in a text-to-text denoising
    generative setting. LongT5 model is an extension of T5 model, and it enables using one of the two different
    efficient attention mechanisms - (1) Local attention, or (2) Transient-Global attention.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LongT5Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
# 定义长T5模型输入文档字符串常量
LONGT5_INPUTS_DOCSTRING = r"""
"""

# 定义长T5编码器输入文档字符串常量
LONGT5_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            输入序列标记在词汇表中的索引。LongT5模型具有相对位置嵌入，因此可以在左右两侧进行填充。

            索引可以使用[`AutoTokenizer`]获得。参见[`PreTrainedTokenizer.encode`]和
            [`PreTrainedTokenizer.__call__`]了解详情。

            如何为预训练准备`input_ids`的更多信息，请参阅[LONGT5
            Training](./longt5#training)。
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            避免在填充标记索引上执行注意力的掩码。掩码的值在`[0, 1]`范围内选择：

            - 1 表示**未被掩码**的标记，
            - 0 表示**被掩码**的标记。

            [什么是注意力掩码？](../glossary#attention-mask)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            用于空置自注意力模块中选择头部的掩码。掩码的值在`[0, 1]`范围内选择：

            - 1 表示头部**未被掩码**，
            - 0 表示头部**被掩码**。

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            可选，可以直接传递嵌入表示而不是`input_ids`。如果您希望更多控制如何将`input_ids`索引转换为关联向量，
            而不是使用模型的内部嵌入查找矩阵，这将非常有用。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关更多细节，请参见返回张量中的`attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关更多细节，请参见返回张量中的`hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回[`~utils.ModelOutput`]而不是普通元组。
"""

# 未来警告的警告消息：head_mask被拆分为两个输入参数 - head_mask和decoder_head_mask
__HEAD_MASK_WARNING_MSG = """
输入参数`head_mask`已分成两个参数`head_mask`和`decoder_head_mask`。当前，
`decoder_head_mask`设置为复制`head_mask`，但此功能已被弃用，并将在未来版本中删除。
如果您现在不想使用任何`decoder_head_mask`，请设置`decoder_head_mask = torch.ones(num_layers, num_heads)`。
"""


@add_start_docstrings(
    "The bare LONGT5 Model transformer outputting raw hidden-states without any specific head on top.",
    LONGT5_START_DOCSTRING,
)
class LongT5Model(LongT5PreTrainedModel):
    # 长T5模型类，输出裸的隐藏状态，没有特定的输出头部。
    # 在加载时忽略的关键键列表，用于意外情况下的模型加载
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    # 被绑定权重的键列表，这些权重将被共享
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: LongT5Config):
        # 调用父类构造函数，初始化模型配置
        super().__init__(config)
        # 共享的词嵌入层，用于编码器和解码器
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 复制编码器配置并初始化编码器
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = LongT5Stack(encoder_config, self.shared)

        # 复制解码器配置并初始化解码器
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = LongT5Stack(decoder_config, self.shared)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回共享的输入词嵌入层
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        # 设置新的输入词嵌入层，并更新编码器和解码器的输入词嵌入层
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        # 如果配置要求，绑定编码器和解码器的词嵌入权重
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    def get_encoder(self):
        # 返回编码器模型
        return self.encoder

    def get_decoder(self):
        # 返回解码器模型
        return self.decoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 剪枝模型中的注意力头
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(LONGT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 使用 `add_start_docstrings` 装饰器为模型添加长文档字符串，描述其作为具有顶部语言建模头的 LONGT5 模型
@add_start_docstrings("""LONGT5 Model with a `language modeling` head on top.""", LONGT5_START_DOCSTRING)
class LongT5ForConditionalGeneration(LongT5PreTrainedModel):
    # 在加载时忽略的键列表，用于意外情况下不加载的键
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    # 要绑定权重的键列表，包括 encoder 和 decoder 的嵌入权重以及 lm_head 的权重
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    # 初始化方法，接受一个 LongT5Config 类型的参数 config
    def __init__(self, config: LongT5Config):
        super().__init__(config)
        # 设置模型维度为配置中的 d_model
        self.model_dim = config.d_model

        # 共享的嵌入层，使用 nn.Embedding 初始化，将词汇表大小设为 config.vocab_size，嵌入维度为 config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 复制 encoder 配置并设置为非解码器，不使用缓存，且非编码-解码器模式
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # 使用 LongT5Stack 类构建编码器
        self.encoder = LongT5Stack(encoder_config, self.shared)

        # 复制 decoder 配置并设置为解码器，且非编码-解码器模式，层数为 config.num_decoder_layers
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        # 使用 LongT5Stack 类构建解码器
        self.decoder = LongT5Stack(decoder_config, self.shared)

        # 使用 nn.Linear 初始化 lm_head，输入维度为 config.d_model，输出维度为 config.vocab_size，无偏置
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层 shared
    def get_input_embeddings(self):
        return self.shared

    # 设置输入嵌入层 shared 的新嵌入
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        # 更新 encoder 和 decoder 的输入嵌入
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    # 绑定权重方法，如果配置中设置了 tie_word_embeddings，则共享 encoder 和 decoder 的嵌入权重
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    # 设置 lm_head 的输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 获取 lm_head 的输出嵌入
    def get_output_embeddings(self):
        return self.lm_head

    # 获取编码器
    def get_encoder(self):
        return self.encoder

    # 获取解码器
    def get_decoder(self):
        return self.decoder

    # 使用 `add_start_docstrings_to_model_forward` 和 `replace_return_docstrings` 装饰器为模型的 forward 方法添加文档字符串，
    # 描述其输入参数和输出类型
    @add_start_docstrings_to_model_forward(LONGT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 定义模型的前向传播方法，接受多个可选的输入参数和掩码
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 如果 past_key_values 不为 None，根据其值裁剪 decoder_input_ids
        if past_key_values is not None:
            # 获取过去键值对的长度
            past_length = past_key_values[0][0].shape[2]

            # 如果 input_ids 的长度大于过去长度，则移除前缀长度
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 否则，默认保留最后一个 ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 裁剪 input_ids，保留后缀部分
            input_ids = input_ids[:, remove_prefix_length:]

        # 返回包含各种输入和掩码信息的字典，用于生成阶段的输入准备
        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    # 根据标签生成解码器的输入 IDs
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        # 调用内部方法 _shift_right 将标签向右移动，用作解码器的输入
        return self._shift_right(labels)
    def _reorder_cache(self, past_key_values, beam_idx):
        # 如果过去的键值不包含在输出中
        # 则关闭快速解码并且无需重新排序
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        # 重新排序后的解码器过去状态
        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # 从层过去状态中获取正确的批次索引
            # `past` 的批次维度在第二个位置
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # 需要为每个键/值状态设置正确的 `past`
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past
# 定义一个LongT5EncoderModel类，继承自LongT5PreTrainedModel，表示不带顶部特定头部的LONGT5模型编码器的原始隐藏状态输出
class LongT5EncoderModel(LongT5PreTrainedModel):
    # 在加载时需要绑定权重的键名列表
    _tied_weights_keys = ["encoder.embed_tokens.weight"]
    # 在加载时忽略的意外键名列表，匹配到"decoder"的键名将被忽略
    _keys_to_ignore_on_load_unexpected = [r"decoder"]

    # 初始化方法，接受一个LongT5Config类型的配置参数config
    def __init__(self, config: LongT5Config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建一个共享的词嵌入层，词汇量大小为config.vocab_size，嵌入维度为config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 复制配置参数config，设置不使用缓存，不是编码器-解码器结构
        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # 创建LongT5Stack编码器，使用上述配置和共享的词嵌入层self.shared
        self.encoder = LongT5Stack(encoder_config, self.shared)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入的词嵌入层对象
    def get_input_embeddings(self):
        return self.shared

    # 设置输入的词嵌入层对象为new_embeddings，并将其传递给编码器self.encoder
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    # 绑定权重的内部方法，如果配置参数中tie_word_embeddings为True，则绑定编码器的嵌入层和共享的词嵌入层self.shared
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)

    # 获取编码器self.encoder
    def get_encoder(self):
        return self.encoder

    # 剪枝模型中的注意力头部方法，heads_to_prune为一个字典，格式为{层编号: 需要剪枝的注意力头部列表}
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历需要剪枝的层和头部信息，调用self.encoder的具体层的注意力对象的prune_heads方法进行剪枝
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 前向传播方法，接受多个可选的输入参数，并返回一个BaseModelOutput类型的对象
    @add_start_docstrings_to_model_forward(LONGT5_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

        # 省略了具体实现细节，将会使用LongT5Stack编码器处理输入，并返回模型输出
        pass
        ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        r"""
        返回一个元组，包含 torch.FloatTensor 或 BaseModelOutput 类型的对象。

        Returns:
        返回模型的输出结果。

        Example:
        示例代码展示如何使用这个模型：

        ```python
        >>> from transformers import AutoTokenizer, LongT5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("google/long-t5-local-base")
        >>> model = LongT5EncoderModel.from_pretrained("google/long-t5-local-base")
        >>> input_ids = tokenizer(
        ...     100 * "Studies have been shown that owning a dog is good for you ", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model(input_ids=input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用编码器（encoder）模块进行前向传播，生成编码器的输出
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 返回编码器的输出作为最终函数的返回值
        return encoder_outputs
```