# `.\transformers\models\longt5\modeling_longt5.py`

```py
# 设置脚本编码为utf-8
# 版权信息声明
# 启用Apache许可证2.0，不得使用此文件除非符合许可证的规定
# 查看许可证的副本，访问http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据"基础上"的条件分发软件
# 按照"原样"分布，没有任何明示或暗示的保证或条件
# 请查看许可证以获取详细的规则和限制
"""PyTorch LongT5 模型。"""


import copy  # 导入copy模块，用于深拷贝对象
import math  # 导入math模块，执行数学运算
import warnings  # 导入warnings模块，用于处理警告
from typing import Any, List, Optional, Tuple, Union  # 导入类型提示需要的模块

import torch  # 导入torch，PyTorch的主要模块
from torch import nn  # 导入torch.nn，PyTorch中神经网络模块
from torch.nn import CrossEntropyLoss  # 导入CrossEntropyLoss，用于计算损失

from ...activations import ACT2FN  # 导入激活函数
from ...modeling_outputs import (  # 导入模型输出相关
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import PreTrainedModel  # 导入预训练模型相关工具
from ...pytorch_utils import ALL_LAYERNORM_LAYERS, find_pruneable_heads_and_indices, prune_linear_layer  # 导入PyTorch工具
from ...utils import (  # 导入常用工具函数
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    logging,
    replace_return_docstrings,
)
from .configuration_longt5 import LongT5Config  # 导入LongT5模型配置


logger = logging.get_logger(__name__)  # 获取logger对象记录日志

_CONFIG_FOR_DOC = "LongT5Config"  # 用于文档的配置信息
_CHECKPOINT_FOR_DOC = "google/long-t5-local-base"  # 用于文档的检查点信息

# TODO: Merge前更新
LONGT5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/long-t5-local-base",
    "google/long-t5-local-large",
    "google/long-t5-tglobal-base",
    "google/long-t5-tglobal-large",
]  # 预训练模型的列表


def _pad_to_multiple(x: torch.Tensor, block_len: int, dim: int, pad_value: int = 0) -> torch.Tensor:
    """将张量填充到序列长度为`block_len`的倍数"""
    pad_len = -x.shape[dim] % block_len
    # 处理给定空输入序列的情况
    if not all(x.shape):
        new_shape = list(x.shape)
        new_shape[dim] += pad_len
        return torch.zeros(new_shape, dtype=x.dtype)

    pad = [(0, 0)] * x.ndim  # 创建填充列表，维度与输入张量相同
    pad[dim] = (0, pad_len)  # 在指定维度上进行填充
    pad = sum(pad[::-1], ())  # 将填充列表压缩成一维
    x = nn.functional.pad(x, pad=pad, mode="constant", value=pad_value)  # 使用常量值填充张量
    return x


def _split_into_blocks(x: torch.Tensor, block_len: int, dim: int) -> torch.Tensor:
    """将输入张量沿指定维度`dim`分割为给定`block_len`大小的块。如果维度长度不是`block_len`的倍数，将首先进行填充"""
    # 将张量填充到`block_len`的倍数
    if x.shape[dim] % block_len != 0:
        x = _pad_to_multiple(x, block_len, dim, pad_value=0)
    num_blocks = x.shape[dim] // block_len
    output_shape = x.shape[:dim] + (num_blocks, block_len) + x.shape[(dim + 1) :]  # 计算输出张量的形状
    # 如果输出形状中包含 0，由于与 ONNX 转换不兼容，无法应用reshape
    if 0 in output_shape:
        # 返回一个空的张量，形状与output_shape相同，dtype与x相同，设备与x相同
        return torch.empty(output_shape, dtype=x.dtype, device=x.device)
    # 使用给定的输出形状对输入张量进行reshape操作
    return x.reshape(output_shape)
# 定义一个函数，将每个输入块进行3块连接，用于局部注意力机制
def _concatenate_3_blocks(x: torch.Tensor, block_dim: int, sequence_dim: int, pad_value: int = 0) -> torch.Tensor:
    """Concatenate three consecutive blocks for each input block for local attentiont.
    详细信息参见: https://arxiv.org/pdf/2112.07916.pdf.
    """
    # 获取输入张量的块数
    num_blocks = x.shape[block_dim]

    # 构造填充值
    pad = [(0, 0)] * x.ndim
    pad[block_dim] = (1, 1)
    pad = sum(pad[::-1], ())
    # 扩展张量的维度 [batch_size, num_blocks, block_len] -> [batch_size, num_blocks + 2, block_len]
    x = nn.functional.pad(x, pad=pad, mode="constant", value=pad_value)

    blocks_list: List[torch.Tensor] = []
    for i in range(3):
        # 使用索引访问张量
        indices = [slice(0, None)] * x.ndim
        indices[block_dim] = slice(i, i + num_blocks)
        indices = tuple(indices)
        blocks_list.append(x[indices])
    # 返回连接后的张量 [batch_size, num_blocks, 3 * block_len, ...]
    return torch.cat(blocks_list, dim=sequence_dim)


# 构造3块相对位置ID用于局部注意力机制
def _make_3block_relative_position_ids(block_len: int) -> torch.Tensor:
    """Makes 3-blocked relative position ids for local attention."""
    # 构造位置ID
    position_ids = torch.arange(3 * block_len, dtype=torch.int32)
    # 获取中心位置的ID
    center_position_ids = position_ids[block_len:-block_len]
    # 构造3块相对位置ID [block_len, 3 * block_len]
    relative_position_ids = position_ids.unsqueeze(0) - center_position_ids.unsqueeze(1)
    return relative_position_ids


# 局部注意力掩码，限制不允许的注意力范围
def _mask_local_attention_mask(local_attention_mask: torch.Tensor, block_len: int) -> torch.Tensor:
    """Mask local attention mask to enforce that tokens are not allowed to attend tokens farther than ``local_radius."""
    # 使用3块相对位置ID构造局部性的掩码
    relative_position_ids = _make_3block_relative_position_ids(block_len)
    locality_mask = torch.abs(relative_position_ids) < block_len
    locality_mask = locality_mask[None, None, :, :]
    locality_mask = locality_mask.to(local_attention_mask.device)
    return torch.logical_and(local_attention_mask, locality_mask)


# 获取适用于局部注意力的注意力掩码
def _get_local_attention_mask(attention_mask: torch.Tensor, block_len: int, device: torch.device) -> torch.Tensor:
    """Prepare attention mask to be applied for a local attention."""
    # 将输入的整体注意力掩码拆分为块状的注意力掩码
    _blocked_attention_mask = _split_into_blocks(attention_mask, block_len, dim=1)
    # 将块状的注意力掩码连接为3块
    _3blocked_attention_mask = _concatenate_3_blocks(_blocked_attention_mask, block_dim=1, sequence_dim=2)

    _blocked_attention_mask = _blocked_attention_mask.unsqueeze(-1)
    _3blocked_attention_mask = _3blocked_attention_mask.unsqueeze(-2)
    # 生成局部注意力掩码 [batch_size, num_block, block_len, 3 * block_len]
    local_attention_mask = torch.logical_and(_blocked_attention_mask, _3blocked_attention_mask)
    # 对局部注意力掩码进行掩码处理，限制注意力范围
    local_attention_mask = _mask_local_attention_mask(local_attention_mask, block_len)
    # 扩展局部注意力掩码的维度 [batch_size, 1, num_block, block_len, 3 * block_len]
    return local_attention_mask.unsqueeze(1).to(device)
# 根据注意力掩码和全局块大小创建每个输入标记对应的“固定块”全局编号
def _make_global_fixed_block_ids(
    attention_mask: torch.Tensor, global_block_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Obtain the "fixed block" global id corresponding to each input token.

    This implementation is a simlified version of the original Flaxformr implementation adopted from:
    https://github.com/google/flaxformer/blob/main/flaxformer/architectures/longt5/long_attention.py.

    在我们的情况下，由于我们仅将此策略用于解码器，孤立标记，即那些不能完全组成固定块的标记，将被分配给前导块。

    原始序列中的填充标记用-1表示。
    """
    # 获取批次大小和序列长度
    batch_size, seq_len = attention_mask.shape[:2]

    # 处理孤立标记的函数
    def handle_orphan_tokens(block_ids: torch.Tensor) -> torch.Tensor:
        block_ends = (torch.arange(seq_len) % global_block_size) == global_block_size - 1
        block_ends = block_ends.to(block_ids.device)
        true_block_ends = torch.logical_and(block_ends, block_ids >= 0)
        full_blocks = true_block_ends.sum(-1).unsqueeze(-1).type(block_ids.dtype) - 1
        block_ids = torch.where(block_ids < full_blocks, block_ids, full_blocks)
        return block_ids

    # 创建固定块掩码
    fixed_block_mask = torch.ones_like(attention_mask, device=attention_mask.device) / global_block_size
    fixed_block_mask = torch.cumsum(fixed_block_mask, axis=1) - fixed_block_mask
    mask = torch.where(attention_mask != 0.0, 1.0, -1000.0).type(attention_mask.dtype)
    global_block_ids = torch.floor(mask + fixed_block_mask - 1.0).type(attention_mask.dtype)
    _global_block_ids_lower_bound = torch.tensor(-1, dtype=global_block_ids.dtype, device=global_block_ids.device)
    global_block_ids = torch.where(
        global_block_ids > _global_block_ids_lower_bound, global_block_ids, _global_block_ids_lower_bound
    )
    # 将填充标记设置为-1
    global_block_ids = (global_block_ids * attention_mask) + (attention_mask - 1)
    # [batch_size, seq_len]
    global_block_ids = handle_orphan_tokens(global_block_ids)
    num_globals = seq_len // global_block_size
    # [batch_size, seq_len // global_block_size]
    if num_globals > 0:
        _sequence_block_ids_max = torch.max(global_block_ids, dim=-1).values.repeat(num_globals, 1).transpose(0, 1)
    else:
        _sequence_block_ids_max = torch.zeros(
            batch_size, 0, dtype=global_block_ids.dtype, device=global_block_ids.device
        )
    global_segment_ids = torch.cumsum(torch.ones(batch_size, num_globals), dim=-1) - 1
    global_segment_ids = global_segment_ids.to(attention_mask.device)
    global_segment_ids = torch.where(global_segment_ids <= _sequence_block_ids_max, 1, 0)
    return global_block_ids.type(torch.int), global_segment_ids.type(torch.int)


def _make_side_relative_position_ids(attention_mask: torch.Tensor, global_block_size: int) -> torch.Tensor:
    """Create the relative position tensor for local -> global attention.
    # 调用_make_global_fixed_block_ids函数获取固定的全局块ID和全局段ID
    block_ids, global_segment_ids = _make_global_fixed_block_ids(attention_mask, global_block_size)
    # 获取全局段ID的长度
    global_seq_len = global_segment_ids.shape[-1]
    # 在设备上创建一个张量，表示全局位置
    global_positions = torch.arange(global_seq_len, device=block_ids.device)
    # 计算全局位置和块ID之间的相对位置
    side_relative_position = global_positions - block_ids[..., None]
    # 将结果转换为torch.int64类型并返回
    return side_relative_position.type(torch.int64)
def _create_global_aggregates(
    hidden_states: torch.Tensor, block_ids: torch.Tensor, global_seq_len: int
) -> torch.Tensor:
    """Compute individual block aggregates by summing over individual blocks."""
    # 为了处理 block_ids 中的负值，将其替换为 global_seq_len，创建一个与 block_ids 相同形状的张量
    block_ids = block_ids.where(
        block_ids >= 0, torch.tensor(global_seq_len, dtype=block_ids.dtype, device=block_ids.device)
    )
    # 将 block_ids 转换为 one-hot 编码，维度为 (batch_size..., seq_len, global_seq_len)
    one_hot_block_ids = nn.functional.one_hot(block_ids.type(torch.int64), global_seq_len + 1)[:, :, :-1]
    # 使用 einsum 计算隐藏状态和 one_hot_block_ids 的张量乘积，并进行汇总
    return torch.einsum("...nd,...ng->...gd", hidden_states, one_hot_block_ids.type(hidden_states.dtype))


# 从 transformers.models.t5.modeling_t5.T5LayerNorm 复制过来，并改名为 LongT5LayerNorm
class LongT5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the LongT5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        # 初始化权重参数
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # 初始化 epsilon 参数
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # LongT5 使用的是只进行缩放而不进行移动的 layer_norm，也称为 Root Mean Square Layer Normalization
        # 因此方差是在没有均值的情况下计算的，也没有偏置项
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # 如果权重的数据类型是半精度浮点类型，则需要将 hidden_states 转换为相同的数据类型
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


try:
    from apex.normalization import FusedRMSNorm

    # 将 LongT5LayerNorm 替换为 FusedRMSNorm，如果能导入的话
    LongT5LayerNorm = FusedRMSNorm  # noqa

    logger.info("Discovered apex.normalization.FusedRMSNorm - will use it instead of LongT5LayerNorm")
except ImportError:
    # 如果无法导入，则使用正常的 LongT5LayerNorm
    pass
except Exception:
    logger.warning("discovered apex but it failed to load, falling back to LongT5LayerNorm")
    pass

# 将 LongT5LayerNorm 添加到 ALL_LAYERNORM_LAYERS 列表中
ALL_LAYERNORM_LAYERS.append(LongT5LayerNorm)


# 从 transformers.models.t5.modeling_t5.T5DenseActDense 复制过来，并改名为 LongT5DenseActDense
class LongT5DenseActDense(nn.Module):
    def __init__(self, config: LongT5Config):
        super().__init__()
        # 输入到中间层的全连接层
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        # 中间层到输出层的全连接层
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        # dropout 层
        self.dropout = nn.Dropout(config.dropout_rate)
        # 激活函数
        self.act = ACT2FN[config.dense_act_fn]
    # 前向传播函数，接受隐藏状态作为输入
    def forward(self, hidden_states):
        # 使用输入状态计算线性变换
        hidden_states = self.wi(hidden_states)
        # 应用激活函数
        hidden_states = self.act(hidden_states)
        # 使用丢弃层进行正则化
        hidden_states = self.dropout(hidden_states)
        # 检查输出权重是否为张量类型，并且隐藏状态的数据类型是否与输出权重的数据类型不同，以及输出权重的数据类型是否不是 torch.int8
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            # 如果条件成立，将隐藏状态转换为输出权重的数据类型
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        # 使用输出权重进行线性变换
        hidden_states = self.wo(hidden_states)
        # 返回隐藏状态
        return hidden_states
# 定义一个包含门控激活函数的全连接层模型
class LongT5DenseGatedActDense(nn.Module):
    def __init__(self, config: LongT5Config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)  # 输入到隐藏层的线性变换
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)  # 输入到隐藏层的线性变换
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)     # 隐藏层到输出的线性变换
        self.dropout = nn.Dropout(config.dropout_rate)                  # 添加丢弃率
        self.act = ACT2FN[config.dense_act_fn]                          # 选择激活函数

    # 前向传播函数
    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))  # 使用激活函数
        hidden_linear = self.wi_1(hidden_states)          # 隐藏层线性变换
        hidden_states = hidden_gelu * hidden_linear       # 门控激活函数操作
        hidden_states = self.dropout(hidden_states)       # 添加丢弃
        hidden_states = self.wo(hidden_states)            # 输出层线性变换
        return hidden_states


# 定义一个长形式的Transformer的Feed Forward层
class LongT5LayerFF(nn.Module):
    def __init__(self, config: LongT5Config):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = LongT5DenseGatedActDense(config)
        else:
            self.DenseReluDense = LongT5DenseActDense(config)

        self.layer_norm = LongT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    # 前向传播函数
    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)        # Layer Normalization
        forwarded_states = self.DenseReluDense(forwarded_states) # 执行全连接和激活函数操作
        hidden_states = hidden_states + self.dropout(forwarded_states)  # 添加丢弃
        return hidden_states


# 定义长形式Transformer的注意力机制层
class LongT5Attention(nn.Module):
    def __init__(self, config: LongT5Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # 为避免在softmax之前进行缩放的Mesh TensorFlow初始化
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)  # 查询键的线性变换
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)  # 键的线性变换
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)  # 值的线性变换
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)  # 输出的线性转换

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)  # 相对位置编码
        self.pruned_heads = set()        # 初始化修剪的头部
        self.gradient_checkpointing = False  # 是否启用梯度检查点
    # 剪枝注意力头部，剔除不需要的头部
    def prune_heads(self, heads):
        # 如果传入的头部列表为空，则直接返回
        if len(heads) == 0:
            return
        # 找到可剪枝的头部和对应索引
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
    # 定义相对位置桶的函数
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        将相对位置转换为相对注意力的桶编号。相对位置定义为 memory_position - query_position，
        即从来源位置到被关注位置的标记距离。
        如果 bidirectional=False，则正的相对位置无效。
        对于较小的绝对相对位置，使用较小的桶，对于较大的绝对相对位置，使用较大的桶。
        所有相对位置大于等于 max_distance 的映射到同一个桶。
        所有相对位置小于等于 -max_distance 的映射到同一个桶。
        这应该能够更好地适应比模型训练时处理的更长序列。

        参数:
            relative_position: 一个 int32 张量
            bidirectional: 一个布尔值，表示注意力是否是双向的
            num_buckets: 一个整数
            max_distance: 一个整数

        返回:
            一个形状与 relative_position 相同的张量，包含范围在 [0, num_buckets) 内的 int32 值
        """
        # 初始化相对位置桶
        relative_buckets = 0
        # 如果是双向的注意力
        if bidirectional:
            # 减少桶数目（除以2）
            num_buckets //= 2
            # 根据相对位置是否大于0，进行相对桶偏移
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            # 获取相对位置的绝对值
            relative_position = torch.abs(relative_position)
        else:
            # 将相对位置转换为负值
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # 现在相对位置在 [0, inf) 范围内

        # 前一半桶用于位置的准确增量
        max_exact = num_buckets // 2
        # 判断相对位置是否小于 max_exact
        is_small = relative_position < max_exact

        # 另一半桶用于位置以 log 比例增加到 max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        # 将大于等于 num_buckets 的相对位置映射为 num_buckets - 1
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        # 将相对位置根据 is_small 和 relative_position_if_large 映射到相对位置桶上
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        # 返回相对位置桶
        return relative_buckets
    # 定义一个方法，用于计算分箱的相对位置偏差
    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        # 如果设备未指定，则使用当前相对注意力偏差的设备
        if device is None:
            device = self.relative_attention_bias.weight.device
        # 生成一个包含查询长度的张量，指定数据类型为long，设备为指定设备
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        # 生成一个包含键长度的张量，指定数据类型为long，设备为指定设备
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        # 计算相对位置，表示查询位置相对于记忆位置的偏差
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        # 将相对位置映射为相对位置桶，考虑是否为双向的，确定桶的数量和最大距离
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # 获取相对位置桶的注意力偏差值
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        # 调整维度，形状变为（1，num_heads，query_length，key_length）
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        # 返回注意力偏差值
        return values

    # 模型的前向传播方法
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
# 定义一个继承自 nn.Module 的类，用于实现 LongT5 的局部注意力机制
class LongT5LocalAttention(nn.Module):
    # 初始化方法，接收配置参数和一个可选的相对注意力偏置标志
    def __init__(self, config: LongT5Config, has_relative_attention_bias: bool = False) -> None:
        # 调用父类初始化方法
        super().__init__()
        # 设置是否为解码器模式
        self.is_decoder = config.is_decoder
        # 设置是否有相对注意力偏置
        self.has_relative_attention_bias = has_relative_attention_bias
        # 设置相对注意力桶的数量
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        # 设置相对注意力的最大距离
        self.relative_attention_max_distance = config.relative_attention_max_distance
        # 设置模型维度
        self.d_model = config.d_model
        # 设置键值对的投影维度
        self.key_value_proj_dim = config.d_kv
        # 设置注意力头数
        self.n_heads = config.num_heads
        # 设置局部半径
        self.local_radius = config.local_radius
        # 计算块长度
        self.block_len = self.local_radius + 1
        # 设置dropout率
        self.dropout = config.dropout_rate
        # 计算内部维度，为头数乘以键值投影维度
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # 初始化查询、键、值、输出的线性层，不带偏置
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        # 如果设置了相对注意力偏置，则初始化相对注意力偏置的嵌入层
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        # 初始化被裁剪头的集合
        self.pruned_heads = set()
        # 设置是否使用梯度检查点
        self.gradient_checkpointing = False

    # 裁剪头部的方法，从模型中移除指定的注意力头
    def prune_heads(self, heads):
        # 如果没有指定头部，则直接返回
        if len(heads) == 0:
            return
        # 查找可裁剪的头部和索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # 裁剪查询、键、值、输出层中对应的头部
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # 更新头数和内部维度
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        # 更新被裁剪头的集合
        self.pruned_heads = self.pruned_heads.union(heads)

    # 从 transformers.models.t5.modeling_t5.T5Attention 复制的静态方法，用于计算相对位置桶
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        将相对位置转换为相对注意力所需的存储桶编号。相对位置定义为memory_position - query_position，即从出现位置到被注意位置的距离。如果bidirectional=False，则正相对位置无效。对于较小的绝对relative_position，我们使用较小的桶，对于较大的绝对relative_positions，我们使用较大的桶。所有相对位置>=max_distance映射到同一个桶。所有相对位置<=-max_distance映射到同一个桶。这应该可以更自然地推广到比模型受过训练的更长序列

        Args:
            relative_position: 一个int32的Tensor
            bidirectional: 一个布尔值 - 注意力是否是双向的
            num_buckets: 一个整数
            max_distance: 一个整数

        Returns:
            一个具有与relative_position相同形状的Tensor，包含值在 [0, num_buckets) 范围内的int32值
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # 现在relative_position在 [0, inf) 范围内

        # 一半的桶用于处理位置的确切增量
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # 另一半的桶用于在位置到max_distance之间按对数更大的倍数处理
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets
    def compute_bias(self, block_length: int):
        """Compute binned relative position bias"""
        # 获取相对位置偏置的目标设备
        target_device = (
            self.relative_attention_bias.weight.device
            if self.relative_attention_bias.weight.device.type != "meta"
            else None
        )
        # 创建内存位置张量，长度为3倍的块长度
        memory_position = torch.arange(3 * block_length, dtype=torch.long, device=target_device)
        # 选择上下文位置，排除首尾块长度的部分
        context_position = memory_position[block_length:-block_length]

        # 计算相对位置
        # (block_length, 3 * block_length)
        relative_position = memory_position[None, :] - context_position[:, None]
        # 将相对位置映射到对应的桶中
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # (block_length, 3 * block_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # 计算相对位置偏置值
        # (block_length, 3 * block_length, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)
        # 重新排列张量维度，以适应后续计算
        # (1, 1, num_heads, block_length, 3 * block_length)
        values = values.permute([2, 0, 1]).unsqueeze(0).unsqueeze(0)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        position_bias=None,
        layer_head_mask=None,
        output_attentions=False,
# 定义一个名为LongT5TransientGlobalAttention的类，继承自nn.Module
class LongT5TransientGlobalAttention(nn.Module):
    def __init__(self, config: LongT5Config, has_relative_attention_bias: bool = False) -> None:
        super().__init__()
        # 根据配置参数实例化对象属性
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

        # 使用线性层初始化参数
        # q表示查询
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        # k表示键
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        # v表示值
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        # o表示输出
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        # 如果存在relative attention bias，则实例化Embedding对象
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()

        # 如果存在relative attention bias，则实例化Embedding对象，并实例化LongT5LayerNorm对象
        if self.has_relative_attention_bias:
            self.global_relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.global_input_layer_norm = LongT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    # 拷贝自transformers.models.t5.modeling_t5.T5Attention.prune_heads方法
    def prune_heads(self, heads):
        # 如果要剪枝的头部数为0，则直接返回
        if len(heads) == 0:
            return
        # 调用函数找到要剪枝的头部和索引
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

    @staticmethod
    # 拷贝自transformers.models.t5.modeling_t5.T5Attention._relative_position_bucket方法
    # 定义一个函数，用于将相对位置转换为相对注意力的桶编号
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
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        # 初始化相对位置所属的桶编号
        relative_buckets = 0
        # 如果是双向的相对注意力，则将桶数减半
        if bidirectional:
            num_buckets //= 2
            # 根据相对位置的正负情况，确定相对位置所属的桶编号
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            # 将相对位置转换为绝对值
            relative_position = torch.abs(relative_position)
        else:
            # 如果是单向的相对注意力，则将相对位置限制在负数范围内
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # 相对位置现在的范围为[0, inf)

        # 一半的桶用于精确增量位置
        max_exact = num_buckets // 2
        # 判断相对位置是否属于较小的范围
        is_small = relative_position < max_exact

        # 另一半的桶用于相对距离更大的位置的对数增量
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        # 将相对位置限制在合法的范围内
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        # 根据相对位置的大小选择相应的桶编号
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        # 返回相对位置所属的桶编号
        return relative_buckets
    def compute_bias(self, block_length: int):
        """计算分组的相对位置偏置"""
        # 获取相对位置偏置张量的设备类型
        target_device = (
            self.relative_attention_bias.weight.device
            if self.relative_attention_bias.weight.device.type != "meta"
            else None
        )
        # 创建存储内存位置的张量，长度为 3 * block_length
        memory_position = torch.arange(3 * block_length, dtype=torch.long, device=target_device)
        # 创建存储上下文位置的张量，长度为(block_length, -block_length)
        context_position = memory_position[block_length:-block_length]
    
        # 计算所有位置的相对位置偏差
        relative_position = memory_position[None, :] - context_position[:, None]
        # 将相对位置离散化为桶，每个桶代表一组相对位置
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # (block_length, 3 * block_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # 使用相对位置桶来获取相对位置偏置
        values = self.relative_attention_bias(relative_position_bucket)
        # 重新排列相对位置偏置的维度顺序
        values = values.permute([2, 0, 1]).unsqueeze(0).unsqueeze(0)
        return values
    
    
    def compute_side_bias(self, mask: torch.Tensor, global_segment_ids: torch.Tensor) -> torch.Tensor:
        """计算侧边相对位置偏置"""
        # 创建侧边注意力掩码
        side_attention_mask = torch.eq(mask[..., None], global_segment_ids[:, None, :])[:, None, ...]
        # 侧边注意力偏置，如果掩码值为真则为0.0，否则为-1e10
        attention_side_bias = torch.where(side_attention_mask > 0, 0.0, -1e10)
        # 计算侧边相对位置 ID
        side_relative_position = _make_side_relative_position_ids(mask, self.global_block_size)
        # 将侧边相对位置离散化为桶，每个桶代表一组相对位置
        side_relative_position_bucket = self._relative_position_bucket(
            side_relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # 获取侧边相对位置偏置
        side_bias = self.global_relative_attention_bias(side_relative_position_bucket)
    
        # 重新排列侧边相对位置偏置的维度顺序
        side_bias = side_bias.permute([0, 3, 1, 2])
        # 将侧边相对位置偏置与侧边注意力偏置相加
        attention_side_bias = attention_side_bias + side_bias
        return attention_side_bias
    
    
    def forward(
        self,
        hidden_states,
        mask=None,
        position_bias=None,
        layer_head_mask=None,
        output_attentions=False,
# 从 transformers.models.t5.modeling_t5.T5LayerSelfAttention 复制，修改 T5->LongT5
class LongT5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        # 调用父类 nn.Module 的初始化方法
        super().__init__()
        # 使用指定配置初始化 LongT5Attention，设置相对注意力偏差的标志
        self.SelfAttention = LongT5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        # 使用配置中的 d_model 和 epsilon 初始化层归一化对象
        self.layer_norm = LongT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 使用配置中的 dropout_rate 初始化 dropout 层
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
        # 对输入的隐藏状态应用层归一化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 调用 SelfAttention 对象，计算注意力输出
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 将注意力输出与原始隐藏状态相加，并应用 dropout 层
        hidden_states = hidden_states + self.dropout(attention_output[0])
        # 将结果存储为元组，包含新的隐藏状态和注意力输出
        outputs = (hidden_states,) + attention_output[1:]  # 如果输出注意力则添加
        # 返回结果
        return outputs


class LongT5LayerLocalSelfAttention(nn.Module):
    """编码器中使用的局部自注意力"""

    def __init__(self, config, has_relative_attention_bias=False):
        # 调用父类 nn.Module 的初始化方法
        super().__init__()
        # 使用指定配置初始化 LongT5LocalAttention，设置相对注意力偏差的标志
        self.LocalSelfAttention = LongT5LocalAttention(config, has_relative_attention_bias=has_relative_attention_bias)
        # 使用配置中的 d_model 和 epsilon 初始化层归一化对象
        self.layer_norm = LongT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 使用配置中的 dropout_rate 初始化 dropout 层
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        output_attentions=False,
        **kwargs: Any,  # 接受 past_key_value 和 use_cache 的关键字参数
    ):
        # 对输入的隐藏状态应用层归一化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 调用 LocalSelfAttention 对象，计算注意力输出
        attention_output = self.LocalSelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 将注意力输出与原始隐藏状态相加，并应用 dropout 层
        hidden_states = hidden_states + self.dropout(attention_output[0])
        # 将结果存储为元组，包含新的隐藏状态和注意力输出
        outputs = (hidden_states,) + attention_output[1:]  # 如果输出注意力则添加
        # 返回结果
        return outputs


class LongT5LayerTransientGlobalSelfAttention(nn.Module):
    """编码器中使用的瞬时全局自注意力"""
    # 初始化函数，用于初始化 LongT5Layer 类的实例
    def __init__(self, config, has_relative_attention_bias=False):
        # 调用父类的初始化函数
        super().__init__()
        # 初始化一个 LongT5TransientGlobalAttention 实例，用于全局自注意力计算
        self.TransientGlobalSelfAttention = LongT5TransientGlobalAttention(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        # 初始化一个 Layer normalization 层，用于规范化输入向量
        self.layer_norm = LongT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 初始化一个 Dropout 层，用于在训练过程中随机丢弃部分神经元，防止过拟合
        self.dropout = nn.Dropout(config.dropout_rate)

    # 前向传播函数，用于模型的前向计算
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        output_attentions=False,
        **kwargs: Any,  # to accept past_key_value and use_cache kwargs
    ):
        # 对输入进行 Layer normalization 规范化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 使用全局自注意力模块进行注意力计算
        attention_output = self.TransientGlobalSelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 将原始隐藏状态与经过注意力计算后的结果相加
        hidden_states = hidden_states + self.dropout(attention_output[0])
        # 如果需要输出注意力值，则在输出中添加注意力值
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        # 返回模型输出
        return outputs
# LongT5LayerCrossAttention
此处定义了一个名为LongT5LayerCrossAttention的类，继承自nn.Module。这个类是从transformers.models.t5.modeling_t5.T5LayerCrossAttention中复制而来，并对其中的T5进行了改动，改为LongT5。

## __init__方法
- 参数:
    - config: 模型的配置信息
- 功能: 初始化函数
- 作用: 
    - 创建一个LongT5Attention对象EncDecAttention，使用配置信息和设定是否使用相对注意力偏置(has_relative_attention_bias)作为参数
    - 创建一个LongT5LayerNorm对象layer_norm，使用配置信息中的d_model和layer_norm_epsilon作为参数
    - 创建一个nn.Dropout对象dropout，使用配置信息中的dropout_rate作为参数

## forward方法
- 参数:
    - hidden_states: 输入的隐藏状态
    - key_value_states: 键值状态
    - attention_mask: 注意力掩码，默认为None
    - position_bias: 位置偏置，默认为None
    - layer_head_mask: 多头注意力掩码，默认为None
    - past_key_value: 缓存的键值对，默认为None
    - use_cache: 是否使用缓存，默认为False
    - query_length: 查询长度，默认为None
    - output_attentions: 是否输出注意力权重，默认为False
- 功能: 执行前向传递操作
- 作用:
    - 对输入的隐藏状态进行层归一化操作
    - 调用EncDecAttention的forward方法，使用归一化后的隐藏状态、注意力掩码、键值状态、位置偏置、多头注意力掩码、缓存的键值对、是否使用缓存、查询长度、是否输出注意力权重作为参数进行前向传递，并将结果赋值给attention_output
    - 将隐藏状态与dropout对attention_output的第一个值进行相加，并将结果赋值给layer_output
    - 将layer_output与attention_output[1:]组成一个元组并赋值给outputs
    - 返回outputs

# LongT5Block
此处定义了一个名为LongT5Block的类，继承自nn.Module。

## __init__方法
- 参数:
    - config: 模型的配置信息
    - has_relative_attention_bias: 是否使用相对注意力偏置，默认为False
- 功能: 初始化函数
- 作用:
    - 设置类的decoder属性为配置信息中的is_decoder属性
    - 根据配置信息中的不同编码器的注意力机制(encoder_attention_type)选择不同的注意力层作为attention_layer
        - 若is_decoder为True，则选择LongT5LayerSelfAttention
        - 若encoder_attention_type为“local”，则选择LongT5LayerLocalSelfAttention
        - 若encoder_attention_type为“transient-global”，则选择LongT5LayerTransientGlobalSelfAttention
        - 若不满足以上三个条件，则抛出ValueError
    - 创建一个nn.ModuleList对象layer
    - 向layer中添加一个attention_layer对象，使用配置信息和是否使用相对注意力偏置作为参数
    - 若is_decoder为True，向layer中添加一个LongT5LayerCrossAttention对象，使用配置信息作为参数
    - 向layer中添加一个LongT5LayerFF对象，使用配置信息作为参数

## forward方法
- 参数:
    - hidden_states: 输入的隐藏状态
    - attention_mask: 注意力掩码，默认为None
    - position_bias: 位置偏置，默认为None
    - encoder_hidden_states: 编码器的隐藏状态，默认为None
    - encoder_attention_mask: 编码器的注意力掩码，默认为None
    - encoder_decoder_position_bias: 编码器和解码器之间的位置偏置，默认为None
    - layer_head_mask: 多头注意力掩码，默认为None
    - cross_attn_layer_head_mask: 交叉注意力的多头注意力掩码，默认为None
    - past_key_value: 缓存的键值对，默认为None
    - use_cache: 是否使用缓存，默认为False
    - output_attentions: 是否输出注意力权重，默认为False
    - return_dict: 是否返回字典类型的结果，默认为True
- 功能: 执行前向传递操作
- 作用:
    - 若encoder_hidden_states不为None，则执行注释部分中的if代码块，该代码块中并未给出具体实现内容，只提示“add encoder modules”。
    - 遍历layer中的每个层，并依次调用每个层的forward方法，使用对应的参数进行前向传递
        - attention_layer: 使用hidden_states、attention_mask、position_bias、layer_head_mask、past_key_value、use_cache、output_attentions作为参数进行前向传递
        - LongT5LayerCrossAttention: 使用hidden_states、attention_mask、position_bias、encoder_hidden_states、encoder_attention_mask、encoder_decoder_position_bias、layer_head_mask、cross_attn_layer_head_mask、past_key_value、use_cache、output_attentions作为参数进行前向传递
        - LongT5LayerFF: 使用前两个层返回的结果(layer_output和attention_output)、encoder_hidden_states、encoder_attention_mask作为参数进行前向传递，并将结果赋值给layer_output
    - 若return_dict为True，则返回一个字典类型的结果，其中包含layer_output和attention_output的值；否则，直接返回layer_output

# LongT5PreTrainedModel
此处定义了一个名为LongT5PreTrainedModel的类，继承自PreTrainedModel类。

## __init__方法
- 功能: 初始化函数
- 作用: 未给出具体的实现内容，只提示“An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained”
    models.
    """
    
    # 定义 LongT5Config 类作为配置类
    config_class = LongT5Config
    # 设定模型前缀为transformer
    base_model_prefix = "transformer"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不需要拆分的模块列表
    _no_split_modules = ["LongT5Block"]

    @property
    # 从transformers.models.t5.modeling_t5.T5PreTrainedModel.dummy_inputs中复制获取dummy_inputs
    def dummy_inputs(self):
        # 创建一个包含DUMMY_INPUTS值的张量
        input_ids = torch.tensor(DUMMY_INPUTS)
        # 创建一个包含DUMMY_MASK值的张量
        input_mask = torch.tensor(DUMMY_MASK)
        # 创建dummy_inputs字典，包含decoder_input_ids、input_ids、decoder_attention_mask这些键值对
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        # 返回dummy_inputs
        return dummy_inputs

    # 从transformers.models.t5.modeling_t5.T5PreTrainedModel._shift_right复制，并替换T5为LongT5
    def _shift_right(self, input_ids):
        # 获取decoder_start_token_id和pad_token_id的值
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        if decoder_start_token_id is None:
            # 如果decoder_start_token_id未定义则引发错误
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In LongT5 it is usually set to the pad_token_id. "
                "See LongT5 docs for more information."
            )

        # 将输入向右移动
        if is_torch_fx_proxy(input_ids):
            # Proxies不支持原生的项目赋值
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            # 如果pad_token_id未定义，引发错误
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # 用pad_token_id替换labels中可能存在的-100值
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids
# 定义一个继承自LongT5PreTrainedModel的LongT5Stack类
class LongT5Stack(LongT5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        # 调用父类的构造方法
        super().__init__(config)

        # 初始化词嵌入tokens
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        # 如果有传入的embed_tokens，则使用传入的
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight
        # 是否为解码器
        self.is_decoder = config.is_decoder

        # 设置本地半径和块长度
        self.local_radius = config.local_radius
        self.block_len = self.local_radius + 1

        # 构建T5块的模块列表
        self.block = nn.ModuleList(
            [LongT5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        # 最终的层归一化
        self.final_layer_norm = LongT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # Dropout层
        self.dropout = nn.Dropout(config.dropout_rate)

        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    # 返回输入词嵌入
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入词嵌入
    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

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

# LongT5模型的文档字符串
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
# 长文本 T5 输入参数的文档字符串
LONGT5_INPUTS_DOCSTRING = r"""
"""

# 长文本 T5 编码器输入参数的文档字符串
LONGT5_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            输入序列标记在词汇表中的索引。LongT5 是一个具有相对位置嵌入的模型，因此您应该能够在右侧和左侧都能够填充输入。
            可以使用 [`AutoTokenizer`] 来获取索引。 进一步细节，请参阅 [`PreTrainedTokenizer.encode`] 和
            [`PreTrainedTokenizer.__call__`]
            有关如何准备预训练的 `input_ids` 的更多信息，请查看 [LONGT5 Training](./longt5#training)。
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            用于避免在填充标记索引上执行注意力的遮罩。 遮罩值选择在 `[0, 1]` 范围内：

            - 1 表示 **未遮罩** 的标记，
            - 0 表示 **遮罩** 的标记。
            
            [什么是注意力遮罩?](../glossary#attention-mask)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            用于将自注意力模块的选定头部置零的遮罩。 遮罩值选择在 `[0, 1]` 范围内：
            
            - 1 表示 **未遮罩** 的头部，
            - 0 表示 **遮罩** 的头部。
            
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            可选择，而不是传递 `input_ids`，您可以选择直接传递嵌入表示。 如果您想要更多控制如何将 `input_ids` 索引转换为关联向量，
            而不是使用模型的内部嵌入查找矩阵，则这将是有用的。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回张量下的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关更多详细信息，请参见返回张量下的 `hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""

# 未来警告：head_mask 已拆分成两个输入参数 - head_mask, decoder_head_mask 的警告消息
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""

# 添��起始文档字符串
@add_start_docstrings(
    "The bare LONGT5 Model transformer outputting raw hidden-states without any specific head on top.",
    LONGT5_START_DOCSTRING,
)
class LongT5Model(LongT5PreTrainedModel):
    # 定义在加载时需要忽略的密钥列表
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    # 定义需要绑定权重的密钥列表
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    # 初始化方法，接受一个 LongT5Config 类型的参数
    def __init__(self, config: LongT5Config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建一个共享的嵌入层
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 复制编码器配置，并设置为非解码器
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # 创建编码器对象
        self.encoder = LongT5Stack(encoder_config, self.shared)

        # 复制解码器配置，并设置为解码器
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        # 创建解码器对象
        self.decoder = LongT5Stack(decoder_config, self.shared)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.shared

    # 设置输入嵌入
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    # 绑定权重
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    # 获取编码器
    def get_encoder(self):
        return self.encoder

    # 获取解码器
    def get_decoder(self):
        return self.decoder

    # 剪枝模型的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 前向传播方法
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
# 为 LongT5ForConditionalGeneration 类添加文档字符串
@add_start_docstrings("""LONGT5 Model with a `language modeling` head on top.""", LONGT5_START_DOCSTRING)
class LongT5ForConditionalGeneration(LongT5PreTrainedModel):
    # 在加载时需要忽略的键值对列表
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    # 需要绑定权重的键列表
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    # 初始化方法，传入 LongT5Config 对象
    def __init__(self, config: LongT5Config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置模型维度属性
        self.model_dim = config.d_model

        # 创建共享的嵌入层
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 复制编码器配置，并设置为非解码器
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = LongT5Stack(encoder_config, self.shared)

        # 复制解码器配置，并设置为解码器
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = LongT5Stack(decoder_config, self.shared)

        # 创建线性层作为语言模型输出层
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 返回输入嵌入层
    def get_input_embeddings(self):
        return self.shared

    # 设置新的输入嵌入层
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    # 绑定权重方法
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    # 设置新的输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 返回输出嵌入层
    def get_output_embeddings(self):
        return self.lm_head

    # 返回编码器对象
    def get_encoder(self):
        return self.encoder

    # 返回解码器对象
    def get_decoder(self):
        return self.decoder

    # 添加输入文档字符串和替换返回文档字符串
    @add_start_docstrings_to_model_forward(LONGT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 此方法用于模型的前向传播，接受多个输入参数，并返回模型的输出结果
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
        # 如果使用了过去的键值，剪切解码器输入的 ID
        if past_key_values is not None:
            # 获取过去键值的长度
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经只传递了最后一个输入 ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认保留最后一个 ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 从输入中剪切掉过去的长度
            input_ids = input_ids[:, remove_prefix_length:]

        # 返回前向传播所需的输入
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

    # 根据标签准备解码器的输入 ID
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        # 调用内部方法将标签向右移动一位作为解码器输入
        return self._shift_right(labels)
    # 重新排列缓存中的过去键值对，根据beam_idx重排
    def _reorder_cache(self, past_key_values, beam_idx):
        # 如果解码器的过去不包含在输出中，关闭速度解码并不需要重新排列
        if past_key_values is None:
            # 输出警告提示用户考虑设置`use_cache=True`以加快解码速度
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values
    
        # 重新排列后的解码器过去状态
        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # 从层次过去批次维度获取正确的批次idx
            # `past`的批次维度在第2个位置
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # 需要为每个四个键/值状态设置正确的`past`
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )
    
            # 断言重新排列后的层次过去状态形状与原状态相同
            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)
    
            # 将重新排列后的层次过去状态加入到重新排列后的解码器过去状态中
            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past
# 添加起始文档字符串到模型的encoder输出未经任何特定标头处理的裸LONGT5模型变压器
@add_start_docstrings(
    "The bare LONGT5 Model transformer outputting encoder's raw hidden-states without any specific head on top.",
    LONGT5_START_DOCSTRING,
)
class LongT5EncoderModel(LongT5PreTrainedModel):
    # 需要共享权重的键
    _tied_weights_keys = ["encoder.embed_tokens.weight"]
    # 加载时需要忽略的键
    _keys_to_ignore_on_load_unexpected = [r"decoder"]

    def __init__(self, config: LongT5Config):
        # 调用父类构造函数
        super().__init__(config)
        # 创建一个共享的嵌入层
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 复制配置并进行一些更改
        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # 创建encoder模块
        self.encoder = LongT5Stack(encoder_config, self.shared)

        # 初始化权重并应用最终处理
        self.post_init()

    # 返回输入的嵌入层
    def get_input_embeddings(self):
        return self.shared

    # 设置新的输入嵌入层
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    # 绑定权重
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)

    # 返回encoder
    def get_encoder(self):
        return self.encoder

    # 精简模型中的头部
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 前向传播函数
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
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        r"""
        Returns: 返回值说明

        Example: 示例用法

        ```python
        >>> from transformers import AutoTokenizer, LongT5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("google/long-t5-local-base")
        >>> model = LongT5EncoderModel.from_pretrained("google/long-t5-local-base")
        >>> input_ids = tokenizer(
        ...     100 * "Studies have been shown that owning a dog is good for you ", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model(input_ids=input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```py"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 确保返回字典的标志已设置，如果未提供，则使用配置中的默认值

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 将输入传递给编码器，获取编码器输出

        return encoder_outputs
        # 返回编码器输出
```