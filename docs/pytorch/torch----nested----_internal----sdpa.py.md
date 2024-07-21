# `.\pytorch\torch\nested\_internal\sdpa.py`

```py
# mypy: allow-untyped-defs
# 导入日志模块和类型相关的库
import logging
from typing import Optional, Tuple

import torch
import torch.nn
import torch.nn.functional as F
from torch.backends.cuda import (
    can_use_efficient_attention,
    can_use_flash_attention,
    flash_sdp_enabled,
    math_sdp_enabled,
    mem_efficient_sdp_enabled,
    SDPAParams,
)
from torch.nn.attention import SDPBackend

# 导入自定义的NestedTensor类
from .nested_tensor import NestedTensor

# 设置日志记录器
log = logging.getLogger(__name__)


def _validate_sdpa_input(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
):
    # 检查query, key, value是否都是NestedTensor类型
    if (
        not isinstance(query, NestedTensor)
        or not isinstance(key, NestedTensor)
        or not isinstance(value, NestedTensor)
    ):
        raise ValueError(
            f"Expected query, key, and value to be nested tensors, "
            f"but got query.is_nested: {query.is_nested}, key.is_nested: {key.is_nested}, "
            f"and value.is_nested: {value.is_nested} instead."
        )
    # 检查query, key, value的数据类型是否一致
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError(
            f"Expected query, key, and value to have the same dtype, "
            f"but got query.dtype: {query.dtype}, key.dtype: {key.dtype}, "
            f"and value.dtype: {value.dtype} instead."
        )
    # 检查query, key, value的设备类型是否一致
    if query.device != key.device or query.device != value.device:
        raise ValueError(
            f"Expected query, key, and value to have the same device type, "
            f"but got query.device: {query.device}, key.device: {key.device}, "
            f"and value.device: {value.device} instead."
        )
    # 检查query, key, value的维度是否都至少为3
    if query.dim() < 3 or key.dim() < 3 or value.dim() < 3:
        raise ValueError(
            f"Expected query, key, and value to all be  at least 3 dimensional, but got query.dim: "
            f"{query.dim()}, key.dim: {key.dim()} and value.dim: {value.dim()} instead."
        )
    # 检查query, key, value是否在相同的ragged维度上
    if query._ragged_idx != key._ragged_idx or query._ragged_idx != value._ragged_idx:
        raise ValueError(
            f"Expected query, key, and value to all be ragged on the same dimension, but got ragged "
            f"dims {query._ragged_idx}, {key._ragged_idx}, and {value._ragged_idx}, respectively."
        )
    # 如果存在attn_mask，则抛出异常，暂不支持使用attn_mask
    if attn_mask is not None:
        # TODO: Figure out whether masks are actually supported for this layout or not
        raise ValueError("Masks are not yet supported!")
        # 如果attn_mask的数据类型既不是bool也不与query的数据类型相同，则抛出异常
        if attn_mask.dtype != torch.bool and attn_mask.dtype != query.dtype:
            raise ValueError(
                f"Expected attn_mask dtype to be bool or to match query dtype, but got attn_mask.dtype: "
                f"{attn_mask.dtype}, and query.dtype: {query.dtype} instead."
            )


def _check_batch_size_nested(params: SDPAParams, debug=False) -> bool:
    # This is expected to be called after check_tensor_shapes ensuring that the
    # size() calls won't error since the inputs are all 4 dimensional
    # 获取查询（query）张量的批量大小
    q_batch_size = params.query.size(0)
    # 获取键（key）张量的批量大小
    k_batch_size = params.key.size(0)
    # 获取值（value）张量的批量大小
    v_batch_size = params.value.size(0)

    # 检查在嵌套输入中是否有 num_heads 逻辑，这个检查在 check_for_seq_len_0_nested_tensor 函数中进行，
    # 以确保 num_heads 不会出现不规则情况（ragged）
    # 返回值：查询、键、值张量的批量大小是否全部相等
    return q_batch_size == k_batch_size and q_batch_size == v_batch_size
# 检查嵌套张量的头维度大小是否符合要求，用于Flash注意力机制
def _check_head_dim_size_flash_nested(params: SDPAParams, debug=False) -> bool:
    # 定义最大尺寸
    max_size = 256
    # 获取查询张量的最后一个维度大小
    query_size_last = params.query.size(-1)
    # 获取键张量的最后一个维度大小
    key_size_last = params.key.size(-1)
    # 获取值张量的最后一个维度大小
    value_size_last = params.value.size(-1)
    # 检查查询、键、值张量的最后一个维度是否相同
    same_head_dim_size = (
        query_size_last == key_size_last and query_size_last == value_size_last
    )
    # 如果不符合Flash注意力机制的要求则返回False
    if not (
        same_head_dim_size
        and (query_size_last % 8 == 0)
        and (query_size_last <= max_size)
    ):
        # 如果处于调试模式，记录警告日志，指出不符合要求的维度情况
        if debug:
            log.warning(
                "For NestedTensor inputs, Flash attention requires q,k,v to have the same "
                "last dimension and to be a multiple of 8 and less than or equal to 256. "
                "Got Query.size(-1): %d, Key.size(-1): %d, Value.size(-1): %d instead.",
                query_size_last,
                key_size_last,
                value_size_last,
            )
        return False
    return True


# 检查嵌套张量的序列长度是否为0，并且检查头维度是否一致的辅助函数
def _check_for_seq_len_0_and_consistent_head_dim_nested_helper(
    param: torch.Tensor, param_name: str, debug=False
) -> bool:
    # 断言参数param是嵌套张量类型（NestedTensor）
    assert isinstance(param, NestedTensor), "param should be a jagged NT"

    # 如果_ragged_idx等于1，表示头维度是不规则的
    if param._ragged_idx == 1:
        # 如果处于调试模式，记录警告日志，指出不支持不规则头维度的情况
        if debug:
            log.warning(
                "Fused kernels do not support ragged num_head_dims, %s has a ragged num_heads.",
                param_name,
            )
        return False

    # 被调用时，param的形状应为[batch, heads, {seq_len}, dim]，检查最小序列长度是否为0
    if param._get_min_seqlen() == 0:
        # 如果处于调试模式，记录警告日志，指出序列长度为0的情况
        if debug:
            log.warning(
                "Fused kernels do not support seq_len == 0, %s has a seq len of 0.",
                param_name,
            )
        return False

    return True


# 尝试广播参数大小以满足要求的辅助函数
def _try_broadcast_param_size(q_size, k_size, v_size, param_name, debug=False) -> bool:
    # 计算三个张量大小的最大值
    max_size = max(q_size, k_size, v_size)
    # 如果三个张量的大小不等于最大值且不等于1，则不满足广播要求
    if (
        (q_size != max_size and q_size != 1)
        or (k_size != max_size and k_size != 1)
        or (v_size != max_size and v_size != 1)
    ):
        # 如果处于调试模式，记录警告日志，指出不满足广播要求的情况
        if debug:
            log.warning(
                "Both fused kernels require query, key and value to have broadcastable %s, "
                "got Query %s %d, Key %s %d, Value %s %d instead.",
                param_name,
                param_name,
                q_size,
                param_name,
                k_size,
                param_name,
                v_size,
            )
        return False
    return True


# 检查嵌套张量的序列长度是否为0的函数
def _check_for_seq_len_0_nested(params: SDPAParams, debug=False) -> bool:
    # 当调用此函数时，确保params.query是嵌套张量且维度为4
    q_is_safe = (
        _check_for_seq_len_0_and_consistent_head_dim_nested_helper(
            params.query, "query", debug
        )
        if params.query.is_nested
        else True
    )
    # 如果任何一个检查不安全，则返回False
    if not q_is_safe:
        return False
    # 检查是否输入的键是安全的，根据是否嵌套决定调用辅助函数或直接返回True
    k_is_safe = (
        _check_for_seq_len_0_and_consistent_head_dim_nested_helper(
            params.key, "key", debug
        )
        if params.key.is_nested  # 如果键是嵌套的，则调用辅助函数检查安全性
        else True  # 如果键不是嵌套的，则默认为安全
    )
    
    # 如果任何一个键不安全，立即返回False
    if not k_is_safe:
        return False
    
    # 检查是否输入的值是安全的，根据是否嵌套决定调用辅助函数或直接返回True
    v_is_safe = (
        _check_for_seq_len_0_and_consistent_head_dim_nested_helper(
            params.value, "value", debug
        )
        if params.value.is_nested  # 如果值是嵌套的，则调用辅助函数检查安全性
        else True  # 如果值不是嵌套的，则默认为安全
    )
    
    # 如果任何一个值不安全，立即返回False
    if not v_is_safe:
        return False
    
    # 现在我们知道输入的所有内容的头维度是一致的，因此可以安全地访问 .size(1)
    q_num_heads = params.query.size(1)  # 获取查询参数的头维度
    k_num_heads = params.key.size(1)  # 获取键参数的头维度
    v_num_heads = params.value.size(1)  # 获取值参数的头维度
    same_num_heads = q_num_heads == k_num_heads and q_num_heads == v_num_heads  # 检查是否所有参数的头维度都相同
    
    # 如果头维度不一致
    if not same_num_heads:
        # 如果任何一个参数需要梯度，且用于调试时，记录警告信息
        if (
            params.query.requires_grad
            or params.key.requires_grad
            or params.value.requires_grad
        ):
            if debug:
                log.warning(
                    "Both fused kernels do not support training with broadcasted NT inputs."
                )
            return False  # 返回False，表示不支持训练
        # 尝试调整参数的尺寸以匹配头维度，返回调整结果的布尔值
        return _try_broadcast_param_size(
            q_num_heads, k_num_heads, v_num_heads, "num heads", debug
        )
    
    # 如果头维度一致，则返回True，表示参数可以安全使用
    return True
def _can_use_flash_sdpa_jagged(params: SDPAParams, debug=False) -> bool:
    # 定义约束条件函数的元组
    constraints = (
        _check_batch_size_nested,
        _check_head_dim_size_flash_nested,
        _check_for_seq_len_0_nested,
    )
    # 遍历每个约束条件函数
    for constraint in constraints:
        # 如果约束条件不满足，则返回 False
        if not constraint(params, debug):
            return False
    # 如果所有约束条件均满足，则返回 True
    return True


def _can_use_efficient_sdpa_jagged(params: SDPAParams, debug=False) -> bool:
    # 定义约束条件函数的元组
    constraints = (
        _check_batch_size_nested,
        _check_for_seq_len_0_nested,
    )
    # 遍历每个约束条件函数
    for constraint in constraints:
        # 如果约束条件不满足，则返回 False
        if not constraint(params, debug):
            return False
    # 如果所有约束条件均满足，则返回 True
    return True


def _can_use_math_sdpa_jagged(params: SDPAParams, debug=False) -> bool:
    # 检查是否需要转置输入张量并检查它们是否连续
    if (
        not params.query.transpose(1, 2).is_contiguous()
        or not params.key.transpose(1, 2).is_contiguous()
        or not params.value.transpose(1, 2).is_contiguous()
    ):
        # 如果需要转置并且不连续，并且启用了 debug 模式，则发出警告
        if debug:
            log.warning(
                "If inputs are nested tensors they must be contiguous after transposing."
            )
        # 返回 False，表示约束条件不满足
        return False
    # 如果 is_causal 标志为 True，则发出警告并返回 False
    if params.is_causal:
        if debug:
            log.warning(
                "Nested tensors for query / key are not supported when is_causal=True."
            )
        return False
    # 如果所有约束条件均满足，则返回 True
    return True


def _select_sdp_backend(query, key, value, attn_mask, dropout, is_causal):
    # 检查是否启用了任何一种 SDP（Self-Attention Dot-Product）后端
    if (
        not flash_sdp_enabled()
        and not mem_efficient_sdp_enabled()
        and not math_sdp_enabled()
    ):
        # 如果没有启用任何一种后端，则返回错误状态
        return SDPBackend.ERROR

    # 指定后端优先级顺序
    ordering = (
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    )

    # 创建 SDPAParams 对象，用于传递给后端选择函数
    params = SDPAParams(query, key, value, attn_mask, dropout, is_causal)

    # 遍历每种后端类型
    for backend in ordering:
        # 如果当前后端是 Flash Attention
        if backend == SDPBackend.FLASH_ATTENTION:
            # 检查是否可以使用 Flash Attention，并且满足 SDPA 的约束条件
            if can_use_flash_attention(params) and _can_use_flash_sdpa_jagged(params):
                return SDPBackend.FLASH_ATTENTION
        # 如果当前后端是 Efficient Attention
        if backend == SDPBackend.EFFICIENT_ATTENTION:
            # 检查是否可以使用 Efficient Attention，并且满足 SDPA 的约束条件
            if can_use_efficient_attention(params) and _can_use_efficient_sdpa_jagged(
                params
            ):
                return SDPBackend.EFFICIENT_ATTENTION
        # 如果当前后端是 Math Attention
        if backend == SDPBackend.MATH:
            # 检查是否启用了 Math SDP，并且满足 SDPA 的约束条件
            if math_sdp_enabled() and _can_use_math_sdpa_jagged(params):
                return SDPBackend.MATH

    # 如果没有任何后端被选中，则发出警告
    log.warning("Memory efficient kernel not used because:")
    # 输出内存效率 attention 内核的使用情况，可能的警告
    can_use_efficient_attention(params, debug=True)
    # 输出 SDPA 约束条件的使用情况，可能的警告
    _can_use_efficient_sdpa_jagged(params, debug=True)
    # 输出 Flash attention 内核的使用情况，可能的警告
    log.warning("Flash attention kernel not used because:")
    can_use_flash_attention(params, debug=True)
    _can_use_flash_sdpa_jagged(params, debug=True)
    # 输出 Math attention 内核的使用情况，可能的警告
    log.warning("Math attention kernel not used because:")
    _can_use_math_sdpa_jagged(params, debug=True)
    # 返回错误状态，表示未选择任何合适的 SDP 后端
    return SDPBackend.ERROR


def _cumulative_and_max_seq_len_nnz(qkv: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    # This function is used to calculate two pieces of metadata that are needed
    # 这个函数用于计算两个必要的元数据
    # 如果输入的 qkv 不是 NestedTensor 类型，则抛出数值错误
    if not isinstance(qkv, NestedTensor):
        raise ValueError("QKV must be nested for flash cumulative_seq_len calculation.")

    # 如果 qkv 的长度信息为 None，则使用偏移值创建累积序列长度
    cumulative_seqlen = qkv.offsets().to(dtype=torch.int32, device=qkv.device)
    
    # 获取 qkv 中最大的序列长度
    max_seqlen = qkv._get_max_seqlen()
    
    # 获取 qkv 的值的形状的第一个维度大小，即 batch 的大小
    n_elem = qkv.values().shape[0]

    # 如果 qkv 的长度信息不为 None，则计算累积序列长度
    else:
        # 计算累积长度并转换为 torch.int32 类型，并将其移动到指定设备上
        cumulative_seqlen = (
            qkv.lengths().cumsum(0).to(dtype=torch.int32, device=qkv.device)
        )
        
        # 获取 batch 大小
        batch_size = qkv.size(0)
        
        # 获取 qkv 中最大的序列长度
        max_seqlen = qkv._get_max_seqlen()
        
        # 获取累积序列长度的最后一个元素，并转换为整数类型
        n_elem = int(cumulative_seqlen[-1].item())
    
    # 返回累积序列长度、最大序列长度和元素数量
    return cumulative_seqlen, max_seqlen, n_elem
# 检查嵌套张量是否适用于flash-attention和efficient_attention内核，无需在嵌套张量输入上调用contiguous。
# 检查存储偏移量的相邻差异是否是前一个张量的常数倍，并且检查步幅是否严格递减。
# 这些检查在对嵌套张量进行转置后进行，结果是形状为[bsz, {seq_len}, num_heads, dim]的Nt。

def _is_safe_to_get_storage_as_tensor(tensor: torch.Tensor):
    assert isinstance(tensor, NestedTensor)
    # 获取嵌套张量的偏移量
    offsets = tensor.offsets()
    # 获取嵌套张量的步幅
    strides = tensor._strides

    # 计算嵌套张量中的张量数量
    n_tensors = offsets.size(0) - 1
    if n_tensors <= 1:
        return True

    # 检查张量的步幅是否严格递减
    prev_stride = strides[1]
    for stride in strides[2:]:
        if prev_stride <= stride:
            # 如果步幅不是严格递减，则返回False
            return False
        prev_stride = stride

    # 如果通过步幅检查，则返回True
    return True


def _view_as_dense(
    tensor: torch.Tensor, Nnz: int, num_heads: int, head_dim: int
) -> torch.Tensor:
    # 如果输入张量是嵌套的，则返回其值部分作为密集张量
    if tensor.is_nested:
        return tensor.values()
    # 否则，按给定的形状(Nnz, num_heads, head_dim)对输入张量进行reshape操作
    return tensor.view(Nnz, num_heads, head_dim)


# TODO: 下一轮迭代应添加测试用例并检查其工作情况
# def _sdpa_nested_preprocessing_with_broadcast(query, key, value):
#     # Query (Batch x Num_heads x {Q_seq_len}  x Dim_per_head)
#     # Key   (Batch x Num_heads x {KV_seq_len} x Dim_per_head)
#     # Value (Batch x Num_heads x {KV_seq_len} x Dim_per_head)
#     q_batch_size = query.size(0)
#     k_batch_size = key.size(0)
#     v_batch_size = value.size(0)

#     output_batch_size = max(q_batch_size, k_batch_size, v_batch_size)

#     q_num_heads = query.size(1)
#     k_num_heads = key.size(1)
#     v_num_heads = value.size(1)

#     output_num_heads = max(q_num_heads, k_num_heads, v_num_heads)

#     head_dim_qk = query.size(3)
#     head_dim_v = value.size(3)

#     q_t = query.transpose(1, 2)
#     k_t = key.transpose(1, 2)
#     v_t = value.transpose(1, 2)

#     # Checks in sdp_utils ensure that if {*}_batch_size/{*}_num_heads !=
#     # output_batch_size/num_heads then they are 1
#     q_batch_size_needs_broadcast = q_batch_size != output_batch_size
#     k_batch_size_needs_broadcast = k_batch_size != output_batch_size
#     v_batch_size_needs_broadcast = v_batch_size != output_batch_size

#     # If {*}_batch_size_needs_broadcast, then
#     # (1) max_seqlen_batch_{*} is given by {*}_t.size(1)
#     #     this is because needs_broadcast indicates that the batch_size is 1
#     #     and hence there is only 1 value for seq_len
#     # (2) The cum_seq_lens are given by [0, {*}_t.size(1), 2 * {*}_t.size(1),
#     # ..., outut_batch_size * {*}_t.size(1)]
# (3) Nnz_{*} is given by output_batch_size * {*}_t.size(1)

if q_batch_size_needs_broadcast or not q_t.is_nested:
    # Determine the maximum sequence length of the queries
    max_seqlen_batch_q = q_t.size(1)
    # Calculate cumulative sequence length for queries
    cumulative_sequence_length_q = torch.arange(
        0,
        (output_batch_size + 1) * max_seqlen_batch_q,
        max_seqlen_batch_q,
        device=q_t.device,
        dtype=torch.int32,
    )
    # Calculate Nnz_q based on output batch size and max sequence length for queries
    Nnz_q = output_batch_size * max_seqlen_batch_q
else:
    # Retrieve cumulative sequence length, max sequence length, and Nnz_q from helper function
    (
        cumulative_sequence_length_q,
        max_seqlen_batch_q,
        Nnz_q,
    ) = _cumulative_and_max_seq_len_nnz(q_t)

if k_batch_size_needs_broadcast and v_batch_size_needs_broadcast:
    # Ensure that the sequence lengths of keys and values match when broadcasting
    assert k_t.size(1) == v_t.size(1)
    # Determine the maximum sequence length of keys and values
    max_seqlen_batch_kv = k_t.size(1)
    # Calculate cumulative sequence length for keys and values
    cumulative_sequence_length_kv = torch.arange(
        0,
        (output_batch_size + 1) * max_seqlen_batch_kv,
        max_seqlen_batch_kv,
        device=k_t.device,
        dtype=torch.int32,
    )
    # Calculate Nnz_kv based on output batch size and max sequence length for keys and values
    Nnz_kv = output_batch_size * max_seqlen_batch_kv
else:
    # Retrieve cumulative sequence length, max sequence length, and Nnz_kv from helper function
    cumulative_sequence_length_kv, max_seqlen_batch_kv, Nnz_kv = (
        _cumulative_and_max_seq_len_nnz(v_t)
        if k_batch_size_needs_broadcast
        else _cumulative_and_max_seq_len_nnz(k_t)
    )

q_num_heads_needs_broadcast = q_num_heads != output_num_heads
k_num_heads_needs_broadcast = k_num_heads != output_num_heads
v_num_heads_needs_broadcast = v_num_heads != output_num_heads

if not q_t.is_nested:
    # Expand queries to match output batch size, sequence length, number of heads, and head dimension
    query_buffer_reshaped = q_t.expand(
        output_batch_size, q_t.size(1), output_num_heads, head_dim_qk
    )
    # Reshape the query buffer to match the flattened structure (Nnz_q, output_num_heads, head_dim_qk)
    query_buffer_reshaped = query_buffer_reshaped.reshape(
        Nnz_q, output_num_heads, head_dim_qk
    )
else:
    if not q_t.is_contiguous() and not _is_safe_to_get_storage_as_tensor(q_t):
        # Ensure contiguous memory layout for nested tensors
        q_t = q_t.contiguous()
    # Determine effective batch size for queries based on broadcasting needs
    effective_batch_size_q = (
        output_batch_size if q_batch_size_needs_broadcast else Nnz_q
    )
    # Reshape queries into a dense format
    query_buffer_reshaped = _view_as_dense(
        q_t, effective_batch_size_q, output_num_heads, head_dim_qk
    )

# Ensure contiguous memory layout for keys and values
if not k_t.is_contiguous() and not _is_safe_to_get_storage_as_tensor(k_t):
    k_t = k_t.contiguous()
if not v_t.is_contiguous() and not _is_safe_to_get_storage_as_tensor(v_t):
    v_t = v_t.contiguous()

# Determine effective batch size for keys based on broadcasting needs
effective_batch_size_k = (
    output_batch_size if k_batch_size_needs_broadcast else Nnz_kv
)
# Reshape keys into a dense format
key_buffer_reshaped = _view_as_dense(
    k_t, effective_batch_size_k, output_num_heads, head_dim_qk
)
# Query (Batch x Num_heads x {Q_seq_len}  x Dim_per_head)
# Key   (Batch x Num_heads x {KV_seq_len} x Dim_per_head)
# Value (Batch x Num_heads x {KV_seq_len} x Dim_per_head)
def _sdpa_nested_preprocessing(query, key, value):
    q_batch_size = query.size(0)  # 获取查询张量的批量大小
    k_batch_size = key.size(0)    # 获取键张量的批量大小
    v_batch_size = value.size(0)  # 获取值张量的批量大小

    q_num_heads = query.size(1)   # 获取查询张量的头数
    k_num_heads = key.size(1)     # 获取键张量的头数
    v_num_heads = value.size(1)   # 获取值张量的头数

    # 如果批量大小或头数不匹配，则抛出运行时错误
    if not (q_batch_size == k_batch_size and q_batch_size == v_batch_size) or not (
        q_num_heads == k_num_heads and k_num_heads == v_num_heads
    ):
        raise RuntimeError(
            "This path is currently not implemented for jagged layout NT."
        )
        # return _sdpa_nested_preprocessing_with_broadcast(query, key, value)

    num_heads = query.size(1)     # 获取查询张量的头数
    head_dim_qk = query.size(3)   # 获取查询张量维度分割后的维度
    head_dim_v = value.size(3)    # 获取值张量维度分割后的维度
    q_t = query.transpose(1, 2)   # 将查询张量转置，交换第1和第2维度
    k_t = key.transpose(1, 2)     # 将键张量转置，交换第1和第2维度
    v_t = value.transpose(1, 2)   # 将值张量转置，交换第1和第2维度

    # 计算查询张量的累积序列长度、批次最大序列长度和非零元素数目
    (
        cumulative_sequence_length_q,
        max_seqlen_batch_q,
        Nnz_q,
    ) = _cumulative_and_max_seq_len_nnz(q_t)
    # 计算键张量的累积序列长度、批次最大序列长度和非零元素数目
    (
        cumulative_sequence_length_kv,
        max_seqlen_batch_kv,
        Nnz_kv,
    ) = _cumulative_and_max_seq_len_nnz(k_t)

    # 如果查询张量、键张量或值张量不是连续的，并且不安全以张量的形式获取它们的存储，则进行连续化操作
    if not q_t.is_contiguous() and not _is_safe_to_get_storage_as_tensor(q_t):
        q_t = q_t.contiguous()
    if not k_t.is_contiguous() and not _is_safe_to_get_storage_as_tensor(k_t):
        k_t = k_t.contiguous()
    if not v_t.is_contiguous() and not _is_safe_to_get_storage_as_tensor(v_t):
        v_t = v_t.contiguous()

    # 将查询张量重塑为稠密格式，以便后续操作
    query_buffer_reshaped = _view_as_dense(q_t, Nnz_q, num_heads, head_dim_qk)
    # 将键张量重塑为稠密格式，以便后续操作
    key_buffer_reshaped = _view_as_dense(k_t, Nnz_kv, num_heads, head_dim_qk)
    # 将输入张量 v_t 重塑为密集形式，以便后续处理
    value_buffer_reshaped = _view_as_dense(v_t, Nnz_kv, num_heads, head_dim_v)
    
    # 创建包含输出注意力信息的字典
    output_nt_info = {
        "offsets": q_t.offsets(),  # 获取查询张量 q_t 的偏移量信息
        "_max_seqlen": q_t._get_max_seqlen(),  # 获取查询张量 q_t 的最大序列长度信息
        "_min_seqlen": q_t._get_min_seqlen(),  # 获取查询张量 q_t 的最小序列长度信息
    }
    
    # 返回多个值，包括重塑后的查询缓冲区、重塑后的键缓冲区、重塑后的值缓冲区，
    # 累积的查询序列长度、累积的键值序列长度、查询批次的最大序列长度、键值批次的最大序列长度，
    # 以及输出的注意力信息字典
    return (
        query_buffer_reshaped,
        key_buffer_reshaped,
        value_buffer_reshaped,
        cumulative_sequence_length_q,
        cumulative_sequence_length_kv,
        max_seqlen_batch_q,
        max_seqlen_batch_kv,
        output_nt_info,
    )
# FlashAttentionV2 要求头维度必须是8的倍数
# 之前在内核中完成了这个操作，但这可能导致内核可能别名查询、键、值
# 因此，我们在复合区域内将头维度填充为8的倍数
# 如果最后一个维度大小已经是alignment_size的倍数，则直接返回输入张量
def _pad_last_dim(
    tensor: torch.Tensor, alignment_size: int, slice: bool
) -> torch.Tensor:
    last_dim_size = tensor.size(-1)  # 获取张量的最后一个维度大小
    if last_dim_size % alignment_size == 0:  # 如果已经是alignment_size的倍数，则直接返回
        return tensor
    pad_count = alignment_size - (last_dim_size % alignment_size)  # 计算需要填充的数量
    tensor = torch.nn.functional.pad(tensor, [0, pad_count])  # 对张量进行填充操作
    if slice:
        return tensor[..., 0:last_dim_size]  # 如果需要切片操作，则返回指定范围内的张量
    return tensor  # 否则返回填充后的张量


# TODO: 与 torch/nn/utils/attention.py 合并
# 计算 softmax 的缩放因子，如果未指定缩放因子，则根据 query 的最后一个维度自动计算
def _calculate_scale(query, scale):
    # TODO: 研究为什么 Dynamo 不能正确处理 math.sqrt()
    softmax_scale = scale if scale is not None else torch.sym_sqrt(1.0 / query.size(-1))
    return softmax_scale  # 返回计算得到的 softmax 缩放因子


# 后处理 Flash 模型的输出，如果输出不是嵌套结构且最后一个维度大小不等于指定的 og_size，则进行切片操作
def _post_process_flash_output(out: torch.Tensor, og_size):
    if not out.is_nested and out.size(-1) != og_size:
        out = out[..., 0:og_size]
    return out  # 返回处理后的输出张量


# 执行不规则的缩放点积注意力操作
def jagged_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
):
    _validate_sdpa_input(query, key, value, attn_mask, dropout_p, is_causal, scale)  # 验证输入的合法性
    # 用于类型检查，确保 query、key、value 是嵌套张量类型
    assert (
        isinstance(query, NestedTensor)
        and isinstance(key, NestedTensor)
        and isinstance(value, NestedTensor)
    )
    from torch.nested._internal.nested_tensor import nested_view_from_values_offsets

    # 特殊路径处理非规则序列长度的情况，例如 SAM 中第二个批次维度是非规则的情况
    # 对于这种情况，可以直接通过标准的缩放点积注意力机制处理
    if query.dim() > 3 and key.dim() > 3 and value.dim() > 3 and query._ragged_idx == 1:
        output = F.scaled_dot_product_attention(
            query.values(),
            key.values(),
            value.values(),
            attn_mask=(
                attn_mask.values() if isinstance(attn_mask, NestedTensor) else attn_mask
            ),
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )
        return nested_view_from_values_offsets(output, query.offsets())  # 返回处理后的输出

    compute_logsumexp = query.requires_grad or key.requires_grad or value.requires_grad  # 是否需要计算 logsumexp

    backend_choice = _select_sdp_backend(
        query, key, value, attn_mask, dropout_p, is_causal
    )  # 选择合适的缩放点积注意力的后端实现
    # 如果选择的后端是 SDPBackend.FLASH_ATTENTION
    if backend_choice == SDPBackend.FLASH_ATTENTION:
        # 获取原始查询张量的最后一个维度大小
        og_size = query.size(-1)
        # 对查询、键、值张量在最后一个维度上进行填充，填充值为8，不包括0填充
        query_padded = _pad_last_dim(query, 8, False)
        key_padded = _pad_last_dim(key, 8, False)
        value_padded = _pad_last_dim(value, 8, False)
        # 根据原始头部维度大小计算比例尺
        og_scale = _calculate_scale(query, scale)
        
        # 调用内部函数进行 Flash Attention 的预处理
        (
            query_buffer_reshaped,
            key_buffer_reshaped,
            value_buffer_reshaped,
            cumulative_sequence_length_q,
            cumulative_sequence_length_kv,
            max_seqlen_batch_q,
            max_seqlen_batch_kv,
            output_nt_info,
        ) = _sdpa_nested_preprocessing(query_padded, key_padded, value_padded)

        # 调用底层 C++ 实现的 Flash Attention 前向计算
        (
            attention,
            logsumexp,
            philox_seed,
            philox_offset,
            debug_attn_mask,
        ) = torch.ops.aten._flash_attention_forward(
            query_buffer_reshaped,
            key_buffer_reshaped,
            value_buffer_reshaped,
            cumulative_sequence_length_q,
            cumulative_sequence_length_kv,
            max_seqlen_batch_q,
            max_seqlen_batch_kv,
            dropout_p,
            is_causal,
            False,
            scale=og_scale,
        )

        # 将注意力张量从压缩状态转换为批量大小和序列长度的形式
        attention = nested_view_from_values_offsets(
            attention.squeeze(0),
            output_nt_info["offsets"],
            min_seqlen=output_nt_info["_min_seqlen"],
            max_seqlen=output_nt_info["_max_seqlen"],
        ).transpose(1, 2)
        
        # 返回经过 Flash Attention 后处理的输出
        return _post_process_flash_output(attention, og_size)
    
    # 如果选择的后端是 SDPBackend.EFFICIENT_ATTENTION
    elif backend_choice == SDPBackend.EFFICIENT_ATTENTION:
        # 进行 Efficient Attention 的预处理
        (
            query_reshaped,
            key_reshaped,
            value_reshaped,
            cumulative_sequence_length_q,
            cumulative_sequence_length_kv,
            max_seqlen_batch_q,
            max_seqlen_batch_kv,
            output_nt_info,
        ) = _sdpa_nested_preprocessing(query, key, value)
        
        # 调用底层 C++ 实现的 Efficient Attention 前向计算
        (
            attention,
            log_sumexp,
            seed,
            offset,
            max_seqlen_q,
            max_seqlen_batch_kv,
        ) = torch.ops.aten._efficient_attention_forward(
            query_reshaped.unsqueeze(0),
            key_reshaped.unsqueeze(0),
            value_reshaped.unsqueeze(0),
            None,
            cumulative_sequence_length_q,
            cumulative_sequence_length_kv,
            max_seqlen_batch_q,
            max_seqlen_batch_kv,
            dropout_p,
            int(is_causal),
            compute_logsumexp,
            scale=scale,
        )

        # 将注意力张量从压缩状态转换为批量大小和序列长度的形式
        return nested_view_from_values_offsets(
            attention.squeeze(0),
            output_nt_info["offsets"],
            min_seqlen=output_nt_info["_min_seqlen"],
            max_seqlen=output_nt_info["_max_seqlen"],
        ).transpose(1, 2)
    elif backend_choice == SDPBackend.MATH:
        # 如果选择的后端是MATH，则执行以下操作

        # 保存输入的偏移和形状信息，以便重塑最终输出
        offsets = query.offsets()
        d1 = query._size[1]
        d2 = value._size[-1]

        # 获取查询和键的最小序列长度和最大序列长度
        min_seqlen_tensor = query._metadata_cache.get(
            "min_seqlen", None
        )  # type: ignore[attr-defined]
        max_seqlen_tensor = query._metadata_cache.get(
            "max_seqlen", None
        )  # type: ignore[attr-defined]

        # 将嵌套张量从不规则布局转换为分步布局，以支持SDPA的数学实现
        def get_strided_layout_nested_tensor(jagged_layout_nt):
            lengths = jagged_layout_nt._offsets[1:] - jagged_layout_nt._offsets[:-1]
            transpose = torch.transpose(jagged_layout_nt, 1, 2)
            tensor_list = transpose.values().split(list(lengths), dim=0)
            strided_nt = torch.nested.as_nested_tensor(list(tensor_list))
            strided_nt = strided_nt.transpose(1, 2).contiguous()
            return strided_nt

        # 转换查询、键和值为分步布局的嵌套张量
        query = get_strided_layout_nested_tensor(query)
        key = get_strided_layout_nested_tensor(key)
        value = get_strided_layout_nested_tensor(value)

        # 使用数学实现的缩放点积注意力机制计算
        attn_out = torch._scaled_dot_product_attention_math(
            query, key, value, attn_mask, dropout_p, is_causal, scale=scale
        )[0]

        # 从张量库中加载值，将分步布局的嵌套张量转换回不规则布局
        from torch.nested._internal.nested_tensor import _load_val_from_tensor
        attn_out = attn_out.transpose(1, 2).contiguous().values()
        attn_out = attn_out.view(-1, d1, d2)
        attn_out = nested_view_from_values_offsets(
            attn_out,
            offsets,
            min_seqlen=(
                None
                if min_seqlen_tensor is None
                else _load_val_from_tensor(min_seqlen_tensor)
            ),
            max_seqlen=(
                None
                if max_seqlen_tensor is None
                else _load_val_from_tensor(max_seqlen_tensor)
            ),
        ).transpose(1, 2)

        # 返回转换后的注意力输出
        return attn_out
    else:
        # 如果没有找到适合 scaled_dot_product_attention 的可行后端，则引发运行时错误
        raise RuntimeError(
            "No viable backend for scaled_dot_product_attention was found."
        )
```