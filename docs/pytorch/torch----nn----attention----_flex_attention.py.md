# `.\pytorch\torch\nn\attention\_flex_attention.py`

```
"""
This module implements the user facing API for flex_attention in PyTorch.
"""
# 导入必要的库和模块
import functools
from typing import Callable, Optional

import torch
from torch._higher_order_ops.flex_attention import flex_attention as flex_attention_hop
from torch._higher_order_ops.utils import _set_compilation_env
from torch.fx.experimental.proxy_tensor import (
    _temp_remove_pre_dispatch_torch_function_mode,
)
from torch.nn.attention._utils import _validate_sdpa_input


def _compose(*fs):
    """Compose a sequence of score_mod functions."""
    # 定义一个函数用于组合多个函数
    def compose2(f, g):
        # 内部函数，对两个函数进行组合
        def inner(score, b, h, m, n):
            return f(g(score, b, h, m, n), b, h, m, n)

        return inner

    return functools.reduce(compose2, fs)


_score_mod_signature = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]


def _identity(
    score: torch.Tensor,
    batch: torch.Tensor,
    head: torch.Tensor,
    token_q: torch.Tensor,
    token_kv: torch.Tensor,
) -> torch.Tensor:
    # 返回输入的分数张量，即身份函数
    return score


_DEFAULT_SPARSE_BLOCK_SIZE = 128


class _BlockSparseMask:
    kv_num_blocks: torch.Tensor
    kv_indices: torch.Tensor
    q_num_blocks: torch.Tensor
    q_indices: torch.Tensor
    KV_BLOCK_SIZE: int
    Q_BLOCK_SIZE: int

    def __init__(
        self,
        kv_num_blocks,
        kv_indices,
        q_num_blocks,
        q_indices,
        KV_BLOCK_SIZE=_DEFAULT_SPARSE_BLOCK_SIZE,
        Q_BLOCK_SIZE=_DEFAULT_SPARSE_BLOCK_SIZE,
    ):
        # 初始化_BlockSparseMask类的实例
        self.kv_num_blocks = kv_num_blocks
        self.kv_indices = kv_indices
        self.q_num_blocks = q_num_blocks
        self.q_indices = q_indices
        self.KV_BLOCK_SIZE = KV_BLOCK_SIZE
        self.Q_BLOCK_SIZE = Q_BLOCK_SIZE


def broadcast_to_dim(x, dim):
    # 将张量 x 广播到指定维度 dim
    while x.dim() < dim:
        x = x.unsqueeze(0)
    return x


def _convert_mask_to_block_mask(
    mask,
    KV_BLOCK_SIZE=_DEFAULT_SPARSE_BLOCK_SIZE,
    Q_BLOCK_SIZE=_DEFAULT_SPARSE_BLOCK_SIZE,
):
    # 将掩码转换为块掩码
    assert mask.dtype == torch.bool
    mask = broadcast_to_dim(mask, 4)
    B, H, Q, KV = mask.shape
    assert Q % Q_BLOCK_SIZE == 0
    assert KV % KV_BLOCK_SIZE == 0
    mask = mask.view(
        B, H, Q // Q_BLOCK_SIZE, Q_BLOCK_SIZE, KV // KV_BLOCK_SIZE, KV_BLOCK_SIZE
    )  # [B, H, Q//Q_BLOCK_SIZE, Q_BLOCK_SIZE, KV//KV_BLOCK_SIZE, KV_BLOCK_SIZE]
    mask = mask.permute(
        0, 1, 2, 4, 3, 5
    )  # [B, H, Q//Q_BLOCK_SIZE, KV//KV_BLOCK_SIZE, Q_BLOCK_SIZE, KV_BLOCK_SIZE]
    mask = mask.sum(dim=[-2, -1]) > 0  # [B, H, Q//Q_BLOCK_SIZE, KV//KV_BLOCK_SIZE]
    return mask


def _convert_block_mask_to_mask(
    block_mask,
    KV_BLOCK_SIZE=_DEFAULT_SPARSE_BLOCK_SIZE,
    Q_BLOCK_SIZE=_DEFAULT_SPARSE_BLOCK_SIZE,
):
    # 将块掩码转换为掩码
    assert block_mask.dim() == 4
    B, H, Q, KV = block_mask.shape
    block_mask = block_mask.expand(Q_BLOCK_SIZE, KV_BLOCK_SIZE, *block_mask.shape)
    block_mask = block_mask.permute(2, 3, 4, 0, 5, 1).reshape(
        B, H, Q * Q_BLOCK_SIZE, KV * KV_BLOCK_SIZE
    )
    # 返回当前函数中的变量 block_mask，即已经处理好的掩码数据
    return block_mask
# 创建一个稀疏块掩码，将输入的掩码转换为块掩码
def _create_block_sparse_mask(
    mask: torch.Tensor,
    KV_BLOCK_SIZE: int = _DEFAULT_SPARSE_BLOCK_SIZE,
    Q_BLOCK_SIZE: int = _DEFAULT_SPARSE_BLOCK_SIZE,
):
    # 调用函数将输入的掩码转换为块掩码
    block_mask = _convert_mask_to_block_mask(
        mask, KV_BLOCK_SIZE=KV_BLOCK_SIZE, Q_BLOCK_SIZE=Q_BLOCK_SIZE
    )
    # 将块掩码转换为 torch.int8 类型
    block_mask = block_mask.to(dtype=torch.int8)
    # 计算每个 kv 方向的块数目
    kv_num_blocks = block_mask.sum(dim=3)
    # 对块掩码在第三个维度（KV方向）按值排序，返回索引
    kv_indices = torch.argsort(block_mask, dim=3, descending=True, stable=True)
    # 计算每个 q 方向的块数目
    q_num_blocks = block_mask.sum(dim=2)
    # 对块掩码在第二个维度（Q方向）按值排序，返回索引，并进行转置
    q_indices = torch.argsort(block_mask, dim=2, descending=True, stable=True).permute(
        0, 1, 3, 2
    )
    # 返回一个包含稀疏块掩码信息的对象
    return _BlockSparseMask(
        kv_num_blocks=kv_num_blocks.to(torch.int32).to(mask.device).contiguous(),
        kv_indices=kv_indices.to(torch.int32).to(mask.device).contiguous(),
        q_num_blocks=q_num_blocks.to(torch.int32).to(mask.device).contiguous(),
        q_indices=q_indices.to(torch.int32).to(mask.device).contiguous(),
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
        Q_BLOCK_SIZE=Q_BLOCK_SIZE,
    )


"""
    使用块稀疏技术实现灵活的注意力机制内核，
    仅计算未掩码的块以获得最佳性能。
    如果用户未指定任何块稀疏掩码信息，
    我们创建此空的块稀疏掩码，
    将所有块都设置为未掩码作为默认值。
"""


# 创建一个空的块稀疏掩码对象
def _create_empty_block_sparse_mask(query, key, value):
    device = query.device
    kv_len = key.size()[-2]
    q_len = query.size()[-2]
    # 返回一个包含全为未掩码的块稀疏掩码对象
    return _BlockSparseMask(
        kv_num_blocks=torch.ones([1, 1, 1], dtype=torch.int32, device=device),
        kv_indices=torch.zeros([1, 1, 1, 1], dtype=torch.int32, device=device),
        q_num_blocks=torch.ones([1, 1, 1], dtype=torch.int32, device=device),
        q_indices=torch.zeros([1, 1, 1, 1], dtype=torch.int32, device=device),
        KV_BLOCK_SIZE=kv_len,
        Q_BLOCK_SIZE=q_len,
    )


# 灵活的注意力函数，实现带有任意注意力分数修改函数的缩放点积注意力
def _flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: _score_mod_signature = _identity,
    block_sparse_mask: Optional[_BlockSparseMask] = None,
) -> torch.Tensor:
    r"""This function implements scaled dot product attention with an arbitrary attention score modification function.

    This function computes the scaled dot product attention between query, key, and value tensors with a user-defined
    attention score modification function. The attention score modification function will be applied after the attention
    scores have been calculated between the query and key tensors. The attention scores are calculated as follows:

    The ``score_mod`` function should have the following signature:

    .. code-block:: python

        def score_mod(
            score: torch.Tensor,
            batch: torch.Tensor,
            head: torch.Tensor,
            token_q: torch.Tensor,
            token_kv: torch.Tensor
        ) -> torch.Tensor:
    """
    if block_sparse_mask is None:
        # 如果块稀疏掩码为空，则创建一个空的块稀疏掩码
        block_sparse_mask = _create_empty_block_sparse_mask(query, key, value)
    if torch.compiler.is_dynamo_compiling():
        # 如果正在使用动态编译，则将 query、key、value 张量标记为静态
        for x in [query, key, value]:
            torch._dynamo.mark_static(x, -1)
        # 调用灵活注意力的单次跳跃计算
        out, _ = flex_attention_hop(
            query,
            key,
            value,
            score_mod,
            block_sparse_mask.kv_num_blocks,
            block_sparse_mask.kv_indices,
            block_sparse_mask.q_num_blocks,
            block_sparse_mask.q_indices,
            block_sparse_mask.KV_BLOCK_SIZE,
            block_sparse_mask.Q_BLOCK_SIZE,
        )
        # 返回计算得到的输出
        return out

    # 对输入进行基本验证
    _validate_sdpa_input(query, key, value)
    # 如果 query 的倒数第二个维度不是 128 的倍数，抛出异常
    if query.size(-2) % 128 != 0:
        raise ValueError("NYI: S and L must be a multiple of 128")

    # 如果动态编译不支持，抛出运行时错误
    if not torch._dynamo.is_dynamo_supported():
        raise RuntimeError("flex_attention requires dynamo support.")
    # 使用 _set_compilation_env() 设置编译环境上下文
    with _set_compilation_env():
        # 使用 torch._dynamo.utils.disable_cache_limit() 禁用缓存限制
        with torch._dynamo.utils.disable_cache_limit():
            # 使用 _temp_remove_pre_dispatch_torch_function_mode() 移除预调度的 Torch 函数模式
            with _temp_remove_pre_dispatch_torch_function_mode():
                # 调用 torch.compile() 进行编译操作，使用 "eager" 后端，并获取完整图形
                out, _ = torch.compile(
                    flex_attention_hop, backend="eager", fullgraph=True
                )(
                    # 传递以下参数给编译函数
                    query,                          # 查询张量
                    key,                            # 键张量
                    value,                          # 值张量
                    score_mod,                      # 分数修正张量
                    block_sparse_mask.kv_num_blocks, # KV 块稀疏掩码的块数
                    block_sparse_mask.kv_indices,    # KV 块稀疏掩码的索引
                    block_sparse_mask.q_num_blocks,  # Q 块稀疏掩码的块数
                    block_sparse_mask.q_indices,     # Q 块稀疏掩码的索引
                    block_sparse_mask.KV_BLOCK_SIZE, # KV 块大小
                    block_sparse_mask.Q_BLOCK_SIZE,  # Q 块大小
                )
                # 返回编译结果 out
                return out
# 定义一个函数 `_causal`，用于计算注意力分数，根据 token_q 和 token_kv 的比较决定是否保留分数或设置为负无穷
def _causal(
    score: torch.Tensor,
    batch: torch.Tensor,
    head: torch.Tensor,
    token_q: torch.Tensor,
    token_kv: torch.Tensor,
) -> torch.Tensor:
    return torch.where(token_q >= token_kv, score, float("-inf"))


# 定义一个函数 `_rel_bias`，用于计算注意力分数，加上 token_q 和 token_kv 的差异
def _rel_bias(
    score: torch.Tensor,
    batch: torch.Tensor,
    head: torch.Tensor,
    token_q: torch.Tensor,
    token_kv: torch.Tensor,
) -> torch.Tensor:
    return score + (token_q - token_kv)


# 定义一个函数 `_rel_causal`，结合 `_causal` 和 `_rel_bias` 的逻辑，根据 token_q 和 token_kv 的比较决定是否保留分数或设置为负无穷，并加上 token_q 和 token_kv 的差异
def _rel_causal(
    score: torch.Tensor,
    batch: torch.Tensor,
    head: torch.Tensor,
    token_q: torch.Tensor,
    token_kv: torch.Tensor,
) -> torch.Tensor:
    return torch.where(token_q >= token_kv, score + (token_q - token_kv), float("-inf"))


# 定义一个函数 `_generate_alibi_bias`，生成一个内部函数 `_alibi_bias`，用于计算注意力分数，加上 token_kv 和 token_q 的差异乘以缩放因子
def _generate_alibi_bias(num_heads: int):
    def _alibi_bias(
        score: torch.Tensor,
        batch: torch.Tensor,
        head: torch.Tensor,
        token_q: torch.Tensor,
        token_kv: torch.Tensor,
    ) -> torch.Tensor:
        scale = torch.exp2(-((head + 1) * 8.0 / num_heads))
        return score + (token_kv - token_q) * scale

    return _alibi_bias
```