# `.\lucidrains\ring-attention-pytorch\ring_attention_pytorch\ring_flash_attention_cuda.py`

```
# 导入数学库
import math
# 导入 functools 库中的 partial 函数
from functools import partial
# 导入 typing 库中的 Optional 和 Tuple 类型
from typing import Optional, Tuple
# 导入 packaging 库中的 version 模块
import packaging.version as pkg_version

# 导入 torch 库
import torch
# 从 torch 库中导入 nn, einsum, Tensor 模块
from torch import nn, einsum, Tensor
# 从 torch 库中导入 F 模块
import torch.nn.functional as F
# 从 torch.autograd.function 中导入 Function 类
from torch.autograd.function import Function

# 从 ring_attention_pytorch.ring 模块中导入相关函数
from ring_attention_pytorch.ring import (
    ring_pass,
    all_ring_pass,
    null_ring_pass,
    one_ring_pass,
    get_rank,
    get_world_size
)

# 从 beartype 库中导入 beartype 函数
from beartype import beartype

# 从 einops 库中导入 repeat, rearrange 函数
from einops import repeat, rearrange

# 定义函数 exists，判断变量是否存在
def exists(v):
    return v is not None

# 定义函数 pad_at_dim，对张量在指定维度进行填充
def pad_at_dim(t, pad: Tuple[int, int], *, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

# 定义函数 is_contiguous，判断张量是否是连续的
def is_contiguous(x):
    return x.stride(-1) == 1

# 确保 flash attention 已安装用于反向传播
import importlib
from importlib.metadata import version

# 断言 flash-attn 必须已安装
assert exists(importlib.util.find_spec('flash_attn')), 'flash-attn must be installed. `pip install flash-attn --no-build-isolation` first'

# 获取 flash-attn 版本信息
flash_attn_version = version('flash_attn')
# 断言 flash-attn 版本大于等于 2.5.1
assert pkg_version.parse(flash_attn_version) >= pkg_version.parse('2.5.1')

# 从 flash_attn.flash_attn_interface 模块中导入相关函数
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_backward,
    _flash_attn_backward
)

# 确保 triton 已安装用于前向传播
assert exists(importlib.util.find_spec('triton')), 'latest triton must be installed. `pip install triton -U` first'

# 获取 triton 版本信息
triton_version = version('triton')
# 断言 triton 版本大于等于 2.1
assert pkg_version.parse(triton_version) >= pkg_version.parse('2.1')

# 导入 triton 库
import triton
# 从 triton.language 中导入 tl 模块

import triton.language as tl

# 从 Tri 的 flash_attn 仓库中获取 flash attention 前向传播代码，并进行修改以返回未归一化的累积值、行最大值和行 lse - 减少通过环传递

@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Bias,
    Out,
    M,
    Lse,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    HAS_BIAS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    CAUSAL_MASK_DIAGONAL: tl.constexpr,
    LOAD_ACCUMULATED: tl.constexpr,
    RETURN_NORMALIZED_OUTPUT: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    k_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    )
    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    )

    if HAS_BIAS:
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n

    # 最大值

    m_ptrs = M + off_hb * seqlen_q_rounded + offs_m

    if LOAD_ACCUMULATED:
        m_i = tl.load(m_ptrs)
    else:
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # 加载 lse

    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m

    if LOAD_ACCUMULATED:
        lse_i = tl.load(lse_ptrs)
    else:
        # 如果条件不成立，创建一个形状为 [BLOCK_M]，数据类型为 float32 的张量，并填充为负无穷大
        lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # 加载累积输出的偏移量
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # 计算输出指针的位置
    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )

    # 如果需要加载累积值
    if LOAD_ACCUMULATED:
        # 如果 BLOCK_M 是偶数
        if EVEN_M:
            # 如果 BLOCK_HEADDIM 是偶数
            if EVEN_HEADDIM:
                acc_o = tl.load(out_ptrs)
            else:
                acc_o = tl.load(out_ptrs, mask=offs_d[None, :] < headdim)
        else:
            # 如果 BLOCK_HEADDIM 是偶数
            if EVEN_HEADDIM:
                acc_o = tl.load(out_ptrs, mask=offs_m[:, None] < seqlen_q)
            else:
                acc_o = tl.load(
                    out_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
                )

        acc_o = acc_o.to(tl.float32)
    else:
        # 创建一个形状为 [BLOCK_M, BLOCK_HEADDIM]，数据类型为 float32 的零张量
        acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    # 加载查询、键、值
    if EVEN_M & EVEN_N:
        # 如果 BLOCK_M 和 BLOCK_N 都是偶数
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        # 如果 BLOCK_M 和 BLOCK_N 不都是偶数
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0
            )

    # 计算结束位置
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    # 循环遍历起始位置，每次增加 BLOCK_N
    for start_n in range(0, end_n, BLOCK_N):
        # 将 start_n 调整为 BLOCK_N 的倍数
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # 根据条件判断是否加载 k
        if EVEN_N & EVEN_M:
            # 根据条件加载 k
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        # 初始化 qk
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # 计算 qk
        qk += tl.dot(q, tl.trans(k))

        # 根据条件判断是否添加特定值到 qk
        if not EVEN_N:
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))

        # 根据条件判断是否添加特定值到 qk
        if IS_CAUSAL:
            if CAUSAL_MASK_DIAGONAL:
                # 为 stripe attention 需要的操作
                qk += tl.where(offs_m[:, None] > (start_n + offs_n)[None, :], 0, float("-inf"))
            else:
                qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))

        # 根据条件判断是否添加偏置到 qk
        if HAS_BIAS:
            if EVEN_N:
                bias = tl.load(b_ptrs + start_n)
            else:
                bias = tl.load(
                    b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0
                )
            bias = bias[None, :]

            bias = bias.to(tl.float32)
            qk = qk * softmax_scale + bias
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])

        # 计算 l_ij
        l_ij = tl.sum(p, 1)

        # 计算 acc_o_scale
        acc_o_scale = tl.exp(m_i - m_ij)
        acc_o = acc_o * acc_o_scale[:, None]

        # 根据条件判断是否加载 v
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )

        # 将 p 转换为与 v 相同的数据类型
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # -- 更新统计信息

        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    # 如果需要返回归一化的输出
    if RETURN_NORMALIZED_OUTPUT:
        acc_o_scale = tl.exp(m_i - lse_i)
        acc_o = acc_o * acc_o_scale[:, None]

    # 计算 m 和 lse 的偏移量

    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # 写回 lse 和 m

    tl.store(lse_ptrs, lse_i)

    if not RETURN_NORMALIZED_OUTPUT:
        tl.store(m_ptrs, m_i)

    # 写入输出

    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(
                out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
            )
# 定义 flash attention 的前向传播函数
def flash_attn_forward(
    q,
    k,
    v,
    bias = None,
    causal = False,
    o = None,
    m = None,
    lse = None,
    softmax_scale = None,
    causal_mask_diagonal = False,
    return_normalized_output = False,
    load_accumulated = True
):
    # 如果输入的张量不是连续的，则将其转换为连续的张量
    q, k, v = [x if is_contiguous(x) else x.contiguous() for x in (q, k, v)]

    # 获取输入张量的形状信息
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape

    # 断言输入张量的形状符合要求
    assert k.shape == (batch, seqlen_k, nheads, d)
    assert v.shape == (batch, seqlen_k, nheads, d)
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda

    # 设置 softmax 的缩放因子
    softmax_scale = default(softmax_scale, d ** -0.5)

    # 检查是否存在偏置项
    has_bias = exists(bias)

    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda

        # 如果偏置项是二维的，则进行扩展
        if bias.ndim == 2:
            bias = repeat(bias, 'b j -> b h i j', h = nheads, i = seqlen_q)

        # 如果偏置项不是连续的，则转换为连续的张量
        if not is_contiguous(bias):
            bias = bias.contiguous()

        assert bias.shape[-2:] == (1, seqlen_k)
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)

    # 记录偏置项的步长信息
    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)

    # 对序列长度进行向上取整，使其能够被 128 整除
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128

    # 初始化 lse 张量
    if not exists(lse):
        max_neg_value = -torch.finfo(torch.float32).max
        init_fn = partial(torch.full, fill_value = max_neg_value) if load_accumulated else torch.empty
        lse = init_fn((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)

    # 初始化 m 张量
    if not exists(m):
        max_neg_value = -torch.finfo(torch.float32).max
        init_fn = partial(torch.full, fill_value = max_neg_value) if load_accumulated else torch.empty
        m = init_fn((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)

    # 初始化输出张量 o
    if not exists(o):
        init_fn = torch.zeros_like if load_accumulated else torch.empty_like
        o = init_fn(q)

    # 设置 BLOCK_HEADDIM 和 BLOCK 的值
    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK = 128
    num_warps = 4 if d <= 64 else 8
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)

    # 调用 _fwd_kernel 函数进行前向传播计算
    _fwd_kernel[grid](
        q,
        k,
        v,
        bias,
        o,
        m,
        lse,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        *bias_strides,
        o.stride(0),
        o.stride(2),
        o.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,
        has_bias,
        causal,
        causal_mask_diagonal,
        load_accumulated,
        return_normalized_output,
        BLOCK_HEADDIM,
        BLOCK_M = BLOCK,
        BLOCK_N = BLOCK,
        num_warps = num_warps,
        num_stages = 1,
    )

    # 返回输出张量 o, m, lse
    return o, m, lse

# 辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 返回默认值
def default(val, d):
    return val if exists(val) else d

# 判断一个数是否能被另一个数整除
def divisible_by(num, den):
    return (num % den) == 0

# ring + (flash) attention forwards and backwards

# flash attention v1 - https://arxiv.org/abs/2205.14135
# flash attention v2 - https://tridao.me/publications/flash2/flash2.pdf
# ring attention - https://arxiv.org/abs/2310.01889

# 定义 RingFlashAttentionCUDAFunction 类
class RingFlashAttentionCUDAFunction(Function):

    # 前向传播函数
    @staticmethod
    @torch.no_grad()
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor],
        causal: bool,
        bucket_size: int,
        ring_reduce_col: bool,
        striped_ring_attn: bool,
        max_lookback_seq_len: Optional[int],
        ring_size: Optional[int]
    @staticmethod
    @torch.no_grad()
# 将自定义的 CUDA 函数应用到环形闪光注意力机制上
ring_flash_attn_cuda_ = RingFlashAttentionCUDAFunction.apply

# 定义环形闪光注意力机制的 CUDA 函数
@beartype
def ring_flash_attn_cuda(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[Tensor] = None,
    causal: bool = False,
    bucket_size: int = 1024,
    ring_reduce_col: bool = False,
    striped_ring_attn: bool = False,
    max_lookback_seq_len: Optional[int] = None,
    ring_size: Optional[int] = None
):
    # 调用环形闪光注意力机制的 CUDA 函数，传入参数并返回结果
    return ring_flash_attn_cuda_(q, k, v, mask, causal, bucket_size, ring_reduce_col, striped_ring_attn, max_lookback_seq_len, ring_size)
```