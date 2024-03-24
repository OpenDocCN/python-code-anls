# `.\lucidrains\ring-attention-pytorch\ring_attention_pytorch\ring_flash_attention.py`

```py
# 导入数学库
import math
# 导入 functools 库中的 partial 函数
from functools import partial
# 导入 typing 库中的 Optional 类型
from typing import Optional

# 导入 torch 库
import torch
# 从 torch 库中导入 nn、einsum、Tensor 类
from torch import nn, einsum, Tensor
# 从 torch.autograd.function 中导入 Function 类
from torch.autograd.function import Function

# 导入 einx 库
import einx
# 从 einx 库中导入 rearrange 函数
from einx import rearrange

# 导入 ring_attention_pytorch.ring 模块中的函数
from ring_attention_pytorch.ring import (
    ring_pass,
    all_ring_pass,
    null_ring_pass,
    one_ring_pass,
    get_rank,
    get_world_size
)

# 导入 beartype 库中的 beartype 装饰器
from beartype import beartype

# 常量定义
EPSILON = 1e-10

# 辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 如果变量存在则返回其值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 判断一个数是否可以被另一个数整除
def divisible_by(num, den):
    return (num % den) == 0

# 生成一个无限循环产生 None 的迭代器
def none_iterator():
    while True:
        yield None

# 根据条件切分张量
def maybe_split(t, size, dim = -2):
    if not exists(t):
        return none_iterator()

    return t.split(size, dim = dim)

# ring + (flash) attention 前向和后向

# flash attention v1 - https://arxiv.org/abs/2205.14135
# flash attention v2 - https://tridao.me/publications/flash2/flash2.pdf
# ring attention - https://arxiv.org/abs/2310.01889

# 定义 RingFlashAttentionFunction 类
class RingFlashAttentionFunction(Function):

    # 静态方法，用于前向传播
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
# 调用 RingFlashAttentionFunction 类的 apply 方法
ring_flash_attn_ = RingFlashAttentionFunction.apply

# 使用 beartype 装饰器定义 ring_flash_attn 函数
@beartype
def ring_flash_attn(
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
    # 调用 ring_flash_attn_ 函数
    return ring_flash_attn_(q, k, v, mask, causal, bucket_size, ring_reduce_col, striped_ring_attn, max_lookback_seq_len, ring_size)
```