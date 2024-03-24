# `.\lucidrains\magvit2-pytorch\magvit2_pytorch\attend.py`

```py
# 导入所需模块和库
from functools import partial
from typing import Optional, Tuple

import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F

from collections import namedtuple
from functools import wraps
from packaging import version
from dataclasses import dataclass

from einops import rearrange, repeat

# 定义一个命名元组EfficientAttentionConfig，包含三个布尔类型的参数
EfficientAttentionConfig = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# 辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 如果变量存在则返回其值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 过滤掉列表中的空值
def compact(arr):
    return [*filter(exists, arr)]

# 保证函数只执行一次
def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

# 打印函数的输出，确保只打印一次
print_once = once(print)

# 用于创建因果掩码的函数
# 针对onnx cpu需要特殊处理（不支持.triu）

# 创建因果掩码
def create_causal_mask(i, j, device):
    return torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)

# 针对onnx创建因果掩码
def onnx_create_causal_mask(i, j, device):
    r = torch.arange(i, device = device)
    causal_mask = rearrange(r, 'i -> i 1') < rearrange(r, 'j -> 1 j')
    causal_mask = F.pad(causal_mask, (j - i, 0), value = False)
    return causal_mask

# 主类

class Attend(nn.Module):
    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        heads = None,
        scale = None,
        flash = False,
        onnxable = False,
        sdp_kwargs: dict = dict(
            enable_flash = True,
            enable_math = True,
            enable_mem_efficient = True
        )
    ):
        super().__init__()
        self.scale = scale

        self.causal = causal
        self.create_causal_mask = onnx_create_causal_mask if onnxable else create_causal_mask

        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        # flash attention

        # 检查是否支持flash attention
        self.flash = flash and torch.cuda.is_available()
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        self.sdp_kwargs = sdp_kwargs

    def flash_attn(
        self,
        q, k, v,
        mask = None,
        attn_bias = None
    ):
        # 解包输入张量的形状和其他属性
        batch, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        # 使输入张量连续
        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        # 处理缩放，因为在 sdp 中缩放不可定制，对其进行处理
        if exists(self.scale):
            q = q * self.scale / (q.shape[-1] ** -0.5)

        # 检查是否存在 mask 并扩展到兼容的形状
        causal = self.causal

        # 如果 q_len == 1 且 causal 为真，则将 causal 设置为 False
        if q_len == 1 and causal:
            causal = False

        # 扩展键填充 mask
        if exists(mask):
            assert mask.ndim == 4
            mask = mask.expand(batch, heads, q_len, k_len)

        # 处理 kv 缓存
        if k_len > q_len and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device=device)
            if not exists(mask):
                mask = ~causal_mask
            else:
                mask = mask & ~causal_mask
            causal = False

        # 手动处理 causal mask，如果给定了另一个 mask
        row_is_entirely_masked = None
        if exists(mask) and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device=device)
            mask = mask & ~causal_mask

            # 防止整行被屏蔽
            row_is_entirely_masked = ~mask.any(dim=-1)
            mask[..., 0] = mask[..., 0] | row_is_entirely_masked
            causal = False

        # 处理 alibi 位置偏差，将 bool 转换为 float
        if exists(attn_bias):
            attn_bias = rearrange(attn_bias, 'h i j -> 1 h i j').expand(batch, heads, -1, -1)

            mask_value = -torch.finfo(q.dtype).max

            if exists(mask):
                attn_bias = attn_bias.masked_fill(~mask, mask_value // 2)
            elif causal:
                causal_mask = self.create_causal_mask(q_len, k_len, device=device)
                attn_bias = attn_bias.masked_fill(causal_mask, mask_value // 2)
                causal = False

            mask = attn_bias

        # 使用 scaled_dot_product_attention 处理注意力
        with torch.backends.cuda.sdp_kernel(**self.sdp_kwargs):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.,
                is_causal=causal
            )

        # 对于整行被完全屏蔽的情况，将输出的该行标记为 0
        if exists(row_is_entirely_masked):
            out = out.masked_fill(row_is_entirely_masked[..., None], 0.)

        return out

    # 前向传播函数
    def forward(
        self,
        q, k, v,
        mask=None,
        attn_bias=None,
        prev_attn=None
        ):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        # 获取输入张量的形状信息
        n, heads, kv_heads, device = q.shape[-2], q.shape[1], k.shape[1], q.device

        # 计算缩放因子
        scale = default(self.scale, q.shape[-1] ** -0.5)

        # 获取是否为因果注意力的标志
        causal = self.causal

        # 处理缓存的键值对解码
        if n == 1 and causal:
            causal = False

        # 处理零键值对，允许网络关注空内容
        if self.flash:
            assert not exists(prev_attn), 'residual attention not compatible with flash attention'
            return self.flash_attn(q, k, v, mask = mask, attn_bias = attn_bias)

        # 计算点积注意力得分
        dots = einsum(f'b h i d, b h j d -> b h i j', q, k) * scale

        # 如果存在先前的注意力，加上先前的注意力得分
        if exists(prev_attn):
            dots = dots + prev_attn

        # 如果存在注意力偏置，加上注意力偏置
        if exists(attn_bias):
            dots = dots + attn_bias

        # 获取点积张量的形状信息和数据类型
        i, j, dtype = *dots.shape[-2:], dots.dtype

        # 定义掩码值
        mask_value = -torch.finfo(dots.dtype).max

        # 如果存在掩码，用掩码值填充不需要关注的位置
        if exists(mask):
            dots = dots.masked_fill(~mask, mask_value)

        # 如果是因果注意力，创建因果掩码并用掩码值填充
        if causal:
            causal_mask = self.create_causal_mask(i, j, device = device)
            dots = dots.masked_fill(causal_mask, mask_value)

        # 计算注意力权重
        attn = dots.softmax(dim = -1)

        # 对注意力权重进行dropout
        attn = self.attn_dropout(attn)

        # 计算输出
        out = einsum(f'b h i j, b h j d -> b h i d', attn, v)

        return out
```