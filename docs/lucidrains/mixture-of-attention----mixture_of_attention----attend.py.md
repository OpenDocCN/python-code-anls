# `.\lucidrains\mixture-of-attention\mixture_of_attention\attend.py`

```py
# 导入必要的库
from collections import namedtuple
from functools import wraps
from packaging import version

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

# 定义一个命名元组EfficientAttentionConfig，用于存储配置信息
EfficientAttentionConfig = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# 定义辅助函数exists，用于检查变量是否存在
def exists(val):
    return val is not None

# 定义装饰器once，确保函数只被调用一次
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

# 用once装饰print函数，确保只打印一次
print_once = once(print)

# 主要类Attend
class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        causal = False,
        flash = False
    ):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.causal = causal
        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # 确定cuda和cpu的高效注意力配置
        self.cpu_config = EfficientAttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = EfficientAttentionConfig(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = EfficientAttentionConfig(False, True, True)

    # 生成mask
    def get_mask(self, i, j, device):
        return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)

    # Flash Attention
    def flash_attn(self, q, k, v, mask = None):
        _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda

        if exists(mask) and mask.ndim != 4:
            mask = rearrange(mask, 'b j -> b 1 1 j')
            mask = mask.expand(-1, heads, q_len, -1)

        config = self.cuda_config if is_cuda else self.cpu_config

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0., 
                is_causal = self.causal
            )

        return out

    # 前向传播函数
    def forward(self, q, k, v, mask = None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device = q.shape[-2], q.device

        scale = q.shape[-1] ** -0.5

        if self.flash:
            return self.flash_attn(q, k, v, mask = mask)

        # 相似度计算
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale

        # key padding mask
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # causal mask
        if self.causal:
            causal_mask = self.get_mask(n, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # 注意力计算
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # 聚合值
        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        return out
```