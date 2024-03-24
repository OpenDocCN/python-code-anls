# `.\lucidrains\iTransformer\iTransformer\attend.py`

```
# 导入所需的库
from functools import partial

import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F

from collections import namedtuple
from functools import wraps
from packaging import version

from einops import rearrange, repeat

# 定义一个命名元组EfficientAttentionConfig，包含三个布尔类型的参数
EfficientAttentionConfig = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# 辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 如果变量存在则返回该变量，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 保证函数只被调用一次
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

# 打印函数，只打印一次
print_once = once(print)

# 主类

class Attend(nn.Module):
    def __init__(
        self,
        *,
        dropout = 0.,
        heads = None,
        scale = None,
        flash = False,
        causal = False
    ):
        super().__init__()
        self.scale = scale

        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.causal = causal

        # flash attention

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = EfficientAttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        major, minor = device_properties.major, device_properties.minor

        if (major, minor) == (8, 0):
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = EfficientAttentionConfig(True, False, False)
        elif (major, minor) == (9, 0):
            print_once('H100 GPU detected, using flash attention')
            self.cuda_config = EfficientAttentionConfig(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = EfficientAttentionConfig(False, True, True)

    # 实现flash attention
    def flash_attn(
        self,
        q, k, v
    ):
        batch, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        # 检查是否有兼容的设备用于flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # 使用torch.backends.cuda.sdp_kernel(**config._asdict())来调用pytorch 2.0的flash attention
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal = self.causal,
                dropout_p = self.dropout if self.training else 0.
            )
        
        return out

    # 前向传播函数
    def forward(
        self,
        q, k, v
    ):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, heads, kv_heads, device, dtype = q.shape[-2], q.shape[1], k.shape[1], q.device, q.dtype

        scale = default(self.scale, q.shape[-1] ** -0.5)

        if self.flash:
            return self.flash_attn(q, k, v)

        sim = einsum(f'b h i d, b h j d -> b h i j', q, k) * scale

        if self.causal:
            i, j, dtype = *sim.shape[-2:], sim.dtype
            mask_value = -torch.finfo(sim.dtype).max
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim = -1)
        attn = attn.type(dtype)

        attn = self.attn_dropout(attn)

        out = einsum(f'b h i j, b h j d -> b h i d', attn, v)

        return out
```