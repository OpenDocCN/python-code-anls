# `.\lucidrains\make-a-video-pytorch\make_a_video_pytorch\attend.py`

```
# 导入必要的库
from functools import wraps
from packaging import version
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

# 定义一个命名元组，用于存储注意力机制的配置信息
AttentionConfig = namedtuple('AttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# 定义一个辅助函数，用于检查变量是否存在
def exists(val):
    return val is not None

# 定义一个装饰器函数，确保被装饰的函数只执行一次
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

# 用装饰器once包装print函数，确保只打印一次
print_once = once(print)

# 主要类定义
class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        flash = False,
        causal = False
    ):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.causal = causal

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # 确定在cuda和cpu上的高效注意力配置

        self.cpu_config = AttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig(False, True, True)

    # 实现flash attention
    def flash_attn(self, q, k, v):
        _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        # 检查是否有兼容的设备用于flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # 使用torch.backends.cuda.sdp_kernel(**config._asdict())来调用pytorch 2.0的flash attention
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p = self.dropout if self.training else 0.,
                is_causal = self.causal
            )

        return out

    # 前向传播函数
    def forward(self, q, k, v, bias = None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        if self.flash:
            assert not exists(bias)
            return self.flash_attn(q, k, v)

        scale = q.shape[-1] ** -0.5

        # 相似度计算
        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # 注意力偏置
        if exists(bias):
            sim = sim + bias

        # 因果关系
        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # 注意力计算
        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # 聚合值
        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out
```