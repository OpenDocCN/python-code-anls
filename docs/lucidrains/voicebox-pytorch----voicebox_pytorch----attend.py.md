# `.\lucidrains\voicebox-pytorch\voicebox_pytorch\attend.py`

```
# 从 functools 模块导入 wraps 函数
# 从 packaging 模块导入 version 类
# 从 collections 模块导入 namedtuple 类
# 导入 torch 库
# 从 torch 模块中导入 nn, einsum 函数
# 从 torch.nn 模块中导入 functional 模块
# 从 einops 模块中导入 rearrange, reduce 函数
from functools import wraps
from packaging import version
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce

# 定义一个命名元组 FlashAttentionConfig，包含三个布尔类型的字段
FlashAttentionConfig = namedtuple('FlashAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# 定义一个辅助函数，判断值是否存在
def exists(val):
    return val is not None

# 定义一个辅助函数，如果值存在则返回该值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 定义一个装饰器函数，确保被装饰的函数只能调用一次
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

# 定义一个打印函数，使用 once 装饰器确保只打印一次
print_once = once(print)

# 主类 Attend
class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        flash = False,
        scale = None
    ):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.scale = scale

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # 确定 CUDA 和 CPU 的高效注意力配置

        self.cpu_config = FlashAttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = FlashAttentionConfig(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = FlashAttentionConfig(False, True, True)

    # Flash Attention 函数
    def flash_attn(self, q, k, v, mask = None):
        _, heads, q_len, dim_head, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        # 如果给定了 scale，将 q 乘以默认的 scale

        if exists(self.scale):
            q = q * (self.scale / (dim_head ** -0.5))

        # 检查 mask 是否存在并扩展到兼容的形状

        if exists(mask):
            mask = mask.expand(-1, heads, q_len, -1)

        # 检查是否有兼容的设备用于 Flash Attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # 使用 torch.backends.cuda.sdp_kernel 函数应用 Flash Attention

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0.
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

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        scale = default(self.scale, q.shape[-1] ** -0.5)

        if exists(mask) and mask.ndim != 4:
            mask = rearrange(mask, 'b j -> b 1 1 j')

        if self.flash:
            return self.flash_attn(q, k, v, mask = mask)

        # 相似度计算

        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # key padding mask

        if exists(mask):
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # 注意力计算

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # 聚合值

        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out
```