# `.\lucidrains\BS-RoFormer\bs_roformer\attend.py`

```py
# 导入必要的库
from functools import wraps
from packaging import version
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce

# 定义一个命名元组 FlashAttentionConfig，包含三个布尔类型的参数
FlashAttentionConfig = namedtuple('FlashAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# 定义一些辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 如果变量存在则返回该变量，否则返回默认值
def default(v, d):
    return v if exists(v) else d

# 保证函数只被调用一次的装饰器
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

# 打印信息，只打印一次
print_once = once(print)

# 主要类

class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        flash = False,
        scale = None
    ):
        super().__init__()
        self.scale = scale
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # 确定在 cuda 和 cpu 上的高效注意力配置

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

    # Flash Attention 方法
    def flash_attn(self, q, k, v):
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        if exists(self.scale):
            default_scale = q.shape[-1] ** -0.5
            q = q * (self.scale / default_scale)

        # 检查是否有兼容的设备用于 Flash Attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # 使用 torch.backends.cuda.sdp_kernel() 方法进行 Flash Attention 计算
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p = self.dropout if self.training else 0.
            )

        return out

    # 前向传播方法
    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        scale = default(self.scale, q.shape[-1] ** -0.5)

        if self.flash:
            return self.flash_attn(q, k, v)

        # 相似度计算

        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # 注意力计算

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # 聚合数值

        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out
```