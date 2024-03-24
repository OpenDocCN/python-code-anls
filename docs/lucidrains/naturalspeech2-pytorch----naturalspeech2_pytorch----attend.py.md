# `.\lucidrains\naturalspeech2-pytorch\naturalspeech2_pytorch\attend.py`

```
# 导入必要的库
from collections import namedtuple
from functools import wraps
from packaging import version

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

# 定义一个命名元组 Config，用于存储 EfficientAttention 的配置信息
Config = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

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

# 用 once 装饰 print 函数，确保只打印一次
print_once = once(print)

# 主要的 Attend 类
class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        causal = False,
        use_flash = False
    ):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.causal = causal
        self.register_buffer("mask", None, persistent=False)

        self.use_flash = use_flash
        assert not (use_flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # 确定在 cuda 和 cpu 上的高效注意力配置

        self.cpu_config = Config(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not use_flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = Config(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = Config(False, True, True)

    # 获取掩码
    def get_mask(self, n, device):
        if exists(self.mask) and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    # Flash Attention 函数
    def flash_attn(self, q, k, v, mask = None):
        _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda

        # 推荐的多查询单键值注意力结构
        if k.ndim == 3:
            k = rearrange(k, 'b ... -> b 1 ...').expand_as(q)

        if v.ndim == 3:
            v = rearrange(v, 'b ... -> b 1 ...').expand_as(q)

        # 检查掩码是否存在并扩展到兼容的形状
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            mask = mask.expand(-1, heads, q_len, -1)

        # 检查是否有兼容的设备用于 Flash Attention
        config = self.cuda_config if is_cuda else self.cpu_config

        # 使用 pytorch 2.0 的 Flash Attention
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0., 
                is_causal = self.causal
            )

        return out
    # 定义一个前向传播函数，实现注意力机制
    def forward(self, q, k, v, mask = None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        # 获取序列长度和设备信息
        n, device = q.shape[-2], q.device

        # 计算缩放因子
        scale = q.shape[-1] ** -0.5

        # 如果使用闪回注意力机制，则调用相应函数
        if self.use_flash:
            return self.flash_attn(q, k, v, mask = mask)

        # 根据输入维度确定键值对的 einsum 方程
        kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

        # 计算相似度
        sim = einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

        # 处理键的填充掩码
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # 处理因果掩码
        if self.causal:
            causal_mask = self.get_mask(n, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # 计算注意力权重
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # 聚合数值
        out = einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)

        return out
```