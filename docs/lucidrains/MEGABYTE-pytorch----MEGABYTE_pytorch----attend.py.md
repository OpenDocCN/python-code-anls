# `.\lucidrains\MEGABYTE-pytorch\MEGABYTE_pytorch\attend.py`

```
# 导入必要的库
from collections import namedtuple
from functools import wraps
from packaging import version

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

# 定义一个命名元组EfficientAttentionConfig，包含三个布尔类型的参数
EfficientAttentionConfig = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# 定义一个辅助函数，用于检查值是否存在
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

# 用once装饰print函数，确保只打印一次
print_once = once(print)

# 主要类Attend
class Attend(nn.Module):
    def __init__(
        self,
        causal = False,
        dropout = 0.,
        flash = False
    ):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.causal = causal
        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # 确定用于cuda和cpu的高效注意力配置

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

    # 生成掩码
    def get_mask(self, i, j, device):
        return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)

    # Flash Attention函数
    def flash_attn(self, q, k, v, mask = None, attn_bias = None):
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        # 单头键/值
        if k.ndim == 3:
            k = rearrange(k, 'b n d -> b 1 n d')

        if v.ndim == 3:
            v = rearrange(v, 'b n d -> b 1 n d')

        # 检查掩码是否存在并扩展到兼容的形状
        if exists(mask) and mask.ndim != 4:
            mask = rearrange(mask, 'b j -> b 1 1 j')
            mask = mask.expand(-1, heads, q_len, -1)

        # 检查是否有兼容的设备用于Flash Attention
        config = self.cuda_config if is_cuda else self.cpu_config

        # 使用torch.backends.cuda.sdp_kernel(**config._asdict())来执行pytorch 2.0的flash attention
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0., 
                is_causal = self.causal
            )

        return out
    # 定义一个前向传播函数，用于计算注意力机制中的查询、键、值以及掩码
    def forward(self, q, k, v, mask = None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        # 获取查询和键的序列长度，以及设备信息
        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        # 计算缩放因子
        scale = q.shape[-1] ** -0.5

        # 根据键的维度确定 einsum 的等式
        kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

        # 如果启用了 flash 注意力机制，则调用相应函数
        if self.flash:
            return self.flash_attn(q, k, v, mask = mask)

        # 计算相似度
        sim = einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

        # 如果启用了因果掩码
        if self.causal:
            # 获取因果掩码
            causal_mask = self.get_mask(q_len, k_len, device)
            # 将掩码应用到相似度矩阵中
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # 计算注意力权重
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # 聚合值
        out = einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)

        return out
```