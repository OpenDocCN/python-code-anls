# `.\lucidrains\classifier-free-guidance-pytorch\classifier_free_guidance_pytorch\attend.py`

```py
# 导入必要的库
from collections import namedtuple
from functools import wraps
from packaging import version

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

# 定义一个命名元组EfficientAttentionConfig，包含三个布尔类型的参数
EfficientAttentionConfig = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# 定义辅助函数exists，用于检查值是否存在
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
        self.register_buffer("mask", None, persistent=False)

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

    # 获取掩码
    def get_mask(self, n, device):
        if exists(self.mask) and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    # Flash Attention函数
    def flash_attn(self, q, k, v, mask = None):
        _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda

        # 推荐使用多查询单键值注意力的Tri Dao
        if k.ndim == 3:
            k = repeat(k, 'b ... -> b h ...', h = heads)

        if v.ndim == 3:
            v = repeat(v, 'b ... -> b h ...', h = heads)

        # 检查掩码是否存在并扩展到兼容的形状
        if exists(mask):
            if mask.ndim == 2:
                mask = rearrange(mask, 'b j -> b 1 1 j')

            mask = mask.expand(-1, heads, q_len, -1)

        # 检查是否有兼容的设备用于Flash Attention
        config = self.cuda_config if is_cuda else self.cpu_config

        # 使用torch.backends.cuda.sdp_kernel(**config._asdict())来调用pytorch 2.0的flash attention
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0., 
                is_causal = self.causal
            )

        return out
    # 定义一个前向传播函数，接受查询(q), 键(k), 值(v)和掩码(mask)作为输入参数
    def forward(self, q, k, v, mask = None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        # 获取序列长度n和设备信息
        n, device = q.shape[-2], q.device

        # 缩放因子，根据特征维度的倒数开根号
        scale = q.shape[-1] ** -0.5

        # 如果启用了flash注意力机制，则调用flash_attn函数
        if self.flash:
            return self.flash_attn(q, k, v, mask = mask)

        # 根据键的维度确定键值对的einsum等式
        kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

        # 计算相似度
        sim = einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

        # 键的填充掩码
        if exists(mask):
            if mask.ndim == 2:
                mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # 因果掩码
        if self.causal:
            causal_mask = self.get_mask(n, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # 注意力权重
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # 聚合值
        out = einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)

        return out
```