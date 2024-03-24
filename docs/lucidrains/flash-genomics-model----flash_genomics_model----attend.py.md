# `.\lucidrains\flash-genomics-model\flash_genomics_model\attend.py`

```py
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

# 定义辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 保证函数只执行一次的装饰器
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

# 用once装饰的print函数，确保只打印一次
print_once = once(print)

# 主要类

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
        # 掩码是B L，因此必须扩展为B H N L

        if exists(mask) and mask.ndim != 4:
            mask = rearrange(mask, 'b j -> b 1 1 j')
            mask = mask.expand(-1, heads, q_len, -1)

        # 检查是否有兼容的设备用于Flash Attention

        config = self.cuda_config if is_cuda else self.cpu_config

        causal = self.causal

        # 处理注意力偏置

        if exists(attn_bias):
            mask_value = -torch.finfo(q.dtype).max // 2
            causal_mask = self.get_mask(q_len, k_len, device)
            attn_bias = attn_bias.masked_fill(causal_mask, mask_value)

            if exists(mask):
                attn_bias = attn_bias.masked_fill(~mask, mask_value)

            mask = attn_bias
            causal = False

        # 使用torch.backends.cuda.sdp_kernel(**config._asdict())来调用Flash Attention
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0., 
                is_causal = causal
            )

        return out
    # 定义一个前向传播函数，接受查询(q)、键(k)、值(v)、掩码(mask)和注意力偏置(attn_bias)作为参数
    def forward(self, q, k, v, mask = None, attn_bias = None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        # 获取查询(q)和键(k)的序列长度以及设备信息
        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        # 计算缩放因子
        scale = q.shape[-1] ** -0.5

        # 根据键(k)的维度确定 einsum 的等式
        kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

        # 如果启用了 flash 模式，则调用 flash_attn 函数
        if self.flash:
            return self.flash_attn(q, k, v, mask = mask, attn_bias = attn_bias)

        # 计算相似度
        sim = einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

        # 添加注意力偏置
        if exists(attn_bias):
            sim = sim + attn_bias

        # 如果启用了因果关系，生成因果掩码
        if self.causal:
            causal_mask = self.get_mask(q_len, k_len, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # 计算注意力权重
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # 聚合值
        out = einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)

        return out
```