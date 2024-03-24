# `.\lucidrains\complex-valued-transformer\complex_valued_transformer\attend.py`

```
from functools import partial  # 导入 functools 模块中的 partial 函数

import torch  # 导入 torch 库
from torch import nn, einsum, Tensor  # 从 torch 库中导入 nn、einsum、Tensor
import torch.nn.functional as F  # 从 torch 库中导入 F

from collections import namedtuple  # 导入 collections 模块中的 namedtuple
from functools import wraps  # 导入 functools 模块中的 wraps
from packaging import version  # 导入 packaging 模块中的 version

from einops import rearrange, repeat  # 从 einops 库中导入 rearrange、repeat

# 定义一个命名元组 EfficientAttentionConfig，包含三个属性
EfficientAttentionConfig = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# 定义辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 如果变量存在则返回该变量，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 仅执行一次的装饰器函数
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

# 仅打印一次的函数
print_once = once(print)

# tensor 函数

# 创建一个因果掩码
def create_causal_mask(i, j, device):
    return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)

# 主类

class Attend(nn.Module):
    def __init__(
        self,
        *,
        dropout=0.,
        causal=False,
        heads=None,
        scale=None,
        flash=False,
    ):
        super().__init__()
        self.scale = scale

        self.causal = causal
        self.create_causal_mask = create_causal_mask

        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

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

    def flash_attn(
        self,
        q, k, v,
        mask=None
    ):
        # 解包 q 的形状，获取 batch, heads, q_len, k_len, is_cuda, device
        batch, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        # 检查是否存在 mask 并扩展到兼容的形状
        # mask 是 B L，因此需要扩展为 B H N L

        causal = self.causal

        # 在 kv 缓存中只有一个令牌的情况下（q_len == 1），只需关闭因果掩码
        # 在推测解码中，这可能会增加到 5-6，因此在那里需要右对齐的因果掩码

        if q_len == 1 and causal:
            causal = False

        # 扩展键填充掩码

        if exists(mask):
            assert mask.ndim == 4
            mask = mask.expand(batch, heads, q_len, k_len)

        # 处理 kv 缓存 - 这应该在更新的 flash attention 2 中可以绕过

        if k_len > q_len and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device=device)
            if not exists(mask):
                mask = ~causal_mask
            else:
                mask = mask & ~causal_mask
            causal = False

        # 手动处理因果掩码，如果给定了另一个掩码

        row_is_entirely_masked = None

        if exists(mask) and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device=device)
            mask = mask & ~causal_mask

            # 防止整行被掩盖

            row_is_entirely_masked = ~mask.any(dim=-1)
            mask[..., 0] = mask[..., 0] | row_is_entirely_masked

            causal = False

        # 检查是否有兼容的设备用于 flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale
        
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0., 
                is_causal=causal
            )

        # 对于整行被完全掩盖的情况，应将该行令牌的输出置零

        if exists(row_is_entirely_masked):
            out = out.masked_fill(row_is_entirely_masked[..., None], 0.)

        return out

    def forward(
        self,
        q, k, v,
        mask=None
    ):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, heads, kv_heads, device = q.shape[-2], q.shape[1], k.shape[1], q.device

        scale = default(self.scale, q.shape[-1] ** -0.5)

        if self.flash:
            return self.flash_attn(q, k, v, mask=mask)

        kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

        sim = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k) * scale

        i, j, dtype = *sim.shape[-2:], sim.dtype

        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            sim = sim.masked_fill(~mask, mask_value)

        if self.causal and n > 1:
            causal_mask = self.create_causal_mask(i, j, device=device)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim=-1)
        attn = attn.type(dtype)

        attn = self.attn_dropout(attn)

        out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)

        return out
```