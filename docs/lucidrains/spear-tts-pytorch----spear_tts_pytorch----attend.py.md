# `.\lucidrains\spear-tts-pytorch\spear_tts_pytorch\attend.py`

```py
import torch
from torch import nn, einsum
import torch.nn.functional as F

from collections import namedtuple
from functools import wraps
from packaging import version

from einops import rearrange, repeat

# 定义一个命名元组 Config，包含三个布尔类型的参数
Config = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# 定义一个辅助函数，用于检查变量是否存在
def exists(val):
    return val is not None

# 定义一个装饰器函数，用于确保被装饰的函数只执行一次
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

# 用装饰器 once 包装 print 函数，确保只打印一次
print_once = once(print)

# 主要类 Attend
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

        # 确定用于 cuda 和 cpu 的高效注意力配置

        self.cpu_config = Config(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = Config(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = Config(False, True, True)

    # 获取掩码
    def get_mask(self, i, j, device):
        n = max(i, j)

        if exists(self.mask) and self.mask.shape[-1] >= n:
            mask = self.mask[:n, :n]
        else:
            mask = torch.ones((n, n), device = device, dtype = torch.bool).triu(1)
            self.register_buffer("mask", mask, persistent = False)

        return mask[-i:, :]

    # Flash Attention 函数
    def flash_attn(self, q, k, v, mask = None):
        _, heads, q_len, _, k_len, causal, is_cuda, device = *q.shape, k.shape[-2], self.causal, q.is_cuda, q.device

        # 检查掩码是否存在并扩展到兼容的形状
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            mask = mask.expand(-1, heads, q_len, -1)

        # 检查是否有兼容的设备用于 Flash Attention
        config = self.cuda_config if is_cuda else self.cpu_config

        # 如果 q 和 k 的长度不同（缓存键/值），并且是因果的，手动构造因果注意力掩码作为浮点数，因为不支持（Flash Attention 2 最终会支持这一点）
        row_is_entirely_masked = None
        if causal and q_len != k_len:
            causal_mask = self.get_mask(q_len, k_len, device = device)

            if exists(mask):
                mask = mask & ~causal_mask
            else:
                mask = ~causal_mask

            row_is_entirely_masked = ~mask.any(dim = -1)
            mask[..., 0] = mask[..., 0] | row_is_entirely_masked

            causal = False

        # 使用 torch.backends.cuda.sdp_kernel 函数应用 PyTorch 2.0 Flash Attention
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0., 
                is_causal = causal
            )

        if exists(row_is_entirely_masked):
            out = out.masked_fill(row_is_entirely_masked[..., None], 0.)

        return out
    # 定义一个前向传播函数，接受查询(q)、键(k)、值(v)和掩码(mask)作为输入参数
    """
    einstein notation
    b - batch
    h - heads
    n, i, j - sequence length (base sequence length, source, target)
    d - feature dimension
    """

    # 获取查询(q)的序列长度和设备信息
    n, device = q.shape[-2], q.device
    # 获取头数和键值对应的头数
    heads, kv_heads = q.shape[1], k.shape[1]

    # 如果键值对应的头数小于总头数，则对键(k)和值(v)进行重复以匹配总头数
    if kv_heads < heads:
        k, v = map(lambda t: repeat(t, 'b h ... -> b (g h) ...', g = heads // kv_heads), (k, v))

    # 缩放因子
    scale = q.shape[-1] ** -0.5

    # 如果启用了flash注意力机制，则调用flash_attn函数
    if self.flash:
        return self.flash_attn(q, k, v, mask = mask)

    # 相似度计算

    sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale

    # 键填充掩码

    # 如果存在掩码，则重新排列掩码并用极小值替换相似度矩阵中的无效位置
    if exists(mask):
        mask = rearrange(mask, 'b j -> b 1 1 j')
        sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

    # 因果掩码

    # 如果启用了因果掩码，则生成因果掩码并用极小值替换相似度矩阵中的无效位置
    if self.causal:
        i, j = sim.shape[-2:]
        causal_mask = self.get_mask(i, j, device)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

    # 注意力权重计算

    # 对相似度矩阵进行softmax操作，得到注意力权重
    attn = sim.softmax(dim = -1)
    # 对注意力权重进行dropout操作
    attn = self.attn_dropout(attn)

    # 聚合值

    # 根据注意力权重对值(v)进行加权求和，得到输出结果
    out = einsum("b h i j, b h j d -> b h i d", attn, v)

    return out
```