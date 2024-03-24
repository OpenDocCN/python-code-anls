# `.\lucidrains\PaLM-rlhf-pytorch\palm_rlhf_pytorch\attention.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块和 einsum 函数
from torch import nn, einsum
# 从 torch 库中导入 nn.functional 模块，并重命名为 F
import torch.nn.functional as F

# 从 collections 模块中导入 namedtuple 类
from collections import namedtuple
# 从 functools 模块中导入 wraps 函数
from functools import wraps
# 从 packaging 模块中导入 version 类
from packaging import version

# 从 einops 库中导入 rearrange 函数
from einops import rearrange

# 定义一个命名元组 Config，包含三个布尔类型的参数
Config = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# 定义一个辅助函数，用于检查值是否存在
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

# 定义一个打印函数，使用 once 装饰器确保只打印一次
print_once = once(print)

# 主要类定义

class Attention(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        causal = False,
        use_flash_attn = False
    ):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.causal = causal
        # 注册一个缓冲区变量 mask，初始值为 None，不会被持久化
        self.register_buffer("mask", None, persistent=False)

        self.use_flash_attn = use_flash_attn
        # 断言条件，如果不满足则抛出异常
        assert not (use_flash_attn and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # 确定 CUDA 和 CPU 的高效注意力配置

        self.cpu_config = Config(True, True, True)
        self.cuda_config = None

        # 如果没有可用的 CUDA 或不使用 flash attention，则直接返回
        if not torch.cuda.is_available() or not use_flash_attn:
            return

        # 获取当前 CUDA 设备的属性
        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        # 根据 CUDA 设备的主要和次要版本号选择配置
        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = Config(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = Config(False, True, True)

    # 获取掩码 mask
    def get_mask(self, n, device):
        if exists(self.mask) and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    # Flash Attention 函数
    def flash_attn(self, q, k, v, mask = None):
        _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda

        # 推荐的多查询单键值注意力重排操作
        k = rearrange(k, 'b ... -> b 1 ...').expand_as(q)
        v = rearrange(v, 'b ... -> b 1 ...').expand_as(q)

        # 检查是否存在 mask 并扩展到兼容的形状
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            mask = mask.expand(-1, heads, q_len, -1)

        # 检查是否有兼容的设备用于 flash attention
        config = self.cuda_config if is_cuda else self.cpu_config

        # 使用 torch.backends.cuda.sdp_kernel 函数应用配置，执行 Flash Attention
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0., 
                is_causal = self.causal
            )

        return out
    # 定义一个前向传播函数，接受查询(q)、键(k)、值(v)和掩码(mask)作为输入参数
    def forward(self, q, k, v, mask = None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        # 获取序列长度(n)和设备信息(device)
        n, device = q.shape[-2], q.device

        # 计算缩放因子
        scale = q.shape[-1] ** -0.5

        # 如果使用闪回注意力机制，则调用flash_attn函数
        if self.use_flash_attn:
            return self.flash_attn(q, k, v, mask = mask)

        # 计算相似度
        sim = einsum("b h i d, b j d -> b h i j", q, k) * scale

        # 键填充掩码
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # 因果掩码
        if self.causal:
            causal_mask = self.get_mask(n, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # 注意力计算
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # 聚合值
        out = einsum("b h i j, b j d -> b h i d", attn, v)

        return out
```