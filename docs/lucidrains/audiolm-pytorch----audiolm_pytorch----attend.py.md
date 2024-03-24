# `.\lucidrains\audiolm-pytorch\audiolm_pytorch\attend.py`

```py
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块和 einsum 函数
from torch import nn, einsum
# 从 torch 库中导入 nn.functional 模块，并重命名为 F
import torch.nn.functional as F

# 导入 namedtuple、wraps 函数
from collections import namedtuple
from functools import wraps
# 从 packaging 库中导入 version 模块
from packaging import version

# 从 einops 库中导入 rearrange 和 repeat 函数
from einops import rearrange, repeat

# 定义 Config 命名元组，包含三个布尔类型的参数
Config = namedtuple('Config', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# 定义 exists 函数，用于判断变量是否存在
def exists(val):
    return val is not None

# 定义 once 装饰器函数，确保函数只被调用一次
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

# 使用 once 装饰器包装 print 函数，确保只打印一次
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
        # 注册缓冲区 mask，初始值为 None
        self.register_buffer("mask", None, persistent=False)

        self.flash = flash
        # 断言条件，如果 flash 为 True 且 torch 版本小于 2.0.0，则抛出异常
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # 确定 cuda 和 cpu 的高效注意力配置

        self.cpu_config = Config(True, True, True)
        self.cuda_config = None

        # 如果没有可用的 CUDA 或不使用 flash，则直接返回
        if not torch.cuda.is_available() or not flash:
            return

        # 获取当前 CUDA 设备的属性
        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        # 如果 CUDA 设备为 A100，则打印信息并设置 cuda_config
        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = Config(True, False, False)
        else:
            # 如果 CUDA 设备不是 A100，则打印信息并设置 cuda_config
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = Config(False, True, True)

    # flash_attn 函数，实现闪存注意力机制
    def flash_attn(self, q, k, v, mask = None):
        _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda

        # 将 k 和 v 重复 heads 次
        k = repeat(k, 'b ... -> b h ...', h = heads)
        v = repeat(v, 'b ... -> b h ...', h = heads)

        causal = self.causal

        # 如果 mask 存在，则根据 mask 设置 mask 和 causal
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            mask = mask.expand(-1, heads, q_len, -1)

            if causal:
                causal_mask = torch.ones((q_len, k_len), device = q.device, dtype = torch.bool).triu(k_len - q_len + 1)
                mask = mask & ~causal_mask                
                causal = False

        # 根据是否在 CUDA 上运行选择配置，使用 torch.backends.cuda.sdp_kernel 函数
        config = self.cuda_config if is_cuda else self.cpu_config

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0., 
                is_causal = causal
            )

        return out

    # 前向传播函数
    def forward(self, q, k, v, mask = None, attn_bias = None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device = q.shape[-2], q.device

        scale = q.shape[-1] ** -0.5

        # 如果使用 flash 注意力机制，则调用 flash_attn 函数
        if self.flash:
            assert not exists(attn_bias), 'attention bias not supported for flash attention'
            return self.flash_attn(q, k, v, mask = mask)

        # 相似度计算
        sim = einsum("b h i d, b j d -> b h i j", q, k) * scale

        # 如果存在 attn_bias，则加到相似度上
        if exists(attn_bias):
            sim = sim + attn_bias

        # 如果存在 mask，则根据 mask 设置 sim
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # 如果是因果关系，则设置因果 mask
        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), device = sim.device, dtype = torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # 注意力计算
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # 聚合值
        out = einsum("b h i j, b j d -> b h i d", attn, v)

        return out
```