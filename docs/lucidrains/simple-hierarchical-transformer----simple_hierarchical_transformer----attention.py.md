# `.\lucidrains\simple-hierarchical-transformer\simple_hierarchical_transformer\attention.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn, einsum 模块
from torch import nn, einsum
# 从 torch.nn 模块中导入 Module 类
from torch.nn import Module
# 从 torch.nn.functional 模块中导入 F 别名
import torch.nn.functional as F

# 导入 namedtuple 类
from collections import namedtuple
# 导入 wraps 函数
from functools import wraps
# 从 packaging 模块中导入 version 类
from packaging import version
# 从 einops 库中导入 rearrange 函数

from einops import rearrange

# 定义常量 Config，使用 namedtuple 创建一个命名元组
Config = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# 定义辅助函数

# 判断值是否存在的函数
def exists(val):
    return val is not None

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

# 主类

# 定义 Attend 类，继承自 Module 类
class Attend(Module):
    # 初始化函数
    def __init__(
        self,
        causal = False,
        use_flash_attn = False
    ):
        super().__init__()
        # 是否是因果关系
        self.causal = causal
        # 注册缓冲区 mask，初始值为 None
        self.register_buffer("mask", None, persistent=False)

        # 是否使用 flash attention
        self.use_flash_attn = use_flash_attn
        # 断言语句，如果使用 flash attention 且 torch 版本小于 2.0，则抛出异常
        assert not (use_flash_attn and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # 确定 cuda 和 cpu 的高效注意力配置

        self.cpu_config = Config(True, True, True)
        self.cuda_config = None

        # 如果没有可用的 cuda 或不使用 flash attention，则直接返回
        if not torch.cuda.is_available() or not use_flash_attn:
            return

        # 获取当前 cuda 设备的属性
        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        # 如果是 A100 GPU，则打印信息并设置 cuda_config
        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = Config(True, False, False)
        else:
            # 如果不是 A100 GPU，则打印信息并设置 cuda_config
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = Config(False, True, True)

    # 获取 mask 函数
    def get_mask(self, n, device):
        # 如果 mask 存在且形状大于等于 n，则返回 mask
        if exists(self.mask) and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        # 创建 mask，上三角矩阵
        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        # 注册缓冲区 mask
        self.register_buffer("mask", mask, persistent=False)
        return mask

    # flash attention 函数
    def flash_attn(self, q, k, v, mask = None):
        _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda

        # 检查 mask 是否存在并扩展到兼容的形状
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            mask = mask.expand(-1, heads, q_len, -1)

        # 检查是否有兼容的设备用于 flash attention
        config = self.cuda_config if is_cuda else self.cpu_config

        # 使用 torch.backends.cuda.sdp_kernel 运行 pytorch 2.0 flash attention
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                is_causal = self.causal
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

        n, device = q.shape[-2], q.device

        scale = q.shape[-1] ** -0.5

        if self.use_flash_attn:
            return self.flash_attn(q, k, v, mask = mask)

        # 相似度
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale

        # key padding mask
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # 因果 mask
        if self.causal:
            causal_mask = self.get_mask(n, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # 注意力
        attn = sim.softmax(dim=-1)

        # 聚合值
        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        return out
```