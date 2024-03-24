# `.\lucidrains\muse-maskgit-pytorch\muse_maskgit_pytorch\attend.py`

```py
# 导入所需的模块和函数
from functools import wraps
from packaging import version
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F

# 导入自定义的 FlashAttentionFunction 函数
from memory_efficient_attention_pytorch.flash_attention import FlashAttentionFunction

# 定义一个命名元组 AttentionConfig，包含三个布尔类型的字段
AttentionConfig = namedtuple('AttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# 定义一个辅助函数，用于检查值是否存在
def exists(val):
    return val is not None

# 定义一个装饰器函数，确保被装饰的函数只能调用一次
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

# 定义一个只能打印一次的函数
print_once = once(print)

# 主要类定义
class Attend(nn.Module):
    def __init__(
        self,
        scale = 8,
        dropout = 0.,
        flash = False
    ):
        super().__init__()
        self.scale = scale
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        # 检查是否启用了 flash attention，且 PyTorch 版本是否大于等于 2.0
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # 确定 CUDA 和 CPU 的高效注意力配置
        self.cuda_config = None
        self.no_hardware_detected = False

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig(False, True, False)

    # 定义 flash attention 函数
    def flash_attn(self, q, k, v, mask = None):
        default_scale = q.shape[-1] ** -0.5

        is_cuda = q.is_cuda

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        # 重新缩放输入张量以适应默认缩放比例
        rescale = self.scale / default_scale
        q = q * (rescale ** 0.5)
        k = k * (rescale ** 0.5)

        # 如果没有检测到正确的硬件或不在 CUDA 上，则使用简单的实现
        use_naive = not is_cuda or not exists(self.cuda_config)

        if not is_cuda or self.no_hardware_detected:
            return FlashAttentionFunction.apply(q, k, v, mask, False, 512, 512)

        # 尝试使用 PyTorch 2.0 的 flash attention 实现
        try:
            raise Exception()
            with torch.backends.cuda.sdp_kernel(**self.cuda_config._asdict()):
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask = mask,
                    dropout_p = self.dropout if self.training else 0.
                )
        except:
            print_once('no hardware detected, falling back to naive implementation from memory-efficient-attention-pytorch library')
            self.no_hardware_detected = True

            out = FlashAttentionFunction.apply(q, k, v, mask, False, 512, 512)

        return out
    # 定义一个前向传播函数，接受查询(q)、键(k)、值(v)、掩码(mask)和是否强制非闪存(force_non_flash)作为参数
    def forward(self, q, k, v, mask = None, force_non_flash = False):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        # 如果启用了flash且不强制使用非flash，则调用flash_attn函数
        if self.flash and not force_non_flash:
            return self.flash_attn(q, k, v, mask = mask)

        # 计算相似度
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        # 掩码处理
        if exists(mask):
            mask_value = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(~mask, mask_value)

        # 注意力计算
        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # 聚合值
        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        return out
```