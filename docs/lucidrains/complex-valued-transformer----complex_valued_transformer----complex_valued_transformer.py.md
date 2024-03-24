# `.\lucidrains\complex-valued-transformer\complex_valued_transformer\complex_valued_transformer.py`

```
from typing import Optional
from functools import partial

import torch
from torch import cfloat
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from complex_valued_transformer.attend import Attend

# helpers

# 检查变量是否存在
def exists(v):
    return v is not None

# 如果变量存在则返回该变量，否则返回默认值
def default(v, d):
    return v if exists(v) else d

# helper tensor functions

# 使用旋转因子调制输入张量
def modulate_with_rotation(x, m):
    if m.dtype == cfloat:
        m = m.abs()

    rot = m.cos() + 1.j * m.sin()
    return x * rot

# complex attention
# https://arxiv.org/abs/2306.09827

# 实部复杂注意力机制
def complex_attention_real(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attend: Attend,
    mask: Optional[Tensor] = None
):
    """
    section 4.1 equation 8
    """

    assert all([t.dtype == cfloat for t in (q, k, v)])
    q, k, v = map(torch.view_as_real, (q, k, v))
    q, k, v = map(lambda t: rearrange(t, '... d c -> ... (d c)'), (q, k, v))

    o = attend(q, k, v, mask = mask)

    o = rearrange(o, '... (d c) -> ... d c', c = 2)
    return torch.view_as_complex(o)

# complex attention - Yang et al
# https://arxiv.org/abs/1910.10202

# 完整复杂注意力机制
def complex_attention_complete(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attend: Attend,
    mask: Optional[Tensor] = None
):
    """
    section 3.2 equation 3
    """
    batch, device = q.shape[0], q.device

    assert all([t.dtype == cfloat for t in (q, k, v)])
    q, k, v = map(torch.view_as_real, (q, k, v))

    # complex attention =    (MH(A, A, A) − MH(A, B, B) − MH(B, A, B) − MH(B, B, A))
    #                     + i(MH(A, A, B) + MH(A, B, A) + MH(B, A, A) − MH(B, B, B))

    q = repeat(q, 'b h n d c -> (c r b) h n d', r = 2)
    k = repeat(k, 'b h n d c -> (r c b) h n d', r = 2)
    v = repeat(v, 'b h n d c -> (r b) h n (d c)', r = 4)

    if exists(mask):
        mask = repeat(mask, 'b ... -> (r b) ...', r = 4)

    o = attend(q, k, v, mask = mask)

    o = rearrange(o, '(r b) ... (d c) -> (r c) b ... d', r = 4, c = 2)

    indices = torch.tensor([0, 3, 5, 6, 1, 2, 4, 7], dtype = torch.long, device = device)

    o = rearrange(o[indices], '(r c) ... -> ... c r', c = 2)

    sign = torch.tensor([
        [1., -1., -1., -1.],   # real component
        [1.,  1.,  1., -1.]    # imag component
    ], dtype = o.dtype, device = device)

    o = (o * sign).sum(dim = -1)

    return torch.view_as_complex(o)

# complex multihead attention

# 复杂多头注意力机制
class ComplexMultiheadAttention(Module):
    def __init__(
        self,
        dim,
        *,
        causal = False,
        dim_head = 32,
        heads = 8,
        complete_complex = False, # whether to use complete complex formulation (Yang et al.) or just the real component, which reduces down to usual dot product on real and imaginary components flattened into the feature dimension
        flash = False
    ):
        super().__init__()
        dim_inner = heads * dim_head

        self.to_q = nn.Linear(dim, dim_inner, bias = False, dtype = cfloat)
        self.to_kv = nn.Linear(dim, dim_inner * 2, bias = False, dtype = cfloat)
        self.to_out = nn.Linear(dim_inner, dim, bias = False, dtype = cfloat)

        maybe_flash_attn = Attend(
            causal = causal,
            heads = heads,
            flash = flash
        )

        complex_attention = complex_attention_complete if complete_complex else complex_attention_real
        self.attend = partial(complex_attention, attend = maybe_flash_attn)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

    def forward(
        self,
        x,
        context = None,
        mask = None,
        rotary_emb = None
        ):
        # 检查是否存在上下文变量
        has_context = exists(context)
        # 如果上下文变量不存在，则使用默认值 x
        context = default(context, x)

        # 将输入 x 转换为查询 q，键 k，值 v
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        # 将查询 q，键 k，值 v 分别拆分为多个头部
        q, k, v = map(self.split_heads, (q, k, v))

        # 如果存在旋转嵌入变量，则将查询 q 和键 k 乘以旋转嵌入
        if exists(rotary_emb):
            q = q * rotary_emb
            k = k * rotary_emb

        # 使用注意力机制计算输出 o
        o = self.attend(q, k, v, mask = mask)

        # 将多个头部的输出 o 合并
        o = self.merge_heads(o)
        # 返回最终输出
        return self.to_out(o)
# 定义一个名为 ComplexRMSNorm 的类，继承自 Module 类
class ComplexRMSNorm(Module):
    # 初始化方法，接受一个参数 dim
    def __init__(self, dim):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化 scale 属性为 dim 的平方根的倒数
        self.scale = dim ** -0.5
        # 初始化 gamma 属性为一个可学习参数，维度为 dim，数据类型为复数
        self.gamma = nn.Parameter(torch.ones(dim, dtype=cfloat))

    # 前向传播方法，接受输入 x
    def forward(self, x):
        # 对输入 x 进行维度为 -1 的标准化，然后乘以 gamma 和 scale
        return F.normalize(x, dim=-1) * self.gamma * self.scale

# 定义一个名为 ModReLU 的类，继承自 Module 类
class ModReLU(Module):
    # 初始化方法，接受一个参数 relu_squared，默认为 False
    def __init__(self, relu_squared=False):
        # 调用父类的初始化方法
        super().__init__()
        # 根据 relu_squared 的值确定 pow 的值为 2 或 1
        self.pow = 2 if relu_squared else 1
        # 初始化 bias 属性为一个可学习参数，值为 0
        self.bias = nn.Parameter(torch.tensor(0.))

    # 前向传播方法，接受输入 x
    def forward(self, x):
        # 计算实部，使用 ReLU 函数对绝对值加上 bias，然后取 pow 次方
        real = F.relu(torch.abs(x) + self.bias) ** self.pow
        # 计算虚部，使用指数函数计算角度
        imag = torch.exp(1.j * torch.angle(x))
        # 返回实部和虚部相加的结果
        return real + imag

# 定义一个名为 ComplexFeedForward 的函数，接受参数 dim、mult 和 relu_squared，默认为 4 和 False
def ComplexFeedForward(dim, mult=4, relu_squared=False):
    # 计算内部维度 dim_inner
    dim_inner = dim * mult
    # 返回一个包含线性层、ModReLU 层和线性层的序列
    return nn.Sequential(
        nn.Linear(dim, dim_inner, dtype=cfloat),
        ModReLU(relu_squared=relu_squared),
        nn.Linear(dim_inner, dim, dtype=cfloat)
    )

# 定义一个名为 RotaryEmbedding 的类，继承自 Module 类
class RotaryEmbedding(Module):
    # 初始化方法，接受参数 dim 和 base，默认为 10000
    def __init__(self, dim, base=10000):
        # 调用父类的初始化方法
        super().__init__()
        # 计算频率的倒数
        inv_freq = 1.0 / (base ** (torch.arange(0, dim).float() / dim))
        # 将频率的倒数作为缓冲区注册为 inv_freq 属性
        self.register_buffer('inv_freq', inv_freq)

    # 定义 device 属性，返回 inv_freq 的设备信息
    @property
    def device(self):
        return self.inv_freq.device

    # 前向传播方法，接受参数 seq_len
    def forward(self, seq_len):
        # 生成序列 t，计算频率，返回余弦和正弦值
        t = torch.arange(seq_len, device=self.device).type_as(self.inv_freq)
        freqs = einsum('i, j -> i j', t, self.inv_freq)
        return torch.cos(freqs) + 1.j * torch.sin(freqs)

# 定义一个名为 ComplexTransformer 的类，继承自 Module 类
class ComplexTransformer(Module):
    # 初始化方法，接受多个参数
    def __init__(
        self,
        dim,
        *,
        depth,
        num_tokens: Optional[int] = None,
        causal=False,
        dim_head=32,
        heads=8,
        ff_mult=4,
        relu_squared=True,
        complete_complex=False,
        rotary_emb=True,
        flash_attn=True
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 判断是否存在 num_tokens
        self.has_embed = exists(num_tokens)

        # 如果存在 num_tokens，则初始化 embed 属性为一个可学习参数
        if exists(num_tokens):
            self.embed = nn.Parameter(torch.randn((num_tokens, dim), dtype=cfloat))

        # 根据 rotary_emb 的值初始化 rotary_emb 属性为 None 或 RotaryEmbedding 对象
        self.rotary_emb = None
        if rotary_emb:
            self.rotary_emb = RotaryEmbedding(dim_head)

        # 初始化 layers 属性为一个模块列表，包含多个复杂层
        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(ModuleList([
                ComplexRMSNorm(dim),
                ComplexMultiheadAttention(dim=dim, dim_head=dim_head, heads=heads, causal=causal, complete_complex=complete_complex, flash=flash_attn),
                ComplexRMSNorm(dim),
                ComplexFeedForward(dim=dim, mult=ff_mult, relu_squared=relu_squared)
            ]))

        # 初始化 norm 属性为 ComplexRMSNorm 对象
        self.norm = ComplexRMSNorm(dim)

        # 初始化 to_logits 属性为一个线性层，用于输出结果
        self.to_logits = nn.Linear(dim, num_tokens, dtype=cfloat)

    # 前向传播方法，接受输入 x、context、mask 和其他参数
    def forward(
        self,
        x,
        context=None,
        mask=None,
        return_abs_logits=False,
        return_real_logits=False
    ):
        # 如果存在 embed 属性，则将 x 替换为 embed[x]
        if self.has_embed:
            x = self.embed[x]

        # 获取序列长度
        seq_len = x.shape[-2]
        rotary_emb = None

        # 如果存在 rotary_emb 属性，则计算 rotary_emb
        if exists(self.rotary_emb):
            rotary_emb = self.rotary_emb(seq_len)

        # 遍历复杂层，进行前向传播
        for attn_norm, attn, ff_norm, ff in self.layers:
            x = attn(attn_norm(x), context=context, mask=mask, rotary_emb=rotary_emb) + x
            x = ff(ff_norm(x)) + x

        # 对结果进行标准化
        x = self.norm(x)

        # 如果不存在 embed 属性，则直接返回结果
        if not self.has_embed:
            return x

        # 计算 logits
        logits = self.to_logits(x)

        # 根据参数选择返回的 logits 类型
        assert (int(return_abs_logits) + int(return_real_logits)) <= 1
        if return_abs_logits:
            logits = logits.abs()
        elif return_real_logits:
            logits = logits.real

        return logits
```