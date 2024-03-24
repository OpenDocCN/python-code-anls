# `.\lucidrains\g-mlp-gpt\g_mlp_gpt\g_mlp_gpt.py`

```py
# 从 math 模块中导入 ceil 函数，用于向上取整
# 从 functools 模块中导入 partial 函数，用于创建偏函数
# 从 random 模块中导入 randrange 函数，用于生成指定范围内的随机整数
# 导入 torch 模块
# 从 torch.nn.functional 模块中导入 F 别名
# 从 torch 模块中导入 nn、einsum 函数
from math import ceil
from functools import partial
from random import randrange
import torch
import torch.nn.functional as F
from torch import nn, einsum

# 从 einops 模块中导入 rearrange、repeat 函数
from einops import rearrange, repeat

# 从 g_mlp_gpt.reversible 模块中导入 ReversibleSequence、SequentialSequence 类

# functions

# 定义函数 exists，用于判断值是否存在
def exists(val):
    return val is not None

# 定义函数 cast_tuple，用于将值转换为元组
def cast_tuple(val, num):
    return ((val,) * num) if not isinstance(val, tuple) else val

# 定义函数 pad_to_multiple，用于将张量填充到指定的倍数
def pad_to_multiple(tensor, multiple, dim = -1, value = 0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return tensor
    remainder = ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(tensor, (*pad_offset, 0, remainder), value = value)

# 定义函数 dropout_layers，用于对层进行随机丢弃
def dropout_layers(layers, prob_survival):
    if prob_survival == 1:
        return layers

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) > prob_survival

    # 确保至少有一层保留
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers

# helper classes

# 定义类 Residual，实现残差连接
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

# 定义类 PreNorm，实现预层归一化
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# 定义类 GEGLU，实现门控线性单元
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

# 定义类 FeedForward，实现前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        inner_dim = int(dim * mult * 2 / 3)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim * 2),
            GEGLU(),
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

# 定义类 Attention，实现注意力机制
class Attention(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner):
        super().__init__()
        self.scale = dim_inner ** -0.5
        self.to_qkv = nn.Linear(dim_in, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim_out)

    def forward(self, x):
        device = x.device
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        mask = torch.ones(sim.shape[-2:], device = device).triu(1).bool()
        sim.masked_fill_(mask[None, ...], -torch.finfo(q.dtype).max)

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        return self.to_out(out)

# 定义类 LocalAttention，实现局部注意力机制
class LocalAttention(nn.Module):
    def __init__(self, dim_in, dim_inner, dim_out, window = 128):
        super().__init__()
        self.scale = dim_inner ** -0.5
        self.window = window

        self.to_qkv = nn.Linear(dim_in, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim_out)
    # 定义前向传播函数，接受输入 x
    def forward(self, x):
        # 获取输入 x 的形状信息，包括 batch size、序列长度、设备信息和窗口大小
        b, n, *_, device, w = *x.shape, x.device, self.window

        # 将输入 x 进行填充，使其长度能够被窗口大小整除
        x = pad_to_multiple(x, w, dim = -2, value = 0.)
        # 将填充后的 x 分别转换为查询、键、值，并按照最后一个维度分割成三部分
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # 定义窗口函数，将输入按照窗口大小重新排列
        window_fn = lambda t: rearrange(t, 'b (w n) d -> b w n d', n = w)
        q, k, v = map(window_fn, (q, k, v))

        # 对键和值进行填充，使其能够进行滑动窗口操作
        k, v = map(lambda t: F.pad(t, (0, 0, 0, 0, 1, 0)), (k, v))
        k, v = map(lambda t: torch.cat((k[:, :-1], k[:, 1:]), dim = 2), (k, v))

        # 计算查询和键之间的相似度，并乘以缩放因子
        sim = einsum('b w i d, b w j d -> b w i j', q, k) * self.scale
        buckets, i, j = sim.shape[-3:]

        # 创建掩码，用于屏蔽无效的位置信息
        mask_value = -torch.finfo(sim.dtype).max
        mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()
        mask = repeat(mask, 'i j -> () u i j', u = buckets)

        # 将掩码应用到相似度矩阵中
        sim.masked_fill_(mask, mask_value)

        # 对相似度矩阵进行 softmax 操作，得到注意力权重
        attn = sim.softmax(dim = -1)

        # 根据注意力权重计算输出
        out = einsum('b w i j, b w j d -> b w i d', attn, v)
        # 将输出重新排列成原始形状
        out = rearrange(out, 'b w n d -> b (w n) d')
        # 将输出传递给输出层，并返回结果
        out = self.to_out(out[:, :n])
        return out
# 定义一个类 CausalSGU，继承自 nn.Module
class CausalSGU(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        dim,
        dim_seq,
        init_eps = 1e-3,
        heads = 4,
        act = nn.Identity()
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 计算输出维度
        dim_out = dim // 2

        # 初始化 LayerNorm 模块
        self.norm = nn.LayerNorm(dim_out)

        # 设置头数和权重、偏置参数
        self.heads = heads
        self.weight = nn.Parameter(torch.zeros(heads, dim_seq, dim_seq))
        self.bias = nn.Parameter(torch.zeros(heads, dim_seq))

        # 初始化权重和偏置参数
        init_eps /= dim_seq
        nn.init.uniform_(self.weight, -init_eps, init_eps)
        nn.init.constant_(self.bias, 1.)

        # 设置激活函数
        self.act = act
        # 创建一个缓冲区，用于存储掩码
        self.register_buffer('mask', ~torch.ones(dim_seq, dim_seq).triu_(1).bool())

    # 前向传播函数，接受输入 x 和 gate_res
    def forward(self, x, gate_res = None):
        # 获取设备信息、输入序列长度和头数
        device, n, h = x.device, x.shape[1], self.heads

        # 将输入 x 分成两部分：res 和 gate
        res, gate = x.chunk(2, dim = -1)
        # 对 gate 进行 LayerNorm 处理
        gate = self.norm(gate)

        # 获取权重和偏置参数
        weight, bias = self.weight, self.bias
        weight, bias = weight[:, :n, :n], bias[:, :n]

        # 对权重参数应用掩码
        weight = weight * self.mask[None, :n, :n].int().float()

        # 重排 gate 的维度
        gate = rearrange(gate, 'b n (h d) -> b h n d', h = h)
        # 执行矩阵乘法操作
        gate = einsum('b h n d, h m n -> b h m d', gate, weight)
        # 添加偏置参数
        gate = gate + rearrange(bias, 'h n -> () h n ()')
        # 重排 gate 的维度
        gate = rearrange(gate, 'b h n d -> b n (h d)')

        # 如果存在 gate_res，则将其加到 gate 上
        if exists(gate_res):
            gate = gate + gate_res

        # 返回激活函数作用后的结果乘以 res
        return self.act(gate) * res

# 定义一个类 CausalLocalSGU，继承自 nn.Module
class CausalLocalSGU(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        dim,
        dim_seq,
        init_eps = 1e-3,
        heads = 4,
        window = 128,
        act = nn.Identity()
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 计算输出维度
        dim_out = dim // 2

        # 初始化 LayerNorm 模块
        self.norm = nn.LayerNorm(dim_out)

        # 设置头数、窗口大小和权重、偏置参数
        self.heads = heads
        self.window = window
        self.weight = nn.Parameter(torch.zeros(heads, window, window * 2))
        self.bias = nn.Parameter(torch.zeros(heads, window))

        # 初始化权重和偏置参数
        init_eps /= window
        nn.init.uniform_(self.weight, -init_eps, init_eps)
        nn.init.constant_(self.bias, 1.)

        # 设置激活函数
        self.act = act
        # 创建一个缓冲区，用于存储掩码
        self.register_buffer('mask', ~torch.ones(window, window * 2).triu_(window + 1).bool())

    # 前向传播函数，接受输入 x 和 gate_res
    def forward(self, x, gate_res = None):
        # 获取设备信息、输入序列长度、头数和窗口大小
        device, n, h, w = x.device, x.shape[1], self.heads, self.window

        # 将输入 x 分成两部分：res 和 gate
        res, gate = x.chunk(2, dim = -1)

        # 对 gate 进行填充和重排
        gate = pad_to_multiple(gate, w, dim = -2)
        gate = rearrange(gate, 'b (w n) d -> b w n d', n = w)

        # 对 gate 进行 LayerNorm 处理
        gate = self.norm(gate)

        # 对 gate 进行填充和拼接
        gate = F.pad(gate, (0, 0, 0, 0, 1, 0), value = 0.)
        gate = torch.cat((gate[:, :-1], gate[:, 1:]), dim = 2)

        # 获取权重和偏置参数
        weight, bias = self.weight, self.bias

        # 对权重参数应用掩码
        weight = weight * self.mask[None, ...].int().float()

        # 重排 gate 的维度
        gate = rearrange(gate, 'b w n (h d) -> b w h n d', h = h)
        # 执行矩阵乘法操作
        gate = einsum('b w h n d, h m n -> b w h m d', gate, weight)
        # 添加偏置参数
        gate = gate + rearrange(bias, 'h n -> () () h n ()')

        # 重排 gate 的维度
        gate = rearrange(gate, 'b w h n d -> b w n (h d)')

        # 重排 gate 的维度
        gate = rearrange(gate, 'b w n d -> b (w n) d')
        gate = gate[:, :n]

        # 如果存在 gate_res，则将其加到 gate 上
        if exists(gate_res):
            gate = gate + gate_res

        # 返回激活函数作用后的结果乘以 res
        return self.act(gate) * res

# 定义一个类 AxiallyFold，继承自 nn.Module
class AxiallyFold(nn.Module):
    # 初始化函数，接受维度、步长和函数参数
    def __init__(self, dim, every, fn):
        # 调用父类的初始化函数
        super().__init__()
        # 设置函数和步长
        self.fn = fn
        self.every = every
        # 如果步长大于 1，则创建一个卷积层
        self.conv = nn.Conv1d(dim, dim, kernel_size = every, groups = dim) if every > 1 else None

    # 前向传播函数，接受输入 x
    def forward(self, x):
        # 获取步长
        every = self.every
        # 如果步长小于等于 1，则直接应用函数
        if every <= 1:
            return self.fn(x)

        # 获取序列长度
        n = x.shape[1]
        # 对输入 x 进行填充和重排
        x = pad_to_multiple(x, self.every, dim = -2)
        x = rearrange(x, 'b (n e) d -> (b e) n d', e = every)
        x = self.fn(x)

        # 重排结果并进行填充
        x = rearrange(x, '(b e) n d -> b d (n e)', e = every)
        x = F.pad(x, (every - 1, 0), value = 0)
        # 对结果应用卷积操作
        out = self.conv(x)
        out = rearrange(out, 'b d n -> b n d')
        return out[:, :n]

# 定义一个类 gMLPBlock，继承自 nn.Module
class gMLPBlock(nn.Module):
    # 初始化函数，设置模型参数
    def __init__(
        self,
        *,
        dim,  # 输入维度
        seq_len,  # 序列长度
        dim_ff,  # FeedForward 层维度
        heads = 4,  # 多头注意力机制的头数，默认为4
        causal = False,  # 是否使用因果关系，默认为False
        window = None,  # 窗口大小，默认为None
        attn_dim = None,  # 注意力机制维度，默认为None
        act = nn.Identity()  # 激活函数，默认为恒等函数
    ):
        super().__init__()
        is_windowed = exists(window) and window < seq_len

        # 根据是否存在窗口大小选择不同的 SGU 类型
        SGU_klass = partial(CausalLocalSGU, window = window) if is_windowed else CausalSGU
        # 根据是否存在窗口大小选择不同的 Attention 类型
        Attention_klass = partial(LocalAttention, window = window) if is_windowed else Attention

        # 如果存在注意力机制维度，则创建注意力层
        self.attn = Attention_klass(dim_in = dim, dim_inner = attn_dim, dim_out = dim_ff // 2) if exists(attn_dim) else None

        # 输入投影层，包含线性层和 GELU 激活函数
        self.proj_in = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.GELU()
        )
        # SGU 层，根据选择的 SGU 类型进行初始化
        self.sgu =  SGU_klass(dim_ff, seq_len, causal, heads = heads, act = act)
        # 输出投影层，线性层
        self.proj_out = nn.Linear(dim_ff // 2, dim)

    # 前向传播函数
    def forward(self, x):
        # 如果存在注意力层，则进行注意力计算
        gate_res = self.attn(x) if exists(self.attn) else None
        # 输入投影
        x = self.proj_in(x)
        # SGU 层计算
        x = self.sgu(x, gate_res = gate_res)
        # 输出投影
        x = self.proj_out(x)
        return x
# 主要类

class gMLPGPT(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,  # 标记的数量
        dim,  # 向量维度
        depth,  # 模型深度
        seq_len,  # 序列长度
        heads = 1,  # 多头注意力机制的头数，默认为1
        ff_mult = 4,  # FeedForward 层的倍数，默认为4
        prob_survival = 1.,  # 存活概率，默认为1
        reversible = False,  # 是否可逆，默认为False
        window = None,  # 窗口大小，默认为None
        attn_dim = None,  # 注意力维度，默认为None
        act = nn.Identity()  # 激活函数，默认为恒等函数
    ):
        super().__init__()
        dim_ff = dim * ff_mult
        self.seq_len = seq_len
        self.prob_survival = prob_survival

        self.to_embed = nn.Embedding(num_tokens, dim)  # 创建嵌入层

        window = cast_tuple(window, depth)  # 将窗口大小转换为元组
        window = tuple(map(lambda t: t if isinstance(t, tuple) else (t, 1), window))  # 将窗口大小转换为元组

        attn_dims = cast_tuple(attn_dim, depth)  # 将注意力维度转换为元组

        assert len(window) == depth, f'num window sizes {len(window)} must be equal to depth {depth}'  # 断言窗口大小数量必须等于深度

        layers = nn.ModuleList([])  # 创建模块列表

        for ind, (w, ax), attn_dim in zip(range(depth), window, attn_dims):
            attn_dim = attn_dim if exists(window) else None
            get_gmlp = lambda: PreNorm(dim, AxiallyFold(dim, ax, gMLPBlock(dim = dim, dim_ff = dim_ff, seq_len = seq_len, heads = heads, window = w, act = act, attn_dim = attn_dim))  # 获取 gMLP 模块

            layer_blocks = nn.ModuleList([
                get_gmlp()
            ])

            if reversible:
                layer_blocks.append(FeedForward(dim, mult = ff_mult))  # 如果是可逆模型，添加 FeedForward 层

            layers.append(layer_blocks)  # 添加模块列表到层列表

        execute_klass = SequentialSequence if not reversible else ReversibleSequence  # 根据是否可逆选择执行类
        self.net = execute_klass(layers)  # 创建执行网络

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),  # 层归一化
            nn.Linear(dim, num_tokens)  # 线性层
        )

    def forward(self, x):
        layer_dropout = 1. - self.prob_survival  # 计算层的丢弃率

        x = self.to_embed(x)  # 嵌入输入序列
        out = self.net(x, layer_dropout = layer_dropout)  # 通过网络传播输入
        return self.to_logits(out)  # 返回输出��果
```