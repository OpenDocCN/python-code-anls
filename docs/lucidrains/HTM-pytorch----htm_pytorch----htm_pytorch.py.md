# `.\lucidrains\HTM-pytorch\htm_pytorch\htm_pytorch.py`

```py
# 从 math 模块中导入 ceil 函数
from math import ceil
# 导入 torch 模块
import torch
# 从 torch 模块中导入 nn 和 einsum
from torch import nn, einsum
# 从 torch.nn.functional 模块中导入 F
import torch.nn.functional as F
# 从 einops 模块中导入 rearrange 和 repeat

from einops import rearrange, repeat

# helpers

# 定义函数 exists，判断值是否存在
def exists(val):
    return val is not None

# 定义函数 default，如果值存在则返回该值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 定义函数 pad_to_multiple，将输入张量在指定维度上填充到指定的倍数长度
def pad_to_multiple(t, multiple, dim = -2, value = 0.):
    seq_len = t.shape[dim]
    pad_to_len = ceil(seq_len / multiple) * multiple
    remainder = pad_to_len - seq_len

    if remainder == 0:
        return t

    zeroes = (0, 0) * (-dim - 1)
    padded_t = F.pad(t, (*zeroes, remainder, 0), value = value)
    return padded_t

# positional encoding

# 定义 SinusoidalPosition 类，用于生成位置编码
class SinusoidalPosition(nn.Module):
    def __init__(
        self,
        dim,
        min_timescale = 2.,
        max_timescale = 1e4
    ):
        super().__init__()
        freqs = torch.arange(0, dim, min_timescale)
        inv_freqs = max_timescale ** (-freqs / dim)
        self.register_buffer('inv_freqs', inv_freqs)

    def forward(self, x):
        seq_len = x.shape[-2]
        seq = torch.arange(seq_len - 1, -1, -1.)
        sinusoidal_inp = rearrange(seq, 'n -> n ()') * rearrange(self.inv_freqs, 'd -> () d')
        pos_emb = torch.cat((sinusoidal_inp.sin(), sinusoidal_inp.cos()), dim = -1)
        return pos_emb

# multi-head attention

# 定义 Attention 类，实现多头注意力机制
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(
        self,
        x,
        mems,
        mask = None
    ):
        h = self.heads
        q, k, v = self.to_q(x), *self.to_kv(mems).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b ... (h d) -> (b h) ... d', h = h), (q, k, v))
        q = q * self.scale

        sim = einsum('b m i d, b m i j d -> b m i j', q, k)

        if exists(mask):
            mask = repeat(mask, 'b ... -> (b h) ...', h = h)
            mask_value = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(~mask, mask_value)

        attn = sim.softmax(dim = -1)

        out = einsum('... i j, ... i j d -> ... i d', attn, v)
        out = rearrange(out, '(b h) ... d -> b ... (h d)', h = h)
        return self.to_out(out)

# main class

# 定义 HTMAttention 类，实现 HTMAttention 模型
class HTMAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        topk_mems = 2,
        mem_chunk_size = 32,
        dim_head = 64,
        add_pos_enc = True,
        eps = 1e-5
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.scale = dim ** -0.5

        self.to_summary_queries = nn.Linear(dim, dim)
        self.to_summary_keys = nn.Linear(dim, dim)

        self.attn = Attention(dim = dim, heads = heads, dim_head = dim_head)

        self.topk_mems = topk_mems
        self.mem_chunk_size = mem_chunk_size
        self.pos_emb = SinusoidalPosition(dim = dim) if add_pos_enc else None

    def forward(
        self,
        queries,
        memories,
        mask = None,
        chunk_attn_mask = None
    ):
        # 解包参数
        dim, query_len, mem_chunk_size, topk_mems, scale, eps = self.dim, queries.shape[1], self.mem_chunk_size, self.topk_mems, self.scale, self.eps

        # 填充记忆，以及如果需要的话，填充记忆掩码，然后分成块

        memories = pad_to_multiple(memories, mem_chunk_size, dim = -2, value = 0.)
        memories = rearrange(memories, 'b (n c) d -> b n c d', c = mem_chunk_size)

        if exists(mask):
            mask = pad_to_multiple(mask, mem_chunk_size, dim = -1, value = False)
            mask = rearrange(mask, 'b (n c) -> b n c', c = mem_chunk_size)

        # 通过均值池化总结记忆，考虑掩码

        if exists(mask):
            mean_mask = rearrange(mask, '... -> ... ()')
            memories = memories.masked_fill(~mean_mask, 0.)
            numer = memories.sum(dim = 2)
            denom = mean_mask.sum(dim = 2)
            summarized_memories = numer / (denom + eps)
        else:
            summarized_memories = memories.mean(dim = 2)

        # 推导查询和总结的记忆键

        summary_queries = self.to_summary_queries(queries)
        summary_keys = self.to_summary_keys(summarized_memories.detach())

        # 对总结的键进行单头注意力

        sim = einsum('b i d, b j d -> b i j', summary_queries, summary_keys) * scale
        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            chunk_mask = mask.any(dim = 2)
            chunk_mask = rearrange(chunk_mask, 'b j -> b () j')
            sim = sim.masked_fill(~chunk_mask, mask_value)

        if exists(chunk_attn_mask):
            sim = sim.masked_fill(~chunk_attn_mask, mask_value)

        topk_logits, topk_indices = sim.topk(k = topk_mems, dim = -1)
        weights = topk_logits.softmax(dim = -1)

        # 为内存注意力准备查询

        queries = repeat(queries, 'b n d -> b k n d', k = topk_mems)

        # 选择前k个记忆

        memories = repeat(memories, 'b m j d -> b m i j d', i = query_len)
        mem_topk_indices = repeat(topk_indices, 'b i m -> b m i j d', j = mem_chunk_size, d = dim)
        selected_memories = memories.gather(1, mem_topk_indices)

        # 位置编码

        if exists(self.pos_emb):
            pos_emb = self.pos_emb(memories)
            selected_memories = selected_memories + rearrange(pos_emb, 'n d -> () () () n d')

        # 选择掩码

        selected_mask = None
        if exists(mask):
            mask = repeat(mask, 'b m j -> b m i j', i = query_len)
            mask_topk_indices = repeat(topk_indices, 'b i m -> b m i j', j = mem_chunk_size)
            selected_mask = mask.gather(1, mask_topk_indices)

        # 现在进行内存注意力

        within_mem_output = self.attn(
            queries,
            selected_memories.detach(),
            mask = selected_mask
        )

        # 对内存注意力输出进行加权

        weighted_output = within_mem_output * rearrange(weights, 'b i m -> b m i ()')
        output = weighted_output.sum(dim = 1)
        return output
# 定义一个 HTMBlock 类，继承自 nn.Module
class HTMBlock(nn.Module):
    # 初始化方法，接受维度参数和其他关键字参数
    def __init__(self, dim, **kwargs):
        super().__init__()
        # 初始化 LayerNorm 层，对输入进行归一化处理
        self.norm = nn.LayerNorm(dim)
        # 初始化 HTMAttention 层，处理注意力机制
        self.attn = HTMAttention(dim=dim, **kwargs)
    # 前向传播方法，接受查询 queries 和记忆 memories，以及其他关键字参数
    def forward(
        self,
        queries,
        memories,
        **kwargs
    ):
        # 对查询 queries 进行归一化处理
        queries = self.norm(queries)
        # 使用 HTMAttention 层处理查询 queries 和记忆 memories，再加上原始查询 queries
        out = self.attn(queries, memories, **kwargs) + queries
        # 返回处理后的结果
        return out
```