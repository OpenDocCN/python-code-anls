# `.\lucidrains\perceiver-pytorch\perceiver_pytorch\perceiver_io.py`

```py
# 从 math 模块中导入 pi 和 log 函数
# 从 functools 模块中导入 wraps 函数
# 导入 torch 模块及其子模块 nn, einsum, functional
# 从 einops 模块中导入 rearrange, repeat 函数
from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

# 定义辅助函数

# 判断值是否存在
def exists(val):
    return val is not None

# 如果值存在则返回该值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 缓存函数的结果
def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

# 结构化的 dropout，比传统的注意力 dropout 更有效

# 对序列进行 dropout
def dropout_seq(seq, mask, dropout):
    b, n, *_, device = *seq.shape, seq.device
    logits = torch.randn(b, n, device = device)

    if exists(mask):
        logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)

    keep_prob = 1. - dropout
    num_keep = max(1,  int(keep_prob * n))
    keep_indices = logits.topk(num_keep, dim = 1).indices

    batch_indices = torch.arange(b, device = device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')

    seq = seq[batch_indices, keep_indices]

    if exists(mask):
        seq_counts = mask.sum(dim = -1)
        seq_keep_counts = torch.ceil(seq_counts * keep_prob).int()
        keep_mask = torch.arange(num_keep, device = device) < rearrange(seq_keep_counts, 'b -> b 1')

        mask = mask[batch_indices, keep_indices] & keep_mask

    return seq, mask

# 辅助类

# 预层归一化
class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

# GEGLU 激活函数
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

# 前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

# 注意力机制
class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # 注意力机制，我们无法获得足够的
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# 主类

class PerceiverIO(nn.Module):
    # 初始化函数，设置模型参数
    def __init__(
        self,
        *,
        depth,
        dim,
        queries_dim,
        logits_dim = None,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        weight_tie_layers = False,
        decoder_ff = False,
        seq_dropout_prob = 0.
    ):
        # 调用父类初始化函数
        super().__init__()
        # 设置序列的dropout概率
        self.seq_dropout_prob = seq_dropout_prob

        # 初始化模型中的可学习参数
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        # 创建交叉注意力块和前馈网络块
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = dim),
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])

        # 定义获取潜在注意力和前馈网络的函数
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        # 使用缓存函数对获取潜在注意力和前馈网络的函数进行缓存
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        # 初始化模型的层
        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        # 循环创建多个层
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        # 创建解码器的交叉注意力块和前馈网络块
        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, latent_dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = latent_dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

        # 创建输出层
        self.to_logits = nn.Linear(queries_dim, logits_dim) if exists(logits_dim) else nn.Identity()

    # 前向传播函数
    def forward(
        self,
        data,
        mask = None,
        queries = None
    ):
        # 获取数据的维度和设备信息
        b, *_, device = *data.shape, data.device

        # 将潜在向量重复扩展到与数据相同的维度
        x = repeat(self.latents, 'n d -> b n d', b = b)

        # 获取交��注意力块和前馈网络块
        cross_attn, cross_ff = self.cross_attend_blocks

        # 结构化的dropout操作
        if self.training and self.seq_dropout_prob > 0.:
            data, mask = dropout_seq(data, mask, self.seq_dropout_prob)

        # 执行交叉注意力操作
        x = cross_attn(x, context = data, mask = mask) + x
        x = cross_ff(x) + x

        # 多层自注意力和前馈网络操作
        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        # 如果没有查询数据，则直接返回结果
        if not exists(queries):
            return x

        # 确保查询数据包含批处理维度
        if queries.ndim == 2:
            queries = repeat(queries, 'n d -> b n d', b = b)

        # 从解码器查询到潜在向量的交叉注意力操作
        latents = self.decoder_cross_attn(queries, context = x)

        # 可选的解码器前馈网络操作
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)

        # 最终的线性输出
        return self.to_logits(latents)
# Perceiver LM 示例

class PerceiverLM(nn.Module):
    def __init__(
        self,
        *,
        dim,  # 定义维度
        num_tokens,  # 定义标记数量
        max_seq_len,  # 定义最大序列长度
        **kwargs  # 其他参数
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)  # 创建标记嵌入层
        self.pos_emb = nn.Embedding(max_seq_len, dim)  # 创建位置嵌入层

        self.perceiver_io = PerceiverIO(  # 创建 PerceiverIO 模块
            dim = dim,
            queries_dim = dim,
            logits_dim = num_tokens,
            **kwargs
        )

    def forward(
        self,
        x,  # 输入张量
        mask = None  # 掩码，默认为空
    ):
        n, device = x.shape[1], x.device  # 获取输入张量的维度和设备信息
        x = self.token_emb(x)  # 对输入张量进行标记嵌入

        pos_emb = self.pos_emb(torch.arange(n, device = device))  # 根据序列长度创建位置嵌入
        pos_emb = rearrange(pos_emb, 'n d -> () n d')  # 重新排列位置嵌入的维度
        x = x + pos_emb  # 将标记嵌入和位置嵌入相加

        logits = self.perceiver_io(x, mask = mask, queries = x)  # 使用 PerceiverIO 模块进行前向传播
        return logits  # 返回输出结果
```