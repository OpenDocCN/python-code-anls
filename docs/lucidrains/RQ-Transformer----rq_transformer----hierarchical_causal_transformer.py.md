# `.\lucidrains\RQ-Transformer\rq_transformer\hierarchical_causal_transformer.py`

```
# 导入数学库
import math
# 导入 functools 库
import functools
# 导入 torch 库
import torch
# 导入 torch.nn.functional 库
import torch.nn.functional as F
# 从 torch 中导入 nn 和 einsum
from torch import nn, einsum
# 从 einops_exts 中导入 rearrange_with_anon_dims
from einops_exts import rearrange_with_anon_dims
# 从 einops 中导入 rearrange, reduce, repeat

# helpers

# 判断值是否存在
def exists(val):
    return val is not None

# 如果值存在则返回该值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 计算 num 与 mult 的余数
def remainder_to_mult(num, mult):
    return (mult - num % mult) % mult

# 将输入转换为元组
def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

# 对多个数进行乘法运算
def reduce_mult(nums):
    return functools.reduce(lambda x, y: x * y, nums, 1)

# tensor helpers

# 计算张量的对数
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# 生成 Gumbel 噪声
def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

# 生成 Gumbel 分布采样
def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

# 获取前 k 个最大值的概率
def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# positional bias

# 定义 Alibi 类
class Alibi(nn.Module):
    def __init__(self, heads, **kwargs):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    def forward(self, i, j, device):
        if exists(self.bias) and self.bias.shape[-1] >= j:
            return self.bias[..., :j]

        bias = torch.arange(j, device = device)
        bias = rearrange(bias, 'j -> 1 1 j')
        bias = bias * self.slopes

        self.register_buffer('bias', bias, persistent = False)
        return self.bias

# norm

# 定义 RMSNorm 类
class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

# helper classes

# 定义 FeedForward 函数
def FeedForward(*, dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        RMSNorm(dim),
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

# 定义 Attention 类
class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)
    # 实现自注意力机制的前向传播
    def forward(self, x, attn_bias = None):
        # 获取头数和设备信息
        h, device = self.heads, x.device

        # 对输入进行归一化处理
        x = self.norm(x)
        # 将输入转换为查询、键、值
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))
        # 将查询向量重新排列为多头形式
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        # 缩放查询向量
        q = q * self.scale
        # 计算注意力分数
        sim = einsum('b h i d, b j d -> b h i j', q, k)

        # 如果存在注意力偏置，则加上
        if exists(attn_bias):
            sim = sim + attn_bias

        # 创建掩码
        i, j = sim.shape[-2:]
        mask_value = -torch.finfo(sim.dtype).max
        mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
        sim = sim.masked_fill(mask, mask_value)

        # 对注意力分数进行归一化处理
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # 计算输出
        out = einsum('b h i j, b j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
# 定义一个名为 Transformer 的类，继承自 nn.Module
class Transformer(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        *,
        dim,  # 维度
        layers,  # 层数
        dim_head = 64,  # 头部维度
        heads = 8,  # 头部数量
        attn_dropout = 0.,  # 注意力机制的 dropout
        ff_dropout = 0.,  # 前馈神经网络的 dropout
        ff_mult = 4,  # 前馈神经网络的倍数
        rel_pos_bias = True  # 是否使用相对位置偏置
    ):
        super().__init__()
        # 如果使用相对位置偏置，则创建 Alibi 对象，否则为 None
        self.alibi = Alibi(heads = heads) if rel_pos_bias else None
        # 创建空的 nn.ModuleList 对象
        self.layers = nn.ModuleList([])

        # 循环创建 layers 个层
        for _ in range(layers):
            # 每个层包含一个注意力机制和一个前馈神经网络
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        # 创建 RMSNorm 对象
        self.norm = RMSNorm(dim)

    # 前向传播函数
    def forward(self, x):
        # 获取输入张量 x 的倒数第二个维度的大小
        n = x.shape[-2]
        # 如果存在相对位置偏置，则根据输入张量 x 的设备创建注意力偏置
        attn_bias = self.alibi(n, n, device = x.device) if exists(self.alibi) else None

        # 遍历每个层中的注意力机制和前馈神经网络
        for attn, ff in self.layers:
            # 使用注意力机制处理输入张量 x，并加上原始输入
            x = attn(x, attn_bias = attn_bias) + x
            # 使用前馈神经网络处理输入张量 x，并加上原始输入
            x = ff(x) + x

        # 返回经过归一化处理后的结果
        return self.norm(x)

# 主类
class HierarchicalCausalTransformer(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        *,
        num_tokens,  # 标记数量
        dim,  # 维度
        depth,  # 深度
        max_seq_len,  # 最大序列长度
        dim_head = 64,  # 头部维度
        heads = 8,  # 头部数量
        attn_dropout = 0.,  # 注意力机制的 dropout
        ff_mult = 4,  # 前馈神经网络的倍数
        ff_dropout = 0.,  # 前馈神经网络的 dropout
        pad_id = 0,  # 填充标记的 id
        rel_pos_bias = True  # 是否使用相对位置偏置
    ):
        super().__init__()

        # 简化每个层次的配置
        # depth = (2, 2, 4) ���示第一阶段深度为 2，第二阶段深度为 2，第三阶段深度为 4
        # max_seq_len = (16, 8, 4) 表示第一阶段最大序列长度为 16，第二阶段为 8，第三阶段为 4

        assert isinstance(depth, tuple) and isinstance(max_seq_len, tuple)
        assert len(depth) == len(max_seq_len)

        # 阶段数量为深度元组的长度
        self.stages = len(depth)

        # 创建标记嵌入层
        self.token_emb = nn.Embedding(num_tokens, dim)
        # 创建起始标记参数
        self.start_tokens = nn.Parameter(torch.randn(dim))

        # 最大序列长度和位置嵌入层列表
        self.max_seq_len = max_seq_len
        self.pos_embs = nn.ModuleList([nn.Embedding(seq_len, dim) for seq_len in max_seq_len])

        # 创建 Transformer 模块列表
        self.transformers = nn.ModuleList([])

        # 遍历每个阶段的深度
        for stage_depth in depth:
            # 创建 Transformer 模块并添加到列表中
            self.transformers.append(Transformer(
                dim = dim,
                layers = stage_depth,
                dim_head = dim_head,
                heads = heads,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                ff_mult = ff_mult,
                rel_pos_bias = rel_pos_bias
            ))

        # 创建线性层用于输出标记
        self.to_logits = nn.Linear(dim, num_tokens)
        # 填充标记的 id
        self.pad_id = pad_id

    # 生成函数
    def generate(self, prime = None, filter_thres = 0.9, temperature = 1., default_batch_size = 1):
        # 计算总序列长度
        total_seq_len = reduce_mult(self.max_seq_len)
        # 获取设备
        device = next(self.parameters()).device

        # 如果 prime 为空，则创建一个空的张量
        if not exists(prime):
            prime = torch.empty((default_batch_size, 0), dtype = torch.long, device = device)

        # 初始化序列为 prime
        seq = prime

        # 循环生成序列
        for _ in range(total_seq_len - seq.shape[-1]):
            # 获取 logits
            logits = self.forward(seq)[:, -1]
            # 根据 filter_thres 过滤 top-k logits
            logits = top_k(logits, thres = filter_thres)
            # 使用 Gumbel 分布采样
            sampled = gumbel_sample(logits, dim = -1, temperature = temperature)
            # 将采样结果拼接到序列中
            seq = torch.cat((seq, rearrange(sampled, 'b -> b 1')), dim = -1)

        # 重新排列序列并返回
        return rearrange_with_anon_dims(seq, 'b (...d) -> b ...d', d = self.max_seq_len)

    # 空输入前向传播函数
    def forward_empty(self, batch_size):
        # 处理特殊情况，从输入为 0（仅起始标记）的样本中采样

        # 重复起始标记，创建 tokens 张量
        tokens = repeat(self.start_tokens, 'd -> b 1 d', b = batch_size)

        # 遍历每个 Transformer 模块
        for transformer in self.transformers:
            tokens = transformer(tokens)

        # 返回 logits
        return self.to_logits(tokens)
    # 定义前向传播函数，接受输入 ids 和是否返回损失值的标志
    def forward(self, ids, return_loss = False):
        # 断言输入 ids 的维度为 2 或者 self.stages + 1
        assert ids.ndim in {2, self.stages + 1}
        # 检查是否为扁平化维度
        flattened_dims = ids.ndim == 2
        # 保存原始 ids 的维度
        ids_orig_ndim = ids.ndim

        # 如果 ids 为空，则调用 forward_empty 函数
        if ids.numel() == 0:
            return self.forward_empty(ids.shape[0])

        # 如果是扁平化维度，则进行自动填充
        if flattened_dims:
            # 获取序列长度
            seq_len = ids.shape[-1]
            # 计算填充值
            multiple_of = reduce_mult(self.max_seq_len[1:])
            padding = remainder_to_mult(seq_len, multiple_of)
            # 对 ids 进行填充和重新排列
            ids = F.pad(ids, (0, padding), value = self.pad_id)
            ids = rearrange_with_anon_dims(ids, 'b (l ...d) -> b l ...d', d = self.max_seq_len[1:])

        # 获取 ids 的形状和设备信息
        b, *prec_dims, device = *ids.shape, ids.device

        # 检查一些维度

        assert prec_dims[0] <= self.max_seq_len[0], 'the first dimension of your axial autoregressive transformer must be less than the first tuple element of max_seq_len (like any autoregressive transformer)'
        assert tuple(prec_dims[1:]) == tuple(self.max_seq_len[1:]), 'all subsequent dimensions must match exactly'

        # 获取 token embeddings

        tokens = self.token_emb(ids)

        # 获取所有层次阶段的 tokens，减少适当的维度并添加绝对位置嵌入

        tokens_at_stages = []
        reduced_tokens = tokens

        for ind, pos_emb in zip(range(len(prec_dims)), reversed(self.pos_embs)):
            is_first = ind == 0

            if not is_first:
                reduced_tokens = reduce(reduced_tokens, 'b ... r d -> b ... d', 'sum')

            positions = pos_emb(torch.arange(reduced_tokens.shape[-2], device = device))
            tokens_with_position = reduced_tokens + positions
            tokens_at_stages.insert(0, tokens_with_position)

        # 获取起始 tokens 并附加到最粗糙的阶段

        start_tokens = repeat(self.start_tokens, 'f -> b 1 f', b = b)

        # 空间 tokens 是在深度 pos 减少的 tokens + 空间位置

        for ind, (stage_tokens, transformer) in enumerate(zip(tokens_at_stages, self.transformers)):
            is_last = ind == (self.stages - 1)

            stage_tokens = torch.cat((
                start_tokens,
                stage_tokens,
            ), dim = -2)

            *prec_dims, _, _ = stage_tokens.shape

            stage_tokens = rearrange(stage_tokens, '... n d -> (...) n d')
            attended = transformer(stage_tokens)
            attended = rearrange_with_anon_dims(attended, '(...b) n d -> ...b n d', b = prec_dims)

            start_tokens = rearrange(attended[..., :-1, :], '... n d -> ... n 1 d')

        logits = self.to_logits(attended)

        logits = logits[..., 1:, :]

        # 如果不需要返回损失值

        if not return_loss:

            if flattened_dims:
                logits = rearrange(logits, 'b ... n -> b (...) n')
                logits = logits[:, :seq_len]

            return logits

        preds = rearrange(logits, 'b ... c -> b c (...)')
        labels = rearrange(ids, 'b ... -> b (...)')

        # 计算交叉熵损失
        loss = F.cross_entropy(
            preds[..., :-1],
            labels[..., 1:],
            ignore_index = self.pad_id
        )
        return loss
```