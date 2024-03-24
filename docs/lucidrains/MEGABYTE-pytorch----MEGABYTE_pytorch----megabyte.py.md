# `.\lucidrains\MEGABYTE-pytorch\MEGABYTE_pytorch\megabyte.py`

```
# 导入数学库
import math
# 导入 functools 库
import functools
# 从 itertools 库中导入 zip_longest 函数
from itertools import zip_longest

# 导入 torch 库
import torch
# 从 torch.nn.functional 中导入 F
import torch.nn.functional as F
# 从 torch 中导入 nn, einsum
from torch import nn, einsum

# 从 einops 库中导入 rearrange, reduce, repeat, pack, unpack
from einops import rearrange, reduce, repeat, pack, unpack
# 从 einops.layers.torch 中导入 Rearrange
from einops.layers.torch import Rearrange

# 从 beartype 库中导入 beartype
from beartype import beartype
# 从 beartype.typing 中导入 Tuple, Union
from beartype.typing import Tuple, Union

# 从 MEGABYTE_pytorch.attend 中导入 Attend
from MEGABYTE_pytorch.attend import Attend

# 从 tqdm 中导入 tqdm
from tqdm import tqdm

# 辅助函数

# 判断值是否存在
def exists(val):
    return val is not None

# 如果值存在则返回该值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 将单个张量按照指定模式打包
def pack_one(t, pattern):
    return pack([t], pattern)

# 将单个张量按照指定模式解包
def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# 计算使 num 变为 mult 的倍数的余数
def remainder_to_mult(num, mult):
    return (mult - num % mult) % mult

# 将输入转换为元组，如果输入不是元组则重复 length 次
def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

# 计算多个数的乘积
def reduce_mult(nums):
    return functools.reduce(lambda x, y: x * y, nums, 1)

# 张量辅助函数

# 计算张量的自然对数，避免小于 eps 的值
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps)

# 生成 Gumbel 噪声
def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

# 生成 Gumbel 分布采样
def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

# 保留前 k 个最大值，其余设为负无穷
def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# Token Shift，从 Peng et al of RWKV 中借鉴
def token_shift(t):
    t, t_shift = t.chunk(2, dim = -1)
    t_shift = F.pad(t_shift, (0, 0, 1, -1))
    return torch.cat((t, t_shift), dim = -1)

# 旋转位置嵌入
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    @property
    def device(self):
        return next(self.buffers()).device

    def forward(self, seq_len):
        t = torch.arange(seq_len, device = self.device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)
        return freqs

# 旋转半个张量
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

# 应用旋转位置嵌入
def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()

# 归一化
class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

# 辅助类

# 创建 FeedForward 网络
def FeedForward(*, dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        RMSNorm(dim),
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

# 注意力机制
class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        flash = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.attend = Attend(
            causal = True,
            flash = flash,
            dropout = dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)
    # 实现 Transformer 模型的前向传播过程
    def forward(self, x, rotary_emb = None):
        # 获取头数和设备信息
        h, device = self.heads, x.device

        # 对输入进行归一化处理
        x = self.norm(x)
        # 将输入 x 分别转换为查询 q、键 k、值 v
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))
        # 将查询 q 重新排列为形状为 'b h n d' 的张量
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        # 如果存在旋转位置编码，则对查询 q 和键 k 应用旋转位置编码
        if exists(rotary_emb):
            q, k = map(lambda t: apply_rotary_pos_emb(rotary_emb, t), (q, k))

        # 使用注意力机制进行注意力计算
        out = self.attend(q, k, v)

        # 将输出重新排列为形状为 'b n (h d)' 的张量
        out = rearrange(out, 'b h n d -> b n (h d)')
        # 将输出转换为最终输出
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
        rel_pos = True,  # 是否使用相对位置编码
        flash_attn = False  # 是否使用 Flash 注意力机制
    ):
        super().__init__()  # 调用父类的初始化函数
        self.rotary_emb = RotaryEmbedding(dim_head) if rel_pos else None  # 如果使用相对位置编码，则创建旋转嵌入对象，否则为 None
        self.layers = nn.ModuleList([])  # 创建一个空的 nn.ModuleList 对象

        # 循环创建指定层数的注意力机制和前馈神经网络
        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, flash = flash_attn),  # 创建注意力机制对象
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)  # 创建前馈神经网络对象
            ]))

        self.norm = RMSNorm(dim)  # 创建 RMS 归一化对象

    # 前向传播函数，接受输入 x
    def forward(self, x):
        n = x.shape[-2]  # 获取输入 x 的倒数第二维度大小
        rotary_emb = self.rotary_emb(n) if exists(self.rotary_emb) else None  # 如果存在旋转嵌入对象，则根据 n 创建旋转嵌入，否则为 None

        # 遍历每一层的注意力机制和前馈神经网络
        for attn, ff in self.layers:
            x = attn(token_shift(x), rotary_emb = rotary_emb) + x  # 执行注意力机制和残差连接
            x = ff(token_shift(x)) + x  # 执行前馈神经网络和残差连接

        return self.norm(x)  # 返回经过归一化的结果

# 主类 MEGABYTE
class MEGABYTE(nn.Module):

    @beartype
    # 初始化函数，接受多个参数
    def __init__(
        self,
        *,
        num_tokens,  # 标记数量
        dim: Union[Tuple, int],  # 维度
        depth: Tuple,  # 深度
        max_seq_len: Tuple,  # 最大序列长度
        dim_head = 64,  # 头部维度
        heads = 8,  # 头部数量
        attn_dropout = 0.,  # 注意力机制的 dropout
        ff_mult = 4,  # 前馈神经网络的倍数
        ff_dropout = 0.,  # 前馈神经网络的 dropout
        pad_id = 0,  # 填充��记的 id
        rel_pos = False,  # 是否使用相对位置编码
        pos_emb = False,  # 是否使用位置嵌入
        flash_attn = False  # 是否使用 Flash 注意力机制
    ):
        # 调用父类的构造函数
        super().__init__()

        # 简化每个层次的配置
        # depth = (2, 2, 4) 表示第一阶段深度为2，第二阶段深度为2，第三阶段深度为4
        # max_seq_len = (16, 8, 4) 表示第一阶段最大序列长度为16，第二阶段为8，最后一阶段为4
        assert isinstance(depth, tuple) and isinstance(max_seq_len, tuple)
        assert len(depth) == len(max_seq_len)

        self.stages = len(depth)
        dim = cast_tuple(dim, self.stages)

        assert len(dim) == self.stages

        coarsest_dim, *_, fine_dim = dim

        self.max_seq_len = max_seq_len

        # 初始化起始 token
        self.start_tokens = nn.ParameterList([nn.Parameter(torch.randn(h_dim)) for h_dim, seq_len in zip(dim, max_seq_len)])
        # 初始化位置嵌入
        self.pos_embs = nn.ModuleList([nn.Embedding(seq_len, h_dim) for h_dim, seq_len in zip(dim, max_seq_len)]) if pos_emb else None

        self.token_embs = nn.ModuleList([])

        patch_size = 1
        # 添加 token 嵌入
        self.token_embs.append(nn.Embedding(num_tokens, fine_dim))

        for dim_out, seq_len in zip(reversed(dim[:-1]), reversed(max_seq_len[1:])):
            patch_size *= seq_len

            # 构建 token 嵌入的序列
            self.token_embs.append(nn.Sequential(
                nn.Embedding(num_tokens, fine_dim),
                Rearrange('... r d -> ... (r d)'),
                nn.LayerNorm(patch_size * fine_dim),
                nn.Linear(patch_size * fine_dim, dim_out),
                nn.LayerNorm(dim_out)
            ))

        self.transformers = nn.ModuleList([])
        self.to_next_transformer_projections = nn.ModuleList([])

        for h_dim, next_h_dim, stage_depth, next_seq_len in zip_longest(dim, dim[1:], depth, max_seq_len[1:]):
            # 添加 Transformer 模块
            self.transformers.append(Transformer(
                dim = h_dim,
                layers = stage_depth,
                dim_head = dim_head,
                heads = heads,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                ff_mult = ff_mult,
                rel_pos = rel_pos,
                flash_attn = flash_attn
            ))

            proj = nn.Identity()

            if exists(next_h_dim) and next_h_dim != dim:
                proj = nn.Sequential(
                    Rearrange('b ... d -> b (...) d'),
                    nn.Linear(h_dim, next_h_dim * next_seq_len),
                    Rearrange('b m (n d) -> (b m) n d', n = next_seq_len)
                )

            self.to_next_transformer_projections.append(proj)

        # 线性层，用于输出 logits
        self.to_logits = nn.Linear(fine_dim, num_tokens)
        self.pad_id = pad_id

    # 生成文本
    def generate(self, prime = None, filter_thres = 0.9, temperature = 1., default_batch_size = 1):
        total_seq_len = reduce_mult(self.max_seq_len)
        device = next(self.parameters()).device

        if not exists(prime):
            prime = torch.empty((default_batch_size, 0), dtype = torch.long, device = device)

        seq = prime
        batch = seq.shape[0]

        # 生成文本序列
        for _ in tqdm(range(total_seq_len - seq.shape[-1])):
            logits = self.forward(seq)[:, -1]
            logits = top_k(logits, thres = filter_thres)
            sampled = gumbel_sample(logits, dim = -1, temperature = temperature)
            seq = torch.cat((seq, rearrange(sampled, 'b -> b 1')), dim = -1)

        return seq.reshape(batch, *self.max_seq_len)
    # 定义一个方法，用于处理特殊情况，即从输入为0（仅起始标记）中进行采样
    def forward_empty(self, batch_size):
        # 初始化前一个阶段的标记表示为空
        prev_stage_tokens_repr = None

        # 遍历起始标记、变换器和投影器，分别对应每个阶段
        for stage_start_tokens, transformer, proj in zip(self.start_tokens, self.transformers, self.to_next_transformer_projections):
            # 将起始标记重复扩展到指定批次大小
            tokens = repeat(stage_start_tokens, 'd -> b 1 d', b = batch_size)

            # 如果前一个阶段的标记表示存在，则将其与当前阶段的标记相加
            if exists(prev_stage_tokens_repr):
                tokens = tokens + prev_stage_tokens_repr[..., :tokens.shape[-2], :]

            # 经过变换器处理标记
            tokens = transformer(tokens)
            # 通过投影器得到当前阶段的标记表示
            prev_stage_tokens_repr = proj(tokens)

        # 返回标记转换为对数概率的结果
        return self.to_logits(tokens)
    # 定义前向传播函数，接受输入 ids 和是否返回损失值的标志
    def forward(self, ids, return_loss = False):
        # 获取批量大小
        batch = ids.shape[0]

        # 断言输入 ids 的维度为 2 或者 self.stages + 1
        assert ids.ndim in {2, self.stages + 1}
        # 检查是否为扁平化维度
        flattened_dims = ids.ndim == 2
        ids_orig_ndim = ids.ndim

        # 如果 ids 为空，则调用 forward_empty 函数
        if ids.numel() == 0:
            return self.forward_empty(ids.shape[0])

        # 如果为扁平化维度，则自动填充到最接近深度序列长度的倍数
        if flattened_dims:
            # 获取序列长度
            seq_len = ids.shape[-1]
            # 计算填充值
            multiple_of = reduce_mult(self.max_seq_len[1:])
            padding = remainder_to_mult(seq_len, multiple_of)
            # 对 ids 进行填充
            ids = F.pad(ids, (0, padding), value = self.pad_id)
            ids = ids.reshape(batch, -1, *self.max_seq_len[1:])

        # 获取 ids 的形状和设备信息
        b, *prec_dims, device = *ids.shape, ids.device

        # 检查一些维度

        assert prec_dims[0] <= self.max_seq_len[0], 'the first dimension of your axial autoregressive transformer must be less than the first tuple element of max_seq_len (like any autoregressive transformer)'
        assert tuple(prec_dims[1:]) == tuple(self.max_seq_len[1:]), 'all subsequent dimensions must match exactly'

        # 获取所有层次阶段的 tokens，减少适当的维度并添加绝对位置嵌入

        tokens_at_stages = []
        pos_embs = default(self.pos_embs, (None,))

        for ind, pos_emb, token_emb in zip_longest(range(len(prec_dims)), pos_embs, self.token_embs):
            is_first = ind == 0

            tokens = token_emb(ids)

            if exists(pos_emb):
                positions = pos_emb(torch.arange(tokens.shape[-2], device = device))
                tokens = tokens + positions

            tokens_at_stages.insert(0, tokens)

            if is_first:
                continue

            ids = rearrange(ids, '... m n -> ... (m n)')

        # 上一个层次结构的未像素化表示，从 None 开始

        prev_stage_tokens_repr = None

        # 空间 tokens 是在深度 pos 减少的 tokens + 空间位置

        for stage_start_tokens, stage_tokens, transformer, proj in zip(self.start_tokens, tokens_at_stages, self.transformers, self.to_next_transformer_projections):
            stage_tokens, ps = pack_one(stage_tokens, '* n d')
            stage_start_tokens = repeat(stage_start_tokens, 'f -> b 1 f', b = stage_tokens.shape[0])

            # 连接起始 token

            stage_tokens = torch.cat((
                stage_start_tokens,
                stage_tokens,
            ), dim = -2)

            # 对上一个层次结构的表示求和

            if exists(prev_stage_tokens_repr):
                prev_stage_tokens_repr = F.pad(prev_stage_tokens_repr, (0, 0, 1, 0), value = 0.)
                stage_tokens = stage_tokens + prev_stage_tokens_repr

            attended = transformer(stage_tokens)

            attended = unpack_one(attended, ps, '* n d')

            # 为下一个层次结构投影

            prev_stage_tokens_repr = proj(attended[..., :-1, :])

        # 投影到 logits

        logits = self.to_logits(attended)

        start_tokens = logits[(slice(None), *((0,) * (logits.ndim - 2)), slice(None)]
        start_tokens = rearrange(start_tokens, 'b d -> b 1 d')

        logits = logits[..., 1:, :]

        if not return_loss:

            if flattened_dims:
                logits = rearrange(logits, 'b ... c -> b (...) c')
                logits = logits[:, :seq_len]

            return logits

        logits = rearrange(logits, 'b ... c -> b (...) c')
        logits = torch.cat((start_tokens, logits), dim = -2)

        preds = rearrange(logits, 'b n c -> b c n')
        labels = rearrange(ids, 'b ... -> b (...)')

        loss = F.cross_entropy(
            preds[..., :-1],
            labels,
            ignore_index = self.pad_id
        )

        return loss
```