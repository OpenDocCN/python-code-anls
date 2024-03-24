# `.\lucidrains\perceiver-ar-pytorch\perceiver_ar_pytorch\perceiver_ar_pytorch.py`

```
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat

# helper functions

# 检查变量是否存在的辅助函数
def exists(val):
    return val is not None

# feedforward

# 定义前馈神经网络层
def FeedForward(dim, mult = 4, dropout = 0.):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),  # 对输入进行 Layer Normalization
        nn.Linear(dim, hidden_dim, bias = False),  # 线性变换
        nn.GELU(),  # GELU 激活函数
        nn.Dropout(dropout),  # Dropout 正则化
        nn.Linear(hidden_dim, dim, bias = False)  # 线性变换
    )

# rotary positional embedding
# https://arxiv.org/abs/2104.09864

# 旋转位置嵌入类
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device = device, dtype = self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim = -1)


# 旋转半个张量
def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)


# 应用旋转位置嵌入
def apply_rotary_pos_emb(pos, t):
    seq_len, rotate_dim = t.shape[-2], pos.shape[-1]
    pos = pos[..., -seq_len:, :]
    t, t_pass = t[..., :rotate_dim], t[..., rotate_dim:]
    t = (t * pos.cos()) + (rotate_half(t) * pos.sin())
    return torch.cat((t, t_pass), dim = -1)

# attention

# 因果注意力机制类
class CausalAttention(nn.Module):
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
        inner_dim = heads * dim_head

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, rotary_pos_emb = None):
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        q = q * self.scale

        if exists(rotary_pos_emb):
            q = apply_rotary_pos_emb(rotary_pos_emb, q)
            k = apply_rotary_pos_emb(rotary_pos_emb, k)

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), device = x.device, dtype = torch.bool).triu(j - i + 1)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# 因果前缀注意力机制类
class CausalPrefixAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        max_heads_process = 2,
        dropout = 0.,
        cross_attn_dropout = 0.
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.max_heads_process = max_heads_process

        inner_dim = heads * dim_head

        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.cross_attn_dropout = cross_attn_dropout # they drop out a percentage of the prefix during training, shown to help prevent overfitting

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)
    # 定义前向传播函数，接受输入 x、上下文 context、上下文掩码 context_mask 和旋转位置嵌入 rotary_pos_emb
    def forward(self, x, context, context_mask = None, rotary_pos_emb = None):
        # 获取输入 x 的批量大小、上下文长度和设备信息
        batch, context_len, device = x.shape[0], context.shape[-2], x.device

        # 复制旋转位置嵌入作为查询和键的旋转位置嵌入
        q_rotary_pos_emb = rotary_pos_emb
        k_rotary_pos_emb = rotary_pos_emb

        # 处理交叉注意力的 dropout

        if self.training and self.cross_attn_dropout > 0.:
            # 生成随机数用于 dropout
            rand = torch.zeros((batch, context_len), device = device).uniform_()
            keep_context_len = context_len - int(context_len * self.cross_attn_dropout)
            keep_indices = rand.topk(keep_context_len, dim = -1).indices
            keep_mask = torch.zeros_like(rand).scatter_(1, keep_indices, 1).bool()

            # 根据掩码保留一部分上下文信息
            context = rearrange(context[keep_mask], '(b n) d -> b n d', b = batch)

            if exists(context_mask):
                context_mask = rearrange(context_mask[keep_mask], '(b n) -> b n', b = batch)

            # 对键的旋转位置嵌入进行操作
            k_rotary_pos_emb = repeat(k_rotary_pos_emb, '... -> b ...', b = batch)
            k_rotary_pos_emb_context, k_rotary_pos_emb_seq = k_rotary_pos_emb[:, :context_len], k_rotary_pos_emb[:, context_len:]
            k_rotary_pos_emb_context = rearrange(k_rotary_pos_emb_context[keep_mask], '(b n) d -> b n d', b = batch)

            k_rotary_pos_emb = torch.cat((k_rotary_pos_emb_context, k_rotary_pos_emb_seq), dim = 1)
            k_rotary_pos_emb = rearrange(k_rotary_pos_emb, 'b n d -> b 1 n d')

        # 归一化处理
        x = self.norm(x)
        context = self.context_norm(context)

        # 获取查询、键、值
        q = self.to_q(x)

        k_input, v_input = self.to_kv(x).chunk(2, dim = -1)
        k_context, v_context = self.to_kv(context).chunk(2, dim = -1)

        k = torch.cat((k_context, k_input), dim = 1)
        v = torch.cat((v_context, v_input), dim = 1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        q = q * self.scale

        # 使用旋转位置嵌入旋转查询和键
        if exists(rotary_pos_emb):
            q = apply_rotary_pos_emb(q_rotary_pos_emb, q)
            k = apply_rotary_pos_emb(k_rotary_pos_emb, k)

        # 处理掩码
        i, j = q.shape[-2], k.shape[-2]
        mask_value = -torch.finfo(q.dtype).max

        if exists(context_mask):
            mask_len = context_mask.shape[-1]
            context_mask = F.pad(context_mask, (0, max(j - mask_len, 0)), value = True)
            context_mask = rearrange(context_mask, 'b j -> b 1 1 j')

        causal_mask = torch.ones((i, j), device = x.device, dtype = torch.bool).triu(j - i + 1)

        # 按头部分块处理
        out = []

        max_heads = self.max_heads_process

        for q_chunk, k_chunk, v_chunk in zip(q.split(max_heads, dim = 1), k.split(max_heads, dim = 1), v.split(max_heads, dim = 1):
            sim = einsum('b h i d, b h j d -> b h i j', q_chunk, k_chunk)

            if exists(context_mask):
                sim = sim.masked_fill(~context_mask, mask_value)

            sim = sim.masked_fill(causal_mask, mask_value)

            attn = sim.softmax(dim = -1)
            attn = self.dropout(attn)

            out_chunk = einsum('b h i j, b h j d -> b h i d', attn, v_chunk)
            out.append(out_chunk)

        # 拼接所有头部
        out = torch.cat(out, dim = 1)

        # 合并头部并与线性层结合
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)
class PerceiverAR(nn.Module):
    # 定义 PerceiverAR 类，继承自 nn.Module
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        max_seq_len,
        cross_attn_seq_len,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        cross_attn_dropout = 0.,
        ff_mult = 4,
        perceive_depth = 1,
        perceive_max_heads_process = 2 # processes the heads in the perceiver layer in chunks to lower peak memory, in the case the prefix is really long
    ):
        # 初始化函数，接受多个参数
        super().__init__()
        # 调用父类的初始化函数
        assert max_seq_len > cross_attn_seq_len, 'max_seq_len must be greater than cross_attn_seq_len, the length of the sequence for which to cross attend to "perceiver" style'
        # 断言，确保 max_seq_len 大于 cross_attn_seq_len
        self.max_seq_len = max_seq_len
        self.cross_attn_seq_len = cross_attn_seq_len

        self.token_emb = nn.Embedding(num_tokens, dim)
        # 创建 token embedding 层
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        # 创建位置 embedding 层

        self.rotary_pos_emb = RotaryEmbedding(dim = max(32, dim_head // 2))
        # 创建旋转位置 embedding 层

        self.perceive_layers  = nn.ModuleList([])
        # 创建感知层的 ModuleList

        for _ in range(perceive_depth):
            # 循环感知深度次数
            self.perceive_layers.append(nn.ModuleList([
                CausalPrefixAttention(dim = dim, dim_head = dim_head, heads = heads, max_heads_process = perceive_max_heads_process, dropout = dropout, cross_attn_dropout = cross_attn_dropout),
                FeedForward(dim, mult = ff_mult, dropout = dropout)
            ]))
            # 将 CausalPrefixAttention 和 FeedForward 添加到感知层中

        self.layers = nn.ModuleList([])
        # 创建层的 ModuleList
        for _ in range(depth):
            # 循环深度次数
            self.layers.append(nn.ModuleList([
                CausalAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim, mult = ff_mult, dropout = dropout),
            ]))
            # 将 CausalAttention 和 FeedForward 添加到层中

        self.to_logits = nn.Linear(dim, num_tokens, bias = False)
        # 创建线性层，用于输出 logits

    def forward(
        self,
        x,
        prefix_mask = None,
        labels = None
    ):
        # 前向传播函数，接受输入 x，前缀掩码和标签
        seq_len, device = x.shape[1], x.device
        # 获取序列长度和设备信息
        assert self.cross_attn_seq_len < seq_len <= self.max_seq_len
        # 断言，确保交叉注意力序列长度小于序列长度且小于等于最大序列长度

        x = self.token_emb(x)
        # 对输入进行 token embedding
        x = x + self.pos_emb(torch.arange(seq_len, device = device))
        # 添加位置 embedding

        # rotary positional embedding

        rotary_pos_emb = self.rotary_pos_emb(seq_len, device = device)
        # 获取旋转位置 embedding

        # divide into prefix to cross attend to and sequence to self attend to

        prefix, x = x[:, :self.cross_attn_seq_len], x[:, self.cross_attn_seq_len:]
        # 将输入分为前缀和序列部分

        # initial perceiver attention and feedforward (one cross attention)

        for cross_attn, ff in self.perceive_layers:
            # 遍历感知层
            x = cross_attn(x, prefix, context_mask = prefix_mask, rotary_pos_emb = rotary_pos_emb) + x
            # 进行交叉注意力操作
            x = ff(x) + x
            # 进行前馈操作

        # layers

        for attn, ff in self.layers:
            # 遍历层
            x = attn(x, rotary_pos_emb = rotary_pos_emb) + x
            # 进行自注意力操作
            x = ff(x) + x
            # 进行前馈操作

        # to logits

        logits = self.to_logits(x)
        # 计算 logits

        # take care of cross entropy loss if labels are provided

        if not exists(labels):
            return logits
        # 如果提供了标签，则处理交叉熵损失

        labels = labels[:, self.cross_attn_seq_len:]
        # 获取标签的序列部分
        return F.cross_entropy(rearrange(logits, 'b n c -> b c n'), labels, ignore_index = 0)
        # 计算交叉熵损失
```