# `.\lucidrains\RQ-Transformer\rq_transformer\rq_transformer.py`

```
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops_exts import rearrange_with_anon_dims
from einops import rearrange, reduce, repeat

# helpers

# 检查值是否存在
def exists(val):
    return val is not None

# 返回值或默认值
def default(val, d):
    return val if exists(val) else d

# 计算余数到最接近的倍数
def remainder_to_mult(num, mult):
    return (mult - num % mult) % mult

# 计算对数，避免值过小
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

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

# helper classes

# 前馈神经网络
def FeedForward(*, dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
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
        dropout = 0.
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        h, device = self.heads, x.device

        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        i, j = sim.shape[-2:]
        mask_value = -torch.finfo(sim.dtype).max
        mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
        sim = sim.masked_fill(mask, mask_value)

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# Transformer 模块
class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        layers,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

# 主类

class RQTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        max_spatial_seq_len,
        depth_seq_len,
        spatial_layers,
        depth_layers,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        pad_id = 0
    ):
        # 调用父类的构造函数
        super().__init__()
        # 初始化模型的维度
        self.dim = dim
        # 初始化空间序列的最大长度
        self.max_spatial_seq_len = max_spatial_seq_len
        # 初始化深度序列的长度
        self.depth_seq_len = depth_seq_len

        # 创建一个词嵌入层，用于将输入的标记转换为向量表示
        self.token_emb = nn.Embedding(num_tokens, dim)
        # 初始化空间序列的起始标记
        self.spatial_start_token = nn.Parameter(torch.randn(dim))

        # 创建一个空间位置编码层
        self.spatial_pos_emb = nn.Embedding(max_spatial_seq_len + 1, dim) # 考虑到一个边界情况
        # 创建一个深度位置编码层
        self.depth_pos_emb = nn.Embedding(depth_seq_len, dim)

        # 创建一个空间变换器，用于处理空间序列的变换
        self.spatial_transformer = Transformer(
            dim = dim,
            layers = spatial_layers,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_mult = ff_mult
        )

        # 创建一个深度变换器，用于处理深度序列的变换
        self.depth_transformer = Transformer(
            dim = dim,
            layers = depth_layers,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_mult = ff_mult
        )

        # 创建一个线性层，用于将模型输出转换为标记的概率分布
        self.to_logits = nn.Linear(dim, num_tokens)
        # 初始化填充标记的ID
        self.pad_id = pad_id

    def generate(self, prime = None, filter_thres = 0.9, temperature = 1., default_batch_size = 1):
        # 计算总的序列长度
        total_seq_len = self.depth_seq_len * self.max_spatial_seq_len
        # 获取模型所在的设备
        device = next(self.parameters()).device

        # 如果没有给定初始输入，则创建一个空的张量作为初始输入
        if not exists(prime):
            prime = torch.empty((default_batch_size, 0), dtype = torch.long, device = device)

        seq = prime

        # 生成序列
        for _ in range(total_seq_len - seq.shape[-1]):
            # 获取模型的预测结果
            logits = self.forward(seq)[:, -1]
            # 通过阈值筛选保留概率较高的标记
            logits = top_k(logits, thres = filter_thres)
            # 通过Gumbel采样获取下一个标记
            sampled = gumbel_sample(logits, dim = -1, temperature = temperature)
            # 将新生成的标记添加到序列中
            seq = torch.cat((seq, rearrange(sampled, 'b -> b 1')), dim = -1)

        # 重新排列生成的序列
        return rearrange(seq, 'b (s d) -> b s d', d = self.depth_seq_len)

    def forward_empty(self, batch_size):
        # 处理特殊情况，当从输入中只采样到0（仅起始标记）时

        # 重复空间起始标记，以匹配指定的批量大小
        spatial_tokens = repeat(self.spatial_start_token, 'd -> b 1 d', b = batch_size)
        # 经过空间变换器处理
        depth_tokens = self.spatial_transformer(spatial_tokens)
        # 经过深度变换器处理
        depth_tokens = self.depth_transformer(depth_tokens)
        # 将处理后的深度标记转换为模型输出
        return self.to_logits(depth_tokens)
    # 定义前向传播函数，接受输入 ids 和是否返回损失值的标志
    def forward(self, ids, return_loss = False):
        # 断言输入 ids 的维度为 2 或 3
        assert ids.ndim in {2, 3}
        # 检查是否为扁平化维度
        flattened_dim = ids.ndim == 2
        # 保存原始 ids 的维度
        ids_orig_ndim = ids.ndim

        # 如果 ids 中元素数量为 0，则调用 forward_empty 函数处理
        if ids.numel() == 0:
            return self.forward_empty(ids.shape[0])

        # 如果是扁平化维度
        if flattened_dim:
            # 允许 ids 的形状为 (batch, seq)，自动填充到最接近深度序列长度的倍数
            seq_len = ids.shape[-1]
            padding = remainder_to_mult(seq_len, self.depth_seq_len)
            ids = F.pad(ids, (0, padding), value = self.pad_id)
            ids = rearrange(ids, 'b (s d) -> b s d', d = self.depth_seq_len)
        else:
            seq_len = ids.shape[1] * ids.shape[2]

        # 获取 ids 的形状、空间维度、深度维度、设备信息
        b, space, depth, device = *ids.shape, ids.device
        # 断言空间维度小于等于最大空间序列长度加一
        assert space <= (self.max_spatial_seq_len + 1), 'spatial dimension is greater than the max_spatial_seq_len set'
        # 断言深度维度等于深度序列长度
        assert depth == self.depth_seq_len, 'depth dimension must be equal to depth_seq_len'

        # 获取 token embeddings
        tokens = self.token_emb(ids)

        # 获取空间位置编码和深度位置编码
        spatial_pos = self.spatial_pos_emb(torch.arange(space, device = device))
        depth_pos = self.depth_pos_emb(torch.arange(depth, device = device))

        # 将 token embeddings 和深度位置编码相加
        tokens_with_depth_pos = tokens + depth_pos

        # 计算空间 tokens
        spatial_tokens = reduce(tokens_with_depth_pos, 'b s d f -> b s f', 'sum') + spatial_pos

        # 在空间 tokens 前添加起始 token
        spatial_tokens = torch.cat((
            repeat(self.spatial_start_token, 'f -> b 1 f', b = b),
            spatial_tokens
        ), dim = -2)        

        # 使用空间 transformer 处理空间 tokens
        spatial_tokens = self.spatial_transformer(spatial_tokens)

        # 重新排列空间 tokens 的维度
        spatial_tokens = rearrange(spatial_tokens, 'b s f -> b s 1 f')

        # 将空间 tokens 变为深度维度的起始 tokens
        tokens_with_depth_pos = F.pad(tokens_with_depth_pos, (0, 0, 0, 0, 0, 1), value = 0.)

        # 拼��深度 tokens
        depth_tokens = torch.cat((spatial_tokens, tokens_with_depth_pos), dim = -2)

        # 重新排列深度 tokens 的维度
        depth_tokens = rearrange(depth_tokens, '... n d -> (...) n d')

        # 使用深度 transformer 处理深度 tokens
        depth_tokens = self.depth_transformer(depth_tokens)

        # 重新排列深度 tokens 的维度
        depth_tokens = rearrange(depth_tokens, '(b s) d f -> b s d f', b = b)

        # 获取 logits
        logits = self.to_logits(depth_tokens)
        logits = rearrange(logits, 'b ... f -> b (...) f')
        logits = logits[:, :(seq_len + 1)]

        # 如果不需要返回损失值
        if not return_loss:
            logits = logits[:, 1:]

            # 如果是扁平化维度，则返回重新排列后的 logits
            if flattened_dim:
                return rearrange(logits, 'b ... n -> b (...) n')

            return logits

        # 如果需要返回损失值
        logits = logits[:, :-1]
        
        # 重新排列 logits 和 ids 的维度
        preds = rearrange(logits, 'b ... c -> b c (...)')
        labels = rearrange(ids, 'b s d -> b (s d)')

        # 计算交叉熵损失
        loss = F.cross_entropy(preds, labels, ignore_index = self.pad_id)
        return loss
```