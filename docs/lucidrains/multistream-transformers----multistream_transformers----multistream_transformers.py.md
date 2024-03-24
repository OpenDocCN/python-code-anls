# `.\lucidrains\multistream-transformers\multistream_transformers\multistream_transformers.py`

```py
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat, reduce

from einops.layers.torch import Rearrange

# 辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 如果变量存在则返回其值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 返回给定数据类型的最小负值
def max_neg_value(t):
    return -torch.finfo(t.dtype).max

# 对所有张量进行重排列
def rearrange_all(tensors, *args, **kwargs):
    return map(lambda t: rearrange(t, *args, **kwargs), tensors)

# 前馈网络

# 分组层归一化
class GroupLayerNorm(nn.Module):
    def __init__(self, dim, groups = 1, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.groups = groups
        self.g = nn.Parameter(torch.ones(1, groups, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, groups, dim, 1))

    def forward(self, x):
        x = rearrange(x, 'b (g d) n -> b g d n', g = self.groups)
        std = torch.var(x, dim = 2, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 2, keepdim = True)
        out = (x - mean) / (std + self.eps) * self.g + self.b
        return rearrange(out, 'b g d n -> b (g d) n')

# 预归一化
class PreNorm(nn.Module):
    def __init__(
        self,
        dim,
        fn,
        groups = 1
    ):
        super().__init__()
        self.norm = GroupLayerNorm(dim, groups = groups)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# 前馈网络
class FeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        mult = 4,
        groups = 1
    ):
        super().__init__()
        input_dim = dim * groups
        hidden_dim = dim * mult * groups

        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1, groups = groups),
            nn.GELU(),
            nn.Conv1d(hidden_dim, input_dim, 1, groups = groups)
        )

    def forward(self, x):
        return self.net(x)

# 注意力机制
class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        causal = False,
        groups = 1
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.groups = groups
        self.heads = heads
        self.causal = causal
        input_dim = dim * groups
        inner_dim = dim_head * heads * groups

        self.to_q = nn.Conv1d(input_dim, inner_dim, 1, bias = False)
        self.to_kv = nn.Conv1d(input_dim, inner_dim * 2, 1, bias = False)
        self.to_out = nn.Conv1d(inner_dim, input_dim, 1)

    def forward(self, x, mask = None, context = None):
        n, device, h, g, causal = x.shape[2], x.device, self.heads, self.groups, self.causal
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = 1))
        q, k, v = rearrange_all((q, k, v), 'b (g h d) n -> (b g h) n d', g = g, h = h)

        q = q * self.scale

        sim = einsum('b i d, b j d -> b i j', q, k)

        if exists(mask):
            mask = repeat(mask, 'b n -> (b g h) n', h = h, g = g)
            mask = rearrange(mask, 'b n -> b n ()') * rearrange(mask, 'b n -> b () n')
            mask_value = max_neg_value(sim)
            sim = sim.masked_fill(~mask, mask_value)

        if causal:
            causal_mask = torch.ones((n, n), device = device).triu(1).bool()
            mask_value = max_neg_value(sim)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b g h) n d -> b (g h d) n', h = h, g = g)
        return self.to_out(out)

# Transformer 块
class TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        causal = False,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        groups = 1
    ):  
        # 调用父类的构造函数
        super().__init__()
        # 初始化注意力层，包括预层归一化、注意力机制和前馈神经网络
        self.attn = PreNorm(dim, Attention(dim = dim, dim_head = dim_head, heads = heads, causal = causal, groups = groups), groups = groups)
        # 初始化前馈神经网络层，包括预层归一化和前馈神经网络
        self.ff = PreNorm(dim, FeedForward(dim = dim, mult = ff_mult, groups = groups), groups = groups)

    def forward(self, x, mask = None):
        # 使用注意力层处理输入数据，并将结果与输入数据相加
        x = self.attn(x, mask = mask) + x
        # 使用前馈神经网络层处理上一步的结果，并将结果与上一步的结果相加
        x = self.ff(x) + x
        # 返回处理后的结果
        return x
# 主类定义

class MultistreamTransformer(nn.Module):
    # 初始化函数
    def __init__(
        self,
        *,
        dim,  # 维度
        depth,  # 深度
        num_tokens,  # 令牌数量
        max_seq_len,  # 最大序列长度
        causal = False,  # 是否因果
        dim_head = 64,  # 头维度
        heads = 8,  # 头数
        ff_mult = 4,  # FeedForward倍数
        num_streams = 1  # 流数量
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.num_streams = num_streams
        self.token_emb = nn.Embedding(num_tokens, dim)  # 令牌嵌入层
        self.pos_emb = nn.Embedding(max_seq_len, dim)  # 位置嵌入层

        self.layers = nn.ModuleList([])
        self.pre_transformer_block = TransformerBlock(dim = dim, causal = causal, dim_head = dim_head, heads = heads)  # 前置Transformer块

        for _ in range(depth):
            self.layers.append(TransformerBlock(dim = dim, causal = causal, dim_head = dim_head, heads = heads, groups = num_streams))  # 添加指定数量的Transformer块

        if num_streams > 1:
            self.query = nn.Parameter(torch.randn(dim))  # 查询参数
            self.attn_pool = Attention(dim = dim, dim_head = dim_head, heads = heads)  # 注意力池化层

        self.post_transformer_block = TransformerBlock(dim = dim, causal = causal, dim_head = dim_head, heads = heads,)  # 后置Transformer块

        self.to_logits = nn.Sequential(
            Rearrange('b d n -> b n d'),  # 重排维度
            nn.LayerNorm(dim),  # 层归一化
            nn.Linear(dim, num_tokens)  # 线性层
        )

    # 前向传播函数
    def forward(self, x, mask = None):
        b, n, d, device, is_multistream = *x.shape, self.dim, x.device, (self.num_streams > 1)  # 获取输入张量的形状和设备信息，判断是否为多流模式
        x = self.token_emb(x)  # 令牌嵌入

        pos_emb = self.pos_emb(torch.arange(n, device = device))  # 位置嵌入
        pos_emb = rearrange(pos_emb, 'n d -> () n d')  # 重排维度

        x = x + pos_emb  # 加上位置嵌入
        x = rearrange(x, 'b n d -> b d n')  # 重排维度

        x = self.pre_transformer_block(x, mask = mask)  # 前置Transformer块处理输入
        layers = [x]  # 存储每一层的输出

        if is_multistream:
            x = repeat(x, 'b d n -> b (s d) n', s = self.num_streams)  # 复制张量以支持多流模式

        for block in self.layers:
            x = block(x, mask = mask)  # 处理每个Transformer块
            layers.append(x)  # 存��每一层的输出

        if is_multistream:
            layers = list(map(lambda t: rearrange(t, 'b (s d) n -> (b n) d s', d = d), layers))  # 重排维度以支持多流模式
            layer_tokens = torch.cat(layers, dim = -1)  # 拼接多个流的输出

            query = repeat(self.query, 'd -> b d ()', b = layer_tokens.shape[0])  # 复制查询参数
            x = self.attn_pool(query, context = layer_tokens)  # 使用注意力池化层
            x = rearrange(x, '(b n) d () -> b d n', n = n)  # 重排维度

        x = self.post_transformer_block(x, mask = mask)  # 后置Transformer块处理输出
        return self.to_logits(x)  # 返回预测结果
```