# `.\lucidrains\linformer\linformer\linformer.py`

```py
import math
import torch
from torch import nn
import torch.nn.functional as F

from linformer.reversible import ReversibleSequence, SequentialSequence

# 辅助函数

# 如果值为 None，则返回默认值
def default(val, default_val):
    return val if val is not None else default_val

# 初始化张量
def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

# 辅助类

# 残差连接
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return x + self.fn(x)

# 预层归一化
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# GELU 激活函数
class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))

# 如果 PyTorch 中有 GELU 函数，则使用，否则使用自定义的 GELU_
GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

# 前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., activation = None, glu = False):
        super().__init__()
        activation = default(activation, GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x

# Linformer 自注意力机制
class LinformerSelfAttention(nn.Module):
    def __init__(self, dim, seq_len, k = 256, heads = 8, dim_head = None, one_kv_head = False, share_kv = False, dropout = 0.):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.k = k

        self.heads = heads

        dim_head = default(dim_head, dim // heads)
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias = False)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias = False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias = False)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)
    # 定义前向传播函数，接受输入 x 和上下文 context，默认参数 kwargs
    def forward(self, x, context = None, **kwargs):
        # 获取输入 x 的形状信息
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k

        # 计算键/值的序列长度
        kv_len = n if context is None else context.shape[1]
        # 断言键/值的序列长度不超过最大序列长度
        assert kv_len <= self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'

        # 将输入 x 转换为查询
        queries = self.to_q(x)

        # 定义函数用于对序列长度进行投影
        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        # 根据是否有上下文选择输入数据
        kv_input = x if context is None else context

        # 将输入数据转换为键和值
        keys = self.to_k(kv_input)
        values = self.to_v(kv_input) if not self.share_kv else keys

        # 定义键和值的投影
        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        # 如果键/值的序列长度小于最大序列长度，则对投影进行切片
        if kv_len < self.seq_len:
            kv_projs = map(lambda t: t[:kv_len], kv_projs)

        # 对键和值沿序列长度维度进行投影
        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # 将查询重塑为 batch, heads, -1 的形状
        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        # 定义函数用于将头部合并到批次中的查询和键/值
        merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        # 注意力计算
        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        # 分割头部
        out = out.transpose(1, 2).reshape(b, n, -1)
        # 返回输出结果
        return self.to_out(out)
class Linformer(nn.Module):
    # 定义 Linformer 类，继承自 nn.Module
    def __init__(self, dim, seq_len, depth, k = 256, heads = 8, dim_head = None, one_kv_head = False, share_kv = False, reversible = False, dropout = 0.):
        # 初始化函数，接受多个参数，包括维度、序列长度、深度等
        super().__init__()
        # 调用父类的初始化函数
        layers = nn.ModuleList([])
        # 创建一个空的模块列表
        for _ in range(depth):
            # 循环 depth 次
            attn = LinformerSelfAttention(dim, seq_len, k = k, heads = heads, dim_head = dim_head, one_kv_head = one_kv_head, share_kv = share_kv, dropout = dropout)
            # 创建 LinformerSelfAttention 注意力机制对象
            ff = FeedForward(dim, dropout = dropout)
            # 创建 FeedForward 前馈神经网络对象

            layers.append(nn.ModuleList([
                PreNorm(dim, attn),
                PreNorm(dim, ff)
            ]))
            # 将 PreNorm 包装的注意力机制和前馈神经网络添加到模块列表中

        execute_type = ReversibleSequence if reversible else SequentialSequence
        # 根据 reversible 参数选择执行类型
        self.net = execute_type(layers)
        # 创建执行类型对象

    def forward(self, x):
        # 前向传播函数
        return self.net(x)
        # 返回执行类型对象对输入 x 的处理结果

class LinformerLM(nn.Module):
    # 定义 LinformerLM 类，继承自 nn.Module
    def __init__(self, num_tokens, dim, seq_len, depth, k = 256, heads = 8, dim_head = None, one_kv_head = False, share_kv = False, reversible = False, dropout = 0.):
        # 初始化函数，接受多个参数，包括标记数量、维度、序列长度、深度等
        super().__init__()
        # 调用父类的初始化函数
        self.token_emb = nn.Embedding(num_tokens, dim)
        # 创建标记嵌入层
        self.pos_emb = nn.Embedding(seq_len, dim)
        # 创建位置嵌入层
        self.linformer = Linformer(dim, seq_len, depth, k = k, heads = heads, dim_head = dim_head,
                one_kv_head = one_kv_head, share_kv = share_kv, reversible = reversible, dropout = dropout)
        # 创建 Linformer 对象
        self.to_logits = nn.Linear(dim, num_tokens)
        # 创建线性层，用于输出标记

    def forward(self, x):
        # 前向传播函数
        x = self.token_emb(x)
        # 对输入 x 进行标记嵌入
        x = self.pos_emb(torch.arange(x.shape[1], device=x.device)) + x
        # 对输入 x 进行位置嵌入
        x = self.linformer(x)
        # 使用 Linformer 处理输入 x
        out = self.to_logits(x)
        # 将���理结果传递给线性层
        return out
        # 返回输出结果
```