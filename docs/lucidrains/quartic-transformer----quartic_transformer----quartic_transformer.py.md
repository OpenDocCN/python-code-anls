# `.\lucidrains\quartic-transformer\quartic_transformer\quartic_transformer.py`

```py
# 导入 torch 库
import torch
# 从 torch 库中导入 nn, einsum 模块
from torch import nn, einsum
# 从 torch.nn 模块中导入 Module, ModuleList 类
from torch.nn import Module, ModuleList

# 从 einops 库中导入 rearrange, repeat, pack, unpack 函数
from einops import rearrange, repeat, pack, unpack
# 从 einops.layers.torch 模块中导入 Rearrange 类
from einops.layers.torch import Rearrange

# 导入 einx 库
import einx
# 从 einx.nn.torch 模块中导入 einn 模块

# 导入 colt5_attention 模块中的 topk 函数

# 导入 taylor_series_linear_attention 模块中的 TaylorSeriesLinearAttn 类

# 从 x_transformers.x_transformers 模块中导入 DynamicPositionBias 类

# 定义辅助函数

# 判断变量是否存在的函数
def exists(v):
    return v is not None

# 返回默认值的函数
def default(v, d):
    return v if exists(v) else d

# 将张量打包成指定模式的函数
def pack_one(t, pattern):
    return pack([t], pattern)

# 将打包的张量解包成指定模式的函数
def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# 定义注意力机制类

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_edges = None,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        causal = False,
        incorporate_edges = True
    ):
        super().__init__()
        dim_edges = default(dim_edges, dim)
        dim_inner = dim_head * heads

        # 定义 QKV 线性层和重排操作
        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias = False),
            Rearrange('b n (qkv h d) -> qkv b h n d', h = heads, qkv = 3)
        )

        # 定义门控线性层和 Sigmoid 激活函数
        self.to_gates = nn.Sequential(
            nn.Linear(dim, heads),
            Rearrange('b n h -> b h n 1'),
            nn.Sigmoid()
        )

        # 定义 RMSNorm 层
        self.rmsnorm = einn.Norm('b... [d]', mean = False, bias = False)

        self.scale = dim_head ** 0.5
        self.causal = causal
        self.dropout = nn.Dropout(dropout)

        self.edges_to_attn_bias = None

        if incorporate_edges:
            # 定义边到注意力偏置的线性层和重排操作
            self.edges_to_attn_bias = nn.Sequential(
                einn.Norm('b... [d]', mean = False, bias = False),
                nn.Linear(dim_edges, heads),
                Rearrange('b i j h -> b h i j')
            )

        # 定义预处理头部的卷积层
        self.pre_talking_heads = nn.Conv2d(heads, heads, 1, bias = False)

        self.to_edges_out = None

        if incorporate_edges:
            # 定义输出到边的线��层和重排操作
            self.to_edges_out = nn.Sequential(
                nn.Conv2d(heads, dim_edges, 1, bias = False),
                Rearrange('b d i j -> b i j d')
            )

        # 定义输出层
        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(dim_inner, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        mask = None,
        edges = None
    ):
        x = self.rmsnorm(x)

        q, k, v = self.to_qkv(x)

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        mask_value = -torch.finfo(sim.dtype).max

        if exists(edges) and exists(self.edges_to_attn_bias):
            attn_bias = self.edges_to_attn_bias(edges)
            sim = sim + attn_bias

        sim = self.pre_talking_heads(sim)

        if exists(mask):
            sim = einx.where('b j, b ... j, ', mask, sim, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = einx.softmax('b h i [j]', sim)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = out * self.to_gates(x)
        out = self.to_out(out)

        edges_out = None
        if exists(self.to_edges_out):
            edges_out = self.to_edges_out(attn)

        if not exists(edges_out):
            return out

        return out, edges_out

# 定义前馈神经网络类

def FeedForward(dim, mult = 4, dropout = 0.):
    dim_inner = int(dim * mult)
    return nn.Sequential(
        einn.Norm('b... [d]', mean = False, bias = False),
        nn.Linear(dim, dim_inner, bias = False),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim_inner, dim, bias = False)
    )

# 定义边嵌入类

class EdgeEmbed(Module):
    # 初始化函数，接受维度参数和可选的边缘维度参数
    def __init__(self, dim, dim_edges = None):
        # 调用父类的初始化函数
        super().__init__()
        # 如果没有提供边缘维度参数，则使用默认值为维度参数
        dim_edges = default(dim_edges, dim)
        # 创建一个线性层，将输入维度映射到边缘维度，不使用偏置
        self.to_rows = nn.Linear(dim, dim_edges, bias = False)
        # 创建另一个线性层，将输入维度映射到边缘维度，不使用偏置
        self.to_cols = nn.Linear(dim, dim_edges, bias = False)

        # 创建一个序列模块，包含一个线性层和一个 LayerNorm 层，用于处理边缘维度数据
        self.to_edges = nn.Sequential(
            nn.Linear(dim_edges, dim_edges, bias = False),
            nn.LayerNorm(dim_edges)
        )

    # 前向传播函数，接受输入张量 x
    def forward(self, x):
        # 将输入张量 x 映射到行维度
        rows = self.to_rows(x)
        # 将输入张量 x 映射到列维度
        cols = self.to_cols(x)
        # 对行和列的外积求和，得到四维张量
        outer_sum = einx.add('b i d, b j d -> b i j d', rows, cols)
        # 将外积求和结果传入边缘处理模块，返回处理后的结果
        return self.to_edges(outer_sum)
# 定义 AxialLinearAttention 类，用于实现轴向线性注意力机制
class AxialLinearAttention(Module):
    def __init__(
        self,
        dim,
        diagonal_attn = True,
        **attn_kwargs
    ):
        super().__init__()

        # 初始化行注意力机制
        self.row_attn = TaylorSeriesLinearAttn(dim = dim, gate_value_heads = True, prenorm = True, **attn_kwargs)
        # 初始化列注意力机制
        self.col_attn = TaylorSeriesLinearAttn(dim = dim, gate_value_heads = True, prenorm = True, **attn_kwargs)

        # 如果设置了对角线注意力机制，则初始化对角线注意力机制
        self.diagonal_attn = Attention(dim = dim, incorporate_edges = False, **attn_kwargs) if diagonal_attn else None

    # 前向传播函数
    def forward(
        self,
        x,
        mask = None
    ):
        # 获取输入张量 x 的形状信息
        b, n, device = *x.shape[:2], x.device

        # 重排输入张量 x 的维度
        x = rearrange(x, 'b i j d -> (b i) j d')

        # 对行进行注意力计算并更新 x
        x = self.row_attn(x, mask = mask) + x

        # 重排 x 的维度
        x = rearrange(x, '(b i) j d -> (b j) i d', b = b)

        # 对列进行注意力计算并更新 x
        x = self.col_attn(x, mask = mask) + x

        # 重排 x 的维度
        x = rearrange(x, '(b j) i d -> b i j d', b = b)

        # 如果没有对角线注意力机制，则直接返回 x
        if not exists(self.diagonal_attn):
            return x

        # 创建对角线掩码
        diagonal_mask = torch.eye(n, dtype = torch.bool, device = device)
        diagonal_mask = rearrange(diagonal_mask, 'i j -> 1 i j')

        # 从 x 中提取对角线元素
        x = rearrange(x[diagonal_mask], '(b n) d -> b n d', b = b)

        # 对对角线元素进行注意力计算并更新 x
        x = self.diagonal_attn(x) + x

        # 重新排列对角线掩码的维度
        diagonal_mask = rearrange(diagonal_mask, '... -> ... 1')
        # 使用对角线掩码更新 x
        x = x.masked_scatter(diagonal_mask, x)
        return x

# 定义 QuarticTransformer 类，用于实现四次方变换器
class QuarticTransformer(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        dim_edges = None,
        dim_head = 64,
        heads = 8,
        causal = False,
        linear_dim_head = 16,
        linear_heads = 16,
        ff_mult = 4,
        dropout = 0.,
        max_seq_len = 2048,
        ablate_edges = False,
        edges_diagonal_attn = True
    ):
        super().__init__()
        dim_edges = default(dim_edges, dim)

        # 初始化类的属性
        self.ablate_edges = ablate_edges
        self.max_seq_len = max_seq_len

        # 初始化 token embedding 和 position embedding
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        # 初始化动态相对位置偏置
        self.dynamic_rel_pos_bias = DynamicPositionBias(dim, depth = 2, heads = dim_edges)

        # 初始化边缘嵌入
        self.to_edge_emb = EdgeEmbed(dim, dim_edges)

        # 初始化层列表
        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(ModuleList([
                ModuleList([
                    Attention(dim = dim, dim_edges = dim_edges, dim_head = dim_head, heads = heads, dropout = dropout, causal = causal),
                    FeedForward(dim = dim, mult = ff_mult, dropout = dropout)
                ]),
                ModuleList([
                    AxialLinearAttention(dim = dim_edges, dim_head = linear_dim_head, heads = linear_heads, causal = causal, diagonal_attn = edges_diagonal_attn),
                    FeedForward(dim = dim_edges, mult = ff_mult)
                ])
            ]))

        # 初始化输出层
        self.to_logits = nn.Sequential(
            einn.Norm('b... [d]', mean = False, bias = False),
            nn.Linear(dim, num_tokens, bias = False)
        )

    # 前向传播函数
    def forward(
        self,
        x,
        mask = None
        ):
        # 获取输入张量的序列长度和设备信息
        seq_len, device = x.shape[-1], x.device
        # 断言序列长度不超过最大序列长度
        assert seq_len <= self.max_seq_len

        # 对输入张量进行 token embedding
        x = self.token_emb(x)

        # 添加位置编码
        x = x + self.pos_emb(torch.arange(seq_len, device=device))
        # 获取边的嵌入表示
        edges = self.to_edge_emb(x)

        # 计算动态相对位置偏置
        edges_rel_pos = self.dynamic_rel_pos_bias(seq_len, seq_len)
        # 将边的嵌入表示与动态相对位置偏置相加
        edges = einx.add('b i j d, d i j -> b i j d', edges, edges_rel_pos)

        # 初始化边的掩码
        edges_mask = None
        # 如果掩码存在，则更新边的掩码
        if exists(mask):
            edges_mask = einx.logical_and('b i, b j -> b (i j)', mask, mask)

        # 遍历每个层
        for (attn, ff), (edges_linear_attn, edges_ff,) in self.layers:

            # 使用注意力机制和前馈网络处理节点和边
            nodes_out, edges_out = attn(x, mask=mask, edges=edges if not self.ablate_edges else None)

            # 更新节点表示
            x = x + nodes_out
            x = ff(x) + x

            # 如果需要剔除边信息，则跳过
            if self.ablate_edges:
                continue

            # 更新边的表示
            edges = edges + edges_out

            # 线性变换边信息
            edges = edges_linear_attn(edges, mask=mask) + edges

            # 使用前馈网络处理边信息
            edges = edges_ff(edges) + edges

        # 返回最终的输出结果
        return self.to_logits(x)
```