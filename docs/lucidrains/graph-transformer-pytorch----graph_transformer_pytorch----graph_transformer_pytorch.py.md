# `.\lucidrains\graph-transformer-pytorch\graph_transformer_pytorch\graph_transformer_pytorch.py`

```py
# 导入 torch 库
import torch
# 从 torch 库中导入 nn, einsum
from torch import nn, einsum
# 从 einops 库中导入 rearrange, repeat

from einops import rearrange, repeat

# 从 rotary_embedding_torch 库中导入 RotaryEmbedding, apply_rotary_emb

# helpers

# 判断变量是否存在的函数
def exists(val):
    return val is not None

# 如果变量存在则返回该变量，否则返回默认值的函数
def default(val, d):
    return val if exists(val) else d

# 定义 nn.ModuleList 类别为 List
List = nn.ModuleList

# normalizations

# 预处理层，包含 LayerNorm 和传入的函数
class PreNorm(nn.Module):
    def __init__(
        self,
        dim,
        fn
    ):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args,**kwargs)

# gated residual

# 残差连接层
class Residual(nn.Module):
    def forward(self, x, res):
        return x + res

# 带门控的残差连接层
class GatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim * 3, 1, bias = False),
            nn.Sigmoid()
        )

    def forward(self, x, res):
        gate_input = torch.cat((x, res, x - res), dim = -1)
        gate = self.proj(gate_input)
        return x * gate + res * (1 - gate)

# attention

# 注意力机制层
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        pos_emb = None,
        dim_head = 64,
        heads = 8,
        edge_dim = None
    ):
        super().__init__()
        edge_dim = default(edge_dim, dim)

        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.pos_emb = pos_emb

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_kv = nn.Linear(dim, inner_dim * 2)
        self.edges_to_kv = nn.Linear(edge_dim, inner_dim)

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, nodes, edges, mask = None):
        h = self.heads

        q = self.to_q(nodes)
        k, v = self.to_kv(nodes).chunk(2, dim = -1)

        e_kv = self.edges_to_kv(edges)

        q, k, v, e_kv = map(lambda t: rearrange(t, 'b ... (h d) -> (b h) ... d', h = h), (q, k, v, e_kv))

        if exists(self.pos_emb):
            freqs = self.pos_emb(torch.arange(nodes.shape[1], device = nodes.device))
            freqs = rearrange(freqs, 'n d -> () n d')
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)

        ek, ev = e_kv, e_kv

        k, v = map(lambda t: rearrange(t, 'b j d -> b () j d '), (k, v))
        k = k + ek
        v = v + ev

        sim = einsum('b i d, b i j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b i -> b i ()') & rearrange(mask, 'b j -> b () j')
            mask = repeat(mask, 'b i j -> (b h) i j', h = h)
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b i j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# optional feedforward

# 可选的前馈神经网络层
def FeedForward(dim, ff_mult = 4):
    return nn.Sequential(
        nn.Linear(dim, dim * ff_mult),
        nn.GELU(),
        nn.Linear(dim * ff_mult, dim)
    )

# classes

# 图形变换器模型
class GraphTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head = 64,
        edge_dim = None,
        heads = 8,
        gated_residual = True,
        with_feedforwards = False,
        norm_edges = False,
        rel_pos_emb = False,
        accept_adjacency_matrix = False
    # 初始化函数，继承父类的初始化方法
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化图神经网络的层列表
        self.layers = List([])
        # 设置边的维度，默认为节点的维度
        edge_dim = default(edge_dim, dim)
        # 如果需要对边进行归一化，则使用 LayerNorm 进行归一化，否则使用恒等映射
        self.norm_edges = nn.LayerNorm(edge_dim) if norm_edges else nn.Identity()

        # 如果需要接受邻接矩阵，则使用 Embedding 层进行嵌入，否则设为 None
        self.adj_emb = nn.Embedding(2, edge_dim) if accept_adjacency_matrix else None

        # 如果需要相对位置编码，则使用 RotaryEmbedding 进行编码，否则设为 None
        pos_emb = RotaryEmbedding(dim_head) if rel_pos_emb else None

        # 循环创建指定深度的图神经网络层
        for _ in range(depth):
            # 添加每一层的注意力机制和前馈网络
            self.layers.append(List([
                List([
                    # 使用预归一化和注意力机制
                    PreNorm(dim, Attention(dim, pos_emb = pos_emb, edge_dim = edge_dim, dim_head = dim_head, heads = heads)),
                    GatedResidual(dim)
                ]),
                List([
                    # 使用预归一化和前馈网络
                    PreNorm(dim, FeedForward(dim)),
                    GatedResidual(dim)
                ]) if with_feedforwards else None
            ]))

    # 前向传播函数
    def forward(
        self,
        nodes,
        edges = None,
        adj_mat = None,
        mask = None
    ):
        # 获取节点的批次大小、序列长度和维度
        batch, seq, _ = nodes.shape

        # 如果存在边信息，则对边进行归一化处理
        if exists(edges):
            edges = self.norm_edges(edges)

        # 如果存在邻接矩阵，则进行相应处理
        if exists(adj_mat):
            assert adj_mat.shape == (batch, seq, seq)
            assert exists(self.adj_emb), 'accept_adjacency_matrix must be set to True'
            adj_mat = self.adj_emb(adj_mat.long())

        # 组合所有边信息
        all_edges = default(edges, 0) + default(adj_mat, 0)

        # 遍历每一层的注意力机制和前馈网络
        for attn_block, ff_block in self.layers:
            attn, attn_residual = attn_block
            # 使用注意力机制和门控残差连接更新节点信息
            nodes = attn_residual(attn(nodes, all_edges, mask = mask), nodes)

            # 如果存在前馈网络，则使用前馈网络和门控残差连接更新节点信息
            if exists(ff_block):
                ff, ff_residual = ff_block
                nodes = ff_residual(ff(nodes), nodes)

        # 返回更新后的节点信息和边信息
        return nodes, edges
```