# `.\lucidrains\egnn-pytorch\egnn_pytorch\egnn_pytorch.py`

```
import torch
from torch import nn, einsum, broadcast_tensors
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# 辅助函数

# 检查值是否存在
def exists(val):
    return val is not None

# 安全除法，避免分母为零
def safe_div(num, den, eps = 1e-8):
    res = num.div(den.clamp(min = eps))
    res.masked_fill_(den == 0, 0.)
    return res

# 在给定维度上批量选择索引
def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

# 傅立叶编码距离
def fourier_encode_dist(x, num_encodings = 4, include_self = True):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device = device, dtype = dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim = -1) if include_self else x
    return x

# 嵌入标记
def embedd_token(x, dims, layers):
    stop_concat = -len(dims)
    to_embedd = x[:, stop_concat:].long()
    for i,emb_layer in enumerate(layers):
        # 与 `to_embedd` 部分对应的部分被丢弃
        x = torch.cat([ x[:, :stop_concat], 
                        emb_layer( to_embedd[:, i] ) 
                      ], dim=-1)
        stop_concat = x.shape[-1]
    return x

# Swish 激活函数回退
class Swish_(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

SiLU = nn.SiLU if hasattr(nn, 'SiLU') else Swish_

# 辅助类

# 这遵循与 SE3 Transformers 中规范化相同的策略
class CoorsNorm(nn.Module):
    def __init__(self, eps = 1e-8, scale_init = 1.):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim = -1, keepdim = True)
        normed_coors = coors / norm.clamp(min = self.eps)
        return normed_coors * self.scale

# 全局线性注意力
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context, mask = None):
        h = self.heads

        q = self.to_q(x)
        kv = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, *kv))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(mask):
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b n -> b () () n')
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)

class GlobalLinearAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64
    # 定义一个 Transformer 模块，继承自 nn.Module
    ):
        # 调用父类的构造函数
        super().__init__()
        # 初始化 LayerNorm 模块，对输入进行归一化
        self.norm_seq = nn.LayerNorm(dim)
        self.norm_queries = nn.LayerNorm(dim)
        # 初始化两个 Attention 模块，用于计算注意力
        self.attn1 = Attention(dim, heads, dim_head)
        self.attn2 = Attention(dim, heads, dim_head)

        # 定义前馈神经网络结构
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),  # 对输入进行归一化
            nn.Linear(dim, dim * 4),  # 线性变换
            nn.GELU(),  # GELU 激活函数
            nn.Linear(dim * 4, dim)  # 线性变换
        )

    # 定义前向传播函数
    def forward(self, x, queries, mask = None):
        # 保存输入的原始值
        res_x, res_queries = x, queries
        # 对输入进行归一化
        x, queries = self.norm_seq(x), self.norm_queries(queries)

        # 计算第一个 Attention 模块的输出
        induced = self.attn1(queries, x, mask = mask)
        # 计算第二个 Attention 模块的输出
        out     = self.attn2(x, induced)

        # 将 Attention 模块的输出与原始输入相加
        x =  out + res_x
        queries = induced + res_queries

        # 经过前馈神经网络处理
        x = self.ff(x) + x
        # 返回处理后的结果
        return x, queries
# 定义 EGNN 类
class EGNN(nn.Module):
    # 初始化函数
    def __init__(
        self,
        dim,
        edge_dim = 0,
        m_dim = 16,
        fourier_features = 0,
        num_nearest_neighbors = 0,
        dropout = 0.0,
        init_eps = 1e-3,
        norm_feats = False,
        norm_coors = False,
        norm_coors_scale_init = 1e-2,
        update_feats = True,
        update_coors = True,
        only_sparse_neighbors = False,
        valid_radius = float('inf'),
        m_pool_method = 'sum',
        soft_edges = False,
        coor_weights_clamp_value = None
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 检查池化方法是否合法
        assert m_pool_method in {'sum', 'mean'}, 'pool method must be either sum or mean'
        # 检查是否需要更新特征或坐标
        assert update_feats or update_coors, 'you must update either features, coordinates, or both'

        # 设置傅立叶特征数量
        self.fourier_features = fourier_features

        # 计算边输入维度
        edge_input_dim = (fourier_features * 2) + (dim * 2) + edge_dim + 1
        # 根据 dropout 值创建 Dropout 层或者恒等映射
        dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 定义边 MLP 网络
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, edge_input_dim * 2),
            dropout,
            SiLU(),
            nn.Linear(edge_input_dim * 2, m_dim),
            SiLU()
        )

        # 如果使用软边，则定义边门控网络
        self.edge_gate = nn.Sequential(
            nn.Linear(m_dim, 1),
            nn.Sigmoid()
        ) if soft_edges else None

        # 如果需要对节点特征进行归一化，则定义 LayerNorm 层，否则为恒等映射
        self.node_norm = nn.LayerNorm(dim) if norm_feats else nn.Identity()
        # 如果需要对坐标进行归一化，则定义 CoorsNorm 层，否则为恒等映射
        self.coors_norm = CoorsNorm(scale_init = norm_coors_scale_init) if norm_coors else nn.Identity()

        # 设置池化方法
        self.m_pool_method = m_pool_method

        # 如果需要更新特征，则定义节点 MLP 网络
        self.node_mlp = nn.Sequential(
            nn.Linear(dim + m_dim, dim * 2),
            dropout,
            SiLU(),
            nn.Linear(dim * 2, dim),
        ) if update_feats else None

        # 如果需要更新坐标，则定义坐标 MLP 网络
        self.coors_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim * 4),
            dropout,
            SiLU(),
            nn.Linear(m_dim * 4, 1)
        ) if update_coors else None

        # 设置最近邻节点数量、是否只使用稀疏邻居、有效半径
        self.num_nearest_neighbors = num_nearest_neighbors
        self.only_sparse_neighbors = only_sparse_neighbors
        self.valid_radius = valid_radius

        # 设置坐标权重截断值
        self.coor_weights_clamp_value = coor_weights_clamp_value

        # 设置初始化值
        self.init_eps = init_eps
        # 应用初始化函数
        self.apply(self.init_)

    # 初始化函数
    def init_(self, module):
        # 如果模块类型为线性层
        if type(module) in {nn.Linear}:
            # 初始化权重，防止网络深度增加导致出现 NaN
            nn.init.normal_(module.weight, std = self.init_eps)

# 定义 EGNN_Network 类
class EGNN_Network(nn.Module):
    # 初始化函数
    def __init__(
        self,
        *,
        depth,
        dim,
        num_tokens = None,
        num_edge_tokens = None,
        num_positions = None,
        edge_dim = 0,
        num_adj_degrees = None,
        adj_dim = 0,
        global_linear_attn_every = 0,
        global_linear_attn_heads = 8,
        global_linear_attn_dim_head = 64,
        num_global_tokens = 4,
        **kwargs
    ):
        # 调用父类的构造函数
        super().__init__()
        # 断言邻接度数不小于1
        assert not (exists(num_adj_degrees) and num_adj_degrees < 1), 'make sure adjacent degrees is greater than 1'
        # 初始化位置数量
        self.num_positions = num_positions

        # 如果存在标记数量，则创建标记嵌入层
        self.token_emb = nn.Embedding(num_tokens, dim) if exists(num_tokens) else None
        # 如果存在位置数量，则创建位置嵌入层
        self.pos_emb = nn.Embedding(num_positions, dim) if exists(num_positions) else None
        # 如果存在边标记数量，则创建边嵌入层
        self.edge_emb = nn.Embedding(num_edge_tokens, edge_dim) if exists(num_edge_tokens) else None
        # 判断是否存在边
        self.has_edges = edge_dim > 0

        # 初始化邻接度数
        self.num_adj_degrees = num_adj_degrees
        # 如果邻接度数存在且邻接维度大于0，则创建邻接嵌入层
        self.adj_emb = nn.Embedding(num_adj_degrees + 1, adj_dim) if exists(num_adj_degrees) and adj_dim > 0 else None

        # 如果存在边，则将边维度赋值给edge_dim，否则为0
        edge_dim = edge_dim if self.has_edges else 0
        # 如果邻接度数存在，则将邻接维度赋值给adj_dim，否则为0
        adj_dim = adj_dim if exists(num_adj_degrees) else 0

        # 判断是否存在全局注意力
        has_global_attn = global_linear_attn_every > 0
        # 初始化全局标记
        self.global_tokens = None
        # 如果存在全局注意力，则初始化全局标记
        if has_global_attn:
            self.global_tokens = nn.Parameter(torch.randn(num_global_tokens, dim))

        # 初始化层列表
        self.layers = nn.ModuleList([])
        # 遍历深度
        for ind in range(depth):
            # 判断是否为全局层
            is_global_layer = has_global_attn and (ind % global_linear_attn_every) == 0

            # 添加全局线性注意力和EGNN层到层列表
            self.layers.append(nn.ModuleList([
                GlobalLinearAttention(dim = dim, heads = global_linear_attn_heads, dim_head = global_linear_attn_dim_head) if is_global_layer else None,
                EGNN(dim = dim, edge_dim = (edge_dim + adj_dim), norm_feats = True, **kwargs),
            ]))

    def forward(
        self,
        feats,
        coors,
        adj_mat = None,
        edges = None,
        mask = None,
        return_coor_changes = False
    ):
        # 获取批次大小和设备
        b, device = feats.shape[0], feats.device

        # 如果存在标记嵌入层，则对特征进行标记嵌入
        if exists(self.token_emb):
            feats = self.token_emb(feats)

        # 如果存在位置嵌入层，则对特征进行位置嵌入
        if exists(self.pos_emb):
            n = feats.shape[1]
            # 断言序列长度小于等于初始化时设置的位置数量
            assert n <= self.num_positions, f'given sequence length {n} must be less than the number of positions {self.num_positions} set at init'
            pos_emb = self.pos_emb(torch.arange(n, device = device))
            feats += rearrange(pos_emb, 'n d -> () n d')

        # 如果存在边并且存在边嵌入层，则对边进行边嵌入
        if exists(edges) and exists(self.edge_emb):
            edges = self.edge_emb(edges)

        # 从一阶连接创建N度邻接矩阵
        if exists(self.num_adj_degrees):
            assert exists(adj_mat), 'adjacency matrix must be passed in (keyword argument adj_mat)'

            if len(adj_mat.shape) == 2:
                adj_mat = repeat(adj_mat.clone(), 'i j -> b i j', b = b)

            adj_indices = adj_mat.clone().long()

            for ind in range(self.num_adj_degrees - 1):
                degree = ind + 2

                next_degree_adj_mat = (adj_mat.float() @ adj_mat.float()) > 0
                next_degree_mask = (next_degree_adj_mat.float() - adj_mat.float()).bool()
                adj_indices.masked_fill_(next_degree_mask, degree)
                adj_mat = next_degree_adj_mat.clone()

            if exists(self.adj_emb):
                adj_emb = self.adj_emb(adj_indices)
                edges = torch.cat((edges, adj_emb), dim = -1) if exists(edges) else adj_emb

        # 设置全局注意力
        global_tokens = None
        if exists(self.global_tokens):
            global_tokens = repeat(self.global_tokens, 'n d -> b n d', b = b)

        # 遍历层
        coor_changes = [coors]

        for global_attn, egnn in self.layers:
            if exists(global_attn):
                feats, global_tokens = global_attn(feats, global_tokens, mask = mask)

            feats, coors = egnn(feats, coors, adj_mat = adj_mat, edges = edges, mask = mask)
            coor_changes.append(coors)

        # 如果需要返回坐标变化，则返回特征、坐标和坐标变化
        if return_coor_changes:
            return feats, coors, coor_changes

        # 否则只返回特征和坐标
        return feats, coors
```