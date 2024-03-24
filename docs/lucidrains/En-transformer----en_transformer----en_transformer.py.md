# `.\lucidrains\En-transformer\en_transformer\en_transformer.py`

```py
# 导入 torch 库
import torch
# 导入 torch 中的函数库
import torch.nn.functional as F
# 从 torch 中导入 nn、einsum 模块
from torch import nn, einsum
# 从 torch.utils.checkpoint 中导入 checkpoint_sequential 函数
from torch.utils.checkpoint import checkpoint_sequential
# 从 einx 中导入 get_at 函数
from einx import get_at
# 从 einops 中导入 rearrange、repeat、reduce 函数，从 einops.layers.torch 中导入 Rearrange 类
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
# 从 taylor_series_linear_attention 中导入 TaylorSeriesLinearAttn 类

# 辅助函数

# 判断变量是否存在的函数
def exists(val):
    return val is not None

# 返回指定数据类型的最小负值的函数
def max_neg_value(t):
    return -torch.finfo(t.dtype).max

# 如果变量存在则返回该变量，否则返回默认值的函数
def default(val, d):
    return val if exists(val) else d

# 对输入张量进行 L2 归一化的函数
def l2norm(t):
    return F.normalize(t, dim = -1)

# 对 nn.Linear 类型的权重进行小范围初始化的函数
def small_init_(t: nn.Linear):
    nn.init.normal_(t.weight, std = 0.02)
    nn.init.zeros_(t.bias)

# 动态位置偏置

class DynamicPositionBias(nn.Module):
    def __init__(
        self,
        dim,
        *,
        heads,
        depth,
        dim_head,
        input_dim = 1,
        norm = True
    ):
        super().__init__()
        assert depth >= 1, 'depth for dynamic position bias MLP must be greater or equal to 1'
        self.mlp = nn.ModuleList([])

        self.mlp.append(nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.LayerNorm(dim) if norm else nn.Identity(),
            nn.SiLU()
        ))

        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim) if norm else nn.Identity(),
                nn.SiLU()
            ))

        self.heads = heads
        self.qk_pos_head = nn.Linear(dim, heads)
        self.value_pos_head = nn.Linear(dim, dim_head * heads)

    def forward(self, pos):
        for layer in self.mlp:
            pos = layer(pos)

        qk_pos = self.qk_pos_head(pos)
        value_pos = self.value_pos_head(pos)

        qk_pos = rearrange(qk_pos, 'b 1 i j h -> b h i j')
        value_pos = rearrange(value_pos, 'b 1 i j (h d) -> b h i j d', h = self.heads)
        return qk_pos, value_pos

# 类

# 此类遵循 SE3 Transformers 中的规范化策略
# https://github.com/lucidrains/se3-transformer-pytorch/blob/main/se3_transformer_pytorch/se3_transformer_pytorch.py#L95

# 层归一化类
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# 坐标归一化类
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

# 残差连接类
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, feats, coors, **kwargs):
        feats_out, coors_delta = self.fn(feats, coors, **kwargs)
        return feats + feats_out, coors + coors_delta

# GEGLU 激活函数类
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

# 前馈神经网络类
class FeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = int(dim * mult * 2 / 3)

        self.net = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, inner_dim * 2, bias = False),
            GEGLU(),
            LayerNorm(inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias = False)
        )

    def forward(self, feats, coors):
        return self.net(feats), 0

class EquivariantAttention(nn.Module):
    # 初始化函数，设置Transformer模型的参数
    def __init__(
        self,
        *,
        dim,  # 输入特征的维度
        dim_head = 64,  # 每个头的维度
        heads = 4,  # 多头注意力机制的头数
        edge_dim = 0,  # 边的特征维度
        coors_hidden_dim = 16,  # 坐标隐藏层的维度
        neighbors = 0,  # 邻居节点的数量
        only_sparse_neighbors = False,  # 是否只使用稀疏邻居
        valid_neighbor_radius = float('inf'),  # 有效邻居的半径
        init_eps = 1e-3,  # 初始化的小量值
        rel_pos_emb = None,  # 相对位置编码
        edge_mlp_mult = 2,  # 边的多层感知机的倍数
        norm_rel_coors = True,  # 是否对相对坐标进行归一化
        norm_coors_scale_init = 1.,  # 归一化坐标的初始值
        use_cross_product = False,  # 是否使用叉积
        talking_heads = False,  # 是否使用Talking Heads
        dropout = 0.,  # Dropout概率
        num_global_linear_attn_heads = 0,  # 全局线性注意力机制的头数
        linear_attn_dim_head = 8,  # 线性注意力机制的头维度
        gate_outputs = True,  # 是否使用门控输出
        gate_init_bias = 10.  # 门控初始化偏置
    # 初始化函数，设置模型参数初始化方式
    def __init__(
        self,
        heads,
        dim,
        dim_head,
        num_global_linear_attn_heads,
        linear_attn_dim_head,
        gate_outputs,
        gate_init_bias,
        talking_heads,
        edge_dim,
        edge_mlp_mult,
        coors_hidden_dim,
        norm_coors,
        norm_coors_scale_init,
        use_cross_product,
        rel_pos_emb,
        dropout,
        init_eps,
        neighbors,
        only_sparse_neighbors,
        valid_neighbor_radius
    ):
        # 调用父类初始化函数
        super().__init__()
        # 设置缩放因子
        self.scale = dim_head ** -0.5
        # 对输入进行归一化
        self.norm = LayerNorm(dim)

        # 设置邻居节点相关参数
        self.neighbors = neighbors
        self.only_sparse_neighbors = only_sparse_neighbors
        self.valid_neighbor_radius = valid_neighbor_radius

        # 计算注意力机制内部维度
        attn_inner_dim = heads * dim_head
        self.heads = heads

        # 判断是否有全局线性注意力机制
        self.has_linear_attn = num_global_linear_attn_heads > 0

        # 初始化全局线性注意力机制
        self.linear_attn = TaylorSeriesLinearAttn(
            dim = dim,
            dim_head = linear_attn_dim_head,
            heads = num_global_linear_attn_heads,
            gate_value_heads = True,
            combine_heads = False
        )

        # 线性变换，将输入转换为查询、键、值
        self.to_qkv = nn.Linear(dim, attn_inner_dim * 3, bias = False)
        # 线性变换，将注意力机制输出转换为模型输出
        self.to_out = nn.Linear(attn_inner_dim + self.linear_attn.dim_hidden, dim)

        # 是否使用门控输出
        self.gate_outputs = gate_outputs
        if gate_outputs:
            # 初始化门控线性层
            gate_linear = nn.Linear(dim, 2 * heads)
            nn.init.zeros_(gate_linear.weight)
            nn.init.constant_(gate_linear.bias, gate_init_bias)

            # 设置输出门控
            self.to_output_gates = nn.Sequential(
                gate_linear,
                nn.Sigmoid(),
                Rearrange('b n (l h) -> l b h n 1', h = heads)
            )

        # 是否使用Talking Heads
        self.talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else None

        # 初始化边缘MLP
        self.edge_mlp = None
        has_edges = edge_dim > 0

        if has_edges:
            edge_input_dim = heads + edge_dim
            edge_hidden = edge_input_dim * edge_mlp_mult

            # 设置边缘MLP
            self.edge_mlp = nn.Sequential(
                nn.Linear(edge_input_dim, edge_hidden, bias = False),
                nn.GELU(),
                nn.Linear(edge_hidden, heads, bias = False)
            )

            # 设置坐标MLP
            self.coors_mlp = nn.Sequential(
                nn.GELU(),
                nn.Linear(heads, heads, bias = False)
            )
        else:
            # 设置坐标MLP
            self.coors_mlp = nn.Sequential(
                nn.Linear(heads, coors_hidden_dim, bias = False),
                nn.GELU(),
                nn.Linear(coors_hidden_dim, heads, bias = False)
            )

        # 设置坐标门控
        self.coors_gate = nn.Linear(heads, heads)
        small_init_(self.coors_gate)

        # 是否使用交叉乘积
        self.use_cross_product = use_cross_product
        if use_cross_product:
            # 设置交叉坐标MLP
            self.cross_coors_mlp = nn.Sequential(
                nn.Linear(heads, coors_hidden_dim, bias = False),
                nn.GELU(),
                nn.Linear(coors_hidden_dim, heads * 2, bias = False)
            )

            # 设置交叉坐标门控
            self.cross_coors_gate_i = nn.Linear(heads, heads)
            self.cross_coors_gate_j = nn.Linear(heads, heads)

            small_init_(self.cross_coors_gate_i)
            small_init_(self.cross_coors_gate_j)

        # 设置坐标归一化
        self.norm_rel_coors = CoorsNorm(scale_init = norm_coors_scale_init) if norm_rel_coors else nn.Identity()

        # 设置坐标组合参数
        num_coors_combine_heads = (2 if use_cross_product else 1) * heads
        self.coors_combine = nn.Parameter(torch.randn(num_coors_combine_heads))

        # 位置嵌入
        # 用于序列和残基/原子之间的相对距离

        self.rel_pos_emb = rel_pos_emb

        # 动态位置偏置MLP
        self.dynamic_pos_bias_mlp = DynamicPositionBias(
            dim = dim // 2,
            heads = heads,
            dim_head = dim_head,
            depth = 3,
            input_dim = (2 if rel_pos_emb else 1)
        )

        # 丢弃层

        self.node_dropout = nn.Dropout(dropout)
        self.coor_dropout = nn.Dropout(dropout)

        # 初始化

        self.init_eps = init_eps
        self.apply(self.init_)

    # 初始化函数，设置模型参数初始化方式
    def init_(self, module):
        if type(module) in {nn.Linear}:
            # 初始化线性层参数
            nn.init.normal_(module.weight, std = self.init_eps)

    # 前向传播函数
    def forward(
        self,
        feats,
        coors,
        edges = None,
        mask = None,
        adj_mat = None
# 定义一个 Transformer 模型的 Block 类，包含注意力机制和前馈神经网络
class Block(nn.Module):
    def __init__(self, attn, ff):
        super().__init__()
        self.attn = attn
        self.ff = ff

    # 前向传播函数，接收输入和坐标变化，返回处理后的特征、坐标、掩码、边缘和邻接矩阵
    def forward(self, inp, coor_changes = None):
        feats, coors, mask, edges, adj_mat = inp
        feats, coors = self.attn(feats, coors, edges = edges, mask = mask, adj_mat = adj_mat)
        feats, coors = self.ff(feats, coors)
        return (feats, coors, mask, edges, adj_mat)

# 定义一个 Encoder Transformer 模型
class EnTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        num_tokens = None,
        rel_pos_emb = False,
        dim_head = 64,
        heads = 8,
        num_edge_tokens = None,
        edge_dim = 0,
        coors_hidden_dim = 16,
        neighbors = 0,
        only_sparse_neighbors = False,
        num_adj_degrees = None,
        adj_dim = 0,
        valid_neighbor_radius = float('inf'),
        init_eps = 1e-3,
        norm_rel_coors = True,
        norm_coors_scale_init = 1.,
        use_cross_product = False,
        talking_heads = False,
        checkpoint = False,
        attn_dropout = 0.,
        ff_dropout = 0.,
        num_global_linear_attn_heads = 0,
        gate_outputs = True
    ):
        super().__init__()
        # 断言维度每个头部应大于等于32，以使旋转嵌入正常工作
        assert dim_head >= 32, 'your dimension per head should be greater than 32 for rotary embeddings to work well'
        # 断言邻接度数大于等于1
        assert not (exists(num_adj_degrees) and num_adj_degrees < 1), 'make sure adjacent degrees is greater than 1'

        # 如果只有稀疏邻居，则将邻接度数设置为1
        if only_sparse_neighbors:
            num_adj_degrees = default(num_adj_degrees, 1)

        # 初始化嵌入层
        self.token_emb = nn.Embedding(num_tokens, dim) if exists(num_tokens) else None
        self.edge_emb = nn.Embedding(num_edge_tokens, edge_dim) if exists(num_edge_tokens) else None

        # 初始化邻接矩阵嵌入层
        self.num_adj_degrees = num_adj_degrees
        self.adj_emb = nn.Embedding(num_adj_degrees + 1, adj_dim) if exists(num_adj_degrees) and adj_dim > 0 else None
        adj_dim = adj_dim if exists(num_adj_degrees) else 0

        self.checkpoint = checkpoint
        self.layers = nn.ModuleList([])

        # 循环创建 Transformer 模型的 Block 层
        for ind in range(depth):
            self.layers.append(Block(
                Residual(EquivariantAttention(
                    dim = dim,
                    dim_head = dim_head,
                    heads = heads,
                    coors_hidden_dim = coors_hidden_dim,
                    edge_dim = (edge_dim + adj_dim),
                    neighbors = neighbors,
                    only_sparse_neighbors = only_sparse_neighbors,
                    valid_neighbor_radius = valid_neighbor_radius,
                    init_eps = init_eps,
                    rel_pos_emb = rel_pos_emb,
                    norm_rel_coors = norm_rel_coors,
                    norm_coors_scale_init = norm_coors_scale_init,
                    use_cross_product = use_cross_product,
                    talking_heads = talking_heads,
                    dropout = attn_dropout,
                    num_global_linear_attn_heads = num_global_linear_attn_heads,
                    gate_outputs = gate_outputs
                )),
                Residual(FeedForward(
                    dim = dim,
                    dropout = ff_dropout
                ))
            ))

    # 前向传播函数，接收特征、坐标、边缘、掩码、邻接矩阵等参数，返回处理后的结果
    def forward(
        self,
        feats,
        coors,
        edges = None,
        mask = None,
        adj_mat = None,
        return_coor_changes = False,
        **kwargs
        ):
            # 获取特征的批次大小
            b = feats.shape[0]

            # 如果存在 token_emb 属性，则对特征进行处理
            if exists(self.token_emb):
                feats = self.token_emb(feats)

            # 如果存在 edge_emb 属性，则对边进行处理
            if exists(self.edge_emb):
                assert exists(edges), 'edges must be passed in as (batch x seq x seq) indicating edge type'
                edges = self.edge_emb(edges)

            # 检查是否存在邻接矩阵，并且 num_adj_degrees 大于 0
            assert not (exists(adj_mat) and (not exists(self.num_adj_degrees) or self.num_adj_degrees == 0)), 'num_adj_degrees must be greater than 0 if you are passing in an adjacency matrix'

            # 如果存在 num_adj_degrees 属性
            if exists(self.num_adj_degrees):
                assert exists(adj_mat), 'adjacency matrix must be passed in (keyword argument adj_mat)'

                # 如果邻接矩阵的维度为 2，则进行扩展
                if len(adj_mat.shape) == 2:
                    adj_mat = repeat(adj_mat.clone(), 'i j -> b i j', b = b)

                # 克隆邻接矩阵并转换为长整型
                adj_indices = adj_mat.clone().long()

                # 遍历 num_adj_degrees - 1 次
                for ind in range(self.num_adj_degrees - 1):
                    degree = ind + 2

                    # 计算下一阶邻接矩阵
                    next_degree_adj_mat = (adj_mat.float() @ adj_mat.float()) > 0
                    next_degree_mask = (next_degree_adj_mat.float() - adj_mat.float()).bool()
                    adj_indices.masked_fill_(next_degree_mask, degree)
                    adj_mat = next_degree_adj_mat.clone()

                # 如果存在 adj_emb 属性，则对邻接矩阵进行处理
                if exists(self.adj_emb):
                    adj_emb = self.adj_emb(adj_indices)
                    edges = torch.cat((edges, adj_emb), dim = -1) if exists(edges) else adj_emb

            # 检查是否需要返回坐标变化，并且模型处于训练模式
            assert not (return_coor_changes and self.training), 'you must be eval mode in order to return coordinates'

            # 遍历层
            coor_changes = [coors]
            inp = (feats, coors, mask, edges, adj_mat)

            # 如果处于训练模式且启用了检查点，则使用检查点跨块进行内存节省
            if self.training and self.checkpoint:
                inp = checkpoint_sequential(self.layers, len(self.layers), inp)
            else:
                # 遍历块
                for layer in self.layers:
                    inp = layer(inp)
                    coor_changes.append(inp[1]) # 为可视化添加坐标

            # 返回
            feats, coors, *_ = inp

            # 如果需要返回坐标变化，则返回特征、坐标和坐标变化
            if return_coor_changes:
                return feats, coors, coor_changes

            # 否则只返回特征和坐标
            return feats, coors
```