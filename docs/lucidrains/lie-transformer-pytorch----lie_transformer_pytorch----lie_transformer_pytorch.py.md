# `.\lucidrains\lie-transformer-pytorch\lie_transformer_pytorch\lie_transformer_pytorch.py`

```
# 导入数学库
import math
# 从 functools 库中导入 partial 函数
from functools import partial
# 导入 PyTorch 库
import torch
import torch.nn.functional as F
# 从 torch 库中导入 nn 模块和 einsum 函数
from torch import nn, einsum
# 从 lie_transformer_pytorch.se3 模块中导入 SE3 类
from lie_transformer_pytorch.se3 import SE3
# 从 einops 库中导入 rearrange 和 repeat 函数
from einops import rearrange, repeat
# 从 lie_transformer_pytorch.reversible 模块中导入 SequentialSequence 和 ReversibleSequence 类

# helpers

# 定义函数，判断变量是否存在
def exists(val):
    return val is not None

# 定义函数，将变量转换为元组
def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else ((val,) * depth)

# 定义函数，返回默认值
def default(val, d):
    return val if exists(val) else d

# 定义函数，对张量进行批量索引选择
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

# helper classes

# 定义 Pass 类，用于对输入进行处理
class Pass(nn.Module):
    def __init__(self, fn, dim = 1):
        super().__init__()
        self.fn = fn
        self.dim = dim

    def forward(self,x):
        dim = self.dim
        xs = list(x)
        xs[dim] = self.fn(xs[dim])
        return xs

# 定义 Lambda 类，用于对输入进行处理
class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)

# 定义 GlobalPool 类，用于计算在掩码中所有空间位置（和群元素）上减少的值
class GlobalPool(nn.Module):
    def __init__(self, mean = False):
        super().__init__()
        self.mean = mean

    def forward(self, x):
        coords, vals, mask = x

        if not exists(mask):
            return val.mean(dim = 1)

        masked_vals = vals.masked_fill_(~mask[..., None], 0.)
        summed = masked_vals.sum(dim = 1)

        if not self.mean:
            return summed

        count = mask.sum(-1).unsqueeze(-1)
        return summed / count

# subsampling code

# 定义 FPSindices 函数，用于根据距离矩阵和掩码进行下采样
def FPSindices(dists, frac, mask):
    """ inputs: pairwise distances DISTS (bs,n,n), downsample_frac (float), valid atom mask (bs,n)
        outputs: chosen_indices (bs,m) """
    m = int(round(frac * dists.shape[1]))
    bs, n, device = *dists.shape[:2], dists.device
    dd_kwargs = {'device': device, 'dtype': torch.long}
    B = torch.arange(bs, **dd_kwargs)

    chosen_indices = torch.zeros(bs, m, **dd_kwargs)
    distances = torch.ones(bs, n, device=device) * 1e8
    a = torch.randint(0, n, (bs,), **dd_kwargs)            # choose random start
    idx = a % mask.sum(-1) + torch.cat([torch.zeros(1, **dd_kwargs), torch.cumsum(mask.sum(-1), dim=0)[:-1]], dim=0)
    farthest = torch.where(mask)[1][idx]

    for i in range(m):
        chosen_indices[:, i] = farthest                    # add point that is farthest to chosen
        dist = dists[B, farthest].masked_fill(~mask, -100) # (bs,n) compute distance from new point to all others
        closer = dist < distances                          # if dist from new point is smaller than chosen points so far
        distances[closer] = dist[closer]                   # update the chosen set's distance to all other points
        farthest = torch.max(distances, -1)[1]             # select the point that is farthest from the set

    return chosen_indices

# 定义 FPSsubsample 类，用于进行 FPS 下采样
class FPSsubsample(nn.Module):
    def __init__(self, ds_frac, cache = False, group = None):
        super().__init__()
        self.ds_frac = ds_frac
        self.cache = cache
        self.cached_indices = None
        self.group = default(group, SE3())
    # 获取查询索引，根据是否启用缓存和缓存文件是否存在来决定是否重新计算
    def get_query_indices(self, abq_pairs, mask):
        # 如果启用缓存并且缓存文件存在，则直接返回缓存的查询索引
        if self.cache and exists(self.cached_indices):
            return self.cached_indices

        # 定义距离函数，如果存在分组则使用分组的距离函数，否则使用默认的 L2 范数
        dist = self.group.distance if self.group else lambda ab: ab.norm(dim=-1)
        # 计算 FPS 索引，根据数据集的分数和掩码值，返回索引值，并且将其从计算图中分离
        value = FPSindices(dist(abq_pairs), self.ds_frac, mask).detach()

        # 如果启用缓存，则将计算得到的索引值缓存起来
        if self.cache:
            self.cached_indices = value

        # 返回计算得到的索引值
        return value

    # 前向传播函数，根据输入数据进行处理并返回结果
    def forward(self, inp, withquery=False):
        # 解包输入数据
        abq_pairs, vals, mask, edges = inp
        # 获取设备信息
        device = vals.device

        # 如果数据子采样比例不为1
        if self.ds_frac != 1:
            # 获取查询索引
            query_idx = self.get_query_indices(abq_pairs, mask)

            # 创建索引 B，用于索引操作
            B = torch.arange(query_idx.shape[0], device=device).long()[:, None]
            # 根据查询索引对 abq_pairs 进行子采样
            subsampled_abq_pairs = abq_pairs[B, query_idx][B, :, query_idx]
            # 根据查询索引对 vals 进行子采样
            subsampled_values = batched_index_select(vals, query_idx, dim=1)
            # 根据查询索引对 mask 进行子采样
            subsampled_mask = batched_index_select(mask, query_idx, dim=1)
            # 如果存在边信息，则根据查询索引对 edges 进行子采样
            subsampled_edges = edges[B, query_idx][B, :, query_idx] if exists(edges) else None
        else:
            # 如果数据子采样比例为1，则不进行子采样操作
            subsampled_abq_pairs = abq_pairs
            subsampled_values = vals
            subsampled_mask = mask
            subsampled_edges = edges
            query_idx = None

        # 将子采样后的数据组合成元组
        ret = (
            subsampled_abq_pairs,
            subsampled_values,
            subsampled_mask,
            subsampled_edges
        )

        # 如果需要查询索引信息，则将查询索引信息添加到返回结果中
        if withquery:
            ret = (*ret, query_idx)

        # 返回处理后的结果
        return ret
# 定义一个自注意力机制的类 LieSelfAttention
class LieSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        edge_dim = None,
        group = None,
        mc_samples = 32,
        ds_frac = 1,
        fill = 1 / 3,
        dim_head = 64,
        heads = 8,
        cache = False
    ):
        super().__init__()
        self.dim = dim

        # 设置用于估计卷积的样本数量
        self.mc_samples = mc_samples
        # 设置 LieConv 的等变性群
        self.group = default(group, SE3())
        # 注册缓冲区变量 r，用于本地邻域半径，由 fill 设置
        self.register_buffer('r',torch.tensor(2.))
        # 设置平均输入进入本地邻域的分数，决定 r
        self.fill_frac = min(fill, 1.)
        
        # 创建 FPSsubsample 对象，用于下采样
        self.subsample = FPSsubsample(ds_frac, cache = cache, group = self.group)
        # 内部系数，用于更新 r
        self.coeff = .5
        # 用于记录平均填充分数，仅用于日志记录
        self.fill_frac_ema = fill

        # 注意力相关参数
        inner_dim = dim_head * heads
        self.heads = heads

        # 线性变换，用于计算查询、键、值和输出
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        edge_dim = default(edge_dim, 0)
        edge_dim_in = self.group.lie_dim + edge_dim

        # 局部注意力 MLP
        self.loc_attn_mlp = nn.Sequential(
            nn.Linear(edge_dim_in, edge_dim_in * 4),
            nn.ReLU(),
            nn.Linear(edge_dim_in * 4, 1),
        )

    # 提取邻域信息
    def extract_neighborhood(self, inp, query_indices):
        """ inputs: [pairs_abq (bs,n,n,d), inp_vals (bs,n,c), mask (bs,n), query_indices (bs,m)]
            outputs: [neighbor_abq (bs,m,mc_samples,d), neighbor_vals (bs,m,mc_samples,c)]"""

        # 从输入中获取数据
        pairs_abq, inp_vals, mask, edges = inp
        device = inp_vals.device

        # 根据查询索引对 pairs_abq、inp_vals、mask 进行下采样
        if exists(query_indices):
            abq_at_query = batched_index_select(pairs_abq, query_indices, dim = 1)
            mask_at_query = batched_index_select(mask, query_indices, dim = 1)
            edges_at_query = batched_index_select(edges, query_indices, dim = 1) if exists(edges) else None
        else:
            abq_at_query = pairs_abq
            mask_at_query = mask
            edges_at_query = edges

        mask_at_query = mask_at_query[..., None]

        vals_at_query = inp_vals
        dists = self.group.distance(abq_at_query)
        mask_value = torch.finfo(dists.dtype).max
        dists = dists.masked_fill(mask[:,None,:], mask_value)

        k = min(self.mc_samples, inp_vals.shape[1])

        # 从距离球中采样
        bs, m, n = dists.shape
        within_ball = (dists < self.r) & mask[:,None,:] & mask_at_query
        noise = torch.zeros((bs, m, n), device = device).uniform_(0, 1)
        valid_within_ball, nbhd_idx = torch.topk(within_ball + noise, k, dim=-1, sorted=False)
        valid_within_ball = (valid_within_ball > 1)

        # 获取邻域位置的 abq_pairs、values 和 mask
        nbhd_abq = batched_index_select(abq_at_query, nbhd_idx, dim = 2)
        nbhd_vals = batched_index_select(vals_at_query, nbhd_idx, dim = 1)
        nbhd_mask = batched_index_select(mask, nbhd_idx, dim = 1)
        nbhd_edges = batched_index_select(edges_at_query, nbhd_idx, dim = 2) if exists(edges) else None

        # 如果处于训练阶段，���新球半径以匹配 fill_frac
        if self.training:
            navg = (within_ball.float()).sum(-1).sum() / mask_at_query.sum()
            avg_fill = (navg / mask.sum(-1).float().mean()).cpu().item()
            self.r +=  self.coeff * (self.fill_frac - avg_fill)
            self.fill_frac_ema += .1 * (avg_fill-self.fill_frac_ema)

        nbhd_mask &= valid_within_ball.bool()

        return nbhd_abq, nbhd_vals, nbhd_mask, nbhd_edges, nbhd_idx
    # 定义前向传播函数，接收输入数据
    def forward(self, inp):
        """inputs: [pairs_abq (bs,n,n,d)], [inp_vals (bs,n,ci)]), [query_indices (bs,m)]
           outputs [subsampled_abq (bs,m,m,d)], [convolved_vals (bs,m,co)]"""
        # 从输入数据中抽取子样本，包括子样本的abq、值、掩码、边缘和查询索引
        sub_abq, sub_vals, sub_mask, sub_edges, query_indices = self.subsample(inp, withquery = True)
        # 从输入数据中提取邻域，包括邻域的abq、值、掩码、边缘和邻域索引
        nbhd_abq, nbhd_vals, nbhd_mask, nbhd_edges, nbhd_indices = self.extract_neighborhood(inp, query_indices)

        # 获取头数、批次大小、节点数、特征维度和设备信息
        h, b, n, d, device = self.heads, *sub_vals.shape, sub_vals.device

        # 将子样本的值转换为查询、键和值
        q, k, v = self.to_q(sub_vals), self.to_k(nbhd_vals), self.to_v(nbhd_vals)

        # 重排查询、键和值的维度
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)
        k, v = map(lambda t: rearrange(t, 'b n m (h d) -> b h n m d', h = h), (k, v))

        # 计算注意力相似度
        sim = einsum('b h i d, b h i j d -> b h i j', q, k) * (q.shape[-1] ** -0.5)

        # 更新边缘信息
        edges = nbhd_abq
        if exists(nbhd_edges):
            edges = torch.cat((nbhd_abq, nbhd_edges), dim = -1)

        # 通过位置注意力MLP更新位置注意力
        loc_attn = self.loc_attn_mlp(edges)
        loc_attn = rearrange(loc_attn, 'b i j () -> b () i j')
        sim = sim + loc_attn

        # 创建掩码值
        mask_value = -torch.finfo(sim.dtype).max

        # 使用掩码值对相似度矩阵进行掩码
        sim.masked_fill_(~rearrange(nbhd_mask, 'b n m -> b () n m'), mask_value)

        # 计算注意力权重
        attn = sim.softmax(dim = -1)
        # 计算输出值
        out = einsum('b h i j, b h i j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        # 将输出值转换为输出维度
        combined = self.to_out(out)

        # 返回子样本的abq、组合值、子样本掩码和子样本边缘
        return sub_abq, combined, sub_mask, sub_edges
class LieSelfAttentionWrapper(nn.Module):
    # 自注意力机制的包装器类
    def __init__(self, dim, attn):
        super().__init__()
        self.dim = dim
        self.attn = attn

        self.net = nn.Sequential(
            Pass(nn.LayerNorm(dim)),  # 添加层归一化
            self.attn
        )

    def forward(self, inp):
        sub_coords, sub_values, mask, edges = self.attn.subsample(inp)
        new_coords, new_values, mask, edges = self.net(inp)
        new_values[..., :self.dim] += sub_values
        return new_coords, new_values, mask, edges

class FeedForward(nn.Module):
    # 前馈神经网络类
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.dim = dim

        self.net = nn.Sequential(
            Pass(nn.LayerNorm(dim)),  # 添加层归一化
            Pass(nn.Linear(dim, mult * dim)),  # 线性变换
            Pass(nn.GELU()),  # GELU激活函数
            Pass(nn.Linear(mult * dim, dim)),  # 线性变换
        )

    def forward(self,inp):
        sub_coords, sub_values, mask, edges = inp
        new_coords, new_values, mask, edges = self.net(inp)
        new_values = new_values + sub_values
        return new_coords, new_values, mask, edges

# transformer class

class LieTransformer(nn.Module):
    """
    [Fill] specifies the fraction of the input which is included in local neighborhood.
            (can be array to specify a different value for each layer)
    [nbhd] number of samples to use for Monte Carlo estimation (p)
    [dim] number of input channels: 1 for MNIST, 3 for RGB images, other for non images
    [ds_frac] total downsampling to perform throughout the layers of the net. In (0,1)
    [num_layers] number of BottleNeck Block layers in the network
    [k] channel width for the network. Can be int (same for all) or array to specify individually.
    [liftsamples] number of samples to use in lifting. 1 for all groups with trivial stabilizer. Otherwise 2+
    [Group] Chosen group to be equivariant to.
    """
    def __init__(
        self,
        dim,
        num_tokens = None,
        num_edge_types = None,
        edge_dim = None,
        heads = 8,
        dim_head = 64,
        depth = 2,
        ds_frac = 1.,
        dim_out = None,
        k = 1536,
        nbhd = 128,
        mean = True,
        per_point = True,
        liftsamples = 4,
        fill = 1 / 4,
        cache = False,
        reversible = False,
        **kwargs
    ):
        super().__init__()
        assert not (ds_frac < 1 and reversible), 'must not downsample if network is reversible'

        dim_out = default(dim_out, dim)
        self.token_emb = nn.Embedding(num_tokens, dim) if exists(num_tokens) else None
        self.edge_emb = nn.Embedding(num_edge_types, edge_dim) if exists(num_edge_types) else None

        group = SE3()
        self.group = group
        self.liftsamples = liftsamples

        layers_fill = cast_tuple(fill, depth)
        layers = nn.ModuleList([])

        for _, layer_fill in zip(range(depth), layers_fill):
            layers.append(nn.ModuleList([
                LieSelfAttentionWrapper(dim, LieSelfAttention(dim, heads = heads, dim_head = dim_head, edge_dim = edge_dim, mc_samples = nbhd, ds_frac = ds_frac, group = group, fill = fill, cache = cache,**kwargs)),
                FeedForward(dim)
            ]))

        execute_class = ReversibleSequence if reversible else SequentialSequence
        self.net = execute_class(layers)

        self.to_logits = nn.Sequential(
            Pass(nn.LayerNorm(dim)),  # 添加层归一化
            Pass(nn.Linear(dim, dim_out))  # 线性变换
        )

        self.pool = GlobalPool(mean = mean)  # 全局池化
    # 定义一个前向传播函数，接受特征、坐标、边缘、掩码等参数，并返回池化结果
    def forward(self, feats, coors, edges = None, mask = None, return_pooled = False):
        # 获取批次大小、节点数等信息
        b, n, *_ = feats.shape

        # 如果存在 token_emb 属性，则对特征进行处理
        if exists(self.token_emb):
            feats = self.token_emb(feats)

        # 如果存在 edge_emb 属性，则对边缘进行处理
        if exists(self.edge_emb):
            # 确保 edges 参数存在
            assert exists(edges), 'edges must be passed in on forward'
            # 确保 edges 的形状符合要求
            assert edges.shape[1] == edges.shape[2] and edges.shape[1] == n, f'edges must be of the shape ({b}, {n}, {n})'
            edges = self.edge_emb(edges)

        # 将坐标、特征、掩码、边缘等参数组合成元组
        inps = (coors, feats, mask, edges)

        # 使用 group 属性对输入进行变换
        lifted_x = self.group.lift(inps, self.liftsamples)
        # 将变换后的输入传入网络进行计算
        out = self.net(lifted_x)

        # 将输出结果转换为 logits
        out = self.to_logits(out)

        # 如果不需要返回池化结果，则直接返回特征
        if not return_pooled:
            features = out[1]
            return features

        # 返回池化结果
        return self.pool(out)
```