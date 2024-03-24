# `.\lucidrains\point-transformer-pytorch\point_transformer_pytorch\multihead_point_transformer_pytorch.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn, einsum 模块
from torch import nn, einsum
# 从 einops 库中导入 repeat, rearrange 函数

# helpers

# 判断值是否存在的函数
def exists(val):
    return val is not None

# 获取张量的最大值
def max_value(t):
    return torch.finfo(t.dtype).max

# 在指定维度上对批量索引进行选择的函数
def batched_index_select(values, indices, dim = 1):
    # 获取值的维度
    value_dims = values.shape[(dim + 1):]
    # 获取值和索引的形状
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    # 将索引扩展到与值相同的维度
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

# classes

# 多头点变换器层类
class MultiheadPointTransformerLayer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 4,
        dim_head = 64,
        pos_mlp_hidden_dim = 64,
        attn_mlp_hidden_mult = 4,
        num_neighbors = None
    ):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads

        self.num_neighbors = num_neighbors

        # 线性变换，将输入维度映射到内部维度的三倍
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        # 线性变换，将内部维度映射回输出维度
        self.to_out = nn.Linear(inner_dim, dim)

        # 位置多层感知机
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_mlp_hidden_dim, inner_dim)
        )

        attn_inner_dim = inner_dim * attn_mlp_hidden_mult

        # 注意力多层感知机
        self.attn_mlp = nn.Sequential(
            nn.Conv2d(inner_dim, attn_inner_dim, 1, groups = heads),
            nn.ReLU(),
            nn.Conv2d(attn_inner_dim, inner_dim, 1, groups = heads),
        )
    # 定义前向传播函数，接受输入 x、位置 pos 和可选的掩码 mask
    def forward(self, x, pos, mask = None):
        # 获取输入 x 的维度信息
        n, h, num_neighbors = x.shape[1], self.heads, self.num_neighbors

        # 获取查询、键、值
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # 将查询、键、值按照头数 h 进行分组
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # 计算相对位置嵌入
        rel_pos = rearrange(pos, 'b i c -> b i 1 c') - rearrange(pos, 'b j c -> b 1 j c')
        rel_pos_emb = self.pos_mlp(rel_pos)

        # 将相对位置嵌入按照头数 h 进行分组
        rel_pos_emb = rearrange(rel_pos_emb, 'b i j (h d) -> b h i j d', h = h)

        # 使用查询减去键。这可能是比点积更好的归纳偏差，适用于点云
        qk_rel = rearrange(q, 'b h i d -> b h i 1 d') - rearrange(k, 'b h j d -> b h 1 j d')

        # 准备掩码
        if exists(mask):
            mask = rearrange(mask, 'b i -> b i 1') * rearrange(mask, 'b j -> b 1 j')

        # 扩展值
        v = repeat(v, 'b h j d -> b h i j d', i = n)

        # 如果指定了 num_neighbors，则确定每个点的 k 近邻
        if exists(num_neighbors) and num_neighbors < n:
            rel_dist = rel_pos.norm(dim = -1)

            if exists(mask):
                mask_value = max_value(rel_dist)
                rel_dist.masked_fill_(~mask, mask_value)

            dist, indices = rel_dist.topk(num_neighbors, largest = False)

            indices_with_heads = repeat(indices, 'b i j -> b h i j', h = h)

            v = batched_index_select(v, indices_with_heads, dim = 3)
            qk_rel = batched_index_select(qk_rel, indices_with_heads, dim = 3)
            rel_pos_emb = batched_index_select(rel_pos_emb, indices_with_heads, dim = 3)

            if exists(mask):
                mask = batched_index_select(mask, indices, dim = 2)

        # 将相对位置嵌入添加到值中
        v = v + rel_pos_emb

        # 使用注意力 MLP，确保先添加相对位置嵌入
        attn_mlp_input = qk_rel + rel_pos_emb
        attn_mlp_input = rearrange(attn_mlp_input, 'b h i j d -> b (h d) i j')

        sim = self.attn_mlp(attn_mlp_input)

        # 掩码
        if exists(mask):
            mask_value = -max_value(sim)
            mask = rearrange(mask, 'b i j -> b 1 i j')
            sim.masked_fill_(~mask, mask_value)

        # 注意力
        attn = sim.softmax(dim = -2)

        # 聚合
        v = rearrange(v, 'b h i j d -> b i j (h d)')
        agg = einsum('b d i j, b i j d -> b i d', attn, v)

        # 合并头
        return self.to_out(agg)
```