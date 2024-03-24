# `.\lucidrains\point-transformer-pytorch\point_transformer_pytorch\point_transformer_pytorch.py`

```py
import torch
from torch import nn, einsum
from einops import repeat

# 辅助函数

# 检查值是否存在
def exists(val):
    return val is not None

# 获取张量的最大值
def max_value(t):
    return torch.finfo(t.dtype).max

# 在给定维度上对批量索引进行选择
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

# 类

class PointTransformerLayer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        pos_mlp_hidden_dim = 64,
        attn_mlp_hidden_mult = 4,
        num_neighbors = None
    ):
        super().__init__()
        self.num_neighbors = num_neighbors

        # 线性变换，将输入维度映射到查询、键、值的维度
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        # 位置信息的多层感知机
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_mlp_hidden_dim, dim)
        )

        # 注意力机制的多层感知机
        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim * attn_mlp_hidden_mult),
            nn.ReLU(),
            nn.Linear(dim * attn_mlp_hidden_mult, dim),
        )

    def forward(self, x, pos, mask = None):
        n, num_neighbors = x.shape[1], self.num_neighbors

        # 获取查询、键、值
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # 计算相对位置嵌入
        rel_pos = pos[:, :, None, :] - pos[:, None, :, :]
        rel_pos_emb = self.pos_mlp(rel_pos)

        # 使用查询减去键。我认为这是点云的更好归纳偏差，而不是点积
        qk_rel = q[:, :, None, :] - k[:, None, :, :]

        # 准备掩码
        if exists(mask):
            mask = mask[:, :, None] * mask[:, None, :]

        # 扩展值
        v = repeat(v, 'b j d -> b i j d', i = n)

        # 如果指定了每个点的 k 近邻数，则确定 k 个最近邻
        if exists(num_neighbors) and num_neighbors < n:
            rel_dist = rel_pos.norm(dim = -1)

            if exists(mask):
                mask_value = max_value(rel_dist)
                rel_dist.masked_fill_(~mask, mask_value)

            dist, indices = rel_dist.topk(num_neighbors, largest = False)

            v = batched_index_select(v, indices, dim = 2)
            qk_rel = batched_index_select(qk_rel, indices, dim = 2)
            rel_pos_emb = batched_index_select(rel_pos_emb, indices, dim = 2)
            mask = batched_index_select(mask, indices, dim = 2) if exists(mask) else None

        # 将相对位置嵌入添加到值中
        v = v + rel_pos_emb

        # 使用注意力多层感知机，确保先添加相对位置嵌入
        sim = self.attn_mlp(qk_rel + rel_pos_emb)

        # 掩码
        if exists(mask):
            mask_value = -max_value(sim)
            sim.masked_fill_(~mask[..., None], mask_value)

        # 注意力
        attn = sim.softmax(dim = -2)

        # 聚合
        agg = einsum('b i j d, b i j d -> b i d', attn, v)
        return agg
```