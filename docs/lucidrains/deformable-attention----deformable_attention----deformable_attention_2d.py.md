# `.\lucidrains\deformable-attention\deformable_attention\deformable_attention_2d.py`

```
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat

# helper functions

# 检查值是否存在
def exists(val):
    return val is not None

# 返回值或默认值
def default(val, d):
    return val if exists(val) else d

# 检查一个数是否可以被另一个数整除
def divisible_by(numer, denom):
    return (numer % denom) == 0

# tensor helpers

# 创建与输入张量相同形状的网格
def create_grid_like(t, dim = 0):
    h, w, device = *t.shape[-2:], t.device

    grid = torch.stack(torch.meshgrid(
        torch.arange(w, device = device),
        torch.arange(h, device = device),
    indexing = 'xy'), dim = dim)

    grid.requires_grad = False
    grid = grid.type_as(t)
    return grid

# 将网格归一化到-1到1的范围
def normalize_grid(grid, dim = 1, out_dim = -1):
    # normalizes a grid to range from -1 to 1
    h, w = grid.shape[-2:]
    grid_h, grid_w = grid.unbind(dim = dim)

    grid_h = 2.0 * grid_h / max(h - 1, 1) - 1.0
    grid_w = 2.0 * grid_w / max(w - 1, 1) - 1.0

    return torch.stack((grid_h, grid_w), dim = out_dim)

# 缩放层
class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

# continuous positional bias from SwinV2

# 连续位置偏置
class CPB(nn.Module):
    """ https://arxiv.org/abs/2111.09883v1 """

    def __init__(self, dim, *, heads, offset_groups, depth):
        super().__init__()
        self.heads = heads
        self.offset_groups = offset_groups

        self.mlp = nn.ModuleList([])

        self.mlp.append(nn.Sequential(
            nn.Linear(2, dim),
            nn.ReLU()
        ))

        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU()
            ))

        self.mlp.append(nn.Linear(dim, heads // offset_groups)

    def forward(self, grid_q, grid_kv):
        device, dtype = grid_q.device, grid_kv.dtype

        grid_q = rearrange(grid_q, 'h w c -> 1 (h w) c')
        grid_kv = rearrange(grid_kv, 'b h w c -> b (h w) c')

        pos = rearrange(grid_q, 'b i c -> b i 1 c') - rearrange(grid_kv, 'b j c -> b 1 j c')
        bias = torch.sign(pos) * torch.log(pos.abs() + 1)  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)

        for layer in self.mlp:
            bias = layer(bias)

        bias = rearrange(bias, '(b g) i j o -> b (g o) i j', g = self.offset_groups)

        return bias

# main class

# 可变形注意力机制
class DeformableAttention2D(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        downsample_factor = 4,
        offset_scale = None,
        offset_groups = None,
        offset_kernel_size = 6,
        group_queries = True,
        group_key_values = True
    ):
        # 调用父类的构造函数
        super().__init__()
        # 设置偏移比例，默认为 downsample_factor
        offset_scale = default(offset_scale, downsample_factor)
        # 断言偏移核大小必须大于或等于下采样因子
        assert offset_kernel_size >= downsample_factor, 'offset kernel size must be greater than or equal to the downsample factor'
        # 断言偏移核大小减去下采样因子必须是偶数
        assert divisible_by(offset_kernel_size - downsample_factor, 2)

        # 设置偏移组数，默认为 heads
        offset_groups = default(offset_groups, heads)
        # 断言 heads 必须是 offset_groups 的倍数
        assert divisible_by(heads, offset_groups)

        # 计算内部维度
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.offset_groups = offset_groups

        # 计算偏移维度
        offset_dims = inner_dim // offset_groups

        self.downsample_factor = downsample_factor

        # 创建偏移量模块
        self.to_offsets = nn.Sequential(
            nn.Conv2d(offset_dims, offset_dims, offset_kernel_size, groups = offset_dims, stride = downsample_factor, padding = (offset_kernel_size - downsample_factor) // 2),
            nn.GELU(),
            nn.Conv2d(offset_dims, 2, 1, bias = False),
            nn.Tanh(),
            Scale(offset_scale)
        )

        # 创建相对位置偏置模块
        self.rel_pos_bias = CPB(dim // 4, offset_groups = offset_groups, heads = heads, depth = 2)

        self.dropout = nn.Dropout(dropout)
        # 创建查询转换模块
        self.to_q = nn.Conv2d(dim, inner_dim, 1, groups = offset_groups if group_queries else 1, bias = False)
        # 创建键转换模块
        self.to_k = nn.Conv2d(dim, inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = False)
        # 创建值转换模块
        self.to_v = nn.Conv2d(dim, inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = False)
        # 创建输出转换模块
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x, return_vgrid = False):
        """
        b - batch
        h - heads
        x - height
        y - width
        d - dimension
        g - offset groups
        """

        heads, b, h, w, downsample_factor, device = self.heads, x.shape[0], *x.shape[-2:], self.downsample_factor, x.device

        # queries

        q = self.to_q(x)

        # 计算偏移量 - 偏移 MLP 在所有组中共享

        group = lambda t: rearrange(t, 'b (g d) ... -> (b g) d ...', g = self.offset_groups)

        grouped_queries = group(q)
        offsets = self.to_offsets(grouped_queries)

        # 计算网格 + 偏移量

        grid = create_grid_like(offsets)
        vgrid = grid + offsets

        vgrid_scaled = normalize_grid(vgrid)

        kv_feats = F.grid_sample(
            group(x),
            vgrid_scaled,
            mode = 'bilinear', padding_mode = 'zeros', align_corners = False)

        kv_feats = rearrange(kv_feats, '(b g) d ... -> b (g d) ...', b = b)

        # 推导键/值

        k, v = self.to_k(kv_feats), self.to_v(kv_feats)

        # 缩放查询

        q = q * self.scale

        # 分割头部

        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h = heads), (q, k, v))

        # 查询/键相似度

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # 相对位置偏置

        grid = create_grid_like(x)
        grid_scaled = normalize_grid(grid, dim = 0)
        rel_pos_bias = self.rel_pos_bias(grid_scaled, vgrid_scaled)
        sim = sim + rel_pos_bias

        # 数值稳定性

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()

        # 注意力

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # 聚合和组合头部

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        out = self.to_out(out)

        if return_vgrid:
            return out, vgrid

        return out
```