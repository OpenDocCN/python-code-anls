# `.\lucidrains\deformable-attention\deformable_attention\deformable_attention_3d.py`

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

# 检查是否可以被整除
def divisible_by(numer, denom):
    return (numer % denom) == 0

# 将输入转换为元组
def cast_tuple(x, length = 1):
    return x if isinstance(x, tuple) else ((x,) * depth)

# tensor helpers

# 创建与输入张量相似的网格
def create_grid_like(t, dim = 0):
    f, h, w, device = *t.shape[-3:], t.device

    grid = torch.stack(torch.meshgrid(
        torch.arange(f, device = device),
        torch.arange(h, device = device),
        torch.arange(w, device = device),
    indexing = 'ij'), dim = dim)

    grid.requires_grad = False
    grid = grid.type_as(t)
    return grid

# 将网格归一化到-1到1的范围
def normalize_grid(grid, dim = 1, out_dim = -1):
    f, h, w = grid.shape[-3:]
    grid_f, grid_h, grid_w = grid.unbind(dim = dim)

    grid_f = 2.0 * grid_f / max(f - 1, 1) - 1.0
    grid_h = 2.0 * grid_h / max(h - 1, 1) - 1.0
    grid_w = 2.0 * grid_w / max(w - 1, 1) - 1.0

    return torch.stack((grid_f, grid_h, grid_w), dim = out_dim)

# 缩放层
class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.register_buffer('scale', torch.tensor(scale, dtype = torch.float32))

    def forward(self, x):
        return x * rearrange(self.scale, 'c -> 1 c 1 1 1')

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
            nn.Linear(3, dim),
            nn.ReLU()
        ))

        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU()
            ))

        self.mlp.append(nn.Linear(dim, heads // offset_groups))

    def forward(self, grid_q, grid_kv):
        device, dtype = grid_q.device, grid_kv.dtype

        grid_q = rearrange(grid_q, '... c -> 1 (...) c')
        grid_kv = rearrange(grid_kv, 'b ... c -> b (...) c')

        pos = rearrange(grid_q, 'b i c -> b i 1 c') - rearrange(grid_kv, 'b j c -> b 1 j c')
        bias = torch.sign(pos) * torch.log(pos.abs() + 1)  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)

        for layer in self.mlp:
            bias = layer(bias)

        bias = rearrange(bias, '(b g) i j o -> b (g o) i j', g = self.offset_groups)

        return bias

# main class

# 可变形注意力机制
class DeformableAttention3D(nn.Module):
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
        # 将下采样因子转换为元组，长度为3
        downsample_factor = cast_tuple(downsample_factor, length = 3)
        # 设置偏移比例，默认为下采样因子
        offset_scale = default(offset_scale, downsample_factor)

        # 计算偏移卷积填充
        offset_conv_padding = tuple(map(lambda x: (x[0] - x[1]) / 2, zip(offset_kernel_size, downsample_factor)))
        # 断言偏移卷积填充大于0且为整数
        assert all([(padding > 0 and padding.is_integer()) for padding in offset_conv_padding])

        # 设置偏移组数，默认为头数
        offset_groups = default(offset_groups, heads)
        # 断言头数可被偏移组数整除
        assert divisible_by(heads, offset_groups)

        # 计算内部维度
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.offset_groups = offset_groups

        offset_dims = inner_dim // offset_groups

        self.downsample_factor = downsample_factor

        # 定义偏移量网络
        self.to_offsets = nn.Sequential(
            nn.Conv3d(offset_dims, offset_dims, offset_kernel_size, groups = offset_dims, stride = downsample_factor, padding = tuple(map(int, offset_conv_padding))),
            nn.GELU(),
            nn.Conv3d(offset_dims, 3, 1, bias = False),
            nn.Tanh(),
            Scale(offset_scale)
        )

        # 定义相对位置偏置
        self.rel_pos_bias = CPB(dim // 4, offset_groups = offset_groups, heads = heads, depth = 2)

        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Conv3d(dim, inner_dim, 1, groups = offset_groups if group_queries else 1, bias = False)
        self.to_k = nn.Conv3d(dim, inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = False)
        self.to_v = nn.Conv3d(dim, inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = False)
        self.to_out = nn.Conv3d(inner_dim, dim, 1)

    def forward(self, x, return_vgrid = False):
        """
        b - batch
        h - heads
        f - frames
        x - height
        y - width
        d - dimension
        g - offset groups
        """

        heads, b, f, h, w, downsample_factor, device = self.heads, x.shape[0], *x.shape[-3:], self.downsample_factor, x.device

        # queries

        q = self.to_q(x)

        # 计算偏移量 - 偏移MLP在所有组中共享

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
        out = rearrange(out, 'b h (f x y) d -> b (h d) f x y', f = f, x = h, y = w)
        out = self.to_out(out)

        if return_vgrid:
            return out, vgrid

        return out
```