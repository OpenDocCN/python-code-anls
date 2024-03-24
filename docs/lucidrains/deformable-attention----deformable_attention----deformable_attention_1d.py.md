# `.\lucidrains\deformable-attention\deformable_attention\deformable_attention_1d.py`

```py
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops.layers.torch import Rearrange
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

# tensor helpers

# 1维网格采样，将网格重塑为2维
def grid_sample_1d(feats, grid, *args, **kwargs):
    grid = rearrange(grid, '... -> ... 1 1')
    grid = F.pad(grid, (0, 1), value = 0.)
    feats = rearrange(feats, '... -> ... 1')
    out = F.grid_sample(feats, grid, **kwargs)
    return rearrange(out, '... 1 -> ...')

# 将1维序列归一化到-1到1的范围
def normalize_grid(arange, dim = 1, out_dim = -1):
    n = arange.shape[-1]
    return 2.0 * arange / max(n - 1, 1) - 1.0

# 缩放层
class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

# 从SwinV2获取连续位置偏差

class CPB(nn.Module):
    """ https://arxiv.org/abs/2111.09883v1 """

    def __init__(self, dim, *, heads, offset_groups, depth, log_distance = True):
        super().__init__()
        self.heads = heads
        self.offset_groups = offset_groups
        self.log_distance = log_distance

        self.mlp = nn.ModuleList([])

        self.mlp.append(nn.Sequential(
            nn.Linear(1, dim),
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

        grid_q = rearrange(grid_q, 'n -> 1 n')
        grid_kv = rearrange(grid_kv, 'b n -> b n')

        pos = rearrange(grid_q, 'b i -> b i 1 1') - rearrange(grid_kv, 'b j -> b 1 j 1')

        if self.log_distance:
            pos = torch.sign(pos) * torch.log(pos.abs() + 1)  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)

        bias = pos

        for layer in self.mlp:
            bias = layer(bias)

        bias = rearrange(bias, '(b g) i j o -> b (g o) i j', g = self.offset_groups)

        return bias

# main class

class DeformableAttention1D(nn.Module):
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
        cpb_log_distance = True,
        group_queries = True,
        group_key_values = True
    ):
        # 调用父类的构造函数
        super().__init__()
        # 设置偏移比例，默认为下采样因子
        offset_scale = default(offset_scale, downsample_factor)
        # 断言偏移核大小必须大于或等于下采样因子
        assert offset_kernel_size >= downsample_factor, 'offset kernel size must be greater than or equal to the downsample factor'
        # 断言偏移核大小减去下采样因子必须是2的倍数
        assert divisible_by(offset_kernel_size - downsample_factor, 2)

        # 设置偏移组数，默认为头数
        offset_groups = default(offset_groups, heads)
        # 断言头数必须是偏移组数的倍数
        assert divisible_by(heads, offset_groups)

        # 计算内部维度
        inner_dim = dim_head * heads
        # 设置缩放因子
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.offset_groups = offset_groups

        # 计算偏移维度
        offset_dims = inner_dim // offset_groups

        self.downsample_factor = downsample_factor

        # 构建偏移网络
        self.to_offsets = nn.Sequential(
            nn.Conv1d(offset_dims, offset_dims, offset_kernel_size, groups = offset_dims, stride = downsample_factor, padding = (offset_kernel_size - downsample_factor) // 2),
            nn.GELU(),
            nn.Conv1d(offset_dims, 1, 1, bias = False),
            Rearrange('b 1 n -> b n'),
            nn.Tanh(),
            Scale(offset_scale)
        )

        # 构建相对位置偏置
        self.rel_pos_bias = CPB(dim // 4, offset_groups = offset_groups, heads = heads, depth = 2, log_distance = cpb_log_distance)

        self.dropout = nn.Dropout(dropout)
        # 构建查询转换层
        self.to_q = nn.Conv1d(dim, inner_dim, 1, groups = offset_groups if group_queries else 1, bias = False)
        # 构建键转换层
        self.to_k = nn.Conv1d(dim, inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = False)
        # 构建值转换层
        self.to_v = nn.Conv1d(dim, inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = False)
        # 构建输出转换层
        self.to_out = nn.Conv1d(inner_dim, dim, 1)

    def forward(self, x, return_vgrid = False):
        """
        b - batch
        h - heads
        n - sequence dimension
        d - dimension
        g - offset groups
        """

        heads, b, n, downsample_factor, device = self.heads, x.shape[0], x.shape[-1], self.downsample_factor, x.device

        # queries

        q = self.to_q(x)

        # calculate offsets - offset MLP shared across all groups

        group = lambda t: rearrange(t, 'b (g d) n -> (b g) d n', g = self.offset_groups)

        grouped_queries = group(q)
        offsets = self.to_offsets(grouped_queries)

        # calculate grid + offsets

        grid = torch.arange(offsets.shape[-1], device = device)
        vgrid = grid + offsets
        vgrid_scaled = normalize_grid(vgrid)

        kv_feats = grid_sample_1d(
            group(x),
            vgrid_scaled,
        mode = 'bilinear', padding_mode = 'zeros', align_corners = False)

        kv_feats = rearrange(kv_feats, '(b g) d n -> b (g d) n', b = b)

        # derive key / values

        k, v = self.to_k(kv_feats), self.to_v(kv_feats)

        # scale queries

        q = q * self.scale

        # split out heads

        q, k, v = map(lambda t: rearrange(t, 'b (h d) n -> b h n d', h = heads), (q, k, v))

        # query / key similarity

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # relative positional bias

        seq_range = torch.arange(n, device = device)
        seq_scaled = normalize_grid(seq_range, dim = 0)
        rel_pos_bias = self.rel_pos_bias(seq_scaled, vgrid_scaled)
        sim = sim + rel_pos_bias

        # numerical stability

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate and combine heads

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b (h d) n')
        out = self.to_out(out)

        if return_vgrid:
            return out, vgrid

        return out
```