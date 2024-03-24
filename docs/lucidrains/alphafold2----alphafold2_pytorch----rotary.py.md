# `.\lucidrains\alphafold2\alphafold2_pytorch\rotary.py`

```
# 从 math 模块中导入 log, sqrt, pi 函数
# 导入 torch 模块
# 从 torch 模块中导入 nn, einsum 函数
# 从 einops 模块中导入 rearrange, repeat 函数

# 旋转嵌入的辅助函数
def rotate_every_two(x):
    # 重新排列张量 x 的维度，将最后一个维度拆分成两个维度
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    # 将 x1, x2 进行旋转操作
    x = torch.stack((-x2, x1), dim = -1)
    # 重新排列张量 x 的维度
    return rearrange(x, '... d j -> ... (d j)')

def apply_rotary_pos_emb(x, sinu_pos):
    # 将 sinu_pos 中的 sin, cos 重新排列维度
    sin, cos = map(lambda t: rearrange(t, 'b ... -> b () ...'), sinu_pos)
    rot_dim = sin.shape[-1]
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # 应用旋转位置嵌入
    x =  x * cos + rotate_every_two(x) * sin
    # 拼接处理后的张量
    return torch.cat((x, x_pass), dim = -1)

# 位置嵌入

class DepthWiseConv1d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding = 0, stride = 1, bias = True, groups = None):
        super().__init__()
        groups = default(groups, dim_in)
        # 定义深度卷积网络
        self.net = nn.Sequential(
            nn.Conv1d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = groups, stride = stride, bias = bias),
            nn.Conv1d(dim_in, dim_out, 1, bias = bias)
        )
    def forward(self, x):
        # 前向传播
        return self.net(x)

class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 计算固定位置嵌入的频率
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, n, device):
        # 生成序列
        seq = torch.arange(n, device = device).type_as(self.inv_freq)
        # 计算频率
        freqs = einsum('i , j -> i j', seq, self.inv_freq)
        freqs = repeat(freqs, 'i j -> () i (j r)', r = 2)
        return [freqs.sin(), freqs.cos()]

class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq = 10):
        super().__init__()
        self.dim = dim // 2
        # 计算轴向旋转嵌入的频率
        inv_freq = 1. / (10000 ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, n, device):
        # 生成序列
        seq = torch.arange(n, device = device).type_as(self.inv_freq)

        x = einsum('n, d -> n d', seq, self.inv_freq)
        y = einsum('n, d -> n d', seq, self.inv_freq)

        x_sinu = repeat(x, 'i d -> i j d', j = n)
        y_sinu = repeat(y, 'j d -> i j d', i = n)

        # 计算 sin, cos
        sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim = -1)
        cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim = -1)

        sin, cos = map(lambda t: repeat(t, 'i j d -> () (i j) (d r)', r = 2), (sin, cos))
        return [sin, cos]
```