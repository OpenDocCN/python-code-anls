# `.\lucidrains\TimeSformer-pytorch\timesformer_pytorch\rotary.py`

```py
# 从 math 模块中导入 log 和 pi 函数
# 从 torch 模块中导入 nn, einsum 和 F
# 从 einops 模块中导入 rearrange 和 repeat 函数
from math import log, pi
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

# 定义函数，用于将输入张量中的每两个元素进行旋转
def rotate_every_two(x):
    # 重新排列输入张量的维度，将每两个元素组成一组
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    # 将每组中的两个元素拆分为两个张量
    x1, x2 = x.unbind(dim = -1)
    # 对每组中的两个元素进行旋转操作
    x = torch.stack((-x2, x1), dim = -1)
    # 重新排列张量的维度，恢复原始形状
    return rearrange(x, '... d j -> ... (d j)')

# 定义函数，应用旋转嵌入到查询和键中
def apply_rot_emb(q, k, rot_emb):
    # 解包旋转嵌入
    sin, cos = rot_emb
    # 获取旋转维度的大小
    rot_dim = sin.shape[-1]
    # 将查询和键张量分为旋转部分和非旋转部分
    (q, q_pass), (k, k_pass) = map(lambda t: (t[..., :rot_dim], t[..., rot_dim:]), (q, k))
    # 对查询和键张量的旋转部分进行旋转操作
    q, k = map(lambda t: t * cos + rotate_every_two(t) * sin, (q, k))
    # 将旋转后的查询和键张量与非旋转部分拼接
    q, k = map(lambda t: torch.cat(t, dim = -1), ((q, q_pass), (k, k_pass)))
    return q, k

# 定义类，实现轴向旋转嵌入
class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq = 10):
        super().__init__()
        self.dim = dim
        # 计算频率范围
        scales = torch.logspace(0., log(max_freq / 2) / log(2), self.dim // 4, base = 2)
        # 将频率范围作为缓冲区存储
        self.register_buffer('scales', scales)

    def forward(self, h, w, device):
        # 重新排列频率范围的维度
        scales = rearrange(self.scales, '... -> () ...')
        # 将频率范围移动到指定设备
        scales = scales.to(device)

        # 生成高度序列
        h_seq = torch.linspace(-1., 1., steps = h, device = device)
        h_seq = h_seq.unsqueeze(-1)

        # 生成宽度序列
        w_seq = torch.linspace(-1., 1., steps = w, device = device)
        w_seq = w_seq.unsqueeze(-1)

        # 对高度和宽度序列应用频率范围和 pi
        h_seq = h_seq * scales * pi
        w_seq = w_seq * scales * pi

        # 生成正弦序列
        x_sinu = repeat(h_seq, 'i d -> i j d', j = w)
        y_sinu = repeat(w_seq, 'j d -> i j d', i = h)

        # 拼接正弦和余弦序列
        sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim = -1)
        cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim = -1)

        # 重新排列正弦和余弦序列的维度
        sin, cos = map(lambda t: rearrange(t, 'i j d -> (i j) d'), (sin, cos))
        sin, cos = map(lambda t: repeat(t, 'n d -> () n (d j)', j = 2), (sin, cos))
        return sin, cos

# 定义类，实现旋转嵌入
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 计算频率的倒数
        inv_freqs = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        # 将频率的倒数作为缓冲区存储
        self.register_buffer('inv_freqs', inv_freqs)

    def forward(self, n, device):
        # 生成序列
        seq = torch.arange(n, device = device)
        # 计算频率
        freqs = einsum('i, j -> i j', seq, self.inv_freqs)
        freqs = torch.cat((freqs, freqs), dim = -1)
        freqs = rearrange(freqs, 'n d -> () n d')
        return freqs.sin(), freqs.cos()
```