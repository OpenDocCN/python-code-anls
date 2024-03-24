# `.\lucidrains\nystrom-attention\nystrom_attention\nystrom_attention.py`

```py
# 从 math 模块中导入 ceil 函数
from math import ceil
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块和 einsum 函数
from torch import nn, einsum
# 从 torch.nn.functional 中导入 F 模块

import torch.nn.functional as F

# 从 einops 库中导入 rearrange 和 reduce 函数

from einops import rearrange, reduce

# 定义一个辅助函数 exists，用于判断变量是否存在
def exists(val):
    return val is not None

# 定义 Moore-Penrose 伪逆的迭代计算函数
def moore_penrose_iter_pinv(x, iters = 6):
    # 获取输入张量 x 的设备信息
    device = x.device

    # 计算 x 的绝对值
    abs_x = torch.abs(x)
    # 沿着最后一个维度求和，得到列和
    col = abs_x.sum(dim = -1)
    # 沿着倒数第二个维度求和，得到行和
    row = abs_x.sum(dim = -2)
    # 对 x 进行重排，转置操作
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    # 创建单位矩阵
    I = torch.eye(x.shape[-1], device = device)
    I = rearrange(I, 'i j -> () i j')

    # 迭代计算 Moore-Penrose 伪逆
    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz))))

    return z

# 主要的注意力类 NystromAttention
class NystromAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        num_landmarks = 256,
        pinv_iterations = 6,
        residual = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head ** -0.5
        # 定义一个线性层，用于将输入维度转换为内部维度的三倍
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        # 定义输出层，包含一个线性层和一个 dropout 层
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.residual = residual
        # 如果启用残差连接
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            # 定义一个卷积层，用于残差连接
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)
    # 定义前向传播函数，接受输入 x，mask 和 return_attn 参数
    def forward(self, x, mask = None, return_attn = False):
        # 解包 x 的形状信息，包括 batch size (b), 序列长度 (n), 头数 (h), 地标数 (m), 伪逆迭代次数 (iters), 以及 epsilon (eps)
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        # 将序列填充，使其可以被均匀地分成 m 个地标
        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value = 0)

            if exists(mask):
                mask = F.pad(mask, (padding, 0), value = False)

        # 派生查询、键、值
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # 将查询、键、值中的掩码位置设为 0
        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        # 通过求和缩减生成地标，然后使用掩码计算均值
        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l = l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l = l)

        # 计算地标掩码，并准备计算掩码均值时的非掩码元素总和
        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l = l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        # 如果存在掩码，则进行掩码均值计算
        q_landmarks = q_landmarks / divisor
        k_landmarks = k_landmarks / divisor

        # 相似度计算
        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)

        # 掩码处理
        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        # 计算公式 (15) 中的等式，并聚合值
        attn1, attn2, attn3 = map(lambda t: t.softmax(dim = -1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        # 添加值的深度卷积残差
        if self.residual:
            out = out + self.res_conv(v)

        # 合并和组合头
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)
        out = out[:, -n:]

        # 如果需要返回注意力权重，则返回输出和注意力权重
        if return_attn:
            attn = attn1 @ attn2_inv @ attn3
            return out, attn

        return out
# transformer

# 定义一个预标准化层，包含一个 LayerNorm 层和一个传入的函数
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 初始化 LayerNorm 层
        self.fn = fn  # 保存传入的函数

    def forward(self, x, **kwargs):
        x = self.norm(x)  # 对输入数据进行标准化
        return self.fn(x, **kwargs)  # 调用传入的函数处理标准化后的数据

# 定义一个前馈神经网络层，包含线性层、GELU 激活函数、Dropout 和另一个线性层
class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),  # 第一个线性层
            nn.GELU(),  # GELU 激活函数
            nn.Dropout(dropout),  # Dropout 层
            nn.Linear(dim * mult, dim)  # 第二个线性层
        )

    def forward(self, x):
        return self.net(x)  # 前馈神经网络的前向传播

# 定义一个 Nystromformer 模型，包含多个层
class Nystromformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_landmarks = 256,
        pinv_iterations = 6,
        attn_values_residual = True,
        attn_values_residual_conv_kernel = 33,
        attn_dropout = 0.,
        ff_dropout = 0.   
    ):
        super().__init__()

        self.layers = nn.ModuleList([])  # 初始化一个空的 ModuleList
        for _ in range(depth):
            # 每一层包含一个 NystromAttention 层和一个 FeedForward 层，都经过预标准化
            self.layers.append(nn.ModuleList([
                PreNorm(dim, NystromAttention(dim = dim, dim_head = dim_head, heads = heads, num_landmarks = num_landmarks, pinv_iterations = pinv_iterations, residual = attn_values_residual, residual_conv_kernel = attn_values_residual_conv_kernel, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim = dim, dropout = ff_dropout))
            ]))

    def forward(self, x, mask = None):
        # 遍历每一层，依次进行注意力计算和前馈神经网络处理
        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x  # 注意力计算后加上残差连接
            x = ff(x) + x  # 前馈神经网络处理后加上残差连接
        return x  # 返回处理后的数据
```