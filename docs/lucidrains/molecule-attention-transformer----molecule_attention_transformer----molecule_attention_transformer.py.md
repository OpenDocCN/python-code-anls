# `.\lucidrains\molecule-attention-transformer\molecule_attention_transformer\molecule_attention_transformer.py`

```py
# 导入 torch 库
import torch
# 导入 torch 中的函数库
import torch.nn.functional as F
# 导入 functools 库中的 partial 函数
from functools import partial
# 从 torch 库中导入 nn 模块和 einsum 函数
from torch import nn, einsum
# 从 einops 库中导入 rearrange 函数
from einops import rearrange

# 常量

# 定义不同距离核函数的字典
DIST_KERNELS = {
    'exp': {
        'fn': lambda t: torch.exp(-t),
        'mask_value_fn': lambda t: torch.finfo(t.dtype).max
    },
    'softmax': {
        'fn': lambda t: torch.softmax(t, dim = -1),
        'mask_value_fn': lambda t: -torch.finfo(t.dtype).max
    }
}

# 辅助函数

# 判断变量是否存在的函数
def exists(val):
    return val is not None

# 返回默认值的函数
def default(val, d):
    return d if not exists(val) else val

# 辅助类

# 残差连接类
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)

# 预层归一化类
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# 前馈神经网络类
class FeedForward(nn.Module):
    def __init__(self, dim, dim_out = None, mult = 4):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim_out)
        )

    def forward(self, x):
        return self.net(x)

# 注意力机制类
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, Lg = 0.5, Ld = 0.5, La = 1, dist_kernel_fn = 'exp'):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads= heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        # 控制加权线性组合的超参数
        self.La = La
        self.Ld = Ld
        self.Lg = Lg

        self.dist_kernel_fn = dist_kernel_fn

    def forward(self, x, mask = None, adjacency_mat = None, distance_mat = None):
        h, La, Ld, Lg, dist_kernel_fn = self.heads, self.La, self.Ld, self.Lg, self.dist_kernel_fn

        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (h qkv d) -> b h n qkv d', h = h, qkv = 3).unbind(dim = -2)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        assert dist_kernel_fn in DIST_KERNELS, f'distance kernel function needs to be one of {DISTANCE_KERNELS.keys()}'
        dist_kernel_config = DIST_KERNELS[dist_kernel_fn]

        if exists(distance_mat):
            distance_mat = rearrange(distance_mat, 'b i j -> b () i j')

        if exists(adjacency_mat):
            adjacency_mat = rearrange(adjacency_mat, 'b i j -> b () i j')

        if exists(mask):
            mask_value = torch.finfo(dots.dtype).max
            mask = mask[:, None, :, None] * mask[:, None, None, :]

            # 屏蔽注意力
            dots.masked_fill_(~mask, -mask_value)

            if exists(distance_mat):
                # 将距离屏蔽为无穷大
                # 待办事项 - 确保对于 softmax 距离核函数，使用 -无穷大
                dist_mask_value = dist_kernel_config['mask_value_fn'](dots)
                distance_mat.masked_fill_(~mask, dist_mask_value)

            if exists(adjacency_mat):
                adjacency_mat.masked_fill_(~mask, 0.)

        attn = dots.softmax(dim = -1)

        # 从邻接矩阵和距离矩阵中汇总贡献
        attn = attn * La

        if exists(adjacency_mat):
            attn = attn + Lg * adjacency_mat

        if exists(distance_mat):
            distance_mat = dist_kernel_config['fn'](distance_mat)
            attn = attn + Ld * distance_mat

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# 主类

class MAT(nn.Module):
    # 初始化函数，设置模型的参数和层结构
    def __init__(
        self,
        *,
        dim_in,  # 输入维度
        model_dim,  # 模型维度
        dim_out,  # 输出维度
        depth,  # 模型深度
        heads = 8,  # 多头注意力机制的头数
        Lg = 0.5,  # 注意力机制中的参数
        Ld = 0.5,  # 注意力机制中的参数
        La = 1,  # 注意力机制中的参数
        dist_kernel_fn = 'exp'  # 距离核函数类型
    ):
        # 调用父类的初始化函数
        super().__init__()

        # 输入到模型的线性变换层
        self.embed_to_model = nn.Linear(dim_in, model_dim)
        # 模型的层列表
        self.layers = nn.ModuleList([])

        # 根据深度循环创建模型的每一层
        for _ in range(depth):
            # 每一层包含一个残差连接和一个预层归一化的注意力机制
            # 以及一个残差连接和一个预层归一化的前馈神经网络
            layer = nn.ModuleList([
                Residual(PreNorm(model_dim, Attention(model_dim, heads = heads, Lg = Lg, Ld = Ld, La = La, dist_kernel_fn = dist_kernel_fn))),
                Residual(PreNorm(model_dim, FeedForward(model_dim)))
            ])
            self.layers.append(layer)

        # 输出的归一化层
        self.norm_out = nn.LayerNorm(model_dim)
        # 输出的前馈神经网络
        self.ff_out = FeedForward(model_dim, dim_out)

    # 前向传播函数
    def forward(
        self,
        x,  # 输入数据
        mask = None,  # 掩码
        adjacency_mat = None,  # 邻接矩阵
        distance_mat = None  # 距离矩阵
    ):
        # 将输入数据进行线性变换
        x = self.embed_to_model(x)

        # 遍历模型的每一层，依次进行注意力机制和前馈神经网络操作
        for (attn, ff) in self.layers:
            x = attn(
                x,
                mask = mask,
                adjacency_mat = adjacency_mat,
                distance_mat = distance_mat
            )
            x = ff(x)

        # 对输出进行归一化
        x = self.norm_out(x)
        # 沿着指定维度求均值
        x = x.mean(dim = -2)
        # 输出的前馈神经网络
        x = self.ff_out(x)
        return x
```