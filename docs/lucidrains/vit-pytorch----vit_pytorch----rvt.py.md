# `.\lucidrains\vit-pytorch\vit_pytorch\rvt.py`

```
# 从 math 模块中导入 sqrt, pi, log 函数
# 从 torch 模块中导入 nn, einsum, F
# 从 einops 模块中导入 rearrange, repeat
# 从 einops.layers.torch 模块中导入 Rearrange 类
from math import sqrt, pi, log

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# 旋转嵌入

# 将输入张量中的每两个元素进行旋转
def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

# 轴向旋转嵌入类
class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq = 10):
        super().__init__()
        self.dim = dim
        scales = torch.linspace(1., max_freq / 2, self.dim // 4)
        self.register_buffer('scales', scales)

    def forward(self, x):
        device, dtype, n = x.device, x.dtype, int(sqrt(x.shape[-2]))

        seq = torch.linspace(-1., 1., steps = n, device = device)
        seq = seq.unsqueeze(-1)

        scales = self.scales[(*((None,) * (len(seq.shape) - 1)), Ellipsis]
        scales = scales.to(x)

        seq = seq * scales * pi

        x_sinu = repeat(seq, 'i d -> i j d', j = n)
        y_sinu = repeat(seq, 'j d -> i j d', i = n)

        sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim = -1)
        cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim = -1)

        sin, cos = map(lambda t: rearrange(t, 'i j d -> (i j) d'), (sin, cos))
        sin, cos = map(lambda t: repeat(t, 'n d -> () n (d j)', j = 2), (sin, cos))
        return sin, cos

# 深度可分离卷积类
class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

# 辅助类

# 空间卷积类
class SpatialConv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel, bias = False):
        super().__init__()
        self.conv = DepthWiseConv2d(dim_in, dim_out, kernel, padding = kernel // 2, bias = False)
        self.cls_proj = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()

    def forward(self, x, fmap_dims):
        cls_token, x = x[:, :1], x[:, 1:]
        x = rearrange(x, 'b (h w) d -> b d h w', **fmap_dims)
        x = self.conv(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        cls_token = self.cls_proj(cls_token)
        return torch.cat((cls_token, x), dim = 1)

# GEGLU 类
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return F.gelu(gates) * x

# 前馈网络类
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., use_glu = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim * 2 if use_glu else hidden_dim),
            GEGLU() if use_glu else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# 注意力类
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., use_rotary = True, use_ds_conv = True, conv_query_kernel = 5):
        super().__init__()
        inner_dim = dim_head *  heads
        self.use_rotary = use_rotary
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.use_ds_conv = use_ds_conv

        self.to_q = SpatialConv(dim, inner_dim, conv_query_kernel, bias = False) if use_ds_conv else nn.Linear(dim, inner_dim, bias = False)

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    # 定义前向传播函数，接受输入 x，位置嵌入 pos_emb，特征图维度 fmap_dims
    def forward(self, x, pos_emb, fmap_dims):
        # 获取输入 x 的形状信息
        b, n, _, h = *x.shape, self.heads

        # 如果使用深度可分离卷积，则传递特定参数给 to_q 函数
        to_q_kwargs = {'fmap_dims': fmap_dims} if self.use_ds_conv else {}

        # 对输入 x 进行归一化处理
        x = self.norm(x)

        # 将 x 传递给 to_q 函数，得到查询向量 q
        q = self.to_q(x, **to_q_kwargs)

        # 将 q 与键值对应的结果拆分为 q, k, v
        qkv = (q, *self.to_kv(x).chunk(2, dim = -1))

        # 将 q, k, v 重排维度，以适应多头注意力机制
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), qkv)

        # 如果使用旋转注意力机制
        if self.use_rotary:
            # 对查询和键应用二维旋转嵌入，不包括 CLS 标记
            sin, cos = pos_emb
            dim_rotary = sin.shape[-1]

            # 拆分 CLS 标记和其余部分
            (q_cls, q), (k_cls, k) = map(lambda t: (t[:, :1], t[:, 1:]), (q, k))

            # 处理旋转维度小于头维度的情况
            (q, q_pass), (k, k_pass) = map(lambda t: (t[..., :dim_rotary], t[..., dim_rotary:]), (q, k))
            q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
            q, k = map(lambda t: torch.cat(t, dim = -1), ((q, q_pass), (k, k_pass)))

            # 拼接回 CLS 标记
            q = torch.cat((q_cls, q), dim = 1)
            k = torch.cat((k_cls, k), dim = 1)

        # 计算点积注意力得分
        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # 经过注意力计算
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # 计算输出
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        # 返回输出结果
        return self.to_out(out)
# 定义一个 Transformer 类，继承自 nn.Module
class Transformer(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, image_size, dropout = 0., use_rotary = True, use_ds_conv = True, use_glu = True):
        # 调用父类的初始化函数
        super().__init__()
        # 初始化一个空的层列表
        self.layers = nn.ModuleList([])
        # 创建 AxialRotaryEmbedding 对象作为位置编码
        self.pos_emb = AxialRotaryEmbedding(dim_head, max_freq = image_size)
        # 循环创建指定数量的层
        for _ in range(depth):
            # 每层包含注意力机制和前馈神经网络
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, use_rotary = use_rotary, use_ds_conv = use_ds_conv),
                FeedForward(dim, mlp_dim, dropout = dropout, use_glu = use_glu)
            ]))
    # 前向传播函数，接受输入 x 和 fmap_dims
    def forward(self, x, fmap_dims):
        # 计算位置编码
        pos_emb = self.pos_emb(x[:, 1:])
        # 遍历每一层，依次进行注意力机制和前馈神经网络操作
        for attn, ff in self.layers:
            x = attn(x, pos_emb = pos_emb, fmap_dims = fmap_dims) + x
            x = ff(x) + x
        # 返回处理后的结果
        return x

# 定义一个 RvT 类，继承自 nn.Module
class RvT(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., use_rotary = True, use_ds_conv = True, use_glu = True):
        # 调用父类的初始化函数
        super().__init__()
        # 断言确保图像尺寸能够被补丁大小整除
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        # 计算补丁数量和补丁维度
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        # 初始化补丁嵌入层
        self.patch_size = patch_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        # 初始化分类令牌
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # 初始化 Transformer 模型
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, image_size, dropout, use_rotary, use_ds_conv, use_glu)

        # 初始化 MLP 头部
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    # 前向传播函数，接受输入图像 img
    def forward(self, img):
        # 获取输入图像的形状信息
        b, _, h, w, p = *img.shape, self.patch_size

        # 将图像转换为补丁嵌入
        x = self.to_patch_embedding(img)
        n = x.shape[1]

        # 重复分类令牌并与补丁嵌入拼接
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)

        # 计算特征图尺寸信息
        fmap_dims = {'h': h // p, 'w': w // p}
        # 使用 Transformer 处理输入数据
        x = self.transformer(x, fmap_dims = fmap_dims)

        # 返回 MLP 头部处理后的结果
        return self.mlp_head(x[:, 0])
```