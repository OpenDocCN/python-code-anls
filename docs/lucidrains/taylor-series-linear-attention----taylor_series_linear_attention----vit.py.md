# `.\lucidrains\taylor-series-linear-attention\taylor_series_linear_attention\vit.py`

```py
# 从 math 模块中导入 sqrt 函数
from math import sqrt

# 导入 torch 库
import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F

# 导入 einops 库中的 rearrange 和 repeat 函数
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

# 导入自定义的注意力模块
from taylor_series_linear_attention.attention import (
    TaylorSeriesLinearAttn,
    ChannelFirstTaylorSeriesLinearAttn
)

# 定义函数 posemb_sincos_2d，用于生成二维的正弦余弦位置编码
def posemb_sincos_2d(
    h, w,
    dim,
    temperature: int = 10000,
    dtype = torch.float32
):
    # 生成网格坐标
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing = "ij")
    # 确保特征维度是4的倍数
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"

    dim //= 4
    omega = torch.arange(dim) / (dim - 1)
    omega = temperature ** -omega

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

# 定义深度可分离卷积函数 DepthWiseConv2d
def DepthWiseConv2d(
    dim_in,
    dim_out,
    kernel_size,
    padding,
    stride = 1,
    bias = True
):
    return nn.Sequential(
        nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
        nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
    )

# 定义前馈神经网络类 FeedForward
class FeedForward(Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        dim_hidden = int(dim * mult)
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim_hidden, 1),
            nn.Hardswish(),
            DepthWiseConv2d(dim_hidden, dim_hidden, 3, padding = 1),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Conv2d(dim_hidden, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        h = w = int(sqrt(x.shape[-2]))
        x = rearrange(x, 'b (h w) c -> b c h w', h = h, w = w)
        x = self.net(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x

# 定义 Transformer 类
class Transformer(Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        ff_mult,
        dropout = 0.
    ):
        super().__init__()

        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                TaylorSeriesLinearAttn(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                nn.LayerNorm(dim),
                FeedForward(dim, ff_mult, dropout = dropout)
            ]))

    def forward(self, x):
        for attn_norm, attn, ff_norm, ff in self.layers:
            x = attn(attn_norm(x)) + x
            x = ff(ff_norm(x)) + x
        return x

# 定义主类 ViT
class ViT(Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        ff_mult = 4,
        heads = 16,
        channels = 3,
        dim_head = 8,
        dropout = 0.,
        emb_dropout = 0.
    ):  # 定义一个类，继承自 nn.Module
        super().__init__()  # 调用父类的构造函数
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size.'  # 断言图片尺寸必须能够被分块尺寸整除
        num_patches = (image_size // patch_size) ** 2  # 计算总的分块数量
        patch_dim = channels * patch_size ** 2  # 计算每个分块的维度

        self.to_patch_embedding = nn.Sequential(  # 定义一个序列模块，用于将图像转换为分块嵌入
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),  # 重新排列输入张量的维度
            nn.LayerNorm(patch_dim),  # 对每个分块进行 LayerNorm
            nn.Linear(patch_dim, dim),  # 线性变换将每个分块的维度映射到指定维度
            nn.LayerNorm(dim),  # 对映射后的维度进行 LayerNorm
        )

        self.register_buffer('pos_embedding', posemb_sincos_2d(  # 注册一个缓冲区，存储位置编码
            h = image_size // patch_size,  # 图像高度上的分块数量
            w = image_size // patch_size,  # 图像宽度上的分块数量
            dim = dim,  # 位置编码的维度
        ), persistent = False)  # 设置缓冲区为非持久性的

        self.dropout = nn.Dropout(emb_dropout)  # 定义一个 Dropout 层，用于在嵌入层上进行随机失活

        self.transformer = Transformer(dim, depth, heads, dim_head, ff_mult, dropout)  # 定义一个 Transformer 模型

        self.mlp_head = nn.Sequential(  # 定义一个序列模块，用于最终的 MLP 头部分类
            Reduce('b n d -> b d', 'mean'),  # 对输入张量进行维度缩减，计算均值
            nn.LayerNorm(dim),  # 对均值后的张量进行 LayerNorm
            nn.Linear(dim, num_classes)  # 线性变换将维度映射到类别数量
        )

    def forward(self, img):  # 定义前向传播函数，接收输入图像
        x = self.to_patch_embedding(img)  # 将输入图像转换为分块嵌入
        x = x + self.pos_embedding  # 添加位置编码到嵌入中
        x = self.dropout(x)  # 对嵌入进行随机失活

        x = self.transformer(x)  # 使用 Transformer 模型进行特征提取和交互

        return self.mlp_head(x)  # 使用 MLP 头部对特征进行分类
```