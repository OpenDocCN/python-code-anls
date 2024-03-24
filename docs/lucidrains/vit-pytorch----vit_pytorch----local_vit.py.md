# `.\lucidrains\vit-pytorch\vit_pytorch\local_vit.py`

```
# 从 math 模块中导入 sqrt 函数
from math import sqrt
# 导入 torch 模块
import torch
# 从 torch 模块中导入 nn, einsum
from torch import nn, einsum
# 从 torch.nn 模块中导入 functional 模块
import torch.nn.functional as F

# 从 einops 模块中导入 rearrange, repeat
from einops import rearrange, repeat
# 从 einops.layers.torch 模块中导入 Rearrange 类

# classes

# 定义 Residual 类，继承自 nn.Module
class Residual(nn.Module):
    # 初始化函数
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    # 前向传播函数
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 定义 ExcludeCLS 类，继承自 nn.Module
class ExcludeCLS(nn.Module):
    # 初始化函数
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    # 前向传播函数
    def forward(self, x, **kwargs):
        cls_token, x = x[:, :1], x[:, 1:]
        x = self.fn(x, **kwargs)
        return torch.cat((cls_token, x), dim = 1)

# feed forward related classes

# 定义 DepthWiseConv2d 类，继承自 nn.Module
class DepthWiseConv2d(nn.Module):
    # 初始化函数
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    # 前向传播函数
    def forward(self, x):
        return self.net(x)

# 定义 FeedForward 类，继承自 nn.Module
class FeedForward(nn.Module):
    # 初始化函数
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Conv2d(dim, hidden_dim, 1),
            nn.Hardswish(),
            DepthWiseConv2d(hidden_dim, hidden_dim, 3, padding = 1),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, 1),
            nn.Dropout(dropout)
        )
    # 前向传播函数
    def forward(self, x):
        h = w = int(sqrt(x.shape[-2]))
        x = rearrange(x, 'b (h w) c -> b c h w', h = h, w = w)
        x = self.net(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x

# attention

# 定义 Attention 类，继承自 nn.Module
class Attention(nn.Module):
    # 初始化函数
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    # 前向传播函数
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads

        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# 定义 Transformer 类，继承自 nn.Module
class Transformer(nn.Module):
    # 初始化函数
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                ExcludeCLS(Residual(FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    # 前向传播函数
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x

# main class

# 定义 LocalViT 类，继承自 nn.Module
class LocalViT(nn.Module):
    # 初始化函数，设置模型参数和层结构
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        # 调用父类的初始化函数
        super().__init__()
        # 检查图像尺寸是否能被分块尺寸整除
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        # 计算图像分块数量
        num_patches = (image_size // patch_size) ** 2
        # 计算每个分块的维度
        patch_dim = channels * patch_size ** 2

        # 定义将图像分块转换为嵌入向量的层序列
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # 初始化位置编码参数
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # 初始化类别标记参数
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # 初始化丢弃层
        self.dropout = nn.Dropout(emb_dropout)

        # 初始化 Transformer 模型
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # 定义 MLP 头部层序列
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    # 前向传播函数
    def forward(self, img):
        # 将图像转换为分块嵌入向量
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # 重复类别标记以匹配批次大小
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # 将类别标记和分块嵌入向量拼接在一起
        x = torch.cat((cls_tokens, x), dim=1)
        # 添加位置编码
        x += self.pos_embedding[:, :(n + 1)]
        # 对结果进行丢弃
        x = self.dropout(x)

        # 使用 Transformer 进行特征变换
        x = self.transformer(x)

        # 返回 MLP 头部的输出结果
        return self.mlp_head(x[:, 0])
```