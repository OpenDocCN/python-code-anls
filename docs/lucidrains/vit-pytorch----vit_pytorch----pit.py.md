# `.\lucidrains\vit-pytorch\vit_pytorch\pit.py`

```
# 从 math 模块中导入 sqrt 函数
from math import sqrt

# 导入 torch 模块及相关子模块
import torch
from torch import nn, einsum
import torch.nn.functional as F

# 导入 einops 模块中的 rearrange 和 repeat 函数，以及 torch 子模块中的 Rearrange 类
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# 定义辅助函数

# 将输入值转换为元组，如果不是元组则重复 num 次
def cast_tuple(val, num):
    return val if isinstance(val, tuple) else (val,) * num

# 计算卷积输出大小
def conv_output_size(image_size, kernel_size, stride, padding = 0):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)

# 定义类

# 前馈神经网络类
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# 注意力机制类
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

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

# Transformer 类
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# 深度卷积，用于池化
class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_out, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

# 池化层
class Pool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.downsample = DepthWiseConv2d(dim, dim * 2, kernel_size = 3, stride = 2, padding = 1)
        self.cls_ff = nn.Linear(dim, dim * 2)

    def forward(self, x):
        cls_token, tokens = x[:, :1], x[:, 1:]

        cls_token = self.cls_ff(cls_token)

        tokens = rearrange(tokens, 'b (h w) c -> b c h w', h = int(sqrt(tokens.shape[1])))
        tokens = self.downsample(tokens)
        tokens = rearrange(tokens, 'b c h w -> b (h w) c')

        return torch.cat((cls_token, tokens), dim = 1)

# 主类
class PiT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        channels = 3
    ):  
        # 初始化函数，继承父类的初始化方法
        super().__init__()
        # 确保图像尺寸能够被分块大小整除
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        # 确保深度是一个整数元组，指定每个下采样之前的块数
        assert isinstance(depth, tuple), 'depth must be a tuple of integers, specifying the number of blocks before each downsizing'
        # 将头数转换为元组
        heads = cast_tuple(heads, len(depth))

        # 计算每个分块的维度
        patch_dim = channels * patch_size ** 2

        # 创建将图像转换为分块嵌入的序列
        self.to_patch_embedding = nn.Sequential(
            nn.Unfold(kernel_size = patch_size, stride = patch_size // 2),
            Rearrange('b c n -> b n c'),
            nn.Linear(patch_dim, dim)
        )

        # 计算输出大小和分块数量
        output_size = conv_output_size(image_size, patch_size, patch_size // 2)
        num_patches = output_size ** 2

        # 初始化位置嵌入和类别令牌
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        layers = []

        # 遍历深度和头数，构建Transformer层和池化层
        for ind, (layer_depth, layer_heads) in enumerate(zip(depth, heads)):
            not_last = ind < (len(depth) - 1)
            
            layers.append(Transformer(dim, layer_depth, layer_heads, dim_head, mlp_dim, dropout))

            if not_last:
                layers.append(Pool(dim))
                dim *= 2

        self.layers = nn.Sequential(*layers)

        # 创建MLP头部
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # 将图像转换为分块嵌入
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # 重复类别令牌并连接到分块嵌入
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :n+1]
        x = self.dropout(x)

        # 通过层堆叠Transformer
        x = self.layers(x)

        # 返回MLP头部的输出
        return self.mlp_head(x[:, 0])
```