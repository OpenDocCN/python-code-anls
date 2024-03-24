# `.\lucidrains\vit-pytorch\vit_pytorch\cait.py`

```
from random import randrange
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

# 检查值是否存在
def exists(val):
    return val is not None

# 对层应用 dropout
def dropout_layers(layers, dropout):
    if dropout == 0:
        return layers

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) < dropout

    # 确保至少有一层保留
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers

# classes

# 缩放层
class LayerScale(nn.Module):
    def __init__(self, dim, fn, depth):
        super().__init__()
        if depth <= 18:  # 根据深度选择初始化值，详见论文第2节
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

# 前馈神经网络
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

# 注意力机制
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.mix_heads_pre_attn = nn.Parameter(torch.randn(heads, heads))
        self.mix_heads_post_attn = nn.Parameter(torch.randn(heads, heads))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None):
        b, n, _, h = *x.shape, self.heads

        x = self.norm(x)
        context = x if not exists(context) else torch.cat((x, context), dim = 1)

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        dots = einsum('b h i j, h g -> b g i j', dots, self.mix_heads_pre_attn)    # talking heads, pre-softmax

        attn = self.attend(dots)
        attn = self.dropout(attn)

        attn = einsum('b h i j, h g -> b g i j', attn, self.mix_heads_post_attn)   # talking heads, post-softmax

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# Transformer 模型
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., layer_dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layer_dropout = layer_dropout

        for ind in range(depth):
            self.layers.append(nn.ModuleList([
                LayerScale(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout), depth = ind + 1),
                LayerScale(dim, FeedForward(dim, mlp_dim, dropout = dropout), depth = ind + 1)
            ]))
    def forward(self, x, context = None):
        layers = dropout_layers(self.layers, dropout = self.layer_dropout)

        for attn, ff in layers:
            x = attn(x, context = context) + x
            x = ff(x) + x
        return x

class CaiT(nn.Module):
    # 初始化函数，设置模型参数
    def __init__(
        self,
        *,
        image_size,  # 图像大小
        patch_size,  # 补丁大小
        num_classes,  # 类别数量
        dim,  # 特征维度
        depth,  # 深度
        cls_depth,  # 分类深度
        heads,  # 多头注意力头数
        mlp_dim,  # MLP隐藏层维度
        dim_head = 64,  # 头维度
        dropout = 0.,  # 丢弃率
        emb_dropout = 0.,  # 嵌入层丢弃率
        layer_dropout = 0.  # 层丢弃率
    ):
        # 调用父类初始化函数
        super().__init__()
        # 检查图像尺寸是否能被补丁大小整除
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        # 计算补丁数量
        num_patches = (image_size // patch_size) ** 2
        # 计算补丁维度
        patch_dim = 3 * patch_size ** 2

        # 补丁嵌入层
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        # 位置嵌入
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        # 分类令牌
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # 丢弃层
        self.dropout = nn.Dropout(emb_dropout)

        # 补丁Transformer
        self.patch_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, layer_dropout)
        # 分类Transformer
        self.cls_transformer = Transformer(dim, cls_depth, heads, dim_head, mlp_dim, dropout, layer_dropout)

        # MLP头
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    # 前向传播函数
    def forward(self, img):
        # 补丁嵌入
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # 添加位置嵌入
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        # 补丁Transformer
        x = self.patch_transformer(x)

        # 重复分类令牌
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # 分类Transformer
        x = self.cls_transformer(cls_tokens, context = x)

        # 返回MLP头的结果
        return self.mlp_head(x[:, 0])
```