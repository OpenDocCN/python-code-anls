# `.\lucidrains\vit-pytorch\vit_pytorch\nest.py`

```
# 导入必要的库
from functools import partial
import torch
from torch import nn, einsum

# 导入 einops 库中的 rearrange 和 Reduce 函数
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

# 定义一个辅助函数，用于将输入值转换为元组
def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else ((val,) * depth)

# 定义 LayerNorm 类，用于实现层归一化
class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

# 定义 FeedForward 类，用于实现前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, dim, mlp_mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            LayerNorm(dim),
            nn.Conv2d(dim, dim * mlp_mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mlp_mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# 定义 Attention 类，用于实现注意力机制
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        dim_head = dim // heads
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, c, h, w, heads = *x.shape, self.heads

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# 定义 Aggregate 函数，用于聚合特征
def Aggregate(dim, dim_out):
    return nn.Sequential(
        nn.Conv2d(dim, dim_out, 3, padding = 1),
        LayerNorm(dim_out),
        nn.MaxPool2d(3, stride = 2, padding = 1)
    )

# 定义 Transformer 类，用于实现 Transformer 模型
class Transformer(nn.Module):
    def __init__(self, dim, seq_len, depth, heads, mlp_mult, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.pos_emb = nn.Parameter(torch.randn(seq_len))

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dropout = dropout),
                FeedForward(dim, mlp_mult, dropout = dropout)
            ]))
    def forward(self, x):
        *_, h, w = x.shape

        pos_emb = self.pos_emb[:(h * w)]
        pos_emb = rearrange(pos_emb, '(h w) -> () () h w', h = h, w = w)
        x = x + pos_emb

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# 定义 NesT 类，用于实现 NesT 模型
class NesT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        heads,
        num_hierarchies,
        block_repeats,
        mlp_mult = 4,
        channels = 3,
        dim_head = 64,
        dropout = 0.
    ):
        # 调用父类的构造函数
        super().__init__()
        # 确保图像尺寸能够被分块尺寸整除
        assert (image_size % patch_size) == 0, 'Image dimensions must be divisible by the patch size.'
        # 计算总的分块数量
        num_patches = (image_size // patch_size) ** 2
        # 计算每个分块的维度
        patch_dim = channels * patch_size ** 2
        # 计算特征图的大小
        fmap_size = image_size // patch_size
        # 计算块的数量
        blocks = 2 ** (num_hierarchies - 1)

        # 计算序列长度，跨层次保持不变
        seq_len = (fmap_size // blocks) ** 2
        # 生成层次列表
        hierarchies = list(reversed(range(num_hierarchies)))
        # 计算每个层次的倍数
        mults = [2 ** i for i in reversed(hierarchies)]

        # 计算每个层次的头数
        layer_heads = list(map(lambda t: t * heads, mults))
        # 计算每个层次的维度
        layer_dims = list(map(lambda t: t * dim, mults))
        # 最后一个维度
        last_dim = layer_dims[-1]

        # 添加最后一个维度到层次维度列表
        layer_dims = [*layer_dims, layer_dims[-1]]
        # 生成维度对
        dim_pairs = zip(layer_dims[:-1], layer_dims[1:])

        # 定义将图像转换为分块嵌入的序列
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1 = patch_size, p2 = patch_size),
            LayerNorm(patch_dim),
            nn.Conv2d(patch_dim, layer_dims[0], 1),
            LayerNorm(layer_dims[0])
        )

        # 将块重复次数转换为元组
        block_repeats = cast_tuple(block_repeats, num_hierarchies)

        # 初始化层次列表
        self.layers = nn.ModuleList([])

        # 遍历层次、头数、维度对、块重复次数
        for level, heads, (dim_in, dim_out), block_repeat in zip(hierarchies, layer_heads, dim_pairs, block_repeats):
            is_last = level == 0
            depth = block_repeat

            # 添加 Transformer 和 Aggregate 模块到层次列表
            self.layers.append(nn.ModuleList([
                Transformer(dim_in, seq_len, depth, heads, mlp_mult, dropout),
                Aggregate(dim_in, dim_out) if not is_last else nn.Identity()
            ]))


        # 定义 MLP 头部
        self.mlp_head = nn.Sequential(
            LayerNorm(last_dim),
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(last_dim, num_classes)
        )

    def forward(self, img):
        # 将图像转换为分块��入
        x = self.to_patch_embedding(img)
        b, c, h, w = x.shape

        # 获取层次数量
        num_hierarchies = len(self.layers)

        # 遍历层次，应用 Transformer 和 Aggregate 模块
        for level, (transformer, aggregate) in zip(reversed(range(num_hierarchies)), self.layers):
            block_size = 2 ** level
            x = rearrange(x, 'b c (b1 h) (b2 w) -> (b b1 b2) c h w', b1 = block_size, b2 = block_size)
            x = transformer(x)
            x = rearrange(x, '(b b1 b2) c h w -> b c (b1 h) (b2 w)', b1 = block_size, b2 = block_size)
            x = aggregate(x)

        # 应用 MLP 头部并返回结果
        return self.mlp_head(x)
```