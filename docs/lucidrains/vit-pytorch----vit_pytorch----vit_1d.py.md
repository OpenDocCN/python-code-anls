# `.\lucidrains\vit-pytorch\vit_pytorch\vit_1d.py`

```py
import torch
from torch import nn

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# 导入所需的库

# 定义 FeedForward 类
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),  # 对输入进行 Layer Normalization
            nn.Linear(dim, hidden_dim),  # 线性变换
            nn.GELU(),  # GELU 激活函数
            nn.Dropout(dropout),  # Dropout 正则化
            nn.Linear(hidden_dim, dim),  # 线性变换
            nn.Dropout(dropout)  # Dropout 正则化
        )
    def forward(self, x):
        return self.net(x)

# 定义 Attention 类
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)  # 对输入进行 Layer Normalization
        self.attend = nn.Softmax(dim = -1)  # Softmax 函数
        self.dropout = nn.Dropout(dropout)  # Dropout 正则化

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)  # 线性变换

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),  # 线性变换
            nn.Dropout(dropout)  # Dropout 正则化
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # 将输入进行线性变换后切分成三部分
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)  # 重排张量形状

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # 点积计算

        attn = self.attend(dots)  # 注意力权重计算
        attn = self.dropout(attn)  # Dropout 正则化

        out = torch.matmul(attn, v)  # 加权求和
        out = rearrange(out, 'b h n d -> b n (h d)')  # 重排张量形状
        return self.to_out(out)

# 定义 Transformer 类
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

# 定义 ViT ��
class ViT(nn.Module):
    def __init__(self, *, seq_len, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert (seq_len % patch_size) == 0

        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p = patch_size),  # 重排张量形状
            nn.LayerNorm(patch_dim),  # 对输入进行 Layer Normalization
            nn.Linear(patch_dim, dim),  # 线性变换
            nn.LayerNorm(dim),  # 对输入进行 Layer Normalization
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # 位置编码
        self.cls_token = nn.Parameter(torch.randn(dim))  # 类别标记
        self.dropout = nn.Dropout(emb_dropout)  # Dropout 正则化

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)  # Transformer 模块

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),  # 对输入进行 Layer Normalization
            nn.Linear(dim, num_classes)  # 线性变换
        )

    def forward(self, series):
        x = self.to_patch_embedding(series)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)  # 重复类别标记

        x, ps = pack([cls_tokens, x], 'b * d')  # 打包张量

        x += self.pos_embedding[:, :(n + 1)]  # 加上位置编码
        x = self.dropout(x)  # Dropout 正则化

        x = self.transformer(x)  # Transformer 模块

        cls_tokens, _ = unpack(x, ps, 'b * d')  # 解包张量

        return self.mlp_head(cls_tokens)  # MLP 头部

if __name__ == '__main__':

    v = ViT(
        seq_len = 256,
        patch_size = 16,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 8,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    time_series = torch.randn(4, 3, 256)
    logits = v(time_series) # (4, 1000)
```