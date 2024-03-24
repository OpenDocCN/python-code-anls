# `.\lucidrains\vit-pytorch\vit_pytorch\deepvit.py`

```py
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# 定义一个前馈神经网络类
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

# 定义一个注意力机制类
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)  # 对输入进行 Layer Normalization
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)  # 线性变换

        self.dropout = nn.Dropout(dropout)  # Dropout 正则化

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))  # 定义可学习参数

        self.reattn_norm = nn.Sequential(
            Rearrange('b h i j -> b i j h'),  # 重新排列张量维度
            nn.LayerNorm(heads),  # 对输入进行 Layer Normalization
            Rearrange('b i j h -> b h i j')  # 重新排列张量维度
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),  # 线性变换
            nn.Dropout(dropout)  # Dropout 正则化
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        x = self.norm(x)  # 对输入进行 Layer Normalization

        qkv = self.to_qkv(x).chunk(3, dim = -1)  # 将线性变换后的结果切分成三部分
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)  # 重新排列张量维度

        # attention

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # 计算点积
        attn = dots.softmax(dim=-1)  # Softmax 操作
        attn = self.dropout(attn)  # Dropout 正则化

        # re-attention

        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)  # 重新排列张量维度
        attn = self.reattn_norm(attn)  # 对输入进行 Layer Normalization

        # aggregate and out

        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # 点积操作
        out = rearrange(out, 'b h n d -> b n (h d)')  # 重新排列张量维度
        out =  self.to_out(out)  # 线性变换
        return out

# 定义一个 Transformer 类
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),  # 注意力机制
                FeedForward(dim, mlp_dim, dropout = dropout)  # 前馈神经网络
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x  # 注意力机制的输出与输入相加
            x = ff(x) + x  # 前馈神经网络的输出与输入相加
        return x

# 定义一个 DeepViT 类
class DeepViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),  # 重新排列张量维度
            nn.LayerNorm(patch_dim),  # 对输入进行 Layer Normalization
            nn.Linear(patch_dim, dim),  # 线性变换
            nn.LayerNorm(dim)  # 对输入进行 Layer Normalization
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # 定义可学习参数
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # 定义可学习参数
        self.dropout = nn.Dropout(emb_dropout)  # Dropout 正则化

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)  # Transformer 模块

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),  # 对输入进行 Layer Normalization
            nn.Linear(dim, num_classes)  # 线性变换
        )
    # 前向传播函数，接收输入图像并返回预测结果
    def forward(self, img):
        # 将输入图像转换为补丁嵌入
        x = self.to_patch_embedding(img)
        # 获取批量大小、补丁数量和嵌入维度
        b, n, _ = x.shape

        # 重复类别标记以匹配批量大小，并与补丁嵌入连接
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        # 添加位置嵌入到输入嵌入中
        x += self.pos_embedding[:, :(n + 1)]
        # 对输入进行 dropout 处理
        x = self.dropout(x)

        # 使用 Transformer 处理输入数据
        x = self.transformer(x)

        # 根据池化方式计算输出结果
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # 将输出结果转换为潜在空间
        x = self.to_latent(x)
        # 使用 MLP 头部处理潜在空间的输出，并返回预测结果
        return self.mlp_head(x)
```