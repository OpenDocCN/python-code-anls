# `.\lucidrains\vit-pytorch\vit_pytorch\vit_with_patch_merger.py`

```py
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

# helpers

# 检查值是否存在
def exists(val):
    return val is not None

# 返回值或默认值
def default(val ,d):
    return val if exists(val) else d

# 将输入转换为元组
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# patch merger class

# 定义 PatchMerger 类
class PatchMerger(nn.Module):
    def __init__(self, dim, num_tokens_out):
        super().__init__()
        self.scale = dim ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.queries = nn.Parameter(torch.randn(num_tokens_out, dim))

    def forward(self, x):
        x = self.norm(x)
        sim = torch.matmul(self.queries, x.transpose(-1, -2)) * self.scale
        attn = sim.softmax(dim = -1)
        return torch.matmul(attn, x)

# classes

# 定义 FeedForward 类
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

# 定义 Attention 类
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
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# 定义 Transformer 类
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., patch_merge_layer = None, patch_merge_num_tokens = 8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        self.patch_merge_layer_index = default(patch_merge_layer, depth // 2) - 1 # default to mid-way through transformer, as shown in paper
        self.patch_merger = PatchMerger(dim = dim, num_tokens_out = patch_merge_num_tokens)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for index, (attn, ff) in enumerate(self.layers):
            x = attn(x) + x
            x = ff(x) + x

            if index == self.patch_merge_layer_index:
                x = self.patch_merger(x)

        return self.norm(x)

class ViT(nn.Module):
    # 初始化函数，设置模型参数
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, patch_merge_layer = None, patch_merge_num_tokens = 8, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        # 调用父类的初始化函数
        super().__init__()
        # 获取图像的高度和宽度
        image_height, image_width = pair(image_size)
        # 获取补丁的高度和宽度
        patch_height, patch_width = pair(patch_size)

        # 检查图像的尺寸是否能被补丁的尺寸整除
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # 计算补丁的数量
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # 计算每个补丁的维度
        patch_dim = channels * patch_height * patch_width

        # 定义将图像转换为补丁嵌入的序列
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        # 初始化位置嵌入参数
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # 定义丢弃层
        self.dropout = nn.Dropout(emb_dropout)

        # 初始化Transformer模型
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, patch_merge_layer, patch_merge_num_tokens)

        # 定义MLP头部
        self.mlp_head = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(dim, num_classes)
        )

    # 前向传播函数
    def forward(self, img):
        # 将图像转换为补丁嵌入
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # 添加位置嵌入到补丁嵌入中
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        # 使用Transformer进行特征提取
        x = self.transformer(x)

        # 使用MLP头部进行分类
        return self.mlp_head(x)
```