# `.\lucidrains\vit-pytorch\vit_pytorch\parallel_vit.py`

```py
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn

# 从 einops 库中导入 rearrange 和 repeat 函数
from einops import rearrange, repeat
# 从 einops.layers.torch 库中导入 Rearrange 类
from einops.layers.torch import Rearrange

# 辅助函数

# 定义一个函数，如果输入参数 t 是元组则返回 t，否则返回一个包含 t 的元组
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# 类定义

# 定义一个并行模块类
class Parallel(nn.Module):
    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        return sum([fn(x) for fn in self.fns])

# 定义一个前馈神经网络类
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

# 定义一个注意力机制类
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

# 定义一个 Transformer 类
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, num_parallel_branches = 2, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        attn_block = lambda: Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)
        ff_block = lambda: FeedForward(dim, mlp_dim, dropout = dropout)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Parallel(*[attn_block() for _ in range(num_parallel_branches)]),
                Parallel(*[ff_block() for _ in range(num_parallel_branches)]),
            ]))

    def forward(self, x):
        for attns, ffs in self.layers:
            x = attns(x) + x
            x = ffs(x) + x
        return x

# 定义一个 ViT 类
class ViT(nn.Module):
    # 初始化函数，设置模型参数
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', num_parallel_branches = 2, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        # 检查图像尺寸是否能被分块尺寸整除
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # 计算分块数量和分块维度
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        # 检查池化类型是否为'cls'或'mean'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # 将图像分块转换为嵌入向量
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        # 初始化位置编码和类别标记
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # 初始化Transformer模型
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, num_parallel_branches, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # MLP头部，用于分类
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    # 前向传播函数
    def forward(self, img):
        # 图像转换为分块嵌入向量
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # 重复类别标记以匹配批次大小
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # Transformer模型处理嵌入向量
        x = self.transformer(x)

        # 池化操作，根据池化类型选择不同的操作
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # 转换为最终输出
        x = self.to_latent(x)
        return self.mlp_head(x)
```