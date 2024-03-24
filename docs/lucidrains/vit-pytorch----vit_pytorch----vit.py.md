# `.\lucidrains\vit-pytorch\vit_pytorch\vit.py`

```py
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn

# 从 einops 库中导入 rearrange 和 repeat 函数
from einops import rearrange, repeat
# 从 einops.layers.torch 库中导入 Rearrange 类
from einops.layers.torch import Rearrange

# 定义辅助函数 pair，用于返回元组形式的输入
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# 定义 FeedForward 类，继承自 nn.Module 类
class FeedForward(nn.Module):
    # 初始化函数，接受维度 dim、隐藏层维度 hidden_dim 和 dropout 概率
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        # 定义神经网络结构
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    # 前向传播函数
    def forward(self, x):
        return self.net(x)

# 定义 Attention 类，继承自 nn.Module 类
class Attention(nn.Module):
    # 初始化函数，接受维度 dim、头数 heads、头维度 dim_head 和 dropout 概率
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

    # 前向传播函数
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

# 定义 Transformer 类，继承自 nn.Module 类
class Transformer(nn.Module):
    # 初始化函数，接受维度 dim、深度 depth、头数 heads、头维度 dim_head、MLP 维度 mlp_dim 和 dropout 概率
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    # 前向传播函数
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

# 定义 ViT 类，继承自 nn.Module 类
class ViT(nn.Module):
    # 初始化函数，接受关键字参数 image_size、patch_size、num_classes、dim、depth、heads、mlp_dim、pool、channels、dim_head、dropout 和 emb_dropout
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)
    # 前向传播函数，接收输入图像并进行处理
    def forward(self, img):
        # 将输入图像转换为补丁嵌入
        x = self.to_patch_embedding(img)
        # 获取补丁嵌入的形状信息
        b, n, _ = x.shape

        # 为每个样本添加类别标记
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        # 将类别标记与补丁嵌入拼接在一起
        x = torch.cat((cls_tokens, x), dim=1)
        # 添加位置编码到输入
        x += self.pos_embedding[:, :(n + 1)]
        # 对输入进行 dropout 处理
        x = self.dropout(x)

        # 使用 Transformer 处理输入
        x = self.transformer(x)

        # 对 Transformer 输出进行池化操作，取平均值或者只取第一个位置的输出
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # 将输出转换为潜在空间
        x = self.to_latent(x)
        # 使用 MLP 头部处理最终输出
        return self.mlp_head(x)
```