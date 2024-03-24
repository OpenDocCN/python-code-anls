# `.\lucidrains\vit-pytorch\vit_pytorch\vit_for_small_dataset.py`

```
# 从 math 模块中导入 sqrt 函数
from math import sqrt
# 导入 torch 模块
import torch
# 从 torch.nn 模块中导入 functional 模块和 nn 模块
import torch.nn.functional as F
from torch import nn
# 从 einops 模块中导入 rearrange 和 repeat 函数，从 einops.layers.torch 模块中导入 Rearrange 类

# 定义辅助函数 pair，如果输入参数 t 是元组则返回 t，否则返回 (t, t)
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# 定义 FeedForward 类，继承自 nn.Module 类
class FeedForward(nn.Module):
    # 初始化函数，接受维度 dim、隐藏层维度 hidden_dim 和 dropout 参数
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

# 定义 LSA 类，继承自 nn.Module 类
class LSA(nn.Module):
    # 初始化函数，接受维度 dim、头数 heads、头维度 dim_head 和 dropout 参数
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))

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
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()

        mask = torch.eye(dots.shape[-1], device = dots.device, dtype = torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# 定义 Transformer 类，继承自 nn.Module 类
class Transformer(nn.Module):
    # 初始化函数，接受维度 dim、深度 depth、头数 heads、头维度 dim_head、MLP维度 mlp_dim 和 dropout 参数
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LSA(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    # 前向传播函数
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# 定义 SPT 类，继承自 nn.Module 类
class SPT(nn.Module):
    # 初始化函数，接受维度 dim、patch 大小 patch_size 和通道数 channels 参数
    def __init__(self, *, dim, patch_size, channels = 3):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels

        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )

    # 前向传播函数
    def forward(self, x):
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        x_with_shifts = torch.cat((x, *shifted_x), dim = 1)
        return self.to_patch_tokens(x_with_shifts)

# 定义 ViT 类
class ViT(nn.Module):
    # 初始化函数，设置模型参数
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
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
        # 检查池化类型是否为 'cls' 或 'mean'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # 创建补丁嵌入层
        self.to_patch_embedding = SPT(dim = dim, patch_size = patch_size, channels = channels)

        # 初始化位置嵌入参数
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # 初始化类别标记参数
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # 初始化丢弃层
        self.dropout = nn.Dropout(emb_dropout)

        # 创建 Transformer 模型
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # 设置池化类型
        self.pool = pool
        # 创建转换到潜在空间的层
        self.to_latent = nn.Identity()

        # 创建 MLP 头部
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    # 前向传播函数
    def forward(self, img):
        # 将图像转换为补丁
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # 重复类别标记以匹配批次大小
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # 将类别标记与补丁连接
        x = torch.cat((cls_tokens, x), dim=1)
        # 添加位置嵌入
        x += self.pos_embedding[:, :(n + 1)]
        # 应用丢弃层
        x = self.dropout(x)

        # 使用 Transformer 进行转换
        x = self.transformer(x)

        # 池化操作，根据池化类型选择不同的方式
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # 转换到潜在空间
        x = self.to_latent(x)
        # 使用 MLP 头部进行分类
        return self.mlp_head(x)
```