# `.\lucidrains\vit-pytorch\vit_pytorch\vit_with_patch_dropout.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn

# 从 einops 库中导入 rearrange 和 repeat 函数，以及 Rearrange 类
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# 辅助函数

# 如果输入 t 是元组，则返回 t，否则返回包含 t 的元组
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# 类定义

# 定义 PatchDropout 类，继承自 nn.Module
class PatchDropout(nn.Module):
    # 初始化函数，接受概率参数 prob
    def __init__(self, prob):
        super().__init__()
        # 断言概率在 [0, 1) 范围内
        assert 0 <= prob < 1.
        self.prob = prob

    # 前向传播函数，接受输入 x
    def forward(self, x):
        # 如果不在训练模式或概率为 0，则直接返回输入 x
        if not self.training or self.prob == 0.:
            return x

        # 获取输入 x 的形状信息
        b, n, _, device = *x.shape, x.device

        # 生成 batch 索引
        batch_indices = torch.arange(b, device = device)
        batch_indices = rearrange(batch_indices, '... -> ... 1')
        # 计算保留的 patch 数量
        num_patches_keep = max(1, int(n * (1 - self.prob)))
        # 生成保留的 patch 索引
        patch_indices_keep = torch.randn(b, n, device = device).topk(num_patches_keep, dim = -1).indices

        return x[batch_indices, patch_indices_keep]

# 定义 FeedForward 类，继承自 nn.Module
class FeedForward(nn.Module):
    # 初始化函数，接受维度 dim、隐藏层维度 hidden_dim 和 dropout 参数
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        # 定义网络结构
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    # 前向传播函数，接受输入 x
    def forward(self, x):
        return self.net(x)

# 定义 Attention 类，继承自 nn.Module
class Attention(nn.Module):
    # 初始化函数，接受维度 dim、头数 heads、头维度 dim_head 和 dropout 参数
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

    # 前向传播函数，接受输入 x
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

# 定义 Transformer 类，继承自 nn.Module
class Transformer(nn.Module):
    # 初始化函数，接受维度 dim、深度 depth、头数 heads、头维度 dim_head、MLP 维度 mlp_dim 和 dropout 参数
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        # 根据深度循环创建多个 Transformer 层
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    # 前向传播函数，接受输入 x
    def forward(self, x):
        # 遍历每个 Transformer 层
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# 定义 ViT 类，继承自 nn.Module
class ViT(nn.Module):
    # 初始化函数，设置模型参数
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., patch_dropout = 0.25):
        # 调用父类的初始化函数
        super().__init__()
        # 获取图像的高度和宽度
        image_height, image_width = pair(image_size)
        # 获取补丁的高度和宽度
        patch_height, patch_width = pair(patch_size)

        # 断言图像的高度和宽度能够被补丁的高度和宽度整除
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # 计算补丁的数量
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # 计算每个补丁的维度
        patch_dim = channels * patch_height * patch_width
        # 断言池化类型只能是'cls'（CLS标记）或'mean'（平均池化）
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # 将图像转换为补丁嵌入
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        # 初始化位置嵌入参数
        self.pos_embedding = nn.Parameter(torch.randn(num_patches, dim))
        # 初始化CLS标记
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # 创建补丁丢弃层
        self.patch_dropout = PatchDropout(patch_dropout)
        # 创建嵌入丢弃层
        self.dropout = nn.Dropout(emb_dropout)

        # 创建Transformer模型
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # 设置池化类型
        self.pool = pool
        # 创建转换到潜在空间的层
        self.to_latent = nn.Identity()

        # 创建MLP头部
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    # 前向传播函数
    def forward(self, img):
        # 将图像转换为补丁嵌入
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # 添加位置嵌入
        x += self.pos_embedding

        # 对补丁进行丢弃
        x = self.patch_dropout(x)

        # 重复CLS标记以匹配批次大小
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)

        # 将CLS标记和补丁连接在一起
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)

        # 使用Transformer进行特征提取
        x = self.transformer(x)

        # 池化操作，根据池化类型选择不同的方式
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # 转换到潜在空间
        x = self.to_latent(x)
        # 使用MLP头部进行分类预测
        return self.mlp_head(x)
```