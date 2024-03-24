# `.\lucidrains\vit-pytorch\vit_pytorch\simple_vit_with_patch_dropout.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn

# 从 einops 库中导入 rearrange 和 Rearrange 函数
from einops import rearrange
from einops.layers.torch import Rearrange

# 辅助函数

# 如果输入 t 是元组，则返回 t，否则返回 (t, t)
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# 生成二维位置编码的正弦和余弦值
def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    # 获取 patches 的形状信息
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    # 创建网格矩阵 y 和 x
    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    # 确保特征维度是 4 的倍数，用于 sincos 编码
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    # 计算 omega 值
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    # 计算位置编码
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

# 补丁丢弃

# 定义 PatchDropout 类
class PatchDropout(nn.Module):
    def __init__(self, prob):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob

    def forward(self, x):
        # 如果不是训练状态或概率为 0，则直接返回输入 x
        if not self.training or self.prob == 0.:
            return x

        # 获取输入 x 的形状信息
        b, n, _, device = *x.shape, x.device

        # 创建批次索引
        batch_indices = torch.arange(b, device = device)
        batch_indices = rearrange(batch_indices, '... -> ... 1')
        # 计算要保留的补丁数量
        num_patches_keep = max(1, int(n * (1 - self.prob)))
        # 随机选择要保留的补丁索引
        patch_indices_keep = torch.randn(b, n, device = device).topk(num_patches_keep, dim = -1).indices

        return x[batch_indices, patch_indices_keep]

# 类

# 定义前馈神经网络类
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

# 定义注意力机制类
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# 定义变换器类
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

# 简单 ViT 模型类
class SimpleViT(nn.Module):
    # 初始化函数，设置模型参数和层结构
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, patch_dropout = 0.5):
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

        # 定义将图像转换为补丁嵌入的层结构
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        # 定义补丁的丢弃层
        self.patch_dropout = PatchDropout(patch_dropout)

        # 定义变换器层
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        # 定义转换为潜在空间的层
        self.to_latent = nn.Identity()
        # 定义线性头层
        self.linear_head = nn.Linear(dim, num_classes)

    # 前向传播函数
    def forward(self, img):
        # 获取图像的形状和数据类型
        *_, h, w, dtype = *img.shape, img.dtype

        # 将图像转换为补丁嵌入
        x = self.to_patch_embedding(img)
        # 获取位置编码
        pe = posemb_sincos_2d(x)
        # 将位置编码添加到补丁嵌入中
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        # 对补丁进行丢弃
        x = self.patch_dropout(x)

        # 使用变换器进行转换
        x = self.transformer(x)
        # 对结果进行平均池化
        x = x.mean(dim = 1)

        # 转换为潜在空间
        x = self.to_latent(x)
        # 使用线性头层进行分类预测
        return self.linear_head(x)
```