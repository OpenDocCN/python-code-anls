# `.\lucidrains\vit-pytorch\vit_pytorch\mp3.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块和 einsum 函数
from torch import nn, einsum
# 从 torch 库中导入 nn.functional 模块并重命名为 F
import torch.nn.functional as F

# 从 einops 库中导入 rearrange 和 repeat 函数，从 einops.layers.torch 中导入 Rearrange 类

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# 辅助函数

# 判断变量是否存在的函数
def exists(val):
    return val is not None

# 如果变量存在则返回变量，否则返回默认值的函数
def default(val, d):
    return val if exists(val) else d

# 将输入转换为元组的函数
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# 位置嵌入

# 生成二维正弦余弦位置嵌入的函数
def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    # 获取 patches 的形状信息
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    # 创建网格矩阵 y 和 x
    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    # 断言特征维度必须是 4 的倍数
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    # 计算 omega
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    # 计算位置嵌入
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

# 前馈网络

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

# (跨)注意力机制

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None):
        b, n, _, h = *x.shape, self.heads

        x = self.norm(x)

        context = self.norm(context) if exists(context) else x

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x, context = None):
        for attn, ff in self.layers:
            x = attn(x, context = context) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    # 初始化函数，定义模型的参数和结构
    def __init__(self, *, num_classes, image_size, patch_size, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0.):
        # 调用父类的初始化函数
        super().__init__()
        # 获取图像的高度和宽度
        image_height, image_width = pair(image_size)
        # 获取补丁的高度和宽度
        patch_height, patch_width = pair(patch_size)

        # 断言图像的高度和宽度必须能够被补丁的高度和宽度整除
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        
        # 计算补丁的数量
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # 计算每个补丁的维度
        patch_dim = channels * patch_height * patch_width

        # 设置模型的维度和补丁数量
        self.dim = dim
        self.num_patches = num_patches

        # 定义将图像转换为补丁嵌入的层序列
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # 创建 Transformer 模型
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # 定义将潜在表示转换为输出类别的层
        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    # 前向传播函数，定义模型的前向计算过程
    def forward(self, img):
        # 获取图像的形状和数据类型
        *_, h, w, dtype = *img.shape, img.dtype

        # 将图像转换为补丁嵌入
        x = self.to_patch_embedding(img)
        # 生成位置编码
        pe = posemb_sincos_2d(x)
        # 将位置编码加到补丁嵌入中
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        # 经过 Transformer 模型处理
        x = self.transformer(x)
        # 对补丁进行平均池化
        x = x.mean(dim = 1)

        # 转换为潜在表示
        x = self.to_latent(x)
        # 经过线性层得到输出类别
        return self.linear_head(x)
# 定义 Masked Position Prediction Pre-Training 类
class MP3(nn.Module):
    # 初始化函数，接受 ViT 模型和 masking 比例作为参数
    def __init__(self, vit: ViT, masking_ratio):
        super().__init__()
        self.vit = vit

        # 断言确保 masking 比例在 0 到 1 之间
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # 获取 ViT 模型的维度
        dim = vit.dim
        # 定义 MLP 头部，包含 LayerNorm 和 Linear 层
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, vit.num_patches)
        )

    # 前向传播函数，接受图像作为输入
    def forward(self, img):
        # 获取输入图像的设备信息
        device = img.device
        # 将图像转换为 token
        tokens = self.vit.to_patch_embedding(img)
        # 重新排列 token 的维度
        tokens = rearrange(tokens, 'b ... d -> b (...) d')

        # 获取 batch 大小和 patch 数量
        batch, num_patches, *_ = tokens.shape

        # Masking
        # 计算需要被 mask 的数量
        num_masked = int(self.masking_ratio * num_patches)
        # 生成随机索引并排序
        rand_indices = torch.rand(batch, num_patches, device=device).argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # 生成 batch 范围的索引
        batch_range = torch.arange(batch, device=device)[:, None]
        # 获取未被 mask 的 token
        tokens_unmasked = tokens[batch_range, unmasked_indices]

        # 使用 ViT 模型的 transformer 进行注意力计算
        attended_tokens = self.vit.transformer(tokens, tokens_unmasked)
        # 将输出结果通过 MLP 头部得到 logits
        logits = rearrange(self.mlp_head(attended_tokens), 'b n d -> (b n) d')
        
        # 定义标签
        labels = repeat(torch.arange(num_patches, device=device), 'n -> (b n)', b=batch)
        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)

        return loss
```