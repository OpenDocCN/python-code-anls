# `.\lucidrains\vit-pytorch\vit_pytorch\simple_vit_3d.py`

```
import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

# 如果输入参数是元组，则返回元组，否则返回包含两个相同元素的元组
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# 生成三维位置编码的正弦和余弦值
def posemb_sincos_3d(patches, temperature = 10000, dtype = torch.float32):
    # 获取 patches 的形状信息
    _, f, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    # 生成三维网格坐标
    z, y, x = torch.meshgrid(
        torch.arange(f, device = device),
        torch.arange(h, device = device),
        torch.arange(w, device = device),
    indexing = 'ij')

    # 计算傅立叶维度
    fourier_dim = dim // 6

    # 计算温度参数
    omega = torch.arange(fourier_dim, device = device) / (fourier_dim - 1)
    omega = 1. / (temperature ** omega)

    # 计算位置编码
    z = z.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim = 1)

    # 如果特征维度不能被6整除，则进行填充
    pe = F.pad(pe, (0, dim - (fourier_dim * 6)))
    return pe.type(dtype)

# classes

# 前馈神经网络类
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

# 注意力机制类
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

# Transformer 模型类
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

class SimpleViT(nn.Module):
    # 初始化函数，设置模型参数和结构
    def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        # 调用父类的初始化函数
        super().__init__()
        # 获取图像大小的高度和宽度
        image_height, image_width = pair(image_size)
        # 获取图像块的高度和宽度
        patch_height, patch_width = pair(image_patch_size)

        # 断言图像高度和宽度能够被图像块的高度和宽度整除
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        # 断言帧数能够被帧块大小整除
        assert frames % frame_patch_size == 0, 'Frames must be divisible by the frame patch size'

        # 计算图像块的数量
        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        # 计算图像块的维度
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        # 将图像块转换为嵌入向量
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b f h w (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # 创建 Transformer 模型
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        # 将嵌入向量转换为潜在向量
        self.to_latent = nn.Identity()
        # 线性层，用于分类
        self.linear_head = nn.Linear(dim, num_classes)

    # 前向传播函数
    def forward(self, video):
        # 获取视频的形状信息
        *_, h, w, dtype = *video.shape, video.dtype

        # 将视频转换为图像块的嵌入向量
        x = self.to_patch_embedding(video)
        # 获取位置编码
        pe = posemb_sincos_3d(x)
        # 将位置编码加到嵌入向量中
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        # 经过 Transformer 模型处理
        x = self.transformer(x)
        # 对结果进行平均池化
        x = x.mean(dim = 1)

        # 转换为潜在向量
        x = self.to_latent(x)
        # 使用线性层进行分类
        return self.linear_head(x)
```