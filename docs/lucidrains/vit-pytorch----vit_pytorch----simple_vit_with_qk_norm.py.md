# `.\lucidrains\vit-pytorch\vit_pytorch\simple_vit_with_qk_norm.py`

```py
import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

# 定义一个函数，如果输入参数是元组则返回元组，否则返回包含两个相同元素的元组
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# 生成二维位置编码的正弦和余弦值
def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# they use a query-key normalization that is equivalent to rms norm (no mean-centering, learned gamma), from vit 22B paper

# in latest tweet, seem to claim more stable training at higher learning rates
# unsure if this has taken off within Brain, or it has some hidden drawback

# 定义一个类，实现 RMS 归一化
class RMSNorm(nn.Module):
    def __init__(self, heads, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim) / self.scale)

    def forward(self, x):
        normed = F.normalize(x, dim = -1)
        return normed * self.scale * self.gamma

# classes

# 定义一个类，实现前馈神经网络
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

# 定义一个类，实现注意力机制
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.q_norm = RMSNorm(heads, dim_head)
        self.k_norm = RMSNorm(heads, dim_head)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        q = self.q_norm(q)
        k = self.k_norm(k)

        dots = torch.matmul(q, k.transpose(-1, -2))

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# 定义一个类，实现 Transformer 模型
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
    # 初始化函数，设置模型参数和层结构
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        # 调用父类的初始化函数
        super().__init__()
        # 获取图像的高度和宽度
        image_height, image_width = pair(image_size)
        # 获取补丁的高度和宽度
        patch_height, patch_width = pair(patch_size)

        # 断言图像的高度和宽度能够被补丁的高度和宽度整除
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # 计算补丁的维度
        patch_dim = channels * patch_height * patch_width

        # 定义将图像转换为补丁嵌入的层结构
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # 生成位置编码
        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        ) 

        # 定义 Transformer 模型
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        # 池化方式为平均池化
        self.pool = "mean"
        # 定义将嵌入转换为潜在表示的层结构
        self.to_latent = nn.Identity()

        # 线性层归一化
        self.linear_head = nn.LayerNorm(dim)

    # 前向传播函数
    def forward(self, img):
        # 获取输入图像的设备信息
        device = img.device

        # 将输入图像转换为补丁嵌入
        x = self.to_patch_embedding(img)
        # 添加位置编码
        x += self.pos_embedding.to(device, dtype=x.dtype)

        # 经过 Transformer 模型
        x = self.transformer(x)
        # 对特征进行平均池化
        x = x.mean(dim = 1)

        # 将特征转换为潜在表示
        x = self.to_latent(x)
        # 返回线性层归一化后的结果
        return self.linear_head(x)
```